from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Mapping

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "RecurrentMarketExpert requires PyTorch. "
        "Install a PyTorch build appropriate for the runtime device."
    ) from exc

from worlds.market.interface import MarketInput, TemporalFeatureWindow
from worlds.market.perception.interface import (
    ExpertOutput,
    ExpertTelemetry,
    MarketExpert,
)


_SCALE_NAMES = ("micro", "short", "medium", "long")
_SCALE_SEED_OFFSETS = {
    "micro": 137,
    "short": 251,
    "medium": 359,
    "long": 479,
}


# ============================================================
# Causal replay trace
# ============================================================


@dataclass(frozen=True, slots=True)
class _WindowTrace:
    scale: str
    feature_count: int
    rows: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class _RecurrentTrace:
    transition_id: str
    windows: tuple[_WindowTrace, ...]


# ============================================================
# Stabilized recurrent cell
# ============================================================


class _StabilizedLSTMCell(nn.Module):
    """
    xLSTM-inspired stabilized recurrent cell.

    This is intentionally implemented locally rather than depending on a
    specific external xLSTM package.

    Core idea:

        input_t + h_(t-1)
            ↓
        candidate z_t
        exponential input gate i_t
        exponential forget gate f_t
        output gate o_t
            ↓
        log-domain stabilizer m_t
            ↓
        numerically stable normalized memory update
            ↓
        h_t

    The exponential gates allow sharper write/retain behavior than a plain
    sigmoid LSTM while the stabilizer prevents those exponentials from
    exploding.

    This is not claimed to be a verbatim reproduction of any particular
    published xLSTM implementation. It is an xLSTM-style recurrent
    specialist with the same architectural motivation: stronger, stabilized
    recurrent memory dynamics.
    """

    def __init__(
        self,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.hidden_size = int(hidden_size)

        # Separate input and recurrent projections make the recurrence
        # explicit and easy to inspect.
        self.x_candidate = nn.Linear(
            hidden_size,
            hidden_size,
        )
        self.h_candidate = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.x_input = nn.Linear(
            hidden_size,
            hidden_size,
        )
        self.h_input = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.x_forget = nn.Linear(
            hidden_size,
            hidden_size,
        )
        self.h_forget = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.x_output = nn.Linear(
            hidden_size,
            hidden_size,
        )
        self.h_output = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.output_projection = nn.Linear(
            hidden_size,
            hidden_size,
        )

    def initial_state(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        zeros = torch.zeros(
            self.hidden_size,
            device=device,
            dtype=dtype,
        )

        # h, c, n, m
        #
        # m starts at a large negative finite value instead of -inf so
        # arithmetic remains simple on all devices/backends.
        return (
            zeros.clone(),
            zeros.clone(),
            zeros.clone(),
            torch.full(
                (self.hidden_size,),
                -30.0,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[
        Tensor,
        tuple[Tensor, Tensor, Tensor, Tensor],
    ]:
        h_prev, c_prev, n_prev, m_prev = state

        candidate = torch.tanh(
            self.x_candidate(x)
            + self.h_candidate(h_prev)
        )

        log_input = (
            self.x_input(x)
            + self.h_input(h_prev)
        )

        # softplus gives a positive retain strength; log(sigmoid()) would
        # constrain it to <= 0 and make strong memory retention harder.
        log_forget = -F.softplus(
            -(
                self.x_forget(x)
                + self.h_forget(h_prev)
            )
        )

        output_gate = torch.sigmoid(
            self.x_output(x)
            + self.h_output(h_prev)
        )

        # Stabilized exponential gates.
        m = torch.maximum(
            log_forget + m_prev,
            log_input,
        )

        input_weight = torch.exp(
            log_input - m
        )

        forget_weight = torch.exp(
            log_forget + m_prev - m
        )

        c = (
            forget_weight * c_prev
            + input_weight * candidate
        )

        n = (
            forget_weight * n_prev
            + input_weight
        )

        normalized_memory = c / n.clamp_min(
            1e-6
        )

        h = output_gate * torch.tanh(
            normalized_memory
        )

        h = self.output_projection(h)

        return h, (h, c, n, m)


class _ResidualStabilizedRecurrentLayer(nn.Module):
    """
    One full recurrent layer over a temporal sequence.

    LayerNorm and residual connection help stacked recurrent layers remain
    trainable without turning this expert into another SSM.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()

        self.hidden_size = int(hidden_size)

        self.norm = nn.LayerNorm(
            hidden_size
        )
        self.cell = _StabilizedLSTMCell(
            hidden_size
        )
        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        sequence: Tensor,
    ) -> Tensor:
        if sequence.ndim != 2:
            raise ValueError(
                "_ResidualStabilizedRecurrentLayer expects "
                "[time, hidden]"
            )

        if sequence.shape[1] != self.hidden_size:
            raise ValueError(
                "Recurrent hidden width changed unexpectedly"
            )

        if sequence.shape[0] <= 0:
            raise ValueError(
                "Recurrent layer cannot process an empty sequence"
            )

        normalized = self.norm(
            sequence
        )

        state = self.cell.initial_state(
            device=sequence.device,
            dtype=sequence.dtype,
        )

        outputs: list[Tensor] = []

        for index in range(
            normalized.shape[0]
        ):
            output, state = self.cell(
                normalized[index],
                state,
            )
            outputs.append(output)

        recurrent = torch.stack(
            outputs,
            dim=0,
        )

        recurrent = self.dropout(
            recurrent
        )

        return sequence + recurrent


class _TemporalRecurrentBranch(nn.Module):
    """
    One stabilized recurrent branch for one Market temporal scale.
    """

    def __init__(
        self,
        *,
        input_features: int,
        hidden_size: int,
        branch_size: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.input_features = int(
            input_features
        )
        self.hidden_size = int(
            hidden_size
        )

        self.input_projection = nn.Sequential(
            nn.Linear(
                input_features,
                hidden_size,
            ),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        self.layers = nn.ModuleList(
            [
                _ResidualStabilizedRecurrentLayer(
                    hidden_size,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.summary_gate = nn.Sequential(
            nn.Linear(
                hidden_size * 3,
                hidden_size,
            ),
            nn.Sigmoid(),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(
                hidden_size * 4,
                branch_size * 2,
            ),
            nn.SiLU(),
            nn.Linear(
                branch_size * 2,
                branch_size,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        rows: Tensor,
    ) -> Tensor:
        if rows.ndim != 2:
            raise ValueError(
                "_TemporalRecurrentBranch expects [time, features]"
            )

        if rows.shape[1] != self.input_features:
            raise ValueError(
                "Recurrent branch feature count changed unexpectedly"
            )

        x = self.input_projection(
            rows
        )

        for layer in self.layers:
            x = layer(x)

        last_state = x[-1]
        mean_state = x.mean(dim=0)
        max_state = x.amax(dim=0)

        summary_source = torch.cat(
            (
                last_state,
                mean_state,
                max_state,
            ),
            dim=0,
        )

        summary_gate = self.summary_gate(
            summary_source
        )

        # Learned interpolation between persistent last recurrent state and
        # broader sequence summary.
        blended = (
            summary_gate * last_state
            + (1.0 - summary_gate)
            * mean_state
        )

        pooled = torch.cat(
            (
                last_state,
                mean_state,
                max_state,
                blended,
            ),
            dim=0,
        )

        return self.output_projection(
            pooled
        )


class _RecurrentNetwork(nn.Module):
    """
    Multi-timescale stabilized recurrent expert network.

    Each scale owns an independent branch. Final expert fusion receives all
    branch representations plus availability bits.
    """

    def __init__(
        self,
        *,
        representation_size: int,
        hidden_size: int,
        branch_size: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.representation_size = int(
            representation_size
        )
        self.hidden_size = int(
            hidden_size
        )
        self.branch_size = int(
            branch_size
        )
        self.depth = int(depth)
        self.dropout = float(
            dropout
        )

        self.branches = nn.ModuleDict()

        fusion_input_size = (
            len(_SCALE_NAMES)
            * branch_size
            + len(_SCALE_NAMES)
        )

        fusion_hidden_size = max(
            representation_size * 2,
            branch_size * 2,
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_input_size,
                fusion_hidden_size,
            ),
            nn.SiLU(),
            nn.Linear(
                fusion_hidden_size,
                representation_size,
            ),
            nn.Tanh(),
        )

    def ensure_branch(
        self,
        scale: str,
        *,
        input_features: int,
        seed: int,
        device: torch.device,
    ) -> None:
        existing = (
            self.branches[scale]
            if scale in self.branches
            else None
        )

        if existing is not None:
            if not isinstance(
                existing,
                _TemporalRecurrentBranch,
            ):
                raise RuntimeError(
                    f"Unexpected recurrent branch type "
                    f"for scale {scale!r}"
                )

            if (
                existing.input_features
                != input_features
            ):
                raise ValueError(
                    f"Recurrent feature width for {scale!r} "
                    f"changed from {existing.input_features} "
                    f"to {input_features}. Keep MarketInput "
                    "feature layout stable or reset/reconfigure "
                    "the expert."
                )

            return

        with torch.random.fork_rng(
            devices=[]
        ):
            torch.manual_seed(
                seed
            )

            branch = _TemporalRecurrentBranch(
                input_features=input_features,
                hidden_size=self.hidden_size,
                branch_size=self.branch_size,
                depth=self.depth,
                dropout=self.dropout,
            )

        self.branches[scale] = branch.to(
            device
        )

    def forward(
        self,
        windows: Mapping[str, Tensor],
    ) -> Tensor:
        reference_parameter = next(
            self.fusion.parameters()
        )

        device = reference_parameter.device
        dtype = reference_parameter.dtype

        branch_vectors: list[Tensor] = []
        availability: list[Tensor] = []

        for scale in _SCALE_NAMES:
            rows = windows.get(
                scale
            )

            if rows is None:
                branch_vectors.append(
                    torch.zeros(
                        self.branch_size,
                        device=device,
                        dtype=dtype,
                    )
                )
                availability.append(
                    torch.zeros(
                        1,
                        device=device,
                        dtype=dtype,
                    )
                )
                continue

            branch = self.branches[
                scale
            ]

            branch_vectors.append(
                branch(rows)
            )

            availability.append(
                torch.ones(
                    1,
                    device=device,
                    dtype=dtype,
                )
            )

        fusion_input = torch.cat(
            (
                *branch_vectors,
                *availability,
            ),
            dim=0,
        )

        return self.fusion(
            fusion_input
        )


# ============================================================
# Recurrent Market expert
# ============================================================


class RecurrentMarketExpert(MarketExpert):
    """
    Trainable multi-timescale stabilized recurrent Market expert.

    Intended specialization:
        - persistent nonlinear sequential memory;
        - event ordering;
        - long-lived recurrent context;
        - dynamics complementary to TCN/SSM/Attention.

    The recurrent cell is xLSTM-inspired but implemented locally so Heiwa is
    not coupled to a particular third-party xLSTM package.

    `training_horizon_minutes=240` is Market-specific experiment metadata.
    It is not a forced holding period and not a shared Heiwa parameter.

    Delayed learning replays the exact normalized input stored under the
    original transition_id. Missing traces return False; current Market
    state is never substituted.
    """

    def __init__(
        self,
        *,
        representation_size: int = 64,
        hidden_size: int = 64,
        branch_size: int = 48,
        depth: int = 3,
        dropout: float = 0.05,
        learning_rate: float = 1e-3,
        gradient_clip: float = 5.0,
        min_steps: int = 8,
        max_steps_per_window: int = 512,
        max_traces: int = 512,
        training_horizon_minutes: int = 240,
        device: str | torch.device | None = None,
        seed: int = 20260830,
    ) -> None:
        if representation_size <= 0:
            raise ValueError(
                "representation_size must be > 0"
            )

        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be > 0"
            )

        if branch_size <= 0:
            raise ValueError(
                "branch_size must be > 0"
            )

        if depth <= 0:
            raise ValueError(
                "depth must be > 0"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        if learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be > 0"
            )

        if gradient_clip <= 0.0:
            raise ValueError(
                "gradient_clip must be > 0"
            )

        if min_steps <= 0:
            raise ValueError(
                "min_steps must be > 0"
            )

        if max_steps_per_window < min_steps:
            raise ValueError(
                "max_steps_per_window must be >= min_steps"
            )

        if max_traces <= 0:
            raise ValueError(
                "max_traces must be > 0"
            )

        if training_horizon_minutes <= 0:
            raise ValueError(
                "training_horizon_minutes must be > 0"
            )

        self._representation_size = int(
            representation_size
        )

        self.hidden_size = int(
            hidden_size
        )
        self.branch_size = int(
            branch_size
        )
        self.depth = int(depth)
        self.dropout = float(
            dropout
        )

        self.learning_rate = float(
            learning_rate
        )
        self.gradient_clip = float(
            gradient_clip
        )

        self.min_steps = int(
            min_steps
        )
        self.max_steps_per_window = int(
            max_steps_per_window
        )
        self.max_traces = int(
            max_traces
        )

        self.training_horizon_minutes = int(
            training_horizon_minutes
        )

        self.seed = int(seed)

        self.device = self._resolve_device(
            device
        )

        self._network = self._new_network().to(
            self.device
        )

        self._branch_feature_counts: dict[
            str,
            int,
        ] = {}

        self._traces: OrderedDict[
            str,
            _RecurrentTrace,
        ] = OrderedDict()

        self._forward_calls = 0
        self._learning_updates = 0

        self._last_representation_norm = 0.0
        self._last_gradient_norm = 0.0
        self._last_available = False
        self._last_active_scales = 0
        self._last_total_steps = 0

    # ========================================================
    # MarketExpert contract
    # ========================================================

    @property
    def name(self) -> str:
        return "recurrent"

    @property
    def representation_size(self) -> int:
        return self._representation_size

    def encode(
        self,
        market_input: MarketInput,
        *,
        transition_id: str,
    ) -> ExpertOutput:
        if not transition_id:
            raise ValueError(
                "transition_id must not be empty"
            )

        trace_windows = self._extract_windows(
            market_input
        )

        self._forward_calls += 1

        self._last_active_scales = len(
            trace_windows
        )
        self._last_total_steps = sum(
            len(window.rows)
            for window in trace_windows
        )

        if not trace_windows:
            self._last_available = False
            self._last_representation_norm = 0.0

            self._traces.pop(
                transition_id,
                None,
            )

            return ExpertOutput(
                expert_name=self.name,
                transition_id=transition_id,
                representation=(),
                available=False,
                uncertainty=None,
                auxiliary={},
                diagnostics={
                    "active_scales": 0.0,
                    "total_steps": 0.0,
                    "training_horizon_minutes": float(
                        self.training_horizon_minutes
                    ),
                },
            )

        self._ensure_branches(
            trace_windows
        )

        tensor_windows = (
            self._trace_windows_to_tensors(
                trace_windows
            )
        )

        self._network.eval()

        with torch.no_grad():
            representation_tensor = self._network(
                tensor_windows
            )

        representation = self._tensor_to_tuple(
            representation_tensor
        )

        self._store_trace(
            _RecurrentTrace(
                transition_id=transition_id,
                windows=trace_windows,
            )
        )

        self._last_available = True
        self._last_representation_norm = (
            self._norm(
                representation
            )
        )

        return ExpertOutput(
            expert_name=self.name,
            transition_id=transition_id,
            representation=representation,
            available=True,
            uncertainty=None,
            auxiliary={},
            diagnostics={
                "active_scales": float(
                    len(trace_windows)
                ),
                "total_steps": float(
                    self._last_total_steps
                ),
                "representation_norm": (
                    self._last_representation_norm
                ),
                "training_horizon_minutes": float(
                    self.training_horizon_minutes
                ),
            },
        )

    def learn(
        self,
        transition_id: str,
        representation_gradient: tuple[
            float,
            ...,
        ],
    ) -> bool:
        trace = self._traces.pop(
            transition_id,
            None,
        )

        if trace is None:
            return False

        gradient = tuple(
            float(value)
            for value
            in representation_gradient
        )

        if (
            len(gradient)
            != self.representation_size
        ):
            raise ValueError(
                "Recurrent representation gradient "
                "size mismatch: "
                f"expected {self.representation_size}, "
                f"got {len(gradient)}"
            )

        for value in gradient:
            if not isfinite(value):
                raise ValueError(
                    "Recurrent representation gradient "
                    "contains non-finite value"
                )

        self._ensure_branches(
            trace.windows
        )

        tensor_windows = (
            self._trace_windows_to_tensors(
                trace.windows
            )
        )

        self._network.train()
        self._zero_grad()

        representation = self._network(
            tensor_windows
        )

        gradient_tensor = torch.tensor(
            gradient,
            dtype=representation.dtype,
            device=representation.device,
        )

        # Local scalar surrogate whose derivative w.r.t. the expert
        # representation equals the requested downstream gradient.
        local_loss = torch.dot(
            representation,
            gradient_tensor.detach(),
        )

        local_loss.backward()

        parameters = tuple(
            parameter
            for parameter
            in self._network.parameters()
            if parameter.grad is not None
        )

        if not parameters:
            self._zero_grad()
            return False

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=self.gradient_clip,
            )
        )

        self._last_gradient_norm = float(
            gradient_norm.detach().cpu().item()
        )

        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is None:
                    continue

                parameter.add_(
                    parameter.grad,
                    alpha=-self.learning_rate,
                )

        self._zero_grad()
        self._learning_updates += 1

        return True

    def telemetry(
        self,
    ) -> ExpertTelemetry:
        return ExpertTelemetry(
            expert_name=self.name,
            available=self._last_available,
            forward_calls=self._forward_calls,
            learning_updates=(
                self._learning_updates
            ),
            last_representation_norm=(
                self._last_representation_norm
            ),
            last_gradient_norm=(
                self._last_gradient_norm
            ),
            uncertainty=None,
            diagnostics={
                "active_scales": float(
                    self._last_active_scales
                ),
                "total_steps": float(
                    self._last_total_steps
                ),
                "pending_traces": float(
                    len(self._traces)
                ),
                "registered_branches": float(
                    len(
                        self._branch_feature_counts
                    )
                ),
                "parameter_count": float(
                    self._parameter_count()
                ),
                "max_steps_per_window": float(
                    self.max_steps_per_window
                ),
                "training_horizon_minutes": float(
                    self.training_horizon_minutes
                ),
            },
        )

    def snapshot(
        self,
    ) -> Mapping[str, object]:
        state_dict = {
            name: tensor.detach().cpu().tolist()
            for name, tensor
            in self._network.state_dict().items()
        }

        return {
            "version": 1,
            "config": {
                "representation_size": (
                    self.representation_size
                ),
                "hidden_size": self.hidden_size,
                "branch_size": self.branch_size,
                "depth": self.depth,
                "dropout": self.dropout,
                "learning_rate": (
                    self.learning_rate
                ),
                "gradient_clip": (
                    self.gradient_clip
                ),
                "min_steps": self.min_steps,
                "max_steps_per_window": (
                    self.max_steps_per_window
                ),
                "max_traces": self.max_traces,
                "training_horizon_minutes": (
                    self.training_horizon_minutes
                ),
                "seed": self.seed,
            },
            "branch_feature_counts": dict(
                self._branch_feature_counts
            ),
            "state_dict": state_dict,
            "counters": {
                "forward_calls": (
                    self._forward_calls
                ),
                "learning_updates": (
                    self._learning_updates
                ),
                "last_representation_norm": (
                    self._last_representation_norm
                ),
                "last_gradient_norm": (
                    self._last_gradient_norm
                ),
            },
        }

    def restore(
        self,
        snapshot: Mapping[str, object],
    ) -> None:
        version = int(
            snapshot.get(
                "version",
                0,
            )
        )

        if version != 1:
            raise ValueError(
                "Unsupported Recurrent snapshot "
                f"version: {version}"
            )

        config = snapshot.get(
            "config"
        )

        if not isinstance(
            config,
            Mapping,
        ):
            raise ValueError(
                "Recurrent snapshot has invalid config"
            )

        self._validate_snapshot_config(
            config
        )

        raw_branch_counts = snapshot.get(
            "branch_feature_counts",
            {},
        )

        if not isinstance(
            raw_branch_counts,
            Mapping,
        ):
            raise ValueError(
                "Recurrent snapshot has invalid "
                "branch_feature_counts"
            )

        branch_counts = {
            str(scale): int(
                feature_count
            )
            for scale, feature_count
            in raw_branch_counts.items()
        }

        for scale in branch_counts:
            if scale not in _SCALE_NAMES:
                raise ValueError(
                    "Unknown Recurrent scale "
                    f"in snapshot: {scale!r}"
                )

        self._network = self._new_network().to(
            self.device
        )

        self._branch_feature_counts = {}

        for scale in _SCALE_NAMES:
            feature_count = branch_counts.get(
                scale
            )

            if feature_count is None:
                continue

            self._ensure_branch(
                scale,
                feature_count=feature_count,
            )

        raw_state_dict = snapshot.get(
            "state_dict"
        )

        if not isinstance(
            raw_state_dict,
            Mapping,
        ):
            raise ValueError(
                "Recurrent snapshot has invalid state_dict"
            )

        current_state = self._network.state_dict()

        restored_state: dict[
            str,
            Tensor,
        ] = {}

        for name, current_tensor in (
            current_state.items()
        ):
            if name not in raw_state_dict:
                raise ValueError(
                    "Recurrent snapshot missing "
                    f"parameter {name!r}"
                )

            restored = torch.tensor(
                raw_state_dict[name],
                dtype=current_tensor.dtype,
                device=self.device,
            )

            if (
                restored.shape
                != current_tensor.shape
            ):
                raise ValueError(
                    "Recurrent snapshot shape mismatch "
                    f"for {name!r}: expected "
                    f"{tuple(current_tensor.shape)}, "
                    f"got {tuple(restored.shape)}"
                )

            restored_state[name] = restored

        extra_keys = (
            set(raw_state_dict)
            - set(current_state)
        )

        if extra_keys:
            raise ValueError(
                "Recurrent snapshot contains "
                "unexpected parameters: "
                + ", ".join(
                    sorted(
                        str(key)
                        for key in extra_keys
                    )
                )
            )

        self._network.load_state_dict(
            restored_state,
            strict=True,
        )

        counters = snapshot.get(
            "counters",
            {},
        )

        if not isinstance(
            counters,
            Mapping,
        ):
            raise ValueError(
                "Recurrent snapshot has invalid counters"
            )

        self._forward_calls = int(
            counters.get(
                "forward_calls",
                0,
            )
        )

        self._learning_updates = int(
            counters.get(
                "learning_updates",
                0,
            )
        )

        self._last_representation_norm = float(
            counters.get(
                "last_representation_norm",
                0.0,
            )
        )

        self._last_gradient_norm = float(
            counters.get(
                "last_gradient_norm",
                0.0,
            )
        )

        self._last_available = False
        self._last_active_scales = 0
        self._last_total_steps = 0

        self._traces.clear()

    def reset(self) -> None:
        branch_counts = dict(
            self._branch_feature_counts
        )

        self._network = self._new_network().to(
            self.device
        )

        self._branch_feature_counts = {}

        for scale in _SCALE_NAMES:
            feature_count = branch_counts.get(
                scale
            )

            if feature_count is not None:
                self._ensure_branch(
                    scale,
                    feature_count=feature_count,
                )

        self._traces.clear()

        self._forward_calls = 0
        self._learning_updates = 0

        self._last_representation_norm = 0.0
        self._last_gradient_norm = 0.0
        self._last_available = False
        self._last_active_scales = 0
        self._last_total_steps = 0

    # ========================================================
    # MarketInput preparation
    # ========================================================

    def _extract_windows(
        self,
        market_input: MarketInput,
    ) -> tuple[
        _WindowTrace,
        ...,
    ]:
        result: list[
            _WindowTrace
        ] = []

        for scale in _SCALE_NAMES:
            window = getattr(
                market_input,
                scale,
            )

            if window is None:
                continue

            trace = self._window_to_trace(
                scale,
                window,
            )

            if trace is not None:
                result.append(
                    trace
                )

        return tuple(
            result
        )

    def _window_to_trace(
        self,
        scale: str,
        window: TemporalFeatureWindow,
    ) -> _WindowTrace | None:
        if len(
            window.rows
        ) < self.min_steps:
            return None

        feature_count = len(
            window.feature_names
        )

        if feature_count <= 0:
            return None

        raw_rows = window.rows[
            -self.max_steps_per_window :
        ]

        raw = torch.tensor(
            raw_rows,
            dtype=torch.float32,
        )

        if raw.ndim != 2:
            raise ValueError(
                f"Recurrent window {scale!r} "
                "is not 2-dimensional"
            )

        if raw.shape[1] != feature_count:
            raise ValueError(
                f"Recurrent window {scale!r} "
                "feature width mismatch"
            )

        if not torch.isfinite(
            raw
        ).all():
            raise ValueError(
                f"Recurrent window {scale!r} "
                "contains non-finite values"
            )

        mean = raw.mean(
            dim=0,
            keepdim=True,
        )

        std = raw.std(
            dim=0,
            unbiased=False,
            keepdim=True,
        ).clamp_min(
            1e-6
        )

        normalized = (
            (raw - mean) / std
        ).clamp(
            min=-8.0,
            max=8.0,
        )

        rows = tuple(
            tuple(
                float(value)
                for value in row
            )
            for row
            in normalized.tolist()
        )

        return _WindowTrace(
            scale=scale,
            feature_count=feature_count,
            rows=rows,
        )

    def _trace_windows_to_tensors(
        self,
        windows: tuple[
            _WindowTrace,
            ...,
        ],
    ) -> dict[
        str,
        Tensor,
    ]:
        result: dict[
            str,
            Tensor,
        ] = {}

        for window in windows:
            result[window.scale] = torch.tensor(
                window.rows,
                dtype=torch.float32,
                device=self.device,
            )

        return result

    # ========================================================
    # Dynamic branches
    # ========================================================

    def _ensure_branches(
        self,
        windows: tuple[
            _WindowTrace,
            ...,
        ],
    ) -> None:
        for window in windows:
            self._ensure_branch(
                window.scale,
                feature_count=(
                    window.feature_count
                ),
            )

    def _ensure_branch(
        self,
        scale: str,
        *,
        feature_count: int,
    ) -> None:
        existing = (
            self._branch_feature_counts.get(
                scale
            )
        )

        if (
            existing is not None
            and existing != feature_count
        ):
            raise ValueError(
                f"Recurrent feature width for "
                f"{scale!r} changed from "
                f"{existing} to {feature_count}. "
                "MarketInput feature layout must "
                "remain stable."
            )

        branch_seed = (
            self.seed
            + _SCALE_SEED_OFFSETS[scale]
            + feature_count * 1021
        )

        self._network.ensure_branch(
            scale,
            input_features=(
                feature_count
            ),
            seed=branch_seed,
            device=self.device,
        )

        self._branch_feature_counts[
            scale
        ] = feature_count

    # ========================================================
    # Trace management
    # ========================================================

    def _store_trace(
        self,
        trace: _RecurrentTrace,
    ) -> None:
        self._traces.pop(
            trace.transition_id,
            None,
        )

        self._traces[
            trace.transition_id
        ] = trace

        while (
            len(self._traces)
            > self.max_traces
        ):
            self._traces.popitem(
                last=False
            )

    # ========================================================
    # Network lifecycle
    # ========================================================

    def _new_network(
        self,
    ) -> _RecurrentNetwork:
        with torch.random.fork_rng(
            devices=[]
        ):
            torch.manual_seed(
                self.seed
            )

            return _RecurrentNetwork(
                representation_size=(
                    self.representation_size
                ),
                hidden_size=(
                    self.hidden_size
                ),
                branch_size=(
                    self.branch_size
                ),
                depth=self.depth,
                dropout=self.dropout,
            )

    def _zero_grad(
        self,
    ) -> None:
        for parameter in (
            self._network.parameters()
        ):
            parameter.grad = None

    # ========================================================
    # Snapshot validation
    # ========================================================

    def _validate_snapshot_config(
        self,
        config: Mapping[
            str,
            object,
        ],
    ) -> None:
        expected_ints = {
            "representation_size": (
                self.representation_size
            ),
            "hidden_size": (
                self.hidden_size
            ),
            "branch_size": (
                self.branch_size
            ),
            "depth": self.depth,
            "min_steps": (
                self.min_steps
            ),
            "max_steps_per_window": (
                self.max_steps_per_window
            ),
            "training_horizon_minutes": (
                self.training_horizon_minutes
            ),
        }

        for name, expected in (
            expected_ints.items()
        ):
            actual = int(
                config[name]
            )

            if actual != expected:
                raise ValueError(
                    f"Recurrent snapshot {name} mismatch: "
                    f"expected {expected}, got {actual}"
                )

        if float(
            config["dropout"]
        ) != self.dropout:
            raise ValueError(
                "Recurrent snapshot dropout mismatch"
            )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _resolve_device(
        device: (
            str
            | torch.device
            | None
        ),
    ) -> torch.device:
        if device is not None:
            return torch.device(
                device
            )

        if torch.cuda.is_available():
            # ROCm-backed PyTorch also uses the "cuda" API.
            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    @staticmethod
    def _tensor_to_tuple(
        tensor: Tensor,
    ) -> tuple[
        float,
        ...,
    ]:
        detached = (
            tensor.detach().cpu()
        )

        if detached.ndim != 1:
            raise RuntimeError(
                "Recurrent representation must be "
                "one-dimensional"
            )

        values = tuple(
            float(value)
            for value
            in detached.tolist()
        )

        for value in values:
            if not isfinite(
                value
            ):
                raise FloatingPointError(
                    "Recurrent expert produced "
                    "non-finite representation"
                )

        return values

    @staticmethod
    def _norm(
        values: tuple[
            float,
            ...,
        ],
    ) -> float:
        return sqrt(
            sum(
                value * value
                for value in values
            )
        )

    def _parameter_count(
        self,
    ) -> int:
        return sum(
            parameter.numel()
            for parameter
            in self._network.parameters()
        )
