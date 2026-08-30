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
        "SSMMarketExpert requires PyTorch. "
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
    "micro": 113,
    "short": 227,
    "medium": 331,
    "long": 443,
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
class _SSMTrace:
    transition_id: str
    windows: tuple[_WindowTrace, ...]


# ============================================================
# Selective diagonal state-space block
# ============================================================


class _SelectiveDiagonalSSMLayer(nn.Module):
    """
    Lightweight selective diagonal state-space layer.

    This is not a wrapper around Mamba and does not require a custom CUDA
    kernel. It implements the core useful idea directly:

        input_t
            ↓
        learned content projection
            ↓
        learned positive step size Δ_t
            ↓
        stable diagonal decay A
            ↓
        recurrent state update
            ↓
        learned input/output gates
            ↓
        residual output

    For each hidden dimension i:

        decay_t[i] = exp(-softplus(A_log[i]) * softplus(delta_t[i]))

        state_t[i] =
            decay_t[i] * state_{t-1}[i]
            + (1 - decay_t[i]) * candidate_t[i] * input_gate_t[i]

        output_t =
            state_t * output_gate_t + residual_t

    The decay is always in (0, 1], so the recurrent dynamics are stable by
    construction while still being input-selective through delta_t.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()

        self.hidden_size = int(hidden_size)

        self.candidate = nn.Linear(hidden_size, hidden_size)
        self.delta = nn.Linear(hidden_size, hidden_size)
        self.input_gate = nn.Linear(hidden_size, hidden_size)
        self.output_gate = nn.Linear(hidden_size, hidden_size)
        self.residual = nn.Linear(hidden_size, hidden_size)

        # Positive continuous-time decay magnitude after softplus.
        self.a_log = nn.Parameter(
            torch.zeros(hidden_size)
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor) -> Tensor:
        """
        sequence:
            [time, hidden]

        returns:
            [time, hidden]
        """

        if sequence.ndim != 2:
            raise ValueError(
                "_SelectiveDiagonalSSMLayer expects [time, hidden]"
            )

        if sequence.shape[1] != self.hidden_size:
            raise ValueError(
                "SSM hidden width changed unexpectedly"
            )

        time_steps = sequence.shape[0]

        if time_steps <= 0:
            raise ValueError(
                "SSM cannot process an empty sequence"
            )

        a = F.softplus(self.a_log) + 1e-4

        state = torch.zeros(
            self.hidden_size,
            dtype=sequence.dtype,
            device=sequence.device,
        )

        outputs: list[Tensor] = []

        for index in range(time_steps):
            x = sequence[index]

            candidate = torch.tanh(
                self.candidate(x)
            )

            delta = F.softplus(
                self.delta(x)
            ) + 1e-4

            decay = torch.exp(
                -a * delta
            )

            input_gate = torch.sigmoid(
                self.input_gate(x)
            )

            output_gate = torch.sigmoid(
                self.output_gate(x)
            )

            state = (
                decay * state
                + (1.0 - decay)
                * candidate
                * input_gate
            )

            y = (
                state * output_gate
                + self.residual(x)
            )

            outputs.append(y)

        stacked = torch.stack(
            outputs,
            dim=0,
        )

        stacked = self.norm(stacked)
        stacked = F.silu(stacked)
        stacked = self.dropout(stacked)

        return sequence + stacked


class _TemporalSSMBranch(nn.Module):
    """
    One recurrent SSM branch for one Market temporal scale.
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

        self.input_features = int(input_features)
        self.hidden_size = int(hidden_size)

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
                _SelectiveDiagonalSSMLayer(
                    hidden_size,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # Last/mean/max exposes both persistent final state and broader
        # temporal activation structure to the expert representation.
        self.output_projection = nn.Sequential(
            nn.Linear(
                hidden_size * 3,
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
                "_TemporalSSMBranch expects [time, features]"
            )

        if rows.shape[1] != self.input_features:
            raise ValueError(
                "SSM branch feature count changed unexpectedly"
            )

        x = self.input_projection(rows)

        for layer in self.layers:
            x = layer(x)

        last_state = x[-1]
        mean_state = x.mean(dim=0)
        max_state = x.amax(dim=0)

        pooled = torch.cat(
            (
                last_state,
                mean_state,
                max_state,
            ),
            dim=0,
        )

        return self.output_projection(
            pooled
        )


class _SSMNetwork(nn.Module):
    """
    Multi-timescale selective diagonal SSM.

    Branches are lazy because Market normalization owns each temporal
    window's feature width.
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
        self.hidden_size = int(hidden_size)
        self.branch_size = int(branch_size)
        self.depth = int(depth)
        self.dropout = float(dropout)

        self.branches = nn.ModuleDict()

        fusion_input_size = (
            len(_SCALE_NAMES) * branch_size
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
                _TemporalSSMBranch,
            ):
                raise RuntimeError(
                    f"Unexpected SSM branch type for "
                    f"scale {scale!r}"
                )

            if (
                existing.input_features
                != input_features
            ):
                raise ValueError(
                    f"SSM feature width for {scale!r} "
                    f"changed from "
                    f"{existing.input_features} "
                    f"to {input_features}. "
                    "Keep MarketInput feature layout stable "
                    "or reset/reconfigure the expert."
                )

            return

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)

            branch = _TemporalSSMBranch(
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
            rows = windows.get(scale)

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

            branch = self.branches[scale]

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
# SSM Market expert
# ============================================================


class SSMMarketExpert(MarketExpert):
    """
    Trainable multi-timescale selective state-space Market expert.

    Compared with TCN, this expert is intended to specialize in persistent
    temporal dynamics and longer sequential state.

    It uses a stable selective diagonal recurrence rather than a hard-coded
    market indicator.

    `training_horizon_minutes=240` is Market-specific experiment metadata.
    It is not a forced holding period and is not a shared Heiwa parameter.

    Delayed learning uses exact transition input replay. We cache normalized
    temporal windows under transition_id and never substitute current Market
    state when an old consequence arrives.
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
        max_traces: int = 512,
        training_horizon_minutes: int = 240,
        device: str | torch.device | None = None,
        seed: int = 20260828,
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

        self.hidden_size = int(hidden_size)
        self.branch_size = int(branch_size)
        self.depth = int(depth)
        self.dropout = float(dropout)

        self.learning_rate = float(
            learning_rate
        )
        self.gradient_clip = float(
            gradient_clip
        )

        self.min_steps = int(min_steps)
        self.max_traces = int(max_traces)
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
            _SSMTrace,
        ] = OrderedDict()

        self._forward_calls = 0
        self._learning_updates = 0

        self._last_representation_norm = 0.0
        self._last_gradient_norm = 0.0
        self._last_available = False
        self._last_active_scales = 0

    @property
    def name(self) -> str:
        return "ssm"

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
            representation_tensor = (
                self._network(
                    tensor_windows
                )
            )

        representation = self._tensor_to_tuple(
            representation_tensor
        )

        self._store_trace(
            _SSMTrace(
                transition_id=transition_id,
                windows=trace_windows,
            )
        )

        self._last_available = True
        self._last_representation_norm = (
            self._norm(representation)
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
            for value in representation_gradient
        )

        if (
            len(gradient)
            != self.representation_size
        ):
            raise ValueError(
                "SSM representation gradient size "
                "mismatch: "
                f"expected "
                f"{self.representation_size}, "
                f"got {len(gradient)}"
            )

        for value in gradient:
            if not isfinite(value):
                raise ValueError(
                    "SSM representation gradient "
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

        # This local surrogate yields exactly the requested downstream
        # representation gradient.
        local_loss = torch.dot(
            representation,
            gradient_tensor.detach(),
        )

        local_loss.backward()

        parameters = tuple(
            parameter
            for parameter in self._network.parameters()
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
            snapshot.get("version", 0)
        )

        if version != 1:
            raise ValueError(
                "Unsupported SSM snapshot "
                f"version: {version}"
            )

        config = snapshot.get("config")

        if not isinstance(config, Mapping):
            raise ValueError(
                "SSM snapshot has invalid config"
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
                "SSM snapshot has invalid "
                "branch_feature_counts"
            )

        branch_counts = {
            str(scale): int(feature_count)
            for scale, feature_count
            in raw_branch_counts.items()
        }

        for scale in branch_counts:
            if scale not in _SCALE_NAMES:
                raise ValueError(
                    "Unknown SSM scale in snapshot: "
                    f"{scale!r}"
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
                "SSM snapshot has invalid state_dict"
            )

        current_state = (
            self._network.state_dict()
        )
        restored_state: dict[str, Tensor] = {}

        for name, current_tensor in (
            current_state.items()
        ):
            if name not in raw_state_dict:
                raise ValueError(
                    "SSM snapshot missing "
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
                    "SSM snapshot shape mismatch "
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
                "SSM snapshot contains unexpected "
                "parameters: "
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
                "SSM snapshot has invalid counters"
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

        # Old delayed-learning traces are never valid after restore.
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

    # ========================================================
    # MarketInput preparation
    # ========================================================

    def _extract_windows(
        self,
        market_input: MarketInput,
    ) -> tuple[_WindowTrace, ...]:
        result: list[_WindowTrace] = []

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
                result.append(trace)

        return tuple(result)

    def _window_to_trace(
        self,
        scale: str,
        window: TemporalFeatureWindow,
    ) -> _WindowTrace | None:
        if (
            len(window.rows)
            < self.min_steps
        ):
            return None

        feature_count = len(
            window.feature_names
        )

        if feature_count <= 0:
            return None

        raw = torch.tensor(
            window.rows,
            dtype=torch.float32,
        )

        if raw.ndim != 2:
            raise ValueError(
                f"SSM window {scale!r} "
                "is not 2-dimensional"
            )

        if raw.shape[1] != feature_count:
            raise ValueError(
                f"SSM window {scale!r} "
                "feature width mismatch"
            )

        if not torch.isfinite(raw).all():
            raise ValueError(
                f"SSM window {scale!r} "
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
        ).clamp_min(1e-6)

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
    ) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}

        for window in windows:
            result[window.scale] = (
                torch.tensor(
                    window.rows,
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        return result

    # ========================================================
    # Dynamic scale branches
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
                f"SSM feature width for {scale!r} "
                f"changed from {existing} "
                f"to {feature_count}. "
                "MarketInput feature layout must "
                "remain stable."
            )

        branch_seed = (
            self.seed
            + _SCALE_SEED_OFFSETS[scale]
            + feature_count * 1013
        )

        self._network.ensure_branch(
            scale,
            input_features=feature_count,
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
        trace: _SSMTrace,
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
    ) -> _SSMNetwork:
        with torch.random.fork_rng(
            devices=[]
        ):
            torch.manual_seed(
                self.seed
            )

            return _SSMNetwork(
                representation_size=(
                    self.representation_size
                ),
                hidden_size=self.hidden_size,
                branch_size=self.branch_size,
                depth=self.depth,
                dropout=self.dropout,
            )

    def _zero_grad(self) -> None:
        for parameter in (
            self._network.parameters()
        ):
            parameter.grad = None

    # ========================================================
    # Snapshot validation
    # ========================================================

    def _validate_snapshot_config(
        self,
        config: Mapping[str, object],
    ) -> None:
        expected_ints = {
            "representation_size": (
                self.representation_size
            ),
            "hidden_size": self.hidden_size,
            "branch_size": self.branch_size,
            "depth": self.depth,
            "min_steps": self.min_steps,
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
                    f"SSM snapshot {name} mismatch: "
                    f"expected {expected}, "
                    f"got {actual}"
                )

        if float(
            config["dropout"]
        ) != self.dropout:
            raise ValueError(
                "SSM snapshot dropout mismatch"
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
            return torch.device(device)

        if torch.cuda.is_available():
            # ROCm builds also expose AMD GPUs through the "cuda" device
            # API in PyTorch.
            return torch.device("cuda")

        return torch.device("cpu")

    @staticmethod
    def _tensor_to_tuple(
        tensor: Tensor,
    ) -> tuple[float, ...]:
        detached = (
            tensor.detach().cpu()
        )

        if detached.ndim != 1:
            raise RuntimeError(
                "SSM representation must be "
                "one-dimensional"
            )

        values = tuple(
            float(value)
            for value
            in detached.tolist()
        )

        for value in values:
            if not isfinite(value):
                raise FloatingPointError(
                    "SSM produced non-finite "
                    "representation"
                )

        return values

    @staticmethod
    def _norm(
        values: tuple[float, ...],
    ) -> float:
        return sqrt(
            sum(
                value * value
                for value in values
            )
        )

    def _parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter
            in self._network.parameters()
        )
