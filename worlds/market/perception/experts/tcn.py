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
        "TCNMarketExpert requires PyTorch. "
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
    "micro": 101,
    "short": 211,
    "medium": 307,
    "long": 401,
}


# ============================================================
# Causal trace
# ============================================================


@dataclass(frozen=True, slots=True)
class _WindowTrace:
    """
    Compact replayable input for one temporal scale.

    We deliberately keep normalized input data rather than an autograd graph.

    Delayed Market consequences can arrive after many newer parameter
    updates. Keeping an old PyTorch graph across those updates is both
    memory-heavy and invalid once parameters change in-place.

    Therefore learning replays the exact transition input through the
    current TCN parameters and applies the downstream representation
    gradient to that replay.

    Causal attribution remains tied to the exact transition_id and exact
    observed input; no current Market state is substituted.
    """

    scale: str
    feature_count: int
    rows: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class _TCNTrace:
    transition_id: str
    windows: tuple[_WindowTrace, ...]


# ============================================================
# Neural building blocks
# ============================================================


class _CausalConv1d(nn.Module):
    """
    Left-padded 1D convolution.

    Output at time t never depends on a future sequence element.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()

        if kernel_size <= 1:
            raise ValueError("kernel_size must be > 1")

        if dilation <= 0:
            raise ValueError("dilation must be > 0")

        self.left_padding = dilation * (kernel_size - 1)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(
            x,
            (self.left_padding, 0),
        )
        return self.conv(x)


class _ResidualTCNBlock(nn.Module):
    """
    Residual dilated causal TCN block.

    GroupNorm is used instead of BatchNorm so behavior does not depend on
    batch statistics. Heiwa may often encode one live transition at a time.
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv1 = _CausalConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(1, channels)

        self.conv2 = _CausalConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(1, channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        y = self.conv1(x)
        y = self.norm1(y)
        y = F.silu(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = F.silu(y)
        y = self.dropout(y)

        return residual + y


class _TemporalScaleBranch(nn.Module):
    """
    One TCN branch for one Market temporal scale.

    Different Market windows may expose different feature widths, so each
    scale owns a separate input projection while producing the same
    branch_size representation.
    """

    def __init__(
        self,
        *,
        input_features: int,
        channels: int,
        branch_size: int,
        depth: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.input_features = int(input_features)

        self.input_projection = nn.Conv1d(
            input_features,
            channels,
            kernel_size=1,
        )

        blocks: list[nn.Module] = []

        for index in range(depth):
            blocks.append(
                _ResidualTCNBlock(
                    channels,
                    kernel_size=kernel_size,
                    dilation=2**index,
                    dropout=dropout,
                )
            )

        self.blocks = nn.Sequential(*blocks)

        # Last/mean/max pooling lets the branch retain:
        # - the most recent causal state;
        # - persistent average context;
        # - strong local activations.
        self.output_projection = nn.Sequential(
            nn.Linear(channels * 3, branch_size * 2),
            nn.SiLU(),
            nn.Linear(branch_size * 2, branch_size),
            nn.Tanh(),
        )

    def forward(self, rows: Tensor) -> Tensor:
        """
        rows:
            [time, features]

        returns:
            [branch_size]
        """

        if rows.ndim != 2:
            raise ValueError(
                "_TemporalScaleBranch expects [time, features]"
            )

        if rows.shape[1] != self.input_features:
            raise ValueError(
                "Temporal branch feature count changed unexpectedly"
            )

        # Conv1d expects [batch, channels, time].
        x = rows.transpose(0, 1).unsqueeze(0)

        x = self.input_projection(x)
        x = self.blocks(x)

        last_state = x[:, :, -1]
        mean_state = x.mean(dim=-1)
        max_state = x.amax(dim=-1)

        pooled = torch.cat(
            (last_state, mean_state, max_state),
            dim=1,
        )

        return self.output_projection(pooled).squeeze(0)


class _TCNNetwork(nn.Module):
    """
    Multi-timescale TCN expert network.

    Scale branches are created lazily because the normalized feature width of
    each Market window is owned by MarketInput preparation, not hardcoded
    here.
    """

    def __init__(
        self,
        *,
        representation_size: int,
        channels: int,
        branch_size: int,
        depth: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.representation_size = int(representation_size)
        self.channels = int(channels)
        self.branch_size = int(branch_size)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)
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
        existing = self.branches[scale] if scale in self.branches else None

        if existing is not None:
            if not isinstance(existing, _TemporalScaleBranch):
                raise RuntimeError(
                    f"Unexpected TCN branch type for scale {scale!r}"
                )

            if existing.input_features != input_features:
                raise ValueError(
                    f"TCN feature width for {scale!r} changed from "
                    f"{existing.input_features} to {input_features}. "
                    "Keep MarketInput feature layout stable or reset/"
                    "reconfigure the expert."
                )

            return

        # Isolate module initialization from the process-global CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)

            branch = _TemporalScaleBranch(
                input_features=input_features,
                channels=self.channels,
                branch_size=self.branch_size,
                depth=self.depth,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            )

        self.branches[scale] = branch.to(device)

    def forward(
        self,
        windows: Mapping[str, Tensor],
    ) -> Tensor:
        branch_vectors: list[Tensor] = []
        availability: list[Tensor] = []

        reference_device = next(self.fusion.parameters()).device
        reference_dtype = next(self.fusion.parameters()).dtype

        for scale in _SCALE_NAMES:
            rows = windows.get(scale)

            if rows is None:
                branch_vectors.append(
                    torch.zeros(
                        self.branch_size,
                        device=reference_device,
                        dtype=reference_dtype,
                    )
                )
                availability.append(
                    torch.zeros(
                        1,
                        device=reference_device,
                        dtype=reference_dtype,
                    )
                )
                continue

            branch = self.branches[scale]
            vector = branch(rows)

            branch_vectors.append(vector)
            availability.append(
                torch.ones(
                    1,
                    device=reference_device,
                    dtype=reference_dtype,
                )
            )

        fused_input = torch.cat(
            (*branch_vectors, *availability),
            dim=0,
        )

        return self.fusion(fused_input)


# ============================================================
# TCNMarketExpert
# ============================================================


class TCNMarketExpert(MarketExpert):
    """
    Trainable multi-timescale causal TCN Market expert.

    The expert consumes normalized temporal windows from MarketInput and
    returns a learned representation.

    It does NOT:
        - call T-Bank;
        - choose BUY/SELL/HOLD;
        - own portfolio policy;
        - access Core internals;
        - decide when a 240-minute position must close.

    `training_horizon_minutes` records the Market-specific prediction/
    training horizon around which downstream training targets may later be
    constructed. It is not a holding timer and is not a shared Heiwa
    contract.

    Delayed learning uses transition-exact input replay:
        encode transition X
            -> cache normalized temporal input for X

        later downstream gradient for X
            -> replay exact cached input X
            -> backprop representation gradient
            -> update TCN

    No current Market state is used as a substitute for a missing trace.
    """

    def __init__(
        self,
        *,
        representation_size: int = 64,
        channels: int = 64,
        branch_size: int = 48,
        depth: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.05,
        learning_rate: float = 1e-3,
        gradient_clip: float = 5.0,
        min_steps: int = 8,
        max_traces: int = 512,
        training_horizon_minutes: int = 240,
        device: str | torch.device | None = None,
        seed: int = 20260827,
    ) -> None:
        if representation_size <= 0:
            raise ValueError("representation_size must be > 0")

        if channels <= 0:
            raise ValueError("channels must be > 0")

        if branch_size <= 0:
            raise ValueError("branch_size must be > 0")

        if depth <= 0:
            raise ValueError("depth must be > 0")

        if kernel_size <= 1:
            raise ValueError("kernel_size must be > 1")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")

        if gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be > 0")

        if min_steps <= 0:
            raise ValueError("min_steps must be > 0")

        if max_traces <= 0:
            raise ValueError("max_traces must be > 0")

        if training_horizon_minutes <= 0:
            raise ValueError(
                "training_horizon_minutes must be > 0"
            )

        self._representation_size = int(representation_size)

        self.channels = int(channels)
        self.branch_size = int(branch_size)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)

        self.learning_rate = float(learning_rate)
        self.gradient_clip = float(gradient_clip)

        self.min_steps = int(min_steps)
        self.max_traces = int(max_traces)
        self.training_horizon_minutes = int(
            training_horizon_minutes
        )

        self.seed = int(seed)
        self.device = self._resolve_device(device)

        self._network = self._new_network().to(self.device)

        self._branch_feature_counts: dict[str, int] = {}

        self._traces: OrderedDict[str, _TCNTrace] = OrderedDict()

        self._forward_calls = 0
        self._learning_updates = 0

        self._last_representation_norm = 0.0
        self._last_gradient_norm = 0.0
        self._last_available = False
        self._last_active_scales = 0

    # ========================================================
    # MarketExpert contract
    # ========================================================

    @property
    def name(self) -> str:
        return "tcn"

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
            raise ValueError("transition_id must not be empty")

        trace_windows = self._extract_windows(market_input)

        self._forward_calls += 1
        self._last_active_scales = len(trace_windows)

        if not trace_windows:
            self._last_available = False
            self._last_representation_norm = 0.0
            self._traces.pop(transition_id, None)

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

        self._ensure_branches(trace_windows)

        tensor_windows = self._trace_windows_to_tensors(
            trace_windows
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
            _TCNTrace(
                transition_id=transition_id,
                windows=trace_windows,
            )
        )

        self._last_available = True
        self._last_representation_norm = self._norm(
            representation
        )

        return ExpertOutput(
            expert_name=self.name,
            transition_id=transition_id,
            representation=representation,
            available=True,
            uncertainty=None,
            auxiliary={},
            diagnostics={
                "active_scales": float(len(trace_windows)),
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
        representation_gradient: tuple[float, ...],
    ) -> bool:
        """
        Apply downstream causal credit to the exact cached transition input.

        `representation_gradient` is interpreted as:

            dLoss / dExpertRepresentation

        We replay the exact transition input through current TCN parameters,
        then use a local scalar surrogate:

            loss_local = dot(
                current_representation,
                representation_gradient,
            )

        whose derivative produces the requested representation gradient.
        """

        trace = self._traces.pop(transition_id, None)

        if trace is None:
            return False

        gradient = tuple(
            float(value)
            for value in representation_gradient
        )

        if len(gradient) != self.representation_size:
            raise ValueError(
                "TCN representation gradient size mismatch: "
                f"expected {self.representation_size}, "
                f"got {len(gradient)}"
            )

        for value in gradient:
            if not isfinite(value):
                raise ValueError(
                    "TCN representation gradient contains "
                    "non-finite value"
                )

        self._ensure_branches(trace.windows)

        tensor_windows = self._trace_windows_to_tensors(
            trace.windows
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

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=self.gradient_clip,
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

    def telemetry(self) -> ExpertTelemetry:
        return ExpertTelemetry(
            expert_name=self.name,
            available=self._last_available,
            forward_calls=self._forward_calls,
            learning_updates=self._learning_updates,
            last_representation_norm=(
                self._last_representation_norm
            ),
            last_gradient_norm=self._last_gradient_norm,
            uncertainty=None,
            diagnostics={
                "active_scales": float(
                    self._last_active_scales
                ),
                "pending_traces": float(
                    len(self._traces)
                ),
                "registered_branches": float(
                    len(self._branch_feature_counts)
                ),
                "parameter_count": float(
                    self._parameter_count()
                ),
                "training_horizon_minutes": float(
                    self.training_horizon_minutes
                ),
            },
        )

    def snapshot(self) -> Mapping[str, object]:
        """
        Return serializable learned TCN state.

        Pending causal traces are transient and intentionally not persisted.
        """

        state_dict = {
            name: tensor.detach().cpu().tolist()
            for name, tensor in self._network.state_dict().items()
        }

        return {
            "version": 1,
            "config": {
                "representation_size": self.representation_size,
                "channels": self.channels,
                "branch_size": self.branch_size,
                "depth": self.depth,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "gradient_clip": self.gradient_clip,
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
                "forward_calls": self._forward_calls,
                "learning_updates": self._learning_updates,
                "last_representation_norm": (
                    self._last_representation_norm
                ),
                "last_gradient_norm": self._last_gradient_norm,
            },
        }

    def restore(
        self,
        snapshot: Mapping[str, object],
    ) -> None:
        version = int(snapshot.get("version", 0))

        if version != 1:
            raise ValueError(
                f"Unsupported TCN snapshot version: {version}"
            )

        config = snapshot.get("config")

        if not isinstance(config, Mapping):
            raise ValueError(
                "TCN snapshot has invalid config"
            )

        self._validate_snapshot_config(config)

        raw_branch_counts = snapshot.get(
            "branch_feature_counts",
            {},
        )

        if not isinstance(raw_branch_counts, Mapping):
            raise ValueError(
                "TCN snapshot has invalid branch_feature_counts"
            )

        branch_counts = {
            str(scale): int(feature_count)
            for scale, feature_count in raw_branch_counts.items()
        }

        for scale in branch_counts:
            if scale not in _SCALE_NAMES:
                raise ValueError(
                    f"Unknown TCN scale in snapshot: {scale!r}"
                )

        # Rebuild architecture before loading dynamically-created branches.
        self._network = self._new_network().to(self.device)
        self._branch_feature_counts = {}

        for scale in _SCALE_NAMES:
            feature_count = branch_counts.get(scale)

            if feature_count is None:
                continue

            self._ensure_branch(
                scale,
                feature_count=feature_count,
            )

        raw_state_dict = snapshot.get("state_dict")

        if not isinstance(raw_state_dict, Mapping):
            raise ValueError(
                "TCN snapshot has invalid state_dict"
            )

        current_state = self._network.state_dict()

        restored_state: dict[str, Tensor] = {}

        for name, current_tensor in current_state.items():
            if name not in raw_state_dict:
                raise ValueError(
                    f"TCN snapshot missing parameter {name!r}"
                )

            restored = torch.tensor(
                raw_state_dict[name],
                dtype=current_tensor.dtype,
                device=self.device,
            )

            if restored.shape != current_tensor.shape:
                raise ValueError(
                    f"TCN snapshot shape mismatch for {name!r}: "
                    f"expected {tuple(current_tensor.shape)}, "
                    f"got {tuple(restored.shape)}"
                )

            restored_state[name] = restored

        extra_keys = (
            set(raw_state_dict)
            - set(current_state)
        )

        if extra_keys:
            raise ValueError(
                "TCN snapshot contains unexpected parameters: "
                + ", ".join(sorted(str(key) for key in extra_keys))
            )

        self._network.load_state_dict(
            restored_state,
            strict=True,
        )

        counters = snapshot.get("counters", {})

        if not isinstance(counters, Mapping):
            raise ValueError(
                "TCN snapshot has invalid counters"
            )

        self._forward_calls = int(
            counters.get("forward_calls", 0)
        )
        self._learning_updates = int(
            counters.get("learning_updates", 0)
        )
        self._last_representation_norm = float(
            counters.get(
                "last_representation_norm",
                0.0,
            )
        )
        self._last_gradient_norm = float(
            counters.get("last_gradient_norm", 0.0)
        )

        self._last_available = False
        self._last_active_scales = 0

        # Never restore stale causal traces.
        self._traces.clear()

    def reset(self) -> None:
        """
        Reset learned TCN parameters while preserving discovered scale shapes.
        """

        branch_counts = dict(
            self._branch_feature_counts
        )

        self._network = self._new_network().to(self.device)
        self._branch_feature_counts = {}

        for scale in _SCALE_NAMES:
            feature_count = branch_counts.get(scale)

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
            window = getattr(market_input, scale)

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
        if len(window.rows) < self.min_steps:
            return None

        feature_count = len(window.feature_names)

        if feature_count <= 0:
            return None

        raw = torch.tensor(
            window.rows,
            dtype=torch.float32,
        )

        if raw.ndim != 2:
            raise ValueError(
                f"TCN window {scale!r} is not 2-dimensional"
            )

        if raw.shape[1] != feature_count:
            raise ValueError(
                f"TCN window {scale!r} feature width mismatch"
            )

        if not torch.isfinite(raw).all():
            raise ValueError(
                f"TCN window {scale!r} contains non-finite values"
            )

        # Per-window feature standardization is mechanical input scaling.
        # It does not inject a trading signal or hand-written indicator.
        mean = raw.mean(dim=0, keepdim=True)
        std = raw.std(
            dim=0,
            unbiased=False,
            keepdim=True,
        ).clamp_min(1e-6)

        normalized = (raw - mean) / std

        # Protect the neural path from rare extreme numeric spikes while
        # keeping their direction and relative magnitude.
        normalized = normalized.clamp(
            min=-8.0,
            max=8.0,
        )

        rows = tuple(
            tuple(float(value) for value in row)
            for row in normalized.tolist()
        )

        return _WindowTrace(
            scale=scale,
            feature_count=feature_count,
            rows=rows,
        )

    def _trace_windows_to_tensors(
        self,
        windows: tuple[_WindowTrace, ...],
    ) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}

        for window in windows:
            result[window.scale] = torch.tensor(
                window.rows,
                dtype=torch.float32,
                device=self.device,
            )

        return result

    # ========================================================
    # Dynamic scale branches
    # ========================================================

    def _ensure_branches(
        self,
        windows: tuple[_WindowTrace, ...],
    ) -> None:
        for window in windows:
            self._ensure_branch(
                window.scale,
                feature_count=window.feature_count,
            )

    def _ensure_branch(
        self,
        scale: str,
        *,
        feature_count: int,
    ) -> None:
        existing = self._branch_feature_counts.get(
            scale
        )

        if existing is not None and existing != feature_count:
            raise ValueError(
                f"TCN feature width for {scale!r} changed from "
                f"{existing} to {feature_count}. "
                "MarketInput feature layout must remain stable."
            )

        branch_seed = (
            self.seed
            + _SCALE_SEED_OFFSETS[scale]
            + feature_count * 1009
        )

        self._network.ensure_branch(
            scale,
            input_features=feature_count,
            seed=branch_seed,
            device=self.device,
        )

        self._branch_feature_counts[scale] = (
            feature_count
        )

    # ========================================================
    # Trace management
    # ========================================================

    def _store_trace(
        self,
        trace: _TCNTrace,
    ) -> None:
        self._traces.pop(
            trace.transition_id,
            None,
        )
        self._traces[trace.transition_id] = trace

        while len(self._traces) > self.max_traces:
            self._traces.popitem(last=False)

    # ========================================================
    # Network lifecycle
    # ========================================================

    def _new_network(self) -> _TCNNetwork:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)

            return _TCNNetwork(
                representation_size=self.representation_size,
                channels=self.channels,
                branch_size=self.branch_size,
                depth=self.depth,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            )

    def _zero_grad(self) -> None:
        for parameter in self._network.parameters():
            parameter.grad = None

    # ========================================================
    # Snapshot validation
    # ========================================================

    def _validate_snapshot_config(
        self,
        config: Mapping[str, object],
    ) -> None:
        expected_ints = {
            "representation_size": self.representation_size,
            "channels": self.channels,
            "branch_size": self.branch_size,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "min_steps": self.min_steps,
            "training_horizon_minutes": (
                self.training_horizon_minutes
            ),
        }

        for name, expected in expected_ints.items():
            actual = int(config[name])

            if actual != expected:
                raise ValueError(
                    f"TCN snapshot {name} mismatch: "
                    f"expected {expected}, got {actual}"
                )

        actual_dropout = float(config["dropout"])

        if actual_dropout != self.dropout:
            raise ValueError(
                "TCN snapshot dropout mismatch"
            )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _resolve_device(
        device: str | torch.device | None,
    ) -> torch.device:
        if device is not None:
            resolved = torch.device(device)
        elif torch.cuda.is_available():
            # ROCm-backed PyTorch also exposes the device through "cuda".
            resolved = torch.device("cuda")
        else:
            resolved = torch.device("cpu")

        return resolved

    @staticmethod
    def _tensor_to_tuple(
        tensor: Tensor,
    ) -> tuple[float, ...]:
        detached = tensor.detach().cpu()

        if detached.ndim != 1:
            raise RuntimeError(
                "TCN representation must be one-dimensional"
            )

        values = tuple(
            float(value)
            for value in detached.tolist()
        )

        for value in values:
            if not isfinite(value):
                raise FloatingPointError(
                    "TCN produced non-finite representation"
                )

        return values

    @staticmethod
    def _norm(
        values: tuple[float, ...],
    ) -> float:
        return sqrt(
            sum(value * value for value in values)
        )

    def _parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self._network.parameters()
        )
