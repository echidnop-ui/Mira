from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite, log, sqrt
from typing import Mapping

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "AttentionMarketExpert requires PyTorch. "
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
    "micro": 127,
    "short": 239,
    "medium": 347,
    "long": 461,
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
class _AttentionTrace:
    transition_id: str
    windows: tuple[_WindowTrace, ...]


# ============================================================
# Attention building blocks
# ============================================================


class _LearnedRelativeTimeBias(nn.Module):
    """
    Small learned relative-distance bias.

    We do not encode absolute wall-clock positions into shared cognition.
    The branch only learns how much relative temporal distance may matter.

    For query i and key j, causal attention allows j <= i. The relative lag
    (i - j) is bucketed logarithmically so long contexts do not require one
    parameter per possible sequence length.
    """

    def __init__(
        self,
        *,
        heads: int,
        buckets: int = 32,
    ) -> None:
        super().__init__()

        if heads <= 0:
            raise ValueError("heads must be > 0")

        if buckets <= 1:
            raise ValueError("buckets must be > 1")

        self.heads = int(heads)
        self.buckets = int(buckets)

        self.bias = nn.Parameter(
            torch.zeros(
                heads,
                buckets,
            )
        )

    def forward(
        self,
        length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if length <= 0:
            raise ValueError(
                "relative attention bias length must be > 0"
            )

        positions = torch.arange(
            length,
            device=device,
        )

        query = positions[:, None]
        key = positions[None, :]

        lag = (query - key).clamp_min(0)

        # Bucket 0 = same step, then increasingly coarse logarithmic ranges.
        bucket = torch.floor(
            torch.log2(
                lag.to(torch.float32) + 1.0
            )
        ).to(torch.long)

        bucket = bucket.clamp_max(
            self.buckets - 1
        )

        # [heads, time, time]
        bias = self.bias[:, bucket]

        return bias.to(dtype=dtype)


class _CausalSelfAttentionBlock(nn.Module):
    """
    Pre-norm causal self-attention block with learned relative time bias.

    This is a compact temporal attention specialist, not an LLM-style giant
    Transformer.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        heads: int,
        feedforward_multiplier: int,
        dropout: float,
        relative_buckets: int,
    ) -> None:
        super().__init__()

        if hidden_size % heads != 0:
            raise ValueError(
                "hidden_size must be divisible by heads"
            )

        self.hidden_size = int(hidden_size)
        self.heads = int(heads)

        self.norm1 = nn.LayerNorm(hidden_size)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.relative_bias = _LearnedRelativeTimeBias(
            heads=heads,
            buckets=relative_buckets,
        )

        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size)

        feedforward_size = (
            hidden_size * feedforward_multiplier
        )

        self.feedforward = nn.Sequential(
            nn.Linear(
                hidden_size,
                feedforward_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                feedforward_size,
                hidden_size,
            ),
        )

        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        sequence: Tensor,
    ) -> Tensor:
        """
        sequence:
            [time, hidden]

        returns:
            [time, hidden]
        """

        if sequence.ndim != 2:
            raise ValueError(
                "_CausalSelfAttentionBlock expects [time, hidden]"
            )

        if sequence.shape[1] != self.hidden_size:
            raise ValueError(
                "Attention hidden width changed unexpectedly"
            )

        length = int(sequence.shape[0])

        if length <= 0:
            raise ValueError(
                "Attention cannot process an empty sequence"
            )

        x = sequence.unsqueeze(0)

        normalized = self.norm1(x)

        # Boolean mask:
        # True means "do not attend".
        causal_mask = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=sequence.device,
            ),
            diagonal=1,
        )

        relative_bias = self.relative_bias(
            length,
            device=sequence.device,
            dtype=sequence.dtype,
        )

        # MultiheadAttention supports a separate mask per
        # batch*head: [batch * heads, target, source].
        #
        # We convert the causal mask into additive -inf and add the learned
        # relative bias.
        additive_mask = torch.zeros(
            self.heads,
            length,
            length,
            dtype=sequence.dtype,
            device=sequence.device,
        )

        additive_mask = additive_mask.masked_fill(
            causal_mask.unsqueeze(0),
            float("-inf"),
        )

        additive_mask = (
            additive_mask + relative_bias
        )

        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=additive_mask,
            need_weights=False,
        )

        x = x + self.dropout1(attended)

        x = x + self.dropout2(
            self.feedforward(
                self.norm2(x)
            )
        )

        return x.squeeze(0)


class _TemporalAttentionBranch(nn.Module):
    """
    One temporal attention branch for one Market scale.
    """

    def __init__(
        self,
        *,
        input_features: int,
        hidden_size: int,
        branch_size: int,
        depth: int,
        heads: int,
        feedforward_multiplier: int,
        dropout: float,
        relative_buckets: int,
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
            nn.GELU(),
        )

        self.blocks = nn.ModuleList(
            [
                _CausalSelfAttentionBlock(
                    hidden_size,
                    heads=heads,
                    feedforward_multiplier=(
                        feedforward_multiplier
                    ),
                    dropout=dropout,
                    relative_buckets=(
                        relative_buckets
                    ),
                )
                for _ in range(depth)
            ]
        )

        self.pool_query = nn.Parameter(
            torch.zeros(hidden_size)
        )

        self.output_projection = nn.Sequential(
            nn.Linear(
                hidden_size * 4,
                branch_size * 2,
            ),
            nn.GELU(),
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
                "_TemporalAttentionBranch expects [time, features]"
            )

        if rows.shape[1] != self.input_features:
            raise ValueError(
                "Attention branch feature count changed unexpectedly"
            )

        x = self.input_projection(rows)

        for block in self.blocks:
            x = block(x)

        last_state = x[-1]
        mean_state = x.mean(dim=0)
        max_state = x.amax(dim=0)

        # Learned summary pooling over the entire causal sequence after all
        # blocks. This gives the branch another way to retain distant
        # evidence without exposing attention internals outside the expert.
        scores = torch.mv(
            x,
            self.pool_query,
        ) / sqrt(
            float(self.hidden_size)
        )

        pool_weights = torch.softmax(
            scores,
            dim=0,
        )

        learned_pool = torch.sum(
            x * pool_weights.unsqueeze(1),
            dim=0,
        )

        pooled = torch.cat(
            (
                last_state,
                mean_state,
                max_state,
                learned_pool,
            ),
            dim=0,
        )

        return self.output_projection(
            pooled
        )


class _AttentionNetwork(nn.Module):
    """
    Multi-timescale causal attention network.

    Each scale has an independent branch because feature widths may differ.
    Final expert representation combines all available scale branches and
    explicit availability bits.
    """

    def __init__(
        self,
        *,
        representation_size: int,
        hidden_size: int,
        branch_size: int,
        depth: int,
        heads: int,
        feedforward_multiplier: int,
        dropout: float,
        relative_buckets: int,
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
        self.heads = int(heads)
        self.feedforward_multiplier = int(
            feedforward_multiplier
        )
        self.dropout = float(dropout)
        self.relative_buckets = int(
            relative_buckets
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
            nn.GELU(),
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
                _TemporalAttentionBranch,
            ):
                raise RuntimeError(
                    f"Unexpected Attention branch type "
                    f"for scale {scale!r}"
                )

            if (
                existing.input_features
                != input_features
            ):
                raise ValueError(
                    f"Attention feature width for "
                    f"{scale!r} changed from "
                    f"{existing.input_features} to "
                    f"{input_features}. Keep MarketInput "
                    "feature layout stable or reset/"
                    "reconfigure the expert."
                )

            return

        with torch.random.fork_rng(
            devices=[]
        ):
            torch.manual_seed(seed)

            branch = _TemporalAttentionBranch(
                input_features=input_features,
                hidden_size=self.hidden_size,
                branch_size=self.branch_size,
                depth=self.depth,
                heads=self.heads,
                feedforward_multiplier=(
                    self.feedforward_multiplier
                ),
                dropout=self.dropout,
                relative_buckets=(
                    self.relative_buckets
                ),
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
# Attention Market expert
# ============================================================


class AttentionMarketExpert(MarketExpert):
    """
    Trainable multi-timescale causal Attention Market expert.

    Intended specialization:
        - non-local temporal relationships;
        - distant event interactions;
        - feature/context relations;
        - selective access to prior sequence elements.

    It is deliberately compact compared with an LLM-style Transformer.

    `training_horizon_minutes=240` remains Market-specific training metadata.
    It is not a forced holding duration and is not part of shared Heiwa.

    Delayed learning uses exact transition input replay:
        transition_id -> normalized temporal windows

    No current Market state is substituted when a delayed gradient arrives.
    """

    def __init__(
        self,
        *,
        representation_size: int = 64,
        hidden_size: int = 64,
        branch_size: int = 48,
        depth: int = 3,
        heads: int = 4,
        feedforward_multiplier: int = 4,
        dropout: float = 0.05,
        relative_buckets: int = 32,
        learning_rate: float = 1e-3,
        gradient_clip: float = 5.0,
        min_steps: int = 8,
        max_steps_per_window: int = 256,
        max_traces: int = 512,
        training_horizon_minutes: int = 240,
        device: str | torch.device | None = None,
        seed: int = 20260829,
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

        if heads <= 0:
            raise ValueError(
                "heads must be > 0"
            )

        if hidden_size % heads != 0:
            raise ValueError(
                "hidden_size must be divisible by heads"
            )

        if feedforward_multiplier <= 0:
            raise ValueError(
                "feedforward_multiplier must be > 0"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        if relative_buckets <= 1:
            raise ValueError(
                "relative_buckets must be > 1"
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
        self.heads = int(heads)
        self.feedforward_multiplier = int(
            feedforward_multiplier
        )
        self.dropout = float(dropout)
        self.relative_buckets = int(
            relative_buckets
        )

        self.learning_rate = float(
            learning_rate
        )
        self.gradient_clip = float(
            gradient_clip
        )

        self.min_steps = int(min_steps)
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
            _AttentionTrace,
        ] = OrderedDict()

        self._forward_calls = 0
        self._learning_updates = 0

        self._last_representation_norm = 0.0
        self._last_gradient_norm = 0.0
        self._last_available = False
        self._last_active_scales = 0
        self._last_total_steps = 0

    @property
    def name(self) -> str:
        return "attention"

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
            representation_tensor = (
                self._network(
                    tensor_windows
                )
            )

        representation = self._tensor_to_tuple(
            representation_tensor
        )

        self._store_trace(
            _AttentionTrace(
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
                "Attention representation gradient "
                "size mismatch: "
                f"expected {self.representation_size}, "
                f"got {len(gradient)}"
            )

        for value in gradient:
            if not isfinite(value):
                raise ValueError(
                    "Attention representation gradient "
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
                "training_horizon_minutes": float(
                    self.training_horizon_minutes
                ),
                "max_steps_per_window": float(
                    self.max_steps_per_window
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
                "heads": self.heads,
                "feedforward_multiplier": (
                    self.feedforward_multiplier
                ),
                "dropout": self.dropout,
                "relative_buckets": (
                    self.relative_buckets
                ),
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
            snapshot.get("version", 0)
        )

        if version != 1:
            raise ValueError(
                "Unsupported Attention snapshot "
                f"version: {version}"
            )

        config = snapshot.get("config")

        if not isinstance(config, Mapping):
            raise ValueError(
                "Attention snapshot has invalid config"
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
                "Attention snapshot has invalid "
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
                    "Unknown Attention scale in snapshot: "
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
                "Attention snapshot has invalid state_dict"
            )

        current_state = (
            self._network.state_dict()
        )

        restored_state: dict[
            str,
            Tensor,
        ] = {}

        for name, current_tensor in (
            current_state.items()
        ):
            if name not in raw_state_dict:
                raise ValueError(
                    "Attention snapshot missing "
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
                    "Attention snapshot shape mismatch "
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
                "Attention snapshot contains unexpected "
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
                "Attention snapshot has invalid counters"
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
        if len(window.rows) < self.min_steps:
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
                f"Attention window {scale!r} "
                "is not 2-dimensional"
            )

        if raw.shape[1] != feature_count:
            raise ValueError(
                f"Attention window {scale!r} "
                "feature width mismatch"
            )

        if not torch.isfinite(raw).all():
            raise ValueError(
                f"Attention window {scale!r} "
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
                f"Attention feature width for "
                f"{scale!r} changed from "
                f"{existing} to {feature_count}. "
                "MarketInput feature layout must "
                "remain stable."
            )

        branch_seed = (
            self.seed
            + _SCALE_SEED_OFFSETS[scale]
            + feature_count * 1019
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
        trace: _AttentionTrace,
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
    ) -> _AttentionNetwork:
        with torch.random.fork_rng(
            devices=[]
        ):
            torch.manual_seed(
                self.seed
            )

            return _AttentionNetwork(
                representation_size=(
                    self.representation_size
                ),
                hidden_size=self.hidden_size,
                branch_size=self.branch_size,
                depth=self.depth,
                heads=self.heads,
                feedforward_multiplier=(
                    self.feedforward_multiplier
                ),
                dropout=self.dropout,
                relative_buckets=(
                    self.relative_buckets
                ),
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
            "heads": self.heads,
            "feedforward_multiplier": (
                self.feedforward_multiplier
            ),
            "relative_buckets": (
                self.relative_buckets
            ),
            "min_steps": self.min_steps,
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
                    f"Attention snapshot {name} "
                    f"mismatch: expected {expected}, "
                    f"got {actual}"
                )

        if float(
            config["dropout"]
        ) != self.dropout:
            raise ValueError(
                "Attention snapshot dropout mismatch"
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
            # ROCm-backed PyTorch also uses the "cuda" API.
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
                "Attention representation must be "
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
                    "Attention produced non-finite "
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
