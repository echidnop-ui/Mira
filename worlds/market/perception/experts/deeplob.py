from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite, log1p, sqrt
from typing import Mapping

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "DeepLOBMarketExpert requires PyTorch. "
        "Install a PyTorch build appropriate for the runtime device."
    ) from exc

from worlds.market.interface import (
    MarketInput,
    OrderBookSnapshot,
)
from worlds.market.perception.interface import (
    ExpertOutput,
    ExpertTelemetry,
    MarketExpert,
)


# ============================================================
# Causal replay trace
# ============================================================


@dataclass(frozen=True, slots=True)
class _BookTrace:
    """
    Exact normalized L2 input used by one causal transition.

    Shape semantics of `rows`:

        time
            ↓
        tuple[
            tuple[
                tuple[
                    bid_relative_price,
                    bid_log_quantity,
                    ask_relative_price,
                    ask_log_quantity,
                ],
                ... one tuple per depth level ...
            ],
            ... one tuple per snapshot ...
        ]

    Storing normalized values instead of an old autograd graph keeps delayed
    learning memory-bounded while preserving the exact transition input.
    """

    rows: tuple[
        tuple[
            tuple[float, float, float, float],
            ...
        ],
        ...
    ]


@dataclass(frozen=True, slots=True)
class _DeepLOBTrace:
    transition_id: str
    book: _BookTrace


# ============================================================
# DeepLOB-style neural network
# ============================================================


class _SamePadConv2d(nn.Module):
    """
    Conv2d with explicit "same" padding for odd kernels.

    We keep padding visible because the temporal/level dimensions have clear
    semantics in L2 data.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int],
    ) -> None:
        super().__init__()

        time_kernel, level_kernel = kernel_size

        if time_kernel <= 0 or level_kernel <= 0:
            raise ValueError(
                "Conv2d kernel dimensions must be > 0"
            )

        if time_kernel % 2 == 0 or level_kernel % 2 == 0:
            raise ValueError(
                "_SamePadConv2d expects odd kernel dimensions"
            )

        self.time_pad = time_kernel // 2
        self.level_pad = level_kernel // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x = F.pad(
            x,
            (
                self.level_pad,
                self.level_pad,
                self.time_pad,
                self.time_pad,
            ),
        )

        return self.conv(x)


class _ResidualBookBlock(nn.Module):
    """
    Residual convolution over time × order-book depth.

    This learns local microstructure patterns without manually encoding
    imbalance rules.
    """

    def __init__(
        self,
        channels: int,
        *,
        temporal_kernel: int,
        level_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv1 = _SamePadConv2d(
            channels,
            channels,
            kernel_size=(
                temporal_kernel,
                level_kernel,
            ),
        )

        self.norm1 = nn.GroupNorm(
            1,
            channels,
        )

        self.conv2 = _SamePadConv2d(
            channels,
            channels,
            kernel_size=(
                temporal_kernel,
                level_kernel,
            ),
        )

        self.norm2 = nn.GroupNorm(
            1,
            channels,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
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


class _TemporalInception(nn.Module):
    """
    DeepLOB-style temporal multi-kernel block.

    Different temporal kernels observe different short microstructure
    durations while sharing the same learned book representation.
    """

    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            _SamePadConv2d(
                in_channels,
                branch_channels,
                kernel_size=(1, 1),
            ),
            nn.GroupNorm(
                1,
                branch_channels,
            ),
            nn.SiLU(),
        )

        self.branch3 = nn.Sequential(
            _SamePadConv2d(
                in_channels,
                branch_channels,
                kernel_size=(3, 1),
            ),
            nn.GroupNorm(
                1,
                branch_channels,
            ),
            nn.SiLU(),
        )

        self.branch5 = nn.Sequential(
            _SamePadConv2d(
                in_channels,
                branch_channels,
                kernel_size=(5, 1),
            ),
            nn.GroupNorm(
                1,
                branch_channels,
            ),
            nn.SiLU(),
        )

        self.pool_projection = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0),
            ),
            _SamePadConv2d(
                in_channels,
                branch_channels,
                kernel_size=(1, 1),
            ),
            nn.GroupNorm(
                1,
                branch_channels,
            ),
            nn.SiLU(),
        )

        self.dropout = nn.Dropout(
            dropout
        )

    @property
    def output_channels(self) -> int:
        return int(
            self.branch1[0].conv.out_channels
            * 4
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        branches = (
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.pool_projection(x),
        )

        return self.dropout(
            torch.cat(
                branches,
                dim=1,
            )
        )


class _DeepLOBNetwork(nn.Module):
    """
    Order-book specialist:

        normalized L2 tensor
            ↓
        side/price/quantity channel projection
            ↓
        residual book convolutions
            ↓
        temporal inception
            ↓
        collapse book-depth dimension
            ↓
        recurrent temporal summarizer
            ↓
        learned pooling
            ↓
        fixed ExpertOutput representation

    The recurrent summarizer is local to the microstructure expert and works
    on CNN-extracted L2 features, so its role is different from the generic
    recurrent Market expert.
    """

    def __init__(
        self,
        *,
        levels: int,
        representation_size: int,
        conv_channels: int,
        inception_channels: int,
        recurrent_size: int,
        recurrent_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.levels = int(
            levels
        )
        self.representation_size = int(
            representation_size
        )
        self.conv_channels = int(
            conv_channels
        )
        self.inception_channels = int(
            inception_channels
        )
        self.recurrent_size = int(
            recurrent_size
        )
        self.recurrent_layers = int(
            recurrent_layers
        )

        # Input channels:
        # 0 bid relative price
        # 1 bid log quantity
        # 2 ask relative price
        # 3 ask log quantity
        self.input_projection = nn.Sequential(
            _SamePadConv2d(
                4,
                conv_channels,
                kernel_size=(1, 1),
            ),
            nn.GroupNorm(
                1,
                conv_channels,
            ),
            nn.SiLU(),
        )

        self.book_blocks = nn.Sequential(
            _ResidualBookBlock(
                conv_channels,
                temporal_kernel=3,
                level_kernel=3,
                dropout=dropout,
            ),
            _ResidualBookBlock(
                conv_channels,
                temporal_kernel=5,
                level_kernel=3,
                dropout=dropout,
            ),
            _ResidualBookBlock(
                conv_channels,
                temporal_kernel=3,
                level_kernel=5,
                dropout=dropout,
            ),
        )

        self.inception = _TemporalInception(
            conv_channels,
            inception_channels,
            dropout=dropout,
        )

        temporal_features = (
            self.inception.output_channels
            * 3
        )

        self.temporal_projection = nn.Sequential(
            nn.Linear(
                temporal_features,
                recurrent_size,
            ),
            nn.LayerNorm(
                recurrent_size
            ),
            nn.SiLU(),
        )

        self.recurrent = nn.LSTM(
            input_size=recurrent_size,
            hidden_size=recurrent_size,
            num_layers=recurrent_layers,
            dropout=(
                dropout
                if recurrent_layers > 1
                else 0.0
            ),
            batch_first=True,
        )

        self.pool_query = nn.Parameter(
            torch.zeros(
                recurrent_size
            )
        )

        self.output_projection = nn.Sequential(
            nn.Linear(
                recurrent_size * 4,
                representation_size * 2,
            ),
            nn.SiLU(),
            nn.Linear(
                representation_size * 2,
                representation_size,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        book: Tensor,
    ) -> Tensor:
        """
        book:
            [time, levels, 4]

        returns:
            [representation_size]
        """

        if book.ndim != 3:
            raise ValueError(
                "_DeepLOBNetwork expects [time, levels, 4]"
            )

        if book.shape[1] != self.levels:
            raise ValueError(
                "DeepLOB depth changed unexpectedly"
            )

        if book.shape[2] != 4:
            raise ValueError(
                "DeepLOB expects four book channels"
            )

        # Conv2d:
        # [batch, channels, time, levels]
        x = book.permute(
            2,
            0,
            1,
        ).unsqueeze(0)

        x = self.input_projection(
            x
        )

        x = self.book_blocks(
            x
        )

        x = self.inception(
            x
        )

        # Preserve temporal order but compress order-book depth three ways.
        #
        # Shapes after reduction:
        # [batch, channels, time]
        mean_depth = x.mean(
            dim=3
        )
        max_depth = x.amax(
            dim=3
        )
        best_depth = x[:, :, :, 0]

        # [batch, time, channels * 3]
        temporal = torch.cat(
            (
                mean_depth,
                max_depth,
                best_depth,
            ),
            dim=1,
        ).transpose(
            1,
            2,
        )

        temporal = self.temporal_projection(
            temporal
        )

        recurrent, _ = self.recurrent(
            temporal
        )

        recurrent = recurrent.squeeze(
            0
        )

        last_state = recurrent[-1]
        mean_state = recurrent.mean(
            dim=0
        )
        max_state = recurrent.amax(
            dim=0
        )

        scores = torch.mv(
            recurrent,
            self.pool_query,
        ) / sqrt(
            float(self.recurrent_size)
        )

        weights = torch.softmax(
            scores,
            dim=0,
        )

        learned_pool = torch.sum(
            recurrent
            * weights.unsqueeze(1),
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


# ============================================================
# DeepLOB Market expert
# ============================================================


class DeepLOBMarketExpert(MarketExpert):
    """
    Trainable order-book microstructure specialist.

    Input:
        MarketInput.order_book_history

    Output:
        learned ExpertOutput representation

    This expert becomes unavailable when there is insufficient usable L2
    history. The rest of Market perception continues normally and Router can
    simply ignore DeepLOB.

    It does NOT:
        - call T-Bank;
        - place orders;
        - choose BUY/SELL/HOLD;
        - treat order-book imbalance as a hand-written trading rule.

    Order-book normalization is intentionally mechanical:

        relative_price =
            (price - mid_price) / mid_price

        quantity =
            log1p(nonnegative quantity)

    Quantities are then standardized per level/channel over the selected
    history. No label or future information is used.

    Delayed causal learning:
        transition_id
            -> exact normalized L2 history
            -> replay on learn()
            -> downstream representation gradient
            -> DeepLOB parameter update

    `training_horizon_minutes=240` is retained as Market-specific experiment
    metadata. DeepLOB itself is a microstructure expert; downstream training
    may discover that its representation matters only for a smaller part of
    the 240-minute episode.
    """

    def __init__(
        self,
        *,
        representation_size: int = 64,
        levels: int = 10,
        conv_channels: int = 48,
        inception_channels: int = 24,
        recurrent_size: int = 64,
        recurrent_layers: int = 1,
        dropout: float = 0.05,
        learning_rate: float = 1e-3,
        gradient_clip: float = 5.0,
        min_snapshots: int = 16,
        max_snapshots: int = 256,
        max_traces: int = 256,
        training_horizon_minutes: int = 240,
        device: str | torch.device | None = None,
        seed: int = 20260831,
    ) -> None:
        if representation_size <= 0:
            raise ValueError(
                "representation_size must be > 0"
            )

        if levels <= 0:
            raise ValueError(
                "levels must be > 0"
            )

        if conv_channels <= 0:
            raise ValueError(
                "conv_channels must be > 0"
            )

        if inception_channels <= 0:
            raise ValueError(
                "inception_channels must be > 0"
            )

        if recurrent_size <= 0:
            raise ValueError(
                "recurrent_size must be > 0"
            )

        if recurrent_layers <= 0:
            raise ValueError(
                "recurrent_layers must be > 0"
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

        if min_snapshots <= 1:
            raise ValueError(
                "min_snapshots must be > 1"
            )

        if max_snapshots < min_snapshots:
            raise ValueError(
                "max_snapshots must be >= min_snapshots"
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

        self.levels = int(
            levels
        )
        self.conv_channels = int(
            conv_channels
        )
        self.inception_channels = int(
            inception_channels
        )
        self.recurrent_size = int(
            recurrent_size
        )
        self.recurrent_layers = int(
            recurrent_layers
        )
        self.dropout = float(
            dropout
        )

        self.learning_rate = float(
            learning_rate
        )
        self.gradient_clip = float(
            gradient_clip
        )

        self.min_snapshots = int(
            min_snapshots
        )
        self.max_snapshots = int(
            max_snapshots
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

        self._traces: OrderedDict[
            str,
            _DeepLOBTrace,
        ] = OrderedDict()

        self._forward_calls = 0
        self._learning_updates = 0

        self._last_available = False
        self._last_snapshot_count = 0
        self._last_representation_norm = 0.0
        self._last_gradient_norm = 0.0

    # ========================================================
    # MarketExpert contract
    # ========================================================

    @property
    def name(self) -> str:
        return "deeplob"

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

        book_trace = self._extract_book(
            market_input.order_book_history
        )

        self._forward_calls += 1

        if book_trace is None:
            self._last_available = False
            self._last_snapshot_count = 0
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
                    "snapshots": 0.0,
                    "levels": float(
                        self.levels
                    ),
                    "training_horizon_minutes": float(
                        self.training_horizon_minutes
                    ),
                },
            )

        book_tensor = self._trace_to_tensor(
            book_trace
        )

        self._network.eval()

        with torch.no_grad():
            representation_tensor = self._network(
                book_tensor
            )

        representation = self._tensor_to_tuple(
            representation_tensor
        )

        self._store_trace(
            _DeepLOBTrace(
                transition_id=transition_id,
                book=book_trace,
            )
        )

        self._last_available = True
        self._last_snapshot_count = len(
            book_trace.rows
        )
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
                "snapshots": float(
                    self._last_snapshot_count
                ),
                "levels": float(
                    self.levels
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

        if len(
            gradient
        ) != self.representation_size:
            raise ValueError(
                "DeepLOB representation gradient size mismatch: "
                f"expected {self.representation_size}, "
                f"got {len(gradient)}"
            )

        for value in gradient:
            if not isfinite(
                value
            ):
                raise ValueError(
                    "DeepLOB representation gradient "
                    "contains non-finite value"
                )

        book_tensor = self._trace_to_tensor(
            trace.book
        )

        self._network.train()
        self._zero_grad()

        representation = self._network(
            book_tensor
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
            learning_updates=self._learning_updates,
            last_representation_norm=(
                self._last_representation_norm
            ),
            last_gradient_norm=(
                self._last_gradient_norm
            ),
            uncertainty=None,
            diagnostics={
                "snapshots": float(
                    self._last_snapshot_count
                ),
                "levels": float(
                    self.levels
                ),
                "pending_traces": float(
                    len(self._traces)
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
                "levels": self.levels,
                "conv_channels": (
                    self.conv_channels
                ),
                "inception_channels": (
                    self.inception_channels
                ),
                "recurrent_size": (
                    self.recurrent_size
                ),
                "recurrent_layers": (
                    self.recurrent_layers
                ),
                "dropout": self.dropout,
                "learning_rate": (
                    self.learning_rate
                ),
                "gradient_clip": (
                    self.gradient_clip
                ),
                "min_snapshots": (
                    self.min_snapshots
                ),
                "max_snapshots": (
                    self.max_snapshots
                ),
                "max_traces": (
                    self.max_traces
                ),
                "training_horizon_minutes": (
                    self.training_horizon_minutes
                ),
                "seed": self.seed,
            },
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
                "Unsupported DeepLOB snapshot "
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
                "DeepLOB snapshot has invalid config"
            )

        self._validate_snapshot_config(
            config
        )

        raw_state_dict = snapshot.get(
            "state_dict"
        )

        if not isinstance(
            raw_state_dict,
            Mapping,
        ):
            raise ValueError(
                "DeepLOB snapshot has invalid state_dict"
            )

        self._network = self._new_network().to(
            self.device
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
                    "DeepLOB snapshot missing "
                    f"parameter {name!r}"
                )

            restored = torch.tensor(
                raw_state_dict[name],
                dtype=current_tensor.dtype,
                device=self.device,
            )

            if restored.shape != current_tensor.shape:
                raise ValueError(
                    "DeepLOB snapshot shape mismatch "
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
                "DeepLOB snapshot contains unexpected "
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
                "DeepLOB snapshot has invalid counters"
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
        self._last_snapshot_count = 0

        self._traces.clear()

    def reset(self) -> None:
        self._network = self._new_network().to(
            self.device
        )

        self._traces.clear()

        self._forward_calls = 0
        self._learning_updates = 0

        self._last_available = False
        self._last_snapshot_count = 0
        self._last_representation_norm = 0.0
        self._last_gradient_norm = 0.0

    # ========================================================
    # Order-book preparation
    # ========================================================

    def _extract_book(
        self,
        history: tuple[
            OrderBookSnapshot,
            ...,
        ],
    ) -> _BookTrace | None:
        if len(
            history
        ) < self.min_snapshots:
            return None

        selected = history[
            -self.max_snapshots :
        ]

        raw_rows: list[
            list[
                tuple[
                    float,
                    float,
                    float,
                    float,
                ]
            ]
        ] = []

        for snapshot in selected:
            normalized_snapshot = (
                self._normalize_snapshot(
                    snapshot
                )
            )

            if normalized_snapshot is None:
                continue

            raw_rows.append(
                normalized_snapshot
            )

        if len(
            raw_rows
        ) < self.min_snapshots:
            return None

        tensor = torch.tensor(
            raw_rows,
            dtype=torch.float32,
        )

        if tensor.ndim != 3:
            raise RuntimeError(
                "DeepLOB normalized book must have "
                "[time, levels, channels] shape"
            )

        if tensor.shape[1] != self.levels:
            raise RuntimeError(
                "DeepLOB normalized book depth mismatch"
            )

        if tensor.shape[2] != 4:
            raise RuntimeError(
                "DeepLOB normalized book channel mismatch"
            )

        if not torch.isfinite(
            tensor
        ).all():
            raise ValueError(
                "DeepLOB normalized book contains "
                "non-finite values"
            )

        # Price channels are already relative to each snapshot mid-price.
        #
        # Quantity channels are log-transformed but can still differ greatly
        # by instrument/book level. Standardize them across selected history
        # for each level independently.
        quantity = tensor[:, :, (1, 3)]

        quantity_mean = quantity.mean(
            dim=0,
            keepdim=True,
        )

        quantity_std = quantity.std(
            dim=0,
            unbiased=False,
            keepdim=True,
        ).clamp_min(
            1e-6
        )

        tensor[:, :, (1, 3)] = (
            quantity - quantity_mean
        ) / quantity_std

        tensor = tensor.clamp(
            min=-12.0,
            max=12.0,
        )

        rows = tuple(
            tuple(
                tuple(
                    float(value)
                    for value in level
                )
                for level in snapshot
            )
            for snapshot
            in tensor.tolist()
        )

        return _BookTrace(
            rows=rows
        )

    def _normalize_snapshot(
        self,
        snapshot: OrderBookSnapshot,
    ) -> list[
        tuple[
            float,
            float,
            float,
            float,
        ]
    ] | None:
        if (
            len(snapshot.bids) < self.levels
            or len(snapshot.asks) < self.levels
        ):
            return None

        best_bid = float(
            snapshot.bids[0].price
        )
        best_ask = float(
            snapshot.asks[0].price
        )

        if (
            not isfinite(best_bid)
            or not isfinite(best_ask)
            or best_bid <= 0.0
            or best_ask <= 0.0
        ):
            return None

        mid = (
            best_bid + best_ask
        ) * 0.5

        if mid <= 0.0:
            return None

        result: list[
            tuple[
                float,
                float,
                float,
                float,
            ]
        ] = []

        for index in range(
            self.levels
        ):
            bid = snapshot.bids[index]
            ask = snapshot.asks[index]

            bid_price = float(
                bid.price
            )
            ask_price = float(
                ask.price
            )
            bid_quantity = float(
                bid.quantity
            )
            ask_quantity = float(
                ask.quantity
            )

            values = (
                bid_price,
                ask_price,
                bid_quantity,
                ask_quantity,
            )

            if not all(
                isfinite(value)
                for value in values
            ):
                return None

            if (
                bid_price <= 0.0
                or ask_price <= 0.0
                or bid_quantity < 0.0
                or ask_quantity < 0.0
            ):
                return None

            result.append(
                (
                    (bid_price - mid) / mid,
                    log1p(
                        bid_quantity
                    ),
                    (ask_price - mid) / mid,
                    log1p(
                        ask_quantity
                    ),
                )
            )

        return result

    def _trace_to_tensor(
        self,
        trace: _BookTrace,
    ) -> Tensor:
        return torch.tensor(
            trace.rows,
            dtype=torch.float32,
            device=self.device,
        )

    # ========================================================
    # Trace management
    # ========================================================

    def _store_trace(
        self,
        trace: _DeepLOBTrace,
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
    ) -> _DeepLOBNetwork:
        with torch.random.fork_rng(
            devices=[]
        ):
            torch.manual_seed(
                self.seed
            )

            return _DeepLOBNetwork(
                levels=self.levels,
                representation_size=(
                    self.representation_size
                ),
                conv_channels=(
                    self.conv_channels
                ),
                inception_channels=(
                    self.inception_channels
                ),
                recurrent_size=(
                    self.recurrent_size
                ),
                recurrent_layers=(
                    self.recurrent_layers
                ),
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
            "levels": self.levels,
            "conv_channels": (
                self.conv_channels
            ),
            "inception_channels": (
                self.inception_channels
            ),
            "recurrent_size": (
                self.recurrent_size
            ),
            "recurrent_layers": (
                self.recurrent_layers
            ),
            "min_snapshots": (
                self.min_snapshots
            ),
            "max_snapshots": (
                self.max_snapshots
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
                    f"DeepLOB snapshot {name} mismatch: "
                    f"expected {expected}, got {actual}"
                )

        if float(
            config["dropout"]
        ) != self.dropout:
            raise ValueError(
                "DeepLOB snapshot dropout mismatch"
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
        detached = tensor.detach().cpu()

        if detached.ndim != 1:
            raise RuntimeError(
                "DeepLOB representation must be "
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
                    "DeepLOB produced non-finite "
                    "representation"
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
