from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable

from contracts.cognition import ActionAttempt, ActionIntent


# ============================================================
# Market mode
# ============================================================


class MarketMode(str, Enum):
    """
    Execution environment used by a Market backend.

    This is market-specific infrastructure state, not a cognitive action.
    """

    SANDBOX = "sandbox"
    REAL = "real"


# ============================================================
# Instruments
# ============================================================


@dataclass(frozen=True, slots=True)
class MarketInstrument:
    """
    Normalized instrument description.

    T-Bank SDK objects must not leak outside the Market module.
    """

    instrument_id: str
    ticker: str
    name: str

    currency: str
    lot_size: int

    min_price_increment: Decimal | None = None

    buy_available: bool = True
    sell_available: bool = True

    payload: Mapping[str, object] = field(default_factory=dict)


# ============================================================
# Backend events
# ============================================================


@dataclass(frozen=True, slots=True)
class MarketBackendEvent:
    """
    One normalized piece of external Market information.

    This object belongs to the Market module.

    It is intentionally more specific than a shared Observation but still
    independent of any particular broker SDK.

    Typical `kind` values:

        quote
        candle
        trade
        orderbook
        portfolio
        position
        order_status
        execution
        trading_status

    The Market World may later convert these events into shared Observation
    objects and/or neural MarketInput.
    """

    id: str
    kind: str
    observed_at: float

    instrument_id: str | None = None

    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MarketBackendFrame:
    """
    Result of one backend poll.

    A frame is external reality collected from the broker/market backend.
    It is not a cognitive transition.
    """

    backend_id: str
    mode: MarketMode
    observed_at: float

    events: tuple[MarketBackendEvent, ...] = ()

    # Instruments may be refreshed less often than quotes/events.
    instruments: tuple[MarketInstrument, ...] = ()


# ============================================================
# Orders
# ============================================================


class MarketOrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class MarketOrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True, slots=True)
class MarketOrderRequest:
    """
    Market-specific execution request produced AFTER cognition chose
    a market Affordance.

    This object does not decide whether to trade.

    `execution.py` will eventually translate ActionIntent into this request.
    """

    request_id: str
    transition_id: str
    intent_id: str

    instrument_id: str
    side: MarketOrderSide
    quantity_lots: int
    order_type: MarketOrderType

    created_at: float

    limit_price: Decimal | None = None

    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MarketCancelRequest:
    """
    Request to cancel an already existing external market order.
    """

    request_id: str
    transition_id: str
    intent_id: str

    external_order_id: str
    created_at: float


@dataclass(frozen=True, slots=True)
class MarketOrderSubmission:
    """
    Immediate result of asking the backend to submit an order.

    `accepted=True` means only that the backend/broker accepted the request
    for processing.

    It does NOT mean the order has been filled.
    """

    request_id: str
    transition_id: str
    intent_id: str

    accepted: bool
    submitted_at: float

    external_order_id: str | None = None
    message: str | None = None

    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MarketCancelSubmission:
    """
    Immediate result of a cancellation request.

    This is not necessarily the final order state.
    """

    request_id: str
    transition_id: str
    intent_id: str

    accepted: bool
    submitted_at: float

    external_order_id: str
    message: str | None = None

    payload: Mapping[str, object] = field(default_factory=dict)


# ============================================================
# Market neural input
# ============================================================


@dataclass(frozen=True, slots=True)
class TemporalFeatureWindow:
    """
    One normalized temporal view of market data.

    Rows are ordered from oldest to newest.

    Example:

        feature_names = (
            "return",
            "volume",
            "spread",
            "range",
        )

        rows = (
            (...),
            (...),
            (...),
        )

    This contract deliberately does not depend on NumPy, PyTorch or any
    specific neural framework.

    Individual experts may convert the tuples into their own tensors.
    """

    name: str

    feature_names: tuple[str, ...]
    timestamps: tuple[float, ...]
    rows: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.rows):
            raise ValueError(
                "TemporalFeatureWindow timestamps/rows length mismatch"
            )

        width = len(self.feature_names)

        for row in self.rows:
            if len(row) != width:
                raise ValueError(
                    "TemporalFeatureWindow row width does not match "
                    "feature_names"
                )


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    """
    One normalized order-book level.

    Quantity is expressed in the Market module's normalized units.
    """

    price: float
    quantity: float


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """
    One normalized L2 snapshot.

    Bids and asks should be ordered best-first by the normalization layer.
    """

    observed_at: float

    bids: tuple[OrderBookLevel, ...] = ()
    asks: tuple[OrderBookLevel, ...] = ()


@dataclass(frozen=True, slots=True)
class MarketInput:
    """
    Neural input produced by Market normalization/perception preparation.

    The public topology supports heterogeneous temporal experts from the
    beginning without forcing every expert to consume every field.

    A source may be absent.

    Example:
        no Level 2 history
            -> order_book_history=()

    The Router/experts must treat absence explicitly.
    """

    instrument_id: str
    as_of: float

    micro: TemporalFeatureWindow | None = None
    short: TemporalFeatureWindow | None = None
    medium: TemporalFeatureWindow | None = None
    long: TemporalFeatureWindow | None = None

    order_book_history: tuple[OrderBookSnapshot, ...] = ()

    portfolio_feature_names: tuple[str, ...] = ()
    portfolio_features: tuple[float, ...] = ()

    context_feature_names: tuple[str, ...] = ()
    context_features: tuple[float, ...] = ()

    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.portfolio_feature_names) != len(self.portfolio_features):
            raise ValueError(
                "portfolio feature names/values length mismatch"
            )

        if len(self.context_feature_names) != len(self.context_features):
            raise ValueError(
                "context feature names/values length mismatch"
            )


# ============================================================
# Market neural output
# ============================================================


@dataclass(frozen=True, slots=True)
class MarketRepresentation:
    """
    Public output of the Market Neural Organ.

    This is what shared Heiwa cognition sees.

    It is NOT BUY/SELL/HOLD.

    The internal expert topology remains private to Market perception.
    """

    transition_id: str
    instrument_id: str
    as_of: float

    vector: tuple[float, ...]

    # Model-level uncertainty or uncertainty summary.
    uncertainty: float | None = None

    # Diagnostics are public enough for Runtime/telemetry but must not be
    # required for cognition to function.
    expert_weights: Mapping[str, float] = field(default_factory=dict)
    expert_available: Mapping[str, bool] = field(default_factory=dict)

    diagnostics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MarketOrganTelemetry:
    """
    Compact diagnostics for live experiments.

    Telemetry must never mutate cognition.
    """

    learning_updates: int = 0
    last_gradient_norm: float = 0.0

    representation_norm: float = 0.0

    expert_weights: Mapping[str, float] = field(default_factory=dict)
    expert_norms: Mapping[str, float] = field(default_factory=dict)
    expert_uncertainty: Mapping[str, float] = field(default_factory=dict)
    expert_available: Mapping[str, bool] = field(default_factory=dict)


# ============================================================
# Market Neural Organ protocol
# ============================================================


@runtime_checkable
class MarketNeuralOrgan(Protocol):
    """
    Public contract for market-specific neural perception.

    Implementations may internally contain:

        TCN
        SSM
        Attention
        xLSTM/recurrent expert
        DeepLOB
        regime/volatility specialists
        trainable Router
        learned fusion

    None of those internal implementation classes should be required by
    shared Runtime/Core code.
    """

    @property
    def representation_size(self) -> int:
        ...

    def encode(
        self,
        market_input: MarketInput,
        *,
        transition_id: str,
        shared_context: tuple[float, ...] | None = None,
    ) -> MarketRepresentation:
        """
        Produce a learned market representation for one causal transition.

        `shared_context` is optional.

        It may later allow Market routing to condition on shared Heiwa state
        through an explicit public contract.

        Implementations must not reach into Core internals.
        """
        ...

    def learn(
        self,
        transition_id: str,
        representation_gradient: tuple[float, ...],
    ) -> bool:
        """
        Apply causal downstream learning to the exact Market representation
        that influenced `transition_id`.

        Return True only if a matching causal trace existed and was updated.
        """
        ...

    def telemetry(self) -> MarketOrganTelemetry:
        ...

    def snapshot(self) -> Mapping[str, object]:
        """
        Return serializable Market-organ state.

        Exact persistence format is owned by the implementation.
        """
        ...

    def restore(self, snapshot: Mapping[str, object]) -> None:
        ...

    def reset(self) -> None:
        """
        Reset learned Market-organ state for experimental Tabula Rasa.
        """
        ...


# ============================================================
# Market backend protocol
# ============================================================


@runtime_checkable
class MarketBackend(Protocol):
    """
    Broker/market transport boundary.

    The backend knows how to communicate with a concrete external service.

    It does NOT know how Heiwa decides whether to buy or sell.

    T-Bank SDK types must be normalized before crossing this interface.
    """

    @property
    def backend_id(self) -> str:
        """
        Stable backend identifier.

        Examples:
            "tbank:sandbox"
            "tbank:real"
        """
        ...

    @property
    def mode(self) -> MarketMode:
        ...

    def connect(self) -> None:
        """
        Establish/prepare technical connectivity.

        This is infrastructure, not cognition.
        """
        ...

    def poll(self, now: float) -> MarketBackendFrame:
        """
        Read newly available normalized market/broker information.
        """
        ...

    def submit_order(
        self,
        request: MarketOrderRequest,
    ) -> MarketOrderSubmission:
        """
        Submit an already-decided market order.
        """
        ...

    def cancel_order(
        self,
        request: MarketCancelRequest,
    ) -> MarketCancelSubmission:
        """
        Request cancellation of an existing external order.
        """
        ...

    def close(self) -> None:
        ...


# ============================================================
# Execution bridge helpers
# ============================================================


def submission_to_action_attempt(
    intent: ActionIntent,
    submission: MarketOrderSubmission | MarketCancelSubmission,
    *,
    world_id: str,
) -> ActionAttempt:
    """
    Convert a Market backend submission into the shared ActionAttempt.

    This helper preserves the causal ids without teaching cognition anything.
    """

    return ActionAttempt(
        id=submission.request_id,
        intent_id=intent.id,
        transition_id=intent.transition_id,
        world=world_id,
        action=intent.action,
        attempted_at=submission.submitted_at,
        accepted=submission.accepted,
        external_id=submission.external_order_id,
        payload={
            "message": submission.message,
            **dict(submission.payload),
        },
    )
