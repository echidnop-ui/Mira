from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import floor, isfinite, log, log1p
from typing import Iterable, Mapping, Sequence

from worlds.market.interface import (
    MarketBackendEvent,
    MarketBackendFrame,
    MarketInput,
    OrderBookLevel,
    OrderBookSnapshot,
    TemporalFeatureWindow,
)


_TEMPORAL_FEATURE_NAMES = (
    "log_return",
    "log_volume",
    "spread_fraction",
    "range_fraction",
    "signed_flow",
    "activity",
)


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True, slots=True)
class TemporalScaleConfig:
    """
    One practical Market timescale.

    These durations are local Market configuration, not shared-Heiwa
    contracts. They may be changed experimentally without changing
    MarketInput or expert public interfaces.
    """

    name: str
    interval_seconds: float
    max_steps: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError(
                "TemporalScaleConfig.name must not be empty"
            )

        if self.interval_seconds <= 0.0:
            raise ValueError(
                "interval_seconds must be > 0"
            )

        if self.max_steps <= 0:
            raise ValueError(
                "max_steps must be > 0"
            )


@dataclass(frozen=True, slots=True)
class MarketInputBuilderConfig:
    """
    Practical defaults for live MarketInput construction.

    The four views are conceptual Market timescales. They are deliberately
    configurable and are not global cognitive constants.

    Default coverage is intentionally overlapping:

        micro   1 s bins
        short  15 s bins
        medium 60 s bins
        long  300 s bins

    Experts remain free to use or ignore any view.
    """

    micro: TemporalScaleConfig = field(
        default_factory=lambda: TemporalScaleConfig(
            name="micro",
            interval_seconds=1.0,
            max_steps=256,
        )
    )

    short: TemporalScaleConfig = field(
        default_factory=lambda: TemporalScaleConfig(
            name="short",
            interval_seconds=15.0,
            max_steps=256,
        )
    )

    medium: TemporalScaleConfig = field(
        default_factory=lambda: TemporalScaleConfig(
            name="medium",
            interval_seconds=60.0,
            max_steps=256,
        )
    )

    long: TemporalScaleConfig = field(
        default_factory=lambda: TemporalScaleConfig(
            name="long",
            interval_seconds=300.0,
            max_steps=256,
        )
    )

    max_order_book_snapshots: int = 512
    order_book_levels: int = 10

    def __post_init__(self) -> None:
        names = (
            self.micro.name,
            self.short.name,
            self.medium.name,
            self.long.name,
        )

        if len(set(names)) != len(names):
            raise ValueError(
                "Market temporal scale names must be unique"
            )

        if self.max_order_book_snapshots <= 0:
            raise ValueError(
                "max_order_book_snapshots must be > 0"
            )

        if self.order_book_levels <= 0:
            raise ValueError(
                "order_book_levels must be > 0"
            )


# ============================================================
# Internal temporal aggregation
# ============================================================


@dataclass(slots=True)
class _Bucket:
    start: float

    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None

    volume: float = 0.0
    signed_volume: float = 0.0

    spread_fraction_sum: float = 0.0
    spread_observations: int = 0

    quote_count: int = 0
    trade_count: int = 0
    candle_count: int = 0

    def observe_price(
        self,
        price: float,
    ) -> None:
        _require_positive_finite(
            price,
            "price",
        )

        if self.open is None:
            self.open = price
            self.high = price
            self.low = price

        self.close = price

        assert self.high is not None
        assert self.low is not None

        if price > self.high:
            self.high = price

        if price < self.low:
            self.low = price

    def observe_quote(
        self,
        *,
        bid: float,
        ask: float,
        last: float | None,
    ) -> None:
        _require_positive_finite(
            bid,
            "bid",
        )
        _require_positive_finite(
            ask,
            "ask",
        )

        if ask < bid:
            raise ValueError(
                "normalized quote ask must be >= bid"
            )

        mid = (
            bid + ask
        ) * 0.5

        spread_fraction = (
            (ask - bid) / mid
            if mid > 0.0
            else 0.0
        )

        self.spread_fraction_sum += (
            spread_fraction
        )
        self.spread_observations += 1
        self.quote_count += 1

        self.observe_price(
            last
            if last is not None
            else mid
        )

    def observe_trade(
        self,
        *,
        price: float,
        quantity: float,
        side: str | None,
    ) -> None:
        _require_positive_finite(
            price,
            "trade price",
        )
        _require_nonnegative_finite(
            quantity,
            "trade quantity",
        )

        self.observe_price(
            price
        )

        self.volume += quantity
        self.trade_count += 1

        if side == "buy":
            self.signed_volume += quantity
        elif side == "sell":
            self.signed_volume -= quantity

    def observe_candle(
        self,
        *,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
    ) -> None:
        for name, value in (
            ("open", open_price),
            ("high", high_price),
            ("low", low_price),
            ("close", close_price),
        ):
            _require_positive_finite(
                value,
                f"candle {name}",
            )

        _require_nonnegative_finite(
            volume,
            "candle volume",
        )

        if high_price < low_price:
            raise ValueError(
                "candle high must be >= low"
            )

        if high_price < max(
            open_price,
            close_price,
        ):
            raise ValueError(
                "candle high is below open/close"
            )

        if low_price > min(
            open_price,
            close_price,
        ):
            raise ValueError(
                "candle low is above open/close"
            )

        if self.open is None:
            self.open = open_price
            self.high = high_price
            self.low = low_price
        else:
            assert self.high is not None
            assert self.low is not None

            self.high = max(
                self.high,
                high_price,
            )
            self.low = min(
                self.low,
                low_price,
            )

        self.close = close_price
        self.volume += volume
        self.candle_count += 1


@dataclass(frozen=True, slots=True)
class _FeatureRow:
    timestamp: float
    values: tuple[float, ...]


class _ScaleAccumulator:
    def __init__(
        self,
        config: TemporalScaleConfig,
    ) -> None:
        self.config = config

        self.rows: deque[_FeatureRow] = deque(
            maxlen=config.max_steps
        )

        self.current: _Bucket | None = None
        self.previous_close: float | None = None

        self.stale_events = 0

    def observe_quote(
        self,
        *,
        observed_at: float,
        bid: float,
        ask: float,
        last: float | None,
    ) -> None:
        bucket = self._bucket_for(
            observed_at
        )

        if bucket is None:
            return

        bucket.observe_quote(
            bid=bid,
            ask=ask,
            last=last,
        )

    def observe_trade(
        self,
        *,
        observed_at: float,
        price: float,
        quantity: float,
        side: str | None,
    ) -> None:
        bucket = self._bucket_for(
            observed_at
        )

        if bucket is None:
            return

        bucket.observe_trade(
            price=price,
            quantity=quantity,
            side=side,
        )

    def observe_candle(
        self,
        *,
        observed_at: float,
        source_interval_seconds: float,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
    ) -> None:
        """
        A coarse candle may seed an equal/coarser scale, but must never be
        expanded into a finer view.

        Example:
            60 s candle
                -> medium 60 s: allowed
                -> long 300 s: allowed
                -> short 15 s: ignored
                -> micro 1 s: ignored
        """

        if (
            source_interval_seconds
            > self.config.interval_seconds
        ):
            return

        bucket = self._bucket_for(
            observed_at
        )

        if bucket is None:
            return

        bucket.observe_candle(
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
        )

    def preview_window(
        self,
    ) -> TemporalFeatureWindow | None:
        rows = list(
            self.rows
        )

        preview = self._feature_row(
            self.current,
            previous_close=self.previous_close,
        )

        if preview is not None:
            rows.append(
                preview
            )

        if not rows:
            return None

        if len(
            rows
        ) > self.config.max_steps:
            rows = rows[
                -self.config.max_steps :
            ]

        return TemporalFeatureWindow(
            name=self.config.name,
            feature_names=(
                _TEMPORAL_FEATURE_NAMES
            ),
            timestamps=tuple(
                item.timestamp
                for item in rows
            ),
            rows=tuple(
                item.values
                for item in rows
            ),
        )

    def reset(
        self,
    ) -> None:
        self.rows.clear()
        self.current = None
        self.previous_close = None
        self.stale_events = 0

    def _bucket_for(
        self,
        observed_at: float,
    ) -> _Bucket | None:
        _require_finite(
            observed_at,
            "observed_at",
        )

        start = (
            floor(
                observed_at
                / self.config.interval_seconds
            )
            * self.config.interval_seconds
        )

        if self.current is None:
            self.current = _Bucket(
                start=start
            )
            return self.current

        if start < self.current.start:
            # Live causal state is append-only. Historical warmup should be
            # ingested oldest-to-newest.
            self.stale_events += 1
            return None

        if start > self.current.start:
            self._finalize_current()

            self.current = _Bucket(
                start=start
            )

        return self.current

    def _finalize_current(
        self,
    ) -> None:
        current = self.current

        if current is None:
            return

        row = self._feature_row(
            current,
            previous_close=self.previous_close,
        )

        if row is not None:
            self.rows.append(
                row
            )

            if current.close is not None:
                self.previous_close = (
                    current.close
                )

    def _feature_row(
        self,
        bucket: _Bucket | None,
        *,
        previous_close: float | None,
    ) -> _FeatureRow | None:
        if (
            bucket is None
            or bucket.close is None
            or bucket.open is None
            or bucket.high is None
            or bucket.low is None
        ):
            return None

        close = bucket.close

        log_return = 0.0

        if (
            previous_close is not None
            and previous_close > 0.0
            and close > 0.0
        ):
            log_return = log(
                close / previous_close
            )

        log_volume = log1p(
            max(
                0.0,
                bucket.volume,
            )
        )

        spread_fraction = (
            bucket.spread_fraction_sum
            / bucket.spread_observations
            if bucket.spread_observations
            else 0.0
        )

        range_fraction = (
            (bucket.high - bucket.low)
            / close
            if close > 0.0
            else 0.0
        )

        signed_flow = (
            bucket.signed_volume
            / bucket.volume
            if bucket.volume > 0.0
            else 0.0
        )

        activity = log1p(
            bucket.quote_count
            + bucket.trade_count
            + bucket.candle_count
        )

        values = (
            log_return,
            log_volume,
            spread_fraction,
            range_fraction,
            signed_flow,
            activity,
        )

        if not all(
            isfinite(value)
            for value in values
        ):
            raise FloatingPointError(
                "Market temporal normalization "
                "produced a non-finite feature"
            )

        # Timestamp is the end of the configured aggregation bucket.
        timestamp = (
            bucket.start
            + self.config.interval_seconds
        )

        return _FeatureRow(
            timestamp=timestamp,
            values=values,
        )


# ============================================================
# Per-instrument state
# ============================================================


class _InstrumentState:
    def __init__(
        self,
        config: MarketInputBuilderConfig,
    ) -> None:
        self.scales = {
            "micro": _ScaleAccumulator(
                config.micro
            ),
            "short": _ScaleAccumulator(
                config.short
            ),
            "medium": _ScaleAccumulator(
                config.medium
            ),
            "long": _ScaleAccumulator(
                config.long
            ),
        }

        self.order_books: deque[
            OrderBookSnapshot
        ] = deque(
            maxlen=(
                config.max_order_book_snapshots
            )
        )

        self.portfolio_features: dict[
            str,
            float,
        ] = {}

        self.context_features: dict[
            str,
            float,
        ] = {}

        self.last_observed_at: float | None = None

        self.events_seen = 0
        self.events_used = 0
        self.events_ignored = 0


# ============================================================
# Telemetry
# ============================================================


@dataclass(frozen=True, slots=True)
class MarketInputBuilderTelemetry:
    instruments: int
    events_seen: int
    events_used: int
    events_ignored: int
    stale_temporal_events: int
    order_book_snapshots: int


# ============================================================
# Builder
# ============================================================


class MarketInputBuilder:
    """
    Stateful broker-neutral MarketInput normalizer.

    Input:
        MarketBackendFrame / MarketBackendEvent

    Output:
        MarketInput

    The backend should already have converted broker SDK objects into simple
    payload mappings. This builder never imports T-Bank SDK types.

    Supported normalized event payloads
    ------------------------------------

    quote:
        {
            "bid_price": number,
            "ask_price": number,
            "last_price": number | optional
        }

    trade:
        {
            "price": number,
            "quantity": number,
            "side": "buy" | "sell" | optional
        }

    candle:
        {
            "open": number,
            "high": number,
            "low": number,
            "close": number,
            "volume": number,
            "interval_seconds": number
        }

    orderbook:
        {
            "bids": [{"price": ..., "quantity": ...}, ...],
            "asks": [{"price": ..., "quantity": ...}, ...]
        }

    `bids` / `asks` may alternatively contain two-item sequences:
        [price, quantity]

    Portfolio/context features are updated explicitly by:
        set_portfolio_features(...)
        set_context_features(...)

    This separation prevents broker-account schema from leaking into neural
    perception.

    Feature engineering here is deliberately mechanical:
        log return
        log volume
        relative spread
        relative high-low range
        signed trade flow
        event activity

    No RSI/MACD/hand-written buy/sell rule is injected.
    """

    def __init__(
        self,
        config: MarketInputBuilderConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else MarketInputBuilderConfig()
        )

        self._states: dict[
            str,
            _InstrumentState,
        ] = {}

    # --------------------------------------------------------
    # Public ingestion
    # --------------------------------------------------------

    def ingest_frame(
        self,
        frame: MarketBackendFrame,
    ) -> None:
        for event in frame.events:
            self.ingest_event(
                event
            )

    def ingest_events(
        self,
        events: Iterable[
            MarketBackendEvent
        ],
    ) -> None:
        for event in events:
            self.ingest_event(
                event
            )

    def ingest_event(
        self,
        event: MarketBackendEvent,
    ) -> bool:
        """
        Normalize one event.

        Returns True when the event affected MarketInput state.

        Unknown event kinds are intentionally ignored rather than treated as
        errors because execution/order-status events may share the same
        backend frame.
        """

        instrument_id = event.instrument_id

        if not instrument_id:
            return False

        state = self._state(
            instrument_id
        )

        state.events_seen += 1

        kind = event.kind.casefold()

        used = False

        if kind == "quote":
            used = self._ingest_quote(
                state,
                event,
            )

        elif kind == "trade":
            used = self._ingest_trade(
                state,
                event,
            )

        elif kind == "candle":
            used = self._ingest_candle(
                state,
                event,
            )

        elif kind == "orderbook":
            used = self._ingest_orderbook(
                state,
                event,
            )

        if used:
            state.events_used += 1

            if (
                state.last_observed_at is None
                or event.observed_at
                > state.last_observed_at
            ):
                state.last_observed_at = (
                    event.observed_at
                )
        else:
            state.events_ignored += 1

        return used

    # --------------------------------------------------------
    # Explicit account/context features
    # --------------------------------------------------------

    def set_portfolio_features(
        self,
        instrument_id: str,
        features: Mapping[
            str,
            float,
        ],
    ) -> None:
        self._state(
            instrument_id
        ).portfolio_features = (
            _validated_feature_mapping(
                features
            )
        )

    def set_context_features(
        self,
        instrument_id: str,
        features: Mapping[
            str,
            float,
        ],
    ) -> None:
        self._state(
            instrument_id
        ).context_features = (
            _validated_feature_mapping(
                features
            )
        )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    def build(
        self,
        instrument_id: str,
        *,
        as_of: float | None = None,
        metadata: Mapping[
            str,
            object,
        ] | None = None,
    ) -> MarketInput:
        if not instrument_id:
            raise ValueError(
                "instrument_id must not be empty"
            )

        state = self._state(
            instrument_id
        )

        resolved_as_of = (
            float(as_of)
            if as_of is not None
            else state.last_observed_at
        )

        if resolved_as_of is None:
            raise ValueError(
                "cannot build MarketInput before "
                "observing market data"
            )

        _require_finite(
            resolved_as_of,
            "as_of",
        )

        portfolio_names, portfolio_values = (
            _ordered_features(
                state.portfolio_features
            )
        )

        context_names, context_values = (
            _ordered_features(
                state.context_features
            )
        )

        builder_metadata: dict[
            str,
            object,
        ] = {
            "input_builder": "MarketInputBuilder",
            "events_seen": state.events_seen,
            "events_used": state.events_used,
            "events_ignored": state.events_ignored,
            "order_book_snapshots": len(
                state.order_books
            ),
        }

        if metadata:
            builder_metadata.update(
                metadata
            )

        return MarketInput(
            instrument_id=instrument_id,
            as_of=resolved_as_of,
            micro=state.scales[
                "micro"
            ].preview_window(),
            short=state.scales[
                "short"
            ].preview_window(),
            medium=state.scales[
                "medium"
            ].preview_window(),
            long=state.scales[
                "long"
            ].preview_window(),
            order_book_history=tuple(
                state.order_books
            ),
            portfolio_feature_names=(
                portfolio_names
            ),
            portfolio_features=(
                portfolio_values
            ),
            context_feature_names=(
                context_names
            ),
            context_features=(
                context_values
            ),
            metadata=builder_metadata,
        )

    # --------------------------------------------------------
    # State / telemetry
    # --------------------------------------------------------

    def telemetry(
        self,
    ) -> MarketInputBuilderTelemetry:
        states = tuple(
            self._states.values()
        )

        return MarketInputBuilderTelemetry(
            instruments=len(states),
            events_seen=sum(
                state.events_seen
                for state in states
            ),
            events_used=sum(
                state.events_used
                for state in states
            ),
            events_ignored=sum(
                state.events_ignored
                for state in states
            ),
            stale_temporal_events=sum(
                scale.stale_events
                for state in states
                for scale in state.scales.values()
            ),
            order_book_snapshots=sum(
                len(
                    state.order_books
                )
                for state in states
            ),
        )

    def reset(
        self,
        instrument_id: str | None = None,
    ) -> None:
        if instrument_id is None:
            self._states.clear()
            return

        self._states.pop(
            instrument_id,
            None,
        )

    # ========================================================
    # Event normalization
    # ========================================================

    def _ingest_quote(
        self,
        state: _InstrumentState,
        event: MarketBackendEvent,
    ) -> bool:
        payload = event.payload

        bid = _required_number(
            payload,
            "bid_price",
        )

        ask = _required_number(
            payload,
            "ask_price",
        )

        last = _optional_number(
            payload,
            "last_price",
        )

        for scale in (
            state.scales.values()
        ):
            scale.observe_quote(
                observed_at=event.observed_at,
                bid=bid,
                ask=ask,
                last=last,
            )

        return True

    def _ingest_trade(
        self,
        state: _InstrumentState,
        event: MarketBackendEvent,
    ) -> bool:
        payload = event.payload

        price = _required_number(
            payload,
            "price",
        )

        quantity = _required_number(
            payload,
            "quantity",
        )

        raw_side = payload.get(
            "side"
        )

        side: str | None

        if raw_side is None:
            side = None
        else:
            side = str(
                raw_side
            ).casefold()

            if side not in {
                "buy",
                "sell",
            }:
                side = None

        for scale in (
            state.scales.values()
        ):
            scale.observe_trade(
                observed_at=event.observed_at,
                price=price,
                quantity=quantity,
                side=side,
            )

        return True

    def _ingest_candle(
        self,
        state: _InstrumentState,
        event: MarketBackendEvent,
    ) -> bool:
        payload = event.payload

        open_price = _required_number(
            payload,
            "open",
        )
        high_price = _required_number(
            payload,
            "high",
        )
        low_price = _required_number(
            payload,
            "low",
        )
        close_price = _required_number(
            payload,
            "close",
        )
        volume = _required_number(
            payload,
            "volume",
        )
        interval_seconds = (
            _required_number(
                payload,
                "interval_seconds",
            )
        )

        if interval_seconds <= 0.0:
            raise ValueError(
                "candle interval_seconds must be > 0"
            )

        used = False

        for scale in (
            state.scales.values()
        ):
            if (
                interval_seconds
                <= scale.config.interval_seconds
            ):
                scale.observe_candle(
                    observed_at=event.observed_at,
                    source_interval_seconds=(
                        interval_seconds
                    ),
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close_price=close_price,
                    volume=volume,
                )

                used = True

        return used

    def _ingest_orderbook(
        self,
        state: _InstrumentState,
        event: MarketBackendEvent,
    ) -> bool:
        payload = event.payload

        bids = _book_side(
            payload.get(
                "bids"
            ),
            side="bids",
            limit=(
                self.config.order_book_levels
            ),
        )

        asks = _book_side(
            payload.get(
                "asks"
            ),
            side="asks",
            limit=(
                self.config.order_book_levels
            ),
        )

        if not bids or not asks:
            return False

        # Normalize ordering even if the backend returned a different order.
        bids = tuple(
            sorted(
                bids,
                key=lambda level: (
                    level.price
                ),
                reverse=True,
            )
        )

        asks = tuple(
            sorted(
                asks,
                key=lambda level: (
                    level.price
                ),
            )
        )

        snapshot = OrderBookSnapshot(
            observed_at=event.observed_at,
            bids=bids,
            asks=asks,
        )

        # Keep L2 history causal and monotonic.
        if (
            state.order_books
            and event.observed_at
            < state.order_books[-1].observed_at
        ):
            return False

        state.order_books.append(
            snapshot
        )

        # Best bid/ask is also mechanically useful to temporal spread input.
        best_bid = bids[0].price
        best_ask = asks[0].price

        for scale in (
            state.scales.values()
        ):
            scale.observe_quote(
                observed_at=event.observed_at,
                bid=best_bid,
                ask=best_ask,
                last=None,
            )

        return True

    # ========================================================
    # Internal
    # ========================================================

    def _state(
        self,
        instrument_id: str,
    ) -> _InstrumentState:
        if not instrument_id:
            raise ValueError(
                "instrument_id must not be empty"
            )

        state = self._states.get(
            instrument_id
        )

        if state is None:
            state = _InstrumentState(
                self.config
            )

            self._states[
                instrument_id
            ] = state

        return state


# ============================================================
# Helpers
# ============================================================


def _required_number(
    payload: Mapping[
        str,
        object,
    ],
    key: str,
) -> float:
    if key not in payload:
        raise ValueError(
            f"market event payload missing {key!r}"
        )

    value = _as_number(
        payload[key],
        key,
    )

    return value


def _optional_number(
    payload: Mapping[
        str,
        object,
    ],
    key: str,
) -> float | None:
    raw = payload.get(
        key
    )

    if raw is None:
        return None

    return _as_number(
        raw,
        key,
    )


def _as_number(
    raw: object,
    name: str,
) -> float:
    try:
        value = float(
            raw
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{name} must be numeric"
        ) from exc

    _require_finite(
        value,
        name,
    )

    return value


def _require_finite(
    value: float,
    name: str,
) -> None:
    if not isfinite(
        value
    ):
        raise ValueError(
            f"{name} must be finite"
        )


def _require_positive_finite(
    value: float,
    name: str,
) -> None:
    _require_finite(
        value,
        name,
    )

    if value <= 0.0:
        raise ValueError(
            f"{name} must be > 0"
        )


def _require_nonnegative_finite(
    value: float,
    name: str,
) -> None:
    _require_finite(
        value,
        name,
    )

    if value < 0.0:
        raise ValueError(
            f"{name} must be >= 0"
        )


def _validated_feature_mapping(
    features: Mapping[
        str,
        float,
    ],
) -> dict[
    str,
    float,
]:
    result: dict[
        str,
        float,
    ] = {}

    for raw_name, raw_value in (
        features.items()
    ):
        name = str(
            raw_name
        ).strip()

        if not name:
            raise ValueError(
                "feature name must not be empty"
            )

        value = _as_number(
            raw_value,
            name,
        )

        result[name] = value

    return result


def _ordered_features(
    features: Mapping[
        str,
        float,
    ],
) -> tuple[
    tuple[str, ...],
    tuple[float, ...],
]:
    names = tuple(
        sorted(
            features
        )
    )

    return (
        names,
        tuple(
            features[name]
            for name in names
        ),
    )


def _book_side(
    raw_side: object,
    *,
    side: str,
    limit: int,
) -> tuple[
    OrderBookLevel,
    ...,
]:
    if raw_side is None:
        return ()

    if not isinstance(
        raw_side,
        Sequence,
    ) or isinstance(
        raw_side,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise ValueError(
            f"orderbook {side} must be a sequence"
        )

    result: list[
        OrderBookLevel
    ] = []

    for raw_level in raw_side:
        price: float
        quantity: float

        if isinstance(
            raw_level,
            Mapping,
        ):
            price = _required_number(
                raw_level,
                "price",
            )
            quantity = _required_number(
                raw_level,
                "quantity",
            )

        elif (
            isinstance(
                raw_level,
                Sequence,
            )
            and not isinstance(
                raw_level,
                (
                    str,
                    bytes,
                    bytearray,
                ),
            )
            and len(
                raw_level
            ) >= 2
        ):
            price = _as_number(
                raw_level[0],
                "orderbook price",
            )
            quantity = _as_number(
                raw_level[1],
                "orderbook quantity",
            )

        else:
            raise ValueError(
                f"invalid orderbook {side} level"
            )

        _require_positive_finite(
            price,
            "orderbook price",
        )
        _require_nonnegative_finite(
            quantity,
            "orderbook quantity",
        )

        result.append(
            OrderBookLevel(
                price=price,
                quantity=quantity,
            )
        )

        if len(
            result
        ) >= limit:
            break

    return tuple(
        result
    )
