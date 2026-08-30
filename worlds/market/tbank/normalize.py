from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from math import isfinite
from typing import Iterable, Mapping, Sequence

from worlds.market.interface import (
    MarketBackendEvent,
    MarketInstrument,
)


# ============================================================
# Generic REST/protobuf-JSON helpers
# ============================================================


_MISSING = object()


def field(
    payload: Mapping[str, object],
    *names: str,
    default: object = _MISSING,
) -> object:
    """
    Read one protobuf JSON field while tolerating camelCase / snake_case.

    T-Bank REST currently exposes protobuf JSON using camelCase in examples,
    while gRPC documentation often names the same field in snake_case.
    Keeping the tolerance here prevents SDK/transport naming from leaking into
    the rest of Heiwa.
    """

    for name in names:
        if name in payload:
            return payload[name]

    if default is not _MISSING:
        return default

    joined = " / ".join(names)
    raise ValueError(
        f"T-Bank payload missing required field: {joined}"
    )


def mapping_field(
    payload: Mapping[str, object],
    *names: str,
    default: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    raw = field(
        payload,
        *names,
        default=default if default is not None else {},
    )

    if raw is None:
        return {}

    if not isinstance(raw, Mapping):
        raise ValueError(
            f"T-Bank field {names[0]!r} must be an object"
        )

    return raw


def sequence_field(
    payload: Mapping[str, object],
    *names: str,
) -> Sequence[object]:
    raw = field(
        payload,
        *names,
        default=(),
    )

    if raw is None:
        return ()

    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes, bytearray))
    ):
        raise ValueError(
            f"T-Bank field {names[0]!r} must be an array"
        )

    return raw


def string_field(
    payload: Mapping[str, object],
    *names: str,
    default: str = "",
) -> str:
    raw = field(
        payload,
        *names,
        default=default,
    )

    if raw is None:
        return default

    return str(raw)


def bool_field(
    payload: Mapping[str, object],
    *names: str,
    default: bool = False,
) -> bool:
    raw = field(
        payload,
        *names,
        default=default,
    )

    if isinstance(raw, bool):
        return raw

    if isinstance(raw, str):
        lowered = raw.strip().casefold()

        if lowered in {"true", "1", "yes"}:
            return True

        if lowered in {"false", "0", "no", ""}:
            return False

    return bool(raw)


def int_field(
    payload: Mapping[str, object],
    *names: str,
    default: int = 0,
) -> int:
    raw = field(
        payload,
        *names,
        default=default,
    )

    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"T-Bank field {names[0]!r} must be an integer"
        ) from exc


def quotation_to_decimal(
    raw: object,
) -> Decimal:
    """
    Convert protobuf Quotation / MoneyValue-like JSON to Decimal.

    Standard protobuf JSON encodes int64 `units` as a string and `nano`
    as an integer.
    """

    if raw is None:
        return Decimal("0")

    if isinstance(raw, Decimal):
        return raw

    if isinstance(raw, (int, float, str)):
        try:
            return Decimal(str(raw))
        except Exception as exc:
            raise ValueError(
                "invalid scalar quotation"
            ) from exc

    if not isinstance(raw, Mapping):
        raise ValueError(
            "quotation must be a scalar or object"
        )

    units_raw = field(
        raw,
        "units",
        default="0",
    )
    nano_raw = field(
        raw,
        "nano",
        default=0,
    )

    try:
        units = Decimal(str(units_raw))
        nano = Decimal(int(nano_raw)) / Decimal(
            1_000_000_000
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "invalid quotation units/nano"
        ) from exc

    return units + nano


def quotation_to_float(
    raw: object,
) -> float:
    result = float(
        quotation_to_decimal(raw)
    )

    if not isfinite(result):
        raise ValueError(
            "quotation converted to a non-finite float"
        )

    return result


def timestamp_to_epoch(
    raw: object,
    *,
    fallback: float | None = None,
) -> float:
    if raw is None or raw == "":
        if fallback is None:
            raise ValueError(
                "T-Bank timestamp is missing"
            )
        return float(fallback)

    if isinstance(raw, (int, float)):
        value = float(raw)

        if not isfinite(value):
            raise ValueError(
                "T-Bank timestamp is non-finite"
            )

        return value

    text = str(raw).strip()

    if not text:
        if fallback is None:
            raise ValueError(
                "T-Bank timestamp is empty"
            )
        return float(fallback)

    # Protobuf REST uses RFC3339. datetime.fromisoformat accepts +00:00,
    # therefore normalize Z explicitly.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"invalid T-Bank timestamp: {raw!r}"
        ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    return dt.timestamp()


def _stable_event_id(
    kind: str,
    instrument_id: str,
    *parts: object,
) -> str:
    payload = "|".join(
        str(part)
        for part in (
            kind,
            instrument_id,
            *parts,
        )
    ).encode("utf-8")

    digest = sha256(
        payload
    ).hexdigest()[:24]

    return (
        f"tbank:{kind}:"
        f"{instrument_id}:{digest}"
    )


# ============================================================
# Instrument normalization
# ============================================================


def normalize_instrument(
    response_or_instrument: Mapping[str, object],
    *,
    requested_id: str | None = None,
) -> MarketInstrument:
    """
    Normalize GetInstrumentBy response or its nested `instrument` object.
    """

    nested = field(
        response_or_instrument,
        "instrument",
        default=None,
    )

    if isinstance(nested, Mapping):
        raw = nested
    else:
        raw = response_or_instrument

    uid = string_field(
        raw,
        "uid",
        "instrumentUid",
        "instrument_uid",
        default="",
    )

    figi = string_field(
        raw,
        "figi",
        default="",
    )

    ticker = string_field(
        raw,
        "ticker",
        default=requested_id or uid or figi,
    )

    class_code = string_field(
        raw,
        "classCode",
        "class_code",
        default="",
    )

    instrument_id = (
        uid
        or figi
        or requested_id
        or (
            f"{ticker}_{class_code}"
            if ticker and class_code
            else ticker
        )
    )

    if not instrument_id:
        raise ValueError(
            "T-Bank instrument has no usable identifier"
        )

    lot_size = int_field(
        raw,
        "lot",
        default=1,
    )

    if lot_size <= 0:
        lot_size = 1

    increment_raw = field(
        raw,
        "minPriceIncrement",
        "min_price_increment",
        default=None,
    )

    min_price_increment = (
        quotation_to_decimal(
            increment_raw
        )
        if increment_raw is not None
        else None
    )

    payload = {
        "figi": figi,
        "uid": uid,
        "class_code": class_code,
        "api_trade_available": bool_field(
            raw,
            "apiTradeAvailableFlag",
            "api_trade_available_flag",
            default=False,
        ),
        "instrument_type": string_field(
            raw,
            "instrumentType",
            "instrument_type",
            default="",
        ),
        "exchange": string_field(
            raw,
            "exchange",
            default="",
        ),
    }

    return MarketInstrument(
        instrument_id=instrument_id,
        ticker=ticker,
        name=string_field(
            raw,
            "name",
            default=ticker,
        ),
        currency=string_field(
            raw,
            "currency",
            default="",
        ),
        lot_size=lot_size,
        min_price_increment=min_price_increment,
        buy_available=bool_field(
            raw,
            "buyAvailableFlag",
            "buy_available_flag",
            default=True,
        ),
        sell_available=bool_field(
            raw,
            "sellAvailableFlag",
            "sell_available_flag",
            default=True,
        ),
        payload=payload,
    )


# ============================================================
# Candle normalization
# ============================================================


def normalize_candles(
    response: Mapping[str, object],
    *,
    instrument_id: str,
    interval_seconds: float,
) -> tuple[MarketBackendEvent, ...]:
    events: list[
        MarketBackendEvent
    ] = []

    for raw_item in sequence_field(
        response,
        "candles",
    ):
        if not isinstance(
            raw_item,
            Mapping,
        ):
            continue

        observed_at = timestamp_to_epoch(
            field(
                raw_item,
                "time",
                default=None,
            )
        )

        open_price = quotation_to_float(
            field(
                raw_item,
                "open",
            )
        )
        high_price = quotation_to_float(
            field(
                raw_item,
                "high",
            )
        )
        low_price = quotation_to_float(
            field(
                raw_item,
                "low",
            )
        )
        close_price = quotation_to_float(
            field(
                raw_item,
                "close",
            )
        )

        volume = float(
            int_field(
                raw_item,
                "volume",
                default=0,
            )
        )

        event_id = _stable_event_id(
            "candle",
            instrument_id,
            observed_at,
            interval_seconds,
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
        )

        events.append(
            MarketBackendEvent(
                id=event_id,
                kind="candle",
                observed_at=observed_at,
                instrument_id=instrument_id,
                payload={
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "interval_seconds": (
                        float(interval_seconds)
                    ),
                    "is_complete": bool_field(
                        raw_item,
                        "isComplete",
                        "is_complete",
                        default=True,
                    ),
                    "source": string_field(
                        raw_item,
                        "candleSource",
                        "candle_source",
                        default="",
                    ),
                    "volume_unit": "lots",
                },
            )
        )

    events.sort(
        key=lambda event: (
            event.observed_at,
            event.id,
        )
    )

    return tuple(events)


# ============================================================
# Trade normalization
# ============================================================


def _trade_side(
    raw: object,
) -> str | None:
    if raw is None:
        return None

    text = str(raw).upper()

    if text.endswith("_BUY") or text == "BUY":
        return "buy"

    if text.endswith("_SELL") or text == "SELL":
        return "sell"

    return None


def normalize_trades(
    response: Mapping[str, object],
    *,
    instrument_id: str,
) -> tuple[MarketBackendEvent, ...]:
    events: list[
        MarketBackendEvent
    ] = []

    # A repeated trade signature is unusual but legal. Include an occurrence
    # index among equal signatures so overlapping REST windows can still be
    # deduplicated deterministically.
    occurrence: dict[
        tuple[object, ...],
        int,
    ] = {}

    for raw_item in sequence_field(
        response,
        "trades",
    ):
        if not isinstance(
            raw_item,
            Mapping,
        ):
            continue

        observed_at = timestamp_to_epoch(
            field(
                raw_item,
                "time",
                default=None,
            )
        )

        price = quotation_to_float(
            field(
                raw_item,
                "price",
            )
        )

        quantity = float(
            int_field(
                raw_item,
                "quantity",
                default=0,
            )
        )

        side = _trade_side(
            field(
                raw_item,
                "direction",
                default=None,
            )
        )

        raw_trade_id = string_field(
            raw_item,
            "tradeId",
            "trade_id",
            default="",
        )

        signature = (
            observed_at,
            price,
            quantity,
            side,
            raw_trade_id,
        )

        ordinal = occurrence.get(
            signature,
            0,
        )
        occurrence[signature] = (
            ordinal + 1
        )

        event_id = _stable_event_id(
            "trade",
            instrument_id,
            *signature,
            ordinal,
        )

        events.append(
            MarketBackendEvent(
                id=event_id,
                kind="trade",
                observed_at=observed_at,
                instrument_id=instrument_id,
                payload={
                    "price": price,
                    "quantity": quantity,
                    "side": side,
                    "trade_id": raw_trade_id,
                    "trade_source": string_field(
                        raw_item,
                        "tradeSource",
                        "trade_source",
                        default="",
                    ),
                    "quantity_unit": "lots",
                },
            )
        )

    events.sort(
        key=lambda event: (
            event.observed_at,
            event.id,
        )
    )

    return tuple(events)


# ============================================================
# Order-book normalization
# ============================================================


def _normalize_book_side(
    raw_side: object,
) -> list[
    dict[str, float]
]:
    if (
        raw_side is None
        or not isinstance(
            raw_side,
            Sequence,
        )
        or isinstance(
            raw_side,
            (str, bytes, bytearray),
        )
    ):
        return []

    result: list[
        dict[str, float]
    ] = []

    for raw_level in raw_side:
        if not isinstance(
            raw_level,
            Mapping,
        ):
            continue

        price = quotation_to_float(
            field(
                raw_level,
                "price",
            )
        )

        quantity = float(
            int_field(
                raw_level,
                "quantity",
                default=0,
            )
        )

        if price <= 0.0 or quantity < 0.0:
            continue

        result.append(
            {
                "price": price,
                "quantity": quantity,
            }
        )

    return result


def normalize_orderbook(
    response: Mapping[str, object],
    *,
    instrument_id: str,
    fallback_observed_at: float,
) -> MarketBackendEvent | None:
    bids = _normalize_book_side(
        field(
            response,
            "bids",
            default=(),
        )
    )
    asks = _normalize_book_side(
        field(
            response,
            "asks",
            default=(),
        )
    )

    if not bids or not asks:
        return None

    bids.sort(
        key=lambda level: level["price"],
        reverse=True,
    )
    asks.sort(
        key=lambda level: level["price"],
    )

    observed_at = timestamp_to_epoch(
        field(
            response,
            "orderbookTs",
            "orderbook_ts",
            "time",
            default=None,
        ),
        fallback=fallback_observed_at,
    )

    last_price_raw = field(
        response,
        "lastPrice",
        "last_price",
        default=None,
    )

    last_price = (
        quotation_to_float(
            last_price_raw
        )
        if last_price_raw is not None
        else None
    )

    event_id = _stable_event_id(
        "orderbook",
        instrument_id,
        observed_at,
        len(bids),
        len(asks),
        bids[0]["price"],
        bids[0]["quantity"],
        asks[0]["price"],
        asks[0]["quantity"],
    )

    return MarketBackendEvent(
        id=event_id,
        kind="orderbook",
        observed_at=observed_at,
        instrument_id=instrument_id,
        payload={
            "bids": bids,
            "asks": asks,
            "last_price": last_price,
            "depth": int_field(
                response,
                "depth",
                default=min(
                    len(bids),
                    len(asks),
                ),
            ),
            "quantity_unit": "lots",
            "book_type": string_field(
                response,
                "orderBookType",
                "order_book_type",
                default="",
            ),
        },
    )


# ============================================================
# Trading-status normalization
# ============================================================


def normalize_trading_status(
    response: Mapping[str, object],
    *,
    instrument_id: str,
    observed_at: float,
) -> MarketBackendEvent:
    status = string_field(
        response,
        "tradingStatus",
        "trading_status",
        default="",
    )

    event_id = _stable_event_id(
        "trading_status",
        instrument_id,
        observed_at,
        status,
        bool_field(
            response,
            "apiTradeAvailableFlag",
            "api_trade_available_flag",
            default=False,
        ),
    )

    return MarketBackendEvent(
        id=event_id,
        kind="trading_status",
        observed_at=observed_at,
        instrument_id=instrument_id,
        payload={
            "trading_status": status,
            "api_trade_available": bool_field(
                response,
                "apiTradeAvailableFlag",
                "api_trade_available_flag",
                default=False,
            ),
            "market_order_available": bool_field(
                response,
                "marketOrderAvailableFlag",
                "market_order_available_flag",
                default=False,
            ),
            "limit_order_available": bool_field(
                response,
                "limitOrderAvailableFlag",
                "limit_order_available_flag",
                default=False,
            ),
        },
    )


# ============================================================
# Generic deterministic event filtering
# ============================================================


def ordered_events(
    groups: Iterable[
        Iterable[MarketBackendEvent]
    ],
) -> tuple[
    MarketBackendEvent,
    ...,
]:
    events = [
        event
        for group in groups
        for event in group
    ]

    events.sort(
        key=lambda event: (
            event.observed_at,
            event.id,
        )
    )

    return tuple(events)
