from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math

from contracts.cognition import Affordance
from worlds.market.interface import MarketInstrument


@dataclass(frozen=True, slots=True)
class MarketAffordanceConfig:
    """
    Mechanical Market capability exposure.

    This config controls only how long one capability snapshot remains valid.
    It does not encode trading preferences or strategy.
    """

    ttl_seconds: float = 5.0

    def __post_init__(self) -> None:
        ttl = float(self.ttl_seconds)
        if not math.isfinite(ttl) or ttl <= 0.0:
            raise ValueError(
                "ttl_seconds must be finite and > 0"
            )


@dataclass(frozen=True, slots=True)
class MarketTradeLimits:
    """
    Broker-neutral current quantity limits for one instrument.

    These values describe account/environment mechanics. They are not Policy
    preferences and do not mean that Heiwa intends to trade.
    """

    currency: str
    buy_max_lots: int
    buy_max_market_lots: int
    sell_max_lots: int

    def __post_init__(self) -> None:
        for name in (
            "buy_max_lots",
            "buy_max_market_lots",
            "sell_max_lots",
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(
                    f"{name} must be an integer >= 0"
                )
            try:
                normalized = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"{name} must be an integer >= 0"
                ) from exc
            if normalized < 0:
                raise ValueError(
                    f"{name} must be >= 0"
                )
            object.__setattr__(
                self,
                name,
                normalized,
            )


class MarketAffordanceProvider:
    """
    Convert current Market mechanics into dynamic shared Affordances.

    The provider answers only:

        "what is technically possible now?"

    It does NOT answer:

        "what should Heiwa do?"

    In particular, this module contains no price-direction rule, no RSI/MACD,
    no BUY/SELL preference and no policy score.

    Current first-slice capabilities:
        market.buy
        market.sell
        market.hold

    `market.cancel_order` should be added only when normalized open-order state
    exists. `market.inspect` should be added only when there is a real
    executable inspection mechanism.
    """

    def __init__(
        self,
        config: MarketAffordanceConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else MarketAffordanceConfig()
        )

    # ========================================================
    # Local/mechanical limits
    # ========================================================

    def build(
        self,
        instrument: MarketInstrument,
        *,
        world_id: str,
        now: float,
        reference_price: Decimal | float | int,
        available_cash: Decimal | float | int,
        held_lots: int,
        trading_open: bool = True,
    ) -> tuple[Affordance, ...]:
        """
        Build capabilities from already-normalized cash and position state.

        This path remains useful for broker-neutral worlds/tests.

        `reference_price` is used only for mechanical affordability. Execution
        must still validate the actual order against live environment state.
        """

        self._validate_common(
            instrument,
            world_id=world_id,
            now=now,
        )

        now = float(now)
        price = self._positive_number(
            reference_price,
            "reference_price",
        )
        cash = self._nonnegative_number(
            available_cash,
            "available_cash",
        )
        held_lots = self._nonnegative_int(
            held_lots,
            "held_lots",
        )

        per_lot_cost = (
            price * instrument.lot_size
        )
        affordable_lots = int(
            math.floor(
                cash / per_lot_cost
            )
        )

        return self._build_from_quantity_limits(
            instrument,
            world_id=world_id,
            now=now,
            trading_open=trading_open,
            buy_max_lots=affordable_lots,
            buy_max_market_lots=affordable_lots,
            sell_max_lots=held_lots,
            payload_extra={
                "reference_price": price,
            },
            constraint_extra={
                "reference_price": price,
                "available_cash": cash,
                "max_quantity_basis": (
                    "reference_price_affordability"
                ),
            },
            limit_source=(
                "normalized_cash_position"
            ),
        )

    # ========================================================
    # Broker-computed limits
    # ========================================================

    def build_from_limits(
        self,
        instrument: MarketInstrument,
        limits: MarketTradeLimits,
        *,
        world_id: str,
        now: float,
        trading_open: bool,
    ) -> tuple[Affordance, ...]:
        """
        Build capabilities from limits calculated by a broker/environment.

        This is preferred when the external environment already knows account,
        blocked-funds, open-order and other execution constraints better than a
        local approximation.

        Policy still only sees Affordances. It does not call the broker.
        """

        self._validate_common(
            instrument,
            world_id=world_id,
            now=now,
        )

        now = float(now)

        if (
            instrument.currency
            and limits.currency
            and instrument.currency.casefold()
            != limits.currency.casefold()
        ):
            raise ValueError(
                "broker trade-limit currency does not match instrument currency"
            )

        return self._build_from_quantity_limits(
            instrument,
            world_id=world_id,
            now=now,
            trading_open=trading_open,
            buy_max_lots=limits.buy_max_lots,
            buy_max_market_lots=(
                limits.buy_max_market_lots
            ),
            sell_max_lots=limits.sell_max_lots,
            payload_extra={},
            constraint_extra={},
            limit_source="broker_max_lots",
        )

    # ========================================================
    # Shared construction
    # ========================================================

    def _build_from_quantity_limits(
        self,
        instrument: MarketInstrument,
        *,
        world_id: str,
        now: float,
        trading_open: bool,
        buy_max_lots: int,
        buy_max_market_lots: int,
        sell_max_lots: int,
        payload_extra: dict[str, object],
        constraint_extra: dict[str, object],
        limit_source: str,
    ) -> tuple[Affordance, ...]:
        buy_max_lots = self._nonnegative_int(
            buy_max_lots,
            "buy_max_lots",
        )
        buy_max_market_lots = self._nonnegative_int(
            buy_max_market_lots,
            "buy_max_market_lots",
        )
        sell_max_lots = self._nonnegative_int(
            sell_max_lots,
            "sell_max_lots",
        )

        expires_at = (
            now + float(self.config.ttl_seconds)
        )
        timestamp_key = int(
            round(now * 1_000_000.0)
        )

        currency = instrument.currency

        common_payload = {
            "instrument_id": instrument.instrument_id,
            "ticker": instrument.ticker,
            "currency": currency,
            **payload_extra,
        }

        common_constraints = {
            "lot_size": instrument.lot_size,
            "currency": currency,
            "limit_source": limit_source,
            **constraint_extra,
        }

        if instrument.min_price_increment is not None:
            increment = float(
                instrument.min_price_increment
            )
            if (
                math.isfinite(increment)
                and increment > 0.0
            ):
                common_constraints[
                    "min_price_increment"
                ] = increment

        affordances: list[Affordance] = []

        if (
            trading_open
            and instrument.buy_available
            and buy_max_lots >= 1
        ):
            affordances.append(
                Affordance(
                    id=self._id(
                        world_id,
                        instrument.instrument_id,
                        "buy",
                        timestamp_key,
                    ),
                    world=world_id,
                    action="market.buy",
                    offered_at=now,
                    payload=dict(
                        common_payload
                    ),
                    constraints={
                        **common_constraints,
                        "min_quantity": 1,
                        "max_quantity": buy_max_lots,
                        "max_market_quantity": (
                            buy_max_market_lots
                        ),
                    },
                    expires_at=expires_at,
                )
            )

        if (
            trading_open
            and instrument.sell_available
            and sell_max_lots >= 1
        ):
            affordances.append(
                Affordance(
                    id=self._id(
                        world_id,
                        instrument.instrument_id,
                        "sell",
                        timestamp_key,
                    ),
                    world=world_id,
                    action="market.sell",
                    offered_at=now,
                    payload=dict(
                        common_payload
                    ),
                    constraints={
                        **common_constraints,
                        "min_quantity": 1,
                        "max_quantity": sell_max_lots,
                    },
                    expires_at=expires_at,
                )
            )

        # HOLD is a valid cognitive option even when external trading is closed
        # or account constraints currently make BUY/SELL impossible. It does
        # not claim that an external broker action occurred.
        affordances.append(
            Affordance(
                id=self._id(
                    world_id,
                    instrument.instrument_id,
                    "hold",
                    timestamp_key,
                ),
                world=world_id,
                action="market.hold",
                offered_at=now,
                payload=dict(
                    common_payload
                ),
                constraints={
                    "trading_open": bool(
                        trading_open
                    ),
                    "limit_source": limit_source,
                },
                expires_at=expires_at,
            )
        )

        return tuple(
            affordances
        )

    # ========================================================
    # Validation/helpers
    # ========================================================

    @staticmethod
    def _validate_common(
        instrument: MarketInstrument,
        *,
        world_id: str,
        now: float,
    ) -> None:
        if not world_id:
            raise ValueError(
                "world_id must not be empty"
            )

        now = float(now)
        if not math.isfinite(now):
            raise ValueError(
                "now must be finite"
            )

        if not instrument.instrument_id:
            raise ValueError(
                "instrument_id must not be empty"
            )

        if instrument.lot_size <= 0:
            raise ValueError(
                "instrument lot_size must be > 0"
            )

    @staticmethod
    def _id(
        world_id: str,
        instrument_id: str,
        action: str,
        timestamp_key: int,
    ) -> str:
        return (
            f"affordance:{world_id}:"
            f"{instrument_id}:{action}:"
            f"{timestamp_key}"
        )

    @staticmethod
    def _positive_number(
        value: Decimal | float | int,
        name: str,
    ) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} must be numeric"
            ) from exc

        if (
            not math.isfinite(numeric)
            or numeric <= 0.0
        ):
            raise ValueError(
                f"{name} must be finite and > 0"
            )

        return numeric

    @staticmethod
    def _nonnegative_number(
        value: Decimal | float | int,
        name: str,
    ) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} must be numeric"
            ) from exc

        if (
            not math.isfinite(numeric)
            or numeric < 0.0
        ):
            raise ValueError(
                f"{name} must be finite and >= 0"
            )

        return numeric

    @staticmethod
    def _nonnegative_int(
        value: int,
        name: str,
    ) -> int:
        if isinstance(value, bool):
            raise ValueError(
                f"{name} must be an integer >= 0"
            )
        try:
            normalized = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} must be an integer >= 0"
            ) from exc
        if normalized < 0:
            raise ValueError(
                f"{name} must be >= 0"
            )
        return normalized
