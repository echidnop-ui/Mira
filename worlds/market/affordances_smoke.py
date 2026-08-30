from __future__ import annotations

from decimal import Decimal

from worlds.market.affordances import (
    MarketAffordanceProvider,
)
from worlds.market.interface import MarketInstrument


def _actions(
    affordances,
) -> tuple[str, ...]:
    return tuple(
        affordance.action
        for affordance in affordances
    )


def _by_action(
    affordances,
    action: str,
):
    return next(
        affordance
        for affordance in affordances
        if affordance.action == action
    )


def main() -> int:
    print("MARKET AFFORDANCES SMOKE")

    provider = MarketAffordanceProvider()

    instrument = MarketInstrument(
        instrument_id="SBER_TQBR",
        ticker="SBER",
        name="Sberbank",
        currency="rub",
        lot_size=10,
        min_price_increment=Decimal("0.01"),
        buy_available=True,
        sell_available=True,
    )

    now = 3000.0

    # --------------------------------------------------------
    # Full mechanically available set
    # --------------------------------------------------------

    affordances = provider.build(
        instrument,
        world_id="market:tbank:sandbox",
        now=now,
        reference_price=Decimal("250"),
        available_cash=Decimal("10000"),
        held_lots=3,
        trading_open=True,
    )

    actions = _actions(
        affordances
    )

    if actions != (
        "market.buy",
        "market.sell",
        "market.hold",
    ):
        raise AssertionError(
            "unexpected full affordance set: "
            f"{actions!r}"
        )

    buy = _by_action(
        affordances,
        "market.buy",
    )
    sell = _by_action(
        affordances,
        "market.sell",
    )

    # 10,000 / (250 * 10) = 4 lots.
    if (
        buy.constraints.get(
            "max_quantity"
        )
        != 4
    ):
        raise AssertionError(
            "buy affordability was not converted to lots"
        )

    if (
        sell.constraints.get(
            "max_quantity"
        )
        != 3
    ):
        raise AssertionError(
            "sell capability ignored held_lots"
        )

    if not all(
        affordance.expires_at is not None
        and affordance.expires_at > now
        for affordance in affordances
    ):
        raise AssertionError(
            "affordances must have a future expiry"
        )

    print(
        "full_set:",
        ", ".join(actions),
    )
    print("buy_cash_limit: OK")
    print("sell_position_limit: OK")
    print("expiry: OK")

    # --------------------------------------------------------
    # No enough cash => BUY is mechanically unavailable
    # --------------------------------------------------------

    no_cash = provider.build(
        instrument,
        world_id="market:tbank:sandbox",
        now=now + 1.0,
        reference_price=Decimal("250"),
        available_cash=Decimal("2499"),
        held_lots=3,
        trading_open=True,
    )

    no_cash_actions = _actions(
        no_cash
    )

    if "market.buy" in no_cash_actions:
        raise AssertionError(
            "BUY was offered without enough cash for one lot"
        )

    if no_cash_actions != (
        "market.sell",
        "market.hold",
    ):
        raise AssertionError(
            "unexpected no-cash affordance set: "
            f"{no_cash_actions!r}"
        )

    print("no_cash_removes_buy: OK")

    # --------------------------------------------------------
    # No position => SELL is mechanically unavailable
    # --------------------------------------------------------

    no_position = provider.build(
        instrument,
        world_id="market:tbank:sandbox",
        now=now + 2.0,
        reference_price=Decimal("250"),
        available_cash=Decimal("10000"),
        held_lots=0,
        trading_open=True,
    )

    no_position_actions = _actions(
        no_position
    )

    if "market.sell" in no_position_actions:
        raise AssertionError(
            "SELL was offered without held lots"
        )

    if no_position_actions != (
        "market.buy",
        "market.hold",
    ):
        raise AssertionError(
            "unexpected no-position affordance set: "
            f"{no_position_actions!r}"
        )

    print("no_position_removes_sell: OK")

    # --------------------------------------------------------
    # Closed trading => no external trade capability, HOLD remains
    # --------------------------------------------------------

    closed = provider.build(
        instrument,
        world_id="market:tbank:sandbox",
        now=now + 3.0,
        reference_price=Decimal("250"),
        available_cash=Decimal("10000"),
        held_lots=3,
        trading_open=False,
    )

    if _actions(
        closed
    ) != (
        "market.hold",
    ):
        raise AssertionError(
            "closed trading should expose only HOLD in this slice"
        )

    print("closed_market: hold_only: OK")

    # --------------------------------------------------------
    # Instrument-side permissions are mechanics, not Policy
    # --------------------------------------------------------

    restricted = MarketInstrument(
        instrument_id="RESTRICTED",
        ticker="REST",
        name="Restricted",
        currency="rub",
        lot_size=1,
        buy_available=False,
        sell_available=False,
    )

    restricted_affordances = provider.build(
        restricted,
        world_id="market:tbank:sandbox",
        now=now + 4.0,
        reference_price=Decimal("10"),
        available_cash=Decimal("10000"),
        held_lots=10,
        trading_open=True,
    )

    if _actions(
        restricted_affordances
    ) != (
        "market.hold",
    ):
        raise AssertionError(
            "instrument permissions were ignored"
        )

    print("instrument_permissions: OK")

    # --------------------------------------------------------
    # IDs must describe concrete current opportunities
    # --------------------------------------------------------

    ids = tuple(
        affordance.id
        for affordance in affordances
    )
    if len(ids) != len(set(ids)):
        raise AssertionError(
            "affordance ids are not unique"
        )

    later_ids = {
        affordance.id
        for affordance in provider.build(
            instrument,
            world_id="market:tbank:sandbox",
            now=now + 0.5,
            reference_price=Decimal("250"),
            available_cash=Decimal("10000"),
            held_lots=3,
            trading_open=True,
        )
    }

    if set(ids) & later_ids:
        raise AssertionError(
            "new capability snapshot reused old affordance ids"
        )

    print("snapshot_ids: OK")

    print(
        "CHAIN: Market mechanics -> dynamic Affordances"
    )
    print("MARKET AFFORDANCES SMOKE: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
