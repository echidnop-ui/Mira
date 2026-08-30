from __future__ import annotations

import os
import time

from worlds.market.affordances import (
    MarketAffordanceProvider,
)
from worlds.market.tbank.sandbox import (
    TBankSandboxObservationBackend,
    TBankSandboxObservationConfig,
)
from worlds.market.tbank.sandbox_capabilities import (
    TBankSandboxCapabilityReader,
)


def _instrument_ids_from_env(
) -> tuple[str, ...]:
    raw_many = os.environ.get(
        "TBANK_INSTRUMENT_IDS",
        "",
    )
    if raw_many.strip():
        result = tuple(
            item.strip()
            for item in raw_many.split(",")
            if item.strip()
        )
        if result:
            return result

    single = os.environ.get(
        "TBANK_INSTRUMENT_ID",
        "SBER_TQBR",
    ).strip()

    return (
        single or "SBER_TQBR",
    )


def main() -> int:
    print(
        "TBANK SANDBOX -> MARKET AFFORDANCES SMOKE"
    )

    instrument_id = (
        _instrument_ids_from_env()[0]
    )

    observation = (
        TBankSandboxObservationBackend(
            TBankSandboxObservationConfig(
                instrument_ids=(
                    instrument_id,
                ),
            )
        )
    )
    capabilities = (
        TBankSandboxCapabilityReader()
    )
    provider = (
        MarketAffordanceProvider()
    )

    try:
        observation.connect()
        capabilities.connect()

        now = time.time()
        frame = observation.poll(
            now
        )

        instruments = {
            item.instrument_id: item
            for item in observation.instruments
        }
        instrument = instruments.get(
            instrument_id
        )

        if instrument is None:
            # T-Bank may normalize the requested alias to its canonical id while
            # retaining only one instrument in this smoke.
            if len(
                observation.instruments
            ) == 1:
                instrument = (
                    observation.instruments[0]
                )
            else:
                raise RuntimeError(
                    "Requested instrument was not resolved uniquely"
                )

        status_events = [
            event
            for event in frame.events
            if (
                event.kind
                == "trading_status"
                and event.instrument_id
                in {
                    instrument_id,
                    instrument.instrument_id,
                }
            )
        ]

        if not status_events:
            raise RuntimeError(
                "Warmup frame contained no normalized trading_status"
            )

        status = max(
            status_events,
            key=lambda event: event.observed_at,
        )

        api_trade_available = bool(
            status.payload.get(
                "api_trade_available",
                False,
            )
        )
        market_order_available = bool(
            status.payload.get(
                "market_order_available",
                False,
            )
        )
        limit_order_available = bool(
            status.payload.get(
                "limit_order_available",
                False,
            )
        )

        trading_open = (
            api_trade_available
            and (
                market_order_available
                or limit_order_available
            )
        )

        limits = (
            capabilities.get_trade_limits(
                instrument_id
            )
        )

        affordances = (
            provider.build_from_limits(
                instrument,
                limits,
                world_id=(
                    "market:tbank:sandbox"
                ),
                now=now,
                trading_open=trading_open,
            )
        )

        actions = tuple(
            affordance.action
            for affordance in affordances
        )

        if not actions:
            raise AssertionError(
                "Market capability provider returned no Affordances"
            )

        if "market.hold" not in actions:
            raise AssertionError(
                "HOLD must remain available in first Market slice"
            )

        if (
            trading_open
            and instrument.buy_available
            and limits.buy_max_lots >= 1
            and "market.buy" not in actions
        ):
            raise AssertionError(
                "broker allows BUY lots but BUY Affordance is absent"
            )

        if (
            trading_open
            and instrument.sell_available
            and limits.sell_max_lots >= 1
            and "market.sell" not in actions
        ):
            raise AssertionError(
                "broker allows SELL lots but SELL Affordance is absent"
            )

        if (
            not trading_open
            and actions != (
                "market.hold",
            )
        ):
            raise AssertionError(
                "closed trading must expose only HOLD"
            )

        if any(
            affordance.constraints.get(
                "limit_source"
            )
            not in {
                "broker_max_lots",
                None,
            }
            for affordance in affordances
        ):
            raise AssertionError(
                "live affordance set lost broker limit provenance"
            )

        print(
            "instrument:",
            instrument.instrument_id,
        )
        print(
            "account:",
            "RESOLVED",
        )
        print(
            "trading_status:",
            status.payload.get(
                "trading_status",
                "",
            ),
        )
        print(
            "api_trade_available:",
            api_trade_available,
        )
        print(
            "market_order_available:",
            market_order_available,
        )
        print(
            "limit_order_available:",
            limit_order_available,
        )
        print(
            "broker_limits:",
            f"buy={limits.buy_max_lots}",
            f"buy_market={limits.buy_max_market_lots}",
            f"sell={limits.sell_max_lots}",
        )
        print(
            "affordances:",
            ", ".join(actions),
        )
        print(
            "broker_limits_to_affordances: OK"
        )
        print(
            "CHAIN: T-Bank sandbox account + trading status "
            "-> MarketTradeLimits -> dynamic Affordances"
        )
        print(
            "TBANK SANDBOX -> MARKET AFFORDANCES SMOKE: OK"
        )

        return 0

    finally:
        capabilities.close()
        observation.close()


if __name__ == "__main__":
    raise SystemExit(main())
