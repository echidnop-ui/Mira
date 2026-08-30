from __future__ import annotations

import argparse

from worlds.market.input_builder import (
    MarketInputBuilder,
)
from worlds.market.moex_history import (
    MoexFuturesHistoryClient,
    MoexIssError,
)


def _steps(window) -> int:
    if window is None:
        return 0
    return len(window.rows)


def run(
    *,
    secid: str,
    from_date: str,
    till_date: str,
) -> None:
    print("MOEX GAZPROM FUTURES HISTORY SMOKE")
    print("secid:", secid)
    print("range:", from_date, "..", till_date)

    client = MoexFuturesHistoryClient()

    # Metadata is current contract metadata; candle data below is
    # historical market data.
    contract = client.contract(secid)

    print(
        "contract:",
        f"short_name={contract.short_name}",
        f"asset={contract.asset_code}",
        f"board={contract.board_id}",
        f"last_trade={contract.last_trade_date}",
    )
    print(
        "mechanics_current:",
        f"min_step={contract.min_step}",
        f"step_price={contract.step_price}",
        f"rub_per_price_unit={contract.rubles_per_price_unit:.6f}",
        f"initial_margin={contract.initial_margin}",
        f"lot_volume={contract.lot_volume}",
        f"exchange_fee={contract.exchange_buy_sell_fee}",
    )

    frame = client.candle_frame(
        secid,
        from_date=from_date,
        till_date=till_date,
        interval_minutes=1,
        board_id=contract.board_id,
    )

    events = frame.events

    if len(events) < 8:
        raise RuntimeError(
            f"only {len(events)} real candles returned"
        )

    previous = None

    for event in events:
        begin = str(
            event.payload["candle_begin"]
        )
        end = str(
            event.payload["candle_end"]
        )

        if not begin or not end:
            raise RuntimeError(
                "candle boundary metadata is missing"
            )

        if previous is not None:
            if event.observed_at < previous:
                raise RuntimeError(
                    "historical candle events are not causal/ordered"
                )
        previous = event.observed_at

        if event.payload.get("is_complete") is not True:
            raise RuntimeError(
                "historical candle is not marked complete"
            )

    print(
        "candles:",
        len(events),
    )
    print(
        "first:",
        events[0].payload["candle_begin"],
        "->",
        events[0].payload["candle_end"],
    )
    print(
        "last:",
        events[-1].payload["candle_begin"],
        "->",
        events[-1].payload["candle_end"],
    )
    print(
        "causal_timestamp_rule:",
        "completed candle exposed at candle END: OK",
    )

    builder = MarketInputBuilder()
    builder.ingest_frame(frame)

    market_input = builder.build(
        secid,
        as_of=frame.observed_at,
        metadata={
            "source": "moex:iss",
            "historical": True,
            "real_market_data": True,
            "history_smoke": True,
        },
    )

    windows = {
        "micro": _steps(market_input.micro),
        "short": _steps(market_input.short),
        "medium": _steps(market_input.medium),
        "long": _steps(market_input.long),
    }

    print(
        "windows:",
        " ".join(
            f"{name}={count}"
            for name, count
            in windows.items()
        ),
    )

    telemetry = builder.telemetry()

    print(
        "builder:",
        f"seen={telemetry.events_seen}",
        f"used={telemetry.events_used}",
        f"ignored={telemetry.events_ignored}",
        f"stale={telemetry.stale_temporal_events}",
    )

    if sum(windows.values()) <= 0:
        raise RuntimeError(
            "real MOEX history produced no MarketInput temporal data"
        )

    if telemetry.stale_temporal_events != 0:
        raise RuntimeError(
            "real MOEX replay generated stale temporal events"
        )

    print(
        "CHAIN: MOEX ISS real futures candles "
        "-> MarketBackendEvent -> MarketInputBuilder -> MarketInput"
    )
    print("NO EXECUTION")
    print("MOEX GAZPROM FUTURES HISTORY SMOKE: OK")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--secid",
        default="GZU6",
        help=(
            "MOEX futures SECID. GZU6 is GAZR-9.26 "
            "and is only the current smoke default."
        ),
    )
    parser.add_argument(
        "--from-date",
        default="2026-08-18",
    )
    parser.add_argument(
        "--till-date",
        default="2026-08-18",
    )

    return parser


def main() -> int:
    args = _parser().parse_args()

    try:
        run(
            secid=args.secid,
            from_date=args.from_date,
            till_date=args.till_date,
        )
    except MoexIssError as exc:
        print(
            "MOEX HISTORY ERROR:",
            str(exc),
        )
        return 2
    except Exception as exc:
        print(
            "MOEX GAZPROM FUTURES HISTORY SMOKE FAILED:",
            f"{type(exc).__name__}: {exc}",
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
