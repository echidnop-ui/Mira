from __future__ import annotations

import argparse
import sys

import torch

from worlds.market.input_builder import MarketInputBuilder
from worlds.market.moex_history import (
    MoexFuturesHistoryClient,
    MoexIssError,
)
from worlds.market.perception.defaults import (
    build_default_market_organ,
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
    print("MOEX REAL HISTORY -> MARKET ORGAN SMOKE")
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print(
        "accelerator_available:",
        torch.cuda.is_available(),
    )
    if torch.cuda.is_available():
        print(
            "accelerator:",
            torch.cuda.get_device_name(0),
        )

    print("secid:", secid)
    print("range:", from_date, "..", till_date)

    client = MoexFuturesHistoryClient()
    contract = client.contract(secid)

    frame = client.candle_frame(
        secid,
        from_date=from_date,
        till_date=till_date,
        interval_minutes=1,
        board_id=contract.board_id,
    )

    if len(frame.events) < 8:
        raise RuntimeError(
            f"only {len(frame.events)} real candles returned"
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
            "moex_perception_smoke": True,
        },
    )

    windows = {
        "micro": _steps(market_input.micro),
        "short": _steps(market_input.short),
        "medium": _steps(market_input.medium),
        "long": _steps(market_input.long),
    }

    print(
        "candles:",
        len(frame.events),
    )
    print(
        "windows:",
        " ".join(
            f"{name}={count}"
            for name, count in windows.items()
        ),
    )
    print(
        "l2_snapshots:",
        len(market_input.order_book_history),
    )

    transition_id = (
        "moex:history:market-organ:"
        f"{secid}:{frame.observed_at:.6f}"
    )

    organ = build_default_market_organ()

    try:
        representation = organ.encode(
            market_input,
            transition_id=transition_id,
            shared_context=None,
        )

        if representation.transition_id != transition_id:
            raise RuntimeError(
                "Market Organ did not preserve transition_id"
            )

        if representation.instrument_id != secid:
            raise RuntimeError(
                "Market Organ did not preserve instrument_id"
            )

        if len(representation.vector) != 96:
            raise RuntimeError(
                "MarketRepresentation size is not 96"
            )

        available = tuple(
            sorted(
                name
                for name, flag
                in representation.expert_available.items()
                if flag
            )
        )
        unavailable = tuple(
            sorted(
                name
                for name, flag
                in representation.expert_available.items()
                if not flag
            )
        )

        if not available:
            raise RuntimeError(
                "no Market expert accepted real MOEX history"
            )

        telemetry = organ.telemetry()

        print(
            "market_representation_size:",
            len(representation.vector),
        )
        print(
            "available_experts:",
            ", ".join(available),
        )
        print(
            "unavailable_experts:",
            (
                ", ".join(unavailable)
                if unavailable
                else "none"
            ),
        )

        weights = representation.expert_weights
        if weights:
            print(
                "expert_weights:",
                " ".join(
                    f"{name}={float(weight):.6f}"
                    for name, weight
                    in sorted(weights.items())
                ),
            )

        print(
            "representation_norm:",
            f"{telemetry.representation_norm:.6f}",
        )
        print(
            "transition_preserved: OK",
        )
        print(
            "real_history_to_neural_perception: OK",
        )
        print(
            "CHAIN: MOEX ISS real GAZR futures candles "
            "-> MarketInputBuilder -> Market Organ "
            "-> MarketRepresentation[96]"
        )
        print("NO LEARNING")
        print("NO EXECUTION")
        print(
            "MOEX REAL HISTORY -> MARKET ORGAN SMOKE: OK"
        )

    finally:
        organ.reset()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--secid",
        default="GZU6",
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
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            "MOEX REAL HISTORY -> MARKET ORGAN SMOKE FAILED:",
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
