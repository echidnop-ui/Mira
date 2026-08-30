from __future__ import annotations

import argparse
from collections import Counter
import os
import sys
import time

from worlds.market.input_builder import MarketInputBuilder
from worlds.market.tbank.sandbox import (
    TBankRestError,
    TBankSandboxObservationBackend,
    TBankSandboxObservationConfig,
)


# ============================================================
# Helpers
# ============================================================


def _required_env(
    name: str,
) -> str:
    value = os.environ.get(
        name,
        "",
    ).strip()

    if not value:
        raise RuntimeError(
            f"required environment variable {name} is missing"
        )

    return value


def _instrument_ids_from_env(
) -> tuple[
    str,
    ...,
]:
    """
    Accept either one instrument or a comma-separated small set.

    Preferred:
        TBANK_INSTRUMENT_ID=SBER_TQBR

    Also accepted:
        TBANK_INSTRUMENT_IDS=SBER_TQBR,GAZP_TQBR
    """

    many = os.environ.get(
        "TBANK_INSTRUMENT_IDS",
        "",
    ).strip()

    one = os.environ.get(
        "TBANK_INSTRUMENT_ID",
        "",
    ).strip()

    raw = many or one

    if not raw:
        raise RuntimeError(
            "set TBANK_INSTRUMENT_ID first, for example:\n"
            "export TBANK_INSTRUMENT_ID='SBER_TQBR'"
        )

    result = tuple(
        item.strip()
        for item in raw.split(",")
        if item.strip()
    )

    if not result:
        raise RuntimeError(
            "no usable T-Bank instrument ids were provided"
        )

    return result


def _window_steps(
    window,
) -> int:
    if window is None:
        return 0

    return len(
        window.rows
    )


def _format_book(
    market_input,
) -> str:
    history = (
        market_input.order_book_history
    )

    if not history:
        return "none"

    latest = history[-1]

    best_bid = (
        latest.bids[0].price
        if latest.bids
        else None
    )

    best_ask = (
        latest.asks[0].price
        if latest.asks
        else None
    )

    return (
        f"snapshots={len(history)} "
        f"best_bid={best_bid!r} "
        f"best_ask={best_ask!r}"
    )


# ============================================================
# Smoke
# ============================================================


def run(
    *,
    polls: int,
    sleep_seconds: float,
) -> None:
    if polls <= 0:
        raise ValueError(
            "polls must be > 0"
        )

    if sleep_seconds < 0.0:
        raise ValueError(
            "sleep_seconds must be >= 0"
        )

    # Do not print the token.
    _required_env(
        "TBANK_INVEST_TOKEN"
    )

    instrument_ids = (
        _instrument_ids_from_env()
    )

    config = (
        TBankSandboxObservationConfig(
            instrument_ids=instrument_ids,
        )
    )

    backend = (
        TBankSandboxObservationBackend(
            config
        )
    )

    builder = MarketInputBuilder()

    print(
        "TBANK SANDBOX OBSERVATION SMOKE"
    )
    print(
        "backend:",
        backend.backend_id,
    )
    print(
        "mode:",
        backend.mode.value,
    )
    print(
        "requested_instruments:",
        ", ".join(
            instrument_ids
        ),
    )
    print(
        "token:",
        "present",
    )

    try:
        print(
            "connect: ..."
        )

        backend.connect()

        print(
            "connect: OK"
        )

        if not backend.instruments:
            raise RuntimeError(
                "T-Bank returned no normalized instruments"
            )

        for instrument in (
            backend.instruments
        ):
            print(
                "instrument:",
                f"id={instrument.instrument_id}",
                f"ticker={instrument.ticker}",
                f"name={instrument.name!r}",
                f"currency={instrument.currency}",
                f"lot={instrument.lot_size}",
            )

        all_counts: Counter[
            str
        ] = Counter()

        total_events = 0

        for poll_index in range(
            polls
        ):
            now = time.time()

            print(
                f"poll[{poll_index + 1}/{polls}]: ..."
            )

            frame = backend.poll(
                now
            )

            counts = Counter(
                event.kind
                for event in frame.events
            )

            all_counts.update(
                counts
            )

            total_events += len(
                frame.events
            )

            builder.ingest_frame(
                frame
            )

            rendered_counts = (
                " ".join(
                    f"{kind}={count}"
                    for kind, count
                    in sorted(
                        counts.items()
                    )
                )
                or "none"
            )

            print(
                f"poll[{poll_index + 1}/{polls}]: OK",
                f"events={len(frame.events)}",
                rendered_counts,
            )

            if (
                poll_index + 1
                < polls
                and sleep_seconds > 0.0
            ):
                time.sleep(
                    sleep_seconds
                )

        if total_events <= 0:
            raise RuntimeError(
                "T-Bank connection worked but no market events "
                "were emitted. Try an actively traded instrument "
                "or repeat during a trading session."
            )

        print(
            "events_total:",
            total_events,
        )
        print(
            "event_kinds:",
            " ".join(
                f"{kind}={count}"
                for kind, count
                in sorted(
                    all_counts.items()
                )
            ),
        )

        builder_telemetry = (
            builder.telemetry()
        )

        print(
            "builder_events:",
            f"seen={builder_telemetry.events_seen}",
            f"used={builder_telemetry.events_used}",
            f"ignored={builder_telemetry.events_ignored}",
            f"stale={builder_telemetry.stale_temporal_events}",
        )

        usable_inputs = 0

        for requested_id in (
            instrument_ids
        ):
            try:
                market_input = (
                    builder.build(
                        requested_id,
                        as_of=time.time(),
                        metadata={
                            "source": (
                                "tbank:sandbox"
                            ),
                            "smoke": True,
                        },
                    )
                )
            except ValueError as exc:
                print(
                    "market_input:",
                    requested_id,
                    "UNAVAILABLE",
                    str(exc),
                )
                continue

            steps = {
                "micro": _window_steps(
                    market_input.micro
                ),
                "short": _window_steps(
                    market_input.short
                ),
                "medium": _window_steps(
                    market_input.medium
                ),
                "long": _window_steps(
                    market_input.long
                ),
            }

            temporal_total = sum(
                steps.values()
            )

            if temporal_total <= 0:
                raise RuntimeError(
                    f"{requested_id}: MarketInput has no temporal data"
                )

            usable_inputs += 1

            print(
                "market_input:",
                requested_id,
                "OK",
            )
            print(
                "windows:",
                " ".join(
                    f"{name}={count}"
                    for name, count
                    in steps.items()
                ),
            )
            print(
                "orderbook:",
                _format_book(
                    market_input
                ),
            )

        if usable_inputs <= 0:
            raise RuntimeError(
                "no usable MarketInput was built"
            )

        telemetry = (
            backend.telemetry()
        )

        print(
            "backend_telemetry:",
            f"polls={telemetry.polls}",
            f"requests={telemetry.requests}",
            f"emitted={telemetry.emitted_events}",
            f"duplicates={telemetry.duplicate_events}",
        )

        print(
            "TBANK OBSERVATION SMOKE: OK"
        )

    finally:
        backend.close()


# ============================================================
# CLI
# ============================================================


def _parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read real market observations from the "
            "T-Bank sandbox endpoint and feed them into "
            "MarketInputBuilder. No orders are submitted."
        )
    )

    parser.add_argument(
        "--polls",
        type=int,
        default=3,
        help=(
            "number of observation polls "
            "(default: 3)"
        ),
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=1.2,
        help=(
            "seconds between polls "
            "(default: 1.2)"
        ),
    )

    return parser


def main(
) -> int:
    args = _parser().parse_args()

    try:
        run(
            polls=args.polls,
            sleep_seconds=args.sleep,
        )

    except KeyboardInterrupt:
        print(
            "\ninterrupted",
            file=sys.stderr,
        )
        return 130

    except TBankRestError as exc:
        print(
            "TBANK REST ERROR:",
            str(exc),
            file=sys.stderr,
        )
        return 2

    except Exception as exc:
        print(
            "TBANK SMOKE FAILED:",
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
