from __future__ import annotations

import argparse
from math import sqrt
import os
import sys
import time

import torch

from worlds.market.input_builder import MarketInputBuilder
from worlds.market.perception.defaults import (
    MarketPerceptionConfig,
    build_default_market_organ,
)
from worlds.market.tbank.sandbox import (
    TBankRestError,
    TBankSandboxObservationBackend,
    TBankSandboxObservationConfig,
)


# ============================================================
# Environment
# ============================================================


def _instrument_ids_from_env() -> tuple[str, ...]:
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


def _require_token() -> None:
    if not os.environ.get(
        "TBANK_INVEST_TOKEN",
        "",
    ).strip():
        raise RuntimeError(
            "TBANK_INVEST_TOKEN is missing from the current environment"
        )


# ============================================================
# Neural smoke configuration
# ============================================================


def _perception_config() -> MarketPerceptionConfig:
    """
    Keep the same heterogeneous architecture while making the diagnostic
    intentionally small enough for a quick real-market forward pass.

    DeepLOB keeps a real minimum-history requirement. The CLI default performs
    enough polls to collect at least 12 real L2 snapshots.
    """

    return MarketPerceptionConfig(
        expert_representation_size=24,
        market_representation_size=32,
        training_horizon_minutes=240,

        expert_max_traces=16,
        router_max_traces=32,
        fusion_max_traces=32,

        expert_learning_rate=1e-3,
        router_learning_rate=1e-3,
        fusion_learning_rate=1e-3,
        gradient_clip=5.0,

        device=None,

        tcn_channels=16,
        tcn_branch_size=16,
        tcn_depth=2,
        tcn_kernel_size=3,
        tcn_dropout=0.0,
        tcn_min_steps=8,

        ssm_hidden_size=16,
        ssm_branch_size=16,
        ssm_depth=2,
        ssm_dropout=0.0,
        ssm_min_steps=8,

        attention_hidden_size=16,
        attention_branch_size=16,
        attention_depth=2,
        attention_heads=4,
        attention_feedforward_multiplier=2,
        attention_dropout=0.0,
        attention_relative_buckets=16,
        attention_min_steps=8,
        attention_max_steps_per_window=64,

        recurrent_hidden_size=16,
        recurrent_branch_size=16,
        recurrent_depth=2,
        recurrent_dropout=0.0,
        recurrent_min_steps=8,
        recurrent_max_steps_per_window=64,

        deeplob_levels=10,
        deeplob_conv_channels=16,
        deeplob_inception_channels=8,
        deeplob_recurrent_size=16,
        deeplob_recurrent_layers=1,
        deeplob_dropout=0.0,
        deeplob_min_snapshots=12,
        deeplob_max_snapshots=24,
        deeplob_max_traces=16,

        router_hidden_size=16,
        router_temperature=1.0,
        router_top_k=None,
    )


# ============================================================
# Helpers
# ============================================================


def _window_steps(window) -> int:
    if window is None:
        return 0

    return len(
        window.rows
    )


def _vector_norm(
    values: tuple[float, ...],
) -> float:
    return sqrt(
        sum(
            value * value
            for value in values
        )
    )


# ============================================================
# Real T-Bank -> Neural Organ smoke
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

    _require_token()

    instrument_ids = (
        _instrument_ids_from_env()
    )

    backend = TBankSandboxObservationBackend(
        TBankSandboxObservationConfig(
            instrument_ids=instrument_ids,
        )
    )

    builder = MarketInputBuilder()

    organ = build_default_market_organ(
        _perception_config()
    )

    print(
        "TBANK SANDBOX -> MARKET PERCEPTION SMOKE"
    )
    print(
        "torch:",
        torch.__version__,
    )
    print(
        "hip:",
        torch.version.hip,
    )
    print(
        "accelerator_available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        print(
            "accelerator:",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "requested_instruments:",
        ", ".join(
            instrument_ids
        ),
    )

    try:
        print(
            "connect: ..."
        )

        backend.connect()

        print(
            "connect: OK"
        )

        total_events = 0
        last_frame = None

        for index in range(
            polls
        ):
            frame = backend.poll(
                time.time()
            )

            last_frame = frame

            builder.ingest_frame(
                frame
            )

            total_events += len(
                frame.events
            )

            print(
                f"poll[{index + 1}/{polls}]:",
                f"events={len(frame.events)}",
                f"total={total_events}",
            )

            if (
                index + 1 < polls
                and sleep_seconds > 0.0
            ):
                time.sleep(
                    sleep_seconds
                )

        if last_frame is None:
            raise RuntimeError(
                "no T-Bank frame was produced"
            )

        if total_events <= 0:
            raise RuntimeError(
                "T-Bank produced no market events"
            )

        builder_telemetry = (
            builder.telemetry()
        )

        print(
            "builder:",
            f"seen={builder_telemetry.events_seen}",
            f"used={builder_telemetry.events_used}",
            f"ignored={builder_telemetry.events_ignored}",
            f"stale={builder_telemetry.stale_temporal_events}",
            f"l2={builder_telemetry.order_book_snapshots}",
        )

        successful = 0

        for instrument_id in (
            instrument_ids
        ):
            market_input = builder.build(
                instrument_id,
                as_of=last_frame.observed_at,
                metadata={
                    "source": "tbank:sandbox",
                    "real_market_data": True,
                    "perception_smoke": True,
                },
            )

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

            print(
                "market_input:",
                instrument_id,
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
                "l2_snapshots:",
                len(
                    market_input.order_book_history
                ),
            )

            transition_id = (
                "tbank:sandbox:perception:"
                f"{instrument_id}:"
                f"{last_frame.observed_at:.6f}"
            )

            representation = organ.encode(
                market_input,
                transition_id=transition_id,
                shared_context=None,
            )

            available = sorted(
                name
                for name, state
                in representation.expert_available.items()
                if state
            )

            unavailable = sorted(
                name
                for name, state
                in representation.expert_available.items()
                if not state
            )

            print(
                "perception_forward: OK"
            )
            print(
                "transition:",
                representation.transition_id,
            )
            print(
                "representation_size:",
                len(
                    representation.vector
                ),
            )
            print(
                "representation_norm:",
                f"{_vector_norm(representation.vector):.6f}",
            )
            print(
                "available:",
                ", ".join(
                    available
                )
                or "none",
            )
            print(
                "unavailable:",
                ", ".join(
                    unavailable
                )
                or "none",
            )
            print(
                "router_weights:",
                " ".join(
                    f"{name}={weight:.4f}"
                    for name, weight
                    in sorted(
                        representation.expert_weights.items()
                    )
                ),
            )

            if not available:
                raise RuntimeError(
                    f"{instrument_id}: no perception expert is available"
                )

            if len(
                representation.vector
            ) != organ.representation_size:
                raise RuntimeError(
                    f"{instrument_id}: MarketRepresentation size mismatch"
                )

            weight_sum = sum(
                representation.expert_weights.values()
            )

            if abs(
                weight_sum - 1.0
            ) > 1e-6:
                raise RuntimeError(
                    f"{instrument_id}: Router weights do not sum to 1"
                )

            successful += 1

        if successful <= 0:
            raise RuntimeError(
                "no real-market MarketRepresentation was produced"
            )

        backend_telemetry = (
            backend.telemetry()
        )

        print(
            "backend_telemetry:",
            f"polls={backend_telemetry.polls}",
            f"requests={backend_telemetry.requests}",
            f"emitted={backend_telemetry.emitted_events}",
            f"duplicates={backend_telemetry.duplicate_events}",
        )

        print(
            "TBANK PERCEPTION SMOKE: OK"
        )

    finally:
        organ.reset()
        backend.close()


# ============================================================
# CLI
# ============================================================


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch real T-Bank sandbox market observations, "
            "build MarketInput, and run the heterogeneous "
            "Market Neural Organ forward pass on the configured GPU."
        )
    )

    parser.add_argument(
        "--polls",
        type=int,
        default=14,
        help=(
            "number of market polls; default 14 collects "
            "enough real L2 snapshots for DeepLOB"
        ),
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=1.1,
        help=(
            "seconds between polls (default: 1.1)"
        ),
    )

    return parser


def main() -> int:
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
            "TBANK PERCEPTION SMOKE FAILED:",
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
