from __future__ import annotations

import os
import sys
import time

from core.personal import PersonalCognitiveCore
from runtime.cycle import MarketCognitionCycle
from worlds.market.input_builder import MarketInputBuilder
from worlds.market.perception.defaults import build_default_market_organ
from worlds.market.tbank.sandbox import (
    TBankRestError,
    TBankSandboxObservationBackend,
    TBankSandboxObservationConfig,
)


def _instrument_ids() -> tuple[str, ...]:
    raw = (
        os.environ.get("TBANK_INSTRUMENT_IDS", "").strip()
        or os.environ.get("TBANK_INSTRUMENT_ID", "").strip()
    )

    result = tuple(
        item.strip()
        for item in raw.split(",")
        if item.strip()
    )

    if not result:
        raise RuntimeError(
            "set TBANK_INSTRUMENT_ID, for example SBER_TQBR"
        )

    return result


def run() -> None:
    if not os.environ.get("TBANK_INVEST_TOKEN", "").strip():
        raise RuntimeError("TBANK_INVEST_TOKEN is missing")

    instrument_ids = _instrument_ids()
    instrument_id = instrument_ids[0]

    backend = TBankSandboxObservationBackend(
        TBankSandboxObservationConfig(
            instrument_ids=instrument_ids,
        )
    )
    builder = MarketInputBuilder()
    organ = build_default_market_organ()
    core = PersonalCognitiveCore()
    cycle = MarketCognitionCycle(core)

    representations = []
    states = []
    available_experts: set[str] = set()

    print("TBANK -> MARKET -> CORE SMOKE")

    try:
        backend.connect()

        for poll_index in range(2):
            frame = backend.poll(time.time())
            builder.ingest_frame(frame)

            if frame.events:
                try:
                    market_input = builder.build(
                        instrument_id,
                        as_of=frame.observed_at,
                        metadata={
                            "source": "tbank:sandbox",
                            "real_market_data": True,
                            "runtime_cycle_smoke": True,
                        },
                    )
                except ValueError:
                    market_input = None

                if market_input is not None:
                    transition_id = (
                        "tbank:sandbox:market-core:"
                        f"{instrument_id}:"
                        f"{frame.observed_at:.6f}:"
                        f"{poll_index + 1}"
                    )

                    representation = organ.encode(
                        market_input,
                        transition_id=transition_id,
                        shared_context=None,
                    )

                    available = {
                        name
                        for name, is_available
                        in representation.expert_available.items()
                        if is_available
                    }

                    if not available:
                        raise RuntimeError(
                            "no Market expert is available"
                        )

                    if len(representation.vector) != 96:
                        raise RuntimeError(
                            "MarketRepresentation size is not 96"
                        )

                    state = cycle.process(
                        representation,
                        remember_for_learning=False,
                    )

                    if len(state.vector) != 96:
                        raise RuntimeError(
                            "IntegratedCognitiveState size is not 96"
                        )

                    if state.transition_id != representation.transition_id:
                        raise RuntimeError(
                            "transition_id was not preserved"
                        )

                    representations.append(representation)
                    states.append(state)
                    available_experts.update(available)

            if poll_index == 0:
                time.sleep(1.1)

        if not states:
            raise RuntimeError(
                "no real MarketRepresentation reached Core"
            )

        telemetry = core.telemetry()

        if len(states) >= 2:
            if telemetry.forward_calls < 2:
                raise RuntimeError(
                    "Core did not process both real representations"
                )

            if telemetry.state_resets != 0:
                raise RuntimeError(
                    "Core was reset between real representations"
                )

            persistent_core = "OK"
        else:
            persistent_core = "NOT_TESTED_REAL_DATA"

        print("market_representation_size:", len(representations[0].vector))
        print("available_experts:", ", ".join(sorted(available_experts)))
        print("core_state_size:", len(states[0].vector))
        print("transition_preserved: OK")
        print("persistent_core:", persistent_core)
        print("TBANK -> MARKET -> CORE SMOKE: OK")

    finally:
        try:
            organ.reset()
        finally:
            backend.close()


def main() -> int:
    try:
        run()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except TBankRestError as exc:
        print("TBANK REST ERROR:", str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            "TBANK -> MARKET -> CORE SMOKE FAILED:",
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
