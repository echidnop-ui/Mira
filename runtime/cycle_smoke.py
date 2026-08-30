from __future__ import annotations

from core.personal import PersonalCognitiveCore
from runtime.cycle import MarketCognitionCycle
from worlds.market.interface import MarketRepresentation


def _representation(
    transition_id: str,
    *,
    reverse: bool,
) -> MarketRepresentation:
    values = range(96, 0, -1) if reverse else range(1, 97)

    return MarketRepresentation(
        transition_id=transition_id,
        instrument_id="SYNTHETIC_TQBR",
        as_of=0.0,
        vector=tuple(
            float(value) / 96.0
            for value in values
        ),
    )


def main() -> int:
    core = PersonalCognitiveCore()
    cycle = MarketCognitionCycle(core)

    first_input = _representation(
        "market-cycle-smoke-1",
        reverse=False,
    )
    second_input = _representation(
        "market-cycle-smoke-2",
        reverse=True,
    )

    first_state = cycle.process(
        first_input,
        remember_for_learning=False,
    )
    second_state = cycle.process(
        second_input,
        remember_for_learning=False,
    )

    if first_state.transition_id != first_input.transition_id:
        raise AssertionError("first transition_id was not preserved")

    if second_state.transition_id != second_input.transition_id:
        raise AssertionError("second transition_id was not preserved")

    core.reset_state(clear_learning_traces=True)

    reset_state = cycle.process(
        second_input,
        remember_for_learning=False,
    )

    difference = max(
        abs(persistent - reset)
        for persistent, reset in zip(
            second_state.vector,
            reset_state.vector,
        )
    )

    if difference <= 1e-7:
        raise AssertionError(
            "second output did not depend on persistent Core state"
        )

    print("transition_ids preserved: OK")
    print("persistent Core state across cycles: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
