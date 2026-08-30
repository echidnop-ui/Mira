from __future__ import annotations

from core.personal import PersonalCognitiveCore
from runtime.market_core import integrate_market_representation
from worlds.market.interface import MarketRepresentation


def main() -> int:
    transition_id = "market-core-integration-smoke"
    market_representation = MarketRepresentation(
        transition_id=transition_id,
        instrument_id="SYNTHETIC_TQBR",
        as_of=0.0,
        vector=tuple(
            float(index) / 96.0
            for index in range(96)
        ),
    )

    core = PersonalCognitiveCore()
    integrated_state = integrate_market_representation(
        core,
        market_representation,
        remember_for_learning=False,
    )

    if len(market_representation.vector) != 96:
        raise AssertionError("MarketRepresentation must contain 96 values")

    if len(integrated_state.vector) != 96:
        raise AssertionError("IntegratedCognitiveState must contain 96 values")

    if integrated_state.transition_id != transition_id:
        raise AssertionError("transition_id was not preserved")

    print(
        "MarketRepresentation[96]"
        " -> PersonalCognitiveCore"
        " -> IntegratedCognitiveState[96]: OK"
    )
    print(
        f"transition_id={integrated_state.transition_id}: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
