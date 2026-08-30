from __future__ import annotations

from core.interface import CognitiveCore, IntegratedCognitiveState
from runtime.market_core import integrate_market_representation
from worlds.market.interface import MarketRepresentation


class MarketCognitionCycle:
    def __init__(self, core: CognitiveCore) -> None:
        self._core = core

    def process(
        self,
        market_representation: MarketRepresentation,
        *,
        remember_for_learning: bool = True,
    ) -> IntegratedCognitiveState:
        integrated_state = integrate_market_representation(
            self._core,
            market_representation,
            remember_for_learning=remember_for_learning,
        )

        if (
            integrated_state.transition_id
            != market_representation.transition_id
        ):
            raise RuntimeError(
                "runtime cycle did not preserve transition_id"
            )

        return integrated_state
