from __future__ import annotations

from typing import Mapping

from core.interface import CognitiveCore, CoreInput, IntegratedCognitiveState
from worlds.market.interface import MarketRepresentation


def integrate_market_representation(
    core: CognitiveCore,
    market_representation: MarketRepresentation,
    *,
    world_id: str = "market",
    memory_context: tuple[float, ...] = (),
    self_context: tuple[float, ...] = (),
    relation_context: tuple[float, ...] = (),
    metadata: Mapping[str, object] | None = None,
    remember_for_learning: bool = True,
) -> IntegratedCognitiveState:
    core_input = CoreInput(
        transition_id=market_representation.transition_id,
        world_id=world_id,
        representation=market_representation.vector,
        memory_context=memory_context,
        self_context=self_context,
        relation_context=relation_context,
        metadata={
            "instrument_id": market_representation.instrument_id,
            "market_as_of": market_representation.as_of,
            "market_uncertainty": market_representation.uncertainty,
            **dict(metadata or {}),
        },
    )

    return core.integrate(
        core_input,
        remember_for_learning=remember_for_learning,
    )
