from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from contracts.cognition import ActionIntent, Affordance
from core.interface import IntegratedCognitiveState
from policy.interface import ActionPolicy
from runtime.cycle import MarketCognitionCycle
from worlds.market.interface import MarketRepresentation


@dataclass(frozen=True, slots=True)
class MarketDecision:
    """
    One runtime-local cognitive decision over Market affordances.

    This object is not a CompletedAction and does not claim that anything was
    sent to the broker. It only preserves the causal bundle needed by the next
    execution stage.
    """

    transition_id: str
    market_representation: MarketRepresentation
    cognitive_state: IntegratedCognitiveState
    affordances: tuple[Affordance, ...]
    intent: ActionIntent


class MarketDecisionCycle:
    """
    Minimal runtime bridge:

        MarketRepresentation
        -> persistent shared Core
        -> IntegratedCognitiveState
        -> dynamic Affordances
        -> shared Policy
        -> ActionIntent

    Responsibilities intentionally stop at ActionIntent. Execution belongs to
    the Market World/execution adapter.
    """

    def __init__(
        self,
        cognition_cycle: MarketCognitionCycle,
        policy: ActionPolicy,
    ) -> None:
        self._cognition_cycle = cognition_cycle
        self._policy = policy

    def process(
        self,
        market_representation: MarketRepresentation,
        affordances: Sequence[Affordance],
        *,
        now: float,
        remember_core_for_learning: bool = True,
    ) -> MarketDecision:
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        candidates = tuple(affordances)
        if not candidates:
            raise ValueError(
                "MarketDecisionCycle requires at least one Affordance"
            )

        self._validate_affordances(
            candidates,
            now=now,
        )

        cognitive_state = self._cognition_cycle.process(
            market_representation,
            remember_for_learning=remember_core_for_learning,
        )

        if (
            cognitive_state.transition_id
            != market_representation.transition_id
        ):
            raise RuntimeError(
                "Core/cognition cycle lost Market transition_id"
            )

        intent = self._policy.choose(
            cognitive_state,
            candidates,
            now=now,
        )

        self._validate_intent(
            intent,
            candidates,
            transition_id=market_representation.transition_id,
        )

        return MarketDecision(
            transition_id=market_representation.transition_id,
            market_representation=market_representation,
            cognitive_state=cognitive_state,
            affordances=candidates,
            intent=intent,
        )

    @staticmethod
    def _validate_affordances(
        affordances: tuple[Affordance, ...],
        *,
        now: float,
    ) -> None:
        seen_ids: set[str] = set()
        active_count = 0

        for affordance in affordances:
            if not affordance.id:
                raise ValueError(
                    "Affordance.id must not be empty"
                )
            if affordance.id in seen_ids:
                raise ValueError(
                    "Affordance ids must be unique inside one decision cycle: "
                    f"{affordance.id!r}"
                )
            seen_ids.add(affordance.id)

            if not affordance.world:
                raise ValueError(
                    f"Affordance {affordance.id!r} has empty world"
                )
            if not affordance.action:
                raise ValueError(
                    f"Affordance {affordance.id!r} has empty action"
                )

            offered_at = float(affordance.offered_at)
            if not math.isfinite(offered_at):
                raise ValueError(
                    f"Affordance {affordance.id!r} has non-finite offered_at"
                )

            if affordance.expires_at is None:
                active_count += 1
                continue

            expires_at = float(
                affordance.expires_at
            )
            if not math.isfinite(expires_at):
                raise ValueError(
                    f"Affordance {affordance.id!r} has non-finite expires_at"
                )
            if expires_at > now:
                active_count += 1

        # Check before mutating shared cognition so an entirely stale capability
        # set does not create a Core learning trace for a decision that cannot
        # actually be made.
        if active_count == 0:
            raise ValueError(
                "All supplied Affordances are expired"
            )

    @staticmethod
    def _validate_intent(
        intent: ActionIntent,
        affordances: tuple[Affordance, ...],
        *,
        transition_id: str,
    ) -> None:
        if intent.transition_id != transition_id:
            raise RuntimeError(
                "Policy lost transition_id"
            )

        chosen = {
            affordance.id: affordance
            for affordance in affordances
        }.get(intent.affordance_id)

        if chosen is None:
            raise RuntimeError(
                "Policy returned ActionIntent for an Affordance that was not "
                "offered in this decision cycle"
            )

        if intent.world != chosen.world:
            raise RuntimeError(
                "ActionIntent.world differs from the chosen Affordance.world"
            )

        if intent.action != chosen.action:
            raise RuntimeError(
                "ActionIntent.action differs from the chosen Affordance.action"
            )
