from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from contracts.cognition import ActionIntent, Affordance
from core.interface import CognitiveCore, CoreInput, IntegratedCognitiveState
from policy.interface import ActionPolicy
from worlds.desktop.representation import (
    DesktopRepresentation,
    represent_desktop_frame,
)
from worlds.interface import WorldFrame


@dataclass(frozen=True, slots=True)
class DesktopDecision:
    transition_id: str
    frame: WorldFrame
    representation: DesktopRepresentation
    cognitive_state: IntegratedCognitiveState
    affordances: tuple[Affordance, ...]
    intent: ActionIntent


class DesktopDecisionCycle:
    """
    Runtime bridge from Desktop external state into shared Heiwa cognition.

        WorldFrame
        -> DesktopRepresentation
        -> shared Core
        -> dynamic Affordances
        -> shared Policy
        -> ActionIntent

    This intentionally stops before action parameterization/execution.
    """

    def __init__(
        self,
        core: CognitiveCore,
        policy: ActionPolicy,
    ) -> None:
        self._core = core
        self._policy = policy

    def process(
        self,
        frame: WorldFrame,
        *,
        transition_id: str,
        now: float,
        remember_core_for_learning: bool = True,
        exploration_temperature: float | None = None,
    ) -> DesktopDecision:
        transition_id = transition_id.strip()
        if not transition_id:
            raise ValueError("transition_id must not be empty")

        now = float(now)
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        affordances = tuple(frame.affordances)
        if not affordances:
            raise ValueError(
                "DesktopDecisionCycle requires at least one Affordance"
            )

        self._validate_affordances(
            affordances,
            world_id=frame.world,
            now=now,
        )

        representation = represent_desktop_frame(
            frame,
            transition_id=transition_id,
            size=self._core.representation_size,
        )

        cognitive_state = self._core.integrate(
            CoreInput(
                transition_id=transition_id,
                world_id=frame.world,
                representation=representation.vector,
                metadata={
                    "desktop_as_of": representation.as_of,
                    "desktop_observation_count": len(frame.observations),
                    "desktop_affordance_count": len(frame.affordances),
                },
            ),
            remember_for_learning=remember_core_for_learning,
        )

        if cognitive_state.transition_id != transition_id:
            raise RuntimeError("Core lost Desktop transition_id")
        if cognitive_state.world_id != frame.world:
            raise RuntimeError("Core lost Desktop world_id")

        if exploration_temperature is None:
            intent = self._policy.choose(
                cognitive_state, affordances, now=now,
            )
        else:
            try:
                intent = self._policy.choose(
                    cognitive_state, affordances, now=now,
                    exploration_temperature=exploration_temperature,
                )
            except TypeError as exc:
                # Backward compatibility for narrow test/adapter policies that
                # implement the older ActionPolicy surface. Production
                # DynamicAffordancePolicy supports exploration_temperature.
                if "exploration_temperature" not in str(exc):
                    raise
                intent = self._policy.choose(
                    cognitive_state, affordances, now=now,
                )

        self._validate_intent(
            intent,
            affordances,
            transition_id=transition_id,
        )

        return DesktopDecision(
            transition_id=transition_id,
            frame=frame,
            representation=representation,
            cognitive_state=cognitive_state,
            affordances=affordances,
            intent=intent,
        )

    @staticmethod
    def _validate_affordances(
        affordances: Sequence[Affordance],
        *,
        world_id: str,
        now: float,
    ) -> None:
        seen: set[str] = set()
        active = 0

        for affordance in affordances:
            if not affordance.id:
                raise ValueError("Affordance.id must not be empty")
            if affordance.id in seen:
                raise ValueError(
                    f"duplicate Affordance id: {affordance.id!r}"
                )
            seen.add(affordance.id)

            if affordance.world != world_id:
                raise ValueError(
                    "Desktop frame contains Affordance for another World"
                )
            if not affordance.action:
                raise ValueError("Affordance.action must not be empty")

            if affordance.expires_at is None:
                active += 1
            else:
                expires_at = float(affordance.expires_at)
                if not math.isfinite(expires_at):
                    raise ValueError("Affordance.expires_at must be finite")
                if expires_at > now:
                    active += 1

        if active == 0:
            raise ValueError("all Desktop Affordances are expired")

    @staticmethod
    def _validate_intent(
        intent: ActionIntent,
        affordances: Sequence[Affordance],
        *,
        transition_id: str,
    ) -> None:
        if intent.transition_id != transition_id:
            raise RuntimeError("Policy lost Desktop transition_id")

        chosen = {
            affordance.id: affordance
            for affordance in affordances
        }.get(intent.affordance_id)

        if chosen is None:
            raise RuntimeError(
                "Policy selected an Affordance that was not offered"
            )

        if intent.world != chosen.world:
            raise RuntimeError(
                "ActionIntent.world differs from chosen Affordance.world"
            )
        if intent.action != chosen.action:
            raise RuntimeError(
                "ActionIntent.action differs from chosen Affordance.action"
            )
