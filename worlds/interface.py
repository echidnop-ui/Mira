from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from contracts.cognition import (
    ActionAttempt,
    ActionIntent,
    Affordance,
    CompletedAction,
    Consequence,
    Observation,
)


@dataclass(frozen=True, slots=True)
class WorldFrame:
    """
    One normalized snapshot/update produced by a World adapter.

    A WorldFrame contains external reality only. It does not choose actions
    and does not mutate shared cognition by itself.
    """

    world: str
    observed_at: float

    observations: tuple[Observation, ...] = ()
    affordances: tuple[Affordance, ...] = ()
    completed_actions: tuple[CompletedAction, ...] = ()
    consequences: tuple[Consequence, ...] = ()


class WorldContractError(ValueError):
    """Raised when a World violates the shared World contract."""


@runtime_checkable
class WorldAdapter(Protocol):
    """
    Public contract implemented by every external Heiwa World.

    Examples:
        Market
        Forum
        Game
        VTuber
        Desktop

    A World adapter:
        - observes its environment;
        - exposes currently available Affordances;
        - executes an already-chosen ActionIntent;
        - reports attempts/completions/consequences.

    A World adapter does NOT make the cognitive decision itself.
    """

    @property
    def world_id(self) -> str:
        ...

    def poll(self, now: float) -> WorldFrame:
        """
        Observe the external World and return normalized events/capabilities.
        """
        ...

    def execute(
        self,
        intent: ActionIntent,
        now: float,
    ) -> ActionAttempt:
        """
        Attempt to execute an already-chosen intent.
        """
        ...

    def close(self) -> None:
        ...


def validate_intent_for_world(
    intent: ActionIntent,
    *,
    world_id: str,
) -> None:
    """
    Mechanical validation that an intent is being sent to the World it
    targets.

    This function does not decide whether the action is cognitively good.
    """

    if not world_id:
        raise WorldContractError("world_id must not be empty")

    if intent.world != world_id:
        raise WorldContractError(
            f"Intent targets world {intent.world!r}, "
            f"but adapter is {world_id!r}"
        )

    if not intent.action:
        raise WorldContractError("Intent action must not be empty")

    if not intent.affordance_id:
        raise WorldContractError(
            "Intent must reference the Affordance that enabled it"
        )

    if not intent.transition_id:
        raise WorldContractError(
            "Intent must preserve transition_id"
        )


def validate_frame(
    frame: WorldFrame,
    *,
    world_id: str,
) -> None:
    """
    Validate shared causal/world identity invariants for a WorldFrame.

    This deliberately checks contracts, not domain semantics.
    """

    if not world_id:
        raise WorldContractError("world_id must not be empty")

    if frame.world != world_id:
        raise WorldContractError(
            f"Frame belongs to world {frame.world!r}, "
            f"expected {world_id!r}"
        )

    for observation in frame.observations:
        if observation.world != world_id:
            raise WorldContractError(
                f"Observation {observation.id!r} belongs to "
                f"{observation.world!r}, expected {world_id!r}"
            )

    for affordance in frame.affordances:
        if affordance.world != world_id:
            raise WorldContractError(
                f"Affordance {affordance.id!r} belongs to "
                f"{affordance.world!r}, expected {world_id!r}"
            )

    for action in frame.completed_actions:
        if action.world != world_id:
            raise WorldContractError(
                f"CompletedAction {action.id!r} belongs to "
                f"{action.world!r}, expected {world_id!r}"
            )

        if not action.transition_id:
            raise WorldContractError(
                f"CompletedAction {action.id!r} lost transition_id"
            )

    for consequence in frame.consequences:
        if consequence.world != world_id:
            raise WorldContractError(
                f"Consequence {consequence.id!r} belongs to "
                f"{consequence.world!r}, expected {world_id!r}"
            )

        if not consequence.transition_id:
            raise WorldContractError(
                f"Consequence {consequence.id!r} lost transition_id"
            )
