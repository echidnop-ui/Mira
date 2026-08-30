from __future__ import annotations

from typing import Protocol, Sequence

from contracts.cognition import ActionIntent, Affordance
from core.interface import IntegratedCognitiveState


class ActionPolicy(Protocol):
    """Shared trainable selector over a dynamic set of World affordances."""

    def choose(
        self,
        state: IntegratedCognitiveState,
        affordances: Sequence[Affordance],
        *,
        now: float,
        remember_for_learning: bool = True,
        exploration_temperature: float | None = None,
    ) -> ActionIntent:
        ...


class TrainableActionPolicy(ActionPolicy, Protocol):
    def learn(self, transition_id: str, advantage: float) -> bool:
        ...

    def has_trace(self, transition_id: str) -> bool:
        """Return whether causal learning credit can still consume this trace."""
        ...

    def discard_trace(self, transition_id: str) -> bool:
        ...

    def export_causal_snapshot(
        self,
        generation_id: str,
        transition_ids: Sequence[str],
    ) -> bytes:
        """Serialize exact private state needed to resume delayed credit."""
        ...

    def restore_causal_snapshot(
        self,
        snapshot: bytes,
        expected_generation_id: str,
        expected_transition_ids: Sequence[str],
    ) -> None:
        """Atomically restore a generation-bound delayed-credit snapshot."""
        ...
