from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class CoreInput:
    """World-agnostic normalized input to the persistent shared Core."""

    transition_id: str
    world_id: str
    representation: tuple[float, ...]
    memory_context: tuple[float, ...] = ()
    self_context: tuple[float, ...] = ()
    relation_context: tuple[float, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IntegratedCognitiveState:
    """Shared cognitive representation produced for one causal transition."""

    transition_id: str
    world_id: str
    vector: tuple[float, ...]
    workspace_vector: tuple[float, ...]
    temporal_vector: tuple[float, ...]
    context_vector: tuple[float, ...]
    novelty: float
    state_change: float
    metadata: Mapping[str, object] = field(default_factory=dict)


class CognitiveCore(Protocol):
    @property
    def representation_size(self) -> int:
        ...

    def integrate(
        self,
        core_input: CoreInput,
        *,
        remember_for_learning: bool = True,
    ) -> IntegratedCognitiveState:
        ...

    def learn(
        self,
        transition_id: str,
        output_gradient: Sequence[float],
    ) -> bool:
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
