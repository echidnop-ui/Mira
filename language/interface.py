from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class LanguageInput:
    transition_id: str
    text: str
    source_kind: str
    source_id: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LanguageRepresentation:
    transition_id: str
    source_kind: str
    vector: tuple[float, ...]
    token_count: int
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExpressionRequest:
    transition_id: str
    cognitive_state: tuple[float, ...]
    max_bytes: int = 192
    temperature: float = 0.8
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RealizedUtterance:
    transition_id: str
    text: str
    token_count: int
    stopped_by_eos: bool
    metadata: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class LanguageOrgan(Protocol):
    @property
    def representation_size(self) -> int:
        ...

    def encode(
        self,
        language_input: LanguageInput,
        *,
        remember_for_learning: bool = True,
    ) -> LanguageRepresentation:
        ...

    def learn_representation(
        self,
        transition_id: str,
        representation_gradient: Sequence[float],
    ) -> bool:
        ...

    def has_trace(self, transition_id: str) -> bool:
        ...

    def discard_trace(self, transition_id: str) -> bool:
        ...

    def realize(self, request: ExpressionRequest) -> RealizedUtterance:
        ...
