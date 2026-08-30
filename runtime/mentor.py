from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
from uuid import uuid4

from core.interface import CognitiveCore, CoreInput, IntegratedCognitiveState
from language.interface import LanguageInput, LanguageOrgan, LanguageRepresentation


@dataclass(frozen=True, slots=True)
class MentorObservationResult:
    """
    Result of bringing one external mentor utterance into current cognition.

    Raw mentor text is intentionally not retained here. The caller may use the
    text while handling the current turn, but this coordinator stores only the
    learned representation and integrated cognitive state.
    """

    transition_id: str
    source_id: str
    language: LanguageRepresentation
    cognition: IntegratedCognitiveState
    raw_text_retained: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)


class MentorIntegrator:
    """
    Mechanical coordinator for ordinary mentor observations.

    Ordinary mentor conversation is NOT automatically language-teacher
    evidence and is NOT automatically autobiographical memory. It enters the
    same temporary cognitive stream as other observations:

        mentor text -> Language.encode -> Core WORK/SLOW/history/workspace

    Whether something is later consolidated into long-term memory, or whether
    a message is an explicit language lesson, belongs to later learning/memory
    policy and must be decided separately.
    """

    def __init__(self, language: LanguageOrgan, core: CognitiveCore) -> None:
        if language.representation_size <= 0:
            raise ValueError("language representation_size must be > 0")
        self.language = language
        self.core = core

    def observe(
        self,
        text: str,
        *,
        source_id: str,
        world_id: str = "desktop.chatgpt",
        transition_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MentorObservationResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("mentor text must be a non-empty str")
        source_id = source_id.strip()
        if not source_id:
            raise ValueError("source_id must not be empty")
        world_id = world_id.strip()
        if not world_id:
            raise ValueError("world_id must not be empty")

        transition = (transition_id or f"mentor-{uuid4().hex}").strip()
        if not transition:
            raise ValueError("transition_id must not be empty")

        observation_metadata = dict(metadata or {})
        observation_metadata.update(
            {
                "mentor_observation": True,
                "temporary_context": True,
                "automatic_teacher_learning": False,
                "automatic_long_term_memory": False,
            }
        )

        language_input = LanguageInput(
            transition_id=transition,
            text=text,
            source_kind="mentor",
            source_id=source_id,
            metadata=observation_metadata,
        )

        # No language trace: ordinary conversation is perception, not an
        # implicit promise to train receptive weights later.
        representation = self.language.encode(
            language_input,
            remember_for_learning=False,
        )

        # No Core learning trace either. The observation still changes the
        # persistent WORK/SLOW/expert state, so it can affect the current and
        # following cognition without altering trainable weights.
        cognition = self.core.integrate(
            CoreInput(
                transition_id=transition,
                world_id=world_id,
                representation=representation.vector,
                metadata=observation_metadata,
            ),
            remember_for_learning=False,
        )

        return MentorObservationResult(
            transition_id=transition,
            source_id=source_id,
            language=representation,
            cognition=cognition,
            raw_text_retained=False,
            metadata=observation_metadata,
        )
