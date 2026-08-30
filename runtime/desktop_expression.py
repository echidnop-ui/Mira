from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence, runtime_checkable

from contracts.cognition import ActionIntent
from language.interface import (
    ExpressionRequest,
    LanguageInput,
    LanguageOrgan,
    RealizedUtterance,
)
from runtime.desktop import DesktopDecision


TYPE_TEXT_ACTION = "desktop.type_text"


class EmptyExpressionError(RuntimeError):
    """Language produced no text for a selected linguistic Desktop action."""


@runtime_checkable
class TrainableExpressionLanguage(LanguageOrgan, Protocol):
    """
    Optional explicit-teacher boundary for expressive language learning.

    Ordinary mentor conversation must not call this automatically.
    """

    def learn_expression(
        self,
        cognitive_state: Sequence[float],
        target: LanguageInput,
    ) -> float:
        ...


@dataclass(frozen=True, slots=True)
class DesktopExpression:
    """
    Result of language parameterization after Policy selected desktop.type_text.

    Policy still owns the action choice. Language owns the concrete utterance.
    """

    transition_id: str
    utterance: RealizedUtterance
    intent: ActionIntent


@dataclass(frozen=True, slots=True)
class ExpressionTeachingResult:
    transition_id: str
    source_kind: str
    loss: float


class DesktopExpressionBridge:
    """
    Narrow boundary:

        DesktopDecision
        -> Policy already selected desktop.type_text
        -> Language.realize(cognitive_state)
        -> parameterized ActionIntent
        -> DesktopWorldAdapter.execute(...)

    It does NOT:
    - decide whether Heiwa should type;
    - execute Desktop actions;
    - treat ordinary mentor text as a teacher target;
    - copy Affordance payload into ActionIntent.parameters;
    - award learning credit.

    Explicit `teach(...)` is early supervised scaffolding for language only.
    """

    def __init__(
        self,
        language: LanguageOrgan,
        *,
        max_bytes: int = 192,
        temperature: float = 0.8,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self.language = language
        self.max_bytes = int(max_bytes)
        self.temperature = float(temperature)

    def realize(
        self,
        decision: DesktopDecision,
        *,
        max_bytes: int | None = None,
        temperature: float | None = None,
    ) -> DesktopExpression:
        intent = decision.intent

        if intent.transition_id != decision.transition_id:
            raise ValueError("DesktopDecision intent lost transition_id")
        if intent.action != TYPE_TEXT_ACTION:
            raise ValueError(
                "DesktopExpressionBridge can parameterize only "
                f"{TYPE_TEXT_ACTION!r}; got {intent.action!r}"
            )
        if intent.parameters:
            raise ValueError(
                "Policy-created ActionIntent already contains parameters; "
                "parameterization boundary was violated"
            )

        requested_max_bytes = (
            self.max_bytes if max_bytes is None else int(max_bytes)
        )
        requested_temperature = (
            self.temperature if temperature is None else float(temperature)
        )
        if requested_max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        if requested_temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        utterance = self.language.realize(
            ExpressionRequest(
                transition_id=decision.transition_id,
                cognitive_state=tuple(
                    float(value)
                    for value in decision.cognitive_state.vector
                ),
                max_bytes=requested_max_bytes,
                temperature=requested_temperature,
                metadata={
                    "world": decision.frame.world,
                    "action": intent.action,
                    "affordance_id": intent.affordance_id,
                    "selected_by_policy": True,
                    "parameterized_by_language": True,
                },
            )
        )

        if utterance.transition_id != decision.transition_id:
            raise RuntimeError("Language realization lost transition_id")
        if not utterance.text:
            raise EmptyExpressionError(
                "Language produced an empty utterance; no Desktop action executed"
            )

        parameterized = replace(
            intent,
            parameters={"text": utterance.text},
        )

        if parameterized.transition_id != decision.transition_id:
            raise RuntimeError("parameterization lost transition_id")
        if parameterized.affordance_id != intent.affordance_id:
            raise RuntimeError("parameterization changed chosen Affordance")
        if parameterized.action != intent.action:
            raise RuntimeError("parameterization changed chosen action")

        return DesktopExpression(
            transition_id=decision.transition_id,
            utterance=utterance,
            intent=parameterized,
        )

    def teach(
        self,
        decision: DesktopDecision,
        target_text: str,
        *,
        source_kind: str = "mentor",
        source_id: str = "chatgpt-visible-ui",
    ) -> ExpressionTeachingResult:
        """
        Explicit supervised correction.

        The caller must invoke this only when a human/mentor intentionally
        supplies the desired utterance for this exact cognitive state.
        Ordinary mentor observations never arrive here automatically.
        """
        if not isinstance(target_text, str) or not target_text:
            raise ValueError("target_text must be a non-empty str")

        source_kind = source_kind.strip().lower()
        if source_kind not in {"mentor", "human"}:
            raise ValueError(
                "explicit expression teaching requires source_kind "
                "'mentor' or 'human'"
            )
        source_id = source_id.strip()
        if not source_id:
            raise ValueError("source_id must not be empty")

        learner = self.language
        if not isinstance(learner, TrainableExpressionLanguage):
            raise TypeError(
                "configured Language organ does not expose learn_expression()"
            )

        target = LanguageInput(
            transition_id=decision.transition_id,
            text=target_text,
            source_kind=source_kind,
            source_id=source_id,
            metadata={
                "explicit_expression_lesson": True,
                "automatic_teacher_learning": False,
                "world": decision.frame.world,
                "action": decision.intent.action,
            },
        )

        loss = float(
            learner.learn_expression(
                decision.cognitive_state.vector,
                target,
            )
        )
        return ExpressionTeachingResult(
            transition_id=decision.transition_id,
            source_kind=source_kind,
            loss=loss,
        )
