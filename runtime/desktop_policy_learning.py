from __future__ import annotations

from dataclasses import dataclass
import math

from contracts.cognition import ActionAttempt, CompletedAction
from policy.interface import TrainableActionPolicy
from runtime.desktop import DesktopDecision


@dataclass(frozen=True, slots=True)
class DesktopPolicyLearningTelemetry:
    learning_updates: int
    positive_updates: int
    negative_updates: int
    rejected_noncausal_updates: int
    missing_policy_traces: int
    last_transition_id: str | None
    last_action: str | None
    last_advantage: float


class CausalDesktopPolicyLearner:
    """
    Narrow learning-credit boundary for Desktop action choice.

    It does not derive reward, decide whether an action was good, execute
    actions, or teach Language/Core. A caller must provide an already-derived
    finite `advantage` (for example explicit mentor feedback now, or a learned
    Value signal later).

    Policy is updated only after a real ActionAttempt has been linked to an
    externally observed CompletedAction for the exact same transition.
    """

    def __init__(self, policy: TrainableActionPolicy) -> None:
        self.policy = policy
        self._learning_updates = 0
        self._positive_updates = 0
        self._negative_updates = 0
        self._rejected_noncausal_updates = 0
        self._missing_policy_traces = 0
        self._last_transition_id: str | None = None
        self._last_action: str | None = None
        self._last_advantage = 0.0

    def learn_from_completed(
        self,
        decision: DesktopDecision,
        attempt: ActionAttempt,
        completed: CompletedAction,
        *,
        advantage: float,
    ) -> bool:
        advantage = float(advantage)
        if not math.isfinite(advantage):
            raise ValueError("advantage must be finite")
        if advantage == 0.0:
            raise ValueError(
                "zero advantage is intentionally rejected because Policy.learn() "
                "would consume the causal trace without useful learning"
            )

        try:
            self._validate_chain(decision, attempt, completed)
        except (ValueError, RuntimeError):
            self._rejected_noncausal_updates += 1
            raise

        updated = bool(
            self.policy.learn(
                decision.transition_id,
                advantage,
            )
        )
        if not updated:
            self._missing_policy_traces += 1
            raise RuntimeError(
                "Policy has no causal decision trace for transition_id "
                f"{decision.transition_id!r}"
            )

        self._learning_updates += 1
        if advantage > 0.0:
            self._positive_updates += 1
        else:
            self._negative_updates += 1
        self._last_transition_id = decision.transition_id
        self._last_action = decision.intent.action
        self._last_advantage = advantage
        return True

    def telemetry(self) -> DesktopPolicyLearningTelemetry:
        return DesktopPolicyLearningTelemetry(
            learning_updates=self._learning_updates,
            positive_updates=self._positive_updates,
            negative_updates=self._negative_updates,
            rejected_noncausal_updates=self._rejected_noncausal_updates,
            missing_policy_traces=self._missing_policy_traces,
            last_transition_id=self._last_transition_id,
            last_action=self._last_action,
            last_advantage=self._last_advantage,
        )

    @staticmethod
    def _validate_chain(
        decision: DesktopDecision,
        attempt: ActionAttempt,
        completed: CompletedAction,
    ) -> None:
        transition_id = decision.transition_id
        intent = decision.intent

        if intent.transition_id != transition_id:
            raise RuntimeError("DesktopDecision intent lost transition_id")
        if not attempt.accepted:
            raise ValueError("rejected ActionAttempt cannot receive causal credit")

        if attempt.intent_id != intent.id:
            raise ValueError("ActionAttempt does not belong to selected ActionIntent")
        if attempt.transition_id != transition_id:
            raise ValueError("ActionAttempt transition_id differs from decision")
        if attempt.world != intent.world:
            raise ValueError("ActionAttempt world differs from ActionIntent")
        if attempt.action != intent.action:
            raise ValueError("ActionAttempt action differs from ActionIntent")

        if completed.attempt_id != attempt.id:
            raise ValueError("CompletedAction does not belong to ActionAttempt")
        if completed.intent_id != intent.id:
            raise ValueError("CompletedAction intent_id differs from decision")
        if completed.transition_id != transition_id:
            raise ValueError("CompletedAction transition_id differs from decision")
        if completed.world != intent.world:
            raise ValueError("CompletedAction world differs from ActionIntent")
        if completed.action != intent.action:
            raise ValueError("CompletedAction action differs from ActionIntent")
        if completed.completed_at < attempt.attempted_at:
            raise ValueError("CompletedAction predates ActionAttempt")
