from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import pickle
import tempfile
from typing import Protocol, Sequence
from uuid import uuid4

from contracts.cognition import (
    ActionAttempt,
    ActionIntent,
    CompletedAction,
    Consequence,
    Observation,
)
from core.interface import CognitiveCore, CoreInput, IntegratedCognitiveState
from policy.interface import TrainableActionPolicy
from worlds.interface import (
    WorldAdapter,
    WorldFrame,
    validate_frame,
    validate_intent_for_world,
)


class CausalCycleError(RuntimeError):
    """Raised when an event cannot belong to its claimed causal transition."""


class CausalCheckpointError(CausalCycleError):
    """Raised when a coordinated causal snapshot cannot be used safely."""


class CoreInputBuilder(Protocol):
    """Interpret World observations at the boundary of the shared Core."""

    def build(
        self,
        observations: tuple[Observation, ...],
        *,
        transition_id: str,
        world_id: str,
    ) -> CoreInput:
        ...


@dataclass(frozen=True, slots=True)
class CausalLearningSignal:
    """World-agnostic credit that may be routed to causal module traces."""

    policy_advantage: float | None = None
    core_output_gradient: tuple[float, ...] = ()
    terminal: bool = True


@dataclass(slots=True)
class PendingTransition:
    transition_id: str
    state: IntegratedCognitiveState
    intent: ActionIntent
    attempt: ActionAttempt
    completed_action: CompletedAction | None = None


class ConsequenceInterpreter(Protocol):
    """Convert an observed consequence into causal, module-specific credit."""

    def evaluate(
        self,
        consequence: Consequence,
        transition: PendingTransition,
    ) -> CausalLearningSignal:
        ...


@dataclass(frozen=True, slots=True)
class CausalLearningResult:
    transition_id: str
    consequence_id: str
    policy_updated: bool
    core_updated: bool
    resolved: bool
    skipped_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CausalStepResult:
    frame: WorldFrame
    integrated_state: IntegratedCognitiveState | None
    started: PendingTransition | None
    learning: tuple[CausalLearningResult, ...]


class CausalCognitionCycle:
    """One world-agnostic persistent observation/action/consequence loop.

    Runtime owns lifecycle bookkeeping only. Observation interpretation,
    action selection, execution and consequence interpretation remain behind
    their respective injected interfaces.
    """

    _CHECKPOINT_FORMAT = 1

    def __init__(
        self,
        *,
        world: WorldAdapter,
        core: CognitiveCore,
        policy: TrainableActionPolicy,
        input_builder: CoreInputBuilder,
        consequence_interpreter: ConsequenceInterpreter,
        transition_prefix: str | None = None,
    ) -> None:
        self._world = world
        self._core = core
        self._policy = policy
        self._input_builder = input_builder
        self._consequence_interpreter = consequence_interpreter
        prefix = transition_prefix or f"causal:{uuid4().hex}"
        self._transition_prefix = prefix.strip()
        if not self._transition_prefix:
            raise ValueError("transition_prefix must not be empty")
        if not self._world.world_id.strip():
            raise ValueError("WorldAdapter.world_id must not be empty")
        self._next_transition_number = 1
        self._pending: dict[str, PendingTransition] = {}
        self._restore_poisoned = False
        self._restore_failure = ""

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def step(self, *, now: float) -> CausalStepResult:
        self._require_usable()
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        frame = self._world.poll(now)
        validate_frame(frame, world_id=self._world.world_id)
        learning = self._ingest_outcomes(frame)

        if not frame.observations:
            return CausalStepResult(
                frame=frame,
                integrated_state=None,
                started=None,
                learning=learning,
            )

        transition_id = self._new_transition_id()
        will_choose_action = bool(frame.affordances)
        core_input = self._input_builder.build(
            frame.observations,
            transition_id=transition_id,
            world_id=self._world.world_id,
        )
        self._validate_core_input(core_input, transition_id=transition_id)
        state = self._core.integrate(
            core_input,
            remember_for_learning=will_choose_action,
        )
        self._validate_state(state, transition_id=transition_id)

        if not will_choose_action:
            return CausalStepResult(
                frame=frame,
                integrated_state=state,
                started=None,
                learning=learning,
            )

        try:
            intent = self._policy.choose(
                state,
                frame.affordances,
                now=now,
                remember_for_learning=True,
            )
            validate_intent_for_world(intent, world_id=self._world.world_id)
            if intent.transition_id != transition_id:
                raise CausalCycleError("Policy intent lost transition_id")
            attempt = self._world.execute(intent, now)
            self._validate_attempt(attempt, intent=intent)
        except Exception:
            self._core.discard_trace(transition_id)
            self._policy.discard_trace(transition_id)
            raise

        started = PendingTransition(
            transition_id=transition_id,
            state=state,
            intent=intent,
            attempt=attempt,
        )
        self._pending[transition_id] = started
        return CausalStepResult(
            frame=frame,
            integrated_state=state,
            started=started,
            learning=learning,
        )

    def save_checkpoint(self, path: Path) -> str:
        """Atomically save Runtime and exact delayed-credit organ traces."""

        self._require_usable()
        path = Path(path)
        if path.exists() and path.is_dir():
            raise CausalCheckpointError("checkpoint path must not be a directory")

        pending = tuple(self._pending.values())
        transition_ids = self._validate_pending_snapshot(
            pending,
            transition_prefix=self._transition_prefix,
            next_transition_number=self._next_transition_number,
            world_id=self._world.world_id,
        )
        generation_id = f"causal-generation:{uuid4().hex}"
        try:
            core_snapshot = self._core.export_causal_snapshot(
                generation_id,
                transition_ids,
            )
            policy_snapshot = self._policy.export_causal_snapshot(
                generation_id,
                transition_ids,
            )
        except Exception as exc:
            raise CausalCheckpointError(
                "an organ refused the coordinated causal snapshot"
            ) from exc
        if not isinstance(core_snapshot, bytes) or not isinstance(
            policy_snapshot, bytes
        ):
            raise CausalCheckpointError("organ causal snapshots must be bytes")

        payload = {
            "format": self._CHECKPOINT_FORMAT,
            "generation_id": generation_id,
            "transition_prefix": self._transition_prefix,
            "next_transition_number": self._next_transition_number,
            "world_id": self._world.world_id,
            "transition_ids": transition_ids,
            "pending": pending,
            "core_snapshot": core_snapshot,
            "core_sha256": hashlib.sha256(core_snapshot).hexdigest(),
            "policy_snapshot": policy_snapshot,
            "policy_sha256": hashlib.sha256(policy_snapshot).hexdigest(),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            try:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        return generation_id

    def restore_checkpoint(self, path: Path) -> str:
        """Restore one coordinated generation or poison this fresh cycle."""

        self._require_usable()
        if self._pending or self._next_transition_number != 1:
            self._poison_restore("restore requires a fresh causal cycle")

        # A later failure may happen after one organ accepted its private blob.
        # Keep the cycle unusable until every participant validates and loads.
        self._restore_poisoned = True
        self._restore_failure = "coordinated restore did not complete"
        try:
            with Path(path).open("rb") as stream:
                payload = pickle.load(stream)
            envelope = self._validate_checkpoint_envelope(payload)
            generation_id = envelope["generation_id"]
            transition_ids = envelope["transition_ids"]
            pending = envelope["pending"]

            # The organ API stages each individual restore, but coordinated
            # commit is still sequential.  Preserve exact fresh-organ backups
            # so a later participant rejection cannot leave an orphan mutated
            # organ reachable through the caller's references.
            rollback_generation = f"causal-rollback:{uuid4().hex}"
            core_rollback = self._core.export_causal_snapshot(
                rollback_generation,
                (),
            )
            policy_rollback = self._policy.export_causal_snapshot(
                rollback_generation,
                (),
            )
            try:
                self._core.restore_causal_snapshot(
                    envelope["core_snapshot"],
                    generation_id,
                    transition_ids,
                )
                self._policy.restore_causal_snapshot(
                    envelope["policy_snapshot"],
                    generation_id,
                    transition_ids,
                )
                missing_core = tuple(
                    value
                    for value in transition_ids
                    if not self._core.has_trace(value)
                )
                missing_policy = tuple(
                    value
                    for value in transition_ids
                    if not self._policy.has_trace(value)
                )
                if missing_core or missing_policy:
                    raise CausalCheckpointError(
                        "restored organs are missing pending causal traces"
                    )
            except Exception as restore_exc:
                rollback_failures: list[str] = []
                try:
                    self._core.restore_causal_snapshot(
                        core_rollback,
                        rollback_generation,
                        (),
                    )
                except Exception as rollback_exc:
                    rollback_failures.append(f"Core: {rollback_exc}")
                try:
                    self._policy.restore_causal_snapshot(
                        policy_rollback,
                        rollback_generation,
                        (),
                    )
                except Exception as rollback_exc:
                    rollback_failures.append(f"Policy: {rollback_exc}")
                if rollback_failures:
                    raise CausalCheckpointError(
                        "coordinated restore failed and rollback was incomplete: "
                        + "; ".join(rollback_failures)
                    ) from restore_exc
                raise

            self._pending = {
                transition.transition_id: transition
                for transition in pending
            }
            self._next_transition_number = envelope[
                "next_transition_number"
            ]
        except Exception as exc:
            self._restore_failure = str(exc) or type(exc).__name__
            if isinstance(exc, CausalCheckpointError):
                raise
            raise CausalCheckpointError(
                f"coordinated causal restore failed: {self._restore_failure}"
            ) from exc

        self._restore_poisoned = False
        self._restore_failure = ""
        return generation_id

    def close(self) -> None:
        self._world.close()

    def _validate_checkpoint_envelope(
        self,
        payload: object,
    ) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise CausalCheckpointError("invalid causal checkpoint envelope")
        if payload.get("format") != self._CHECKPOINT_FORMAT:
            raise CausalCheckpointError("unsupported causal checkpoint format")

        generation_id = payload.get("generation_id")
        if not isinstance(generation_id, str) or not generation_id.strip():
            raise CausalCheckpointError("checkpoint generation_id is invalid")
        if payload.get("transition_prefix") != self._transition_prefix:
            raise CausalCheckpointError("checkpoint transition_prefix mismatch")
        if payload.get("world_id") != self._world.world_id:
            raise CausalCheckpointError("checkpoint world_id mismatch")

        next_number = payload.get("next_transition_number")
        if not isinstance(next_number, int) or isinstance(next_number, bool):
            raise CausalCheckpointError("checkpoint transition counter is invalid")
        pending = payload.get("pending")
        if not isinstance(pending, tuple):
            raise CausalCheckpointError("checkpoint pending records are invalid")
        transition_ids = self._validate_pending_snapshot(
            pending,
            transition_prefix=self._transition_prefix,
            next_transition_number=next_number,
            world_id=self._world.world_id,
        )
        if payload.get("transition_ids") != transition_ids:
            raise CausalCheckpointError("checkpoint transition index mismatch")

        core_snapshot = payload.get("core_snapshot")
        policy_snapshot = payload.get("policy_snapshot")
        if not isinstance(core_snapshot, bytes) or not isinstance(
            policy_snapshot, bytes
        ):
            raise CausalCheckpointError("checkpoint organ blobs are invalid")
        if payload.get("core_sha256") != hashlib.sha256(
            core_snapshot
        ).hexdigest():
            raise CausalCheckpointError("Core causal blob checksum mismatch")
        if payload.get("policy_sha256") != hashlib.sha256(
            policy_snapshot
        ).hexdigest():
            raise CausalCheckpointError("Policy causal blob checksum mismatch")

        return {
            "generation_id": generation_id,
            "transition_ids": transition_ids,
            "pending": pending,
            "next_transition_number": next_number,
            "core_snapshot": core_snapshot,
            "policy_snapshot": policy_snapshot,
        }

    @classmethod
    def _validate_pending_snapshot(
        cls,
        pending: tuple[PendingTransition, ...],
        *,
        transition_prefix: str,
        next_transition_number: int,
        world_id: str,
    ) -> tuple[str, ...]:
        if next_transition_number <= 0:
            raise CausalCheckpointError("transition counter must be positive")
        marker = f"{transition_prefix}:"
        transition_ids: list[str] = []
        intent_ids: set[str] = set()
        attempt_ids: set[str] = set()
        completed_ids: set[str] = set()

        for transition in pending:
            if not isinstance(transition, PendingTransition):
                raise CausalCheckpointError("invalid pending transition record")
            transition_id = transition.transition_id
            if transition_id in transition_ids:
                raise CausalCheckpointError("duplicate pending transition_id")
            if not transition_id.startswith(marker):
                raise CausalCheckpointError("pending transition prefix mismatch")
            suffix = transition_id[len(marker):]
            if not suffix.isdigit() or not 0 < int(suffix) < next_transition_number:
                raise CausalCheckpointError("pending transition number is invalid")
            if transition.state.transition_id != transition_id:
                raise CausalCheckpointError("pending Core state lost transition_id")
            if transition.state.world_id != world_id:
                raise CausalCheckpointError("pending Core state changed world_id")
            if transition.intent.transition_id != transition_id:
                raise CausalCheckpointError("pending intent lost transition_id")
            validate_intent_for_world(transition.intent, world_id=world_id)
            cls._validate_attempt(transition.attempt, intent=transition.intent)
            if transition.intent.id in intent_ids:
                raise CausalCheckpointError("duplicate pending ActionIntent id")
            if transition.attempt.id in attempt_ids:
                raise CausalCheckpointError("duplicate pending ActionAttempt id")
            intent_ids.add(transition.intent.id)
            attempt_ids.add(transition.attempt.id)

            completed = transition.completed_action
            if completed is not None:
                cls._validate_completed_snapshot(
                    completed,
                    transition=transition,
                )
                if completed.id in completed_ids:
                    raise CausalCheckpointError("duplicate CompletedAction id")
                completed_ids.add(completed.id)
            transition_ids.append(transition_id)
        return tuple(transition_ids)

    @staticmethod
    def _validate_completed_snapshot(
        completed: CompletedAction,
        *,
        transition: PendingTransition,
    ) -> None:
        if not transition.attempt.accepted:
            raise CausalCheckpointError(
                "rejected ActionAttempt cannot have CompletedAction"
            )
        if completed.transition_id != transition.transition_id:
            raise CausalCheckpointError("CompletedAction lost transition_id")
        if completed.attempt_id != transition.attempt.id:
            raise CausalCheckpointError(
                "CompletedAction does not reference its ActionAttempt"
            )
        if completed.intent_id != transition.intent.id:
            raise CausalCheckpointError(
                "CompletedAction does not reference its ActionIntent"
            )
        if completed.world != transition.intent.world:
            raise CausalCheckpointError("CompletedAction changed world")
        if completed.action != transition.intent.action:
            raise CausalCheckpointError("CompletedAction changed action")

    def _require_usable(self) -> None:
        if self._restore_poisoned:
            reason = self._restore_failure or "restore failed"
            raise CausalCheckpointError(
                f"causal cycle is unusable after failed restore: {reason}"
            )

    def _poison_restore(self, reason: str) -> None:
        self._restore_poisoned = True
        self._restore_failure = reason
        raise CausalCheckpointError(reason)

    def _ingest_outcomes(
        self,
        frame: WorldFrame,
    ) -> tuple[CausalLearningResult, ...]:
        for completed in frame.completed_actions:
            self._accept_completed_action(completed)

        return tuple(
            self._accept_consequence(consequence)
            for consequence in frame.consequences
        )

    def _accept_completed_action(self, completed: CompletedAction) -> None:
        transition = self._require_pending(completed.transition_id)
        if transition.completed_action is not None:
            raise CausalCycleError(
                f"transition {completed.transition_id!r} already completed"
            )
        if not transition.attempt.accepted:
            raise CausalCycleError(
                "rejected ActionAttempt cannot become CompletedAction"
            )
        expected = transition
        if completed.attempt_id != expected.attempt.id:
            raise CausalCycleError("CompletedAction does not reference its ActionAttempt")
        if completed.intent_id != expected.intent.id:
            raise CausalCycleError("CompletedAction does not reference its ActionIntent")
        if completed.world != expected.intent.world:
            raise CausalCycleError("CompletedAction changed world")
        if completed.action != expected.intent.action:
            raise CausalCycleError("CompletedAction changed action")
        transition.completed_action = completed

    def _accept_consequence(
        self,
        consequence: Consequence,
    ) -> CausalLearningResult:
        transition = self._require_pending(consequence.transition_id)
        self._validate_consequence(consequence, transition=transition)
        signal = self._consequence_interpreter.evaluate(consequence, transition)
        self._validate_learning_signal(signal)

        if not signal.terminal:
            if signal.policy_advantage is not None or signal.core_output_gradient:
                raise CausalCycleError(
                    "non-terminal consequence cannot consume one-shot learning traces"
                )
            return CausalLearningResult(
                transition_id=transition.transition_id,
                consequence_id=consequence.id,
                policy_updated=False,
                core_updated=False,
                resolved=False,
            )

        missing_traces: list[str] = []
        if (
            signal.policy_advantage is not None
            and not self._policy.has_trace(transition.transition_id)
        ):
            missing_traces.append("policy")
        if (
            signal.core_output_gradient
            and not self._core.has_trace(transition.transition_id)
        ):
            missing_traces.append("core")

        if missing_traces:
            # Learning calls consume traces and mutate parameters.  Preflight
            # every credited organ before the first call so trace eviction can
            # never produce a partial cross-organ update.  The terminal event
            # still resolves deterministically: any surviving traces are
            # discarded because this causal transition cannot be replayed.
            self._policy.discard_trace(transition.transition_id)
            self._core.discard_trace(transition.transition_id)
            del self._pending[transition.transition_id]
            return CausalLearningResult(
                transition_id=transition.transition_id,
                consequence_id=consequence.id,
                policy_updated=False,
                core_updated=False,
                resolved=True,
                skipped_reason=(
                    "missing_causal_trace:" + ",".join(missing_traces)
                ),
            )

        if signal.policy_advantage is None:
            self._policy.discard_trace(transition.transition_id)
            policy_updated = False
        else:
            policy_updated = self._policy.learn(
                transition.transition_id,
                signal.policy_advantage,
            )
            if not policy_updated:
                raise CausalCycleError(
                    "Policy causal trace was missing for consequence"
                )

        if not signal.core_output_gradient:
            self._core.discard_trace(transition.transition_id)
            core_updated = False
        else:
            core_updated = self._core.learn(
                transition.transition_id,
                signal.core_output_gradient,
            )
            if not core_updated:
                raise CausalCycleError(
                    "Core causal trace was missing for consequence"
                )

        del self._pending[transition.transition_id]
        return CausalLearningResult(
            transition_id=transition.transition_id,
            consequence_id=consequence.id,
            policy_updated=policy_updated,
            core_updated=core_updated,
            resolved=True,
        )

    def _require_pending(self, transition_id: str) -> PendingTransition:
        transition = self._pending.get(transition_id)
        if transition is None:
            raise CausalCycleError(
                f"event references unknown transition_id {transition_id!r}"
            )
        return transition

    def _validate_core_input(
        self,
        value: CoreInput,
        *,
        transition_id: str,
    ) -> None:
        if value.transition_id != transition_id:
            raise CausalCycleError("CoreInput builder changed transition_id")
        if value.world_id != self._world.world_id:
            raise CausalCycleError("CoreInput builder changed world_id")

    def _validate_state(
        self,
        state: IntegratedCognitiveState,
        *,
        transition_id: str,
    ) -> None:
        if state.transition_id != transition_id:
            raise CausalCycleError("Core state lost transition_id")
        if state.world_id != self._world.world_id:
            raise CausalCycleError("Core state changed world_id")

    @staticmethod
    def _validate_attempt(
        attempt: ActionAttempt,
        *,
        intent: ActionIntent,
    ) -> None:
        if attempt.intent_id != intent.id:
            raise CausalCycleError("ActionAttempt does not reference its ActionIntent")
        if attempt.transition_id != intent.transition_id:
            raise CausalCycleError("ActionAttempt lost transition_id")
        if attempt.world != intent.world:
            raise CausalCycleError("ActionAttempt changed world")
        if attempt.action != intent.action:
            raise CausalCycleError("ActionAttempt changed action")

    @staticmethod
    def _validate_consequence(
        consequence: Consequence,
        *,
        transition: PendingTransition,
    ) -> None:
        if consequence.intent_id is not None and consequence.intent_id != transition.intent.id:
            raise CausalCycleError("Consequence does not reference its ActionIntent")
        if (
            consequence.attempt_id is not None
            and consequence.attempt_id != transition.attempt.id
        ):
            raise CausalCycleError("Consequence does not reference its ActionAttempt")
        if consequence.completed_action_id is not None:
            completed = transition.completed_action
            if completed is None:
                raise CausalCycleError(
                    "Consequence references a CompletedAction not yet observed"
                )
            if consequence.completed_action_id != completed.id:
                raise CausalCycleError(
                    "Consequence does not reference its CompletedAction"
                )

    def _validate_learning_signal(self, signal: CausalLearningSignal) -> None:
        if signal.policy_advantage is not None:
            advantage = float(signal.policy_advantage)
            if not math.isfinite(advantage):
                raise CausalCycleError("policy_advantage must be finite")
        gradient: Sequence[float] = signal.core_output_gradient
        if gradient and len(gradient) != self._core.representation_size:
            raise CausalCycleError(
                "core_output_gradient size does not match Core representation"
            )
        if not all(math.isfinite(float(value)) for value in gradient):
            raise CausalCycleError("core_output_gradient must contain finite values")

    def _new_transition_id(self) -> str:
        transition_id = (
            f"{self._transition_prefix}:{self._next_transition_number}"
        )
        self._next_transition_number += 1
        if transition_id in self._pending:
            raise CausalCycleError(f"duplicate transition_id {transition_id!r}")
        return transition_id
