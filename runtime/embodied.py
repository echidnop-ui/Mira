from __future__ import annotations

from collections import deque
import argparse
from dataclasses import asdict, replace
from difflib import SequenceMatcher
from hashlib import sha256
import json
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Protocol
from uuid import uuid4

import torch

from core.personal import PersonalCognitiveCore
from core.interface import CoreInput
from language.organ import NeuralLanguageConfig, NeuralLanguageOrgan
from language.interface import LanguageInput
from policy.dynamic import DynamicAffordancePolicy, DynamicPolicyConfig
from runtime.desktop import DesktopDecision, DesktopDecisionCycle
from runtime.desktop_expression import DesktopExpressionBridge, EmptyExpressionError
from runtime.mentor import MentorIntegrator
from runtime.teacher import (
    CommandTeacherAdapter, OllamaTeacherAdapter, TeacherAdapter, TeacherContext,
    TeacherMessage, TeacherSessionStore,
)
from runtime.autonomy import AutonomyConfig, load_config
from worlds.desktop.inference import NeuralDesktopInference
from worlds.desktop.neural_adapter import NeuralDesktopWorldAdapter, NeuralSafetyGate
from worlds.desktop.ownership import PhysicalActionArbiter
from worlds.desktop.portal_screencast import PortalScreenCastError


class EmbodiedWorld(Protocol):
    @property
    def world_id(self) -> str: ...
    def poll(self, now: float): ...
    def execute(self, intent, now: float): ...
    def close(self) -> None: ...


class NullTeacherAdapter:
    teacher_id = "none"
    def respond(self, context: TeacherContext) -> TeacherMessage | None:
        return None


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        import os
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, default=str); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _AsyncTeacherChannel:
    """One bounded in-flight teacher request; transport failure never owns the organism loop."""
    def __init__(self, adapter: TeacherAdapter, *, interval: float) -> None:
        self.adapter = adapter; self.interval = float(interval)
        self._results: queue.SimpleQueue[tuple[TeacherContext, TeacherMessage | None, BaseException | None]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None; self.next_request_at = 0.0
        self.status = "offline" if adapter.teacher_id == "none" else "healthy"
        self.last_error = ""; self.requests = 0; self.failures = 0

    def submit(self, context: TeacherContext, *, now: float) -> bool:
        if self.adapter.teacher_id == "none" or now < self.next_request_at:
            return False
        if self._thread is not None and self._thread.is_alive(): return False
        self.next_request_at = now + self.interval; self.requests += 1; self.status = "waiting"
        def run() -> None:
            try: self._results.put((context, self.adapter.respond(context), None))
            except BaseException as exc: self._results.put((context, None, exc))
        self._thread = threading.Thread(target=run, name="heiwa-teacher-channel", daemon=True)
        self._thread.start(); return True

    def poll(self) -> tuple[TeacherContext, TeacherMessage | None, BaseException | None] | None:
        try: result = self._results.get_nowait()
        except queue.Empty: return None
        if result[2] is None:
            self.status = "healthy"; self.last_error = ""
        else:
            self.status = "degraded"; self.failures += 1
            self.last_error = f"{type(result[2]).__name__}: {' '.join(str(result[2]).split())[:300]}"
        return result

    def close(self) -> None:
        close = getattr(self.adapter, "close", None)
        if callable(close):
            try: close()
            except Exception: pass


def _event_id(payload: dict[str, object]) -> str:
    stable = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return "event-" + sha256(stable.encode()).hexdigest()[:24]


class EmbodiedRuntime:
    """One Heiwa: shared Core with Vision, Language, Desktop and Teacher channels."""

    def __init__(
        self, *, core, language, policy, world: EmbodiedWorld,
        teacher: TeacherAdapter, teacher_store: TeacherSessionStore,
        event_log: Path, expression: DesktopExpressionBridge | None = None,
        state_dir: Path | None = None, desktop_checkpoint: Path | None = None,
        language_checkpoint: Path | None = None, core_checkpoint: Path | None = None,
        policy_checkpoint: Path | None = None, checkpoint_interval: float = 300.0,
        teacher_interval: float = 30.0, autonomous: bool = True,
        reading_minimum_confidence: float = 0.35,
        policy_exploration_temperature: float = 1.0,
        stagnation_noop_steps: int = 3,
        stagnation_noop_penalty: float = 0.35,
        reader_trainer: subprocess.Popen | None = None,
        reader_candidate: Path | None = None, reader_report: Path | None = None,
        reader_command: tuple[str, ...] | None = None,
        language_trainer: subprocess.Popen | None = None,
        language_candidate: Path | None = None, language_report: Path | None = None,
        language_command: tuple[str, ...] | None = None,
        language_observation_queue: Path | None = None,
        language_observation_visual_confidence: float = 0.80,
    ) -> None:
        self.core = core; self.language = language; self.policy = policy; self.world = world
        self.teacher = teacher; self.teacher_store = teacher_store
        self.mentor = MentorIntegrator(language, core)
        self.cycle = DesktopDecisionCycle(core, policy)
        self.expression = expression or DesktopExpressionBridge(language)
        self.event_log = event_log; self.event_log.parent.mkdir(parents=True, exist_ok=True)
        self.state_dir = state_dir or event_log.parent
        self.desktop_checkpoint = desktop_checkpoint
        self.language_checkpoint = language_checkpoint
        self.core_checkpoint = core_checkpoint or self.state_dir / "core.pt"
        self.policy_checkpoint = policy_checkpoint or self.state_dir / "policy.pt"
        self.status_path = self.state_dir / "status.json"
        self.checkpoint_interval = float(checkpoint_interval); self._next_checkpoint = time.monotonic() + self.checkpoint_interval
        self.autonomous = bool(autonomous)
        self.reading_minimum_confidence = float(reading_minimum_confidence)
        self.policy_exploration_temperature = float(policy_exploration_temperature)
        self.stagnation_noop_steps = max(1, int(stagnation_noop_steps))
        self.stagnation_noop_penalty = min(1.0, max(1e-6, float(stagnation_noop_penalty)))
        self._unchanged_noop_streak = 0
        self.teacher_channel = _AsyncTeacherChannel(teacher, interval=teacher_interval)
        self.episode_id = f"desktop-episode-{uuid4().hex}"
        self.last_attempt = None; self.last_consequence = None; self.last_heiwa_utterance = ""
        self.last_heiwa_message_id = ""
        self.last_teacher_utterance = ""; self.last_expression_decision: DesktopDecision | None = None
        # Technical provenance for self-expression. This is deliberately NOT a
        # Language teacher buffer: own generated speech can be compared with
        # later pixels to diagnose motor/reader corruption, but never becomes
        # true merely because nobody corrected it.
        self._pending_expression_audit: dict[str, object] | None = None
        self.steps = 0; self.started_at = time.monotonic(); self.last_step_latency_ms = 0.0
        self.last_stage_latency_ms: dict[str, float] = {}
        self.reader_trainer = reader_trainer; self.reader_candidate = reader_candidate; self.reader_report = reader_report
        self.reader_command = reader_command; self._reader_restart_at = 0.0
        self._reader_adopted = False
        self.language_trainer = language_trainer
        self.language_candidate = language_candidate
        self.language_report = language_report
        self.language_command = language_command
        self.language_observation_queue = language_observation_queue or (self.state_dir / "language_observations.jsonl")
        self.language_observation_queue.parent.mkdir(parents=True, exist_ok=True)
        self.language_observation_visual_confidence = float(language_observation_visual_confidence)
        self._language_candidate_mtime_ns = 0
        self._language_restart_at = 0.0
        self._last_language_observation_hash = ""
        self._last_language_observation_at = 0.0
        # Desktop F21..F24 are organ-local visual controls. Runtime deliberately
        # receives no Policy reward or Language teacher evidence from them.
        # Ordinary Backspace/Enter remain application keys. Silence remains unknown.
        self._feedback_trace_window_seconds = 15.0
        # Keep enough context to make explicit F23 feedback affect the very
        # next decision, not merely future weights. target_xy is technical
        # motor context; it does not imply what the correct target should be.
        self._recent_feedback_traces: deque[tuple[str, str, float, tuple[float, float] | None]] = deque(maxlen=8)
        self._pending_immediate_rejection: dict[str, object] | None = None
        # F23 must survive more than one policy pass.  A one-step affordance
        # removal let the policy immediately fall back into the same attractor
        # on the following frame.  Keep a short, explicit working-memory taboo
        # for rejected action kinds.  This is feedback-driven, generic, and
        # expires automatically; it does not prescribe an opposite action.
        self._rejection_cooldowns: dict[str, dict[str, object]] = {}
        self._rejection_base_decisions = 4
        self._rejection_max_decisions = 10
        self._rejection_ttl_seconds = 12.0
        self._expression_confirmation_window_seconds = 30.0
        self._recent_expression_traces: deque[dict[str, object]] = deque(maxlen=4)
        self._last_user_correction_at = float("-inf")
        # HEIWA_PENDING_EXTERNAL_RESPONSE_V1
        # A self-expression followed by a click is treated as a generic commit
        # action.  After that commit the organism waits for the world to answer
        # instead of immediately generating more keyboard/mouse interaction.
        # This is stateful consequence waiting, not a hardcoded Send-button rule.
        self._pending_external_response: dict[str, object] | None = None
        self._pending_external_response_expression_window_seconds = 10.0
        self._pending_external_response_local_change_grace_seconds = 1.5
        # Do not release merely because the UI pauses briefly. During a streamed
        # answer the Stop control can remain visible while the pixels are quiet.
        # Require a real observation interval plus several distinct visual changes.
        self._pending_external_response_min_wait_seconds = 8.0
        self._pending_external_response_stable_seconds = 5.0
        self._pending_external_response_min_change_count = 3
        self._pending_external_response_max_seconds = 45.0
        # Repeated physical pointer takeovers are stronger evidence than a
        # single aiming adjustment. Track them by behavioural family + target
        # region, not only exact action kind: left/right/double click on the same
        # UI object are one repeated pattern from the human's point of view.
        self._pointer_behavior_corrections: dict[str, dict[str, object]] = {}
        self._pointer_behavior_repeat_seconds = 15.0
        self._pointer_behavior_target_radius = 0.14

    def _write(self, kind: str, **payload: object) -> None:
        record = {"kind": kind, "timestamp": time.time(), "episode_id": self.episode_id, **payload}
        with self.event_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n"); stream.flush()

    def _queue_language_observation(
        self, *, text: str, source_kind: str, source_id: str, event_id: str,
        provenance: str, confidence: float | None = None,
    ) -> bool:
        """Queue external text for self-supervised surface learning.

        Queueing is intentionally truth-neutral: it says only that this text was
        externally observed.  It never marks it correct, useful, or factual.
        """
        text = " ".join(str(text).split()).strip()
        if not text or source_kind == "self":
            return False
        digest = sha256((source_kind + "\0" + source_id + "\0" + text).encode()).hexdigest()
        now = time.time()
        if digest == self._last_language_observation_hash and now - self._last_language_observation_at < 30.0:
            return False
        record = {
            "id": event_id or f"language-observed-{digest[:20]}",
            "timestamp": now, "text": text, "source_kind": source_kind,
            "source_id": source_id, "provenance": provenance,
            "confidence": confidence, "observational_only": True,
            "truth_claim": False, "explicit_teacher_data": False,
            "own_speech": False, "self_generated": False,
        }
        with self.language_observation_queue.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            stream.flush()
        self._last_language_observation_hash = digest
        self._last_language_observation_at = now
        self._write(
            "language_observation_queued", observation_id=record["id"],
            source_kind=source_kind, source_id=source_id, provenance=provenance,
            confidence=confidence, text=text, observational_only=True,
            truth_claim=False, language_teacher=False,
        )
        return True

    def _queue_explicit_language_example(
        self, *, context_text: str, target_text: str, event_id: str, kind: str,
    ) -> bool:
        context_text = " ".join(str(context_text).split()).strip()
        target_text = " ".join(str(target_text).split()).strip()
        if not target_text:
            return False
        record = {
            "id": event_id, "timestamp": time.time(),
            "learning_kind": "explicit_conditioned",
            "context_text": context_text, "target_text": target_text,
            "text": target_text, "source_kind": "mentor",
            "source_id": self.teacher.teacher_id, "teacher_kind": kind,
            "observational_only": False, "truth_claim": False,
            "explicit_teacher_data": True, "own_speech": False,
            "self_generated": False,
        }
        with self.language_observation_queue.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            stream.flush()
        self._write(
            "language_teacher_example_queued", observation_id=event_id,
            teacher_kind=kind, context_text=context_text, target_text=target_text,
            explicit_teacher_data=True, own_speech_teacher=False,
        )
        return True

    def _queue_human_confirmed_expression(
        self, *, text: str, transition_id: str, message_id: str,
    ) -> bool:
        """Queue explicit human evidence that Heiwa's written form was correct.

        A physical F24 command is deliberately narrower than HNT3 teach:
        it validates the surface expression only. It does not claim factual
        truth, usefulness, or that choosing to speak was the right action.
        """
        text = " ".join(str(text).split()).strip()
        if not text:
            return False
        event_id = f"human-f24-confirm-{message_id or transition_id}"
        record = {
            "id": event_id, "timestamp": time.time(),
            "learning_kind": "explicit_surface_confirmation",
            "context_text": "", "target_text": text, "text": text,
            "source_kind": "human", "source_id": "input_remapper_f24",
            "teacher_kind": "human-f24-surface-confirmation",
            "observational_only": False, "truth_claim": False,
            "usefulness_claim": False, "explicit_teacher_data": True,
            "own_speech": True, "self_generated": True,
            "human_confirmed": True, "surface_correctness_only": True,
            "transition_id": transition_id, "message_id": message_id,
        }
        with self.language_observation_queue.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            stream.flush()
        self._write(
            "language_human_expression_confirmation_queued",
            observation_id=event_id, transition_id=transition_id,
            message_id=message_id, text=text, source="input_remapper_f24",
            explicit_teacher_data=True, human_confirmed=True,
            surface_correctness_only=True, truth_claim=False,
            usefulness_claim=False, own_speech_teacher=False,
        )
        return True

    def _teacher_context(self, frame, now: float) -> TeacherContext:
        action = {} if self.last_attempt is None else {
            "transition_id": self.last_attempt.transition_id,
            "action": self.last_attempt.action, "accepted": self.last_attempt.accepted,
        }
        consequence = {} if self.last_consequence is None else {
            "transition_id": self.last_consequence.transition_id,
            "kind": self.last_consequence.kind, "payload": dict(self.last_consequence.payload),
        }
        summary = {
            "observation_kinds": [value.kind for value in frame.observations],
            "available_actions": [value.action for value in frame.affordances],
            "completed_actions": len(frame.completed_actions),
            "visual_digest": next((str(value.payload.get("frame_digest", ""))
                                   for value in frame.observations
                                   if value.kind == "desktop.neural_visual_state"), ""),
        }
        # Request identity is tied to one real outbound Heiwa utterance, not to
        # a changing screen frame.  This prevents the teacher channel from
        # becoming a periodic <OBSERVE> clock and preserves the same rid across
        # transport retries for the same utterance.
        request_identity = {
            "conversation_id": self.teacher_store.conversation_id,
            "heiwa_message_id": self.last_heiwa_message_id,
            "heiwa": self.last_heiwa_utterance,
        }
        summary["heiwa_message_id"] = self.last_heiwa_message_id
        return TeacherContext(
            conversation_id=self.teacher_store.conversation_id, world_id=frame.world,
            observed_at=now, event_id=_event_id(request_identity), episode_id=self.episode_id,
            heiwa_utterance=self.last_heiwa_utterance,
            teacher_utterance=self.last_teacher_utterance, recent_action=action,
            consequence=consequence, state_summary=summary,
        )

    def _teacher_learning_input(
        self, *, context: TeacherContext, text: str, kind: str, index: int,
    ) -> LanguageInput:
        return LanguageInput(
            transition_id=f"{context.event_id}-{kind}-{index}", text=text,
            source_kind="mentor", source_id=self.teacher.teacher_id,
            metadata={
                "language_teacher": True, "explicit_teacher_data": True,
                "rid": context.event_id, "teacher_kind": kind,
                "own_speech_teacher": False,
            },
        )

    def _apply_teacher_message(self, context: TeacherContext, message: TeacherMessage) -> None:
        if message.reply_to_event_id != context.event_id:
            raise ValueError("teacher response lost request/event correlation")
        rid = str(message.metadata.get("rid", context.event_id))
        if rid != context.event_id:
            raise ValueError("teacher response rid does not match current request")
        if not self.teacher_store.accept(message):
            self._write("teacher_duplicate_ignored", rid=context.event_id, message_id=message.id)
            return

        self._write(
            "teacher_received", rid=context.event_id, message_id=message.id,
            reply_to_heiwa_message_id=self.last_heiwa_message_id,
            text=message.text,
        )
        previous_teacher_utterance = self.last_teacher_utterance
        result = self.mentor.observe(
            message.text, source_id=self.teacher.teacher_id,
            transition_id=f"teacher-observe-{context.event_id}",
            metadata={"conversation_id": message.conversation_id,
                      "inbound_message_id": message.id, "language_teacher": False,
                      "declared_teacher_kind": message.kind,
                      "rid": context.event_id},
        )
        self.last_teacher_utterance = message.text
        self._queue_language_observation(
            text=message.text, source_kind="mentor", source_id=self.teacher.teacher_id,
            event_id=f"mentor-observed-{context.event_id}", provenance="teacher_say",
        )
        self._write(
            "teacher_say_observed", rid=context.event_id,
            transition_id=result.transition_id, text=message.text,
            language_source=result.language.source_kind,
            core_transition_id=result.cognition.transition_id,
            language_teacher=False,
        )

        # Legacy explicit correction remains supported, but ordinary mentor
        # speech is never silently promoted to a training target.
        taught = False
        if message.language_teacher and self.last_expression_decision is not None:
            if self.language_command is None:
                self.expression.teach(
                    self.last_expression_decision, message.text, source_kind="mentor",
                    source_id=self.teacher.teacher_id,
                )
                taught = True
            else:
                taught = self._queue_explicit_language_example(
                    context_text=previous_teacher_utterance, target_text=message.text,
                    event_id=f"{context.event_id}-legacy-explicit-correction",
                    kind="explicit-linguistic-correction",
                )
        if message.kind == "competence_feedback":
            transition = str(message.metadata.get("transition_id", ""))
            useful = message.metadata.get("useful")
            if transition and isinstance(useful, bool):
                self.policy.learn(transition, 1.0 if useful else -1.0)

        raw_teach = message.metadata.get("teach", ())
        raw_fix = message.metadata.get("fix", ())
        teach = tuple(dict.fromkeys(
            str(value).strip() for value in raw_teach
            if isinstance(raw_teach, (tuple, list)) and str(value).strip()
        )) if isinstance(raw_teach, (tuple, list)) else ()
        fixes = tuple(dict.fromkeys(
            str(value).strip() for value in raw_fix
            if isinstance(raw_fix, (tuple, list)) and str(value).strip()
        )) if isinstance(raw_fix, (tuple, list)) else ()

        teacher_losses: list[float] = []
        conditioned_losses: list[float] = []
        queued_examples = 0

        # One language learner owns trainable Language weights at a time.  With
        # the background learner enabled, even explicit HNT3 teach/fix examples
        # are queued instead of racing a second optimizer in the organism loop.
        if self.language_command is not None:
            for index, target in enumerate(teach):
                if self._queue_explicit_language_example(
                    context_text=message.text, target_text=target,
                    event_id=f"{context.event_id}-teaching-example-{index}",
                    kind="teaching-example",
                ):
                    queued_examples += 1
            for index, target in enumerate(fixes):
                if self._queue_explicit_language_example(
                    context_text=previous_teacher_utterance, target_text=target,
                    event_id=f"{context.event_id}-correction-{index}",
                    kind="correction",
                ):
                    queued_examples += 1
        else:
            learn_conditioned = getattr(self.language, "learn_conditioned_expression", None)
            learn_surface = getattr(self.language, "learn_teacher_text", None)
            for index, target in enumerate(teach):
                target_input = self._teacher_learning_input(
                    context=context, text=target, kind="teaching-example", index=index)
                if callable(learn_conditioned):
                    context_input = self._teacher_learning_input(
                        context=context, text=message.text, kind="teaching-context", index=index)
                    conditioned_losses.append(learn_conditioned(context_input, target_input))
                if callable(learn_surface):
                    teacher_losses.append(learn_surface(target_input))
            for index, target in enumerate(fixes):
                target_input = self._teacher_learning_input(
                    context=context, text=target, kind="correction", index=index)
                if callable(learn_conditioned) and previous_teacher_utterance:
                    context_input = self._teacher_learning_input(
                        context=context, text=previous_teacher_utterance,
                        kind="correction-context", index=index)
                    conditioned_losses.append(learn_conditioned(context_input, target_input))
                if callable(learn_surface):
                    teacher_losses.append(learn_surface(target_input))

        self._write(
            "teacher_examples_learned", rid=context.event_id,
            teach_count=len(teach), fix_count=len(fixes),
            conditioned_losses=conditioned_losses, surface_losses=teacher_losses,
            queued_examples=queued_examples,
            language_updated=bool(taught or conditioned_losses or teacher_losses),
            language_update_queued=bool(queued_examples),
        )
        self._write(
            "teacher_message", message_id=message.id, conversation_id=message.conversation_id,
            transition_id=result.transition_id, text=message.text,
            language_teacher=bool(message.language_teacher or teach or fixes),
            language_updated=bool(taught or conditioned_losses or teacher_losses),
            language_update_queued=bool(queued_examples),
            teacher_losses=teacher_losses, conditioned_losses=conditioned_losses,
            rid=context.event_id,
        )

    def _teacher_tick(self, frame, now: float) -> None:
        result = self.teacher_channel.poll()
        if result is not None:
            context, message, error = result
            if error is not None:
                self._write("teacher_degraded", rid=context.event_id,
                            error=self.teacher_channel.last_error, organism_continues=True)
            else:
                if message is None:
                    self.teacher_store.mark_event(context.event_id)
                    self._write("teacher_silence", rid=context.event_id)
                else:
                    try:
                        self._apply_teacher_message(context, message)
                    except (ValueError, RuntimeError) as exc:
                        self.teacher_channel.status = "degraded"
                        self._write("teacher_invalid", rid=context.event_id,
                                    error=f"{type(exc).__name__}: {exc}", organism_continues=True)
                    else:
                        self.teacher_store.mark_event(context.event_id)

        # Teacher dialogue is event-driven.  A changing frame is not a reason
        # to ask the mentor anything.  Only a real, successfully executed Heiwa
        # utterance opens/retries one correlated teacher turn.
        if not self.last_heiwa_utterance or not self.last_heiwa_message_id:
            return
        context = self._teacher_context(frame, now)
        if self.teacher_store.should_send_event(context.event_id):
            if self.teacher_channel.submit(context, now=now):
                self._write(
                    "teacher_request", rid=context.event_id, reason="heiwa_utterance",
                    heiwa_message_id=self.last_heiwa_message_id,
                    heiwa_text=self.last_heiwa_utterance,
                )

    @staticmethod
    def _surface_text(value: str) -> str:
        return " ".join(str(value).split()).strip()

    def _audit_expression_against_visual_text(
        self, *, text: str, confidence: float, bbox: object,
    ) -> None:
        """Compare intended self-expression with later visual text.

        This is technical causal evidence only. It never calls a Language
        learning method and never turns Heiwa's own utterance (or human
        silence) into a truth/teacher target.
        """
        pending = self._pending_expression_audit
        if pending is None:
            return
        age_steps = self.steps - int(pending.get("created_step", self.steps))
        checks = int(pending.get("checks", 0)) + 1
        intended = self._surface_text(str(pending.get("intended_text", "")))
        observed = self._surface_text(text)
        ratio = 0.0 if not intended or not observed else SequenceMatcher(
            None, intended.casefold(), observed.casefold()
        ).ratio()
        self._write(
            "expression_visual_audit",
            transition_id=str(pending.get("transition_id", "")),
            message_id=str(pending.get("message_id", "")),
            intended_text=intended,
            observed_visual_text=observed,
            visual_confidence=float(confidence),
            surface_similarity=round(float(ratio), 6),
            bbox=bbox,
            technical_evidence_only=True,
            language_teacher=False,
            language_updated=False,
            silence_is_confirmation=False,
        )
        pending["checks"] = checks
        # Keep several post-action visual samples because the neural fovea may
        # initially look at another region. A strong textual overlap or a small
        # bounded observation window closes the audit without assigning reward.
        if ratio >= 0.82 or checks >= 5 or age_steps >= 8:
            self._pending_expression_audit = None

    def _integrate_visual_reading(self, frame) -> None:
        observation = next((value for value in frame.observations
                            if value.kind == "desktop.neural_visual_state"), None)
        if observation is None: return
        confidence = float(observation.payload.get("visual_read_confidence", 0.0))
        if confidence < self.reading_minimum_confidence: return
        raw_ids = observation.payload.get("visual_read_byte_ids", ())
        if not isinstance(raw_ids, (tuple, list)): return
        tokenizer = getattr(self.language, "tokenizer", None)
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode): return
        text = " ".join(str(decode(tuple(int(value) for value in raw_ids))).split()).strip()
        if not text: return
        self._audit_expression_against_visual_text(
            text=text, confidence=confidence,
            bbox=observation.payload.get("visual_read_bbox"),
        )
        transition = f"desktop-reading-{uuid4().hex}"
        language = self.language.encode(LanguageInput(
            transition_id=transition, text=text, source_kind="world",
            source_id="desktop.neural.fovea",
            metadata={"language_teacher": False, "neural_reading": True,
                      "confidence": confidence,
                      "bbox": observation.payload.get("visual_read_bbox")},
        ), remember_for_learning=False)
        self.core.integrate(CoreInput(
            transition_id=transition, world_id=frame.world,
            representation=language.vector,
            metadata={"language_teacher": False, "neural_reading": True,
                      "confidence": confidence,
                      "bbox": observation.payload.get("visual_read_bbox")},
        ), remember_for_learning=False)
        self._write("visual_reading", transition_id=transition, text=text,
                    confidence=confidence, bbox=observation.payload.get("visual_read_bbox"),
                    language_teacher=False)
        # High-confidence pixel text may teach only surface language statistics.
        # While a self-expression audit is open we refuse it completely, so
        # Heiwa cannot immediately train on text she just generated herself.
        own_similarity = 0.0
        if self.last_heiwa_utterance:
            own_similarity = SequenceMatcher(
                None, self._surface_text(self.last_heiwa_utterance).casefold(),
                self._surface_text(text).casefold(),
            ).ratio()
        if (confidence >= self.language_observation_visual_confidence
                and self._pending_expression_audit is None
                and own_similarity < 0.45):
            self._queue_language_observation(
                text=text, source_kind="world", source_id="desktop.neural.fovea",
                event_id=f"visual-language-{transition}", provenance="pixel_text",
                confidence=confidence,
            )

    def _log_visual_reading_feedback(self, frame) -> None:
        for observation in frame.observations:
            if observation.kind != "desktop.visual_reading_feedback":
                continue
            payload = observation.payload
            self._write(
                "visual_reading_feedback",
                key=payload.get("key"), accepted=bool(payload.get("accepted")),
                decoded_byte_ids=payload.get("decoded_byte_ids"),
                bbox=payload.get("bbox"), metrics=payload.get("metrics"),
                language_teacher=False, policy_feedback=False,
            )

    def _expire_feedback_traces(self, now: float) -> None:
        cutoff = float(now) - self._feedback_trace_window_seconds
        while self._recent_feedback_traces and self._recent_feedback_traces[0][2] < cutoff:
            transition_id, _, _, _ = self._recent_feedback_traces.popleft()
            self.policy.discard_trace(transition_id)
        expression_cutoff = float(now) - self._expression_confirmation_window_seconds
        while self._recent_expression_traces and float(self._recent_expression_traces[0].get("at", 0.0)) < expression_cutoff:
            self._recent_expression_traces.popleft()

    def _forget_feedback_trace(self, transition_id: str) -> None:
        if not transition_id:
            return
        self._recent_feedback_traces = deque(
            (item for item in self._recent_feedback_traces if item[0] != transition_id),
            maxlen=8,
        )

    @staticmethod
    def _normalized_surface(text: str) -> str:
        return " ".join(str(text).split()).strip().casefold()

    def _arm_immediate_rejection(
        self, *, transition_id: str, action: str, target_xy: tuple[float, float] | None, now: float,
    ) -> None:
        rejected_text = ""
        if action == "desktop.type_text":
            kept: deque[dict[str, object]] = deque(maxlen=4)
            for trace in self._recent_expression_traces:
                if str(trace.get("transition_id", "")) == transition_id:
                    rejected_text = str(trace.get("text", ""))
                    # A phrase explicitly rejected with F23 must never remain
                    # eligible for a later F24 confirmation.
                    continue
                kept.append(trace)
            self._recent_expression_traces = kept
        self._pending_immediate_rejection = {
            "transition_id": transition_id,
            "action": action,
            "target_xy": target_xy,
            "rejected_text": rejected_text,
            "at": float(now),
        }

    def _register_rejection_cooldown(
        self, *, action: str, transition_id: str, target_xy: tuple[float, float] | None, now: float,
    ) -> None:
        if not action or action == "desktop.type_text":
            return
        previous = self._rejection_cooldowns.get(action)
        strikes = 1
        if previous is not None and float(previous.get("until", 0.0)) > float(now):
            strikes = int(previous.get("strikes", 1)) + 1
        remaining = min(
            self._rejection_max_decisions,
            self._rejection_base_decisions + max(0, strikes - 1) * 2,
        )
        self._rejection_cooldowns[action] = {
            "action": action,
            "transition_id": transition_id,
            "target_xy": target_xy,
            "strikes": strikes,
            "remaining": remaining,
            "until": float(now) + self._rejection_ttl_seconds,
        }
        self._write(
            "user_correction_working_memory_armed",
            rejected_transition_id=transition_id, rejected_action=action,
            rejected_target_xy=target_xy, strikes=strikes,
            remaining_decisions=remaining, ttl_seconds=self._rejection_ttl_seconds,
            meaning="temporarily_avoid_rejected_behavior", scripted_alternative=False,
            truth_confirmation=False,
        )

    def _clear_rejection_cooldown(self, action: str) -> None:
        if action:
            self._rejection_cooldowns.pop(action, None)

    def _constrain_frame_after_explicit_rejection(self, frame, now: float):
        """Keep explicit F23 rejection active across several decisions.

        Earlier code removed the rejected action for only one policy pass.  If
        the policy had a strong attractor, it selected the same action again on
        the next frame.  Here F23 creates short-lived working memory: rejected
        non-language action kinds remain unavailable for several decisions or
        until their TTL expires/F24 explicitly accepts that action.  The system
        still never invents the correct alternative.
        """
        expired = [
            action for action, item in self._rejection_cooldowns.items()
            if int(item.get("remaining", 0)) <= 0 or float(item.get("until", 0.0)) <= float(now)
        ]
        for action in expired:
            self._rejection_cooldowns.pop(action, None)

        blocked = set(self._rejection_cooldowns)
        if not blocked:
            return frame
        filtered = tuple(a for a in frame.affordances if a.action not in blocked)
        # Never create a frame with no possible action. If everything is blocked,
        # retain noop as a neutral escape hatch rather than silently restoring
        # the rejected attractor.
        if not filtered:
            filtered = tuple(a for a in frame.affordances if a.action == "desktop.noop")
        if not filtered:
            return frame

        removed = len(frame.affordances) - len(filtered)
        if removed <= 0:
            return frame
        active = []
        for action in sorted(blocked):
            item = self._rejection_cooldowns.get(action)
            if item is None:
                continue
            item["remaining"] = max(0, int(item.get("remaining", 0)) - 1)
            active.append({
                "action": action,
                "remaining": int(item["remaining"]),
                "strikes": int(item.get("strikes", 1)),
                "target_xy": item.get("target_xy"),
            })
        self._write(
            "user_correction_persistent_constraint",
            blocked=active, removed_affordances=removed,
            remaining_affordances=len(filtered),
            meaning="do_not_fall_back_into_recent_rejected_behavior",
            scripted_alternative=False, truth_confirmation=False,
        )
        return replace(frame, affordances=filtered)

    def _realize_after_explicit_rejection(self, decision: DesktopDecision):
        rejection = self._pending_immediate_rejection
        rejected_text = "" if not rejection else str(rejection.get("rejected_text", ""))
        if not rejection or str(rejection.get("action", "")) != "desktop.type_text" or not rejected_text:
            return self.expression.realize(decision)

        # F23 says the previous wording was not acceptable, but supplies no
        # correct target. Sample alternatives at a somewhat higher temperature
        # and refuse an immediate near-duplicate rather than teaching an invented
        # opposite.
        temperature = max(float(getattr(self.expression, "temperature", 0.8)) * 1.5, 1.15)
        rejected_norm = self._normalized_surface(rejected_text)
        last = None
        for attempt_index in range(3):
            candidate = self.expression.realize(decision, temperature=temperature + 0.15 * attempt_index)
            last = candidate
            candidate_norm = self._normalized_surface(candidate.utterance.text)
            similarity = SequenceMatcher(None, rejected_norm, candidate_norm).ratio() if rejected_norm and candidate_norm else 0.0
            if candidate_norm != rejected_norm and similarity < 0.90:
                self._write(
                    "user_correction_expression_changed",
                    rejected_transition_id=rejection.get("transition_id"),
                    rejected_text=rejected_text, replacement_text=candidate.utterance.text,
                    similarity=similarity, attempts=attempt_index + 1,
                    explicit_target=False, truth_confirmation=False,
                )
                return candidate
        raise EmptyExpressionError("F23 rejected the previous wording and Language repeated a near-duplicate")

    def _apply_user_physical_feedback(self, frame, now: float) -> tuple[bool, bool]:
        """Apply explicit physical feedback without turning silence into reward.

        Only a physical pointer takeover reaches shared Policy as a bounded
        motor correction. F21/F22 stay in the reader and F23/F24 stay in cursor
        guidance; legacy universal key observations are ignored here.
        """
        corrections = ()
        pointer_corrections = tuple(
            observation for observation in frame.observations
            if observation.kind == "desktop.user_pointer_correction"
        )
        confirmations = ()
        self._expire_feedback_traces(now)
        corrected = bool(corrections or pointer_corrections)
        confirmed = bool(confirmations)
        if corrections:
            self._last_user_correction_at = float(now)
            if not self._recent_feedback_traces:
                self._pending_immediate_rejection = None
                self._write(
                    "user_universal_correction", source="input_remapper_f23",
                    target_transition_id=None, target_action=None, updated=False,
                    immediate_replan=True, immediate_rejection_armed=False,
                    meaning="change_behavior", truth_confirmation=False,
                )
            else:
                transition_id, action, _, target_xy = self._recent_feedback_traces.pop()
                updated = bool(self.policy.learn(transition_id, -1.0))
                self._arm_immediate_rejection(
                    transition_id=transition_id, action=action, target_xy=target_xy, now=now,
                )
                self._register_rejection_cooldown(
                    action=action, transition_id=transition_id, target_xy=target_xy, now=now,
                )
                self._write(
                    "user_universal_correction", source="input_remapper_f23",
                    target_transition_id=transition_id, target_action=action,
                    target_xy=target_xy, advantage=-1.0, updated=updated,
                    immediate_replan=True, immediate_rejection_armed=True,
                    meaning="change_behavior", truth_confirmation=False,
                )
        elif pointer_corrections:
            self._last_user_correction_at = float(now)
        for observation in pointer_corrections:
            transition_id = str(observation.payload.get("target_transition_id", ""))
            action = str(observation.payload.get("target_action", ""))
            target_xy = observation.payload.get("previous_target_xy")
            if not (isinstance(target_xy, (tuple, list)) and len(target_xy) >= 2):
                target_xy = None
            else:
                target_xy = (float(target_xy[0]), float(target_xy[1]))

            # One takeover is a soft "adjust this" signal. Repeated takeover
            # of the same behavioural family near the same target means the
            # whole interaction pattern is stuck. In particular, left_click,
            # double_click and right_click on the same button are one pattern;
            # alternating click variants must not reset the strike counter.
            click_family = {
                "desktop.left_click", "desktop.double_click", "desktop.right_click",
            }
            behavior_family = "desktop.pointer_click" if action in click_family else action
            previous = self._pointer_behavior_corrections.get(behavior_family) if behavior_family else None
            same_target = False
            if previous is not None:
                age = float(now) - float(previous.get("at", float("-inf")))
                if age <= self._pointer_behavior_repeat_seconds:
                    old_xy = previous.get("target_xy")
                    if target_xy is None or old_xy is None:
                        same_target = True
                    else:
                        try:
                            dx = float(target_xy[0]) - float(old_xy[0])
                            dy = float(target_xy[1]) - float(old_xy[1])
                            same_target = (dx * dx + dy * dy) ** 0.5 <= self._pointer_behavior_target_radius
                        except (TypeError, ValueError, IndexError):
                            same_target = True
            strikes = int(previous.get("strikes", 1)) + 1 if (previous is not None and same_target) else 1
            if behavior_family:
                self._pointer_behavior_corrections[behavior_family] = {
                    "at": float(now), "strikes": strikes, "target_xy": target_xy,
                }

            advantage = -0.15 if strikes == 1 else -1.0
            updated = bool(self.policy.learn(transition_id, advantage)) if transition_id else False
            escalated = strikes >= 2 and bool(action)
            blocked_actions = tuple(sorted(click_family)) if (escalated and behavior_family == "desktop.pointer_click") else ((action,) if escalated else ())
            for blocked_action in blocked_actions:
                self._register_rejection_cooldown(
                    action=blocked_action, transition_id=transition_id, target_xy=target_xy, now=now,
                )
            self._forget_feedback_trace(transition_id)
            self._write(
                "user_pointer_correction", source="physical_pointer_motion",
                target_transition_id=transition_id or None, target_action=action or None,
                advantage=advantage, updated=updated, meaning=(
                    "change_repeated_behavior" if escalated else "adjust_motor_target_and_reconsider"
                ),
                directional_correction=True, correction_strikes=strikes,
                action_level_escalated=escalated,
                behavior_family=behavior_family or None,
                blocked_actions=list(blocked_actions),
                relative_dx=observation.payload.get("relative_dx"),
                relative_dy=observation.payload.get("relative_dy"),
                previous_target_xy=observation.payload.get("previous_target_xy"),
                corrected_target_xy=observation.payload.get("corrected_target_xy"),
                trajectory_duration_ms=observation.payload.get("trajectory_duration_ms"),
                finalized_by=observation.payload.get("finalized_by"),
                motor_learning=observation.payload.get("motor_learning"),
                truth_confirmation=False, silence_is_confirmation=False,
            )
        if confirmed:
            # F24 is universal explicit acceptance of the most recent Heiwa
            # behaviour.  It is positive evidence for the local action policy,
            # never evidence that a proposition is factually true.
            confirmed_transition_id = ""
            confirmed_action = ""
            confirmed_target_xy = None
            policy_updated = False
            if self._recent_feedback_traces:
                confirmed_transition_id, confirmed_action, _, confirmed_target_xy = self._recent_feedback_traces.pop()
                policy_updated = bool(self.policy.learn(confirmed_transition_id, +0.5))
                # Explicit F24 acceptance cancels any temporary F23 taboo for
                # this action kind. The human has now supplied newer evidence.
                self._clear_rejection_cooldown(confirmed_action)
                self._write(
                    "user_universal_confirmation", source="input_remapper_f24",
                    target_transition_id=confirmed_transition_id, target_action=confirmed_action,
                    target_xy=confirmed_target_xy, advantage=0.5, updated=policy_updated,
                    meaning="accept_recent_behavior", truth_confirmation=False,
                    silence_is_confirmation=False,
                )
            else:
                self._write(
                    "user_universal_confirmation", source="input_remapper_f24",
                    target_transition_id=None, target_action=None, updated=False,
                    meaning="accept_recent_behavior", truth_confirmation=False,
                    silence_is_confirmation=False,
                )

            # When the accepted behaviour was speech, F24 additionally makes the
            # exact surface form eligible for Language learning.  This remains
            # surface/correctness evidence only, not a truth label.
            expression_trace = None
            if confirmed_transition_id:
                kept = deque(maxlen=4)
                for trace in self._recent_expression_traces:
                    if str(trace.get("transition_id", "")) == confirmed_transition_id:
                        expression_trace = trace
                    else:
                        kept.append(trace)
                self._recent_expression_traces = kept
            elif self._recent_expression_traces:
                expression_trace = self._recent_expression_traces.pop()

            if expression_trace is not None:
                queued = self._queue_human_confirmed_expression(
                    text=str(expression_trace.get("text", "")),
                    transition_id=str(expression_trace.get("transition_id", "")),
                    message_id=str(expression_trace.get("message_id", "")),
                )
                self._write(
                    "user_expression_confirmation", source="input_remapper_f24",
                    target_transition_id=str(expression_trace.get("transition_id", "")),
                    message_id=str(expression_trace.get("message_id", "")),
                    text=str(expression_trace.get("text", "")), queued=bool(queued),
                    meaning="written_correctly", surface_correctness_only=True,
                    truth_confirmation=False, usefulness_confirmation=False,
                    policy_updated=policy_updated,
                )
        return corrected, confirmed

    def _visual_frame_digest(self, frame) -> str:
        return next((
            str(value.payload.get("frame_digest", ""))
            for value in frame.observations
            if value.kind == "desktop.neural_visual_state"
        ), "")

    def _arm_pending_external_response(
        self, *, transition_id: str, action: str, frame_digest: str,
        target_xy: tuple[float, float] | None, now: float,
    ) -> None:
        if self._pending_external_response is not None:
            return
        recent_expression = next((
            trace for trace in reversed(self._recent_expression_traces)
            if now - float(trace.get("at", float("-inf")))
            <= self._pending_external_response_expression_window_seconds
        ), None)
        if recent_expression is None:
            return
        self._pending_external_response = {
            "transition_id": transition_id,
            "commit_action": action,
            "expression_transition_id": str(recent_expression.get("transition_id", "")),
            "expression_message_id": str(recent_expression.get("message_id", "")),
            "armed_at": float(now),
            "baseline_digest": frame_digest,
            "last_digest": frame_digest,
            "last_change_at": float(now),
            "change_count": 0,
            "response_observed": False,
            "target_xy": target_xy,
            "last_block_log_step": -1,
        }
        self._write(
            "pending_external_response_armed",
            transition_id=transition_id, commit_action=action,
            expression_transition_id=str(recent_expression.get("transition_id", "")),
            expression_message_id=str(recent_expression.get("message_id", "")),
            target_xy=target_xy,
            semantics="commit_then_wait_for_world_consequence",
            hardcoded_button=False,
        )

    def _update_pending_external_response(self, frame, now: float) -> None:
        pending = self._pending_external_response
        if pending is None:
            return
        armed_at = float(pending.get("armed_at", now))
        age = max(0.0, now - armed_at)
        if age >= self._pending_external_response_max_seconds:
            self._write(
                "pending_external_response_released",
                transition_id=pending.get("transition_id"), reason="timeout",
                waited_seconds=age,
                response_observed=bool(pending.get("response_observed", False)),
                change_count=int(pending.get("change_count", 0)),
            )
            self._pending_external_response = None
            return

        digest = self._visual_frame_digest(frame)
        if digest and digest != str(pending.get("last_digest", "")):
            pending["last_digest"] = digest
            pending["last_change_at"] = float(now)
            pending["change_count"] = int(pending.get("change_count", 0)) + 1
            # The first very fast visual mutation is commonly the local commit
            # effect (input cleared/button changed).  A later mutation is the
            # evidence that the external world has begun answering.
            if age >= self._pending_external_response_local_change_grace_seconds:
                pending["response_observed"] = True
                self._write(
                    "pending_external_response_observed",
                    transition_id=pending.get("transition_id"),
                    waited_seconds=age,
                    change_count=int(pending.get("change_count", 0)),
                    evidence="delayed_visual_world_change",
                )

        last_change_at = float(pending.get("last_change_at", armed_at))
        # A stable frame is not enough on its own: streaming UIs can expose a
        # persistent Stop control during a pause. Keep all interactive actions
        # blocked until the world has produced several distinct changes and then
        # remained settled for the full stabilization interval.
        if bool(pending.get("response_observed", False)) and int(pending.get("change_count", 0)) < self._pending_external_response_min_change_count:
            if int(pending.get("last_block_log_step", -1)) != int(self.steps):
                pending["last_block_log_step"] = int(self.steps)
                self._write(
                    "pending_external_response_wait",
                    transition_id=pending.get("transition_id"),
                    blocked_affordances="interactive",
                    waited_seconds=age,
                    response_observed=True,
                    change_count=int(pending.get("change_count", 0)),
                    reason="waiting_for_more_world_changes",
                )
        if (bool(pending.get("response_observed", False))
                and int(pending.get("change_count", 0)) >= self._pending_external_response_min_change_count
                and age >= self._pending_external_response_min_wait_seconds
                and now - last_change_at >= self._pending_external_response_stable_seconds):
            self._write(
                "pending_external_response_released",
                transition_id=pending.get("transition_id"), reason="world_responded_and_settled",
                waited_seconds=age,
                change_count=int(pending.get("change_count", 0)),
            )
            self._pending_external_response = None

    def _constrain_frame_while_awaiting_external_response(self, frame, now: float):
        pending = self._pending_external_response
        if pending is None:
            return frame
        # Waiting is an actual organism state.  Do not let Policy turn it into
        # another click/key action.  noop remains available so Core keeps
        # observing and integrating the changing world while it waits.
        filtered = tuple(a for a in frame.affordances if a.action == "desktop.noop")
        if not filtered:
            return frame
        removed = len(frame.affordances) - len(filtered)
        if removed <= 0:
            return frame
        if int(pending.get("last_block_log_step", -1)) != int(self.steps):
            pending["last_block_log_step"] = int(self.steps)
            self._write(
                "pending_external_response_wait",
                transition_id=pending.get("transition_id"),
                blocked_affordances=removed,
                waited_seconds=max(0.0, now - float(pending.get("armed_at", now))),
                response_observed=bool(pending.get("response_observed", False)),
                change_count=int(pending.get("change_count", 0)),
            )
        return replace(frame, affordances=filtered)

    def _apply_desktop_policy_feedback(self, frame) -> None:
        """Learn causal competence and avoid persistent inertness.

        Screen change is predictive evidence, not a generic reward. A changed
        screen after an action proves causal effect only. Persistent owned noop
        on an unchanged desktop is the one local activity signal treated as bad.
        """
        for completed in frame.completed_actions:
            transition_id = str(completed.transition_id)
            if not transition_id:
                continue
            ownership = str(completed.payload.get("ownership", ""))
            if ownership != "heiwa":
                self.policy.discard_trace(transition_id)
                continue

            if completed.action == "desktop.noop":
                if completed.succeeded:
                    self._unchanged_noop_streak = 0
                    self.policy.discard_trace(transition_id)
                    self._write(
                        "policy_activity_feedback", transition_id=transition_id,
                        action=completed.action, outcome="world_changed_while_waiting",
                        advantage=0.0, updated=False,
                    )
                    continue

                self._unchanged_noop_streak += 1
                if self._unchanged_noop_streak < self.stagnation_noop_steps:
                    self.policy.discard_trace(transition_id)
                    self._write(
                        "policy_activity_feedback", transition_id=transition_id,
                        action=completed.action, outcome="brief_stillness",
                        streak=self._unchanged_noop_streak, advantage=0.0, updated=False,
                    )
                    continue

                updated = bool(self.policy.learn(transition_id, -self.stagnation_noop_penalty))
                self._write(
                    "policy_stagnation_feedback", transition_id=transition_id,
                    action=completed.action, outcome="persistent_stagnation",
                    streak=self._unchanged_noop_streak,
                    advantage=-self.stagnation_noop_penalty, updated=updated,
                )
                continue

            self._unchanged_noop_streak = 0
            if completed.succeeded:
                # A visual effect is causal evidence only, not approval. Keep the
                # policy trace alive for the explicit F23/F24 feedback window so
                # the human can still reject or accept this concrete action.
                self._write(
                    "policy_causal_feedback", transition_id=transition_id,
                    action=completed.action, outcome="observed_effect",
                    advantage=0.0, updated=False, explicit_feedback_trace_retained=True,
                )
                continue

            updated = bool(self.policy.learn(transition_id, -1.0))
            self._forget_feedback_trace(transition_id)
            self._write(
                "policy_causal_feedback", transition_id=transition_id,
                action=completed.action, outcome="no_observed_effect",
                advantage=-1.0, updated=updated,
            )

    def _maybe_adopt_background_reader(self) -> None:
        # The developmental trainer is subordinate to life, never the other way
        # around. A failed trainer cannot kill the organism; it is restarted with
        # backoff and resumes from its epoch checkpoint.
        if not self._reader_adopted and self.reader_candidate is not None and self.reader_report is not None:
            if self.reader_candidate.exists() and self.reader_report.exists():
                try:
                    report = json.loads(self.reader_report.read_text(encoding="utf-8"))
                    if bool(report.get("passed")):
                        inference = getattr(self.world, "inference", None)
                        if inference is not None and inference.adopt_reader_checkpoint(self.reader_candidate):
                            self._reader_adopted = True
                            self._write("reader_hot_adopted", candidate=str(self.reader_candidate), validation=report.get("validation", {}))
                            if self.desktop_checkpoint is not None: inference.save_checkpoint(self.desktop_checkpoint)
                except Exception as exc:
                    self._write("reader_hot_adopt_failed", error=f"{type(exc).__name__}: {exc}")

        if self._reader_adopted or self.reader_command is None:
            return
        if self.reader_trainer is not None and self.reader_trainer.poll() is None:
            return
        now = time.monotonic()
        if now < self._reader_restart_at:
            return
        previous_code = None if self.reader_trainer is None else self.reader_trainer.returncode
        try:
            self.reader_trainer = subprocess.Popen(self.reader_command)
            self._reader_restart_at = now + 30.0
            self._write("reader_background_started", resumed=True, previous_returncode=previous_code)
        except Exception as exc:
            self._reader_restart_at = now + 60.0
            self._write("reader_background_failed", error=f"{type(exc).__name__}: {exc}", organism_continues=True)

    def _maybe_adopt_background_language(self) -> None:
        candidate = self.language_candidate
        report_path = self.language_report
        if candidate is not None and report_path is not None and candidate.exists() and report_path.exists():
            try:
                mtime = candidate.stat().st_mtime_ns
                if mtime > self._language_candidate_mtime_ns:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    if bool(report.get("passed")) and bool(report.get("safe_language_candidate")):
                        result = self.language.load_checkpoint(candidate, load_optimizer=False)
                        if not result.compatible:
                            raise RuntimeError(result.reason)
                        # Optimizer moments from pre-adoption weights are stale.
                        self.language.optimizer = torch.optim.AdamW(
                            self.language.parameters(), lr=self.language.config.learning_rate
                        )
                        self._language_candidate_mtime_ns = mtime
                        self._write(
                            "language_hot_adopted", candidate=str(candidate),
                            updates=report.get("updates"), last_loss=report.get("last_loss"),
                            safe_language_candidate=True, truth_claim=False,
                        )
            except Exception as exc:
                self._write(
                    "language_hot_adopt_failed", error=f"{type(exc).__name__}: {exc}",
                    organism_continues=True,
                )

        if self.language_command is None:
            return
        if self.language_trainer is not None and self.language_trainer.poll() is None:
            return
        now = time.monotonic()
        if now < self._language_restart_at:
            return
        previous_code = None if self.language_trainer is None else self.language_trainer.returncode
        try:
            self.language_trainer = subprocess.Popen(self.language_command)
            self._language_restart_at = now + 30.0
            self._write(
                "language_background_started", resumed=True,
                previous_returncode=previous_code, observational_only=True,
            )
        except Exception as exc:
            self._language_restart_at = now + 60.0
            self._write(
                "language_background_failed", error=f"{type(exc).__name__}: {exc}",
                organism_continues=True,
            )

    def step(self, *, now: float | None = None):
        step_started = time.monotonic()
        now = time.monotonic() if now is None else float(now)
        self._maybe_adopt_background_reader()
        self._maybe_adopt_background_language()
        stage_started = time.monotonic()
        frame = self.world.poll(now)
        self._update_pending_external_response(frame, now)
        user_corrected, user_confirmed_expression = self._apply_user_physical_feedback(frame, now)
        self._apply_desktop_policy_feedback(frame)
        self._log_visual_reading_feedback(frame)
        observation_ms = (time.monotonic() - stage_started) * 1000.0
        stage_started = time.monotonic()
        self._integrate_visual_reading(frame)
        if frame.consequences:
            self.last_consequence = frame.consequences[-1]
            for consequence in frame.consequences:
                self._write("consequence", transition_id=consequence.transition_id,
                            consequence=asdict(consequence))
        self._teacher_tick(frame, now)
        integration_ms = (time.monotonic() - stage_started) * 1000.0
        transition_id = f"desktop-transition-{uuid4().hex}"
        stage_started = time.monotonic()
        decision_frame = self._constrain_frame_after_explicit_rejection(frame, now)
        decision_frame = self._constrain_frame_while_awaiting_external_response(decision_frame, now)
        decision = self.cycle.process(
            decision_frame, transition_id=transition_id, now=now,
            remember_core_for_learning=False,
            exploration_temperature=(
                max(self.policy_exploration_temperature, 1.5)
                if user_corrected else self.policy_exploration_temperature
            ),
        )
        decision_ms = (time.monotonic() - stage_started) * 1000.0
        diagnostics = getattr(self.policy, "score_diagnostics", None)
        if callable(diagnostics):
            score_log = diagnostics()
            if score_log:
                self._write("policy_scores", transition_id=transition_id, **score_log)
        self.last_stage_latency_ms = {
            "observation_and_consequence": observation_ms,
            "reading_core_teacher_tick": integration_ms,
            "decision": decision_ms,
            "motor_execution": 0.0,
        }
        intent = decision.intent
        active_rejection = self._pending_immediate_rejection
        if active_rejection is not None:
            self._write(
                "user_correction_immediate_replan",
                rejected_transition_id=active_rejection.get("transition_id"),
                rejected_action=active_rejection.get("action"),
                selected_action=intent.action,
                same_action=(intent.action == str(active_rejection.get("action", ""))),
                core_saw_feedback=True, scripted_alternative=False,
                truth_confirmation=False,
            )
        if not self.autonomous:
            self.policy.discard_trace(transition_id)
            discard = getattr(self.core, "discard_trace", None)
            if callable(discard): discard(transition_id)
            self._write("observation_only", transition_id=transition_id)
            self._pending_immediate_rejection = None
            self._finish_step(step_started); return None
        utterance = ""
        if intent.action == "desktop.type_text":
            try:
                expression = self._realize_after_explicit_rejection(decision)
            except EmptyExpressionError:
                self.policy.discard_trace(transition_id)
                self._write("empty_language_intention", transition_id=transition_id,
                            language_teacher=False,
                            explicit_user_rejection=bool(self._pending_immediate_rejection))
                self._pending_immediate_rejection = None
                self._finish_step(step_started)
                return None
            intent = expression.intent; utterance = expression.utterance.text
            self.last_expression_decision = decision
            self._write(
                "heiwa_expression_intended", transition_id=transition_id,
                text=utterance, codepoints=[f"U+{ord(ch):04X}" for ch in utterance],
                source="language.realize", self_generated=True,
                language_teacher=False, language_updated=False,
                silence_is_confirmation=False,
            )
        frame_digest = next((str(value.payload.get("frame_digest", ""))
                             for value in frame.observations
                             if value.kind == "desktop.neural_visual_state"), "")
        # Ten-second bucket prevents immediate crash/restart replay without
        # permanently banning a legitimate repeated action on an unchanged UI.
        action_id = _event_id({"frame_digest": frame_digest, "action": intent.action,
                               "parameters": dict(intent.parameters), "time_bucket": int(now // 10)})
        if self.teacher_store.action_was_executed(action_id):
            self.policy.discard_trace(transition_id)
            self._write("duplicate_action_suppressed", transition_id=transition_id,
                        action_id=action_id, action=intent.action)
            self._pending_immediate_rejection = None
            self._finish_step(step_started)
            return None
        stage_started = time.monotonic()
        attempt = self.world.execute(intent, now=time.monotonic())
        self.last_stage_latency_ms["motor_execution"] = (time.monotonic() - stage_started) * 1000.0
        self.last_attempt = attempt
        if attempt.accepted:
            self.teacher_store.record_action(action_id)
            # Preserve a short causal Policy trace for explicit F23 correction.
            # Mere passage of time/silence never becomes positive evidence.
            raw_target = attempt.payload.get("neural_target")
            target_xy = None
            if isinstance(raw_target, (tuple, list)) and len(raw_target) >= 2:
                try:
                    target_xy = (float(raw_target[0]), float(raw_target[1]))
                except (TypeError, ValueError):
                    target_xy = None
            self._recent_feedback_traces.append((transition_id, intent.action, float(now), target_xy))
            if intent.action in {"desktop.left_click", "desktop.double_click", "desktop.right_click"}:
                self._arm_pending_external_response(
                    transition_id=transition_id, action=intent.action,
                    frame_digest=frame_digest, target_xy=target_xy, now=float(now),
                )
        else:
            reason = str(attempt.payload.get("reason", ""))
            if "action rate limited" in reason:
                # A scheduler throttle says nothing about whether the selected
                # action was sensible. Do not train Policy with a false -1.
                self.policy.discard_trace(transition_id)
                self._write(
                    "policy_scheduler_feedback", transition_id=transition_id,
                    action=intent.action, outcome="rate_limited",
                    advantage=0.0, updated=False,
                )
            else:
                updated = bool(self.policy.learn(transition_id, -1.0))
                self._write(
                    "policy_causal_feedback", transition_id=transition_id,
                    action=intent.action, outcome="execution_rejected",
                    advantage=-1.0, updated=updated,
                )
        self._write(
            "action_attempt", transition_id=transition_id, intent=asdict(intent),
            attempt=asdict(attempt), heiwa_text=utterance,
            language_teacher=False, selected_by_same_core=True,
        )
        if utterance and attempt.accepted:
            message_id = f"heiwa-{transition_id}"
            if self.teacher_store.record_heiwa(message_id):
                self.last_heiwa_utterance = utterance
                self.last_heiwa_message_id = message_id
                self._pending_expression_audit = {
                    "transition_id": transition_id,
                    "message_id": message_id,
                    "intended_text": utterance,
                    "created_step": self.steps,
                    "checks": 0,
                }
                self._write(
                    "heiwa_expression_generated", transition_id=transition_id,
                    message_id=message_id, text=utterance,
                    physical_action_accepted=True, language_teacher=False,
                )
                # The utterance remains context for visual consequence auditing,
                # never teacher data merely because a desktop key was pressed.
                self._recent_expression_traces.append({
                    "transition_id": transition_id, "message_id": message_id,
                    "text": utterance, "at": float(now),
                })
        self._pending_immediate_rejection = None
        self._finish_step(step_started)
        return attempt

    def _finish_step(self, started: float) -> None:
        self.steps += 1; self.last_step_latency_ms = (time.monotonic() - started) * 1000
        if time.monotonic() >= self._next_checkpoint:
            self.save_state(reason="periodic")
            self._next_checkpoint = time.monotonic() + self.checkpoint_interval
        self._write_status(running=True)

    def _write_status(self, *, running: bool) -> None:
        learner = getattr(getattr(self.world, "inference", None), "learner", None)
        _atomic_json(self.status_path, {
            "running": running, "pid": __import__("os").getpid(),
            "episode_id": self.episode_id, "steps": self.steps,
            "uptime_seconds": round(time.monotonic() - self.started_at, 3),
            "last_step_latency_ms": round(self.last_step_latency_ms, 3),
            "last_stage_latency_ms": {
                key: round(value, 3) for key, value in self.last_stage_latency_ms.items()
            },
            "teacher_status": self.teacher_channel.status,
            "teacher_error": self.teacher_channel.last_error,
            "teacher_requests": self.teacher_channel.requests,
            "teacher_failures": self.teacher_channel.failures,
            "language_background": {
                "enabled": self.language_command is not None,
                "trainer_alive": self.language_trainer is not None and self.language_trainer.poll() is None,
                "candidate": None if self.language_candidate is None else str(self.language_candidate),
                "queue": str(self.language_observation_queue),
                "candidate_mtime_ns": self._language_candidate_mtime_ns,
                "visual_learning_confidence": self.language_observation_visual_confidence,
                "observational_only": True, "truth_claim": False,
            },
            "online_learning": None if learner is None else learner.learning_state(),
            "online_learning_last": None if learner is None else learner.last_metrics,
            "replay_size": None if learner is None else len(learner.replay),
            "autonomous": self.autonomous,
            "pending_expression_audit": None if self._pending_expression_audit is None else {
                "transition_id": self._pending_expression_audit.get("transition_id"),
                "message_id": self._pending_expression_audit.get("message_id"),
                "checks": self._pending_expression_audit.get("checks", 0),
            },
        })

    def save_state(self, *, reason: str) -> None:
        started = time.monotonic()
        inference = getattr(self.world, "inference", None)
        if inference is not None and self.desktop_checkpoint is not None:
            inference.save_checkpoint(self.desktop_checkpoint)
        save_core = getattr(self.core, "save_checkpoint", None)
        if callable(save_core): save_core(self.core_checkpoint)
        save_policy = getattr(self.policy, "save_checkpoint", None)
        if callable(save_policy): save_policy(self.policy_checkpoint)
        if self.language_checkpoint is not None:
            self.language.save_checkpoint(
                self.language_checkpoint,
                training_metadata={"runtime": "embodied", "online": True,
                                   "steps": self.steps, "own_speech_teacher": False},
            )
        self._write("checkpoint", reason=reason,
                    latency_ms=round((time.monotonic() - started) * 1000, 3))

    def run(self, *, interval: float = 1.0, max_steps: int | None = None) -> int:
        stopped = False
        def stop(*_: object) -> None:
            nonlocal stopped; stopped = True
        signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
        steps = 0
        try:
            while not stopped and (max_steps is None or steps < max_steps):
                try:
                    attempt = self.step()
                except PortalScreenCastError as exc:
                    # A portal/PipeWire stream can disappear when the desktop
                    # session or compositor transiently changes. Reset only
                    # that transport and keep the organism process alive.
                    portal = getattr(getattr(self.world, "inference", None), "portal", None)
                    close_portal = getattr(portal, "close", None)
                    if callable(close_portal):
                        try: close_portal()
                        except Exception: pass
                    self._write("world_degraded", channel="desktop_capture",
                                error=f"{type(exc).__name__}: {exc}",
                                retrying=True, organism_continues=True)
                    self._write_status(running=True)
                    time.sleep(min(max(interval, 0.1), 2.0))
                    continue
                steps += 1
                print(json.dumps({"step": steps, "attempt": None if attempt is None else
                                  {"action": attempt.action, "accepted": attempt.accepted,
                                   "transition_id": attempt.transition_id}}, ensure_ascii=False), flush=True)
                if interval: time.sleep(interval)
        finally:
            try:
                # An accepted final attempt must receive its causal after-frame
                # before persistence; shutdown itself never selects a new action.
                if getattr(self.world, "_pending", None) is not None:
                    try:
                        final_frame = self.world.poll(time.monotonic())
                        for consequence in final_frame.consequences:
                            self.last_consequence = consequence
                            self._write("consequence", transition_id=consequence.transition_id,
                                        consequence=asdict(consequence), shutdown_flush=True)
                    except Exception as exc:
                        self._write("shutdown_flush_failed", error=f"{type(exc).__name__}: {exc}")
                self.save_state(reason="graceful_shutdown")
            finally:
                self._write_status(running=False)
                if self.reader_trainer is not None and self.reader_trainer.poll() is None:
                    try: self.reader_trainer.terminate(); self.reader_trainer.wait(timeout=5)
                    except Exception:
                        try: self.reader_trainer.kill()
                        except Exception: pass
                if self.language_trainer is not None and self.language_trainer.poll() is None:
                    try: self.language_trainer.terminate(); self.language_trainer.wait(timeout=5)
                    except Exception:
                        try: self.language_trainer.kill()
                        except Exception: pass
                self.teacher_channel.close(); self.world.close()
        return 0


def _load_language(path: Path) -> NeuralLanguageOrgan:
    organ = NeuralLanguageOrgan(NeuralLanguageConfig.local())
    if path.exists():
        result = organ.load_checkpoint(path)
        if not result.compatible:
            raise RuntimeError(f"incompatible Language checkpoint {path}: {result.reason}")
    return organ


def _teacher(args, backend: str, config: AutonomyConfig, action_arbiter: PhysicalActionArbiter) -> TeacherAdapter:
    if backend == "none": return NullTeacherAdapter()
    if backend == "ollama": return OllamaTeacherAdapter(args.teacher_model)
    if backend == "command": return CommandTeacherAdapter(args.teacher_command)
    raise ValueError(backend)


def main() -> int:
    parser = argparse.ArgumentParser(description="One multimodal embodied Heiwa runtime")
    parser.add_argument("--config", type=Path, default=Path("state/runtime_config.json"))
    parser.add_argument("--desktop-checkpoint", type=Path, default=Path("state/neural_desktop.pt"))
    parser.add_argument("--language-checkpoint", type=Path, default=Path("state/language_developmental_dialogue.v5-byte.pt"))
    parser.add_argument("--policy-checkpoint", type=Path, default=Path("state/embodied_runtime/policy.pt"))
    parser.add_argument("--core-checkpoint", type=Path, default=Path("state/embodied_runtime/core.pt"))
    parser.add_argument("--state-dir", type=Path, default=Path("state/embodied_runtime"))
    parser.add_argument("--teacher", choices=("none", "ollama", "command"))
    parser.add_argument("--teacher-model", default="qwen3:8b")
    parser.add_argument("--teacher-command", default="")
    parser.add_argument("--allow-clicks", action="store_true")
    parser.add_argument("--allow-right-clicks", action="store_true")
    parser.add_argument("--allow-typing", action="store_true")
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument("--interval", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        status = args.state_dir / "status.json"
        print(status.read_text(encoding="utf-8") if status.exists() else '{"running": false, "status": "not_initialized"}')
        return 0
    config = load_config(args.config)
    backend = args.teacher or (config.teacher_backend if config.teacher_channel else "none")
    if backend == "command" and not args.teacher_command:
        parser.error("--teacher command requires --teacher-command")
    if not config.vision: parser.error("unified embodied runtime requires vision=true")
    action_arbiter = PhysicalActionArbiter()
    teacher = _teacher(args, backend, config, action_arbiter)
    language = _load_language(args.language_checkpoint)
    core = PersonalCognitiveCore()
    policy = DynamicAffordancePolicy(DynamicPolicyConfig(
        specialist_prior_weight=config.policy_specialist_prior_weight,
        exploration_mixture=config.policy_exploration_mixture,
    ))
    core.load_checkpoint(args.core_checkpoint)
    policy.load_checkpoint(args.policy_checkpoint)
    inference = NeuralDesktopInference(
        args.desktop_checkpoint, replay_path=args.state_dir / "desktop_replay.pt"
    )
    confidence = config.minimum_action_confidence if args.min_confidence is None else args.min_confidence
    world = NeuralDesktopWorldAdapter(
        inference, safety=NeuralSafetyGate(
            confidence=confidence,
            max_actions_per_second=config.maximum_actions_per_second,
        ),
        allow_clicks=config.clicks or args.allow_clicks,
        allow_right_clicks=config.right_clicks or args.allow_right_clicks,
        allow_typing=config.typing or args.allow_typing,
        allow_mouse=config.mouse,
        allow_keyboard=config.keyboard or args.allow_typing,
        capture_input=config.capture_user_input,
        online_learning=config.online_learning,
        learn_stationary_desktop=config.learn_stationary_desktop,
        action_arbiter=action_arbiter,
    )
    teacher_key = sha256(teacher.teacher_id.encode()).hexdigest()[:16]
    store = TeacherSessionStore(
        args.state_dir / "teacher_sessions" / f"{teacher_key}.json",
        teacher_id=teacher.teacher_id,
    )
    reader_trainer = None
    reader_command: tuple[str, ...] | None = None
    reader_candidate = args.state_dir / "reader_candidate.pt"
    reader_progress = args.state_dir / "reader_progress.pt"
    reader_report = Path("state/reader_training/report.json")
    if config.background_reader_training:
        # Resume developmental progress across login/service/computer restarts.
        # The live checkpoint remains usable immediately; progress is adopted
        # only after the reader's validation gate passes.
        reader_source = reader_progress if reader_progress.exists() else args.desktop_checkpoint
        command = [
            sys.executable, "-m", "worlds.desktop.training.train_reader",
            "--checkpoint", str(reader_source),
            "--output", "state/reader_training",
            "--diagnostics", "state/vision_diagnostics/20260829-205522",
            "--train-samples", str(config.reader_train_samples),
            "--val-samples", str(config.reader_val_samples),
            "--epochs-per-stage", str(config.reader_epochs_per_stage),
            "--batch-size", str(config.reader_batch_size),
            "--lr", str(config.reader_learning_rate),
            "--save-on-pass", "--save-checkpoint", str(reader_candidate),
            "--progress-checkpoint", str(reader_progress),
        ]
        reader_command = tuple(command)
        try:
            reader_trainer = subprocess.Popen(command)
        except Exception as exc:
            print(json.dumps({"reader_background":"degraded","error":f"{type(exc).__name__}: {exc}","organism_continues":True}), flush=True)

    language_trainer = None
    language_command: tuple[str, ...] | None = None
    language_observation_queue = args.state_dir / "language_observations.jsonl"
    language_candidate = args.state_dir / "language_candidate.pt"
    language_progress = args.state_dir / "language_continual_progress.pt"
    language_learning_state = args.state_dir / "language_continual_state.json"
    language_report = args.state_dir / "language_continual_report.json"
    if config.background_language_learning:
        language_source = language_progress if language_progress.exists() else args.language_checkpoint
        command = [
            sys.executable, "-m", "language.continual",
            "--checkpoint", str(language_source),
            "--progress-checkpoint", str(language_progress),
            "--candidate", str(language_candidate),
            "--queue", str(language_observation_queue),
            "--state", str(language_learning_state),
            "--report", str(language_report),
            "--poll-seconds", str(config.language_learning_poll_seconds),
            "--save-every", str(config.language_learning_save_every),
        ]
        language_command = tuple(command)
        try:
            language_trainer = subprocess.Popen(command)
        except Exception as exc:
            print(json.dumps({
                "language_background":"degraded",
                "error":f"{type(exc).__name__}: {exc}",
                "organism_continues":True,
            }), flush=True)

    runtime = EmbodiedRuntime(
        core=core, language=language, policy=policy, world=world,
        teacher=teacher, teacher_store=store, event_log=args.state_dir / "events.jsonl",
        state_dir=args.state_dir, desktop_checkpoint=args.desktop_checkpoint,
        language_checkpoint=args.language_checkpoint, core_checkpoint=args.core_checkpoint,
        policy_checkpoint=args.policy_checkpoint,
        checkpoint_interval=config.checkpoint_interval_seconds,
        teacher_interval=config.teacher_interval_seconds,
        autonomous=config.autonomous,
        reading_minimum_confidence=config.reading_minimum_confidence,
        policy_exploration_temperature=config.policy_exploration_temperature,
        stagnation_noop_steps=config.stagnation_noop_steps,
        stagnation_noop_penalty=config.stagnation_noop_penalty,
        reader_trainer=reader_trainer, reader_candidate=reader_candidate, reader_report=reader_report,
        reader_command=reader_command,
        language_trainer=language_trainer, language_candidate=language_candidate,
        language_report=language_report, language_command=language_command,
        language_observation_queue=language_observation_queue,
        language_observation_visual_confidence=config.language_observation_visual_confidence,
    )
    print(json.dumps({"runtime": "embodied", "one_core": True, "teacher": teacher.teacher_id,
                      "conversation_id": store.conversation_id, "pixel_only": True,
                      "autonomous": config.autonomous,
                      "allow_clicks": config.clicks or args.allow_clicks,
                      "allow_typing": config.typing or args.allow_typing,
                      "online_learning": config.online_learning,
                      "config": str(args.config)}))
    interval = config.observation_interval_seconds if args.interval is None else args.interval
    return runtime.run(interval=interval, max_steps=args.steps)


if __name__ == "__main__":
    raise SystemExit(main())
