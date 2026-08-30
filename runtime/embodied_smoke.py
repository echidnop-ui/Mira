from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import math
import tempfile
import time

import torch

from contracts.cognition import ActionIntent
from core.personal import PersonalCognitiveCore, PersonalCoreConfig
from language.interface import LanguageRepresentation, RealizedUtterance
from runtime.embodied import EmbodiedRuntime, NullTeacherAdapter
from runtime.teacher import TeacherMessage, TeacherSessionStore
from worlds.desktop.inference import NeuralMotorProposal
from worlds.desktop.neural import NeuralDesktopBody, NeuralDesktopConfig
from worlds.desktop.online_learning import OnlineDesktopLearner
from worlds.desktop.motor import MotorResult
from worlds.desktop.neural_adapter import NeuralDesktopWorldAdapter, NeuralSafetyGate
from worlds.desktop.training.input_events import EV_REL, REL_X, RawInputEvent
from worlds.desktop.portal_screencast import PortalScreenCastError


class _Language:
    representation_size = 96
    def __init__(self) -> None: self.inputs = []; self.generated = []; self.teacher_targets = []
    def encode(self, value, *, remember_for_learning=True):
        self.inputs.append(value)
        vector = tuple((len(value.text) % 7) / 7 for _ in range(96))
        return LanguageRepresentation(value.transition_id, value.source_kind, vector, len(value.text), value.metadata)
    def learn_representation(self, transition_id, representation_gradient): return False
    def has_trace(self, transition_id): return False
    def discard_trace(self, transition_id): return True
    def realize(self, request):
        self.generated.append(request)
        return RealizedUtterance(request.transition_id, "живой ответ", 11, True,
                                 {"language_teacher": False})
    def learn_expression(self, cognitive_state, target):
        self.teacher_targets.append(target); return 0.25


class _Policy:
    def __init__(self) -> None:
        self.actions = ["desktop.move_pointer", "desktop.key_press", "desktop.type_text"]
        self.states = []; self.feedback = []
    def choose(self, state, affordances, *, now):
        self.states.append(state)
        # Neural Desktop now exposes the low-level action selected by its
        # motor head; shared Policy still owns the intention boundary.
        chosen = tuple(affordances)[0]
        return ActionIntent(f"intent-{len(self.states)}", state.transition_id, chosen.id,
                            chosen.world, chosen.action, now, {})
    def learn(self, transition_id, advantage): self.feedback.append((transition_id, advantage)); return True
    def discard_trace(self, transition_id): return True


class _Inference:
    class _Model:
        def predict_next(self, before, kind, parameters):
            return before.latent + 0.05, torch.zeros(before.latent.shape[0], before.patch_tokens.shape[1])
    def __init__(self) -> None:
        self.tick = 0; self.state = None; self.previous_state = None
        self.last_frame = None; self.previous_frame = None
        self.remembered = []; self.device = torch.device("cpu"); self.model = self._Model()
        self.portal = SimpleNamespace(close=lambda: None)
    def capture_and_propose(self):
        self.tick += 1
        value = self.tick * 0.1
        self.previous_state = self.state; self.previous_frame = self.last_frame
        self.last_frame = torch.full((1, 3, 16, 16), value)
        self.state = SimpleNamespace(
            latent=torch.full((1, 192), value),
            patch_tokens=torch.full((1, 24, 192), value),
            region_positions=torch.zeros(1, 12, 4), salience=torch.zeros(1, 12),
        )
        self.state.detached = lambda: self.state
        action = ("move_pointer", "key_press", "type_text")[(self.tick - 1) % 3]
        action_distribution = tuple(1.0 if kind == action else 0.0 for kind in
                                    ("noop", "move_pointer", "left_click", "double_click", "right_click",
                                     "scroll", "key_press", "type_text", "key_chord"))
        return NeuralMotorProposal(
            action=action, x=120, y=80, confidence=0.9,
            action_distribution=action_distribution, target_distribution=(1.0,),
            screen_width=800, screen_height=600, key_code=30,
            frame_digest=f"frame-{self.tick}",
            target_bbox=(0.1, 0.1, 0.3, 0.2), trajectory=(0.0, 0.0, 0.0),
            read_byte_ids=(), read_confidence=0.0,
        )
    def remember_action(self, kind, parameters): self.remembered.append((kind, parameters))
    def learn_consequence(self, before, action_kind, parameters, after):
        return {"prediction_loss": 0.1, "temporal_error": 0.05, "change_error": 0.05}
    def learn_transition(self, **kwargs):
        return {"fast_loss": 0.1, "fast_updated": True, "fast_parameter_delta": 0.001,
                "target_error_px": 4.0, "replay_size": 1, "ownership": kwargs["ownership"]}
    def close(self): pass


class _Motor:
    def __init__(self) -> None: self.calls = []
    def move_pointer_absolute(self, x, y): self.calls.append(("mouse", x, y)); return MotorResult(0, "", "")
    def move_pointer_trajectory(self, points, *, duration):
        self.calls.append(("mouse", *points[-1])); return MotorResult(0, "", "")
    def press_key(self, code): self.calls.append(("keyboard", code)); return MotorResult(0, "", "")
    def type_unicode_linux(self, text): self.calls.append(("type", text)); return MotorResult(0, "", "")


class _UserInput:
    def __init__(self) -> None: self.sent = False
    def drain(self, before_ns, after_ns):
        if self.sent: return ()
        self.sent = True
        return (RawInputEvent((before_ns + after_ns) // 2, "/dev/input/event-test", "physical mouse", EV_REL, REL_X, 4),)
    def close(self): pass


class _FakeTeacher:
    teacher_id = "fake-test-teacher"
    def __init__(self) -> None: self.calls = 0; self.actuator = None
    def respond(self, context):
        self.calls += 1
        if self.calls == 1:
            return TeacherMessage("teacher-1", context.conversation_id,
                                  "Передвинь курсор в другую область.", "ordinary", context.event_id)
        if self.calls == 2:
            raise TimeoutError("synthetic teacher timeout")
        if self.calls == 3:
            return TeacherMessage("teacher-wrong-rid", context.conversation_id,
                                  "stale", "ordinary", context.event_id, {"rid": "wrong"})
        if self.calls == 4:
            return TeacherMessage("teacher-4", context.conversation_id,
                                  "живой исправленный ответ", "explicit_linguistic_correction", context.event_id)
        return None


class _TransientCaptureWorld:
    def __init__(self, world) -> None:
        self.world = world; self.world_id = world.world_id; self.inference = world.inference
        self.failed = False
    def poll(self, now):
        if not self.failed:
            self.failed = True
            raise PortalScreenCastError("synthetic transient PipeWire loss")
        return self.world.poll(now)
    def execute(self, intent, now): return self.world.execute(intent, now)
    def close(self): self.world.close()


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); language = _Language(); policy = _Policy(); teacher = _FakeTeacher()
        core = PersonalCognitiveCore(PersonalCoreConfig.smoke(device="cpu"))
        inference = _Inference(); motor = _Motor()
        world = NeuralDesktopWorldAdapter(
            inference, motor=motor, safety=NeuralSafetyGate(confidence=0.0, max_actions_per_second=1000),
            allow_clicks=True, allow_typing=True, capture_input=False,
        )
        store_path = root / "teacher.json"
        store = TeacherSessionStore(store_path, teacher_id=teacher.teacher_id)
        runtime = EmbodiedRuntime(
            core=core, language=language, policy=policy, world=world, teacher=teacher,
            teacher_store=store, event_log=root / "events.jsonl",
            state_dir=root, teacher_interval=0.001,
        )
        feedback_before = list(policy.feedback)
        local_visual_keys = SimpleNamespace(observations=(
            SimpleNamespace(kind="desktop.cursor_rejection", payload={}),
            SimpleNamespace(kind="desktop.cursor_acceptance", payload={}),
            SimpleNamespace(kind="desktop.visual_reading_feedback", payload={"accepted": True}),
        ))
        corrected, confirmed = runtime._apply_user_physical_feedback(
            local_visual_keys, time.monotonic(),
        )
        assert not corrected and not confirmed and policy.feedback == feedback_before
        observable_records = []
        runtime.step(now=time.monotonic()); observable_records.append(runtime.last_observable_record); time.sleep(0.01)
        # A physical event in the pending causal window is external action,
        # never approval/rejection or a clean Heiwa completion.
        world._input = _UserInput(); world._input_status = "linux-evdev"
        for _ in range(8):
            runtime.step(now=time.monotonic()); observable_records.append(runtime.last_observable_record); time.sleep(0.01)
        # Teacher work is asynchronous. On a loaded ROCm/desktop host the
        # fourth synthetic reply may arrive after eight fast runtime steps,
        # so wait for the explicit correction instead of making this test
        # scheduler-speed dependent.
        deadline = time.monotonic() + 3.0
        while not language.teacher_targets and time.monotonic() < deadline:
            runtime.step(now=time.monotonic()); observable_records.append(runtime.last_observable_record); time.sleep(0.01)
        assert policy.states and core.telemetry().forward_calls >= 4
        assert language.inputs and language.inputs[0].source_kind == "mentor"
        assert any(call[0] == "mouse" for call in motor.calls)
        assert any(call[0] == "keyboard" for call in motor.calls)
        assert any(call[0] == "type" for call in motor.calls)
        records = [__import__("json").loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
        own = [r for r in records if r["kind"] == "action_attempt" and r.get("heiwa_text")]
        physical = [r for r in records if r["kind"] == "action_attempt"]
        assert own and own[-1]["language_teacher"] is False
        assert physical and all(r["language_teacher"] is False for r in physical)
        corrections = [r for r in records if r["kind"] == "teacher_message" and r["language_teacher"]]
        assert corrections and corrections[-1]["language_updated"] is True
        assert language.teacher_targets and language.teacher_targets[-1].source_kind == "mentor"
        assert teacher.actuator is None
        assert any(r["kind"] == "teacher_degraded" and r["organism_continues"] for r in records)
        assert any(r["kind"] == "teacher_invalid" and r["organism_continues"] for r in records)
        consequences = [r for r in records if r["kind"] == "consequence"]
        assert consequences[0]["consequence"]["payload"]["ownership"] == "mixed_user_interference"
        observable = next(item for item in reversed(observable_records)
                          if item and item["causal"]["consequences"] and item["online"]["updates"])
        assert observable["frame"]["id"].startswith("frame-")
        assert observable["neural"] == {"target": [120, 80], "bbox": (0.1, 0.1, 0.3, 0.2),
                                        "confidence": 0.9}
        assert observable["causal"]["attempt"]["id"].startswith("desktop-attempt-")
        linked = observable["causal"]["completions"][0]
        consequence_link = observable["causal"]["consequences"][0]
        assert linked["transition_id"] == consequence_link["transition_id"]
        assert linked["id"] == consequence_link["completed_action_id"]
        source_consequence = next(r["consequence"] for r in consequences
                                  if r["consequence"]["id"] == consequence_link["id"])
        assert source_consequence["transition_id"] == consequence_link["transition_id"]
        assert source_consequence["completed_action_id"] == consequence_link["completed_action_id"]
        assert observable["online"]["updates"]["fast_updated"] is True
        assert observable["online"]["parameter_delta"]["fast_parameter_delta"] == 0.001
        assert observable["rolling"]["pointer_correction_samples"] > 0
        move_only_world = NeuralDesktopWorldAdapter(
            _Inference(), motor=_Motor(),
            safety=NeuralSafetyGate(confidence=0.0, max_actions_per_second=1000),
            allow_clicks=True, allow_typing=True, move_only=True, capture_input=False,
        )
        move_only_frame = move_only_world.poll(time.monotonic())
        assert {item.action for item in move_only_frame.affordances} == {
            "desktop.noop", "desktop.move_pointer"
        }
        move_only_world.close()
        torch.manual_seed(17)
        tiny_config = NeuralDesktopConfig(
            image_height=64, image_width=96, hidden_size=64, token_size=64,
            region_count=4, transformer_layers=1, transformer_heads=4,
            action_embed_size=16, task_context_size=32, max_read_bytes=12,
        )
        tiny_model = NeuralDesktopBody(tiny_config)
        tiny_learner = OnlineDesktopLearner(tiny_model, device=torch.device("cpu"), work_batch=1)
        before_pixels = torch.rand(1, 3, 64, 96)
        after_pixels = (before_pixels + 0.05).clamp(0, 1)
        before_state = tiny_model.observe(before_pixels).detached()
        after_state = tiny_model.observe(after_pixels).detached()
        actual_metrics = tiny_learner.process_transition(
            before_frame=before_pixels, after_frame=after_pixels,
            before_state=before_state, after_state=after_state,
            action_kind=1, parameters=(0.25, 0.5, 0.0, 0.0),
            ownership="heiwa", completion=True,
        )
        assert actual_metrics["fast_updated"] is True
        assert math.isfinite(float(actual_metrics["fast_parameter_delta"]))
        assert float(actual_metrics["fast_parameter_delta"]) > 0.0
        # Persisted IDs survive restart and reject duplicate inbound/outbound messages.
        resumed = TeacherSessionStore(store_path, teacher_id=teacher.teacher_id)
        assert resumed.conversation_id == store.conversation_id
        duplicate = TeacherMessage("teacher-1", resumed.conversation_id, "duplicate")
        assert resumed.accept(duplicate) is False
        assert resumed.record_heiwa("heiwa-duplicate") is True
        resumed_again = TeacherSessionStore(store_path, teacher_id=teacher.teacher_id)
        assert resumed_again.record_heiwa("heiwa-duplicate") is False
        resumed_again.record_action("action-duplicate")
        final_resume = TeacherSessionStore(store_path, teacher_id=teacher.teacher_id)
        assert final_resume.action_was_executed("action-duplicate") is True
        # A transient portal/PipeWire failure degrades and retries the World;
        # it does not terminate the unified organism process.
        resilience_world = _TransientCaptureWorld(NeuralDesktopWorldAdapter(
            _Inference(), motor=_Motor(),
            safety=NeuralSafetyGate(confidence=0.0, max_actions_per_second=1000),
            allow_clicks=True, allow_typing=True, capture_input=False,
        ))
        resilience_log = root / "resilience.jsonl"
        resilience = EmbodiedRuntime(
            core=PersonalCognitiveCore(PersonalCoreConfig.smoke(device="cpu")),
            language=_Language(), policy=_Policy(), world=resilience_world,
            teacher=NullTeacherAdapter(),
            teacher_store=TeacherSessionStore(root / "none-teacher.json", teacher_id="none"),
            event_log=resilience_log, state_dir=root / "resilience",
        )
        assert resilience.run(interval=0.0, max_steps=1) == 0
        resilience_records = [__import__("json").loads(line) for line in resilience_log.read_text().splitlines()]
        assert any(r["kind"] == "world_degraded" and r["organism_continues"] for r in resilience_records)
        assert any(r["kind"] == "action_attempt" for r in resilience_records)
        print({"ok": True, "same_core_states": len(policy.states),
               "vision_core_forwards": core.telemetry().forward_calls,
               "teacher_inputs": len(language.inputs), "motor_calls": motor.calls,
               "restart_dedup": True, "teacher_has_actuator": False,
               "visual_feedback_is_organ_local": True,
               "capture_failure_recovered": True})
    return 0


if __name__ == "__main__": raise SystemExit(main())
