from __future__ import annotations

from dataclasses import dataclass
import ctypes
import ctypes.util
from pathlib import Path
import json
import select
import struct
import threading
import time
from uuid import uuid4

import torch
import torch.nn.functional as F

from contracts.cognition import ActionAttempt, ActionIntent, Affordance, CompletedAction, Consequence, Observation
from worlds.desktop.inference import NeuralDesktopInference
from worlds.desktop.motor import DesktopMotor, DesktopMotorError, MotorResult
from worlds.desktop.neural import ACTION_KINDS, VisualState
from worlds.desktop.ownership import PhysicalActionArbiter
from worlds.desktop.training.input_events import EV_ABS, EV_KEY, EV_REL, REL_X, REL_Y, EvdevInputCollector, InputCaptureUnavailable, RawInputEvent
from worlds.interface import WorldFrame, validate_intent_for_world


WORLD_ID = "desktop:neural"

# Dedicated local visual-feedback keys emitted by Input Remapper.
# F21/F22 supervise only pixel reading; F23/F24 supervise only cursor targets.
# They never become Policy reward or Language teacher evidence.
KEY_F21 = 191
KEY_F22 = 192
KEY_F23 = 193
KEY_F24 = 194


@dataclass(slots=True)
class _Pending:
    attempt: ActionAttempt
    before_state: VisualState
    action_kind: int
    parameters: tuple[float, float, float, float]
    predicted_latent: torch.Tensor
    before_frame: torch.Tensor
    action_started_ns: int
    action_finished_ns: int


class NeuralSafetyGate:
    """Mechanical action limiter. It can reject but never select a target."""
    def __init__(self, *, emergency_stop: str | Path = "state/desktop.EMERGENCY_STOP", confidence: float = 0.55, max_actions_per_second: float = 3.0, emergency_key_code: int = 88) -> None:
        self.emergency_stop = Path(emergency_stop)
        self.minimum_confidence = float(confidence)
        self.minimum_interval = 1.0 / float(max_actions_per_second)
        self.last_action_at = float("-inf")
        self._key_stop = _EmergencyKeyMonitor(emergency_key_code)

    def check(self, *, confidence: float, now: float) -> tuple[bool, str]:
        if self.emergency_stop.exists() or self._key_stop.triggered: return False, f"emergency stop active: {self.emergency_stop} or key"
        if confidence < self.minimum_confidence: return False, "neural motor confidence below safety threshold"
        if now - self.last_action_at < self.minimum_interval: return False, "action rate limited"
        return True, ""

    def allow(self, *, confidence: float, now: float) -> tuple[bool, str]:
        allowed, reason = self.check(confidence=confidence, now=now)
        if not allowed: return False, reason
        self.last_action_at = now; return True, ""


def _forwarded_pointer_duplicate(event: RawInputEvent) -> bool:
    """Input Remapper pointer output duplicates the physical mouse endpoint."""
    name = event.device_name.casefold()
    if "input-remapper" not in name and "input remapper" not in name:
        return False
    if "forwarded" not in name:
        return False
    return event.event_type in {EV_REL, EV_ABS} or (
        event.event_type == EV_KEY and 272 <= int(event.code) <= 279
    )


class _EmergencyKeyMonitor:
    """Best-effort Linux evdev emergency key listener (F12 by default)."""
    _EVENT = struct.Struct("llHHI")
    def __init__(self, key_code: int) -> None:
        self.triggered = False; self.key_code = int(key_code); self._files: list[object] = []
        input_root = Path("/dev/input")
        for path in input_root.glob("event*") if input_root.exists() else ():
            try: self._files.append(path.open("rb", buffering=0))
            except OSError: pass
        if self._files:
            threading.Thread(target=self._run, name="heiwa-emergency-key", daemon=True).start()

    def _run(self) -> None:
        while not self.triggered:
            readable, _, _ = select.select(self._files, [], [], 0.5)
            for stream in readable:
                data = stream.read(self._EVENT.size)
                if len(data) != self._EVENT.size: continue
                _, _, event_type, code, value = self._EVENT.unpack(data)
                if event_type == 1 and code == self.key_code and value == 1:
                    self.triggered = True; return


class NeuralDesktopWorldAdapter:
    """Pixel-first Desktop World. AT-SPI is absent from this production path."""
    def __init__(
        self, inference: NeuralDesktopInference, *, motor: DesktopMotor | None = None,
        safety: NeuralSafetyGate | None = None, allow_clicks: bool = False,
        allow_right_clicks: bool = False, allow_typing: bool = False,
        allow_mouse: bool = True, allow_keyboard: bool = True,
        capture_input: bool = True, online_learning: bool = True,
        learn_stationary_desktop: bool = True,
        action_arbiter: PhysicalActionArbiter | None = None,
    ) -> None:
        self.inference = inference; self.motor = motor or DesktopMotor(); self.safety = safety or NeuralSafetyGate()
        self.allow_clicks = bool(allow_clicks); self.allow_right_clicks = bool(allow_right_clicks)
        self.allow_typing = bool(allow_typing); self.allow_mouse = bool(allow_mouse)
        self.allow_keyboard = bool(allow_keyboard)
        self._affordances: dict[str, Affordance] = {}; self._pending: _Pending | None = None; self._closed = False
        self._pointer: tuple[int, int] | None = None
        self.capture_input = bool(capture_input); self._input: EvdevInputCollector | None = None
        self.online_learning = bool(online_learning)
        self.learn_stationary_desktop = bool(learn_stationary_desktop)
        self.action_arbiter = action_arbiter or PhysicalActionArbiter()
        self._input_status = "starting" if self.capture_input else "disabled"
        self._last_capture_ns: int | None = None
        # Physical local supervision is carried by dedicated remapped keys:
        # F21/F22 = current cursor-fovea reading correct/incorrect;
        # F23/F24 = current cursor target reject/accept.
        # None of these keys becomes a general Policy or Language signal.
        # Ordinary Backspace and Enter are deliberately not feedback channels.
        # Pointer takeover is direct supervision for the motor target branch.
        # A correction is collected as one short human trajectory instead of
        # turning every REL packet into an independent (and contradictory)
        # target.  Wayland REL units are intentionally never treated as exact
        # screen coordinates: only the accumulated direction is supervised.
        self._pointer_feedback_deadzone = 2.0
        self._pointer_correction_idle_ns = int(0.12 * 1_000_000_000)
        self._pointer_correction_max_ns = int(0.70 * 1_000_000_000)
        self._pointer_correction_session: dict[str, object] | None = None
        # Keep the most recent Heiwa pointer action available long enough for a
        # human to react. Previously correction attribution lived only inside
        # the single `_pending` completion poll, so any hand movement arriving
        # one poll later was silently ignored.
        self._pointer_feedback_window_ns = int(3.0 * 1_000_000_000)
        self._recent_pointer_action: _Pending | None = None
        # Explicit F23 starts a short correction episode.  The previous one-shot
        # reflex could visibly leave a bad target and then stop while the neural
        # proposal immediately returned to the same attractor.  Keep rejected
        # target cells in fast motor memory and verify subsequent proposals.
        self._active_cursor_correction: dict[str, object] | None = None
        self._cursor_correction_ttl_ns = int(12.0 * 1_000_000_000)
        self._cursor_correction_max_hops = 18
        # Interactive F23 search is deliberately slow enough for a person to
        # steer it between hops.  Human pointer motion after F23 refreshes the
        # spatial hint; Heiwa then chooses another neural target near that hint.
        self._cursor_correction_min_hop_ns = int(0.22 * 1_000_000_000)
        # A recent human pointer gesture is a spatial hint, not a target label.
        # F23 uses it to search the neural target distribution in the region
        # the user pointed toward instead of jumping to a globally unrelated mode.
        self._cursor_search_hint: dict[str, object] | None = None
        self._cursor_search_hint_ttl_ns = int(3.5 * 1_000_000_000)
        # Guided search should converge rather than jump between broad regions.
        # Human motion nudges a persistent normalized search center; repeated
        # corrections shrink a local radius around that center.
        self._cursor_guidance_step = 0.075
        self._cursor_guidance_radius_start = 0.20
        self._cursor_guidance_radius_min = 0.045
        # Strong teacher mode: F23 rejects a *neighbourhood*, not one exact
        # target cell. Rejections survive steering turns and are also retained
        # briefly for similar visual target-distribution contexts.
        self._cursor_hard_reject_radius_cells = 1
        self._cursor_pattern_memory_path = Path("state/embodied_runtime/cursor_pattern_memory.json")
        self._cursor_pattern_memory_limit = 96
        self._cursor_pattern_memory_ttl_ns = int(10.0 * 60.0 * 1_000_000_000)
        self._cursor_pattern_memory = self._load_cursor_pattern_memory()
        # A held F23 is a continuous visual-pattern eraser.  Key repeat is not
        # required: evdev press/release state is retained across visual polls.
        self._f23_held = False
        self._f23_blocked_until_release = False
        self._f23_last_erase_ns = 0
        self._f23_erase_interval_ns = int(0.25 * 1_000_000_000)
        self._f23_devices: tuple[str, ...] = ()
        self._last_physical_pointer_delta: dict[str, object] | None = None
        self._cursor_overlay_path = Path("state/embodied_runtime/cursor_guidance_overlay.json")
        self._shell_pointer_path = Path("state/embodied_runtime/shell_pointer.json")
        self._visual_focus_xy: tuple[float, float] | None = None

    @property
    def world_id(self) -> str: return WORLD_ID

    def poll(self, now: float) -> WorldFrame:
        proposal = self.inference.capture_and_propose()
        capture_ns = time.monotonic_ns()
        state = self.inference.state
        assert state is not None
        if self._input is None and self.capture_input and self._input_status == "starting":
            try:
                self._input = EvdevInputCollector(
                    screen_width=proposal.screen_width, screen_height=proposal.screen_height
                ); self._input.start(); self._input_status = "linux-evdev"
            except InputCaptureUnavailable as exc:
                self._input_status = f"unavailable: {exc}"
        physical_events: tuple[RawInputEvent, ...] = ()
        previous_capture_ns = self._last_capture_ns
        if self._input is not None and self._last_capture_ns is not None:
            physical_events = self._input.drain(self._last_capture_ns, capture_ns)
        # Input Remapper may grab the physical mouse, leaving only its
        # forwarded virtual endpoint visible to this process.  The previous
        # code discarded every forwarded pointer packet unconditionally; on
        # such setups that discarded *all* human pointer corrections.
        # Prefer raw physical pointer packets when they exist, but retain the
        # forwarded stream when it is the only observable human pointer source.
        raw_pointer_events = tuple(
            event for event in physical_events
            if (event.event_type in {EV_REL, EV_ABS}
                or (event.event_type == EV_KEY and 272 <= int(event.code) <= 279))
            and not _forwarded_pointer_duplicate(event)
        )
        if raw_pointer_events:
            attributed_events = tuple(
                event for event in physical_events if not _forwarded_pointer_duplicate(event)
            )
            forwarded_pointer_fallback = False
        else:
            attributed_events = physical_events
            forwarded_pointer_fallback = any(_forwarded_pointer_duplicate(event) for event in physical_events)
        self._last_capture_ns = capture_ns
        self._integrate_physical_pointer(attributed_events, proposal, capture_ns)
        self._update_cursor_search_hint_from_events(attributed_events, proposal, capture_ns)
        visual_focus = self._read_shell_pointer(capture_ns) or self._read_xwayland_pointer(proposal)
        if visual_focus is None and self._pointer is not None:
            visual_focus = (
                float(self._pointer[0]) / max(int(proposal.screen_width) - 1, 1),
                float(self._pointer[1]) / max(int(proposal.screen_height) - 1, 1),
            )
        self._visual_focus_xy = visual_focus
        refocus = getattr(self.inference, "refocus_reading", None)
        if visual_focus is not None and callable(refocus):
            proposal = refocus(proposal, visual_focus)
        completed: tuple[CompletedAction, ...] = (); consequences: tuple[Consequence, ...] = ()
        pointer_correction_payload: dict[str, object] | None = None

        mouse_button_down = any(
            event.event_type == EV_KEY and event.value == 1 and 272 <= int(event.code) <= 279
            for event in attributed_events
        )

        # Continue an already-started human correction across poll boundaries.
        # This is important: a real hand movement arrives as many REL packets,
        # often split over several frames.  Training each fragment separately
        # made the target oscillate and repeatedly fall back into the same mode.
        if self._pointer_correction_session is not None:
            session = self._pointer_correction_session
            rel = tuple(
                event for event in attributed_events
                if event.event_type == EV_REL and event.code in {REL_X, REL_Y}
                and event.timestamp_ns >= int(session["started_ns"])
            )
            if rel:
                dx = sum(event.value for event in rel if event.code == REL_X)
                dy = sum(event.value for event in rel if event.code == REL_Y)
                session["relative_dx"] = int(session["relative_dx"]) + int(dx)
                session["relative_dy"] = int(session["relative_dy"]) + int(dy)
                session["last_event_ns"] = max(event.timestamp_ns for event in rel)
                trajectory = session["trajectory"]
                if isinstance(trajectory, list) and len(trajectory) < 48:
                    trajectory.append((int(session["last_event_ns"]), int(session["relative_dx"]), int(session["relative_dy"])))
                devices = list(session["devices"])
                for event in rel:
                    if event.device_name not in devices:
                        devices.append(event.device_name)
                session["devices"] = tuple(devices)

            idle = capture_ns - int(session["last_event_ns"])
            age = capture_ns - int(session["started_ns"])
            finish = mouse_button_down or idle >= self._pointer_correction_idle_ns or age >= self._pointer_correction_max_ns
            if finish:
                dx = int(session["relative_dx"]); dy = int(session["relative_dy"])
                magnitude = (float(dx) ** 2 + float(dy) ** 2) ** 0.5
                if magnitude >= self._pointer_feedback_deadzone:
                    direction_x = float(dx) / magnitude; direction_y = float(dy) / magnitude
                    previous_xy = tuple(session["previous_target_xy"])
                    # Fixed nearby anchor encodes direction only. REL magnitude is
                    # deliberately absent because evdev counts are not pixels.
                    anchor_step = 0.10
                    corrected_x = min(max(float(previous_xy[0]) + direction_x * anchor_step, 0.0), 1.0)
                    corrected_y = min(max(float(previous_xy[1]) + direction_y * anchor_step, 0.0), 1.0)
                    motor_metrics = self.inference.learn_pointer_correction(
                        before_state=session["before_state"],
                        target_xy=(corrected_x, corrected_y),
                        repeats=2,
                    ) if self.online_learning else {"pointer_feedback": "online_learning_disabled"}
                    pointer_correction_payload = {
                        "signal": "follow_human_pointer_trajectory",
                        "negative_evidence": False,
                        "truth_value": "unknown",
                        "source": "physical_pointer_motion",
                        "target_transition_id": session["target_transition_id"],
                        "target_action": session["target_action"],
                        "relative_dx": dx, "relative_dy": dy,
                        "relative_magnitude": magnitude,
                        "direction": (direction_x, direction_y),
                        "previous_target_xy": previous_xy,
                        "corrected_target_xy": (corrected_x, corrected_y),
                        "target_is_directional_anchor": True,
                        "trajectory": tuple(session["trajectory"]),
                        "trajectory_duration_ms": (int(session["last_event_ns"]) - int(session["started_ns"])) / 1_000_000.0,
                        "reaction_latency_ms": float(session.get("reaction_latency_ms", 0.0)),
                        "attribution_window_ms": self._pointer_feedback_window_ns / 1_000_000.0,
                        "finalized_by": "click" if mouse_button_down else ("idle" if idle >= self._pointer_correction_idle_ns else "timeout"),
                        "devices": session["devices"],
                        "motor_learning": motor_metrics,
                        "deadzone_raw": self._pointer_feedback_deadzone,
                    }
                self._pointer_correction_session = None

        # Attribute human pointer takeover to the most recent Heiwa pointer
        # action for a short reaction window, not merely to the immediately
        # pending completion poll. This makes ordinary human reaction latency
        # observable even when one or more noop/visual polls happen first.
        correction_source = self._recent_pointer_action
        if correction_source is not None:
            source_age_ns = capture_ns - correction_source.action_finished_ns
            if source_age_ns < 0 or source_age_ns > self._pointer_feedback_window_ns:
                self._recent_pointer_action = None
                correction_source = None
        if self._pointer_correction_session is None and correction_source is not None:
            correction_events = tuple(
                event for event in attributed_events
                if event.timestamp_ns > correction_source.action_finished_ns
                and event.event_type == EV_REL and event.code in {REL_X, REL_Y}
            )
            correction_dx = sum(event.value for event in correction_events if event.code == REL_X)
            correction_dy = sum(event.value for event in correction_events if event.code == REL_Y)
            correction_magnitude = (float(correction_dx) ** 2 + float(correction_dy) ** 2) ** 0.5
            if correction_magnitude >= self._pointer_feedback_deadzone:
                first_ns = min(event.timestamp_ns for event in correction_events)
                last_ns = max(event.timestamp_ns for event in correction_events)
                self._pointer_correction_session = {
                    "started_ns": first_ns,
                    "last_event_ns": last_ns,
                    "target_transition_id": correction_source.attempt.transition_id,
                    "target_action": correction_source.attempt.action,
                    "previous_target_xy": (float(correction_source.parameters[0]), float(correction_source.parameters[1])),
                    "relative_dx": int(correction_dx),
                    "relative_dy": int(correction_dy),
                    "trajectory": [(last_ns, int(correction_dx), int(correction_dy))],
                    "devices": tuple(dict.fromkeys(event.device_name for event in correction_events)),
                    "before_state": correction_source.before_state,
                    "reaction_latency_ms": (first_ns - correction_source.action_finished_ns) / 1_000_000.0,
                }
                # One physical takeover consumes this attribution target. A new
                # Heiwa pointer action will arm a fresh target.
                self._recent_pointer_action = None

        if self._pending is not None:
            pending = self._pending; self._pending = None
            # Patch tokens are a direct function of pixels. Recurrent latent
            # state may drift between identical frames and is therefore not
            # sufficient evidence that a physical action changed the screen.
            patch_change = 1.0 - F.cosine_similarity(
                pending.before_state.patch_tokens, state.patch_tokens, dim=-1
            )
            change = float(patch_change.mean())
            # Any physical event after the causal before-frame contaminates
            # attribution, including a user action just before actuation.
            interfering = tuple(
                event for event in attributed_events
                if event.event_type in {EV_KEY, EV_REL, EV_ABS}
            )
            causal_before_ns = previous_capture_ns or pending.action_started_ns
            foreign_windows = self.action_arbiter.foreign_windows("heiwa", causal_before_ns, capture_ns)
            if self._input is None:
                ownership = "unknown_input_attribution"
            elif interfering:
                ownership = "mixed_user_interference"
            elif foreign_windows:
                ownership = "mixed_" + "+".join(sorted({window.source for window in foreign_windows})) + "_interference"
            else:
                ownership = "heiwa"
            succeeded = change > 1e-4 if ownership == "heiwa" else False
            interference_summary = tuple({"device": event.device_name, "type": event.event_type,
                                           "code": event.code, "value": event.value,
                                           "timestamp_ns": event.timestamp_ns}
                                          for event in interfering[:16])
            ignored_forwarded_duplicates = len(physical_events) - len(attributed_events)
            assert self.inference.last_frame is not None
            errors = self.inference.learn_transition(
                before_frame=pending.before_frame, after_frame=self.inference.last_frame,
                before_state=pending.before_state, after_state=state,
                action_kind=pending.action_kind, parameters=pending.parameters,
                ownership=ownership, completion=succeeded if ownership == "heiwa" else None,
            ) if self.online_learning else {"online_learning": "disabled"}
            if "observed_target_x" in errors and "observed_target_y" in errors:
                observed_x = float(errors["observed_target_x"]); observed_y = float(errors["observed_target_y"])
                errors["target_error_px"] = (
                    ((observed_x - pending.parameters[0]) * proposal.screen_width) ** 2
                    + ((observed_y - pending.parameters[1]) * proposal.screen_height) ** 2
                ) ** 0.5
            predicted_error = float(F.smooth_l1_loss(pending.predicted_latent, state.latent).cpu())
            action = CompletedAction(
                    id=f"desktop-completed-{uuid4().hex}", attempt_id=pending.attempt.id,
                    intent_id=pending.attempt.intent_id, transition_id=pending.attempt.transition_id,
                    world=self.world_id, action=pending.attempt.action, completed_at=now,
                    status=("observed_effect" if succeeded else "no_observed_effect") if ownership == "heiwa" else "ownership_uncertain",
                    succeeded=succeeded,
                    payload={"verified_by": "neural_visual_state_change", "latent_change": change,
                             "predicted_consequence_error": predicted_error,
                             "ownership": ownership,
                             "external_event_count": len(interfering),
                             "ignored_forwarded_pointer_duplicates": ignored_forwarded_duplicates,
                             "external_events": interference_summary, **errors},
                )
            completed = (action,); consequences = (Consequence(
                id=f"desktop-consequence-{uuid4().hex}", transition_id=action.transition_id,
                world=self.world_id, kind="desktop.visual_effect", observed_at=now,
                completed_action_id=action.id, payload={"latent_change": change,
                                                       "predicted_consequence_error": predicted_error,
                                                       "ownership": ownership,
                                                       "external_event_count": len(interfering),
                                                       "ignored_forwarded_pointer_duplicates": ignored_forwarded_duplicates,
                                                       "external_events": interference_summary,
                                                       "foreign_action_sources": tuple(window.source for window in foreign_windows),
                                                       "local_prediction_losses": errors},
            ),)
        elif self.inference.previous_frame is not None and self.inference.previous_state is not None:
            patch_change = 1.0 - F.cosine_similarity(
                self.inference.previous_state.patch_tokens, state.patch_tokens, dim=-1
            )
            passive_change = float(patch_change.mean())
            # Learn the temporal law of the desktop from lived frames, including
            # stationary ones. This teaches prediction of both change and stillness;
            # motion itself is not treated as a positive reward.
            if self.online_learning and (self.learn_stationary_desktop or physical_events or passive_change > 1e-4):
                assert self.inference.last_frame is not None
                self.inference.learn_transition(
                    before_frame=self.inference.previous_frame,
                    after_frame=self.inference.last_frame,
                    before_state=self.inference.previous_state,
                    after_state=state, action_kind=ACTION_KINDS.index("noop"),
                    parameters=(0.0, 0.0, 0.0, 0.0),
                    ownership="user" if physical_events else "environment",
                    completion=None,
                )
        # Dedicated commands stay local to the corresponding Desktop organ.
        key_downs = tuple(
            event for event in physical_events
            if event.event_type == EV_KEY and event.value == 1
        )
        f23_erase_tick, f23_press_count = self._update_f23_eraser(physical_events, capture_ns)
        cursor_accept_commands = tuple(event for event in key_downs if event.code == KEY_F24)
        reading_correct_commands = tuple(event for event in key_downs if event.code == KEY_F21)
        reading_incorrect_commands = tuple(event for event in key_downs if event.code == KEY_F22)
        # Contradictory same-frame signals can be produced by a remapper when a
        # held key changes layer on release. A hard rejection always wins; F24
        # must be a separate explicit confirmation of an active session.
        valid_cursor_accept_commands = () if f23_erase_tick else cursor_accept_commands
        reading_feedback = None
        if reading_correct_commands or reading_incorrect_commands:
            accepted_reading = not bool(reading_incorrect_commands)
            learn_reading = getattr(self.inference, "learn_reading_feedback", None)
            metrics = (
                learn_reading(
                    bbox=proposal.read_bbox,
                    read_byte_ids=proposal.read_byte_ids,
                    accepted=accepted_reading,
                )
                if self.online_learning and callable(learn_reading)
                else {"reader_feedback": "online_learning_disabled"}
            )
            reading_feedback = {
                "accepted": accepted_reading,
                "key": "F21" if accepted_reading else "F22",
                "decoded_byte_ids": proposal.read_byte_ids,
                "bbox": proposal.read_bbox,
                "metrics": metrics,
                "language_teacher": False,
                "policy_feedback": False,
            }

        # During an active F23 episode, physical mouse motion is live steering.
        # It does not assert an exact screen coordinate; it only moves the local
        # search region in the direction indicated by the person.
        live_cursor_guidance = self._refresh_active_cursor_guidance(capture_ns)

        # While an F23 correction episode is active, inspect the *new* neural
        # proposal before giving ordinary autonomy another turn.  If it falls
        # back into a rejected target cell, immediately try another neural
        # alternative instead of stopping after the first cursor displacement.
        correction_followup = None
        if not f23_erase_tick:
            correction_followup = self._continue_cursor_feedback_replan(proposal, capture_ns)

        observation = Observation(
            id=f"desktop-observation-{uuid4().hex}", world=self.world_id,
            kind="desktop.neural_visual_state", observed_at=now,
            payload={"latent": tuple(float(v) for v in state.latent[0].cpu()),
                     "frame_digest": proposal.frame_digest,
                     "regions": tuple(tuple(float(v) for v in box) for box in state.region_positions[0].cpu()),
                     "salience": tuple(float(v) for v in state.salience[0].cpu()),
                     "input_attribution": self._input_status,
                     "external_input_events": len(physical_events),
                     "external_action_is_feedback": bool(f23_erase_tick or valid_cursor_accept_commands or reading_feedback or pointer_correction_payload),
                     "cursor_rejection": bool(f23_erase_tick),
                     "cursor_rejection_count": int(f23_press_count or f23_erase_tick),
                     "cursor_rejection_held": self._f23_held,
                     "cursor_acceptance": bool(valid_cursor_accept_commands),
                     "cursor_acceptance_count": len(valid_cursor_accept_commands),
                     "visual_reading_feedback": reading_feedback,
                     "user_pointer_correction": bool(pointer_correction_payload),
                     "physical_pointer_events": sum(1 for event in physical_events if event.event_type in {EV_REL, EV_ABS}),
                     "attributed_pointer_events": sum(1 for event in attributed_events if event.event_type in {EV_REL, EV_ABS}),
                     "forwarded_pointer_fallback": forwarded_pointer_fallback,
                     "cursor_correction_active": self._active_cursor_correction is not None,
                     "cursor_correction_followup": correction_followup,
                     "cursor_search_hint": self._recent_cursor_search_hint(capture_ns),
                     "cursor_live_guidance": live_cursor_guidance,
                     "visual_read_byte_ids": proposal.read_byte_ids,
                     "visual_read_confidence": 0.0 if reading_incorrect_commands else proposal.read_confidence,
                     "visual_read_bbox": proposal.read_bbox,
                     "visual_read_focus_source": proposal.read_focus_source,
                     "visual_cursor_focus_xy": self._visual_focus_xy,
                     "neural_proposed_action": proposal.action,
                     "neural_action_probabilities": proposal.action_distribution},
        )
        observations = [observation]
        if reading_feedback is not None:
            observations.append(Observation(
                id=f"desktop-reading-feedback-{uuid4().hex}", world=self.world_id,
                kind="desktop.visual_reading_feedback", observed_at=now,
                payload=reading_feedback,
            ))
        if pointer_correction_payload is not None:
            observations.append(Observation(
                id=f"desktop-user-pointer-correction-{uuid4().hex}", world=self.world_id,
                kind="desktop.user_pointer_correction", observed_at=now,
                payload=pointer_correction_payload,
            ))
        if f23_erase_tick:
            immediate_pointer_feedback = self._immediate_pointer_feedback_replan(proposal, arm_episode=True)
            observations.append(Observation(
                id=f"desktop-cursor-rejection-{uuid4().hex}", world=self.world_id,
                kind="desktop.cursor_rejection", observed_at=now,
                payload={
                    "signal": "reject_cursor_target",
                    "source": "input_remapper_f23",
                    "immediate_pointer_feedback": immediate_pointer_feedback,
                    "held_eraser": self._f23_held,
                    "count": int(f23_press_count or 1),
                    "devices": self._f23_devices,
                },
            ))
        if valid_cursor_accept_commands:
            accepted_cursor_pattern = self._accept_cursor_feedback(proposal)
            if self._f23_held:
                self._f23_blocked_until_release = True
                self._f23_held = False
            observations.append(Observation(
                id=f"desktop-cursor-acceptance-{uuid4().hex}", world=self.world_id,
                kind="desktop.cursor_acceptance", observed_at=now,
                payload={
                    "signal": "accept_cursor_target",
                    "source": "input_remapper_f24",
                    "dedicated_command_key": "F24",
                    "accepted_cursor_pattern": accepted_cursor_pattern,
                    "count": len(valid_cursor_accept_commands),
                    "devices": tuple(dict.fromkeys(event.device_name for event in valid_cursor_accept_commands)),
                    "policy_feedback": False,
                    "language_teacher": False,
                },
            ))
        self._publish_cursor_overlay(proposal)
        effective_x, effective_y, effective_cell = self._effective_teacher_target(proposal)
        # NeuralDesktopBody is a specialist: its motor head proposes a
        # distribution, while shared Heiwa Policy remains the final cognitive
        # selector.  Offer every mechanically executable primitive as a dynamic
        # affordance and attach the specialist probability as a generic prior.
        self._affordances = {}
        if len(proposal.action_distribution) == len(ACTION_KINDS):
            probabilities = {
                action: max(float(proposal.action_distribution[index]), 1e-8)
                for index, action in enumerate(ACTION_KINDS)
            }
        else:
            # Compatibility with older/fake inference providers that exposed
            # only the selected action.  Keep alternatives alive with a tiny
            # prior instead of collapsing the World back to one affordance.
            probabilities = {action: 1e-8 for action in ACTION_KINDS}
            if proposal.action in probabilities:
                probabilities[proposal.action] = 1.0

        candidate_actions: list[str] = ["noop"]
        motor_ready, _ = self.safety.check(confidence=proposal.confidence, now=now)
        if motor_ready and self.allow_mouse:
            candidate_actions.extend(("move_pointer", "scroll"))
            if self.allow_clicks:
                candidate_actions.extend(("left_click", "double_click"))
            if self.allow_right_clicks:
                candidate_actions.append("right_click")
        if motor_ready and self.allow_keyboard and self.allow_typing:
            candidate_actions.extend(("key_press", "type_text"))
        # key_chord is intentionally not offered yet: the current neural motor
        # head does not parameterize a complete chord, so exposing it would
        # create an affordance that the executor cannot realize causally.

        ranked = sorted(candidate_actions, key=lambda action: probabilities[action], reverse=True)
        for rank, action in enumerate(ranked):
            affordance = Affordance(
                id=f"desktop-affordance-{uuid4().hex}", world=self.world_id,
                action=f"desktop.{action}", offered_at=now,
                payload={
                    "neural_target_x": effective_x, "neural_target_y": effective_y,
                    "neural_raw_target_x": proposal.x, "neural_raw_target_y": proposal.y,
                    "teacher_routed_target_cell": effective_cell,
                    "neural_key_code": proposal.key_code,
                    "frame_digest": proposal.frame_digest,
                    "neural_confidence": proposal.confidence,
                    "neural_proposed_action": proposal.action,
                    "neural_action_probability": probabilities[action],
                    "policy_prior_probability": probabilities[action],
                    "neural_action_rank": rank,
                    "neural_action_probabilities": proposal.action_distribution,
                    "neural_target_bbox": proposal.target_bbox,
                    "neural_trajectory": proposal.trajectory,
                    "visual_read_byte_ids": proposal.read_byte_ids,
                    "visual_read_confidence": proposal.read_confidence,
                    "screen_width": proposal.screen_width, "screen_height": proposal.screen_height,
                },
                constraints={
                    "pixel_only": True, "physical_input": True,
                    "neural_motor_candidate": True, "shared_policy_selects": True,
                },
            )
            self._affordances[affordance.id] = affordance
        return WorldFrame(self.world_id, now, tuple(observations), tuple(self._affordances.values()), completed, consequences)

    def _load_cursor_pattern_memory(self) -> list[dict[str, object]]:
        try:
            raw = json.loads(self._cursor_pattern_memory_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return []
        if not isinstance(raw, list):
            return []
        now_ns = time.monotonic_ns()
        kept = []
        for item in raw[-self._cursor_pattern_memory_limit:]:
            if not isinstance(item, dict):
                continue
            # Files survive restarts while monotonic clocks do not. Treat loaded
            # items as fresh for this process and let the bounded list age out.
            item = dict(item)
            item["loaded_ns"] = now_ns
            kept.append(item)
        return kept

    def _update_f23_eraser(self, events, capture_ns: int) -> tuple[bool, int]:
        """Track physical F23 press/release and emit bounded continuous erase ticks."""
        presses = 0
        devices = list(self._f23_devices)
        for event in events:
            if event.event_type != EV_KEY or int(event.code) != KEY_F23:
                continue
            if event.device_name not in devices:
                devices.append(event.device_name)
            if int(event.value) == 0:
                self._f23_held = False
                self._f23_blocked_until_release = False
            elif int(event.value) in {1, 2} and not self._f23_blocked_until_release:
                self._f23_held = True
                if int(event.value) == 1:
                    presses += 1
        self._f23_devices = tuple(devices[-8:])
        due = self._f23_held and (
            presses > 0
            or not self._f23_last_erase_ns
            or int(capture_ns) - self._f23_last_erase_ns >= self._f23_erase_interval_ns
        )
        if due:
            self._f23_last_erase_ns = int(capture_ns)
        return bool(due), int(presses)

    def _integrate_physical_pointer(self, events, proposal, capture_ns: int) -> None:
        """Keep the internal cursor estimate aligned with real human REL motion."""
        rel = tuple(
            event for event in events
            if event.event_type == EV_REL and event.code in {REL_X, REL_Y}
        )
        if not rel:
            return
        dx = sum(int(event.value) for event in rel if event.code == REL_X)
        dy = sum(int(event.value) for event in rel if event.code == REL_Y)
        if not dx and not dy:
            return
        width = max(int(proposal.screen_width), 1)
        height = max(int(proposal.screen_height), 1)
        if self._pointer is None:
            self._pointer = (width // 2, height // 2)
        px = min(max(int(self._pointer[0]) + dx, 0), width - 1)
        py = min(max(int(self._pointer[1]) + dy, 0), height - 1)
        self._pointer = (px, py)
        self._last_physical_pointer_delta = {
            "at_ns": int(capture_ns),
            "dx": int(dx), "dy": int(dy),
            "pointer_xy": (int(px), int(py)),
            "devices": tuple(dict.fromkeys(event.device_name for event in rel)),
        }

    @staticmethod
    def _format_reading_for_display(token_ids) -> str:
        """Show valid printable UTF-8 literally and escape only undisplayable bytes."""
        raw = bytearray()
        for value in token_ids:
            token = int(value)
            if token == 2:
                break
            if 4 <= token <= 259:
                raw.append(token - 4)
        output: list[str] = []
        index = 0
        while index < len(raw):
            first = raw[index]
            if first < 0x80:
                width = 1
            elif 0xC2 <= first <= 0xDF:
                width = 2
            elif 0xE0 <= first <= 0xEF:
                width = 3
            elif 0xF0 <= first <= 0xF4:
                width = 4
            else:
                output.append(f"\\x{first:02X}")
                index += 1
                continue
            chunk = bytes(raw[index:index + width])
            try:
                character = chunk.decode("utf-8")
            except UnicodeDecodeError:
                output.append(f"\\x{first:02X}")
                index += 1
                continue
            if character.isprintable():
                output.append(character)
            elif character == "\n":
                output.append("\\n")
            elif character == "\t":
                output.append("\\t")
            elif ord(character) <= 0xFFFF:
                output.append(f"\\u{ord(character):04X}")
            else:
                output.append(f"\\U{ord(character):08X}")
            index += width
        return "".join(output)

    def _publish_cursor_overlay(self, proposal) -> None:
        """Atomically publish visual-only teacher diagnostics for the overlay."""
        try:
            probabilities, gh, gw = self._target_grid(proposal)
        except AttributeError:
            # Minimal/fake inference providers need not expose visual geometry.
            return
        if not probabilities or len(probabilities) != gh * gw:
            return
        episode = self._active_cursor_correction or {}
        remembered_rejected = self._remembered_rejected_cells(probabilities, gw, gh)
        active_rejected = set(int(v) for v in episode.get("rejected_cells", ()))
        accepted = self._remembered_accepted_cells(probabilities, gw, gh)
        ranked = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:8]
        effective_x, effective_y, effective_cell = self._effective_teacher_target(proposal)
        reading_text = self._format_reading_for_display(proposal.read_byte_ids)
        if accepted:
            erow, ecol = divmod(effective_cell, gw)
            display_bbox = [ecol/gw, erow/gh, (ecol+1)/gw, (erow+1)/gh]
        else:
            display_bbox = [float(v) for v in proposal.target_bbox]
        payload = {
            "format": 1,
            "updated_monotonic_ns": time.monotonic_ns(),
            "screen": [int(proposal.screen_width), int(proposal.screen_height)],
            "grid": [int(gw), int(gh)],
            "f23_held": bool(self._f23_held),
            "session_active": self._active_cursor_correction is not None,
            "proposal_cell": int(effective_cell),
            "proposal_xy": [float(effective_x) / max(int(proposal.screen_width)-1, 1),
                            float(effective_y) / max(int(proposal.screen_height)-1, 1)],
            "raw_proposal_xy": [float(proposal.x) / max(int(proposal.screen_width)-1, 1),
                                float(proposal.y) / max(int(proposal.screen_height)-1, 1)],
            "teacher_routed": bool(accepted),
            "proposal_bbox": display_bbox,
            "top_candidates": [{"cell": int(i), "probability": float(probabilities[i])} for i in ranked],
            "rejected_cells": sorted(remembered_rejected | active_rejected),
            "hard_rejected_cells": sorted(int(v) for v in episode.get("hard_rejected_cells", ())),
            "attempted_cells": sorted(int(v) for v in episode.get("attempted_cells", ())),
            "accepted_cells": sorted(accepted),
            "guided_xy": episode.get("guided_xy"),
            "search_radius": float(episode.get("search_radius", 0.0)),
            "guidance_trace": list(episode.get("guidance_trace", ()))[-32:],
            "last_physical_pointer_delta": self._last_physical_pointer_delta,
            "reading_bbox": [float(v) for v in proposal.read_bbox],
            "reading_focus_source": proposal.read_focus_source,
            "reading_text": reading_text,
            "reading_confidence": float(proposal.read_confidence),
            "visual_focus_xy": self._visual_focus_xy,
        }
        try:
            self._cursor_overlay_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._cursor_overlay_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self._cursor_overlay_path)
        except OSError:
            pass

    def _save_cursor_pattern_memory(self) -> None:
        try:
            self._cursor_pattern_memory_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = []
            for item in self._cursor_pattern_memory[-self._cursor_pattern_memory_limit:]:
                clean = {k: v for k, v in item.items() if k != "loaded_ns"}
                serializable.append(clean)
            self._cursor_pattern_memory_path.write_text(
                json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def _target_context_signature(self, probabilities: tuple[float, ...], *, top_k: int = 12) -> tuple[int, ...]:
        if not probabilities:
            return ()
        ranked = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)
        return tuple(int(v) for v in ranked[:min(top_k, len(ranked))])

    @staticmethod
    def _context_overlap(a, b) -> float:
        sa, sb = set(int(v) for v in a), set(int(v) for v in b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(len(sa), len(sb), 1)

    def _cell_neighbourhood(self, cell: int, gw: int, gh: int, radius: int | None = None) -> set[int]:
        radius = self._cursor_hard_reject_radius_cells if radius is None else max(int(radius), 0)
        row, col = divmod(int(cell), gw)
        out: set[int] = set()
        for rr in range(max(0, row-radius), min(gh, row+radius+1)):
            for cc in range(max(0, col-radius), min(gw, col+radius+1)):
                out.add(rr * gw + cc)
        return out

    def _remember_rejected_pattern(self, probabilities, cell: int, gw: int, gh: int) -> None:
        signature = self._target_context_signature(tuple(probabilities))
        cells = tuple(sorted(self._cell_neighbourhood(cell, gw, gh)))
        self._cursor_pattern_memory.append({
            "kind": "rejected",
            "context": signature,
            "cells": cells,
            "created_unix": time.time(),
            "loaded_ns": time.monotonic_ns(),
        })
        self._cursor_pattern_memory = self._cursor_pattern_memory[-self._cursor_pattern_memory_limit:]
        self._save_cursor_pattern_memory()

    def _remember_accepted_pattern(
        self, proposal, *, teacher_xy: tuple[float, float], source: str,
    ) -> dict[str, object] | None:
        probabilities, gh, gw = self._target_grid(proposal)
        if not probabilities or len(probabilities) != gh * gw:
            return None
        width = max(int(proposal.screen_width), 1)
        height = max(int(proposal.screen_height), 1)
        nx = min(max(float(teacher_xy[0]), 0.0), 1.0)
        ny = min(max(float(teacher_xy[1]), 0.0), 1.0)
        px, py = nx * max(width - 1, 1), ny * max(height - 1, 1)
        col = min(gw - 1, max(0, int(nx * gw)))
        row = min(gh - 1, max(0, int(ny * gh)))
        cell = row * gw + col
        signature = self._target_context_signature(tuple(probabilities))
        trace = []
        if self._active_cursor_correction is not None:
            trace = [tuple(float(v) for v in xy) for xy in self._active_cursor_correction.get("guidance_trace", ())][-32:]
        item = {
            "kind": "accepted",
            "confirmed_teacher_signal": True,
            "confirmed_active_session": self._active_cursor_correction is not None,
            "source": str(source),
            "context": signature,
            "cell": int(cell),
            "xy": (float(px) / max(width - 1, 1), float(py) / max(height - 1, 1)),
            "guidance_trace": trace,
            "created_unix": time.time(),
            "loaded_ns": time.monotonic_ns(),
        }
        self._cursor_pattern_memory.append(item)
        self._cursor_pattern_memory = self._cursor_pattern_memory[-self._cursor_pattern_memory_limit:]
        self._save_cursor_pattern_memory()
        return {"cell": int(cell), "xy": item["xy"], "trace_points": len(trace)}

    def _accept_cursor_feedback(self, proposal) -> dict[str, object] | None:
        """Accept either an active search or the visual cell explicitly hovered at F24."""
        capture_ns = time.monotonic_ns()
        teacher_xy = self._read_shell_pointer(capture_ns) or self._read_xwayland_pointer(proposal)
        active = self._active_cursor_correction is not None
        if teacher_xy is None:
            if not active or self._pointer is None:
                return None
            teacher_xy = (
                float(self._pointer[0]) / max(int(proposal.screen_width)-1, 1),
                float(self._pointer[1]) / max(int(proposal.screen_height)-1, 1),
            )
        accepted = self._remember_accepted_pattern(
            proposal,
            teacher_xy=teacher_xy,
            source="guided_f24" if active else "direct_hover_f24",
        )
        self._active_cursor_correction = None
        return accepted

    def _remembered_rejected_cells(self, probabilities, gw: int, gh: int) -> set[int]:
        signature = self._target_context_signature(tuple(probabilities))
        if not signature:
            return set()
        now_ns = time.monotonic_ns()
        rejected: set[int] = set()
        kept = []
        for item in self._cursor_pattern_memory:
            loaded_ns = int(item.get("loaded_ns", now_ns))
            if now_ns - loaded_ns > self._cursor_pattern_memory_ttl_ns:
                continue
            kept.append(item)
            if item.get("kind") != "rejected":
                continue
            if self._context_overlap(signature, item.get("context", ())) < 0.58:
                continue
            rejected.update(int(v) for v in item.get("cells", ()) if 0 <= int(v) < gw * gh)
        self._cursor_pattern_memory = kept[-self._cursor_pattern_memory_limit:]
        return rejected

    def _remembered_accepted_cells(self, probabilities, gw: int, gh: int) -> set[int]:
        """Return locally supervised targets for a similar neural visual context."""
        return {
            int(item["cell"])
            for item, _overlap in self._matching_accepted_patterns(probabilities, gw, gh)
        }

    def _matching_accepted_patterns(self, probabilities, gw: int, gh: int):
        """Return confirmed targets and their visual-context similarity."""
        signature = self._target_context_signature(tuple(probabilities))
        if not signature:
            return []
        now_ns = time.monotonic_ns()
        accepted = []
        for item in self._cursor_pattern_memory:
            if item.get("kind") != "accepted":
                continue
            if item.get("confirmed_teacher_signal") is not True:
                continue
            if now_ns - int(item.get("loaded_ns", now_ns)) > self._cursor_pattern_memory_ttl_ns:
                continue
            overlap = self._context_overlap(signature, item.get("context", ()))
            if overlap < 0.58:
                continue
            cell = int(item.get("cell", -1))
            if 0 <= cell < gw * gh:
                accepted.append((item, overlap))
        return accepted

    def _effective_teacher_target(self, proposal) -> tuple[int, int, int]:
        """Route motor/overlay target to accepted neural cell in matching context."""
        try:
            probabilities, gh, gw = self._target_grid(proposal)
        except AttributeError:
            return int(proposal.x), int(proposal.y), 0
        raw_cell = self._proposal_target_cell(proposal, gw, gh)
        if not probabilities or len(probabilities) != gh * gw:
            return int(proposal.x), int(proposal.y), int(raw_cell)
        accepted = self._matching_accepted_patterns(probabilities, gw, gh)
        if not accepted:
            return int(proposal.x), int(proposal.y), int(raw_cell)
        # F24 stores an exact compositor-space point. Keep the neural cell as
        # the visual association, but do not quantize the teacher's requested
        # motor destination back to the cell centre.
        item, _overlap = max(
            accepted,
            key=lambda pair: (pair[1], float(pair[0].get("created_unix", 0.0))),
        )
        cell = int(item["cell"])
        row, col = divmod(cell, gw)
        width = max(int(proposal.screen_width), 1)
        height = max(int(proposal.screen_height), 1)
        xy = item.get("xy")
        if isinstance(xy, (tuple, list)) and len(xy) == 2:
            x = int(round(min(max(float(xy[0]), 0.0), 1.0) * max(width - 1, 1)))
            y = int(round(min(max(float(xy[1]), 0.0), 1.0) * max(height - 1, 1)))
        else:
            x = int(round(((col + 0.5) / gw) * max(width - 1, 1)))
            y = int(round(((row + 0.5) / gh) * max(height - 1, 1)))
        return x, y, int(cell)

    def _update_cursor_search_hint_from_events(self, events, proposal, capture_ns: int) -> None:
        """Turn human REL motion into a stable *incremental* search-center hint.

        Raw evdev REL counts are not pixels, so magnitude is never interpreted as
        an absolute cursor coordinate.  Instead each meaningful human gesture
        nudges a normalized guidance center by a bounded step.  This lets the
        user steer Heiwa progressively toward the wanted neural pattern without
        the old large region jumps.
        """
        rel = tuple(
            event for event in events
            if event.event_type == EV_REL and event.code in {REL_X, REL_Y}
        )
        if not rel:
            return
        dx = sum(int(event.value) for event in rel if event.code == REL_X)
        dy = sum(int(event.value) for event in rel if event.code == REL_Y)
        magnitude = (float(dx) ** 2 + float(dy) ** 2) ** 0.5
        if magnitude < self._pointer_feedback_deadzone:
            return
        ux, uy = float(dx) / magnitude, float(dy) / magnitude
        width = max(int(proposal.screen_width), 1)
        height = max(int(proposal.screen_height), 1)

        episode = self._active_cursor_correction
        shell_pointer = self._read_shell_pointer(capture_ns)
        compositor_pointer = shell_pointer or self._read_xwayland_pointer(proposal)
        if compositor_pointer is not None:
            # GNOME Shell supplies compositor coordinates only as an anchor for
            # the teacher's local search region. It is never a target label;
            # neural top-k still chooses the actual purple target nearby.
            base_x, base_y = compositor_pointer
        elif episode is not None and episode.get("guided_xy") is not None:
            base_x, base_y = (float(v) for v in episode["guided_xy"])
        elif self._cursor_search_hint is not None:
            old = self._cursor_search_hint.get("xy", (0.5, 0.5))
            base_x, base_y = float(old[0]), float(old[1])
        else:
            if self._pointer is None:
                self._pointer = (int(proposal.x), int(proposal.y))
            base_x = float(self._pointer[0]) / max(width - 1, 1)
            base_y = float(self._pointer[1]) / max(height - 1, 1)

        # Keep every correction small and predictable. A stronger/faster hand
        # gesture modestly increases the nudge, but cannot teleport the region.
        strength = min(1.8, max(0.65, magnitude ** 0.25))
        step = self._cursor_guidance_step * strength
        hx = min(max(base_x + ux * step, 0.0), 1.0)
        hy = min(max(base_y + uy * step, 0.0), 1.0)
        self._cursor_search_hint = {
            "at_ns": int(capture_ns),
            "xy": (hx, hy),
            "direction": (ux, uy),
            "raw_magnitude": magnitude,
            "step": step,
            "anchor_source": "gnome_shell" if shell_pointer is not None else (
                "xwayland_pointer" if compositor_pointer is not None else "integrated_relative"
            ),
        }
        if episode is not None:
            trace = list(episode.get("guidance_trace", ()))
            trace.append((hx, hy))
            episode["guidance_trace"] = trace[-64:]
            episode["last_activity_ns"] = int(capture_ns)

    def _read_shell_pointer(self, capture_ns: int) -> tuple[float, float] | None:
        try:
            payload = json.loads(self._shell_pointer_path.read_text(encoding="utf-8"))
            updated_ns = int(payload["updated_monotonic_us"]) * 1000
            xy = payload["xy"]
            if payload.get("format") != 1 or int(capture_ns) - updated_ns > 1_000_000_000:
                return None
            if not isinstance(xy, list) or len(xy) != 2:
                return None
            x, y = float(xy[0]), float(xy[1])
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return None
            return x, y
        except (OSError, ValueError, TypeError, KeyError):
            return None

    @staticmethod
    def _read_xwayland_pointer(proposal) -> tuple[float, float] | None:
        """Read GNOME Wayland's XWayland pointer mirror as a spatial anchor.

        This is not screen capture, UI semantics, or a target label.  It merely
        prevents raw evdev units from being mistaken for compositor pixels when
        the user physically steers a correction region.
        """
        library_name = ctypes.util.find_library("X11")
        if not library_name:
            return None
        try:
            x11 = ctypes.CDLL(library_name)
            x11.XOpenDisplay.argtypes = (ctypes.c_char_p,)
            x11.XOpenDisplay.restype = ctypes.c_void_p
            x11.XDefaultRootWindow.argtypes = (ctypes.c_void_p,)
            x11.XDefaultRootWindow.restype = ctypes.c_ulong
            x11.XQueryPointer.argtypes = (
                ctypes.c_void_p, ctypes.c_ulong,
                ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_uint),
            )
            x11.XQueryPointer.restype = ctypes.c_int
            x11.XCloseDisplay.argtypes = (ctypes.c_void_p,)
            display = x11.XOpenDisplay(None)
            if not display:
                return None
            try:
                root = x11.XDefaultRootWindow(display)
                root_return = ctypes.c_ulong(); child_return = ctypes.c_ulong()
                root_x = ctypes.c_int(); root_y = ctypes.c_int()
                win_x = ctypes.c_int(); win_y = ctypes.c_int(); mask = ctypes.c_uint()
                ok = x11.XQueryPointer(
                    display, root, ctypes.byref(root_return), ctypes.byref(child_return),
                    ctypes.byref(root_x), ctypes.byref(root_y),
                    ctypes.byref(win_x), ctypes.byref(win_y), ctypes.byref(mask),
                )
                if not ok:
                    return None
                width = max(int(proposal.screen_width) - 1, 1)
                height = max(int(proposal.screen_height) - 1, 1)
                return (
                    min(max(float(root_x.value) / width, 0.0), 1.0),
                    min(max(float(root_y.value) / height, 0.0), 1.0),
                )
            finally:
                x11.XCloseDisplay(display)
        except (OSError, AttributeError, TypeError, ValueError):
            return None

    def _recent_cursor_search_hint(self, capture_ns: int):
        hint = self._cursor_search_hint
        if hint is None:
            return None
        if int(capture_ns) - int(hint.get("at_ns", 0)) > self._cursor_search_hint_ttl_ns:
            return None
        return {
            "xy": tuple(float(v) for v in hint.get("xy", (0.5, 0.5))),
            "direction": tuple(float(v) for v in hint.get("direction", (0.0, 0.0))),
            "age_ms": (int(capture_ns) - int(hint.get("at_ns", capture_ns))) / 1_000_000.0,
        }

    def _refresh_active_cursor_guidance(self, capture_ns: int) -> dict[str, object] | None:
        """Apply a fresh human pointer gesture to an already-active F23 search.

        The gesture is a direction/region hint only.  We intentionally do not
        reinterpret evdev REL counts as compositor pixels.  A meaningful change
        of hint re-opens the local candidate set so the user can steer, observe
        Heiwa's next attempt, and steer again.
        """
        episode = self._active_cursor_correction
        if episode is None:
            return None
        hint = self._recent_cursor_search_hint(capture_ns)
        if hint is None:
            return {"active": True, "updated": False, "reason": "no_fresh_human_motion"}
        hint_ns = int(self._cursor_search_hint.get("at_ns", 0)) if self._cursor_search_hint else 0
        if hint_ns <= int(episode.get("last_guidance_ns", 0)):
            return {"active": True, "updated": False, "reason": "guidance_already_consumed"}

        new_xy = tuple(float(v) for v in hint["xy"])
        old_xy = episode.get("guided_xy")
        direction = tuple(float(v) for v in hint.get("direction", (0.0, 0.0)))
        old_direction = episode.get("guidance_direction")
        direction_changed = False
        if old_direction is not None:
            dot = direction[0] * float(old_direction[0]) + direction[1] * float(old_direction[1])
            direction_changed = dot < 0.35
        episode["guided_xy"] = new_xy
        episode["guidance_direction"] = direction
        episode["last_guidance_ns"] = hint_ns
        episode["last_activity_ns"] = int(capture_ns)
        episode["awaiting_guidance"] = False
        refinements = int(episode.get("guidance_refinements", 0)) + 1
        episode["guidance_refinements"] = refinements
        # Repeated steering should *narrow* the local search. Only a real turn
        # re-opens candidate memory; small corrections keep prior misses taboo.
        if direction_changed:
            # A turn changes where we search, but does not forgive already
            # tried/rejected motor patterns during this teaching episode.
            refinements = 1
            episode["guidance_refinements"] = refinements
        radius = max(
            self._cursor_guidance_radius_min,
            self._cursor_guidance_radius_start * (0.78 ** max(refinements - 1, 0)),
        )
        episode["search_radius"] = float(radius)
        return {
            "active": True,
            "updated": True,
            "guided_xy": new_xy,
            "direction": direction,
            "search_radius": float(radius),
            "guidance_refinements": refinements,
            "candidate_memory_reset": bool(direction_changed),
        }

    def _target_grid(self, proposal):
        probabilities = tuple(float(v) for v in getattr(proposal, "target_distribution", ()) or ())
        gh = max(int(self.inference.config.image_height) // 16, 1)
        gw = max(int(self.inference.config.image_width) // 16, 1)
        if not probabilities or len(probabilities) != gh * gw:
            return probabilities, gh, gw
        return probabilities, gh, gw

    def _proposal_target_cell(self, proposal, gw: int, gh: int) -> int:
        width = max(int(proposal.screen_width), 1)
        height = max(int(proposal.screen_height), 1)
        col = min(gw - 1, max(0, int((float(proposal.x) / max(width - 1, 1)) * gw)))
        row = min(gh - 1, max(0, int((float(proposal.y) / max(height - 1, 1)) * gh)))
        return row * gw + col

    def _continue_cursor_feedback_replan(self, proposal, capture_ns: int) -> dict[str, object] | None:
        episode = self._active_cursor_correction
        if episode is None:
            return None
        activity_ns = int(episode.get("last_activity_ns", episode.get("started_ns", capture_ns)))
        age_ns = int(capture_ns) - activity_ns
        hops = int(episode.get("hops", 0))
        if age_ns > self._cursor_correction_ttl_ns or hops >= self._cursor_correction_max_hops:
            self._active_cursor_correction = None
            return {"active": False, "released": True, "reason": "idle_ttl_or_hop_limit", "hops": hops}

        # F23 may be pressed first and the pointing gesture made afterwards.
        # In that case Heiwa waits instead of jumping to an unrelated global mode.
        if episode.get("guided_xy") is None:
            return {"active": True, "released": False, "reason": "waiting_for_user_guidance", "hops": hops}

        last_hop_ns = int(episode.get("last_hop_ns", 0))
        if last_hop_ns and int(capture_ns) - last_hop_ns < self._cursor_correction_min_hop_ns:
            return {"active": True, "released": False, "reason": "watching_between_hops", "hops": hops}

        probabilities, gh, gw = self._target_grid(proposal)
        if not probabilities or len(probabilities) != gh * gw:
            self._active_cursor_correction = None
            return {"active": False, "released": True, "reason": "target_distribution_unavailable", "hops": hops}

        # Do not terminate merely because the network changed one grid cell.
        # The correction episode is an interactive search: the human can keep
        # nudging the direction, F24 explicitly accepts, or the short TTL ends.
        result = self._immediate_pointer_feedback_replan(proposal, arm_episode=False)
        if self._active_cursor_correction is not None:
            self._active_cursor_correction["last_hop_ns"] = int(capture_ns)
            self._active_cursor_correction["last_activity_ns"] = int(capture_ns)
        result["active"] = self._active_cursor_correction is not None
        result["followup"] = True
        result["interactive_guided_search"] = True
        return result

    def _immediate_pointer_feedback_replan(self, proposal, *, arm_episode: bool = False) -> dict[str, object]:
        """Immediately leave a rejected neural target and remember the taboo briefly.

        F23 is not treated as a correct destination label.  It rejects the
        current motor pattern.  A short-lived correction episode remembers
        rejected target cells and, on following frames, watches whether the
        neural proposal falls back into them.  Repeated fallback triggers another
        visible cursor move to the next non-rejected neural candidate.
        """
        result: dict[str, object] = {"attempted": False, "changed": False, "reason": "unavailable"}
        if not self.allow_mouse:
            result["reason"] = "mouse_disabled"
            return result
        probabilities, gh, gw = self._target_grid(proposal)
        if not probabilities:
            result["reason"] = "no_target_distribution"
            return result
        if len(probabilities) != gh * gw:
            result["reason"] = "target_distribution_shape_mismatch"
            return result

        now_ns = time.monotonic_ns()
        rejected_cell = self._proposal_target_cell(proposal, gw, gh)
        existing = self._active_cursor_correction
        if existing is not None:
            activity_ns = int(existing.get("last_activity_ns", existing.get("started_ns", 0)))
            if now_ns - activity_ns > self._cursor_correction_ttl_ns:
                self._active_cursor_correction = None
                existing = None
        # Only the first F23 opens the teacher session. Further F23 presses in
        # that session must accumulate A/B/C rejections instead of replacing
        # the session state and allowing an A -> B -> A loop.
        if existing is None:
            hint = self._recent_cursor_search_hint(now_ns)
            hint_ns = int(self._cursor_search_hint.get("at_ns", 0)) if (hint is not None and self._cursor_search_hint) else 0
            self._active_cursor_correction = {
                "started_ns": now_ns,
                "last_activity_ns": now_ns,
                "last_hop_ns": 0,
                "last_guidance_ns": hint_ns,
                "rejected_cells": set(self._cell_neighbourhood(rejected_cell, gw, gh)),
                "hard_rejected_cells": set(self._cell_neighbourhood(rejected_cell, gw, gh)),
                "attempted_cells": set(),
                "guidance_trace": [],
                "guided_xy": tuple(hint["xy"]) if hint is not None else None,
                "guidance_direction": tuple(hint["direction"]) if hint is not None else None,
                "guidance_refinements": 1 if hint is not None else 0,
                "search_radius": self._cursor_guidance_radius_start,
                "awaiting_guidance": hint is None,
                "hops": 0,
            }
        episode = self._active_cursor_correction
        assert episode is not None
        rejected = set(int(v) for v in episode.get("rejected_cells", ()))
        hard_rejected = set(int(v) for v in episode.get("hard_rejected_cells", ()))
        new_reject = self._cell_neighbourhood(rejected_cell, gw, gh)
        rejected.update(new_reject)
        hard_rejected.update(new_reject)
        rejected.update(self._remembered_rejected_cells(probabilities, gw, gh))
        if arm_episode:
            self._remember_rejected_pattern(probabilities, rejected_cell, gw, gh)
        episode["rejected_cells"] = rejected
        episode["hard_rejected_cells"] = hard_rejected
        episode["last_activity_ns"] = now_ns
        if arm_episode and episode.get("guided_xy") is None:
            result.update({
                "reason": "waiting_for_user_guidance",
                "correction_armed": True,
                "interactive_guided_search": True,
                "semantic_target_asserted": False,
                "rejected_cells": tuple(sorted(rejected)),
                "hard_rejected_cells": tuple(sorted(hard_rejected)),
                "rejected_pattern_count": len(hard_rejected),
            })
            return result

        width = max(int(proposal.screen_width), 1)
        height = max(int(proposal.screen_height), 1)
        rejected_x = float(proposal.x) / max(width - 1, 1)
        rejected_y = float(proposal.y) / max(height - 1, 1)
        accepted = self._remembered_accepted_cells(probabilities, gw, gh)
        # F24 is local teacher evidence, not a scalar reward. For a matching
        # visual target distribution it promotes the accepted neural mode
        # within top-k search. Any current/persistent F23 rejection below still
        # excludes it absolutely.
        ranked = sorted(
            range(len(probabilities)),
            key=lambda i: (i in accepted, probabilities[i]),
            reverse=True,
        )
        attempted = set(int(v) for v in episode.get("attempted_cells", ()))
        guided_xy = episode.get("guided_xy")

        chosen = None
        chosen_distance = -1.0
        guided_distance = None
        candidates = [index for index in ranked if index not in rejected and index not in attempted]
        if guided_xy is not None and candidates:
            gx, gy = float(guided_xy[0]), float(guided_xy[1])
            radius = float(episode.get("search_radius", self._cursor_guidance_radius_start))
            local = []
            for index in candidates:
                row, col = divmod(index, gw)
                nx, ny = (col + 0.5) / gw, (row + 0.5) / gh
                d = ((nx - gx) ** 2 + (ny - gy) ** 2) ** 0.5
                if d <= radius:
                    local.append((index, d))
            # If the current radius contains no neural mode, expand only enough
            # to reach the nearest available one; never jump to a remote global mode.
            if not local:
                nearest = []
                for index in candidates:
                    row, col = divmod(index, gw)
                    nx, ny = (col + 0.5) / gw, (row + 0.5) / gh
                    d = ((nx - gx) ** 2 + (ny - gy) ** 2) ** 0.5
                    nearest.append((index, d))
                nearest.sort(key=lambda item: item[1])
                local = nearest[:max(1, min(4, len(nearest)))]
            # Distance dominates, while neural probability chooses among nearby
            # modes. This makes manual steering spatially faithful but still lets
            # Heiwa recognize its own learned pattern rather than copying a point.
            best_probability = max(probabilities[index] for index, _ in local) or 1.0
            def local_score(item):
                index, d = item
                probability_penalty = 1.0 - (probabilities[index] / best_probability)
                return d + 0.035 * probability_penalty
            chosen, guided_distance = min(local, key=local_score)
            row, col = divmod(chosen, gw)
            nx, ny = (col + 0.5) / gw, (row + 0.5) / gh
            chosen_distance = ((nx - rejected_x) ** 2 + (ny - rejected_y) ** 2) ** 0.5
            result["search_radius"] = radius
            result["local_candidate_count"] = len(local)
        else:
            for index in candidates:
                row, col = divmod(index, gw)
                nx = (col + 0.5) / gw
                ny = (row + 0.5) / gh
                distance = ((nx - rejected_x) ** 2 + (ny - rejected_y) ** 2) ** 0.5
                if distance >= 0.18:
                    chosen = index
                    chosen_distance = distance
                    break
            if chosen is None:
                for index in candidates:
                    row, col = divmod(index, gw)
                    nx = (col + 0.5) / gw
                    ny = (row + 0.5) / gh
                    distance = ((nx - rejected_x) ** 2 + (ny - rejected_y) ** 2) ** 0.5
                    if distance > chosen_distance:
                        chosen = index
                        chosen_distance = distance
        if chosen is None:
            self._active_cursor_correction = None
            result["reason"] = "no_nonrejected_target"
            return result

        attempted.add(chosen)
        episode["attempted_cells"] = attempted
        row, col = divmod(chosen, gw)
        tx = int(round(((col + 0.5) / gw) * max(width - 1, 1)))
        ty = int(round(((row + 0.5) / gh) * max(height - 1, 1)))
        if self._pointer is None:
            self._pointer = (int(proposal.x), int(proposal.y))
        sx, sy = self._pointer
        if (sx, sy) == (tx, ty):
            # Do not accept this candidate as progress; taboo it and let the
            # next follow-up choose another one.
            rejected.add(chosen)
            episode["rejected_cells"] = rejected
            result["reason"] = "alternate_equals_current_pointer"
            return result

        points = []
        for step in range(1, 6):
            t = step / 5.0
            points.append((round(sx + (tx - sx) * t), round(sy + (ty - sy) * t)))
        result["attempted"] = True
        try:
            with self.action_arbiter.claim("heiwa"):
                motor_result = self.motor.move_pointer_trajectory(tuple(points), duration=0.08)
        except (DesktopMotorError, ValueError, TypeError) as exc:
            result["reason"] = f"motor_error:{type(exc).__name__}"
            return result
        self._pointer = (tx, ty)
        episode["rejected_cells"] = rejected
        episode["hard_rejected_cells"] = hard_rejected
        episode["hops"] = int(episode.get("hops", 0)) + 1
        result.update({
            "changed": int(getattr(motor_result, "returncode", 1)) == 0,
            "reason": "alternate_neural_target",
            "from_xy": (sx, sy), "to_xy": (tx, ty),
            "target_rank": ranked.index(chosen) + 1,
            "target_probability": probabilities[chosen],
            "normalized_distance_from_rejected_target": chosen_distance,
            "rejected_cells": tuple(sorted(rejected)),
            "hard_rejected_cells": tuple(sorted(hard_rejected)),
            "guidance_trace_points": len(episode.get("guidance_trace", ())),
            "correction_hops": int(episode["hops"]),
            "semantic_target_asserted": False,
            "interactive_guided_search": True,
            "guided_distance": guided_distance,
        })
        return result

    def execute(self, intent: ActionIntent, now: float) -> ActionAttempt:
        validate_intent_for_world(intent, world_id=self.world_id)
        offered = self._affordances.get(intent.affordance_id)
        if offered is None or offered.action != intent.action: return self._reject(intent, now, "stale or mismatched affordance")
        confidence = float(offered.payload["neural_confidence"])
        x, y = int(offered.payload["neural_target_x"]), int(offered.payload["neural_target_y"])
        if self._pointer is None:
            self._pointer = (int(offered.payload["screen_width"]) // 2, int(offered.payload["screen_height"]) // 2)
        action = intent.action.removeprefix("desktop.")
        if action in {"move_pointer", "left_click", "double_click", "right_click", "scroll"} and not self.allow_mouse:
            return self._reject(intent, now, "mouse execution disabled by persistent autonomy config")
        if action in {"key_press", "type_text", "key_chord"} and not self.allow_keyboard:
            return self._reject(intent, now, "keyboard execution disabled by persistent autonomy config")
        if action in {"left_click", "double_click"} and not self.allow_clicks:
            return self._reject(intent, now, "click execution requires --allow-clicks")
        if action == "right_click" and not self.allow_right_clicks:
            return self._reject(intent, now, "right click execution requires --allow-right-clicks")
        if action in {"key_press", "type_text", "key_chord"} and not self.allow_typing:
            return self._reject(intent, now, "keyboard execution requires --allow-typing")
        if action != "noop":
            allowed, reason = self.safety.allow(confidence=confidence, now=now)
            if not allowed: return self._reject(intent, now, reason)
        if action == "key_press" and "key_code" not in intent.parameters:
            from dataclasses import replace
            intent = replace(intent, parameters={"key_code": int(offered.payload["neural_key_code"])})
        action_started_ns = time.monotonic_ns()
        try:
            with self.action_arbiter.claim("heiwa"):
                result = self._actuate(action, x, y, intent, offered)
        except (DesktopMotorError, ValueError, TypeError) as exc: return self._reject(intent, now, str(exc))
        action_finished_ns = time.monotonic_ns()
        attempt = ActionAttempt(
            id=f"desktop-attempt-{uuid4().hex}", intent_id=intent.id, transition_id=intent.transition_id,
            world=self.world_id, action=intent.action, attempted_at=now, accepted=True,
            payload={"motor_returncode": result.returncode, "neural_target": (x, y),
                     "neural_confidence": confidence, "completion_unverified": True},
        )
        assert self.inference.state is not None
        assert self.inference.last_frame is not None
        normalized = (x / max(int(offered.payload["screen_width"]), 1), y / max(int(offered.payload["screen_height"]), 1), 0.0, 0.0)
        action_index = ACTION_KINDS.index(action)
        before_state = self.inference.state.detached()
        with torch.inference_mode():
            predicted, _ = self.inference.model.predict_next(
                before_state, torch.tensor([action_index], device=self.inference.device),
                torch.tensor([normalized], device=self.inference.device),
            )
        self._pending = _Pending(
            attempt, before_state, action_index, normalized, predicted.detach(),
            self.inference.last_frame.detach().clone(), action_started_ns, action_finished_ns,
        )
        if action in {"move_pointer", "left_click", "double_click", "right_click", "scroll"}:
            self._recent_pointer_action = self._pending
        self.inference.remember_action(action_index, normalized)
        return attempt

    def _move_neurally(self, x: int, y: int, offered: Affordance) -> MotorResult:
        assert self._pointer is not None
        sx, sy = self._pointer; width = int(offered.payload["screen_width"]); height = int(offered.payload["screen_height"])
        duration_raw, bend_x, bend_y = tuple(float(v) for v in offered.payload["neural_trajectory"])
        duration = 0.03 + 0.17 * (duration_raw + 1.0) / 2.0
        cx = (sx + x) / 2 + bend_x * width * 0.12; cy = (sy + y) / 2 + bend_y * height * 0.12
        points = []
        for step in range(1, 7):
            t = step / 6; u = 1 - t
            points.append((round(u*u*sx + 2*u*t*cx + t*t*x), round(u*u*sy + 2*u*t*cy + t*t*y)))
        result = self.motor.move_pointer_trajectory(tuple(points), duration=duration)
        self._pointer = (x, y); return result

    def _actuate(self, action: str, x: int, y: int, intent: ActionIntent, offered: Affordance) -> MotorResult:
        if action == "noop": return MotorResult(0, "", "")
        if action == "move_pointer": return self._move_neurally(x, y, offered)
        if action in {"left_click", "double_click", "right_click"}:
            self._move_neurally(x, y, offered)
            return {"left_click": self.motor.click_left, "double_click": self.motor.double_click_left, "right_click": self.motor.click_right}[action]()
        if action == "scroll": return self.motor.scroll(int(intent.parameters.get("dy", 1)))
        if action == "key_press": return self.motor.press_key(int(intent.parameters["key_code"]))
        if action == "type_text": return self.motor.type_unicode_linux(str(intent.parameters["text"]))
        if action == "key_chord": return self.motor.press_chord(tuple(int(v) for v in intent.parameters["key_codes"]))
        raise ValueError(f"unsupported neural Desktop action: {action}")

    def _reject(self, intent: ActionIntent, now: float, reason: str) -> ActionAttempt:
        return ActionAttempt(id=f"desktop-attempt-{uuid4().hex}", intent_id=intent.id, transition_id=intent.transition_id, world=self.world_id, action=intent.action, attempted_at=now, accepted=False, payload={"reason": reason})

    def close(self) -> None:
        if self._closed: return
        self._closed = True; self._affordances.clear(); self._pending = None; self._recent_pointer_action = None; self._active_cursor_correction = None; self._cursor_search_hint = None
        if self._input is not None: self._input.close(); self._input = None
        self.inference.close()
