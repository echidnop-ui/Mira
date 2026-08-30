from __future__ import annotations

import time
import torch

from contracts.cognition import ActionIntent
from runtime.embodied_smoke import _Inference, _Motor
from worlds.desktop.neural import NeuralDesktopBody, NeuralDesktopConfig
from worlds.desktop.neural_adapter import NeuralDesktopWorldAdapter, NeuralSafetyGate, _forwarded_pointer_duplicate
from worlds.desktop.online_learning import OnlineDesktopLearner
from worlds.desktop.training.input_events import EV_REL, REL_X, RawInputEvent


class _ForwardedInput:
    def drain(self, before_ns: int, after_ns: int):
        return (RawInputEvent((before_ns + after_ns) // 2, "/dev/input/event-forwarded",
                              "input-remapper Mouse forwarded", EV_REL, REL_X, 400),)
    def close(self): pass


def _intent(affordance, transition: str, now: float) -> ActionIntent:
    return ActionIntent(f"intent-{transition}", transition, affordance.id,
                        affordance.world, affordance.action, now, {})


def main() -> int:
    # Scheduler exposes physical affordances only when they can execute. Noop is
    # always available and does not consume the physical motor interval.
    motor = _Motor(); inference = _Inference()
    world = NeuralDesktopWorldAdapter(
        inference, motor=motor,
        safety=NeuralSafetyGate(confidence=0.0, max_actions_per_second=0.5),
        allow_clicks=True, allow_right_clicks=True, allow_typing=True,
        capture_input=False, online_learning=False,
    )
    first = world.poll(100.0)
    move = next(value for value in first.affordances if value.action == "desktop.move_pointer")
    assert world.execute(_intent(move, "move", 100.0), 100.0).accepted
    assert motor.calls and motor.calls[-1][0] == "mouse"
    world._input = _ForwardedInput(); world._input_status = "linux-evdev"
    cooldown = world.poll(100.1)
    assert cooldown.completed_actions[0].payload["ownership"] == "heiwa"
    assert cooldown.completed_actions[0].payload["ignored_forwarded_pointer_duplicates"] == 1
    assert tuple(value.action for value in cooldown.affordances) == ("desktop.noop",)
    noop = cooldown.affordances[0]
    assert world.execute(_intent(noop, "noop", 100.1), 100.1).accepted
    ready = world.poll(102.1)
    assert any(value.action == "desktop.move_pointer" for value in ready.affordances)
    world.close()

    forwarded = RawInputEvent(1, "x", "input-remapper Mouse forwarded", EV_REL, REL_X, 4)
    physical = RawInputEvent(1, "y", "CX Wireless mouse", EV_REL, REL_X, 4)
    assert _forwarded_pointer_duplicate(forwarded)
    assert not _forwarded_pointer_duplicate(physical)

    # Repeated supervised correction of the same visual state must reduce its
    # actual decoded target error; autonomous optimizer does not own these heads.
    torch.manual_seed(13)
    config = NeuralDesktopConfig(image_height=64, image_width=96, hidden_size=64,
        token_size=64, region_count=4, transformer_layers=1, transformer_heads=4,
        action_embed_size=16, task_context_size=32, max_read_bytes=12)
    model = NeuralDesktopBody(config)
    learner = OnlineDesktopLearner(model, device=torch.device("cpu"), learning_rate=1e-5)
    state = model.observe(torch.rand(1, 3, 64, 96)).detached()
    with torch.no_grad(): point = model.propose(state).target_point[0]
    target = (min(float(point[0]) + .35, .95), max(float(point[1]) - .30, .05))
    metrics = learner.learn_pointer_correction(before_state=state, target_xy=target, repeats=3)
    assert metrics["pointer_feedback_target_error_after"] < metrics["pointer_feedback_target_error_before"]
    print({"ok": True, "motor_reached": True, "cooldown_affordances": ["desktop.noop"],
           "forwarded_ownership": "heiwa", "pointer_error_before": metrics["pointer_feedback_target_error_before"],
           "pointer_error_after": metrics["pointer_feedback_target_error_after"],
           "cell_loss": metrics["pointer_feedback_cell_loss"], "updates": metrics["pointer_feedback_repeats"]})
    return 0


if __name__ == "__main__": raise SystemExit(main())
