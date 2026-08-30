from __future__ import annotations

from worlds.desktop.training.input_events import BTN_LEFT, EV_KEY, EV_REL, REL_X, REL_Y, RawInputEvent, PointerState, aggregate_events
from worlds.desktop.training.record_demonstration import _transition
from pathlib import Path


def main() -> int:
    before, after = 1_000_000_000, 1_100_000_000
    events = (
        RawInputEvent(before + 10_000_000, "/dev/input/event-test", "mouse", EV_REL, REL_X, 2),
        RawInputEvent(before + 12_000_000, "/dev/input/event-test", "mouse", EV_REL, REL_X, 9),
        RawInputEvent(before + 20_000_000, "/dev/input/event-test", "mouse", EV_REL, REL_Y, 5),
        RawInputEvent(before + 30_000_000, "/dev/input/event-test", "mouse", EV_KEY, BTN_LEFT, 1),
        RawInputEvent(before + 48_000_000, "/dev/input/event-test", "mouse", EV_KEY, BTN_LEFT, 0),
        RawInputEvent(before + 55_000_000, "/dev/input/event-test", "teacher", EV_KEY, 191, 1),
        RawInputEvent(before + 56_000_000, "/dev/input/event-test", "teacher", EV_KEY, 192, 1),
        RawInputEvent(before + 57_000_000, "/dev/input/event-test", "teacher", EV_KEY, 193, 1),
        RawInputEvent(before + 58_000_000, "/dev/input/event-test", "teacher", EV_KEY, 194, 1),
    )
    pointer = PointerState(1920, 1080, 960, 540); aggregated = aggregate_events(events, pointer)
    assert len([item for item in aggregated if item["action"] == "move_pointer"]) == 1
    press = next(item for item in aggregated if item["action"] == "left_click")
    release = next(item for item in aggregated if item.get("motor_event", {}).get("state") == "release")
    assert before < int(press["timestamp_ns"]) < after
    assert release["motor_event"]["press_release_duration_ns"] == 18_000_000
    assert all(0 <= value <= 1 for value in press["target_xy"])
    assert release["trainable"] is False and release["motor_event"]["language_teacher"] is False
    feedback = {
        item["motor_event"]["feedback_command"]: item
        for item in aggregated if item.get("motor_event", {}).get("feedback_command")
    }
    assert set(feedback) == {"F21", "F22", "F23", "F24"}
    assert all(not item["trainable"] for item in feedback.values())
    transition = _transition(Path("root/frames/a.webp"), Path("root/frames/b.webp"), Path("root"), press, before, after)
    assert transition.frame_before_ns < transition.action_timestamp_ns < transition.frame_after_ns
    print({"ok": True, "aggregated_events": len(aggregated), "trajectory_points": len(aggregated[0]["motor_event"]["trajectory"])})
    return 0


if __name__ == "__main__": raise SystemExit(main())
