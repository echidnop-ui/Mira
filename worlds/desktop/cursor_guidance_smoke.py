from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile

from worlds.desktop.inference import NeuralMotorProposal
from worlds.desktop.motor import MotorResult
from worlds.desktop.neural import ACTION_KINDS
from worlds.desktop.neural_adapter import NeuralDesktopWorldAdapter
from worlds.desktop.neural_adapter import KEY_F23
from worlds.desktop.training.input_events import EV_KEY, EV_REL, REL_X, RawInputEvent


class _Inference:
    config = SimpleNamespace(image_height=64, image_width=64)

    def close(self) -> None:
        pass


class _Motor:
    def __init__(self) -> None:
        self.trajectories: list[tuple[tuple[int, int], ...]] = []

    def move_pointer_trajectory(self, points, *, duration: float) -> MotorResult:
        self.trajectories.append(tuple(points))
        return MotorResult(0, "", "")


def _proposal(x: int, y: int, peak: int) -> NeuralMotorProposal:
    targets = [0.01] * 16
    targets[peak] = 0.85
    actions = tuple(1.0 / len(ACTION_KINDS) for _ in ACTION_KINDS)
    return NeuralMotorProposal(
        action="move_pointer", x=x, y=y, confidence=0.7,
        action_distribution=actions, target_distribution=tuple(targets),
        screen_width=400, screen_height=400, key_code=30,
        frame_digest=f"frame-{x}-{y}", target_bbox=(0.0, 0.0, 0.1, 0.1),
        trajectory=(0.0, 0.0, 0.0), read_byte_ids=(), read_confidence=0.0,
    )


def main() -> int:
    motor = _Motor()
    world = NeuralDesktopWorldAdapter(
        _Inference(), motor=motor, capture_input=False, online_learning=False,
        allow_mouse=True,
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            world._cursor_pattern_memory_path = Path(directory) / "patterns.json"
            world._cursor_overlay_path = Path(directory) / "overlay.json"
            world._cursor_pattern_memory = []
            world._read_shell_pointer = lambda _capture_ns: (0.8, 0.8)
            direct = world._accept_cursor_feedback(_proposal(25, 25, 0))
            assert direct is not None and direct["cell"] == 15
            assert world._cursor_pattern_memory[-1]["source"] == "direct_hover_f24"
            routed_x, routed_y, routed_cell = world._effective_teacher_target(_proposal(25, 25, 0))
            assert routed_cell == 15
            assert abs(routed_x - round(.8 * 399)) <= 1
            assert abs(routed_y - round(.8 * 399)) <= 1
            display = world._format_reading_for_display((0xD0 + 4, 0x96 + 4, ord("7") + 4, 0x07 + 4, 0xFF + 4, 2))
            assert display == "Ж7\\u0007\\xFF", display
            world._cursor_pattern_memory = []
            world._pointer = (100, 100)
            right = RawInputEvent(900_000_000, "/dev/input/mouse", "physical mouse", EV_REL, REL_X, 40)
            world._integrate_physical_pointer((right,), _proposal(25, 25, 0), 900_000_000)
            assert world._pointer == (140, 100)
            now_ns = 1_000_000_000
            press = RawInputEvent(now_ns, "/dev/input/test", "teacher", EV_KEY, KEY_F23, 1)
            due, presses = world._update_f23_eraser((press,), now_ns)
            assert due and presses == 1 and world._f23_held
            due, presses = world._update_f23_eraser((), now_ns + world._f23_erase_interval_ns + 1)
            assert due and presses == 0 and world._f23_held
            release = RawInputEvent(now_ns + 2, "/dev/input/test", "teacher", EV_KEY, KEY_F23, 0)
            due, _ = world._update_f23_eraser((release,), now_ns + world._f23_erase_interval_ns * 2)
            assert not due and not world._f23_held
            first = world._immediate_pointer_feedback_replan(
                _proposal(25, 25, 0), arm_episode=True,
            )
            episode = world._active_cursor_correction
            assert episode is not None
            first_rejected = set(episode["hard_rejected_cells"])
            assert first["reason"] == "waiting_for_user_guidance"

            # This is the second F23 in the same correction session. Its remote
            # visual neighbourhood must be accumulated, not replace the first.
            second = world._immediate_pointer_feedback_replan(
                _proposal(375, 375, 15), arm_episode=True,
            )
            assert world._active_cursor_correction is episode
            second_rejected = set(episode["hard_rejected_cells"])
            assert first_rejected < second_rejected
            assert second["rejected_pattern_count"] == len(second_rejected)

            world._pointer = (375, 375)
            accepted = world._accept_cursor_feedback(_proposal(375, 375, 15))
            assert accepted is not None
            assert world._active_cursor_correction is None
            assert any(item["kind"] == "accepted" for item in world._cursor_pattern_memory)
            assert all(item.get("kind") != "accepted" or item.get("confirmed_active_session") is True
                       for item in world._cursor_pattern_memory)
            assert all("context" in item for item in world._cursor_pattern_memory)
            probabilities, gh, gw = world._target_grid(_proposal(375, 375, 15))
            assert 15 in world._remembered_accepted_cells(probabilities, gw, gh)
            world._publish_cursor_overlay(_proposal(375, 375, 15))
            assert world._cursor_overlay_path.exists()

            print({
                "ok": True,
                "f23_rejected_before": len(first_rejected),
                "f23_rejected_after": len(second_rejected),
                "same_session": True,
                "f24_closed_session": True,
                "visual_context_memory": True,
                "accepted_pattern_reused": True,
                "held_f23_continuous_eraser": True,
                "overlay_state_published": True,
                "physical_pointer_direction_integrated": True,
                "f24_without_f23_uses_hovered_cell": True,
                "direct_hover_f24_routes_target": True,
                "exact_hover_coordinate_preserved": True,
                "readable_utf8_overlay_text": True,
            })
            return 0
    finally:
        world.close()


if __name__ == "__main__":
    raise SystemExit(main())
