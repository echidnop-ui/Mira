from __future__ import annotations

import argparse
from pathlib import Path

import torch

from worlds.desktop.inference import NeuralDesktopInference
from worlds.desktop.motor import DesktopMotor


def _pixel_change(before: torch.Tensor, after: torch.Tensor) -> tuple[float, float]:
    difference = (after.float() - before.float()).abs()
    return float(difference.mean()), float(difference.mean(dim=1).gt(0.03).float().mean())


def main() -> int:
    parser = argparse.ArgumentParser(description="Pixel-only gated neural visual-control live probe")
    parser.add_argument("--checkpoint", type=Path, default=Path("state/neural_desktop.pt"))
    parser.add_argument("--allow-clicks", action="store_true")
    parser.add_argument("--allow-typing", action="store_true")
    parser.add_argument("--text", default="heiwa")
    args = parser.parse_args()
    if args.allow_typing and not args.allow_clicks:
        parser.error("visual typing probe requires --allow-clicks to focus the predicted region")
    inference = NeuralDesktopInference(args.checkpoint); motor = DesktopMotor()
    try:
        proposal = inference.capture_and_propose(); before = inference.state
        assert before is not None
        assert inference.last_frame is not None
        initial_frame = inference.last_frame.clone()
        duration_raw, bend_x, bend_y = proposal.trajectory
        sx, sy = proposal.screen_width // 2, proposal.screen_height // 2
        cx = (sx + proposal.x) / 2 + bend_x * proposal.screen_width * .12
        cy = (sy + proposal.y) / 2 + bend_y * proposal.screen_height * .12
        points = tuple((round((1-t)**2*sx + 2*(1-t)*t*cx + t*t*proposal.x),
                        round((1-t)**2*sy + 2*(1-t)*t*cy + t*t*proposal.y))
                       for t in (1/6, 2/6, 3/6, 4/6, 5/6, 1.0))
        motor.move_pointer_trajectory(points, duration=.03 + .17 * (duration_raw + 1) / 2)
        executed = "move_pointer"
        inference.capture_and_propose()
        assert inference.last_frame is not None
        moved_frame = inference.last_frame.clone()
        move_mae, move_fraction = _pixel_change(initial_frame, moved_frame)
        click_mae = click_fraction = 0.0
        typing_mae = typing_fraction = 0.0
        if args.allow_clicks:
            motor.click_left(); executed = "left_click"
            inference.capture_and_propose()
            assert inference.last_frame is not None
            clicked_frame = inference.last_frame.clone()
            click_mae, click_fraction = _pixel_change(moved_frame, clicked_frame)
        if args.allow_typing:
            motor.type_unicode_linux(args.text); executed = "type_text"
            inference.capture_and_propose()
            assert inference.last_frame is not None
            typed_frame = inference.last_frame.clone()
            typing_mae, typing_fraction = _pixel_change(clicked_frame, typed_frame)
        after_proposal = inference.capture_and_propose(); after = inference.state
        assert after is not None
        print({"ok": True, "pixel_only": True, "atspi": False, "executed": executed,
               "target": [proposal.x, proposal.y], "target_bbox": proposal.target_bbox,
               "confidence": round(proposal.confidence, 4),
               "move_pixel_mae": move_mae, "move_changed_fraction": move_fraction,
               "click_pixel_mae": click_mae, "click_changed_fraction": click_fraction,
               "visual_click_confirmed": bool(args.allow_clicks and click_fraction > 1e-4),
               "typing_pixel_mae": typing_mae, "typing_changed_fraction": typing_fraction,
               "visual_typing_confirmed": bool(args.allow_typing and typing_fraction > 1e-4),
               "next_target": [after_proposal.x, after_proposal.y]})
    finally:
        inference.portal.close()
    return 0


if __name__ == "__main__": raise SystemExit(main())
