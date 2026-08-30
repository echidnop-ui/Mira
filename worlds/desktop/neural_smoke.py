from __future__ import annotations

from pathlib import Path
import tempfile

import torch

from worlds.desktop.neural import NeuralDesktopBody, NeuralDesktopConfig
from worlds.desktop.inference import NeuralDesktopInference
from worlds.desktop.training.train import load_checkpoint, save_checkpoint


def main() -> int:
    torch.manual_seed(3)
    config = NeuralDesktopConfig(image_height=64, image_width=96, hidden_size=64, token_size=64, region_count=4, transformer_layers=1, transformer_heads=4, action_embed_size=16)
    model = NeuralDesktopBody(config)
    before = torch.rand(2, 3, 64, 96); after = before.clone(); after[:, :, 20:38, 35:58] += 0.1; after.clamp_(0, 1)
    actions = torch.tensor([1, 2]); parameters = torch.zeros(2, 4); target = torch.tensor([8, 9])
    first = model.observe(before); second = model.observe(after, first, actions, parameters)
    assert first.region_tokens.shape == (2, 4, 64) and second.temporal_state.shape == (2, 64)
    motor = model.propose(first, torch.zeros(2, config.task_context_size))
    assert motor.target_logits.shape == (2, 24) and motor.target_bbox.shape == (2, 4)
    assert motor.target_point.shape == (2, 2) and motor.trajectory.shape == (2, 3)
    target_xy = torch.tensor([[0.45, 0.4], [0.6, 0.5]])
    bbox = torch.tensor([[0.3, 0.2, 0.6, 0.5], [0.45, 0.3, 0.75, 0.65]])
    read = torch.full((2, config.max_read_bytes + 2), -100); read[:, :4] = torch.tensor([1, 212, 213, 2])
    trajectory = torch.tensor([[0.1, 0.05, 0.0], [0.2, -0.03, 0.04]])
    losses = model.training_losses(before, after, actions, parameters, target,
                                   target_xy=target_xy, target_bbox=bbox,
                                   read_tokens=read, trajectory_target=trajectory)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3); optimizer.zero_grad(); losses["total"].backward(); optimizer.step()
    model.zero_grad(set_to_none=True)
    reading = model.training_losses(
        before, after, actions, parameters, target,
        target_bbox=torch.full_like(bbox, float("nan")), read_tokens=read,
    )["foveal_byte_reading"]
    reading.backward()
    assert model.grounding_box.weight.grad is not None
    assert float(model.grounding_box.weight.grad.abs().sum()) > 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "body.pt"; save_checkpoint(path, model, optimizer, step=1)
        restored = NeuralDesktopBody(config); assert load_checkpoint(path, restored) == 1
        for left, right in zip(model.parameters(), restored.parameters()): assert torch.equal(left, right)
        inference = NeuralDesktopInference(path, portal=object(), device="cpu")
        inference.last_frame = before[:1]
        from worlds.desktop.inference import NeuralMotorProposal
        focus_proposal = NeuralMotorProposal(
            action="move_pointer", x=20, y=20, confidence=.5,
            action_distribution=(1.0,), target_distribution=(1.0,),
            screen_width=1920, screen_height=1080, key_code=30,
            frame_digest="focus", target_bbox=(.1, .1, .9, .9),
            trajectory=(0.0, 0.0, 0.0), read_byte_ids=(), read_confidence=.4,
            read_bbox=(.1, .1, .9, .9),
        )
        focused = inference.refocus_reading(focus_proposal, (.72, .42))
        fx0, fy0, fx1, fy1 = focused.read_bbox
        assert focused.read_focus_source == "cursor_fovea"
        assert abs((fx0 + fx1) / 2 - .72) < 1e-6
        assert abs((fy0 + fy1) / 2 - .42) < 1e-6
        assert fx1 - fx0 <= .38 + 1e-6 and fy1 - fy0 <= .22 + 1e-6
        frozen_action = inference.model.action_head.weight.detach().clone()
        positive = inference.learn_reading_feedback(
            bbox=focused.read_bbox, read_byte_ids=(ord("7") + 4, 2), accepted=True,
        )
        assert positive["reader_feedback"] == "updated_verified", positive
        assert positive["loss_after"] < positive["loss_before"]
        assert torch.equal(frozen_action, inference.model.action_head.weight)
        negative = inference.learn_reading_feedback(
            bbox=focused.read_bbox, read_byte_ids=focused.read_byte_ids, accepted=False,
        )
        assert negative["reader_feedback"] in {"updated_verified", "rolled_back_no_improvement"}
        assert torch.equal(frozen_action, inference.model.action_head.weight)
        consequence_before = inference.model.observe(before[:1]).detached()
        consequence_after = inference.model.observe(
            after[:1], consequence_before, actions[:1], parameters[:1]
        ).detached()
        consequence = inference.learn_consequence(
            consequence_before, int(actions[0]), tuple(float(v) for v in parameters[0]),
            consequence_after,
        )
        assert consequence["prediction_loss"] >= 0 and consequence["change_error"] >= 0
    assert {"target_point", "target_bbox", "foveal_byte_reading", "trajectory"} <= losses.keys()
    print({"ok": True, "parameters": model.parameter_count, "loss": float(losses["total"].detach()),
           "losses": sorted(losses)})
    return 0


if __name__ == "__main__": raise SystemExit(main())
