from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch

from worlds.desktop.neural import ACTION_KINDS, NeuralDesktopBody, NeuralDesktopConfig, select_device
from worlds.desktop.training.dataset import DesktopDemonstrationDataset
from worlds.desktop.training.train import load_checkpoint


def _point_error(model, sample, device, config) -> float:
    frame = sample["before"].unsqueeze(0).to(device)
    with torch.inference_mode(): point = model.propose(model.observe(frame)).target_point[0].cpu()
    target = sample["target_xy"]
    delta = (point - target) * torch.tensor((config.image_width, config.image_height))
    return float(delta.square().sum().sqrt())


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded visual-control demonstration overfit/eval")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("state/neural_desktop.pt"))
    parser.add_argument("--samples", type=int, default=8); parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--action", choices=ACTION_KINDS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(); torch.manual_seed(31); random.seed(31)
    config = NeuralDesktopConfig(); dataset = DesktopDemonstrationDataset(args.manifest, config)
    indices = [i for i, record in enumerate(dataset.records)
               if record.get("target_xy") is not None
               and (args.action is None or record.get("action") == args.action)]
    if len(indices) < 2: raise SystemExit("need at least two spatial demonstrations")
    train_indices = indices[:max(1, min(args.samples, len(indices) - 1))]; unseen_index = indices[len(train_indices)]
    device = select_device(args.device); model = NeuralDesktopBody(config).to(device)
    if args.checkpoint.exists(): load_checkpoint(args.checkpoint, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
    initial = sum(_point_error(model, dataset[i], device, config) for i in train_indices) / len(train_indices)
    model.train(); last = {}
    for step in range(args.steps):
        sample = dataset[train_indices[step % len(train_indices)]]
        before = sample["before"].unsqueeze(0).to(device); after = sample["after"].unsqueeze(0).to(device)
        action = sample["action"].view(1).to(device); parameters = sample["parameters"].unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = model.training_losses(
            before, after, action, parameters, sample["target_cell"].view(1).to(device),
            target_xy=sample["target_xy"].unsqueeze(0).to(device),
            target_bbox=sample["target_bbox"].unsqueeze(0).to(device),
            read_tokens=sample["read_tokens"].unsqueeze(0).to(device),
            trajectory_target=sample["trajectory_target"].unsqueeze(0).to(device),
        )
        losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        last = {key: float(value.detach()) for key, value in losses.items()}
    model.eval()
    final = sum(_point_error(model, dataset[i], device, config) for i in train_indices) / len(train_indices)
    unseen = _point_error(model, dataset[unseen_index], device, config)
    click_indices = [i for i in train_indices if dataset.records[i]["action"] in {"left_click", "right_click", "double_click"}]
    click_ok = 0
    for i in click_indices:
        sample = dataset[i]
        with torch.inference_mode(): output = model.propose(model.observe(sample["before"].unsqueeze(0).to(device)))
        click_ok += ACTION_KINDS[int(output.action_logits.argmax(-1)[0])] == dataset.records[i]["action"]
    print({"ok": True, "samples": len(train_indices), "steps": args.steps,
           "initial_target_error_px": round(initial, 2), "final_target_error_px": round(final, 2),
           "unseen_target_error_px": round(unseen, 2),
           "click_success_rate": None if not click_indices else round(click_ok / len(click_indices), 3),
           "losses": {k: round(v, 5) for k, v in last.items()}, "device": str(device)})
    return 0


if __name__ == "__main__": raise SystemExit(main())
