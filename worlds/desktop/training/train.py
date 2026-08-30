from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from worlds.desktop.neural import NeuralDesktopBody, NeuralDesktopConfig, select_device
from worlds.desktop.training.dataset import DesktopDemonstrationDataset


def save_checkpoint(path: Path, model: NeuralDesktopBody, optimizer: torch.optim.Optimizer, *, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format": 1, "config": model.config.__dict__ if hasattr(model.config, "__dict__") else {
            field: getattr(model.config, field) for field in model.config.__dataclass_fields__
        },
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
    }, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, model: NeuralDesktopBody, optimizer: torch.optim.Optimizer | None = None) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = dict(payload["model"]); fresh = model.state_dict()
    # Curriculum heads are explicitly allowed to evolve while the shared body
    # checkpoint remains loadable; shape-changed symbol tensors start fresh.
    for name, value in tuple(state.items()):
        if name.startswith("symbol_") and name in fresh and fresh[name].shape != value.shape:
            state.pop(name)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_prefixes = ("key_head.", "task_projection.", "motor_fusion.", "grounding_box.",
                        "trajectory_head.", "foveal_reader.", "byte_reader.",
                        "visual_reader_v2.", "reading_", "symbol_")
    # Reader v1 was a single Linear layer. Its two tensors are intentionally
    # ignored when initializing the isolated sequence reader.
    unexpected = set(incompatible.unexpected_keys) - {"byte_reader.weight", "byte_reader.bias"}
    missing = set(incompatible.missing_keys)
    unsupported_missing = {name for name in missing if not name.startswith(allowed_prefixes)}
    if unexpected or unsupported_missing:
        raise RuntimeError(f"incompatible checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    # An old optimizer has no slots or parameter-group entry for the new key
    # head. Preserve all learned model weights and initialize a fresh optimizer.
    if optimizer is not None and "optimizer" in payload and not missing:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Heiwa's neural Desktop vision and motor body")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("state/neural_desktop.pt"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=50, help="optimizer steps; 0 disables intra-epoch saves")
    parser.add_argument("--max-steps", type=int, help="stop cleanly after this many optimizer steps and save")
    parser.add_argument("--heads-only", action="store_true", help="train only Desktop motor/reader heads from an interactive school")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be > 0")
    torch.manual_seed(17); random.seed(17)
    config = NeuralDesktopConfig()
    dataset = DesktopDemonstrationDataset(args.manifest, config)
    train_indices, validation_indices = dataset.split_indices()
    train_subset = Subset(dataset, train_indices)
    counts: dict[str, int] = {}
    for index in train_indices:
        action = str(dataset.records[index]["action"]); counts[action] = counts.get(action, 0) + 1
    sample_weights = [1.0 / counts[str(dataset.records[index]["action"])] for index in train_indices]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_indices), replacement=True)
    loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    device = select_device(args.device)
    model = NeuralDesktopBody(config).to(device)
    head_prefixes = ("action_head.", "target_query.", "target_key.", "offset_head.",
                     "key_head.", "confidence_head.", "motor_fusion.", "grounding_box.",
                     "trajectory_head.", "foveal_reader.", "visual_reader_v2.", "reading_")
    if args.heads_only:
        for name, parameter in model.named_parameters(): parameter.requires_grad = name.startswith(head_prefixes)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)
    step = load_checkpoint(args.checkpoint, model, None if args.heads_only else optimizer) if args.resume and args.checkpoint.exists() else 0
    starting_step = step
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    stop_training = False
    for epoch in range(args.epochs):
        model.train()
        epoch_started = time.monotonic()
        for batch_index, batch in enumerate(loader):
            before, after = batch["before"].to(device), batch["after"].to(device)
            action, parameters = batch["action"].to(device), batch["parameters"].to(device)
            targets = batch["target_cell"].to(device)
            selection_feedback = batch["selection_feedback"].to(device)
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with context:
                losses = model.training_losses(
                    before, after, action, parameters, target_cell=targets,
                    success_target=selection_feedback, selection_feedback=selection_feedback,
                    target_xy=batch["target_xy"].to(device), target_bbox=batch["target_bbox"].to(device),
                    read_tokens=batch["read_tokens"].to(device),
                    trajectory_target=batch["trajectory_target"].to(device),
                    read_bbox=batch["read_bbox"].to(device),
                )
                loss = losses["total"] / max(args.accumulate, 1)
            scaler.scale(loss).backward()
            if (batch_index + 1) % max(args.accumulate, 1) == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); step += 1
                if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                    save_checkpoint(args.checkpoint, model, optimizer, step=step)
                if args.max_steps is not None and step - starting_step >= args.max_steps:
                    stop_training = True
            if args.progress_every > 0 and (batch_index + 1) % args.progress_every == 0:
                print({"epoch": epoch + 1, "batch": batch_index + 1, "batches": len(loader),
                       "step": step, "loss": round(float(losses["total"].detach()), 5),
                       "elapsed_seconds": round(time.monotonic() - epoch_started, 1)}, flush=True)
            if stop_training:
                break
        print({"epoch": epoch + 1, "step": step, **{name: round(float(value.detach()), 5) for name, value in losses.items()}})
        save_checkpoint(args.checkpoint, model, optimizer, step=step)
        if stop_training:
            break
    print({"checkpoint": str(args.checkpoint), "parameters": model.parameter_count, "device": str(device), "validation_samples": len(validation_indices)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
