from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torch.nn.functional as F

from worlds.desktop.neural import NeuralDesktopBody, NeuralDesktopConfig, select_device
from worlds.desktop.training.dataset import _pillow_image, load_rgb


TRAINABLE_PREFIXES = (
    "region_queries", "region_attention.",
    "target_query.", "target_key.", "offset_head.", "grounding_box.",
)

TEXTS = (
    "HEIWA 7319", "RED DRAGON", "BLUE TIGER", "START", "OPEN", "SAVE",
    "ПРИВЕТ", "СИНИЙ ДРАКОН", "КНОПКА 8427", "TEST 5931",
)


def _font(size: int):
    Image = _pillow_image()
    del Image
    from PIL import ImageFont
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(path, size=max(10, int(size)))


def _clamp_center(center: float, extent: float) -> float:
    return min(max(center, extent / 2 + .01), 1 - extent / 2 - .01)


class ControlledGroundingDataset(Dataset[dict[str, Tensor]]):
    """Deterministic balanced scenes with generator-owned geometry labels."""

    anchors = ((.13, .13), (.50, .13), (.87, .13),
               (.13, .50), (.50, .50), (.87, .50),
               (.13, .87), (.50, .87), (.87, .87))

    def __init__(self, count: int, config: NeuralDesktopConfig, *, seed: int,
                 paired_holdout: bool = False,
                 exact_x_pair: tuple[float, float] | None = None) -> None:
        self.count = int(count); self.config = config; self.seed = int(seed)
        self.paired_holdout = bool(paired_holdout)
        self.exact_x_pair = exact_x_pair

    def __len__(self) -> int: return self.count

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        pair = index // 2 if self.paired_holdout else index
        rng = random.Random(self.seed + pair * 104729)
        width, height = self.config.image_width, self.config.image_height
        Image = _pillow_image(); from PIL import ImageDraw
        dark = rng.random() < .55
        base = rng.randint(0, 35) if dark else rng.randint(185, 245)
        tint = [min(255, max(0, base + rng.randint(-15, 15))) for _ in range(3)]
        image = Image.new("RGB", (width, height), tuple(tint)); draw = ImageDraw.Draw(image)
        # Window/panel backgrounds of different sizes and colors.
        for _ in range(rng.randint(0, 4)):
            x0 = rng.randint(0, width - 30); y0 = rng.randint(0, height - 20)
            x1 = rng.randint(x0 + 20, width); y1 = rng.randint(y0 + 15, height)
            delta = rng.randint(-35, 35)
            color = tuple(min(255, max(0, value + delta)) for value in tint)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=rng.randint(0, 14), fill=color)
        object_width = rng.uniform(.11, .38); object_height = rng.uniform(.09, .25)
        if self.paired_holdout:
            center_x = (self.exact_x_pair[index % 2] if self.exact_x_pair is not None
                        else (rng.uniform(.14, .30) if index % 2 == 0 else rng.uniform(.70, .86)))
            center_y = rng.uniform(.14, .86)
        else:
            anchor_x, anchor_y = self.anchors[index % len(self.anchors)]
            center_x = anchor_x + rng.uniform(-.075, .075)
            center_y = anchor_y + rng.uniform(-.075, .075)
        center_x = _clamp_center(center_x, object_width)
        center_y = _clamp_center(center_y, object_height)
        bbox = (center_x - object_width / 2, center_y - object_height / 2,
                center_x + object_width / 2, center_y + object_height / 2)
        px = tuple(round(value * dimension) for value, dimension in
                   zip(bbox, (width, height, width, height)))
        # Low-contrast distractors cannot be the intended high-contrast target.
        for _ in range(rng.randint(0, 5)):
            dw, dh = rng.randint(18, 100), rng.randint(10, 55)
            dx, dy = rng.randint(0, max(0, width - dw)), rng.randint(0, max(0, height - dh))
            color = tuple(min(255, max(0, value + rng.randint(-28, 28))) for value in tint)
            draw.rectangle((dx, dy, dx + dw, dy + dh), fill=color)
        target_color = tuple(rng.randint(150, 255) if dark else rng.randint(0, 95) for _ in range(3))
        text_color = (5, 8, 12) if sum(target_color) > 380 else (248, 250, 255)
        style = rng.randrange(3)
        if style == 0:
            draw.rounded_rectangle(px, radius=max(2, int((px[3] - px[1]) * .18)), fill=target_color,
                                   outline=text_color, width=max(1, height // 144))
        elif style == 1:
            draw.rectangle(px, fill=target_color)
        else:
            draw.rounded_rectangle(px, radius=max(2, int((px[3] - px[1]) * .08)),
                                   outline=target_color, width=max(3, height // 72))
        text = TEXTS[rng.randrange(len(TEXTS))]
        font = _font(min(int((px[3] - px[1]) * .52), int((px[2] - px[0]) / max(len(text) * .62, 1))))
        text_box = draw.textbbox((0, 0), text, font=font)
        tx = (px[0] + px[2] - (text_box[2] - text_box[0])) // 2
        ty = (px[1] + px[3] - (text_box[3] - text_box[1])) // 2 - text_box[1]
        draw.text((tx, ty), text, fill=text_color if style != 2 else target_color, font=font)
        raw = bytearray(image.tobytes())
        frame = torch.frombuffer(raw, dtype=torch.uint8).reshape(height, width, 3).permute(2, 0, 1).clone()
        return {"frame": frame, "bbox": torch.tensor(bbox, dtype=torch.float32),
                "pair": torch.tensor(pair), "source": torch.tensor(0)}


class RealDiagnosticDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, root: Path, config: NeuralDesktopConfig, *, repeats: int,
                 names: Iterable[str]) -> None:
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        self.examples = []
        for name in names:
            scene = summary["scenes"][name]
            frame = load_rgb(root / name / "frame.png", width=config.image_width,
                             height=config.image_height).mul(255).byte()
            self.examples.append((frame, torch.tensor(scene["controlled_bbox"], dtype=torch.float32)))
        self.repeats = int(repeats)

    def __len__(self) -> int: return len(self.examples) * self.repeats
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        frame, bbox = self.examples[index % len(self.examples)]
        return {"frame": frame, "bbox": bbox, "pair": torch.tensor(-1), "source": torch.tensor(1)}


class CachedGroundingDataset(Dataset[dict[str, Tensor]]):
    """Materialize deterministic uint8 scenes once; epochs remain geometry-identical."""
    def __init__(self, source: Dataset[dict[str, Tensor]]) -> None:
        self.items = [source[index] for index in range(len(source))]
    def __len__(self) -> int: return len(self.items)
    def __getitem__(self, index: int) -> dict[str, Tensor]: return self.items[index]


def _box_iou(predicted: Tensor, target: Tensor) -> Tensor:
    top_left = torch.maximum(predicted[:, :2], target[:, :2])
    bottom_right = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    predicted_area = (predicted[:, 2:] - predicted[:, :2]).clamp_min(0).prod(-1)
    target_area = (target[:, 2:] - target[:, :2]).clamp_min(0).prod(-1)
    return intersection / (predicted_area + target_area - intersection).clamp_min(1e-7)


def _generalized_iou_loss(predicted: Tensor, target: Tensor) -> Tensor:
    iou = _box_iou(predicted, target)
    outer_top_left = torch.minimum(predicted[:, :2], target[:, :2])
    outer_bottom_right = torch.maximum(predicted[:, 2:], target[:, 2:])
    outer_area = (outer_bottom_right - outer_top_left).clamp_min(0).prod(-1)
    predicted_area = (predicted[:, 2:] - predicted[:, :2]).clamp_min(0).prod(-1)
    target_area = (target[:, 2:] - target[:, :2]).clamp_min(0).prod(-1)
    top_left = torch.maximum(predicted[:, :2], target[:, :2])
    bottom_right = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    union = predicted_area + target_area - intersection
    giou = iou - (outer_area - union) / outer_area.clamp_min(1e-7)
    return 1 - giou


def _soft_target(model: NeuralDesktopBody, logits: Tensor, raw_offsets: Tensor) -> Tensor:
    gh, gw = model.config.image_height // 16, model.config.image_width // 16
    yy, xx = torch.meshgrid(torch.arange(gh, device=logits.device),
                            torch.arange(gw, device=logits.device), indexing="ij")
    centers = torch.stack(((xx.flatten() + .5) / gw, (yy.flatten() + .5) / gh), dim=-1)
    offsets = torch.tanh(raw_offsets) * .5
    scale = torch.tensor((gw, gh), device=logits.device)
    positions = (centers.unsqueeze(0) + offsets / scale).clamp(0, 1)
    return torch.einsum("bn,bnd->bd", logits.softmax(-1), positions)


def grounding_loss(model: NeuralDesktopBody, frame: Tensor, bbox: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    visual = model.observe(frame)
    output = model.propose(visual)
    center = (bbox[:, :2] + bbox[:, 2:]) / 2
    gh, gw = visual.grid_size
    cell = (center[:, 1].clamp(max=.999999).mul(gh).long() * gw
            + center[:, 0].clamp(max=.999999).mul(gw).long())
    soft = _soft_target(model, output.target_logits, output.target_offsets)
    spatial_ce = F.cross_entropy(output.target_logits, cell)
    center_loss = F.smooth_l1_loss(soft, center)
    hard_center_loss = F.smooth_l1_loss(output.target_point, center)
    bbox_l1 = F.smooth_l1_loss(output.target_bbox, bbox)
    giou = _generalized_iou_loss(output.target_bbox, bbox).mean()
    total = spatial_ce + 8 * center_loss + 2 * hard_center_loss + 8 * bbox_l1 + 2 * giou
    return total, {"spatial_ce": spatial_ce, "center": center_loss,
                   "hard_center": hard_center_loss, "bbox_l1": bbox_l1, "giou": giou}


def _rank(values: Tensor) -> Tensor:
    order = values.argsort(); ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(len(values), dtype=torch.float32)
    return ranks


def _correlation(left: Tensor, right: Tensor) -> float:
    left = left.float() - left.float().mean(); right = right.float() - right.float().mean()
    return float((left * right).sum() / (left.square().sum().sqrt() * right.square().sum().sqrt()).clamp_min(1e-9))


@torch.inference_mode()
def evaluate(model: NeuralDesktopBody, loader: DataLoader, *, device: torch.device,
             screen_width: int = 2560, screen_height: int = 1440) -> dict[str, float | list[float]]:
    expected, predicted, expected_boxes, predicted_boxes, pairs = [], [], [], [], []
    model.eval()
    for batch in loader:
        frame = batch["frame"].to(device, non_blocking=True).float().div_(255)
        bbox = batch["bbox"].to(device, non_blocking=True)
        output = model.propose(model.observe(frame))
        expected.append((bbox[:, :2] + bbox[:, 2:]).div(2).cpu())
        predicted.append(output.target_point.cpu()); expected_boxes.append(bbox.cpu())
        predicted_boxes.append(output.target_bbox.cpu()); pairs.append(batch["pair"].cpu())
    target = torch.cat(expected); point = torch.cat(predicted)
    target_bbox = torch.cat(expected_boxes); point_bbox = torch.cat(predicted_boxes)
    pair_ids = torch.cat(pairs)
    errors = torch.sqrt(((point[:, 0] - target[:, 0]) * screen_width).square()
                        + ((point[:, 1] - target[:, 1]) * screen_height).square())
    iou = _box_iou(point_bbox, target_bbox)
    ordering = []
    for pair in pair_ids.unique():
        if int(pair) < 0: continue
        rows = (pair_ids == pair).nonzero().flatten()
        if len(rows) == 2:
            left, right = rows[target[rows, 0].argsort()]
            ordering.append(float(point[right, 0] > point[left, 0]))
    width_error = ((point_bbox[:, 2] - point_bbox[:, 0]) - (target_bbox[:, 2] - target_bbox[:, 0])).abs()
    height_error = ((point_bbox[:, 3] - point_bbox[:, 1]) - (target_bbox[:, 3] - target_bbox[:, 1])).abs()
    quantiles = torch.quantile(iou, torch.tensor((0., .1, .25, .5, .75, .9, 1.)))
    return {
        "pearson_x": _correlation(target[:, 0], point[:, 0]),
        "spearman_x": _correlation(_rank(target[:, 0]), _rank(point[:, 0])),
        "median_center_error_px": float(errors.median()),
        "p90_center_error_px": float(torch.quantile(errors, .9)),
        "median_width_error": float(width_error.median()),
        "median_height_error": float(height_error.median()),
        "left_right_ordering_accuracy": sum(ordering) / max(len(ordering), 1),
        "iou_mean": float(iou.mean()), "iou_quantiles": [float(v) for v in quantiles],
    }


@torch.inference_mode()
def evaluate_real(model: NeuralDesktopBody, diagnostics: Path, *, device: torch.device) -> dict[str, object]:
    summary = json.loads((diagnostics / "summary.json").read_text(encoding="utf-8"))
    values = {}
    for name in ("left", "right", "heiwa", "red_dragon", "blue_tiger"):
        scene = summary["scenes"][name]
        frame = load_rgb(diagnostics / name / "frame.png", width=model.config.image_width,
                         height=model.config.image_height).unsqueeze(0).to(device)
        output = model.propose(model.observe(frame))
        target_bbox = torch.tensor([scene["controlled_bbox"]], device=device)
        target_center = (target_bbox[:, :2] + target_bbox[:, 2:]) / 2
        error = math.hypot(float((output.target_point[0, 0] - target_center[0, 0]) * 2560),
                           float((output.target_point[0, 1] - target_center[0, 1]) * 1440))
        values[name] = {"expected_center": [float(v) for v in target_center[0]],
                        "predicted_point": [float(v) for v in output.target_point[0]],
                        "predicted_bbox": [float(v) for v in output.target_bbox[0]],
                        "center_error_px": error,
                        "iou": float(_box_iou(output.target_bbox, target_bbox)[0])}
    left_x = values["left"]["predicted_point"][0]  # type: ignore[index]
    right_x = values["right"]["predicted_point"][0]  # type: ignore[index]
    return {"scenes": values, "left_right_ordered": bool(right_x > left_x),
            "predicted_x_shift": float(right_x - left_x)}


@torch.inference_mode()
def evaluate_exact_pair(model: NeuralDesktopBody, *, device: torch.device, seed: int) -> dict[str, object]:
    dataset = ControlledGroundingDataset(
        2, model.config, seed=seed, paired_holdout=True, exact_x_pair=(.2, .8)
    )
    expected, predicted, boxes = [], [], []
    for index in range(2):
        item = dataset[index]
        frame = item["frame"].unsqueeze(0).to(device).float().div_(255)
        target_bbox = item["bbox"].unsqueeze(0).to(device)
        output = model.propose(model.observe(frame))
        expected.append(float((target_bbox[0, 0] + target_bbox[0, 2]) / 2))
        predicted.append(float(output.target_point[0, 0]))
        boxes.append(float(_box_iou(output.target_bbox, target_bbox)[0]))
    return {"expected_x": expected, "predicted_x": predicted,
            "predicted_shift": predicted[1] - predicted[0],
            "ordered": predicted[1] > predicted[0], "bbox_iou": boxes}


def _state_hash(state: dict[str, Tensor], names: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode()); digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Grounding-only training; no reader/action/Core updates")
    parser.add_argument("--checkpoint", type=Path, default=Path("state/neural_desktop.pt"))
    parser.add_argument("--diagnostics", type=Path, default=Path("state/vision_diagnostics/20260829-205522"))
    parser.add_argument("--output", type=Path, default=Path("state/grounding_training"))
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--real-repeats", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7319)
    parser.add_argument("--save-on-pass", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    before_sha = _sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = NeuralDesktopConfig(**payload["config"]); device = select_device("auto")
    model = NeuralDesktopBody(config).to(device); model.load_state_dict(payload["model"], strict=False)
    if args.evaluate_only:
        validation = CachedGroundingDataset(ControlledGroundingDataset(
            args.val_samples, config, seed=args.seed + 10_000_000, paired_holdout=True
        ))
        loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)
        evaluation = evaluate(model, loader, device=device)
        real = evaluate_real(model, args.diagnostics, device=device)
        exact_pair = evaluate_exact_pair(model, device=device, seed=args.seed + 20_000_000)
        report_path = args.output / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        report["post_save_evaluation"] = evaluation
        report["test_a_exact_x_0_2_0_8"] = exact_pair
        report["post_save_real_screen"] = real
        report["checkpoint_sha_after"] = before_sha
        backup = args.checkpoint.with_name(args.checkpoint.stem + ".pre_grounding" + args.checkpoint.suffix)
        if backup.exists():
            old = torch.load(backup, map_location="cpu", weights_only=False)["model"]
            new = payload["model"]
            trained = {name for name in new if name == "region_queries" or name.startswith(TRAINABLE_PREFIXES[1:])}
            frozen = [name for name in new if name not in trained]
            tensors_equal = lambda left, right: torch.allclose(left, right, rtol=0, atol=0, equal_nan=True)
            report["serialized_frozen_state_unchanged"] = all(tensors_equal(old[name], new[name]) for name in frozen)
            report["serialized_reader_unchanged"] = all(tensors_equal(old[name], new[name]) for name in new
                                                        if name.startswith(("foveal_reader.", "byte_reader.")))
            report["serialized_action_head_unchanged"] = all(tensors_equal(old[name], new[name]) for name in new
                                                             if name.startswith("action_head."))
            report["preexisting_nan_tensors_unchanged"] = [name for name in frozen
                                                           if torch.isnan(old[name]).any()
                                                           and torch.equal(torch.isnan(old[name]), torch.isnan(new[name]))]
        args.output.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"validation": evaluation, "test_a": exact_pair, "real_screen": real,
                          "checkpoint_sha": before_sha,
                          "frozen_unchanged": report.get("serialized_frozen_state_unchanged")},
                         ensure_ascii=False), flush=True)
        return 0
    trainable_names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "region_queries" or name.startswith(TRAINABLE_PREFIXES[1:]))
        if parameter.requires_grad: trainable_names.append(name)
    frozen_names = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    original_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    frozen_before = _state_hash(original_state, [name for name in frozen_names if name in original_state])
    train_synthetic = ControlledGroundingDataset(args.train_samples, config, seed=args.seed)
    train_real = RealDiagnosticDataset(
        args.diagnostics, config, repeats=args.real_repeats,
        names=("heiwa", "red_dragon", "blue_tiger", "russian_heiwa", "russian_dragon"),
    )
    train = CachedGroundingDataset(ConcatDataset((train_synthetic, train_real)))
    validation = CachedGroundingDataset(ControlledGroundingDataset(
        args.val_samples, config, seed=args.seed + 10_000_000, paired_holdout=True
    ))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=.01)
    baseline = evaluate(model, validation_loader, device=device)
    baseline_real = evaluate_real(model, args.diagnostics, device=device)
    print(json.dumps({"baseline": baseline, "baseline_real": baseline_real}, ensure_ascii=False), flush=True)
    best_score = -float("inf"); best_state = None; best_metrics = None; history = []
    for epoch in range(1, args.epochs + 1):
        model.eval(); totals = {key: 0.0 for key in ("total", "spatial_ce", "center", "hard_center", "bbox_l1", "giou")}
        batches = 0; started = time.perf_counter()
        for batch in train_loader:
            frame = batch["frame"].to(device, non_blocking=True).float().div_(255)
            bbox = batch["bbox"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = grounding_loss(model, frame, bbox)
            loss.backward(); torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step(); batches += 1; totals["total"] += float(loss.detach())
            for key, value in parts.items(): totals[key] += float(value.detach())
        metrics = evaluate(model, validation_loader, device=device)
        score = (float(metrics["pearson_x"]) + float(metrics["spearman_x"])
                 + float(metrics["left_right_ordering_accuracy"]) + float(metrics["iou_mean"])
                 - float(metrics["median_center_error_px"]) / 2560)
        row = {"epoch": epoch, "seconds": time.perf_counter() - started,
               "losses": {key: value / batches for key, value in totals.items()}, "validation": metrics}
        history.append(row); print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score; best_metrics = metrics
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if (float(metrics["pearson_x"]) > .92 and float(metrics["spearman_x"]) > .92
                and float(metrics["median_center_error_px"]) < 100
                and float(metrics["left_right_ordering_accuracy"]) > .97
                and float(metrics["iou_mean"]) > .35):
            break
    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state); real = evaluate_real(model, args.diagnostics, device=device)
    frozen_after = _state_hash(best_state, [name for name in frozen_names if name in best_state])
    frozen_unchanged = frozen_before == frozen_after
    passed = (float(best_metrics["pearson_x"]) > .9 and float(best_metrics["spearman_x"]) > .9
              and float(best_metrics["median_center_error_px"]) < 128
              and float(best_metrics["left_right_ordering_accuracy"]) > .95
              and float(best_metrics["iou_mean"]) > .35
              and bool(real["left_right_ordered"]) and frozen_unchanged)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "passed": passed, "device": str(device), "rocm": torch.version.hip,
        "checkpoint_sha_before": before_sha, "checkpoint_sha_after": before_sha,
        "trainable_parameters": trainable_names,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameter_count": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "frozen_state_unchanged": frozen_unchanged,
        "reader_trained": False, "action_head_trained": False, "core_trained": False,
        "dataset": {"synthetic_train": len(train_synthetic), "real_train_repeated": len(train_real),
                    "real_unique_train": len(train_real.examples), "unseen_validation": len(validation),
                    "real_holdout": ["left", "right"]},
        "baseline": baseline, "baseline_real": baseline_real,
        "best_validation": best_metrics, "real_screen": real, "history": history,
    }
    if passed and args.save_on_pass:
        backup = args.checkpoint.with_name(args.checkpoint.stem + ".pre_grounding" + args.checkpoint.suffix)
        if not backup.exists(): shutil.copy2(args.checkpoint, backup)
        payload["model"] = best_state
        payload["grounding_training"] = {"report": str(args.output / "report.json"),
                                          "reader_trained": False, "action_head_trained": False}
        with tempfile.NamedTemporaryFile(dir=args.checkpoint.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try: torch.save(payload, temporary); temporary.replace(args.checkpoint)
        finally: temporary.unlink(missing_ok=True)
        report["checkpoint_sha_after"] = _sha256(args.checkpoint)
        report["checkpoint_backup"] = str(backup)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "checkpoint_sha_before", "checkpoint_sha_after",
                                                    "best_validation", "real_screen")}, ensure_ascii=False), flush=True)
    return 0 if passed else 2


if __name__ == "__main__": raise SystemExit(main())
