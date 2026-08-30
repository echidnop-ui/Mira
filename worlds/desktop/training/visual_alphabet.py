from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import time

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from worlds.desktop.neural import NeuralDesktopBody, NeuralDesktopConfig, select_device
from worlds.desktop.training.train import load_checkpoint


SYMBOLS = ("е", "а", "о", "и", "й", "ш", "щ", "ь", "ъ")
STAGES = (
    ("е",),
    ("е", "а"),
    ("е", "а", "о"),
    ("е", "а", "о", "и", "й"),
    ("е", "а", "о", "и", "й", "ш", "щ"),
    SYMBOLS,
)
SIMILAR_PAIRS = (("е", "с"), ("и", "й"), ("ш", "щ"), ("ь", "ъ"))
FIND_DISTRACTORS = ("с", "ё", "х", "0", "+", "?", "#")
SYMBOL_PREFIX = "symbol_"
ABSENT_CLASS = len(SYMBOLS)


def _pil():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError:
        # Ubuntu installs python3-pil in dist-packages, while the project's
        # ROCm venv is intentionally isolated from system packages.
        import sys
        ubuntu_dist_packages = Path("/usr/lib/python3/dist-packages")
        if ubuntu_dist_packages.is_dir():
            sys.path.append(str(ubuntu_dist_packages))
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "RGB curriculum requires Pillow; install it with "
                ".venv-rocm/bin/python -m pip install Pillow"
            ) from error
    return Image, ImageDraw, ImageFont


def discover_fonts() -> tuple[str, ...]:
    paths: list[str] = []
    try:
        result = subprocess.run(
            ("fc-list", ":lang=ru", "-f", "%{file}\n"), check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        paths.extend(line.strip() for line in result.stdout.splitlines())
    except (OSError, subprocess.CalledProcessError):
        pass
    paths.extend((
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ))
    return tuple(dict.fromkeys(path for path in paths if path and Path(path).is_file()))


def _real_source_split(manifest: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    if not manifest.exists(): return (), ()
    payload = json.loads(manifest.read_text(encoding="utf-8")); root = manifest.parent
    train = tuple(root/path for path in payload.get("train", []))
    holdout = tuple(root/path for path in payload.get("holdout", []))
    if set(train) & set(holdout): raise ValueError("real screenshot train/holdout sources overlap")
    missing = [str(path) for path in (*train, *holdout) if not path.is_file()]
    if missing: raise FileNotFoundError(f"missing real screenshot sources: {missing}")
    return train, holdout


def _rgb_tensor(image) -> Tensor:
    raw = bytearray(image.convert("RGB").tobytes())
    return torch.frombuffer(raw, dtype=torch.uint8).reshape(image.height, image.width, 3).permute(2, 0, 1).clone()


@dataclass(frozen=True, slots=True)
class RenderedSymbol:
    pixels: Tensor
    bbox: tuple[float, float, float, float]


class AlphabetRenderer:
    """Trainer-owned labels over ordinary RGB canvases; never an inference dependency."""

    def __init__(
        self, fonts: tuple[str, ...], *, crop_size: int = 96,
        backgrounds: tuple[Path, ...] = (),
    ) -> None:
        if not fonts:
            raise RuntimeError("no system font with Cyrillic support was found")
        self.fonts = fonts
        self.crop_size = int(crop_size)
        self.backgrounds = tuple(path for path in backgrounds if path.is_file())

    def _canvas_with_source(
        self, size: tuple[int, int], rng: random.Random, background: tuple[int, int, int],
    ):
        Image, _, _ = _pil()
        if not self.backgrounds:
            return Image.new("RGB", size, background), "synthetic"
        source_path = rng.choice(self.backgrounds)
        source = Image.open(source_path).convert("RGB")
        width, height = size
        scale = max(width / source.width, height / source.height)
        resized = source.resize((max(width, round(source.width * scale)),
                                 max(height, round(source.height * scale))))
        x = rng.randint(0, max(0, resized.width - width)); y = rng.randint(0, max(0, resized.height - height))
        image = resized.crop((x, y, x + width, y + height))
        # Keep the screenshot recognizable while giving the glyph a locally
        # legible, randomized surface. The model never receives this metadata.
        tint = Image.new("RGB", size, background)
        return Image.blend(image, tint, rng.uniform(.08, .28)), str(source_path)

    def _canvas(self, size: tuple[int, int], rng: random.Random, background: tuple[int, int, int]):
        return self._canvas_with_source(size, rng, background)[0]

    @staticmethod
    def _colors(rng: random.Random) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        background = tuple(rng.randint(8, 245) for _ in range(3))
        if sum(background) / 3 > 128:
            foreground = tuple(rng.randint(0, 60) for _ in range(3))
        else:
            foreground = tuple(rng.randint(195, 255) for _ in range(3))
        return background, foreground

    @staticmethod
    def _font_style(path: str) -> str:
        name = Path(path).name.lower()
        italic = "italic" in name or "oblique" in name or "ita." in name
        thin = "thin" in name or "light" in name
        narrow = "narrow" in name or "condensed" in name
        parts = [label for label, enabled in (("thin", thin), ("italic", italic), ("narrow", narrow)) if enabled]
        return "_".join(parts) if parts else "regular"

    def symbol(self, value: str, seed: int, *, size: int | None = None) -> RenderedSymbol:
        Image, ImageDraw, ImageFont = _pil()
        rng = random.Random(int(seed)); side = self.crop_size if size is None else int(size)
        background, foreground = self._colors(rng)
        image = self._canvas((side, side), rng, background); draw = ImageDraw.Draw(image)
        # Low-amplitude RGB structure prevents a single solid background from
        # becoming a shortcut while remaining a real pixel canvas.
        for _ in range(rng.randint(0, 4)):
            shade = tuple(min(255, max(0, channel + rng.randint(-18, 18))) for channel in background)
            x0, y0 = rng.randrange(side), rng.randrange(side)
            draw.rectangle((x0, y0, min(side, x0 + rng.randint(4, side // 2)),
                            min(side, y0 + rng.randint(3, side // 3))), fill=shade)
        scale = rng.uniform(.62, 1.18)
        font_size = max(18, int(side * rng.uniform(.46, .76) * scale))
        font = ImageFont.truetype(rng.choice(self.fonts), font_size)
        box = draw.textbbox((0, 0), value, font=font, spacing=rng.randint(0, 5))
        width, height = max(1, box[2] - box[0]), max(1, box[3] - box[1])
        room_x, room_y = max(0, side - width - 4), max(0, side - height - 4)
        x = 2 + rng.randint(0, room_x) - box[0] + rng.randint(-2, 2)
        y = 2 + rng.randint(0, room_y) - box[1] + rng.randint(-2, 2)
        if self.backgrounds:
            local = image.getpixel((min(side-1, max(0, round(x))), min(side-1, max(0, round(y)))))
            opacity = rng.uniform(.58, 1.0)
            foreground = tuple(round(opacity*fg + (1-opacity)*bg) for fg, bg in zip(foreground, local, strict=True))
        draw_x, draw_y = x+rng.uniform(-.45,.45), y+rng.uniform(-.45,.45)
        draw.text((draw_x, draw_y), value, font=font,
                  fill=foreground, spacing=rng.randint(0, 5), stroke_width=rng.choice((0, 0, 0, 1)))
        actual = draw.textbbox((draw_x, draw_y), value, font=font)
        if self.backgrounds:
            from PIL import ImageFilter
            if rng.random() < .65: image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.12, .65)))
            if rng.random() < .55:
                stream = io.BytesIO(); image.save(stream, format="JPEG", quality=rng.randint(72, 97))
                stream.seek(0); image = Image.open(stream).convert("RGB")
        bbox = tuple(min(max(float(v), 0.0), float(side)) / side for v in actual)
        return RenderedSymbol(_rgb_tensor(image), bbox)  # type: ignore[arg-type]

    def blank(self, seed: int, *, size: int | None = None) -> Tensor:
        Image, ImageDraw, _ = _pil()
        rng = random.Random(int(seed)); side = self.crop_size if size is None else int(size)
        background, _ = self._colors(rng); image = self._canvas((side, side), rng, background)
        draw = ImageDraw.Draw(image)
        for _ in range(rng.randint(1, 5)):
            color = tuple(rng.randint(0, 255) for _ in range(3))
            x0, y0 = rng.randrange(side), rng.randrange(side)
            draw.line((x0, y0, rng.randrange(side), rng.randrange(side)), fill=color, width=rng.randint(1, 3))
        return _rgb_tensor(image)

    def find_canvas(
        self, target: str, active: tuple[str, ...], seed: int,
        *, width: int = 256, height: int = 160,
    ) -> tuple[Tensor, Tensor, tuple[float, float], tuple[float, float, float, float], dict[str, object]]:
        Image, ImageDraw, ImageFont = _pil()
        rng = random.Random(int(seed)); background, foreground = self._colors(rng)
        canvas, background_source = self._canvas_with_source((width, height), rng, background)
        draw = ImageDraw.Draw(canvas)
        count = rng.randint(4, 7)
        distractors = [value for value in (*active, *FIND_DISTRACTORS) if value != target]
        values = [target, *(rng.choice(distractors) for _ in range(count - 1))]
        rng.shuffle(values); target_slot = values.index(target)
        columns = 4; rows = 2; cell_w, cell_h = width / columns, height / rows
        boxes: list[tuple[float, float, float, float]] = []
        slots = rng.sample(range(columns * rows), count)
        target_font = ""; target_font_size = 0
        for index, (value, slot) in enumerate(zip(values, slots, strict=True)):
            row, col = divmod(slot, columns)
            font_path = rng.choice(self.fonts); font_size = rng.randint(30, 58)
            font = ImageFont.truetype(font_path, font_size)
            box = draw.textbbox((0, 0), value, font=font)
            glyph_w, glyph_h = box[2] - box[0], box[3] - box[1]
            left, top = col * cell_w, row * cell_h
            x = left + rng.uniform(5, max(6, cell_w - glyph_w - 4)) - box[0]
            y = top + rng.uniform(5, max(6, cell_h - glyph_h - 4)) - box[1]
            color = foreground if rng.random() < .7 else tuple(255 - v for v in foreground)
            if background_source != "synthetic":
                local = canvas.getpixel((min(width-1,max(0,round(x))), min(height-1,max(0,round(y)))))
                opacity = rng.uniform(.58, 1.0)
                color = tuple(round(opacity*fg + (1-opacity)*bg) for fg,bg in zip(color,local,strict=True))
            draw_x, draw_y = x+rng.uniform(-.45,.45), y+rng.uniform(-.45,.45)
            draw.text((draw_x, draw_y), value, font=font, fill=color,
                      stroke_width=rng.choice((0, 0, 0, 1)))
            actual = draw.textbbox((draw_x, draw_y), value, font=font)
            boxes.append(tuple(float(v) for v in actual))
            if index == target_slot:
                target_font = font_path; target_font_size = font_size
        if background_source != "synthetic":
            from PIL import ImageFilter
            if rng.random() < .65: canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(.12, .65)))
            if rng.random() < .55:
                stream = io.BytesIO(); canvas.save(stream, format="JPEG", quality=rng.randint(72, 97))
                stream.seek(0); canvas = Image.open(stream).convert("RGB")
        target_box_px = boxes[target_slot]
        x0, y0, x1, y1 = target_box_px
        target_box = (x0 / width, y0 / height, x1 / width, y1 / height)
        target_xy = ((x0 + x1) / (2 * width), (y0 + y1) / (2 * height))
        query = self.symbol(target, seed + 10_000_019).pixels
        area = max(0.0, target_box[2]-target_box[0]) * max(0.0, target_box[3]-target_box[1])
        metadata: dict[str, object] = {
            "source_seed": int(seed), "target": target, "font": target_font,
            "font_style": self._font_style(target_font),
            "font_size": target_font_size, "scale": target_font_size / 44.0,
            "position": list(target_xy), "bbox": list(target_box), "area": area,
            "size_bin": "small" if area < .012 else ("large" if area > .035 else "medium"),
            "near_edge": bool(x0/width < .08 or y0/height < .08 or x1/width > .92 or y1/height > .92),
            "background": background_source,
            "background_type": "synthetic" if background_source == "synthetic" else "real_screenshot",
            "distractor_count": count - 1,
            "distractors": [value for index, value in enumerate(values) if index != target_slot],
        }
        return query, _rgb_tensor(canvas), target_xy, target_box, metadata


class RealScreenshotPool:
    """Unlabelled RGB regions from train-only screenshot sources."""

    def __init__(self, paths: tuple[Path, ...]) -> None:
        Image, _, _ = _pil()
        self.images = tuple(Image.open(path).convert("RGB") for path in paths)

    def batch(self, seeds: list[int]) -> tuple[Tensor, Tensor, Tensor]:
        anchors: list[Tensor] = []; positives: list[Tensor] = []; negatives: list[Tensor] = []
        for seed in seeds:
            rng = random.Random(seed); image = rng.choice(self.images)
            side = rng.randint(96, min(240, image.width, image.height))
            x = rng.randint(0, image.width-side); y = rng.randint(0, image.height-side)
            anchor = _rgb_tensor(image.crop((x, y, x+side, y+side)).resize((96, 96)))
            positive = anchor.float().mul(rng.uniform(.82, 1.18)).add(rng.uniform(-15, 15)).clamp(0, 255).byte()
            mask_side = rng.randint(5, 14); mx = rng.randint(0, 96-mask_side); my = rng.randint(0, 96-mask_side)
            positive[:, my:my+mask_side, mx:mx+mask_side] = positive.float().mean((1, 2), keepdim=True).byte()
            other = rng.choice(self.images)
            other_side = rng.randint(96, min(240, other.width, other.height))
            ox = rng.randint(0, other.width-other_side); oy = rng.randint(0, other.height-other_side)
            negative = _rgb_tensor(other.crop((ox, oy, ox+other_side, oy+other_side)).resize((96, 96)))
            anchors.append(anchor); positives.append(positive); negatives.append(negative)
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


class RecognitionDataset(Dataset):
    def __init__(
        self, renderer: AlphabetRenderer, active: tuple[str, ...], count: int, seed: int,
    ) -> None:
        self.renderer = renderer; self.active = active; self.count = int(count); self.seed = int(seed)
        self.indices = {value: SYMBOLS.index(value) for value in active}

    def __len__(self) -> int: return self.count

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        symbol = self.active[index % len(self.active)]
        seed = self.seed + index * 104729
        return {
            "first": self.renderer.symbol(symbol, seed).pixels,
            "second": self.renderer.symbol(symbol, seed + 97_409).pixels,
            "label": torch.tensor(self.indices[symbol], dtype=torch.long),
        }


def _encoder_hash(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        if name.startswith(SYMBOL_PREFIX):
            continue
        digest.update(name.encode()); digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _contrastive(embeddings: Tensor, labels: Tensor) -> Tensor:
    similarity = (embeddings @ embeddings.T) / .12
    eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    same = labels[:, None].eq(labels[None, :]) & ~eye
    candidates = similarity.masked_fill(eye, -torch.inf)
    positive = similarity.masked_fill(~same, -torch.inf)
    valid = same.any(-1)
    if not bool(valid.any()): return similarity.new_zeros(())
    return -(torch.logsumexp(positive[valid], -1) - torch.logsumexp(candidates[valid], -1)).mean()


def _encode(model: NeuralDesktopBody, images: Tensor) -> tuple[Tensor, Tensor]:
    visual = model.observe(images)
    return model.visual_symbol(visual)


def _real_region_loss(
    model: NeuralDesktopBody, pool: RealScreenshotPool, seeds: list[int], device: torch.device,
) -> Tensor:
    anchor, positive, negative = pool.batch(seeds)
    joined = torch.cat((anchor, positive, negative)).to(device).float().div(255)
    embedding, _ = _encode(model, joined); count = len(seeds)
    first, second, other = embedding[:count], embedding[count:2*count], embedding[2*count:]
    positive_similarity = (first * second).sum(-1)
    negative_similarity = (first * other).sum(-1)
    return (1.0 - positive_similarity).mean() + F.relu(negative_similarity-positive_similarity+.20).mean()


def _find_batch(
    model: NeuralDesktopBody, renderer: AlphabetRenderer, active: tuple[str, ...],
    seeds: list[int], device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    rows = [renderer.find_canvas(active[index % len(active)], active, seed)
            for index, seed in enumerate(seeds)]
    queries = torch.stack([row[0] for row in rows]).to(device).float().div(255)
    canvases = torch.stack([row[1] for row in rows]).to(device).float().div(255)
    targets = torch.tensor([row[2] for row in rows], device=device)
    boxes = torch.tensor([row[3] for row in rows], device=device)
    query_embeddings, _ = _encode(model, queries)
    canvas_visual = model.observe(canvases)
    logits, points = model.find_visual_symbol(canvas_visual, query_embeddings)
    gh, gw = canvas_visual.grid_size
    yy, xx = torch.meshgrid(
        (torch.arange(gh, device=device) + .5) / gh,
        (torch.arange(gw, device=device) + .5) / gw, indexing="ij",
    )
    locations = torch.stack((xx.flatten(), yy.flatten()), -1).unsqueeze(0)
    positive = ((locations[..., 0] >= boxes[:, None, 0]) & (locations[..., 0] <= boxes[:, None, 2])
                & (locations[..., 1] >= boxes[:, None, 1]) & (locations[..., 1] <= boxes[:, None, 3]))
    # Very small glyphs can sit between cell centers; include their nearest cell.
    nearest = (targets[:, 1].mul(gh).long().clamp(max=gh - 1) * gw
               + targets[:, 0].mul(gw).long().clamp(max=gw - 1))
    positive[torch.arange(len(rows), device=device), nearest] = True
    log_prob = logits.float().log_softmax(-1)
    region_probability_loss = -torch.logsumexp(log_prob.masked_fill(~positive, -torch.inf), dim=-1).mean()
    return region_probability_loss, F.smooth_l1_loss(points, targets), points


def _load_failure_conditions(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.exists(): return []
    payload = json.loads(path.read_text(encoding="utf-8")); result: list[dict[str, object]] = []
    for sample in payload.get("samples", []):
        metadata = sample.get("metadata", {})
        if metadata.get("background_type") != "synthetic": continue
        result.append({"parent_source_seed": metadata.get("source_seed"),
                       "target": metadata.get("target"),
                       "size_bin": metadata.get("size_bin"), "near_edge": metadata.get("near_edge"),
                       "font_style": metadata.get("font_style") or AlphabetRenderer._font_style(str(metadata.get("font", ""))),
                       "distractor_count": metadata.get("distractor_count")})
    return result


def _mine_independent_seeds(
    renderer: AlphabetRenderer, active: tuple[str, ...], seeds: list[int],
    conditions: list[dict[str, object]], fraction: float,
) -> tuple[list[int], int]:
    if not conditions or fraction <= 0: return seeds, 0
    result = list(seeds); wanted_count = min(len(seeds), max(1, round(len(seeds)*fraction))); matched = 0
    for index in range(wanted_count):
        target = active[index % len(active)]
        matching_target = [condition for condition in conditions if condition.get("target") == target]
        condition_pool = matching_target or conditions
        condition = condition_pool[index % len(condition_pool)]
        # Validation source seeds are never reused. Search only new training
        # seeds for the same renderer condition, capped to prevent tail takeover.
        for attempt in range(32):
            candidate = seeds[index] + 31_337 + attempt * 104_729
            metadata = renderer.find_canvas(target, active, candidate)[4]
            if (metadata.get("size_bin") == condition.get("size_bin")
                    and metadata.get("near_edge") == condition.get("near_edge")
                    and metadata.get("font_style") == condition.get("font_style")
                    and metadata.get("distractor_count") == condition.get("distractor_count")):
                result[index] = candidate; matched += 1; break
    return result, matched


def _sanity_gate(
    model: NeuralDesktopBody, renderer: AlphabetRenderer, optimizer: torch.optim.Optimizer,
    trainable: list[Tensor], *, device: torch.device, output: Path, seed: int,
) -> dict[str, object]:
    """Prove gradients, parameter updates and save/reload before curriculum."""
    original_model = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    model.train()
    positives = torch.stack([renderer.symbol("е", seed + index * 37).pixels for index in range(12)])
    negatives = torch.stack([renderer.blank(seed + 1_000_000 + index * 37) for index in range(12)])
    images = torch.cat((positives, negatives)).to(device).float().div(255)
    labels = torch.tensor([SYMBOLS.index("е")] * 12 + [ABSENT_CLASS] * 12, device=device)
    with torch.no_grad():
        before_logits = _encode(model, images)[1]
        loss_before = float(F.cross_entropy(before_logits[:, :ABSENT_CLASS + 1].float(), labels))
        probability_before = before_logits[:, :ABSENT_CLASS + 1].softmax(-1)[:, 0].detach().cpu()
    snapshots = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
    gradient_tensors = 0; finite_gradients = True
    for _ in range(192):
        _, logits = _encode(model, images)
        loss = F.cross_entropy(logits[:, :ABSENT_CLASS + 1].float(), labels)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        current = [parameter.grad for parameter in trainable if parameter.grad is not None]
        gradient_tensors = max(gradient_tensors, len(current))
        finite_gradients = finite_gradients and all(bool(torch.isfinite(value).all()) for value in current)
        torch.nn.utils.clip_grad_norm_(trainable, 2.0); optimizer.step()
    model.eval()
    with torch.no_grad():
        after_logits = _encode(model, images)[1]
        loss_after = float(F.cross_entropy(after_logits[:, :ABSENT_CLASS + 1].float(), labels))
        probability_after = after_logits[:, :ABSENT_CLASS + 1].softmax(-1)[:, 0].detach().cpu()
        accuracy_after = float(after_logits[:, :ABSENT_CLASS + 1].argmax(-1).eq(labels).float().mean())
    changed = [name for name, value in model.named_parameters()
               if value.requires_grad and not torch.equal(snapshots[name], value.detach().cpu())]
    finite_weights = all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "sanity.pt"
    with tempfile.NamedTemporaryFile(dir=output, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save({"config": {name: getattr(model.config, name) for name in model.config.__dataclass_fields__},
                    "model": model.state_dict()}, temporary)
        temporary.replace(checkpoint)
    finally:
        temporary.unlink(missing_ok=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = NeuralDesktopBody(NeuralDesktopConfig(**payload["config"])).to(device)
    restored.load_state_dict(payload["model"]); restored.eval()
    with torch.no_grad(): reload_logits = _encode(restored, images)[1]
    reload_equal = bool(torch.equal(after_logits.detach().cpu(), reload_logits.detach().cpu()))
    prediction_change = float((probability_after - probability_before).abs().mean())
    passed = (loss_after < loss_before * .55 and accuracy_after >= .95
              and gradient_tensors > 0 and finite_gradients and bool(changed)
              and finite_weights and (prediction_change > .05 or loss_before < .50)
              and reload_equal)
    report = {
        "passed": passed, "loss_before": loss_before, "loss_after": loss_after,
        "accuracy_after": accuracy_after,
        "gradient_tensors": gradient_tensors, "finite_gradients": finite_gradients,
        "changed_parameter_tensors": len(changed), "prediction_mean_abs_change": prediction_change,
        "finite_weights": finite_weights, "reload_exact": reload_equal, "checkpoint": str(checkpoint),
    }
    # This is a diagnostic transaction, not curriculum data. Repeated resumes
    # must not bias the shared representation toward the single probe glyph.
    model.load_state_dict(original_model)
    optimizer.load_state_dict(original_optimizer)
    report["state_restored_exact"] = all(
        torch.equal(original_model[name], value.detach().cpu())
        for name, value in model.state_dict().items()
    )
    passed = passed and bool(report["state_restored_exact"])
    report["passed"] = passed
    if not passed:
        raise RuntimeError(f"visual alphabet sanity gate failed: {report}")
    return report


def _localization_gradient_probe(
    model: NeuralDesktopBody, renderer: AlphabetRenderer, *, device: torch.device, seed: int,
) -> dict[str, object]:
    model.train(); model.zero_grad(set_to_none=True)
    region_loss, point_loss, _ = _find_batch(model, renderer, ("е", "а", "о"),
                                              [seed + index * 101 for index in range(8)], device)
    (region_loss + 4.0 * point_loss).backward()
    shared = [(name, parameter.grad) for name, parameter in model.named_parameters()
              if name.startswith(("stem.", "spatial.", "position_projection."))]
    nonzero = [(name, grad) for name, grad in shared
               if grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0]
    first = renderer.symbol("е", seed + 91).pixels.unsqueeze(0).to(device).float().div(255)
    second = renderer.symbol("о", seed + 193).pixels.unsqueeze(0).to(device).float().div(255)
    with torch.no_grad():
        first_visual = model.observe(first); second_visual = model.observe(second)
        patch_rgb_delta = float((first_visual.patch_tokens - second_visual.patch_tokens).abs().mean())
        query, _ = model.visual_symbol(first_visual)
        key_dimension = int(model.symbol_find_key(first_visual.patch_tokens).shape[-1])
    model.zero_grad(set_to_none=True)
    return {
        "passed": bool(nonzero) and patch_rgb_delta > 1e-5 and int(query.shape[-1]) == key_dimension,
        "shared_visual_gradient_tensors": len(nonzero),
        "shared_visual_gradient_l1": sum(float(grad.abs().sum()) for _, grad in nonzero),
        "patch_embedding_rgb_delta": patch_rgb_delta,
        "query_dimension": int(query.shape[-1]), "patch_key_dimension": key_dimension,
        "space": "shared_cosine",
    }


def _tail_modes(record: dict[str, object]) -> list[str]:
    modes: list[str] = []
    if bool(record["target_in_top5"]) and float(record["error"]) >= .20: modes.append("A_topk_but_point_far")
    if not bool(record["target_in_top5"]): modes.append("B_target_absent_topk")
    if bool(record["multi_modal"]): modes.append("C_multi_modal")
    if float(record["entropy"]) >= .72: modes.append("D_high_entropy")
    if float(record["confidence"]) >= .70 and not bool(record["top1_correct"]): modes.append("E_confident_miss")
    metadata = record["metadata"]
    if isinstance(metadata, dict):
        if metadata.get("size_bin") == "small": modes.append("F_small_glyph")
        if bool(metadata.get("near_edge")): modes.append("G_edge_crop")
    return modes


def _write_tail_diagnostics(records: list[dict[str, object]], output: Path) -> dict[str, object]:
    Image, ImageDraw, ImageFont = _pil(); output.mkdir(parents=True, exist_ok=True)
    worst = sorted(records, key=lambda row: float(row["error"]), reverse=True)[:max(1, math.ceil(len(records)*.10))]
    mode_counts: Counter[str] = Counter(); condition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    fields = ("size_bin", "near_edge", "font", "font_style", "background_type", "distractor_count")
    all_conditions: dict[str, Counter[str]] = defaultdict(Counter)
    tail_conditions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        metadata = dict(row["metadata"])
        metadata["font"] = Path(str(metadata.get("font", ""))).name
        for field in fields: all_conditions[field][str(metadata.get(field))] += 1
    for row in worst:
        metadata = dict(row["metadata"]); metadata["font"] = Path(str(metadata.get("font", ""))).name
        for field in fields: tail_conditions[field][str(metadata.get(field))] += 1
    overrepresented = {
        (field, value) for field in fields for value, count in tail_conditions[field].items()
        if count >= 2 and (count / len(worst)) >= 1.5 * (all_conditions[field][value] / len(records))
    }
    manifest: list[dict[str, object]] = []
    for rank, row in enumerate(worst, 1):
        modes = _tail_modes(row)
        metadata = dict(row["metadata"]); metadata["font"] = Path(str(metadata.get("font", ""))).name
        if any((field, str(metadata.get(field))) in overrepresented for field in fields):
            modes.append("H_render_condition")
        mode_counts.update(modes)
        for field in fields:
            condition_counts[field][str(metadata.get(field))] += 1
        canvas = row.pop("canvas"); probability = row.pop("probability_map")
        assert isinstance(canvas, Tensor) and isinstance(probability, Tensor)
        image = Image.fromarray(canvas.permute(1, 2, 0).numpy()).convert("RGB")
        heat_values = probability.div(probability.max().clamp_min(1e-9)).mul(255).byte().numpy()
        heat = Image.fromarray(heat_values, mode="L").resize(image.size)
        red = Image.new("RGB", image.size, (255, 0, 30)); overlay = Image.composite(red, image, heat)
        image = Image.blend(image, overlay, .38); draw = ImageDraw.Draw(image); width, height = image.size
        x0, y0, x1, y1 = row["target_bbox"]
        draw.rectangle((x0*width, y0*height, x1*width, y1*height), outline=(20,255,80), width=3)
        colors = ((255,0,255), (0,220,255), (255,180,0))
        for index, point in enumerate(row["top3_points"]):
            px, py = point; radius = 6 if index == 0 else 4
            draw.ellipse((px*width-radius, py*height-radius, px*width+radius, py*height+radius),
                         outline=colors[index], width=3)
        ex, ey = row["expected_point"]
        draw.line((ex*width-7, ey*height, ex*width+7, ey*height), fill=(255,255,0), width=2)
        draw.line((ex*width, ey*height-7, ex*width, ey*height+7), fill=(255,255,0), width=2)
        filename = f"{rank:02d}-seed-{metadata['source_seed']}"
        image.save(output / f"{filename}.png")
        probability.mul(65535).clamp(0,65535).to(torch.uint16).numpy().tofile(output / f"{filename}.probability.u16")
        serializable = {key: value for key, value in row.items() if key not in ("canvas", "probability_map")}
        serializable.update({"metadata": metadata, "failure_modes": modes,
                             "rgb": f"{filename}.png", "probability_map": f"{filename}.probability.u16",
                             "probability_shape": list(probability.shape)})
        manifest.append(serializable)
    summary = {"count": len(records), "tail_count": len(worst), "failure_modes": dict(mode_counts),
               "conditions": {name: dict(values) for name, values in condition_counts.items()},
               "overrepresented_conditions": sorted([list(value) for value in overrepresented]),
               "samples": manifest}
    (output/"tail_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return summary


@torch.no_grad()
def pure_real_consistency(
    model: NeuralDesktopBody, sources: tuple[Path, ...], *, device: torch.device,
    seed: int, samples_per_source: int = 16,
) -> dict[str, object] | None:
    if not sources: return None
    Image, _, _ = _pil(); rng = random.Random(seed)
    positives: list[Tensor] = []; variants: list[Tensor] = []; negatives: list[Tensor] = []
    loaded = [Image.open(path).convert("RGB") for path in sources]
    for source_index, image in enumerate(loaded):
        for _ in range(samples_per_source):
            side = rng.randint(96, min(240, image.width, image.height))
            x = rng.randint(0, image.width-side); y = rng.randint(0, image.height-side)
            crop = _rgb_tensor(image.crop((x,y,x+side,y+side)).resize((96,96)))
            variant = crop.float().mul(rng.uniform(.82,1.18)).add(rng.uniform(-15,15)).clamp(0,255).byte()
            mask_side = rng.randint(5,14); mx, my = rng.randint(0,96-mask_side), rng.randint(0,96-mask_side)
            variant[:,my:my+mask_side,mx:mx+mask_side] = variant.float().mean((1,2),keepdim=True).byte()
            other = loaded[(source_index+1) % len(loaded)] if len(loaded)>1 else image
            oside = rng.randint(96, min(240, other.width, other.height))
            ox, oy = rng.randint(0,other.width-oside), rng.randint(0,other.height-oside)
            negative = _rgb_tensor(other.crop((ox,oy,ox+oside,oy+oside)).resize((96,96)))
            positives.append(crop); variants.append(variant); negatives.append(negative)
    first, first_logits = _encode(model, torch.stack(positives).to(device).float().div(255))
    second, second_logits = _encode(model, torch.stack(variants).to(device).float().div(255))
    other, _ = _encode(model, torch.stack(negatives).to(device).float().div(255))
    positive_similarity = (first*second).sum(-1); negative_similarity = (first*other).sum(-1)
    first_probability = first_logits[:, :len(SYMBOLS)].softmax(-1)
    second_probability = second_logits[:, :len(SYMBOLS)].softmax(-1)
    entropy = -(first_probability*first_probability.clamp_min(1e-12).log()).sum(-1) / math.log(len(SYMBOLS))
    return {"sources": [str(path) for path in sources], "samples": len(positives),
            "augmentation_consistency": float(positive_similarity.mean()),
            "different_region_similarity": float(negative_similarity.mean()),
            "separation_margin": float((positive_similarity-negative_similarity).mean()),
            "pairwise_retrieval_accuracy": float(positive_similarity.gt(negative_similarity).float().mean()),
            "prediction_stability": float(first_probability.argmax(-1).eq(second_probability.argmax(-1)).float().mean()),
            "mean_prediction_entropy": float(entropy.mean()),
            "mean_prediction_confidence": float(first_probability.max(-1).values.mean())}


@torch.no_grad()
def ui_control_benchmark(model: NeuralDesktopBody, *, device: torch.device) -> dict[str, object]:
    from torch.utils.data import DataLoader
    from worlds.desktop.training.train_grounding import (
        CachedGroundingDataset, ControlledGroundingDataset, evaluate as evaluate_grounding,
    )
    dataset = CachedGroundingDataset(ControlledGroundingDataset(
        90, model.config, seed=711931, paired_holdout=True,
    ))
    return dict(evaluate_grounding(model, DataLoader(dataset, batch_size=10), device=device))


def evaluate_real_sources(
    model: NeuralDesktopBody, fonts: tuple[str, ...], sources: tuple[Path, ...],
    active: tuple[str, ...], *, seed: int, device: torch.device, batch_size: int,
    samples: int = 120, find_samples: int = 45,
) -> dict[str, object]:
    return {
        path.name: evaluate(
            model, AlphabetRenderer(fonts, backgrounds=(path,)), active,
            samples=samples, find_samples=find_samples, seed=seed+index*104_729,
            device=device, batch_size=batch_size,
        )
        for index, path in enumerate(sources)
    }


@torch.no_grad()
def evaluate(
    model: NeuralDesktopBody, renderer: AlphabetRenderer, active: tuple[str, ...],
    *, samples: int, find_samples: int, seed: int, device: torch.device, batch_size: int,
    tail_dir: Path | None = None,
) -> dict[str, object]:
    model.eval(); dataset = RecognitionDataset(renderer, active, samples, seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    n = len(active); confusion = torch.zeros(n, n, dtype=torch.long)
    all_embeddings: list[Tensor] = []; all_labels: list[Tensor] = []
    same_correct = different_correct = same_total = different_total = 0
    for batch in loader:
        first = batch["first"].to(device).float().div(255)
        second = batch["second"].to(device).float().div(255)
        labels = batch["label"].to(device)
        first_e, logits = _encode(model, first); second_e, _ = _encode(model, second)
        predicted = logits[:, :len(SYMBOLS)].argmax(-1)
        for expected, actual in zip(labels.cpu(), predicted.cpu(), strict=True):
            if int(expected) < n and int(actual) < n: confusion[int(expected), int(actual)] += 1
        positive = (first_e * second_e).sum(-1)
        same_correct += int(positive.ge(.55).sum()); same_total += len(positive)
        if n > 1:
            negative_indices = []
            for i, label in enumerate(labels.tolist()):
                negative_indices.append(next(j for j, other in enumerate(labels.tolist()) if other != label))
            negative = (first_e * second_e[torch.tensor(negative_indices, device=device)]).sum(-1)
            different_correct += int(negative.lt(.55).sum()); different_total += len(negative)
        all_embeddings.append(first_e.cpu()); all_labels.append(labels.cpu())
    per_symbol = {}
    for index, symbol in enumerate(active):
        total = int(confusion[index].sum()); per_symbol[symbol] = float(confusion[index, index] / max(total, 1))
    total = int(confusion.sum()); accuracy = float(confusion.diag().sum() / max(total, 1))
    embeddings, labels = torch.cat(all_embeddings), torch.cat(all_labels)
    prototypes = torch.stack([embeddings[labels.eq(SYMBOLS.index(symbol))].mean(0) for symbol in active])
    prototypes = F.normalize(prototypes, dim=-1); prototype_similarity = prototypes @ prototypes.T
    nearest = {}
    for index, symbol in enumerate(active):
        candidates = [(float(prototype_similarity[index, j]), active[j]) for j in range(n) if j != index]
        nearest[symbol] = sorted(candidates, reverse=True)[:3]
    match_accuracy = (same_correct + different_correct) / max(same_total + different_total, 1)
    find_hits = 0; find_cell_hits = 0; find_top3 = 0; find_top5 = 0
    find_target_mass: list[float] = []; find_errors: list[float] = []
    find_entropies: list[float] = []; correct_confidences: list[float] = []
    catastrophic_confidences: list[float] = []; find_records: list[dict[str, object]] = []
    spatial_by_symbol: dict[str, dict[str, object]] = {
        symbol: {"count": 0, "top1": 0, "top3": 0, "top5": 0, "errors": [], "mass": []}
        for symbol in active
    }
    for start in range(0, find_samples, batch_size):
        count = min(batch_size, find_samples - start)
        rows = [renderer.find_canvas(active[(start+i) % n], active, seed + 50_000_000 + (start+i)*7919)
                for i in range(count)]
        queries = torch.stack([row[0] for row in rows]).to(device).float().div(255)
        canvases = torch.stack([row[1] for row in rows]).to(device).float().div(255)
        target = torch.tensor([row[2] for row in rows], device=device)
        boxes = torch.tensor([row[3] for row in rows], device=device)
        query_e, _ = _encode(model, queries); canvas_v = model.observe(canvases)
        logits, points = model.find_visual_symbol(canvas_v, query_e)
        gh, gw = canvas_v.grid_size
        expected = target[:, 1].mul(gh).long().clamp(max=gh-1)*gw + target[:, 0].mul(gw).long().clamp(max=gw-1)
        probabilities = logits.softmax(-1); top_values, top = probabilities.topk(min(5, logits.shape[-1]), dim=-1)
        top1_correct = logits.argmax(-1).eq(expected)
        find_cell_hits += int(top1_correct.sum())
        find_top3 += int(top[:, :min(3, top.shape[1])].eq(expected[:, None]).any(-1).sum())
        find_top5 += int(top.eq(expected[:, None]).any(-1).sum())
        yy, xx = torch.meshgrid(
            (torch.arange(gh, device=device) + .5) / gh,
            (torch.arange(gw, device=device) + .5) / gw, indexing="ij",
        )
        locations = torch.stack((xx.flatten(), yy.flatten()), -1).unsqueeze(0)
        region = ((locations[..., 0] >= boxes[:, None, 0]) & (locations[..., 0] <= boxes[:, None, 2])
                  & (locations[..., 1] >= boxes[:, None, 1]) & (locations[..., 1] <= boxes[:, None, 3]))
        region[torch.arange(count, device=device), expected] = True
        find_target_mass.extend((logits.softmax(-1) * region).sum(-1).cpu().tolist())
        inside = ((points[:, 0] >= boxes[:, 0]) & (points[:, 0] <= boxes[:, 2])
                  & (points[:, 1] >= boxes[:, 1]) & (points[:, 1] <= boxes[:, 3]))
        find_hits += int(inside.sum())
        batch_errors = torch.linalg.vector_norm(points-target, dim=-1)
        find_errors.extend(batch_errors.cpu().tolist())
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1) / math.log(probabilities.shape[-1])
        find_entropies.extend(entropy.cpu().tolist())
        correct_confidences.extend(top_values[top1_correct, 0].cpu().tolist())
        catastrophic = batch_errors.ge(.20)
        catastrophic_confidences.extend(top_values[catastrophic, 0].cpu().tolist())
        for index, row in enumerate(rows):
            symbol = str(row[4]["target"]); symbol_stats = spatial_by_symbol[symbol]
            symbol_stats["count"] = int(symbol_stats["count"]) + 1
            symbol_stats["top1"] = int(symbol_stats["top1"]) + int(bool(top1_correct[index]))
            symbol_stats["top3"] = int(symbol_stats["top3"]) + int(bool(top[index,:3].eq(expected[index]).any()))
            symbol_stats["top5"] = int(symbol_stats["top5"]) + int(bool(top[index].eq(expected[index]).any()))
            symbol_stats["errors"].append(float(batch_errors[index]))
            symbol_stats["mass"].append(float((probabilities[index] * region[index]).sum()))
            top_indices = top[index, :3]
            top_points = [[float((int(cell) % gw + .5) / gw), float((int(cell) // gw + .5) / gh)]
                          for cell in top_indices.cpu().tolist()]
            cell_x = (torch.arange(gw, device=device, dtype=probabilities.dtype) + .5) / gw
            cell_y = (torch.arange(gh, device=device, dtype=probabilities.dtype) + .5) / gh
            yy_expected, xx_expected = torch.meshgrid(cell_y, cell_x, indexing="ij")
            expected_point = torch.stack((xx_expected.flatten(), yy_expected.flatten()), -1)
            expected_point = (probabilities[index, :, None] * expected_point).sum(0)
            first_point, second_point = top_points[0], top_points[min(1, len(top_points)-1)]
            peak_distance = math.dist(first_point, second_point)
            record: dict[str, object] = {
                "metadata": dict(row[4]), "target_bbox": list(row[3]), "target_xy": list(row[2]),
                "top1_cell": int(top[index,0]), "top1_correct": bool(top1_correct[index]),
                "target_in_top3": bool(top[index,:3].eq(expected[index]).any()),
                "target_in_top5": bool(top[index].eq(expected[index]).any()),
                "top3_cells": top_indices.cpu().tolist(), "top3_points": top_points,
                "top3_probabilities": top_values[index,:3].cpu().tolist(),
                "decoded_point": points[index].cpu().tolist(), "expected_point": expected_point.cpu().tolist(),
                "error": float(batch_errors[index]), "entropy": float(entropy[index]),
                "confidence": float(top_values[index,0]),
                "target_mass": float((probabilities[index] * region[index]).sum()),
                "multi_modal": bool(float(top_values[index,1] / top_values[index,0].clamp_min(1e-9)) > .70
                                    and peak_distance > .18),
            }
            if tail_dir is not None:
                record["canvas"] = row[1].clone(); record["probability_map"] = probabilities[index].reshape(gh,gw).cpu()
            find_records.append(record)
    errors = sorted(find_errors)
    tail_summary = _write_tail_diagnostics(find_records, tail_dir) if tail_dir is not None else None
    def condition_slice(rows: list[dict[str, object]]) -> dict[str, object]:
        slice_errors = sorted(float(row["error"]) for row in rows)
        count = max(len(rows), 1)
        return {
            "samples": len(rows),
            "top1": sum(bool(row["top1_correct"]) for row in rows) / count,
            "top3": sum(bool(row["target_in_top3"]) for row in rows) / count,
            "top5": sum(bool(row["target_in_top5"]) for row in rows) / count,
            "median_error": slice_errors[len(slice_errors)//2] if slice_errors else None,
            "p90_error": slice_errors[min(len(slice_errors)-1, int(len(slice_errors)*.90))] if slice_errors else None,
            "target_mass": sum(float(row["target_mass"]) for row in rows) / count,
        }
    clean_records: list[dict[str, object]] = []
    hard_records: list[dict[str, object]] = []
    for row in find_records:
        metadata = dict(row["metadata"])
        hard = (metadata.get("size_bin") == "small" or bool(metadata.get("near_edge"))
                or metadata.get("font_style") != "regular"
                or int(metadata.get("distractor_count", 0)) >= 6)
        (hard_records if hard else clean_records).append(row)
    localization_per_symbol: dict[str, object] = {}
    for symbol, values in spatial_by_symbol.items():
        symbol_errors = sorted(values["errors"]); count = max(int(values["count"]), 1)
        localization_per_symbol[symbol] = {
            "count": int(values["count"]), "top1": int(values["top1"])/count,
            "top3": int(values["top3"])/count, "top5": int(values["top5"])/count,
            "median_error": symbol_errors[len(symbol_errors)//2] if symbol_errors else None,
            "p90_error": symbol_errors[min(len(symbol_errors)-1, int(len(symbol_errors)*.90))] if symbol_errors else None,
            "target_mass": sum(values["mass"])/max(len(values["mass"]), 1),
        }
    similar_errors = {}
    for left, right in SIMILAR_PAIRS:
        if left in active and right in active:
            li, ri = active.index(left), active.index(right)
            similar_errors[f"{left}/{right}"] = {
                "left_as_right": int(confusion[li, ri]), "right_as_left": int(confusion[ri, li]),
            }
    return {
        "samples": total, "accuracy": accuracy, "per_symbol": per_symbol,
        "confusion_symbols": list(active), "confusion_matrix": confusion.tolist(),
        "match_accuracy": match_accuracy, "nearest_embeddings": nearest,
        "similar_symbol_errors": similar_errors,
        "find_accuracy": find_hits / max(find_samples, 1),
        "find_center_cell_accuracy": find_cell_hits / max(find_samples, 1),
        "find_top3_recall": find_top3 / max(find_samples, 1),
        "find_top5_recall": find_top5 / max(find_samples, 1),
        "find_target_region_probability": sum(find_target_mass) / max(len(find_target_mass), 1),
        "find_median_error": errors[len(errors)//2] if errors else None,
        "find_p90_error": errors[min(len(errors)-1, int(len(errors)*.90))] if errors else None,
        "find_p95_error": errors[min(len(errors)-1, int(len(errors)*.95))] if errors else None,
        "find_mean_entropy": sum(find_entropies) / max(len(find_entropies), 1),
        "find_confidence_correct": sum(correct_confidences) / max(len(correct_confidences), 1),
        "find_confidence_catastrophic": sum(catastrophic_confidences) / max(len(catastrophic_confidences), 1),
        "tail_failure_modes": None if tail_summary is None else tail_summary["failure_modes"],
        "localization_conditions": {
            "clean": condition_slice(clean_records), "hard": condition_slice(hard_records),
        },
        "localization_per_symbol": localization_per_symbol,
    }


@torch.no_grad()
def diagnostic(
    model: NeuralDesktopBody, renderer: AlphabetRenderer, active: tuple[str, ...],
    output: Path, device: torch.device, seed: int,
) -> dict[str, object]:
    Image, ImageDraw, ImageFont = _pil(); target = active[-1]
    query, canvas, target_xy, target_box, _ = renderer.find_canvas(target, active, seed)
    query_e, query_logits = _encode(model, query.unsqueeze(0).to(device).float().div(255))
    canvas_v = model.observe(canvas.unsqueeze(0).to(device).float().div(255))
    spatial, point = model.find_visual_symbol(canvas_v, query_e)
    predicted_index = int(query_logits[0, :len(SYMBOLS)].argmax())
    predicted_symbol = SYMBOLS[predicted_index] if predicted_index < len(SYMBOLS) else "?"
    px, py = (float(v) for v in point[0].cpu())
    canvas_image = Image.fromarray(canvas.permute(1, 2, 0).numpy()).resize((768, 480))
    draw = ImageDraw.Draw(canvas_image); w, h = canvas_image.size
    x0, y0, x1, y1 = target_box
    draw.rectangle((x0*w, y0*h, x1*w, y1*h), outline=(20, 255, 80), width=5)
    draw.ellipse((px*w-9, py*h-9, px*w+9, py*h+9), outline=(255, 20, 240), width=5)
    font = ImageFont.truetype(renderer.fonts[0], 28)
    draw.text((12, 10), f"GT {target}  PRED {predicted_symbol}", font=font, fill=(255, 230, 20),
              stroke_width=2, stroke_fill=(0, 0, 0))
    query_image = Image.fromarray(query.permute(1, 2, 0).numpy()).resize((240, 240))
    combined = Image.new("RGB", (1008, 480), (25, 25, 25)); combined.paste(query_image, (0, 120)); combined.paste(canvas_image, (240, 0))
    output.parent.mkdir(parents=True, exist_ok=True); combined.save(output)
    return {"ground_truth": target, "prediction": predicted_symbol,
            "target_xy": target_xy, "predicted_xy": [px, py], "image": str(output)}


def _save(path: Path, model: NeuralDesktopBody, optimizer, report: dict[str, object], active: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({
        "format": "heiwa_visual_alphabet_v1",
        "config": {name: getattr(model.config, name) for name in model.config.__dataclass_fields__},
        "symbols": list(SYMBOLS), "active_symbols": list(active),
        "encoder_sha256": _encoder_hash(state), "model": state,
        "symbol_state": {name: value for name, value in state.items() if name.startswith(SYMBOL_PREFIX)},
        "optimizer": optimizer.state_dict(), "report": report,
    }, temporary); temporary.replace(path)


@torch.no_grad()
def _reload_probe(
    path: Path, model: NeuralDesktopBody, renderer: AlphabetRenderer,
    active: tuple[str, ...], device: torch.device, seed: int,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = NeuralDesktopBody(NeuralDesktopConfig(**payload["config"])).to(device)
    restored.load_state_dict(payload["model"]); restored.eval(); model.eval()
    query, canvas, _, _, _ = renderer.find_canvas(active[-1], active, seed)
    query = query.unsqueeze(0).to(device).float().div(255)
    canvas = canvas.unsqueeze(0).to(device).float().div(255)
    left_embedding, left_class = _encode(model, query)
    right_embedding, right_class = _encode(restored, query)
    left_spatial, left_point = model.find_visual_symbol(canvas, left_embedding)
    right_spatial, right_point = restored.find_visual_symbol(canvas, right_embedding)
    maximum_delta = max(float((left_class-right_class).abs().max()),
                        float((left_spatial-right_spatial).abs().max()),
                        float((left_point-right_point).abs().max()))
    metric_keys = ("accuracy", "match_accuracy", "find_center_cell_accuracy", "find_top3_recall",
                   "find_top5_recall", "find_median_error", "find_p90_error",
                   "find_target_region_probability")
    before_metrics = evaluate(model, renderer, active, samples=60, find_samples=60,
                              seed=seed+700_001, device=device, batch_size=30)
    after_metrics = evaluate(restored, renderer, active, samples=60, find_samples=60,
                             seed=seed+700_001, device=device, batch_size=30)
    metric_delta = {name: abs(float(before_metrics[name])-float(after_metrics[name])) for name in metric_keys}
    metrics_equal = before_metrics["confusion_matrix"] == after_metrics["confusion_matrix"] and max(metric_delta.values()) == 0.0
    return {"passed": maximum_delta == 0.0 and metrics_equal, "maximum_output_delta": maximum_delta,
            "metric_delta": metric_delta, "confusion_equal": before_metrics["confusion_matrix"] == after_metrics["confusion_matrix"],
            "prediction": SYMBOLS[int(right_class[0, :len(SYMBOLS)].argmax())],
            "target_point": right_point[0].cpu().tolist()}


def main() -> int:
    parser = argparse.ArgumentParser(description="RGB-only visual alphabet curriculum for NeuralDesktopBody")
    parser.add_argument("command", nargs="?", choices=("train", "evaluate"), default="train")
    parser.add_argument("--output", type=Path, default=Path("state/visual_alphabet"))
    parser.add_argument("--base-checkpoint", type=Path, help="optional initialization; encoder remains trainable unless --freeze-encoder")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint for evaluate (defaults to OUTPUT/visual_alphabet.pt)")
    parser.add_argument("--report-output", type=Path,
                        help="optional JSON result path for read-only evaluation")
    parser.add_argument("--evaluate", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--fresh", action="store_true", help="hard-reset every learned Heiwa state before training; no backup")
    parser.add_argument("--freeze-encoder", action="store_true", help="ablation only: train symbol heads without shared visual encoder")
    parser.add_argument("--real-background-dir", type=Path,
                        default=Path("assets/visual_alphabet/real_holdout"))
    parser.add_argument("--real-source-manifest", type=Path,
                        default=Path("assets/visual_alphabet/sources.json"))
    parser.add_argument("--max-stage", type=int, choices=range(1, 4), default=3,
                        help="localization gate currently limits curriculum to е/а/о")
    parser.add_argument("--start-stage", type=int, choices=range(1, 4), default=1,
                        help="resume a clean cumulative run from an already reached stage")
    parser.add_argument("--train-samples", type=int, default=1200)
    parser.add_argument("--val-samples", type=int, default=360)
    parser.add_argument("--find-val-samples", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--find-batch-size", type=int, default=8)
    parser.add_argument("--max-epochs-per-stage", type=int, default=12)
    parser.add_argument("--accuracy-threshold", type=float, default=.94)
    parser.add_argument("--per-symbol-threshold", type=float, default=.90)
    parser.add_argument("--match-threshold", type=float, default=.90)
    parser.add_argument("--find-threshold", type=float, default=.75)
    parser.add_argument("--spatial-error-threshold", type=float, default=.10)
    parser.add_argument("--mine-tail", type=Path,
                        help="tail_summary.json; mines conditions with new seeds, never validation seeds")
    parser.add_argument("--hard-negative-fraction", type=float, default=.30)
    parser.add_argument("--holdout-round", type=int, default=0,
                        help="rotates final source seeds; mined seeds are never reused")
    parser.add_argument("--domain-adapt", action="store_true",
                        help="interleave synthetic replay with train-only real screenshot overlays/consistency")
    parser.add_argument("--domain-only", action="store_true",
                        help="domain ablation B: omit synthetic replay from optimizer loss")
    parser.add_argument("--domain-weight", type=float, default=.25,
                        help="weight of domain loss when synthetic replay is enabled")
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=240731)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.hard_negative_fraction <= .40:
        parser.error("--hard-negative-fraction must be in [0, 0.40]")
    if args.domain_only and not args.domain_adapt:
        parser.error("--domain-only requires --domain-adapt")
    if not 0 < args.domain_weight <= 1:
        parser.error("--domain-weight must be in (0, 1]")
    if args.smoke:
        args.max_stage = min(args.max_stage, 3); args.train_samples = min(args.train_samples, 600)
        args.val_samples = min(args.val_samples, 240); args.find_val_samples = min(args.find_val_samples, 100)
        args.max_epochs_per_stage = max(6, min(args.max_epochs_per_stage, 8))
        args.accuracy_threshold = min(args.accuracy_threshold, .88)
        args.per_symbol_threshold = min(args.per_symbol_threshold, .82)
        args.match_threshold = min(args.match_threshold, .82); args.find_threshold = min(args.find_threshold, .60)
    torch.manual_seed(args.seed); random.seed(args.seed)
    project = Path(__file__).resolve().parents[3]
    if args.fresh and args.command == "evaluate": parser.error("--fresh cannot be used with evaluate")
    if args.fresh and args.start_stage != 1: parser.error("--fresh requires --start-stage 1")
    if args.start_stage > args.max_stage: parser.error("--start-stage cannot exceed --max-stage")
    if args.fresh:
        from worlds.desktop.training.visual_school import _fresh_state
        _fresh_state(project)
        args.base_checkpoint = project / "state/neural_desktop.pt"
    device = select_device(args.device); fonts = discover_fonts(); renderer = AlphabetRenderer(fonts)
    real_train_sources, real_backgrounds = _real_source_split(args.real_source_manifest)
    if not real_backgrounds and args.real_background_dir.exists():
        real_backgrounds = tuple(sorted(args.real_background_dir.glob("*.png")))
    real_renderer = AlphabetRenderer(fonts, backgrounds=real_backgrounds) if real_backgrounds else None
    real_train_renderer = AlphabetRenderer(fonts, backgrounds=real_train_sources) if real_train_sources else None
    real_train_pool = RealScreenshotPool(real_train_sources) if real_train_sources else None
    if args.domain_adapt and (real_train_renderer is None or real_train_pool is None):
        parser.error("--domain-adapt requires non-overlapping train sources in --real-source-manifest")
    evaluate_path = args.evaluate or args.checkpoint or args.output / "visual_alphabet.pt"
    if args.command == "evaluate" or args.evaluate:
        payload = torch.load(evaluate_path, map_location="cpu", weights_only=False)
        model = NeuralDesktopBody(NeuralDesktopConfig(**payload["config"])).to(device)
        model.load_state_dict(payload["model"]); active = tuple(payload["active_symbols"])
        evaluation_seed = args.seed + 900_000_000 + args.holdout_round * 100_000_003
        metrics = evaluate(model, renderer, active, samples=args.val_samples,
                           find_samples=args.find_val_samples, seed=evaluation_seed,
                           device=device, batch_size=args.batch_size,
                           tail_dir=args.output/"tail"/"evaluate_synthetic")
        real_metrics = None if real_renderer is None else evaluate(
            model, real_renderer, active, samples=args.val_samples,
            find_samples=args.find_val_samples, seed=evaluation_seed+1_000_000_000,
            device=device, batch_size=args.batch_size,
            tail_dir=args.output/"tail"/"evaluate_real_screenshots",
        )
        real_consistency = pure_real_consistency(model, real_backgrounds, device=device,
                                                 seed=args.seed+2_900_000_000)
        ui_metrics = ui_control_benchmark(model, device=device)
        train_metrics = None if real_train_renderer is None else evaluate(
            model, real_train_renderer, active, samples=args.val_samples,
            find_samples=args.find_val_samples, seed=evaluation_seed+1_500_000_000,
            device=device, batch_size=args.batch_size,
        )
        train_by_source = evaluate_real_sources(
            model, fonts, real_train_sources, active, seed=evaluation_seed+1_600_000_000,
            device=device, batch_size=args.batch_size,
            samples=args.val_samples, find_samples=args.find_val_samples,
        )
        holdout_by_source = evaluate_real_sources(
            model, fonts, real_backgrounds, active, seed=evaluation_seed+1_700_000_000,
            device=device, batch_size=args.batch_size,
            samples=args.val_samples, find_samples=args.find_val_samples,
        )
        reload_probe = _reload_probe(evaluate_path, model, renderer, active, device,
                                     args.seed+3_000_000_000)
        diag = diagnostic(model, renderer, active, args.output/"diagnostic.png", device, args.seed+91_000_001)
        result = {"metrics": metrics,
                  "real_screenshot_train_overlay": train_metrics,
                  "real_screenshot_train_by_source": train_by_source,
                  "real_screenshot_holdout": real_metrics,
                  "real_screenshot_holdout_by_source": holdout_by_source,
                  "pure_real_consistency": real_consistency, "ui_control_benchmark": ui_metrics,
                  "reload_probe": reload_probe, "diagnostic": diag}
        if args.report_output is not None:
            args.report_output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report_output.with_suffix(args.report_output.suffix+".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            temporary.replace(args.report_output)
        print(json.dumps(result, ensure_ascii=False)); return 0
    config = NeuralDesktopConfig(); model = NeuralDesktopBody(config).to(device)
    source = "fresh"
    if args.base_checkpoint:
        load_checkpoint(args.base_checkpoint, model); source = str(args.base_checkpoint)
    ui_control_before = ui_control_benchmark(model, device=device)
    shared_visual_prefixes = ("stem.", "spatial.", "position_projection.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (name.startswith(SYMBOL_PREFIX)
                                   or (not args.freeze_encoder and name.startswith(shared_visual_prefixes)))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=.005)
    failure_conditions = _load_failure_conditions(args.mine_tail)
    sanity = _sanity_gate(model, renderer, optimizer, trainable, device=device,
                          output=args.output, seed=args.seed + 7_000_000)
    localization_probe = _localization_gradient_probe(
        model, renderer, device=device, seed=args.seed + 8_000_000,
    )
    if not bool(localization_probe["passed"]):
        raise RuntimeError(f"localization is detached from shared RGB encoder: {localization_probe}")
    print(json.dumps({"sanity": sanity, "localization_probe": localization_probe}, ensure_ascii=False), flush=True)
    history: list[dict[str, object]] = []; started = time.monotonic(); final_active = STAGES[0]
    retained_domain_state: dict[str, Tensor] | None = None
    retained_domain_optimizer: dict[str, object] | None = None
    retained_domain_epoch: int | None = None
    passed_all = True
    for stage_index in range(args.start_stage, args.max_stage + 1):
        active = STAGES[stage_index - 1]
        final_active = active; stable = 0; stage_passed = False
        for epoch in range(1, args.max_epochs_per_stage + 1):
            model.train(); dataset = RecognitionDataset(renderer, active, args.train_samples,
                                                         args.seed + stage_index*10_000_000 + epoch*100_000)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            totals = {"loss": 0., "classification": 0., "contrastive": 0., "find": 0., "domain": 0.}; batches = 0
            mined_matched = 0; mined_requested = 0
            for batch_index, batch in enumerate(loader):
                first = batch["first"].to(device).float().div(255)
                second = batch["second"].to(device).float().div(255)
                labels = batch["label"].to(device); images = torch.cat((first, second)); doubled = torch.cat((labels, labels))
                embeddings, logits = _encode(model, images)
                classification = F.cross_entropy(logits[:, :len(SYMBOLS)].float(), doubled)
                contrastive = _contrastive(embeddings, doubled)
                find_count = min(args.find_batch_size, len(labels))
                seeds = [args.seed + stage_index*100_000_000 + epoch*1_000_000 + batch_index*1000+i for i in range(find_count)]
                if failure_conditions:
                    mined_requested += max(1, round(len(seeds)*args.hard_negative_fraction))
                    seeds, matched = _mine_independent_seeds(
                        renderer, active, seeds, failure_conditions, args.hard_negative_fraction,
                    )
                    mined_matched += matched
                find_class, find_point, _ = _find_batch(model, renderer, active, seeds, device)
                loss = classification + .65*contrastive + 1.15*find_class + 4.0*find_point
                domain_loss = loss.new_zeros(())
                if args.domain_adapt:
                    assert real_train_renderer is not None and real_train_pool is not None
                    overlay_seeds = [seed + 4_000_000_019 for seed in seeds]
                    overlay_class, overlay_point, _ = _find_batch(
                        model, real_train_renderer, active, overlay_seeds, device,
                    )
                    overlay_count = min(8, len(labels))
                    overlay_symbols = [active[int(labels[index]) % len(active)] for index in range(overlay_count)]
                    overlay_first = torch.stack([
                        real_train_renderer.symbol(symbol, seeds[index]+5_000_000_021).pixels
                        for index, symbol in enumerate(overlay_symbols)
                    ]).to(device).float().div(255)
                    overlay_second = torch.stack([
                        real_train_renderer.symbol(symbol, seeds[index]+6_000_000_023).pixels
                        for index, symbol in enumerate(overlay_symbols)
                    ]).to(device).float().div(255)
                    overlay_labels = labels[:overlay_count]
                    overlay_embedding, overlay_logits = _encode(
                        model, torch.cat((overlay_first, overlay_second)),
                    )
                    doubled_overlay_labels = torch.cat((overlay_labels, overlay_labels))
                    overlay_recognition = F.cross_entropy(
                        overlay_logits[:, :len(SYMBOLS)].float(), doubled_overlay_labels,
                    )
                    overlay_contrastive = _contrastive(overlay_embedding, doubled_overlay_labels)
                    region_seeds = [seed + 7_000_000_027 for seed in seeds[:min(8, len(seeds))]]
                    region_consistency = _real_region_loss(model, real_train_pool, region_seeds, device)
                    domain_loss = (overlay_recognition + .65*overlay_contrastive
                                   + 1.15*overlay_class + 4.0*overlay_point
                                   + .25*region_consistency)
                    loss = domain_loss if args.domain_only else loss + args.domain_weight*domain_loss
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
                totals["loss"] += float(loss.detach()); totals["classification"] += float(classification.detach())
                totals["contrastive"] += float(contrastive.detach()); totals["find"] += float((find_class+find_point).detach()); batches += 1
                totals["domain"] += float(domain_loss.detach())
            metrics = evaluate(model, renderer, active, samples=args.val_samples,
                               find_samples=args.find_val_samples,
                               seed=args.seed + 800_000_000 + stage_index*10_000_000,
                               device=device, batch_size=args.batch_size)
            gate = (float(metrics["accuracy"]) >= args.accuracy_threshold
                    and min(float(v) for v in metrics["per_symbol"].values()) >= args.per_symbol_threshold
                    and float(metrics["match_accuracy"]) >= args.match_threshold
                    and float(metrics["find_center_cell_accuracy"]) >= args.find_threshold
                    and float(metrics["find_median_error"]) < args.spatial_error_threshold)
            if stage_index == 3:
                gate = (gate and float(metrics["find_center_cell_accuracy"]) >= .70
                        and float(metrics["find_top3_recall"]) >= .90
                        and float(metrics["find_median_error"]) <= .07
                        and float(metrics["find_p90_error"]) <= .20)
            if args.domain_adapt:
                per_location = metrics["localization_per_symbol"]
                retention = (float(metrics["accuracy"]) >= .95
                             and min(float(value) for value in metrics["per_symbol"].values()) >= .95
                             and float(metrics["find_center_cell_accuracy"]) >= .75
                             and float(metrics["find_top3_recall"]) >= .95
                             and float(metrics["find_p90_error"]) <= .15
                             and max(float(value["p90_error"]) for value in per_location.values()) <= .15)
                gate = gate and retention
                if retention:
                    retained_domain_state = {
                        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                    }
                    retained_domain_optimizer = copy.deepcopy(optimizer.state_dict())
                    retained_domain_epoch = epoch
            stable = stable + 1 if gate else 0
            minimum_epochs = 6 if (failure_conditions or args.domain_adapt) else 2
            stage_passed = stable >= 2 and epoch >= minimum_epochs
            row = {"stage": stage_index, "symbols": list(active), "epoch": epoch,
                   **{name: value/max(batches,1) for name, value in totals.items()},
                   "hard_negative": {"conditions": len(failure_conditions),
                                     "matched": mined_matched, "requested": mined_requested,
                                     "fraction_cap": args.hard_negative_fraction},
                   "validation": metrics, "gate": gate, "stable_passes": stable}
            history.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
            if stage_passed: break
        if not stage_passed:
            passed_all = False; break
    if args.domain_adapt and retained_domain_state is not None:
        model.load_state_dict(retained_domain_state)
        if retained_domain_optimizer is not None: optimizer.load_state_dict(retained_domain_optimizer)
    final_seed = args.seed + 990_000_000 + args.holdout_round * 100_000_003
    final_metrics = evaluate(model, renderer, final_active, samples=args.val_samples,
                             find_samples=args.find_val_samples, seed=final_seed,
                             device=device, batch_size=args.batch_size,
                             tail_dir=args.output/"tail"/"synthetic")
    real_metrics = None if real_renderer is None else evaluate(
        model, real_renderer, final_active, samples=args.val_samples,
        find_samples=args.find_val_samples, seed=final_seed+1_000_000_000,
        device=device, batch_size=args.batch_size, tail_dir=args.output/"tail"/"real_screenshots",
    )
    real_train_metrics = None if real_train_renderer is None else evaluate(
        model, real_train_renderer, final_active, samples=args.val_samples,
        find_samples=args.find_val_samples, seed=final_seed+1_500_000_000,
        device=device, batch_size=args.batch_size,
    )
    real_train_by_source = evaluate_real_sources(
        model, fonts, real_train_sources, final_active, seed=final_seed+1_600_000_000,
        device=device, batch_size=args.batch_size,
    )
    real_holdout_by_source = evaluate_real_sources(
        model, fonts, real_backgrounds, final_active, seed=final_seed+1_700_000_000,
        device=device, batch_size=args.batch_size,
    )
    per_symbol_localization = final_metrics["localization_per_symbol"]
    final_gate = (float(final_metrics["accuracy"]) >= .95
                  and min(float(value) for value in final_metrics["per_symbol"].values()) >= .95
                  and float(final_metrics["match_accuracy"]) >= args.match_threshold
                  and float(final_metrics["find_center_cell_accuracy"]) >= .75
                  and float(final_metrics["find_top3_recall"]) >= .95
                  and float(final_metrics["find_median_error"]) <= .07
                  and float(final_metrics["find_p90_error"]) <= .15
                  and min(float(value["top3"]) for value in per_symbol_localization.values()) >= .90
                  and max(float(value["p90_error"]) for value in per_symbol_localization.values()) <= .15)
    passed_all = passed_all and final_gate
    real_consistency = pure_real_consistency(
        model, real_backgrounds, device=device, seed=final_seed + 2_000_000_000,
    )
    ui_control_after = ui_control_benchmark(model, device=device)
    ui_metric_deltas = {
        key: float(ui_control_after[key]) - float(ui_control_before[key])
        for key in ("pearson_x", "spearman_x", "median_center_error_px",
                    "p90_center_error_px", "left_right_ordering_accuracy", "iou_mean")
        if key in ui_control_before and key in ui_control_after
    }
    report = {"passed": passed_all, "final_gate": final_gate, "final_source_seed": final_seed,
              "source": source, "stages_completed": len({row["stage"] for row in history if row["gate"]}),
              "domain_adaptation": bool(args.domain_adapt), "domain_only": bool(args.domain_only),
              "retained_domain_epoch": retained_domain_epoch,
              "active_symbols": list(final_active), "trainable_parameters": sum(p.numel() for p in trainable),
              "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
              "seconds": time.monotonic()-started, "sanity": sanity,
              "localization_probe": localization_probe,
              "split": {"train": "generated seeds by stage/epoch", "validation": "fixed unseen generated seeds",
                        "test": "disjoint seeds and source screenshots"},
              "real_background_sources": [str(path) for path in real_backgrounds],
              "real_train_sources": [str(path) for path in real_train_sources],
              "final": final_metrics, "real_screenshot_train_overlay": real_train_metrics,
              "real_screenshot_train_by_source": real_train_by_source,
              "real_screenshot_holdout": real_metrics,
              "real_screenshot_holdout_by_source": real_holdout_by_source,
              "pure_real_consistency": real_consistency,
              "ui_control_benchmark_before": ui_control_before,
              "ui_control_benchmark_after": ui_control_after,
              "ui_control_metric_deltas": ui_metric_deltas,
              "history": history}
    args.output.mkdir(parents=True, exist_ok=True); checkpoint = args.output/"visual_alphabet.pt"
    _save(checkpoint, model, optimizer, report, final_active)
    reload_probe = _reload_probe(checkpoint, model, renderer, final_active, device,
                                 args.seed + 2_000_000_000)
    if not bool(reload_probe["passed"]):
        raise RuntimeError(f"visual alphabet save/reload changed outputs: {reload_probe}")
    report["reload_probe"] = reload_probe
    _save(checkpoint, model, optimizer, report, final_active)
    # This is the same full NeuralDesktopBody consumed by normal RGB runtime,
    # not an isolated classifier checkpoint.
    runtime_checkpoint = project / "state/neural_desktop.pt"
    if passed_all:
        runtime_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        runtime_tmp = runtime_checkpoint.with_suffix(".pt.tmp")
        torch.save({"format": 1, "config": {name: getattr(model.config, name) for name in model.config.__dataclass_fields__},
                    "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "step": len(history), "visual_alphabet": report}, runtime_tmp)
        runtime_tmp.replace(runtime_checkpoint)
    (args.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    diag = diagnostic(model, renderer, final_active, args.output/"diagnostic.png", device, args.seed+91_000_001)
    print(json.dumps({"passed": passed_all, "checkpoint": str(checkpoint),
                      "runtime_checkpoint": str(runtime_checkpoint) if passed_all else None,
                      "report": str(args.output/"report.json"),
                      "diagnostic": diag, "final": final_metrics}, ensure_ascii=False), flush=True)
    return 0 if passed_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
