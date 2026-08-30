from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Mapping

import torch
from torch import Tensor
from torch.utils.data import Dataset

from worlds.desktop.neural import ACTION_KINDS, NeuralDesktopConfig


@dataclass(frozen=True, slots=True)
class DemonstrationTransition:
    before: str
    after: str
    action: str
    parameters: tuple[float, float, float, float]
    timestamp: float
    frame_before_ns: int = 0
    action_timestamp_ns: int = 0
    frame_after_ns: int = 0
    target_xy: tuple[float, float] | None = None
    trainable: bool = True
    motor_event: Mapping[str, object] | None = None
    teacher: dict[str, object] | None = None
    human_feedback: float | None = None


def append_manifest(path: Path, transition: DemonstrationTransition) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(transition), ensure_ascii=False) + "\n")


class AtomicManifestWriter:
    """Session writer: existing records are preserved and final publish is atomic."""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path); self.partial = self.path.with_suffix(self.path.suffix + ".partial")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.partial.open("w", encoding="utf-8")
        if self.path.exists(): self.stream.write(self.path.read_text(encoding="utf-8"))

    def append(self, transition: DemonstrationTransition) -> None:
        self.stream.write(json.dumps(asdict(transition), ensure_ascii=False) + "\n"); self.stream.flush()

    def close(self, *, commit: bool = True) -> None:
        if self.stream.closed: return
        self.stream.flush(); import os; os.fsync(self.stream.fileno()); self.stream.close()
        if commit: self.partial.replace(self.path)

    def __enter__(self) -> "AtomicManifestWriter": return self
    def __exit__(self, exc_type: object, *_: object) -> None: self.close(commit=exc_type is None)


def _pillow_image():
    try:
        from PIL import Image
    except ModuleNotFoundError:
        # Ubuntu's system Pillow is ABI-independent Python for this use and is
        # already required by the capture worker. Avoid one subprocess/frame.
        system_packages = "/usr/lib/python3/dist-packages"
        if system_packages not in sys.path: sys.path.append(system_packages)
        from PIL import Image
    return Image


def load_rgb(path: Path, *, width: int, height: int, system_python: str = "/usr/bin/python3") -> Tensor:
    del system_python  # Retained for call compatibility with older callers.
    Image = _pillow_image()
    with Image.open(path) as source:
        image = source.convert("RGB").resize((width, height))
        raw = bytearray(image.tobytes())
    expected = width * height * 3
    if len(raw) != expected:
        raise ValueError(f"decoded frame has {len(raw)} bytes, expected {expected}")
    data = torch.frombuffer(raw, dtype=torch.uint8)
    return data.reshape(height, width, 3).permute(2, 0, 1).float().div_(255.0)


class DesktopDemonstrationDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, manifest: str | Path, config: NeuralDesktopConfig = NeuralDesktopConfig()) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.config = config
        self.records = [item for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()
                        for item in (json.loads(line),) if item.get("trainable", True)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = self.records[index]
        before = load_rgb(self.root / item["before"], width=self.config.image_width, height=self.config.image_height)
        after = load_rgb(self.root / item["after"], width=self.config.image_width, height=self.config.image_height)
        action = torch.tensor(ACTION_KINDS.index(item["action"]), dtype=torch.long)
        parameters = torch.tensor(item.get("parameters", [0, 0, 0, 0]), dtype=torch.float32)
        target = item.get("target_xy")
        if target is None:
            target_cell = torch.tensor(-1, dtype=torch.long)
            target_xy = torch.full((2,), float("nan"))
        else:
            gw, gh = self.config.image_width // 16, self.config.image_height // 16
            x, y = min(max(float(target[0]), 0.0), 0.999999), min(max(float(target[1]), 0.0), 0.999999)
            target_cell = torch.tensor(int(y * gh) * gw + int(x * gw), dtype=torch.long)
            target_xy = torch.tensor((x, y), dtype=torch.float32)
        teacher = item.get("teacher") if isinstance(item.get("teacher"), dict) else {}
        teacher_visual = bool(teacher.get("language_teacher") or teacher.get("pixel_supervision"))
        raw_bbox = teacher.get("region_bbox") if teacher_visual else None
        raw_motor_bbox = teacher.get("motor_bbox") if teacher_visual else None
        target_bbox = torch.full((4,), float("nan"))
        if isinstance(raw_motor_bbox, (list, tuple)) and len(raw_motor_bbox) == 4:
            target_bbox = torch.tensor([min(max(float(v), 0.0), 1.0) for v in raw_motor_bbox])
        read_bbox = torch.full((4,), float("nan"))
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            read_bbox = torch.tensor([min(max(float(v), 0.0), 1.0) for v in raw_bbox])
        read_tokens = torch.full((self.config.max_read_bytes + 2,), -100, dtype=torch.long)
        visual_text = teacher.get("visual_text") if teacher_visual else None
        if isinstance(visual_text, str):
            encoded = [1, *(value + 4 for value in visual_text.encode("utf-8")[:self.config.max_read_bytes]), 2]
            read_tokens[:len(encoded)] = torch.tensor(encoded)
        trajectory_target = torch.full((3,), float("nan"))
        event = item.get("motor_event") if isinstance(item.get("motor_event"), dict) else {}
        start, end = event.get("start_position"), event.get("end_position")
        if isinstance(start, list) and isinstance(end, list) and len(start) == len(end) == 2:
            import math
            trajectory_target = torch.tensor((
                math.tanh(float(event.get("duration_ns", 0)) / 1e9),
                float(end[0]) - float(start[0]), float(end[1]) - float(start[1]),
            ), dtype=torch.float32)
        feedback = item.get("human_feedback")
        selection_feedback = torch.tensor(float("nan") if feedback is None else float(feedback), dtype=torch.float32)
        return {"before": before, "after": after, "action": action, "parameters": parameters,
                "target_cell": target_cell, "target_xy": target_xy, "target_bbox": target_bbox,
                "read_bbox": read_bbox, "read_tokens": read_tokens, "trajectory_target": trajectory_target,
                "selection_feedback": selection_feedback}

    def split_indices(self, validation_fraction: float = 0.1) -> tuple[list[int], list[int]]:
        train, validation = [], []
        for index, item in enumerate(self.records):
            key = f"{item['before']}:{item['timestamp']}".encode()
            import hashlib
            bucket = int.from_bytes(hashlib.blake2b(key, digest_size=2).digest(), "big") / 65535
            (validation if bucket < validation_fraction else train).append(index)
        if not train and validation:
            train.append(validation.pop())
        return train, validation
