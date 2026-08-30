from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Mapping

from worlds.interface import WorldFrame, validate_frame


_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-zА-Яа-яЁё]+")
DEFAULT_SIZE = 96


@dataclass(frozen=True, slots=True)
class DesktopRepresentation:
    """
    World-local normalized Desktop representation for shared cognition.

    The representation contains external Desktop state only. It does not choose
    an action and does not generate language.
    """

    transition_id: str
    world_id: str
    as_of: float
    vector: tuple[float, ...]


def represent_desktop_frame(
    frame: WorldFrame,
    *,
    transition_id: str,
    size: int = DEFAULT_SIZE,
) -> DesktopRepresentation:
    transition_id = transition_id.strip()
    if not transition_id:
        raise ValueError("transition_id must not be empty")
    if size <= 0:
        raise ValueError("size must be > 0")

    validate_frame(frame, world_id=frame.world)

    values = [0.0] * size

    _add_numeric(values, "observation_count", len(frame.observations))
    _add_numeric(values, "affordance_count", len(frame.affordances))
    _add_numeric(values, "completed_count", len(frame.completed_actions))
    _add_numeric(values, "consequence_count", len(frame.consequences))

    _add_text(values, "world", frame.world)

    for observation in frame.observations:
        _add_text(values, "observation.kind", observation.kind)
        if observation.kind == "desktop.neural_visual_state":
            latent = observation.payload.get("latent")
            if isinstance(latent, (tuple, list)):
                _add_dense_vector(values, latent)
        _add_mapping(values, "observation.payload", observation.payload)

    for affordance in frame.affordances:
        _add_text(values, "affordance.action", affordance.action)
        _add_mapping(values, "affordance.payload", affordance.payload)
        _add_mapping(values, "affordance.constraints", affordance.constraints)

    for completed in frame.completed_actions:
        _add_text(values, "completed.action", completed.action)
        _add_text(values, "completed.status", completed.status)
        _add(values, f"completed.succeeded:{int(completed.succeeded)}")
        _add_mapping(values, "completed.payload", completed.payload)

    for consequence in frame.consequences:
        _add_text(values, "consequence.kind", consequence.kind)
        _add_mapping(values, "consequence.payload", consequence.payload)

    norm = math.sqrt(sum(value * value for value in values))
    if norm > 0.0:
        values = [value / norm for value in values]

    return DesktopRepresentation(
        transition_id=transition_id,
        world_id=frame.world,
        as_of=float(frame.observed_at),
        vector=tuple(values),
    )


def _slot(key: str, size: int) -> tuple[int, float]:
    digest = hashlib.blake2b(
        key.encode("utf-8", errors="replace"),
        digest_size=16,
        person=b"heiwa-desktop",
    ).digest()

    index = int.from_bytes(digest[:8], "little") % size
    sign = 1.0 if (digest[8] & 1) == 0 else -1.0
    return index, sign


def _add(values: list[float], key: str, amount: float = 1.0) -> None:
    index, sign = _slot(key, len(values))
    values[index] += sign * float(amount)


def _add_numeric(values: list[float], key: str, value: float) -> None:
    value = float(value)
    if not math.isfinite(value):
        return

    # Keep arbitrary external magnitudes bounded.
    bounded = math.tanh(value)
    _add(values, f"numeric:{key}", bounded)


def _add_dense_vector(values: list[float], vector: tuple[object, ...] | list[object]) -> None:
    """Preserve learned visual geometry instead of reducing it to text/hash features."""
    for index, raw in enumerate(vector):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if math.isfinite(value):
            values[index % len(values)] += value / math.sqrt(1 + index // len(values))


def _add_text(values: list[float], key: str, text: object) -> None:
    normalized = str(text).strip().lower()
    if not normalized:
        return

    _add(values, f"text:{key}:whole:{normalized}")

    for token in _TOKEN_SPLIT.split(normalized):
        if token:
            _add(values, f"text:{key}:token:{token}")


def _add_mapping(
    values: list[float],
    prefix: str,
    mapping: Mapping[str, object],
    *,
    depth: int = 0,
) -> None:
    if depth > 3:
        return

    for raw_key in sorted(mapping, key=str):
        key = str(raw_key)
        value = mapping[raw_key]
        path = f"{prefix}.{key}"

        if isinstance(value, bool):
            _add(values, f"bool:{path}:{int(value)}")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            _add_numeric(values, path, float(value))
        elif isinstance(value, str):
            _add_text(values, path, value)
        elif isinstance(value, Mapping):
            _add_mapping(
                values,
                path,
                value,
                depth=depth + 1,
            )
        elif isinstance(value, (tuple, list)):
            _add_numeric(values, f"{path}.length", len(value))
            for index, item in enumerate(value[:16]):
                if isinstance(item, str):
                    _add_text(values, f"{path}[{index}]", item)
                elif isinstance(item, bool):
                    _add(values, f"bool:{path}[{index}]:{int(item)}")
                elif isinstance(item, (int, float)):
                    _add_numeric(values, f"{path}[{index}]", float(item))
        elif value is not None:
            _add_text(values, path, type(value).__name__)
