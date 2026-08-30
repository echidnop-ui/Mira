from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True, slots=True)
class AutonomyConfig:
    autonomous: bool = True
    vision: bool = True
    mouse: bool = True
    clicks: bool = True
    right_clicks: bool = True
    keyboard: bool = True
    typing: bool = True
    online_learning: bool = True
    capture_user_input: bool = True
    teacher_channel: bool = True
    teacher_backend: str = "chatgpt"
    observation_interval_seconds: float = 1.0
    minimum_action_confidence: float = 0.0
    maximum_actions_per_second: float = 0.5
    checkpoint_interval_seconds: float = 300.0
    teacher_interval_seconds: float = 30.0
    teacher_reply_timeout_seconds: float = 45.0
    reading_minimum_confidence: float = 0.35
    policy_exploration_temperature: float = 1.0
    policy_exploration_mixture: float = 0.25
    policy_specialist_prior_weight: float = 0.25
    stagnation_noop_steps: int = 3
    stagnation_noop_penalty: float = 0.35
    learn_stationary_desktop: bool = True
    background_reader_training: bool = True
    reader_train_samples: int = 16000
    reader_val_samples: int = 1005
    reader_epochs_per_stage: int = 20
    reader_batch_size: int = 64
    reader_learning_rate: float = 0.001
    background_language_learning: bool = True
    language_observation_visual_confidence: float = 0.80
    language_learning_poll_seconds: float = 3.0
    language_learning_save_every: int = 4

    def __post_init__(self) -> None:
        if self.teacher_backend not in {"none", "chatgpt", "ollama", "command"}:
            raise ValueError("unsupported teacher_backend")
        positive = (self.observation_interval_seconds, self.maximum_actions_per_second,
                    self.checkpoint_interval_seconds, self.teacher_interval_seconds,
                    self.teacher_reply_timeout_seconds)
        if any(value <= 0 for value in positive): raise ValueError("runtime intervals must be > 0")
        if not 0 <= self.minimum_action_confidence <= 1: raise ValueError("confidence must be in [0,1]")
        if not 0 <= self.reading_minimum_confidence <= 1: raise ValueError("reading confidence must be in [0,1]")
        if self.policy_exploration_temperature <= 0:
            raise ValueError("policy_exploration_temperature must be > 0")
        if not 0.0 <= self.policy_exploration_mixture < 1.0:
            raise ValueError("policy_exploration_mixture must be in [0,1)")
        if self.policy_specialist_prior_weight < 0.0:
            raise ValueError("policy_specialist_prior_weight must be >= 0")
        if self.stagnation_noop_steps < 1:
            raise ValueError("stagnation_noop_steps must be >= 1")
        if not 0.0 < self.stagnation_noop_penalty <= 1.0:
            raise ValueError("stagnation_noop_penalty must be in (0,1]")
        if min(self.reader_train_samples, self.reader_val_samples, self.reader_epochs_per_stage, self.reader_batch_size) <= 0:
            raise ValueError("reader training sizes must be > 0")
        if self.reader_learning_rate <= 0:
            raise ValueError("reader_learning_rate must be > 0")
        if not 0.0 <= self.language_observation_visual_confidence <= 1.0:
            raise ValueError("language_observation_visual_confidence must be in [0,1]")
        if self.language_learning_poll_seconds <= 0:
            raise ValueError("language_learning_poll_seconds must be > 0")
        if self.language_learning_save_every < 1:
            raise ValueError("language_learning_save_every must be >= 1")


def save_config(path: Path, config: AutonomyConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(asdict(config), stream, ensure_ascii=False, indent=2); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: Path) -> AutonomyConfig:
    if not path.exists():
        config = AutonomyConfig(); save_config(path, config); return config
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise ValueError("runtime config root must be an object")
    allowed = {field.name for field in fields(AutonomyConfig)}
    unknown = set(payload) - allowed
    if unknown: raise ValueError(f"unknown runtime config keys: {sorted(unknown)}")
    return AutonomyConfig(**payload)
