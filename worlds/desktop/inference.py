from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import tempfile
import threading

import torch

from worlds.desktop.neural import ACTION_KINDS, KEYBOARD_KEY_CODES, MotorOutput, NeuralDesktopBody, NeuralDesktopConfig, VisualState, select_device
from worlds.desktop.online_learning import OnlineDesktopLearner
from worlds.desktop.portal_screencast import PortalScreenCast
from worlds.desktop.training.dataset import load_rgb


@dataclass(frozen=True, slots=True)
class NeuralMotorProposal:
    action: str
    x: int
    y: int
    confidence: float
    action_distribution: tuple[float, ...]
    target_distribution: tuple[float, ...]
    screen_width: int
    screen_height: int
    key_code: int
    frame_digest: str
    target_bbox: tuple[float, float, float, float]
    trajectory: tuple[float, float, float]
    read_byte_ids: tuple[int, ...]
    read_confidence: float
    read_bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    read_focus_source: str = "neural"


class NeuralDesktopInference:
    """Normal inference path: pixels and prior motor action only; never AT-SPI."""

    def __init__(self, checkpoint: str | Path, *, portal: PortalScreenCast | None = None, device: str = "auto", replay_path: str | Path | None = None) -> None:
        self.checkpoint = Path(checkpoint)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.config = NeuralDesktopConfig(**payload["config"])
        self.device = select_device(device)
        self.model = NeuralDesktopBody(self.config).to(self.device)
        # Keep the freshly initialized values so a legacy checkpoint with fully
        # non-finite world-model tensors can be repaired without touching motor,
        # grounding, reader, or visual encoder weights.
        fresh_state = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        checkpoint_state = dict(payload["model"])
        for name, value in tuple(checkpoint_state.items()):
            if name.startswith("symbol_") and name in fresh_state and fresh_state[name].shape != value.shape:
                checkpoint_state.pop(name)
        repairable_prefixes = ("action_embedding.", "action_parameters.", "next_latent.", "change_query.")
        repaired: list[str] = []
        fatal: list[str] = []
        for name, value in checkpoint_state.items():
            if not torch.is_tensor(value) or bool(torch.isfinite(value).all()):
                continue
            if name.startswith(repairable_prefixes) and name in fresh_state and fresh_state[name].shape == value.shape:
                checkpoint_state[name] = fresh_state[name].to(device=value.device, dtype=value.dtype)
                repaired.append(name)
            else:
                fatal.append(name)
        if fatal:
            raise RuntimeError(f"non-finite neural Desktop checkpoint tensors outside repairable world model: {fatal}")
        if repaired:
            print({"desktop_checkpoint_repair": "nonfinite_world_model", "reinitialized": repaired}, flush=True)
            # Optimizer moments corresponding to NaN weights are untrustworthy.
            payload.pop("online_optimizer", None); payload.pop("optimizer", None)
        incompatible = self.model.load_state_dict(checkpoint_state, strict=False)
        allowed = ("key_head.", "task_projection.", "motor_fusion.", "grounding_box.",
                   "trajectory_head.", "foveal_reader.", "byte_reader.",
                   "visual_reader_v2.", "reading_", "symbol_")
        unexpected = set(incompatible.unexpected_keys) - {"byte_reader.weight", "byte_reader.bias"}
        if {name for name in incompatible.missing_keys if not name.startswith(allowed)} or unexpected:
            raise RuntimeError(f"incompatible neural Desktop checkpoint: {incompatible}")
        self.model.eval()
        self.model_lock = threading.RLock()
        self.portal = portal or PortalScreenCast()
        self.state: VisualState | None = None
        self.previous_state: VisualState | None = None
        self._last_action: tuple[int, tuple[float, float, float, float]] | None = None
        self.last_frame: torch.Tensor | None = None
        self.previous_frame: torch.Tensor | None = None
        resolved_replay = Path(replay_path) if replay_path is not None else self.checkpoint.with_suffix(".replay.pt")
        self.learner = OnlineDesktopLearner(self.model, device=self.device, replay_path=resolved_replay)
        self.learner.load_optimizer_state(payload.get("online_optimizer", payload.get("optimizer")))
        self.learner.load_reader_feedback_optimizer_state(payload.get("reader_feedback_optimizer"))
        self.learner.load_learning_state(payload.get("online_learning"))

    def remember_action(self, kind: int, parameters: tuple[float, float, float, float]) -> None:
        self._last_action = (int(kind), parameters)

    def adopt_reader_checkpoint(self, candidate: str | Path) -> bool:
        """Atomically adopt only the isolated reader branch into the live organism."""
        candidate = Path(candidate)
        if not candidate.exists(): return False
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        state = payload.get("model", {})
        reader = {k: v for k, v in state.items() if k.startswith(
            ("foveal_reader.", "visual_reader_v2.", "reading_")
        )}
        if not reader: return False
        with self.model_lock:
            current = self.model.state_dict()
            for key, value in reader.items():
                if key not in current or current[key].shape != value.shape:
                    raise RuntimeError(f"reader candidate incompatible at {key}")
            self.model.load_state_dict(reader, strict=False)
            self.model.eval()
        return True

    @torch.inference_mode()
    def observe_path(self, path: Path, *, last_action_kind: int | None = None, last_parameters: tuple[float, float, float, float] | None = None) -> VisualState:
        frame = load_rgb(path, width=self.config.image_width, height=self.config.image_height).unsqueeze(0).to(self.device)
        self.previous_frame = self.last_frame
        self.last_frame = frame.detach()
        kind = None if last_action_kind is None else torch.tensor([last_action_kind], device=self.device)
        params = None if last_parameters is None else torch.tensor([last_parameters], device=self.device)
        self.previous_state = self.state
        with self.model_lock:
            self.state = self.model.observe(frame, self.state, kind, params).detached()
        return self.state

    @torch.inference_mode()
    def capture_and_propose(self) -> NeuralMotorProposal:
        with tempfile.TemporaryDirectory(prefix="heiwa-neural-desktop-") as directory:
            frame = Path(directory) / "frame.png"; self.portal.capture_png(frame)
            import hashlib
            frame_digest = hashlib.sha256(frame.read_bytes()).hexdigest()
            # Native dimensions are retained solely for coordinate scaling.
            import subprocess
            size = subprocess.run(("/usr/bin/python3", "-c", "from PIL import Image; import sys; print(*Image.open(sys.argv[1]).size)", str(frame)), check=True, text=True, stdout=subprocess.PIPE).stdout.split()
            width, height = int(size[0]), int(size[1])
            last = self._last_action
            visual = self.observe_path(
                frame,
                last_action_kind=None if last is None else last[0],
                last_parameters=None if last is None else last[1],
            )
            self._last_action = None
        with self.model_lock:
            output = self.model.propose(visual)
            x, y = self.model.decode_target(output, screen_width=width, screen_height=height)
            # The Desktop motor head is a specialist prior, not the final
            # cognitive selector.  Preserve its unmodified distribution and
            # expose that distribution to the World adapter; shared Policy will
            # choose among the mechanically available motor affordances.
            action_probabilities = torch.softmax(output.action_logits[0].float(), dim=-1)
            selected_action = int(torch.argmax(action_probabilities).item())
            target_probabilities = output.target_logits.softmax(-1)[0]
            assert self.last_frame is not None
            read_bbox, focus_confidence = self.model.locate_reading_region(visual)
            read_logits = self.model.read_region(self.last_frame, read_bbox)
            read_probabilities = read_logits.softmax(-1)
            read_ids = read_probabilities.argmax(-1)[0]
            sequence_confidence = read_probabilities.max(-1).values[0]
            eos = int((read_ids == 2).nonzero()[0]) + 1 if bool((read_ids == 2).any()) else len(read_ids)
            read_confidence = float(sequence_confidence[:eos].mean() * focus_confidence[0])
        return NeuralMotorProposal(
            action=ACTION_KINDS[selected_action], x=int(x[0]), y=int(y[0]),
            confidence=float(output.confidence[0]),
            action_distribution=tuple(float(value) for value in action_probabilities.cpu()),
            target_distribution=tuple(float(value) for value in target_probabilities.cpu()),
            screen_width=width, screen_height=height,
            key_code=KEYBOARD_KEY_CODES[int(output.key_logits.argmax(-1)[0])],
            frame_digest=frame_digest,
            target_bbox=tuple(float(value) for value in output.target_bbox[0].cpu()),
            trajectory=tuple(float(value) for value in output.trajectory[0].cpu()),
            read_byte_ids=tuple(int(value) for value in read_ids.cpu()),
            read_confidence=read_confidence,
            read_bbox=tuple(float(value) for value in read_bbox[0].cpu()),
        )

    @staticmethod
    def cursor_fovea_bbox(
        proposal: NeuralMotorProposal, focus_xy: tuple[float, float],
    ) -> tuple[float, float, float, float]:
        """Place a compact 4:1 physical-pixel fovea around the cursor."""
        cx = min(max(float(focus_xy[0]), 0.0), 1.0)
        cy = min(max(float(focus_xy[1]), 0.0), 1.0)
        learned_width = max(float(proposal.read_bbox[2]) - float(proposal.read_bbox[0]), 0.0)
        width = min(max(learned_width, 0.18), 0.38)
        screen_aspect = max(float(proposal.screen_width), 1.0) / max(float(proposal.screen_height), 1.0)
        height = min(max(width * screen_aspect / 4.0, 0.07), 0.22)
        x0 = min(max(cx - width / 2.0, 0.0), 1.0 - width)
        y0 = min(max(cy - height / 2.0, 0.0), 1.0 - height)
        return x0, y0, x0 + width, y0 + height

    @torch.inference_mode()
    def refocus_reading(
        self, proposal: NeuralMotorProposal, focus_xy: tuple[float, float],
    ) -> NeuralMotorProposal:
        """Re-run only the neural high-resolution reader at the cursor fovea."""
        if self.last_frame is None:
            return proposal
        bbox_values = self.cursor_fovea_bbox(proposal, focus_xy)
        bbox = torch.tensor([bbox_values], device=self.device, dtype=self.last_frame.dtype)
        with self.model_lock:
            logits = self.model.read_region(self.last_frame, bbox)
            probabilities = logits.softmax(-1)
            ids = probabilities.argmax(-1)[0]
            token_confidence = probabilities.max(-1).values[0]
            eos = int((ids == 2).nonzero()[0]) + 1 if bool((ids == 2).any()) else len(ids)
            sequence_confidence = float(token_confidence[:eos].mean())
        # Cursor guidance chooses where to sense, not whether decoded bytes are
        # correct. Never inflate reader confidence merely because a person
        # pointed at the region.
        confidence = min(float(proposal.read_confidence), sequence_confidence)
        return replace(
            proposal,
            read_byte_ids=tuple(int(value) for value in ids.cpu()),
            read_confidence=confidence,
            read_bbox=tuple(float(value) for value in bbox_values),
            read_focus_source="cursor_fovea",
        )

    def learn_consequence(
        self, before: VisualState, action_kind: int, parameters: tuple[float, float, float, float],
        after: VisualState,
    ) -> dict[str, float]:
        """Backward-compatible direct FAST update for one exact causal transition."""
        with self.model_lock:
            result = self.learner._fast(before, after, action_kind, parameters)
        return {"prediction_loss": float(result["fast_loss"]),
                "temporal_error": float(result["fast_temporal"]),
                "change_error": float(result["fast_change"])}

    def learn_pointer_correction(
        self, *, before_state: VisualState, target_xy: tuple[float, float], repeats: int = 1,
    ) -> dict[str, float | int | str]:
        with self.model_lock:
            return self.learner.learn_pointer_correction(
                before_state=before_state, target_xy=target_xy, repeats=repeats,
            )

    def learn_reading_feedback(
        self, *, bbox: tuple[float, float, float, float],
        read_byte_ids: tuple[int, ...], accepted: bool,
    ) -> dict[str, float | int | str | bool]:
        if self.last_frame is None:
            return {"reader_feedback": "no_frame", "accepted": bool(accepted)}
        with self.model_lock:
            return self.learner.learn_reading_feedback(
                frame=self.last_frame, bbox=bbox,
                read_byte_ids=read_byte_ids, accepted=accepted,
            )

    def learn_transition(
        self, *, before_frame: torch.Tensor, after_frame: torch.Tensor,
        before_state: VisualState, after_state: VisualState, action_kind: int,
        parameters: tuple[float, float, float, float], ownership: str,
        completion: bool | None,
    ) -> dict[str, float | int | str | None]:
        with self.model_lock:
            return self.learner.process_transition(
                before_frame=before_frame, after_frame=after_frame,
                before_state=before_state, after_state=after_state,
                action_kind=action_kind, parameters=parameters,
                ownership=ownership, completion=completion,
            )

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        destination = self.checkpoint if path is None else Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with self.model_lock:
            torch.save({
                "format": 1,
                "config": {field: getattr(self.config, field) for field in self.config.__dataclass_fields__},
                "model": self.model.state_dict(),
                "online_optimizer": self.learner.optimizer.state_dict(),
                "reader_feedback_optimizer": self.learner.reader_feedback_optimizer.state_dict(),
                "online_learning": self.learner.learning_state(),
                "step": self.learner.work_updates + self.learner.slow_updates,
            }, temporary)
            temporary.replace(destination)
            self.learner.save_replay()
        return destination

    def close(self) -> None:
        self.save_checkpoint(); self.portal.close()
