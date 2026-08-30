from __future__ import annotations

from collections import defaultdict, deque
import copy
from dataclasses import dataclass
from pathlib import Path
import random
import tempfile
import time
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from worlds.desktop.neural import NeuralDesktopBody, VisualState


@dataclass(slots=True)
class OnlineExperience:
    before: Tensor
    after: Tensor
    action_kind: int
    parameters: tuple[float, float, float, float]
    ownership: str
    completion: bool | None

    @property
    def bucket(self) -> str:
        outcome = "unknown" if self.completion is None else ("changed" if self.completion else "unchanged")
        return f"{self.action_kind}:{self.ownership}:{outcome}"


class BalancedDesktopReplay:
    """Bounded action/ownership/outcome-stratified CPU replay."""

    def __init__(self, *, capacity: int = 96, per_bucket: int = 16) -> None:
        self.capacity = int(capacity); self.per_bucket = int(per_bucket)
        self._buckets: dict[str, deque[OnlineExperience]] = defaultdict(
            lambda: deque(maxlen=self.per_bucket)
        )

    def __len__(self) -> int:
        return sum(len(values) for values in self._buckets.values())

    def add(self, experience: OnlineExperience) -> None:
        self._buckets[experience.bucket].append(experience)
        while len(self) > self.capacity:
            largest = max(self._buckets, key=lambda key: len(self._buckets[key]))
            self._buckets[largest].popleft()
            if not self._buckets[largest]:
                del self._buckets[largest]

    def sample(self, count: int) -> list[OnlineExperience]:
        active = [values for values in self._buckets.values() if values]
        random.shuffle(active); selected: list[OnlineExperience] = []
        while active and len(selected) < count:
            next_active = []
            for values in active:
                selected.append(random.choice(tuple(values)))
                if len(selected) >= count:
                    break
                if values:
                    next_active.append(values)
            active = next_active
        return selected

    def state_dict(self) -> dict[str, object]:
        return {
            "format": 1, "capacity": self.capacity, "per_bucket": self.per_bucket,
            "items": [item for values in self._buckets.values() for item in values],
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if int(payload.get("format", -1)) != 1:
            raise ValueError("unsupported Desktop replay format")
        self._buckets.clear()
        for item in payload.get("items", ()):  # type: ignore[assignment]
            if isinstance(item, OnlineExperience):
                self.add(item)


class OnlineDesktopLearner:
    """FAST/WORK/SLOW plasticity for the one live neural Desktop model."""

    def __init__(
        self, model: NeuralDesktopBody, *, device: torch.device,
        replay_path: Path | None = None, learning_rate: float = 1e-5,
        replay_capacity: int = 96, work_batch: int = 2, slow_every: int = 32,
    ) -> None:
        self.model = model; self.device = device; self.replay_path = replay_path
        self.replay = BalancedDesktopReplay(capacity=replay_capacity)
        # Consequence learning must not rewrite the action/target policy from its
        # own consequences.  Without an external intended-action label this is a
        # self-confirming loop: a mistaken click changes pixels, then that same
        # change trains the selector toward the mistaken click/target.
        # Keep motor policy plasticity for demonstration/explicit-feedback paths,
        # but exclude it from autonomous FAST/WORK/SLOW optimizers.
        policy_prefixes = (
            "motor_fusion.", "action_head.", "target_query.", "target_key.",
            "offset_head.", "key_head.", "grounding_box.", "trajectory_head.",
        )
        online_parameters = [
            parameter for name, parameter in model.named_parameters()
            if not name.startswith(policy_prefixes)
        ]
        self.optimizer = torch.optim.AdamW(online_parameters, lr=learning_rate, weight_decay=0.01)
        # Explicit human pointer corrections are the one place where the motor
        # target branch receives a direct supervised signal.  Keep this optimizer
        # isolated from autonomous consequence learning so screen change can
        # never self-confirm a bad motor target.
        feedback_parameters = [
            parameter for name, parameter in model.named_parameters()
            if name.startswith(("target_query.", "target_key.", "offset_head."))
        ]
        self.pointer_feedback_optimizer = torch.optim.AdamW(
            feedback_parameters, lr=max(float(learning_rate) * 100.0, 5e-3), weight_decay=0.001
        )
        self.pointer_feedback_updates = 0
        reader_feedback_parameters = [
            parameter for name, parameter in model.named_parameters()
            if name.startswith(("foveal_reader.", "visual_reader_v2."))
        ]
        self.reader_feedback_optimizer = torch.optim.AdamW(
            reader_feedback_parameters, lr=max(float(learning_rate) * 20.0, 2e-4), weight_decay=0.001
        )
        self.reader_feedback_updates = 0
        self.work_batch = max(1, int(work_batch)); self.slow_every = max(1, int(slow_every))
        self.experiences = 0; self.fast_updates = 0; self.work_updates = 0; self.slow_updates = 0
        self.skipped_nonfinite = 0
        self.last_metrics: dict[str, float | int | str | None] = {}
        if replay_path is not None and replay_path.exists():
            try:
                payload = torch.load(replay_path, map_location="cpu", weights_only=False)
                if isinstance(payload, dict): self.replay.load_state_dict(payload)
            except (OSError, RuntimeError, TypeError, ValueError):
                # Corrupt replay is expendable experience, never a reason to lose the organism.
                self.replay = BalancedDesktopReplay(capacity=replay_capacity)

    @staticmethod
    def _compact(frame: Tensor) -> Tensor:
        resized = F.interpolate(frame.detach().float(), size=(144, 256), mode="bilinear", align_corners=False)
        return resized.mul(255).round().clamp(0, 255).to(device="cpu", dtype=torch.uint8)[0]

    @staticmethod
    def _normal_state(state: VisualState) -> VisualState:
        """Clone inference-mode tensors into ordinary tensors usable by autograd."""
        with torch.inference_mode(False):
            return VisualState(
                state.patch_tokens.detach().clone(), state.region_tokens.detach().clone(),
                state.region_positions.detach().clone(), state.salience.detach().clone(),
                state.persistence.detach().clone(), state.latent.detach().clone(),
                state.temporal_state.detach().clone(), state.grid_size,
            )

    def load_optimizer_state(self, payload: object) -> bool:
        if not isinstance(payload, dict): return False
        try:
            self.optimizer.load_state_dict(payload)
            if not self._optimizer_finite():
                # Never carry poisoned Adam moments into a live organism.
                self.optimizer.state.clear()
                return False
            return True
        except (KeyError, RuntimeError, ValueError):
            self.optimizer.state.clear()
            return False

    def load_learning_state(self, payload: object) -> None:
        if not isinstance(payload, dict): return
        self.experiences = max(0, int(payload.get("experiences", 0)))
        self.fast_updates = max(0, int(payload.get("fast_updates", 0)))
        self.work_updates = max(0, int(payload.get("work_updates", 0)))
        self.slow_updates = max(0, int(payload.get("slow_updates", 0)))
        self.skipped_nonfinite = max(0, int(payload.get("skipped_nonfinite", 0)))
        self.pointer_feedback_updates = max(0, int(payload.get("pointer_feedback_updates", 0)))
        self.reader_feedback_updates = max(0, int(payload.get("reader_feedback_updates", 0)))

    def learning_state(self) -> dict[str, int]:
        return {"experiences": self.experiences, "fast_updates": self.fast_updates,
                "work_updates": self.work_updates, "slow_updates": self.slow_updates,
                "skipped_nonfinite": self.skipped_nonfinite,
                "pointer_feedback_updates": self.pointer_feedback_updates,
                "reader_feedback_updates": self.reader_feedback_updates}

    def load_reader_feedback_optimizer_state(self, payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        try:
            self.reader_feedback_optimizer.load_state_dict(payload)
            return True
        except (KeyError, RuntimeError, ValueError):
            self.reader_feedback_optimizer.state.clear()
            return False

    @staticmethod
    def _all_finite(*values: Tensor) -> bool:
        return all(bool(torch.isfinite(value.detach()).all()) for value in values)

    def _optimizer_finite(self) -> bool:
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    return False
        return True

    def _gradients_finite(self) -> bool:
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    return False
        return True

    def _skip_nonfinite(self, stage: str, reason: str, started: float) -> dict[str, float | str]:
        self.optimizer.zero_grad(set_to_none=True)
        self.skipped_nonfinite += 1
        self.model.eval()
        return {f"{stage}_status": "skipped_nonfinite", f"{stage}_nonfinite": reason,
                f"{stage}_updated": False, f"{stage}_parameter_delta": 0.0,
                f"{stage}_latency_ms": (time.perf_counter() - started) * 1000.0}

    def _online_parameter_snapshot(self) -> list[Tensor]:
        return [parameter.detach().clone() for group in self.optimizer.param_groups
                for parameter in group["params"]]

    def _online_parameter_delta(self, before: list[Tensor]) -> float:
        current = [parameter.detach() for group in self.optimizer.param_groups
                   for parameter in group["params"]]
        squared = sum(float(torch.sum((new - old).float().square()).cpu())
                      for old, new in zip(before, current, strict=True))
        return squared ** 0.5

    def learn_pointer_correction(
        self, *, before_state: VisualState, target_xy: tuple[float, float], repeats: int = 1,
    ) -> dict[str, float | int | str]:
        """Learn a bounded human-directed motor target correction.

        The signal says only that the previous target should move in this
        direction.  It is not a reward, a truth label, or an imitation label for
        which action kind should have been chosen.
        """
        started = time.perf_counter()
        x = min(max(float(target_xy[0]), 0.0), 1.0)
        y = min(max(float(target_xy[1]), 0.0), 1.0)
        gh = self.model.config.image_height // 16
        gw = self.model.config.image_width // 16
        col = min(max(int(x * gw), 0), gw - 1)
        row = min(max(int(y * gh), 0), gh - 1)
        cell = row * gw + col
        desired_offset = torch.tensor(
            [[x * gw - (col + 0.5), y * gh - (row + 0.5)]],
            device=self.device, dtype=torch.float32,
        ).clamp(-0.5, 0.5)
        visual = self._normal_state(before_state)
        target_cell = torch.tensor([cell], device=self.device)
        feedback_params = [p for group in self.pointer_feedback_optimizer.param_groups for p in group["params"]]
        # Human correction runs on the live interaction path.  Keep it small:
        # the old 24-step loop could freeze perception/action for seconds on
        # consumer GPUs.  Repeated human corrections are better supervision
        # than one huge synchronous update.
        minimum_steps = min(max(int(repeats), 1), 3)
        maximum_steps = 6
        performed_steps = 0
        initial_selection: float | None = None
        last_loss = last_selection = last_offset = None
        with torch.no_grad():
            before_point = self.model.propose(visual).target_point.detach().clone()
        # Roll back an update that fails to move the predicted target toward the
        # human-provided directional anchor.  This prevents a correction from
        # being recorded as successful merely because gradients were finite.
        feedback_snapshot = [parameter.detach().clone() for parameter in feedback_params]
        optimizer_snapshot = copy.deepcopy(self.pointer_feedback_optimizer.state_dict())
        self.model.train()
        for _ in range(maximum_steps):
            self.pointer_feedback_optimizer.zero_grad(set_to_none=True)
            motor = self.model.propose(visual)
            selection = F.cross_entropy(motor.target_logits.float(), target_cell)
            offset = F.smooth_l1_loss(
                motor.target_offsets[:, cell, :].float(), desired_offset
            )
            loss = selection + offset
            if not self._all_finite(motor.target_logits, motor.target_offsets, selection, offset, loss):
                self.pointer_feedback_optimizer.zero_grad(set_to_none=True)
                self.model.eval(); self.skipped_nonfinite += 1
                return {"pointer_feedback": "skipped_nonfinite"}
            loss.backward()
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in feedback_params):
                self.pointer_feedback_optimizer.zero_grad(set_to_none=True)
                self.model.eval(); self.skipped_nonfinite += 1
                return {"pointer_feedback": "skipped_nonfinite_gradient"}
            torch.nn.utils.clip_grad_norm_(feedback_params, 0.5)
            self.pointer_feedback_optimizer.step()
            self.pointer_feedback_updates += 1
            performed_steps += 1
            last_loss, last_selection, last_offset = loss, selection, offset
            if initial_selection is None:
                initial_selection = float(selection.detach())
            if performed_steps >= minimum_steps and float(selection.detach()) <= max(2.0, initial_selection * 0.5):
                break
        self.model.eval()
        with torch.no_grad():
            after_point = self.model.propose(visual).target_point.detach().clone()
        target = torch.tensor([[x, y]], device=self.device)
        error_before = float(torch.linalg.vector_norm(before_point - target, dim=-1).mean())
        error_after = float(torch.linalg.vector_norm(after_point - target, dim=-1).mean())
        improvement = error_before - error_after
        requested = target - before_point
        requested_norm = torch.linalg.vector_norm(requested, dim=-1, keepdim=True).clamp_min(1e-6)
        requested_direction = requested / requested_norm
        actual_delta = after_point - before_point
        directional_progress = float((actual_delta * requested_direction).sum(dim=-1).mean())
        # A correction is successful only when the decoded motor target actually
        # moves in the human-requested direction.  Falling loss by itself is not
        # enough because target-cell logits can improve without changing argmax.
        if not (directional_progress > 1e-4 and error_after + 1e-5 < error_before):
            with torch.no_grad():
                for parameter, saved in zip(feedback_params, feedback_snapshot, strict=True):
                    parameter.copy_(saved)
            self.pointer_feedback_optimizer.load_state_dict(optimizer_snapshot)
            self.pointer_feedback_updates = max(0, self.pointer_feedback_updates - performed_steps)
            self.model.eval()
            return {
                "pointer_feedback": "rolled_back_no_improvement",
                "pointer_feedback_repeats": performed_steps,
                "pointer_feedback_target_error_before": error_before,
                "pointer_feedback_target_error_after": error_after,
                "pointer_feedback_improvement": improvement,
                "pointer_feedback_directional_progress": directional_progress,
                "pointer_feedback_updates": self.pointer_feedback_updates,
                "pointer_feedback_latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        return {
            "pointer_feedback": "updated_verified",
            "pointer_feedback_loss": float(last_loss.detach()),
            "pointer_feedback_cell_loss": float(last_selection.detach()),
            "pointer_feedback_offset_loss": float(last_offset.detach()),
            "pointer_feedback_repeats": performed_steps,
            "pointer_feedback_updates": self.pointer_feedback_updates,
            "pointer_feedback_target_error_before": error_before,
            "pointer_feedback_target_error_after": error_after,
            "pointer_feedback_improvement": improvement,
            "pointer_feedback_directional_progress": directional_progress,
            "pointer_feedback_latency_ms": (time.perf_counter() - started) * 1000.0,
        }

    def learn_reading_feedback(
        self, *, frame: Tensor, bbox: tuple[float, float, float, float],
        read_byte_ids: tuple[int, ...], accepted: bool,
    ) -> dict[str, float | int | str | bool]:
        """Apply explicit F21/F22 supervision only to the cursor foveal reader."""
        started = time.perf_counter()
        clean: list[int] = []
        for raw in read_byte_ids:
            token = int(raw)
            if token == 2:
                clean.append(token)
                break
            if token >= 4:
                clean.append(token)
        if accepted and not clean:
            return {"reader_feedback": "ignored_empty_prediction", "accepted": True}
        if accepted and clean[-1] != 2:
            clean.append(2)
        box = torch.tensor([bbox], device=self.device, dtype=torch.float32)
        pixels = frame.detach().to(self.device).float()
        parameters = [
            parameter for group in self.reader_feedback_optimizer.param_groups
            for parameter in group["params"]
        ]
        snapshot = [parameter.detach().clone() for parameter in parameters]
        optimizer_snapshot = copy.deepcopy(self.reader_feedback_optimizer.state_dict())
        rejected_ids = rejected_mask = None
        if not accepted:
            self.model.eval()
            with torch.no_grad():
                rejected_logits = self.model.read_region_raw(pixels, box)
                rejected_ids = rejected_logits.argmax(-1)
                rejected_mask = rejected_ids.ne(0)
                if not bool(rejected_mask.any()):
                    rejected_mask = torch.ones_like(rejected_mask, dtype=torch.bool)

        def objective() -> Tensor:
            raw_logits = self.model.read_region_raw(pixels, box)
            if accepted:
                targets = torch.tensor(clean, device=self.device, dtype=torch.long)
                input_lengths = torch.tensor([raw_logits.shape[1]], device=self.device)
                target_lengths = torch.tensor([len(clean)], device=self.device)
                return F.ctc_loss(
                    raw_logits.float().log_softmax(-1).transpose(0, 1), targets,
                    input_lengths, target_lengths, blank=0, zero_infinity=True,
                )
            assert rejected_ids is not None and rejected_mask is not None
            probabilities = raw_logits.float().softmax(-1)
            selected = probabilities.gather(-1, rejected_ids.unsqueeze(-1)).squeeze(-1)[rejected_mask]
            return -torch.log1p(-selected.clamp(max=0.9999)).mean()

        self.model.eval()
        with torch.no_grad():
            before = float(objective())
        self.model.train()
        last_loss = None
        performed_steps = 0
        for _ in range(2):
            self.reader_feedback_optimizer.zero_grad(set_to_none=True)
            loss = objective()
            if not bool(torch.isfinite(loss)):
                break
            loss.backward()
            if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
                break
            torch.nn.utils.clip_grad_norm_(parameters, 0.5)
            self.reader_feedback_optimizer.step()
            self.reader_feedback_updates += 1
            performed_steps += 1
            last_loss = loss
        self.model.eval()
        with torch.no_grad():
            after = float(objective())
        if last_loss is None or not after < before:
            with torch.no_grad():
                for parameter, saved in zip(parameters, snapshot, strict=True):
                    parameter.copy_(saved)
            self.reader_feedback_optimizer.load_state_dict(optimizer_snapshot)
            self.reader_feedback_updates = max(0, self.reader_feedback_updates - performed_steps)
            self.model.eval()
            return {
                "reader_feedback": "rolled_back_no_improvement",
                "accepted": bool(accepted), "loss_before": before, "loss_after": after,
                "reader_feedback_updates": self.reader_feedback_updates,
            }
        return {
            "reader_feedback": "updated_verified",
            "accepted": bool(accepted), "loss_before": before, "loss_after": after,
            "reader_feedback_updates": self.reader_feedback_updates,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }

    def process_transition(
        self, *, before_frame: Tensor, after_frame: Tensor,
        before_state: VisualState, after_state: VisualState,
        action_kind: int, parameters: tuple[float, float, float, float],
        ownership: str, completion: bool | None,
    ) -> dict[str, float | int | str | None]:
        update_started = time.perf_counter()
        metrics: dict[str, float | int | str | None] = {
            "ownership": ownership, "completion": completion,
        }
        if ownership == "heiwa" and completion is not None:
            metrics.update(self._fast(before_state, after_state, action_kind, parameters))
        self.replay.add(OnlineExperience(
            self._compact(before_frame), self._compact(after_frame), int(action_kind),
            tuple(float(value) for value in parameters), ownership, completion,
        ))
        self.experiences += 1
        # Meaningful own transitions update immediately. Passive/external life
        # contributes at a lower cadence to world representation learning.
        if ownership == "heiwa" or self.experiences % 4 == 0:
            metrics.update(self._work(self.work_batch, stage="work"))
            if self.work_updates and self.work_updates % self.slow_every == 0:
                metrics.update(self._work(max(4, self.work_batch), stage="slow"))
        metrics["replay_size"] = len(self.replay)
        metrics["online_total_ms"] = (time.perf_counter() - update_started) * 1000.0
        self.last_metrics = metrics
        return metrics

    def _fast(
        self, before: VisualState, after: VisualState, action_kind: int,
        parameters: tuple[float, float, float, float],
    ) -> dict[str, float]:
        started = time.perf_counter()
        before = self._normal_state(before); after = self._normal_state(after)
        kind = torch.tensor([action_kind], device=self.device)
        values = torch.tensor([parameters], device=self.device)
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        parameter_before = self._online_parameter_snapshot()
        predicted, change_logits = self.model.predict_next(before, kind, values)
        patch_change = 1.0 - F.cosine_similarity(before.patch_tokens, after.patch_tokens, dim=-1)
        temporal = F.smooth_l1_loss(predicted, after.latent.detach())
        change = F.binary_cross_entropy_with_logits(change_logits, patch_change.gt(0.02).float())
        loss = temporal + change
        if not self._all_finite(predicted, change_logits, temporal, change, loss):
            return self._skip_nonfinite("fast", "forward_or_loss", started)
        loss.backward()
        if not self._gradients_finite():
            return self._skip_nonfinite("fast", "gradient", started)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if not self._gradients_finite():
            return self._skip_nonfinite("fast", "gradient_after_clip", started)
        self.optimizer.step(); self.model.eval()
        self.fast_updates += 1
        return {"fast_loss": float(loss.detach()), "fast_temporal": float(temporal.detach()),
                "fast_change": float(change.detach()),
                "fast_updated": True,
                "fast_parameter_delta": self._online_parameter_delta(parameter_before),
                "fast_latency_ms": (time.perf_counter() - started) * 1000.0}

    def _work(self, count: int, *, stage: str) -> dict[str, float]:
        started = time.perf_counter()
        items = self.replay.sample(count)
        if not items: return {}
        height, width = self.model.config.image_height, self.model.config.image_width
        before = torch.stack([item.before for item in items]).to(self.device).float().div_(255)
        after = torch.stack([item.after for item in items]).to(self.device).float().div_(255)
        before = F.interpolate(before, size=(height, width), mode="bilinear", align_corners=False)
        after = F.interpolate(after, size=(height, width), mode="bilinear", align_corners=False)
        actions = torch.tensor([item.action_kind for item in items], device=self.device)
        parameters = torch.tensor([item.parameters for item in items], device=self.device)
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        parameter_before = self._online_parameter_snapshot()
        # Live autonomous learning stays in fp32. Mixed-precision GradScaler state
        # previously made WORK crash after FAST used the same optimizer.
        with torch.autocast(device_type="cuda", enabled=False):
            # Autonomous WORK/SLOW learning has no ground-truth action label.
            # Do not move the shared visual/temporal representation underneath
            # an action_head that receives no corresponding action-selection
            # gradient: that caused motor-policy collapse (e.g. right_click for
            # almost every frame).  Predictor/competence/grounding heads can
            # still learn from detached visual states.
            with torch.no_grad():
                first_observed = self.model.observe(before)
                second_observed = self.model.observe(after, first_observed, actions, parameters)
            first = self._normal_state(first_observed)
            second = self._normal_state(second_observed)
            predicted, change_logits = self.model.predict_next(first, actions, parameters)
            motor = self.model.propose(first)
            difference = (after - before).abs().mean(dim=1)
            patch_target = F.adaptive_avg_pool2d(
                difference.unsqueeze(1), first.grid_size
            ).flatten(1).gt(0.04).float()
            temporal = F.smooth_l1_loss(predicted, second.latent.detach())
            change = F.binary_cross_entropy_with_logits(change_logits, patch_target)
            feature = F.smooth_l1_loss(
                self.model.feature_predictor(first.patch_tokens.mean(1)),
                second.patch_tokens.mean(1).detach(),
            )
            losses = [temporal, change, feature]
            own = torch.tensor([item.ownership == "heiwa" and item.completion is not None for item in items], device=self.device)
            competence = torch.zeros((), device=self.device)
            if bool(own.any()):
                target = torch.tensor([bool(item.completion) for item in items], device=self.device, dtype=torch.float32)
                competence = F.binary_cross_entropy_with_logits(motor.confidence_logits[own], target[own])
                losses.append(competence)
            points: list[Tensor] = []; boxes: list[Tensor] = []; rows: list[int] = []
            for index, item in enumerate(items):
                mask = difference[index].gt(0.03)
                if item.ownership != "heiwa" or item.completion is None or int(mask.sum()) < 4: continue
                yy, xx = mask.nonzero(as_tuple=True)
                points.append(torch.stack((xx.float().mean() / max(width - 1, 1), yy.float().mean() / max(height - 1, 1))))
                boxes.append(torch.stack((xx.min() / max(width - 1, 1), yy.min() / max(height - 1, 1),
                                          xx.max() / max(width - 1, 1), yy.max() / max(height - 1, 1))))
                rows.append(index)
            grounding = torch.zeros((), device=self.device)
            if rows:
                row_tensor = torch.tensor(rows, device=self.device)
                grounding = F.smooth_l1_loss(motor.target_point[row_tensor], torch.stack(points))
                grounding = grounding + F.smooth_l1_loss(motor.target_bbox[row_tensor], torch.stack(boxes))
                losses.append(grounding)
            total = sum(losses)
        finite_terms = [predicted, change_logits, temporal, change, feature, total]
        if bool(own.any()): finite_terms.append(competence)
        if rows: finite_terms.append(grounding)
        if not self._all_finite(*finite_terms):
            return self._skip_nonfinite(stage, "forward_or_loss", started)
        total.backward()
        if not self._gradients_finite():
            return self._skip_nonfinite(stage, "gradient", started)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if not self._gradients_finite():
            return self._skip_nonfinite(stage, "gradient_after_clip", started)
        self.optimizer.step(); self.model.eval()
        if not self._optimizer_finite():
            # A non-finite optimizer state must never be persisted.  Drop moments;
            # model parameters remain protected by the finite forward/gradient checks.
            self.optimizer.state.clear()
        if stage == "slow": self.slow_updates += 1
        else: self.work_updates += 1
        result = {f"{stage}_loss": float(total.detach()), f"{stage}_temporal": float(temporal.detach()),
                  f"{stage}_change": float(change.detach()), f"{stage}_feature": float(feature.detach()),
                  f"{stage}_updated": True,
                  f"{stage}_parameter_delta": self._online_parameter_delta(parameter_before),
                  f"{stage}_latency_ms": (time.perf_counter() - started) * 1000.0}
        if bool(own.any()): result[f"{stage}_competence"] = float(competence.detach())
        if rows:
            result[f"{stage}_grounding"] = float(grounding.detach())
            result["observed_target_x"] = float(torch.stack(points)[0, 0])
            result["observed_target_y"] = float(torch.stack(points)[0, 1])
        return result

    def save_replay(self) -> None:
        if self.replay_path is None: return
        self.replay_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.replay_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(self.replay.state_dict(), temporary); temporary.replace(self.replay_path)
        finally:
            temporary.unlink(missing_ok=True)
