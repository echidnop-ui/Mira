from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ACTION_KINDS = (
    "noop", "move_pointer", "left_click", "double_click", "right_click",
    "scroll", "key_press", "type_text", "key_chord",
)

# Layout-independent Linux evdev codes for printable motor exploration.
# Modifiers and Enter are deliberately excluded: the motor may learn characters
# without accidentally creating shortcuts or submitting focused forms.
KEYBOARD_KEY_CODES = (
    *range(2, 14), *range(16, 28), *range(30, 54), 57,
)


@dataclass(frozen=True, slots=True)
class NeuralDesktopConfig:
    image_height: int = 288
    image_width: int = 512
    hidden_size: int = 192
    token_size: int = 192
    region_count: int = 12
    transformer_layers: int = 2
    transformer_heads: int = 6
    action_embed_size: int = 48
    task_context_size: int = 96
    byte_vocab_size: int = 260
    max_read_bytes: int = 64
    visual_symbol_capacity: int = 64
    visual_symbol_embedding_size: int = 64


@dataclass(slots=True)
class VisualState:
    patch_tokens: Tensor
    region_tokens: Tensor
    region_positions: Tensor
    salience: Tensor
    persistence: Tensor
    latent: Tensor
    temporal_state: Tensor
    grid_size: tuple[int, int]

    def detached(self) -> "VisualState":
        return VisualState(
            self.patch_tokens.detach(), self.region_tokens.detach(),
            self.region_positions.detach(), self.salience.detach(),
            self.persistence.detach(), self.latent.detach(),
            self.temporal_state.detach(), self.grid_size,
        )


@dataclass(slots=True)
class MotorOutput:
    action_logits: Tensor
    target_logits: Tensor
    target_offsets: Tensor
    key_logits: Tensor
    confidence_logits: Tensor
    confidence: Tensor
    target_point: Tensor
    target_bbox: Tensor
    trajectory: Tensor
    motor_state: Tensor


class ConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels * 3, 1), nn.GELU(),
            nn.Conv2d(channels * 3, channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pointwise(self.norm(self.depthwise(x)))


class ByteSequenceReader(nn.Module):
    """CTC-aligned UTF-8 byte reader over horizontal foveal RGB features."""
    def __init__(self, width: int, vocab_size: int, positions: int) -> None:
        super().__init__()
        self.positions = positions
        self.column_projection = nn.Sequential(nn.Linear(width * 8, 128), nn.GELU())
        self.encoder = nn.LSTM(128, 128, num_layers=2, dropout=.1,
                               bidirectional=True, batch_first=True)
        self.norm = nn.LayerNorm(256)
        self.output = nn.Linear(256, vocab_size)

    def forward(self, memory: Tensor, teacher_tokens: Tensor | None = None) -> Tensor:
        del teacher_tokens  # Ground-truth bytes never enter the visual reader.
        encoded, _ = self.encoder(self.column_projection(memory))
        return self.output(self.norm(encoded))

    def collapse(self, raw_logits: Tensor) -> Tensor:
        """Expose CTC greedy bytes through the existing fixed logits interface."""
        batch, _, vocab = raw_logits.shape
        result = raw_logits.new_full((batch, self.positions, vocab), -30.0)
        result[:, :, 0] = 0.0
        ids = raw_logits.argmax(-1)
        for b in range(batch):
            selected: list[int] = []; previous = -1
            for index, token in enumerate(ids[b].tolist()):
                if token != 0 and token != previous: selected.append(index)
                previous = token
            for position, index in enumerate(selected[:self.positions]):
                result[b, position] = raw_logits[b, index]
        return result


class VisualByteDecoder(nn.Module):
    """Monotonic CTC UTF-8 decoder over foveal visual columns."""

    def __init__(self, width: int, vocab_size: int, positions: int, hidden: int = 192) -> None:
        super().__init__()
        self.positions = positions; self.vocab_size = vocab_size
        self.column_projection = nn.Sequential(
            nn.Linear(width * 8, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.column_positions = nn.Parameter(torch.randn(128, hidden) * 0.01)
        self.encoder = nn.GRU(hidden, hidden, num_layers=2, dropout=.1,
                              bidirectional=True, batch_first=True)
        self.norm = nn.LayerNorm(hidden * 2)
        self.output = nn.Linear(hidden * 2, vocab_size)

    def forward(self, memory: Tensor, teacher_tokens: Tensor | None = None) -> Tensor:
        del teacher_tokens
        columns = self.column_projection(memory)
        encoded, _ = self.encoder(columns + self.column_positions[:columns.shape[1]])
        return self.output(self.norm(encoded))

    def collapse(self, raw_logits: Tensor) -> Tensor:
        batch, _, vocab = raw_logits.shape
        result = raw_logits.new_full((batch, self.positions, vocab), -30.0)
        result[:, :, 0] = 0.0
        ids = raw_logits.argmax(-1)
        for b in range(batch):
            selected: list[int] = []; previous = -1
            for index, token in enumerate(ids[b].tolist()):
                if token != 0 and token != previous: selected.append(index)
                previous = token
            for position, index in enumerate(selected[:self.positions]):
                result[b, position] = raw_logits[b, index]
        return result


class NeuralDesktopBody(nn.Module):
    """Trainable pixel vision, temporal state, world model and spatial motor policy."""

    def __init__(self, config: NeuralDesktopConfig = NeuralDesktopConfig()) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, 4, 4), nn.GELU(), ConvBlock(48),
            nn.Conv2d(48, 96, 2, 2), nn.GELU(), ConvBlock(96),
            nn.Conv2d(96, d, 2, 2), nn.GELU(), ConvBlock(d),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d, config.transformer_heads, d * 3, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.spatial = nn.TransformerEncoder(encoder_layer, config.transformer_layers)
        self.position_projection = nn.Linear(2, d)
        self.region_queries = nn.Parameter(torch.randn(config.region_count, d) * 0.02)
        self.region_attention = nn.MultiheadAttention(
            d, config.transformer_heads, batch_first=True,
        )
        self.region_geometry = nn.Linear(d, 4)
        self.region_salience = nn.Linear(d, 1)
        self.temporal = nn.GRUCell(d + config.action_embed_size, d)
        self.action_embedding = nn.Embedding(len(ACTION_KINDS), config.action_embed_size)
        self.action_parameters = nn.Linear(4, config.action_embed_size)
        self.action_head = nn.Linear(d, len(ACTION_KINDS))
        self.task_projection = nn.Linear(config.task_context_size, d)
        self.motor_fusion = nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, d), nn.GELU())
        self.target_query = nn.Linear(d, d)
        self.target_key = nn.Linear(d, d)
        self.offset_head = nn.Linear(d, 2)
        self.key_head = nn.Linear(d, len(KEYBOARD_KEY_CODES))
        self.confidence_head = nn.Linear(d, 1)
        self.next_latent = nn.Sequential(nn.Linear(d + config.action_embed_size, d), nn.GELU(), nn.Linear(d, d))
        self.change_query = nn.Linear(d + config.action_embed_size, d)
        self.feature_predictor = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.grounding_box = nn.Linear(d, 4)
        self.trajectory_head = nn.Linear(d, 3)
        # The coarse encoder stays at 1/16 resolution. Only one selected crop
        # enters this small high-resolution pixel reader.
        self.foveal_reader = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=(2, 1), padding=1), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=(2, 1), padding=1), nn.GELU(),
        )
        self.byte_reader = ByteSequenceReader(
            96, config.byte_vocab_size, config.max_read_bytes + 2,
        )
        # V2 is intentionally a new checkpoint namespace.  Legacy collapsed
        # reader tensors can be loaded without shape surgery and are ignored.
        self.visual_reader_v2 = VisualByteDecoder(
            96, config.byte_vocab_size, config.max_read_bytes + 2,
        )
        # Reading focus is independent from the motor target.  It learns which
        # visible region contains the current textual observation.
        self.reading_query = nn.Parameter(torch.randn(1, d) * 0.02)
        self.reading_key = nn.Linear(d, d)
        self.reading_offset = nn.Linear(d, 2)
        self.reading_size = nn.Linear(d, 2)
        self.reading_confidence = nn.Linear(d, 1)
        # Pixel-native visual concepts. Symbol labels exist only in curriculum
        # losses; normal inference receives RGB and (for find) an RGB query crop.
        symbol_d = config.visual_symbol_embedding_size
        self.symbol_attention = nn.Linear(d, 1)
        self.symbol_projection = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, symbol_d), nn.GELU(),
            nn.Linear(symbol_d, symbol_d),
        )
        self.symbol_classifier = nn.Linear(symbol_d, config.visual_symbol_capacity)
        self.symbol_logit_scale = nn.Parameter(torch.tensor(2.3025851))
        self.symbol_find_query = nn.Linear(symbol_d, symbol_d)
        self.symbol_find_key = nn.Linear(d, symbol_d)
        self.symbol_find_scale = nn.Parameter(torch.tensor(2.3025851))
        self.symbol_find_offset = nn.Linear(d, 2)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode_action(
        self, action_kind: Tensor | None, action_parameters: Tensor | None,
        *, batch: int, device: torch.device,
    ) -> Tensor:
        if action_kind is None:
            return torch.zeros(batch, self.config.action_embed_size, device=device)
        encoded = self.action_embedding(action_kind.long())
        if action_parameters is not None:
            encoded = encoded + self.action_parameters(action_parameters.float())
        return encoded

    def observe(
        self, frame: Tensor, previous_state: VisualState | None = None,
        last_action_kind: Tensor | None = None,
        last_action_parameters: Tensor | None = None,
    ) -> VisualState:
        if frame.ndim != 4 or frame.shape[1] != 3:
            raise ValueError("frame must have shape [batch, 3, height, width]")
        features = self.stem(frame.float())
        batch, channels, gh, gw = features.shape
        patches = features.flatten(2).transpose(1, 2)
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, gh, device=frame.device),
            torch.linspace(0, 1, gw, device=frame.device), indexing="ij",
        )
        positions = torch.stack((xx, yy), dim=-1).reshape(1, gh * gw, 2)
        patches = self.spatial(patches + self.position_projection(positions))
        queries = self.region_queries.unsqueeze(0).expand(batch, -1, -1)
        regions, weights = self.region_attention(queries, patches, patches)
        geometry = self.region_geometry(regions).sigmoid()
        salience = self.region_salience(regions).sigmoid().squeeze(-1)
        action = self.encode_action(last_action_kind, last_action_parameters, batch=batch, device=frame.device)
        pooled = regions.mean(dim=1)
        prior = torch.zeros_like(pooled) if previous_state is None else previous_state.temporal_state
        temporal = self.temporal(torch.cat((pooled, action), dim=-1), prior)
        if previous_state is None:
            persistence = torch.zeros_like(salience)
        else:
            similarity = F.cosine_similarity(
                regions[:, :, None, :], previous_state.region_tokens[:, None, :, :], dim=-1,
            )
            persistence = similarity.max(dim=-1).values.sigmoid()
        return VisualState(
            patch_tokens=patches, region_tokens=regions,
            region_positions=geometry, salience=salience,
            persistence=persistence, latent=temporal,
            temporal_state=temporal, grid_size=(gh, gw),
        )

    def propose(self, visual: VisualState, task_context: Tensor | None = None) -> MotorOutput:
        grounded = visual.latent
        if task_context is not None:
            grounded = grounded + self.task_projection(task_context.float())
        query = self.target_query(grounded).unsqueeze(1)
        keys = self.target_key(visual.patch_tokens)
        logits = (query * keys).sum(dim=-1) / self.config.hidden_size ** 0.5
        target_token = torch.einsum("bn,bnd->bd", logits.softmax(-1), visual.patch_tokens)
        motor_state = self.motor_fusion(torch.cat((grounded, target_token), dim=-1))
        confidence_logits = self.confidence_head(motor_state).squeeze(-1)
        bbox_raw = self.grounding_box(motor_state).sigmoid()
        center, size = bbox_raw[:, :2], 0.02 + 0.98 * bbox_raw[:, 2:]
        bbox = torch.cat(((center - size / 2).clamp(0, 1), (center + size / 2).clamp(0, 1)), dim=-1)
        point = self.decode_normalized_target(logits, self.offset_head(visual.patch_tokens))
        return MotorOutput(
            action_logits=self.action_head(motor_state), target_logits=logits,
            target_offsets=torch.tanh(self.offset_head(visual.patch_tokens)) * 0.5,
            key_logits=self.key_head(motor_state),
            confidence_logits=confidence_logits,
            confidence=confidence_logits.sigmoid(),
            target_point=point, target_bbox=bbox,
            trajectory=torch.tanh(self.trajectory_head(motor_state)), motor_state=motor_state,
        )

    def decode_normalized_target(self, logits: Tensor, raw_offsets: Tensor) -> Tensor:
        index = logits.argmax(dim=-1); gh = self.config.image_height // 16; gw = self.config.image_width // 16
        row, col = torch.div(index, gw, rounding_mode="floor"), index % gw
        offsets = torch.tanh(raw_offsets[torch.arange(index.shape[0], device=index.device), index]) * 0.5
        return torch.stack(((col.float() + 0.5 + offsets[:, 0]) / gw,
                            (row.float() + 0.5 + offsets[:, 1]) / gh), dim=-1).clamp(0, 1)

    def read_region_raw(self, frame: Tensor, bbox: Tensor, teacher_tokens: Tensor | None = None) -> Tensor:
        """Foveate one normalized bbox and emit Language-v5-compatible byte logits."""
        batch = frame.shape[0]
        unit_x = torch.linspace(0, 1, 128, device=frame.device).view(1, 1, 128)
        unit_y = torch.linspace(0, 1, 32, device=frame.device).view(1, 32, 1)
        x0, y0, x1, y1 = bbox.unbind(dim=-1)
        xs = x0[:, None, None] + (x1 - x0)[:, None, None] * unit_x
        ys = y0[:, None, None] + (y1 - y0)[:, None, None] * unit_y
        xx = xs.expand(batch, 32, 128); yy = ys.expand(batch, 32, 128)
        grid = torch.stack((xx * 2 - 1, yy * 2 - 1), dim=-1)
        crop = F.grid_sample(frame.float(), grid, align_corners=True)
        features = self.foveal_reader(crop).permute(0, 3, 1, 2).flatten(2)
        return self.visual_reader_v2(features, teacher_tokens)

    def read_region(self, frame: Tensor, bbox: Tensor, teacher_tokens: Tensor | None = None) -> Tensor:
        return self.visual_reader_v2.collapse(self.read_region_raw(frame, bbox, teacher_tokens))

    def locate_reading_region(self, visual: VisualState) -> tuple[Tensor, Tensor]:
        """Predict one normalized pixel-reading bbox without using motor state."""
        batch, patches, _ = visual.patch_tokens.shape
        query = self.reading_query.expand(batch, -1).unsqueeze(1)
        logits = (query * self.reading_key(visual.patch_tokens)).sum(-1) / self.config.hidden_size ** .5
        weights = logits.softmax(-1)
        gh, gw = visual.grid_size
        yy, xx = torch.meshgrid(
            torch.linspace(.5 / gh, 1 - .5 / gh, gh, device=weights.device),
            torch.linspace(.5 / gw, 1 - .5 / gw, gw, device=weights.device),
            indexing="ij",
        )
        positions = torch.stack((xx, yy), -1).reshape(1, patches, 2)
        token = torch.einsum("bn,bnd->bd", weights, visual.patch_tokens)
        center = (torch.einsum("bn,bnd->bd", weights, positions)
                  + torch.tanh(self.reading_offset(token)) * .15).clamp(0, 1)
        size = (.03 + .94 * self.reading_size(token).sigmoid()).clamp(.03, .97)
        bbox = torch.cat(((center - size / 2).clamp(0, 1),
                          (center + size / 2).clamp(0, 1)), -1)
        return bbox, self.reading_confidence(token).squeeze(-1).sigmoid()

    def visual_symbol(self, frame_or_visual: Tensor | VisualState) -> tuple[Tensor, Tensor]:
        """Build a position-tolerant symbol concept from RGB-derived features."""
        visual = self.observe(frame_or_visual) if isinstance(frame_or_visual, Tensor) else frame_or_visual
        weights = self.symbol_attention(visual.patch_tokens).squeeze(-1).softmax(-1)
        pooled = torch.einsum("bn,bnd->bd", weights, visual.patch_tokens)
        embedding = F.normalize(self.symbol_projection(pooled), dim=-1)
        classifier = F.normalize(self.symbol_classifier.weight, dim=-1)
        scale = self.symbol_logit_scale.exp().clamp(1.0, 100.0)
        logits = F.linear(embedding, classifier, self.symbol_classifier.bias) * scale
        return embedding, logits

    def find_visual_symbol(
        self, canvas: Tensor | VisualState, query_embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Ground an RGB-query symbol into the shared pixel patch grid."""
        visual = self.observe(canvas) if isinstance(canvas, Tensor) else canvas
        keys = F.normalize(self.symbol_find_key(visual.patch_tokens), dim=-1)
        offsets = torch.tanh(self.symbol_find_offset(visual.patch_tokens)) * 0.5
        gh, gw = visual.grid_size
        # Query crops and canvas patches meet in the same visual-concept space;
        # an extra unconstrained query rotation made localization needlessly
        # underdetermined on small curricula.
        query = F.normalize(query_embedding, dim=-1).unsqueeze(1)
        scale = self.symbol_find_scale.exp().clamp(1.0, 100.0)
        logits = (query * keys).sum(-1) * scale
        yy, xx = torch.meshgrid(
            torch.arange(gh, device=logits.device, dtype=logits.dtype),
            torch.arange(gw, device=logits.device, dtype=logits.dtype), indexing="ij",
        )
        cells = torch.stack(((xx.flatten()[None] + .5 + offsets[..., 0]) / gw,
                             (yy.flatten()[None] + .5 + offsets[..., 1]) / gh), -1)
        expected_point = torch.einsum("bn,bnd->bd", logits.softmax(-1), cells)
        selected = cells[torch.arange(logits.shape[0], device=logits.device), logits.argmax(-1)]
        # Execute the concrete top-1 target while retaining gradients from the
        # full spatial distribution during curriculum learning.
        point = (selected.detach() + expected_point - expected_point.detach()).clamp(0, 1)
        return logits, point

    def predict_next(self, visual: VisualState, action_kind: Tensor, action_parameters: Tensor) -> tuple[Tensor, Tensor]:
        action = self.encode_action(action_kind, action_parameters, batch=visual.latent.shape[0], device=visual.latent.device)
        conditioned = torch.cat((visual.latent, action), dim=-1)
        next_latent = self.next_latent(conditioned)
        change_logits = torch.einsum("bd,bnd->bn", self.change_query(conditioned), visual.patch_tokens)
        return next_latent, change_logits

    def decode_target(self, output: MotorOutput, *, screen_width: int, screen_height: int, target_index: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if target_index is None:
            return output.target_point[:, 0] * (screen_width - 1), output.target_point[:, 1] * (screen_height - 1)
        index = output.target_logits.argmax(dim=-1) if target_index is None else target_index.long()
        gh = self.config.image_height // 16
        gw = self.config.image_width // 16
        row, col = torch.div(index, gw, rounding_mode="floor"), index % gw
        offsets = output.target_offsets[torch.arange(index.shape[0], device=index.device), index]
        x = ((col.float() + 0.5 + offsets[:, 0]) / gw * screen_width).clamp(0, screen_width - 1)
        y = ((row.float() + 0.5 + offsets[:, 1]) / gh * screen_height).clamp(0, screen_height - 1)
        return x, y

    def training_losses(self, before: Tensor, after: Tensor, action_kind: Tensor, action_parameters: Tensor, target_cell: Tensor | None = None, success_target: Tensor | None = None, selection_feedback: Tensor | None = None, target_xy: Tensor | None = None, target_bbox: Tensor | None = None, read_tokens: Tensor | None = None, trajectory_target: Tensor | None = None, read_bbox: Tensor | None = None) -> Mapping[str, Tensor]:
        first = self.observe(before)
        second = self.observe(after, first, action_kind, action_parameters)
        predicted, change_logits = self.predict_next(first, action_kind, action_parameters)
        motor = self.propose(first)
        actual_change = (F.adaptive_avg_pool2d((after - before).abs().mean(1, keepdim=True), first.grid_size).flatten(1) > 0.04).float()
        action_loss = F.cross_entropy(motor.action_logits, action_kind.long())
        if selection_feedback is not None:
            explicit = torch.isfinite(selection_feedback)
            if bool(explicit.any()):
                action_loss = self._rated_choice_loss(
                    motor.action_logits[explicit], action_kind.long()[explicit], selection_feedback[explicit]
                )
        losses: dict[str, Tensor] = {
            "temporal_prediction": F.smooth_l1_loss(predicted, second.latent.detach()),
            "change_prediction": F.binary_cross_entropy_with_logits(change_logits, actual_change),
            "temporal_consistency": 1.0 - F.cosine_similarity(first.latent, second.latent, dim=-1).mean(),
            "feature_prediction": F.smooth_l1_loss(self.feature_predictor(first.patch_tokens.mean(1)), second.patch_tokens.mean(1).detach()),
            "action_imitation": action_loss,
        }
        if target_xy is not None:
            supervised_points = torch.isfinite(target_xy).all(dim=-1)
            if bool(supervised_points.any()):
                losses["target_point"] = F.smooth_l1_loss(
                    motor.target_point[supervised_points], target_xy[supervised_points]
                )
        if target_bbox is not None:
            supervised_boxes = torch.isfinite(target_bbox).all(dim=-1)
            if bool(supervised_boxes.any()):
                losses["target_bbox"] = F.smooth_l1_loss(
                    motor.target_bbox[supervised_boxes], target_bbox[supervised_boxes]
                )
        if read_tokens is not None:
            readable = read_tokens.ge(0).any(dim=-1)
            if bool(readable.any()):
                if read_bbox is None:
                    boxes = motor.target_bbox[readable]
                else:
                    boxes = torch.where(torch.isfinite(read_bbox[readable]), read_bbox[readable], motor.target_bbox[readable])
                    predicted_read_boxes, predicted_read_confidence = self.locate_reading_region(first)
                    supervised_read_boxes = torch.isfinite(read_bbox).all(-1)
                    if bool(supervised_read_boxes.any()):
                        losses["reading_focus"] = F.smooth_l1_loss(
                            predicted_read_boxes[supervised_read_boxes], read_bbox[supervised_read_boxes]
                        ) - .1 * torch.log(
                            predicted_read_confidence[supervised_read_boxes].float().clamp_min(1e-6)
                        ).mean()
                byte_logits = self.read_region_raw(before[readable], boxes)
                tokens = read_tokens[readable]
                lengths = tokens.ge(0).sum(-1)
                flat_targets = torch.cat([row[:length] for row, length in zip(tokens, lengths.tolist())])
                input_lengths = torch.full_like(lengths, byte_logits.shape[1])
                losses["foveal_byte_reading"] = F.ctc_loss(
                    byte_logits.float().log_softmax(-1).transpose(0, 1), flat_targets,
                    input_lengths, lengths, blank=0, zero_infinity=True,
                )
        if trajectory_target is not None:
            trajectory_supervised = torch.isfinite(trajectory_target).all(dim=-1)
            if bool(trajectory_supervised.any()):
                losses["trajectory"] = F.smooth_l1_loss(
                    motor.trajectory[trajectory_supervised], trajectory_target[trajectory_supervised]
                )
        correct = motor.action_logits.detach().argmax(dim=-1).eq(action_kind.long())
        if target_cell is not None:
            spatial = target_cell >= 0
            predicted_cell = motor.target_logits.detach().argmax(dim=-1)
            gw = self.config.image_width // 16
            predicted_row, predicted_col = predicted_cell // gw, predicted_cell % gw
            target_row, target_col = target_cell.clamp_min(0) // gw, target_cell.clamp_min(0) % gw
            near_target = (predicted_row - target_row).abs().le(1) & (predicted_col - target_col).abs().le(1)
            correct = correct & (~spatial | near_target)
        derived_success = correct.float()
        if success_target is None:
            success_target = derived_success
        else:
            success_target = torch.where(torch.isfinite(success_target), success_target, derived_success)
        if success_target is not None:
            losses["confidence_calibration"] = F.binary_cross_entropy_with_logits(
                motor.confidence_logits, success_target.float().clamp(0, 1)
            )
        if target_cell is not None:
            supervised = target_cell >= 0
            if bool(supervised.any()):
                spatial_feedback = None if selection_feedback is None else selection_feedback[supervised]
                if spatial_feedback is not None and bool(torch.isfinite(spatial_feedback).any()):
                    explicit = torch.isfinite(spatial_feedback)
                    losses["motor_imitation"] = self._rated_choice_loss(
                        motor.target_logits[supervised][explicit], target_cell[supervised].long()[explicit], spatial_feedback[explicit]
                    )
                else:
                    losses["motor_imitation"] = F.cross_entropy(
                        motor.target_logits[supervised], target_cell[supervised].long()
                    )
        key_actions = action_kind.eq(ACTION_KINDS.index("key_press")) & action_parameters[:, 1].gt(0.5)
        if bool(key_actions.any()):
            key_codes = action_parameters[key_actions, 0].long()
            key_indices = torch.full_like(key_codes, -1)
            for index, code in enumerate(KEYBOARD_KEY_CODES):
                key_indices[key_codes.eq(code)] = index
            known = key_indices.ge(0)
            if bool(known.any()):
                key_logits = motor.key_logits[key_actions][known]
                key_targets = key_indices[known]
                key_feedback = None if selection_feedback is None else selection_feedback[key_actions][known]
                if key_feedback is not None and bool(torch.isfinite(key_feedback).any()):
                    explicit = torch.isfinite(key_feedback)
                    losses["key_imitation"] = self._rated_choice_loss(
                        key_logits[explicit], key_targets[explicit], key_feedback[explicit]
                    )
                else:
                    losses["key_imitation"] = F.cross_entropy(key_logits, key_targets)
        losses["total"] = sum(losses.values())
        return losses

    @staticmethod
    def _rated_choice_loss(logits: Tensor, chosen: Tensor, feedback: Tensor) -> Tensor:
        """Positive feedback imitates a choice; negative feedback suppresses it."""
        positive = feedback >= 0.5
        terms: list[Tensor] = []
        if bool(positive.any()):
            terms.append(F.cross_entropy(logits[positive], chosen[positive], reduction="none"))
        negative = ~positive
        if bool(negative.any()):
            probabilities = logits[negative].float().softmax(dim=-1)
            selected = probabilities.gather(1, chosen[negative].unsqueeze(1)).squeeze(1)
            terms.append(-torch.log1p(-selected.clamp(max=1.0 - 1e-6)))
        return torch.cat(terms).mean()


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    # ROCm intentionally uses PyTorch's CUDA-compatible API.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
