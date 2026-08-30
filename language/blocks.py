from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from language.ssm import LanguageSSMBlock
from language.stateweave import StateWeaveBlock, StateWeaveState


class GatedRecurrentMemoryBlock(nn.Module):
    """xLSTM-inspired normalized gated memory; not an official xLSTM."""

    def __init__(self, width: int, *, dropout: float, bidirectional: bool = False) -> None:
        super().__init__()
        self.width = width
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(width)
        self.gates = nn.Linear(width, width * 4)
        self.memory_projection = nn.Linear(width, width)
        self.output_projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def _scan(
        self,
        sequence: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, length, width = sequence.shape
        if state is None:
            memory = sequence.new_zeros(batch, width)
            normalizer = sequence.new_ones(batch, width)
        else:
            memory, normalizer = state
        projected = self.gates(sequence)
        input_gate, forget_gate, candidate, output_gate = projected.chunk(4, dim=-1)
        values: list[Tensor] = []
        for index in range(length):
            incoming = torch.sigmoid(input_gate[:, index, :])
            retained = torch.sigmoid(forget_gate[:, index, :])
            previous_weight = retained * normalizer
            normalizer = previous_weight + incoming + 1.0e-6
            memory = (
                previous_weight * memory
                + incoming * torch.tanh(candidate[:, index, :])
            ) / normalizer
            values.append(
                torch.sigmoid(output_gate[:, index, :])
                * self.memory_projection(memory)
            )
        return torch.stack(values, dim=1), (memory, normalizer)

    def forward(self, x: Tensor) -> Tensor:
        value = self.norm(x)
        forward, _ = self._scan(value)
        if self.bidirectional:
            backward, _ = self._scan(torch.flip(value, dims=(1,)))
            forward = 0.5 * (forward + torch.flip(backward, dims=(1,)))
        return x + self.dropout(self.output_projection(forward))

    def step(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        value, state = self._scan(self.norm(x), state)
        return x + self.output_projection(value), state


class LocalAttentionBlock(nn.Module):
    """Windowed attention whose sequence cost is O(length * window)."""

    def __init__(
        self,
        width: int,
        heads: int,
        *,
        window: int,
        causal: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("attention width must be divisible by heads")
        if window <= 0:
            raise ValueError("attention window must be > 0")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.window = window
        self.causal = causal
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def _split(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.heads, self.head_width).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        value = self.norm(x)
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        length = x.shape[1]
        outputs: list[Tensor] = []
        chunk_size = self.window
        for start in range(0, length, chunk_size):
            end = min(length, start + chunk_size)
            key_start = max(0, start - self.window + 1)
            key_end = end if self.causal else min(length, end + self.window - 1)
            query = q[:, :, start:end, :]
            keys = k[:, :, key_start:key_end, :]
            values = v[:, :, key_start:key_end, :]
            query_positions = torch.arange(start, end, device=x.device).unsqueeze(1)
            key_positions = torch.arange(key_start, key_end, device=x.device).unsqueeze(0)
            if self.causal:
                allowed = (key_positions <= query_positions) & (
                    query_positions - key_positions < self.window
                )
            else:
                allowed = (query_positions - key_positions).abs() < self.window
            mask = torch.zeros_like(allowed, dtype=x.dtype)
            mask = mask.masked_fill(~allowed, -torch.inf)
            attended = F.scaled_dot_product_attention(
                query,
                keys,
                values,
                attn_mask=mask,
                dropout_p=0.0,
            )
            outputs.append(attended)
        merged = torch.cat(outputs, dim=2).transpose(1, 2).contiguous()
        merged = merged.view(x.shape[0], length, self.width)
        return x + self.dropout(self.output(merged))

    def step(self, x: Tensor, history: Tensor | None) -> tuple[Tensor, Tensor]:
        value = self.norm(x)
        combined = value if history is None else torch.cat((history, value), dim=1)
        combined = combined[:, -self.window :, :]
        q = self._split(self.qkv(value).chunk(3, dim=-1)[0])
        projected = self.qkv(combined)
        _, k, v = projected.chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(
            q,
            self._split(k),
            self._split(v),
            dropout_p=0.0,
        )
        merged = attended.transpose(1, 2).contiguous().view(x.shape[0], 1, self.width)
        return x + self.output(merged), combined.detach()


@dataclass(slots=True)
class HybridBlockState:
    stateweave: StateWeaveState | None = None
    ssm: object | None = None
    recurrent: tuple[Tensor, Tensor] | None = None
    attention_history: Tensor | None = None


class HybridLanguageBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        *,
        attention_window: int,
        ssm_backend: str,
        causal: bool,
        dropout: float,
        expansion: int,
        stateweave_plastic_rank: int = 32,
        stateweave_dynamic_rank: int = 32,
    ) -> None:
        super().__init__()
        self.stateweave = StateWeaveBlock(
            width,
            dropout=dropout,
            plastic_rank=stateweave_plastic_rank,
            dynamic_rank=stateweave_dynamic_rank,
            bidirectional=not causal,
        )
        self.ssm = LanguageSSMBlock(
            width,
            backend=ssm_backend,
            dropout=dropout,
            bidirectional=not causal and ssm_backend == "selective_ssm",
        )
        self.recurrent = GatedRecurrentMemoryBlock(
            width, dropout=dropout, bidirectional=not causal
        )
        self.attention = LocalAttentionBlock(
            width,
            heads,
            window=attention_window,
            causal=causal,
            dropout=dropout,
        )
        self.fusion_gate = nn.Linear(width, 4)
        # StateWeave is the primary temporal prior, but the preference is fully
        # trainable.  Other branches remain active from initialization.
        nn.init.zeros_(self.fusion_gate.weight)
        with torch.no_grad():
            self.fusion_gate.bias.copy_(
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=self.fusion_gate.bias.dtype)
            )
        self.fusion_projection = nn.Linear(width, width)
        self.fusion_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * expansion, width),
            nn.Dropout(dropout),
        )

    def _fuse(self, x: Tensor, branches: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        weights = torch.softmax(self.fusion_gate(x), dim=-1)
        # Every branch is a residual block and already contains x.  Route the
        # innovations rather than blending four duplicate identity paths.
        deltas = tuple(branch - x for branch in branches)
        stacked = torch.stack(deltas, dim=-2)
        fused_delta = (stacked * weights.unsqueeze(-1)).sum(dim=-2)
        value = self.fusion_norm(x + self.fusion_projection(fused_delta))
        return value + self.ffn(value)

    def forward(self, x: Tensor) -> Tensor:
        return self._fuse(
            x,
            (self.stateweave(x), self.ssm(x), self.recurrent(x), self.attention(x)),
        )

    def step(self, x: Tensor, state: HybridBlockState) -> tuple[Tensor, HybridBlockState]:
        stateweave, stateweave_state = self.stateweave.step(x, state.stateweave)
        ssm, ssm_state = self.ssm.step(x, state.ssm)
        recurrent, recurrent_state = self.recurrent.step(x, state.recurrent)
        attention, history = self.attention.step(x, state.attention_history)
        return self._fuse(x, (stateweave, ssm, recurrent, attention)), HybridBlockState(
            stateweave=stateweave_state,
            ssm=ssm_state,
            recurrent=recurrent_state,
            attention_history=history,
        )


class LanguageWorkspace(nn.Module):
    def __init__(self, width: int, output_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(width, 1)
        self.summary = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, output_size),
        )

    def forward(self, sequence: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        scores = self.score(sequence).squeeze(-1) / sqrt(sequence.shape[-1])
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.summary(pooled)
