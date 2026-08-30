from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn
import torch.nn.functional as F


EXPERT_NAMES: Final[tuple[str, ...]] = (
    "ssm",
    "recurrent",
    "tcn",
    "attention",
)


class SelectiveSSMExpert(nn.Module):
    """Persistent selective state-space temporal expert.

    This is deliberately pure PyTorch and kernel-portable.  The expert boundary
    is stable so a Mamba/Mamba-2 implementation can replace or join it later
    without changing PersonalCognitiveCore or its public interface.
    """

    def __init__(self, size: int, *, depth: int, dropout: float) -> None:
        super().__init__()
        self.size = int(size)
        self.depth = int(depth)
        self.layers = nn.ModuleList(
            [_SelectiveSSMLayer(size, dropout=dropout) for _ in range(depth)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(size),
            nn.Linear(size, size),
            nn.SiLU(),
        )

    def initial_state(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(
            (self.depth, 1, self.size),
            device=device,
            dtype=dtype,
        )

    def forward_step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        if tuple(state.shape) != (self.depth, 1, self.size):
            raise ValueError("invalid SSM expert state shape")

        y = x
        next_states: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            y, next_state = layer(y, state[index])
            next_states.append(next_state)

        return self.out(y), torch.stack(next_states, dim=0)


class _SelectiveSSMLayer(nn.Module):
    def __init__(self, size: int, *, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.candidate = nn.Linear(size, size)
        self.delta = nn.Linear(size, size)
        self.input_gate = nn.Linear(size, size)
        self.output_gate = nn.Linear(size, size)
        self.residual = nn.Linear(size, size)
        self.a_log = nn.Parameter(torch.zeros(size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        z = self.norm(x)
        candidate = torch.tanh(self.candidate(z))
        delta = F.softplus(self.delta(z)) + 1e-4
        decay_rate = F.softplus(self.a_log).unsqueeze(0) + 1e-4
        decay = torch.exp(-decay_rate * delta)
        write = torch.sigmoid(self.input_gate(z))
        next_state = decay * state + (1.0 - decay) * write * candidate
        value = torch.sigmoid(self.output_gate(z)) * next_state
        y = x + self.dropout(value + self.residual(z))
        return y, next_state


class XLSTMStyleExpert(nn.Module):
    """Persistent stabilized recurrent expert inspired by xLSTM-style gating."""

    def __init__(self, size: int, *, depth: int, dropout: float) -> None:
        super().__init__()
        self.size = int(size)
        self.depth = int(depth)
        self.layers = nn.ModuleList(
            [_StabilizedRecurrentLayer(size, dropout=dropout) for _ in range(depth)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(size),
            nn.Linear(size, size),
            nn.SiLU(),
        )

    def initial_state(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        shape = (self.depth, 1, self.size)
        zeros = torch.zeros(shape, device=device, dtype=dtype)
        m = torch.full(shape, -30.0, device=device, dtype=dtype)
        return zeros.clone(), zeros.clone(), zeros.clone(), m

    def forward_step(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        h_state, c_state, n_state, m_state = state
        expected = (self.depth, 1, self.size)
        if any(tuple(value.shape) != expected for value in state):
            raise ValueError("invalid recurrent expert state shape")

        y = x
        next_h: list[Tensor] = []
        next_c: list[Tensor] = []
        next_n: list[Tensor] = []
        next_m: list[Tensor] = []

        for index, layer in enumerate(self.layers):
            y, local = layer(
                y,
                (
                    h_state[index],
                    c_state[index],
                    n_state[index],
                    m_state[index],
                ),
            )
            h, c, n, m = local
            next_h.append(h)
            next_c.append(c)
            next_n.append(n)
            next_m.append(m)

        return self.out(y), (
            torch.stack(next_h, dim=0),
            torch.stack(next_c, dim=0),
            torch.stack(next_n, dim=0),
            torch.stack(next_m, dim=0),
        )


class _StabilizedRecurrentLayer(nn.Module):
    def __init__(self, size: int, *, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.x_candidate = nn.Linear(size, size)
        self.h_candidate = nn.Linear(size, size, bias=False)
        self.x_input = nn.Linear(size, size)
        self.h_input = nn.Linear(size, size, bias=False)
        self.x_forget = nn.Linear(size, size)
        self.h_forget = nn.Linear(size, size, bias=False)
        self.x_output = nn.Linear(size, size)
        self.h_output = nn.Linear(size, size, bias=False)
        self.projection = nn.Linear(size, size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        h_prev, c_prev, n_prev, m_prev = state
        z = self.norm(x)
        candidate = torch.tanh(self.x_candidate(z) + self.h_candidate(h_prev))
        log_input = self.x_input(z) + self.h_input(h_prev)
        log_forget = -F.softplus(-(self.x_forget(z) + self.h_forget(h_prev)))
        output_gate = torch.sigmoid(self.x_output(z) + self.h_output(h_prev))

        m = torch.maximum(log_forget + m_prev, log_input)
        input_weight = torch.exp(log_input - m)
        forget_weight = torch.exp(log_forget + m_prev - m)
        c = forget_weight * c_prev + input_weight * candidate
        n = forget_weight * n_prev + input_weight
        normalized = c / n.clamp_min(1e-6)
        h = self.projection(output_gate * torch.tanh(normalized))
        y = x + self.dropout(h)
        return y, (h, c, n, m)


class CausalTCNExpert(nn.Module):
    """Dilated causal convolution expert over recent Core event history."""

    def __init__(
        self,
        size: int,
        *,
        depth: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.blocks = nn.ModuleList(
            [
                _TCNBlock(
                    size,
                    kernel_size=kernel_size,
                    dilation=2**index,
                    dropout=dropout,
                )
                for index in range(depth)
            ]
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(size * 3),
            nn.Linear(size * 3, size * 2),
            nn.SiLU(),
            nn.Linear(size * 2, size),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 2 or sequence.shape[1] != self.size:
            raise ValueError("TCN expert expects [time, hidden]")
        x = sequence.transpose(0, 1).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        last = x[:, :, -1]
        mean = x.mean(dim=-1)
        maximum = x.amax(dim=-1)
        return self.summary(torch.cat((last, mean, maximum), dim=-1))


class _TCNBlock(nn.Module):
    def __init__(
        self,
        size: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(size, size, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(size, size, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, size)
        self.norm2 = nn.GroupNorm(1, size)
        self.dropout = nn.Dropout(dropout)

    def _causal(self, conv: nn.Conv1d, x: Tensor) -> Tensor:
        return conv(F.pad(x, (self.padding, 0)))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        y = self._causal(self.conv1, x)
        y = self.dropout(F.silu(self.norm1(y)))
        y = self._causal(self.conv2, y)
        y = self.dropout(F.silu(self.norm2(y)))
        return residual + y


class HistoryAttentionExpert(nn.Module):
    """Small causal/history attention specialist, not an LLM-style Transformer."""

    def __init__(
        self,
        size: int,
        *,
        depth: int,
        heads: int,
        feedforward_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.blocks = nn.ModuleList(
            [
                _AttentionBlock(
                    size,
                    heads=heads,
                    feedforward_multiplier=feedforward_multiplier,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.pool_query = nn.Parameter(torch.zeros(size))
        self.summary = nn.Sequential(
            nn.LayerNorm(size * 3),
            nn.Linear(size * 3, size * 2),
            nn.GELU(),
            nn.Linear(size * 2, size),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 2 or sequence.shape[1] != self.size:
            raise ValueError("attention expert expects [time, hidden]")
        x = sequence.unsqueeze(0)
        length = int(sequence.shape[0])
        mask = torch.triu(
            torch.ones(length, length, device=sequence.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, mask)
        x = x[0]
        last = x[-1]
        scores = torch.mv(x, self.pool_query) * (self.size ** -0.5)
        pooled = torch.sum(x * torch.softmax(scores, dim=0).unsqueeze(1), dim=0)
        maximum = x.amax(dim=0)
        return self.summary(torch.cat((last, pooled, maximum), dim=-1).unsqueeze(0))


class _AttentionBlock(nn.Module):
    def __init__(
        self,
        size: int,
        *,
        heads: int,
        feedforward_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(size)
        self.attention = nn.MultiheadAttention(
            size,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(size)
        self.ff = nn.Sequential(
            nn.Linear(size, size * feedforward_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(size * feedforward_multiplier, size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        z = self.norm1(x)
        attended, _ = self.attention(z, z, z, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attended)
        return x + self.dropout(self.ff(self.norm2(x)))
