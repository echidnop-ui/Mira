from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _logit(probability: float) -> float:
    probability = min(max(probability, 1.0e-5), 1.0 - 1.0e-5)
    return log(probability / (1.0 - probability))


@dataclass(slots=True)
class StateWeaveState:
    """Persistent state for one StateWeave block.

    fast/medium/slow carry different temporal scales. ``plastic`` is a small
    delta-rule associative fast-weight matrix. ``convolution_history`` keeps
    only the local causal frontend context needed by ``step``.
    """

    fast: Tensor
    medium: Tensor
    slow: Tensor
    plastic: Tensor
    convolution_history: Tensor

    def detached(self) -> "StateWeaveState":
        """Detach persistent state when a caller wants to cut BPTT history."""
        return StateWeaveState(
            fast=self.fast.detach(),
            medium=self.medium.detach(),
            slow=self.slow.detach(),
            plastic=self.plastic.detach(),
            convolution_history=self.convolution_history.detach(),
        )


class StateWeaveBlock(nn.Module):
    """Multi-timescale recurrent state with input-conditioned dynamics.

    This is a Heiwa-native temporal block rather than a Mamba imitation.  It
    combines three persistent state rates, low-rank input-conditioned state
    corrections, and a small plastic associative memory updated by a delta
    rule.  The implementation intentionally uses ordinary PyTorch ops so the
    same architecture can run on ROCm without custom CUDA kernels.
    """

    def __init__(
        self,
        width: int,
        *,
        dropout: float,
        plastic_rank: int = 32,
        dynamic_rank: int = 32,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be > 0")
        if plastic_rank <= 0 or dynamic_rank <= 0:
            raise ValueError("StateWeave ranks must be > 0")

        self.width = width
        self.plastic_rank = plastic_rank
        self.dynamic_rank = dynamic_rank
        self.bidirectional = bidirectional

        self.norm = nn.LayerNorm(width)
        self.local = nn.Conv1d(width, width, 3, groups=width, padding=2)
        self.local_mix = nn.Linear(width, width)

        # Three candidate states and three input-dependent retention controls.
        self.candidate = nn.Linear(width, width * 3)
        self.retention = nn.Linear(width, width * 3)
        self.base_retention = nn.Parameter(
            torch.tensor(
                [
                    _logit(0.55),   # fast: changes readily
                    _logit(0.90),   # medium: sentence / local thought
                    _logit(0.985),  # slow: persistent conversational context
                ],
                dtype=torch.float32,
            ).view(3, 1).repeat(1, width)
        )

        # Low-rank dynamic corrections let the transition itself depend on
        # current input instead of only changing a forget gate.
        self.state_to_dynamic = nn.ModuleList(
            nn.Linear(width, dynamic_rank, bias=False) for _ in range(3)
        )
        self.input_to_dynamic = nn.Linear(width, dynamic_rank * 3)
        self.dynamic_to_state = nn.ModuleList(
            nn.Linear(dynamic_rank, width, bias=False) for _ in range(3)
        )
        self.dynamic_to_retention = nn.ModuleList(
            nn.Linear(dynamic_rank, width, bias=False) for _ in range(3)
        )
        self.dynamic_strength = nn.Parameter(torch.full((3, 1), -1.5))

        # Low-rank cross-scale exchange lets fast evidence influence slower
        # state and slow context bias faster processing without a full
        # width*3 -> width*3 projection.  The small initial output keeps this
        # path stable while still allowing gradients from the first step.
        self.cross_scale_down = nn.Linear(width * 3, dynamic_rank, bias=False)
        self.cross_scale_input = nn.Linear(width, dynamic_rank, bias=False)
        self.cross_scale_up = nn.Linear(dynamic_rank, width * 3, bias=False)
        self.cross_scale_strength = nn.Parameter(torch.tensor(-1.5))
        nn.init.normal_(self.cross_scale_up.weight, mean=0.0, std=0.01)
        for layer in self.dynamic_to_retention:
            nn.init.zeros_(layer.weight)

        # The network chooses which timescale matters for each token.
        self.scale_gate = nn.Linear(width, 3)
        self.state_output = nn.Linear(width, width)
        self.output_gate = nn.Linear(width, width)

        # Small fast-weight memory.  It learns what to write/query; the matrix
        # itself is recurrent state, not a permanent parameter table.
        self.plastic_key = nn.Linear(width, plastic_rank)
        self.plastic_value = nn.Linear(width, plastic_rank)
        self.plastic_query = nn.Linear(width, plastic_rank)
        self.plastic_control = nn.Linear(width, 2)
        self.plastic_output = nn.Linear(plastic_rank, width, bias=False)
        self.plastic_base_decay = nn.Parameter(torch.tensor(_logit(0.97)))

        self.output_projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def _empty_state(self, sequence: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = sequence.shape[0]
        zeros = sequence.new_zeros(batch, self.width)
        plastic = sequence.new_zeros(batch, self.plastic_rank, self.plastic_rank)
        return zeros, zeros.clone(), zeros.clone(), plastic

    def _scan(
        self,
        sequence: Tensor,
        state: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        if sequence.ndim != 3 or sequence.shape[-1] != self.width:
            raise ValueError("StateWeave expects [batch, length, width]")
        if state is None:
            fast, medium, slow, plastic = self._empty_state(sequence)
        else:
            fast, medium, slow, plastic = state

        candidates = self.candidate(sequence).view(
            sequence.shape[0], sequence.shape[1], 3, self.width
        )
        retention_delta = self.retention(sequence).view(
            sequence.shape[0], sequence.shape[1], 3, self.width
        )
        dynamic_input = self.input_to_dynamic(sequence).view(
            sequence.shape[0], sequence.shape[1], 3, self.dynamic_rank
        )
        scale_weights = torch.softmax(self.scale_gate(sequence), dim=-1)
        output_gate = torch.sigmoid(self.output_gate(sequence))

        keys = F.normalize(self.plastic_key(sequence), dim=-1, eps=1.0e-6)
        values = torch.tanh(self.plastic_value(sequence))
        queries = F.normalize(self.plastic_query(sequence), dim=-1, eps=1.0e-6)
        plastic_controls = self.plastic_control(sequence)

        states = [fast, medium, slow]
        outputs: list[Tensor] = []
        inv_sqrt_rank = 1.0 / sqrt(float(self.plastic_rank))

        for index in range(sequence.shape[1]):
            previous_states = states
            joined = torch.cat(previous_states, dim=-1)
            cross_scale = self.cross_scale_down(joined)
            cross_scale = cross_scale * torch.tanh(
                self.cross_scale_input(sequence[:, index, :])
            )
            cross_scale = self.cross_scale_up(torch.tanh(cross_scale)).view(
                sequence.shape[0], 3, self.width
            )
            cross_scale = torch.sigmoid(self.cross_scale_strength) * cross_scale

            next_states: list[Tensor] = []
            for scale in range(3):
                previous = previous_states[scale]
                dynamic = self.state_to_dynamic[scale](previous)
                dynamic = torch.tanh(
                    dynamic * torch.tanh(dynamic_input[:, index, scale, :])
                )
                correction = self.dynamic_to_state[scale](dynamic)
                correction = torch.sigmoid(self.dynamic_strength[scale]) * correction
                proposal = torch.tanh(
                    candidates[:, index, scale, :]
                    + correction
                    + cross_scale[:, scale, :]
                )
                retention_feedback = self.dynamic_to_retention[scale](dynamic)
                keep = torch.sigmoid(
                    self.base_retention[scale].unsqueeze(0)
                    + retention_delta[:, index, scale, :]
                    + retention_feedback
                )
                next_states.append(keep * previous + (1.0 - keep) * proposal)
            states = next_states

            weights = scale_weights[:, index, :]
            woven = (
                weights[:, 0:1] * states[0]
                + weights[:, 1:2] * states[1]
                + weights[:, 2:3] * states[2]
            )

            key = keys[:, index, :]
            value = values[:, index, :]
            query = queries[:, index, :]
            write = torch.sigmoid(plastic_controls[:, index, 0]).view(-1, 1, 1)
            decay = torch.sigmoid(
                self.plastic_base_decay + plastic_controls[:, index, 1]
            ).view(-1, 1, 1)

            # Delta-rule associative update: write only the part not already
            # predicted by the current memory for this key.
            predicted = torch.bmm(key.unsqueeze(1), plastic).squeeze(1)
            error = value - predicted
            update = torch.bmm(key.unsqueeze(2), error.unsqueeze(1))
            plastic = decay * plastic + write * update * inv_sqrt_rank

            read = torch.bmm(query.unsqueeze(1), plastic).squeeze(1)
            combined = self.state_output(woven) + self.plastic_output(torch.tanh(read))
            outputs.append(output_gate[:, index, :] * combined)

        return torch.stack(outputs, dim=1), (states[0], states[1], states[2], plastic)

    def _prepare(self, x: Tensor) -> Tensor:
        value = self.norm(x)
        local = self.local(value.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2)
        return value + F.silu(self.local_mix(local))

    def forward(self, x: Tensor) -> Tensor:
        prepared = self._prepare(x)
        forward, _ = self._scan(prepared)
        if self.bidirectional:
            backward, _ = self._scan(torch.flip(prepared, dims=(1,)))
            forward = 0.5 * (forward + torch.flip(backward, dims=(1,)))
        return x + self.dropout(self.output_projection(forward))

    def step(
        self,
        x: Tensor,
        state: StateWeaveState | None,
    ) -> tuple[Tensor, StateWeaveState]:
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] != self.width:
            raise ValueError("StateWeave.step expects [batch, 1, width]")

        value = self.norm(x)
        history = None if state is None else state.convolution_history
        combined = value if history is None else torch.cat((history, value), dim=1)
        if combined.shape[1] < 3:
            padded = F.pad(
                combined.transpose(1, 2),
                (3 - combined.shape[1], 0),
            ).transpose(1, 2)
        else:
            padded = combined
        local = F.conv1d(
            padded[:, -3:, :].transpose(1, 2),
            self.local.weight,
            self.local.bias,
            groups=self.width,
        ).transpose(1, 2)
        prepared = value + F.silu(self.local_mix(local))

        recurrent = None
        if state is not None:
            recurrent = (state.fast, state.medium, state.slow, state.plastic)
        scanned, (fast, medium, slow, plastic) = self._scan(prepared, recurrent)
        next_state = StateWeaveState(
            fast=fast,
            medium=medium,
            slow=slow,
            plastic=plastic,
            convolution_history=combined[:, -2:, :],
        )
        return x + self.output_projection(scanned), next_state
