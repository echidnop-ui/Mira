from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class Mamba2Probe:
    status: str
    import_ok: bool
    forward_ok: bool
    backward_ok: bool
    reason: str


@lru_cache(maxsize=4)
def probe_mamba2_backend(device_name: str) -> Mamba2Probe:
    try:
        from mamba_ssm import Mamba2  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        return Mamba2Probe(
            status="UNAVAILABLE",
            import_ok=False,
            forward_ok=False,
            backward_ok=False,
            reason=f"mamba_ssm.Mamba2 import unavailable: {type(exc).__name__}: {exc}",
        )

    device = torch.device(device_name)
    try:
        block = Mamba2(d_model=64, d_state=64, d_conv=4, expand=2, headdim=32)
        block = block.to(device=device, dtype=torch.float32)
        value = torch.randn(1, 8, 64, device=device, requires_grad=True)
        output = block(value)
        if not torch.isfinite(output).all():
            raise RuntimeError("Mamba2 forward produced non-finite output")
    except Exception as exc:
        return Mamba2Probe(
            status="ROCM_INCOMPATIBLE",
            import_ok=True,
            forward_ok=False,
            backward_ok=False,
            reason=f"isolated Mamba2 forward failed: {type(exc).__name__}: {exc}",
        )
    try:
        output.square().mean().backward()
        if value.grad is None or not torch.isfinite(value.grad).all():
            raise RuntimeError("Mamba2 backward produced missing/non-finite gradients")
    except Exception as exc:
        return Mamba2Probe(
            status="ROCM_INCOMPATIBLE",
            import_ok=True,
            forward_ok=True,
            backward_ok=False,
            reason=f"isolated Mamba2 backward failed: {type(exc).__name__}: {exc}",
        )
    return Mamba2Probe(
        status="AVAILABLE_AND_WORKING",
        import_ok=True,
        forward_ok=True,
        backward_ok=True,
        reason="isolated Mamba2 forward/backward passed",
    )


def resolve_ssm_backend(requested: str, device: torch.device) -> tuple[str, Mamba2Probe]:
    if requested not in {"auto", "mamba2", "selective_ssm"}:
        raise ValueError("ssm_backend must be auto, mamba2, or selective_ssm")
    probe = probe_mamba2_backend(str(device))
    if requested == "mamba2":
        if probe.status != "AVAILABLE_AND_WORKING":
            raise RuntimeError(f"Mamba2 backend requested but unavailable: {probe.reason}")
        return "mamba2", probe
    if requested == "auto" and probe.status == "AVAILABLE_AND_WORKING":
        return "mamba2", probe
    return "selective_ssm", probe


@dataclass(slots=True)
class SelectiveSSMState:
    recurrent: Tensor
    convolution_history: Tensor


class SelectiveSSMBlock(nn.Module):
    """Portable fallback with selective gates and recurrent temporal state."""

    def __init__(self, width: int, *, dropout: float, bidirectional: bool = False) -> None:
        super().__init__()
        self.width = width
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(width)
        self.local = nn.Conv1d(width, width, 3, groups=width, padding=2)
        self.input_projection = nn.Linear(width, width * 3)
        self.state_projection = nn.Linear(width, width)
        self.decay = nn.Parameter(torch.full((width,), 1.5))
        self.output_projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def _scan(self, sequence: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch, length, width = sequence.shape
        if state is None:
            state = sequence.new_zeros(batch, width)
        candidate, update, output = self.input_projection(sequence).chunk(3, dim=-1)
        decay = torch.sigmoid(self.decay).unsqueeze(0)
        values: list[Tensor] = []
        for index in range(length):
            gate = torch.sigmoid(update[:, index, :])
            keep = decay * (1.0 - gate)
            state = keep * state + (1.0 - keep) * torch.tanh(candidate[:, index, :])
            values.append(torch.sigmoid(output[:, index, :]) * self.state_projection(state))
        return torch.stack(values, dim=1), state

    def forward(self, x: Tensor) -> Tensor:
        value = self.norm(x)
        local = self.local(value.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2)
        forward, _ = self._scan(value + F.silu(local))
        if self.bidirectional:
            backward, _ = self._scan(torch.flip(value + F.silu(local), dims=(1,)))
            forward = 0.5 * (forward + torch.flip(backward, dims=(1,)))
        return x + self.dropout(self.output_projection(forward))

    def step(
        self,
        x: Tensor,
        state: SelectiveSSMState | None,
    ) -> tuple[Tensor, SelectiveSSMState]:
        value = self.norm(x)
        history = None if state is None else state.convolution_history
        combined = value if history is None else torch.cat((history, value), dim=1)
        if combined.shape[1] < 3:
            combined = F.pad(
                combined.transpose(1, 2), (3 - combined.shape[1], 0)
            ).transpose(1, 2)
        local = F.conv1d(
            combined[:, -3:, :].transpose(1, 2),
            self.local.weight,
            self.local.bias,
            groups=self.width,
        ).transpose(1, 2)
        recurrent = None if state is None else state.recurrent
        scanned, recurrent = self._scan(value + F.silu(local), recurrent)
        next_state = SelectiveSSMState(
            recurrent=recurrent,
            convolution_history=combined[:, -2:, :].detach(),
        )
        return x + self.output_projection(scanned), next_state


class Mamba2LanguageBlock(nn.Module):
    """Thin adapter isolating the optional official mamba_ssm.Mamba2 API."""

    def __init__(self, width: int, *, dropout: float, bidirectional: bool = False) -> None:
        super().__init__()
        if bidirectional:
            raise ValueError("official Mamba2 adapter is causal; receptive fusion supplies other context")
        from mamba_ssm import Mamba2  # type: ignore[import-not-found]

        head_width = 64 if width % 64 == 0 else 32
        self.norm = nn.LayerNorm(width)
        self.backend = Mamba2(
            d_model=width,
            d_state=128,
            d_conv=4,
            expand=2,
            headdim=head_width,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.dropout(self.backend(self.norm(x)))

    def step(self, x: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]:
        history = x if state is None else torch.cat((state, x), dim=1)
        output = self.forward(history)[:, -1:, :]
        return output, history.detach()


class LanguageSSMBlock(nn.Module):
    """Stable Language adapter over official Mamba2 or PyTorch fallback."""

    def __init__(
        self,
        width: int,
        *,
        backend: str,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.backend_name = backend
        if backend == "mamba2":
            self.block = Mamba2LanguageBlock(
                width, dropout=dropout, bidirectional=bidirectional
            )
        elif backend == "selective_ssm":
            self.block = SelectiveSSMBlock(
                width, dropout=dropout, bidirectional=bidirectional
            )
        else:
            raise ValueError(f"unsupported resolved SSM backend: {backend}")

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def step(self, x: Tensor, state: object | None) -> tuple[Tensor, object]:
        return self.block.step(x, state)  # type: ignore[arg-type, no-any-return]
