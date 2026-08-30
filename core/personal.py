from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import io
from math import isfinite
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from core.experts import (
    EXPERT_NAMES,
    CausalTCNExpert,
    HistoryAttentionExpert,
    SelectiveSSMExpert,
    XLSTMStyleExpert,
)
from core.interface import CoreInput, IntegratedCognitiveState


_CAUSAL_SNAPSHOT_FORMAT = "heiwa.core.causal-snapshot"
_CAUSAL_SNAPSHOT_VERSION = 1


@dataclass(frozen=True, slots=True)
class PersonalCoreConfig:
    input_size: int = 96
    context_size: int = 96
    hidden_size: int = 512
    output_size: int = 96

    workspace_heads: int = 8
    attention_heads: int = 8
    router_hidden_size: int = 512
    router_top_k: int = 2

    ssm_depth: int = 3
    recurrent_depth: int = 3
    tcn_depth: int = 4
    tcn_kernel_size: int = 3
    attention_depth: int = 3
    attention_feedforward_multiplier: int = 4
    history_size: int = 128

    dropout: float = 0.05
    slow_decay_floor: float = 0.90
    slow_decay_ceiling: float = 0.999
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 5.0
    max_traces: int = 64
    device: str | None = None
    dtype: torch.dtype = torch.float32

    @classmethod
    def smoke(cls, *, device: str | None = None) -> "PersonalCoreConfig":
        return cls(
            hidden_size=128,
            workspace_heads=4,
            attention_heads=4,
            router_hidden_size=128,
            ssm_depth=2,
            recurrent_depth=2,
            tcn_depth=2,
            attention_depth=2,
            history_size=32,
            dropout=0.0,
            max_traces=16,
            device=device,
        )

    @classmethod
    def rx6800(cls) -> "PersonalCoreConfig":
        return cls()

    @classmethod
    def rx6800_heavy(cls) -> "PersonalCoreConfig":
        return cls(
            hidden_size=768,
            workspace_heads=12,
            attention_heads=12,
            router_hidden_size=768,
            router_top_k=3,
            ssm_depth=4,
            recurrent_depth=4,
            tcn_depth=5,
            attention_depth=4,
            history_size=192,
            max_traces=32,
        )


@dataclass(frozen=True, slots=True)
class PersonalCoreTelemetry:
    forward_calls: int
    learning_updates: int
    traces_created: int
    traces_consumed: int
    traces_evicted: int
    duplicate_transition_rejections: int
    state_resets: int
    parameter_count: int
    last_active_experts: tuple[str, ...]
    last_router_weights: Mapping[str, float]


@dataclass(slots=True)
class _PersistentState:
    work: Tensor
    slow: Tensor
    ssm: Tensor
    recurrent: tuple[Tensor, Tensor, Tensor, Tensor]
    history: Tensor
    previous_output: Tensor


@dataclass(frozen=True, slots=True)
class _StateSnapshot:
    work: Tensor
    slow: Tensor
    ssm: Tensor
    recurrent: tuple[Tensor, Tensor, Tensor, Tensor]
    history: Tensor
    previous_output: Tensor


@dataclass(frozen=True, slots=True)
class _LearningTrace:
    core_input: CoreInput
    state_snapshot: _StateSnapshot
    active_experts: tuple[str, ...]


class _ContextIntegrator(nn.Module):
    def __init__(self, context_size: int, hidden_size: int) -> None:
        super().__init__()
        self.memory = nn.Linear(context_size, hidden_size)
        self.self_model = nn.Linear(context_size, hidden_size)
        self.relation = nn.Linear(context_size, hidden_size)
        self.missing = nn.Parameter(torch.zeros(3, hidden_size))
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.scale = hidden_size ** -0.5

    def _token(self, value: Tensor | None, encoder: nn.Module, missing: int) -> Tensor:
        if value is None:
            return self.missing[missing].unsqueeze(0)
        return torch.nn.functional.gelu(encoder(value))

    def forward(
        self,
        observation: Tensor,
        memory: Tensor | None,
        self_context: Tensor | None,
        relation: Tensor | None,
    ) -> Tensor:
        tokens = torch.stack(
            (
                self._token(memory, self.memory, 0),
                self._token(self_context, self.self_model, 1),
                self._token(relation, self.relation, 2),
            ),
            dim=1,
        )
        query = self.query(observation).unsqueeze(1)
        keys = self.key(tokens)
        values = self.value(tokens)
        weights = torch.softmax(
            (query * keys).sum(dim=-1) * self.scale,
            dim=-1,
        ).unsqueeze(-1)
        return self.out((weights * values).sum(dim=1))


class _SparseExpertRouter(nn.Module):
    def __init__(self, hidden_size: int, router_hidden_size: int, top_k: int) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, router_hidden_size),
            nn.SiLU(),
            nn.Linear(router_hidden_size, router_hidden_size),
            nn.SiLU(),
            nn.Linear(router_hidden_size, len(EXPERT_NAMES)),
        )

    def forward(
        self,
        current: Tensor,
        context: Tensor,
        work: Tensor,
        slow: Tensor,
        *,
        forced_experts: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], Tensor, Tensor]:
        logits = self.network(torch.cat((current, context, work, slow), dim=-1))[0]

        if forced_experts is None:
            indices = torch.topk(logits, k=self.top_k, dim=0).indices
            active = tuple(EXPERT_NAMES[int(index)] for index in indices.detach().cpu())
        else:
            if not forced_experts:
                raise ValueError("forced_experts must not be empty")
            if len(set(forced_experts)) != len(forced_experts):
                raise ValueError("forced_experts contains duplicates")
            try:
                raw = [EXPERT_NAMES.index(name) for name in forced_experts]
            except ValueError as exc:
                raise ValueError("forced_experts contains unknown expert") from exc
            indices = torch.tensor(raw, device=logits.device, dtype=torch.long)
            active = tuple(forced_experts)

        active_logits = logits.index_select(0, indices)
        weights = torch.softmax(active_logits, dim=0)
        return active, indices, weights


class _ExpertFusion(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                )
                for name in EXPERT_NAMES
            }
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        active: tuple[str, ...],
        weights: Tensor,
    ) -> Tensor:
        fused: Tensor | None = None
        for index, name in enumerate(active):
            projected = self.projections[name](outputs[name])
            contribution = projected * weights[index]
            fused = contribution if fused is None else fused + contribution
        if fused is None:
            raise RuntimeError("expert fusion received no active experts")
        return self.out(fused)


class _Workspace(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, tokens: Tensor) -> Tensor:
        query = self.query.expand(tokens.shape[0], -1, -1)
        attended, _ = self.attention(query, tokens, tokens, need_weights=False)
        x = self.norm1(query + attended)
        return self.norm2(x + self.ff(x))[:, 0, :]


class PersonalCognitiveCore(nn.Module):
    """World-agnostic persistent heterogeneous neural Core of Heiwa.

    The outer contract stays CoreInput -> IntegratedCognitiveState.  Internally,
    learned routing chooses a sparse subset of heterogeneous temporal experts.
    Mechanism dispatch is deterministic; the selection itself is neural.
    """

    def __init__(self, config: PersonalCoreConfig = PersonalCoreConfig()) -> None:
        super().__init__()
        self.config = config
        self._validate_config(config)
        self.device = self._resolve_device(config.device)
        h = config.hidden_size

        self.input_projection = nn.Sequential(
            nn.LayerNorm(config.input_size),
            nn.Linear(config.input_size, h * 2),
            nn.GELU(),
            nn.Linear(h * 2, h),
        )
        self.context_integrator = _ContextIntegrator(config.context_size, h)

        self.fast = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h * 3),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h * 3, h),
        )

        self.router = _SparseExpertRouter(h, config.router_hidden_size, config.router_top_k)
        self.ssm_expert = SelectiveSSMExpert(
            h, depth=config.ssm_depth, dropout=config.dropout
        )
        self.recurrent_expert = XLSTMStyleExpert(
            h, depth=config.recurrent_depth, dropout=config.dropout
        )
        self.tcn_expert = CausalTCNExpert(
            h,
            depth=config.tcn_depth,
            kernel_size=config.tcn_kernel_size,
            dropout=config.dropout,
        )
        self.attention_expert = HistoryAttentionExpert(
            h,
            depth=config.attention_depth,
            heads=config.attention_heads,
            feedforward_multiplier=config.attention_feedforward_multiplier,
            dropout=config.dropout,
        )
        self.expert_fusion = _ExpertFusion(h)

        self.temporal_fusion = nn.Sequential(
            nn.LayerNorm(h * 3),
            nn.Linear(h * 3, h * 2),
            nn.GELU(),
            nn.Linear(h * 2, h),
        )
        self.work_input = nn.Sequential(
            nn.LayerNorm(h * 3),
            nn.Linear(h * 3, h),
            nn.GELU(),
        )
        self.work_cell = nn.GRUCell(h, h)
        self.slow_candidate = nn.Sequential(
            nn.LayerNorm(h * 2),
            nn.Linear(h * 2, h),
            nn.Tanh(),
        )
        self.slow_gate = nn.Sequential(
            nn.LayerNorm(h * 2),
            nn.Linear(h * 2, h),
            nn.Sigmoid(),
        )
        self.workspace = _Workspace(h, config.workspace_heads, config.dropout)
        self.output_head = nn.Sequential(
            nn.LayerNorm(h * 4),
            nn.Linear(h * 4, h * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h * 2, config.output_size),
        )

        self.to(device=self.device, dtype=config.dtype)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self._state = self._new_state()
        self._traces: OrderedDict[str, _LearningTrace] = OrderedDict()
        self._forward_calls = 0
        self._learning_updates = 0
        self._traces_created = 0
        self._traces_consumed = 0
        self._traces_evicted = 0
        self._duplicate_transition_rejections = 0
        self._state_resets = 0
        self._last_active_experts: tuple[str, ...] = ()
        self._last_router_weights: dict[str, float] = {}

    @property
    def representation_size(self) -> int:
        return self.config.output_size

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def integrate(
        self,
        core_input: CoreInput,
        *,
        remember_for_learning: bool = True,
    ) -> IntegratedCognitiveState:
        value = self._normalize_input(core_input)
        if remember_for_learning and value.transition_id in self._traces:
            self._duplicate_transition_rejections += 1
            raise ValueError(
                "transition_id already has an unconsumed core trace: "
                f"{value.transition_id}"
            )

        snapshot = self._snapshot_state(self._state)
        self.eval()
        with torch.no_grad():
            (
                output,
                workspace,
                temporal,
                context,
                next_state,
                active,
                router_weights,
            ) = self._forward_tensors(self._input_tensors(value), self._state)
            novelty = self._cosine_novelty(output, self._state.previous_output)
            state_change = self._rms_distance(output, self._state.previous_output)
            self._state = self._detach_state(next_state)

        self._forward_calls += 1
        self._last_active_experts = active
        self._last_router_weights = {
            name: float(router_weights[index].detach().cpu().item())
            for index, name in enumerate(active)
        }

        if remember_for_learning:
            self._store_trace(
                value.transition_id,
                _LearningTrace(value, snapshot, active),
            )

        metadata = dict(value.metadata)
        metadata["core_active_experts"] = active
        metadata["core_router_weights"] = dict(self._last_router_weights)

        return IntegratedCognitiveState(
            transition_id=value.transition_id,
            world_id=value.world_id,
            vector=self._tensor_tuple(output),
            workspace_vector=self._tensor_tuple(workspace),
            temporal_vector=self._tensor_tuple(temporal),
            context_vector=self._tensor_tuple(context),
            novelty=novelty,
            state_change=state_change,
            metadata=metadata,
        )

    def learn(self, transition_id: str, output_gradient: Sequence[float]) -> bool:
        transition_id = transition_id.strip()
        if not transition_id:
            raise ValueError("transition_id must not be empty")
        trace = self._traces.pop(transition_id, None)
        if trace is None:
            return False

        gradient = self._vector_tensor(
            output_gradient,
            self.config.output_size,
            "output_gradient",
        )
        replay_state = self._state_from_snapshot(trace.state_snapshot)
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        output, _, _, _, _, _, _ = self._forward_tensors(
            self._input_tensors(trace.core_input),
            replay_state,
            forced_experts=trace.active_experts,
        )
        output.backward(gradient)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        self._learning_updates += 1
        self._traces_consumed += 1
        return True

    def discard_trace(self, transition_id: str) -> bool:
        return self._traces.pop(transition_id.strip(), None) is not None

    def has_trace(self, transition_id: str) -> bool:
        return transition_id.strip() in self._traces

    def export_causal_snapshot(
        self,
        generation_id: str,
        transition_ids: Sequence[str],
    ) -> bytes:
        generation, expected_ids = self._causal_identity(
            generation_id, transition_ids
        )
        actual_ids = tuple(self._traces)
        if expected_ids != actual_ids:
            raise ValueError(
                "Core causal snapshot trace IDs do not match live traces: "
                f"expected={expected_ids!r}, actual={actual_ids!r}"
            )
        payload = {
            "format": _CAUSAL_SNAPSHOT_FORMAT,
            "version": _CAUSAL_SNAPSHOT_VERSION,
            "config": self.config,
            "generation_id": generation,
            "transition_ids": expected_ids,
            "model": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
            "persistent_state": self._snapshot_state(self._state),
            "traces": tuple(self._traces.items()),
            "telemetry": {
                "forward_calls": self._forward_calls,
                "learning_updates": self._learning_updates,
                "traces_created": self._traces_created,
                "traces_consumed": self._traces_consumed,
                "traces_evicted": self._traces_evicted,
                "duplicate_transition_rejections": self._duplicate_transition_rejections,
                "state_resets": self._state_resets,
                "last_active_experts": self._last_active_experts,
                "last_router_weights": dict(self._last_router_weights),
                "training": self.training,
            },
        }
        stream = io.BytesIO()
        torch.save(payload, stream)
        return stream.getvalue()

    def restore_causal_snapshot(
        self,
        snapshot: bytes,
        expected_generation_id: str,
        expected_transition_ids: Sequence[str],
    ) -> None:
        generation, expected_ids = self._causal_identity(
            expected_generation_id, expected_transition_ids
        )
        if not isinstance(snapshot, bytes):
            raise ValueError("Core causal snapshot must be bytes")
        try:
            payload = torch.load(
                io.BytesIO(snapshot), map_location="cpu", weights_only=False
            )
        except Exception as exc:
            raise ValueError(f"invalid Core causal snapshot: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("invalid Core causal snapshot root")
        if payload.get("format") != _CAUSAL_SNAPSHOT_FORMAT:
            raise ValueError("unsupported Core causal snapshot format")
        if payload.get("version") != _CAUSAL_SNAPSHOT_VERSION:
            raise ValueError("unsupported Core causal snapshot version")
        if payload.get("config") != self.config:
            raise ValueError("Core causal snapshot config mismatch")
        if payload.get("generation_id") != generation:
            raise ValueError("Core causal snapshot generation mismatch")
        if payload.get("transition_ids") != expected_ids:
            raise ValueError("Core causal snapshot transition IDs mismatch")

        model_state = self._validate_causal_model_state(payload.get("model"))
        optimizer_state = payload.get("optimizer")
        if not isinstance(optimizer_state, dict):
            raise ValueError("invalid Core causal optimizer state")
        staged_optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        try:
            staged_optimizer.load_state_dict(optimizer_state)
        except Exception as exc:
            raise ValueError(f"invalid Core causal optimizer state: {exc}") from exc
        self._validate_causal_optimizer(staged_optimizer)

        persistent_snapshot = payload.get("persistent_state")
        self._validate_causal_state_snapshot(
            persistent_snapshot, name="persistent_state"
        )
        staged_state = self._state_from_snapshot(persistent_snapshot)

        raw_traces = payload.get("traces")
        if not isinstance(raw_traces, tuple) or len(raw_traces) != len(expected_ids):
            raise ValueError("invalid Core causal traces")
        staged_traces: OrderedDict[str, _LearningTrace] = OrderedDict()
        for index, item in enumerate(raw_traces):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("invalid Core causal trace entry")
            transition_id, trace = item
            if transition_id != expected_ids[index]:
                raise ValueError("Core causal trace order mismatch")
            if not isinstance(trace, _LearningTrace):
                raise ValueError("invalid Core causal trace type")
            if not isinstance(trace.core_input, CoreInput):
                raise ValueError("invalid Core causal trace input type")
            normalized_input = self._normalize_input(trace.core_input)
            if normalized_input != trace.core_input:
                raise ValueError("invalid Core causal trace input")
            if normalized_input.transition_id != transition_id:
                raise ValueError("Core causal trace transition mismatch")
            self._validate_causal_state_snapshot(
                trace.state_snapshot, name=f"trace {transition_id!r}"
            )
            if (
                not isinstance(trace.active_experts, tuple)
                or not trace.active_experts
                or any(name not in EXPERT_NAMES for name in trace.active_experts)
            ):
                raise ValueError("invalid Core causal trace experts")
            staged_traces[transition_id] = trace

        telemetry = payload.get("telemetry")
        if not isinstance(telemetry, dict):
            raise ValueError("invalid Core causal telemetry")
        counter_names = (
            "forward_calls", "learning_updates", "traces_created",
            "traces_consumed", "traces_evicted",
            "duplicate_transition_rejections", "state_resets",
        )
        staged_counters: dict[str, int] = {}
        for name in counter_names:
            value = telemetry.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid Core causal telemetry {name!r}")
            staged_counters[name] = value
        last_active = telemetry.get("last_active_experts")
        if (
            not isinstance(last_active, tuple)
            or any(name not in EXPERT_NAMES for name in last_active)
        ):
            raise ValueError("invalid Core last_active_experts")
        raw_router = telemetry.get("last_router_weights")
        if not isinstance(raw_router, dict) or any(
            name not in EXPERT_NAMES
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for name, value in raw_router.items()
        ):
            raise ValueError("invalid Core last_router_weights")
        training = telemetry.get("training")
        if type(training) is not bool:
            raise ValueError("invalid Core training mode")

        self.load_state_dict(model_state)
        self.optimizer = staged_optimizer
        self._state = staged_state
        self._traces = staged_traces
        self._forward_calls = staged_counters["forward_calls"]
        self._learning_updates = staged_counters["learning_updates"]
        self._traces_created = staged_counters["traces_created"]
        self._traces_consumed = staged_counters["traces_consumed"]
        self._traces_evicted = staged_counters["traces_evicted"]
        self._duplicate_transition_rejections = staged_counters[
            "duplicate_transition_rejections"
        ]
        self._state_resets = staged_counters["state_resets"]
        self._last_active_experts = last_active
        self._last_router_weights = {
            name: float(value) for name, value in raw_router.items()
        }
        self.train(training)

    def reset_state(self, *, clear_learning_traces: bool = False) -> None:
        self._state = self._new_state()
        if clear_learning_traces:
            self._traces.clear()
        self._state_resets += 1

    def telemetry(self) -> PersonalCoreTelemetry:
        return PersonalCoreTelemetry(
            forward_calls=self._forward_calls,
            learning_updates=self._learning_updates,
            traces_created=self._traces_created,
            traces_consumed=self._traces_consumed,
            traces_evicted=self._traces_evicted,
            duplicate_transition_rejections=self._duplicate_transition_rejections,
            state_resets=self._state_resets,
            parameter_count=self.parameter_count,
            last_active_experts=self._last_active_experts,
            last_router_weights=dict(self._last_router_weights),
        )

    def save_checkpoint(self, path: Path) -> None:
        """Atomically persist weights, optimizer and the live FAST/WORK/SLOW state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self._snapshot_state(self._state)
        payload = {
            "format": 1, "config": self.config,
            "model": self.state_dict(), "optimizer": self.optimizer.state_dict(),
            "persistent_state": state,
            "telemetry": {"forward_calls": self._forward_calls,
                          "learning_updates": self._learning_updates},
        }
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(payload, temporary); temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path: Path) -> bool:
        if not path.exists(): return False
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or int(payload.get("format", -1)) != 1:
            raise ValueError("unsupported Core checkpoint")
        self.load_state_dict(payload["model"])
        if isinstance(payload.get("optimizer"), dict): self.optimizer.load_state_dict(payload["optimizer"])
        state = payload.get("persistent_state")
        if isinstance(state, _StateSnapshot): self._state = self._state_from_snapshot(state)
        telemetry = payload.get("telemetry")
        if isinstance(telemetry, dict):
            self._forward_calls = max(0, int(telemetry.get("forward_calls", 0)))
            self._learning_updates = max(0, int(telemetry.get("learning_updates", 0)))
        self._traces.clear(); return True

    def _forward_tensors(
        self,
        tensors: tuple[Tensor, Tensor | None, Tensor | None, Tensor | None],
        state: _PersistentState,
        *,
        forced_experts: tuple[str, ...] | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        _PersistentState,
        tuple[str, ...],
        Tensor,
    ]:
        representation, memory, self_context, relation = tensors
        current = self.input_projection(representation)
        context = self.context_integrator(current, memory, self_context, relation)
        fast = current + self.fast(current)

        active, _, router_weights = self.router(
            current,
            context,
            state.work,
            state.slow,
            forced_experts=forced_experts,
        )

        sequence = torch.cat((state.history, current), dim=0)
        outputs: dict[str, Tensor] = {}
        next_ssm = state.ssm
        next_recurrent = state.recurrent

        for name in active:
            if name == "ssm":
                outputs[name], next_ssm = self.ssm_expert.forward_step(current, state.ssm)
            elif name == "recurrent":
                outputs[name], next_recurrent = self.recurrent_expert.forward_step(
                    current, state.recurrent
                )
            elif name == "tcn":
                outputs[name] = self.tcn_expert(sequence)
            elif name == "attention":
                outputs[name] = self.attention_expert(sequence)
            else:  # pragma: no cover - guarded by router names
                raise RuntimeError(f"unknown Core expert {name!r}")

        expert = self.expert_fusion(outputs, active, router_weights)
        temporal = self.temporal_fusion(torch.cat((fast, expert, current), dim=-1))

        work_input = self.work_input(torch.cat((temporal, context, fast), dim=-1))
        work = self.work_cell(work_input, state.work)

        slow_source = torch.cat((work, temporal), dim=-1)
        slow_gate = self.slow_gate(slow_source)
        decay = self.config.slow_decay_floor + slow_gate * (
            self.config.slow_decay_ceiling - self.config.slow_decay_floor
        )
        slow = decay * state.slow + (1.0 - decay) * self.slow_candidate(slow_source)

        workspace_tokens = [current, fast, temporal, work, slow, context]
        workspace_tokens.extend(outputs[name] for name in active)
        workspace = self.workspace(torch.stack(workspace_tokens, dim=1))

        output = self.output_head(
            torch.cat((workspace, temporal, context, slow), dim=-1)
        )
        history = sequence[-self.config.history_size :]

        return (
            output,
            workspace,
            temporal,
            context,
            _PersistentState(
                work=work,
                slow=slow,
                ssm=next_ssm,
                recurrent=next_recurrent,
                history=history,
                previous_output=output,
            ),
            active,
            router_weights,
        )

    def _normalize_input(self, value: CoreInput) -> CoreInput:
        transition_id = value.transition_id.strip()
        world_id = value.world_id.strip()
        if not transition_id:
            raise ValueError("CoreInput.transition_id must not be empty")
        if not world_id:
            raise ValueError("CoreInput.world_id must not be empty")
        return CoreInput(
            transition_id=transition_id,
            world_id=world_id,
            representation=self._normalize_vector(
                value.representation, self.config.input_size, "representation"
            ),
            memory_context=self._normalize_optional_context(
                value.memory_context, "memory_context"
            ),
            self_context=self._normalize_optional_context(value.self_context, "self_context"),
            relation_context=self._normalize_optional_context(
                value.relation_context, "relation_context"
            ),
            metadata=dict(value.metadata),
        )

    def _normalize_optional_context(
        self, values: Sequence[float], name: str
    ) -> tuple[float, ...]:
        if not values:
            return ()
        return self._normalize_vector(values, self.config.context_size, name)

    @staticmethod
    def _normalize_vector(
        values: Sequence[float], expected_size: int, name: str
    ) -> tuple[float, ...]:
        if len(values) != expected_size:
            raise ValueError(f"{name} must have {expected_size} values; got {len(values)}")
        result = tuple(float(x) for x in values)
        if not all(isfinite(x) for x in result):
            raise ValueError(f"{name} contains non-finite values")
        return result

    def _input_tensors(
        self, value: CoreInput
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        return (
            self._vector_tensor(value.representation, self.config.input_size, "representation"),
            self._optional_tensor(value.memory_context),
            self._optional_tensor(value.self_context),
            self._optional_tensor(value.relation_context),
        )

    def _optional_tensor(self, values: Sequence[float]) -> Tensor | None:
        if not values:
            return None
        return self._vector_tensor(values, self.config.context_size, "context")

    def _vector_tensor(self, values: Sequence[float], expected_size: int, name: str) -> Tensor:
        values = self._normalize_vector(values, expected_size, name)
        return torch.tensor(values, device=self.device, dtype=self.config.dtype).unsqueeze(0)

    @staticmethod
    def _tensor_tuple(tensor: Tensor) -> tuple[float, ...]:
        return tuple(
            float(x)
            for x in tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1).tolist()
        )

    def _new_state(self) -> _PersistentState:
        h = self.config.hidden_size
        return _PersistentState(
            work=torch.zeros((1, h), device=self.device, dtype=self.config.dtype),
            slow=torch.zeros((1, h), device=self.device, dtype=self.config.dtype),
            ssm=self.ssm_expert.initial_state(device=self.device, dtype=self.config.dtype),
            recurrent=self.recurrent_expert.initial_state(
                device=self.device, dtype=self.config.dtype
            ),
            history=torch.empty((0, h), device=self.device, dtype=self.config.dtype),
            previous_output=torch.zeros(
                (1, self.config.output_size), device=self.device, dtype=self.config.dtype
            ),
        )

    @staticmethod
    def _cpu_clone(value: Tensor) -> Tensor:
        return value.detach().to(device="cpu", dtype=torch.float32).clone()

    def _snapshot_state(self, state: _PersistentState) -> _StateSnapshot:
        return _StateSnapshot(
            work=self._cpu_clone(state.work),
            slow=self._cpu_clone(state.slow),
            ssm=self._cpu_clone(state.ssm),
            recurrent=tuple(self._cpu_clone(value) for value in state.recurrent),
            history=self._cpu_clone(state.history),
            previous_output=self._cpu_clone(state.previous_output),
        )

    @staticmethod
    def _causal_identity(
        generation_id: str,
        transition_ids: Sequence[str],
    ) -> tuple[str, tuple[str, ...]]:
        if (
            not isinstance(generation_id, str)
            or not generation_id
            or generation_id != generation_id.strip()
        ):
            raise ValueError("generation_id must be a non-empty normalized string")
        if isinstance(transition_ids, (str, bytes)):
            raise ValueError("transition_ids must be a sequence of strings")
        values = tuple(transition_ids)
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in values
        ):
            raise ValueError("transition_ids contain an invalid ID")
        if len(set(values)) != len(values):
            raise ValueError("transition_ids contain duplicates")
        return generation_id, values

    def _validate_causal_model_state(self, value: object) -> Mapping[str, Tensor]:
        if not isinstance(value, Mapping):
            raise ValueError("invalid Core causal model state")
        expected = self.state_dict()
        if value.keys() != expected.keys():
            raise ValueError("Core causal model keys mismatch")
        for name, current in expected.items():
            candidate = value[name]
            if (
                not isinstance(candidate, Tensor)
                or candidate.shape != current.shape
                or candidate.layout != current.layout
                or candidate.dtype != current.dtype
            ):
                raise ValueError(f"Core causal model tensor mismatch for {name!r}")
        return value

    @staticmethod
    def _validate_causal_optimizer(optimizer: torch.optim.AdamW) -> None:
        moment_names = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        for parameter, state in optimizer.state.items():
            if not isinstance(state, dict):
                raise ValueError("invalid Core causal optimizer entry")
            if any(name not in moment_names | {"step"} for name in state):
                raise ValueError("invalid Core causal optimizer field")
            for name, value in state.items():
                if not isinstance(value, Tensor) or value.layout != torch.strided:
                    raise ValueError("invalid Core causal optimizer tensor")
                if name == "step":
                    if value.numel() != 1:
                        raise ValueError("invalid Core causal optimizer step")
                elif value.shape != parameter.shape:
                    raise ValueError("invalid Core causal optimizer moment shape")

    def _validate_causal_state_snapshot(
        self,
        value: object,
        *,
        name: str,
    ) -> None:
        if not isinstance(value, _StateSnapshot):
            raise ValueError(f"invalid Core {name} snapshot")
        reference = self._snapshot_state(self._new_state())
        fixed = (
            ("work", value.work, reference.work),
            ("slow", value.slow, reference.slow),
            ("ssm", value.ssm, reference.ssm),
            ("previous_output", value.previous_output, reference.previous_output),
        )
        if not isinstance(value.recurrent, tuple) or len(value.recurrent) != 4:
            raise ValueError(f"invalid Core {name} recurrent snapshot")
        fixed += tuple(
            (f"recurrent[{index}]", candidate, expected)
            for index, (candidate, expected) in enumerate(
                zip(value.recurrent, reference.recurrent, strict=True)
            )
        )
        for field, candidate, expected in fixed:
            if (
                not isinstance(candidate, Tensor)
                or candidate.shape != expected.shape
                or candidate.layout != torch.strided
                or candidate.dtype != torch.float32
                or not bool(torch.isfinite(candidate).all())
            ):
                raise ValueError(f"invalid Core {name} tensor {field!r}")
        history = value.history
        if (
            not isinstance(history, Tensor)
            or history.layout != torch.strided
            or history.dtype != torch.float32
            or history.ndim != 2
            or history.shape[1] != self.config.hidden_size
            or history.shape[0] > self.config.history_size
            or not bool(torch.isfinite(history).all())
        ):
            raise ValueError(f"invalid Core {name} history")

    def _restore_tensor(self, value: Tensor) -> Tensor:
        return value.to(device=self.device, dtype=self.config.dtype)

    def _state_from_snapshot(self, snapshot: _StateSnapshot) -> _PersistentState:
        return _PersistentState(
            work=self._restore_tensor(snapshot.work),
            slow=self._restore_tensor(snapshot.slow),
            ssm=self._restore_tensor(snapshot.ssm),
            recurrent=tuple(self._restore_tensor(value) for value in snapshot.recurrent),
            history=self._restore_tensor(snapshot.history),
            previous_output=self._restore_tensor(snapshot.previous_output),
        )

    @staticmethod
    def _detach_state(state: _PersistentState) -> _PersistentState:
        return _PersistentState(
            work=state.work.detach(),
            slow=state.slow.detach(),
            ssm=state.ssm.detach(),
            recurrent=tuple(value.detach() for value in state.recurrent),
            history=state.history.detach(),
            previous_output=state.previous_output.detach(),
        )

    def _store_trace(self, transition_id: str, trace: _LearningTrace) -> None:
        self._traces[transition_id] = trace
        self._traces.move_to_end(transition_id)
        self._traces_created += 1
        while len(self._traces) > self.config.max_traces:
            self._traces.popitem(last=False)
            self._traces_evicted += 1

    @staticmethod
    def _rms_distance(left: Tensor, right: Tensor) -> float:
        value = torch.sqrt(torch.mean((left - right) ** 2))
        return float(value.detach().cpu().item())

    @staticmethod
    def _cosine_novelty(current: Tensor, previous: Tensor) -> float:
        current_norm = torch.linalg.vector_norm(current)
        previous_norm = torch.linalg.vector_norm(previous)
        if float(current_norm.detach().cpu().item()) <= 1e-12:
            return 0.0
        if float(previous_norm.detach().cpu().item()) <= 1e-12:
            return 1.0
        cosine = torch.sum(current * previous) / (current_norm * previous_norm + 1e-12)
        novelty = torch.clamp(1.0 - cosine, min=0.0, max=2.0)
        return float(novelty.detach().cpu().item())

    @staticmethod
    def _validate_config(config: PersonalCoreConfig) -> None:
        positive = (
            config.input_size,
            config.context_size,
            config.hidden_size,
            config.output_size,
            config.workspace_heads,
            config.attention_heads,
            config.router_hidden_size,
            config.router_top_k,
            config.ssm_depth,
            config.recurrent_depth,
            config.tcn_depth,
            config.tcn_kernel_size,
            config.attention_depth,
            config.attention_feedforward_multiplier,
            config.history_size,
            config.max_traces,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("core sizes/counts must be > 0")
        if config.hidden_size % config.workspace_heads != 0:
            raise ValueError("hidden_size must be divisible by workspace_heads")
        if config.hidden_size % config.attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if config.router_top_k > len(EXPERT_NAMES):
            raise ValueError("router_top_k exceeds available expert count")
        if config.tcn_kernel_size <= 1:
            raise ValueError("tcn_kernel_size must be > 1")
        if not (0.0 <= config.dropout < 1.0):
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if not (
            0.0 <= config.slow_decay_floor < config.slow_decay_ceiling <= 1.0
        ):
            raise ValueError("invalid slow decay range")
        if config.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if config.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if config.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be > 0")

    @staticmethod
    def _resolve_device(requested: str | None) -> torch.device:
        if requested:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
