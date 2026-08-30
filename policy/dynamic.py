from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import math
import re
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor, nn

from contracts.cognition import ActionIntent, Affordance
from core.interface import IntegratedCognitiveState


_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-zА-Яа-яЁё]+")
_CAUSAL_SNAPSHOT_FORMAT = "heiwa.policy.causal-snapshot"
_CAUSAL_SNAPSHOT_VERSION = 1


@dataclass(frozen=True, slots=True)
class DynamicPolicyConfig:
    state_size: int = 96
    affordance_feature_size: int = 64
    hidden_size: int = 96

    learning_rate: float = 1e-3
    gradient_clip: float = 5.0
    max_traces: int = 256
    specialist_prior_weight: float = 0.25
    entropy_coefficient: float = 0.01
    exploration_mixture: float = 0.25

    device: str | None = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.state_size <= 0:
            raise ValueError("state_size must be > 0")
        if self.affordance_feature_size <= 0:
            raise ValueError("affordance_feature_size must be > 0")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be > 0")
        if self.max_traces <= 0:
            raise ValueError("max_traces must be > 0")
        if not math.isfinite(self.specialist_prior_weight) or self.specialist_prior_weight < 0.0:
            raise ValueError("specialist_prior_weight must be finite and >= 0")
        if not math.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be finite and >= 0")
        if not math.isfinite(self.exploration_mixture) or not 0.0 <= self.exploration_mixture < 1.0:
            raise ValueError("exploration_mixture must be in [0,1)")


@dataclass(frozen=True, slots=True)
class DynamicPolicyTelemetry:
    decision_calls: int
    learning_updates: int

    traces_created: int
    traces_consumed: int
    traces_evicted: int
    duplicate_transition_rejections: int

    last_available_count: int
    last_chosen_action: str | None
    last_score_margin: float
    last_gradient_norm: float


@dataclass(frozen=True, slots=True)
class _PolicyTrace:
    transition_id: str
    state_vector: tuple[float, ...]
    affordances: tuple[Affordance, ...]
    chosen_index: int
    now: float


class NoAvailableAffordanceError(ValueError):
    """Raised when Policy has no currently executable candidate to score."""


class DynamicAffordancePolicy(nn.Module):
    """
    Trainable shared Policy over a variable set of Affordances.

    There is deliberately no global action enum and no BUY/SELL-specific head.

    The same network can score:
        market.buy
        forum.reply
        game.move
        desktop.open
        future actions that did not exist when this class was written

    Affordances are converted into a stable, world-agnostic hashed feature
    representation. Shared cognition and each candidate representation are then
    jointly scored by one trainable network.

    Delayed learning is tied to transition_id. The Policy stores only the
    minimal exact decision input needed to replay the decision under current
    weights. It does not claim historical-weight-exact backpropagation.
    """

    def __init__(
        self,
        config: DynamicPolicyConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config if config is not None else DynamicPolicyConfig()
        self.device = torch.device(
            self.config.device
            if self.config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        h = self.config.hidden_size
        s = self.config.state_size
        a = self.config.affordance_feature_size

        self.state_encoder = nn.Sequential(
            nn.LayerNorm(s),
            nn.Linear(s, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.Tanh(),
        )

        self.affordance_encoder = nn.Sequential(
            nn.LayerNorm(a),
            nn.Linear(a, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.Tanh(),
        )

        self.scorer = nn.Sequential(
            nn.Linear(h * 4, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

        self.to(
            device=self.device,
            dtype=self.config.dtype,
        )

        self._optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
        )

        self._traces: OrderedDict[str, _PolicyTrace] = OrderedDict()

        self._decision_calls = 0
        self._learning_updates = 0
        self._traces_created = 0
        self._traces_consumed = 0
        self._traces_evicted = 0
        self._duplicate_transition_rejections = 0

        self._last_available_count = 0
        self._last_chosen_action: str | None = None
        self._last_score_margin = 0.0
        self._last_gradient_norm = 0.0
        self._last_score_diagnostics: dict[str, object] = {}

    # ========================================================
    # Public decision boundary
    # ========================================================

    def choose(
        self,
        state: IntegratedCognitiveState,
        affordances: Sequence[Affordance],
        *,
        now: float,
        remember_for_learning: bool = True,
        exploration_temperature: float | None = None,
    ) -> ActionIntent:
        self._validate_state(state)

        now = float(now)
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        if exploration_temperature is not None:
            exploration_temperature = float(exploration_temperature)
            if (
                not math.isfinite(exploration_temperature)
                or exploration_temperature <= 0.0
            ):
                raise ValueError(
                    "exploration_temperature must be finite and > 0 "
                    "when provided"
                )

        candidates = self._available_affordances(
            affordances,
            now=now,
        )
        if not candidates:
            raise NoAvailableAffordanceError(
                "Policy received no currently available Affordances"
            )

        if (
            remember_for_learning
            and state.transition_id in self._traces
        ):
            self._duplicate_transition_rejections += 1
            raise ValueError(
                "Policy already has an unconsumed trace for transition_id "
                f"{state.transition_id!r}"
            )

        state_tensor = self._state_tensor(
            state.vector
        )
        affordance_tensor = self._affordance_tensor(
            candidates,
            now=now,
        )

        with torch.no_grad():
            scores = self._combined_scores(state_tensor, affordance_tensor, candidates)
            log_probs = torch.log_softmax(scores, dim=0)
            probabilities = log_probs.exp()
            self._last_score_diagnostics = {
                "scores": tuple((item.action, float(scores[index]), float(probabilities[index]))
                                for index, item in enumerate(candidates)),
                "entropy": float(-(probabilities * log_probs).sum()),
                "top1_margin": self._score_margin(scores),
            }

            if exploration_temperature is None:
                chosen_index = int(
                    torch.argmax(scores).item()
                )
            else:
                probabilities = torch.softmax(scores / exploration_temperature, dim=0)
                mixture = float(self.config.exploration_mixture)
                if mixture > 0.0 and probabilities.numel() > 1:
                    probabilities = (
                        (1.0 - mixture) * probabilities
                        + mixture / probabilities.numel()
                    )
                chosen_index = int(
                    torch.multinomial(
                        probabilities,
                        num_samples=1,
                    ).item()
                )

        chosen = candidates[chosen_index]

        self._decision_calls += 1
        self._last_available_count = len(candidates)
        self._last_chosen_action = chosen.action
        self._last_score_margin = self._score_margin(scores)

        if remember_for_learning:
            self._remember_trace(
                _PolicyTrace(
                    transition_id=state.transition_id,
                    state_vector=tuple(
                        float(value)
                        for value in state.vector
                    ),
                    affordances=candidates,
                    chosen_index=chosen_index,
                    now=now,
                )
            )

        return ActionIntent(
            id=(
                f"intent:{state.transition_id}:"
                f"{chosen.id}"
            ),
            transition_id=state.transition_id,
            affordance_id=chosen.id,
            world=chosen.world,
            action=chosen.action,
            created_at=now,
            # Parameterization is a separate responsibility. Do not copy the
            # Affordance payload and pretend it was cognitively chosen.
            parameters={},
        )

    # ========================================================
    # Causal learning
    # ========================================================

    def learn(
        self,
        transition_id: str,
        advantage: float,
    ) -> bool:
        if not transition_id:
            raise ValueError("transition_id must not be empty")

        advantage = float(advantage)
        if not math.isfinite(advantage):
            raise ValueError("advantage must be finite")

        trace = self._traces.get(
            transition_id
        )
        if trace is None:
            return False

        state_tensor = self._state_tensor(
            trace.state_vector
        )
        affordance_tensor = self._affordance_tensor(
            trace.affordances,
            now=trace.now,
        )

        scores = self._combined_scores(state_tensor, affordance_tensor, trace.affordances)
        log_probs = torch.log_softmax(
            scores,
            dim=0,
        )

        chosen_log_prob = log_probs[
            trace.chosen_index
        ]
        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum()
        loss = (
            -advantage
            * chosen_log_prob
            - self.config.entropy_coefficient * entropy
        )

        self._optimizer.zero_grad(
            set_to_none=True
        )
        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=self.config.gradient_clip,
        )
        self._optimizer.step()

        self._last_gradient_norm = float(
            gradient_norm.detach().cpu().item()
            if isinstance(gradient_norm, Tensor)
            else gradient_norm
        )

        self._learning_updates += 1
        self._traces_consumed += 1
        del self._traces[transition_id]

        return True

    def discard_trace(
        self,
        transition_id: str,
    ) -> bool:
        if transition_id in self._traces:
            del self._traces[transition_id]
            return True
        return False

    def has_trace(
        self,
        transition_id: str,
    ) -> bool:
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
                "Policy causal snapshot trace IDs do not match live traces: "
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
            "optimizer": self._optimizer.state_dict(),
            "traces": tuple(self._traces.items()),
            "telemetry": {
                "decision_calls": self._decision_calls,
                "learning_updates": self._learning_updates,
                "traces_created": self._traces_created,
                "traces_consumed": self._traces_consumed,
                "traces_evicted": self._traces_evicted,
                "duplicate_transition_rejections": self._duplicate_transition_rejections,
                "last_available_count": self._last_available_count,
                "last_chosen_action": self._last_chosen_action,
                "last_score_margin": self._last_score_margin,
                "last_gradient_norm": self._last_gradient_norm,
                "last_score_diagnostics": dict(self._last_score_diagnostics),
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
            raise ValueError("Policy causal snapshot must be bytes")
        try:
            payload = torch.load(
                io.BytesIO(snapshot), map_location="cpu", weights_only=False
            )
        except Exception as exc:
            raise ValueError(f"invalid Policy causal snapshot: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("invalid Policy causal snapshot root")
        if payload.get("format") != _CAUSAL_SNAPSHOT_FORMAT:
            raise ValueError("unsupported Policy causal snapshot format")
        if payload.get("version") != _CAUSAL_SNAPSHOT_VERSION:
            raise ValueError("unsupported Policy causal snapshot version")
        if payload.get("config") != self.config:
            raise ValueError("Policy causal snapshot config mismatch")
        if payload.get("generation_id") != generation:
            raise ValueError("Policy causal snapshot generation mismatch")
        if payload.get("transition_ids") != expected_ids:
            raise ValueError("Policy causal snapshot transition IDs mismatch")

        model_state = self._validate_causal_model_state(payload.get("model"))
        optimizer_state = payload.get("optimizer")
        if not isinstance(optimizer_state, dict):
            raise ValueError("invalid Policy causal optimizer state")
        staged_optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.learning_rate
        )
        try:
            staged_optimizer.load_state_dict(optimizer_state)
        except Exception as exc:
            raise ValueError(f"invalid Policy causal optimizer state: {exc}") from exc
        self._validate_causal_optimizer(staged_optimizer)

        raw_traces = payload.get("traces")
        if not isinstance(raw_traces, tuple) or len(raw_traces) != len(expected_ids):
            raise ValueError("invalid Policy causal traces")
        staged_traces: OrderedDict[str, _PolicyTrace] = OrderedDict()
        for index, item in enumerate(raw_traces):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("invalid Policy causal trace entry")
            transition_id, trace = item
            if transition_id != expected_ids[index]:
                raise ValueError("Policy causal trace order mismatch")
            if not isinstance(trace, _PolicyTrace):
                raise ValueError("invalid Policy causal trace type")
            if trace.transition_id != transition_id:
                raise ValueError("Policy causal trace transition mismatch")
            if (
                not isinstance(trace.state_vector, tuple)
                or not isinstance(trace.affordances, tuple)
                or len(trace.state_vector) != self.config.state_size
                or not all(math.isfinite(float(value)) for value in trace.state_vector)
                or not math.isfinite(float(trace.now))
                or type(trace.chosen_index) is not int
            ):
                raise ValueError("invalid Policy causal trace state")
            candidates = self._available_affordances(
                trace.affordances, now=float(trace.now)
            )
            if (
                candidates != trace.affordances
                or not candidates
                or not 0 <= trace.chosen_index < len(candidates)
            ):
                raise ValueError("invalid Policy causal trace affordances")
            staged_traces[transition_id] = trace

        telemetry = payload.get("telemetry")
        if not isinstance(telemetry, dict):
            raise ValueError("invalid Policy causal telemetry")
        counter_names = (
            "decision_calls", "learning_updates", "traces_created",
            "traces_consumed", "traces_evicted",
            "duplicate_transition_rejections", "last_available_count",
        )
        staged_counters: dict[str, int] = {}
        for name in counter_names:
            value = telemetry.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid Policy causal telemetry {name!r}")
            staged_counters[name] = value
        last_action = telemetry.get("last_chosen_action")
        if last_action is not None and not isinstance(last_action, str):
            raise ValueError("invalid Policy last_chosen_action")
        last_margin = telemetry.get("last_score_margin")
        last_gradient = telemetry.get("last_gradient_norm")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (last_margin, last_gradient)
        ):
            raise ValueError("invalid Policy causal floating telemetry")
        diagnostics = telemetry.get("last_score_diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError("invalid Policy score diagnostics")
        training = telemetry.get("training")
        if type(training) is not bool:
            raise ValueError("invalid Policy training mode")

        self.load_state_dict(model_state)
        self._optimizer = staged_optimizer
        self._traces = staged_traces
        self._decision_calls = staged_counters["decision_calls"]
        self._learning_updates = staged_counters["learning_updates"]
        self._traces_created = staged_counters["traces_created"]
        self._traces_consumed = staged_counters["traces_consumed"]
        self._traces_evicted = staged_counters["traces_evicted"]
        self._duplicate_transition_rejections = staged_counters[
            "duplicate_transition_rejections"
        ]
        self._last_available_count = staged_counters["last_available_count"]
        self._last_chosen_action = last_action
        self._last_score_margin = float(last_margin)
        self._last_gradient_norm = float(last_gradient)
        self._last_score_diagnostics = dict(diagnostics)
        self.train(training)

    def score_diagnostics(self) -> dict[str, object]:
        """Compact diagnostics for the most recent decision; no learning state."""
        return dict(self._last_score_diagnostics)

    def reset_scorer(self) -> None:
        """Reinitialize only a collapsed final scoring head, preserving encoders."""
        for module in self.scorer.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        for parameter in self.scorer.parameters():
            self._optimizer.state.pop(parameter, None)

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
            raise ValueError("invalid Policy causal model state")
        expected = self.state_dict()
        if value.keys() != expected.keys():
            raise ValueError("Policy causal model keys mismatch")
        for name, current in expected.items():
            candidate = value[name]
            if (
                not isinstance(candidate, Tensor)
                or candidate.shape != current.shape
                or candidate.layout != current.layout
                or candidate.dtype != current.dtype
            ):
                raise ValueError(f"Policy causal model tensor mismatch for {name!r}")
        return value

    @staticmethod
    def _validate_causal_optimizer(optimizer: torch.optim.AdamW) -> None:
        moment_names = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        for parameter, state in optimizer.state.items():
            if not isinstance(state, dict):
                raise ValueError("invalid Policy causal optimizer entry")
            if any(name not in moment_names | {"step"} for name in state):
                raise ValueError("invalid Policy causal optimizer field")
            for name, value in state.items():
                if not isinstance(value, Tensor) or value.layout != torch.strided:
                    raise ValueError("invalid Policy causal optimizer tensor")
                if name == "step":
                    if value.numel() != 1:
                        raise ValueError("invalid Policy causal optimizer step")
                elif value.shape != parameter.shape:
                    raise ValueError("invalid Policy causal optimizer moment shape")

    # ========================================================
    # Telemetry
    # ========================================================

    def telemetry(
        self,
    ) -> DynamicPolicyTelemetry:
        return DynamicPolicyTelemetry(
            decision_calls=self._decision_calls,
            learning_updates=self._learning_updates,
            traces_created=self._traces_created,
            traces_consumed=self._traces_consumed,
            traces_evicted=self._traces_evicted,
            duplicate_transition_rejections=(
                self._duplicate_transition_rejections
            ),
            last_available_count=self._last_available_count,
            last_chosen_action=self._last_chosen_action,
            last_score_margin=self._last_score_margin,
            last_gradient_norm=self._last_gradient_norm,
        )

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save({"format": 1, "model": self.state_dict(),
                        "optimizer": self._optimizer.state_dict(),
                        "learning_updates": self._learning_updates}, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path: Path) -> bool:
        if not path.exists(): return False
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict): raise ValueError("invalid Policy checkpoint")
        self.load_state_dict(payload.get("model", payload.get("state_dict", payload)))
        if isinstance(payload.get("optimizer"), dict): self._optimizer.load_state_dict(payload["optimizer"])
        self._learning_updates = max(0, int(payload.get("learning_updates", 0)))
        self._traces.clear(); return True

    # ========================================================
    # Neural scoring
    # ========================================================

    def _score_tensors(
        self,
        state: Tensor,
        affordances: Tensor,
    ) -> Tensor:
        state_hidden = self.state_encoder(
            state
        )
        affordance_hidden = self.affordance_encoder(
            affordances
        )

        state_batch = state_hidden.expand(
            affordance_hidden.shape[0],
            -1,
        )

        interaction = torch.cat(
            (
                state_batch,
                affordance_hidden,
                state_batch * affordance_hidden,
                torch.abs(
                    state_batch - affordance_hidden
                ),
            ),
            dim=-1,
        )

        return self.scorer(
            interaction
        ).squeeze(-1)

    def _combined_scores(self, state: Tensor, affordances: Tensor,
                         candidates: Sequence[Affordance]) -> Tensor:
        """Keep learned logits trainable without allowing softmax saturation."""
        learned = self._score_tensors(state, affordances)
        centered = learned - learned.mean()
        # Detached scaling preserves score ordering and gradient direction while
        # preventing an old head with huge logits from becoming unlearnable.
        scale = centered.detach().square().mean().sqrt().clamp_min(1.0)
        return centered / scale + self._specialist_prior_logits(candidates)

    def _specialist_prior_logits(
        self, affordances: Sequence[Affordance],
    ) -> Tensor:
        """Optional generic specialist prior supplied by a World affordance.

        The shared Policy remains the final selector.  A specialist may expose
        ``policy_prior_probability`` in the affordance payload; this function
        converts the probabilities to centered log-priors so they guide an
        initially untrained shared Policy without becoming a hard decision.
        """
        weight = float(self.config.specialist_prior_weight)
        result = torch.zeros(len(affordances), device=self.device, dtype=self.config.dtype)
        if weight == 0.0 or not affordances:
            return result
        indexed: list[tuple[int, float]] = []
        for index, affordance in enumerate(affordances):
            raw = affordance.payload.get("policy_prior_probability")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            probability = float(raw)
            if math.isfinite(probability) and probability > 0.0:
                indexed.append((index, math.log(max(probability, 1e-8))))
        if not indexed:
            return result
        mean = sum(value for _, value in indexed) / len(indexed)
        for index, value in indexed:
            result[index] = (value - mean) * weight
        return result

    # ========================================================
    # Mechanical feature encoding
    # ========================================================

    def _state_tensor(
        self,
        values: Sequence[float],
    ) -> Tensor:
        if len(values) != self.config.state_size:
            raise ValueError(
                "IntegratedCognitiveState vector size mismatch: "
                f"expected {self.config.state_size}, got {len(values)}"
            )

        numeric = tuple(
            float(value)
            for value in values
        )
        if not all(
            math.isfinite(value)
            for value in numeric
        ):
            raise ValueError(
                "IntegratedCognitiveState vector contains non-finite values"
            )

        return torch.tensor(
            numeric,
            device=self.device,
            dtype=self.config.dtype,
        ).unsqueeze(0)

    def _affordance_tensor(
        self,
        affordances: Sequence[Affordance],
        *,
        now: float,
    ) -> Tensor:
        rows = [
            self._affordance_features(
                affordance,
                now=now,
            )
            for affordance in affordances
        ]

        return torch.tensor(
            rows,
            device=self.device,
            dtype=self.config.dtype,
        )

    def _affordance_features(
        self,
        affordance: Affordance,
        *,
        now: float,
    ) -> tuple[float, ...]:
        size = self.config.affordance_feature_size
        values = [0.0] * size

        self._add_text(
            values,
            "world",
            affordance.world,
        )
        self._add_text(
            values,
            "action",
            affordance.action,
        )

        self._add_numeric(
            values,
            "offered_age",
            max(
                0.0,
                now - float(affordance.offered_at),
            ),
        )

        if affordance.expires_at is not None:
            self._add_numeric(
                values,
                "time_to_expiry",
                float(affordance.expires_at) - now,
            )

        self._add_mapping(
            values,
            "payload",
            affordance.payload,
            depth=0,
        )
        self._add_mapping(
            values,
            "constraints",
            affordance.constraints,
            depth=0,
        )

        norm = math.sqrt(
            sum(
                value * value
                for value in values
            )
        )
        if norm > 1e-12:
            values = [
                value / norm
                for value in values
            ]

        return tuple(values)

    def _add_mapping(
        self,
        features: list[float],
        prefix: str,
        mapping: Mapping[str, object],
        *,
        depth: int,
    ) -> None:
        if depth > 2:
            return

        for index, key in enumerate(
            sorted(
                mapping,
                key=lambda item: str(item),
            )
        ):
            if index >= 16:
                break

            value = mapping[key]
            path = f"{prefix}.{key}"
            self._add_value(
                features,
                path,
                value,
                depth=depth,
            )

    def _add_value(
        self,
        features: list[float],
        path: str,
        value: object,
        *,
        depth: int,
    ) -> None:
        if value is None:
            self._add_text(
                features,
                path,
                "<none>",
            )
            return

        if isinstance(value, bool):
            self._add_numeric(
                features,
                path,
                1.0 if value else -1.0,
            )
            return

        if isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                self._add_numeric(
                    features,
                    path,
                    numeric,
                )
            return

        if isinstance(value, str):
            self._add_text(
                features,
                path,
                value,
            )
            return

        if isinstance(value, Mapping):
            self._add_mapping(
                features,
                path,
                value,
                depth=depth + 1,
            )
            return

        if (
            isinstance(value, Sequence)
            and not isinstance(
                value,
                (str, bytes, bytearray),
            )
        ):
            for index, item in enumerate(value):
                if index >= 8:
                    break
                self._add_value(
                    features,
                    f"{path}[{index}]",
                    item,
                    depth=depth + 1,
                )
            return

        self._add_text(
            features,
            path,
            type(value).__name__,
        )

    def _add_text(
        self,
        features: list[float],
        path: str,
        text: str,
    ) -> None:
        folded = text.strip().casefold()
        if not folded:
            return

        self._hash_add(
            features,
            f"{path}={folded}",
            1.0,
        )

        tokens = tuple(
            token
            for token in _TOKEN_SPLIT.split(folded)
            if token
        )
        for token in tokens[:8]:
            self._hash_add(
                features,
                f"{path}:token={token}",
                0.5,
            )

    def _add_numeric(
        self,
        features: list[float],
        path: str,
        value: float,
    ) -> None:
        # Generic compression: keeps sign/order-of-magnitude information
        # without imposing a market-specific scale.
        compressed = math.copysign(
            min(
                8.0,
                math.log1p(abs(value)),
            ),
            value,
        )
        self._hash_add(
            features,
            f"{path}:numeric",
            compressed,
        )

    @staticmethod
    def _hash_add(
        features: list[float],
        key: str,
        value: float,
    ) -> None:
        digest = hashlib.blake2b(
            key.encode("utf-8"),
            digest_size=16,
        ).digest()

        index = int.from_bytes(
            digest[:8],
            "little",
        ) % len(features)

        sign = (
            1.0
            if (digest[8] & 1) == 0
            else -1.0
        )

        features[index] += (
            sign * float(value)
        )

    # ========================================================
    # Validation / traces
    # ========================================================

    @staticmethod
    def _validate_state(
        state: IntegratedCognitiveState,
    ) -> None:
        if not state.transition_id:
            raise ValueError(
                "IntegratedCognitiveState.transition_id must not be empty"
            )

    @staticmethod
    def _available_affordances(
        affordances: Sequence[Affordance],
        *,
        now: float,
    ) -> tuple[Affordance, ...]:
        result: list[Affordance] = []

        for affordance in affordances:
            if not affordance.id:
                raise ValueError(
                    "Affordance.id must not be empty"
                )
            if not affordance.world:
                raise ValueError(
                    "Affordance.world must not be empty"
                )
            if not affordance.action:
                raise ValueError(
                    "Affordance.action must not be empty"
                )

            offered_at = float(
                affordance.offered_at
            )
            if not math.isfinite(offered_at):
                raise ValueError(
                    "Affordance.offered_at must be finite"
                )

            expires_at = affordance.expires_at
            if expires_at is not None:
                expires_at = float(
                    expires_at
                )
                if not math.isfinite(expires_at):
                    raise ValueError(
                        "Affordance.expires_at must be finite when present"
                    )
                if expires_at <= now:
                    continue

            result.append(
                affordance
            )

        return tuple(result)

    def _remember_trace(
        self,
        trace: _PolicyTrace,
    ) -> None:
        self._traces[
            trace.transition_id
        ] = trace
        self._traces_created += 1

        while (
            len(self._traces)
            > self.config.max_traces
        ):
            self._traces.popitem(
                last=False
            )
            self._traces_evicted += 1

    @staticmethod
    def _score_margin(
        scores: Tensor,
    ) -> float:
        detached = scores.detach()

        if detached.numel() <= 1:
            return 0.0

        top = torch.topk(
            detached,
            k=2,
        ).values

        return float(
            (top[0] - top[1]).cpu().item()
        )
