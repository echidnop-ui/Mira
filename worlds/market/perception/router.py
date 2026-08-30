from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import exp, isfinite, log, sqrt, tanh
from random import Random
from typing import Mapping

from worlds.market.perception.interface import (
    ExpertOutput,
    MarketRouter,
    RouterDecision,
    RouterInput,
    RouterTelemetry,
)


@dataclass(slots=True)
class _ExpertParameters:
    input_size: int
    projection: list[list[float]]
    projection_bias: list[float]
    score_bias: float


@dataclass(frozen=True, slots=True)
class _ExpertRouteTrace:
    input_vector: tuple[float, ...]
    hidden: tuple[float, ...]
    weight: float


@dataclass(frozen=True, slots=True)
class _RouteTrace:
    transition_id: str
    experts: Mapping[str, _ExpertRouteTrace]
    temperature: float


class AdaptiveMarketRouter(MarketRouter):
    """
    Trainable neural Router for heterogeneous Market experts.

    Each available expert contributes:
        expert representation
        + explicit uncertainty state
        + market context
        + optional shared Heiwa context

    The Router maps those inputs to expert logits and then to normalized
    weights. It never produces BUY/SELL/HOLD directly.

    Learning is tied to the exact transition_id used during routing.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 24,
        learning_rate: float = 1e-3,
        temperature: float = 1.0,
        top_k: int | None = None,
        gradient_clip: float = 5.0,
        max_traces: int = 4096,
        seed: int = 20260827,
    ) -> None:
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0 or None")
        if gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be > 0")
        if max_traces <= 0:
            raise ValueError("max_traces must be > 0")

        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)
        self.top_k = top_k
        self.gradient_clip = float(gradient_clip)
        self.max_traces = int(max_traces)
        self.seed = int(seed)

        self._rng = Random(self.seed)

        self._score_weights = self._random_vector(self.hidden_size)
        self._experts: dict[str, _ExpertParameters] = {}
        self._traces: OrderedDict[str, _RouteTrace] = OrderedDict()

        self._forward_calls = 0
        self._learning_updates = 0
        self._last_entropy = 0.0
        self._last_gradient_norm = 0.0
        self._weight_sums: dict[str, float] = {}
        self._selection_counts: dict[str, int] = {}

    def route(self, router_input: RouterInput) -> RouterDecision:
        available = tuple(
            output
            for output in router_input.expert_outputs
            if output.available
        )

        if not available:
            self._forward_calls += 1
            self._last_entropy = 0.0
            self._traces.pop(router_input.transition_id, None)

            return RouterDecision(
                transition_id=router_input.transition_id,
                weights={},
                active_experts=(),
                entropy=0.0,
                diagnostics={
                    "available_experts": 0.0,
                    "active_experts": 0.0,
                },
            )

        shared_context = self._validated_context(
            router_input.shared_context,
            name="shared_context",
        )
        market_context = self._validated_context(
            router_input.market_context,
            name="market_context",
        )

        logits: dict[str, float] = {}
        local: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}

        for output in available:
            input_vector = self._build_input_vector(
                output,
                market_context=market_context,
                shared_context=shared_context,
            )

            parameters = self._ensure_expert(
                output.expert_name,
                input_size=len(input_vector),
            )

            hidden = self._project(parameters, input_vector)

            logit = (
                self._dot(self._score_weights, hidden)
                + parameters.score_bias
            )

            logits[output.expert_name] = logit
            local[output.expert_name] = (input_vector, hidden)

        active_names = self._select_active(logits)
        weights = self._softmax(logits, active_names)
        entropy = self._entropy(weights)

        trace_experts = {
            name: _ExpertRouteTrace(
                input_vector=local[name][0],
                hidden=local[name][1],
                weight=weights[name],
            )
            for name in active_names
        }

        self._store_trace(
            _RouteTrace(
                transition_id=router_input.transition_id,
                experts=trace_experts,
                temperature=self.temperature,
            )
        )

        self._forward_calls += 1
        self._last_entropy = entropy

        for name, weight in weights.items():
            self._weight_sums[name] = (
                self._weight_sums.get(name, 0.0) + weight
            )

        for name in active_names:
            self._selection_counts[name] = (
                self._selection_counts.get(name, 0) + 1
            )

        return RouterDecision(
            transition_id=router_input.transition_id,
            weights=dict(weights),
            active_experts=tuple(active_names),
            entropy=entropy,
            diagnostics={
                "available_experts": float(len(available)),
                "active_experts": float(len(active_names)),
                "max_weight": max(weights.values()) if weights else 0.0,
                "temperature": self.temperature,
            },
        )

    def learn(
        self,
        transition_id: str,
        weight_gradients: Mapping[str, float],
    ) -> bool:
        trace = self._traces.pop(transition_id, None)

        if trace is None or not trace.experts:
            return False

        for name, gradient in weight_gradients.items():
            if not isfinite(float(gradient)):
                raise ValueError(
                    f"Non-finite Router weight gradient for {name!r}"
                )

        active_names = tuple(trace.experts)

        gradients = {
            name: float(weight_gradients.get(name, 0.0))
            for name in active_names
        }

        weighted_mean = sum(
            trace.experts[name].weight * gradients[name]
            for name in active_names
        )

        logit_gradients: dict[str, float] = {}

        for name in active_names:
            route_trace = trace.experts[name]

            gradient = (
                route_trace.weight
                * (gradients[name] - weighted_mean)
                / trace.temperature
            )

            logit_gradients[name] = self._clip(gradient)

        self._last_gradient_norm = sqrt(
            sum(value * value for value in logit_gradients.values())
        )

        old_score_weights = tuple(self._score_weights)

        score_weight_gradient = [0.0] * self.hidden_size

        for name in active_names:
            route_trace = trace.experts[name]
            dlogit = logit_gradients[name]

            for index, hidden_value in enumerate(route_trace.hidden):
                score_weight_gradient[index] += dlogit * hidden_value

        for name in active_names:
            route_trace = trace.experts[name]
            parameters = self._experts[name]
            dlogit = logit_gradients[name]

            parameters.score_bias -= self.learning_rate * dlogit

            for hidden_index, hidden_value in enumerate(route_trace.hidden):
                dhidden = dlogit * old_score_weights[hidden_index]
                dz = self._clip(
                    dhidden * (1.0 - hidden_value * hidden_value)
                )

                parameters.projection_bias[hidden_index] -= (
                    self.learning_rate * dz
                )

                row = parameters.projection[hidden_index]

                for input_index, input_value in enumerate(
                    route_trace.input_vector
                ):
                    row[input_index] -= (
                        self.learning_rate * dz * input_value
                    )

        for index, gradient in enumerate(score_weight_gradient):
            self._score_weights[index] -= (
                self.learning_rate * self._clip(gradient)
            )

        self._learning_updates += 1
        return True

    def telemetry(self) -> RouterTelemetry:
        if self._forward_calls:
            average_weights = {
                name: total / self._forward_calls
                for name, total in self._weight_sums.items()
            }
        else:
            average_weights = {}

        return RouterTelemetry(
            forward_calls=self._forward_calls,
            learning_updates=self._learning_updates,
            last_entropy=self._last_entropy,
            last_gradient_norm=self._last_gradient_norm,
            average_weights=average_weights,
            selection_counts=dict(self._selection_counts),
            diagnostics={
                "registered_experts": float(len(self._experts)),
                "pending_traces": float(len(self._traces)),
                "temperature": self.temperature,
                "top_k": (
                    -1.0
                    if self.top_k is None
                    else float(self.top_k)
                ),
            },
        )

    def snapshot(self) -> Mapping[str, object]:
        return {
            "version": 1,
            "config": {
                "hidden_size": self.hidden_size,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "gradient_clip": self.gradient_clip,
                "max_traces": self.max_traces,
                "seed": self.seed,
            },
            "score_weights": list(self._score_weights),
            "experts": {
                name: {
                    "input_size": parameters.input_size,
                    "projection": [
                        list(row)
                        for row in parameters.projection
                    ],
                    "projection_bias": list(
                        parameters.projection_bias
                    ),
                    "score_bias": parameters.score_bias,
                }
                for name, parameters in self._experts.items()
            },
            "counters": {
                "forward_calls": self._forward_calls,
                "learning_updates": self._learning_updates,
                "last_entropy": self._last_entropy,
                "last_gradient_norm": self._last_gradient_norm,
                "weight_sums": dict(self._weight_sums),
                "selection_counts": dict(self._selection_counts),
            },
        }

    def restore(self, snapshot: Mapping[str, object]) -> None:
        version = int(snapshot.get("version", 0))

        if version != 1:
            raise ValueError(
                f"Unsupported router snapshot version: {version}"
            )

        config = snapshot.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("Router snapshot has invalid config")

        if int(config["hidden_size"]) != self.hidden_size:
            raise ValueError(
                "Router snapshot hidden_size does not match current Router"
            )

        raw_score_weights = snapshot.get("score_weights")
        if not isinstance(raw_score_weights, list):
            raise ValueError("Router snapshot has invalid score_weights")

        score_weights = [float(value) for value in raw_score_weights]

        if len(score_weights) != self.hidden_size:
            raise ValueError("Router score_weights size mismatch")

        raw_experts = snapshot.get("experts")
        if not isinstance(raw_experts, Mapping):
            raise ValueError("Router snapshot has invalid experts")

        experts: dict[str, _ExpertParameters] = {}

        for raw_name, raw_parameters in raw_experts.items():
            name = str(raw_name)

            if not isinstance(raw_parameters, Mapping):
                raise ValueError(
                    f"Invalid Router parameters for expert {name!r}"
                )

            input_size = int(raw_parameters["input_size"])
            raw_projection = raw_parameters["projection"]
            raw_bias = raw_parameters["projection_bias"]

            if not isinstance(raw_projection, list):
                raise ValueError(
                    f"Invalid projection for expert {name!r}"
                )

            if not isinstance(raw_bias, list):
                raise ValueError(
                    f"Invalid projection bias for expert {name!r}"
                )

            projection = [
                [float(value) for value in row]
                for row in raw_projection
            ]
            projection_bias = [float(value) for value in raw_bias]

            if len(projection) != self.hidden_size:
                raise ValueError(
                    f"Projection hidden size mismatch for {name!r}"
                )

            if len(projection_bias) != self.hidden_size:
                raise ValueError(
                    f"Projection bias size mismatch for {name!r}"
                )

            for row in projection:
                if len(row) != input_size:
                    raise ValueError(
                        f"Projection input size mismatch for {name!r}"
                    )

            experts[name] = _ExpertParameters(
                input_size=input_size,
                projection=projection,
                projection_bias=projection_bias,
                score_bias=float(raw_parameters["score_bias"]),
            )

        counters = snapshot.get("counters", {})
        if not isinstance(counters, Mapping):
            raise ValueError("Router snapshot has invalid counters")

        raw_weight_sums = counters.get("weight_sums", {})
        raw_selection_counts = counters.get("selection_counts", {})

        if not isinstance(raw_weight_sums, Mapping):
            raise ValueError("Router snapshot has invalid weight_sums")

        if not isinstance(raw_selection_counts, Mapping):
            raise ValueError(
                "Router snapshot has invalid selection_counts"
            )

        self._score_weights = score_weights
        self._experts = experts

        self._forward_calls = int(counters.get("forward_calls", 0))
        self._learning_updates = int(
            counters.get("learning_updates", 0)
        )
        self._last_entropy = float(
            counters.get("last_entropy", 0.0)
        )
        self._last_gradient_norm = float(
            counters.get("last_gradient_norm", 0.0)
        )

        self._weight_sums = {
            str(name): float(value)
            for name, value in raw_weight_sums.items()
        }
        self._selection_counts = {
            str(name): int(value)
            for name, value in raw_selection_counts.items()
        }

        self._traces.clear()

    def reset(self) -> None:
        self._rng = Random(self.seed)
        self._score_weights = self._random_vector(self.hidden_size)
        self._experts.clear()
        self._traces.clear()

        self._forward_calls = 0
        self._learning_updates = 0
        self._last_entropy = 0.0
        self._last_gradient_norm = 0.0
        self._weight_sums.clear()
        self._selection_counts.clear()

    def _build_input_vector(
        self,
        output: ExpertOutput,
        *,
        market_context: tuple[float, ...],
        shared_context: tuple[float, ...],
    ) -> tuple[float, ...]:
        if output.uncertainty is None:
            uncertainty = 0.0
            uncertainty_known = 0.0
        else:
            uncertainty = float(output.uncertainty)
            uncertainty_known = 1.0

        return (
            *output.representation,
            uncertainty,
            uncertainty_known,
            *market_context,
            *shared_context,
        )

    @staticmethod
    def _validated_context(
        context: tuple[float, ...] | None,
        *,
        name: str,
    ) -> tuple[float, ...]:
        if context is None:
            return ()

        result = tuple(float(value) for value in context)

        for value in result:
            if not isfinite(value):
                raise ValueError(
                    f"{name} contains non-finite value"
                )

        return result

    def _ensure_expert(
        self,
        name: str,
        *,
        input_size: int,
    ) -> _ExpertParameters:
        existing = self._experts.get(name)

        if existing is not None:
            if existing.input_size != input_size:
                raise ValueError(
                    f"Router input size for expert {name!r} changed from "
                    f"{existing.input_size} to {input_size}. "
                    "Keep context dimensions stable or reset/reconfigure "
                    "the Router."
                )
            return existing

        if input_size <= 0:
            raise ValueError(
                f"Cannot register expert {name!r} with empty input"
            )

        scale = sqrt(2.0 / (input_size + self.hidden_size))

        projection = [
            [
                self._rng.uniform(-scale, scale)
                for _ in range(input_size)
            ]
            for _ in range(self.hidden_size)
        ]

        parameters = _ExpertParameters(
            input_size=input_size,
            projection=projection,
            projection_bias=[0.0] * self.hidden_size,
            score_bias=0.0,
        )

        self._experts[name] = parameters
        return parameters

    def _random_vector(self, size: int) -> list[float]:
        scale = sqrt(2.0 / max(1, size))

        return [
            self._rng.uniform(-scale, scale)
            for _ in range(size)
        ]

    @staticmethod
    def _dot(
        left: list[float] | tuple[float, ...],
        right: list[float] | tuple[float, ...],
    ) -> float:
        if len(left) != len(right):
            raise ValueError("dot product size mismatch")

        return sum(
            a * b
            for a, b in zip(left, right)
        )

    def _project(
        self,
        parameters: _ExpertParameters,
        input_vector: tuple[float, ...],
    ) -> tuple[float, ...]:
        return tuple(
            tanh(bias + self._dot(row, input_vector))
            for row, bias in zip(
                parameters.projection,
                parameters.projection_bias,
            )
        )

    def _select_active(
        self,
        logits: Mapping[str, float],
    ) -> tuple[str, ...]:
        names = tuple(logits)

        if self.top_k is None or self.top_k >= len(names):
            return names

        return tuple(
            sorted(
                names,
                key=lambda name: logits[name],
                reverse=True,
            )[: self.top_k]
        )

    def _softmax(
        self,
        logits: Mapping[str, float],
        active_names: tuple[str, ...],
    ) -> dict[str, float]:
        if not active_names:
            return {}

        scaled = {
            name: logits[name] / self.temperature
            for name in active_names
        }

        maximum = max(scaled.values())

        exponentials = {
            name: exp(value - maximum)
            for name, value in scaled.items()
        }

        total = sum(exponentials.values())

        if total <= 0.0 or not isfinite(total):
            raise FloatingPointError("Router softmax became invalid")

        return {
            name: value / total
            for name, value in exponentials.items()
        }

    @staticmethod
    def _entropy(weights: Mapping[str, float]) -> float:
        return -sum(
            weight * log(weight)
            for weight in weights.values()
            if weight > 0.0
        )

    def _store_trace(self, trace: _RouteTrace) -> None:
        self._traces.pop(trace.transition_id, None)
        self._traces[trace.transition_id] = trace

        while len(self._traces) > self.max_traces:
            self._traces.popitem(last=False)

    def _clip(self, value: float) -> float:
        if value > self.gradient_clip:
            return self.gradient_clip
        if value < -self.gradient_clip:
            return -self.gradient_clip
        return value
