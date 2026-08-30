from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite, sqrt, tanh
from random import Random
from typing import Mapping

from worlds.market.perception.interface import (
    ExpertOutput,
    FusionBackward,
    FusionResult,
    FusionTelemetry,
    MarketFusion,
    RouterDecision,
    output_map,
    validate_router_decision,
)


# ============================================================
# Internal trainable state
# ============================================================


@dataclass(slots=True)
class _ExpertProjection:
    input_size: int
    weights: list[list[float]]
    bias: list[float]


@dataclass(frozen=True, slots=True)
class _ExpertFusionTrace:
    representation: tuple[float, ...]
    projected: tuple[float, ...]
    weight: float


@dataclass(frozen=True, slots=True)
class _FusionTrace:
    transition_id: str
    experts: Mapping[str, _ExpertFusionTrace]
    pre_activation: tuple[float, ...]
    output: tuple[float, ...]


@dataclass(slots=True)
class _ExpertProjectionGradient:
    weights: list[list[float]]
    bias: list[float]


@dataclass(slots=True)
class _PendingFusionUpdate:
    transition_id: str
    expert_gradients: dict[str, _ExpertProjectionGradient]


# ============================================================
# AdaptiveMarketFusion
# ============================================================


class AdaptiveMarketFusion(MarketFusion):
    """
    Trainable fusion layer for heterogeneous Market experts.

    Each expert may have a different representation size.

    For every active expert:

        expert representation
                ↓
        expert-specific projection
                ↓
         projected expert vector
                ↓
          multiply by Router weight
                ↓
             weighted sum
                ↓
               tanh
                ↓
          fixed-size fusion vector

    This design gives us three useful properties:

    1. Expert implementations remain independent.
       TCN, SSM, Attention, xLSTM and DeepLOB may expose different
       representation sizes.

    2. Router controls *how much* each expert contributes.

    3. Fusion learns *how* each expert representation should be translated
       into the common Market representation space.

    Backward pass decomposes downstream gradient into:

        dLoss / dExpertRepresentation

    and:

        dLoss / dRouterWeight

    while retaining Fusion's own parameter gradients until apply_update().
    """

    def __init__(
        self,
        *,
        output_size: int = 64,
        learning_rate: float = 1e-3,
        gradient_clip: float = 5.0,
        max_traces: int = 4096,
        seed: int = 20260827,
    ) -> None:
        if output_size <= 0:
            raise ValueError("output_size must be > 0")

        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")

        if gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be > 0")

        if max_traces <= 0:
            raise ValueError("max_traces must be > 0")

        self._output_size = int(output_size)
        self.learning_rate = float(learning_rate)
        self.gradient_clip = float(gradient_clip)
        self.max_traces = int(max_traces)
        self.seed = int(seed)

        self._rng = Random(self.seed)

        self._projections: dict[str, _ExpertProjection] = {}

        self._traces: OrderedDict[str, _FusionTrace] = OrderedDict()
        self._pending_updates: dict[str, _PendingFusionUpdate] = {}

        self._forward_calls = 0
        self._learning_updates = 0

        self._last_output_norm = 0.0
        self._last_gradient_norm = 0.0

        self._expert_projection_norms: dict[str, float] = {}

    # ========================================================
    # Public API
    # ========================================================

    @property
    def output_size(self) -> int:
        return self._output_size

    def fuse(
        self,
        expert_outputs: tuple[ExpertOutput, ...],
        decision: RouterDecision,
    ) -> FusionResult:
        """
        Fuse the exact set of experts selected by Router.

        A causal trace is stored under decision.transition_id so delayed
        downstream learning can return to this exact forward pass.
        """

        if not expert_outputs:
            raise ValueError("Fusion requires at least one ExpertOutput")

        router_input_like = self._make_router_input_like(
            expert_outputs,
            decision.transition_id,
        )
        validate_router_decision(router_input_like, decision)

        outputs = output_map(expert_outputs)

        active_names = tuple(decision.active_experts)

        if not active_names:
            raise ValueError(
                "Fusion cannot produce MarketRepresentation without "
                "at least one active expert"
            )

        pre_activation = [0.0] * self._output_size
        trace_experts: dict[str, _ExpertFusionTrace] = {}

        for expert_name in active_names:
            output = outputs[expert_name]

            if not output.available:
                raise ValueError(
                    f"Fusion received unavailable active expert "
                    f"{expert_name!r}"
                )

            weight = float(decision.weights.get(expert_name, 0.0))

            if weight <= 0.0:
                raise ValueError(
                    f"Active expert {expert_name!r} has non-positive "
                    "Router weight"
                )

            projection = self._ensure_projection(
                expert_name,
                input_size=len(output.representation),
            )

            projected = self._project(
                projection,
                output.representation,
            )

            for index, value in enumerate(projected):
                pre_activation[index] += weight * value

            trace_experts[expert_name] = _ExpertFusionTrace(
                representation=output.representation,
                projected=projected,
                weight=weight,
            )

        output_vector = tuple(tanh(value) for value in pre_activation)

        self._store_trace(
            _FusionTrace(
                transition_id=decision.transition_id,
                experts=trace_experts,
                pre_activation=tuple(pre_activation),
                output=output_vector,
            )
        )

        self._forward_calls += 1
        self._last_output_norm = self._norm(output_vector)

        diagnostics = {
            "active_experts": float(len(active_names)),
            "router_entropy": float(decision.entropy),
            "pre_activation_norm": self._norm(pre_activation),
            "output_norm": self._last_output_norm,
        }

        return FusionResult(
            transition_id=decision.transition_id,
            vector=output_vector,
            diagnostics=diagnostics,
        )

    def backward(
        self,
        transition_id: str,
        output_gradient: tuple[float, ...],
    ) -> FusionBackward | None:
        """
        Backpropagate a gradient from MarketRepresentation into:

        - each contributing expert representation;
        - each Router weight;
        - Fusion's own projection parameters.

        Fusion parameter gradients are retained until apply_update() so
        organ.py can first pass expert/router gradients to their owners.
        """

        trace = self._traces.get(transition_id)

        if trace is None:
            return None

        if transition_id in self._pending_updates:
            raise RuntimeError(
                f"Fusion backward already called for transition "
                f"{transition_id!r}; call apply_update() before repeating"
            )

        if len(output_gradient) != self._output_size:
            raise ValueError(
                "Fusion output gradient size mismatch: "
                f"expected {self._output_size}, "
                f"got {len(output_gradient)}"
            )

        gradient = tuple(float(value) for value in output_gradient)

        for value in gradient:
            if not isfinite(value):
                raise ValueError(
                    "Fusion output gradient contains non-finite value"
                )

        # y = tanh(z)
        #
        # dL/dz = dL/dy * (1 - y^2)
        pre_gradient = tuple(
            self._clip(
                downstream * (1.0 - output_value * output_value)
            )
            for downstream, output_value in zip(
                gradient,
                trace.output,
            )
        )

        expert_representation_gradients: dict[str, tuple[float, ...]] = {}
        router_weight_gradients: dict[str, float] = {}
        own_gradients: dict[str, _ExpertProjectionGradient] = {}

        for expert_name, expert_trace in trace.experts.items():
            projection = self._projections[expert_name]
            weight = expert_trace.weight

            # z = Σ_e w_e * projected_e
            #
            # dL/d(projected_e) = w_e * dL/dz
            projected_gradient = tuple(
                self._clip(weight * value)
                for value in pre_gradient
            )

            # dL/dw_e = dot(dL/dz, projected_e)
            router_weight_gradients[expert_name] = self._clip(
                self._dot(
                    pre_gradient,
                    expert_trace.projected,
                )
            )

            # projected_e = P_e @ representation_e + bias_e
            #
            # dL/drepresentation = P_e^T @ dL/dprojected
            representation_gradient = [0.0] * projection.input_size

            for output_index, row in enumerate(projection.weights):
                dprojected = projected_gradient[output_index]

                for input_index, parameter in enumerate(row):
                    representation_gradient[input_index] += (
                        parameter * dprojected
                    )

            expert_representation_gradients[expert_name] = tuple(
                self._clip(value)
                for value in representation_gradient
            )

            projection_weight_gradient = [
                [0.0] * projection.input_size
                for _ in range(self._output_size)
            ]
            projection_bias_gradient = [0.0] * self._output_size

            for output_index, dprojected in enumerate(projected_gradient):
                projection_bias_gradient[output_index] = dprojected

                row_gradient = projection_weight_gradient[output_index]

                for input_index, input_value in enumerate(
                    expert_trace.representation
                ):
                    row_gradient[input_index] = self._clip(
                        dprojected * input_value
                    )

            own_gradients[expert_name] = _ExpertProjectionGradient(
                weights=projection_weight_gradient,
                bias=projection_bias_gradient,
            )

        self._pending_updates[transition_id] = _PendingFusionUpdate(
            transition_id=transition_id,
            expert_gradients=own_gradients,
        )

        self._last_gradient_norm = self._norm(gradient)

        return FusionBackward(
            transition_id=transition_id,
            expert_gradients=expert_representation_gradients,
            weight_gradients=router_weight_gradients,
            diagnostics={
                "output_gradient_norm": self._last_gradient_norm,
                "pre_activation_gradient_norm": self._norm(
                    pre_gradient
                ),
                "expert_count": float(len(trace.experts)),
            },
        )

    def apply_update(
        self,
        transition_id: str,
    ) -> bool:
        """
        Apply Fusion's own parameter update for a previously-backpropagated
        transition and consume the causal trace.
        """

        pending = self._pending_updates.pop(transition_id, None)

        if pending is None:
            return False

        trace = self._traces.pop(transition_id, None)

        if trace is None:
            raise RuntimeError(
                "Fusion pending update exists without its causal trace"
            )

        for expert_name, gradient in pending.expert_gradients.items():
            projection = self._projections.get(expert_name)

            if projection is None:
                raise RuntimeError(
                    f"Fusion projection for {expert_name!r} disappeared "
                    "before update"
                )

            for output_index in range(self._output_size):
                projection.bias[output_index] -= (
                    self.learning_rate
                    * self._clip(gradient.bias[output_index])
                )

                row = projection.weights[output_index]
                row_gradient = gradient.weights[output_index]

                for input_index in range(projection.input_size):
                    row[input_index] -= (
                        self.learning_rate
                        * self._clip(row_gradient[input_index])
                    )

            self._expert_projection_norms[expert_name] = (
                self._matrix_norm(projection.weights)
            )

        self._learning_updates += 1
        return True

    def telemetry(self) -> FusionTelemetry:
        diagnostics: dict[str, float] = {
            "registered_experts": float(len(self._projections)),
            "pending_traces": float(len(self._traces)),
            "pending_updates": float(len(self._pending_updates)),
        }

        for expert_name, norm in self._expert_projection_norms.items():
            diagnostics[f"projection_norm:{expert_name}"] = norm

        return FusionTelemetry(
            forward_calls=self._forward_calls,
            learning_updates=self._learning_updates,
            last_output_norm=self._last_output_norm,
            last_gradient_norm=self._last_gradient_norm,
            diagnostics=diagnostics,
        )

    def snapshot(self) -> Mapping[str, object]:
        """
        Serialize learned Fusion parameters.

        Causal traces and pending gradients are transient and are not
        persisted.
        """

        return {
            "version": 1,
            "config": {
                "output_size": self._output_size,
                "learning_rate": self.learning_rate,
                "gradient_clip": self.gradient_clip,
                "max_traces": self.max_traces,
                "seed": self.seed,
            },
            "projections": {
                expert_name: {
                    "input_size": projection.input_size,
                    "weights": [
                        list(row)
                        for row in projection.weights
                    ],
                    "bias": list(projection.bias),
                }
                for expert_name, projection in self._projections.items()
            },
            "counters": {
                "forward_calls": self._forward_calls,
                "learning_updates": self._learning_updates,
                "last_output_norm": self._last_output_norm,
                "last_gradient_norm": self._last_gradient_norm,
                "expert_projection_norms": dict(
                    self._expert_projection_norms
                ),
            },
        }

    def restore(self, snapshot: Mapping[str, object]) -> None:
        version = int(snapshot.get("version", 0))

        if version != 1:
            raise ValueError(
                f"Unsupported AdaptiveMarketFusion snapshot version: "
                f"{version}"
            )

        config = snapshot.get("config")

        if not isinstance(config, Mapping):
            raise ValueError("Fusion snapshot has invalid config")

        output_size = int(config["output_size"])

        if output_size != self._output_size:
            raise ValueError(
                "Fusion snapshot output_size does not match current Fusion"
            )

        raw_projections = snapshot.get("projections")

        if not isinstance(raw_projections, Mapping):
            raise ValueError("Fusion snapshot has invalid projections")

        projections: dict[str, _ExpertProjection] = {}

        for raw_name, raw_projection in raw_projections.items():
            expert_name = str(raw_name)

            if not isinstance(raw_projection, Mapping):
                raise ValueError(
                    f"Invalid Fusion projection for {expert_name!r}"
                )

            input_size = int(raw_projection["input_size"])

            raw_weights = raw_projection["weights"]
            raw_bias = raw_projection["bias"]

            if not isinstance(raw_weights, list):
                raise ValueError(
                    f"Invalid Fusion weights for {expert_name!r}"
                )

            if not isinstance(raw_bias, list):
                raise ValueError(
                    f"Invalid Fusion bias for {expert_name!r}"
                )

            weights = [
                [float(value) for value in row]
                for row in raw_weights
            ]
            bias = [float(value) for value in raw_bias]

            if len(weights) != self._output_size:
                raise ValueError(
                    f"Fusion output size mismatch for {expert_name!r}"
                )

            if len(bias) != self._output_size:
                raise ValueError(
                    f"Fusion bias size mismatch for {expert_name!r}"
                )

            for row in weights:
                if len(row) != input_size:
                    raise ValueError(
                        f"Fusion input size mismatch for {expert_name!r}"
                    )

                for value in row:
                    if not isfinite(value):
                        raise ValueError(
                            f"Non-finite Fusion parameter for "
                            f"{expert_name!r}"
                        )

            for value in bias:
                if not isfinite(value):
                    raise ValueError(
                        f"Non-finite Fusion bias for {expert_name!r}"
                    )

            projections[expert_name] = _ExpertProjection(
                input_size=input_size,
                weights=weights,
                bias=bias,
            )

        counters = snapshot.get("counters", {})

        if not isinstance(counters, Mapping):
            raise ValueError("Fusion snapshot has invalid counters")

        raw_norms = counters.get("expert_projection_norms", {})

        if not isinstance(raw_norms, Mapping):
            raise ValueError(
                "Fusion snapshot has invalid projection norms"
            )

        self._projections = projections

        self._forward_calls = int(
            counters.get("forward_calls", 0)
        )
        self._learning_updates = int(
            counters.get("learning_updates", 0)
        )
        self._last_output_norm = float(
            counters.get("last_output_norm", 0.0)
        )
        self._last_gradient_norm = float(
            counters.get("last_gradient_norm", 0.0)
        )

        self._expert_projection_norms = {
            str(name): float(value)
            for name, value in raw_norms.items()
        }

        # Old causal traces must never survive process restore.
        self._traces.clear()
        self._pending_updates.clear()

    def reset(self) -> None:
        """
        Full Fusion Tabula Rasa.

        Projections are discarded and lazily reinitialized when experts
        appear again.
        """

        self._rng = Random(self.seed)

        self._projections.clear()
        self._traces.clear()
        self._pending_updates.clear()

        self._forward_calls = 0
        self._learning_updates = 0

        self._last_output_norm = 0.0
        self._last_gradient_norm = 0.0

        self._expert_projection_norms.clear()

    # ========================================================
    # Projection management
    # ========================================================

    def _ensure_projection(
        self,
        expert_name: str,
        *,
        input_size: int,
    ) -> _ExpertProjection:
        existing = self._projections.get(expert_name)

        if existing is not None:
            if existing.input_size != input_size:
                raise ValueError(
                    f"Fusion input size for expert {expert_name!r} changed "
                    f"from {existing.input_size} to {input_size}. "
                    "Reset/reconfigure Fusion or keep expert representation "
                    "size stable."
                )

            return existing

        if input_size <= 0:
            raise ValueError(
                f"Cannot create Fusion projection for empty expert "
                f"{expert_name!r}"
            )

        # Xavier-like initialization.
        scale = sqrt(
            2.0 / (input_size + self._output_size)
        )

        weights = [
            [
                self._rng.uniform(-scale, scale)
                for _ in range(input_size)
            ]
            for _ in range(self._output_size)
        ]

        projection = _ExpertProjection(
            input_size=input_size,
            weights=weights,
            bias=[0.0] * self._output_size,
        )

        self._projections[expert_name] = projection
        self._expert_projection_norms[expert_name] = (
            self._matrix_norm(weights)
        )

        return projection

    @staticmethod
    def _project(
        projection: _ExpertProjection,
        representation: tuple[float, ...],
    ) -> tuple[float, ...]:
        if len(representation) != projection.input_size:
            raise ValueError(
                "Expert representation size does not match Fusion "
                "projection"
            )

        result: list[float] = []

        for row, bias in zip(
            projection.weights,
            projection.bias,
        ):
            value = bias + sum(
                parameter * input_value
                for parameter, input_value in zip(
                    row,
                    representation,
                )
            )

            result.append(value)

        return tuple(result)

    # ========================================================
    # Trace management
    # ========================================================

    def _store_trace(
        self,
        trace: _FusionTrace,
    ) -> None:
        transition_id = trace.transition_id

        # A new forward for the same transition invalidates an older pending
        # update. Keeping both would make causal credit ambiguous.
        self._pending_updates.pop(transition_id, None)
        self._traces.pop(transition_id, None)

        self._traces[transition_id] = trace

        while len(self._traces) > self.max_traces:
            oldest_transition_id, _ = self._traces.popitem(
                last=False
            )
            self._pending_updates.pop(
                oldest_transition_id,
                None,
            )

    # ========================================================
    # Math helpers
    # ========================================================

    def _clip(
        self,
        value: float,
    ) -> float:
        if value > self.gradient_clip:
            return self.gradient_clip

        if value < -self.gradient_clip:
            return -self.gradient_clip

        return value

    @staticmethod
    def _dot(
        left: tuple[float, ...] | list[float],
        right: tuple[float, ...] | list[float],
    ) -> float:
        if len(left) != len(right):
            raise ValueError("dot product size mismatch")

        return sum(
            a * b
            for a, b in zip(left, right)
        )

    @staticmethod
    def _norm(
        values: tuple[float, ...] | list[float],
    ) -> float:
        return sqrt(
            sum(float(value) * float(value) for value in values)
        )

    @staticmethod
    def _matrix_norm(
        matrix: list[list[float]],
    ) -> float:
        return sqrt(
            sum(
                value * value
                for row in matrix
                for value in row
            )
        )

    # ========================================================
    # Small validation adapter
    # ========================================================

    @staticmethod
    def _make_router_input_like(
        expert_outputs: tuple[ExpertOutput, ...],
        transition_id: str,
    ):
        """
        Avoid duplicating Router validation logic.

        Import is local to keep the top of the file focused on the Fusion
        contracts.
        """

        from worlds.market.perception.interface import RouterInput

        return RouterInput(
            transition_id=transition_id,
            expert_outputs=expert_outputs,
        )
