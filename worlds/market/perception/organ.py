from __future__ import annotations

from math import isfinite, sqrt
from typing import Iterable, Mapping

from worlds.market.interface import (
    MarketInput,
    MarketNeuralOrgan,
    MarketOrganTelemetry,
    MarketRepresentation,
)
from worlds.market.perception.interface import (
    ExpertOutput,
    MarketExpert,
    MarketFusion,
    MarketRouter,
    RouterInput,
    validate_router_decision,
)


class MarketPerceptionUnavailable(RuntimeError):
    """
    Raised when no Market expert can produce usable perception.

    This is different from a valid low-confidence representation.

    The Runtime/World integration may later use this condition to avoid
    exposing actions that require perception rather than fabricating a zero
    market state.
    """


class HeterogeneousMarketOrgan(MarketNeuralOrgan):
    """
    Coordinates the complete heterogeneous Market Neural Organ.

    Forward path:

        MarketInput
            ↓
        independent MarketExpert implementations
            ↓
        ExpertOutput[]
            ↓
        MarketRouter
            ↓
        RouterDecision
            ↓
        MarketFusion
            ↓
        MarketRepresentation

    Backward path:

        dLoss/dMarketRepresentation
            ↓
        Fusion.backward(transition_id)
            ├── dLoss/dExpertRepresentation
            └── dLoss/dRouterWeight
                    ↓
              Router.learn(...)
                    ↓
              Expert.learn(...)
                    ↓
              Fusion.apply_update(...)

    The organ does not choose BUY/SELL/HOLD and does not execute orders.

    It is a market-specific perception organ of the shared Heiwa.
    """

    def __init__(
        self,
        *,
        experts: Iterable[MarketExpert],
        router: MarketRouter,
        fusion: MarketFusion,
    ) -> None:
        expert_map: dict[str, MarketExpert] = {}

        for expert in experts:
            name = str(expert.name)

            if not name:
                raise ValueError("Market expert name must not be empty")

            if name in expert_map:
                raise ValueError(
                    f"Duplicate Market expert name: {name!r}"
                )

            if expert.representation_size <= 0:
                raise ValueError(
                    f"Market expert {name!r} has invalid "
                    "representation_size"
                )

            expert_map[name] = expert

        if not expert_map:
            raise ValueError(
                "HeterogeneousMarketOrgan requires at least one expert"
            )

        if fusion.output_size <= 0:
            raise ValueError(
                "Market Fusion output_size must be > 0"
            )

        self._experts = expert_map
        self._router = router
        self._fusion = fusion

        self._learning_updates = 0
        self._last_gradient_norm = 0.0
        self._last_representation_norm = 0.0

        self._last_expert_weights: dict[str, float] = {}
        self._last_expert_norms: dict[str, float] = {}
        self._last_expert_uncertainty: dict[str, float] = {}
        self._last_expert_available: dict[str, bool] = {}

        self._missing_expert_traces = 0
        self._missing_router_traces = 0
        self._missing_fusion_updates = 0

    # ========================================================
    # Public API
    # ========================================================

    @property
    def representation_size(self) -> int:
        return self._fusion.output_size

    @property
    def expert_names(self) -> tuple[str, ...]:
        """
        Stable names of experts currently owned by the organ.

        This is primarily useful for configuration and telemetry.
        Shared cognition should not depend on particular expert names.
        """

        return tuple(self._experts)

    def encode(
        self,
        market_input: MarketInput,
        *,
        transition_id: str,
        shared_context: tuple[float, ...] | None = None,
    ) -> MarketRepresentation:
        if not transition_id:
            raise ValueError("transition_id must not be empty")

        validated_shared_context = self._validate_vector(
            shared_context,
            name="shared_context",
            allow_none=True,
        )

        outputs = self._encode_experts(
            market_input,
            transition_id=transition_id,
        )

        available_outputs = tuple(
            output
            for output in outputs
            if output.available
        )

        self._capture_expert_telemetry(outputs)

        if not available_outputs:
            self._last_expert_weights = {}

            raise MarketPerceptionUnavailable(
                "No Market expert produced an available representation "
                f"for transition {transition_id!r}"
            )

        market_context = self._build_router_market_context(
            market_input
        )

        router_input = RouterInput(
            transition_id=transition_id,
            expert_outputs=outputs,
            shared_context=validated_shared_context,
            market_context=market_context,
        )

        decision = self._router.route(router_input)

        validate_router_decision(
            router_input,
            decision,
        )

        if not decision.active_experts:
            self._last_expert_weights = dict(decision.weights)

            raise MarketPerceptionUnavailable(
                "Market Router selected no active expert for "
                f"transition {transition_id!r}"
            )

        fusion_result = self._fusion.fuse(
            outputs,
            decision,
        )

        if fusion_result.transition_id != transition_id:
            raise RuntimeError(
                "Fusion returned a result from another transition"
            )

        if len(fusion_result.vector) != self.representation_size:
            raise RuntimeError(
                "Fusion returned unexpected representation size: "
                f"expected {self.representation_size}, "
                f"got {len(fusion_result.vector)}"
            )

        self._last_expert_weights = dict(decision.weights)
        self._last_representation_norm = self._norm(
            fusion_result.vector
        )

        diagnostics = dict(fusion_result.diagnostics)
        diagnostics["router_entropy"] = float(
            decision.entropy
        )
        diagnostics["active_experts"] = float(
            len(decision.active_experts)
        )
        diagnostics["available_experts"] = float(
            len(available_outputs)
        )

        return MarketRepresentation(
            transition_id=transition_id,
            instrument_id=market_input.instrument_id,
            as_of=market_input.as_of,
            vector=fusion_result.vector,
            uncertainty=None,
            expert_weights=dict(decision.weights),
            expert_available={
                output.expert_name: output.available
                for output in outputs
            },
            diagnostics=diagnostics,
        )

    def learn(
        self,
        transition_id: str,
        representation_gradient: tuple[float, ...],
    ) -> bool:
        """
        Apply a downstream gradient to the exact causal Market transition.

        Update order matters:

        1. Fusion decomposes the gradient using its forward-time parameters.
        2. Router receives only Router-weight gradients.
        3. Each contributing expert receives only its own representation
           gradient.
        4. Fusion applies its own parameter update last.

        No current-state fallback is allowed when a trace is missing.
        """

        if not transition_id:
            raise ValueError("transition_id must not be empty")

        gradient = self._validate_vector(
            representation_gradient,
            name="representation_gradient",
            allow_none=False,
        )

        assert gradient is not None

        if len(gradient) != self.representation_size:
            raise ValueError(
                "Market representation gradient size mismatch: "
                f"expected {self.representation_size}, "
                f"got {len(gradient)}"
            )

        backward = self._fusion.backward(
            transition_id,
            gradient,
        )

        if backward is None:
            return False

        if backward.transition_id != transition_id:
            raise RuntimeError(
                "Fusion backward returned another transition"
            )

        router_updated = self._router.learn(
            transition_id,
            backward.weight_gradients,
        )

        if not router_updated:
            self._missing_router_traces += 1

        expert_updates = 0

        for expert_name, expert_gradient in (
            backward.expert_gradients.items()
        ):
            expert = self._experts.get(expert_name)

            if expert is None:
                raise RuntimeError(
                    "Fusion produced gradient for unknown expert "
                    f"{expert_name!r}"
                )

            if expert.learn(
                transition_id,
                expert_gradient,
            ):
                expert_updates += 1
            else:
                self._missing_expert_traces += 1

        fusion_updated = self._fusion.apply_update(
            transition_id
        )

        if not fusion_updated:
            self._missing_fusion_updates += 1
            return False

        self._learning_updates += 1
        self._last_gradient_norm = self._norm(gradient)

        return True

    def telemetry(self) -> MarketOrganTelemetry:
        """
        Return compact organ-level telemetry.

        Detailed Router/Fusion telemetry remains available from those
        components locally and is not duplicated into the public Market
        contract.
        """

        return MarketOrganTelemetry(
            learning_updates=self._learning_updates,
            last_gradient_norm=self._last_gradient_norm,
            representation_norm=self._last_representation_norm,
            expert_weights=dict(self._last_expert_weights),
            expert_norms=dict(self._last_expert_norms),
            expert_uncertainty=dict(
                self._last_expert_uncertainty
            ),
            expert_available=dict(
                self._last_expert_available
            ),
        )

    def snapshot(self) -> Mapping[str, object]:
        """
        Serialize learned Market perception state.

        Causal traces are intentionally owned by components and should not
        survive restore.
        """

        return {
            "version": 1,
            "representation_size": self.representation_size,
            "experts": {
                name: {
                    "representation_size": expert.representation_size,
                    "state": dict(expert.snapshot()),
                }
                for name, expert in self._experts.items()
            },
            "router": dict(self._router.snapshot()),
            "fusion": dict(self._fusion.snapshot()),
            "counters": {
                "learning_updates": self._learning_updates,
                "last_gradient_norm": self._last_gradient_norm,
                "last_representation_norm": (
                    self._last_representation_norm
                ),
                "missing_expert_traces": (
                    self._missing_expert_traces
                ),
                "missing_router_traces": (
                    self._missing_router_traces
                ),
                "missing_fusion_updates": (
                    self._missing_fusion_updates
                ),
            },
        }

    def restore(
        self,
        snapshot: Mapping[str, object],
    ) -> None:
        version = int(snapshot.get("version", 0))

        if version != 1:
            raise ValueError(
                "Unsupported HeterogeneousMarketOrgan snapshot "
                f"version: {version}"
            )

        snapshot_representation_size = int(
            snapshot.get("representation_size", -1)
        )

        if (
            snapshot_representation_size
            != self.representation_size
        ):
            raise ValueError(
                "Market organ snapshot representation_size does not "
                "match current Fusion"
            )

        raw_experts = snapshot.get("experts", {})

        if not isinstance(raw_experts, Mapping):
            raise ValueError(
                "Market organ snapshot has invalid experts"
            )

        # Restore only currently installed experts.
        #
        # This allows a newer topology to introduce an additional expert
        # without making an older snapshot unusable. Unknown old experts
        # are ignored because no implementation exists to receive them.
        for name, expert in self._experts.items():
            raw_entry = raw_experts.get(name)

            if raw_entry is None:
                continue

            if not isinstance(raw_entry, Mapping):
                raise ValueError(
                    f"Invalid snapshot entry for expert {name!r}"
                )

            saved_size = int(
                raw_entry.get("representation_size", -1)
            )

            if saved_size != expert.representation_size:
                raise ValueError(
                    f"Snapshot representation_size mismatch for "
                    f"expert {name!r}"
                )

            raw_state = raw_entry.get("state")

            if not isinstance(raw_state, Mapping):
                raise ValueError(
                    f"Invalid saved state for expert {name!r}"
                )

            expert.restore(raw_state)

        raw_router = snapshot.get("router")
        raw_fusion = snapshot.get("fusion")

        if not isinstance(raw_router, Mapping):
            raise ValueError(
                "Market organ snapshot has invalid Router state"
            )

        if not isinstance(raw_fusion, Mapping):
            raise ValueError(
                "Market organ snapshot has invalid Fusion state"
            )

        self._router.restore(raw_router)
        self._fusion.restore(raw_fusion)

        counters = snapshot.get("counters", {})

        if not isinstance(counters, Mapping):
            raise ValueError(
                "Market organ snapshot has invalid counters"
            )

        self._learning_updates = int(
            counters.get("learning_updates", 0)
        )
        self._last_gradient_norm = float(
            counters.get("last_gradient_norm", 0.0)
        )
        self._last_representation_norm = float(
            counters.get("last_representation_norm", 0.0)
        )
        self._missing_expert_traces = int(
            counters.get("missing_expert_traces", 0)
        )
        self._missing_router_traces = int(
            counters.get("missing_router_traces", 0)
        )
        self._missing_fusion_updates = int(
            counters.get("missing_fusion_updates", 0)
        )

        self._clear_live_telemetry()

    def reset(self) -> None:
        """
        Full Market Neural Organ Tabula Rasa.

        External Market/portfolio state is untouched.
        """

        for expert in self._experts.values():
            expert.reset()

        self._router.reset()
        self._fusion.reset()

        self._learning_updates = 0
        self._last_gradient_norm = 0.0
        self._last_representation_norm = 0.0

        self._missing_expert_traces = 0
        self._missing_router_traces = 0
        self._missing_fusion_updates = 0

        self._clear_live_telemetry()

    # ========================================================
    # Expert forward path
    # ========================================================

    def _encode_experts(
        self,
        market_input: MarketInput,
        *,
        transition_id: str,
    ) -> tuple[ExpertOutput, ...]:
        outputs: list[ExpertOutput] = []

        for name, expert in self._experts.items():
            output = expert.encode(
                market_input,
                transition_id=transition_id,
            )

            if output.expert_name != name:
                raise RuntimeError(
                    f"Expert registered as {name!r} returned name "
                    f"{output.expert_name!r}"
                )

            if output.transition_id != transition_id:
                raise RuntimeError(
                    f"Expert {name!r} returned output from another "
                    "transition"
                )

            if (
                output.available
                and len(output.representation)
                != expert.representation_size
            ):
                raise RuntimeError(
                    f"Expert {name!r} returned representation size "
                    f"{len(output.representation)}, expected "
                    f"{expert.representation_size}"
                )

            outputs.append(output)

        return tuple(outputs)

    # ========================================================
    # Router context
    # ========================================================

    @staticmethod
    def _build_router_market_context(
        market_input: MarketInput,
    ) -> tuple[float, ...]:
        """
        Build a compact normalized context already present in MarketInput.

        This is mechanical data plumbing, not a cognitive rule.

        Raw sequence/order-book data stays inside experts. The Router sees
        the compact portfolio/context vectors plus expert representations.
        """

        return (
            *market_input.portfolio_features,
            *market_input.context_features,
        )

    # ========================================================
    # Telemetry helpers
    # ========================================================

    def _capture_expert_telemetry(
        self,
        outputs: tuple[ExpertOutput, ...],
    ) -> None:
        self._last_expert_norms = {}
        self._last_expert_uncertainty = {}
        self._last_expert_available = {}

        for output in outputs:
            name = output.expert_name

            self._last_expert_available[name] = (
                output.available
            )

            if output.available:
                self._last_expert_norms[name] = self._norm(
                    output.representation
                )
            else:
                self._last_expert_norms[name] = 0.0

            if output.uncertainty is not None:
                self._last_expert_uncertainty[name] = float(
                    output.uncertainty
                )

    def _clear_live_telemetry(self) -> None:
        self._last_expert_weights = {}
        self._last_expert_norms = {}
        self._last_expert_uncertainty = {}
        self._last_expert_available = {}

    # ========================================================
    # Validation / math
    # ========================================================

    @staticmethod
    def _validate_vector(
        values: tuple[float, ...] | None,
        *,
        name: str,
        allow_none: bool,
    ) -> tuple[float, ...] | None:
        if values is None:
            if allow_none:
                return None

            raise ValueError(f"{name} must not be None")

        result = tuple(float(value) for value in values)

        for value in result:
            if not isfinite(value):
                raise ValueError(
                    f"{name} contains non-finite value"
                )

        return result

    @staticmethod
    def _norm(
        values: tuple[float, ...] | list[float],
    ) -> float:
        return sqrt(
            sum(
                float(value) * float(value)
                for value in values
            )
        )
