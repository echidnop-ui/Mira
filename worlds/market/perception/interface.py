from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, Protocol, runtime_checkable

from worlds.market.interface import MarketInput


# ============================================================
# Expert output
# ============================================================


@dataclass(frozen=True, slots=True)
class ExpertOutput:
    """
    Public output of one Market perception expert.

    The expert may be:

        TCN
        SSM
        Attention
        recurrent/xLSTM
        DeepLOB
        regime/volatility specialist
        another future expert

    Experts do NOT return BUY/SELL/HOLD.

    They return a learned representation that may later be routed and fused.

    If `available=False`, `representation` should normally be empty.
    Absence is explicit so the Router can mask the expert rather than
    interpreting a zero vector as meaningful information.
    """

    expert_name: str
    transition_id: str

    representation: tuple[float, ...]

    available: bool = True
    uncertainty: float | None = None

    auxiliary: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.expert_name:
            raise ValueError("ExpertOutput.expert_name must not be empty")

        if not self.transition_id:
            raise ValueError("ExpertOutput.transition_id must not be empty")

        if self.available and not self.representation:
            raise ValueError(
                f"Available expert {self.expert_name!r} returned "
                "an empty representation"
            )

        if not self.available and self.representation:
            raise ValueError(
                f"Unavailable expert {self.expert_name!r} must not expose "
                "a meaningful representation"
            )

        for value in self.representation:
            if not isfinite(value):
                raise ValueError(
                    f"Expert {self.expert_name!r} produced a non-finite value"
                )

        if self.uncertainty is not None and not isfinite(self.uncertainty):
            raise ValueError(
                f"Expert {self.expert_name!r} produced non-finite uncertainty"
            )


# ============================================================
# Expert telemetry
# ============================================================


@dataclass(frozen=True, slots=True)
class ExpertTelemetry:
    """
    Small diagnostics surface for one expert.

    Telemetry must not influence cognition unless a value is explicitly
    routed back through a public learning/input contract.
    """

    expert_name: str

    available: bool = False

    forward_calls: int = 0
    learning_updates: int = 0

    last_representation_norm: float = 0.0
    last_gradient_norm: float = 0.0

    uncertainty: float | None = None

    diagnostics: Mapping[str, float] = field(default_factory=dict)


# ============================================================
# Market expert protocol
# ============================================================


@runtime_checkable
class MarketExpert(Protocol):
    """
    Public local contract implemented by every Market neural expert.

    An expert owns its internal architecture and trainable state.

    Shared code must not depend on whether the implementation is:
        convolutional,
        recurrent,
        SSM,
        attention-based,
        order-book specific,
        or something introduced later.
    """

    @property
    def name(self) -> str:
        ...

    @property
    def representation_size(self) -> int:
        ...

    def encode(
        self,
        market_input: MarketInput,
        *,
        transition_id: str,
    ) -> ExpertOutput:
        """
        Produce a representation for one causal transition.

        If required input is unavailable, return:

            ExpertOutput(
                ...,
                representation=(),
                available=False,
            )

        Do not fabricate a meaningful zero representation for missing data.
        """
        ...

    def learn(
        self,
        transition_id: str,
        representation_gradient: tuple[float, ...],
    ) -> bool:
        """
        Apply causal learning to the exact expert trace that contributed to
        `transition_id`.

        Return True only when a matching trainable trace existed.

        The expert must not train the current state using an unrelated old
        consequence.
        """
        ...

    def telemetry(self) -> ExpertTelemetry:
        ...

    def snapshot(self) -> Mapping[str, object]:
        """
        Return serializable expert state.
        """
        ...

    def restore(self, snapshot: Mapping[str, object]) -> None:
        ...

    def reset(self) -> None:
        """
        Reset learned expert state for experimental Tabula Rasa.
        """
        ...


# ============================================================
# Router input
# ============================================================


@dataclass(frozen=True, slots=True)
class RouterInput:
    """
    Everything the Router is allowed to use for one routing decision.

    `expert_outputs` is the primary input.

    `shared_context` may later contain a public representation of the shared
    Heiwa state. It must never be a reference to Core internals.

    `market_context` may contain compact normalized Market context useful for
    routing without forcing the Router to parse full raw MarketInput again.

    Both optional contexts are plain vectors and are intentionally opaque to
    the Router interface.
    """

    transition_id: str

    expert_outputs: tuple[ExpertOutput, ...]

    shared_context: tuple[float, ...] | None = None
    market_context: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise ValueError("RouterInput.transition_id must not be empty")

        names: set[str] = set()

        for output in self.expert_outputs:
            if output.transition_id != self.transition_id:
                raise ValueError(
                    "RouterInput contains ExpertOutput from another transition"
                )

            if output.expert_name in names:
                raise ValueError(
                    f"Duplicate expert output: {output.expert_name!r}"
                )

            names.add(output.expert_name)


# ============================================================
# Router decision
# ============================================================


@dataclass(frozen=True, slots=True)
class RouterDecision:
    """
    Trainable Router result.

    `weights` contains normalized contribution weights for available experts.

    Unavailable experts should either:
        - be absent from `weights`, or
        - have weight 0.0.

    `active_experts` is the actual routed subset.

    Sparse routing may later use only top-k experts.
    Dense routing may include every available expert.
    """

    transition_id: str

    weights: Mapping[str, float]
    active_experts: tuple[str, ...]

    entropy: float = 0.0

    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise ValueError("RouterDecision.transition_id must not be empty")

        active = set(self.active_experts)

        if len(active) != len(self.active_experts):
            raise ValueError("RouterDecision.active_experts contains duplicates")

        total = 0.0

        for name, weight in self.weights.items():
            if not isfinite(weight):
                raise ValueError(
                    f"Router weight for {name!r} is non-finite"
                )

            if weight < 0.0:
                raise ValueError(
                    f"Router weight for {name!r} must be non-negative"
                )

            if weight > 0.0 and name not in active:
                raise ValueError(
                    f"Expert {name!r} has positive router weight but is not "
                    "listed in active_experts"
                )

            total += weight

        if active and total <= 0.0:
            raise ValueError(
                "RouterDecision has active experts but zero total weight"
            )


# ============================================================
# Router telemetry
# ============================================================


@dataclass(frozen=True, slots=True)
class RouterTelemetry:
    """
    Diagnostics for Router behavior and collapse detection.
    """

    forward_calls: int = 0
    learning_updates: int = 0

    last_entropy: float = 0.0
    last_gradient_norm: float = 0.0

    average_weights: Mapping[str, float] = field(default_factory=dict)
    selection_counts: Mapping[str, int] = field(default_factory=dict)

    diagnostics: Mapping[str, float] = field(default_factory=dict)


# ============================================================
# Router protocol
# ============================================================


@runtime_checkable
class MarketRouter(Protocol):
    """
    Router chooses how available expert representations contribute.

    It does not execute trades and does not choose a final market action.
    """

    def route(self, router_input: RouterInput) -> RouterDecision:
        ...

    def learn(
        self,
        transition_id: str,
        weight_gradients: Mapping[str, float],
    ) -> bool:
        """
        Learn from gradient with respect to Router expert weights.

        The Router implementation owns its exact parameterization.

        Return True only if a matching causal routing trace existed.
        """
        ...

    def telemetry(self) -> RouterTelemetry:
        ...

    def snapshot(self) -> Mapping[str, object]:
        ...

    def restore(self, snapshot: Mapping[str, object]) -> None:
        ...

    def reset(self) -> None:
        ...


# ============================================================
# Fusion result
# ============================================================


@dataclass(frozen=True, slots=True)
class FusionResult:
    """
    Result of combining routed expert representations.

    This is still perception-local.

    `MarketNeuralOrgan` later packages this vector into the public
    `MarketRepresentation` from worlds/market/interface.py.
    """

    transition_id: str
    vector: tuple[float, ...]

    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise ValueError("FusionResult.transition_id must not be empty")

        if not self.vector:
            raise ValueError("FusionResult.vector must not be empty")

        for value in self.vector:
            if not isfinite(value):
                raise ValueError("FusionResult contains a non-finite value")


# ============================================================
# Fusion backward result
# ============================================================


@dataclass(frozen=True, slots=True)
class FusionBackward:
    """
    Gradient decomposition produced by Fusion.

    Fusion receives gradient of its output representation and returns:

    - gradient for each expert representation that actually contributed;
    - gradient for Router weights.

    This keeps causal credit explicit and allows `organ.py` to route learning
    without knowing Fusion internals.
    """

    transition_id: str

    expert_gradients: Mapping[str, tuple[float, ...]]
    weight_gradients: Mapping[str, float]

    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise ValueError("FusionBackward.transition_id must not be empty")

        for expert_name, gradient in self.expert_gradients.items():
            if not expert_name:
                raise ValueError("Expert gradient has an empty expert name")

            for value in gradient:
                if not isfinite(value):
                    raise ValueError(
                        f"Non-finite expert gradient for {expert_name!r}"
                    )

        for expert_name, value in self.weight_gradients.items():
            if not isfinite(value):
                raise ValueError(
                    f"Non-finite Router weight gradient for {expert_name!r}"
                )


# ============================================================
# Fusion telemetry
# ============================================================


@dataclass(frozen=True, slots=True)
class FusionTelemetry:
    """
    Compact diagnostics for the learned fusion mechanism.
    """

    forward_calls: int = 0
    learning_updates: int = 0

    last_output_norm: float = 0.0
    last_gradient_norm: float = 0.0

    diagnostics: Mapping[str, float] = field(default_factory=dict)


# ============================================================
# Fusion protocol
# ============================================================


@runtime_checkable
class MarketFusion(Protocol):
    """
    Learned mechanism that combines Router-selected expert representations.

    Router decides contribution weights.

    Fusion decides how routed information becomes one fixed-size vector.
    """

    @property
    def output_size(self) -> int:
        ...

    def fuse(
        self,
        expert_outputs: tuple[ExpertOutput, ...],
        decision: RouterDecision,
    ) -> FusionResult:
        """
        Fuse expert outputs for the exact routing decision.

        Implementations should retain only the causal trace needed for later
        learning.
        """
        ...

    def backward(
        self,
        transition_id: str,
        output_gradient: tuple[float, ...],
    ) -> FusionBackward | None:
        """
        Backpropagate a downstream representation gradient through Fusion.

        Return None when there is no matching causal trace.
        """
        ...

    def apply_update(
        self,
        transition_id: str,
    ) -> bool:
        """
        Apply Fusion's own parameter update for an already-backpropagated
        causal trace.

        This separation allows `organ.py` to obtain expert/router gradients
        before the Fusion trace is discarded.

        Implementations that update during `backward` may return True here
        as a no-op, but must document that behavior locally.
        """
        ...

    def telemetry(self) -> FusionTelemetry:
        ...

    def snapshot(self) -> Mapping[str, object]:
        ...

    def restore(self, snapshot: Mapping[str, object]) -> None:
        ...

    def reset(self) -> None:
        ...


# ============================================================
# Helper functions
# ============================================================


def available_expert_outputs(
    outputs: tuple[ExpertOutput, ...],
) -> tuple[ExpertOutput, ...]:
    """
    Return only outputs that represent actual available expert information.

    This is a mechanical mask, not a routing decision.
    """

    return tuple(output for output in outputs if output.available)


def output_map(
    outputs: tuple[ExpertOutput, ...],
) -> dict[str, ExpertOutput]:
    """
    Build a unique expert-name map.

    Raises on duplicate names because ambiguous expert identity would break
    causal routing and learning.
    """

    result: dict[str, ExpertOutput] = {}

    for output in outputs:
        if output.expert_name in result:
            raise ValueError(
                f"Duplicate ExpertOutput for {output.expert_name!r}"
            )

        result[output.expert_name] = output

    return result


def validate_router_decision(
    router_input: RouterInput,
    decision: RouterDecision,
) -> None:
    """
    Validate only the perception-local routing contract.

    This does not decide whether the Router made a good decision.
    """

    if decision.transition_id != router_input.transition_id:
        raise ValueError(
            "RouterDecision belongs to another transition"
        )

    outputs = output_map(router_input.expert_outputs)

    active = set(decision.active_experts)

    for expert_name in active:
        output = outputs.get(expert_name)

        if output is None:
            raise ValueError(
                f"Router activated unknown expert {expert_name!r}"
            )

        if not output.available:
            raise ValueError(
                f"Router activated unavailable expert {expert_name!r}"
            )

    for expert_name, weight in decision.weights.items():
        if weight <= 0.0:
            continue

        output = outputs.get(expert_name)

        if output is None:
            raise ValueError(
                f"Router weighted unknown expert {expert_name!r}"
            )

        if not output.available:
            raise ValueError(
                f"Router gave positive weight to unavailable expert "
                f"{expert_name!r}"
            )
