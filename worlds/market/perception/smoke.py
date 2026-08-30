from __future__ import annotations

from math import cos, sin, sqrt

from worlds.market.interface import (
    MarketInput,
    OrderBookLevel,
    OrderBookSnapshot,
    TemporalFeatureWindow,
)
from worlds.market.perception.defaults import (
    MarketPerceptionConfig,
    build_default_market_organ,
)


def _temporal_window(
    name: str,
    *,
    steps: int,
    start_time: float,
    cadence_seconds: float,
    phase: float,
) -> TemporalFeatureWindow:
    """
    Build deterministic synthetic temporal data.

    This is not a price simulator and not training data. Its only purpose is
    to exercise the Market perception interfaces with non-constant,
    structurally valid input.
    """

    feature_names = (
        "return",
        "volume",
        "spread",
        "range",
        "momentum",
        "activity",
    )

    timestamps: list[float] = []
    rows: list[tuple[float, ...]] = []

    for index in range(steps):
        t = float(index)

        wave_fast = sin(
            phase + t * 0.37
        )
        wave_slow = cos(
            phase * 0.7 + t * 0.11
        )

        synthetic_return = (
            0.0025 * wave_fast
            + 0.0012 * wave_slow
            + 0.00005 * t
        )

        synthetic_volume = (
            1.0
            + 0.30 * sin(
                phase + t * 0.19
            )
            + 0.02 * t
        )

        synthetic_spread = (
            0.0008
            + 0.0002
            * (
                1.0
                + cos(
                    phase + t * 0.23
                )
            )
        )

        synthetic_range = (
            0.003
            + 0.0015
            * (
                1.0
                + sin(
                    phase * 0.5
                    + t * 0.17
                )
            )
        )

        synthetic_momentum = (
            0.6 * wave_slow
            + 0.4
            * sin(
                phase + t * 0.07
            )
        )

        synthetic_activity = (
            0.5
            + 0.25
            * cos(
                phase * 1.3
                + t * 0.29
            )
            + 0.005 * t
        )

        timestamps.append(
            start_time
            + index
            * cadence_seconds
        )

        rows.append(
            (
                synthetic_return,
                synthetic_volume,
                synthetic_spread,
                synthetic_range,
                synthetic_momentum,
                synthetic_activity,
            )
        )

    return TemporalFeatureWindow(
        name=name,
        feature_names=feature_names,
        timestamps=tuple(
            timestamps
        ),
        rows=tuple(
            rows
        ),
    )


def _order_book_history(
    *,
    snapshots: int,
    levels: int,
    start_time: float,
    cadence_seconds: float,
) -> tuple[OrderBookSnapshot, ...]:
    """
    Build deterministic best-first L2 snapshots.

    Bids decrease with depth; asks increase with depth.
    """

    history: list[
        OrderBookSnapshot
    ] = []

    for time_index in range(
        snapshots
    ):
        t = float(
            time_index
        )

        mid = (
            100.0
            + 0.04 * t
            + 0.25
            * sin(
                t * 0.21
            )
        )

        half_spread = (
            0.01
            + 0.003
            * (
                1.0
                + cos(
                    t * 0.17
                )
            )
        )

        bids: list[
            OrderBookLevel
        ] = []

        asks: list[
            OrderBookLevel
        ] = []

        for level_index in range(
            levels
        ):
            depth = float(
                level_index + 1
            )

            price_step = (
                0.01 * depth
            )

            bid_price = (
                mid
                - half_spread
                - price_step
            )

            ask_price = (
                mid
                + half_spread
                + price_step
            )

            bid_quantity = (
                40.0
                + 3.0 * depth
                + 8.0
                * (
                    1.0
                    + sin(
                        t * 0.31
                        + depth * 0.43
                    )
                )
            )

            ask_quantity = (
                38.0
                + 3.5 * depth
                + 7.0
                * (
                    1.0
                    + cos(
                        t * 0.27
                        + depth * 0.37
                    )
                )
            )

            bids.append(
                OrderBookLevel(
                    price=bid_price,
                    quantity=bid_quantity,
                )
            )

            asks.append(
                OrderBookLevel(
                    price=ask_price,
                    quantity=ask_quantity,
                )
            )

        history.append(
            OrderBookSnapshot(
                observed_at=(
                    start_time
                    + time_index
                    * cadence_seconds
                ),
                bids=tuple(
                    bids
                ),
                asks=tuple(
                    asks
                ),
            )
        )

    return tuple(
        history
    )


def _market_input(
    *,
    include_l2: bool,
) -> MarketInput:
    """
    Construct one structurally valid synthetic MarketInput.
    """

    as_of = 2_000_000_000.0

    return MarketInput(
        instrument_id="SYNTHETIC:TEST",
        as_of=as_of,
        micro=_temporal_window(
            "micro",
            steps=24,
            start_time=(
                as_of - 24.0
            ),
            cadence_seconds=1.0,
            phase=0.2,
        ),
        short=_temporal_window(
            "short",
            steps=28,
            start_time=(
                as_of - 28.0 * 15.0
            ),
            cadence_seconds=15.0,
            phase=0.7,
        ),
        medium=_temporal_window(
            "medium",
            steps=32,
            start_time=(
                as_of - 32.0 * 60.0
            ),
            cadence_seconds=60.0,
            phase=1.1,
        ),
        long=_temporal_window(
            "long",
            steps=36,
            start_time=(
                as_of - 36.0 * 300.0
            ),
            cadence_seconds=300.0,
            phase=1.6,
        ),
        order_book_history=(
            _order_book_history(
                snapshots=20,
                levels=10,
                start_time=(
                    as_of - 20.0
                ),
                cadence_seconds=1.0,
            )
            if include_l2
            else ()
        ),
        portfolio_feature_names=(
            "cash_fraction",
            "position_fraction",
            "unrealized_return",
        ),
        portfolio_features=(
            0.80,
            0.20,
            0.012,
        ),
        context_feature_names=(
            "session_progress",
            "recent_volatility",
            "liquidity_state",
        ),
        context_features=(
            0.55,
            0.30,
            0.75,
        ),
        metadata={
            "synthetic": True,
            "purpose": "market perception smoke",
        },
    )


def _smoke_config(
) -> MarketPerceptionConfig:
    """
    Small dimensions keep the diagnostic quick while preserving the same
    heterogeneous architecture and code paths as the default organ.
    """

    return MarketPerceptionConfig(
        expert_representation_size=24,
        market_representation_size=32,
        training_horizon_minutes=240,
        expert_max_traces=16,
        router_max_traces=32,
        fusion_max_traces=32,
        expert_learning_rate=1e-3,
        router_learning_rate=1e-3,
        fusion_learning_rate=1e-3,
        gradient_clip=5.0,

        tcn_channels=16,
        tcn_branch_size=16,
        tcn_depth=2,
        tcn_kernel_size=3,
        tcn_dropout=0.0,
        tcn_min_steps=8,

        ssm_hidden_size=16,
        ssm_branch_size=16,
        ssm_depth=2,
        ssm_dropout=0.0,
        ssm_min_steps=8,

        attention_hidden_size=16,
        attention_branch_size=16,
        attention_depth=2,
        attention_heads=4,
        attention_feedforward_multiplier=2,
        attention_dropout=0.0,
        attention_relative_buckets=16,
        attention_min_steps=8,
        attention_max_steps_per_window=64,

        recurrent_hidden_size=16,
        recurrent_branch_size=16,
        recurrent_depth=2,
        recurrent_dropout=0.0,
        recurrent_min_steps=8,
        recurrent_max_steps_per_window=64,

        deeplob_levels=10,
        deeplob_conv_channels=16,
        deeplob_inception_channels=8,
        deeplob_recurrent_size=16,
        deeplob_recurrent_layers=1,
        deeplob_dropout=0.0,
        deeplob_min_snapshots=12,
        deeplob_max_snapshots=24,
        deeplob_max_traces=16,

        router_hidden_size=16,
        router_temperature=1.0,
        router_top_k=None,
    )


def _vector_norm(
    values: tuple[float, ...],
) -> float:
    return sqrt(
        sum(
            value * value
            for value in values
        )
    )


def _synthetic_gradient(
    size: int,
) -> tuple[float, ...]:
    """
    Small deterministic downstream gradient.

    It has no trading meaning.
    """

    return tuple(
        (
            0.01
            * (
                1.0
                + (index % 5)
            )
            * (
                -1.0
                if index % 2
                else 1.0
            )
        )
        for index in range(
            size
        )
    )


def run_smoke() -> None:
    """
    First runnable integration diagnostic for Market perception.

    It exercises:

        five experts
        → Router
        → Fusion
        → MarketRepresentation
        → causal backward on same transition_id

    Then it checks the expected degraded mode where L2 is absent and only
    DeepLOB becomes unavailable.
    """

    print(
        "MARKET PERCEPTION SMOKE"
    )

    config = _smoke_config()

    print(
        "build: "
        f"expert_repr={config.expert_representation_size}, "
        f"market_repr={config.market_representation_size}"
    )

    organ = build_default_market_organ(
        config
    )

    expected_experts = {
        "tcn",
        "ssm",
        "attention",
        "recurrent",
        "deeplob",
    }

    actual_experts = set(
        organ.expert_names
    )

    if actual_experts != expected_experts:
        raise RuntimeError(
            "Unexpected expert set: "
            f"{sorted(actual_experts)!r}"
        )

    # --------------------------------------------------------
    # Forward with all five experts available
    # --------------------------------------------------------

    transition_id = (
        "smoke:market:full"
    )

    market_input = _market_input(
        include_l2=True
    )

    representation = organ.encode(
        market_input,
        transition_id=transition_id,
        shared_context=(
            0.15,
            -0.20,
            0.35,
            0.10,
        ),
    )

    if (
        len(
            representation.vector
        )
        != config.market_representation_size
    ):
        raise RuntimeError(
            "Unexpected MarketRepresentation size: "
            f"{len(representation.vector)}"
        )

    unavailable = {
        name
        for name, available
        in representation.expert_available.items()
        if not available
    }

    if unavailable:
        raise RuntimeError(
            "Experts unexpectedly unavailable "
            f"with full synthetic input: "
            f"{sorted(unavailable)!r}"
        )

    weight_sum = sum(
        representation.expert_weights.values()
    )

    if abs(
        weight_sum - 1.0
    ) > 1e-6:
        raise RuntimeError(
            "Router weights do not sum to 1: "
            f"{weight_sum}"
        )

    print(
        "forward: OK"
    )
    print(
        "transition:",
        representation.transition_id,
    )
    print(
        "representation_size:",
        len(
            representation.vector
        ),
    )
    print(
        "representation_norm:",
        f"{_vector_norm(representation.vector):.6f}",
    )
    print(
        "available:",
        ", ".join(
            sorted(
                name
                for name, available
                in representation.expert_available.items()
                if available
            )
        ),
    )
    print(
        "router_weights:",
        " ".join(
            f"{name}={weight:.4f}"
            for name, weight
            in sorted(
                representation.expert_weights.items()
            )
        ),
    )

    # --------------------------------------------------------
    # Causal backward on the exact same transition
    # --------------------------------------------------------

    gradient = _synthetic_gradient(
        organ.representation_size
    )

    learned = organ.learn(
        transition_id,
        gradient,
    )

    if not learned:
        raise RuntimeError(
            "Causal Market learn() returned False "
            "for a valid forward transition"
        )

    # The trace must have been consumed. A second update to the same
    # transition is not allowed to silently reuse current state.
    learned_twice = organ.learn(
        transition_id,
        gradient,
    )

    if learned_twice:
        raise RuntimeError(
            "Causal Market trace was reused after consumption"
        )

    telemetry = organ.telemetry()

    if telemetry.learning_updates != 1:
        raise RuntimeError(
            "Unexpected Market organ learning update count: "
            f"{telemetry.learning_updates}"
        )

    print(
        "backward: OK"
    )
    print(
        "causal_trace_consumed: OK"
    )
    print(
        "learning_updates:",
        telemetry.learning_updates,
    )

    # --------------------------------------------------------
    # Missing-L2 degraded mode
    # --------------------------------------------------------

    no_l2_transition = (
        "smoke:market:no-l2"
    )

    no_l2_representation = organ.encode(
        _market_input(
            include_l2=False
        ),
        transition_id=no_l2_transition,
        # Keep non-L2 context identical. This block is specifically testing
        # graceful DeepLOB unavailability, not context-shape changes.
        shared_context=(
            0.15,
            -0.20,
            0.35,
            0.10,
        ),
    )

    if no_l2_representation.expert_available.get(
        "deeplob",
        True,
    ):
        raise RuntimeError(
            "DeepLOB should be unavailable without L2 history"
        )

    other_unavailable = {
        name
        for name, available
        in no_l2_representation.expert_available.items()
        if (
            name != "deeplob"
            and not available
        )
    }

    if other_unavailable:
        raise RuntimeError(
            "Non-DeepLOB experts unexpectedly became unavailable "
            "when only L2 was removed: "
            f"{sorted(other_unavailable)!r}"
        )

    print(
        "missing_l2: OK "
        "(DeepLOB unavailable, other experts still active)"
    )

    # Remove transient traces before process exit.
    organ.reset()

    print(
        "SMOKE: OK"
    )


if __name__ == "__main__":
    run_smoke()
