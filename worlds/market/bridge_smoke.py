from __future__ import annotations

from math import cos, sin, sqrt

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "Market bridge smoke requires the same PyTorch environment "
        "as Market perception."
    ) from exc

from worlds.market.input_builder import (
    MarketInputBuilder,
    MarketInputBuilderConfig,
    TemporalScaleConfig,
)
from worlds.market.interface import (
    MarketBackendEvent,
    MarketBackendFrame,
    MarketMode,
)
from worlds.market.perception.defaults import (
    MarketPerceptionConfig,
    build_default_market_organ,
)


_INSTRUMENT_ID = "SYNTHETIC:BRIDGE"


# ============================================================
# Synthetic broker-normalized events
# ============================================================


def _candle_event(
    event_id: str,
    *,
    observed_at: float,
    interval_seconds: float,
    base_price: float,
    phase: float,
) -> MarketBackendEvent:
    """
    Create one deterministic normalized candle event.

    It intentionally looks like an event already emitted by a broker adapter;
    the MarketInputBuilder does not know which broker produced it.
    """

    drift = (
        0.04 * sin(phase)
        + 0.02 * cos(phase * 0.7)
    )

    open_price = (
        base_price + drift
    )

    close_price = (
        open_price
        + 0.05 * sin(
            phase * 1.31
        )
    )

    high_price = (
        max(
            open_price,
            close_price,
        )
        + 0.04
        + 0.01
        * (
            1.0
            + sin(
                phase * 0.43
            )
        )
    )

    low_price = (
        min(
            open_price,
            close_price,
        )
        - 0.04
        - 0.01
        * (
            1.0
            + cos(
                phase * 0.37
            )
        )
    )

    volume = (
        100.0
        + 15.0
        * (
            1.0
            + sin(
                phase * 0.53
            )
        )
    )

    return MarketBackendEvent(
        id=event_id,
        kind="candle",
        observed_at=observed_at,
        instrument_id=_INSTRUMENT_ID,
        payload={
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "interval_seconds": (
                interval_seconds
            ),
        },
    )


def _quote_event(
    event_id: str,
    *,
    observed_at: float,
    mid: float,
    phase: float,
) -> MarketBackendEvent:
    half_spread = (
        0.012
        + 0.003
        * (
            1.0
            + cos(
                phase
            )
        )
    )

    return MarketBackendEvent(
        id=event_id,
        kind="quote",
        observed_at=observed_at,
        instrument_id=_INSTRUMENT_ID,
        payload={
            "bid_price": (
                mid - half_spread
            ),
            "ask_price": (
                mid + half_spread
            ),
            "last_price": (
                mid
                + 0.004
                * sin(
                    phase * 1.7
                )
            ),
        },
    )


def _trade_event(
    event_id: str,
    *,
    observed_at: float,
    price: float,
    index: int,
) -> MarketBackendEvent:
    side = (
        "buy"
        if index % 3 != 0
        else "sell"
    )

    quantity = (
        1.0
        + float(
            index % 5
        )
    )

    return MarketBackendEvent(
        id=event_id,
        kind="trade",
        observed_at=observed_at,
        instrument_id=_INSTRUMENT_ID,
        payload={
            "price": price,
            "quantity": quantity,
            "side": side,
        },
    )


def _orderbook_event(
    event_id: str,
    *,
    observed_at: float,
    mid: float,
    phase: float,
    levels: int,
) -> MarketBackendEvent:
    bids: list[
        dict[str, float]
    ] = []

    asks: list[
        dict[str, float]
    ] = []

    half_spread = (
        0.01
        + 0.002
        * (
            1.0
            + sin(
                phase
            )
        )
    )

    for level_index in range(
        levels
    ):
        depth = float(
            level_index + 1
        )

        step = (
            0.01 * depth
        )

        bids.append(
            {
                "price": (
                    mid
                    - half_spread
                    - step
                ),
                "quantity": (
                    35.0
                    + 4.0 * depth
                    + 6.0
                    * (
                        1.0
                        + sin(
                            phase
                            + depth * 0.41
                        )
                    )
                ),
            }
        )

        asks.append(
            {
                "price": (
                    mid
                    + half_spread
                    + step
                ),
                "quantity": (
                    33.0
                    + 4.5 * depth
                    + 5.0
                    * (
                        1.0
                        + cos(
                            phase
                            + depth * 0.37
                        )
                    )
                ),
            }
        )

    return MarketBackendEvent(
        id=event_id,
        kind="orderbook",
        observed_at=observed_at,
        instrument_id=_INSTRUMENT_ID,
        payload={
            "bids": bids,
            "asks": asks,
        },
    )


def _event_frame(
) -> MarketBackendFrame:
    """
    Build chronologically ordered normalized backend events.

    We deliberately seed coarse history first and progressively move toward
    finer live data.

    This verifies an important MarketInputBuilder property:

        coarse history may seed equal/coarser views,
        but it must not be fabricated into a finer view.
    """

    as_of = 2_100_000_000.0

    events: list[
        MarketBackendEvent
    ] = []

    # --------------------------------------------------------
    # Long history: 300-second candles
    # --------------------------------------------------------

    long_start = (
        as_of - 7200.0
    )

    for index in range(
        12
    ):
        observed_at = (
            long_start
            + index * 300.0
        )

        events.append(
            _candle_event(
                f"long:{index}",
                observed_at=observed_at,
                interval_seconds=300.0,
                base_price=(
                    98.0
                    + index * 0.08
                ),
                phase=(
                    0.2 + index * 0.31
                ),
            )
        )

    # --------------------------------------------------------
    # Medium history: 60-second candles
    # --------------------------------------------------------

    medium_start = (
        as_of - 1800.0
    )

    for index in range(
        12
    ):
        observed_at = (
            medium_start
            + index * 60.0
        )

        events.append(
            _candle_event(
                f"medium:{index}",
                observed_at=observed_at,
                interval_seconds=60.0,
                base_price=(
                    99.2
                    + index * 0.045
                ),
                phase=(
                    0.7 + index * 0.27
                ),
            )
        )

    # --------------------------------------------------------
    # Short history: 15-second candles
    # --------------------------------------------------------

    short_start = (
        as_of - 600.0
    )

    for index in range(
        16
    ):
        observed_at = (
            short_start
            + index * 15.0
        )

        events.append(
            _candle_event(
                f"short:{index}",
                observed_at=observed_at,
                interval_seconds=15.0,
                base_price=(
                    99.7
                    + index * 0.025
                ),
                phase=(
                    1.1 + index * 0.23
                ),
            )
        )

    # --------------------------------------------------------
    # Micro history: quotes + trades
    # --------------------------------------------------------

    micro_start = (
        as_of - 120.0
    )

    for index in range(
        24
    ):
        observed_at = (
            micro_start
            + index * 2.0
        )

        mid = (
            100.1
            + 0.012 * index
            + 0.04
            * sin(
                index * 0.35
            )
        )

        events.append(
            _quote_event(
                f"quote:{index}",
                observed_at=observed_at,
                mid=mid,
                phase=(
                    0.3 + index * 0.19
                ),
            )
        )

        events.append(
            _trade_event(
                f"trade:{index}",
                observed_at=(
                    observed_at + 0.2
                ),
                price=(
                    mid
                    + 0.003
                    * sin(
                        index * 0.51
                    )
                ),
                index=index,
            )
        )

    # --------------------------------------------------------
    # L2 history: order-book snapshots
    # --------------------------------------------------------

    l2_start = (
        as_of - 60.0
    )

    for index in range(
        20
    ):
        observed_at = (
            l2_start
            + index * 2.0
        )

        mid = (
            100.4
            + 0.008 * index
            + 0.025
            * sin(
                index * 0.29
            )
        )

        events.append(
            _orderbook_event(
                f"book:{index}",
                observed_at=observed_at,
                mid=mid,
                phase=(
                    0.5 + index * 0.17
                ),
                levels=10,
            )
        )

    # All event groups were generated in chronological non-overlapping
    # ranges, but sort explicitly so future edits cannot accidentally break
    # causal ingestion order.
    events.sort(
        key=lambda event: (
            event.observed_at,
            event.id,
        )
    )

    return MarketBackendFrame(
        backend_id="synthetic-bridge",
        mode=MarketMode.SANDBOX,
        observed_at=as_of,
        events=tuple(
            events
        ),
    )


# ============================================================
# Small neural config
# ============================================================


def _perception_config(
) -> MarketPerceptionConfig:
    """
    Same architecture as the real default organ, smaller dimensions for a
    fast bridge diagnostic.

    Device stays None so the normal expert device-selection path is tested.
    In the intended TheRock/ROCm environment this should select the AMD GPU.
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

        device=None,

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


def _builder_config(
) -> MarketInputBuilderConfig:
    """
    Keep enough history for every expert while making the diagnostic small.
    """

    return MarketInputBuilderConfig(
        micro=TemporalScaleConfig(
            name="micro",
            interval_seconds=1.0,
            max_steps=64,
        ),
        short=TemporalScaleConfig(
            name="short",
            interval_seconds=15.0,
            max_steps=64,
        ),
        medium=TemporalScaleConfig(
            name="medium",
            interval_seconds=60.0,
            max_steps=64,
        ),
        long=TemporalScaleConfig(
            name="long",
            interval_seconds=300.0,
            max_steps=64,
        ),
        max_order_book_snapshots=32,
        order_book_levels=10,
    )


# ============================================================
# Diagnostic helpers
# ============================================================


def _window_steps(
    window,
) -> int:
    return (
        len(window.rows)
        if window is not None
        else 0
    )


def _vector_norm(
    values: tuple[
        float,
        ...,
    ],
) -> float:
    return sqrt(
        sum(
            value * value
            for value in values
        )
    )


def _gradient(
    size: int,
) -> tuple[
    float,
    ...,
]:
    return tuple(
        (
            0.01
            * (
                1.0
                + index % 4
            )
            * (
                1.0
                if index % 2 == 0
                else -1.0
            )
        )
        for index in range(
            size
        )
    )


# ============================================================
# Full bridge smoke
# ============================================================


def run_bridge_smoke(
) -> None:
    print(
        "MARKET BACKEND -> INPUT -> PERCEPTION BRIDGE SMOKE"
    )

    print(
        "torch:",
        torch.__version__,
    )
    print(
        "hip:",
        torch.version.hip,
    )
    print(
        "accelerator_available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        print(
            "accelerator:",
            torch.cuda.get_device_name(
                0
            ),
        )

    frame = _event_frame()

    builder = MarketInputBuilder(
        _builder_config()
    )

    builder.ingest_frame(
        frame
    )

    builder.set_portfolio_features(
        _INSTRUMENT_ID,
        {
            "cash_fraction": 0.72,
            "position_fraction": 0.28,
            "unrealized_return": 0.008,
        },
    )

    builder.set_context_features(
        _INSTRUMENT_ID,
        {
            "session_progress": 0.48,
            "recent_volatility": 0.34,
            "liquidity_state": 0.77,
        },
    )

    market_input = builder.build(
        _INSTRUMENT_ID,
        as_of=frame.observed_at,
        metadata={
            "bridge_smoke": True,
            "backend_id": (
                frame.backend_id
            ),
            "backend_mode": (
                frame.mode.value
            ),
        },
    )

    temporal_steps = {
        "micro": _window_steps(
            market_input.micro
        ),
        "short": _window_steps(
            market_input.short
        ),
        "medium": _window_steps(
            market_input.medium
        ),
        "long": _window_steps(
            market_input.long
        ),
    }

    for name, steps in (
        temporal_steps.items()
    ):
        if steps < 8:
            raise RuntimeError(
                f"{name} window has only "
                f"{steps} usable steps"
            )

    if len(
        market_input.order_book_history
    ) < 12:
        raise RuntimeError(
            "insufficient L2 history after "
            "MarketInputBuilder"
        )

    telemetry = builder.telemetry()

    print(
        "builder: OK"
    )
    print(
        "events:",
        f"seen={telemetry.events_seen}",
        f"used={telemetry.events_used}",
        f"ignored={telemetry.events_ignored}",
        f"stale={telemetry.stale_temporal_events}",
    )
    print(
        "windows:",
        " ".join(
            f"{name}={steps}"
            for name, steps
            in temporal_steps.items()
        ),
    )
    print(
        "l2_snapshots:",
        len(
            market_input.order_book_history
        ),
    )

    if telemetry.stale_temporal_events != 0:
        raise RuntimeError(
            "bridge smoke generated stale temporal events"
        )

    organ = build_default_market_organ(
        _perception_config()
    )

    transition_id = (
        "smoke:market:bridge"
    )

    representation = organ.encode(
        market_input,
        transition_id=transition_id,
        shared_context=(
            0.10,
            -0.05,
            0.20,
            0.15,
        ),
    )

    expected_experts = {
        "tcn",
        "ssm",
        "attention",
        "recurrent",
        "deeplob",
    }

    available_experts = {
        name
        for name, available
        in representation.expert_available.items()
        if available
    }

    if (
        available_experts
        != expected_experts
    ):
        raise RuntimeError(
            "unexpected expert availability after "
            "MarketInputBuilder: "
            f"{sorted(available_experts)!r}"
        )

    if (
        len(
            representation.vector
        )
        != organ.representation_size
    ):
        raise RuntimeError(
            "MarketRepresentation size mismatch"
        )

    weight_sum = sum(
        representation.expert_weights.values()
    )

    if abs(
        weight_sum - 1.0
    ) > 1e-6:
        raise RuntimeError(
            "Router weights do not sum to 1"
        )

    print(
        "perception_forward: OK"
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
                available_experts
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

    learned = organ.learn(
        transition_id,
        _gradient(
            organ.representation_size
        ),
    )

    if not learned:
        raise RuntimeError(
            "Market perception did not learn from "
            "the bridge transition"
        )

    if organ.learn(
        transition_id,
        _gradient(
            organ.representation_size
        ),
    ):
        raise RuntimeError(
            "bridge causal trace was reused"
        )

    print(
        "perception_backward: OK"
    )
    print(
        "causal_trace_consumed: OK"
    )

    organ.reset()
    builder.reset()

    print(
        "BRIDGE SMOKE: OK"
    )


if __name__ == "__main__":
    run_bridge_smoke()
