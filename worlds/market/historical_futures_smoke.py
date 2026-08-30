from __future__ import annotations

from contracts.cognition import ActionIntent

from worlds.market.historical_futures import (
    CLOSE_SHORT,
    HOLD,
    OPEN_SHORT,
    FuturesAccountState,
    FuturesContractSpec,
    HistoricalFuturesWorld,
)


def _pick(affordances, action: str):
    for item in affordances:
        if item.action == action:
            return item
    raise RuntimeError(
        f"affordance {action!r} is unavailable"
    )


def _intent(
    *,
    transition_id: str,
    affordance,
    created_at: float,
) -> ActionIntent:
    return ActionIntent(
        id=f"intent:{transition_id}",
        transition_id=transition_id,
        affordance_id=affordance.id,
        world=affordance.world,
        action=affordance.action,
        created_at=created_at,
        parameters={
            "quantity_contracts": 1,
        },
    )


def _new_world() -> HistoricalFuturesWorld:
    # Synthetic contract mechanics only.
    # These are NOT hard-coded GAZP futures specifications.
    return HistoricalFuturesWorld(
        spec=FuturesContractSpec(
            instrument_id="SYNTHETIC:GAZP-FUT",
            point_value=10.0,
            initial_margin_per_contract=1_000.0,
            commission_per_contract=1.0,
            slippage_points=0.10,
        ),
        account=FuturesAccountState(
            cash=10_000.0,
        ),
    )


def _profitable_short() -> None:
    world = _new_world()

    open_transition = "hist:short-profit:open"
    open_affordances = world.build_affordances(
        now=1_000.0,
        mark_price=130.0,
    )

    open_short = _pick(
        open_affordances,
        OPEN_SHORT,
    )

    open_intent = _intent(
        transition_id=open_transition,
        affordance=open_short,
        created_at=1_000.0,
    )

    opened = world.execute_market_intent(
        open_intent,
        mark_price=130.0,
        now=1_000.0,
    )
    if opened is None:
        raise RuntimeError(
            "open_short produced no external execution"
        )

    if not opened.attempt.accepted:
        raise RuntimeError(
            "profitable short open was rejected"
        )
    if not opened.completed.succeeded:
        raise RuntimeError(
            "profitable short open did not complete"
        )
    if world.account.position_contracts != -1:
        raise RuntimeError(
            "short position was not opened"
        )

    # The market moves down after the SHORT.
    close_transition = "hist:short-profit:close"
    close_affordances = world.build_affordances(
        now=2_000.0,
        mark_price=120.0,
    )
    close_short = _pick(
        close_affordances,
        CLOSE_SHORT,
    )

    close_intent = _intent(
        transition_id=close_transition,
        affordance=close_short,
        created_at=2_000.0,
    )

    closed = world.execute_market_intent(
        close_intent,
        mark_price=120.0,
        now=2_000.0,
    )
    if closed is None:
        raise RuntimeError(
            "close_short produced no external execution"
        )

    consequence = world.observe_consequence(
        transition_id=close_transition,
        mark_price=120.0,
        now=2_001.0,
        completed_action=closed.completed,
    )

    if not closed.attempt.accepted:
        raise RuntimeError(
            "profitable short close was rejected"
        )
    if not closed.completed.succeeded:
        raise RuntimeError(
            "profitable short close did not complete"
        )
    if world.account.position_contracts != 0:
        raise RuntimeError(
            "short position was not closed"
        )
    if world.account.realized_pnl <= 0.0:
        raise RuntimeError(
            "falling price did not produce positive short PnL"
        )
    if consequence.transition_id != close_transition:
        raise RuntimeError(
            "Consequence lost transition_id"
        )
    if (
        consequence.completed_action_id
        != closed.completed.id
    ):
        raise RuntimeError(
            "Consequence lost CompletedAction link"
        )

    if (
        open_intent.id == opened.attempt.id
        or opened.attempt.id == opened.completed.id
        or open_intent.id == opened.completed.id
    ):
        raise RuntimeError(
            "Intent/Attempt/CompletedAction were conflated"
        )

    print("profitable_short: OK")
    print(
        "profit_realized:",
        f"{world.account.realized_pnl:.2f}",
    )


def _losing_short() -> None:
    world = _new_world()

    open_transition = "hist:short-loss:open"
    open_affordances = world.build_affordances(
        now=3_000.0,
        mark_price=130.0,
    )
    open_short = _pick(
        open_affordances,
        OPEN_SHORT,
    )

    opened = world.execute_market_intent(
        _intent(
            transition_id=open_transition,
            affordance=open_short,
            created_at=3_000.0,
        ),
        mark_price=130.0,
        now=3_000.0,
    )
    if opened is None or not opened.completed.succeeded:
        raise RuntimeError(
            "losing short could not be opened"
        )

    # The market moves UP against the SHORT.
    close_transition = "hist:short-loss:close"
    close_affordances = world.build_affordances(
        now=4_000.0,
        mark_price=140.0,
    )
    close_short = _pick(
        close_affordances,
        CLOSE_SHORT,
    )

    closed = world.execute_market_intent(
        _intent(
            transition_id=close_transition,
            affordance=close_short,
            created_at=4_000.0,
        ),
        mark_price=140.0,
        now=4_000.0,
    )
    if closed is None or not closed.completed.succeeded:
        raise RuntimeError(
            "losing short could not be closed"
        )

    consequence = world.observe_consequence(
        transition_id=close_transition,
        mark_price=140.0,
        now=4_001.0,
        completed_action=closed.completed,
    )

    if world.account.realized_pnl >= 0.0:
        raise RuntimeError(
            "rising price did not produce negative short PnL"
        )
    if consequence.transition_id != close_transition:
        raise RuntimeError(
            "losing consequence lost transition_id"
        )

    print("losing_short: OK")
    print(
        "loss_realized:",
        f"{world.account.realized_pnl:.2f}",
    )


def _hold_is_not_fake_execution() -> None:
    world = _new_world()

    affordances = world.build_affordances(
        now=5_000.0,
        mark_price=130.0,
    )
    hold = _pick(
        affordances,
        HOLD,
    )

    result = world.execute_market_intent(
        _intent(
            transition_id="hist:hold",
            affordance=hold,
            created_at=5_000.0,
        ),
        mark_price=130.0,
        now=5_000.0,
    )

    if result is not None:
        raise RuntimeError(
            "HOLD fabricated external execution"
        )

    print("hold_no_fake_execution: OK")


def run() -> None:
    print("HISTORICAL FUTURES EXECUTION SMOKE")
    _profitable_short()
    _losing_short()
    _hold_is_not_fake_execution()

    print(
        "CHAIN: ActionIntent -> ActionAttempt -> "
        "CompletedAction -> Consequence"
    )
    print("HISTORICAL FUTURES EXECUTION SMOKE: OK")


if __name__ == "__main__":
    run()
