from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import math
import sys
from zoneinfo import ZoneInfo

import torch

from core.personal import PersonalCognitiveCore
from policy.dynamic import DynamicAffordancePolicy
from runtime.cycle import MarketCognitionCycle
from worlds.market.historical_futures import (
    FuturesAccountState,
    FuturesContractSpec,
    HistoricalFuturesWorld,
)
from worlds.market.input_builder import MarketInputBuilder
from worlds.market.interface import MarketBackendFrame
from worlds.market.moex_history import (
    MoexFuturesHistoryClient,
    MoexIssError,
)
from worlds.market.perception.defaults import (
    build_default_market_organ,
)


_MOSCOW = ZoneInfo("Europe/Moscow")


def _parse_moscow(value: object) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError("candle_begin is empty")

    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_MOSCOW)

    return parsed.timestamp()


def _price(event, name: str) -> float:
    value = float(event.payload[name])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"candle {name} must be finite and > 0"
        )
    return value


def _window_steps(window) -> int:
    return 0 if window is None else len(window.rows)


def _perception_ready(market_input) -> bool:
    return max(
        _window_steps(market_input.micro),
        _window_steps(market_input.short),
        _window_steps(market_input.medium),
        _window_steps(market_input.long),
    ) >= 8


def _single_event_frame(
    source: MarketBackendFrame,
    event,
) -> MarketBackendFrame:
    return MarketBackendFrame(
        backend_id=source.backend_id,
        mode=source.mode,
        observed_at=event.observed_at,
        events=(event,),
        instruments=(),
    )


def _policy_advantage_from_consequence(
    consequence,
    *,
    risk_unit_rub: float,
) -> float:
    """
    Explicit learning-objective adapter.

    Consequence remains factual external evidence.
    Policy advantage is a separate training signal:

        equity_delta / one-contract margin

    clipped to [-1, +1].

    This is deliberately small and inspectable rather than a giant
    handcrafted reward formula.
    """
    if risk_unit_rub <= 0.0:
        raise ValueError("risk_unit_rub must be > 0")

    equity_delta = float(
        consequence.payload["equity_delta"]
    )

    if not math.isfinite(equity_delta):
        raise ValueError(
            "Consequence equity_delta is not finite"
        )

    raw = equity_delta / risk_unit_rub
    return max(-1.0, min(1.0, raw))


def run(
    *,
    secid: str,
    from_date: str,
    till_date: str,
    starting_cash: float,
    slippage_ticks: float,
    extra_commission_per_contract: float,
    exploration_temperature: float,
    max_execution_delay_seconds: float,
    max_decisions: int,
    seed: int,
) -> None:
    print("MOEX HISTORICAL CAUSAL REPLAY")
    print("secid:", secid)
    print("range:", from_date, "..", till_date)
    print("starting_cash:", f"{starting_cash:.2f}")
    print(
        "exploration_temperature:",
        exploration_temperature,
    )
    if max_decisions <= 0:
        raise ValueError("max_decisions must be > 0")

    print("max_decisions:", max_decisions)
    print("seed:", seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    client = MoexFuturesHistoryClient()
    contract = client.contract(secid)

    source_frame = client.candle_frame(
        secid,
        from_date=from_date,
        till_date=till_date,
        interval_minutes=1,
        board_id=contract.board_id,
    )

    if len(source_frame.events) < 16:
        raise RuntimeError(
            "not enough real MOEX candles for replay"
        )

    point_value = float(
        contract.rubles_per_price_unit
    )
    margin = float(
        contract.initial_margin
    )
    commission = (
        float(contract.exchange_buy_sell_fee)
        + float(extra_commission_per_contract)
    )
    slippage_points = (
        float(contract.min_step)
        * float(slippage_ticks)
    )

    spec = FuturesContractSpec(
        instrument_id=secid,
        point_value=point_value,
        initial_margin_per_contract=margin,
        commission_per_contract=commission,
        slippage_points=slippage_points,
    )
    world = HistoricalFuturesWorld(
        spec=spec,
        account=FuturesAccountState(
            cash=float(starting_cash),
        ),
    )

    builder = MarketInputBuilder()
    organ = build_default_market_organ()
    core = PersonalCognitiveCore()
    cycle = MarketCognitionCycle(core)
    policy = DynamicAffordancePolicy()

    pending_intent = None
    equity_baselined = False

    decisions = 0
    consequences = 0
    completed_actions = 0
    rejected_actions = 0
    holds = 0
    gap_discards = 0
    nonzero_advantages = 0

    advantage_sum = 0.0
    advantage_abs_sum = 0.0

    action_counts: Counter[str] = Counter()
    first_actions: list[str] = []
    last_actions: list[str] = []

    initial_equity = float(starting_cash)
    last_close = None

    print(
        "mechanics:",
        f"point_value={point_value:.6f}",
        f"margin={margin:.2f}",
        f"commission_per_side={commission:.2f}",
        f"slippage_points={slippage_points:.6f}",
    )
    print(
        "mechanics_note:",
        "margin/fee are current MOEX contract metadata; "
        "price candles are historical",
    )
    print(
        "objective:",
        "advantage = clip(equity_delta / one_contract_margin, -1, +1)",
    )
    print(
        "execution_rule:",
        "decision after candle close -> execute at NEXT candle open",
    )

    try:
        for index, event in enumerate(
            source_frame.events
        ):
            if event.kind != "candle":
                continue

            close_price = _price(
                event,
                "close",
            )
            open_price = _price(
                event,
                "open",
            )
            begin_time = _parse_moscow(
                event.payload["candle_begin"]
            )
            end_time = float(
                event.observed_at
            )
            last_close = close_price

            # ------------------------------------------------
            # 1. Execute the PREVIOUS decision only after time
            #    advances to this candle's open.
            # ------------------------------------------------
            if pending_intent is not None:
                delay = (
                    begin_time
                    - float(
                        pending_intent.created_at
                    )
                )

                if delay < -1e-6:
                    raise RuntimeError(
                        "historical time moved backwards "
                        "between decision and execution"
                    )

                if (
                    delay
                    > max_execution_delay_seconds
                ):
                    policy.discard_trace(
                        pending_intent.transition_id
                    )
                    pending_intent = None
                    equity_baselined = False
                    gap_discards += 1

                else:
                    execution = (
                        world.execute_market_intent(
                            pending_intent,
                            mark_price=open_price,
                            now=begin_time,
                        )
                    )

                    completed = (
                        execution.completed
                        if execution is not None
                        else None
                    )

                    if execution is None:
                        holds += 1
                    else:
                        completed_actions += 1
                        if not execution.completed.succeeded:
                            rejected_actions += 1

                    consequence = (
                        world.observe_consequence(
                            transition_id=(
                                pending_intent.transition_id
                            ),
                            mark_price=close_price,
                            now=end_time,
                            completed_action=completed,
                        )
                    )
                    consequences += 1
                    equity_baselined = True

                    advantage = (
                        _policy_advantage_from_consequence(
                            consequence,
                            risk_unit_rub=margin,
                        )
                    )

                    if abs(advantage) > 1e-12:
                        nonzero_advantages += 1

                    advantage_sum += advantage
                    advantage_abs_sum += abs(
                        advantage
                    )

                    learned = policy.learn(
                        pending_intent.transition_id,
                        advantage,
                    )
                    if not learned:
                        raise RuntimeError(
                            "Policy lost the exact causal "
                            "transition trace"
                        )

                    pending_intent = None

                    if decisions >= max_decisions:
                        break

            # ------------------------------------------------
            # 2. Only NOW expose this completed candle.
            # ------------------------------------------------
            builder.ingest_frame(
                _single_event_frame(
                    source_frame,
                    event,
                )
            )

            try:
                market_input = builder.build(
                    secid,
                    as_of=end_time,
                    metadata={
                        "source": "moex:iss",
                        "historical": True,
                        "real_market_data": True,
                        "causal_replay": True,
                    },
                )
            except ValueError:
                continue

            if not _perception_ready(
                market_input
            ):
                continue

            transition_id = (
                "moex:history:replay:"
                f"{secid}:{end_time:.6f}:{index}"
            )

            representation = organ.encode(
                market_input,
                transition_id=transition_id,
                shared_context=None,
            )

            if len(representation.vector) != 96:
                raise RuntimeError(
                    "MarketRepresentation size is not 96"
                )

            available = tuple(
                name
                for name, flag
                in representation.expert_available.items()
                if flag
            )
            if not available:
                raise RuntimeError(
                    "no Market expert accepted "
                    "historical replay state"
                )

            state = cycle.process(
                representation,
                # Current Policy learning does not yet expose
                # a downstream state gradient for Core.
                remember_for_learning=False,
            )

            if len(state.vector) != 96:
                raise RuntimeError(
                    "Core state size is not 96"
                )

            # If a previous decision was intentionally discarded
            # across a session gap, establish a fresh factual
            # equity baseline. This event is not used as reward.
            if not equity_baselined:
                world.observe_consequence(
                    transition_id=(
                        f"replay:equity-baseline:"
                        f"{secid}:{end_time:.6f}"
                    ),
                    mark_price=close_price,
                    now=end_time,
                    completed_action=None,
                )
                equity_baselined = True

            affordances = world.build_affordances(
                now=end_time,
                mark_price=close_price,
            )

            intent = policy.choose(
                state,
                affordances,
                now=end_time,
                remember_for_learning=True,
                exploration_temperature=(
                    exploration_temperature
                ),
            )

            pending_intent = intent

            decisions += 1
            action_counts[intent.action] += 1

            if len(first_actions) < 20:
                first_actions.append(
                    intent.action
                )

            last_actions.append(
                intent.action
            )
            if len(last_actions) > 20:
                del last_actions[0]

        # The final decision has no next candle in this replay.
        if pending_intent is not None:
            policy.discard_trace(
                pending_intent.transition_id
            )
            pending_intent = None

        if decisions <= 0:
            raise RuntimeError(
                "replay produced no Policy decisions"
            )

        if consequences <= 0:
            raise RuntimeError(
                "replay produced no causal consequences"
            )

        if policy.telemetry().learning_updates <= 0:
            raise RuntimeError(
                "Policy received no learning updates"
            )

        if last_close is None:
            raise RuntimeError(
                "replay has no final market price"
            )

        final_equity = world.equity(
            last_close
        )
        pnl = (
            final_equity
            - initial_equity
        )

        policy_telemetry = policy.telemetry()
        core_telemetry = core.telemetry()
        builder_telemetry = builder.telemetry()

        print(
            "decisions:",
            decisions,
        )
        print(
            "consequences:",
            consequences,
        )
        print(
            "policy_learning_updates:",
            policy_telemetry.learning_updates,
        )
        print(
            "nonzero_advantages:",
            nonzero_advantages,
        )
        print(
            "mean_abs_advantage:",
            f"{advantage_abs_sum / max(1, consequences):.9f}",
        )
        print(
            "mean_advantage:",
            f"{advantage_sum / max(1, consequences):.9f}",
        )
        print(
            "completed_external_actions:",
            completed_actions,
        )
        print(
            "holds_without_fake_completion:",
            holds,
        )
        print(
            "rejected_actions:",
            rejected_actions,
        )
        print(
            "gap_discards:",
            gap_discards,
        )

        print(
            "action_counts:",
            " ".join(
                f"{name}={count}"
                for name, count
                in sorted(
                    action_counts.items()
                )
            ),
        )
        print(
            "first_20_actions:",
            ", ".join(first_actions),
        )
        print(
            "last_20_actions:",
            ", ".join(last_actions),
        )

        print(
            "cash:",
            f"{world.account.cash:.2f}",
        )
        print(
            "position_contracts:",
            world.account.position_contracts,
        )
        print(
            "realized_pnl:",
            f"{world.account.realized_pnl:.2f}",
        )
        print(
            "unrealized_pnl:",
            f"{world.unrealized_pnl(last_close):.2f}",
        )
        print(
            "commissions_paid:",
            f"{world.account.commissions_paid:.2f}",
        )
        print(
            "final_equity:",
            f"{final_equity:.2f}",
        )
        print(
            "net_pnl:",
            f"{pnl:.2f}",
        )

        print(
            "builder_events:",
            f"seen={builder_telemetry.events_seen}",
            f"used={builder_telemetry.events_used}",
            f"ignored={builder_telemetry.events_ignored}",
            f"stale={builder_telemetry.stale_temporal_events}",
        )
        print(
            "core_forward_calls:",
            core_telemetry.forward_calls,
        )
        print(
            "policy_last_action:",
            policy_telemetry.last_chosen_action,
        )
        print(
            "policy_last_score_margin:",
            f"{policy_telemetry.last_score_margin:.9f}",
        )

        print(
            "LEARNING_SCOPE:",
            "Policy only; Core and Market Organ are inference-only "
            "in this slice",
        )
        print(
            "CAUSAL_CHAIN:",
            "completed candle -> MarketInput -> Market Organ -> Core "
            "-> Policy -> ActionIntent -> next-open execution "
            "-> CompletedAction/hold -> Consequence -> Policy.learn",
        )
        print(
            "NO_LOOKAHEAD_RULE:",
            "decision at candle END, execution at NEXT candle BEGIN: OK",
        )
        print(
            "MOEX HISTORICAL CAUSAL REPLAY: OK"
        )

    finally:
        organ.reset()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--secid",
        default="GZU6",
    )
    parser.add_argument(
        "--from-date",
        default="2026-08-18",
    )
    parser.add_argument(
        "--till-date",
        default="2026-08-18",
    )
    parser.add_argument(
        "--starting-cash",
        type=float,
        default=10_000.0,
    )
    parser.add_argument(
        "--slippage-ticks",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--extra-commission-per-contract",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--exploration-temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-execution-delay-seconds",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260828,
    )

    return parser


def main() -> int:
    args = _parser().parse_args()

    try:
        run(
            secid=args.secid,
            from_date=args.from_date,
            till_date=args.till_date,
            starting_cash=args.starting_cash,
            slippage_ticks=args.slippage_ticks,
            extra_commission_per_contract=(
                args.extra_commission_per_contract
            ),
            exploration_temperature=(
                args.exploration_temperature
            ),
            max_execution_delay_seconds=(
                args.max_execution_delay_seconds
            ),
            max_decisions=args.max_decisions,
            seed=args.seed,
        )
    except MoexIssError as exc:
        print(
            "MOEX HISTORY ERROR:",
            str(exc),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            "MOEX HISTORICAL CAUSAL REPLAY FAILED:",
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
