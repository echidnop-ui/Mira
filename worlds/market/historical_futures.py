from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping
from uuid import uuid4

from contracts.cognition import (
    ActionAttempt,
    ActionIntent,
    Affordance,
    CompletedAction,
    Consequence,
)


WORLD_ID = "market:history:futures"

OPEN_LONG = "market.open_long"
CLOSE_LONG = "market.close_long"
OPEN_SHORT = "market.open_short"
CLOSE_SHORT = "market.close_short"
HOLD = "market.hold"


@dataclass(frozen=True, slots=True)
class FuturesContractSpec:
    instrument_id: str
    point_value: float
    initial_margin_per_contract: float
    commission_per_contract: float = 0.0
    slippage_points: float = 0.0

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must not be empty")
        if self.point_value <= 0.0 or not isfinite(self.point_value):
            raise ValueError("point_value must be finite and > 0")
        if (
            self.initial_margin_per_contract <= 0.0
            or not isfinite(self.initial_margin_per_contract)
        ):
            raise ValueError(
                "initial_margin_per_contract must be finite and > 0"
            )
        if (
            self.commission_per_contract < 0.0
            or not isfinite(self.commission_per_contract)
        ):
            raise ValueError(
                "commission_per_contract must be finite and >= 0"
            )
        if (
            self.slippage_points < 0.0
            or not isfinite(self.slippage_points)
        ):
            raise ValueError("slippage_points must be finite and >= 0")


@dataclass(slots=True)
class FuturesAccountState:
    cash: float
    position_contracts: int = 0
    average_entry_price: float | None = None
    blocked_margin: float = 0.0
    realized_pnl: float = 0.0
    commissions_paid: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.cash):
            raise ValueError("cash must be finite")


@dataclass(frozen=True, slots=True)
class HistoricalExecution:
    attempt: ActionAttempt
    completed: CompletedAction


@dataclass(slots=True)
class HistoricalFuturesWorld:
    spec: FuturesContractSpec
    account: FuturesAccountState
    world_id: str = WORLD_ID
    _affordances: dict[str, Affordance] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _last_equity: float | None = field(
        default=None,
        init=False,
        repr=False,
    )

    # --------------------------------------------------------
    # Mechanical market state
    # --------------------------------------------------------

    @property
    def free_cash(self) -> float:
        return self.account.cash - self.account.blocked_margin

    def unrealized_pnl(self, mark_price: float) -> float:
        self._require_price(mark_price)

        position = self.account.position_contracts
        entry = self.account.average_entry_price

        if position == 0 or entry is None:
            return 0.0

        return (
            (mark_price - entry)
            * position
            * self.spec.point_value
        )

    def equity(self, mark_price: float) -> float:
        return self.account.cash + self.unrealized_pnl(mark_price)

    # --------------------------------------------------------
    # Dynamic affordances
    # --------------------------------------------------------

    def build_affordances(
        self,
        *,
        now: float,
        mark_price: float,
    ) -> tuple[Affordance, ...]:
        self._require_time(now)
        self._require_price(mark_price)

        result: list[Affordance] = []

        def add(
            action: str,
            *,
            constraints: Mapping[str, object],
        ) -> None:
            affordance = Affordance(
                id=(
                    f"hist-fut:{self.spec.instrument_id}:"
                    f"{action}:{now:.6f}:{uuid4().hex[:8]}"
                ),
                world=self.world_id,
                action=action,
                offered_at=now,
                payload={
                    "instrument_id": self.spec.instrument_id,
                    "mark_price": mark_price,
                },
                constraints=dict(constraints),
            )
            result.append(affordance)

        position = self.account.position_contracts

        if position == 0:
            margin = self.spec.initial_margin_per_contract
            commission = self.spec.commission_per_contract

            if self.free_cash >= margin + commission:
                common = {
                    "instrument_id": self.spec.instrument_id,
                    "min_quantity_contracts": 1,
                    "max_quantity_contracts": max(
                        1,
                        int(
                            self.free_cash
                            // (
                                margin
                                + commission
                            )
                        ),
                    ),
                    "initial_margin_per_contract": margin,
                    "commission_per_contract": commission,
                }
                add(
                    OPEN_LONG,
                    constraints=common,
                )
                add(
                    OPEN_SHORT,
                    constraints=common,
                )

        elif position > 0:
            add(
                CLOSE_LONG,
                constraints={
                    "instrument_id": self.spec.instrument_id,
                    "max_quantity_contracts": position,
                },
            )

        else:
            add(
                CLOSE_SHORT,
                constraints={
                    "instrument_id": self.spec.instrument_id,
                    "max_quantity_contracts": abs(position),
                },
            )

        add(
            HOLD,
            constraints={
                "instrument_id": self.spec.instrument_id,
            },
        )

        self._affordances = {
            item.id: item
            for item in result
        }

        return tuple(result)

    # --------------------------------------------------------
    # Historical execution
    # --------------------------------------------------------

    def execute_market_intent(
        self,
        intent: ActionIntent,
        *,
        mark_price: float,
        now: float,
    ) -> HistoricalExecution | None:
        """
        Execute one already-chosen historical futures intent.

        `market.hold` is deliberately NOT converted into a fake external
        ActionAttempt/CompletedAction. It is a cognitive decision to do
        nothing externally and therefore returns None.
        """
        self._require_time(now)
        self._require_price(mark_price)

        if intent.world != self.world_id:
            raise ValueError(
                f"intent world {intent.world!r} does not match "
                f"{self.world_id!r}"
            )

        affordance = self._affordances.get(
            intent.affordance_id
        )
        if affordance is None:
            raise ValueError(
                "intent references an unknown/stale affordance"
            )

        if affordance.action != intent.action:
            raise ValueError(
                "intent action does not match referenced affordance"
            )

        if intent.action == HOLD:
            return None

        quantity = self._quantity_from_intent(intent)

        accepted, reason = self._can_execute(
            action=intent.action,
            quantity=quantity,
        )

        attempt = ActionAttempt(
            id=f"attempt:{uuid4().hex}",
            intent_id=intent.id,
            transition_id=intent.transition_id,
            world=self.world_id,
            action=intent.action,
            attempted_at=now,
            accepted=accepted,
            external_id=None,
            payload={
                "instrument_id": self.spec.instrument_id,
                "quantity_contracts": quantity,
                "mark_price": mark_price,
                "reason": reason,
                "historical": True,
            },
        )

        if not accepted:
            completed = CompletedAction(
                id=f"completed:{uuid4().hex}",
                attempt_id=attempt.id,
                intent_id=intent.id,
                transition_id=intent.transition_id,
                world=self.world_id,
                action=intent.action,
                completed_at=now,
                status="rejected",
                succeeded=False,
                external_id=None,
                payload={
                    "instrument_id": self.spec.instrument_id,
                    "quantity_contracts": quantity,
                    "reason": reason,
                    "historical": True,
                },
            )
            return HistoricalExecution(
                attempt=attempt,
                completed=completed,
            )

        fill_price = self._fill_price(
            intent.action,
            mark_price,
        )
        realized_delta = self._apply_fill(
            action=intent.action,
            quantity=quantity,
            fill_price=fill_price,
        )

        completed = CompletedAction(
            id=f"completed:{uuid4().hex}",
            attempt_id=attempt.id,
            intent_id=intent.id,
            transition_id=intent.transition_id,
            world=self.world_id,
            action=intent.action,
            completed_at=now,
            status="filled",
            succeeded=True,
            external_id=f"historical-fill:{uuid4().hex}",
            payload={
                "instrument_id": self.spec.instrument_id,
                "quantity_contracts": quantity,
                "fill_price": fill_price,
                "mark_price": mark_price,
                "realized_pnl_delta": realized_delta,
                "position_contracts": self.account.position_contracts,
                "blocked_margin": self.account.blocked_margin,
                "cash": self.account.cash,
                "historical": True,
            },
        )

        return HistoricalExecution(
            attempt=attempt,
            completed=completed,
        )

    # --------------------------------------------------------
    # Consequence
    # --------------------------------------------------------

    def observe_consequence(
        self,
        *,
        transition_id: str,
        mark_price: float,
        now: float,
        completed_action: CompletedAction | None = None,
    ) -> Consequence:
        self._require_time(now)
        self._require_price(mark_price)

        if not transition_id:
            raise ValueError("transition_id must not be empty")

        equity = self.equity(mark_price)
        previous = self._last_equity
        equity_delta = (
            0.0
            if previous is None
            else equity - previous
        )
        self._last_equity = equity

        return Consequence(
            id=f"consequence:{uuid4().hex}",
            transition_id=transition_id,
            world=self.world_id,
            kind="market.futures.account_mark",
            observed_at=now,
            completed_action_id=(
                completed_action.id
                if completed_action is not None
                else None
            ),
            payload={
                "instrument_id": self.spec.instrument_id,
                "mark_price": mark_price,
                "cash": self.account.cash,
                "blocked_margin": self.account.blocked_margin,
                "free_cash": self.free_cash,
                "position_contracts": (
                    self.account.position_contracts
                ),
                "average_entry_price": (
                    self.account.average_entry_price
                ),
                "realized_pnl": self.account.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl(
                    mark_price
                ),
                "commissions_paid": (
                    self.account.commissions_paid
                ),
                "equity": equity,
                "equity_delta": equity_delta,
                "historical": True,
            },
        )

    # --------------------------------------------------------
    # Validation / mechanics
    # --------------------------------------------------------

    def _can_execute(
        self,
        *,
        action: str,
        quantity: int,
    ) -> tuple[bool, str]:
        if quantity <= 0:
            return False, "quantity must be > 0"

        position = self.account.position_contracts

        if action in {OPEN_LONG, OPEN_SHORT}:
            if position != 0:
                return False, "open requires flat position"

            required = (
                quantity
                * (
                    self.spec.initial_margin_per_contract
                    + self.spec.commission_per_contract
                )
            )
            if self.free_cash < required:
                return False, "insufficient free cash for margin"

            return True, "accepted"

        if action == CLOSE_LONG:
            if position <= 0:
                return False, "no long position to close"
            if quantity > position:
                return False, "close quantity exceeds long position"
            return True, "accepted"

        if action == CLOSE_SHORT:
            if position >= 0:
                return False, "no short position to close"
            if quantity > abs(position):
                return False, "close quantity exceeds short position"
            return True, "accepted"

        return False, "unsupported action"

    def _apply_fill(
        self,
        *,
        action: str,
        quantity: int,
        fill_price: float,
    ) -> float:
        commission = (
            quantity
            * self.spec.commission_per_contract
        )
        self.account.cash -= commission
        self.account.commissions_paid += commission

        if action == OPEN_LONG:
            self.account.position_contracts = quantity
            self.account.average_entry_price = fill_price
            self.account.blocked_margin = (
                quantity
                * self.spec.initial_margin_per_contract
            )
            self.account.realized_pnl -= commission
            return -commission

        if action == OPEN_SHORT:
            self.account.position_contracts = -quantity
            self.account.average_entry_price = fill_price
            self.account.blocked_margin = (
                quantity
                * self.spec.initial_margin_per_contract
            )
            self.account.realized_pnl -= commission
            return -commission

        entry = self.account.average_entry_price
        if entry is None:
            raise RuntimeError(
                "closing position has no average entry price"
            )

        if action == CLOSE_LONG:
            gross = (
                (fill_price - entry)
                * quantity
                * self.spec.point_value
            )
            self.account.position_contracts -= quantity

        elif action == CLOSE_SHORT:
            gross = (
                (entry - fill_price)
                * quantity
                * self.spec.point_value
            )
            self.account.position_contracts += quantity

        else:
            raise RuntimeError(
                f"unsupported fill action: {action}"
            )

        self.account.cash += gross
        self.account.realized_pnl += gross - commission

        if self.account.position_contracts == 0:
            self.account.average_entry_price = None
            self.account.blocked_margin = 0.0
        else:
            self.account.blocked_margin = (
                abs(self.account.position_contracts)
                * self.spec.initial_margin_per_contract
            )

        return gross - commission

    def _fill_price(
        self,
        action: str,
        mark_price: float,
    ) -> float:
        slip = self.spec.slippage_points

        if action in {
            OPEN_LONG,
            CLOSE_SHORT,
        }:
            return mark_price + slip

        if action in {
            OPEN_SHORT,
            CLOSE_LONG,
        }:
            return mark_price - slip

        raise ValueError(
            f"unsupported fill action: {action}"
        )

    @staticmethod
    def _quantity_from_intent(
        intent: ActionIntent,
    ) -> int:
        raw = intent.parameters.get(
            "quantity_contracts",
            1,
        )

        if isinstance(raw, bool):
            raise ValueError(
                "quantity_contracts must be an integer"
            )

        quantity = int(raw)
        if quantity <= 0:
            raise ValueError(
                "quantity_contracts must be > 0"
            )

        return quantity

    @staticmethod
    def _require_price(value: float) -> None:
        if not isfinite(value) or value <= 0.0:
            raise ValueError(
                "mark_price must be finite and > 0"
            )

    @staticmethod
    def _require_time(value: float) -> None:
        if not isfinite(value):
            raise ValueError("time must be finite")
