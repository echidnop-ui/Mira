from __future__ import annotations

from dataclasses import dataclass
from time import time

import torch

from contracts.cognition import (
    ActionAttempt,
    Affordance,
    CompletedAction,
    Consequence,
    ExecutionResult,
    Observation,
)


@dataclass(slots=True)
class MarketState:
    step: int
    price: float
    previous_price: float
    cash: float
    position: int
    equity: float


class SimulatedMarketWorld:
    """
    Minimal local Market World.

    It provides market observations and mechanically possible affordances.

    It deliberately contains NO trading strategy.

    In particular, this class never says:

        if price rises:
            buy()

    BUY/SELL/HOLD are merely possible actions.
    Policy decides which affordance Heiwa selects.
    """

    world_id = "market.local"

    observation_size = 8
    affordance_feature_size = 6

    def __init__(
        self,
        prices: list[float] | tuple[float, ...],
        *,
        initial_cash: float = 1000.0,
        max_position: int = 10,
    ) -> None:
        if len(prices) == 0:
            raise ValueError(
                "prices must contain at least one value"
            )

        if any(price <= 0 for price in prices):
            raise ValueError(
                "all prices must be > 0"
            )

        if initial_cash <= 0:
            raise ValueError(
                "initial_cash must be > 0"
            )

        if max_position <= 0:
            raise ValueError(
                "max_position must be > 0"
            )

        self._prices = tuple(
            float(price)
            for price in prices
        )

        self.initial_cash = float(initial_cash)
        self.max_position = int(max_position)

        self.cash = float(initial_cash)
        self.position = 0

        self._index = 0
        self._previous_price = self._prices[0]

    # ------------------------------------------------------------
    # MARKET STATE
    # ------------------------------------------------------------

    @property
    def price(self) -> float:
        return self._prices[self._index]

    @property
    def step_index(self) -> int:
        return self._index

    @property
    def equity(self) -> float:
        return (
            self.cash
            + self.position * self.price
        )

    def state(self) -> MarketState:
        return MarketState(
            step=self._index,
            price=self.price,
            previous_price=self._previous_price,
            cash=self.cash,
            position=self.position,
            equity=self.equity,
        )

    # ------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------

    def observe(self) -> Observation:
        state = self.state()

        return Observation(
            observation_id=(
                f"{self.world_id}:obs:{self._index}"
            ),
            world_id=self.world_id,
            kind="market.snapshot",
            payload={
                "step": state.step,
                "price": state.price,
                "previous_price": state.previous_price,
                "cash": state.cash,
                "position": state.position,
                "equity": state.equity,
            },
        )

    def encode_observation(
        self,
        observation: Observation,
    ) -> torch.Tensor:
        if observation.world_id != self.world_id:
            raise ValueError(
                "Observation belongs to another World"
            )

        payload = observation.payload

        price = float(payload["price"])
        previous_price = float(
            payload["previous_price"]
        )
        cash = float(payload["cash"])
        position = int(payload["position"])
        equity = float(payload["equity"])
        step = int(payload["step"])

        price_return = (
            (price - previous_price)
            / previous_price
        )

        initial_price = self._prices[0]

        price_relative = (
            price / initial_price
        ) - 1.0

        cash_relative = (
            cash / self.initial_cash
        )

        position_value_relative = (
            position * price
            / self.initial_cash
        )

        equity_relative = (
            equity / self.initial_cash
        ) - 1.0

        position_relative = (
            position / self.max_position
        )

        can_buy = float(
            cash >= price
            and position < self.max_position
        )

        # Only current time is represented.
        # No future market information is used.
        time_signal = float(step) / 1000.0

        return torch.tensor(
            [
                price_return,
                price_relative,
                cash_relative,
                position_value_relative,
                equity_relative,
                position_relative,
                can_buy,
                time_signal,
            ],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------
    # DYNAMIC AFFORDANCES
    # ------------------------------------------------------------

    def affordances(
        self,
    ) -> tuple[Affordance, ...]:
        actions: list[Affordance] = []

        # Waiting/holding is always mechanically possible.
        actions.append(
            self._affordance(
                name="hold",
                parameters={
                    "reference_step": self._index,
                    "reference_price": self.price,
                },
            )
        )

        # This is a capability check, not a trading decision.
        if (
            self.cash >= self.price
            and self.position < self.max_position
        ):
            actions.append(
                self._affordance(
                    name="buy",
                    parameters={
                        "quantity": 1,
                        "reference_step": self._index,
                        "reference_price": self.price,
                    },
                )
            )

        # SELL is only meaningful if an asset exists to sell.
        if self.position > 0:
            actions.append(
                self._affordance(
                    name="sell",
                    parameters={
                        "quantity": 1,
                        "reference_step": self._index,
                        "reference_price": self.price,
                    },
                )
            )

        return tuple(actions)

    def encode_affordances(
        self,
        affordances: tuple[Affordance, ...],
    ) -> torch.Tensor:
        rows: list[list[float]] = []

        for affordance in affordances:
            if affordance.world_id != self.world_id:
                raise ValueError(
                    "Affordance belongs to another World"
                )

            action = affordance.action

            is_hold = float(
                action == "market.hold"
            )
            is_buy = float(
                action == "market.buy"
            )
            is_sell = float(
                action == "market.sell"
            )

            if not (
                is_hold
                or is_buy
                or is_sell
            ):
                raise ValueError(
                    f"unknown market action: {action}"
                )

            quantity = float(
                affordance.parameters.get(
                    "quantity",
                    0,
                )
            )

            notional_relative = (
                quantity * self.price
                / self.initial_cash
            )

            rows.append(
                [
                    is_hold,
                    is_buy,
                    is_sell,
                    self.position
                    / self.max_position,
                    self.cash
                    / self.initial_cash,
                    notional_relative,
                ]
            )

        return torch.tensor(
            rows,
            dtype=torch.float32,
        )

    def _affordance(
        self,
        *,
        name: str,
        parameters: dict,
    ) -> Affordance:
        return Affordance(
            affordance_id=(
                f"{self.world_id}:"
                f"{self._index}:"
                f"{name}"
            ),
            world_id=self.world_id,
            action=f"market.{name}",
            parameters=parameters,
        )

    # ------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------

    def execute(
        self,
        attempt: ActionAttempt,
    ) -> ExecutionResult:
        if attempt.world_id != self.world_id:
            return ExecutionResult(
                completed=False,
                failure_kind=(
                    "execution.wrong_world"
                ),
                failure={
                    "expected_world": self.world_id,
                    "received_world": attempt.world_id,
                },
            )

        stale = self._stale_reason(
            attempt
        )

        if stale is not None:
            return ExecutionResult(
                completed=False,
                failure_kind=(
                    "execution.stale_affordance"
                ),
                failure={
                    "reason": stale,
                },
            )

        before = self.state()

        if attempt.action == "market.hold":
            pass

        elif attempt.action == "market.buy":
            quantity = self._quantity(
                attempt
            )

            cost = quantity * self.price

            if (
                self.position + quantity
                > self.max_position
            ):
                return ExecutionResult(
                    completed=False,
                    failure_kind=(
                        "execution.position_limit"
                    ),
                    failure={
                        "position": self.position,
                        "quantity": quantity,
                        "max_position": self.max_position,
                    },
                )

            if cost > self.cash:
                return ExecutionResult(
                    completed=False,
                    failure_kind=(
                        "execution.insufficient_cash"
                    ),
                    failure={
                        "cash": self.cash,
                        "required": cost,
                    },
                )

            self.cash -= cost
            self.position += quantity

        elif attempt.action == "market.sell":
            quantity = self._quantity(
                attempt
            )

            if quantity > self.position:
                return ExecutionResult(
                    completed=False,
                    failure_kind=(
                        "execution.insufficient_position"
                    ),
                    failure={
                        "position": self.position,
                        "quantity": quantity,
                    },
                )

            self.position -= quantity
            self.cash += (
                quantity * self.price
            )

        else:
            return ExecutionResult(
                completed=False,
                failure_kind=(
                    "execution.unknown_action"
                ),
                failure={
                    "action": attempt.action,
                },
            )

        after = self.state()

        return ExecutionResult(
            completed=True,
            result={
                "price": self.price,
                "cash_before": before.cash,
                "cash_after": after.cash,
                "position_before": before.position,
                "position_after": after.position,
                "equity_after": after.equity,
            },
        )

    def _stale_reason(
        self,
        attempt: ActionAttempt,
    ) -> str | None:
        reference_step = (
            attempt.arguments.get(
                "reference_step"
            )
        )

        if reference_step != self._index:
            return (
                "market step changed before execution"
            )

        reference_price = (
            attempt.arguments.get(
                "reference_price"
            )
        )

        if reference_price is None:
            return (
                "missing reference_price"
            )

        if abs(
            float(reference_price)
            - self.price
        ) > 1e-9:
            return (
                "market price changed before execution"
            )

        return None

    @staticmethod
    def _quantity(
        attempt: ActionAttempt,
    ) -> int:
        raw = attempt.arguments.get(
            "quantity"
        )

        if (
            not isinstance(raw, int)
            or isinstance(raw, bool)
            or raw <= 0
        ):
            raise ValueError(
                "quantity must be a positive integer"
            )

        return raw

    # ------------------------------------------------------------
    # CONSEQUENCE
    # ------------------------------------------------------------

    def consequence(
        self,
        completed_action: CompletedAction,
        next_observation: Observation,
    ) -> Consequence:
        """
        Describe what objectively happened after an earlier action.

        This is environment evidence, NOT a cognitive reward formula.

        Market World reports measurable facts such as:

            equity before environment evolution
            equity after environment evolution
            price change

        A learner may later decide how those facts should train cognition.
        """

        if completed_action.world_id != self.world_id:
            raise ValueError(
                "CompletedAction belongs to another World"
            )

        if next_observation.world_id != self.world_id:
            raise ValueError(
                "Observation belongs to another World"
            )

        result = completed_action.result

        if "equity_after" not in result:
            raise ValueError(
                "CompletedAction lacks equity_after"
            )

        previous_equity = float(
            result["equity_after"]
        )

        next_equity = float(
            next_observation.payload["equity"]
        )

        previous_price = float(
            result["price"]
        )

        next_price = float(
            next_observation.payload["price"]
        )

        equity_delta = (
            next_equity - previous_equity
        )

        equity_delta_relative = (
            equity_delta / self.initial_cash
        )

        price_delta = (
            next_price - previous_price
        )

        price_return = (
            price_delta / previous_price
        )

        return Consequence(
            consequence_id=(
                f"{self.world_id}:"
                f"consequence:"
                f"{completed_action.transition_id}"
            ),
            transition_id=(
                completed_action.transition_id
            ),
            world_id=self.world_id,
            kind="market.equity_changed",
            payload={
                "action": completed_action.action,
                "equity_before": previous_equity,
                "equity_after": next_equity,
                "equity_delta": equity_delta,
                "equity_delta_relative": (
                    equity_delta_relative
                ),
                "price_before": previous_price,
                "price_after": next_price,
                "price_delta": price_delta,
                "price_return": price_return,
            },
            intent_id=completed_action.intent_id,
            attempt_id=completed_action.attempt_id,
            completed_action_id=(
                completed_action.completed_action_id
            ),
            observed_at=time(),
        )

    # ------------------------------------------------------------
    # ENVIRONMENT TIME
    # ------------------------------------------------------------

    def advance(self) -> bool:
        """
        Move external market time forward.

        Returns False when the local price sequence is exhausted.
        """

        if (
            self._index + 1
            >= len(self._prices)
        ):
            return False

        self._previous_price = self.price
        self._index += 1

        return True
