from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time

import torch

from contracts.cognition import (
    ActionAttempt,
    Affordance,
    CompletedAction,
    Consequence,
    Observation,
)
from core.interface import CoreInput
from core.personal import PersonalCognitiveCore, PersonalCoreConfig
from policy.dynamic import DynamicAffordancePolicy, DynamicPolicyConfig
from runtime.causal import (
    CausalCognitionCycle,
    CausalLearningSignal,
    PendingTransition,
)
from worlds.interface import WorldFrame


class _ControlledInputBuilder:
    def build(
        self,
        observations: tuple[Observation, ...],
        *,
        transition_id: str,
        world_id: str,
    ) -> CoreInput:
        if len(observations) != 1:
            raise AssertionError("controlled World must expose one observation")
        signal = float(observations[0].payload["signal"])
        return CoreInput(
            transition_id=transition_id,
            world_id=world_id,
            representation=(signal, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            metadata={"observation_id": observations[0].id},
        )


class _RewardInterpreter:
    def evaluate(
        self,
        consequence: Consequence,
        transition: PendingTransition,
    ) -> CausalLearningSignal:
        assert consequence.transition_id == transition.transition_id
        reward = float(consequence.payload["reward"])
        # Controlled auxiliary target: the first shared Core coordinate learns
        # to predict transition utility.  This is a real dL/dCoreOutput vector,
        # not reward copied blindly into the Core optimizer.
        core_gradient = (
            transition.state.vector[0] - reward,
        ) + (0.0,) * 7
        return CausalLearningSignal(
            policy_advantage=reward,
            core_output_gradient=core_gradient,
            terminal=True,
        )


class _NoLearningInterpreter:
    def evaluate(
        self,
        consequence: Consequence,
        transition: PendingTransition,
    ) -> CausalLearningSignal:
        assert consequence.transition_id == transition.transition_id
        return CausalLearningSignal(terminal=True)


@dataclass(slots=True)
class _QueuedOutcome:
    completed: CompletedAction
    consequence: Consequence


class _ControlledWorld:
    """One cue, two actions; the initially selected action becomes harmful."""

    def __init__(self, *, max_actions: int) -> None:
        self.max_actions = max_actions
        self.actions = 0
        self.initial_action: str | None = None
        self.choices: list[str] = []
        self.causal_rows: list[tuple[str, str, str, str]] = []
        self._queued: _QueuedOutcome | None = None

    @property
    def world_id(self) -> str:
        return "controlled"

    def poll(self, now: float) -> WorldFrame:
        completed_actions: tuple[CompletedAction, ...] = ()
        consequences: tuple[Consequence, ...] = ()
        if self._queued is not None:
            completed_actions = (self._queued.completed,)
            consequences = (self._queued.consequence,)
            self._queued = None

        if self.actions >= self.max_actions:
            return WorldFrame(
                world=self.world_id,
                observed_at=now,
                completed_actions=completed_actions,
                consequences=consequences,
            )

        observation = Observation(
            id=f"observation:{self.actions}",
            world=self.world_id,
            kind="controlled.cue",
            observed_at=now,
            payload={"signal": 1.0},
        )
        affordances = (
            Affordance("left", self.world_id, "choose.left", now),
            Affordance("right", self.world_id, "choose.right", now),
        )
        return WorldFrame(
            world=self.world_id,
            observed_at=now,
            observations=(observation,),
            affordances=affordances,
            completed_actions=completed_actions,
            consequences=consequences,
        )

    def execute(self, intent, now: float) -> ActionAttempt:
        if self.initial_action is None:
            self.initial_action = intent.action
        self.choices.append(intent.action)
        self.actions += 1

        attempt = ActionAttempt(
            id=f"attempt:{intent.transition_id}",
            intent_id=intent.id,
            transition_id=intent.transition_id,
            world=self.world_id,
            action=intent.action,
            attempted_at=now,
            accepted=True,
        )
        completed = CompletedAction(
            id=f"completed:{intent.transition_id}",
            attempt_id=attempt.id,
            intent_id=intent.id,
            transition_id=intent.transition_id,
            world=self.world_id,
            action=intent.action,
            completed_at=now + 0.01,
            status="completed",
            succeeded=True,
        )
        reward = -1.0 if intent.action == self.initial_action else 1.0
        consequence = Consequence(
            id=f"consequence:{intent.transition_id}",
            transition_id=intent.transition_id,
            world=self.world_id,
            kind="controlled.reward",
            observed_at=now + 0.02,
            completed_action_id=completed.id,
            intent_id=intent.id,
            attempt_id=attempt.id,
            payload={"reward": reward},
        )
        self.causal_rows.append(
            (intent.transition_id, attempt.transition_id,
             completed.transition_id, consequence.transition_id)
        )
        self._queued = _QueuedOutcome(completed, consequence)
        return attempt

    def close(self) -> None:
        return None


def _core() -> PersonalCognitiveCore:
    return PersonalCognitiveCore(
        PersonalCoreConfig(
            input_size=8,
            context_size=8,
            hidden_size=16,
            output_size=8,
            workspace_heads=4,
            attention_heads=4,
            router_hidden_size=16,
            router_top_k=2,
            ssm_depth=1,
            recurrent_depth=1,
            tcn_depth=1,
            tcn_kernel_size=2,
            attention_depth=1,
            attention_feedforward_multiplier=2,
            history_size=8,
            dropout=0.0,
            learning_rate=1e-3,
            max_traces=32,
            device="cpu",
        )
    )


def _policy() -> DynamicAffordancePolicy:
    return DynamicAffordancePolicy(
        DynamicPolicyConfig(
            state_size=8,
            affordance_feature_size=16,
            hidden_size=16,
            learning_rate=0.05,
            entropy_coefficient=0.0,
            exploration_mixture=0.0,
            device="cpu",
        )
    )


def _probe(
    core: PersonalCognitiveCore,
    policy: DynamicAffordancePolicy,
    *,
    signal: float,
    transition_id: str,
) -> tuple[str, tuple[str, ...], float]:
    core.reset_state()
    state = core.integrate(
        CoreInput(
            transition_id=transition_id,
            world_id="controlled",
            representation=(signal, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        remember_for_learning=False,
    )
    intent = policy.choose(
        state,
        (
            Affordance("probe-left", "controlled", "choose.left", 0.0),
            Affordance("probe-right", "controlled", "choose.right", 0.0),
        ),
        now=0.0,
        remember_for_learning=False,
    )
    return (
        intent.action,
        tuple(state.metadata["core_active_experts"]),
        policy.telemetry().last_score_margin,
    )


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _parameter_delta(
    before: dict[str, torch.Tensor],
    module: torch.nn.Module,
) -> float:
    squared = 0.0
    for name, value in module.state_dict().items():
        difference = value.detach().cpu().to(torch.float64) - before[name].to(
            torch.float64
        )
        squared += float(torch.sum(difference * difference))
    return math.sqrt(squared)


def _pause(*, step: bool, delay: float) -> None:
    if step:
        input("  [Enter for next event]")
    elif delay:
        time.sleep(delay)


def _print_step(
    result,
    *,
    policy_score_margin: float | None,
    step: bool,
    delay: float,
) -> None:
    for completed in result.frame.completed_actions:
        print(
            f"COMPLETED id={completed.id} attempt_id={completed.attempt_id} "
            f"transition_id={completed.transition_id} action={completed.action}"
        )
        _pause(step=step, delay=delay)
    for consequence in result.frame.consequences:
        print(
            f"CONSEQUENCE id={consequence.id} completed_action_id="
            f"{consequence.completed_action_id} transition_id="
            f"{consequence.transition_id} reward={consequence.payload['reward']:+.1f}"
        )
        _pause(step=step, delay=delay)
    for learned in result.learning:
        print(
            f"LEARNING transition_id={learned.transition_id} consequence_id="
            f"{learned.consequence_id} policy_updated={learned.policy_updated} "
            f"core_updated={learned.core_updated} resolved={learned.resolved}"
        )
        _pause(step=step, delay=delay)
    if result.frame.observations:
        observation = result.frame.observations[0]
        print(
            f"OBSERVATION id={observation.id} kind={observation.kind} "
            f"signal={observation.payload['signal']:.1f}"
        )
        _pause(step=step, delay=delay)
    if result.started is not None:
        started = result.started
        state = result.integrated_state
        assert state is not None
        print(
            f"ORGANS transition_id={started.transition_id} Core experts="
            f"{tuple(state.metadata['core_active_experts'])} Policy "
            f"score_margin={policy_score_margin:.6f}"
        )
        _pause(step=step, delay=delay)
        print(
            f"INTENT id={started.intent.id} transition_id="
            f"{started.intent.transition_id} action={started.intent.action}"
        )
        _pause(step=step, delay=delay)
        print(
            f"ATTEMPT id={started.attempt.id} intent_id="
            f"{started.attempt.intent_id} transition_id="
            f"{started.attempt.transition_id} accepted={started.attempt.accepted}"
        )
        _pause(step=step, delay=delay)


def _run_learning_demo(*, step: bool, delay: float):
    torch.manual_seed(7)
    world = _ControlledWorld(max_actions=8)
    core = _core()
    policy = _policy()
    before_action, before_experts, before_margin = _probe(
        core, policy, signal=0.9, transition_id="probe:before"
    )
    core_before = _snapshot(core)
    policy_before = _snapshot(policy)
    print(
        f"BEFORE signal=0.9 action={before_action} Core experts="
        f"{before_experts} Policy score_margin={before_margin:.6f}"
    )
    _pause(step=step, delay=delay)
    cycle = CausalCognitionCycle(
        world=world,
        core=core,
        policy=policy,
        input_builder=_ControlledInputBuilder(),
        consequence_interpreter=_RewardInterpreter(),
        transition_prefix="controlled-test",
    )

    results = []
    for number in range(9):
        result = cycle.step(now=float(number))
        results.append(result)
        _print_step(
            result,
            policy_score_margin=(
                policy.telemetry().last_score_margin
                if result.started is not None
                else None
            ),
            step=step,
            delay=delay,
        )
    learned = tuple(
        item
        for result in results
        for item in result.learning
    )

    assert len(world.choices) == 8
    assert world.initial_action is not None
    assert world.choices[-1] != world.initial_action, world.choices
    assert all(len(set(row)) == 1 for row in world.causal_rows)
    assert len(learned) == 8
    assert all(item.policy_updated for item in learned)
    assert all(item.core_updated for item in learned)
    assert cycle.pending_count == 0
    assert policy.telemetry().learning_updates == 8
    # Eight causal decisions plus the read-only BEFORE probe.
    assert core.telemetry().forward_calls == 9
    assert core.telemetry().learning_updates == 8
    assert core.telemetry().traces_consumed == 8

    core_delta = _parameter_delta(core_before, core)
    policy_delta = _parameter_delta(policy_before, policy)
    after_action, after_experts, after_margin = _probe(
        core, policy, signal=0.9, transition_id="probe:after"
    )
    assert math.isfinite(core_delta) and core_delta > 0.0
    assert math.isfinite(policy_delta) and policy_delta > 0.0
    assert after_action != before_action

    return (
        before_action,
        before_experts,
        before_margin,
        after_action,
        after_experts,
        after_margin,
        core_delta,
        policy_delta,
        world,
        learned,
        cycle,
    )


def _run_no_learning_control():
    torch.manual_seed(7)
    world = _ControlledWorld(max_actions=8)
    core = _core()
    policy = _policy()
    before_action, _, _ = _probe(
        core, policy, signal=0.9, transition_id="control:before"
    )
    core_before = _snapshot(core)
    policy_before = _snapshot(policy)
    cycle = CausalCognitionCycle(
        world=world,
        core=core,
        policy=policy,
        input_builder=_ControlledInputBuilder(),
        consequence_interpreter=_NoLearningInterpreter(),
        transition_prefix="controlled-no-learning",
    )
    results = [cycle.step(now=float(number)) for number in range(9)]
    learned = tuple(item for result in results for item in result.learning)
    after_action, _, _ = _probe(
        core, policy, signal=0.9, transition_id="control:after"
    )
    core_delta = _parameter_delta(core_before, core)
    policy_delta = _parameter_delta(policy_before, policy)
    assert all(not item.policy_updated and not item.core_updated for item in learned)
    assert before_action == after_action
    assert core_delta == 0.0
    assert policy_delta == 0.0
    return before_action, after_action, core_delta, policy_delta, world.choices


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch Heiwa's causal learning loop")
    parser.add_argument(
        "--step", action="store_true", help="wait for Enter after each causal event"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="seconds to pause after each causal event",
    )
    args = parser.parse_args()
    if not math.isfinite(args.delay) or args.delay < 0.0:
        parser.error("--delay must be a finite non-negative number")

    (
        before_action,
        _,
        _,
        after_action,
        after_experts,
        after_margin,
        core_delta,
        policy_delta,
        world,
        learned,
        cycle,
    ) = _run_learning_demo(step=args.step, delay=args.delay)
    control = _run_no_learning_control()

    print(
        f"AFTER signal=0.9 action={after_action} Core experts={after_experts} "
        f"Policy score_margin={after_margin:.6f}"
    )
    print(
        f"PARAMETER DELTA Core={core_delta:.12f} finite={math.isfinite(core_delta)} "
        f"Policy={policy_delta:.12f} finite={math.isfinite(policy_delta)}"
    )
    print(
        f"NO-LEARNING CONTROL before={control[0]} after={control[1]} "
        f"Core_delta={control[2]:.1f} Policy_delta={control[3]:.1f} "
        f"choices={control[4]}"
    )
    print(
        "SUMMARY "
        f"BEFORE={before_action} LEARNING=8 linked transitions "
        f"(Core+Policy updated={len(learned)}) AFTER={after_action} "
        f"pending={cycle.pending_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
