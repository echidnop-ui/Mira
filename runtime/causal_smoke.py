from __future__ import annotations

from dataclasses import dataclass

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


def main() -> int:
    torch.manual_seed(7)
    world = _ControlledWorld(max_actions=8)
    core = _core()
    policy = DynamicAffordancePolicy(
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
    cycle = CausalCognitionCycle(
        world=world,
        core=core,
        policy=policy,
        input_builder=_ControlledInputBuilder(),
        consequence_interpreter=_RewardInterpreter(),
        transition_prefix="controlled-test",
    )

    results = [cycle.step(now=float(step)) for step in range(9)]
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
    assert core.telemetry().forward_calls == 8
    assert core.telemetry().learning_updates == 8
    assert core.telemetry().traces_consumed == 8

    print(
        {
            "choices": world.choices,
            "initial_harmful_action": world.initial_action,
            "final_action": world.choices[-1],
            "causal_resolutions": len(learned),
            "policy_updates": policy.telemetry().learning_updates,
            "core_updates": core.telemetry().learning_updates,
            "pending": cycle.pending_count,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
