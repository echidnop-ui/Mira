from __future__ import annotations

import torch

from contracts.cognition import (
    ActionAttempt,
    Affordance,
    CompletedAction,
    Consequence,
    Observation,
)
from core.personal import PersonalCognitiveCore, PersonalCoreConfig
from policy.dynamic import DynamicAffordancePolicy, DynamicPolicyConfig
from runtime.causal import CausalCognitionCycle
from runtime.causal_smoke import _ControlledInputBuilder, _RewardInterpreter
from worlds.interface import WorldFrame


class _EvictingTraceWorld:
    """Starts two transitions before releasing either terminal outcome."""

    world_id = "controlled"

    def __init__(self) -> None:
        self._polls = 0
        self._outcomes: list[tuple[CompletedAction, Consequence]] = []

    def poll(self, now: float) -> WorldFrame:
        poll_number = self._polls
        self._polls += 1
        if poll_number < 2:
            return WorldFrame(
                world=self.world_id,
                observed_at=now,
                observations=(
                    Observation(
                        id=f"observation:{poll_number}",
                        world=self.world_id,
                        kind="controlled.cue",
                        observed_at=now,
                        payload={"signal": 1.0},
                    ),
                ),
                affordances=(
                    Affordance("left", self.world_id, "choose.left", now),
                    Affordance("right", self.world_id, "choose.right", now),
                ),
            )

        completed, consequence = self._outcomes[poll_number - 2]
        return WorldFrame(
            world=self.world_id,
            observed_at=now,
            completed_actions=(completed,),
            consequences=(consequence,),
        )

    def execute(self, intent, now: float) -> ActionAttempt:
        index = len(self._outcomes)
        attempt = ActionAttempt(
            id=f"attempt:{index}",
            intent_id=intent.id,
            transition_id=intent.transition_id,
            world=self.world_id,
            action=intent.action,
            attempted_at=now,
            accepted=True,
        )
        completed = CompletedAction(
            id=f"completed:{index}",
            attempt_id=attempt.id,
            intent_id=intent.id,
            transition_id=intent.transition_id,
            world=self.world_id,
            action=intent.action,
            completed_at=now + 0.01,
            status="completed",
            succeeded=True,
        )
        consequence = Consequence(
            id=f"consequence:{index}",
            transition_id=intent.transition_id,
            world=self.world_id,
            kind="controlled.reward",
            observed_at=now + 0.02,
            completed_action_id=completed.id,
            intent_id=intent.id,
            attempt_id=attempt.id,
            payload={"reward": 1.0},
        )
        self._outcomes.append((completed, consequence))
        return attempt

    def close(self) -> None:
        return None


def _core_with_one_trace() -> PersonalCognitiveCore:
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
            max_traces=1,
            device="cpu",
        )
    )


def main() -> int:
    torch.manual_seed(1)
    core = _core_with_one_trace()
    policy = DynamicAffordancePolicy(
        DynamicPolicyConfig(
            state_size=8,
            affordance_feature_size=16,
            hidden_size=16,
            learning_rate=0.05,
            entropy_coefficient=0.0,
            exploration_mixture=0.0,
            max_traces=4,
            device="cpu",
        )
    )
    cycle = CausalCognitionCycle(
        world=_EvictingTraceWorld(),
        core=core,
        policy=policy,
        input_builder=_ControlledInputBuilder(),
        consequence_interpreter=_RewardInterpreter(),
        transition_prefix="atomic-regression",
    )

    cycle.step(now=0.0)
    cycle.step(now=1.0)
    assert cycle.pending_count == 2
    assert core.telemetry().traces_evicted == 1

    skipped = cycle.step(now=2.0).learning
    assert len(skipped) == 1
    assert skipped[0].resolved
    assert skipped[0].skipped_reason == "missing_causal_trace:core"
    assert not skipped[0].policy_updated
    assert not skipped[0].core_updated
    assert policy.telemetry().learning_updates == 0
    assert core.telemetry().learning_updates == 0
    assert cycle.pending_count == 1

    learned = cycle.step(now=3.0).learning
    assert len(learned) == 1
    assert learned[0].resolved
    assert learned[0].skipped_reason is None
    assert learned[0].policy_updated
    assert learned[0].core_updated
    assert policy.telemetry().learning_updates == 1
    assert core.telemetry().learning_updates == 1
    assert cycle.pending_count == 0

    print(
        {
            "evicted_transition": skipped[0].transition_id,
            "skipped_reason": skipped[0].skipped_reason,
            "partial_updates": 0,
            "next_transition_learned": learned[0].transition_id,
            "pending": cycle.pending_count,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
