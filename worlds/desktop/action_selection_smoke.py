from __future__ import annotations

import time

import torch

from core.interface import IntegratedCognitiveState
from policy.dynamic import DynamicAffordancePolicy, DynamicPolicyConfig
from worlds.desktop.inference import NeuralMotorProposal
from worlds.desktop.neural import ACTION_KINDS, VisualState
from worlds.desktop.neural_adapter import NeuralDesktopWorldAdapter


class _FakeInference:
    def __init__(self) -> None:
        self.state = None
        self.previous_state = None
        self.previous_frame = None
        self.last_frame = torch.zeros(1, 3, 16, 16)
        self.device = torch.device("cpu")
        self.model = None

    def capture_and_propose(self) -> NeuralMotorProposal:
        self.state = VisualState(
            patch_tokens=torch.zeros(1, 6, 8),
            region_tokens=torch.zeros(1, 2, 8),
            region_positions=torch.zeros(1, 2, 4),
            salience=torch.zeros(1, 2),
            persistence=torch.zeros(1, 2),
            latent=torch.zeros(1, 8),
            temporal_state=torch.zeros(1, 8),
            grid_size=(2, 3),
        )
        raw = {action: 0.01 for action in ACTION_KINDS}
        raw["right_click"] = 0.92
        raw["noop"] = 0.02
        total = sum(raw.values())
        distribution = tuple(raw[action] / total for action in ACTION_KINDS)
        return NeuralMotorProposal(
            action="right_click",
            x=100,
            y=100,
            confidence=0.9,
            action_distribution=distribution,
            target_distribution=(1.0,),
            screen_width=1920,
            screen_height=1080,
            key_code=30,
            frame_digest="synthetic-frame",
            target_bbox=(0.1, 0.1, 0.2, 0.2),
            trajectory=(0.0, 0.0, 0.0),
            read_byte_ids=(),
            read_confidence=0.0,
        )

    def close(self) -> None:
        pass


def _state(index: int, world: str) -> IntegratedCognitiveState:
    zeros = (0.0,) * 96
    return IntegratedCognitiveState(
        transition_id=f"desktop-action-selection-{index}",
        world_id=world,
        vector=zeros,
        workspace_vector=zeros,
        temporal_vector=zeros,
        context_vector=zeros,
        novelty=0.0,
        state_change=0.0,
        metadata={},
    )


def main() -> int:
    torch.manual_seed(7)
    world = NeuralDesktopWorldAdapter(
        _FakeInference(),
        capture_input=False,
        online_learning=False,
        allow_mouse=True,
        allow_clicks=True,
        allow_right_clicks=True,
        allow_keyboard=True,
        allow_typing=True,
    )
    try:
        frame = world.poll(time.monotonic())
        actions = tuple(affordance.action for affordance in frame.affordances)
        if len(actions) < 6:
            raise AssertionError(f"expected multiple motor affordances, got {actions}")
        if "desktop.right_click" not in actions or "desktop.noop" not in actions:
            raise AssertionError(f"missing expected alternatives: {actions}")

        priors = {
            affordance.action: float(affordance.payload["policy_prior_probability"])
            for affordance in frame.affordances
        }
        if not priors["desktop.right_click"] > priors["desktop.noop"]:
            raise AssertionError("synthetic collapsed specialist prior is invalid")

        policy = DynamicAffordancePolicy(
            DynamicPolicyConfig(
                state_size=96,
                affordance_feature_size=64,
                hidden_size=64,
                learning_rate=2e-2,
                max_traces=96,
                specialist_prior_weight=0.25,
                device="cpu",
            )
        )

        first = policy.choose(
            _state(0, frame.world), frame.affordances, now=time.monotonic()
        )
        if first.action != "desktop.right_click":
            raise AssertionError(
                "specialist prior did not guide the initially untrained shared Policy"
            )
        before_diagnostics = policy.score_diagnostics()
        before_scores = {action: (score, probability) for action, score, probability in before_diagnostics["scores"]}
        if len({round(value[0], 6) for value in before_scores.values()}) == 1:
            raise AssertionError("different Desktop affordances received identical scores")
        if not policy.learn(first.transition_id, -1.0):
            raise AssertionError("negative Desktop correction missed its causal trace")
        policy.choose(_state(999, frame.world), frame.affordances, now=time.monotonic(),
                      remember_for_learning=False)
        after_scores = {action: (score, probability) for action, score, probability in policy.score_diagnostics()["scores"]}
        if not after_scores[first.action][1] < before_scores[first.action][1]:
            raise AssertionError(
                f"negative correction did not reduce rejected probability: "
                f"{before_scores[first.action][1]} -> {after_scores[first.action][1]}"
            )

        switched_to: str | None = None
        for index in range(1, 80):
            intent = policy.choose(
                _state(index, frame.world), frame.affordances, now=time.monotonic()
            )
            if intent.action != "desktop.right_click":
                switched_to = intent.action
                policy.discard_trace(intent.transition_id)
                break
            if not policy.learn(intent.transition_id, -1.0):
                raise AssertionError("missing causal Policy trace")

        if switched_to is None:
            raise AssertionError("shared Policy failed to override collapsed motor prior")

        print({
            "ok": True,
            "affordance_count": len(actions),
            "initial_action": first.action,
            "switched_to": switched_to,
            "right_click_prior": priors["desktop.right_click"],
            "rejected_probability_before": before_scores[first.action][1],
            "rejected_probability_after": after_scores[first.action][1],
            "entropy": policy.score_diagnostics()["entropy"],
            "top1_margin": policy.score_diagnostics()["top1_margin"],
        })
        return 0
    finally:
        world.close()


if __name__ == "__main__":
    raise SystemExit(main())
