from __future__ import annotations

from dataclasses import replace
import math

import torch

from contracts.cognition import Observation
from core.interface import CoreInput
from language import LanguageInput, NeuralLanguageConfig, NeuralLanguageOrgan
from policy.dynamic import DynamicAffordancePolicy, DynamicPolicyConfig
from runtime.causal import CausalCognitionCycle
from runtime.causal_smoke import _ControlledWorld, _RewardInterpreter, _core


class _LanguageWorld(_ControlledWorld):
    """Controlled causal World whose observation is external language."""

    def __init__(self, *, max_actions: int, text: str) -> None:
        super().__init__(max_actions=max_actions)
        self.text = text

    def poll(self, now: float):
        frame = super().poll(now)
        if not frame.observations:
            return frame
        observation = frame.observations[0]
        return replace(
            frame,
            observations=(
                Observation(
                    id=observation.id,
                    world=observation.world,
                    kind="controlled.language",
                    observed_at=observation.observed_at,
                    payload={"text": self.text},
                ),
            ),
        )


class _LanguageInputBuilder:
    def __init__(self, language: NeuralLanguageOrgan) -> None:
        self.language = language
        self.representations: list[tuple[float, ...]] = []

    def build(
        self,
        observations: tuple[Observation, ...],
        *,
        transition_id: str,
        world_id: str,
    ) -> CoreInput:
        if len(observations) != 1:
            raise AssertionError("controlled World must expose one observation")
        text = observations[0].payload.get("text")
        if not isinstance(text, str) or not text:
            raise AssertionError("controlled language observation needs text")
        representation = self.language.encode(
            LanguageInput(
                transition_id=transition_id,
                text=text,
                source_kind="world",
                source_id=observations[0].id,
                metadata={"automatic_teacher_learning": False},
            ),
            # Ordinary World language is perception, not teacher evidence.
            remember_for_learning=False,
        )
        self.representations.append(representation.vector)
        return CoreInput(
            transition_id=transition_id,
            world_id=world_id,
            representation=representation.vector,
            metadata={
                "observation_id": observations[0].id,
                "language_token_count": representation.token_count,
                "automatic_teacher_learning": False,
            },
        )


def _language() -> NeuralLanguageOrgan:
    return NeuralLanguageOrgan(
        NeuralLanguageConfig(
            representation_size=8,
            cognitive_size=8,
            embedding_size=16,
            hidden_size=16,
            recurrent_layers=1,
            attention_heads=4,
            attention_window=8,
            ffn_expansion=2,
            ssm_backend="selective_ssm",
            stateweave_plastic_rank=4,
            stateweave_dynamic_rank=4,
            dropout=0.0,
            max_input_bytes=32,
            max_generation_bytes=16,
            max_traces=16,
            device="cpu",
        )
    )


def _rms_delta(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left, right, strict=True))
        / len(left)
    )


def _language_to_core_counterfactual(
    language: NeuralLanguageOrgan,
) -> tuple[float, float]:
    left = language.encode(
        LanguageInput("probe:left", "опасность слева", "world"),
        remember_for_learning=False,
    )
    right = language.encode(
        LanguageInput("probe:right", "опасность справа", "world"),
        remember_for_learning=False,
    )
    language_delta = _rms_delta(left.vector, right.vector)

    # Identical Core weights and initial persistent state.  The Language
    # representation is the only changed input, so a different shared state is
    # causal influence rather than two sequential independent calls.
    torch.manual_seed(31)
    left_core = _core()
    torch.manual_seed(31)
    right_core = _core()
    left_state = left_core.integrate(
        CoreInput("probe:left", "controlled", left.vector),
        remember_for_learning=False,
    )
    right_state = right_core.integrate(
        CoreInput("probe:right", "controlled", right.vector),
        remember_for_learning=False,
    )
    core_delta = _rms_delta(left_state.vector, right_state.vector)
    return language_delta, core_delta


def main() -> int:
    torch.manual_seed(23)
    language = _language()
    language_delta, core_delta = _language_to_core_counterfactual(language)
    assert language_delta > 1e-4
    assert core_delta > 1e-5

    torch.manual_seed(7)
    world = _LanguageWorld(max_actions=8, text="опасность слева")
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
    builder = _LanguageInputBuilder(language)
    cycle = CausalCognitionCycle(
        world=world,
        core=core,
        policy=policy,
        input_builder=builder,
        consequence_interpreter=_RewardInterpreter(),
        transition_prefix="language-causal",
    )

    results = [cycle.step(now=float(step)) for step in range(9)]
    learned = tuple(item for result in results for item in result.learning)

    assert len(builder.representations) == 8
    assert all(
        vector == builder.representations[0]
        for vector in builder.representations[1:]
    )
    assert world.initial_action is not None
    assert world.choices[-1] != world.initial_action
    assert all(len(set(row)) == 1 for row in world.causal_rows)
    assert len(learned) == 8
    assert all(item.policy_updated and item.core_updated for item in learned)
    assert all(item.skipped_reason is None for item in learned)
    assert cycle.pending_count == 0
    assert language.telemetry().traces_created == 0

    print(
        {
            "language_representation_delta": language_delta,
            "caused_core_state_delta": core_delta,
            "choices": world.choices,
            "initial_harmful_action": world.initial_action,
            "final_action": world.choices[-1],
            "language_teacher_updates": language.telemetry().teacher_language_updates,
            "language_learning_traces": language.telemetry().traces_created,
            "policy_updates": policy.telemetry().learning_updates,
            "core_updates": core.telemetry().learning_updates,
            "pending": cycle.pending_count,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
