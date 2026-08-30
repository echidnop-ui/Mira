from __future__ import annotations

import math
import sys

import torch

from contracts.cognition import Affordance
from core.personal import PersonalCognitiveCore, PersonalCoreConfig
from policy.dynamic import DynamicAffordancePolicy, DynamicPolicyConfig
from runtime.cycle import MarketCognitionCycle
from runtime.market_decision import MarketDecisionCycle
from worlds.market.interface import MarketRepresentation


def _vector(
    size: int,
    phase: float,
) -> tuple[float, ...]:
    return tuple(
        math.sin(
            (index + 1) * 0.113 + phase
        )
        + 0.25
        * math.cos(
            (index + 1) * 0.047 - phase
        )
        for index in range(size)
    )


def _market_representation(
    transition_id: str,
    *,
    phase: float,
) -> MarketRepresentation:
    return MarketRepresentation(
        transition_id=transition_id,
        instrument_id="SBER_TQBR",
        as_of=1700.0 + phase,
        vector=_vector(
            96,
            phase,
        ),
        uncertainty=0.25,
        expert_weights={
            "attention": 0.25,
            "recurrent": 0.25,
            "ssm": 0.25,
            "tcn": 0.25,
        },
        expert_available={
            "attention": True,
            "deeplob": False,
            "recurrent": True,
            "ssm": True,
            "tcn": True,
        },
        diagnostics={
            "smoke": 1.0,
        },
    )


def _affordances(
    now: float,
) -> tuple[Affordance, ...]:
    return (
        Affordance(
            id="market:SBER:buy",
            world="market:tbank:sandbox",
            action="market.buy",
            offered_at=now,
            payload={
                "instrument_id": "SBER_TQBR",
            },
            constraints={
                "min_quantity": 1,
                "max_quantity": 4,
                "lot_size": 10,
            },
            expires_at=now + 5.0,
        ),
        Affordance(
            id="market:SBER:sell",
            world="market:tbank:sandbox",
            action="market.sell",
            offered_at=now,
            payload={
                "instrument_id": "SBER_TQBR",
            },
            constraints={
                "min_quantity": 1,
                "max_quantity": 2,
                "lot_size": 10,
            },
            expires_at=now + 5.0,
        ),
        Affordance(
            id="market:SBER:hold",
            world="market:tbank:sandbox",
            action="market.hold",
            offered_at=now,
            payload={
                "instrument_id": "SBER_TQBR",
            },
            constraints={},
            expires_at=now + 5.0,
        ),
    )


def main() -> int:
    print("MARKET DECISION CYCLE SMOKE")
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"hip: {getattr(torch.version, 'hip', None)}")
    print(f"accelerator_available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(
            "accelerator:",
            torch.cuda.get_device_name(0),
        )
    else:
        print("accelerator: CPU")

    core = PersonalCognitiveCore(
        PersonalCoreConfig(
            input_size=96,
            context_size=96,
            hidden_size=96,
            output_size=96,
            workspace_heads=4,
            dropout=0.0,
            learning_rate=1e-3,
            max_traces=32,
        )
    )

    policy = DynamicAffordancePolicy(
        DynamicPolicyConfig(
            state_size=96,
            affordance_feature_size=64,
            hidden_size=96,
            learning_rate=1e-3,
            max_traces=32,
        )
    )

    cognition_cycle = MarketCognitionCycle(
        core
    )
    decision_cycle = MarketDecisionCycle(
        cognition_cycle,
        policy,
    )

    print(f"core_device: {core.device}")
    print(f"policy_device: {policy.device}")

    now = 2000.0
    representation = _market_representation(
        "market-decision-smoke-1",
        phase=0.35,
    )
    affordances = _affordances(
        now
    )

    decision = decision_cycle.process(
        representation,
        affordances,
        now=now,
    )

    if (
        decision.transition_id
        != representation.transition_id
    ):
        raise AssertionError(
            "MarketDecision lost transition_id"
        )

    if (
        decision.cognitive_state.transition_id
        != representation.transition_id
    ):
        raise AssertionError(
            "IntegratedCognitiveState lost transition_id"
        )

    if (
        decision.intent.transition_id
        != representation.transition_id
    ):
        raise AssertionError(
            "ActionIntent lost transition_id"
        )

    if len(
        decision.cognitive_state.vector
    ) != 96:
        raise AssertionError(
            "unexpected cognitive state size"
        )

    offered = {
        affordance.id: affordance
        for affordance in affordances
    }
    if (
        decision.intent.affordance_id
        not in offered
    ):
        raise AssertionError(
            "Policy chose an Affordance that was not offered"
        )

    chosen = offered[
        decision.intent.affordance_id
    ]
    if (
        chosen.action
        != decision.intent.action
        or chosen.world
        != decision.intent.world
    ):
        raise AssertionError(
            "ActionIntent does not match chosen Affordance"
        )

    print("market_representation_size: 96")
    print("core_state_size: 96")
    print(
        "available_affordances:",
        ", ".join(
            affordance.action
            for affordance in affordances
        ),
    )
    print(
        "chosen_action:",
        decision.intent.action,
    )
    print("transition_preserved: OK")
    print("intent_matches_affordance: OK")

    # The integration must have created causal traces in both trainable
    # components using the same transition_id.
    policy_learned = policy.learn(
        representation.transition_id,
        advantage=1.0,
    )
    if not policy_learned:
        raise AssertionError(
            "Policy did not retain the decision trace"
        )

    output_gradient = tuple(
        0.01
        * math.sin(
            (index + 1) * 0.19
        )
        for index in range(96)
    )
    core_learned = core.learn(
        representation.transition_id,
        output_gradient,
    )
    if not core_learned:
        raise AssertionError(
            "Core did not retain the same causal transition"
        )

    if policy.learn(
        representation.transition_id,
        advantage=1.0,
    ):
        raise AssertionError(
            "Policy causal trace was reusable"
        )

    if core.learn(
        representation.transition_id,
        output_gradient,
    ):
        raise AssertionError(
            "Core causal trace was reusable"
        )

    print("shared_causal_transition: OK")
    print("policy_trace_consumed_once: OK")
    print("core_trace_consumed_once: OK")

    # A completely stale capability set must fail before Core integration, so
    # it cannot manufacture an unused Core causal trace.
    stale_transition = "market-decision-smoke-stale"
    stale_representation = _market_representation(
        stale_transition,
        phase=0.91,
    )
    stale_affordances = (
        Affordance(
            id="market:SBER:expired",
            world="market:tbank:sandbox",
            action="market.buy",
            offered_at=now - 10.0,
            expires_at=now,
        ),
    )

    stale_rejected = False
    try:
        decision_cycle.process(
            stale_representation,
            stale_affordances,
            now=now,
        )
    except ValueError:
        stale_rejected = True

    if not stale_rejected:
        raise AssertionError(
            "all-expired Affordance set was not rejected"
        )

    # No Core trace should have been created for that rejected decision cycle.
    if core.learn(
        stale_transition,
        output_gradient,
    ):
        raise AssertionError(
            "stale affordances created an unintended Core trace"
        )

    print("all_expired_precore_guard: OK")

    core_telemetry = core.telemetry()
    policy_telemetry = policy.telemetry()

    print(
        "telemetry:",
        f"core_learning_updates={core_telemetry.learning_updates}",
        f"policy_learning_updates={policy_telemetry.learning_updates}",
        f"policy_decisions={policy_telemetry.decision_calls}",
    )

    print(
        "CHAIN: MarketRepresentation[96] -> Core[96] -> "
        "Affordances -> Policy -> ActionIntent"
    )
    print("MARKET DECISION CYCLE SMOKE: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
