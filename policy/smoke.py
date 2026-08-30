from __future__ import annotations

import math
import sys

import torch

from contracts.cognition import Affordance
from core.interface import IntegratedCognitiveState
from policy.dynamic import (
    DynamicAffordancePolicy,
    DynamicPolicyConfig,
)


def _vector(
    size: int,
    phase: float,
) -> tuple[float, ...]:
    return tuple(
        math.sin(
            (index + 1) * 0.137 + phase
        )
        + 0.35
        * math.cos(
            (index + 1) * 0.071 - phase
        )
        for index in range(size)
    )


def _state(
    transition_id: str,
    *,
    phase: float,
    size: int,
) -> IntegratedCognitiveState:
    vector = _vector(
        size,
        phase,
    )
    return IntegratedCognitiveState(
        transition_id=transition_id,
        world_id="market",
        vector=vector,
        workspace_vector=_vector(
            size,
            phase + 0.2,
        ),
        temporal_vector=_vector(
            size,
            phase + 0.4,
        ),
        context_vector=_vector(
            size,
            phase + 0.6,
        ),
        novelty=0.4,
        state_change=0.2,
        metadata={"smoke": True},
    )


def _parameter_snapshot(
    policy: DynamicAffordancePolicy,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }


def _changed_parameters(
    before: dict[str, torch.Tensor],
    policy: DynamicAffordancePolicy,
) -> list[str]:
    changed: list[str] = []
    for name, parameter in policy.named_parameters():
        if name not in before:
            continue
        if not torch.equal(
            before[name],
            parameter.detach().cpu(),
        ):
            changed.append(name)
    return changed


def main() -> int:
    print("DYNAMIC POLICY SMOKE")
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

    config = DynamicPolicyConfig(
        state_size=96,
        affordance_feature_size=64,
        hidden_size=96,
        learning_rate=1e-3,
        max_traces=32,
    )
    policy = DynamicAffordancePolicy(
        config
    )

    print(f"policy_device: {policy.device}")

    now = 1000.0

    market_affordances = (
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

    first_state = _state(
        "policy-smoke-1",
        phase=0.15,
        size=config.state_size,
    )

    before = _parameter_snapshot(
        policy
    )

    intent = policy.choose(
        first_state,
        market_affordances,
        now=now,
    )

    if intent.transition_id != first_state.transition_id:
        raise AssertionError(
            "Policy lost transition_id"
        )

    candidate_ids = {
        item.id
        for item in market_affordances
    }
    if intent.affordance_id not in candidate_ids:
        raise AssertionError(
            "Policy chose an Affordance that was not offered"
        )

    chosen = next(
        item
        for item in market_affordances
        if item.id == intent.affordance_id
    )

    if intent.world != chosen.world:
        raise AssertionError(
            "ActionIntent world does not match chosen Affordance"
        )
    if intent.action != chosen.action:
        raise AssertionError(
            "ActionIntent action does not match chosen Affordance"
        )
    if intent.parameters:
        raise AssertionError(
            "Initial Policy must not fabricate action parameters"
        )

    print("decision: OK")
    print(f"chosen_action: {intent.action}")
    print(f"affordance_count: {len(market_affordances)}")

    # An unconsumed causal transition may not be silently overwritten.
    duplicate_rejected = False
    try:
        policy.choose(
            first_state,
            market_affordances,
            now=now,
        )
    except ValueError:
        duplicate_rejected = True

    if not duplicate_rejected:
        raise AssertionError(
            "duplicate transition_id was not rejected"
        )

    print("duplicate_transition_guard: OK")

    # Causal learning changes the actual trainable Policy and consumes the
    # exact decision trace once.
    learned = policy.learn(
        first_state.transition_id,
        advantage=1.0,
    )
    if not learned:
        raise AssertionError(
            "known policy trace did not learn"
        )

    changed = _changed_parameters(
        before,
        policy,
    )
    if not changed:
        raise AssertionError(
            "Policy learning changed no trainable parameter"
        )

    if policy.learn(
        first_state.transition_id,
        advantage=1.0,
    ):
        raise AssertionError(
            "consumed Policy trace was reusable"
        )

    print("causal_learning: OK")
    print(
        "changed_parameter_tensors:",
        len(changed),
    )
    print("causal_trace_consumed_once: OK")

    # The Policy must accept an action vocabulary that did not exist when the
    # class was written. No central enum is edited.
    second_state = _state(
        "policy-smoke-2",
        phase=0.77,
        size=config.state_size,
    )

    dynamic_affordances = (
        Affordance(
            id="future:inspect",
            world="future:world",
            action="future.inspect_unknown_object",
            offered_at=now + 1.0,
            payload={
                "target": "object-17",
                "novel_field": "anything",
            },
            constraints={
                "distance": 2.5,
            },
            expires_at=now + 10.0,
        ),
        Affordance(
            id="future:expired",
            world="future:world",
            action="future.expired_action",
            offered_at=now - 5.0,
            expires_at=now,
        ),
        Affordance(
            id="desktop:open",
            world="desktop",
            action="desktop.open",
            offered_at=now + 1.0,
            payload={
                "target": "notes.txt",
            },
            constraints={},
            expires_at=None,
        ),
    )

    second_intent = policy.choose(
        second_state,
        dynamic_affordances,
        now=now + 1.0,
        remember_for_learning=False,
    )

    if (
        second_intent.affordance_id
        == "future:expired"
    ):
        raise AssertionError(
            "expired Affordance was selectable"
        )

    if second_intent.affordance_id not in {
        "future:inspect",
        "desktop:open",
    }:
        raise AssertionError(
            "Policy failed on dynamic action vocabulary"
        )

    print("dynamic_affordances: OK")
    print("expired_affordance_filter: OK")
    print(
        "dynamic_chosen_action:",
        second_intent.action,
    )

    telemetry = policy.telemetry()

    if telemetry.learning_updates != 1:
        raise AssertionError(
            "unexpected Policy learning update count"
        )
    if telemetry.traces_consumed != 1:
        raise AssertionError(
            "unexpected Policy consumed trace count"
        )
    if telemetry.duplicate_transition_rejections != 1:
        raise AssertionError(
            "unexpected duplicate rejection count"
        )

    print(
        "telemetry:",
        f"decisions={telemetry.decision_calls}",
        f"learning_updates={telemetry.learning_updates}",
        f"traces_created={telemetry.traces_created}",
        f"traces_consumed={telemetry.traces_consumed}",
        f"score_margin={telemetry.last_score_margin:.6f}",
        f"gradient_norm={telemetry.last_gradient_norm:.6f}",
    )

    # A generation-bound causal snapshot must restore the exact replay trace,
    # optimizer and counters needed for delayed continuation.
    snapshot_state = _state(
        "policy-smoke-snapshot",
        phase=1.21,
        size=config.state_size,
    )
    policy.choose(
        snapshot_state,
        market_affordances,
        now=now + 2.0,
        remember_for_learning=True,
    )
    trace_ids = (snapshot_state.transition_id,)
    mismatch_rejected = False
    try:
        policy.export_causal_snapshot("policy-generation", ("wrong-trace",))
    except ValueError:
        mismatch_rejected = True
    if not mismatch_rejected:
        raise AssertionError("Policy snapshot export accepted mismatched trace IDs")
    snapshot = policy.export_causal_snapshot("policy-generation", trace_ids)
    restored = DynamicAffordancePolicy(config)
    restored.restore_causal_snapshot(
        snapshot,
        "policy-generation",
        trace_ids,
    )
    if not restored.has_trace(snapshot_state.transition_id):
        raise AssertionError("Policy causal trace was not restored")
    if policy.telemetry() != restored.telemetry():
        raise AssertionError("Policy causal telemetry was not restored exactly")
    if _changed_parameters(_parameter_snapshot(policy), restored):
        raise AssertionError("Policy causal model was not restored exactly")

    before_rejected_restore = _parameter_snapshot(restored)
    before_rejected_telemetry = restored.telemetry()
    wrong_generation_rejected = False
    try:
        restored.restore_causal_snapshot(snapshot, "wrong-generation", trace_ids)
    except ValueError:
        wrong_generation_rejected = True
    if not wrong_generation_rejected:
        raise AssertionError("Policy snapshot accepted wrong generation")
    if _changed_parameters(before_rejected_restore, restored):
        raise AssertionError("rejected Policy restore mutated model")
    if restored.telemetry() != before_rejected_telemetry:
        raise AssertionError("rejected Policy restore mutated telemetry")
    if not restored.has_trace(snapshot_state.transition_id):
        raise AssertionError("rejected Policy restore mutated traces")

    if not policy.learn(snapshot_state.transition_id, advantage=-0.75):
        raise AssertionError("original Policy snapshot trace did not learn")
    if not restored.learn(snapshot_state.transition_id, advantage=-0.75):
        raise AssertionError("restored Policy snapshot trace did not learn")
    if _changed_parameters(_parameter_snapshot(policy), restored):
        raise AssertionError("restored Policy continuation diverged")
    if policy.telemetry() != restored.telemetry():
        raise AssertionError("restored Policy counters diverged")
    print("causal_snapshot_exact_continuation: OK")
    print("causal_snapshot_atomic_rejection: OK")

    print("DYNAMIC POLICY SMOKE: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
