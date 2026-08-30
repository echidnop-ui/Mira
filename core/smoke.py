from __future__ import annotations

import math
import sys
from dataclasses import asdict

import torch

from core.personal import CoreInput, PersonalCognitiveCore, PersonalCoreConfig


def _vector(size: int, phase: float) -> tuple[float, ...]:
    return tuple(
        math.sin((index + 1) * 0.173 + phase) * 0.7
        + math.cos((index + 1) * 0.071 - phase) * 0.3
        for index in range(size)
    )


def _rms_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise AssertionError(
            f"vector size mismatch: {len(left)} != {len(right)}"
        )
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left, right, strict=True))
        / max(1, len(left))
    )


def _parameter_snapshot(
    core: PersonalCognitiveCore,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in core.named_parameters()
        if parameter.requires_grad
    }


def _changed_parameters(
    before: dict[str, torch.Tensor],
    core: PersonalCognitiveCore,
) -> list[str]:
    changed: list[str] = []
    for name, parameter in core.named_parameters():
        if name not in before:
            continue
        current = parameter.detach().cpu()
        if not torch.equal(before[name], current):
            changed.append(name)
    return changed


def _assert_finite(name: str, values: tuple[float, ...]) -> None:
    if not values:
        raise AssertionError(f"{name} is empty")
    if not all(math.isfinite(value) for value in values):
        raise AssertionError(f"{name} contains non-finite values")


def _make_input(
    transition_id: str,
    *,
    phase: float,
    size: int,
    with_context: bool = True,
) -> CoreInput:
    if with_context:
        memory = _vector(size, phase + 0.31)
        self_context = _vector(size, phase + 0.67)
        relation = _vector(size, phase + 1.03)
    else:
        memory = ()
        self_context = ()
        relation = ()

    return CoreInput(
        transition_id=transition_id,
        world_id="market",
        representation=_vector(size, phase),
        memory_context=memory,
        self_context=self_context,
        relation_context=relation,
        metadata={"smoke": True, "phase": phase},
    )


def main() -> int:
    print("PERSONAL CORE SMOKE")
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"hip: {getattr(torch.version, 'hip', None)}")
    print(f"accelerator_available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"accelerator: {torch.cuda.get_device_name(0)}")
    else:
        print("accelerator: CPU")

    config = PersonalCoreConfig(
        input_size=96,
        context_size=96,
        hidden_size=96,
        output_size=96,
        workspace_heads=4,
        dropout=0.0,
        learning_rate=1e-3,
        max_traces=32,
        device=None,
    )
    core = PersonalCognitiveCore(config)

    print(f"core_device: {core.device}")
    print(f"representation_size: {core.representation_size}")

    # 1. Forward integration with all contexts present.
    first_input = _make_input(
        "core-smoke-forward-1",
        phase=0.10,
        size=config.input_size,
        with_context=True,
    )
    first = core.integrate(first_input)

    if first.transition_id != first_input.transition_id:
        raise AssertionError("transition_id was not preserved")
    if first.world_id != "market":
        raise AssertionError("world_id was not preserved")
    if len(first.vector) != config.output_size:
        raise AssertionError("unexpected integrated vector size")
    if len(first.workspace_vector) != config.hidden_size:
        raise AssertionError("unexpected workspace vector size")
    if len(first.temporal_vector) != config.hidden_size:
        raise AssertionError("unexpected temporal vector size")
    if len(first.context_vector) != config.hidden_size:
        raise AssertionError("unexpected context vector size")

    _assert_finite("vector", first.vector)
    _assert_finite("workspace_vector", first.workspace_vector)
    _assert_finite("temporal_vector", first.temporal_vector)
    _assert_finite("context_vector", first.context_vector)

    if not math.isfinite(first.novelty):
        raise AssertionError("novelty is non-finite")
    if not math.isfinite(first.state_change):
        raise AssertionError("state_change is non-finite")

    print("forward: OK")
    print(
        "forward_metrics: "
        f"novelty={first.novelty:.6f} "
        f"state_change={first.state_change:.6f}"
    )

    # The first trace is not needed for the rest of this smoke.
    if not core.discard_trace(first_input.transition_id):
        raise AssertionError("failed to discard known trace")

    # 2. Missing Memory/Self/Relation context must remain a valid path.
    missing_context = core.integrate(
        _make_input(
            "core-smoke-missing-context",
            phase=0.25,
            size=config.input_size,
            with_context=False,
        ),
        remember_for_learning=False,
    )
    _assert_finite("missing_context.vector", missing_context.vector)
    _assert_finite(
        "missing_context.context_vector",
        missing_context.context_vector,
    )
    print("missing_context: OK")

    # 3. Persistent temporal state must affect the next integration.
    persistence_input = _make_input(
        "core-smoke-persistence",
        phase=0.41,
        size=config.input_size,
        with_context=True,
    )

    core.reset_state(clear_learning_traces=True)
    baseline = core.integrate(
        persistence_input,
        remember_for_learning=False,
    )

    # Prime persistent state with a different event, then replay the same input.
    core.reset_state(clear_learning_traces=True)
    core.integrate(
        _make_input(
            "core-smoke-prime",
            phase=1.20,
            size=config.input_size,
            with_context=True,
        ),
        remember_for_learning=False,
    )
    after_history = core.integrate(
        persistence_input,
        remember_for_learning=False,
    )

    persistence_delta = _rms_distance(
        baseline.vector,
        after_history.vector,
    )
    if persistence_delta <= 1e-8:
        raise AssertionError(
            "persistent state did not measurably affect integrated output"
        )

    print("persistent_state: OK")
    print(f"persistent_state_delta: {persistence_delta:.9f}")

    # 4. Duplicate unconsumed transition IDs must be rejected.
    core.reset_state(clear_learning_traces=True)
    duplicate_input = _make_input(
        "core-smoke-duplicate",
        phase=0.73,
        size=config.input_size,
        with_context=True,
    )
    core.integrate(duplicate_input)

    duplicate_rejected = False
    try:
        core.integrate(duplicate_input)
    except ValueError:
        duplicate_rejected = True

    if not duplicate_rejected:
        raise AssertionError(
            "duplicate unconsumed transition_id was not rejected"
        )

    if not core.discard_trace(duplicate_input.transition_id):
        raise AssertionError("failed to discard duplicate smoke trace")

    print("duplicate_transition_guard: OK")

    # 5. Delayed causal learning must use the exact transition trace,
    # consume it once, and update trainable parameters.
    core.reset_state(clear_learning_traces=True)

    learning_input = _make_input(
        "core-smoke-learning",
        phase=0.94,
        size=config.input_size,
        with_context=True,
    )
    before = _parameter_snapshot(core)

    learning_state = core.integrate(
        learning_input,
        remember_for_learning=True,
    )
    _assert_finite("learning_state.vector", learning_state.vector)

    # This is deliberately only a synthetic smoke gradient.
    # It verifies the causal learning path, not task quality.
    output_gradient = tuple(
        0.01 * math.sin((index + 1) * 0.19)
        for index in range(config.output_size)
    )

    learned = core.learn(
        learning_input.transition_id,
        output_gradient,
    )
    if not learned:
        raise AssertionError("known transition trace did not learn")

    changed = _changed_parameters(before, core)
    if not changed:
        raise AssertionError("learning step changed no parameters")

    learned_again = core.learn(
        learning_input.transition_id,
        output_gradient,
    )
    if learned_again:
        raise AssertionError(
            "consumed transition trace was incorrectly reusable"
        )

    print("causal_learning: OK")
    print(f"changed_parameter_tensors: {len(changed)}")
    print("causal_trace_consumed_once: OK")

    telemetry = core.telemetry()
    telemetry_dict = asdict(telemetry)

    if telemetry.learning_updates != 1:
        raise AssertionError(
            f"unexpected learning_updates={telemetry.learning_updates}"
        )
    if telemetry.traces_consumed != 1:
        raise AssertionError(
            f"unexpected traces_consumed={telemetry.traces_consumed}"
        )
    if telemetry.duplicate_transition_rejections != 1:
        raise AssertionError(
            "duplicate rejection telemetry mismatch: "
            f"{telemetry.duplicate_transition_rejections}"
        )

    print(
        "telemetry: "
        + " ".join(
            f"{name}={value}"
            for name, value in telemetry_dict.items()
        )
    )

    # 6. A generation-bound causal snapshot must restore the private trace,
    # optimizer, persistent state and counters for exact delayed continuation.
    snapshot_input = _make_input(
        "core-smoke-snapshot",
        phase=1.37,
        size=config.input_size,
        with_context=True,
    )
    core.integrate(snapshot_input, remember_for_learning=True)
    trace_ids = (snapshot_input.transition_id,)
    mismatch_rejected = False
    try:
        core.export_causal_snapshot("core-generation", ("wrong-trace",))
    except ValueError:
        mismatch_rejected = True
    if not mismatch_rejected:
        raise AssertionError("Core snapshot export accepted mismatched trace IDs")
    snapshot = core.export_causal_snapshot("core-generation", trace_ids)
    restored = PersonalCognitiveCore(config)
    restored.restore_causal_snapshot(
        snapshot,
        "core-generation",
        trace_ids,
    )
    if not restored.has_trace(snapshot_input.transition_id):
        raise AssertionError("Core causal trace was not restored")
    if core.telemetry() != restored.telemetry():
        raise AssertionError("Core causal telemetry was not restored exactly")
    if _changed_parameters(_parameter_snapshot(core), restored):
        raise AssertionError("Core causal model was not restored exactly")

    before_rejected_restore = _parameter_snapshot(restored)
    before_rejected_telemetry = restored.telemetry()
    wrong_generation_rejected = False
    try:
        restored.restore_causal_snapshot(snapshot, "wrong-generation", trace_ids)
    except ValueError:
        wrong_generation_rejected = True
    if not wrong_generation_rejected:
        raise AssertionError("Core snapshot accepted wrong generation")
    if _changed_parameters(before_rejected_restore, restored):
        raise AssertionError("rejected Core restore mutated model")
    if restored.telemetry() != before_rejected_telemetry:
        raise AssertionError("rejected Core restore mutated telemetry")
    if not restored.has_trace(snapshot_input.transition_id):
        raise AssertionError("rejected Core restore mutated traces")

    if not core.learn(snapshot_input.transition_id, output_gradient):
        raise AssertionError("original Core snapshot trace did not learn")
    if not restored.learn(snapshot_input.transition_id, output_gradient):
        raise AssertionError("restored Core snapshot trace did not learn")
    if _changed_parameters(_parameter_snapshot(core), restored):
        raise AssertionError("restored Core continuation diverged")
    continuation_input = _make_input(
        "core-smoke-snapshot-next",
        phase=1.61,
        size=config.input_size,
        with_context=True,
    )
    original_next = core.integrate(
        continuation_input, remember_for_learning=False
    )
    restored_next = restored.integrate(
        continuation_input, remember_for_learning=False
    )
    if original_next.vector != restored_next.vector:
        raise AssertionError("restored Core persistent continuation diverged")
    print("causal_snapshot_exact_continuation: OK")
    print("causal_snapshot_atomic_rejection: OK")

    print("PERSONAL CORE SMOKE: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
