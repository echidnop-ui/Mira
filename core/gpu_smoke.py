from __future__ import annotations

import math
import time

import torch

from core.interface import CoreInput
from core.personal import PersonalCognitiveCore, PersonalCoreConfig


def _vector(size: int, phase: float) -> tuple[float, ...]:
    return tuple(
        math.sin((index + 1) * 0.173 + phase) * 0.7
        + math.cos((index + 1) * 0.071 - phase) * 0.3
        for index in range(size)
    )


def _input(transition_id: str, phase: float, cfg: PersonalCoreConfig) -> CoreInput:
    return CoreInput(
        transition_id=transition_id,
        world_id="desktop",
        representation=_vector(cfg.input_size, phase),
        memory_context=_vector(cfg.context_size, phase + 0.31),
        self_context=_vector(cfg.context_size, phase + 0.67),
        relation_context=_vector(cfg.context_size, phase + 1.03),
        metadata={"gpu_smoke": True},
    )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    print("HEIWA SCALABLE CORE GPU SMOKE")
    print("torch:", torch.__version__)
    print("hip:", getattr(torch.version, "hip", None))
    print("accelerator_available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("GPU SMOKE FAILED: torch.cuda.is_available() is False")
        return 2

    print("accelerator:", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = PersonalCoreConfig.rx6800()
    core = PersonalCognitiveCore(cfg)
    print("core_device:", core.device)
    print("parameters:", core.parameter_count)
    print("router_top_k:", cfg.router_top_k)
    print("expert_count: 4")
    print("history_size:", cfg.history_size)

    # Prime history/state without retaining training traces.
    for index in range(8):
        core.integrate(
            _input(f"gpu-prime-{index}", 0.1 + index * 0.13, cfg),
            remember_for_learning=False,
        )

    _sync()
    started = time.perf_counter()
    state = core.integrate(_input("gpu-forward", 1.7, cfg), remember_for_learning=True)
    _sync()
    forward_ms = (time.perf_counter() - started) * 1000.0

    active = tuple(state.metadata.get("core_active_experts", ()))
    weights = state.metadata.get("core_router_weights", {})
    print("forward: OK")
    print("active_experts:", ", ".join(active))
    print("router_weights:", weights)
    print(f"forward_ms: {forward_ms:.3f}")

    gradient = tuple(
        0.01 * math.sin((index + 1) * 0.19)
        for index in range(cfg.output_size)
    )

    _sync()
    started = time.perf_counter()
    learned = core.learn("gpu-forward", gradient)
    _sync()
    backward_ms = (time.perf_counter() - started) * 1000.0

    if not learned:
        print("GPU SMOKE FAILED: causal learning trace missing")
        return 3

    if core.learn("gpu-forward", gradient):
        print("GPU SMOKE FAILED: causal trace reused")
        return 4

    print("backward: OK")
    print("causal_trace_consumed_once: OK")
    print(f"backward_update_ms: {backward_ms:.3f}")

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    print(f"memory_allocated_mib: {allocated / 1024**2:.1f}")
    print(f"memory_reserved_mib: {reserved / 1024**2:.1f}")
    print(f"peak_memory_allocated_mib: {peak / 1024**2:.1f}")

    telemetry = core.telemetry()
    print("learning_updates:", telemetry.learning_updates)
    print("last_active_experts:", ", ".join(telemetry.last_active_experts))
    print(
        "CHAIN: CoreInput -> GPU input/context -> learned sparse Router -> "
        "heterogeneous temporal experts -> fusion -> WORK/SLOW/workspace -> "
        "IntegratedCognitiveState -> exact transition replay -> GPU backward"
    )
    print("HEIWA SCALABLE CORE GPU SMOKE: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
