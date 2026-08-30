from __future__ import annotations

import argparse
import math
import time

import torch

from core.interface import CoreInput
from core.personal import PersonalCognitiveCore, PersonalCoreConfig


def _vector(size: int, phase: float) -> tuple[float, ...]:
    return tuple(math.sin((i + 1) * 0.113 + phase) for i in range(size))


def _input(tid: str, phase: float, cfg: PersonalCoreConfig) -> CoreInput:
    return CoreInput(
        transition_id=tid,
        world_id="benchmark",
        representation=_vector(cfg.input_size, phase),
        memory_context=_vector(cfg.context_size, phase + 0.2),
        self_context=_vector(cfg.context_size, phase + 0.4),
        relation_context=_vector(cfg.context_size, phase + 0.6),
    )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run(profile: str, steps: int) -> None:
    if profile == "local":
        cfg = PersonalCoreConfig.rx6800()
    elif profile == "heavy":
        cfg = PersonalCoreConfig.rx6800_heavy()
    else:
        raise ValueError(profile)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    core = PersonalCognitiveCore(cfg)

    for index in range(8):
        core.integrate(_input(f"warm-{profile}-{index}", index * 0.1, cfg), remember_for_learning=False)
    _sync()

    forward_times: list[float] = []
    backward_times: list[float] = []

    for index in range(steps):
        tid = f"bench-{profile}-{index}"
        _sync()
        start = time.perf_counter()
        core.integrate(_input(tid, 1.0 + index * 0.07, cfg), remember_for_learning=True)
        _sync()
        forward_times.append((time.perf_counter() - start) * 1000.0)

        gradient = tuple(0.005 * math.cos(j * 0.17 + index) for j in range(cfg.output_size))
        _sync()
        start = time.perf_counter()
        if not core.learn(tid, gradient):
            raise RuntimeError("missing benchmark learning trace")
        _sync()
        backward_times.append((time.perf_counter() - start) * 1000.0)

    print(f"PROFILE: {profile}")
    print(f"parameters: {core.parameter_count}")
    print(f"hidden_size: {cfg.hidden_size}")
    print(f"router_top_k: {cfg.router_top_k}/4")
    print(f"history_size: {cfg.history_size}")
    print(f"mean_forward_ms: {sum(forward_times) / len(forward_times):.3f}")
    print(f"mean_backward_update_ms: {sum(backward_times) / len(backward_times):.3f}")
    print(f"peak_memory_allocated_mib: {torch.cuda.max_memory_allocated() / 1024**2:.1f}")
    print(f"memory_reserved_mib: {torch.cuda.memory_reserved() / 1024**2:.1f}")
    print("last_active_experts:", ", ".join(core.telemetry().last_active_experts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("local", "heavy", "both"), default="local")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be > 0")

    print("HEIWA SCALABLE CORE GPU BENCHMARK")
    print("torch:", torch.__version__)
    print("hip:", getattr(torch.version, "hip", None))
    print("accelerator_available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("BENCHMARK FAILED: accelerator unavailable")
        return 2
    print("accelerator:", torch.cuda.get_device_name(0))

    profiles = ("local", "heavy") if args.profile == "both" else (args.profile,)
    for profile in profiles:
        _run(profile, args.steps)

    print("HEIWA SCALABLE CORE GPU BENCHMARK: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
