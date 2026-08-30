from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import dataclass

import torch

from core.interface import CoreInput
from core.personal import PersonalCognitiveCore, PersonalCoreConfig


@dataclass(frozen=True, slots=True)
class _CapacityProfile:
    name: str
    hidden_size: int
    heads: int
    ssm_depth: int
    recurrent_depth: int
    tcn_depth: int
    attention_depth: int
    history_size: int


# Every profile routes all four temporal experts.  Width/depth grows while the
# public Core contract stays identical.  The last profile is intentionally near
# the useful training-memory region of a 16 GiB card; the sweep stops early if
# the measured peak reaches the requested safety fraction.
_PROFILES = (
    _CapacityProfile("full4_131m", 768, 12, 4, 4, 5, 4, 192),
    _CapacityProfile("full4_233m", 1024, 16, 4, 4, 5, 4, 256),
    _CapacityProfile("full4_415m", 1280, 16, 5, 5, 6, 5, 256),
    _CapacityProfile("full4_597m", 1536, 16, 5, 5, 6, 5, 256),
    _CapacityProfile("full4_756m", 1728, 16, 5, 5, 6, 5, 256),
)


def _vector(size: int, phase: float) -> tuple[float, ...]:
    return tuple(
        math.sin((index + 1) * 0.113 + phase) * 0.7
        + math.cos((index + 1) * 0.071 - phase) * 0.3
        for index in range(size)
    )


def _input(
    transition_id: str,
    phase: float,
    cfg: PersonalCoreConfig,
) -> CoreInput:
    return CoreInput(
        transition_id=transition_id,
        world_id="capacity-benchmark",
        representation=_vector(cfg.input_size, phase),
        memory_context=_vector(cfg.context_size, phase + 0.31),
        self_context=_vector(cfg.context_size, phase + 0.67),
        relation_context=_vector(cfg.context_size, phase + 1.03),
        metadata={"capacity_benchmark": True},
    )


def _sync() -> None:
    torch.cuda.synchronize()


def _mib(value: int) -> float:
    return float(value) / 1024.0**2


def _gib(value: int) -> float:
    return float(value) / 1024.0**3


def _config(profile: _CapacityProfile) -> PersonalCoreConfig:
    return PersonalCoreConfig(
        input_size=96,
        context_size=96,
        hidden_size=profile.hidden_size,
        output_size=96,
        workspace_heads=profile.heads,
        attention_heads=profile.heads,
        router_hidden_size=profile.hidden_size,
        router_top_k=4,
        ssm_depth=profile.ssm_depth,
        recurrent_depth=profile.recurrent_depth,
        tcn_depth=profile.tcn_depth,
        tcn_kernel_size=3,
        attention_depth=profile.attention_depth,
        attention_feedforward_multiplier=4,
        history_size=profile.history_size,
        dropout=0.05,
        learning_rate=3e-4,
        weight_decay=1e-2,
        gradient_clip=5.0,
        # Capacity probing needs only one live causal trace at a time.  Keeping
        # this small also avoids confusing CPU trace storage with GPU capacity.
        max_traces=4,
        device="cuda",
        dtype=torch.float32,
    )


def _is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


def _cleanup(core: PersonalCognitiveCore | None) -> None:
    if core is not None:
        try:
            core.optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
    del core
    gc.collect()
    torch.cuda.empty_cache()
    _sync()


def _run_profile(
    profile: _CapacityProfile,
    *,
    steps: int,
    context_fill: int,
    total_memory: int,
) -> tuple[bool, float]:
    cfg = _config(profile)
    core: PersonalCognitiveCore | None = None

    print()
    print(f"PROFILE: {profile.name}")
    print(f"hidden_size: {cfg.hidden_size}")
    print(f"router_top_k: {cfg.router_top_k}/4")
    print(f"history_size: {cfg.history_size}")
    print(f"context_fill_steps: {context_fill}")

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        core = PersonalCognitiveCore(cfg)
        _sync()

        model_allocated = torch.cuda.memory_allocated()
        print(f"parameters: {core.parameter_count}")
        print(f"model_allocated_mib: {_mib(model_allocated):.1f}")

        # Build a non-empty rolling temporal context through the public Core API.
        # The complete long-term/autobiographical context is intentionally NOT a
        # giant GPU window: that belongs to persistent state + MemoryContext.
        fill_steps = min(context_fill, cfg.history_size)
        for index in range(fill_steps):
            core.integrate(
                _input(
                    f"capacity-fill-{profile.name}-{index}",
                    0.03 * index,
                    cfg,
                ),
                remember_for_learning=False,
            )
        _sync()

        forward_times: list[float] = []
        backward_times: list[float] = []

        for index in range(steps):
            transition_id = f"capacity-{profile.name}-{index}"

            _sync()
            started = time.perf_counter()
            state = core.integrate(
                _input(transition_id, 1.0 + 0.07 * index, cfg),
                remember_for_learning=True,
            )
            _sync()
            forward_times.append((time.perf_counter() - started) * 1000.0)

            active = tuple(state.metadata.get("core_active_experts", ()))
            if set(active) != {"ssm", "recurrent", "tcn", "attention"}:
                raise RuntimeError(
                    "capacity profile did not execute all four experts: "
                    f"{active!r}"
                )

            gradient = tuple(
                0.005 * math.cos(element * 0.17 + index)
                for element in range(cfg.output_size)
            )

            _sync()
            started = time.perf_counter()
            if not core.learn(transition_id, gradient):
                raise RuntimeError("missing capacity benchmark causal trace")
            _sync()
            backward_times.append((time.perf_counter() - started) * 1000.0)

        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()
        fraction = peak / max(1, total_memory)

        print(
            "mean_forward_ms: "
            f"{sum(forward_times) / len(forward_times):.3f}"
        )
        print(
            "mean_backward_update_ms: "
            f"{sum(backward_times) / len(backward_times):.3f}"
        )
        print(f"peak_memory_allocated_mib: {_mib(peak):.1f}")
        print(f"memory_reserved_mib: {_mib(reserved):.1f}")
        print(f"vram_peak_fraction: {fraction:.4f}")
        print("active_experts: ssm, recurrent, tcn, attention")
        print("STATUS: OK")
        return True, fraction

    except torch.OutOfMemoryError as exc:
        print(f"STATUS: OOM ({type(exc).__name__})")
        return False, 1.0
    except RuntimeError as exc:
        if _is_oom(exc):
            print(f"STATUS: OOM ({type(exc).__name__})")
            return False, 1.0
        raise
    finally:
        _cleanup(core)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find a safe all-four-expert training capacity for the scalable "
            "Heiwa Core on the current GPU."
        )
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--context-fill",
        type=int,
        default=16,
        help=(
            "Number of real public-API Core steps used to populate recent "
            "temporal history before each measured training step."
        ),
    )
    parser.add_argument(
        "--target-vram-fraction",
        type=float,
        default=0.75,
        help=(
            "Stop after a successful profile reaches this fraction of total "
            "VRAM. The default intentionally leaves room for Worlds, Language, "
            "ROCm workspace and future context machinery."
        ),
    )
    parser.add_argument(
        "--max-profile",
        choices=tuple(profile.name for profile in _PROFILES),
        default=_PROFILES[-1].name,
    )
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be > 0")
    if args.context_fill < 0:
        raise SystemExit("--context-fill must be >= 0")
    if not (0.10 <= args.target_vram_fraction <= 0.95):
        raise SystemExit("--target-vram-fraction must be between 0.10 and 0.95")

    print("HEIWA CORE FULL-4 CAPACITY BENCHMARK")
    print("torch:", torch.__version__)
    print("hip:", getattr(torch.version, "hip", None))
    print("accelerator_available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("CAPACITY BENCHMARK FAILED: accelerator unavailable")
        return 2

    print("accelerator:", torch.cuda.get_device_name(0))
    properties = torch.cuda.get_device_properties(0)
    total_memory = int(properties.total_memory)
    print(f"total_vram_gib: {_gib(total_memory):.2f}")
    print(f"target_vram_fraction: {args.target_vram_fraction:.2f}")
    print(
        "NOTE: target is deliberately below 100%; Heiwa still needs headroom "
        "for Worlds, future Language, ROCm workspaces and transient context."
    )

    last_success: str | None = None
    last_fraction = 0.0

    for profile in _PROFILES:
        success, fraction = _run_profile(
            profile,
            steps=args.steps,
            context_fill=args.context_fill,
            total_memory=total_memory,
        )

        if not success:
            print()
            print("SWEEP_STOP: first OOM")
            break

        last_success = profile.name
        last_fraction = fraction

        if fraction >= args.target_vram_fraction:
            print()
            print("SWEEP_STOP: target VRAM fraction reached")
            break

        if profile.name == args.max_profile:
            print()
            print("SWEEP_STOP: max profile reached")
            break

    print()
    print("LAST_SUCCESS:", last_success or "none")
    print(f"LAST_SUCCESS_VRAM_FRACTION: {last_fraction:.4f}")
    print(
        "CONTEXT_MODEL: recent temporal history is bounded; WORK/SLOW/SSM/"
        "recurrent states compress ongoing context; autobiographical detail "
        "belongs to external MemoryContext rather than one giant GPU window."
    )
    print("HEIWA CORE FULL-4 CAPACITY BENCHMARK: DONE")
    return 0 if last_success is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
