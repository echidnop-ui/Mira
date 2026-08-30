from __future__ import annotations

import hashlib
import io
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory

import torch

from contracts.cognition import (
    ActionAttempt,
    Affordance,
    CompletedAction,
    Consequence,
    Observation,
)
from core.personal import PersonalCognitiveCore
from policy.dynamic import DynamicAffordancePolicy, DynamicPolicyConfig
from runtime.causal import (
    CausalCheckpointError,
    CausalCognitionCycle,
    CausalCycleError,
)
from runtime.causal_smoke import _ControlledInputBuilder, _RewardInterpreter, _core
from worlds.interface import WorldFrame


class _CompletionBeforeShutdownWorld:
    world_id = "controlled"

    def __init__(self) -> None:
        self._poll_number = 0
        self.completed: CompletedAction | None = None
        self.consequence: Consequence | None = None

    def poll(self, now: float) -> WorldFrame:
        poll_number = self._poll_number
        self._poll_number += 1
        if poll_number == 0:
            return WorldFrame(
                world=self.world_id,
                observed_at=now,
                observations=(
                    Observation(
                        "observation:restart",
                        self.world_id,
                        "controlled.cue",
                        now,
                        {"signal": 1.0},
                    ),
                ),
                affordances=(
                    Affordance("left", self.world_id, "choose.left", now),
                    Affordance("right", self.world_id, "choose.right", now),
                ),
            )
        if poll_number == 1:
            assert self.completed is not None
            return WorldFrame(
                world=self.world_id,
                observed_at=now,
                completed_actions=(self.completed,),
            )
        return WorldFrame(world=self.world_id, observed_at=now)

    def execute(self, intent, now: float) -> ActionAttempt:
        attempt = ActionAttempt(
            "attempt:restart",
            intent.id,
            intent.transition_id,
            self.world_id,
            intent.action,
            now,
            True,
        )
        self.completed = CompletedAction(
            "completed:restart",
            attempt.id,
            intent.id,
            intent.transition_id,
            self.world_id,
            intent.action,
            now + 0.01,
            "completed",
            True,
        )
        self.consequence = Consequence(
            "consequence:restart",
            intent.transition_id,
            self.world_id,
            "controlled.reward",
            now + 0.02,
            self.completed.id,
            {"reward": 1.0},
            intent.id,
            attempt.id,
        )
        return attempt

    def close(self) -> None:
        return None


class _DelayedConsequenceWorld:
    world_id = "controlled"

    def __init__(self, consequence: Consequence) -> None:
        self._consequence = consequence

    def poll(self, now: float) -> WorldFrame:
        # Repeated delivery on a later poll simulates an external retry.
        return WorldFrame(
            world=self.world_id,
            observed_at=now,
            consequences=(self._consequence,),
        )

    def execute(self, intent, now: float) -> ActionAttempt:
        raise AssertionError("outcome-only World must not execute")

    def close(self) -> None:
        return None


def _policy() -> DynamicAffordancePolicy:
    return DynamicAffordancePolicy(
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


def _cycle(
    world,
    core: PersonalCognitiveCore,
    policy: DynamicAffordancePolicy,
    *,
    prefix: str = "stable",
) -> CausalCognitionCycle:
    return CausalCognitionCycle(
        world=world,
        core=core,
        policy=policy,
        input_builder=_ControlledInputBuilder(),
        consequence_interpreter=_RewardInterpreter(),
        transition_prefix=prefix,
    )


def _read_payload(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    assert isinstance(payload, dict)
    return payload


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _tamper_inner_policy_version(
    payload: dict[str, object],
) -> dict[str, object]:
    result = dict(payload)
    raw_blob = result["policy_snapshot"]
    assert isinstance(raw_blob, bytes)
    inner = torch.load(io.BytesIO(raw_blob), map_location="cpu", weights_only=False)
    assert isinstance(inner, dict)
    inner["version"] = 999
    stream = io.BytesIO()
    torch.save(inner, stream)
    tampered_blob = stream.getvalue()
    result["policy_snapshot"] = tampered_blob
    result["policy_sha256"] = hashlib.sha256(tampered_blob).hexdigest()
    return result


def _assert_poisoned(cycle: CausalCognitionCycle, path: Path) -> None:
    try:
        cycle.step(now=99.0)
    except CausalCheckpointError:
        pass
    else:
        raise AssertionError("failed restore cycle accepted step")
    try:
        cycle.save_checkpoint(path)
    except CausalCheckpointError:
        pass
    else:
        raise AssertionError("failed restore cycle accepted save")


def main() -> int:
    torch.manual_seed(41)
    initial_world = _CompletionBeforeShutdownWorld()
    initial_core = _core()
    initial_policy = _policy()
    initial_cycle = _cycle(initial_world, initial_core, initial_policy)
    started = initial_cycle.step(now=0.0).started
    assert started is not None and started.transition_id == "stable:1"
    completed_step = initial_cycle.step(now=1.0)
    assert not completed_step.learning
    assert initial_cycle.pending_count == 1
    assert started.completed_action is initial_world.completed
    assert initial_world.consequence is not None

    with TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "causal.snapshot"
        generation = initial_cycle.save_checkpoint(checkpoint)

        restored_core = _core()
        restored_policy = _policy()
        restored_cycle = _cycle(
            _DelayedConsequenceWorld(initial_world.consequence),
            restored_core,
            restored_policy,
        )
        assert restored_cycle.restore_checkpoint(checkpoint) == generation
        assert restored_cycle.pending_count == 1
        learned = restored_cycle.step(now=2.0).learning
        assert len(learned) == 1
        assert learned[0].transition_id == "stable:1"
        assert learned[0].policy_updated and learned[0].core_updated
        assert learned[0].skipped_reason is None
        assert restored_cycle.pending_count == 0
        assert restored_core.telemetry().learning_updates == 1
        assert restored_policy.telemetry().learning_updates == 1

        # External retry after terminal resolution is rejected before learning.
        try:
            restored_cycle.step(now=3.0)
        except CausalCycleError:
            pass
        else:
            raise AssertionError("duplicate consequence was accepted")
        assert restored_core.telemetry().learning_updates == 1
        assert restored_policy.telemetry().learning_updates == 1

        payload = _read_payload(checkpoint)
        corrupt_payload = dict(payload)
        core_blob = bytearray(corrupt_payload["core_snapshot"])
        core_blob[len(core_blob) // 2] ^= 1
        corrupt_payload["core_snapshot"] = bytes(core_blob)
        corrupt_path = root / "corrupt.snapshot"
        _write_payload(corrupt_path, corrupt_payload)
        corrupt_cycle = _cycle(
            _DelayedConsequenceWorld(initial_world.consequence),
            _core(),
            _policy(),
        )
        try:
            corrupt_cycle.restore_checkpoint(corrupt_path)
        except CausalCheckpointError:
            pass
        else:
            raise AssertionError("corrupt causal blob was accepted")
        _assert_poisoned(corrupt_cycle, root / "forbidden-corrupt.snapshot")

        prefix_cycle = _cycle(
            _DelayedConsequenceWorld(initial_world.consequence),
            _core(),
            _policy(),
            prefix="other",
        )
        try:
            prefix_cycle.restore_checkpoint(checkpoint)
        except CausalCheckpointError:
            pass
        else:
            raise AssertionError("transition prefix mismatch was accepted")
        _assert_poisoned(prefix_cycle, root / "forbidden-prefix.snapshot")

        generation_payload = dict(payload)
        generation_payload["generation_id"] = "causal-generation:mismatch"
        mismatch_path = root / "generation-mismatch.snapshot"
        _write_payload(mismatch_path, generation_payload)
        mismatch_core = _core()
        mismatch_policy = _policy()
        mismatch_cycle = _cycle(
            _DelayedConsequenceWorld(initial_world.consequence),
            mismatch_core,
            mismatch_policy,
        )
        try:
            mismatch_cycle.restore_checkpoint(mismatch_path)
        except CausalCheckpointError:
            pass
        else:
            raise AssertionError("organ generation mismatch was accepted")
        assert mismatch_core.telemetry().learning_updates == 0
        assert mismatch_policy.telemetry().learning_updates == 0
        _assert_poisoned(mismatch_cycle, root / "forbidden-generation.snapshot")

        # Outer envelope and checksum are valid, so Core accepts first.  A bad
        # inner Policy version must roll Core back exactly before failing closed.
        inner_policy_path = root / "inner-policy-mismatch.snapshot"
        _write_payload(inner_policy_path, _tamper_inner_policy_version(payload))
        rollback_core = _core()
        rollback_policy = _policy()
        core_before = {
            name: value.detach().clone()
            for name, value in rollback_core.state_dict().items()
        }
        policy_before = {
            name: value.detach().clone()
            for name, value in rollback_policy.state_dict().items()
        }
        core_telemetry_before = rollback_core.telemetry()
        policy_telemetry_before = rollback_policy.telemetry()
        rollback_cycle = _cycle(
            _DelayedConsequenceWorld(initial_world.consequence),
            rollback_core,
            rollback_policy,
        )
        try:
            rollback_cycle.restore_checkpoint(inner_policy_path)
        except CausalCheckpointError:
            pass
        else:
            raise AssertionError("invalid inner Policy snapshot was accepted")
        assert all(
            torch.equal(value, core_before[name])
            for name, value in rollback_core.state_dict().items()
        )
        assert all(
            torch.equal(value, policy_before[name])
            for name, value in rollback_policy.state_dict().items()
        )
        assert rollback_core.telemetry() == core_telemetry_before
        assert rollback_policy.telemetry() == policy_telemetry_before
        assert not rollback_core.has_trace("stable:1")
        assert not rollback_policy.has_trace("stable:1")
        _assert_poisoned(rollback_cycle, root / "forbidden-rollback.snapshot")

    print(
        {
            "transition": "stable:1",
            "completed_before_shutdown": True,
            "generation_restored": generation,
            "core_updates_after_restore": 1,
            "policy_updates_after_restore": 1,
            "duplicate_updates": 0,
            "corrupt_blob_failed_closed": True,
            "prefix_mismatch_failed_closed": True,
            "generation_mismatch_failed_closed": True,
            "inner_policy_rejection_rolled_back_both": True,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
