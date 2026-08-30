from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile

import torch

from language.interface import ExpressionRequest, LanguageInput
from language.organ import NeuralLanguageConfig, NeuralLanguageOrgan


def _clone_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return value


def _state_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _state_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _state(size: int, phase: float) -> tuple[float, ...]:
    return tuple(math.sin((i + 1) * 0.07 + phase) for i in range(size))


def main() -> int:
    print("HEIWA LANGUAGE ORGAN SMOKE")
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"hip: {getattr(torch.version, 'hip', None)}")
    print(f"accelerator_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"accelerator: {torch.cuda.get_device_name(0)}")

    config = NeuralLanguageConfig(
        representation_size=96,
        cognitive_size=96,
        embedding_size=96,
        hidden_size=128,
        recurrent_layers=2,
        attention_heads=4,
        dropout=0.0,
        max_input_bytes=256,
        max_generation_bytes=64,
        learning_rate=1e-3,
        max_traces=16,
    )
    organ = NeuralLanguageOrgan(config)
    print(f"language_device: {organ.device}")
    print(f"parameters: {sum(p.numel() for p in organ.parameters())}")

    incoming = LanguageInput(
        transition_id="lang-smoke-1",
        text="Это кнопка. Нажми её.",
        source_kind="human",
        source_id="teacher",
    )
    representation = organ.encode(incoming)
    assert len(representation.vector) == 96
    assert all(math.isfinite(x) for x in representation.vector)
    print("receptive_forward: OK")

    gradient = tuple(0.01 * math.sin(i * 0.11) for i in range(96))
    assert organ.learn_representation(incoming.transition_id, gradient)
    assert not organ.learn_representation(incoming.transition_id, gradient)
    print("causal_receptive_learning: OK")

    teacher = LanguageInput(
        transition_id="teacher-text-1",
        text="Привет. Это поле ввода.",
        source_kind="human",
    )
    loss = organ.learn_teacher_text(teacher)
    assert math.isfinite(loss)
    print(f"external_teacher_language_learning: OK loss={loss:.6f}")

    rejected = False
    try:
        organ.learn_teacher_text(
            LanguageInput(
                transition_id="self-text-1",
                text="мой собственный текст",
                source_kind="self",
            )
        )
    except ValueError:
        rejected = True
    assert rejected
    print("self_teacher_guard: OK")

    cognitive = _state(96, 0.3)
    expression_loss = organ.learn_expression(
        cognitive,
        LanguageInput(
            transition_id="expression-teacher-1",
            text="Я вижу кнопку.",
            source_kind="mentor",
        ),
    )
    assert math.isfinite(expression_loss)
    print(f"external_expression_teaching: OK loss={expression_loss:.6f}")

    utterance = organ.realize(
        ExpressionRequest(
            transition_id="realize-1",
            cognitive_state=cognitive,
            max_bytes=24,
            temperature=1.0,
        )
    )
    assert isinstance(utterance.text, str)
    print("expressive_forward: OK")
    print(f"generated_bytes: {utterance.token_count}")

    t = organ.telemetry()
    assert t.representation_learning_updates == 1
    assert t.teacher_language_updates == 1
    assert t.expression_learning_updates == 1
    assert t.rejected_self_teacher_updates == 1
    print(f"telemetry: {t}")

    with tempfile.TemporaryDirectory() as directory:
        incompatible_path = Path(directory) / "incompatible.pt"
        payload = organ.checkpoint_payload()
        model_state = dict(payload["model_state"])
        model_state["teacher_null_context"] = model_state["teacher_null_context"][:, :-1]
        payload["model_state"] = model_state
        torch.save(payload, incompatible_path)
        before_load = {
            name: value.detach().clone()
            for name, value in organ.state_dict().items()
        }
        load = organ.load_checkpoint(incompatible_path)
        assert not load.compatible
        assert "model_state shape" in load.reason
        assert all(
            torch.equal(value, before_load[name])
            for name, value in organ.state_dict().items()
        )
    print("incompatible_checkpoint_rejected_without_mutation: OK")

    with tempfile.TemporaryDirectory() as directory:
        incompatible_path = Path(directory) / "incompatible-optimizer.pt"
        payload = organ.checkpoint_payload()
        model_state = dict(payload["model_state"])
        model_state["teacher_null_context"] = (
            model_state["teacher_null_context"] + 1.0
        )
        payload["model_state"] = model_state
        payload["optimizer_state"] = {"state": {}, "param_groups": []}
        torch.save(payload, incompatible_path)
        before_model = {
            name: value.detach().clone()
            for name, value in organ.state_dict().items()
        }
        before_optimizer = _clone_state(organ.optimizer.state_dict())
        load = organ.load_checkpoint(incompatible_path, load_optimizer=True)
        assert not load.compatible
        assert "optimizer_state" in load.reason
        assert all(
            torch.equal(value, before_model[name])
            for name, value in organ.state_dict().items()
        )
        assert _state_equal(organ.optimizer.state_dict(), before_optimizer)
    print("incompatible_optimizer_rejected_without_mutation: OK")

    with tempfile.TemporaryDirectory() as directory:
        incompatible_path = Path(directory) / "incompatible-sparse-model.pt"
        payload = organ.checkpoint_payload(
            training_metadata={"sentinel": "sparse-model"}
        )
        model_state = dict(payload["model_state"])
        model_state["teacher_null_context"] = model_state[
            "teacher_null_context"
        ].to_sparse()
        payload["model_state"] = model_state
        torch.save(payload, incompatible_path)
        before_model = {
            name: value.detach().clone()
            for name, value in organ.state_dict().items()
        }
        before_optimizer = _clone_state(organ.optimizer.state_dict())
        load = organ.load_checkpoint(incompatible_path, load_optimizer=True)
        assert not load.compatible
        assert "model_state layout" in load.reason
        assert load.training_metadata == {"sentinel": "sparse-model"}
        assert all(
            torch.equal(value, before_model[name])
            for name, value in organ.state_dict().items()
        )
        assert _state_equal(organ.optimizer.state_dict(), before_optimizer)
    print("sparse_model_state_rejected_without_mutation: OK")

    with tempfile.TemporaryDirectory() as directory:
        incompatible_path = Path(directory) / "malformed-optimizer.pt"
        payload = organ.checkpoint_payload(
            training_metadata={"sentinel": "malformed-optimizer"}
        )
        model_state = dict(payload["model_state"])
        model_state["teacher_null_context"] = (
            model_state["teacher_null_context"] + 1.0
        )
        payload["model_state"] = model_state
        payload["optimizer_state"] = "bad"
        torch.save(payload, incompatible_path)
        before_model = {
            name: value.detach().clone()
            for name, value in organ.state_dict().items()
        }
        before_optimizer = _clone_state(organ.optimizer.state_dict())
        load = organ.load_checkpoint(incompatible_path, load_optimizer=True)
        assert not load.compatible
        assert "optimizer_state" in load.reason
        assert load.training_metadata == {"sentinel": "malformed-optimizer"}
        assert all(
            torch.equal(value, before_model[name])
            for name, value in organ.state_dict().items()
        )
        assert _state_equal(organ.optimizer.state_dict(), before_optimizer)
    print("malformed_optimizer_rejected_without_mutation: OK")
    print("HEIWA LANGUAGE ORGAN SMOKE: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
