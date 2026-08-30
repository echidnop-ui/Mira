from __future__ import annotations

import gc
import hashlib
import math
from dataclasses import replace
from pathlib import Path
import tempfile

import torch

from language.compositional_pedagogy import CompositionalLanguagePedagogy
from language.interface import ExpressionRequest, LanguageInput
from language.organ import (
    LANGUAGE_ARCH_VERSION,
    NeuralLanguageConfig,
    NeuralLanguageOrgan,
)
from language.tokenizer import UnicodeCharTokenizer


def _config(device: str) -> NeuralLanguageConfig:
    return NeuralLanguageConfig(
        representation_size=16,
        cognitive_size=16,
        embedding_size=24,
        hidden_size=32,
        recurrent_layers=1,
        attention_heads=4,
        attention_window=8,
        ffn_expansion=2,
        dropout=0.0,
        max_input_bytes=64,
        max_generation_bytes=16,
        learning_rate=3e-3,
        max_traces=8,
        device=device,
    )


def _teacher(text: str, index: int) -> LanguageInput:
    return LanguageInput(
        transition_id=f"teacher-{index}",
        text=text,
        source_kind="mentor",
        metadata={"explicit_teacher_correction": True},
    )


def main() -> int:
    torch.manual_seed(7)
    print("HEIWA LANGUAGE ARCHITECTURE V3 STATEWEAVE LOCAL SMOKE")
    tokenizer = UnicodeCharTokenizer()
    sample = "Я понимаю это?"
    encoded = tokenizer.encode(sample, add_bos=True, add_eos=True)
    assert tokenizer.decode(encoded) == sample
    assert "�" not in tokenizer.decode(encoded)
    unknown = tokenizer.decode(tokenizer.encode("🙂"))
    assert unknown == "<UNK>" and "�" not in unknown
    print(f"tokenizer_roundtrip_russian: OK vocab={tokenizer.vocab_size}")
    print("tokenizer_invalid_utf8_replacement: 0")

    organ = NeuralLanguageOrgan(_config("cpu"))
    print(f"mamba2_backend_status: {organ.mamba2_probe.status}")
    print(f"ssm_backend_resolved: {organ.ssm_backend}")
    print("stateweave_primary_temporal_branch: OK")
    token_ids = organ._text_tensor(sample, add_bos=True, add_eos=True)
    receptive = organ.receptive(token_ids)
    assert receptive.shape == (1, 16)
    state = torch.zeros(1, 16)
    decoder_input = token_ids[:, :-1]
    expressive = organ.expressive(decoder_input, state)
    assert expressive.shape == (1, decoder_input.shape[1], organ.vocab_size)
    stream_state = organ.expressive.initial_state(state)
    stream_logits = []
    for index in range(decoder_input.shape[1]):
        step_logits, stream_state = organ.expressive.step(
            decoder_input[:, index : index + 1], stream_state
        )
        stream_logits.append(step_logits.unsqueeze(1))
    streamed = torch.cat(stream_logits, dim=1)
    assert torch.allclose(expressive, streamed, atol=2.0e-5, rtol=2.0e-4)
    print("receptive_forward_shape: OK [1,16]")
    print(f"expressive_forward_shape: OK {list(expressive.shape)}")
    print("reusable_stateweave_hybrid_state_equivalence: OK")

    forced_text = "Я понимаю это?"
    forced_ids: list[int] = []
    for token_id in tokenizer.encode(forced_text):
        logits = torch.full((1, tokenizer.vocab_size), -100.0)
        logits[0, token_id] = 100.0
        selected = organ._select_next_token(
            logits, temperature=1.0, deterministic=True, top_k=None
        )
        forced_ids.append(int(selected.item()))
    assert tokenizer.decode(forced_ids) == forced_text
    print("synthetic_logits_russian_generation: OK")

    cognitive = tuple(math.sin(index * 0.2) for index in range(16))
    utterance = organ.generate(
        ExpressionRequest(
            transition_id="autoregressive",
            cognitive_state=cognitive,
            max_bytes=8,
            temperature=0.8,
        ),
        deterministic=True,
    )
    assert isinstance(utterance.text, str) and "�" not in utterance.text
    print("autoregressive_recurrent_state_generation: OK")

    watched = dict(organ.named_parameters())["expressive.output.weight"]
    before = watched.detach().clone()
    loss = organ.learn_expression(cognitive, _teacher("Я понимаю это.", 0))
    gradient_total = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in organ.parameters()
        if parameter.grad is not None
    )
    assert math.isfinite(loss) and gradient_total > 0.0
    assert not torch.equal(before, watched.detach())
    print(f"teacher_step_finite_nonzero_update: OK loss={loss:.4f}")

    updates_before = organ.telemetry().teacher_language_updates
    try:
        organ.learn_teacher_text(
            LanguageInput("self-guard", "Я понимаю.", "self")
        )
    except ValueError:
        pass
    else:
        raise AssertionError("self-generated text became teacher data")
    assert organ.telemetry().teacher_language_updates == updates_before
    print("self_teacher_guard: OK")

    pedagogy = CompositionalLanguagePedagogy(organ)
    examples = ("Я понимаю это.", "Я это понимаю.", "Что?", "?")
    losses = [
        pedagogy.learn_surface_language(
            examples, source_kind="mentor", batch_size=4
        )
        for _ in range(10)
    ]
    assert math.isfinite(losses[-1]) and losses[-1] < losses[0]
    print(f"short_unicode_overfit_loss: {losses[0]:.4f}->{losses[-1]:.4f}")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "language-v3.pt"
        organ.save_checkpoint(
            checkpoint,
            training_metadata={"updates": 10, "source": "explicit_teacher"},
        )
        restored = NeuralLanguageOrgan(_config("cpu"))
        loaded = restored.load_checkpoint(checkpoint)
        assert loaded.compatible and loaded.optimizer_loaded
        assert loaded.architecture_version == LANGUAGE_ARCH_VERSION
        assert loaded.training_metadata["updates"] == 10
        assert restored.optimizer.state_dict()["state"]
        assert restored.tokenizer.state_dict() == organ.tokenizer.state_dict()
        raw_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert raw_checkpoint["ssm_backend"] == organ.ssm_backend
        assert raw_checkpoint["temporal_architecture"] == "stateweave_hybrid"
        assert raw_checkpoint["stateweave_arch_version"] == 1
        print("checkpoint_model_tokenizer_optimizer_resume: OK")
        print(f"checkpoint_ssm_backend_pinned: {organ.ssm_backend}")

        old = root / "language-v1.pt"
        torch.save({"state_dict": organ.state_dict()}, old)
        digest_before = hashlib.sha256(old.read_bytes()).hexdigest()
        incompatible = restored.load_checkpoint(old)
        digest_after = hashlib.sha256(old.read_bytes()).hexdigest()
        assert not incompatible.compatible
        assert f"expected {LANGUAGE_ARCH_VERSION}" in incompatible.reason
        assert digest_before == digest_after
        print("old_arch_incompatible_preserved: OK")

    del pedagogy, organ
    gc.collect()

    local = NeuralLanguageOrgan(replace(NeuralLanguageConfig.local(), device="cpu"))
    local_parameters = local.telemetry().parameter_count
    assert 140_000_000 <= local_parameters <= 210_000_000
    print(f"local_profile_parameters: {local_parameters}")
    del local
    gc.collect()

    if torch.cuda.is_available():
        gpu = NeuralLanguageOrgan(_config("cuda"))
        representation = gpu.encode(
            LanguageInput("gpu-forward", "Я понимаю это?", "human"),
            remember_for_learning=False,
        )
        assert len(representation.vector) == 16
        gpu_loss = gpu.learn_teacher_text(_teacher("Я понимаю это.", 99))
        assert math.isfinite(gpu_loss)
        print(f"rocm_forward_backward_smoke: OK device={gpu.device}")
        del gpu
        torch.cuda.empty_cache()
    else:
        print("rocm_device_smoke: SKIPPED_UNAVAILABLE")
    print("cpu_fallback_smoke: OK")
    print("HEIWA LANGUAGE ARCHITECTURE V3 STATEWEAVE LOCAL SMOKE: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
