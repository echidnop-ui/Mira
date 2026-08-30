from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from language.blocks import HybridBlockState, HybridLanguageBlock, LanguageWorkspace
from language.interface import (
    ExpressionRequest,
    LanguageInput,
    LanguageRepresentation,
    RealizedUtterance,
)
from language.tokenizer import Utf8ByteTokenizer, UnicodeCharTokenizer, tokenizer_from_state_dict
from language.ssm import resolve_ssm_backend


LANGUAGE_ARCH_VERSION = 5
STATEWEAVE_ARCH_VERSION = 2
_DEFAULT_TOKENIZER = Utf8ByteTokenizer()
PAD = _DEFAULT_TOKENIZER.pad_id
BOS = _DEFAULT_TOKENIZER.bos_id
EOS = _DEFAULT_TOKENIZER.eos_id
UNK = _DEFAULT_TOKENIZER.unk_id
VOCAB_SIZE = _DEFAULT_TOKENIZER.vocab_size

_ALLOWED_EXTERNAL_TEACHERS = frozenset({"human", "mentor"})
_ALLOWED_SOURCES = frozenset({"human", "mentor", "self", "system", "world"})


@dataclass(frozen=True, slots=True)
class NeuralLanguageConfig:
    representation_size: int = 96
    cognitive_size: int = 96
    embedding_size: int = 256
    hidden_size: int = 384
    recurrent_layers: int = 3
    attention_heads: int = 6
    attention_window: int = 32
    ffn_expansion: int = 4
    ssm_backend: str = "auto"
    stateweave_plastic_rank: int = 32
    stateweave_dynamic_rank: int = 32
    dropout: float = 0.05
    max_input_bytes: int = 512
    max_generation_bytes: int = 256
    learning_rate: float = 3e-4
    gradient_clip: float = 1.0
    max_traces: int = 256
    extra_characters: str = ""
    device: str | None = None
    dtype: torch.dtype = torch.float32

    @classmethod
    def local(cls) -> "NeuralLanguageConfig":
        # Approximately 168M trainable parameters with StateWeave v2. Width/depth can be raised
        # independently without changing the organ's public API.
        return cls(
            embedding_size=384,
            hidden_size=768,
            recurrent_layers=4,
            attention_heads=12,
            attention_window=48,
            max_input_bytes=1024,
            max_generation_bytes=384,
        )

    def checkpoint_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dtype"] = str(self.dtype).removeprefix("torch.")
        return payload


@dataclass(frozen=True, slots=True)
class LanguageTelemetry:
    forward_calls: int
    representation_learning_updates: int
    teacher_language_updates: int
    expression_learning_updates: int
    traces_created: int
    traces_consumed: int
    traces_evicted: int
    rejected_self_teacher_updates: int
    generated_utterances: int
    parameter_count: int


@dataclass(frozen=True, slots=True)
class LanguageCheckpointLoad:
    compatible: bool
    optimizer_loaded: bool
    architecture_version: int | None
    reason: str
    training_metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _EncodingTrace:
    language_input: LanguageInput


@dataclass(slots=True)
class ExpressiveRecurrentState:
    cognitive: Tensor
    layers: list[HybridBlockState]


class _ReceptiveEncoder(nn.Module):
    def __init__(
        self,
        config: NeuralLanguageConfig,
        embedding: nn.Embedding,
        ssm_backend: str,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.input_projection = nn.Linear(config.embedding_size, config.hidden_size)
        self.local_frontend = nn.Sequential(
            nn.Conv1d(
                config.hidden_size,
                config.hidden_size,
                kernel_size=5,
                padding=2,
                groups=config.hidden_size,
            ),
            nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.blocks = nn.ModuleList(
            HybridLanguageBlock(
                config.hidden_size,
                config.attention_heads,
                attention_window=config.attention_window,
                ssm_backend=ssm_backend,
                causal=False,
                dropout=config.dropout,
                expansion=config.ffn_expansion,
                stateweave_plastic_rank=config.stateweave_plastic_rank,
                stateweave_dynamic_rank=config.stateweave_dynamic_rank,
            )
            for _ in range(config.recurrent_layers)
        )
        self.workspace = LanguageWorkspace(config.hidden_size, config.representation_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.input_projection(self.embedding(token_ids))
        x = x + self.local_frontend(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.workspace(x, token_ids.eq(PAD))


class _ExpressiveDecoder(nn.Module):
    def __init__(
        self,
        config: NeuralLanguageConfig,
        embedding: nn.Embedding,
        vocab_size: int,
        ssm_backend: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding = embedding
        self.input_projection = nn.Linear(config.embedding_size, config.hidden_size)
        self.condition = nn.Sequential(
            nn.LayerNorm(config.cognitive_size),
            nn.Linear(config.cognitive_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.blocks = nn.ModuleList(
            HybridLanguageBlock(
                config.hidden_size,
                config.attention_heads,
                attention_window=config.attention_window,
                ssm_backend=ssm_backend,
                causal=True,
                dropout=config.dropout,
                expansion=config.ffn_expansion,
                stateweave_plastic_rank=config.stateweave_plastic_rank,
                stateweave_dynamic_rank=config.stateweave_dynamic_rank,
            )
            for _ in range(config.recurrent_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, vocab_size)

    def forward(self, input_ids: Tensor, cognitive_state: Tensor) -> Tensor:
        condition = self.condition(cognitive_state).unsqueeze(1)
        x = self.input_projection(self.embedding(input_ids)) + condition
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))

    def initial_state(self, cognitive_state: Tensor) -> ExpressiveRecurrentState:
        return ExpressiveRecurrentState(
            cognitive=self.condition(cognitive_state).unsqueeze(1),
            layers=[HybridBlockState() for _ in self.blocks],
        )

    def step(
        self,
        token_id: Tensor,
        state: ExpressiveRecurrentState,
    ) -> tuple[Tensor, ExpressiveRecurrentState]:
        x = self.input_projection(self.embedding(token_id)) + state.cognitive
        next_layers: list[HybridBlockState] = []
        for block, layer_state in zip(self.blocks, state.layers, strict=True):
            x, next_state = block.step(x, layer_state)
            next_layers.append(next_state)
        state.layers = next_layers
        return self.output(self.norm(x[:, -1, :])), state


class NeuralLanguageOrgan(nn.Module):
    """Scalable StateWeave hybrid Language architecture with teacher-only guards."""

    language_arch_version = LANGUAGE_ARCH_VERSION

    def __init__(
        self,
        config: NeuralLanguageConfig = NeuralLanguageConfig(),
        *,
        tokenizer: Utf8ByteTokenizer | UnicodeCharTokenizer | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._validate_config(config)
        self.device = self._resolve_device(config.device)
        self.ssm_backend, self.mamba2_probe = resolve_ssm_backend(
            config.ssm_backend, self.device
        )
        self.tokenizer = tokenizer or Utf8ByteTokenizer()
        self.embedding = nn.Embedding(
            self.tokenizer.vocab_size,
            config.embedding_size,
            padding_idx=self.tokenizer.pad_id,
        )
        self.receptive = _ReceptiveEncoder(
            config, self.embedding, self.ssm_backend
        )
        self.expressive = _ExpressiveDecoder(
            config, self.embedding, self.tokenizer.vocab_size, self.ssm_backend
        )
        self.teacher_null_context = nn.Parameter(torch.zeros(1, config.cognitive_size))
        self.to(device=self.device, dtype=config.dtype)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate)
        self._traces: OrderedDict[str, _EncodingTrace] = OrderedDict()
        self._forward_calls = 0
        self._representation_learning_updates = 0
        self._teacher_language_updates = 0
        self._expression_learning_updates = 0
        self._traces_created = 0
        self._traces_consumed = 0
        self._traces_evicted = 0
        self._rejected_self_teacher_updates = 0
        self._generated_utterances = 0

    @property
    def representation_size(self) -> int:
        return self.config.representation_size

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(
        self,
        language_input: LanguageInput,
        *,
        remember_for_learning: bool = True,
    ) -> LanguageRepresentation:
        value = self._normalize_input(language_input)
        if remember_for_learning and value.transition_id in self._traces:
            raise ValueError(
                "transition_id already has an unconsumed language trace: "
                f"{value.transition_id}"
            )
        token_ids = self._text_tensor(value.text, add_bos=True, add_eos=True)
        self.eval()
        with torch.no_grad():
            representation = self.receptive(token_ids)
        self._forward_calls += 1
        if remember_for_learning:
            self._store_trace(value.transition_id, _EncodingTrace(value))
        return LanguageRepresentation(
            transition_id=value.transition_id,
            source_kind=value.source_kind,
            vector=self._tensor_tuple(representation),
            token_count=int(token_ids.shape[1]),
            metadata={
                **dict(value.metadata),
                "language_arch_version": LANGUAGE_ARCH_VERSION,
                "stateweave_arch_version": STATEWEAVE_ARCH_VERSION,
            },
        )

    def learn_representation(
        self,
        transition_id: str,
        representation_gradient: Sequence[float],
    ) -> bool:
        transition_id = transition_id.strip()
        if not transition_id:
            raise ValueError("transition_id must not be empty")
        trace = self._traces.pop(transition_id, None)
        if trace is None:
            return False
        gradient = self._vector_tensor(
            representation_gradient, self.config.representation_size, "representation_gradient"
        )
        tokens = self._text_tensor(trace.language_input.text, add_bos=True, add_eos=True)
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        representation = self.receptive(tokens)
        representation.backward(gradient)
        self._step_optimizer()
        self._representation_learning_updates += 1
        self._traces_consumed += 1
        return True

    def learn_observed_text(self, language_input: LanguageInput) -> float:
        """Self-supervised surface-language update from an external observation.

        This teaches byte/character regularities only.  The observation is NOT
        promoted to truth, correction, reward, semantic target, or teacher
        evidence.  Own generated speech is categorically rejected so Heiwa
        cannot bootstrap its current mistakes into a self-confirming corpus.
        """
        value = self._normalize_input(language_input)
        if value.source_kind == "self":
            self._rejected_self_teacher_updates += 1
            raise ValueError("own speech cannot be observational language data")
        if value.source_kind not in {"human", "mentor", "world"}:
            raise ValueError(
                "observational language learning accepts only external human/mentor/world text; "
                f"got source_kind={value.source_kind!r}"
            )
        full = self._tokenize(value.text, add_bos=True, add_eos=True)
        if len(full) < 2:
            return 0.0
        input_ids = torch.tensor(full[:-1], device=self.device).unsqueeze(0)
        targets = torch.tensor(full[1:], device=self.device).unsqueeze(0)
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Null cognitive context deliberately limits this update to surface
        # language statistics. Meaning/correctness still comes from grounded
        # Core context and explicit HNT3 teach/fix paths.
        logits = self.expressive(input_ids, self.teacher_null_context)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite observational language loss")
        loss.backward()
        gradients = [p.grad for p in self.parameters() if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in gradients):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite observational language gradient")
        self._step_optimizer()
        return float(loss.detach().cpu())

    def learn_teacher_text(self, language_input: LanguageInput) -> float:
        value = self._normalize_input(language_input)
        self._require_external_teacher(value, "teacher language learning")
        full = self._tokenize(value.text, add_bos=True, add_eos=True)
        if len(full) < 2:
            return 0.0
        input_ids = torch.tensor(full[:-1], device=self.device).unsqueeze(0)
        targets = torch.tensor(full[1:], device=self.device).unsqueeze(0)
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.expressive(input_ids, self.teacher_null_context)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        loss.backward()
        self._step_optimizer()
        self._teacher_language_updates += 1
        return float(loss.detach().cpu())

    def learn_conditioned_expression(
        self,
        context: LanguageInput,
        target: LanguageInput,
    ) -> float:
        """Learn an explicit teacher-demonstrated response from raw context.

        Unlike ``learn_expression``, this keeps the receptive encoder in the
        autograd graph. The contextual representation and expressive decoder
        therefore learn together instead of treating the 96-D context as a
        detached constant. This is the developmental path used by HNT3.
        """
        context_value = self._normalize_input(context)
        target_value = self._normalize_input(target)
        self._require_external_teacher(context_value, "conditioned expression context")
        self._require_external_teacher(target_value, "conditioned expression target")
        context_ids = self._text_tensor(context_value.text, add_bos=True, add_eos=True)
        full = self._tokenize(target_value.text, add_bos=True, add_eos=True)
        if len(full) < 2:
            return 0.0
        input_ids = torch.tensor(full[:-1], device=self.device).unsqueeze(0)
        targets = torch.tensor(full[1:], device=self.device).unsqueeze(0)
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        cognitive = self.receptive(context_ids)
        logits = self.expressive(input_ids, cognitive)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        loss.backward()
        self._step_optimizer()
        self._expression_learning_updates += 1
        return float(loss.detach().cpu())

    def learn_expression(
        self,
        cognitive_state: Sequence[float],
        target: LanguageInput,
    ) -> float:
        value = self._normalize_input(target)
        self._require_external_teacher(value, "expression targets")
        state = self._vector_tensor(cognitive_state, self.config.cognitive_size, "cognitive_state")
        full = self._tokenize(value.text, add_bos=True, add_eos=True)
        input_ids = torch.tensor(full[:-1], device=self.device).unsqueeze(0)
        targets = torch.tensor(full[1:], device=self.device).unsqueeze(0)
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.expressive(input_ids, state)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        loss.backward()
        self._step_optimizer()
        self._expression_learning_updates += 1
        return float(loss.detach().cpu())

    @torch.no_grad()
    def generate(
        self,
        request: ExpressionRequest,
        *,
        deterministic: bool = False,
        top_k: int | None = None,
    ) -> RealizedUtterance:
        value = self._normalize_request(request)
        cognitive = self._vector_tensor(value.cognitive_state, self.config.cognitive_size, "cognitive_state")
        limit = min(value.max_bytes, self.config.max_generation_bytes)
        state = self.expressive.initial_state(cognitive)
        token = torch.tensor([[BOS]], device=self.device, dtype=torch.long)
        generated: list[int] = []
        stopped = False
        self.eval()
        for _ in range(limit):
            logits, state = self.expressive.step(token, state)
            logits[:, PAD] = -torch.inf
            logits[:, BOS] = -torch.inf
            next_token = self._select_next_token(
                logits,
                temperature=value.temperature,
                deterministic=deterministic,
                top_k=top_k,
            )
            token_id = int(next_token.item())
            if token_id == EOS:
                stopped = True
                break
            generated.append(token_id)
            token = next_token
        text = self.tokenizer.decode(generated)
        self._generated_utterances += 1
        return RealizedUtterance(
            transition_id=value.transition_id,
            text=text,
            token_count=len(generated),
            stopped_by_eos=stopped,
            metadata={
                **dict(value.metadata),
                "language_arch_version": LANGUAGE_ARCH_VERSION,
                "stateweave_arch_version": STATEWEAVE_ARCH_VERSION,
            },
        )

    def realize(self, request: ExpressionRequest) -> RealizedUtterance:
        deterministic = bool(request.metadata.get("deterministic", False))
        raw_top_k = request.metadata.get("top_k")
        top_k = int(raw_top_k) if raw_top_k is not None else None
        return self.generate(request, deterministic=deterministic, top_k=top_k)

    @staticmethod
    def _select_next_token(
        logits: Tensor,
        *,
        temperature: float,
        deterministic: bool,
        top_k: int | None,
    ) -> Tensor:
        if deterministic:
            return logits.argmax(dim=-1, keepdim=True)
        scaled = logits / temperature
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be > 0")
            k = min(top_k, scaled.shape[-1])
            threshold = torch.topk(scaled, k=k, dim=-1).values[:, -1:]
            scaled = scaled.masked_fill(scaled < threshold, -torch.inf)
        return torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)

    def initial_expression_state(self, cognitive_state: Sequence[float]) -> ExpressiveRecurrentState:
        """Create reusable SSM/recurrent/local-attention state for streaming."""
        state = self._vector_tensor(cognitive_state, self.config.cognitive_size, "cognitive_state")
        return self.expressive.initial_state(state)

    def checkpoint_payload(
        self,
        *,
        training_metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "format": "heiwa.language.v5.byte-context",
            "language_arch_version": LANGUAGE_ARCH_VERSION,
            "stateweave_arch_version": STATEWEAVE_ARCH_VERSION,
            "temporal_architecture": "stateweave_hybrid",
            "ssm_backend": self.ssm_backend,
            "config": self.config.checkpoint_dict(),
            "tokenizer": self.tokenizer.state_dict(),
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "training_metadata": dict(training_metadata or {}),
        }

    def save_checkpoint(
        self,
        path: Path,
        *,
        training_metadata: Mapping[str, object] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(self.checkpoint_payload(training_metadata=training_metadata), temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(
        self,
        path: Path,
        *,
        load_optimizer: bool = True,
    ) -> LanguageCheckpointLoad:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict):
            return LanguageCheckpointLoad(False, False, None, "invalid checkpoint root", {})
        raw_version = payload.get("language_arch_version")
        version = int(raw_version) if isinstance(raw_version, int) else None
        metadata = payload.get("training_metadata")
        training_metadata = metadata if isinstance(metadata, dict) else {}
        if version != LANGUAGE_ARCH_VERSION:
            return LanguageCheckpointLoad(
                False,
                False,
                version,
                f"incompatible Language architecture: expected {LANGUAGE_ARCH_VERSION}, found {version or 1}",
                training_metadata,
            )
        checkpoint_backend = payload.get("ssm_backend")
        if checkpoint_backend != self.ssm_backend:
            return LanguageCheckpointLoad(
                False,
                False,
                version,
                "checkpoint SSM backend mismatch: "
                f"checkpoint={checkpoint_backend!r}, current={self.ssm_backend!r}",
                training_metadata,
            )
        try:
            checkpoint_tokenizer = tokenizer_from_state_dict(payload["tokenizer"])
        except (KeyError, TypeError, ValueError) as exc:
            return LanguageCheckpointLoad(False, False, version, str(exc), training_metadata)
        if checkpoint_tokenizer.state_dict() != self.tokenizer.state_dict():
            return LanguageCheckpointLoad(
                False, False, version, "checkpoint tokenizer/vocabulary mismatch", training_metadata
            )
        model_state = payload.get("model_state")
        if not isinstance(model_state, dict):
            return LanguageCheckpointLoad(False, False, version, "model_state missing", training_metadata)
        expected_state = self.state_dict()
        missing = sorted(expected_state.keys() - model_state.keys())
        unexpected = sorted(model_state.keys() - expected_state.keys())
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing keys: {missing[:3]}")
            if unexpected:
                details.append(f"unexpected keys: {unexpected[:3]}")
            return LanguageCheckpointLoad(
                False,
                False,
                version,
                "incompatible model_state keys (" + "; ".join(details) + ")",
                training_metadata,
            )
        for name, expected in expected_state.items():
            checkpoint_value = model_state[name]
            if not isinstance(checkpoint_value, Tensor):
                return LanguageCheckpointLoad(
                    False,
                    False,
                    version,
                    f"model_state value for {name!r} is not a tensor",
                    training_metadata,
                )
            if checkpoint_value.shape != expected.shape:
                return LanguageCheckpointLoad(
                    False,
                    False,
                    version,
                    "incompatible model_state shape for "
                    f"{name!r}: checkpoint={tuple(checkpoint_value.shape)}, "
                    f"current={tuple(expected.shape)}",
                    training_metadata,
                )
            if checkpoint_value.layout != expected.layout:
                return LanguageCheckpointLoad(
                    False,
                    False,
                    version,
                    "incompatible model_state layout for "
                    f"{name!r}: checkpoint={checkpoint_value.layout}, "
                    f"current={expected.layout}",
                    training_metadata,
                )

        staged_optimizer: torch.optim.AdamW | None = None
        optimizer_state = payload.get("optimizer_state")
        if load_optimizer and "optimizer_state" in payload and not isinstance(
            optimizer_state, dict
        ):
            return LanguageCheckpointLoad(
                False,
                False,
                version,
                "incompatible optimizer_state: expected a mapping",
                training_metadata,
            )
        if load_optimizer and isinstance(optimizer_state, dict):
            staged_optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
            )
            try:
                staged_optimizer.load_state_dict(optimizer_state)
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                return LanguageCheckpointLoad(
                    False,
                    False,
                    version,
                    f"incompatible optimizer_state: {exc}",
                    training_metadata,
                )

        self.load_state_dict(model_state)
        optimizer_loaded = staged_optimizer is not None
        if staged_optimizer is not None:
            self.optimizer = staged_optimizer
        return LanguageCheckpointLoad(True, optimizer_loaded, version, "", training_metadata)

    def discard_trace(self, transition_id: str) -> bool:
        return self._traces.pop(transition_id.strip(), None) is not None

    def has_trace(self, transition_id: str) -> bool:
        return transition_id.strip() in self._traces

    def telemetry(self) -> LanguageTelemetry:
        return LanguageTelemetry(
            forward_calls=self._forward_calls,
            representation_learning_updates=self._representation_learning_updates,
            teacher_language_updates=self._teacher_language_updates,
            expression_learning_updates=self._expression_learning_updates,
            traces_created=self._traces_created,
            traces_consumed=self._traces_consumed,
            traces_evicted=self._traces_evicted,
            rejected_self_teacher_updates=self._rejected_self_teacher_updates,
            generated_utterances=self._generated_utterances,
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
        )

    def _require_external_teacher(self, value: LanguageInput, purpose: str) -> None:
        if value.source_kind not in _ALLOWED_EXTERNAL_TEACHERS:
            if value.source_kind == "self":
                self._rejected_self_teacher_updates += 1
            raise ValueError(
                f"{purpose} accepts only external human/mentor text; got source_kind={value.source_kind!r}"
            )

    def _step_optimizer(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip)
        self.optimizer.step()

    def _store_trace(self, transition_id: str, trace: _EncodingTrace) -> None:
        self._traces[transition_id] = trace
        self._traces.move_to_end(transition_id)
        self._traces_created += 1
        while len(self._traces) > self.config.max_traces:
            self._traces.popitem(last=False)
            self._traces_evicted += 1

    def _normalize_input(self, value: LanguageInput) -> LanguageInput:
        transition_id = value.transition_id.strip()
        source_kind = value.source_kind.strip().lower()
        if not transition_id:
            raise ValueError("LanguageInput.transition_id must not be empty")
        if source_kind not in _ALLOWED_SOURCES:
            raise ValueError(f"unsupported source_kind={value.source_kind!r}")
        if not isinstance(value.text, str):
            raise TypeError("LanguageInput.text must be str")
        return LanguageInput(
            transition_id=transition_id,
            text=self.tokenizer.normalize(value.text),
            source_kind=source_kind,
            source_id=value.source_id.strip(),
            metadata=dict(value.metadata),
        )

    def _normalize_request(self, value: ExpressionRequest) -> ExpressionRequest:
        transition_id = value.transition_id.strip()
        if not transition_id:
            raise ValueError("ExpressionRequest.transition_id must not be empty")
        state = self._normalize_vector(value.cognitive_state, self.config.cognitive_size, "cognitive_state")
        if value.max_bytes <= 0:
            raise ValueError("ExpressionRequest.max_bytes must be > 0")
        if not isfinite(value.temperature) or value.temperature <= 0.0:
            raise ValueError("ExpressionRequest.temperature must be finite and > 0")
        return ExpressionRequest(
            transition_id=transition_id,
            cognitive_state=state,
            max_bytes=value.max_bytes,
            temperature=value.temperature,
            metadata=dict(value.metadata),
        )

    def _tokenize(self, text: str, *, add_bos: bool, add_eos: bool) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
            max_characters=self.config.max_input_bytes,
        )

    def _text_tensor(self, text: str, *, add_bos: bool, add_eos: bool) -> Tensor:
        return torch.tensor(
            self._tokenize(text, add_bos=add_bos, add_eos=add_eos),
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)

    @staticmethod
    def _tensor_tuple(tensor: Tensor) -> tuple[float, ...]:
        return tuple(
            float(value)
            for value in tensor.detach().to("cpu", torch.float32).reshape(-1).tolist()
        )

    def _vector_tensor(self, values: Sequence[float], expected: int, name: str) -> Tensor:
        normalized = self._normalize_vector(values, expected, name)
        return torch.tensor(normalized, device=self.device, dtype=self.config.dtype).unsqueeze(0)

    @staticmethod
    def _normalize_vector(
        values: Sequence[float], expected: int, name: str
    ) -> tuple[float, ...]:
        if len(values) != expected:
            raise ValueError(f"{name} must have {expected} values; got {len(values)}")
        result = tuple(float(value) for value in values)
        if not all(isfinite(value) for value in result):
            raise ValueError(f"{name} contains non-finite values")
        return result

    @staticmethod
    def _validate_config(config: NeuralLanguageConfig) -> None:
        positive = (
            config.representation_size,
            config.cognitive_size,
            config.embedding_size,
            config.hidden_size,
            config.recurrent_layers,
            config.attention_heads,
            config.attention_window,
            config.ffn_expansion,
            config.max_input_bytes,
            config.max_generation_bytes,
            config.max_traces,
            config.stateweave_plastic_rank,
            config.stateweave_dynamic_rank,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("language sizes/counts must be > 0")
        if config.hidden_size % config.attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if config.ssm_backend not in {"auto", "mamba2", "selective_ssm"}:
            raise ValueError("ssm_backend must be auto, mamba2, or selective_ssm")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if config.learning_rate <= 0.0 or config.gradient_clip <= 0.0:
            raise ValueError("learning_rate and gradient_clip must be > 0")

    @staticmethod
    def _resolve_device(requested: str | None) -> torch.device:
        if requested:
            return torch.device(requested)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
