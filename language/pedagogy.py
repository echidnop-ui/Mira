from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean
from typing import Iterable, Sequence
from uuid import uuid4

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from language.interface import LanguageInput
from language.organ import NeuralLanguageOrgan


@dataclass(frozen=True, slots=True)
class LanguageLesson:
    """One explicit external-teacher language lesson.

    `phrases` are paraphrases/near-equivalents of one meaning. They train the
    receptive side to put equivalent language closer without installing a
    fixed intent taxonomy. `expressions` are utterances an external teacher has
    explicitly approved for supervised expression practice.
    """

    meaning: str
    phrases: tuple[str, ...]
    expressions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.meaning.strip():
            raise ValueError("lesson meaning must not be empty")
        if len(self.phrases) < 2:
            raise ValueError("each lesson needs at least two equivalent phrases")
        if not self.expressions:
            raise ValueError("each lesson needs at least one expression target")
        if any(not value.strip() for value in (*self.phrases, *self.expressions)):
            raise ValueError("lesson text must not be empty")


@dataclass(frozen=True, slots=True)
class SemanticBatchResult:
    loss: float
    positive_similarity: float
    negative_similarity: float
    groups: int


@dataclass(frozen=True, slots=True)
class PedagogyTelemetry:
    semantic_updates: int
    teacher_text_updates: int
    expression_updates: int


class LanguagePedagogy:
    """Explicit language-teaching mechanics for NeuralLanguageOrgan.

    This object is intentionally inside the language module: it may train the
    organ's private receptive network, but it does not choose when Heiwa should
    speak, ask, wait, or act. Ordinary conversation must not call these methods.
    """

    def __init__(self, organ: NeuralLanguageOrgan, *, temperature: float = 0.12) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self.organ = organ
        self.temperature = float(temperature)
        self._semantic_updates = 0
        self._teacher_text_updates = 0
        self._expression_updates = 0

    def telemetry(self) -> PedagogyTelemetry:
        return PedagogyTelemetry(
            semantic_updates=self._semantic_updates,
            teacher_text_updates=self._teacher_text_updates,
            expression_updates=self._expression_updates,
        )

    def learn_semantic_equivalence(
        self,
        lessons: Sequence[LanguageLesson],
        *,
        view_offset: int = 0,
    ) -> SemanticBatchResult:
        """One in-batch contrastive update over teacher-marked paraphrases.

        No fixed concept IDs are learned. The only supervision is that two
        teacher-provided phrases in the same lesson should encode closer than
        phrases sampled from other lessons.
        """

        usable = [lesson for lesson in lessons if len(lesson.phrases) >= 2]
        if len(usable) < 2:
            raise ValueError("semantic equivalence batch needs at least two lessons")

        first: list[Tensor] = []
        second: list[Tensor] = []
        self.organ.train()
        for lesson in usable:
            count = len(lesson.phrases)
            a = lesson.phrases[view_offset % count]
            b = lesson.phrases[(view_offset + 1) % count]
            first.append(self.organ.receptive(self.organ._text_tensor(a, add_bos=True, add_eos=True)))
            second.append(self.organ.receptive(self.organ._text_tensor(b, add_bos=True, add_eos=True)))

        left = F.normalize(torch.cat(first, dim=0), dim=-1)
        right = F.normalize(torch.cat(second, dim=0), dim=-1)
        embeddings = torch.cat((left, right), dim=0)
        groups = left.shape[0]

        logits = embeddings @ embeddings.transpose(0, 1)
        logits = logits / self.temperature
        diagonal = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(diagonal, -torch.inf)
        targets = torch.cat(
            (
                torch.arange(groups, 2 * groups, device=logits.device),
                torch.arange(0, groups, device=logits.device),
            )
        )
        loss = F.cross_entropy(logits, targets)

        self.organ.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.organ.parameters(), self.organ.config.gradient_clip)
        self.organ.optimizer.step()
        self._semantic_updates += 1

        with torch.no_grad():
            positive = torch.sum(left * right, dim=-1).mean()
            negatives = left @ right.transpose(0, 1)
            mask = ~torch.eye(groups, device=negatives.device, dtype=torch.bool)
            negative = negatives[mask].mean()

        return SemanticBatchResult(
            loss=float(loss.detach().cpu().item()),
            positive_similarity=float(positive.detach().cpu().item()),
            negative_similarity=float(negative.detach().cpu().item()),
            groups=groups,
        )

    def learn_teacher_language(self, texts: Iterable[str], *, source_id: str) -> float:
        losses: list[float] = []
        seen: set[str] = set()
        for raw in texts:
            text = raw.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            loss = self.organ.learn_teacher_text(
                LanguageInput(
                    transition_id=f"language-teacher-{uuid4().hex}",
                    text=text,
                    source_kind="mentor",
                    source_id=source_id,
                    metadata={"explicit_language_lesson": True},
                )
            )
            if math.isfinite(loss):
                losses.append(loss)
                self._teacher_text_updates += 1
        return fmean(losses) if losses else 0.0

    def learn_expression(
        self,
        cognitive_state: Sequence[float],
        target_text: str,
        *,
        source_id: str,
        meaning: str,
    ) -> float:
        loss = self.organ.learn_expression(
            cognitive_state,
            LanguageInput(
                transition_id=f"language-expression-{uuid4().hex}",
                text=target_text,
                source_kind="mentor",
                source_id=source_id,
                metadata={
                    "explicit_language_lesson": True,
                    "lesson_meaning": meaning,
                },
            ),
        )
        self._expression_updates += 1
        return loss
