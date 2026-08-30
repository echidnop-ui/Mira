from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from language.compositional_pedagogy import (
    CompositionalCurriculum,
    CompositionalLanguagePedagogy,
    CompositionExample,
    LexicalAtom,
    SemanticSlot,
)
from language.interface import ExpressionRequest
from language.organ import NeuralLanguageOrgan


@dataclass(frozen=True, slots=True)
class DevelopmentalTarget:
    meaning: tuple[SemanticSlot, ...]
    seed_phrases: tuple[str, ...]

    @property
    def signature(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((item.slot, item.concept) for item in self.meaning))

    def as_jsonable_meaning(self) -> list[dict[str, str]]:
        return [
            {"slot": item.slot, "concept": item.concept}
            for item in self.meaning
        ]

    @property
    def minimal_question_context(self) -> bool:
        """Whether this explicit teacher scaffold represents minimal inquiry."""

        concepts = {(item.slot, item.concept) for item in self.meaning}
        return (
            concepts
            == {("interrogative", "WHAT"), ("mode", "QUESTION")}
            and "?" in self.seed_phrases
        )


@dataclass(frozen=True, slots=True)
class TeacherCorrection:
    understood_before_training: bool
    teacher_forms: tuple[str, ...]
    word_focus: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class RetryEvaluation:
    understood: bool
    natural: bool
    teacher_form: str


def load_curriculum(path: Path) -> CompositionalCurriculum:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("curriculum root must be an object")

    raw_atoms = payload.get("atoms")
    if not isinstance(raw_atoms, list):
        raise ValueError("curriculum needs atoms[]")

    atoms: list[LexicalAtom] = []
    known: set[str] = set()
    for index, raw in enumerate(raw_atoms):
        if not isinstance(raw, dict):
            raise ValueError(f"atoms[{index}] must be an object")
        raw_id = raw.get("id")
        if raw_id is None:
            raw_id = raw.get("concept_id", "")
        concept_id = str(raw_id).strip().upper()
        if not concept_id:
            raise ValueError(f"atoms[{index}] has no concept id")
        meaning = str(raw.get("meaning", "")).strip()
        forms_raw = raw.get("forms")
        if not isinstance(forms_raw, list):
            raise ValueError(f"atoms[{index}].forms must be a list")
        forms = tuple(str(value).strip() for value in forms_raw if str(value).strip())
        atom = LexicalAtom(concept_id=concept_id, meaning=meaning, forms=forms)
        atoms.append(atom)
        known.add(concept_id)

    def parse_examples(raw: Any, label: str) -> tuple[CompositionExample, ...]:
        if not isinstance(raw, list):
            raise ValueError(f"{label} must be a list")
        result: list[CompositionExample] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"{label}[{index}] must be an object")
            raw_meaning = item.get("meaning")
            if not isinstance(raw_meaning, list) or not raw_meaning:
                raise ValueError(f"{label}[{index}].meaning must be a non-empty list")
            slots: list[SemanticSlot] = []
            for slot_index, raw_slot in enumerate(raw_meaning):
                if not isinstance(raw_slot, dict):
                    raise ValueError(
                        f"{label}[{index}].meaning[{slot_index}] must be an object"
                    )
                slot = str(raw_slot.get("slot", "")).strip().lower()
                concept = str(raw_slot.get("concept", "")).strip().upper()
                if concept not in known:
                    raise ValueError(
                        f"{label}[{index}] references unknown concept {concept!r}"
                    )
                slots.append(SemanticSlot(slot=slot, concept=concept))
            raw_phrases = item.get("phrases")
            if not isinstance(raw_phrases, list):
                raise ValueError(f"{label}[{index}].phrases must be a list")
            phrases = tuple(
                str(value).strip() for value in raw_phrases if str(value).strip()
            )
            result.append(CompositionExample(meaning=tuple(slots), phrases=phrases))
        return tuple(result)

    return CompositionalCurriculum(
        atoms=tuple(atoms),
        train=parse_examples(payload.get("train"), "train"),
        holdout=parse_examples(payload.get("holdout"), "holdout"),
    )


def ordered_targets(curriculum: CompositionalCurriculum) -> tuple[DevelopmentalTarget, ...]:
    indexed = list(enumerate(curriculum.train))
    indexed.sort(key=lambda pair: (len(pair[1].meaning), pair[0]))
    return tuple(
        DevelopmentalTarget(example.meaning, example.phrases)
        for _, example in indexed
    )


def semantic_state(
    pedagogy: CompositionalLanguagePedagogy,
    target: DevelopmentalTarget,
) -> tuple[float, ...]:
    tensor = pedagogy.scaffold.composition(
        target.meaning,
        device=pedagogy.organ.device,
    )
    return tuple(float(value) for value in tensor.detach().cpu().tolist())


def generate_attempt(
    organ: NeuralLanguageOrgan,
    state: Sequence[float],
    *,
    transition_id: str,
    max_bytes: int = 64,
    temperature: float = 0.55,
) -> str:
    result = organ.realize(
        ExpressionRequest(
            transition_id=transition_id,
            cognitive_state=tuple(float(value) for value in state),
            max_bytes=max_bytes,
            temperature=temperature,
            metadata={
                "developmental_language_attempt": True,
                "teacher_generated": False,
            },
        )
    )
    # Keep generated content honest: no canned fallback when the model emits
    # nothing or invalid UTF-8. Only remove terminal-hostile controls/newlines.
    value = result.text.replace("\x00", "").replace("\r", " ").replace("\n", " ")
    return " ".join(value.split()).strip()


def train_from_correction(
    pedagogy: CompositionalLanguagePedagogy,
    curriculum: CompositionalCurriculum,
    target: DevelopmentalTarget,
    correction: TeacherCorrection,
    *,
    local_steps: int,
    batch_size: int = 8,
) -> dict[str, float | int]:
    if local_steps <= 0:
        raise ValueError("local_steps must be > 0")
    if not correction.teacher_forms:
        raise ValueError("teacher correction needs at least one teacher form")

    state = semantic_state(pedagogy, target)
    expression_losses: list[float] = []
    surface_losses: list[float] = []

    # Explicit correction teaches the stable teacher-only semantic scaffold ->
    # Russian expression mapping. This is not a Core state and not a desire.
    states = [state for _ in correction.teacher_forms]
    for step in range(local_steps):
        expression_losses.append(
            pedagogy.learn_expression_batch(
                states,
                correction.teacher_forms,
                source_kind="mentor",
                batch_size=batch_size,
            )
        )
        if step == 0 or (step + 1) % 4 == 0:
            surface_texts = list(correction.teacher_forms)
            for _, forms in correction.word_focus:
                surface_texts.extend(forms)
            surface_losses.append(
                pedagogy.learn_surface_language(
                    surface_texts,
                    source_kind="mentor",
                    batch_size=batch_size,
                )
            )

    # Receptive grounding uses only explicit known lexical atoms plus the
    # corrected meaning from this live turn. It does not train on heldout data.
    micro = CompositionalCurriculum(
        atoms=curriculum.atoms,
        train=(
            CompositionExample(
                meaning=target.meaning,
                phrases=correction.teacher_forms,
            ),
        ),
        holdout=(),
    )
    receptive = pedagogy.learn_receptive_composition(
        micro,
        view_offset=0,
        batch_size=batch_size,
        passes=2,
    )

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "expression_loss": avg(expression_losses),
        "surface_loss": avg(surface_losses),
        "receptive_loss": receptive.loss,
        "receptive_retrieval": receptive.retrieval_accuracy,
        "local_expression_updates": len(expression_losses),
        "local_surface_updates": len(surface_losses),
        "local_receptive_updates": receptive.updates,
    }
