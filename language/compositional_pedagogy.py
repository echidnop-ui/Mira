from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from statistics import fmean
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from language.organ import BOS, EOS, PAD, NeuralLanguageOrgan

_ALLOWED_EXTERNAL_TEACHERS = frozenset({"human", "mentor"})


@dataclass(frozen=True, slots=True)
class LexicalAtom:
    """Teacher-scaffolded atomic meaning with observed Russian word forms."""

    concept_id: str
    meaning: str
    forms: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.concept_id.strip():
            raise ValueError("concept_id must not be empty")
        if not self.meaning.strip():
            raise ValueError("atom meaning must not be empty")
        if not self.forms or any(not value.strip() for value in self.forms):
            raise ValueError("atom needs at least one non-empty word form")


@dataclass(frozen=True, slots=True)
class SemanticSlot:
    slot: str
    concept: str

    def __post_init__(self) -> None:
        if not self.slot.strip() or not self.concept.strip():
            raise ValueError("semantic slot and concept must not be empty")


@dataclass(frozen=True, slots=True)
class CompositionExample:
    """One meaning assembled from reusable atoms, with surface paraphrases."""

    meaning: tuple[SemanticSlot, ...]
    phrases: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.meaning:
            raise ValueError("composition meaning must not be empty")
        if len(self.phrases) < 2:
            raise ValueError("composition needs at least two surface phrases")
        if any(not phrase.strip() for phrase in self.phrases):
            raise ValueError("composition phrases must not be empty")
        slots = [item.slot for item in self.meaning]
        if len(slots) != len(set(slots)):
            raise ValueError("one composition may not repeat a semantic slot")

    @property
    def signature(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((item.slot, item.concept) for item in self.meaning))


@dataclass(frozen=True, slots=True)
class CompositionalCurriculum:
    atoms: tuple[LexicalAtom, ...]
    train: tuple[CompositionExample, ...]
    holdout: tuple[CompositionExample, ...]


@dataclass(frozen=True, slots=True)
class ReceptiveTrainResult:
    loss: float
    retrieval_accuracy: float
    cosine_to_target: float
    updates: int
    classes: int


@dataclass(frozen=True, slots=True)
class CompositionalEvalResult:
    atom_top1: float
    train_top1: float
    train_cosine: float
    holdout_top1: float
    holdout_cosine: float
    paraphrase_cosine: float
    contrast_margin: float
    contrast_pairs: int


@dataclass(frozen=True, slots=True)
class CompositionalPedagogyTelemetry:
    receptive_updates: int
    surface_updates: int
    expression_updates: int


class SemanticScaffold:
    """Deterministic, non-trainable *compositional* teacher scaffold.

    The previous scaffold accidentally assigned a fresh random vector to every
    slot=concept binding and even to the whole composition. That made an unseen
    holdout binding mathematically unknowable from learned atoms.

    V2 binds a reusable concept to a reusable semantic slot with a deterministic
    signed permutation. No vector is created for a specific slot=concept pair or
    for a whole sentence. Therefore an unseen combination is constructed only
    from parts that can already have been learned. The scaffold is teacher-only;
    it is not runtime Policy, desire, or an intent taxonomy.
    """

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("semantic scaffold size must be > 0")
        self.size = int(size)
        self._cache: dict[str, Tensor] = {}
        self._slot_cache: dict[str, tuple[Tensor, Tensor]] = {}

    def _seed(self, label: str) -> int:
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)

    def _vector(self, label: str, *, device: torch.device) -> Tensor:
        cached = self._cache.get(label)
        if cached is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._seed(label))
            cached = F.normalize(torch.randn(self.size, generator=generator), dim=0)
            self._cache[label] = cached
        return cached.to(device=device)

    def _slot_transform(
        self, slot: str, *, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        cached = self._slot_cache.get(slot)
        if cached is None:
            perm_gen = torch.Generator(device="cpu")
            perm_gen.manual_seed(self._seed(f"slot-permutation:{slot}"))
            permutation = torch.randperm(self.size, generator=perm_gen)

            sign_gen = torch.Generator(device="cpu")
            sign_gen.manual_seed(self._seed(f"slot-sign:{slot}"))
            signs = torch.randint(0, 2, (self.size,), generator=sign_gen)
            signs = signs.to(dtype=torch.float32).mul_(2.0).sub_(1.0)
            cached = (permutation, signs)
            self._slot_cache[slot] = cached
        permutation, signs = cached
        return permutation.to(device=device), signs.to(device=device)

    def atom(self, concept: str, *, device: torch.device) -> Tensor:
        return self._vector(f"concept:{concept}", device=device)

    def bind(self, slot: str, concept: str, *, device: torch.device) -> Tensor:
        value = self.atom(concept, device=device)
        permutation, signs = self._slot_transform(slot, device=device)
        role_bound = value[permutation] * signs
        # Keep some role-independent lexical identity while role_bound carries
        # who-did-what information. Both terms are reusable for unseen pairs.
        return F.normalize(0.25 * value + 0.75 * role_bound, dim=0)

    def composition(
        self,
        meaning: Sequence[SemanticSlot],
        *,
        device: torch.device,
    ) -> Tensor:
        pieces = [
            self.bind(item.slot, item.concept, device=device)
            for item in sorted(meaning, key=lambda value: value.slot)
        ]
        return F.normalize(torch.stack(pieces, dim=0).sum(dim=0), dim=0)


class CompositionalLanguagePedagogy:
    """Explicit lexical/compositional teacher mechanics for Language.

    No ordinary dialogue enters this object. It trains word/meaning grounding,
    novel combinations, Russian byte sequence structure, and expression from
    cognitive state. It never decides when Heiwa should speak or what goal she
    should pursue.
    """

    def __init__(
        self,
        organ: NeuralLanguageOrgan,
        *,
        temperature: float = 0.10,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self.organ = organ
        self.temperature = float(temperature)
        self.scaffold = SemanticScaffold(organ.representation_size)
        self._receptive_updates = 0
        self._surface_updates = 0
        self._expression_updates = 0

    def telemetry(self) -> CompositionalPedagogyTelemetry:
        return CompositionalPedagogyTelemetry(
            receptive_updates=self._receptive_updates,
            surface_updates=self._surface_updates,
            expression_updates=self._expression_updates,
        )

    def _receptive(self, text: str) -> Tensor:
        token_ids = self.organ._text_tensor(text, add_bos=True, add_eos=True)
        return self.organ.receptive(token_ids)[0]

    def _receptive_batch(self, texts: Sequence[str]) -> Tensor:
        """Encode equal-length groups together without introducing padding."""

        sequences = [
            self.organ._tokenize(text, add_bos=True, add_eos=True)
            for text in texts
        ]
        groups: dict[int, list[int]] = {}
        for index, sequence in enumerate(sequences):
            groups.setdefault(len(sequence), []).append(index)

        representations: list[Tensor | None] = [None] * len(sequences)
        for indices in groups.values():
            token_ids = torch.tensor(
                [sequences[index] for index in indices],
                device=self.organ.device,
                dtype=torch.long,
            )
            encoded = self.organ.receptive(token_ids)
            for row, index in enumerate(indices):
                representations[index] = encoded[row]

        if any(value is None for value in representations):
            raise RuntimeError("receptive batch did not encode every text")
        return torch.stack([value for value in representations if value is not None])

    def learn_receptive_composition(
        self,
        curriculum: CompositionalCurriculum,
        *,
        view_offset: int,
        batch_size: int = 12,
        passes: int = 2,
    ) -> ReceptiveTrainResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if passes <= 0:
            raise ValueError("passes must be > 0")

        device = self.organ.device
        class_anchors: list[Tensor] = []
        selected: list[tuple[str, int, Tensor]] = []

        # Every observed inflection/paraphrase participates each epoch. This is
        # still a tiny curriculum, but it gives the network many more lexical
        # views without inventing extra teacher facts.
        for atom in curriculum.atoms:
            class_index = len(class_anchors)
            anchor = self.scaffold.atom(atom.concept_id, device=device)
            class_anchors.append(anchor)
            for form in atom.forms:
                selected.append((form, class_index, anchor))

        for example in curriculum.train:
            class_index = len(class_anchors)
            anchor = self.scaffold.composition(example.meaning, device=device)
            class_anchors.append(anchor)
            for phrase in example.phrases:
                selected.append((phrase, class_index, anchor))

        target_matrix = F.normalize(torch.stack(class_anchors, dim=0), dim=-1)
        losses: list[float] = []
        accuracies: list[float] = []
        cosines: list[float] = []

        self.organ.train()
        total = len(selected)
        for pass_index in range(passes):
            # Rotation prevents the same neighbouring examples from sharing each
            # mini-batch forever while staying deterministic/reproducible.
            offset = ((view_offset + pass_index) * max(1, batch_size // 2)) % total
            ordered = selected[offset:] + selected[:offset]
            for start in range(0, len(ordered), batch_size):
                chunk = ordered[start : start + batch_size]
                representations = self._receptive_batch(
                    [text for text, _, _ in chunk]
                )
                normalized = F.normalize(representations, dim=-1)
                labels = torch.tensor(
                    [class_index for _, class_index, _ in chunk],
                    device=device,
                    dtype=torch.long,
                )
                own_targets = torch.stack([anchor for _, _, anchor in chunk], dim=0)
                raw_scores = normalized @ target_matrix.transpose(0, 1)
                logits = raw_scores / self.temperature

                retrieval = F.cross_entropy(logits, labels)
                own_score = raw_scores.gather(1, labels.unsqueeze(1)).squeeze(1)
                alignment = (1.0 - own_score).mean()

                wrong_scores = raw_scores.clone()
                wrong_scores.scatter_(1, labels.unsqueeze(1), -1.0e9)
                hard_negative = wrong_scores.max(dim=1).values
                margin = F.relu(0.20 + hard_negative - own_score).mean()

                # Alignment is deliberately stronger than in V1. The margin
                # prevents the trivial "everything points roughly together"
                # solution that produced chance-level atom/holdout retrieval.
                loss = retrieval + 2.0 * alignment + 0.75 * margin

                self.organ.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.organ.parameters(), self.organ.config.gradient_clip
                )
                self.organ.optimizer.step()
                self._receptive_updates += 1

                with torch.no_grad():
                    accuracy = (raw_scores.argmax(dim=-1) == labels).float().mean()
                losses.append(float(loss.detach().cpu().item()))
                accuracies.append(float(accuracy.detach().cpu().item()))
                cosines.append(float(own_score.detach().mean().cpu().item()))

        return ReceptiveTrainResult(
            loss=fmean(losses) if losses else 0.0,
            retrieval_accuracy=fmean(accuracies) if accuracies else 0.0,
            cosine_to_target=fmean(cosines) if cosines else 0.0,
            updates=len(losses),
            classes=len(class_anchors),
        )

    def learn_surface_language(
        self,
        texts: Iterable[str],
        *,
        source_kind: str,
        batch_size: int = 16,
    ) -> float:
        if source_kind not in _ALLOWED_EXTERNAL_TEACHERS:
            raise ValueError("surface language teacher must be human or mentor")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        unique: list[str] = []
        seen: set[str] = set()
        for raw in texts:
            text = raw.strip()
            if text and text not in seen:
                seen.add(text)
                unique.append(text)
        if not unique:
            return 0.0

        losses: list[float] = []
        self.organ.train()
        for start in range(0, len(unique), batch_size):
            chunk = unique[start : start + batch_size]
            sequences = [
                self.organ._tokenize(text, add_bos=True, add_eos=True)
                for text in chunk
            ]
            max_len = max(len(sequence) for sequence in sequences)
            input_ids = torch.full(
                (len(chunk), max_len - 1),
                PAD,
                device=self.organ.device,
                dtype=torch.long,
            )
            targets = torch.full_like(input_ids, PAD)
            for row, sequence in enumerate(sequences):
                inp = sequence[:-1]
                out = sequence[1:]
                input_ids[row, : len(inp)] = torch.tensor(
                    inp, device=self.organ.device, dtype=torch.long
                )
                targets[row, : len(out)] = torch.tensor(
                    out, device=self.organ.device, dtype=torch.long
                )

            context = self.organ.teacher_null_context.expand(len(chunk), -1)
            logits = self.organ.expressive(input_ids, context)
            loss = F.cross_entropy(
                logits.reshape(-1, self.organ.vocab_size),
                targets.reshape(-1),
                ignore_index=PAD,
            )
            self.organ.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.organ.parameters(), self.organ.config.gradient_clip
            )
            self.organ.optimizer.step()
            self._surface_updates += 1
            losses.append(float(loss.detach().cpu().item()))

        return fmean(losses)

    def learn_expression_batch(
        self,
        cognitive_states: Sequence[Sequence[float]],
        targets_text: Sequence[str],
        *,
        source_kind: str,
        batch_size: int = 12,
    ) -> float:
        if source_kind not in _ALLOWED_EXTERNAL_TEACHERS:
            raise ValueError("expression teacher must be human or mentor")
        if len(cognitive_states) != len(targets_text):
            raise ValueError("states and expression targets must have equal length")
        if not cognitive_states:
            return 0.0
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        losses: list[float] = []
        self.organ.train()
        for start in range(0, len(cognitive_states), batch_size):
            state_chunk = cognitive_states[start : start + batch_size]
            text_chunk = targets_text[start : start + batch_size]
            states = torch.tensor(
                state_chunk,
                device=self.organ.device,
                dtype=self.organ.config.dtype,
            )
            sequences = [
                self.organ._tokenize(text.strip(), add_bos=True, add_eos=True)
                for text in text_chunk
            ]
            max_len = max(len(sequence) for sequence in sequences)
            input_ids = torch.full(
                (len(text_chunk), max_len - 1),
                PAD,
                device=self.organ.device,
                dtype=torch.long,
            )
            targets = torch.full_like(input_ids, PAD)
            for row, sequence in enumerate(sequences):
                inp = sequence[:-1]
                out = sequence[1:]
                input_ids[row, : len(inp)] = torch.tensor(
                    inp, device=self.organ.device, dtype=torch.long
                )
                targets[row, : len(out)] = torch.tensor(
                    out, device=self.organ.device, dtype=torch.long
                )

            logits = self.organ.expressive(input_ids, states)
            loss = F.cross_entropy(
                logits.reshape(-1, self.organ.vocab_size),
                targets.reshape(-1),
                ignore_index=PAD,
            )
            self.organ.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.organ.parameters(), self.organ.config.gradient_clip
            )
            self.organ.optimizer.step()
            self._expression_updates += 1
            losses.append(float(loss.detach().cpu().item()))

        return fmean(losses)

    @torch.no_grad()
    def evaluate(self, curriculum: CompositionalCurriculum) -> CompositionalEvalResult:
        self.organ.eval()
        device = self.organ.device

        atom_targets = torch.stack(
            [self.scaffold.atom(atom.concept_id, device=device) for atom in curriculum.atoms],
            dim=0,
        )
        atom_correct = 0
        atom_total = 0
        for index, atom in enumerate(curriculum.atoms):
            for form in atom.forms:
                representation = F.normalize(self._receptive(form), dim=0)
                scores = representation @ atom_targets.transpose(0, 1)
                atom_correct += int(scores.argmax().item() == index)
                atom_total += 1
        atom_top1 = atom_correct / max(1, atom_total)

        train_targets = torch.stack(
            [self.scaffold.composition(example.meaning, device=device) for example in curriculum.train],
            dim=0,
        )
        train_correct = 0
        train_total = 0
        train_cosines: list[float] = []
        paraphrase_cosines: list[float] = []
        for index, example in enumerate(curriculum.train):
            reps = [F.normalize(self._receptive(phrase), dim=0) for phrase in example.phrases]
            for representation in reps:
                scores = representation @ train_targets.transpose(0, 1)
                train_correct += int(scores.argmax().item() == index)
                train_total += 1
                train_cosines.append(float(scores[index].cpu().item()))
            if len(reps) >= 2:
                paraphrase_cosines.append(float((reps[0] * reps[1]).sum().cpu().item()))

        holdout_targets = torch.stack(
            [self.scaffold.composition(example.meaning, device=device) for example in curriculum.holdout],
            dim=0,
        )
        holdout_correct = 0
        holdout_total = 0
        holdout_cosines: list[float] = []
        for index, example in enumerate(curriculum.holdout):
            for phrase in example.phrases:
                representation = F.normalize(self._receptive(phrase), dim=0)
                scores = representation @ holdout_targets.transpose(0, 1)
                holdout_correct += int(scores.argmax().item() == index)
                holdout_total += 1
                holdout_cosines.append(float(scores[index].cpu().item()))

        margins: list[float] = []
        pairs = _minimal_contrast_pairs(curriculum.train)
        for left_index, right_index in pairs:
            left = curriculum.train[left_index]
            right = curriculum.train[right_index]
            left_rep = F.normalize(self._receptive(left.phrases[0]), dim=0)
            right_rep = F.normalize(self._receptive(right.phrases[0]), dim=0)
            left_anchor = self.scaffold.composition(left.meaning, device=device)
            right_anchor = self.scaffold.composition(right.meaning, device=device)
            left_margin = (left_rep * left_anchor).sum() - (left_rep * right_anchor).sum()
            right_margin = (right_rep * right_anchor).sum() - (right_rep * left_anchor).sum()
            margins.append(float(((left_margin + right_margin) * 0.5).cpu().item()))

        return CompositionalEvalResult(
            atom_top1=float(atom_top1),
            train_top1=train_correct / max(1, train_total),
            train_cosine=fmean(train_cosines) if train_cosines else 0.0,
            holdout_top1=holdout_correct / max(1, holdout_total),
            holdout_cosine=fmean(holdout_cosines) if holdout_cosines else 0.0,
            paraphrase_cosine=fmean(paraphrase_cosines) if paraphrase_cosines else 0.0,
            contrast_margin=fmean(margins) if margins else 0.0,
            contrast_pairs=len(pairs),
        )

    @torch.no_grad()
    def semantic_similarity(
        self,
        text: str,
        meaning: Sequence[SemanticSlot],
    ) -> float:
        if not text.strip():
            return 0.0
        representation = F.normalize(self._receptive(text), dim=0)
        anchor = self.scaffold.composition(meaning, device=self.organ.device)
        value = float((representation * anchor).sum().cpu().item())
        return value if math.isfinite(value) else 0.0


def _minimal_contrast_pairs(
    examples: Sequence[CompositionExample],
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    sets = [set(example.signature) for example in examples]
    for left in range(len(examples)):
        for right in range(left + 1, len(examples)):
            difference = sets[left].symmetric_difference(sets[right])
            # One added/removed slot (e.g. negation), or one slot changing its
            # concept produces 1 or 2 differing (slot, concept) pairs.
            if 0 < len(difference) <= 2:
                pairs.append((left, right))
    return pairs
