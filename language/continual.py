from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import torch

from language.interface import LanguageInput
from language.organ import NeuralLanguageConfig, NeuralLanguageOrgan


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def _iter_new(queue: Path, offset: int):
    if not queue.exists():
        return [], offset
    rows = []
    with queue.open('rb') as stream:
        size = stream.seek(0, 2)
        if offset > size:
            offset = 0
        stream.seek(offset)
        while True:
            line = stream.readline()
            if not line:
                break
            next_offset = stream.tell()
            try:
                raw = json.loads(line.decode('utf-8'))
            except Exception:
                offset = next_offset
                continue
            if isinstance(raw, dict):
                rows.append((raw, next_offset))
            offset = next_offset
    return rows, offset


def _load_organ(path: Path) -> NeuralLanguageOrgan:
    organ = NeuralLanguageOrgan(NeuralLanguageConfig.local())
    if path.exists():
        result = organ.load_checkpoint(path, load_optimizer=True)
        if not result.compatible:
            raise RuntimeError(f'incompatible Language checkpoint {path}: {result.reason}')
    return organ


def main() -> int:
    ap = argparse.ArgumentParser(description='Background observational Language learner')
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--progress-checkpoint', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--queue', type=Path, required=True)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--poll-seconds', type=float, default=3.0)
    ap.add_argument('--save-every', type=int, default=4)
    ap.add_argument('--max-text-bytes', type=int, default=1024)
    args = ap.parse_args()

    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = {'offset': 0, 'updates': 0, 'last_loss': None, 'last_id': '', 'skipped': 0}
    if args.state.exists():
        try:
            loaded = json.loads(args.state.read_text(encoding='utf-8'))
            if isinstance(loaded, dict): state.update(loaded)
        except Exception:
            pass

    source = args.progress_checkpoint if args.progress_checkpoint.exists() else args.checkpoint
    organ = _load_organ(source)
    recent = deque(maxlen=256)
    offset = int(state.get('committed_offset', state.get('offset', 0)))
    updates = int(state.get('updates', 0))
    unsaved = 0

    def save_candidate() -> None:
        nonlocal unsaved
        metadata = {
            'runtime': 'background_continual_language',
            'observational_learning_truth_claim': False,
            'truth_claim': False,
            'own_speech_teacher': False,
            'updates': updates,
            'queue_offset': offset,
        }
        organ.save_checkpoint(args.progress_checkpoint, training_metadata=metadata)
        # Candidate is a separately named atomic checkpoint so live runtime
        # never loads a file while it is being written.
        organ.save_checkpoint(args.candidate, training_metadata=metadata)
        _atomic_json(args.report, {
            'passed': True, 'updates': updates, 'last_loss': state.get('last_loss'),
            'queue_offset': offset, 'safe_language_candidate': True,
            'observational_learning_truth_claim': False,
            'truth_claim': False, 'own_speech_teacher': False,
            'candidate': str(args.candidate), 'updated_at': time.time(),
        })
        state['committed_offset'] = offset
        _atomic_json(args.state, state)
        unsaved = 0

    while True:
        rows, next_offset = _iter_new(args.queue, offset)
        if not rows:
            time.sleep(max(0.25, args.poll_seconds))
            continue
        for row, row_offset in rows:
            offset = row_offset
            text = str(row.get('text', '')).strip()
            source_kind = str(row.get('source_kind', '')).strip().lower()
            source_id = str(row.get('source_id', '')).strip()
            event_id = str(row.get('id', '')).strip()
            learning_kind = str(row.get('learning_kind', 'observation')).strip()
            if not text or source_kind not in {'human', 'mentor', 'world'}:
                state['skipped'] = int(state.get('skipped', 0)) + 1
                continue
            human_confirmed = bool(row.get('human_confirmed'))
            if (bool(row.get('self_generated')) or bool(row.get('own_speech'))) and not human_confirmed:
                state['skipped'] = int(state.get('skipped', 0)) + 1
                continue
            raw = text.encode('utf-8')[: max(1, args.max_text_bytes)]
            text = raw.decode('utf-8', errors='ignore').strip()
            if not text:
                continue
            digest_basis = learning_kind + '\0' + source_kind + '\0' + text + '\0' + str(row.get('context_text',''))
            if learning_kind in {'explicit_conditioned', 'explicit_surface_confirmation'}:
                digest_basis += '\0' + event_id
            digest = hashlib.sha256(digest_basis.encode('utf-8')).hexdigest()
            if digest in recent:
                continue
            recent.append(digest)
            try:
                if learning_kind in {'explicit_conditioned', 'explicit_surface_confirmation'}:
                    is_surface_confirmation = learning_kind == 'explicit_surface_confirmation'
                    target = LanguageInput(
                        transition_id=event_id or f'explicit-{digest[:20]}', text=text,
                        source_kind=('human' if is_surface_confirmation else 'mentor'),
                        source_id=source_id,
                        metadata={
                            'explicit_teacher_data': True,
                            'own_speech_teacher': False,
                            'human_confirmed': bool(row.get('human_confirmed')),
                            'surface_correctness_only': is_surface_confirmation,
                            'truth_claim': False,
                            'usefulness_claim': False,
                        },
                    )
                    context_text = str(row.get('context_text', '')).strip()
                    # Standalone Enter confirms only written form, not that the
                    # response was semantically correct in its preceding context.
                    if context_text and not is_surface_confirmation:
                        context = LanguageInput(
                            transition_id=(event_id or digest[:20]) + '-context', text=context_text,
                            source_kind='mentor', source_id=source_id,
                            metadata={'explicit_teacher_data': True, 'own_speech_teacher': False},
                        )
                        conditioned = organ.learn_conditioned_expression(context, target)
                    else:
                        conditioned = None
                    loss = organ.learn_teacher_text(target)
                    state['last_conditioned_loss'] = conditioned
                    if is_surface_confirmation:
                        state['human_surface_confirmations'] = int(state.get('human_surface_confirmations', 0)) + 1
                else:
                    value = LanguageInput(
                        transition_id=event_id or f'observed-{digest[:20]}',
                        text=text, source_kind=source_kind, source_id=source_id,
                        metadata={
                            'observational_language': True,
                            'explicit_teacher_data': False,
                            'truth_claim': False,
                            'own_speech_teacher': False,
                        },
                    )
                    loss = organ.learn_observed_text(value)
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                state['skipped'] = int(state.get('skipped', 0)) + 1
                _atomic_json(args.report, {
                    'passed': False, 'organism_continues': True,
                    'error': f'{type(exc).__name__}: {exc}', 'updates': updates,
                    'observational_only': True, 'truth_claim': False,
                    'updated_at': time.time(),
                })
                continue
            updates += 1
            unsaved += 1
            state.update({'updates': updates, 'last_loss': loss, 'last_id': event_id, 'offset': offset})
            _atomic_json(args.state, state)
            if learning_kind in {'explicit_conditioned', 'explicit_surface_confirmation'} or unsaved >= max(1, args.save_every):
                save_candidate()
        # `_iter_new` consumed the complete file snapshot; store its current EOF.
        try:
            offset = args.queue.stat().st_size
        except FileNotFoundError:
            offset = 0
        state['offset'] = offset
        _atomic_json(args.state, state)


if __name__ == '__main__':
    raise SystemExit(main())
