# ACTIVE DESKTOP FIX — 2026-08-29

Current user-directed task: neural Desktop action repetition/collapse.

Verified local source fix:
- restored the established `CoreInput/IntegratedCognitiveState/CognitiveCore` and shared contract compatibility after accidental parallel-interface overwrites;
- preserved the existing pixel-first `NeuralDesktopBody`;
- removed the inference-time repetition penalty/sampling as the final selector;
- `NeuralDesktopWorldAdapter` now exposes multiple mechanically valid motor Affordances with per-action neural prior probabilities;
- `DynamicAffordancePolicy` is again the final selector and can learn to override a collapsed specialist prior;
- regression with a synthetic ~91% `right_click` specialist prior switched away after negative causal feedback;
- `core.smoke`, `policy.smoke`, `worlds.desktop.neural_smoke`, and `runtime.embodied_smoke` pass locally.
- AT-SPI desktop cognition smoke is environment-blocked in the review container because `pyatspi` is not installed.

Important distinction: repeated atomic click/move/scroll decisions are NOT `ActiveAction`; continue/cancel lifecycle is reserved for operations that physically remain active across transitions.

---

# HEIWA V2 — CURRENT STATE

Snapshot: 2026-08-28. Operational routing note added: 2026-08-31.

This is a dated state snapshot, not the live task queue. Active work is tracked
in private staged GitHub Issues under `automation/symphony/WORKFLOW.md`. See
`docs/SYMPHONY_OPERATIONS.md` and do not let an older NEXT entry override a
newer Issue handoff.

Current priority:

`LANGUAGE -> lexical/compositional semantics -> stable Core bridge -> expression -> autonomous inquiry`

Desktop expansion is paused except mentor communication/regression.

## Status

| Area | Status |
|---|---|
| Shared scalable Core | VERIFIED |
| Neural Language architecture v3 / StateWeave hybrid | IMPLEMENTED_LOCAL_VERIFIED |
| Visible Firefox/ChatGPT mentor bridge | VERIFIED |
| Full mentor->Policy->Desktop->feedback loop | VERIFIED |
| Fast local desktop focus practice | VERIFIED, regression only |
| Real blind focus practice | OBSOLETE |
| 8-lesson phrase bootstrap | OBSOLETE |
| Compositional V1 | VERIFIED_FAILURE |
| Compositional Semantic V2 code | VERIFIED_PRESENT, RUN_BLOCKED |
| Persistent Core<->Language semantic bridge | FUTURE |
| Policy-chosen autonomous mentor inquiry | FUTURE |

Expected curriculum:
`state/language_curriculum/compositional.json`

Historical checkpoints:
- `state/language_bootstrap.pt`
- `state/language_compositional.pt`

Expected V2 checkpoint only after an actual V2 run:
`state/language_compositional_v2.pt`

Random baselines for the current 18/24/8 curriculum:
- atom_top1 ~= 0.056
- train_top1 ~= 0.042
- holdout_top1 = 0.125

Last verified V1 result:
- atom_top1 0.056
- train cosine 0.192
- holdout_top1 0.125
- holdout cosine 0.184
- contrast margin approximately 0

Latest V2 attempt:

- V2 code was definitely loaded (`semantic_scaffold: reusable_signed_permutation_v2`).
- Training did NOT start.
- Loader failed on the saved curriculum with `ValueError: invalid concept id ''`.
- Root cause: the earlier saver persisted atom key `concept_id`, while V2 loader expected only public key `id`.
- A schema-compatibility fix was produced to accept both and save canonical `id`.

Immediate question for the next chat:

**Was the curriculum schema-compatibility patch applied, and what are the first real V2 training metrics?**

Answer from the local tree and fresh run output, not from chat history assumptions.

## LANGUAGE ARCHITECTURE V3 — STATEWEAVE

The current Language line is `language_arch_version = 3`. Every hybrid block has
a Heiwa-native StateWeave branch with fast/medium/slow recurrent states,
input-conditioned low-rank transition corrections and delta-rule plastic
fast-weight memory. It is fused with the existing SSM, recurrent-memory and
local-attention branches.

`mamba_ssm.Mamba2` remains optional in the separate SSM branch. The current
environment still resolves that branch to `selective_ssm` when Mamba2 is not
available; this no longer defines the primary temporal architecture. Local
profile size is ~166M parameters.

## CURRENT NEXT ACTION — LIVE LANGUAGE

`runtime/language_live_dialogue.py` completed one real developmental turn on
the user's machine and is VERIFIED as the single-turn regression path.

It should perform exactly one live developmental turn:

1. load current Language (prefer developmental checkpoint, otherwise V2 checkpoint);
2. select one simple curriculum meaning as a teacher-only scaffold;
3. generate Heiwa's own attempt;
4. physically send that attempt to visible ChatGPT;
5. receive explicit structured correction;
6. train locally on GPU;
7. generate a retry from the same meaning;
8. physically send retry to ChatGPT;
9. receive understood/natural evaluation;
10. save Language optimizer/weights and compact progress.

That one-turn live path is verified; this does not verify the newer continuous
multi-turn orchestrator.

## EXACT FIREFOX WINDOW ACTIVATION

Status: VERIFIED for the single live developmental turn.

The bridge now address-activates only the top-level Firefox frame that already
contains the showing intended ChatGPT document, then focuses the composer and
rechecks `window_active`, `document_showing` and `control_focused`. It uses
AT-SPI `Component.grabFocus()` plus the frame `click` action, with at most three
attempts and no automatic tab switching. Both the focused regression and the
subsequent real single-turn send passed. The intended ChatGPT conversation must
still be the visible tab in that exact Firefox window.

## PERSISTENT DEVELOPMENTAL LANGUAGE RUNTIME

Status: `IMPLEMENTED_NOT_VERIFIED_LIVE`.

The previously verified single live turn remains valid. The new continuous
orchestrator reuses one Language model, optimizer, mentor bridge/session and
Core for all turns in one process. It repeats a target after
`understood=false`, advances exactly once after `understood=true`, and stops
without wrapping when the curriculum is complete.

Production launch: `python -m runtime.language_development_loop`

Status only: `python -m runtime.language_development_loop --status`

Persistence:
- checkpoint: `state/language_developmental_dialogue.pt`
- session: `state/language_developmental/session.json`
- status: `state/language_developmental/status.json`
- explicit structured corrections only:
  `state/language_curriculum/developmental_corrections.jsonl`

Local fake regressions verify progression, rollback, model/optimizer resume,
graceful stop, completion without wrap, lightweight status and one-model
lifecycle. They do not call Firefox or ChatGPT. Own attempts remain non-teacher
data, ordinary mentor dialogue remains non-teacher data, and no raw mentor
transcript is persisted. Curriculum completion does not invent new goals.

## NATURAL CHATGPT TEACHER SESSION

Status: EXPERIMENTAL / LOCAL_SMOKE_VERIFIED, REAL_TAB_NOT_YET_VERIFIED.

A narrow language-only runtime now exists at `runtime/language_natural_teacher.py`.
It removes the fixed `TARGET_MEANING` curriculum from the live teacher loop. One
visible ChatGPT tab dynamically chooses what to teach from Heiwa's actual own
utterance. Explicit teacher utterances and post-attempt corrections update
Language online; Heiwa's own output never becomes teacher data.

This is deliberately not claimed as the final Core-language bridge. Expression
is conditioned on the current stable Language receptive vector while the same
mentor message is also observed by a process-local Core. The persistent learned
Core<->Language semantic bridge remains future work.
