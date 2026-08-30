# HEIWA V2 — AI HANDOFF

Snapshot: 2026-08-28. Operational routing note added: 2026-08-31.

Read this after `AGENTS.md`. The local tree at `~/heiwa_v2` is authoritative.

Operational note: this file contains dated experiment history. The active work
queue is the set of open private GitHub Issues carrying `symphony-ready`; their
stage comments are newer handoff evidence. See `docs/SYMPHONY_OPERATIONS.md`.
Do not execute an older NEXT section when it conflicts with an active Issue.

Status words:
- VERIFIED: user actually ran it and showed successful output.
- IMPLEMENTED_NOT_VERIFIED: code/patch exists, but no successful local run was shown.
- VERIFIED_FAILURE: experiment ran and its result was inadequate.
- FUTURE: intended, not implemented.

## Project goal

Heiwa is one persistent artificial companion/agent across Worlds. Desktop, chat, games, VTuber, markets and future environments are organs/adapters of one Heiwa, not separate personalities.

Current priority: LANGUAGE FIRST.

Desktop should currently be treated mainly as a communication body for reaching the mentor. Do not spend the next cycle expanding random computer actions unless language-driven tasks require them.

## Architectural invariants

`Intention != Attempt != CompletedAction != Consequence`.

Delayed learning must return to the exact causal `transition_id`.

Mechanisms may be deterministic. Cognitive decisions should preferably be learned.

World:
- observes reality;
- exposes dynamic affordances;
- executes chosen actions;
- reports attempts/completion/consequences;
- does not decide what Heiwa wants.

Core:
- shared, persistent, world-agnostic;
- must not know Firefox/T-Bank/UI implementation details.

Language:
- an organ, not personality or Policy;
- own generated speech is NOT teacher evidence;
- ordinary mentor dialogue is NOT teacher evidence;
- only explicit teacher data may perform teacher updates.

Do not add permanent rules such as:
`if unknown: ask()`, `if greeting: say(...)`, `if curiosity > x: explore()`.

## Hardware/runtime

- project: `~/heiwa_v2`
- env: `~/heiwa_v2/.venv-rocm`
- Ubuntu / GNOME / Wayland
- AMD Radeon RX 6800 gfx1030 ~16 GiB
- Ryzen 7 2700X
- ~16 GiB RAM

ROCm PyTorch may report the device through the `cuda` API.

Known MIOpen gfx1030 warnings are non-blocking unless they cause measured correctness/performance problems.

## Core — VERIFIED

Shared scalable Core exists and has GPU forward/learning smokes.

Concept:
`CoreInput -> learned integration -> FAST -> heterogeneous experts -> fusion -> WORK -> SLOW -> Global Workspace -> IntegratedCognitiveState`.

Expert families include SSM, recurrent/xLSTM-style, causal TCN and history attention.

Core is not the present bottleneck. Do not redesign it while language is being debugged.

## Language architecture v3 / StateWeave — LOCAL_PATCH_VERIFIED

The public Language API and Unicode tokenizer are retained. The current local
patch sets `language_arch_version = 3` and adds StateWeave as the primary
Heiwa-native temporal branch: fast/medium/slow persistent states,
input-conditioned low-rank transition corrections, and delta-rule associative
fast-weight memory. Each layer learns to fuse StateWeave with the existing SSM,
gated recurrent-memory and local-attention branches.

The old `selective_ssm` is retained only as the current secondary SSM
baseline/fallback. `mamba_ssm.Mamba2` may later replace that SSM branch if it
passes a real local backend probe. StateWeave itself does not claim to be
Mamba-2. Local profile: ~166M trainable parameters.

## Visible ChatGPT mentor channel — VERIFIED

Real Firefox/ChatGPT messaging has worked.

Important facts:
- Firefox is Snap;
- AT-SPI helpers run in the Firefox Snap environment;
- Unicode input uses Wayland clipboard (`wl-copy`/`wl-paste`) with a temporary owner;
- keep the intended ChatGPT conversation visibly open;
- auto tab switching was removed;
- clipboard race was fixed.

UI/AT-SPI is slow. Correct training pattern:
`rare mentor request -> structured teacher curriculum -> many local GPU updates`.

Never use one ChatGPT UI round-trip per gradient step.

## Desktop causal learning — VERIFIED, now regression only

A full live chain worked:
`mentor instruction -> temporary Language/Core -> neural Policy -> real Desktop action -> AT-SPI completion -> mentor advantage -> Policy.learn`.

Observed:
- `desktop.type_text` for a focus task -> mentor `-1`;
- later `desktop.focus_next` -> completion -> mentor `+1`.

Fast local practice later ran 20/20 positive focus_next trials, mean ~358.9 ms/trial.

Real Firefox practice proved physical tabbing but also showed why the objective was bad:
`Chat with ChatGPT -> High -> Start dictation -> Start Voice -> ...`

Rewarding "any focus change" teaches blind tabbing. Freeze this as regression, not primary learning.

## Phrase bootstrap — OBSOLETE

An 8-lesson explicit language bootstrap ran successfully as infrastructure, but it was too phrase-centric and generation stayed mostly byte noise.

Historical checkpoint:
`state/language_bootstrap.pt`

Do not use it as the current language state.

## Compositional Language V1 — VERIFIED_FAILURE

A real 18 atom / 24 train / 8 holdout run completed.

Final observed metrics:
- atom_top1 = 0.056 ~= random 1/18
- train cosine = 0.192
- holdout_top1 = 0.125 = random 1/8
- holdout cosine = 0.184
- contrast_margin ~= 0

Conclusion: semantic composition did not learn despite falling losses.

Historical checkpoint:
`state/language_compositional.pt`

Structured curriculum is useful and expected at:
`state/language_curriculum/compositional.json`

Holdout was not used for training.

## Root cause found in V1

V1 semantic scaffold created fresh random vectors for slot/concept bindings and/or whole compositions.

Thus a heldout combination of known atoms still contained a new unseen random target. That defeats composition.

A second issue: expression training used a newly initialized `PersonalCognitiveCore()` conditioning space. That space could change across runs.

Do not resume Core-conditioned expression until the real persistent Core-language bridge is explicit and stable.

## CURRENT WORK — Compositional Semantic V2

Status: VERIFIED_PRESENT, RUN_BLOCKED.

Package generated earlier:
`HEIWA_COMPOSITIONAL_SEMANTIC_V2_FIX.zip`

The user started V2 and its runtime printed:
- `semantic_scaffold: reusable_signed_permutation_v2`
- `unseen_binding_random_vectors: False`
- `expression_conditioning: deferred_until_persistent_core_bridge`

Therefore V2 code is installed. However training did not begin because curriculum loading failed with `ValueError: invalid concept id ''`.

Root cause: the earlier curriculum saver serialized internal dataclass field `concept_id`, while the V2 parser only accepted mentor/public schema field `id`. A compatibility patch was produced to accept either key and always save canonical `id` in the future.

Do not call V2 training verified until that patch is applied and real training metrics are shown.

V2 intent:
- semantic targets are built only from reusable concept + role/slot components;
- no fresh whole-sentence random meaning vector for unseen combinations;
- surface LM update occurs before semantic grounding;
- Core-conditioned expression is temporarily disabled;
- V2 checkpoint is separate.

Expected V2 checkpoint:
`state/language_compositional_v2.pt`

Next command, after verifying the V2 code is actually installed:

```bash
cd ~/heiwa_v2
source .venv-rocm/bin/activate
python tools/verify_ai_state.py

python -m runtime.language_compositional_school   --curriculum-file state/language_curriculum/compositional.json   --epochs 6
```

Important random baselines:
- atom_top1 ~= 0.056
- train_top1 ~= 0.042
- holdout_top1 = 0.125

Do not call V2 successful merely because loss falls or a checkpoint appears. Metrics must move materially above random and minimal contrast margin should become positive/useful.

## Next work

1. Run/measure Compositional V2.
2. If still random: inspect gradients, target geometry, normalization/temperature, objective interference and similarity matrices. Do not blindly add 100 epochs.
3. Once receptive composition works: build a stable persistent Language <-> real Core semantic bridge.
4. Then train expressive language from that stable space.
5. Test generation on novel meaning combinations, not only memorized phrases.
6. Then expose "ask mentor" as an affordance that Policy can choose. Do NOT hardcode `if unknown: ask()`.
7. Expand Desktop mouse/click/scroll/vision only when Heiwa needs them for actual tasks.

## Do not do

- no canned reply table;
- no self-output teacher loop;
- no automatic teacher update from ordinary ChatGPT dialogue;
- no raw mentor transcript as default permanent memory;
- no blind focus_next curriculum;
- no giant handcrafted curiosity reward;
- no Runtime god object;
- no redesign of working Core during language debugging;
- no assumption that a generated ZIP was applied;
- no old GitHub repo search unless user explicitly asks.

## New-chat procedure

Tell a new assistant:

"Work on my local Heiwa v2 project. First read AGENTS.md, AI_HANDOFF.md, AI_STATE.md and the relevant MODULE.md. Run `python tools/verify_ai_state.py`. The local tree is authoritative. Do not guess that generated patches were applied. Continue from the first unverified/current NEXT item whose prerequisites are actually present."

## LIVE DEVELOPMENTAL LANGUAGE DIALOGUE — CURRENT NEXT STEP

Status: VERIFIED for one real single developmental turn.

The project direction was corrected again after the user pointed out that offline curriculum metrics are not the desired developmental loop.

New intended loop:

`Heiwa semantic training state -> Heiwa's own generated attempt -> visible ChatGPT mentor -> explicit correction -> local GPU learning -> Heiwa retry -> visible mentor evaluation`

Key rules:
- the first attempted text is generated by the current Language network, never a canned fallback;
- the failed attempt is NOT teacher data;
- the mentor correction IS explicit teacher data;
- ordinary MentorSession observation still performs zero automatic teacher updates;
- the semantic target is a temporary curriculum scaffold, explicitly NOT Heiwa's desire or autonomous goal;
- expression training is conditioned on the deterministic reusable SemanticScaffold, not on a freshly initialized Core;
- raw mentor transcript and failed raw attempts are not persisted as autobiography;
- only compact structured teacher corrections are persisted;
- after local learning Heiwa sends a real retry back to the mentor for evaluation;
- if the retry is understood, the curriculum scheduler advances; otherwise it repeats the same developmental target next turn.

Files:
- `language/developmental.py`
- `runtime/language_live_dialogue.py`

Single-turn regression command:

```bash
python -m runtime.language_live_dialogue --local-steps 24
```

This path supersedes further blind offline-epoch iteration as the immediate development priority. V2 compositional training remains useful as a seed/diagnostic, but language progress should now be observed through real attempts and corrections.

## LIVE CHATGPT WINDOW ACTIVATION FIX

Status: VERIFIED for exact activation and one real developmental turn.

Cause: launching the developmental loop from Terminal left the already-visible
Firefox ChatGPT window inactive. AT-SPI could locate the intended composer, but
`window_active` and `control_focused` correctly remained false.

Fix: `ChatGPTDesktopBootstrap.activate_composer()` now targets the exact
top-level Firefox frame already containing the showing ChatGPT document. It
uses the frame's AT-SPI `Component.grabFocus()` and, if needed, its `click`
action, then re-focuses the exact composer and re-reads all activation state.
The invariant remains mandatory and activation is bounded to three attempts.
No ChatGPT tab switching was added.

Focused deterministic regression: VERIFIED. A later real single developmental
turn also completed through the exact Firefox activation path. This does not
verify the new continuous multi-turn runtime live.

## CONTINUOUS DEVELOPMENTAL LANGUAGE HANDOFF

Status: `IMPLEMENTED_NOT_VERIFIED_LIVE`. One complete single developmental turn
was physically VERIFIED earlier; no new live mentor calls were made while
implementing the continuous process.

- Run: `python -m runtime.language_development_loop`
- Read status without loading GPU/mentor:
  `python -m runtime.language_development_loop --status`
- Checkpoint: `state/language_developmental_dialogue.pt`
- Session: `state/language_developmental/session.json`
- Status: `state/language_developmental/status.json`
- Install service: `./tools/install_heiwa_language_service.sh`
- Start service manually: `systemctl --user start heiwa-language`

The production path uses the real `ChatGPTDesktopBootstrap`; fake mentors exist
only in the local regression smoke. A single process owns one Language model,
optimizer and mentor session. Invalid mentor data rolls back to the last
consistent checkpoint, cannot advance the target, and cannot survive as a
teacher update. Own attempts and raw mentor transcripts are never teacher data.
Curriculum completion stops without wrapping or generating new goals.

## NATURAL CHATGPT TEACHER SESSION — EXPERIMENTAL

A minimal bounded no-curriculum teacher loop is now present in
`runtime/language_natural_teacher.py`. It uses the already-visible ChatGPT tab,
has no canned Heiwa replies and no production lesson phrase list. ChatGPT returns
a dynamic teacher utterance plus optional explicit corrections to Heiwa's actual
previous attempt. Teacher material trains Language online; own utterances do not.

The mode intentionally avoids fresh-Core expression conditioning: until the
persistent learned Core<->Language bridge exists, generation is conditioned on
Language's receptive vector for the current teacher message. The observation is
also fed through the process-local Core, but that Core vector is not used as a
stable supervised expression target. Treat this as EXPERIMENTAL and verify on the
real Firefox tab before raising status.
