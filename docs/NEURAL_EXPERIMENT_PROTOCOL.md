# Heiwa Neural Experiment Protocol

This protocol applies when a Codex development agent trains or changes a
neural Heiwa organ. "Agent" below means the coding agent. "Model" or "organ"
means Heiwa's trainable system. Do not confuse the two.

## Required loop

```text
EXPERIMENT -> ANALYSIS -> ONE FIX -> EXPERIMENT
```

Before editing neural code, establish a sufficiently large, reproducible
baseline on the current checkpoint. One run must answer one declared
hypothesis.

Record before the run:

- experiment ID and date;
- git commit and checkpoint hash;
- hypothesis;
- trainable and frozen parameter groups;
- optimizer/scaler/scheduler state;
- dataset sources, sizes and source-separated splits;
- seeds and training budget;
- success gate and failure gate;
- old capabilities that must be retained.

## Evidence before architecture

Classify the bottleneck before changing anything:

- data or leakage;
- metric/decoder/selection;
- optimizer or gradient path;
- representation/resolution;
- causal credit;
- persistence/state restoration;
- domain shift;
- continual-learning forgetting;
- architecture capacity.

Architecture changes require evidence that a bounded data/loss/decoder/domain
fix cannot solve the measured failure. Do not change architecture, objective,
optimizer, curriculum and augmentation in the same experiment.

## Neural causality

For every trained organ answer:

```text
WHAT DID THIS ORGAN CAUSE?
WHAT EXACT EVIDENCE TRAINS IT?
WHICH transition_id RECEIVES CREDIT?
```

No gradient or learning update may be justified by an outcome the organ did
not causally influence. Preserve:

```text
ActionIntent != ActionAttempt != CompletedAction != Consequence
```

Screen/state change alone is not positive reward. User silence or absence of a
correction is not confirmation. Own generated language is not teacher data.

## Data and holdout hygiene

- Split by original source/scene/session, not augmented crop.
- Never optimize, mine hard negatives, tune thresholds or select epochs on the
  final holdout.
- If a holdout result changes the design, rotate a new final holdout when
  possible.
- Report synthetic, real-background and pure-real diagnostics separately.
- Labels from a renderer or explicit teacher are targets only; they are not
  model inputs.
- OCR, DOM, AT-SPI or hidden simulator metadata may be teacher-only evidence
  only when the task explicitly allows it. They must not silently become an
  inference dependency.

## Gradient and representation checks

Before a long run verify:

- finite non-zero gradient reaches the intended shared representation, not
  only a shortcut head;
- input-dependent embeddings actually change when RGB/text/state changes;
- frozen modules stay frozen;
- intended trainable parameters change;
- a debug probe restores model, optimizer, scheduler/scaler and runtime state;
- save/reload reproduces fixed probe predictions within tolerance.

## Evaluation

Compare BEFORE and AFTER on exactly the same independent benchmark, then run a
new-seed/source-separated final test.

Always report distributions and tails, not only means. Depending on task:

- classification: overall/per-class accuracy and confusion matrix;
- localization: top-1/top-k, median/p90/p95 spatial error, mass, entropy and
  catastrophic confident misses;
- sequence/language: exact match, character/byte accuracy, held-out
  composition and contrast margins;
- policy/motor: per-affordance scores, entropy/margin, target error, execution
  rate and correction response;
- causal runtime: linked transition counts, duplicate/delayed consequence
  controls and changed behavior after reload;
- continual learning: retention benchmark for every previously accepted skill.

Loss decreasing is never sufficient evidence.

## Online learning

Online updates must distinguish explicit evidence:

- accepted action/pattern;
- rejected action/pattern;
- local physical correction;
- environmental completion;
- unrelated external interference.

Forwarded duplicates from the model's own motor path are not human guidance.
Physical user correction must remain distinguishable. Learning must be visible
in a subsequent similar but non-identical state and survive persistence reload.

## Checkpoints

- Do not reset unrelated organs to fix one collapsed head.
- Do not overwrite a trusted checkpoint until the candidate passes reload and
  retention gates.
- Save model and all required training/runtime state atomically.
- On interruption, either preserve a valid previous checkpoint or a complete
  recoverable candidate; never leave a partially written authoritative file.
- Record exact trainable parameter groups and checkpoint hashes before/after.

## Experiment journal

Each substantial run must leave:

```text
EXPERIMENT ID
CHECKPOINT BEFORE
GIT COMMIT
HYPOTHESIS
ONE CHANGE
DATASET/SPLITS/SEEDS
TRAINING BUDGET
BASELINE METRICS
RESULT METRICS
TAIL/FAILURE ANALYSIS
RETENTION
RELOAD RESULT
ACCEPTED OR REJECTED
CHECKPOINT AFTER
```

## Stop conditions

Stop and report evidence instead of continuing when:

- the next step needs architecture redesign;
- the required holdout or teacher evidence does not exist;
- another active experiment owns the same GPU/checkpoint/files;
- results are too noisy to distinguish the hypothesis;
- the task would cross a forbidden module boundary;
- continuing would merely add epochs without new information.

An accepted neural change must improve the declared gate without destroying
causal invariants, persistence or previously accepted capability.
