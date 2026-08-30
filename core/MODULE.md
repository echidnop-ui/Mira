# Core

## PURPOSE

Persistent shared trainable cognition of Heiwa. It integrates normalized
experience over time and returns one world-agnostic cognitive representation.

## PUBLIC API

`core/interface.py`:

- `CoreInput`
- `IntegratedCognitiveState`
- `CognitiveCore`

`CognitiveCore` also exposes generation-bound opaque causal snapshots:

- `export_causal_snapshot(generation_id, transition_ids) -> bytes`
- `restore_causal_snapshot(snapshot, expected_generation_id, expected_transition_ids)`

These snapshots persist the exact private model, optimizer, temporal state,
learning traces and counters needed to resume delayed causal credit. Export
requires the supplied IDs to exactly match live traces. Restore validates and
stages the complete blob before mutating Core.

Production implementation: `core/personal.py::PersonalCognitiveCore`.

## INTERNAL ARCHITECTURE

The current scalable Core owns persistent work/slow/temporal/history state and a
sparse expert router. Expert families in `core/experts.py` are:

- selective SSM;
- xLSTM-style stabilized recurrent expert;
- causal TCN;
- history attention.

Context integration accepts optional memory/self/relation vectors through the
public `CoreInput` boundary.

## INPUT

A normalized fixed-size representation plus optional context vectors. The Core
does not capture pixels, parse Desktop APIs, tokenize language or call brokers.

## OUTPUT

`IntegratedCognitiveState` with main vector, workspace, temporal/context vectors,
novelty, state change and telemetry metadata.

## WHAT TRAINS IT?

Only causal gradients tied to stored transition traces. `PersonalCognitiveCore`
replays the exact stored input/state snapshot for that transition and consumes a
trace at most once. Runtime can preflight a delayed update through the read-only
`has_trace(transition_id)` public method before any cross-organ trace is consumed.

## MUST NOT

Core must not choose external actions, execute motor commands, contain Desktop or
Market strategy, or become a second Language organ.

## FILES

- `interface.py` — stable public contract.
- `personal.py` — production persistent Core.
- `experts.py` — heterogeneous temporal experts.
- `smoke.py`, `gpu_smoke.py`, benchmarks — verification/performance tools.

## READ SCOPE

For Core changes: this file, `interface.py`, `personal.py`, and the specific
expert file. Do not scan Worlds unless the public contract changes.
