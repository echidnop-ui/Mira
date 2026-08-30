# Policy

## PURPOSE

Shared trainable selection among the dynamic Affordances currently exposed by
one or more Worlds.

## PUBLIC API

`policy/interface.py`:

- `ActionPolicy`
- `TrainableActionPolicy`

Production implementation: `policy/dynamic.py::DynamicAffordancePolicy`.

`TrainableActionPolicy` exposes generation-bound opaque causal snapshots:

- `export_causal_snapshot(generation_id, transition_ids) -> bytes`
- `restore_causal_snapshot(snapshot, expected_generation_id, expected_transition_ids)`

The snapshot owns the exact private model, optimizer, decision traces and
counters required for delayed continuation. Export requires an exact live trace
ID match. Restore validates and stages the entire blob before mutating Policy.

## INPUT

- `IntegratedCognitiveState` from the shared Core;
- variable-length sequence of current `Affordance` objects.

## OUTPUT

One `ActionIntent` that references an actually offered Affordance.

## SPECIALIST PRIORS

A World specialist may put `policy_prior_probability` in an Affordance payload.
Dynamic Policy converts valid probabilities to centered log-priors and adds them
softly to its learned shared score. The specialist therefore guides an untrained
Policy but cannot force the final choice.

This is currently used by neural Desktop: `NeuralDesktopBody.action_head`
produces a motor distribution; Desktop exposes mechanically valid motor choices
as multiple Affordances; shared Policy selects the final action.

## LEARNING

Policy stores a minimal transition trace and can receive a later causal
advantage through `learn(transition_id, advantage)`. A trace is consumed once.
Runtime can preflight a delayed cross-organ update through the read-only
`has_trace(transition_id)` public method.

## MUST NOT

- fixed global action enum;
- Desktop scripts or trading strategy;
- "same action N times => choose something else" cognition;
- execution of the selected action;
- treating a specialist proposal as a compulsory command.

## READ SCOPE

Usually `MODULE.md`, `interface.py`, `dynamic.py`, plus shared contracts.
