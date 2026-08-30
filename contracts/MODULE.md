# Contracts

## PURPOSE

Stable world-agnostic causal vocabulary between Worlds, Runtime, Core and Policy.

## PUBLIC API

`cognition.py` defines:

- `Observation(id, world, kind, observed_at, payload)`
- `Affordance(id, world, action, offered_at, payload, constraints, expires_at)`
- `ActionIntent(id, transition_id, affordance_id, world, action, created_at, parameters)`
- `ActionAttempt(id, intent_id, transition_id, world, action, attempted_at, accepted, ...)`
- `CompletedAction(id, attempt_id, intent_id, transition_id, world, action, completed_at, status, succeeded, ...)`
- `Consequence(id, transition_id, world, kind, observed_at, completed_action_id, payload)`

Experimental `ActiveAction/CancelledAction/ExecutionResult` types may be used only
for operations that genuinely span cognitive transitions.

## CAUSAL INVARIANTS

`ActionIntent != ActionAttempt != CompletedAction`.

Rejected attempts do not become successful completed actions. Delayed
consequences retain the transition that caused the action.

## MUST NOT

Contracts contain no Desktop strategy, no Market strategy, no language rules and
no global action enum.

## READ SCOPE

Usually only `contracts/MODULE.md` and `contracts/cognition.py`.
