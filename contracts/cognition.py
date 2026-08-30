from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias

TransitionId: TypeAlias = str
ObservationId: TypeAlias = str
AffordanceId: TypeAlias = str
IntentId: TypeAlias = str
AttemptId: TypeAlias = str
CompletedActionId: TypeAlias = str
ConsequenceId: TypeAlias = str
Payload: TypeAlias = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Observation:
    id: str
    world: str
    kind: str
    observed_at: float
    payload: Payload = field(default_factory=dict)

    @property
    def observation_id(self) -> str: return self.id
    @property
    def world_id(self) -> str: return self.world


@dataclass(frozen=True, slots=True)
class Affordance:
    id: str
    world: str
    action: str
    offered_at: float
    payload: Payload = field(default_factory=dict)
    constraints: Payload = field(default_factory=dict)
    expires_at: float | None = None

    @property
    def affordance_id(self) -> str: return self.id
    @property
    def world_id(self) -> str: return self.world
    @property
    def parameters(self) -> Payload: return self.payload


@dataclass(frozen=True, slots=True)
class ActionIntent:
    id: str
    transition_id: TransitionId
    affordance_id: AffordanceId
    world: str
    action: str
    created_at: float
    parameters: Payload = field(default_factory=dict)

    @property
    def intent_id(self) -> str: return self.id
    @property
    def world_id(self) -> str: return self.world
    @property
    def arguments(self) -> Payload: return self.parameters


@dataclass(frozen=True, slots=True)
class ActionAttempt:
    id: str
    intent_id: IntentId
    transition_id: TransitionId
    world: str
    action: str
    attempted_at: float
    accepted: bool
    external_id: str | None = None
    payload: Payload = field(default_factory=dict)

    @property
    def attempt_id(self) -> str: return self.id
    @property
    def world_id(self) -> str: return self.world
    @property
    def arguments(self) -> Payload: return self.payload


@dataclass(frozen=True, slots=True)
class CompletedAction:
    id: str
    attempt_id: AttemptId
    intent_id: IntentId
    transition_id: TransitionId
    world: str
    action: str
    completed_at: float
    status: str
    succeeded: bool
    external_id: str | None = None
    payload: Payload = field(default_factory=dict)

    @property
    def completed_action_id(self) -> str: return self.id
    @property
    def world_id(self) -> str: return self.world
    @property
    def arguments(self) -> Payload: return self.payload
    @property
    def result(self) -> Payload: return self.payload


@dataclass(frozen=True, slots=True)
class Consequence:
    id: str
    transition_id: TransitionId
    world: str
    kind: str
    observed_at: float
    completed_action_id: CompletedActionId | None = None
    payload: Payload = field(default_factory=dict)
    intent_id: IntentId | None = None
    attempt_id: AttemptId | None = None

    @property
    def consequence_id(self) -> str: return self.id
    @property
    def world_id(self) -> str: return self.world


# Experimental long-running action lifecycle.  The established Worlds do not
# need to use this for atomic click/key/move actions; it is reserved for real
# operations whose execution genuinely spans cognitive transitions.
@dataclass(frozen=True, slots=True)
class ExecutionResult:
    completed: bool
    active: bool = False
    result: Payload = field(default_factory=dict)
    active_payload: Payload = field(default_factory=dict)
    failure_kind: str | None = None
    failure: Payload = field(default_factory=dict)
    cancelled_active_action_id: str | None = None


@dataclass(frozen=True, slots=True)
class ActiveAction:
    active_action_id: str
    attempt_id: AttemptId
    intent_id: IntentId
    transition_id: TransitionId
    affordance_id: AffordanceId
    world_id: str
    action: str
    arguments: Payload = field(default_factory=dict)
    state: Payload = field(default_factory=dict)
    started_at: float | None = None


@dataclass(frozen=True, slots=True)
class CancelledAction:
    cancelled_action_id: str
    active_action_id: str
    original_transition_id: TransitionId
    cancellation_transition_id: TransitionId
    cancellation_intent_id: IntentId
    cancellation_attempt_id: AttemptId
    world_id: str
    action: str
    result: Payload = field(default_factory=dict)
    cancelled_at: float | None = None
