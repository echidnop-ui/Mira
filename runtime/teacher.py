from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Mapping, Protocol, runtime_checkable
from urllib.request import Request, urlopen
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class TeacherContext:
    conversation_id: str
    world_id: str
    observed_at: float
    event_id: str
    episode_id: str
    heiwa_utterance: str = ""
    teacher_utterance: str = ""
    recent_action: Mapping[str, object] = field(default_factory=dict)
    consequence: Mapping[str, object] = field(default_factory=dict)
    state_summary: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TeacherMessage:
    id: str
    conversation_id: str
    text: str
    kind: str = "ordinary"
    reply_to_event_id: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def language_teacher(self) -> bool:
        return self.kind == "explicit_linguistic_correction"


@runtime_checkable
class TeacherAdapter(Protocol):
    @property
    def teacher_id(self) -> str: ...

    def respond(self, context: TeacherContext) -> TeacherMessage | None: ...


_SYSTEM = """You are an external developmental teacher talking naturally to Heiwa.
You receive a compact event from one embodied agent. You may respond, explain,
ask, correct, demonstrate linguistically, or remain silent. Never output mouse
coordinates, key codes, shell commands, hidden motor commands, or scalar reward.
Return one JSON object only: {"id": string, "kind": "ordinary" or
"explicit_linguistic_correction" or "competence_feedback", "text": string,
"metadata": object}. For silence return {"kind":"silence","text":""}.
Use explicit_linguistic_correction only when the text is intentionally a target
utterance for language learning; ordinary conversation is not teacher data.
For competence_feedback metadata may contain {"useful": boolean,
"transition_id": string}; this is local competence evidence, not global reward."""


def _parse_message(raw: object, context: TeacherContext) -> TeacherMessage | None:
    if not isinstance(raw, dict):
        raise ValueError("teacher response must be a JSON object")
    kind = str(raw.get("kind", "ordinary")).strip()
    text = str(raw.get("text", "")).strip()
    if kind == "silence" or not text:
        return None
    if kind not in {"ordinary", "explicit_linguistic_correction", "competence_feedback"}:
        raise ValueError(f"unsupported teacher message kind: {kind!r}")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("teacher metadata must be an object")
    return TeacherMessage(
        id=str(raw.get("id") or f"teacher-{uuid4().hex}"),
        conversation_id=context.conversation_id,
        text=text,
        kind=kind,
        reply_to_event_id=context.event_id,
        metadata=metadata,
    )


class CommandTeacherAdapter:
    """Provider-neutral JSON subprocess boundary; the command never sees actuators."""

    def __init__(self, command: str, *, teacher_id: str = "external-command") -> None:
        self.command = tuple(shlex.split(command))
        if not self.command:
            raise ValueError("teacher command must not be empty")
        self._teacher_id = teacher_id

    @property
    def teacher_id(self) -> str:
        return self._teacher_id

    def respond(self, context: TeacherContext) -> TeacherMessage | None:
        request = {"system": _SYSTEM, "context": asdict(context)}
        result = subprocess.run(
            self.command, input=json.dumps(request, ensure_ascii=False), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"teacher command failed ({result.returncode}): {result.stderr.strip()}")
        output = result.stdout.strip()
        return None if not output else _parse_message(json.loads(output), context)


class OllamaTeacherAdapter:
    """Local production provider using Ollama's documented HTTP chat boundary."""

    def __init__(self, model: str, *, endpoint: str = "http://127.0.0.1:11434/api/chat") -> None:
        if not model.strip():
            raise ValueError("Ollama model must not be empty")
        self.model = model.strip(); self.endpoint = endpoint

    @property
    def teacher_id(self) -> str:
        return f"ollama:{self.model}"

    def respond(self, context: TeacherContext) -> TeacherMessage | None:
        body = json.dumps({
            "model": self.model, "stream": False, "format": "json",
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps(asdict(context), ensure_ascii=False)},
            ],
        }).encode()
        request = Request(self.endpoint, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode())
        content = payload.get("message", {}).get("content", "")
        return None if not str(content).strip() else _parse_message(json.loads(content), context)


class TeacherSessionStore:
    """Atomic conversation identity and message/event deduplication state."""

    def __init__(self, path: Path, *, teacher_id: str) -> None:
        self.path = path
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = {}
        existing_teacher = str(raw.get("teacher_id", teacher_id))
        if existing_teacher != teacher_id:
            raise ValueError(f"teacher state belongs to {existing_teacher!r}, not {teacher_id!r}")
        self.teacher_id = teacher_id
        self.conversation_id = str(raw.get("conversation_id") or f"conversation-{uuid4().hex}")
        self.processed_event_ids = set(map(str, raw.get("processed_event_ids", ())))
        self.inbound_message_ids = set(map(str, raw.get("inbound_message_ids", ())))
        self.outbound_message_ids = set(map(str, raw.get("outbound_message_ids", ())))
        self.executed_action_ids = set(map(str, raw.get("executed_action_ids", ())))
        self.save()

    def should_send_event(self, event_id: str) -> bool:
        return event_id not in self.processed_event_ids

    def mark_event(self, event_id: str) -> None:
        self.processed_event_ids.add(event_id); self.save()

    def accept(self, message: TeacherMessage) -> bool:
        if message.conversation_id != self.conversation_id:
            raise ValueError("teacher message lost conversation identity")
        if message.id in self.inbound_message_ids:
            return False
        self.inbound_message_ids.add(message.id); self.save(); return True

    def record_heiwa(self, message_id: str) -> bool:
        if message_id in self.outbound_message_ids:
            return False
        self.outbound_message_ids.add(message_id); self.save(); return True

    def action_was_executed(self, action_id: str) -> bool:
        return action_id in self.executed_action_ids

    def record_action(self, action_id: str) -> None:
        self.executed_action_ids.add(action_id); self.save()

    def save(self) -> None:
        payload = {
            "teacher_id": self.teacher_id, "conversation_id": self.conversation_id,
            "processed_event_ids": sorted(self.processed_event_ids)[-2048:],
            "inbound_message_ids": sorted(self.inbound_message_ids)[-2048:],
            "outbound_message_ids": sorted(self.outbound_message_ids)[-2048:],
            "executed_action_ids": sorted(self.executed_action_ids)[-2048:],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2); stream.write("\n")
                stream.flush(); os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)
