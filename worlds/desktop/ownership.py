from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from typing import Iterator


@dataclass(frozen=True, slots=True)
class OwnedActionWindow:
    source: str
    started_ns: int
    finished_ns: int


class PhysicalActionArbiter:
    """Serialize physical injectors and retain bounded monotonic ownership markers."""

    def __init__(self, *, max_markers: int = 256) -> None:
        self._lock = threading.RLock(); self._markers: deque[OwnedActionWindow] = deque(maxlen=max_markers)

    @contextmanager
    def claim(self, source: str) -> Iterator[None]:
        if not source.strip(): raise ValueError("physical action source must not be empty")
        with self._lock:
            started = time.monotonic_ns()
            try: yield
            finally: self._markers.append(OwnedActionWindow(source, started, time.monotonic_ns()))

    def foreign_windows(self, source: str, before_ns: int, after_ns: int) -> tuple[OwnedActionWindow, ...]:
        return tuple(marker for marker in self._markers
                     if marker.source != source and marker.finished_ns > before_ns
                     and marker.started_ns < after_ns)
