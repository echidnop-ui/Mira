from __future__ import annotations

import json
from pathlib import Path
import tempfile

from worlds.desktop.portal_screencast import PortalScreenCast


class _Pipe:
    def write(self, _: str) -> int: return 1
    def flush(self) -> None: pass
    def close(self) -> None: pass


class _Output(_Pipe):
    def readline(self) -> str: return json.dumps({"ready": True}) + "\n"


class _Worker:
    def __init__(self) -> None:
        self.stdin = _Pipe(); self.stdout = _Output(); self.stderr = _Pipe(); self.returncode = None
    def poll(self): return self.returncode
    def wait(self, timeout=None): self.returncode = 0; return 0
    def terminate(self): self.returncode = -15
    def kill(self): self.returncode = -9


class _LifecycleProbe(PortalScreenCast):
    def __init__(self) -> None:
        super().__init__(); self.remote_opens = 0; self.worker_spawns = 0; self.frame_requests = 0
    def available(self) -> bool: return True
    def _open_remote(self) -> tuple[int, int]:
        import os
        self.remote_opens += 1; read_fd, write_fd = os.pipe(); os.close(write_fd); return read_fd, 77
    def _spawn_stream_worker(self, fd: int, node_id: int):
        assert node_id == 77; self.worker_spawns += 1; return _Worker()
    def _request_worker_frame(self, destination: Path) -> None:
        assert self._worker is not None and self._worker.poll() is None
        self.frame_requests += 1; destination.write_bytes(b"synthetic-png")
    def close(self) -> None:
        self._stop_stream_worker()


def main() -> int:
    sensor = _LifecycleProbe()
    with tempfile.TemporaryDirectory() as directory:
        sensor.capture_png(Path(directory) / "first.png")
        worker_id = id(sensor._worker)
        sensor.capture_png(Path(directory) / "second.png")
        assert id(sensor._worker) == worker_id
    assert (sensor.remote_opens, sensor.worker_spawns, sensor.frame_requests) == (1, 1, 2)
    sensor.close(); assert sensor._worker is None
    print({"ok": True, "remote_opens": 1, "persistent_worker": True, "frames": 2, "closed": True})
    return 0


if __name__ == "__main__": raise SystemExit(main())

