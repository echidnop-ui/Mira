from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid


PORTAL_DEST = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST_IFACE = "org.freedesktop.portal.ScreenCast"
REQUEST_IFACE = "org.freedesktop.portal.Request"
DBUS_DEST = "org.freedesktop.DBus"
DBUS_PATH = "/org/freedesktop/DBus"
DBUS_IFACE = "org.freedesktop.DBus"


class PortalScreenCastError(RuntimeError):
    """Raised when the XDG ScreenCast/PipeWire body sensor cannot operate."""


_GST_FRAME_WORKER = r'''
import json
import os
from pathlib import Path
import sys

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
from PIL import Image

Gst.init(None)
fd, node_id = int(sys.argv[1]), int(sys.argv[2])
failures = []
pipeline = sink = None
for selector in ("path", "target-object"):
    try:
        candidate = Gst.parse_launch(
            "pipewiresrc fd=%d %s=%d do-timestamp=true ! "
            "queue leaky=downstream max-size-buffers=2 ! videoconvert ! "
            "video/x-raw,format=RGB ! appsink name=sink max-buffers=2 drop=true sync=false"
            % (fd, selector, node_id)
        )
        candidate_sink = candidate.get_by_name("sink")
        result = candidate.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("pipeline rejected PLAYING")
        state_result, _, _ = candidate.get_state(5 * Gst.SECOND)
        if state_result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("pipeline failed preroll")
        pipeline, sink = candidate, candidate_sink
        break
    except Exception as exc:
        failures.append("%s: %s" % (selector, exc))
        try: candidate.set_state(Gst.State.NULL)
        except Exception: pass
if pipeline is None or sink is None:
    print(json.dumps({"ready": False, "error": " | ".join(failures)}), flush=True)
    raise SystemExit(2)
print(json.dumps({"ready": True}), flush=True)
try:
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") == "close": break
        destination = Path(request["destination"])
        sample = sink.emit("try-pull-sample", int(float(request["timeout"]) * Gst.SECOND))
        if sample is None:
            print(json.dumps({"ok": False, "error": "timed out waiting for PipeWire frame"}), flush=True)
            continue
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value("width"), caps.get_value("height")
        buffer = sample.get_buffer(); ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            print(json.dumps({"ok": False, "error": "failed to map GStreamer buffer"}), flush=True); continue
        try:
            stride = len(mapped.data) // height
            image = Image.frombytes("RGB", (width, height), bytes(mapped.data), "raw", "RGB", stride, 1)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            image.save(temporary, format="PNG"); os.replace(temporary, destination)
            print(json.dumps({"ok": True, "width": width, "height": height}), flush=True)
        finally:
            buffer.unmap(mapped)
finally:
    pipeline.set_state(Gst.State.NULL)
    os.close(fd)
'''


def _unwrap(value):
    """Recursively unwrap dbus-next Variants without importing at module import."""
    if hasattr(value, "value") and hasattr(value, "signature"):
        return _unwrap(value.value)
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_unwrap(v) for v in value]
    return value


class _PortalClient:
    def __init__(self, *, state_path: Path, request_timeout: float) -> None:
        self.state_path = state_path
        self.request_timeout = float(request_timeout)
        self.bus = None
        self.session_handle: str | None = None
        self.node_id: int | None = None
        self._responses: dict[str, tuple[int, dict]] = {}
        self._waiters: dict[str, asyncio.Future] = {}

    async def connect(self) -> None:
        if self.bus is not None:
            return
        try:
            from dbus_next.aio import MessageBus
            from dbus_next.constants import BusType
        except ModuleNotFoundError as exc:
            raise PortalScreenCastError(
                "Python dependency 'dbus-next' is missing in .venv-rocm"
            ) from exc

        self.bus = await MessageBus(
            bus_type=BusType.SESSION,
            negotiate_unix_fd=True,
        ).connect()
        self.bus.add_message_handler(self._on_message)
        await self._add_match(
            "type='signal',interface='org.freedesktop.portal.Request',member='Response'"
        )
        await self._add_match(
            "type='signal',interface='org.freedesktop.portal.Session',member='Closed'"
        )

    async def _add_match(self, rule: str) -> None:
        from dbus_next import Message
        from dbus_next.constants import MessageType

        assert self.bus is not None
        reply = await self.bus.call(
            Message(
                destination=DBUS_DEST,
                path=DBUS_PATH,
                interface=DBUS_IFACE,
                member="AddMatch",
                signature="s",
                body=[rule],
            )
        )
        if reply.message_type == MessageType.ERROR:
            raise PortalScreenCastError(
                f"D-Bus AddMatch failed: {reply.error_name}: {reply.body}"
            )

    def _on_message(self, message):
        if message.interface == REQUEST_IFACE and message.member == "Response":
            try:
                code = int(message.body[0])
                results = _unwrap(message.body[1])
            except Exception:
                return None
            waiter = self._waiters.pop(message.path, None)
            if waiter is not None and not waiter.done():
                waiter.set_result((code, results))
            else:
                self._responses[message.path] = (code, results)
            return None

        if (
            message.interface == "org.freedesktop.portal.Session"
            and message.member == "Closed"
            and message.path == self.session_handle
        ):
            self.session_handle = None
            self.node_id = None
        return None

    async def _call(self, *, member: str, signature: str, body: list):
        from dbus_next import Message
        from dbus_next.constants import MessageType

        assert self.bus is not None
        reply = await self.bus.call(
            Message(
                destination=PORTAL_DEST,
                path=PORTAL_PATH,
                interface=SCREENCAST_IFACE,
                member=member,
                signature=signature,
                body=body,
            )
        )
        if reply.message_type == MessageType.ERROR:
            raise PortalScreenCastError(
                f"ScreenCast.{member} failed: {reply.error_name}: {reply.body}"
            )
        return reply

    async def _request(self, *, member: str, signature: str, body: list) -> dict:
        reply = await self._call(member=member, signature=signature, body=body)
        if not reply.body:
            raise PortalScreenCastError(f"ScreenCast.{member} returned no request handle")
        handle = str(reply.body[0])

        cached = self._responses.pop(handle, None)
        if cached is not None:
            code, results = cached
        else:
            loop = asyncio.get_running_loop()
            waiter = loop.create_future()
            self._waiters[handle] = waiter
            try:
                code, results = await asyncio.wait_for(
                    waiter,
                    timeout=self.request_timeout,
                )
            except asyncio.TimeoutError as exc:
                self._waiters.pop(handle, None)
                raise PortalScreenCastError(
                    f"ScreenCast.{member} portal response timed out"
                ) from exc

        if int(code) != 0:
            if int(code) == 1:
                reason = "cancelled by user"
            elif int(code) == 2:
                reason = "portal rejected the request"
            else:
                reason = f"portal response code {code}"
            raise PortalScreenCastError(f"ScreenCast.{member}: {reason}")
        return results

    def _token(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex}"

    def _load_restore_token(self) -> str | None:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            token = payload.get("restore_token")
            if isinstance(token, str) and token:
                return token
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        return None

    def _save_restore_token(self, token: str | None) -> None:
        if not token:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temp.write_text(
            json.dumps(
                {
                    "restore_token": token,
                    "updated_at": time.time(),
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        temp.replace(self.state_path)

    async def ensure_session(self) -> None:
        if self.session_handle is not None and self.node_id is not None:
            return
        await self.connect()
        from dbus_next import Variant

        create_options = {
            "handle_token": Variant("s", self._token("heiwa_create")),
            "session_handle_token": Variant("s", self._token("heiwa_session")),
        }
        created = await self._request(
            member="CreateSession",
            signature="a{sv}",
            body=[create_options],
        )
        session = created.get("session_handle")
        if not isinstance(session, str) or not session.startswith("/"):
            raise PortalScreenCastError(
                f"CreateSession returned invalid session handle: {session!r}"
            )
        self.session_handle = session

        select_options = {
            "handle_token": Variant("s", self._token("heiwa_select")),
            "types": Variant("u", 1),  # MONITOR
            "multiple": Variant("b", False),
            "persist_mode": Variant("u", 2),
        }
        restore_token = self._load_restore_token()
        if restore_token:
            select_options["restore_token"] = Variant("s", restore_token)

        await self._request(
            member="SelectSources",
            signature="oa{sv}",
            body=[self.session_handle, select_options],
        )

        start_options = {
            "handle_token": Variant("s", self._token("heiwa_start")),
        }
        started = await self._request(
            member="Start",
            signature="osa{sv}",
            body=[self.session_handle, "", start_options],
        )

        streams = started.get("streams")
        if not isinstance(streams, list) or not streams:
            raise PortalScreenCastError("ScreenCast.Start returned no PipeWire streams")
        first = streams[0]
        if not isinstance(first, list) or len(first) < 1:
            raise PortalScreenCastError(f"invalid PipeWire stream descriptor: {first!r}")
        try:
            self.node_id = int(first[0])
        except (TypeError, ValueError) as exc:
            raise PortalScreenCastError(
                f"invalid PipeWire node id: {first[0]!r}"
            ) from exc

        token = started.get("restore_token")
        if isinstance(token, str) and token:
            self._save_restore_token(token)

    async def open_pipewire_remote(self) -> tuple[int, int]:
        await self.ensure_session()
        assert self.session_handle is not None
        assert self.node_id is not None

        reply = await self._call(
            member="OpenPipeWireRemote",
            signature="oa{sv}",
            body=[self.session_handle, {}],
        )
        if not reply.body:
            raise PortalScreenCastError("OpenPipeWireRemote returned no fd index")
        try:
            fd_index = int(reply.body[0])
            received_fd = int(reply.unix_fds[fd_index])
        except (TypeError, ValueError, IndexError) as exc:
            raise PortalScreenCastError(
                "OpenPipeWireRemote returned an invalid Unix fd"
            ) from exc

        # Own a duplicate independent from dbus-next's Message object lifetime.
        duplicate = os.dup(received_fd)
        try:
            os.close(received_fd)
        except OSError:
            pass
        return duplicate, self.node_id

    async def reset_session(self) -> None:
        self.session_handle = None
        self.node_id = None

    async def close_session(self) -> None:
        if self.bus is None or self.session_handle is None:
            self.session_handle = None; self.node_id = None; return
        from dbus_next import Message
        try:
            await self.bus.call(Message(
                destination=PORTAL_DEST, path=self.session_handle,
                interface="org.freedesktop.portal.Session", member="Close",
            ))
        finally:
            self.session_handle = None; self.node_id = None


class PortalScreenCast:
    """Synchronous body-sensor facade over XDG ScreenCast + PipeWire.

    D-Bus runs on a dedicated asyncio thread. One PipeWire remote fd and one
    system-Python/GStreamer appsink remain alive for the Portal session. Frame
    requests sample that persistent stream instead of repeatedly attaching
    short-lived readers to a Portal-owned node.
    """

    def __init__(
        self,
        *,
        state_path: str | Path = "state/desktop_screencast.json",
        portal_timeout: float = 60.0,
        frame_timeout: float = 8.0,
    ) -> None:
        self.state_path = Path(state_path)
        self.portal_timeout = float(portal_timeout)
        self.frame_timeout = float(frame_timeout)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: _PortalClient | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._worker: subprocess.Popen[str] | None = None

    def available(self) -> bool:
        return bool(shutil.which("gst-launch-1.0"))

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready.clear()
            self._startup_error = None
            self._thread = threading.Thread(
                target=self._thread_main,
                name="heiwa-screencast-portal",
                daemon=True,
            )
            self._thread.start()
            if not self._ready.wait(timeout=5.0):
                raise PortalScreenCastError("ScreenCast portal thread did not start")
            if self._startup_error is not None:
                raise PortalScreenCastError(
                    f"ScreenCast portal thread failed: {self._startup_error}"
                )

    def _thread_main(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._client = _PortalClient(
                state_path=self.state_path,
                request_timeout=self.portal_timeout,
            )
            self._ready.set()
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()

    def _run(self, coroutine, *, timeout: float):
        self._ensure_thread()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError as exc:
            future.cancel()
            raise PortalScreenCastError("ScreenCast portal operation timed out") from exc

    def capture_png(self, destination: Path) -> None:
        if not self.available():
            raise PortalScreenCastError("gst-launch-1.0 is not installed")
        with self._capture_lock:
            self._ensure_stream_worker()
            self._request_worker_frame(destination)

    def _open_remote(self) -> tuple[int, int]:
        self._ensure_thread(); assert self._client is not None
        return self._run(self._client.open_pipewire_remote(), timeout=self.portal_timeout + 5.0)

    def _spawn_stream_worker(self, fd: int, node_id: int) -> subprocess.Popen[str]:
        return subprocess.Popen(
            ("/usr/bin/python3", "-c", _GST_FRAME_WORKER, str(fd), str(node_id)),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, pass_fds=(fd,),
        )

    def _ensure_stream_worker(self) -> None:
        if self._worker is not None and self._worker.poll() is None: return
        self._stop_stream_worker()
        fd, node_id = self._open_remote()
        try:
            worker = self._spawn_stream_worker(fd, node_id)
        finally:
            os.close(fd)
        assert worker.stdout is not None
        ready = worker.stdout.readline()
        try: payload = json.loads(ready)
        except json.JSONDecodeError as exc:
            worker.terminate()
            raise PortalScreenCastError(f"persistent GStreamer worker gave invalid startup response: {ready!r}") from exc
        if not payload.get("ready"):
            worker.terminate()
            raise PortalScreenCastError(f"persistent GStreamer worker failed: {payload.get('error', 'unknown error')}")
        self._worker = worker

    def _request_worker_frame(self, destination: Path) -> None:
        worker = self._worker
        if worker is None or worker.stdin is None or worker.stdout is None:
            raise PortalScreenCastError("persistent GStreamer worker is unavailable")
        destination.parent.mkdir(parents=True, exist_ok=True); destination.unlink(missing_ok=True)
        try:
            worker.stdin.write(json.dumps({"command": "capture", "destination": str(destination), "timeout": self.frame_timeout}) + "\n")
            worker.stdin.flush(); raw = worker.stdout.readline()
            payload = json.loads(raw)
        except (BrokenPipeError, OSError, json.JSONDecodeError) as exc:
            self._stop_stream_worker()
            raise PortalScreenCastError(f"persistent GStreamer frame worker failed: {exc}") from exc
        if not payload.get("ok") or not destination.exists() or destination.stat().st_size <= 0:
            raise PortalScreenCastError(f"persistent PipeWire frame capture failed: {payload.get('error', 'empty frame')}")

    def _stop_stream_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is None: return
        if worker.poll() is None and worker.stdin is not None:
            try: worker.stdin.write('{"command":"close"}\n'); worker.stdin.flush(); worker.wait(timeout=3.0)
            except Exception:
                worker.terminate()
                try: worker.wait(timeout=1.0)
                except subprocess.TimeoutExpired: worker.kill()
        for stream in (worker.stdin, worker.stdout, worker.stderr):
            if stream is not None:
                try: stream.close()
                except OSError: pass

    def close(self) -> None:
        with self._capture_lock:
            self._stop_stream_worker()
            if self._client is not None and self._loop is not None:
                try: self._run(self._client.close_session(), timeout=3.0)
                except Exception: pass
            if self._loop is not None: self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None: self._thread.join(timeout=2.0)
            self._thread = None; self._loop = None; self._client = None
