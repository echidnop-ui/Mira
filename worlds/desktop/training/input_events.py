from __future__ import annotations

from dataclasses import dataclass, field
import errno
import fcntl
from pathlib import Path
import select
import struct
import threading
import time
from typing import Iterable


EV_SYN, EV_KEY, EV_REL, EV_ABS = 0, 1, 2, 3
REL_X, REL_Y, REL_WHEEL, REL_HWHEEL = 0, 1, 8, 6
ABS_X, ABS_Y = 0, 1
BTN_LEFT, BTN_RIGHT, BTN_MIDDLE = 272, 273, 274
FEEDBACK_KEYS = {191: "F21", 192: "F22", 193: "F23", 194: "F24"}
MODIFIERS = {29: 1, 97: 1, 42: 2, 54: 2, 56: 4, 100: 4, 125: 8, 126: 8}
EVIOCSCLOCKID = 0x400445A0
EVIOCGRAB = 0x40044590
_INPUT_EVENT = struct.Struct("llHHi")


class InputCaptureUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RawInputEvent:
    timestamp_ns: int
    device: str
    device_name: str
    event_type: int
    code: int
    value: int


@dataclass(slots=True)
class PointerState:
    width: int
    height: int
    x: float
    y: float
    coordinate_source: str = "integrated_relative"
    buttons_pressed: dict[int, int] = field(default_factory=dict)
    modifiers_pressed: set[int] = field(default_factory=set)

    def relative(self, dx: int, dy: int) -> None:
        self.x = min(max(self.x + dx, 0.0), max(self.width - 1, 0))
        self.y = min(max(self.y + dy, 0.0), max(self.height - 1, 0))

    def normalized(self) -> tuple[float, float]:
        return self.x / max(self.width - 1, 1), self.y / max(self.height - 1, 1)


def permission_help() -> str:
    return (
        "No readable physical /dev/input/event* devices. Add the current user to the input group, then log out and back in:\n"
        "  sudo usermod -aG input \"$USER\"\n"
        "Verify after login with:\n  id -nG | tr ' ' '\\n' | grep '^input$'"
    )


def _device_name(path: Path) -> str:
    sys_name = Path("/sys/class/input") / path.name / "device/name"
    try: return sys_name.read_text(encoding="utf-8").strip()
    except OSError: return path.name


def _supports_key(path: str, key_code: int) -> bool:
    capability = Path("/sys/class/input") / Path(path).name / "device/capabilities/key"
    try: words = capability.read_text(encoding="utf-8").split()
    except OSError: return False
    # sysfs prints unsigned-long words high-to-low without zero padding.
    word_index, bit_index = divmod(key_code, 64)
    if word_index >= len(words): return False
    try: word = int(words[-1 - word_index], 16)
    except ValueError: return False
    return bool(word & (1 << bit_index))


def discover_devices(root: Path = Path("/dev/input")) -> tuple[Path, ...]:
    readable: list[Path] = []
    denied = 0
    for path in sorted(root.glob("event*")):
        name = _device_name(path).lower()
        # Ignore Heiwa's own actuation devices and unrelated virtual devices,
        # but keep Input Remapper's virtual output: the two dedicated feedback
        # buttons intentionally arrive there as F21..F24.
        is_input_remapper = "input-remapper" in name or "input remapper" in name
        if "ydotool" in name or "heiwa" in name or ("virtual" in name and not is_input_remapper):
            continue
        try:
            descriptor = path.open("rb", buffering=0); descriptor.close(); readable.append(path)
        except PermissionError: denied += 1
        except OSError: continue
    if not readable:
        detail = permission_help() if denied else "No physical evdev devices were found under /dev/input."
        raise InputCaptureUnavailable(detail)
    return tuple(readable)


class EvdevInputCollector:
    """Non-grabbing Linux evdev observer with kernel monotonic timestamps."""
    def __init__(self, *, screen_width: int, screen_height: int, devices: Iterable[Path] | None = None) -> None:
        self.devices = tuple(devices) if devices is not None else discover_devices()
        self.pointer = PointerState(screen_width, screen_height, screen_width / 2, screen_height / 2)
        self._streams: list[object] = []; self._names: dict[int, tuple[str, str]] = {}
        self._events: list[RawInputEvent] = []; self._lock = threading.Lock(); self._stop = threading.Event()
        self._modifiers: set[int] = set(); self._thread: threading.Thread | None = None

    def _open_device(self, path: Path) -> bool:
        """Attach one evdev source without letting a broken endpoint disable all input."""
        if any(self._names.get(stream.fileno(), (None, None))[0] == str(path) for stream in self._streams):
            return True
        try:
            stream = path.open("rb", buffering=0)
        except (PermissionError, FileNotFoundError, OSError):
            return False
        try:
            fcntl.ioctl(stream.fileno(), EVIOCSCLOCKID, struct.pack("i", time.CLOCK_MONOTONIC))
        except OSError:
            # Some virtual/HID endpoints do not implement EVIOCSCLOCKID.  They
            # must not disable capture from working devices such as the
            # Input Remapper keyboard that carries F21..F24.
            stream.close()
            return False
        self._streams.append(stream)
        self._names[stream.fileno()] = (str(path), _device_name(path))
        return True

    def _rescan_devices(self) -> None:
        try:
            candidates = discover_devices()
        except InputCaptureUnavailable:
            return
        for path in candidates:
            self._open_device(path)

    def start(self) -> None:
        for path in self.devices:
            self._open_device(path)
        if not self._streams:
            raise InputCaptureUnavailable(
                permission_help() + "\nReadable devices exist, but none could be attached as monotonic evdev sources."
            )
        self._thread = threading.Thread(target=self._run, name="heiwa-evdev-recorder", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        next_rescan = time.monotonic() + 1.0
        while not self._stop.is_set():
            if time.monotonic() >= next_rescan:
                self._rescan_devices()
                next_rescan = time.monotonic() + 1.0
            streams = tuple(self._streams)
            if not streams:
                time.sleep(0.2)
                continue
            try:
                readable, _, _ = select.select(streams, [], [], 0.2)
            except (OSError, ValueError):
                continue
            for stream in readable:
                try:
                    data = stream.read(_INPUT_EVENT.size * 64)
                except OSError as exc:
                    if exc.errno in {errno.EAGAIN, errno.ENODEV, errno.EBADF}:
                        try: self._streams.remove(stream)
                        except ValueError: pass
                        self._names.pop(stream.fileno(), None)
                        try: stream.close()
                        except OSError: pass
                        continue
                    continue
                identity = self._names.get(stream.fileno())
                if identity is None:
                    continue
                path, name = identity
                for offset in range(0, len(data) - _INPUT_EVENT.size + 1, _INPUT_EVENT.size):
                    seconds, micros, event_type, code, value = _INPUT_EVENT.unpack_from(data, offset)
                    timestamp_ns = seconds * 1_000_000_000 + micros * 1_000
                    event = RawInputEvent(timestamp_ns, path, name, event_type, code, value)
                    with self._lock:
                        self._events.append(event)
                        if len(self._events) > 100_000:
                            del self._events[:50_000]

    def drain(self, before_ns: int, after_ns: int) -> tuple[RawInputEvent, ...]:
        if after_ns <= before_ns:
            raise ValueError("frame timestamps must be increasing")
        # The evdev reader runs on a separate thread. A kernel event can have a
        # monotonic timestamp just before the previous frame boundary but be
        # appended to `_events` just after that frame called drain(). Filtering
        # again by `before_ns` would then drop that event forever on the next
        # drain. This is especially visible with Input Remapper's forwarded
        # mouse endpoint: real human REL_X/REL_Y packets can disappear before
        # NeuralDesktopWorldAdapter gets a chance to interpret them as feedback.
        #
        # Queue membership already guarantees exactly-once delivery. Therefore
        # use only the upper timestamp watermark here: events that arrived late
        # are delivered on the next poll with their original kernel timestamp,
        # while events already returned are removed from the queue and cannot be
        # duplicated.
        with self._lock:
            selected = tuple(event for event in self._events if event.timestamp_ns < after_ns)
            self._events = [event for event in self._events if event.timestamp_ns >= after_ns]
        return selected

    def grab_feedback_keyboards(self, enabled: bool) -> None:
        """Temporarily keep rating keys from reaching the focused application."""
        # EVIOCGRAB is declared _IOW but evdev tests the ioctl argument itself
        # as 0/1. Passing struct.pack(0) supplies a non-null buffer address and
        # is therefore interpreted as another grab, not an ungrab.
        value = int(enabled)
        candidates = [
            stream for stream in self._streams
            if _supports_key(self._names[stream.fileno()][0], 2)
            and _supports_key(self._names[stream.fileno()][0], 3)
        ]
        if not enabled:
            for stream in candidates:
                try: fcntl.ioctl(stream.fileno(), EVIOCGRAB, value)
                except OSError: pass
            return

        # GNOME may briefly own a HID keyboard endpoint while finishing the
        # preceding key transition. Acquire the set transactionally: never
        # leave half the keyboards grabbed, and tolerate that short hand-off.
        last_error: tuple[str, OSError] | None = None
        for attempt in range(11):
            acquired: list[object] = []
            for stream in candidates:
                path, _ = self._names[stream.fileno()]
                try:
                    fcntl.ioctl(stream.fileno(), EVIOCGRAB, value)
                    acquired.append(stream)
                except OSError as exc:
                    last_error = (path, exc)
                    for owned in acquired:
                        try: fcntl.ioctl(owned.fileno(), EVIOCGRAB, 0)
                        except OSError: pass
                    break
            else:
                return
            if attempt < 10:
                time.sleep(0.05)
        assert last_error is not None
        path, exc = last_error
        raise InputCaptureUnavailable(
            f"cannot exclusively capture feedback keys on {path} after 0.5s retry: {exc}. "
            "Check that no other live_guidance/evtest process is running."
        ) from exc

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout=1.0)
        for stream in self._streams:
            try: stream.close()
            except OSError: pass
        self._streams.clear()

    def __enter__(self) -> "EvdevInputCollector": self.start(); return self
    def __exit__(self, *_: object) -> None: self.close()


def aggregate_events(events: Iterable[RawInputEvent], pointer: PointerState, *, movement_threshold: float = 3.0, trajectory_interval_ns: int = 12_000_000) -> list[dict[str, object]]:
    """Collapse motion noise while retaining trajectory, buttons, wheel and physical keys."""
    ordered = sorted(events, key=lambda event: event.timestamp_ns)
    output: list[dict[str, object]] = []; trajectory: list[list[float | int]] = []
    movement_start: tuple[float, float, int] | None = None; last_sample = -1
    modifiers = pointer.modifiers_pressed
    typed_key_presses: list[tuple[int, int]] = []

    def flush_motion(end_ns: int) -> None:
        nonlocal movement_start, trajectory, last_sample
        if movement_start is None: return
        sx, sy, start_ns = movement_start; distance = ((pointer.x - sx) ** 2 + (pointer.y - sy) ** 2) ** 0.5
        if distance >= movement_threshold:
            x, y = pointer.normalized()
            output.append({"action": "move_pointer", "timestamp_ns": end_ns, "timestamp": end_ns / 1e9,
                "parameters": [x, y, (end_ns - start_ns) / 1e9, distance], "target_xy": [x, y],
                "motor_event": {"start_position": [sx / max(pointer.width - 1, 1), sy / max(pointer.height - 1, 1)],
                    "end_position": [x, y], "duration_ns": end_ns - start_ns, "trajectory": trajectory,
                    "coordinate_source": pointer.coordinate_source}})
        movement_start = None; trajectory = []; last_sample = -1

    for event in ordered:
        if event.event_type == EV_REL and event.code in {REL_X, REL_Y}:
            if movement_start is None: movement_start = (pointer.x, pointer.y, event.timestamp_ns)
            pointer.relative(event.value if event.code == REL_X else 0, event.value if event.code == REL_Y else 0)
            if last_sample < 0 or event.timestamp_ns - last_sample >= trajectory_interval_ns:
                x, y = pointer.normalized(); trajectory.append([event.timestamp_ns, x, y]); last_sample = event.timestamp_ns
            continue
        flush_motion(event.timestamp_ns)
        if event.event_type == EV_REL and event.code in {REL_WHEEL, REL_HWHEEL}:
            output.append({"action": "scroll", "timestamp_ns": event.timestamp_ns, "timestamp": event.timestamp_ns / 1e9,
                "parameters": [event.value if event.code == REL_HWHEEL else 0, event.value if event.code == REL_WHEEL else 0, 0, 0],
                "motor_event": {"axis": "horizontal" if event.code == REL_HWHEEL else "vertical", "delta": event.value}})
        elif event.event_type == EV_ABS and event.code in {ABS_X, ABS_Y}:
            # Raw ABS range is device-specific and retained without pretending it is screen geometry.
            output.append({"action": "move_pointer", "timestamp_ns": event.timestamp_ns, "timestamp": event.timestamp_ns / 1e9,
                "parameters": [0, 0, 0, 0], "motor_event": {"absolute_axis": event.code, "raw_value": event.value, "coordinate_source": "device_absolute_raw"}, "trainable": False})
        elif event.event_type == EV_KEY:
            if event.code in MODIFIERS:
                (modifiers.add if event.value else modifiers.discard)(event.code)
            modifier_mask = sum(MODIFIERS[code] for code in modifiers)
            if event.code in {BTN_LEFT, BTN_RIGHT, BTN_MIDDLE}:
                button = {BTN_LEFT: "left", BTN_RIGHT: "right", BTN_MIDDLE: "middle"}[event.code]
                if event.value == 1: pointer.buttons_pressed[event.code] = event.timestamp_ns
                pressed_at = pointer.buttons_pressed.get(event.code)
                duration_ns = None if pressed_at is None or event.value != 0 else event.timestamp_ns - pressed_at
                if event.value == 0: pointer.buttons_pressed.pop(event.code, None)
                x, y = pointer.normalized(); supported = button in {"left", "right"}
                output.append({"action": f"{button}_click" if supported and event.value == 1 else "noop", "timestamp_ns": event.timestamp_ns,
                    "timestamp": event.timestamp_ns / 1e9, "parameters": [x, y, float(event.value), 0], "target_xy": [x, y],
                    "trainable": supported and event.value == 1, "motor_event": {"kind": "mouse_button", "button": button,
                        "state": "press" if event.value else "release", "value": event.value,
                        "press_timestamp_ns": pressed_at, "press_release_duration_ns": duration_ns,
                        "modifiers": sorted(modifiers),
                        "coordinate_source": pointer.coordinate_source, "language_teacher": False}})
            elif event.code < BTN_LEFT:
                is_feedback = event.code in FEEDBACK_KEYS
                if event.value == 1 and event.code not in MODIFIERS and not is_feedback:
                    typed_key_presses.append((event.timestamp_ns, event.code))
                output.append({"action": "key_press", "timestamp_ns": event.timestamp_ns, "timestamp": event.timestamp_ns / 1e9,
                    "parameters": [event.code, event.value, modifier_mask, 0],
                    "trainable": event.value in {0, 1} and not is_feedback,
                    "motor_event": {"kind": "physical_key", "code": event.code,
                        "state": {0: "release", 1: "press", 2: "repeat"}.get(event.value, "unknown"),
                        "modifiers": sorted(modifiers), "layout_independent": True,
                        "feedback_command": FEEDBACK_KEYS.get(event.code), "language_teacher": False}})
    if ordered: flush_motion(ordered[-1].timestamp_ns)
    if len(typed_key_presses) >= 2:
        start_ns, end_ns = typed_key_presses[0][0], typed_key_presses[-1][0]
        output.append({"action": "type_text", "timestamp_ns": end_ns, "timestamp": end_ns / 1e9,
            "parameters": [len(typed_key_presses), (end_ns - start_ns) / 1e9, 0, 0], "target_xy": None,
            "trainable": True, "motor_event": {"kind": "physical_key_sequence",
                "key_codes": [code for _, code in typed_key_presses], "duration_ns": end_ns - start_ns,
                "layout_independent": True, "text_content_recorded": False, "language_teacher": False}})
    return output
