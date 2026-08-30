from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

from worlds.desktop.portal_screencast import PortalScreenCast
from worlds.desktop.training.dataset import AtomicManifestWriter, DemonstrationTransition
from worlds.desktop.training.input_events import EvdevInputCollector, InputCaptureUnavailable, aggregate_events


def _compress(source: Path, destination: Path, width: int, height: int) -> tuple[int, int]:
    script = (
        "from PIL import Image; import sys; im=Image.open(sys.argv[1]).convert('RGB'); "
        "print(*im.size); im.thumbnail((int(sys.argv[3]),int(sys.argv[4]))); "
        "im.save(sys.argv[2],format='WEBP',quality=78,method=4)"
    )
    result = subprocess.run(("/usr/bin/python3", "-c", script, str(source), str(destination), str(width), str(height)),
                            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    native = result.stdout.split(); return int(native[0]), int(native[1])


def _external_events(path: Path | None, offset: int) -> tuple[list[dict[str, object]], int]:
    if path is None or not path.exists(): return [], offset
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(offset); chunk = stream.read(); offset = stream.tell()
    return [json.loads(line) for line in chunk.splitlines() if line.strip()], offset


def _apply_controlled_event(event: dict[str, object], controlled: list[dict[str, object]]) -> None:
    """Replace drifting Wayland REL targets with generator-known click geometry."""
    if event.get("action") == "move_pointer":
        event["trainable"] = False
        motor = event.get("motor_event") if isinstance(event.get("motor_event"), dict) else {}
        motor["school_absolute_target_supervision"] = False
        event["motor_event"] = motor
        return
    timestamp = int(event.get("timestamp_ns", 0)); action = str(event.get("action", ""))
    candidates = [row for row in controlled if row.get("action") == action
                  and abs(int(row.get("timestamp_ns", 0)) - timestamp) <= 300_000_000]
    if not candidates: return
    match = min(candidates, key=lambda row: abs(int(row["timestamp_ns"]) - timestamp))
    target, bbox = match.get("target_xy"), match.get("target_bbox")
    if isinstance(target, list) and len(target) == 2:
        event["target_xy"] = [float(target[0]), float(target[1])]
        parameters = list(event.get("parameters", [0, 0, 0, 0])); parameters[:2] = event["target_xy"]
        event["parameters"] = parameters
    if isinstance(bbox, list) and len(bbox) == 4:
        event["controlled_target_bbox"] = [float(value) for value in bbox]


def _attach_pixel_supervision(event: dict[str, object], state_path: Path | None) -> None:
    if state_path is None:
        return
    try: state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return
    text, bbox = state.get("message_text"), state.get("message_bbox")
    if not isinstance(text, str) or not isinstance(bbox, list) or len(bbox) != 4: return
    teacher: dict[str, object] = {"pixel_supervision": True, "language_teacher": False,
                                  "provenance": "controlled_visual_school",
                                  "visual_text": text, "region_bbox": bbox}
    controlled_bbox = event.get("controlled_target_bbox")
    if isinstance(controlled_bbox, list) and len(controlled_bbox) == 4:
        teacher["motor_bbox"] = controlled_bbox
        teacher["motor_role"] = "controlled_widget"
    target = event.get("target_xy")
    if isinstance(target, (list, tuple)) and len(target) == 2:
        x, y = float(target[0]), float(target[1])
        for name in ("composer_bbox", "send_bbox"):
            candidate = state.get(name)
            if (isinstance(candidate, list) and len(candidate) == 4
                    and candidate[0] <= x <= candidate[2] and candidate[1] <= y <= candidate[3]):
                teacher["motor_bbox"] = candidate; teacher["motor_role"] = name.removesuffix("_bbox"); break
    event["teacher"] = teacher


def _transition(before: Path, after: Path, root: Path, event: dict[str, object], before_ns: int, after_ns: int) -> DemonstrationTransition:
    action_ns = int(event.get("timestamp_ns", int(float(event.get("timestamp", (before_ns + after_ns) / 2e9)) * 1e9)))
    if not before_ns < action_ns < after_ns: raise ValueError("action timestamp is not between its frame timestamps")
    parameters = tuple(float(value) for value in event.get("parameters", [0, 0, 0, 0]))
    if len(parameters) != 4: raise ValueError("action parameters must contain four values")
    target = event.get("target_xy")
    return DemonstrationTransition(
        before=str(before.relative_to(root)), after=str(after.relative_to(root)), action=str(event["action"]),
        parameters=parameters, timestamp=action_ns / 1e9, frame_before_ns=before_ns,
        action_timestamp_ns=action_ns, frame_after_ns=after_ns,
        target_xy=None if target is None else (float(target[0]), float(target[1])),
        trainable=bool(event.get("trainable", True)),
        motor_event=event.get("motor_event") if isinstance(event.get("motor_event"), dict) else None,
        teacher=event.get("teacher") if isinstance(event.get("teacher"), dict) else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Record synchronized pixel/input/pixel Desktop demonstrations")
    parser.add_argument("--output", type=Path, default=Path("state/desktop_demonstrations"))
    parser.add_argument("--interval", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=512); parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--capture-input", action="store_true", help="capture physical devices directly through Linux evdev")
    parser.add_argument("--action-log", type=Path, help="offline/debug JSONL input instead of evdev")
    parser.add_argument("--pixel-supervision-state", type=Path, help="controlled renderer state; never read in inference")
    parser.add_argument("--controlled-events", type=Path, help="controlled renderer interaction geometry")
    args = parser.parse_args()
    if args.capture_input and args.action_log is not None: parser.error("choose --capture-input or --action-log")
    args.output.mkdir(parents=True, exist_ok=True); frames = args.output / "frames"; frames.mkdir(exist_ok=True)
    portal = PortalScreenCast(); stopped = False
    def stop(*_: object) -> None:
        nonlocal stopped; stopped = True
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    print("Recording synchronized pixel/input/pixel transitions; Ctrl+C checkpoints and stops.")
    raw = args.output / ".capture.png"; portal.capture_png(raw)
    previous_ns = time.monotonic_ns(); previous = frames / f"{previous_ns:020d}.webp"
    native_width, native_height = _compress(raw, previous, args.width, args.height)
    raw.unlink(missing_ok=True); sequence = 1; action_offset = 0; controlled_offset = 0
    collector: EvdevInputCollector | None = None
    try:
        if args.capture_input:
            collector = EvdevInputCollector(screen_width=native_width, screen_height=native_height); collector.start()
            print(json.dumps({"input_backend": "linux-evdev", "devices": [str(path) for path in collector.devices],
                              "relative_pointer_coordinates": "integrated; Wayland compositor position is unavailable"}))
        with AtomicManifestWriter(args.output / "manifest.jsonl") as manifest:
            while not stopped:
                time.sleep(max(args.interval, 0.05)); portal.capture_png(raw)
                current_ns = time.monotonic_ns(); current = frames / f"{current_ns:020d}.webp"
                _compress(raw, current, args.width, args.height); raw.unlink(missing_ok=True)
                if collector is not None:
                    events = aggregate_events(collector.drain(previous_ns, current_ns), collector.pointer)
                else:
                    events, action_offset = _external_events(args.action_log, action_offset)
                    events = [event for event in events if previous_ns < int(event.get("timestamp_ns", 0)) < current_ns]
                if args.controlled_events is not None:
                    controlled, controlled_offset = _external_events(args.controlled_events, controlled_offset)
                    for event in events: _apply_controlled_event(event, controlled)
                if not events:
                    midpoint = (previous_ns + current_ns) // 2
                    events = [{"action": "noop", "timestamp_ns": midpoint, "parameters": [0, 0, 0, 0],
                               "motor_event": {"kind": "passive_interval", "language_teacher": False}}]
                for event in events:
                    _attach_pixel_supervision(event, args.pixel_supervision_state)
                    manifest.append(_transition(previous, current, args.output, event, previous_ns, current_ns))
                previous, previous_ns, sequence = current, current_ns, sequence + 1
    except InputCaptureUnavailable as exc:
        print(f"INPUT_CAPTURE_UNAVAILABLE:\n{exc}"); return 2
    finally:
        if collector is not None: collector.close()
        portal.close()
        raw.unlink(missing_ok=True)
    print(json.dumps({"frames": sequence, "manifest": str(args.output / "manifest.jsonl"), "stopped_cleanly": True}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
