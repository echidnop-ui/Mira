from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from worlds.desktop.neural import ACTION_KINDS, NeuralDesktopBody, NeuralDesktopConfig, select_device
from worlds.desktop.portal_screencast import PortalScreenCast, PortalScreenCastError
from worlds.desktop.training.dataset import _pillow_image, load_rgb


SCENES = (
    ("blank", "", "center"),
    ("heiwa", "HEIWA TEST 7319", "center"),
    ("red_dragon", "RED DRAGON 7319", "center"),
    ("blue_tiger", "BLUE TIGER 4826", "center"),
    ("left", "HEIWA TEST 7319", "left"),
    ("right", "HEIWA TEST 7319", "right"),
    ("russian_heiwa", "ПРИВЕТ HEIWA 8427", "center"),
    ("russian_dragon", "СИНИЙ ДРАКОН 5931", "center"),
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _presenter(command_path: Path, telemetry_path: Path) -> int:
    # This process only draws known test stimuli. It never inspects the desktop.
    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk, GLib, Gtk

    window = Gtk.Window(title="Heiwa pixel vision diagnostic")
    window.set_decorated(False)
    fixed = Gtk.Fixed(); label = Gtk.Label(); pulse = Gtk.Label(label="●")
    pulse.set_name("pulse")
    fixed.put(label, 0, 0); fixed.put(pulse, 8, 8); window.add(fixed)
    css = Gtk.CssProvider()
    css.load_from_data(
        b"window, fixed { background-color: #040405; } "
        b"label { color: #f5faff; font-family: 'DejaVu Sans'; font-weight: bold; font-size: 132px; } "
        b"#pulse { color: #ff3355; font-size: 18px; }"
    )
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(), css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )
    state: dict[str, Any] = {"id": -1, "text": "", "align": "center"}
    pulse_step = [0]

    def reload_state() -> bool:
        try:
            payload = json.loads(command_path.read_text(encoding="utf-8"))
            if payload != state:
                state.clear(); state.update(payload); label.set_text(str(state.get("text", "")))
                GLib.idle_add(place_label)
        except (OSError, json.JSONDecodeError):
            pass
        return True

    def place_label() -> bool:
        width, height = fixed.get_allocated_width(), fixed.get_allocated_height()
        text = str(state.get("text", "")); bbox = None
        if text:
            minimum, natural = label.get_preferred_size()
            text_width, text_height = natural.width, natural.height
            align = str(state.get("align", "center"))
            if align == "left": x = int(width * 0.035)
            elif align == "right": x = int(width * 0.965 - text_width)
            else: x = int((width - text_width) / 2)
            y = int((height - text_height) / 2)
            fixed.move(label, x, y)
            bbox = [x, y, x + text_width, y + text_height]
        _atomic_json(telemetry_path, {
            "id": int(state.get("id", -1)), "text": text, "align": state.get("align"),
            "window_size": [width, height], "text_bbox_px": bbox,
            "rendered_monotonic_ns": time.monotonic_ns(),
        })
        return False

    def animate_pulse() -> bool:
        width, height = fixed.get_allocated_width(), fixed.get_allocated_height()
        pulse_step[0] += 1
        pulse.set_text("●" if pulse_step[0] % 2 else "○")
        fixed.move(pulse, 8 + (pulse_step[0] % 24) * 5, max(8, height - 32))
        return True

    window.connect("destroy", Gtk.main_quit)
    # Maximized (not direct-scanout fullscreen) keeps GNOME's composited
    # PipeWire monitor stream producing damage buffers.
    window.maximize(); window.present(); window.show_all()
    GLib.timeout_add(50, reload_state); GLib.timeout_add(250, place_label)
    GLib.timeout_add(80, animate_pulse)
    Gtk.main()
    return 0


def _wait_presented(path: Path, scene_id: int, *, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("id", -1)) == scene_id: return payload
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        time.sleep(0.05)
    raise RuntimeError(f"diagnostic presenter did not render scene {scene_id}")


def _save_tensor_image(frame: Tensor, path: Path) -> None:
    Image = _pillow_image()
    pixels = frame.detach().clamp(0, 1).mul(255).byte()[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(pixels, mode="RGB").save(path)


def _crop(frame: Tensor, bbox: Tensor) -> Tensor:
    batch = frame.shape[0]
    unit_x = torch.linspace(0, 1, 128, device=frame.device).view(1, 1, 128)
    unit_y = torch.linspace(0, 1, 32, device=frame.device).view(1, 32, 1)
    x0, y0, x1, y1 = bbox.unbind(dim=-1)
    xs = x0[:, None, None] + (x1 - x0)[:, None, None] * unit_x
    ys = y0[:, None, None] + (y1 - y0)[:, None, None] * unit_y
    grid = torch.stack((xs.expand(batch, 32, 128) * 2 - 1,
                        ys.expand(batch, 32, 128) * 2 - 1), dim=-1)
    return F.grid_sample(frame.float(), grid, align_corners=True)


def _bbox_from_presenter(telemetry: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float] | None:
    raw = telemetry.get("text_bbox_px"); window = telemetry.get("window_size")
    if not isinstance(raw, list) or not isinstance(window, list) or len(raw) != 4 or len(window) != 2:
        return None
    logical_width, logical_height = float(window[0]), float(window[1])
    # GTK reports logical pixels while PipeWire may report physical pixels.
    # Aspect-ratio equality is enough; normalization cancels the scale factor.
    if logical_width <= 0 or logical_height <= 0 or abs(logical_width / logical_height - width / height) > 0.02:
        return None
    x0, y0, x1, y1 = map(float, raw)
    pad_x, pad_y = logical_width * 0.015, logical_height * 0.03
    return (max(0.0, (x0 - pad_x) / logical_width), max(0.0, (y0 - pad_y) / logical_height),
            min(1.0, (x1 + pad_x) / logical_width), min(1.0, (y1 + pad_y) / logical_height))


def _decode(logits: Tensor, expected: str) -> dict[str, Any]:
    probabilities = logits.float().softmax(-1)[0]
    ids = probabilities.argmax(-1); maxima = probabilities.max(-1).values
    raw_bytes: list[int] = []; eos_index = None; positions = []
    for index, token in enumerate(ids.tolist()):
        top_probability, top_ids = probabilities[index].topk(3)
        positions.append({"index": index, "token": token, "probability": float(maxima[index]),
                          "top3": [[int(code), float(prob)] for code, prob in zip(top_ids, top_probability)]})
        if token == 2:
            eos_index = index; break
        if 4 <= token < 260: raw_bytes.append(token - 4)
    decoded = bytes(raw_bytes).decode("utf-8", errors="replace")
    expected_bytes = expected.encode("utf-8")
    byte_correct = sum(a == b for a, b in zip(raw_bytes, expected_bytes))
    byte_accuracy = byte_correct / max(len(raw_bytes), len(expected_bytes), 1)
    return {
        "expected_text": expected, "decoded_token_ids": ids.tolist(),
        "decoded_bytes": raw_bytes, "decoded_utf8": decoded,
        "sequence_confidence": float(maxima[:(eos_index + 1 if eos_index is not None else len(maxima))].mean()),
        "eos": eos_index is not None, "eos_index": eos_index,
        "byte_probabilities": positions, "exact_match": decoded == expected,
        "character_error_rate": _cer(decoded, expected), "byte_accuracy": byte_accuracy,
    }


def _cer(predicted: str, expected: str) -> float:
    previous = list(range(len(predicted) + 1))
    for row, expected_char in enumerate(expected, 1):
        current = [row]
        for column, predicted_char in enumerate(predicted, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (expected_char != predicted_char)))
        previous = current
    return previous[-1] / max(len(expected), 1)


def _distance(left: Tensor, right: Tensor) -> dict[str, float]:
    left, right = left.float().flatten(), right.float().flatten()
    return {"cosine_distance": float(1 - F.cosine_similarity(left, right, dim=0)),
            "l2": float(torch.linalg.vector_norm(left - right))}


def _known_patch_feature(state, bbox: tuple[float, float, float, float] | None) -> Tensor:
    if bbox is None: return state.patch_tokens.mean(dim=1)[0]
    gh, gw = state.grid_size; x0, y0, x1, y1 = bbox
    yy, xx = torch.meshgrid(torch.arange(gh, device=state.patch_tokens.device),
                            torch.arange(gw, device=state.patch_tokens.device), indexing="ij")
    centers_x, centers_y = (xx.flatten() + .5) / gw, (yy.flatten() + .5) / gh
    mask = (centers_x >= x0) & (centers_x <= x1) & (centers_y >= y0) & (centers_y <= y1)
    return state.patch_tokens[0, mask].mean(dim=0) if bool(mask.any()) else state.patch_tokens.mean(dim=1)[0]


def _capture_with_retry(portal: PortalScreenCast, destination: Path, *, attempts: int = 3) -> None:
    last_error: PortalScreenCastError | None = None
    for attempt in range(attempts):
        try:
            portal.capture_png(destination); return
        except PortalScreenCastError as exc:
            last_error = exc
            if attempt == 1: portal.close()
            if attempt + 1 < attempts: time.sleep(0.2)
    assert last_error is not None
    raise last_error


def run_diagnostic(checkpoint: Path, output_root: Path, *, settle_seconds: float) -> Path:
    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    command_path, presenter_path = run_dir / "presenter_command.json", run_dir / "presenter_state.json"
    _atomic_json(command_path, {"id": -1, "text": "", "align": "center"})
    project_root = Path(__file__).resolve().parents[2]
    presenter = subprocess.Popen(("/usr/bin/python3", "-m", "worlds.desktop.vision_diagnostic", "--presenter",
                                  str(command_path), str(presenter_path)), cwd=project_root)
    portal = PortalScreenCast(); payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = NeuralDesktopConfig(**payload["config"]); device = select_device("auto")
    model = NeuralDesktopBody(config).to(device); model.load_state_dict(payload["model"], strict=False); model.eval()
    scenes: dict[str, dict[str, Any]] = {}; previous_native = None; previous_embedding = None
    try:
        for scene_id, (name, expected, align) in enumerate(SCENES):
            _atomic_json(command_path, {"id": scene_id, "text": expected, "align": align})
            telemetry = _wait_presented(presenter_path, scene_id); time.sleep(settle_seconds)
            scene_dir = run_dir / name; scene_dir.mkdir()
            frame_path = scene_dir / "frame.png"; _capture_with_retry(portal, frame_path)
            timestamp_ns = time.monotonic_ns()
            Image = _pillow_image()
            with Image.open(frame_path) as image:
                native_width, native_height = image.size
                native_raw = bytearray(image.convert("RGB").tobytes())
            native = torch.frombuffer(native_raw, dtype=torch.uint8).reshape(native_height, native_width, 3).float().div(255)
            frame = load_rgb(frame_path, width=config.image_width, height=config.image_height).unsqueeze(0).to(device)
            torch.cuda.synchronize() if device.type == "cuda" else None
            started = time.perf_counter()
            with torch.inference_mode():
                state = model.observe(frame); motor = model.propose(state)
                predicted_logits = model.read_region(frame, motor.target_bbox)
            torch.cuda.synchronize() if device.type == "cuda" else None
            inference_ms = (time.perf_counter() - started) * 1000
            controlled_bbox = _bbox_from_presenter(telemetry, native_width, native_height)
            if controlled_bbox is None:
                controlled_bbox = tuple(float(v) for v in motor.target_bbox[0].cpu())
            controlled_bbox_tensor = torch.tensor([controlled_bbox], device=device)
            with torch.inference_mode():
                controlled_logits = model.read_region(frame, controlled_bbox_tensor)
                predicted_crop = _crop(frame, motor.target_bbox)
                controlled_crop = _crop(frame, controlled_bbox_tensor)
                foveal_feature = model.foveal_reader(controlled_crop).mean(dim=(2, 3))[0]
            _save_tensor_image(predicted_crop, scene_dir / "predicted_foveal_crop.png")
            _save_tensor_image(controlled_crop, scene_dir / "controlled_text_crop.png")
            embedding = state.latent[0]
            change = None if previous_native is None else {
                "mean_absolute": float((native - previous_native).abs().mean()),
                "changed_pixel_fraction": float((native - previous_native).abs().amax(dim=-1).gt(2 / 255).float().mean()),
            }
            text_bbox_raw = telemetry.get("text_bbox_px")
            logical_size = telemetry.get("window_size")
            scale_y = (native_height / float(logical_size[1])
                       if isinstance(logical_size, list) and len(logical_size) == 2 and float(logical_size[1]) > 0 else 1.0)
            character_height_native = (0 if not isinstance(text_bbox_raw, list)
                                       else max(0.0, float(text_bbox_raw[3] - text_bbox_raw[1]) * scale_y))
            controlled_native_height = max(1.0, (controlled_bbox[3] - controlled_bbox[1]) * native_height)
            reader_character_height = 32 * character_height_native / controlled_native_height
            result = {
                "name": name, "expected_text": expected, "alignment": align,
                "timestamp_monotonic_ns": timestamp_ns,
                "native_frame_shape": [native_height, native_width, 3],
                "model_frame_shape": [1, 3, config.image_height, config.image_width],
                "pixel": {"min": float(native.min()), "max": float(native.max()),
                          "mean": float(native.mean()), "std": float(native.std())},
                "frame_sha256": hashlib.sha256(bytes(native_raw)).hexdigest(),
                "change_from_previous": change,
                "coarse_embedding_norm": float(torch.linalg.vector_norm(embedding)),
                "embedding_from_previous": None if previous_embedding is None else _distance(previous_embedding, embedding),
                "coarse_grid": list(state.grid_size), "coarse_stride_pixels_at_model_input": 16,
                "regions": [{"geometry": [float(v) for v in box], "salience": float(salience)}
                            for box, salience in zip(state.region_positions[0].cpu(), state.salience[0].cpu())],
                "target": {"point": [float(v) for v in motor.target_point[0].cpu()],
                           "bbox": [float(v) for v in motor.target_bbox[0].cpu()],
                           "confidence": float(motor.confidence[0]),
                           "action": ACTION_KINDS[int(motor.action_logits.argmax(-1)[0])]},
                "controlled_bbox": list(controlled_bbox),
                "controlled_coarse_feature": [float(v) for v in _known_patch_feature(state, controlled_bbox).cpu()],
                "controlled_foveal_feature": [float(v) for v in foveal_feature.cpu()],
                "resolution_trace": {"physical_monitor": [native_width, native_height],
                                     "captured_frame": [native_width, native_height],
                                     "coarse_resize": [config.image_width, config.image_height],
                                     "coarse_tokens": [state.grid_size[1], state.grid_size[0]],
                                     "controlled_crop_native_approx": [round((controlled_bbox[2]-controlled_bbox[0])*native_width), round(controlled_native_height)],
                                     "reader_input": [128, 32], "reader_feature_map": [128, 8],
                                     "glyph_height_native_px": character_height_native,
                                     "glyph_height_reader_input_px": reader_character_height,
                                     "glyph_height_reader_feature_cells": reader_character_height / 4},
                "reader_on_predicted_bbox": _decode(predicted_logits, expected),
                "reader_on_controlled_text_bbox": _decode(controlled_logits, expected),
                "inference_ms": inference_ms, "device": str(next(model.parameters()).device),
                "rocm": torch.version.hip,
            }
            (scene_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            scenes[name] = result; previous_native = native; previous_embedding = embedding.detach().clone()
            print(json.dumps({"scene": name, "hash": result["frame_sha256"][:16],
                              "pixel_change": change, "embedding_norm": result["coarse_embedding_norm"],
                              "target": result["target"], "decoded": result["reader_on_controlled_text_bbox"]["decoded_utf8"],
                              "cer": result["reader_on_controlled_text_bbox"]["character_error_rate"],
                              "inference_ms": inference_ms}, ensure_ascii=False), flush=True)
        comparisons = {
            "blank_vs_heiwa_coarse": _distance(torch.tensor(scenes["blank"]["controlled_coarse_feature"]), torch.tensor(scenes["heiwa"]["controlled_coarse_feature"])),
            "red_vs_blue_coarse": _distance(torch.tensor(scenes["red_dragon"]["controlled_coarse_feature"]), torch.tensor(scenes["blue_tiger"]["controlled_coarse_feature"])),
            "red_vs_blue_foveal": _distance(torch.tensor(scenes["red_dragon"]["controlled_foveal_feature"]), torch.tensor(scenes["blue_tiger"]["controlled_foveal_feature"])),
            "left_vs_right_local_coarse": _distance(torch.tensor(scenes["left"]["controlled_coarse_feature"]), torch.tensor(scenes["right"]["controlled_coarse_feature"])),
            "left_vs_right_full_embedding": scenes["right"]["embedding_from_previous"],
            "left_target": scenes["left"]["target"]["point"], "right_target": scenes["right"]["target"]["point"],
        }
        changed = float(scenes["heiwa"]["change_from_previous"]["changed_pixel_fraction"])
        coarse_distance = float(comparisons["red_vs_blue_coarse"]["l2"])
        target_shift = abs(float(comparisons["right_target"][0]) - float(comparisons["left_target"][0]))
        controlled_reader_good = sum(bool(scenes[name]["reader_on_controlled_text_bbox"]["exact_match"])
                                     for name in ("heiwa", "red_dragon", "blue_tiger", "russian_heiwa", "russian_dragon")) >= 4
        if changed < 0.001: diagnosis = "CAPTURE BROKEN"
        elif coarse_distance < 1e-4: diagnosis = "COARSE VISION BROKEN"
        elif target_shift < 0.15: diagnosis = "FOVEATION BROKEN"
        elif not controlled_reader_good: diagnosis = "READER UNTRAINED"
        else: diagnosis = "READER WORKS"
        summary = {"diagnosis": diagnosis, "checkpoint_read_only": True,
                   "atspi": False, "ocr": False, "accessibility": False,
                   "device": str(device), "config": asdict(config),
                   "comparisons": comparisons, "scenes": scenes}
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"diagnosis": diagnosis, "output": str(run_dir), "comparisons": comparisons}, ensure_ascii=False), flush=True)
        return run_dir
    finally:
        # Deliberately close only the read-only sensor. Never call
        # NeuralDesktopInference.close(), which persists a checkpoint.
        portal.close(); presenter.terminate()
        try: presenter.wait(timeout=3)
        except subprocess.TimeoutExpired: presenter.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only real Wayland pixel/vision diagnostic")
    parser.add_argument("--checkpoint", type=Path, default=Path("state/neural_desktop.pt"))
    parser.add_argument("--output", type=Path, default=Path("state/vision_diagnostics"))
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--presenter", nargs=2, metavar=("COMMAND_JSON", "TELEMETRY_JSON"))
    args = parser.parse_args()
    if args.presenter:
        return _presenter(Path(args.presenter[0]), Path(args.presenter[1]))
    run_diagnostic(args.checkpoint, args.output, settle_seconds=args.settle_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
