from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

from worlds.desktop.portal_screencast import PortalScreenCast, PortalScreenCastError


class ScreenVisionError(RuntimeError):
    """Raised when the visible desktop cannot be captured or measured."""


@dataclass(frozen=True, slots=True)
class ScreenRect:
    x: int
    y: int
    width: int
    height: int

    def clipped_positive(self) -> "ScreenRect":
        return ScreenRect(
            max(0, int(self.x)),
            max(0, int(self.y)),
            max(1, int(self.width)),
            max(1, int(self.height)),
        )


@dataclass(frozen=True, slots=True)
class VisualFingerprint:
    cells: tuple[int, ...]
    width: int
    height: int

    def distance(self, other: "VisualFingerprint") -> float:
        if len(self.cells) != len(other.cells) or not self.cells:
            return 1.0
        mse = sum((a - b) ** 2 for a, b in zip(self.cells, other.cells)) / len(self.cells)
        return math.sqrt(mse) / 255.0


_PIXBUF_FINGERPRINT = r'''
import json
import sys
import gi
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf

path, x, y, w, h, cols, rows = sys.argv[1:]
x, y, w, h, cols, rows = map(int, (x, y, w, h, cols, rows))
pix = GdkPixbuf.Pixbuf.new_from_file(path)
W, H = pix.get_width(), pix.get_height()
x = max(0, min(x, W - 1))
y = max(0, min(y, H - 1))
w = max(1, min(w, W - x))
h = max(1, min(h, H - y))
channels = pix.get_n_channels()
rowstride = pix.get_rowstride()
pixels = pix.get_pixels()
values = []
for gy in range(rows):
    py = y + min(h - 1, int((gy + 0.5) * h / rows))
    for gx in range(cols):
        px = x + min(w - 1, int((gx + 0.5) * w / cols))
        base = py * rowstride + px * channels
        r = pixels[base]
        g = pixels[base + 1]
        b = pixels[base + 2]
        values.append((77 * r + 150 * g + 29 * b) >> 8)
print(json.dumps({"cells": values, "width": W, "height": H}))
'''


_PIXBUF_CHAT_COMPOSER = r'''
import json
import math
import sys
import gi
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf

path, wx, wy, ww, wh = sys.argv[1:]
wx, wy, ww, wh = map(int, (wx, wy, ww, wh))
pix = GdkPixbuf.Pixbuf.new_from_file(path)
W, H = pix.get_width(), pix.get_height()
channels = pix.get_n_channels()
rowstride = pix.get_rowstride()
pixels = pix.get_pixels()

if ww <= 0 or wh <= 0:
    wx, wy, ww, wh = 0, 0, W, H

wx = max(0, min(wx, W - 1))
wy = max(0, min(wy, H - 1))
ww = max(1, min(ww, W - wx))
wh = max(1, min(wh, H - wy))

def gray(x, y):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    base = y * rowstride + x * channels
    r = pixels[base]
    g = pixels[base + 1]
    b = pixels[base + 2]
    return (77 * r + 150 * g + 29 * b) >> 8

def mean_abs_diff(a, b):
    if not a:
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

def edge_score(x, y, w, h):
    step_x = max(10, w // 42)
    step_y = max(8, h // 10)
    xs = list(range(x + 12, x + w - 12, step_x))
    ys = list(range(y + 8, y + h - 8, step_y))
    if len(xs) < 8 or len(ys) < 3:
        return -1e9

    top_in = [gray(px, y + 8) for px in xs]
    top_out = [gray(px, y - 5) for px in xs]
    bottom_in = [gray(px, y + h - 8) for px in xs]
    bottom_out = [gray(px, y + h + 5) for px in xs]
    left_in = [gray(x + 8, py) for py in ys]
    left_out = [gray(x - 5, py) for py in ys]
    right_in = [gray(x + w - 8, py) for py in ys]
    right_out = [gray(x + w + 5, py) for py in ys]

    boundary = (
        mean_abs_diff(top_in, top_out)
        + mean_abs_diff(bottom_in, bottom_out)
        + mean_abs_diff(left_in, left_out)
        + mean_abs_diff(right_in, right_out)
    ) / 4.0

    interior = []
    for py in (y + h // 3, y + 2 * h // 3):
        interior.extend(gray(px, py) for px in xs)
    if interior:
        mean = sum(interior) / len(interior)
        rough = math.sqrt(sum((v - mean) ** 2 for v in interior) / len(interior))
    else:
        rough = 255.0

    bottom = wy + wh
    center_x = x + w / 2.0
    desired_x = wx + ww * 0.56
    horizontal_penalty = abs(center_x - desired_x) / max(1.0, ww) * 18.0
    bottom_margin = bottom - (y + h)
    vertical_penalty = abs(bottom_margin - max(28.0, wh * 0.055)) / max(1.0, wh) * 28.0
    width_ratio = w / max(1.0, ww)
    width_penalty = abs(width_ratio - 0.52) * 16.0
    rough_penalty = max(0.0, rough - 32.0) * 0.08
    return boundary - horizontal_penalty - vertical_penalty - width_penalty - rough_penalty

best = None
best_score = -1e9
width_ratios = (0.40, 0.48, 0.56, 0.64)
height_ratios = (0.070, 0.090, 0.115, 0.140)
center_ratios = (0.50, 0.56, 0.60)
bottom_ratios = (0.035, 0.055, 0.080, 0.110, 0.145)

for wr in width_ratios:
    w = int(max(380, min(1100, ww * wr)))
    if w >= ww - 20:
        continue
    for hr in height_ratios:
        h = int(max(58, min(170, wh * hr)))
        for cr in center_ratios:
            cx = int(wx + ww * cr)
            x = cx - w // 2
            if x < wx + 8 or x + w > wx + ww - 8:
                continue
            for br in bottom_ratios:
                bottom_margin = int(max(24, wh * br))
                y = wy + wh - bottom_margin - h
                if y < wy + int(wh * 0.50):
                    continue
                score = edge_score(x, y, w, h)
                if score > best_score:
                    best_score = score
                    best = (x, y, w, h)

if best is None:
    w = int(max(420, min(900, ww * 0.52)))
    h = int(max(72, min(130, wh * 0.105)))
    cx = int(wx + ww * 0.56)
    x = max(wx + 8, min(cx - w // 2, wx + ww - w - 8))
    y = max(wy + int(wh * 0.55), wy + wh - h - max(32, int(wh * 0.055)))
    best = (x, y, w, h)
    best_score = -999.0

print(json.dumps({
    "ok": True,
    "composer": list(best),
    "window": [wx, wy, ww, wh],
    "score": float(best_score),
    "screen": [W, H],
}))
'''



class ScreenSensor:
    """Visible-screen sensor.

    It observes pixels only. It does not choose actions or infer goals.
    """

    def __init__(
        self,
        *,
        timeout: float = 6.0,
        system_python: str = "/usr/bin/python3",
        portal: PortalScreenCast | None = None,
    ) -> None:
        self.timeout = float(timeout)
        self.system_python = system_python
        self.portal = portal or PortalScreenCast(
            portal_timeout=max(60.0, self.timeout),
            frame_timeout=max(6.0, self.timeout),
        )

    def available(self) -> bool:
        return self.portal.available()

    def fingerprint(
        self,
        rect: ScreenRect,
        *,
        cols: int = 16,
        rows: int = 8,
    ) -> VisualFingerprint:
        rect = rect.clipped_positive()
        with tempfile.TemporaryDirectory(prefix="heiwa-screen-") as temp_dir:
            image = Path(temp_dir) / "screen.png"
            self._capture(image)
            result = subprocess.run(
                (
                    self.system_python,
                    "-c",
                    _PIXBUF_FINGERPRINT,
                    str(image),
                    str(rect.x),
                    str(rect.y),
                    str(rect.width),
                    str(rect.height),
                    str(int(cols)),
                    str(int(rows)),
                ),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                raise ScreenVisionError(
                    "GdkPixbuf fingerprint failed: "
                    + (result.stderr.strip() or f"returncode={result.returncode}")
                )
            try:
                payload = json.loads(result.stdout.strip())
                cells = tuple(int(value) for value in payload["cells"])
                return VisualFingerprint(
                    cells=cells,
                    width=int(payload["width"]),
                    height=int(payload["height"]),
                )
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ScreenVisionError(
                    f"invalid screen fingerprint payload: {result.stdout[:200]!r}"
                ) from exc


    def locate_chat_composer(
        self,
        *,
        window: ScreenRect | None = None,
    ) -> tuple[ScreenRect, ScreenRect, float]:
        """Estimate the visible ChatGPT composer directly from screen pixels.

        `window` is only a crop hint. If unavailable, the full visible desktop
        is searched. No accessibility editable node is required.
        """

        with tempfile.TemporaryDirectory(prefix="heiwa-composer-") as temp_dir:
            image = Path(temp_dir) / "screen.png"
            self._capture(image)
            if window is None:
                crop = ScreenRect(0, 0, 0, 0)
            else:
                crop = window.clipped_positive()
            result = subprocess.run(
                (
                    self.system_python,
                    "-c",
                    _PIXBUF_CHAT_COMPOSER,
                    str(image),
                    str(crop.x),
                    str(crop.y),
                    str(crop.width if window is not None else 0),
                    str(crop.height if window is not None else 0),
                ),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                raise ScreenVisionError(
                    "visual composer locator failed: "
                    + (result.stderr.strip() or f"returncode={result.returncode}")
                )
            try:
                payload = json.loads(result.stdout.strip())
                composer = payload["composer"]
                resolved_window = payload["window"]
                if not (
                    isinstance(composer, list)
                    and len(composer) == 4
                    and isinstance(resolved_window, list)
                    and len(resolved_window) == 4
                ):
                    raise ValueError("missing geometry")
                return (
                    ScreenRect(*(int(v) for v in composer)),
                    ScreenRect(*(int(v) for v in resolved_window)),
                    float(payload.get("score", 0.0)),
                )
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ScreenVisionError(
                    f"invalid visual composer payload: {result.stdout[:200]!r}"
                ) from exc

    def _capture(self, destination: Path) -> None:
        try:
            self.portal.capture_png(destination)
        except PortalScreenCastError as exc:
            raise ScreenVisionError(f"portal/PipeWire capture failed: {exc}") from exc
