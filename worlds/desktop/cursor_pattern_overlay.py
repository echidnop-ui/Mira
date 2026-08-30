from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
from pathlib import Path
import time


COLORS = {
    "candidate": (0.15, 0.65, 1.0),
    "rejected": (1.0, 0.12, 0.12),
    "attempted": (1.0, 0.58, 0.08),
    "accepted": (0.12, 1.0, 0.32),
    "guided": (0.05, 0.95, 1.0),
    "proposal": (1.0, 0.10, 0.95),
}


def _load(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) and value.get("format") == 1 else None


def _cell_rect(cell: int, gw: int, gh: int, width: float, height: float):
    row, col = divmod(int(cell), gw)
    return col * width / gw, row * height / gh, width / gw, height / gh


def _validate(payload: dict[str, object]) -> None:
    gw, gh = (int(v) for v in payload["grid"])
    if gw <= 0 or gh <= 0:
        raise ValueError("invalid overlay grid")
    for key in ("rejected_cells", "hard_rejected_cells", "attempted_cells", "accepted_cells"):
        if any(not 0 <= int(cell) < gw * gh for cell in payload.get(key, ())):
            raise ValueError(f"invalid {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Click-through visualizer for Heiwa neural cursor patterns")
    parser.add_argument("--state", type=Path, default=Path("state/embodied_runtime/cursor_guidance_overlay.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        sample = {
            "format": 1, "grid": [4, 3], "rejected_cells": [0, 1],
            "hard_rejected_cells": [0], "attempted_cells": [5], "accepted_cells": [11],
        }
        _validate(sample)
        assert _cell_rect(11, 4, 3, 400, 300) == (300.0, 200.0, 100.0, 100.0)
        print({"ok": True, "click_through_overlay_geometry": True})
        return 0

    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    try:
        gi.require_foreign("cairo")
        import cairo
    except ImportError as exc:
        raise SystemExit(
            "Transparent overlay requires python3-gi-cairo. Install it with: "
            "sudo apt install -y python3-gi-cairo"
        ) from exc
    from gi.repository import Gdk, GLib, Gtk

    library_name = ctypes.util.find_library("gtk-layer-shell") or "libgtk-layer-shell.so.0"
    try:
        layer_shell = ctypes.CDLL(library_name)
    except OSError as exc:
        raise SystemExit(
            "Native Wayland overlay requires gtk-layer-shell. Install it with: "
            "sudo apt install -y libgtk-layer-shell0"
        ) from exc

    def gobject_pointer(value) -> int:
        capsule_get = ctypes.pythonapi.PyCapsule_GetPointer
        capsule_get.restype = ctypes.c_void_p
        capsule_get.argtypes = (ctypes.py_object, ctypes.c_char_p)
        pointer = capsule_get(value.__gpointer__, None)
        if not pointer:
            raise RuntimeError("cannot resolve GTK window pointer")
        return int(pointer)

    layer_shell.gtk_layer_is_supported.restype = ctypes.c_int
    layer_shell.gtk_layer_init_for_window.argtypes = (ctypes.c_void_p,)
    layer_shell.gtk_layer_set_namespace.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    layer_shell.gtk_layer_set_layer.argtypes = (ctypes.c_void_p, ctypes.c_int)
    layer_shell.gtk_layer_set_keyboard_mode.argtypes = (ctypes.c_void_p, ctypes.c_int)
    layer_shell.gtk_layer_set_exclusive_zone.argtypes = (ctypes.c_void_p, ctypes.c_int)
    layer_shell.gtk_layer_set_anchor.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    if not layer_shell.gtk_layer_is_supported():
        raise SystemExit("The current compositor does not expose the Wayland layer-shell protocol")

    window = Gtk.Window(type=Gtk.WindowType.TOPLEVEL)
    window.set_title("Heiwa cursor pattern overlay")
    window.set_decorated(False)
    window.set_accept_focus(False)
    window.set_focus_on_map(False)
    window.set_skip_taskbar_hint(True)
    window.set_skip_pager_hint(True)
    window.set_app_paintable(True)
    screen = window.get_screen()
    visual = screen.get_rgba_visual()
    if visual is not None:
        window.set_visual(visual)
    # Native Wayland overlay surface. Unlike fullscreen/maximized toplevels,
    # layer-shell preserves the alpha channel and never receives keyboard input.
    window_pointer = gobject_pointer(window)
    layer_shell.gtk_layer_init_for_window(window_pointer)
    layer_shell.gtk_layer_set_namespace(window_pointer, b"heiwa-cursor-patterns")
    layer_shell.gtk_layer_set_layer(window_pointer, 3)  # OVERLAY
    layer_shell.gtk_layer_set_keyboard_mode(window_pointer, 0)  # NONE
    layer_shell.gtk_layer_set_exclusive_zone(window_pointer, 0)
    for edge in range(4):  # LEFT, RIGHT, TOP, BOTTOM
        layer_shell.gtk_layer_set_anchor(window_pointer, edge, 1)
    fixed = Gtk.Fixed()
    fixed.set_name("overlay-root")
    window.add(fixed)
    css = Gtk.CssProvider()
    css.load_from_data(b"""
        window, #overlay-root { background-color: rgba(0,0,0,0); }
        #candidate { border: 2px solid rgba(38,166,255,0.55); background-color: rgba(38,166,255,0.04); }
        #rejected { border: 2px solid rgba(255,30,30,0.82); background-color: rgba(255,30,30,0.20); }
        #hard-rejected { border: 4px solid rgba(255,20,20,0.98); background-color: rgba(255,20,20,0.25); }
        #attempted { border: 4px solid rgba(255,148,20,0.95); background-color: rgba(255,148,20,0.08); }
        #accepted { border: 5px solid rgba(30,255,82,0.98); background-color: rgba(30,255,82,0.09); }
        #proposal { border: 4px solid rgba(255,25,242,0.98); background-color: rgba(255,25,242,0.05); }
        #guided { border: 4px solid rgba(12,242,255,0.92); background-color: rgba(12,242,255,0.04); }
        #trace { border-radius: 5px; background-color: rgba(12,242,255,0.95); }
        #status { color: white; background-color: rgba(5,5,5,0.82); padding: 8px 12px; font-size: 18px; font-weight: bold; }
        #status-held { color: #ff3030; background-color: rgba(5,5,5,0.88); padding: 8px 12px; font-size: 18px; font-weight: bold; }
    """)
    Gtk.StyleContext.add_provider_for_screen(screen, css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    state: dict[str, object] = {"payload": None, "loaded_at": 0.0}

    def realized(widget) -> None:
        surface = widget.get_window()
        if surface is not None:
            surface.set_pass_through(True)

    def clear_transparent_background(_widget, cr) -> bool:
        # CSS transparency alone may leave a black Wayland backing surface.
        # Explicit SOURCE-alpha clearing makes untouched pixels genuinely
        # transparent while child widgets draw their coloured outlines later.
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0.0, 0.0, 0.0, 0.0)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)
        return False

    def add_box(name: str, x: float, y: float, width: float, height: float):
        box = Gtk.EventBox()
        box.set_name(name)
        box.set_size_request(max(int(width), 2), max(int(height), 2))
        fixed.put(box, int(x), int(y))
        box.show()

    def render(payload: dict[str, object] | None) -> None:
        for child in tuple(fixed.get_children()):
            fixed.remove(child)
        if not isinstance(payload, dict):
            return
        allocation = window.get_allocation()
        width, height = float(allocation.width), float(allocation.height)
        try:
            _validate(payload)
            gw, gh = (int(v) for v in payload["grid"])
        except (KeyError, TypeError, ValueError):
            return

        for item in payload.get("top_candidates", ()):
            x, y, w, h = _cell_rect(int(item["cell"]), gw, gh, width, height)
            add_box("candidate", x+2, y+2, w-4, h-4)
        for cell in payload.get("rejected_cells", ()):
            x, y, w, h = _cell_rect(int(cell), gw, gh, width, height)
            add_box("rejected", x+2, y+2, w-4, h-4)
        for cell in payload.get("hard_rejected_cells", ()):
            x, y, w, h = _cell_rect(int(cell), gw, gh, width, height)
            add_box("hard-rejected", x+2, y+2, w-4, h-4)
        for cell in payload.get("attempted_cells", ()):
            x, y, w, h = _cell_rect(int(cell), gw, gh, width, height)
            add_box("attempted", x+2, y+2, w-4, h-4)
        for cell in payload.get("accepted_cells", ()):
            x, y, w, h = _cell_rect(int(cell), gw, gh, width, height)
            add_box("accepted", x+2, y+2, w-4, h-4)

        bbox = payload.get("proposal_bbox", ())
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            add_box("proposal", x1*width, y1*height, max((x2-x1)*width, 4), max((y2-y1)*height, 4))
        xy = payload.get("proposal_xy")
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            x, y = float(xy[0]) * width, float(xy[1]) * height
            add_box("proposal", x-9, y-9, 18, 18)

        guided = payload.get("guided_xy")
        radius = float(payload.get("search_radius", 0.0))
        if isinstance(guided, (list, tuple)) and len(guided) == 2:
            gx, gy = float(guided[0]) * width, float(guided[1]) * height
            rr = radius * min(width, height)
            add_box("guided", gx-rr, gy-rr, 2*rr, 2*rr)
        trace = payload.get("guidance_trace", ())
        for point in trace:
            x, y = float(point[0]) * width, float(point[1]) * height
            add_box("trace", x-4, y-4, 8, 8)

        held = bool(payload.get("f23_held"))
        rejected_count = len(payload.get("hard_rejected_cells", ()))
        attempted_count = len(payload.get("attempted_cells", ()))
        label = f"F21=text OK F22=text wrong   F23 cursor erase: {'HELD' if held else 'released'}   F24=cursor accept"
        status = Gtk.Label(label=label)
        status.set_name("status-held" if held else "status")
        fixed.put(status, 14, 14)
        status.show()

    def refresh() -> bool:
        payload = _load(args.state)
        if payload is not None:
            state["payload"] = payload
            state["loaded_at"] = time.monotonic()
        elif time.monotonic() - float(state["loaded_at"]) > 2.0:
            state["payload"] = None
        render(state["payload"])
        return True

    window.connect("realize", realized)
    window.connect("draw", clear_transparent_background)
    window.connect("destroy", Gtk.main_quit)
    GLib.timeout_add(100, refresh)
    window.show_all()
    Gtk.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
