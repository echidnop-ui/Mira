from __future__ import annotations

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402

from worlds.desktop.practice_contract import (
    FIRST_CONTROL,
    SECOND_CONTROL,
    WINDOW_TITLE,
)


def main() -> int:
    window = Gtk.Window(title=WINDOW_TITLE)
    window.set_default_size(380, 130)
    window.set_border_width(18)

    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    first = Gtk.Entry()
    second = Gtk.Entry()
    first.set_placeholder_text("First")
    second.set_placeholder_text("Second")
    first.get_accessible().set_name(FIRST_CONTROL)
    second.get_accessible().set_name(SECOND_CONTROL)

    box.pack_start(first, True, True, 0)
    box.pack_start(second, True, True, 0)
    window.add(box)
    window.connect("destroy", Gtk.main_quit)
    window.show_all()

    # Initial fixture state only; later resets are owned by the training
    # harness, never by Policy or DesktopWorldAdapter.
    GLib.idle_add(first.grab_focus)
    Gtk.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
