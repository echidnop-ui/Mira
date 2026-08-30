from __future__ import annotations

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402


def main() -> int:
    window = Gtk.Window(title="Heiwa Key Smoke")
    window.set_default_size(360, 120)
    window.set_border_width(18)

    box = Gtk.Box(
        orientation=Gtk.Orientation.VERTICAL,
        spacing=8,
    )

    first = Gtk.Entry()
    second = Gtk.Entry()

    first.set_placeholder_text("First")
    second.set_placeholder_text("Second")

    # Stable semantic names for motor targets.
    first.get_accessible().set_name("Heiwa Key First")
    second.get_accessible().set_name("Heiwa Key Second")

    box.pack_start(first, True, True, 0)
    box.pack_start(second, True, True, 0)
    window.add(box)

    result = {"focus": ""}

    def finish() -> bool:
        if first.has_focus():
            result["focus"] = "first"
        elif second.has_focus():
            result["focus"] = "second"
        else:
            result["focus"] = "none"

        print(result["focus"], flush=True)
        Gtk.main_quit()
        return False

    window.connect("destroy", Gtk.main_quit)
    window.show_all()

    GLib.timeout_add_seconds(6, finish)
    Gtk.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
