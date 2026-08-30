from __future__ import annotations

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib


EXPECTED_TIMEOUT_MS = 6000


def main() -> int:
    window = Gtk.Window(title="Heiwa Motor Smoke")
    window.set_default_size(520, 130)

    entry = Gtk.Entry()
    entry.set_name("heiwa-motor-smoke-entry")
    entry.set_placeholder_text("Waiting for Heiwa motor input")
    window.add(entry)

    result = {"text": ""}

    def finish() -> bool:
        result["text"] = entry.get_text()
        print(result["text"], flush=True)
        Gtk.main_quit()
        return False

    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    window.present()
    entry.grab_focus()

    GLib.timeout_add(EXPECTED_TIMEOUT_MS, finish)
    Gtk.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
