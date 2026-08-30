from __future__ import annotations

import os

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk


PROMPT_ENV = "HEIWA_NEURAL_LANGUAGE_PROMPT"
TIMEOUT_MS = 60000
COMPLETION_GRACE_MS = 2500


def main() -> int:
    initial = os.environ.get(PROMPT_ENV, "Человек: Ты меня видишь?")

    window = Gtk.Window(title="Heiwa Neural Language Bridge")
    window.set_default_size(760, 130)

    entry = Gtk.Entry()
    entry.set_name("heiwa-neural-language-entry")
    entry.set_text(initial)
    entry.set_position(-1)
    window.add(entry)

    result = {"printed": False}

    def finish() -> bool:
        if not result["printed"]:
            result["printed"] = True
            print(entry.get_text(), flush=True)
        Gtk.main_quit()
        return False

    change_seen = {"value": False}

    def observe_change() -> bool:
        if entry.get_text() != initial and not change_seen["value"]:
            # Keep the accessibility object alive long enough for
            # DesktopWorldAdapter.poll() to re-read the exact target text
            # and emit CompletedAction. Closing immediately here races observation.
            change_seen["value"] = True
            GLib.timeout_add(COMPLETION_GRACE_MS, finish)
            return False
        return True

    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    window.present()
    entry.grab_focus()

    GLib.timeout_add(50, observe_change)
    GLib.timeout_add(TIMEOUT_MS, finish)
    Gtk.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
