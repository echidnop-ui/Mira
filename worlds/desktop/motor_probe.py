from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    print("HEIWA DESKTOP MOTOR PROBE")

    ydotool = shutil.which("ydotool")
    print("ydotool:", ydotool)
    print(
        "uinput_exists:",
        Path("/dev/uinput").exists(),
    )
    print(
        "uinput_writable:",
        os.access("/dev/uinput", os.W_OK),
    )

    if ydotool is None:
        print("DESKTOP MOTOR PROBE: YDOTOOL MISSING")
        return 2

    if not os.access("/dev/uinput", os.W_OK):
        print("DESKTOP MOTOR PROBE: UINPUT PERMISSION NEEDED")
        return 3

    # Deliberately do not generate any input in this probe.
    # The first real motor smoke will target a disposable text window.
    print("DESKTOP MOTOR PROBE: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
