from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


DEFAULT_SOCKET = "/run/heiwa-ydotool.sock"
KEY_LEFTSHIFT = 42
KEY_RIGHTSHIFT = 54
KEY_LEFTCTRL = 29
KEY_RIGHTCTRL = 97
KEY_LEFTALT = 56
KEY_RIGHTALT = 100
KEY_LEFTMETA = 125
KEY_RIGHTMETA = 126
KEY_A = 30
KEY_ENTER = 28
KEY_BACKSPACE = 14

_ASCII_KEY_CODES = {
    "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
    "6": 7, "7": 8, "8": 9, "9": 10, "0": 11,
    "-": 12, "=": 13,
    "q": 16, "w": 17, "e": 18, "r": 19, "t": 20,
    "y": 21, "u": 22, "i": 23, "o": 24, "p": 25,
    "[": 26, "]": 27,
    "a": 30, "s": 31, "d": 32, "f": 33, "g": 34,
    "h": 35, "j": 36, "k": 37, "l": 38, ";": 39,
    "'": 40, "`": 41, "\\": 43,
    "z": 44, "x": 45, "c": 46, "v": 47, "b": 48,
    "n": 49, "m": 50, ",": 51, ".": 52, "/": 53,
    " ": 57,
}
_SHIFTED_ASCII = {
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5",
    "^": "6", "&": "7", "*": "8", "(": "9", ")": "0",
    "_": "-", "+": "=", "{": "[", "}": "]", ":": ";",
    '"': "'", "~": "`", "|": "\\", "<": ",", ">": ".",
    "?": "/",
}

# Physical JCUKEN keys.  Cyrillic letters are emitted as real key events under
# GNOME's Russian input source instead of relying on GTK Ctrl+Shift+U compose.
# This avoids leaking fragments such as ``u43...`` into chat when compose is
# not accepted by the focused Wayland/Firefox widget.
_RUSSIAN_KEY_CODES = {
    "й": 16, "ц": 17, "у": 18, "к": 19, "е": 20, "н": 21,
    "г": 22, "ш": 23, "щ": 24, "з": 25, "х": 26, "ъ": 27,
    "ф": 30, "ы": 31, "в": 32, "а": 33, "п": 34, "р": 35,
    "о": 36, "л": 37, "д": 38, "ж": 39, "э": 40,
    "я": 44, "ч": 45, "с": 46, "м": 47, "и": 48, "т": 49,
    "ь": 50, "б": 51, "ю": 52, "ё": 41,
}


class DesktopMotorError(RuntimeError):
    """Raised when the low-level Desktop motor cannot execute input."""


@dataclass(frozen=True, slots=True)
class MotorResult:
    returncode: int
    stdout: str
    stderr: str


class DesktopMotor:
    """
    Low-level deterministic Desktop motor.

    It executes already-selected physical input.
    It does not decide what Heiwa should do.
    """

    def __init__(
        self,
        *,
        socket_path: str = DEFAULT_SOCKET,
        timeout: float = 5.0,
    ) -> None:
        self.socket_path = socket_path
        self.timeout = timeout

    def ready(self) -> bool:
        return (
            shutil.which("ydotool") is not None
            and Path(self.socket_path).is_socket()
        )

    def _run_session_command(
        self,
        *command: str,
        timeout: float = 2.0,
    ) -> subprocess.CompletedProcess[str]:
        """Run a small desktop-session helper command.

        This is motor configuration, not text transport. The actual text is
        still emitted as physical uinput key events through ydotool.
        """

        return subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )

    @staticmethod
    def _is_us_input_source(source: tuple[str, str]) -> bool:
        kind, name = source
        return kind == "xkb" and (
            name == "us"
            or name.startswith("us+")
            or name.startswith("us(")
        )

    @staticmethod
    def _is_ru_input_source(source: tuple[str, str]) -> bool:
        kind, name = source
        return kind == "xkb" and (
            name == "ru"
            or name.startswith("ru+")
            or name.startswith("ru(")
        )

    def _gnome_input_source_state(
        self,
    ) -> tuple[tuple[str, str], tuple[tuple[str, str], ...]] | None:
        """Read GNOME's MRU input-source state.

        GNOME's old ``current`` gsettings key is deprecated/ignored. The
        shell maintains the active source and writes the most-recently-used
        source first in ``mru-sources`` when the source changes.
        """

        gsettings = shutil.which("gsettings")
        if gsettings is None:
            return None
        schema = "org.gnome.desktop.input-sources"

        def read(key: str) -> str | None:
            try:
                result = self._run_session_command(
                    gsettings, "get", schema, key
                )
            except (OSError, subprocess.TimeoutExpired):
                return None
            if result.returncode != 0:
                return None
            return result.stdout.strip()

        raw_sources = read("sources")
        raw_mru = read("mru-sources")
        if raw_sources is None or raw_mru is None:
            return None

        pattern = (
            r"\(\s*['\"]([^'\"]+)['\"]\s*,\s*"
            r"['\"]([^'\"]+)['\"]\s*\)"
        )
        sources = tuple(re.findall(pattern, raw_sources))
        mru = tuple(re.findall(pattern, raw_mru))
        if not sources or not mru:
            return None
        current = mru[0]
        if current not in sources:
            return None
        return current, sources

    def _gnome_next_input_source(self) -> None:
        """Use GNOME's physical next-input-source shortcut (Super+Space)."""

        self.release_modifiers()
        self._run(
            "key",
            f"{KEY_LEFTMETA}:1",
            f"{_ASCII_KEY_CODES[' ']}:1",
            f"{_ASCII_KEY_CODES[' ']}:0",
            f"{KEY_LEFTMETA}:0",
        )
        import time
        time.sleep(0.22)

    def _gnome_switch_to(self, predicate) -> tuple[str, str] | None:
        state = self._gnome_input_source_state()
        if state is None:
            return None
        previous, sources = state
        encoded_previous = previous[0] + "\t" + previous[1]
        if predicate(previous):
            return ("gnome-shortcut", encoded_previous)

        # Cycle only through configured sources and verify GNOME's own state.
        for _ in range(len(sources) + 1):
            self._gnome_next_input_source()
            refreshed = self._gnome_input_source_state()
            if refreshed is None:
                continue
            current, _ = refreshed
            if predicate(current):
                return ("gnome-shortcut", encoded_previous)
        return None

    def _gnome_switch_to_us(self) -> tuple[str, str] | None:
        return self._gnome_switch_to(self._is_us_input_source)

    def _gnome_switch_to_ru(self) -> tuple[str, str] | None:
        return self._gnome_switch_to(self._is_ru_input_source)

    def _restore_keyboard_layout(self, lease: tuple[str, str]) -> None:
        backend, encoded_previous = lease
        if backend != "gnome-shortcut":
            return
        try:
            previous_kind, previous_name = encoded_previous.split("\t", 1)
        except ValueError:
            return
        previous = (previous_kind, previous_name)

        try:
            state = self._gnome_input_source_state()
            if state is None:
                return
            current, sources = state
            if current == previous:
                return
            for _ in range(len(sources) + 1):
                self._gnome_next_input_source()
                refreshed = self._gnome_input_source_state()
                if refreshed is not None and refreshed[0] == previous:
                    return
        except (DesktopMotorError, OSError, subprocess.TimeoutExpired):
            # The actual text input result must not be hidden by a best-effort
            # restoration failure.
            return

    @contextmanager
    def us_keyboard_layout(self):
        """Temporarily select a verified US source using GNOME itself.

        Physical uinput key codes are interpreted by the compositor's active
        keymap. We therefore use GNOME's own input-source shortcut and verify
        the resulting MRU state instead of writing the deprecated ``current``
        gsettings key or trusting an unrelated IBus engine state.
        """

        lease = self._gnome_switch_to_us()
        if lease is None:
            raise DesktopMotorError(
                "cannot establish verified US keyboard layout through GNOME "
                "input-source switching"
            )
        try:
            yield lease[0]
        finally:
            self.release_modifiers()
            self._restore_keyboard_layout(lease)

    @contextmanager
    def ru_keyboard_layout(self):
        """Temporarily select and verify a configured Russian XKB source."""

        lease = self._gnome_switch_to_ru()
        if lease is None:
            raise DesktopMotorError(
                "cannot establish verified Russian keyboard layout through GNOME "
                "input-source switching"
            )
        try:
            yield lease[0]
        finally:
            self.release_modifiers()
            self._restore_keyboard_layout(lease)

    def type_text(self, text: str) -> MotorResult:
        """Type text through the same layout-safe physical keyboard path."""

        return self.type_unicode_linux(text)

    def press_key(self, key_code: int) -> MotorResult:
        """
        Press and release one Linux input-event key code.

        Example:
            KEY_TAB = 15
            KEY_ENTER = 28

        This is a mechanical primitive, not a cognitive action vocabulary.
        """

        return self.key_press(key_code)

    def key_down(self, key_code: int) -> MotorResult:
        self._validate_key_code(key_code)
        return self._run("key", f"{key_code}:1")

    def key_up(self, key_code: int) -> MotorResult:
        self._validate_key_code(key_code)
        return self._run("key", f"{key_code}:0")

    def key_press(self, key_code: int) -> MotorResult:
        self._validate_key_code(key_code)
        return self._run("key", f"{key_code}:1", f"{key_code}:0")

    def release_modifiers(self) -> MotorResult:
        """Release common keyboard modifiers before a bounded UI action.

        This is a mechanical recovery primitive. It does not select or infer
        an action; it only prevents stale uinput modifier state from changing
        the meaning of a subsequent physical key press.
        """

        return self._run(
            "key",
            f"{KEY_LEFTSHIFT}:0",
            f"{KEY_RIGHTSHIFT}:0",
            f"{KEY_LEFTCTRL}:0",
            f"{KEY_RIGHTCTRL}:0",
            f"{KEY_LEFTALT}:0",
            f"{KEY_RIGHTALT}:0",
            f"{KEY_LEFTMETA}:0",
            f"{KEY_RIGHTMETA}:0",
        )

    def clear_focused_editable(self) -> MotorResult:
        """Physically select all and delete text in the focused editor.

        Callers must establish that overwriting the focused content is safe.
        The temporary US layout keeps Ctrl+A deterministic under RU/US input
        source switching.
        """

        with self.us_keyboard_layout():
            return self._run(
                "key",
                f"{KEY_LEFTCTRL}:1",
                f"{KEY_A}:1",
                f"{KEY_A}:0",
                f"{KEY_LEFTCTRL}:0",
                f"{KEY_BACKSPACE}:1",
                f"{KEY_BACKSPACE}:0",
            )

    def type_ascii(self, text: str) -> MotorResult:
        """Type ASCII as explicit Linux input key transitions.

        This is physical uinput execution. It deliberately does not use the
        clipboard and does not decide what text should be expressed.
        """

        if not isinstance(text, str):
            raise TypeError("text must be str")
        events: list[str] = []
        for character in text:
            if character == "\n":
                # Text newline in a chat composer is Shift+Enter; bare Enter
                # remains available through key_press(KEY_ENTER) for submit.
                code, shifted = KEY_ENTER, True
            elif character == "\b":
                code, shifted = KEY_BACKSPACE, False
            elif character == "\t":
                code, shifted = 15, False
            elif character.isascii() and character.isalpha():
                code = _ASCII_KEY_CODES[character.lower()]
                shifted = character.isupper()
            elif character in _SHIFTED_ASCII:
                code = _ASCII_KEY_CODES[_SHIFTED_ASCII[character]]
                shifted = True
            elif character in _ASCII_KEY_CODES:
                code, shifted = _ASCII_KEY_CODES[character], False
            else:
                raise ValueError(
                    f"physical ASCII keyboard cannot type {character!r}"
                )
            if shifted:
                events.append(f"{KEY_LEFTSHIFT}:1")
            events.extend((f"{code}:1", f"{code}:0"))
            if shifted:
                events.append(f"{KEY_LEFTSHIFT}:0")
        if not events:
            return MotorResult(returncode=0, stdout="", stderr="")
        with self.us_keyboard_layout():
            return self._run("key", *events)

    def move_pointer_absolute(self, x: int, y: int) -> MotorResult:
        """Move the physical pointer to desktop coordinates."""

        if isinstance(x, bool) or isinstance(y, bool):
            raise TypeError("pointer coordinates must be integers")
        x, y = int(x), int(y)
        last_error: Exception | None = None
        with self.flat_mouse_acceleration():
            for args in (
                ("mousemove", "--absolute", str(x), str(y)),
                ("mousemove", "-a", str(x), str(y)),
            ):
                try:
                    return self._run(*args)
                except DesktopMotorError as exc:
                    last_error = exc
        raise DesktopMotorError(f"ydotool absolute mouse move failed: {last_error}")

    def move_pointer_trajectory(
        self, points: tuple[tuple[int, int], ...], *, duration: float,
    ) -> MotorResult:
        """Execute already-selected neural waypoints; this method makes no target decision."""
        if not points:
            raise ValueError("pointer trajectory must contain at least one point")
        result = MotorResult(0, "", ""); pause = max(float(duration), 0.0) / len(points)
        for x, y in points:
            result = self.move_pointer_absolute(int(x), int(y))
            if pause: time.sleep(pause)
        return result

    @contextmanager
    def flat_mouse_acceleration(self):
        """Temporarily satisfy ydotool absolute movement's flat-profile requirement."""
        gsettings = shutil.which("gsettings")
        if gsettings is None:
            raise DesktopMotorError("gsettings is required for calibrated absolute pointer movement")
        schema, key = "org.gnome.desktop.peripherals.mouse", "accel-profile"
        current = self._run_session_command(gsettings, "get", schema, key)
        if current.returncode != 0:
            raise DesktopMotorError(f"cannot read mouse acceleration profile: {current.stderr.strip()}")
        original = current.stdout.strip().strip("'")
        changed = original != "flat"
        if changed:
            result = self._run_session_command(gsettings, "set", schema, key, "flat")
            if result.returncode != 0:
                raise DesktopMotorError(f"cannot set temporary flat mouse acceleration: {result.stderr.strip()}")
            time.sleep(0.05)
        try:
            yield
        finally:
            if changed:
                self._run_session_command(gsettings, "set", schema, key, original)

    def click_left(self) -> MotorResult:
        """Press/release the primary physical pointer button."""

        return self._run("click", "0xC0")

    def click_right(self) -> MotorResult:
        return self._run("click", "0xC1")

    def double_click_left(self) -> MotorResult:
        first = self.click_left()
        self.click_left()
        return first

    def scroll(self, dy: int) -> MotorResult:
        if isinstance(dy, bool) or not isinstance(dy, int):
            raise TypeError("scroll delta must be int")
        if dy == 0:
            return MotorResult(returncode=0, stdout="", stderr="")
        button = "0xC4" if dy > 0 else "0xC5"
        result = MotorResult(returncode=0, stdout="", stderr="")
        for _ in range(min(abs(dy), 20)):
            result = self._run("click", button)
        return result

    def press_chord(self, key_codes: tuple[int, ...]) -> MotorResult:
        if not key_codes:
            raise ValueError("key chord must not be empty")
        for code in key_codes:
            self._validate_key_code(code)
        events = [f"{code}:1" for code in key_codes]
        events.extend(f"{code}:0" for code in reversed(key_codes))
        return self._run("key", *events)

    def type_unicode_linux(self, text: str) -> MotorResult:
        """Type text as verified physical key events.

        ASCII is emitted under US XKB. Russian Cyrillic is emitted directly
        under RU XKB (JCUKEN). Only characters without a physical mapping use
        Linux/GTK Ctrl+Shift+U as a bounded fallback. This keeps ordinary
        Russian dialogue away from the fragile compose path that can surface
        literal ``u43...`` fragments when a widget rejects compose.
        """

        if not isinstance(text, str):
            raise TypeError("text must be str")
        if not text:
            return MotorResult(returncode=0, stdout="", stderr="")

        def mode(character: str) -> str:
            if character.lower() in _RUSSIAN_KEY_CODES:
                return "ru"
            if character.isascii():
                return "us"
            return "unicode"

        runs: list[tuple[str, str]] = []
        for character in text:
            current = mode(character)
            if runs and runs[-1][0] == current:
                runs[-1] = (current, runs[-1][1] + character)
            else:
                runs.append((current, character))

        last = MotorResult(returncode=0, stdout="", stderr="")

        def append_ascii(events: list[str], character: str) -> None:
            if character == "\n":
                code, shifted = KEY_ENTER, True
            elif character == "\b":
                code, shifted = KEY_BACKSPACE, False
            elif character == "\t":
                code, shifted = 15, False
            elif character.isascii() and character.isalpha():
                code = _ASCII_KEY_CODES[character.lower()]
                shifted = character.isupper()
            elif character in _SHIFTED_ASCII:
                code = _ASCII_KEY_CODES[_SHIFTED_ASCII[character]]
                shifted = True
            elif character in _ASCII_KEY_CODES:
                code, shifted = _ASCII_KEY_CODES[character], False
            else:
                raise ValueError(f"cannot physically type ASCII {character!r}")
            if shifted:
                events.append(f"{KEY_LEFTSHIFT}:1")
            events.extend((f"{code}:1", f"{code}:0"))
            if shifted:
                events.append(f"{KEY_LEFTSHIFT}:0")

        for run_mode, chunk in runs:
            events: list[str] = []
            if run_mode == "us":
                for character in chunk:
                    append_ascii(events, character)
                with self.us_keyboard_layout():
                    last = self._run("key", *events)
                continue

            if run_mode == "ru":
                for character in chunk:
                    lower = character.lower()
                    code = _RUSSIAN_KEY_CODES[lower]
                    shifted = character != lower
                    if shifted:
                        events.append(f"{KEY_LEFTSHIFT}:1")
                    events.extend((f"{code}:1", f"{code}:0"))
                    if shifted:
                        events.append(f"{KEY_LEFTSHIFT}:0")
                with self.ru_keyboard_layout():
                    last = self._run("key", *events)
                continue

            # Rare unmapped Unicode fallback. Keep it isolated from ordinary
            # Cyrillic so one failed compose cannot corrupt an entire message.
            for character in chunk:
                events.extend((
                    f"{KEY_LEFTCTRL}:1", f"{KEY_LEFTSHIFT}:1",
                    f"{_ASCII_KEY_CODES['u']}:1", f"{_ASCII_KEY_CODES['u']}:0",
                    f"{KEY_LEFTSHIFT}:0", f"{KEY_LEFTCTRL}:0",
                ))
                for digit in f"{ord(character):x}":
                    append_ascii(events, digit)
                events.extend((
                    f"{_ASCII_KEY_CODES[' ']}:1",
                    f"{_ASCII_KEY_CODES[' ']}:0",
                ))
            with self.us_keyboard_layout():
                last = self._run("key", *events)

        return last

    @staticmethod
    def _validate_key_code(key_code: int) -> None:
        if isinstance(key_code, bool) or not isinstance(key_code, int):
            raise TypeError("key_code must be int")
        if key_code < 0 or key_code > 767:
            raise ValueError("key_code must be between 0 and 767")

    def _run(
        self,
        *arguments: str,
    ) -> MotorResult:
        ydotool = shutil.which("ydotool")
        if ydotool is None:
            raise DesktopMotorError("ydotool is not installed")

        if not Path(self.socket_path).is_socket():
            raise DesktopMotorError(
                f"ydotool socket is not available: {self.socket_path}"
            )

        env = os.environ.copy()
        env["YDOTOOL_SOCKET"] = self.socket_path

        try:
            result = subprocess.run(
                (ydotool, *arguments),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise DesktopMotorError(
                f"ydotool {' '.join(arguments)} timed out "
                f"after {self.timeout}s"
            ) from exc
        except OSError as exc:
            raise DesktopMotorError(
                f"failed to execute ydotool: {exc}"
            ) from exc

        motor_result = MotorResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

        if result.returncode != 0:
            raise DesktopMotorError(
                "ydotool failed: "
                + (
                    result.stderr.strip()
                    or f"returncode={result.returncode}"
                )
            )

        return motor_result
