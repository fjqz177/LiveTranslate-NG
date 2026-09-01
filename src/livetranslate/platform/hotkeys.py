"""Global hotkey protocol and the cross-platform combo model.

The GUI binds user-facing names (e.g. "Ctrl+Alt+P") to callbacks through a
backend; per-OS backends live in platform/hotkey_backends/. Callbacks are
invoked from the backend's own thread — the Qt layer marshals them onto the
UI thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class HotkeyCombo:
    """Platform-independent hotkey expression.

    key is a canonical key name ("A".."Z", "0".."9", "F1".."F24", "SPACE",
    "ENTER", ...); mods holds the canonical modifier names. The platform
    mapping tables (one per backend) translate these to OS keycodes.
    """

    key: str
    mods: frozenset[str] = field(default_factory=frozenset)

    _DISPLAY_MODS: ClassVar[dict[str, str]] = {
        "ctrl": "Ctrl",
        "alt": "Alt",
        "shift": "Shift",
        "super": "Super",
    }
    _MOD_ORDER: ClassVar[tuple[str, ...]] = ("ctrl", "alt", "shift", "super")

    def __str__(self) -> str:
        parts = [self._DISPLAY_MODS[m] for m in self._MOD_ORDER if m in self.mods]
        parts.append(self.key)
        return "+".join(parts)

    @classmethod
    def parse(cls, text: str) -> HotkeyCombo:
        """Parse a canonical hotkey string ("Ctrl+Alt+P", "F9", "Shift+SPACE").

        Case-insensitive on modifiers and single-letter keys.
        """
        if any(not t.strip() for t in text.split("+")):
            raise ValueError(f"empty token in hotkey: {text!r}")
        tokens = [t.strip().lower() for t in text.split("+")]
        if not tokens:
            raise ValueError(f"empty hotkey: {text!r}")
        key = tokens[-1].upper()
        if (
            len(key) != 1
            and not (key.startswith("F") and key[1:].isdigit())
            and key not in ("SPACE", "ENTER", "TAB", "ESC", "BACKSPACE")
        ):
            raise ValueError(f"unsupported hotkey key: {text!r}")
        mods: set[str] = set()
        for t in tokens[:-1]:
            if t in ("ctrl", "control"):
                mods.add("ctrl")
            elif t in ("alt", "option"):
                mods.add("alt")
            elif t == "shift":
                mods.add("shift")
            elif t in ("super", "win", "cmd", "meta"):
                mods.add("super")
            else:
                raise ValueError(f"unsupported hotkey modifier: {t!r}")
        return cls(key=key, mods=frozenset(mods))


@dataclass(frozen=True)
class HotkeyStatus:
    """Registration outcome with a user-facing explanation."""

    ok: bool
    reason: str = ""  # why it failed / why the platform cannot do hotkeys
    degraded: bool = False  # works but limited (e.g. focus-only on XWayland)
    note: str = ""  # degraded-mode description


class HotkeyBackend(Protocol):
    """Platform hotkey registration."""

    name: str

    def register(
        self, name: str, combo: HotkeyCombo, callback: Callable[[], None]
    ) -> HotkeyStatus: ...

    def unregister(self, name: str) -> None: ...

    def stop(self) -> None: ...

    def capability(self) -> HotkeyStatus:
        """Platform-level availability (before any registration)."""
        ...
