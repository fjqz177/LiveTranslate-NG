"""Global-hotkey adapter: Qt signals over the platform backend.

The concrete registration lives in livetranslate.platform (Win32 on
Windows, Carbon on macOS, X11 on Linux; Wayland reports unavailable). This
module only adapts: backend callbacks arrive on the backend's thread and
are marshalled onto the Qt main thread before triggered(name) fires.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Q_ARG, QMetaObject, QObject, Qt, pyqtSignal, pyqtSlot

from livetranslate.platform.hotkeys import HotkeyCombo, HotkeyStatus
from livetranslate.platform.registry import create_hotkey_backend

log = logging.getLogger("LiveTranslate.Hotkeys")

# Default bindings (plan §3.7 三平台一致项). Stored per-user under
# settings["hotkeys"]; the panel edits them and the shell re-registers.
DEFAULT_HOTKEYS = {
    "pause": "Ctrl+Alt+P",
    "overlay": "Ctrl+Alt+H",
    "subtitle": "Ctrl+Alt+S",
    "clear": "Ctrl+Alt+C",
}

HOTKEY_ACTION_KEYS = {  # action name -> i18n key for its label
    "pause": "hotkey_pause",
    "overlay": "hotkey_overlay",
    "subtitle": "hotkey_subtitle",
    "clear": "hotkey_clear",
}


class HotkeyManager(QObject):
    """Registry of global hotkeys with Qt-signal dispatch."""

    triggered = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None, backend: object | None = None) -> None:
        super().__init__(parent)
        self._backend = backend if backend is not None else create_hotkey_backend()
        self._combos: dict[str, HotkeyCombo] = {}

    def register(self, name: str, combo: HotkeyCombo | str) -> HotkeyStatus:
        """Register name -> combo; returns a status with a user-facing reason
        on failure (already in use / platform unsupported)."""
        parsed = combo if isinstance(combo, HotkeyCombo) else HotkeyCombo.parse(combo)
        status = self._backend.register(name, parsed, lambda n=name: self._on_hotkey(n))
        if status.ok:
            self._combos[name] = parsed
            log.info(f"Hotkey registered: {name} ({parsed})")
        else:
            log.warning(f"Hotkey registration failed: {name} ({parsed}): {status.reason}")
        return status

    def unregister(self, name: str) -> None:
        self._backend.unregister(name)
        self._combos.pop(name, None)

    def unregister_all(self) -> None:
        for name in list(self._combos):
            self.unregister(name)

    def stop(self) -> None:
        self.unregister_all()
        self._backend.stop()

    # Called on the backend's thread.
    def _on_hotkey(self, name: str) -> None:
        QMetaObject.invokeMethod(
            self, "_emit_triggered", Qt.ConnectionType.QueuedConnection, Q_ARG(str, name)
        )

    @pyqtSlot(str)
    def _emit_triggered(self, name: str) -> None:
        self.triggered.emit(name)
