"""Hotkey settings group (识别 page, §3.2/§3.7): per-action rebinding.

Each action row shows a capture button. Clicking arms it (keyboard grab);
the next non-modifier keypress becomes the new combo (Esc cancels).
Intra-app duplicates are rejected inline with the §3.6 conflict copy;
OS-level conflicts surface through the shell's re-registration path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout

if TYPE_CHECKING:
    from PyQt6.QtGui import QKeyEvent

from livetranslate.core.i18n import t
from livetranslate.platform.hotkeys import HotkeyCombo
from livetranslate.ui.hotkeys import DEFAULT_HOTKEYS, HOTKEY_ACTION_KEYS
from livetranslate.ui.panel._tab_base import TabBase

log = logging.getLogger("LiveTranslate.Panel")

_MOD_MAP = {
    Qt.KeyboardModifier.ControlModifier: "ctrl",
    Qt.KeyboardModifier.AltModifier: "alt",
    Qt.KeyboardModifier.ShiftModifier: "shift",
    Qt.KeyboardModifier.MetaModifier: "super",
}


def _qt_key_to_name(key: int) -> str | None:
    """Map a Qt key code (int) to a HotkeyCombo canonical key name."""
    if Qt.Key.Key_A.value <= key <= Qt.Key.Key_Z.value:
        return chr(key)
    if Qt.Key.Key_0.value <= key <= Qt.Key.Key_9.value:
        return chr(key)
    if Qt.Key.Key_F1.value <= key <= Qt.Key.Key_F24.value:
        return f"F{key - Qt.Key.Key_F1.value + 1}"
    names = {
        Qt.Key.Key_Space.value: "SPACE",
        Qt.Key.Key_Return.value: "ENTER",
        Qt.Key.Key_Enter.value: "ENTER",
        Qt.Key.Key_Tab.value: "TAB",
        Qt.Key.Key_Backspace.value: "BACKSPACE",
    }
    return names.get(key)


class HotkeyCaptureButton(QPushButton):
    """Button that shows a combo and captures the next keypress on click."""

    captured = pyqtSignal(str)

    def __init__(self, combo: str, parent=None):
        super().__init__(combo, parent)
        self._combo = combo
        self._capturing = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._start_capture)

    @property
    def combo(self) -> str:
        return self._combo

    def set_combo(self, combo: str) -> None:
        self._combo = combo
        self._capturing = False
        self.releaseKeyboard()
        self.setText(combo)

    def _start_capture(self) -> None:
        self._capturing = True
        self.setText(t("hotkey_capture_hint"))
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.grabKeyboard()

    def _cancel_capture(self) -> None:
        self._capturing = False
        self.releaseKeyboard()
        self.setText(self._combo)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not self._capturing:
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key.Key_Escape:
            self._cancel_capture()
            return
        mods = [name for mod, name in _MOD_MAP.items() if event.modifiers() & mod]
        key_name = _qt_key_to_name(event.key())
        if key_name is None:
            return  # modifier-only or unsupported key: keep waiting
        combo = HotkeyCombo(key=key_name, mods=frozenset(mods))
        self._capturing = False
        self.releaseKeyboard()
        self._combo = str(combo)
        self.setText(self._combo)
        self.captured.emit(self._combo)

    def focusOutEvent(self, event) -> None:
        if self._capturing:
            self._cancel_capture()
        super().focusOutEvent(event)


class HotkeysGroup(TabBase):
    """识别 page group: rebind the four global hotkeys (plan §3.7)."""

    def __init__(self, panel):
        super().__init__(panel)
        group = QGroupBox(t("group_hotkeys"))
        layout = QVBoxLayout(group)
        self._buttons: dict[str, HotkeyCaptureButton] = {}
        self._warning = QLabel()
        self._warning.setObjectName("hintLabel")
        self._warning.setWordWrap(True)
        self._warning.hide()

        combos = self.settings.get("hotkeys") or dict(DEFAULT_HOTKEYS)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        for row, name in enumerate(DEFAULT_HOTKEYS):
            label = QLabel(t(HOTKEY_ACTION_KEYS[name]))
            btn = HotkeyCaptureButton(combos.get(name, DEFAULT_HOTKEYS[name]))
            btn.captured.connect(lambda text, n=name: self._on_captured(n, text))
            grid.addWidget(label, row, 0)
            grid.addWidget(btn, row, 1)
            self._buttons[name] = btn
        layout.addLayout(grid)
        layout.addWidget(self._warning)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(group)
        outer.addStretch(1)

    def combos(self) -> dict[str, str]:
        return {name: btn.combo for name, btn in self._buttons.items()}

    def set_combo(self, name: str, combo: str) -> None:
        """Revert a button from outside (OS-level conflict feedback)."""
        if name in self._buttons:
            self._buttons[name].set_combo(combo)
            hotkeys = dict(self.settings.get("hotkeys") or {})
            hotkeys[name] = combo
            self.settings["hotkeys"] = hotkeys
            self.store_save()

    def _on_captured(self, name: str, combo: str) -> None:
        # Intra-app duplicate check (both would fire the same action)
        for other_name, btn in self._buttons.items():
            if other_name != name and btn.combo == combo:
                self._buttons[name].set_combo(
                    self.settings.get("hotkeys", {}).get(name, DEFAULT_HOTKEYS[name])
                )
                self._warning.setText(
                    t("hotkey_duplicate").format(action=t(HOTKEY_ACTION_KEYS[other_name]))
                )
                self._warning.show()
                return
        self._warning.hide()
        hotkeys = dict(self.settings.get("hotkeys") or dict(DEFAULT_HOTKEYS))
        hotkeys[name] = combo
        self.settings["hotkeys"] = hotkeys
        self.auto_save()
        self.panel.hotkeys_changed.emit(dict(hotkeys))
