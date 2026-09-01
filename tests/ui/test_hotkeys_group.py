"""Hotkey settings group tests (识别 page rebinding, §3.2/§3.7)."""

import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent

from livetranslate.ui.hotkeys import DEFAULT_HOTKEYS
from livetranslate.ui.panel.panel import ControlPanel

CONFIG = {
    "translation": {
        "model": "gpt-test",
        "api_base": "https://example.com/v1",
        "api_key": "sk-test",
        "target_language": "zh",
    },
    "asr": {
        "vad_threshold": 0.5,
        "min_speech_duration": 0.3,
        "max_speech_duration": 20.0,
        "language": "auto",
        "device": "cpu",
    },
}


@pytest.fixture
def panel(qapp, tmp_path, monkeypatch):
    assert qapp is not None
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "s.json")
    p = ControlPanel(CONFIG)
    yield p
    p.close()


def _press(button, key, modifiers=Qt.KeyboardModifier.NoModifier):
    event = QKeyEvent(QEvent.Type.KeyPress, key, modifiers)
    button.keyPressEvent(event)


def test_defaults_from_settings(panel):
    group = panel._hotkeys_group
    assert set(group._buttons) == set(DEFAULT_HOTKEYS)
    assert group._buttons["pause"].combo == "Ctrl+Alt+P"


def test_capture_updates_setting_and_emits(panel):
    group = panel._hotkeys_group
    seen = []
    panel.hotkeys_changed.connect(lambda c: seen.append(c))
    btn = group._buttons["pause"]
    btn._start_capture()
    _press(
        btn,
        Qt.Key.Key_P,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier,
    )
    assert btn.combo == "Ctrl+Alt+P"
    assert panel.get_settings()["hotkeys"]["pause"] == "Ctrl+Alt+P"
    assert seen and seen[-1]["pause"] == "Ctrl+Alt+P"


def test_escape_cancels_capture(panel):
    group = panel._hotkeys_group
    btn = group._buttons["clear"]
    original = btn.combo
    btn._start_capture()
    _press(btn, Qt.Key.Key_Escape)
    assert btn.combo == original
    assert btn.text() == "Ctrl+Alt+C"


def test_modifier_only_keypress_is_ignored(panel):
    group = panel._hotkeys_group
    btn = group._buttons["clear"]
    btn._start_capture()
    _press(btn, Qt.Key.Key_Control, Qt.KeyboardModifier.ControlModifier)
    assert btn._capturing is True  # still waiting for a real key
    _press(btn, Qt.Key.Key_F9)
    assert btn.combo == "F9"


def test_duplicate_combo_rejected_inline(panel):
    group = panel._hotkeys_group
    btn = group._buttons["overlay"]
    btn._start_capture()
    _press(
        btn,
        Qt.Key.Key_P,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier,
    )
    # Reverted to the previous overlay binding, warning visible
    assert btn.combo == "Ctrl+Alt+H"
    assert not group._warning.isHidden()
    assert panel.get_settings()["hotkeys"]["overlay"] == "Ctrl+Alt+H"


def test_panel_revert_api_updates_button(panel):
    group = panel._hotkeys_group
    panel.set_hotkey_combo("pause", "F5")
    assert group._buttons["pause"].combo == "F5"
    assert panel.get_settings()["hotkeys"]["pause"] == "F5"
