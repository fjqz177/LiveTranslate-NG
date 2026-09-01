"""About-tab update check flow tests (plan §4.9 rewrite wiring)."""

import time

import pytest
from PyQt6.QtWidgets import QMessageBox

from livetranslate.core.updater import UpdateCheckResult
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


def test_check_update_reports_new_version(panel, qapp, monkeypatch):
    from livetranslate.core.i18n import t

    result = UpdateCheckResult(
        kind="new",
        new_version="9.9.9",
        url="https://github.com/fjqz177/LiveTranslate/releases/latest",
    )
    monkeypatch.setattr(
        "livetranslate.core.updater.check_latest_release",
        lambda current, timeout=10.0: result,
    )
    boxes = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: boxes.append(self) or 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: None)
    panel._about_tab._check_update()
    deadline = time.monotonic() + 5
    while not boxes and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert boxes, "update dialog never appeared"
    assert boxes[0].windowTitle() == t("update_available_title")
    assert "9.9.9" in boxes[0].text()
    assert panel._about_tab._check_btn.isEnabled()


def test_check_update_reports_failure(panel, qapp, monkeypatch):
    monkeypatch.setattr(
        "livetranslate.core.updater.check_latest_release",
        lambda current, timeout=10.0: UpdateCheckResult(kind="error", detail="boom"),
    )
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a)))
    panel._about_tab._check_update()
    deadline = time.monotonic() + 5
    while not warned and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert warned, "failure dialog never appeared"
    assert "boom" in str(warned[0])
