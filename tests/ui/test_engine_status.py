"""Engine status row + one-click dependency install tests (识别页 §3.2)."""

import sys

import pytest
from PyQt6.QtWidgets import QMessageBox

from livetranslate.core.i18n import t
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


def _status(status: str, monkeypatch) -> None:
    monkeypatch.setattr(
        "livetranslate.asr.availability.engine_status",
        lambda engine_id, platform: status,
    )


def test_available_engine_hides_install_button(panel, monkeypatch):
    _status("available", monkeypatch)
    panel._vad_tab._refresh_engine_status()
    # The label now carries hardware/runtime context after the status text.
    assert panel._vad_tab._engine_status_label.text().startswith(t("engine_status_available"))
    assert panel._vad_tab._engine_install_btn.isHidden()


def test_needs_extras_shows_install_button(panel, monkeypatch):
    _status("needs-extras", monkeypatch)
    panel._vad_tab._refresh_engine_status()
    assert not panel._vad_tab._engine_install_btn.isHidden()
    assert panel._vad_tab._engine_install_btn.isEnabled()
    assert "GB" in panel._vad_tab._engine_status_label.text()


def test_frozen_needs_extras_enables_runtime_install(panel, monkeypatch):
    """SelfServe P1-B5: the frozen disabled-button placeholder is gone —
    frozen builds install the recommended runtime variant instead."""
    _status("needs-extras", monkeypatch)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    panel._vad_tab._refresh_engine_status()
    assert panel._vad_tab._engine_install_btn.isEnabled()
    assert panel._vad_tab._engine_install_btn.toolTip() == ""
    assert (
        "runtime" in panel._vad_tab._engine_install_btn.text().lower()
        or "运行时" in panel._vad_tab._engine_install_btn.text()
    )


def test_install_requires_uv(panel, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a)))
    panel._vad_tab._install_engine_deps()
    assert warned, "no uv -> user-facing warning expected"


def test_sensevoice_onnx_needs_model_shows_honest_copy(panel, monkeypatch):
    """M-MATRIX honesty: the default CPU-recommended engine is sensevoice-onnx,
    which has NO auto-downloader. A needs-model status must show the
    export/community guidance, not the generic "model downloads on switch"
    promise (which is false for the ONNX path)."""
    # Default panel selects sensevoice-onnx (no recommended_engine).
    assert panel._vad_tab._selected_engine_id() == "sensevoice-onnx"
    _status("needs-model", monkeypatch)
    panel._vad_tab._refresh_engine_status()
    text = panel._vad_tab._engine_status_label.text()
    assert text.startswith(t("engine_sensevoice_onnx_missing"))
    assert not text.startswith(t("engine_status_needs_model"))
