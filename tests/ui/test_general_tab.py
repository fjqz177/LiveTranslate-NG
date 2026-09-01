"""Tests for the General tab: language, autostart, reduced motion,
window reset and settings import/export (plan §3.2 item 1).
"""

import json

import pytest
from PyQt6.QtWidgets import QFileDialog, QMessageBox

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
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "settings.json")
    p = ControlPanel(CONFIG)
    yield p
    p.close()


def test_collect_writes_general_keys(panel):
    tab = panel._general_tab
    tab._ui_lang.setCurrentIndex(tab._ui_lang.findData("en"))
    tab._start_hidden.setChecked(True)
    tab._reduce_motion.setChecked(True)
    tab.collect()
    s = panel.get_settings()
    assert s["ui_lang"] == "en"
    assert s["start_hidden"] is True
    assert s["reduce_motion"] is True


def test_ui_lang_choices_are_system_zh_en(panel):
    tab = panel._general_tab
    assert [tab._ui_lang.itemData(i) for i in range(tab._ui_lang.count())] == [
        "system",
        "zh",
        "en",
    ]


def test_autostart_failure_disables_control(qapp, tmp_path, monkeypatch):
    assert qapp is not None

    class Boom:
        def autostart_enabled(self):
            raise OSError("no login items here")

    monkeypatch.setattr(
        "livetranslate.ui.panel.tabs.general_tab.create_system_integration",
        lambda: Boom(),
    )
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "settings.json")
    p = ControlPanel(CONFIG)
    tab = p._general_tab
    assert tab._system is None
    assert not tab._autostart.isEnabled()
    p.close()


def test_autostart_toggle_calls_backend(qapp, tmp_path, monkeypatch):
    assert qapp is not None

    class Fake:
        def __init__(self):
            self.enabled = True

        def autostart_enabled(self):
            return self.enabled

        def set_autostart(self, enabled):
            self.enabled = enabled

    fake = Fake()
    monkeypatch.setattr(
        "livetranslate.ui.panel.tabs.general_tab.create_system_integration",
        lambda: fake,
    )
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "settings.json")
    p = ControlPanel(CONFIG)
    tab = p._general_tab
    assert tab._autostart.isChecked() is True
    tab._autostart.setChecked(False)
    assert fake.enabled is False
    p.close()


def test_autostart_failure_reverts_checkbox(qapp, tmp_path, monkeypatch):
    assert qapp is not None

    class Flaky:
        def autostart_enabled(self):
            return False

        def set_autostart(self, enabled):
            raise OSError("write denied")

    monkeypatch.setattr(
        "livetranslate.ui.panel.tabs.general_tab.create_system_integration",
        lambda: Flaky(),
    )
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
    )
    p = ControlPanel(CONFIG)
    tab = p._general_tab
    tab._autostart.setChecked(True)  # should be reverted
    assert tab._autostart.isChecked() is False
    p.close()


def test_export_settings_writes_redacted_free_json(panel, monkeypatch, tmp_path):
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    out = tmp_path / "exported.json"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out), "JSON (*.json)")),
    )
    panel._general_tab._export_settings()
    data = json.loads(out.read_text(encoding="utf-8"))
    # UI-3: first-run fallbacks are the safe CPU default, never hardcoded cuda.
    assert data["asr_engine"] == "sensevoice-onnx"
    assert data["asr_device"] == "cpu"
    assert data["models"][0]["api_base"] == "https://example.com/v1"


def test_first_run_defaults_follow_recommendations(qapp, tmp_path, monkeypatch):
    """UI-3: the composition root's recommendation reaches the fallback
    settings dict (CUDA machine -> whisper/cuda; CPU -> onnx/cpu)."""
    assert qapp is not None
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "settings.json")
    p = ControlPanel(CONFIG, recommended_engine="whisper", recommended_device="cuda")
    assert p._current_settings["asr_engine"] == "whisper"
    assert p._current_settings["asr_device"] == "cuda"
    p.close()
    p2 = ControlPanel(CONFIG, recommended_engine="sensevoice-onnx", recommended_device="cpu")
    assert p2._current_settings["asr_engine"] == "sensevoice-onnx"
    assert p2._current_settings["asr_device"] == "cpu"
    p2.close()


def test_export_settings_declines_on_cancel(panel, monkeypatch, tmp_path):
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )
    out = tmp_path / "never.json"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out), "")),
    )
    panel._general_tab._export_settings()
    assert not out.exists()


def test_import_settings_seeds_store(panel, monkeypatch, tmp_path):
    src = tmp_path / "imported.json"
    src.write_text(json.dumps({"target_language": "ja", "models": []}), encoding="utf-8")
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(src), "")),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    panel._general_tab._import_settings()
    assert panel._store.snapshot()["target_language"] == "ja"


def test_import_invalid_file_reports_without_seeding(panel, monkeypatch, tmp_path):
    src = tmp_path / "broken.json"
    src.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(src), "")),
    )
    warned = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: warned.append(a) or QMessageBox.StandardButton.Ok),
    )
    panel._general_tab._import_settings()
    assert warned
    assert panel._store.snapshot().get("target_language") != "ja"


def test_reset_positions_button_emits_panel_signal(panel):
    seen = []
    panel.reset_positions.connect(lambda: seen.append(True))
    tab = panel._general_tab
    import PyQt6.QtWidgets as w

    buttons = tab.findChildren(w.QPushButton)
    reset = next(b for b in buttons if b.text() == t("btn_reset_positions"))
    reset.click()
    assert seen == [True]


def test_platform_notes_shown_when_degraded(qapp, tmp_path, monkeypatch):
    assert qapp is not None
    monkeypatch.setattr(
        "livetranslate.ui.platform_notes.platform_notes",
        lambda: ["note_wayland_positioning"],
    )
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "s.json")
    p = ControlPanel(CONFIG)
    from PyQt6.QtWidgets import QGroupBox

    titles = [b.title() for b in p._general_tab.findChildren(QGroupBox)]
    assert t("platform_notes_title") in titles
    p.close()
