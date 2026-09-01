"""M-SPLIT smoke: vad_tab composes _EngineRuntimeMixin + _WhisperDownloadMixin.

Guards the split contract (zero public-symbol change):
- VadTab is actually an instance of both plain-object mixins (MRO valid,
  mixin methods resolve on the concrete instance);
- the moved engine-status / whisper-download methods still run without
  raising on a fully-built panel;
- the engine dropdown is built from the single-source registry table
  (GUI_ENGINE_ORDER / ENGINE_REGISTRY) — 5 items, userData = registry id.
"""

import pytest

from livetranslate.asr.registry import ENGINE_REGISTRY, GUI_ENGINE_ORDER
from livetranslate.ui.panel.panel import ControlPanel
from livetranslate.ui.panel.tabs.vad_engine import _EngineRuntimeMixin
from livetranslate.ui.panel.tabs.vad_tab import VadTab
from livetranslate.ui.panel.tabs.vad_whisper import _WhisperDownloadMixin

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


def test_vadtab_mixins_and_mro(panel):
    tab = panel._vad_tab
    assert isinstance(tab, VadTab)
    assert isinstance(tab, _EngineRuntimeMixin)
    assert isinstance(tab, _WhisperDownloadMixin)
    # The moved methods resolve through the instance (mixin methods run on the
    # concrete VadTab), and the signals stay on VadTab.
    assert callable(tab._refresh_engine_status)
    assert callable(tab._populate_whisper_models)
    assert callable(tab._download_whisper)


def test_engine_order_matches_gui_registry(panel):
    """M-MATRIX: the dropdown is built from the single-source GUI_ENGINE_ORDER
    (5 items, userData = registry id) — the old 4-item _ENGINE_IDS is gone."""
    ids = [
        panel._vad_tab._asr_engine.itemData(i) for i in range(panel._vad_tab._asr_engine.count())
    ]
    assert ids == list(GUI_ENGINE_ORDER)
    for engine_id in ids:
        assert engine_id in ENGINE_REGISTRY


def test_refresh_engine_status_no_raise(panel, monkeypatch):
    monkeypatch.setattr(
        "livetranslate.asr.availability.engine_status",
        lambda engine_id, platform: "available",
    )
    tab = panel._vad_tab
    tab._refresh_engine_status()  # must not raise
    assert tab._engine_status_label.text()


def test_populate_whisper_models_no_raise(panel):
    tab = panel._vad_tab
    tab._populate_whisper_models(tab.settings.get("whisper_model_size", "medium"))
    assert tab._whisper_size_combo.count() > 0
