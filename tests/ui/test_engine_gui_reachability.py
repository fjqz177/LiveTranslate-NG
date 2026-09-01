"""M-MATRIX GUI reachability: sensevoice-onnx is genuinely selectable.

Guards the M1 fix end-to-end at the GUI layer:
- a default (no ``recommended_engine``) panel selects ``sensevoice-onnx`` on the
  CPU recommendation path and persists the *worker-frontier* engine_type;
- the CUDA recommendation path (``recommend_engine`` -> faster-whisper ->
  ``engine_type "whisper"``) selects ``faster-whisper`` in the dropdown and
  persists ``asr_engine == "whisper"``.
No `_UI_ENGINE_ALIAS` is involved — the choice comes straight from the registry.
"""

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


def _panel(qapp, tmp_path, monkeypatch, *, recommended_engine=None):
    assert qapp is not None
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "s.json")
    return ControlPanel(CONFIG, recommended_engine=recommended_engine)


def test_cpu_default_selects_and_persists_sensevoice_onnx(qapp, tmp_path, monkeypatch):
    """No recommended_engine -> the CPU fallback (sensevoice-onnx) is selected
    in the dropdown and written back as the worker-frontier engine_type."""
    panel = _panel(qapp, tmp_path, monkeypatch)
    try:
        tab = panel._vad_tab
        assert tab._selected_engine_id() == "sensevoice-onnx"
        # M-MATRIX regression guard: a non-remote default must NOT show the
        # Remote ASR URL/token group. The first engine-visibility pass runs
        # after every group is built, so the method (single source of the
        # decision) hides remote_group for non-remote engine types.
        # isHidden() (not isVisible()) is the right probe: a child of an unshown
        # parent reads isVisible()==False either way, while isHidden() is True
        # only when setVisible(False) was actually called.
        assert tab._remote_group.isHidden() is True
        assert tab._whisper_group.isHidden() is True
        tab.collect()
        assert panel.get_settings()["asr_engine"] == "sensevoice-onnx"
    finally:
        panel.close()


def test_cuda_recommendation_selects_and_persists_whisper(qapp, tmp_path, monkeypatch):
    """recommend_engine(cuda) resolves to faster-whisper (engine_type "whisper")
    through the registry; the dropdown selects faster-whisper and collect()
    persists the worker-frontier "whisper" value."""
    panel = _panel(qapp, tmp_path, monkeypatch, recommended_engine="whisper")
    try:
        tab = panel._vad_tab
        assert tab._selected_engine_id() == "faster-whisper"
        tab.collect()
        assert panel.get_settings()["asr_engine"] == "whisper"
    finally:
        panel.close()
