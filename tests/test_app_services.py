"""Tests for ui/app_services (M-COMPOSE): the pure ``build_worker_config`` +
``apply_settings`` extraction from ``livetranslate.app``.

``build_worker_config`` is Qt-free; ``apply_settings`` is a plain function
driven by fake collaborators. Neither module imports ``livetranslate.app``, so
these tests need no QApplication — the plan/routing logic under test is pure.
"""

from __future__ import annotations

import pytest

from livetranslate.ui.app_services.settings_applier import apply_settings
from livetranslate.ui.app_services.worker_config import build_worker_config

# ── Shared fixtures / fakes ──────────────────────────────────────────────


def _base_config():
    return {
        "asr": {
            "asr_engine": "funasr",
            "model_size": "medium",
            "compute_type": "int8",
            "language": "auto",
            "sensevoice_pad_seconds": 0.5,
            "whisper_pad_seconds": 0.4,
            "remote_asr_url": "http://127.0.0.1:8765",
        },
    }


class _FakeCtl:
    """Minimal AsrController stand-in exposing the read surface the plan uses."""

    def __init__(self, ready: bool = False):
        self.funasr_model_key = "sensevoice-small"
        self.device = "cpu"
        self.whisper_model_size = "medium"
        self._ready = ready

    def is_ready_with_signature(self, signature) -> bool:
        return self._ready


@pytest.fixture
def _patch_state(monkeypatch):
    """Pin the filesystem-ish helpers the plan reads to deterministic values."""
    monkeypatch.setattr(
        "livetranslate.ui.app_services.worker_config.resolve_custom_whisper_model",
        lambda size: None,
    )
    monkeypatch.setattr(
        "livetranslate.ui.app_services.worker_config.is_asr_cached",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "livetranslate.ui.app_services.worker_config.sensevoice_onnx_model_present",
        lambda: True,
    )


# ── build_worker_config: pure plan ───────────────────────────────────────


def test_whisper_plan_signature_and_fields(_patch_state):
    plan = build_worker_config(
        _base_config(),
        {"asr_engine": "whisper", "whisper_model_size": "small", "hub": "hf"},
        _FakeCtl(),
        "whisper",
    )
    assert plan.engine_type == "whisper"
    # signature = (engine_type, signature_model, device, hub, compute)
    assert plan.signature == ("whisper", "small", "cpu", "hf", "int8")
    assert plan.worker_config["engine_type"] == "whisper"
    assert plan.worker_config["model_size"] == "small"
    assert plan.worker_config["pad_seconds"] == 0.4  # whisper pad key
    assert plan.target_state["whisper_model_size"] == "small"
    assert plan.target_state["display_name"] == "Whisper small"
    assert plan.sensevoice_missing is False
    assert plan.already_ready is False


def test_funasr_plan_signature_and_fields(_patch_state):
    plan = build_worker_config(
        _base_config(),
        {"asr_engine": "funasr", "funasr_model": "funasr-nano-2512", "hub": "ms"},
        _FakeCtl(),
        "funasr",
    )
    assert plan.engine_type == "funasr"
    assert plan.funasr_model == "funasr-nano-2512"
    # signature model is the (normalized) funasr model key
    assert plan.signature == ("funasr", "funasr-nano-2512", "cpu", "ms", "int8")
    assert plan.worker_config["pad_seconds"] == 0.5  # sensevoice/funasr pad key
    assert plan.target_state["funasr_model_key"] == "funasr-nano-2512"
    assert plan.target_state["display_name"] == "Fun-ASR-Nano"


def test_remote_whisper_plan_url_and_token_signature(_patch_state):
    plan = build_worker_config(
        _base_config(),
        {
            "asr_engine": "remote-whisper",
            "remote_asr_url": "http://10.0.0.5:9000",
            "remote_asr_token": "tok123",
        },
        _FakeCtl(),
        "remote-whisper",
    )
    assert plan.engine_type == "remote-whisper"
    # URL + token are part of the identity (SEC-5).
    assert plan.signature[1] == "http://10.0.0.5:9000|tok123"
    assert plan.worker_config["remote_asr_url"] == "http://10.0.0.5:9000"
    assert plan.worker_config["remote_asr_token"] == "tok123"
    assert plan.worker_config["pad_seconds"] is None
    assert plan.target_state["device_label"] == "http://10.0.0.5:9000"


@pytest.mark.parametrize(
    "engine,input_settings,expected",
    [
        # funasr & sensevoice-onnx share the sensevoice pad key
        ("funasr", {}, 0.5),
        ("sensevoice-onnx", {}, 0.5),
        ("whisper", {}, 0.4),
        ("whisper", {"whisper_pad_seconds": 0.9}, 0.9),
        ("funasr", {"sensevoice_pad_seconds": 0.7}, 0.7),
        # remote-whisper has no pad key at all
        ("remote-whisper", {"whisper_pad_seconds": 0.9}, None),
    ],
)
def test_pad_seconds_keyed_by_engine(_patch_state, engine, input_settings, expected):
    plan = build_worker_config(
        _base_config(), {"asr_engine": engine, **input_settings}, _FakeCtl(), engine
    )
    assert plan.worker_config["pad_seconds"] == expected


def test_sensevoice_onnx_missing_flag(_patch_state, monkeypatch):
    monkeypatch.setattr(
        "livetranslate.ui.app_services.worker_config.sensevoice_onnx_model_present",
        lambda: False,
    )
    plan = build_worker_config(
        _base_config(), {"asr_engine": "sensevoice-onnx"}, _FakeCtl(), "sensevoice-onnx"
    )
    assert plan.sensevoice_missing is True
    # A different engine is never gateway on the artifact.
    plan = build_worker_config(_base_config(), {"asr_engine": "funasr"}, _FakeCtl(), "funasr")
    assert plan.sensevoice_missing is False


def test_already_ready_flag(_patch_state):
    plan = build_worker_config(
        _base_config(), {"asr_engine": "whisper"}, _FakeCtl(ready=True), "whisper"
    )
    assert plan.already_ready is True


# ── apply_settings: routing to collaborators ─────────────────────────────


class _Recorder:
    """Base fake: records every (method, args) call."""

    def __init__(self):
        self.calls: list[tuple] = []

    def _push(self, name, *args):
        self.calls.append((name, *args))


class _FakeVad(_Recorder):
    def update_settings(self, settings):
        self._push("update_settings", settings)

    def reset(self):
        self._push("reset")


class _FakeAsrCtl(_Recorder):
    def __init__(self, type=None):
        super().__init__()
        self.type = type

    def set_language(self, lang):
        self._push("set_language", lang)

    def set_padding(self, engine_type, pad):
        self._push("set_padding", engine_type, pad)


class _FakeTranslator(_Recorder):
    def set_timeout(self, timeout):
        self._push("set_timeout", timeout)


class _FakePipeline(_Recorder):
    def __init__(self):
        super().__init__()
        self.translator = _FakeTranslator()

    def set_asr_language(self, lang):
        self._push("set_asr_language", lang)

    def set_incremental(self, value):
        self._push("set_incremental", value)

    def set_interim_interval(self, value):
        self._push("set_interim_interval", value)

    def set_target_language(self, lang):
        self._push("set_target_language", lang)

    def set_log_transcript(self, value):
        self._push("set_log_transcript", value)


class _FakeAudio(_Recorder):
    def __init__(self, device_id="old"):
        super().__init__()
        self.device_id = device_id

    def switch_device(self, device):
        self._push("switch_device", device)

    def switch_mic(self, mic):
        self._push("switch_mic", mic)


class _FakeOverlay(_Recorder):
    def apply_style(self, style):
        self._push("apply_style", style)

    def update_monitor(self, a, b):
        self._push("update_monitor", a, b)

    def set_target_language(self, lang):
        self._push("set_target_language", lang)

    def set_reduce_motion(self, value):
        self._push("set_reduce_motion", value)


class _FakeSubwin(_Recorder):
    def set_reduce_motion(self, value):
        self._push("set_reduce_motion", value)


class _FakeTranscript(_Recorder):
    def set_enabled(self, value):
        self._push("set_enabled", value)


class _FakeSwitcher(_Recorder):
    def switch(self, engine_type):
        self._push("switch", engine_type)


def _collaborators(**overrides):
    """Build a full collaborator bundle, overriding individual fakes."""
    bundle = {
        "vad": _FakeVad(),
        "asr_ctl": _FakeAsrCtl(),
        "pipeline": _FakePipeline(),
        "audio": _FakeAudio(),
        "overlay": _FakeOverlay(),
        "subwin": _FakeSubwin(),
        "transcript": _FakeTranscript(),
        "config": _base_config(),
        "switcher": _FakeSwitcher(),
    }
    bundle.update(overrides)
    return bundle


def _names(calls) -> list[str]:
    return [call[0] for call in calls]


def test_each_settings_key_routes_to_its_collaborator():
    c = _collaborators()
    apply_settings(
        {
            "style": "dark",
            "asr_language": "zh",
            "whisper_pad_seconds": 0.9,
            "audio_device": "dev2",
            "mic_device": "mic1",
            "incremental_asr": True,
            "interim_interval": 250,
            "target_language": "en",
            "timeout": 30,
            "auto_save_transcript": True,
            "log_transcript": True,
            "reduce_motion": True,
        },
        **c,
    )
    assert "update_settings" in _names(c["vad"].calls)
    assert ("apply_style", "dark") in c["overlay"].calls
    assert ("set_language", "zh") in c["asr_ctl"].calls
    assert ("set_padding", "whisper", 0.9) in c["asr_ctl"].calls
    assert ("switch_device", "dev2") in c["audio"].calls
    assert ("switch_mic", "mic1") in c["audio"].calls
    assert ("set_incremental", True) in c["pipeline"].calls
    assert ("set_interim_interval", 250) in c["pipeline"].calls
    assert ("set_target_language", "en") in c["pipeline"].calls
    assert ("set_target_language", "en") in c["overlay"].calls
    assert ("set_timeout", 30) in c["pipeline"].translator.calls
    assert ("set_enabled", True) in c["transcript"].calls
    assert ("set_log_transcript", True) in c["pipeline"].calls
    assert ("set_reduce_motion", True) in c["overlay"].calls
    assert ("set_reduce_motion", True) in c["subwin"].calls


def test_asr_keys_call_switcher_switch():
    c = _collaborators()
    apply_settings({"asr_engine": "whisper"}, **c)
    assert c["switcher"].calls == [("switch", "whisper")]


def test_asr_switch_uses_current_engine_when_key_absent():
    # "asr_device" is an ASR key but the engine comes from ctl/current config.
    c = _collaborators(asr_ctl=_FakeAsrCtl(type="funasr"))
    apply_settings({"asr_device": "cuda"}, **c)
    assert c["switcher"].calls == [("switch", "funasr")]


def test_pipeline_set_asr_language_receives_snapshot():
    c = _collaborators()
    apply_settings({"asr_language": "ja"}, **c)
    assert ("set_asr_language", "ja") in c["pipeline"].calls
    # It must be the literal snapshot value, not a re-read from a live store.
    assert ("set_asr_language", "ja") in c["pipeline"].calls


def test_sensevoice_pad_keyed_by_current_engine_type():
    c = _collaborators(asr_ctl=_FakeAsrCtl(type="sensevoice-onnx"))
    apply_settings({"sensevoice_pad_seconds": 0.7}, **c)
    assert ("set_padding", "sensevoice-onnx", 0.7) in c["asr_ctl"].calls


def test_sensevoice_pad_skipped_for_other_engines():
    c = _collaborators(asr_ctl=_FakeAsrCtl(type="whisper"))
    apply_settings({"sensevoice_pad_seconds": 0.7}, **c)
    assert c["asr_ctl"].calls == []


def test_audio_device_resets_vad_when_device_changes():
    c = _collaborators(audio=_FakeAudio(device_id="dev1"))
    apply_settings({"audio_device": "dev2"}, **c)
    assert ("switch_device", "dev2") in c["audio"].calls
    assert ("reset",) in c["vad"].calls
    assert ("update_monitor", 0.0, 0.0) in c["overlay"].calls


def test_audio_device_skips_reset_when_unchanged():
    c = _collaborators(audio=_FakeAudio(device_id="dev2"))
    apply_settings({"audio_device": "dev2"}, **c)
    assert ("switch_device", "dev2") in c["audio"].calls
    assert ("reset",) not in c["vad"].calls


def test_none_overlay_and_subwin_are_guarded():
    c = _collaborators(overlay=None, subwin=None)
    # These keys touch overlay/subwin; None must not crash.
    apply_settings({"style": "dark", "target_language": "en", "reduce_motion": True}, **c)
    assert ("set_target_language", "en") in c["pipeline"].calls
