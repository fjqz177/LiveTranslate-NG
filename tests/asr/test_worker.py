"""Tests for the ASR worker command handlers and engine factory registry."""

import numpy as np
import pytest

import livetranslate.asr.worker as asr_worker
from livetranslate.asr.protocol import EngineCapabilities, TranscriptionResult
from livetranslate.asr.worker import ConfigError, _load_engine, _transcribe, handle_message


class _RecordingEngine:
    """Fake engine that records transcribe kwargs."""

    capabilities = EngineCapabilities(word_timestamps=True, input_padding=True)
    language = None

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, word_timestamps=False):
        self.calls.append((audio, word_timestamps))
        return TranscriptionResult(text="ok", language="en")

    def set_language(self, language):
        self.language = language

    def set_input_padding(self, pad_seconds):
        self.pad_seconds = pad_seconds

    def unload(self):
        pass


class _NoWordTsEngine(_RecordingEngine):
    capabilities = EngineCapabilities(word_timestamps=False)


def _audio():
    return np.zeros(1600, dtype=np.float32)


class TestTranscribe:
    def test_passes_word_timestamps_when_capable(self):
        engine = _RecordingEngine()
        result = _transcribe(engine, {"audio": _audio(), "word_timestamps": True})
        assert isinstance(result, TranscriptionResult)
        assert engine.calls[0][1] is True

    def test_omits_word_timestamps_when_not_capable(self):
        engine = _NoWordTsEngine()
        _transcribe(engine, {"audio": _audio(), "word_timestamps": True})
        assert engine.calls[0][1] is False

    @pytest.mark.parametrize(
        "audio",
        [
            "not an array",
            np.zeros(10, dtype=np.float64),  # wrong dtype
            np.zeros((2, 512), dtype=np.float32),  # not 1-D
            np.zeros(0, dtype=np.float32),  # empty
        ],
    )
    def test_rejects_invalid_payloads(self, audio):
        with pytest.raises((TypeError, ValueError)):
            _transcribe(_RecordingEngine(), {"audio": audio})


class TestHandleMessage:
    def test_ping(self):
        r = handle_message(_RecordingEngine(), {"id": "1", "type": "ping"})
        assert r == {"id": "1", "ok": True, "type": "pong", "payload": None}

    def test_transcribe(self):
        r = handle_message(
            _RecordingEngine(), {"id": "2", "type": "transcribe", "payload": {"audio": _audio()}}
        )
        assert r["ok"] and r["type"] == "result"
        assert r["payload"].text == "ok"

    def test_set_language(self):
        engine = _RecordingEngine()
        r = handle_message(
            engine, {"id": "3", "type": "set_language", "payload": {"language": "ja"}}
        )
        assert r["ok"] and r["type"] == "ack"
        assert engine.language == "ja"

    def test_set_input_padding(self):
        engine = _RecordingEngine()
        r = handle_message(
            engine, {"id": "4", "type": "set_input_padding", "payload": {"pad_seconds": 0.25}}
        )
        assert r["ok"]
        assert engine.pad_seconds == 0.25

    def test_shutdown(self):
        r = handle_message(_RecordingEngine(), {"id": "5", "type": "shutdown"})
        assert r["ok"] and r["type"] == "shutdown"

    def test_unknown_command_raises(self):
        with pytest.raises(ValueError):
            handle_message(_RecordingEngine(), {"id": "6", "type": "nonsense"})

    def test_missing_payload_defaults_to_empty(self):
        engine = _RecordingEngine()
        r = handle_message(engine, {"id": "7", "type": "set_language"})
        assert r["ok"]
        assert engine.language == "auto"


class TestLoadEngine:
    def test_unknown_engine_type_raises_config_error(self, monkeypatch):
        monkeypatch.setattr(asr_worker, "_ENGINE_FACTORIES", {"whisper": lambda c: None})
        with pytest.raises(ConfigError) as excinfo:
            _load_engine({"engine_type": "no-such-engine"})
        assert "no-such-engine" in str(excinfo.value)
        assert "whisper" in str(excinfo.value)

    def test_factory_result_gets_language(self, monkeypatch):
        engine = _RecordingEngine()
        monkeypatch.setattr(asr_worker, "_ENGINE_FACTORIES", {"fake": lambda c: engine})
        # apply_cache_env is imported lazily inside _load_engine; stub it
        import livetranslate.modeling.manager as model_manager

        monkeypatch.setattr(model_manager, "apply_cache_env", lambda: None)
        loaded = _load_engine({"engine_type": "fake", "language": "ja"})
        assert loaded is engine
        assert engine.language == "ja"


def test_factory_registry_covers_expected_engines():
    assert set(asr_worker._ENGINE_FACTORIES) == {
        "whisper",
        "funasr",
        "anime-whisper",
        "sensevoice-onnx",
    }


# ── _load_whisper local-cache resolution ──


def _fake_whisper_engine_module(monkeypatch, captured):
    """Stub livetranslate.asr.engines.whisper without importing faster_whisper."""
    import sys
    import types

    class FakeASREngine:
        def __init__(self, model_size, **kwargs):
            captured.update(model_size=model_size, kwargs=kwargs)

        def set_language(self, language):
            pass

        def unload(self):
            pass

    module = types.ModuleType("livetranslate.asr.engines.whisper")
    module.ASREngine = FakeASREngine
    monkeypatch.setitem(sys.modules, "livetranslate.asr.engines.whisper", module)
    return FakeASREngine


class TestLoadWhisperLocalPath:
    def test_uses_local_cache_path_when_present(self, monkeypatch):
        import livetranslate.modeling.manager as model_manager

        monkeypatch.setattr(
            model_manager,
            "get_whisper_local_path",
            lambda size, hub="ms": f"C:/fake/ms-{size}",
        )
        captured = {}
        _fake_whisper_engine_module(monkeypatch, captured)
        asr_worker._load_whisper({"model_size": "tiny", "hub": "ms"})
        assert captured["model_size"] == "C:/fake/ms-tiny"

    def test_keeps_builtin_size_when_no_cache(self, monkeypatch):
        import livetranslate.modeling.manager as model_manager

        monkeypatch.setattr(model_manager, "get_whisper_local_path", lambda *a, **k: None)
        captured = {}
        _fake_whisper_engine_module(monkeypatch, captured)
        asr_worker._load_whisper({"model_size": "tiny", "hub": "hf"})
        assert captured["model_size"] == "tiny"

    def test_hub_forwarded_to_resolver(self, monkeypatch):
        import livetranslate.modeling.manager as model_manager

        seen = {}
        monkeypatch.setattr(
            model_manager,
            "get_whisper_local_path",
            lambda size, hub="ms": seen.update(hub=hub) or None,
        )
        captured = {}
        _fake_whisper_engine_module(monkeypatch, captured)
        asr_worker._load_whisper({"model_size": "tiny", "hub": "hf"})
        assert seen == {"hub": "hf"}


# ── End-to-end protocol round-trip through a real spawn worker process ──


def _fake_worker_main(conn):
    """Runs inside a spawned subprocess: installs a fake engine factory and
    serves the real worker_main command loop."""
    from livetranslate.asr.protocol import EngineCapabilities, TranscriptionResult

    class FakeEngine:
        capabilities = EngineCapabilities()
        language = None

        def transcribe(self, audio, word_timestamps=False):
            return TranscriptionResult(text="hello world", language=self.language or "en")

        def set_language(self, language):
            self.language = None if language == "auto" else language

        def set_input_padding(self, pad_seconds):
            self.pad_seconds = pad_seconds

        def unload(self):
            pass

    asr_worker._ENGINE_FACTORIES = {"fake": lambda config: FakeEngine()}
    asr_worker.worker_main(conn, {"engine_type": "fake", "language": "ja"})


def test_worker_protocol_end_to_end():
    import multiprocessing as mp
    import uuid

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=True)
    proc = ctx.Process(target=_fake_worker_main, args=(child,))
    proc.start()
    child.close()
    try:
        # Ready handshake
        assert parent.poll(30)
        ready = parent.recv()
        assert ready["ok"] and ready["type"] == "ready"

        def request(msg_type, payload):
            parent.send({"id": uuid.uuid4().hex, "type": msg_type, "payload": payload})
            assert parent.poll(30), f"no response for {msg_type}"
            return parent.recv()

        r = request("ping", {})
        assert r["ok"] and r["type"] == "pong"

        r = request("transcribe", {"audio": _audio()})
        assert r["ok"] and r["type"] == "result"
        result = r["payload"]
        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        assert result.language == "ja"  # applied at load time

        assert request("set_language", {"language": "auto"})["ok"]
        assert request("set_input_padding", {"pad_seconds": 0.25})["ok"]

        r = request("shutdown", {})
        assert r["ok"] and r["type"] == "shutdown"
    finally:
        parent.close()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        assert proc.exitcode == 0
