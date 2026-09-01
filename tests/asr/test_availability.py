"""Tests for engine availability probing."""

import pytest

from livetranslate.asr import availability


class TestProbeMap:
    def test_sensevoice_onnx_engine_type(self):
        from livetranslate.asr.registry import ENGINE_REGISTRY

        assert ENGINE_REGISTRY["sensevoice-onnx"].engine_type == "sensevoice-onnx"


class TestEngineStatus:
    def test_unknown_engine_raises(self):
        with pytest.raises(KeyError):
            availability.engine_status("no-such", "win32")

    def test_unsupported_platform(self):
        # All registry engines are Windows-only; a non-win32 platform is unsupported.
        assert availability.engine_status("faster-whisper", "linux") == "unsupported"

    def test_available_on_win32(self):
        # Full install: engine deps ship with the build, so a supported engine
        # is always available on win32 (no needs-extras per-extra probe).
        assert availability.engine_status("faster-whisper", "win32") == "available"

    def test_sensevoice_onnx_without_model(self, monkeypatch):
        monkeypatch.setattr(availability, "sensevoice_onnx_model_present", lambda: False)
        assert availability.engine_status("sensevoice-onnx", "win32") == "needs-model"

    def test_sensevoice_onnx_available(self, monkeypatch):
        monkeypatch.setattr(availability, "sensevoice_onnx_model_present", lambda: True)
        assert availability.engine_status("sensevoice-onnx", "win32") == "available"
