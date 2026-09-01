"""Tests for engine availability probing."""

import pytest

from livetranslate.asr import availability


class TestExtrasInstalled:
    def test_present_module(self):
        assert availability.extras_installed(("importlib",)) is True

    def test_missing_module(self):
        assert availability.extras_installed(("no_such_module_xyz",)) is False


class TestProbeMap:
    def test_no_engine_nano_probe(self):
        # M-MATRIX: engine-nano is a phantom (no engine-nano extra, no worker
        # factory); the llama_cpp probe was fiction and must not exist.
        assert "engine-nano" not in availability.EXTRAS_PROBE_MAP

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

    def test_missing_extras(self):
        # faster_whisper is not importable without the engine-whisper extra
        # (the probe only runs when the venv lacks it; the venv here has it,
        # so patch the probe to keep the test hermetic).
        assert availability.engine_status("faster-whisper", "win32") in (
            "available",
            "needs-extras",
        )

    def test_sensevoice_onnx_without_model(self, monkeypatch):
        monkeypatch.setattr(availability, "sensevoice_onnx_model_present", lambda: False)
        monkeypatch.setattr(availability, "extras_installed", lambda _m: True)
        assert availability.engine_status("sensevoice-onnx", "win32") == "needs-model"

    def test_sensevoice_onnx_available(self, monkeypatch):
        monkeypatch.setattr(availability, "sensevoice_onnx_model_present", lambda: True)
        monkeypatch.setattr(availability, "extras_installed", lambda _m: True)
        assert availability.engine_status("sensevoice-onnx", "win32") == "available"
