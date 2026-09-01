"""Contract tests for the null audio backend."""

import numpy as np

from livetranslate.audio.backends.null import NullAudioBackend


def test_read_chunk_shape_and_dtype():
    backend = NullAudioBackend()
    backend.start(sample_rate=16000, chunk_ms=32)
    chunk, mic_rms = backend.read_chunk()
    assert isinstance(chunk, np.ndarray)
    assert chunk.dtype == np.float32
    assert chunk.shape == (512,)
    assert mic_rms is None
    assert not chunk.any()  # silence


def test_read_chunk_respects_configured_rate():
    backend = NullAudioBackend()
    backend.start(sample_rate=8000, chunk_ms=50)
    chunk, _ = backend.read_chunk()
    assert chunk.shape == (400,)


def test_stopped_backend_yields_none():
    backend = NullAudioBackend()
    backend.start()
    backend.stop()
    assert backend.read_chunk() is None


def test_no_devices_are_reported():
    backend = NullAudioBackend()
    assert backend.list_outputs() == []
    assert backend.list_inputs() == []


def test_diagnostics_are_stable():
    backend = NullAudioBackend()
    backend.start()
    d = backend.diagnostics()
    assert d["backend"] == "null"
    assert d["rate"] == 16000
    assert d["status"] == "running"
