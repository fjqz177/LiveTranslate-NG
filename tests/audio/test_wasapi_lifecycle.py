"""AUD-2 regression: the WASAPI backend must terminate the PortAudio
context explicitly on every start/stop cycle and device restart — relying
on __del__ leaks one native PortAudio context per restart on CPython
timing that is not guaranteed (and never fires on cycles where the backend
object survives)."""

import sys
import types
from typing import ClassVar

import pytest

from livetranslate.audio.backends.wasapi import WasapiBackend


class _FakeStream:
    def __init__(self):
        self.closed = False
        self.stopped = False

    def get_read_available(self):
        return 0

    def read(self, _n, exception_on_overflow=False):
        return b""

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _FakePA:
    instances: ClassVar[list["_FakePA"]] = []

    def __init__(self):
        self.terminated = False
        self.streams: list[_FakeStream] = []
        _FakePA.instances.append(self)

    def terminate(self):
        self.terminated = True

    def get_host_api_count(self):
        return 1

    def get_host_api_info_by_index(self, i):
        return {"name": "WASAPI", "index": i, "defaultOutputDevice": 0}

    def get_device_count(self):
        return 1

    def get_device_info_by_index(self, i):
        return {
            "index": i,
            "name": "Speakers [Loopback]",
            "hostApi": 0,
            "isLoopbackDevice": True,
            "maxInputChannels": 2,
            "defaultSampleRate": 48000,
        }

    def open(self, **kwargs):
        stream = _FakeStream()
        self.streams.append(stream)
        return stream


@pytest.fixture()
def fake_pyaudio(monkeypatch):
    """Replace both the importable module and the already-bound
    wasapi.pyaudio reference (the module-level `import pyaudiowpatch as
    pyaudio` happened when the test file was imported)."""
    module = types.ModuleType("pyaudiowpatch")
    module.PyAudio = _FakePA
    module.paFloat32 = 1
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", module)
    monkeypatch.setattr("livetranslate.audio.backends.wasapi.pyaudio", module)
    _FakePA.instances = []
    return module


@pytest.mark.usefixtures("fake_pyaudio")
def _backend() -> WasapiBackend:
    backend = WasapiBackend()
    backend.sample_rate = 16000
    backend.chunk_duration = 0.032
    return backend


class TestPortAudioLifecycle:
    @pytest.mark.usefixtures("fake_pyaudio")
    def test_start_terminates_previous_context(self):
        backend = _backend()
        backend.start(device_id="Speakers [Loopback]")
        assert len(_FakePA.instances) == 1

        backend.start(device_id="Speakers [Loopback]")
        assert len(_FakePA.instances) == 2
        assert _FakePA.instances[0].terminated is True, "old context must be terminated"
        backend.stop()

    @pytest.mark.usefixtures("fake_pyaudio")
    def test_stop_terminates_context(self):
        backend = _backend()
        backend.start(device_id="Speakers [Loopback]")
        assert not _FakePA.instances[0].terminated

        backend.stop()
        assert _FakePA.instances[0].terminated is True
        assert backend._pa is None

    @pytest.mark.usefixtures("fake_pyaudio")
    def test_device_restart_terminates_context(self):
        backend = _backend()
        backend.start(device_id="Speakers [Loopback]")
        assert len(_FakePA.instances) == 1

        backend._restart_stream()
        assert _FakePA.instances[0].terminated is True
        assert len(_FakePA.instances) == 2
        backend.stop()

    @pytest.mark.usefixtures("fake_pyaudio")
    def test_restart_after_stop_does_not_crash(self):
        """_restart_stream may race stop() (join timeout); a None context
        must not crash the read thread."""
        backend = _backend()
        backend.start(device_id="Speakers [Loopback]")
        backend.stop()
        assert backend._pa is None
        backend._restart_stream()  # defensive path: re-creates a context
        assert backend._pa is not None
        backend.stop()
