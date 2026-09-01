"""Process-level ASRClient tests (P3): real spawn + Pipe round-trips.

The worker child re-imports everything fresh (spawn context), so the fake
engine is injected two ways: the module lives on `pythonpaths` (inserted
by worker_main before the factory import) and the factory override is
selected with the LIVETRANSLATE_TEST_ENGINE_FACTORY seam. No real engine
dependencies are needed. Spawns are slow (~1s each on Windows), so the
suite stays small.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from livetranslate.asr.client import ASRClient, ASRWorkerTimeout

FAKE_ENGINE = """\
import time
from types import SimpleNamespace

# Echo engine for process-level client tests: returns
# {"text": "echo-<n>", "language": ...} verbatim over the wire.


class EchoEngine:
    capabilities = SimpleNamespace(word_timestamps=True)

    def __init__(self, **kwargs):
        self._hang = bool(kwargs.get("hang"))
        self.language = None

    def set_language(self, language):
        self.language = language

    def set_input_padding(self, pad_seconds):
        pass

    def transcribe(self, audio, word_timestamps=False):
        if self._hang:
            time.sleep(60)
        return {
            "text": f"echo-{len(audio)}",
            "language": self.language or "en",
            "language_name": self.language or "English",
            "words": (),
        }

    def unload(self):
        pass


def make_echo(config):
    return EchoEngine(**config)
"""


@pytest.fixture()
def fake_engine_dir(tmp_path: Path) -> Path:
    mod = tmp_path / "fake_echo_engine.py"
    mod.write_text(FAKE_ENGINE, encoding="utf-8")
    return tmp_path


def _client(fake_engine_dir: Path, monkeypatch, request_timeout=10.0, use_seam=True, **overrides):
    if use_seam:
        monkeypatch.setenv("LIVETRANSLATE_TEST_ENGINE_FACTORY", "fake_echo_engine:make_echo")
    else:
        monkeypatch.delenv("LIVETRANSLATE_TEST_ENGINE_FACTORY", raising=False)
    config = {
        "engine_type": "not-a-real-engine",  # overridden by the test seam
        "device": "cpu",
        "pythonpaths": [str(fake_engine_dir)],
        "display_name": "test-echo",
        "language": "auto",
    }
    config.update(overrides)
    return ASRClient(config, ready_timeout=60.0, request_timeout=request_timeout)


class TestProcessRoundTrip:
    def test_spawn_ready_transcribe_shutdown(self, fake_engine_dir, monkeypatch):
        client = _client(fake_engine_dir, monkeypatch)
        client.start()
        client.wait_ready()
        assert client.status == "ready"
        assert client.pid is not None

        result = client.transcribe(np.zeros(16000, dtype=np.float32))
        assert result["text"] == "echo-16000"

        client.set_language("ja")
        result = client.transcribe(np.zeros(8000, dtype=np.float32))
        assert result["language"] == "ja"

        client.shutdown()
        assert client.status in ("stopped", "exited")

    def test_hung_worker_times_out_and_is_terminated(self, fake_engine_dir, monkeypatch):
        client = _client(fake_engine_dir, monkeypatch, request_timeout=3.0, hang=True)
        client.start()
        client.wait_ready()
        started = time.monotonic()
        with pytest.raises(ASRWorkerTimeout):
            client.transcribe(np.zeros(16000, dtype=np.float32))
        # Bound: the 3s request timeout must dominate, not the 60s hang.
        assert time.monotonic() - started < 20
        client.shutdown()

    def test_worker_death_surfaces_as_exited(self, fake_engine_dir, monkeypatch):
        from livetranslate.asr.client import ASRClientError

        client = _client(fake_engine_dir, monkeypatch)
        client.start()
        client.wait_ready()
        # Kill the worker out from under the client: the death must surface
        # immediately (status flips to "exited", requests fail fast) — not
        # spin to the request deadline.
        client._process.kill()
        client._process.join(timeout=15)
        assert client.status == "exited"
        started = time.monotonic()
        with pytest.raises(ASRClientError, match="not ready"):
            client.transcribe(np.zeros(16000, dtype=np.float32))
        assert time.monotonic() - started < 5
        client.shutdown()

    def test_unknown_engine_without_seam_raises_config_error(self, fake_engine_dir, monkeypatch):
        client = _client(fake_engine_dir, monkeypatch, use_seam=False)
        client.start()
        from livetranslate.asr.client import ASRWorkerError

        with pytest.raises(ASRWorkerError):
            client.wait_ready()
        client.shutdown()
