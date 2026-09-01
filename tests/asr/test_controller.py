"""Unit tests for AsrController (fake clients, no real worker processes)."""

import numpy as np
import pytest

from livetranslate.asr.client import ASRWorkerError, ASRWorkerExited
from livetranslate.asr.controller import AsrController
from livetranslate.asr.protocol import TranscriptionResult


class FakeClient:
    """Minimal ASRClient/RemoteASREngine surface."""

    status = "ready"
    pid = 12345

    def __init__(self, fail_transcribe=None):
        self.language = "auto"
        self.padding = None
        self.ping_ok = True
        self.shutdown_calls = 0
        self.terminate_calls = 0
        self._fail_transcribe = fail_transcribe
        self.transcribe_calls = 0

    def transcribe(self, audio, **kwargs):
        self.transcribe_calls += 1
        if self._fail_transcribe is not None:
            raise self._fail_transcribe
        return TranscriptionResult(text="hello", language="en")

    def set_language(self, language):
        self.language = language

    def set_input_padding(self, pad_seconds):
        self.padding = pad_seconds

    def ping(self):
        if not self.ping_ok:
            raise ASRWorkerExited("pipe closed")

    def shutdown(self):
        self.shutdown_calls += 1
        self.status = "stopped"

    def terminate(self):
        self.terminate_calls += 1
        self.status = "failed"


STATE = {
    "type": "whisper",
    "signature": ("whisper", "medium", "cuda", "ms", "float16"),
    "device": "cuda",
    "funasr_model_key": "sensevoice-small",
    "whisper_model_size": "medium",
    "config": {"engine_type": "whisper", "display_name": "Whisper medium"},
    "display_name": "Whisper medium",
    "device_label": "cuda",
}


def _make_controller(**kwargs):
    defaults = {
        "initial_device": "cuda",
        "initial_whisper_model_size": "medium",
        "initial_funasr_model_key": "sensevoice-small",
    }
    defaults.update(kwargs)
    return AsrController(**defaults)


class TestLifecycle:
    def test_activate_sets_state_and_ready(self):
        ctl = _make_controller()
        client = FakeClient()
        ctl.activate(client, STATE)
        assert ctl.ready is True
        assert ctl.client is client
        assert ctl.type == "whisper"
        assert ctl.device == "cuda"

    def test_detach_returns_old_client_and_state(self):
        ctl = _make_controller()
        client = FakeClient()
        ctl.activate(client, STATE)
        old_client, old_state = ctl.detach_current()
        assert old_client is client
        assert old_state["type"] == "whisper"
        assert old_state["config"]["engine_type"] == "whisper"
        assert ctl.ready is False
        assert ctl.client is None

    def test_snapshot_none_when_idle(self):
        ctl = _make_controller()
        assert ctl.snapshot_state() is None

    def test_is_ready_with_signature(self):
        ctl = _make_controller()
        ctl.activate(FakeClient(), STATE)
        assert ctl.is_ready_with_signature(STATE["signature"])
        assert not ctl.is_ready_with_signature(("whisper", "small", "cuda", "ms", "f16"))

    def test_mark_unavailable_shuts_down_and_reports(self):
        ctl = _make_controller()
        client = FakeClient()
        ctl.activate(client, STATE)
        statuses = []
        ctl._status_cb = statuses.append
        ctl.mark_unavailable("boom", client)
        assert ctl.ready is False
        assert client.shutdown_calls == 1
        assert statuses == ["ASR unavailable"]

    def test_refresh_ready(self):
        ctl = _make_controller()
        client = FakeClient()
        ctl.activate(client, STATE)
        client.status = "exited"
        ctl.refresh_ready()
        assert ctl.ready is False


class TestTranscribe:
    def test_success_reports_mem_and_clears_errors(self):
        mem = []
        ctl = _make_controller(mem_cb=lambda k, s, m: mem.append(k))
        client = FakeClient()
        ctl.activate(client, STATE)
        result, _ = ctl.transcribe(np.zeros(1600, np.float32), "segment")
        assert result.text == "hello"
        assert mem[-1] == "segment"

    def test_not_ready_returns_none(self):
        ctl = _make_controller()
        result, asr_ms = ctl.transcribe(np.zeros(1600, np.float32), "segment")
        assert result is None and asr_ms == 0.0

    def test_worker_exited_recovers(self, monkeypatch):
        statuses = []
        ctl = _make_controller(status_cb=statuses.append)
        dead = FakeClient(fail_transcribe=ASRWorkerExited("pipe closed"))
        replacement = FakeClient()
        ctl.activate(dead, STATE)
        monkeypatch.setattr(ctl, "load_engine_client", lambda config: replacement)
        with pytest.raises(ASRWorkerExited):
            ctl.transcribe(np.zeros(1600, np.float32), "segment")
        assert ctl.client is replacement
        assert ctl.ready is True
        assert statuses[-1].startswith("Whisper medium")

    def test_recoverable_errors_count_to_fatal(self):
        statuses = []
        ctl = _make_controller(status_cb=statuses.append)
        exc = ASRWorkerError({"message": "engine hiccup", "recoverable": True})
        for i in range(3):
            client = FakeClient() if i > 0 else FakeClient(fail_transcribe=exc)
            if i == 0:
                ctl.activate(client, STATE)
            with pytest.raises(ASRWorkerError):
                ctl.transcribe(np.zeros(1600, np.float32), "segment")
        # Third consecutive recoverable error marks the worker unavailable
        assert ctl.ready is False
        assert statuses[-1] == "ASR unavailable"

    def test_pending_settings_applied_before_transcribe(self):
        ctl = _make_controller()
        client = FakeClient()
        ctl.activate(client, STATE)
        ctl.set_language("ja")
        ctl.set_padding("whisper", 0.25)
        ctl.transcribe(np.zeros(1600, np.float32), "segment")
        assert client.language == "ja"
        assert client.padding == 0.25


class TestRecovery:
    def test_recover_noop_when_client_replaced(self, monkeypatch):
        ctl = _make_controller()
        dead = FakeClient()
        ctl.activate(dead, STATE)
        ctl.detach_current()
        monkeypatch.setattr(ctl, "load_engine_client", lambda c: pytest.fail("must not reload"))
        ctl.recover_worker(dead, "late failure")
        assert ctl.client is None

    def test_recover_gives_up_without_restart_state(self, monkeypatch):
        statuses = []
        ctl = _make_controller(status_cb=statuses.append)
        dead = FakeClient()
        ctl.activate(dead, STATE)
        ctl._restart_state = None  # state lost -> cannot restart
        monkeypatch.setattr(ctl, "load_engine_client", lambda c: pytest.fail("must not reload"))
        ctl.recover_worker(dead, "crashed")
        assert ctl.client is None
        assert statuses[-1] == "ASR unavailable"

    def test_generation_guard_discards_stale_restart(self, monkeypatch):
        ctl = _make_controller()
        ctl.activate(FakeClient(), STATE)
        # A newer switch bumps the generation past the restart's expectation
        stale_gen = ctl._generation
        ctl.detach_current()
        stale_client = FakeClient()
        monkeypatch.setattr(ctl, "load_engine_client", lambda c: stale_client)
        started = ctl.start_worker_from_state(STATE, stale_gen)
        assert started is False
        assert stale_client.shutdown_calls == 1
        assert ctl.client is None


class TestPing:
    def test_ping_recovers_dead_worker(self, monkeypatch):
        ctl = _make_controller()
        dead = FakeClient()
        dead.ping_ok = False
        replacement = FakeClient()
        ctl.activate(dead, STATE)
        monkeypatch.setattr(ctl, "load_engine_client", lambda c: replacement)
        ctl.maybe_ping_worker()
        assert ctl.client is replacement

    def test_ping_rate_limited(self, monkeypatch):
        ctl = _make_controller()
        dead = FakeClient()
        dead.ping_ok = False
        replacement = FakeClient()
        ctl.activate(dead, STATE)
        monkeypatch.setattr(ctl, "load_engine_client", lambda c: replacement)
        ctl.maybe_ping_worker()  # first call probes and recovers
        assert ctl.client is replacement
        replacement2 = FakeClient()
        monkeypatch.setattr(ctl, "load_engine_client", lambda c: replacement2)
        ctl.maybe_ping_worker()  # within 5s -> skipped
        assert ctl.client is replacement


class TestRssRecycle:
    """P3: the worker RSS recycle path (bound native-side leaks in the
    long-lived worker process)."""

    @staticmethod
    def _fake_psutil(monkeypatch, rss_mb):
        from types import SimpleNamespace

        def _process(_pid):
            return SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=rss_mb * 1024 * 1024))

        # The controller does `import psutil` inside the method, so the real
        # module's Process symbol is the injection point.
        monkeypatch.setattr("psutil.Process", _process)

    def test_first_call_establishes_baseline_without_recycle(self, monkeypatch):
        ctl = _make_controller()
        client = FakeClient()
        ctl.activate(client, STATE)
        self._fake_psutil(monkeypatch, rss_mb=100)
        ctl.maybe_recycle_worker()
        assert ctl._worker_baseline_mb == 100
        assert ctl.client is client  # no recycle on the first probe

    def test_rss_growth_under_delta_is_left_alone(self, monkeypatch):
        ctl = _make_controller()
        client = FakeClient()
        ctl.activate(client, STATE)
        ctl._worker_baseline_mb = 100
        self._fake_psutil(monkeypatch, rss_mb=100 + ctl._recycle_delta_mb - 1)
        monkeypatch.setattr(ctl, "recycle_worker", lambda *a: pytest.fail("must not recycle"))
        ctl.maybe_recycle_worker()
        assert ctl.client is client

    def test_rss_growth_over_delta_recycles(self, monkeypatch):
        ctl = _make_controller()
        client = FakeClient()
        ctl.activate(client, STATE)
        ctl._worker_baseline_mb = 100
        recycled = []
        monkeypatch.setattr(ctl, "recycle_worker", lambda old, state: recycled.append(old))
        self._fake_psutil(monkeypatch, rss_mb=100 + ctl._recycle_delta_mb + 1)
        ctl.maybe_recycle_worker()
        assert recycled == [client]

    def test_recycle_skipped_for_in_process_remote_engine(self, monkeypatch):
        """pid=None (remote) and unknown pids must never recycle."""
        ctl = _make_controller()
        client = FakeClient()
        client.pid = None
        ctl.activate(client, STATE)
        monkeypatch.setattr(ctl, "recycle_worker", lambda *a: pytest.fail("must not recycle"))
        ctl.maybe_recycle_worker()
        assert ctl.client is client

    def test_psutil_failure_is_silent(self, monkeypatch):
        ctl = _make_controller()
        client = FakeClient()
        ctl.activate(client, STATE)
        monkeypatch.setattr(
            "psutil.Process",
            lambda pid: (_ for _ in ()).throw(OSError()),
        )
        monkeypatch.setattr(ctl, "recycle_worker", lambda *a: pytest.fail("must not recycle"))
        ctl.maybe_recycle_worker()
        assert ctl.client is client
