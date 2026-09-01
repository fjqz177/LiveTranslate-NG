"""Tests for pipeline.py orchestration with fake components (no Qt, no audio)."""

import logging
import queue
import time

import numpy as np

from livetranslate.asr.protocol import TranscriptionResult
from livetranslate.core.pipeline import Pipeline


class FakeCtl:
    ready = True

    def __init__(self, result=None):
        self._result = result
        self.kinds = []
        self.shutdown_calls = 0
        self.recycle_calls = 0
        self.ping_calls = 0

    def transcribe(self, audio, kind, **kwargs):
        self.kinds.append(kind)
        if self._result is None:
            return None, 0.0
        return self._result, 12.0

    def maybe_recycle_worker(self):
        self.recycle_calls += 1

    def maybe_ping_worker(self):
        self.ping_calls += 1

    def shutdown_worker(self):
        self.shutdown_calls += 1


class FakeVAD:
    def __init__(self):
        self.chunks = []
        self.reset_calls = 0

    def is_speaking(self):
        return False

    def effective_silence_limit(self):
        return 25

    def process_chunk(self, chunk):
        self.chunks.append(chunk)
        return

    def current_confidence(self):
        return 0.0

    def buffered_samples(self):
        return 0

    def peek_buffer(self):
        return None

    def force_flush(self):
        return None

    def flush(self):
        return None

    def reset(self):
        self.reset_calls += 1

    def update_settings(self, settings):
        pass


class FakeAudio:
    """Backend-contract fake: the app layer owns start/stop; the pipeline
    only consumes chunks."""

    def read_chunk(self):
        return None  # eternal silence


class FakeOverlay:
    def __init__(self):
        self.messages = []
        self.translations = []
        self.streaming = []
        self.stats = []
        self.show_errors = []

    def add_message(self, *args):
        self.messages.append(args)

    def update_translation(self, *args):
        self.translations.append(args)

    def show_error(self, text):
        self.show_errors.append(text)

    def update_streaming(self, *args):
        self.streaming.append(args)

    def update_stats(self, *args):
        self.stats.append(args)

    def update_monitor(self, *args):
        pass

    def message_count(self):
        return len(self.messages)


class FakePanel:
    def __init__(self, asr_language="auto"):
        self._settings = {"asr_language": asr_language}

    def get_settings(self):
        return dict(self._settings)


class FakeTranscript:
    def __init__(self):
        self.originals = []
        self.translations = []
        self.finalized = []
        self.closed = False

    def write_original(self, *args):
        self.originals.append(args)

    def write_translation(self, *args):
        self.translations.append(args)

    def finalize_no_translation(self, *args):
        self.finalized.append(args)

    def close(self):
        self.closed = True

    def set_enabled(self, enabled):
        pass


class FakeSubwin:
    def __init__(self, visible=False, languages=("zh",)):
        self.visible = visible
        self.languages = set(languages)
        self.texts = []

    def isVisible(self):
        return self.visible

    def get_target_languages(self):
        return set(self.languages)

    def update_text(self, *args):
        self.texts.append(args)


class FakeTranslator:
    def __init__(self):
        self.last_usage = (10, 5)
        self.target_language = None

    def translate_iter(self, text, source_lang):
        yield "\u8bd1\u6587-partial"  # 译文-partial
        yield "\u8bd1\u6587-final"

    def translate(self, text, source_lang):
        return "\u8bd1\u6587"

    def with_target_language(self, lang):
        return self

    def set_target_language(self, lang):
        self.target_language = lang

    def set_timeout(self, timeout):
        pass


CONFIG = {
    "translation": {"target_language": "zh"},
    "audio": {"sample_rate": 16000, "chunk_duration": 0.032},
}


def _make(**overrides):
    defaults = {
        "vad": FakeVAD(),
        "audio": FakeAudio(),
        "asr_ctl": FakeCtl(),
        "translator": FakeTranslator(),
        "transcript": FakeTranscript(),
    }
    defaults.update(overrides)
    return Pipeline(CONFIG, **defaults)


class TestSegmentText:
    def test_same_language_skips_translation(self):
        overlay, transcript = FakeOverlay(), FakeTranscript()
        p = _make(transcript=transcript)
        p.set_overlay(overlay)
        p.set_panel(FakePanel())
        p._process_segment_text("hello there", "zh")
        assert len(overlay.messages) == 1
        assert len(transcript.finalized) == 1
        assert overlay.translations[0][1] == ""  # same-language marker
        assert p.asr_count == 1

    def test_different_language_submits_translation(self):
        overlay, transcript = FakeOverlay(), FakeTranscript()
        p = _make(transcript=transcript)
        p.set_overlay(overlay)
        p.set_panel(FakePanel())
        p._process_segment_text("hello world", "en")
        # Translation runs on the executor; wait for it to finish
        deadline = time.monotonic() + 5
        while not transcript.translations and time.monotonic() < deadline:
            time.sleep(0.01)
        assert transcript.translations, "translation was not submitted"
        assert p.translate_count == 1

    def test_language_filter_discards_mismatch(self):
        overlay, transcript = FakeOverlay(), FakeTranscript()
        p = _make(transcript=transcript)
        p.set_overlay(overlay)
        p.set_panel(FakePanel(asr_language="ja"))
        # CORE-11: the filter reads the pipeline snapshot (pushed by the app
        # layer), never the panel dict directly.
        p.set_asr_language("ja")
        p._process_segment_text("english text here", "en")
        assert overlay.messages == []
        assert p.asr_count == 0

    def test_language_filter_ignores_panel_until_snapshot_set(self):
        overlay = FakeOverlay()
        p = _make()
        p.set_overlay(overlay)
        p.set_panel(FakePanel(asr_language="ja"))
        # Snapshot defaults to "auto": the panel's dict must not influence
        # the ASR thread even though it was set.
        p._process_segment_text("english text here", "en")
        assert len(overlay.messages) == 1
        assert p.asr_count == 1
        p.set_asr_language("ja")
        p._process_segment_text("more english", "en")
        assert p.asr_count == 1  # discarded: count did not grow

    def test_empty_text_ignored(self):
        overlay = FakeOverlay()
        p = _make()
        p.set_overlay(overlay)
        p._process_segment_text("   ...  ", "en")
        assert overlay.messages == []


class TestProcessSegment:
    def test_transcribes_and_adds_message(self):
        ctl = FakeCtl(result=TranscriptionResult(text="Hello world", language="en"))
        overlay, transcript = FakeOverlay(), FakeTranscript()
        p = _make(asr_ctl=ctl, transcript=transcript)
        p.set_overlay(overlay)
        p.set_panel(FakePanel())
        audio = np.zeros(16000, dtype=np.float32)  # 1.0s
        p._process_segment(audio)
        assert ctl.kinds == ["segment"]
        assert len(overlay.messages) == 1
        assert overlay.messages[0][2] == "Hello world"

    def test_noise_filter_discards_short_text_from_long_segment(self):
        ctl = FakeCtl(result=TranscriptionResult(text="hi", language="en"))
        overlay = FakeOverlay()
        p = _make(asr_ctl=ctl)
        p.set_overlay(overlay)
        p.set_panel(FakePanel())
        p._process_segment(np.zeros(3 * 16000, dtype=np.float32))  # 3s segment
        assert overlay.messages == []

    def test_no_result_is_ignored(self):
        ctl = FakeCtl(result=None)
        overlay = FakeOverlay()
        p = _make(asr_ctl=ctl)
        p.set_overlay(overlay)
        p.set_panel(FakePanel())
        p._process_segment(np.zeros(16000, dtype=np.float32))
        assert overlay.messages == []


class TestQueueHelpers:
    def test_enqueue_drops_oldest_when_full(self):
        p = _make()
        for _i in range(16):
            p._enqueue_asr("interim", _i)
        p._enqueue_asr("vad_flush", "newest")
        items = []
        while not p._asr_queue.empty():
            items.append(p._asr_queue.get_nowait())
        assert items[0] == ("interim", 1)  # oldest interim dropped
        assert items[-1] == ("vad_flush", "newest")

    def test_drain_interim_duplicates_stops_at_non_interim(self):
        p = _make()
        for _ in range(3):
            p._asr_queue.put(("interim", None))
        p._asr_queue.put(("vad_flush", "seg"))
        p._drain_interim_duplicates()
        assert p._asr_queue.qsize() == 1
        assert p._asr_queue.get_nowait() == ("vad_flush", "seg")


class TestLifecycle:
    def test_start_stop_runs_threads_and_shuts_down(self):
        ctl = FakeCtl()
        audio, transcript = FakeAudio(), FakeTranscript()
        started, stopped = [], []
        p = _make(
            asr_ctl=ctl,
            audio=audio,
            transcript=transcript,
            start_hook=lambda: started.append(True),
            stop_hook=lambda: stopped.append(True),
        )
        p.start()
        assert p.running
        # The ASR loop idles and performs recycle/ping maintenance
        deadline = time.monotonic() + 3
        while ctl.recycle_calls == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ctl.recycle_calls > 0
        p.stop()
        assert not p.running
        assert transcript.closed
        assert ctl.shutdown_calls == 1
        assert started == [True] and stopped == [True]

    def test_pause_resume(self):
        p = _make()
        p.pause()
        assert p._paused
        p.resume()
        assert not p._paused

    def test_setters(self):
        p = _make()
        translator = FakeTranslator()
        p.set_translator(translator)
        assert p.translator is translator
        p.set_prices(1.0, 2.0)
        assert p._input_price == 1.0
        p.set_target_language("ja")
        assert p.target_language == "ja"
        assert translator.target_language == "ja"
        p.set_incremental(True)
        p.set_interim_interval(3.0)
        assert p._incremental_enabled is True
        assert p._interim_interval == 3.0


class _AuthError(Exception):
    status_code = 401


class BoomTranslator(FakeTranslator):
    def translate_iter(self, text, source_lang):
        raise _AuthError()
        yield  # pragma: no cover

    def translate(self, text, source_lang):
        raise _AuthError()


def test_translate_error_classified_reported_and_inline():
    """§3.6: a 401 becomes the canonical copy on all three surfaces."""
    from livetranslate.core.i18n import t

    overlay, transcript = FakeOverlay(), FakeTranscript()
    seen = []
    p = _make(translator=BoomTranslator(), transcript=transcript)
    p.set_overlay(overlay)
    p.set_error_reporter(seen.append)
    p.set_panel(FakePanel())
    p._process_segment_text("hello world", "en")
    deadline = time.monotonic() + 5
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen, "error was not reported"
    expected = t("err_401")
    assert seen[0] == expected
    assert overlay.show_errors == [expected]
    assert any(tr[1] == f"[{expected}]" for tr in overlay.translations)


class TestLogTranscriptGate:
    """SEC-1: speech/translation content must not reach the logs by default;
    the opt-in switch enables it explicitly. Gated at the call site because
    the file handler logs at DEBUG level."""

    def _records(self, caplog):
        return [
            r.getMessage()
            for r in caplog.records
            if r.name == "LiveTranslate.Pipeline" and r.levelno >= logging.INFO
        ]

    def test_info_logs_omit_content_by_default(self, caplog):
        p = _make()
        p.set_overlay(FakeOverlay())
        p.set_panel(FakePanel())
        with caplog.at_level(logging.INFO):
            p._process_segment_text("top secret speech", "en")
        for msg in self._records(caplog):
            assert "top secret speech" not in msg

    def test_translate_log_omits_content_by_default(self, caplog):
        p = _make()
        p.set_overlay(FakeOverlay())
        p.set_panel(FakePanel())
        with caplog.at_level(logging.INFO):
            p._process_segment_text("hello world", "en")
        for msg in self._records(caplog):
            assert "译文-final" not in msg

    def test_content_logged_when_opted_in(self, caplog):
        p = _make(log_transcript=True)
        p.set_overlay(FakeOverlay())
        p.set_panel(FakePanel())
        with caplog.at_level(logging.INFO):
            p._process_segment_text("top secret speech", "en")
        assert any("top secret speech" in m for m in self._records(caplog))

    def test_setter_toggles_gate(self, caplog):
        p = _make()
        p.set_overlay(FakeOverlay())
        p.set_panel(FakePanel())
        p.set_log_transcript(True)
        with caplog.at_level(logging.INFO):
            p._process_segment_text("top secret speech", "en")
        assert any("top secret speech" in m for m in self._records(caplog))
        caplog.clear()
        p.set_log_transcript(False)
        with caplog.at_level(logging.INFO):
            p._process_segment_text("top secret speech again", "en")
        for msg in self._records(caplog):
            assert "top secret speech again" not in msg


class TestQueueDiscipline:
    """CORE-2: the ASR queue must never block a producer — full means
    drop-oldest, and drain/stop replays respect the same discipline."""

    def test_enqueue_drops_oldest_when_full(self):
        p = _make()
        p._asr_queue = queue.Queue(maxsize=3)
        for _i in range(3):
            p._enqueue_asr("segment", np.zeros(16, dtype=np.float32))
        # Now full; the 4th entry must evict the oldest.
        p._enqueue_asr("vad_flush", np.zeros(16, dtype=np.float32))
        items = [p._asr_queue.get_nowait() for _ in range(3)]
        assert items[0][0] == "segment"  # first two segments survive...
        assert items[1][0] == "segment"
        assert items[2][0] == "vad_flush"  # ...oldest was dropped

    def test_drain_replays_non_interim_without_blocking(self):
        p = _make()
        p._asr_queue = queue.Queue(maxsize=4)
        p._enqueue_asr("interim", None)
        p._enqueue_asr("interim", None)
        p._enqueue_asr("vad_flush", np.zeros(16, dtype=np.float32))
        p._drain_interim_duplicates()
        remaining = [p._asr_queue.get_nowait() for _ in range(1)]
        assert remaining[0][0] == "vad_flush"

    def test_drain_empties_when_only_interim(self):
        p = _make()
        p._asr_queue = queue.Queue(maxsize=2)
        p._enqueue_asr("interim", None)
        p._enqueue_asr("interim", None)
        p._drain_interim_duplicates()
        assert p._asr_queue.empty()


class TestStopBoundedness:
    """CORE-2/CORE-10: stop() must return in bounded time even with a full
    queue, a dead ASR thread or a hung worker."""

    def test_stop_with_full_queue_returns_promptly(self):
        p = _make()
        p._asr_queue = queue.Queue(maxsize=16)
        for _i in range(16):
            p._asr_queue.put_nowait(("segment", np.zeros(16, dtype=np.float32)))
        start = time.monotonic()
        p.stop()
        assert time.monotonic() - start < 5

    def test_stop_with_dead_asr_thread_returns_promptly(self):
        p = _make()
        p._asr_queue = queue.Queue(maxsize=16)
        # ASR thread "died" without consuming: queue stays full.
        for _i in range(16):
            p._asr_queue.put_nowait(("segment", np.zeros(16, dtype=np.float32)))
        start = time.monotonic()
        p.stop()
        assert time.monotonic() - start < 5
        # The sentinel made it in (evicting one backlog slot); leftovers
        # stay queued but are harmless — the ASR loop exits on _running.
        items = []
        while not p._asr_queue.empty():
            items.append(p._asr_queue.get_nowait())
        assert items[-1] is None  # sentinel at the tail
        assert len(items) == 16  # 15 leftover segments + sentinel

    def test_stop_flush_tail_is_bounded(self):
        """A hung worker must not freeze stop(): the VAD tail flush runs on
        a bounded helper thread."""

        class SlowCtl(FakeCtl):
            ready = True

            def transcribe(self, audio, kind, **kwargs):
                time.sleep(30)  # simulate a hung worker
                return None, 0.0

        class FlushVAD(FakeVAD):
            def force_flush(self):
                return np.zeros(16000, dtype=np.float32)

        p = _make(asr_ctl=SlowCtl(), vad=FlushVAD())
        p._interim_active = True  # force the flush path
        start = time.monotonic()
        p.stop()
        assert time.monotonic() - start < 10


class TestResetInterim:
    def test_resets_interim_state_and_vad(self):
        """reset_interim() must clear the interim-ASR fields (the real state
        lives in the Pipeline, not the app) and reset the VAD buffer."""
        vad = FakeVAD()
        p = _make(vad=vad)
        p._interim_active = True
        p._interim_pending = "pending utterance"
        p._last_interim_samples = 42
        p._last_interim_check_time = 99.0
        p._interim_committed_tail = "committed tail"
        p.reset_interim()
        assert p._interim_active is False
        assert p._interim_pending == ""
        assert p._last_interim_samples == 0
        assert p._last_interim_check_time == 0.0
        assert p._interim_committed_tail == ""
        assert vad.reset_calls == 1

    def test_reset_interim_is_idempotent(self):
        p = _make()
        p.reset_interim()
        p.reset_interim()
        assert p._interim_active is False
        assert p._interim_pending == ""
