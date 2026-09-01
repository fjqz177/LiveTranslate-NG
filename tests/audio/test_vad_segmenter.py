"""Unit tests for the pure VAD state machine (no torch, scripted confidences).

Exercises the segmenter branches that historically had zero coverage:
onset/pre-speech, silence flush, short-segment merge, trimmed remainder,
backtrack split, density filter, adaptive and progressive silence.

Semantics reminder (mirrors the implementation):
- silence chunks while speaking ARE appended to the buffer and history;
- the short-segment soft reset only triggers when total buffered samples
  (speech + trailing silence) stay below min_speech_samples;
- the density filter runs inside _flush_segment, i.e. only for segments
  that already passed the min-duration check.
"""

import numpy as np
import pytest

from livetranslate.audio.vad.segmenter import SpeechSegmenter

CHUNK = 512  # samples per 32ms chunk at 16kHz


class ScriptedScorer:
    """Yields a fixed confidence sequence, then a default value."""

    def __init__(self, confidences, default=0.0, threshold=0.5):
        self._seq = list(confidences)
        self._i = 0
        self._default = default
        self._threshold = threshold

    @property
    def onset_threshold(self) -> float:
        return self._threshold

    def score(self, chunk):
        v = self._seq[self._i] if self._i < len(self._seq) else self._default
        self._i += 1
        return v


def _chunk() -> np.ndarray:
    return np.zeros(CHUNK, dtype=np.float32)


def _make(confidences=(), default=0.0, threshold=0.5, **kw):
    scorer = ScriptedScorer(confidences, default=default, threshold=threshold)
    defaults = {
        "min_speech_duration": 0.32,  # 10 chunks = 5120 samples
        "max_speech_duration": 4.0,
        "chunk_duration": 0.032,
    }
    defaults.update(kw)
    return SpeechSegmenter(scorer=scorer, **defaults)


def _feed_many(seg, confidences):
    """Feed chunks scored by the given sequence; afterwards score = 0.0."""
    seg.set_scorer(ScriptedScorer(confidences))
    for _ in confidences:
        seg.process_chunk(_chunk())


def _silence_out(seg, n=5):
    """Feed n silence chunks; return the first emitted segment (if any)."""
    out = None
    for _ in range(n):
        r = seg.process_chunk(_chunk())
        if r is not None:
            out = r
    return out


class TestOnsetAndFlush:
    def test_speech_then_silence_emits_segment(self):
        seg = _make(default=0.0)
        seg.update_settings({"silence_mode": "fixed", "silence_duration": 0.16})
        _feed_many(seg, [1.0] * 10)  # 320ms speech >= min
        assert seg.is_speaking()
        out = _silence_out(seg)  # 5 silence chunks = 0.16s limit
        assert out is not None
        assert len(out) == (10 + 5) * CHUNK  # trailing silence is included
        assert not seg.is_speaking()
        assert seg.buffered_samples() == 0

    def test_pre_speech_ring_buffer_prepended(self):
        seg = _make(default=0.0)
        # Not speaking: chunks go to the pre-speech ring buffer
        for _ in range(3):
            seg.process_chunk(_chunk())
        assert seg.buffered_samples() == 0
        # Onset: pre-chunks prepended with synthetic threshold confidence
        _feed_many(seg, [1.0, 1.0])
        assert seg.buffered_samples() == 5 * CHUNK

    def test_short_segment_soft_reset_merges_with_next_onset(self):
        seg = _make(default=0.0)
        seg.update_settings({"silence_mode": "fixed", "silence_duration": 0.16})
        _feed_many(seg, [1.0] * 2)  # 64ms speech
        out = _silence_out(seg)  # total (2+5)*512=3584 < min 5120
        assert out is None
        assert not seg.is_speaking()
        assert seg.buffered_samples() == 7 * CHUNK  # kept for merge
        # Next onset merges naturally
        _feed_many(seg, [1.0] * 3)
        assert seg.buffered_samples() == 10 * CHUNK


class TestTrimmedRemainder:
    def test_trim_then_short_silence_force_flushes(self):
        seg = _make(default=0.0)
        seg.update_settings({"silence_mode": "fixed", "silence_duration": 0.16})
        _feed_many(seg, [1.0] * 10)
        seg.trim_front(9 * CHUNK)  # interim ASR consumed 9 chunks
        assert seg.buffered_samples() == CHUNK
        out = _silence_out(seg)
        # Remainder is below min_speech but _was_trimmed forces the flush
        assert out is not None
        assert len(out) == (1 + 5) * CHUNK

    def test_partial_chunk_trim(self):
        seg = _make(default=0.0)
        _feed_many(seg, [1.0] * 4)
        seg.trim_front(CHUNK + 100)  # remove one full chunk + 100 samples
        assert seg.buffered_samples() == 3 * CHUNK - 100
        audio, _duration = seg.peek_buffer()
        assert len(audio) == 3 * CHUNK - 100


class TestBacktrackSplit:
    def test_max_duration_splits_at_valley(self):
        # 4 chunks hit the max; the dip at index 1 gives a split point
        seg = _make(
            confidences=[1.0, 0.1, 1.0, 1.0],
            max_speech_duration=4 * CHUNK / 16000,
        )
        out = None
        for _ in range(4):
            r = seg.process_chunk(_chunk())
            if r is not None:
                out = r
        # Ties in the smoothed minimum resolve to the LATER index (<=),
        # so the split lands at index 2 here: two chunks emitted.
        assert out is not None
        assert len(out) == 2 * CHUNK
        assert seg.buffered_samples() == 2 * CHUNK  # remainder kept
        assert seg.is_speaking()

    def test_max_duration_hard_flush_when_no_valley(self):
        seg = _make(
            confidences=[1.0, 1.0, 1.0, 1.0],
            max_speech_duration=3 * CHUNK / 16000,
        )
        out = None
        for _ in range(4):
            r = seg.process_chunk(_chunk())
            if r is not None:
                out = r
        # The 3rd chunk already hits max; n<4 means no split point -> hard flush
        assert out is not None
        assert len(out) == 3 * CHUNK
        assert seg.buffered_samples() == CHUNK  # 4th chunk restarts a segment


class TestDensityFilter:
    def test_low_density_segment_discarded(self):
        # Auto mode keeps the 25-chunk silence limit: the low-confidence
        # chunks count as silence, so the flush decision only fires at the
        # 25th one — by then 6/31 chunks are voiced (< 0.25 density).
        seg = _make(default=0.0)
        _feed_many(seg, [1.0] * 6 + [0.1] * 25)
        out = _silence_out(seg)
        assert out is None  # discarded inside the final feed/flush
        assert seg.buffered_samples() == 0
        assert not seg.is_speaking()

    def test_high_density_segment_kept(self):
        seg = _make(default=0.0)
        seg.update_settings({"silence_mode": "fixed", "silence_duration": 0.16})
        _feed_many(seg, [1.0] * 10 + [0.1])
        # The trailing 0.1 chunk already counts one silence; flush comes
        # after 4 more silence chunks.
        out = _silence_out(seg)
        assert out is not None
        assert len(out) == (11 + 4) * CHUNK


class TestAdaptiveSilence:
    def test_pause_history_shrinks_limit(self):
        seg = _make(default=0.0)
        seg.update_settings({"silence_mode": "auto"})
        initial = seg.effective_silence_limit()
        # 4 cycles of speech + short pause (>=0.1s = 4+ silence chunks);
        # pauses are recorded at cycles 2..4 -> 3 entries -> adaptive update
        for _ in range(4):
            _feed_many(seg, [1.0] * 5)
            for _ in range(4):
                seg.process_chunk(_chunk())
        # P75 of ~0.13s pauses * 1.2 clamps to the 0.3s minimum
        assert seg.effective_silence_limit() < initial


class TestProgressiveSilence:
    def test_long_buffer_accepts_shorter_pauses(self):
        seg = _make(default=1.0, max_speech_duration=30.0)
        seg.update_settings({"silence_mode": "fixed", "silence_duration": 0.8})
        full = seg.effective_silence_limit()
        for _ in range(320):  # 10.24s of speech -> quarter multiplier
            seg.process_chunk(_chunk())
        quarter = max(1, round(full * 0.25))
        assert seg.effective_silence_limit() == quarter
        assert quarter < full


class TestPeekAndReset:
    def test_peek_none_when_idle(self):
        seg = _make()
        assert seg.peek_buffer() is None

    def test_peek_returns_audio_without_flushing(self):
        seg = _make(default=1.0)
        _feed_many(seg, [1.0] * 3)
        audio, duration = seg.peek_buffer()
        assert len(audio) == 3 * CHUNK
        assert duration == pytest.approx(3 * CHUNK / 16000)
        assert seg.buffered_samples() == 3 * CHUNK  # untouched

    def test_reset_discards(self):
        seg = _make(default=1.0)
        _feed_many(seg, [1.0] * 3)
        seg.reset()
        assert seg.buffered_samples() == 0
        assert not seg.is_speaking()


class TestViews:
    def test_buffered_duration_and_confidence(self):
        seg = _make(default=0.7)
        _feed_many(seg, [0.9, 0.8])
        assert seg.buffered_duration() == pytest.approx(2 * CHUNK / 16000)
        assert seg.current_confidence() == 0.8

    def test_settings_update_min_max(self):
        seg = _make()
        seg.update_settings({"min_speech_duration": 1.0, "max_speech_duration": 8.0})
        assert seg.min_speech_samples == 16000
        assert seg.max_speech_samples == 128000
