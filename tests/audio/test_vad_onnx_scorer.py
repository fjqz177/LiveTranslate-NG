"""Tests for the ONNX Silero scorer (real bundled model, CPU).

These run the actual 512-sample streaming path so state handling and
determinism are verified against the shipped model, not a mock.
"""

import numpy as np
import pytest

from livetranslate.audio.vad.scorer import SileroConfidenceScorer


@pytest.fixture(scope="module")
def scorer():
    return SileroConfidenceScorer()


def _silence() -> np.ndarray:
    return np.zeros(512, dtype=np.float32)


def _tone() -> np.ndarray:
    t = np.arange(512, dtype=np.float32) / 16000
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class TestOnnxScorer:
    def test_scores_are_probabilities(self, scorer):
        scorer.reset()
        for chunk in (_silence(), _tone(), _silence(), _tone()):
            score = scorer.score(chunk)
            assert 0.0 <= score <= 1.0

    def test_silence_scores_low(self, scorer):
        scorer.reset()
        scores = [scorer.score(_silence()) for _ in range(10)]
        assert max(scores) < 0.35

    def test_tone_scores_higher_than_silence(self, scorer):
        scorer.reset()
        silence_score = scorer.score(_silence())
        tone_score = scorer.score(_tone())
        assert tone_score > silence_score

    def test_reset_restores_determinism(self, scorer):
        scorer.reset()
        first = [scorer.score(_tone()) for _ in range(3)]
        scorer.reset()
        second = [scorer.score(_tone()) for _ in range(3)]
        assert first == pytest.approx(second, abs=1e-6)

    def test_state_advances_between_chunks(self, scorer):
        scorer.reset()
        a = scorer.score(_tone())
        b = scorer.score(_tone())
        assert a != b  # LSTM state + context roll forward

    def test_short_chunk_is_padded(self, scorer):
        scorer.reset()
        score = scorer.score(np.zeros(100, dtype=np.float32))
        assert 0.0 <= score <= 1.0
