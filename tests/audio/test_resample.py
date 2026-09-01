"""Tests for the pure audio transforms (behavior locked to the
original WASAPI capture implementation)."""

import numpy as np
import pytest

from livetranslate.audio.resample import (
    mic_rms,
    resample_linear,
    resample_to_mono,
    to_mono,
)


class TestToMono:
    def test_mono_passthrough(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = to_mono(a, 1)
        assert np.array_equal(out, a)

    def test_stereo_average(self):
        a = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)  # 2 frames stereo
        out = to_mono(a, 2)
        assert np.array_equal(out, np.array([2.0, 6.0], dtype=np.float32))


class TestResampleLinear:
    def test_same_rate_is_identity(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.array_equal(resample_linear(a, 16000, 16000), a)

    def test_downsample_length(self):
        rng = np.random.default_rng(42)
        a = rng.random(48000).astype(np.float32)
        out = resample_linear(a, 48000, 16000)
        assert out.shape == (16000,)
        assert out.dtype == np.float32

    def test_upsample_length(self):
        rng = np.random.default_rng(7)
        a = rng.random(8000).astype(np.float32)
        out = resample_linear(a, 8000, 16000)
        assert out.shape == (16000,)

    def test_empty_input(self):
        out = resample_linear(np.array([], dtype=np.float32), 48000, 16000)
        assert out.size == 0


class TestResampleToMono:
    def test_bytes_stereo_downsample_shape(self):
        rng = np.random.default_rng(1)
        stereo = rng.random(96000).astype(np.float32)  # 48000 frames stereo
        out = resample_to_mono(stereo.tobytes(), 2, 48000, 16000)
        assert out.shape == (16000,)

    def test_matches_reference_implementation(self):
        # Compare against the straightforward numpy pipeline (the original
        # AudioCapture math, expressed independently).
        rng = np.random.default_rng(3)
        stereo = rng.random(48000 * 2).astype(np.float32)
        out = resample_to_mono(stereo.tobytes(), 2, 48000, 16000)
        mono = stereo.reshape(-1, 2).mean(axis=1)
        ratio = 16000 / 48000
        n_out = int(len(mono) * ratio)
        indices = np.clip(np.arange(n_out) / ratio, 0, len(mono) - 1)
        idx_floor = indices.astype(np.int64)
        idx_ceil = np.minimum(idx_floor + 1, len(mono) - 1)
        frac = (indices - idx_floor).astype(np.float32)
        expected = mono[idx_floor] * (1 - frac) + mono[idx_ceil] * frac
        assert np.allclose(out, expected, atol=1e-6)


class TestMicRms:
    def test_silence(self):
        assert mic_rms(np.zeros(512, dtype=np.float32)) == 0.0

    def test_known_value(self):
        chunk = np.array([1.0, -1.0], dtype=np.float32)
        assert mic_rms(chunk) == pytest.approx(1.0)

    def test_empty(self):
        assert mic_rms(np.array([], dtype=np.float32)) == 0.0
