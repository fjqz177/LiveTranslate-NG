"""Pure-numpy fbank tests — no torch, run in every CI environment.

asr/fbank.py is the hand-written numpy port of torchaudio.compliance.kaldi.fbank
(SelfServe P0-A3). The torchaudio parity file (test_fbank_parity.py) is gated on
``importorskip("torch")`` and silently skipped whenever torch is absent, which is
exactly the CI/dev test environment — so the *numerical* parity checks never run.
This file splits out the pure-numpy behaviours (shape, window validation, short
waveform) that CAN and SHOULD run unconditionally, so a regression in the port
doesn't pass CI just because torch wasn't installed.

The 1e-4 numerical equivalence vs torchaudio remains in test_fbank_parity.py
(opt-in on a torch-enabled machine); here we lock the torch-free surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from livetranslate.asr.fbank import fbank

SENSEVOICE_KW = {
    "num_mel_bins": 80,
    "frame_length": 25,
    "frame_shift": 10,
    "dither": 0.0,
    "energy_floor": 0.0,
    "window_type": "hamming",
    "sample_frequency": 16000.0,
}


def test_fbank_rejects_bad_window_type():
    """The port must validate the window type itself (the ValueErro is raised
    by _feature_window_function), independent of torch."""
    with pytest.raises(ValueError, match="Invalid window type"):
        fbank(np.zeros(16000, dtype=np.float32), window_type="gaussian")


def test_fbank_returns_expected_frames():
    """2.0s @16kHz, 25ms frame / 10ms shift -> 198 frames of 80 bins."""
    wave = np.zeros(16000 * 2, dtype=np.float32)
    out = fbank(wave, **SENSEVOICE_KW)
    assert out.ndim == 2
    assert out.shape[0] == 198  # floor((32000 - 400) / 160) + 1
    assert out.shape[1] == SENSEVOICE_KW["num_mel_bins"]
    assert np.isfinite(out).all(), "silence must produce finite (zero) fbank"


def test_fbank_short_waveform_yields_empty():
    """Shorter than min_duration returns an empty (0,) tensor, not a crash."""
    wave = np.zeros(8000, dtype=np.float32)
    out = fbank(wave, **{**SENSEVOICE_KW, "min_duration": 1.0})
    assert out.size == 0


def test_fbank_use_energy_appends_column():
    """use_energy=True appends one log-energy column (bins+1)."""
    wave = np.zeros(16000, dtype=np.float32)
    kw = {**SENSEVOICE_KW, "use_energy": True}
    out = fbank(wave, **kw)
    assert out.shape[1] == kw["num_mel_bins"] + 1
