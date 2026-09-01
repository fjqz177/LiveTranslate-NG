"""Parity tests: numpy fbank vs torchaudio.compliance.kaldi (SelfServe P0-A3).

Marked `engine`: requires torchaudio (engine-sensevoice-onnx extra). The
SenseVoice ONNX frontend runs against the numpy port in production, so this
file is the gate that proves the port is numerically identical (1e-4).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.engine

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")
from torchaudio.compliance import kaldi  # noqa: E402

from livetranslate.asr.fbank import fbank as np_fbank  # noqa: E402

SENSEVOICE_KW = {
    "num_mel_bins": 80,
    "frame_length": 25,
    "frame_shift": 10,
    "dither": 0.0,
    "energy_floor": 0.0,
    "window_type": "hamming",
    "sample_frequency": 16000.0,
}


def _ref(wave: np.ndarray, **kw) -> np.ndarray:
    tensor = torch.from_numpy(wave.reshape(1, -1))
    return kaldi.fbank(tensor, **kw).numpy()


@pytest.mark.parametrize("seed,n", [(0, 16000), (1, 48000), (2, 12345)])
def test_parity_sensevoice_parameters(seed: int, n: int):
    rng = np.random.default_rng(seed)
    wave = (rng.standard_normal(n) * 0.1).astype(np.float32)
    ref = _ref(wave, **SENSEVOICE_KW)
    got = np_fbank(wave, **SENSEVOICE_KW)
    assert ref.shape == got.shape
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "overrides",
    [
        {"window_type": "povey"},
        {"use_power": False},
        {"use_energy": True, "htk_compat": False},
        {"subtract_mean": True},
        {"energy_floor": 1.0},
        {"snip_edges": False},
        {"round_to_power_of_two": False},
        {"preemphasis_coefficient": 0.0},
    ],
)
def test_parity_parameter_matrix(overrides: dict):
    rng = np.random.default_rng(7)
    wave = (rng.standard_normal(24000) * 0.1).astype(np.float32)
    kw = {**SENSEVOICE_KW, **overrides}
    ref = _ref(wave, **kw)
    got = np_fbank(wave, **kw)
    assert ref.shape == got.shape
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_parity_short_waveform_yields_empty():
    # Shorter than min_duration: both implementations return an empty (0,) tensor.
    wave = np.zeros(8000, dtype=np.float32)
    kw = {**SENSEVOICE_KW, "min_duration": 1.0}
    assert np_fbank(wave, **kw).shape == (0,)
    assert _ref(wave, **kw).shape == (0,)


def test_fbank_rejects_bad_window_type():
    with pytest.raises(ValueError, match="Invalid window type"):
        np_fbank(np.zeros(16000, dtype=np.float32), window_type="gaussian")
