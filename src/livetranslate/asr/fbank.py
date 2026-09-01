"""Kaldi-compliant fbank in pure numpy (SelfServe P0-A3).

Line-by-line port of ``torchaudio.compliance.kaldi.fbank`` (BSD-3-Clause,
(c) 2017 Facebook Inc.), scoped to the parameter set the SenseVoice ONNX
frontend needs. Keeping the math identical is load-bearing: the ONNX export
was trained/calibrated against torchaudio features, and the parity test
(tests/audio/test_fbank_parity.py) pins 1e-4 agreement.

This module is the reason the base install needs no torch/torchaudio: VAD
and SenseVoice inference run on onnxruntime, and the acoustic frontend now
runs here.
"""

from __future__ import annotations

import math

import numpy as np

EPSILON = float(np.finfo(np.float32).eps)
MILLISECONDS_TO_SECONDS = 0.001
HANNING = "hanning"
HAMMING = "hamming"
POVEY = "povey"
RECTANGULAR = "rectangular"
BLACKMAN = "blackman"


def _next_power_of_2(x: int) -> int:
    """Smallest power of 2 that is greater than x."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _get_strided(
    waveform: np.ndarray, window_size: int, window_shift: int, snip_edges: bool
) -> np.ndarray:
    """(num_samples,) -> (m, window_size) frames, mirroring torchaudio's as_strided."""
    num_samples = waveform.shape[0]
    if snip_edges:
        if num_samples < window_size:
            return np.empty((0, 0), dtype=waveform.dtype)
        m = 1 + (num_samples - window_size) // window_shift
        if m <= 0:
            return np.empty((0, 0), dtype=waveform.dtype)
        return np.lib.stride_tricks.as_strided(
            waveform,
            shape=(m, window_size),
            strides=(window_shift * waveform.strides[0], waveform.strides[0]),
        )
    # snip_edges=False: reflect-pad at the ends (torchaudio's manual pad).
    reversed_waveform = waveform[::-1]
    m = (num_samples + (window_shift // 2)) // window_shift
    pad = window_size // 2 - window_shift // 2
    pad_right = reversed_waveform
    if pad > 0:
        pad_left = (
            reversed_waveform[-pad:]
            if pad <= len(reversed_waveform)
            else np.pad(reversed_waveform, (0, pad - len(reversed_waveform)))
        )
        waveform = np.concatenate((pad_left, waveform, pad_right))
    else:
        start = -pad
        waveform = np.concatenate((waveform[start:] if start > 0 else waveform, pad_right))
    return np.lib.stride_tricks.as_strided(
        waveform,
        shape=(m, window_size),
        strides=(window_shift * waveform.strides[0], waveform.strides[0]),
    )


def _feature_window_function(
    window_type: str, window_size: int, blackman_coeff: float
) -> np.ndarray:
    if window_type == HANNING:
        n = np.arange(window_size, dtype=np.float64)
        return (0.5 - 0.5 * np.cos(2 * math.pi * n / (window_size - 1))).astype(np.float32)
    if window_type == HAMMING:
        n = np.arange(window_size, dtype=np.float64)
        return (0.54 - 0.46 * np.cos(2 * math.pi * n / (window_size - 1))).astype(np.float32)
    if window_type == POVEY:
        n = np.arange(window_size, dtype=np.float64)
        return (0.5 - 0.5 * np.cos(2 * math.pi * n / (window_size - 1))).astype(np.float32) ** 0.85
    if window_type == RECTANGULAR:
        return np.ones(window_size, dtype=np.float32)
    if window_type == BLACKMAN:
        a = 2 * math.pi / (window_size - 1)
        n = np.arange(window_size, dtype=np.float32)
        return (
            blackman_coeff - 0.5 * np.cos(a * n) + (0.5 - blackman_coeff) * np.cos(2 * a * n)
        ).astype(np.float32)
    raise ValueError(f"Invalid window type {window_type}")


def _get_log_energy(strided_input: np.ndarray, energy_floor: float) -> np.ndarray:
    log_energy = np.log(np.maximum(np.sum(strided_input**2, axis=1), EPSILON))
    if energy_floor == 0.0:
        return log_energy
    return np.maximum(log_energy, math.log(energy_floor))


def _get_window(
    waveform: np.ndarray,
    padded_window_size: int,
    window_size: int,
    window_shift: int,
    window_type: str,
    blackman_coeff: float,
    snip_edges: bool,
    raw_energy: bool,
    energy_floor: float,
    dither: float,
    remove_dc_offset: bool,
    preemphasis_coefficient: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(strided_input, signal_log_energy) — exact port of torchaudio's _get_window."""
    strided_input = _get_strided(waveform, window_size, window_shift, snip_edges)

    if dither != 0.0:
        rng = np.random.default_rng()
        strided_input = (
            strided_input + rng.standard_normal(strided_input.shape).astype(np.float32) * dither
        )

    if remove_dc_offset:
        strided_input = strided_input - np.mean(strided_input, axis=1, keepdims=True)

    signal_log_energy = np.empty(0, dtype=np.float32)
    if raw_energy:
        signal_log_energy = _get_log_energy(strided_input, energy_floor)

    if preemphasis_coefficient != 0.0:
        # strided_input[i, j] -= coeff * strided_input[i, max(0, j-1)]
        offset = np.concatenate(
            (strided_input[:, :1], strided_input[:, :-1]), axis=1
        )  # replicate pad (1, 0)
        strided_input = strided_input - preemphasis_coefficient * offset

    window_function = _feature_window_function(window_type, window_size, blackman_coeff)
    strided_input = strided_input * window_function[np.newaxis, :]

    if padded_window_size != window_size:
        padding_right = padded_window_size - window_size
        strided_input = np.pad(strided_input, ((0, 0), (0, padding_right)), mode="constant")

    if not raw_energy:
        signal_log_energy = _get_log_energy(strided_input, energy_floor)

    return strided_input.astype(np.float32, copy=False), signal_log_energy.astype(
        np.float32, copy=False
    )


def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def inverse_mel_scale_scalar(mel_freq: float) -> float:
    return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)


def mel_scale(freq: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + freq / 700.0)


def inverse_mel_scale(mel_freq: np.ndarray) -> np.ndarray:
    return 700.0 * (np.exp(mel_freq / 1127.0) - 1.0)


def get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
    vtln_warp_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(bins (num_bins, num_fft_bins), center_freqs) — port of torchaudio's get_mel_banks.

    VTLN warping is not implemented (warp factor must be 1.0): the SenseVoice
    frontend never uses it.
    """
    assert num_bins > 3, "Must have at least 3 mel bins"
    assert window_length_padded % 2 == 0
    assert vtln_warp_factor == 1.0, "VTLN warping is not implemented"
    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq

    if high_freq <= 0.0:
        high_freq += nyquist

    assert 0.0 <= low_freq < nyquist and 0.0 < high_freq <= nyquist and low_freq < high_freq, (
        f"Bad values in options: low-freq {low_freq} and high-freq {high_freq} vs. nyquist {nyquist}"
    )

    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    bins_idx = np.arange(num_bins)[:, np.newaxis].astype(np.float64)
    left_mel = mel_low_freq + bins_idx * mel_freq_delta
    center_mel = mel_low_freq + (bins_idx + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bins_idx + 2.0) * mel_freq_delta

    center_freqs = inverse_mel_scale(center_mel)
    mel = mel_scale(fft_bin_width * np.arange(num_fft_bins, dtype=np.float64))[np.newaxis, :]

    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    bins = np.maximum(np.zeros(1, dtype=np.float64), np.minimum(up_slope, down_slope))
    return bins.astype(np.float32), center_freqs.astype(np.float32)


def fbank(
    waveform: np.ndarray,
    num_mel_bins: int = 23,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    window_type: str = POVEY,
    sample_frequency: float = 16000.0,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    preemphasis_coefficient: float = 0.97,
    remove_dc_offset: bool = True,
    raw_energy: bool = True,
    round_to_power_of_two: bool = True,
    snip_edges: bool = True,
    subtract_mean: bool = False,
    use_energy: bool = False,
    use_log_fbank: bool = True,
    use_power: bool = True,
    htk_compat: bool = False,
    min_duration: float = 0.0,
    blackman_coeff: float = 0.42,
) -> np.ndarray:
    """Kaldi-compliant fbank, identical math to torchaudio.compliance.kaldi.fbank.

    ``waveform``: 1-D float32 array at ``sample_frequency``. Returns (m, num_mel_bins + use_energy).
    """
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
    window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
    padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size

    assert 2 <= window_size <= len(waveform), (
        f"choose a window size {window_size} that is [2, {len(waveform)}]"
    )
    assert window_shift > 0, "`window_shift` must be greater than 0"
    assert padded_window_size % 2 == 0, (
        "the padded `window_size` must be divisible by two; use `round_to_power_of_two` "
        "or change `frame_length`"
    )
    assert 0.0 <= preemphasis_coefficient <= 1.0, "`preemphasis_coefficient` must be between [0,1]"
    assert sample_frequency > 0, "`sample_frequency` must be greater than zero"

    if len(waveform) < min_duration * sample_frequency:
        return np.empty(0, dtype=np.float32)

    strided_input, signal_log_energy = _get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        window_type,
        blackman_coeff,
        snip_edges,
        raw_energy,
        energy_floor,
        dither,
        remove_dc_offset,
        preemphasis_coefficient,
    )

    spectrum = np.abs(np.fft.rfft(strided_input, n=padded_window_size, axis=1))
    if use_power:
        spectrum = spectrum**2

    mel_energies, _ = get_mel_banks(
        num_mel_bins, padded_window_size, sample_frequency, low_freq, high_freq
    )
    # pad right column with zeros: (num_mel_bins, padded_window_size // 2 + 1)
    mel_energies = np.pad(mel_energies, ((0, 0), (0, 1)), mode="constant")

    mel_energies_out = spectrum @ mel_energies.T
    if use_log_fbank:
        mel_energies_out = np.log(np.maximum(mel_energies_out, EPSILON))

    if use_energy:
        signal_log_energy = signal_log_energy[:, np.newaxis]
        if htk_compat:
            mel_energies_out = np.concatenate((mel_energies_out, signal_log_energy), axis=1)
        else:
            mel_energies_out = np.concatenate((signal_log_energy, mel_energies_out), axis=1)

    if subtract_mean:
        mel_energies_out = mel_energies_out - np.mean(mel_energies_out, axis=0, keepdims=True)

    return mel_energies_out.astype(np.float32)
