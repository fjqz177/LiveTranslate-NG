"""Pure audio transforms shared by every capture backend.

Backends deliver raw float32 frames at the device's native rate/channels;
these functions convert them to the pipeline contract: float32, mono,
resampled to the target rate. Extracted from the WASAPI capture loop so
macOS/Linux backends reuse the exact same math (and its tests).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Float32Array = npt.NDArray[np.float32]


def to_mono(audio: Float32Array, channels: int) -> Float32Array:
    """Average interleaved channel data down to mono."""
    if channels <= 1:
        return audio
    return audio.reshape(-1, channels).mean(axis=1).astype(np.float32)


def resample_linear(audio: Float32Array, native_rate: int, target_rate: int) -> Float32Array:
    """Linear-interpolation resample float32 mono audio."""
    if native_rate == target_rate or len(audio) == 0:
        return audio
    ratio = target_rate / native_rate
    n_out = int(len(audio) * ratio)
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices, 0, len(audio) - 1)
    idx_floor = indices.astype(np.int64)
    idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
    frac = (indices - idx_floor).astype(np.float32)
    return audio[idx_floor] * (1 - frac) + audio[idx_ceil] * frac


def resample_to_mono(
    data: bytes, native_channels: int, native_rate: int, target_rate: int
) -> Float32Array:
    """Convert raw float32 interleaved bytes to mono at the target rate."""
    audio = np.frombuffer(data, dtype=np.float32)
    audio = to_mono(audio, native_channels)
    return resample_linear(audio, native_rate, target_rate)


def mic_rms(chunk: Float32Array) -> float:
    """RMS of a mono chunk (0.0 for silence/empty)."""
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
