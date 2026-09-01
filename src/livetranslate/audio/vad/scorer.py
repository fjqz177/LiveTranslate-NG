"""Confidence scorers used by SpeechSegmenter (see vad_segmenter.py).

Scorer protocol:
    score(chunk) -> float in [0, 1]: how "speech-like" the chunk is
    onset_threshold -> float: boundary used by the segmenter for onset,
    valley and density decisions
    reset() -> None (optional): scorers with internal state (the ONNX
    Silero model carries LSTM h/c and a context window) reset it here;
    the segmenter calls it on VAD reset and scorer switches.

Torch-free by design: the Silero scorer runs the model through
onnxruntime, so the base install needs no GPU framework.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from livetranslate.core.paths import models_dir

log = logging.getLogger("LiveTranslate.VAD")


def default_onnx_path() -> Path:
    """Locate the Silero VAD ONNX model.

    Preference order: the silero-vad pip package's bundled export (dev),
    the frozen bundle's copy (spec collects it into models/vad), then the
    per-user model cache.
    """
    try:
        import silero_vad  # heavy import only when present

        bundled = Path(silero_vad.__file__).parent / "data" / "silero_vad.onnx"
        if bundled.exists():
            return bundled
    except ImportError:
        pass
    if getattr(sys, "frozen", False):
        bundled = Path(getattr(sys, "_MEIPASS", "")) / "models" / "vad" / "silero_vad.onnx"
        if bundled.exists():
            return bundled
    return models_dir() / "vad" / "silero_vad.onnx"


class AlwaysSpeechScorer:
    """'disabled' VAD mode: everything counts as speech."""

    onset_threshold = 0.5

    def score(self, audio_chunk: np.ndarray) -> float:
        return 1.0


class EnergyConfidenceScorer:
    """RMS-energy scorer with saturation mapping ('energy' VAD mode)."""

    onset_threshold = 0.5

    def __init__(self, energy_threshold: float = 0.02):
        self.energy_threshold = energy_threshold

    def score(self, audio_chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(audio_chunk**2)))
        return min(1.0, rms / (self.energy_threshold * 2))


class SileroConfidenceScorer:
    """Silero VAD via onnxruntime ('silero' mode, the default).

    Mirrors the official OnnxWrapper streaming semantics: 512-sample
    windows (16 kHz) with a rolling 64-sample context and the LSTM state
    tensor carried across calls. Deterministic once reset().
    """

    def __init__(
        self,
        threshold: float = 0.50,
        sample_rate: int = 16000,
        model_path: Path | str | None = None,
    ):
        import onnxruntime as ort

        self.threshold = threshold
        self.sample_rate = sample_rate
        path = str(model_path) if model_path else str(default_onnx_path())
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {path}. Install the "
                "silero-vad package or download the ONNX model into the "
                "model cache."
            )
        opts = ort.SessionOptions()
        # The segmenter runs inside the capture thread of the GUI process;
        # keep inference single-threaded like the old torch path.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.reset()
        log.info(f"Silero VAD (onnx) loaded: {path}")

    @property
    def onset_threshold(self) -> float:
        return self.threshold

    @property
    def _window_size(self) -> int:
        return 512 if self.sample_rate == 16000 else 256

    @property
    def _context_size(self) -> int:
        return 64 if self.sample_rate == 16000 else 32

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def score(self, audio_chunk: np.ndarray) -> float:
        window = self._window_size
        chunk = audio_chunk[:window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        if self._context.shape[1] != self._context_size:
            self.reset()

        x = np.concatenate([self._context, chunk.reshape(1, -1)], axis=1)
        out, state = self._session.run(
            None,
            {
                "input": x.astype(np.float32),
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            },
        )
        self._state = state
        self._context = x[:, -self._context_size :]
        return float(out[0, 0])
