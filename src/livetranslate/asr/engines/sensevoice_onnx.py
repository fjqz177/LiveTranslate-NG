"""SenseVoice via onnxruntime (torch-free inference, Phase 5).

Frontend (fbank + LFR stacking) runs on the pure-numpy port in
livetranslate.asr.fbank (SelfServe P0-A3, parity-tested against
torchaudio at 1e-4) — the same math funasr uses — while the neural part
runs in onnxruntime, so the engine needs no funasr/modelscope/torch. The
ONNX model is produced by scripts/export_sensevoice_onnx.py and cached at
models_dir()/sensevoice/sensevoice-small.onnx.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from livetranslate.asr.fbank import fbank
from livetranslate.asr.protocol import (
    ASREngineBase,
    EngineCapabilities,
    TranscriptionResult,
)

log = logging.getLogger("LiveTranslate.SenseVoiceOnnx")

SAMPLE_RATE = 16000
DEFAULT_PAD_SECONDS = 0.5
PAD_SECONDS_ENV = "LIVETRANS_SENSEVOICE_PAD_SECONDS"
LFR_M = 7
LFR_N = 6
FEAT_DIM = 80

LANG_MAP = {
    "<|zh|>": "zh",
    "<|en|>": "en",
    "<|ja|>": "ja",
    "<|ko|>": "ko",
    "<|yue|>": "yue",
}


def _fbank(waveform: np.ndarray) -> np.ndarray:
    """Kaldi-compliant fbank, 80-dim (25ms/10ms hamming, 16kHz) — numpy port."""
    return fbank(
        waveform,
        num_mel_bins=FEAT_DIM,
        frame_length=25,
        frame_shift=10,
        dither=0.0,  # deterministic features (legacy default 1.0 adds RNG noise)
        energy_floor=0.0,
        window_type="hamming",
        sample_frequency=SAMPLE_RATE,
    )


def _apply_lfr(feats: np.ndarray) -> np.ndarray:
    """LFR stacking (m=7, n=6): numpy port of funasr apply_lfr."""
    t, dim = feats.shape
    t_lfr = int(np.ceil(t / LFR_N))
    left_padding = np.repeat(feats[0:1], (LFR_M - 1) // 2, axis=0)
    padded = np.vstack([left_padding, feats])
    t_pad = t + (LFR_M - 1) // 2
    last_idx = (t_pad - LFR_M) // LFR_N + 1
    num_padding = LFR_M - (t_pad - last_idx * LFR_N)
    if num_padding > 0:
        num_padding = int(
            (2 * LFR_M - 2 * t_pad + (t_lfr - 1 + last_idx) * LFR_N) / 2 * (t_lfr - last_idx)
        )
        padded = np.vstack([padded] + [padded[-1:]] * num_padding)
    return np.lib.stride_tricks.as_strided(
        padded, shape=(t_lfr, LFR_M * dim), strides=(LFR_N * dim * 4, 4)
    ).copy()


def _decode(logits: np.ndarray, tokenizer: object) -> tuple[str, str]:
    """Greedy CTC decode + language-tag extraction.

    Special token ids (>= 25000) are emitted once each; the sentencepiece
    tokenizer decodes the full id sequence including tags.
    """
    ids = logits[0].argmax(axis=-1).tolist()
    # Legacy decode: collapse consecutive repeats, drop blank (0). The
    # special tokens (<|zh|> etc.) are ordinary ids in the bpe vocab.
    filtered: list[int] = []
    prev = -1
    for tok in ids:
        if tok == prev or tok == 0:
            continue
        prev = tok
        filtered.append(tok)
    if not filtered:
        return "", "auto"
    text = tokenizer.decode(filtered)
    detected = "auto"
    for tag, lang in LANG_MAP.items():
        if tag in text:
            detected = lang
            text = text.replace(tag, "")
            break
    text = re.sub(r"<\|[^|]+\|>", "", text).strip()
    return text, detected


class SenseVoiceOnnxEngine(ASREngineBase):
    """SenseVoiceSmall through onnxruntime."""

    capabilities = EngineCapabilities(input_padding=True)

    def __init__(self, model_path: str | Path | None = None, pad_seconds: float | None = None):
        import onnxruntime as ort

        from livetranslate.core.paths import models_dir

        path = (
            Path(model_path)
            if model_path
            else models_dir() / "sensevoice" / "sensevoice-small.onnx"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"SenseVoice ONNX model not found at {path}. "
                "Run scripts/export_sensevoice_onnx.py on a machine with the "
                "model cached (or download a community export)."
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 4
        self._session = ort.InferenceSession(str(path), sess_options=opts)
        self._tokenizer = self._load_tokenizer(path.parent)
        self._pad_seconds = self._read_pad_seconds(pad_seconds)
        self._pad_quantum = round(SAMPLE_RATE * self._pad_seconds)
        self.language: str | None = None
        log.info(f"SenseVoice (onnx) loaded: {path}")

    @staticmethod
    def _load_tokenizer(model_dir: Path) -> object:
        import sentencepiece as spm

        from livetranslate.modeling.manager import get_local_model_path

        bpe = model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"
        if not bpe.exists():
            local = get_local_model_path("sensevoice", hub="ms")
            if local is not None:
                bpe = Path(local) / "chn_jpn_yue_eng_ko_spectok.bpe.model"
        if not bpe.exists():
            raise FileNotFoundError(f"SenseVoice bpe model not found: {bpe}")
        return spm.SentencePieceProcessor(model_file=str(bpe))

    @staticmethod
    def _read_pad_seconds(value: float | None = None) -> float:
        import os

        if value is None:
            value = os.environ.get(PAD_SECONDS_ENV)
        if value is None:
            return DEFAULT_PAD_SECONDS
        try:
            return float(value)
        except (TypeError, ValueError):
            return DEFAULT_PAD_SECONDS

    def set_input_padding(self, pad_seconds: float) -> None:
        old = self._pad_quantum
        self._pad_seconds = self._read_pad_seconds(pad_seconds)
        self._pad_quantum = round(SAMPLE_RATE * self._pad_seconds)
        if self._pad_quantum != old:
            log.info(f"SenseVoice pad bucket: {self._pad_seconds:g}s")

    def set_language(self, language: str) -> None:
        self.language = None if language == "auto" else language

    def unload(self) -> None:
        self._session = None

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult | None:
        if audio.size == 0:
            return None
        padded = audio
        if self._pad_quantum > 0:
            rem = audio.shape[0] % self._pad_quantum
            if rem:
                padded = np.pad(audio, (0, self._pad_quantum - rem), mode="constant")
        feats = _fbank(padded.astype(np.float32))
        if feats.shape[0] < 4:
            return None
        speech = _apply_lfr(feats)[None, :, :].astype(np.float32)
        logits = self._session.run(None, {"speech": speech})[0]
        text, detected_lang = _decode(logits, self._tokenizer)
        if not text:
            return None
        return TranscriptionResult(text=text, language=detected_lang, language_name=detected_lang)
