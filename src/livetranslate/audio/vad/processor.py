"""Voice Activity Detection facade with multiple modes.

Thin backward-compatible facade: picks the confidence scorer for the
selected mode (silero / energy / disabled) and delegates all segmentation
state to SpeechSegmenter (pure logic, see vad_segmenter.py).
"""

import logging

from livetranslate.audio.vad.scorer import (
    AlwaysSpeechScorer,
    EnergyConfidenceScorer,
    SileroConfidenceScorer,
)
from livetranslate.audio.vad.segmenter import SpeechSegmenter

log = logging.getLogger("LiveTranslate.VAD")


class VADProcessor:
    """Voice Activity Detection with multiple modes."""

    def __init__(
        self,
        sample_rate=16000,
        threshold=0.50,
        min_speech_duration=1.0,
        max_speech_duration=15.0,
        chunk_duration=0.032,
    ):
        self.sample_rate = sample_rate
        self._silero_scorer = SileroConfidenceScorer(threshold=threshold, sample_rate=sample_rate)
        self._energy_scorer = EnergyConfidenceScorer()
        self._disabled_scorer = AlwaysSpeechScorer()
        self._scorers = {
            "silero": self._silero_scorer,
            "energy": self._energy_scorer,
            "disabled": self._disabled_scorer,
        }
        self._segmenter = SpeechSegmenter(
            scorer=self._silero_scorer,
            sample_rate=sample_rate,
            min_speech_duration=min_speech_duration,
            max_speech_duration=max_speech_duration,
            chunk_duration=chunk_duration,
        )

    def update_settings(self, settings: dict):
        mode = settings.get("vad_mode")
        if mode in self._scorers:
            self._segmenter.set_scorer(self._scorers[mode])
        if "vad_threshold" in settings:
            self._silero_scorer.threshold = settings["vad_threshold"]
        if "energy_threshold" in settings:
            self._energy_scorer.energy_threshold = settings["energy_threshold"]
        self._segmenter.update_settings(settings)
        seg = self._segmenter
        log.info(
            f"VAD settings updated: mode={mode}, "
            f"silence={seg.silence_mode} "
            f"({seg.silence_limit} chunks = "
            f"{seg.silence_limit * seg.chunk_duration:.2f}s)"
        )

    # --- Delegation to SpeechSegmenter ---

    def process_chunk(self, audio_chunk):
        return self._segmenter.process_chunk(audio_chunk)

    def peek_buffer(self):
        return self._segmenter.peek_buffer()

    def trim_front(self, n_samples: int):
        return self._segmenter.trim_front(n_samples)

    def force_flush(self):
        return self._segmenter.force_flush()

    def flush(self):
        return self._segmenter.flush()

    def reset(self):
        return self._segmenter.reset()

    def is_speaking(self) -> bool:
        return self._segmenter.is_speaking()

    def buffered_samples(self) -> int:
        return self._segmenter.buffered_samples()

    def buffered_duration(self) -> float:
        return self._segmenter.buffered_duration()

    def current_confidence(self) -> float:
        return self._segmenter.current_confidence()

    def effective_silence_limit(self) -> int:
        return self._segmenter.effective_silence_limit()
