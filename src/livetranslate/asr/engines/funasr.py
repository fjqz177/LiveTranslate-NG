import logging

import numpy as np

from livetranslate.asr.protocol import ASREngineBase, EngineCapabilities
from livetranslate.modeling.manager import funasr_profile, normalize_funasr_model_key

log = logging.getLogger("LiveTranslate.FunASR")


class FunASREngine(ASREngineBase):
    """Unified FunASR engine that dispatches to model-family adapters."""

    def __init__(
        self,
        model_key: str = "sensevoice-small",
        device: str = "cuda",
        hub: str = "ms",
        pad_seconds: float | None = None,
    ):
        self.model_key = normalize_funasr_model_key(model_key)
        self.profile = funasr_profile(self.model_key)
        self.family = self.profile["family"]
        self.capabilities = EngineCapabilities(
            input_padding=bool(self.profile.get("supports_padding"))
        )

        if self.family == "sensevoice":
            from livetranslate.asr.engines.sensevoice import SenseVoiceEngine

            self._engine = SenseVoiceEngine(device=device, hub=hub, pad_seconds=pad_seconds)
        elif self.family == "funasr-nano":
            from livetranslate.asr.engines.funasr_nano import FunASRNanoEngine

            self._engine = FunASRNanoEngine(
                device=device,
                hub=hub,
                engine_type=self.profile["legacy_engine"],
            )
        else:
            raise ValueError(f"Unsupported FunASR model family: {self.family}")

        log.info(
            f"FunASR loaded: {self.profile['display_name']} "
            f"({self.model_key}, family={self.family})"
        )

    def set_language(self, language: str):
        self._engine.set_language(language)

    def set_input_padding(self, pad_seconds):
        if self.capabilities.input_padding:
            self._engine.set_input_padding(pad_seconds)

    def unload(self):
        self._engine.unload()

    def transcribe(self, audio: np.ndarray):
        return self._engine.transcribe(audio)
