import contextlib
import logging

import numpy as np

from livetranslate.asr.protocol import ASREngineBase, TranscriptionResult
from livetranslate.core.privacy import redact_text

log = logging.getLogger("LiveTranslate.AnimeWhisper")

MODEL_ID = "litagin/anime-whisper"


class AnimeWhisperEngine(ASREngineBase):
    """Speech-to-text using litagin/anime-whisper (kotoba-whisper-v2.0 fine-tune).

    Japanese-only, specialized for anime / galgame speech (sighs, breaths, etc.).
    Loaded via transformers pipeline; no faster-whisper / ctranslate2 path.
    """

    def __init__(self, device="cuda", hub="hf"):
        import torch
        from transformers import pipeline

        from livetranslate.modeling.manager import get_local_model_path

        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        local = get_local_model_path("anime-whisper", hub=hub)
        model = local or MODEL_ID

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self._device = device
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=device,
            torch_dtype=dtype,
            chunk_length_s=30.0,
            batch_size=1,
        )
        self.language = "ja"
        log.info(f"AnimeWhisper loaded from {redact_text(str(model))} on {device}")

    def set_language(self, language: str):
        # Model is Japanese-only; ignore attempts to change
        if language not in ("auto", "ja", None):
            log.info(f"AnimeWhisper is Japanese-only, ignoring language={language}")
        self.language = "ja"

    def unload(self):
        if hasattr(self, "_pipe") and self._pipe is not None:
            with contextlib.suppress(Exception):
                self._pipe.model.to("cpu")
            self._pipe = None

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult | None:
        """Transcribe audio segment (float32, 16kHz mono)."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        import torch

        # anime-whisper README: disable initial_prompt, suppress repetitions
        with torch.inference_mode():
            result = self._pipe(
                audio,
                generate_kwargs={
                    "language": "Japanese",
                    "task": "transcribe",
                    "do_sample": False,
                    "num_beams": 1,
                    "no_repeat_ngram_size": 5,
                    "repetition_penalty": 1.0,
                },
            )

        text = (result or {}).get("text", "").strip()
        if not text:
            return None

        return TranscriptionResult(
            text=text,
            language="ja",
            language_name="ja",
        )
