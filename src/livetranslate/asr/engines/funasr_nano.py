import contextlib
import logging
import os
import re
import sys
import tempfile
import wave

import numpy as np
import torch

from livetranslate.asr.protocol import ASREngineBase, TranscriptionResult

log = logging.getLogger("LiveTranslate.FunASR-Nano")

# Add bundled code to path so model.py can resolve its imports (ctc, tools.utils).
# The vendored package is a sibling of engines/ (asr/vendor/funasr_nano), not a
# subdirectory of engines/.
_NANO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "vendor", "funasr_nano"))


class FunASRNanoEngine(ASREngineBase):
    """Speech-to-text using Fun-ASR-Nano-2512 or Fun-ASR-MLT-Nano-2512."""

    def __init__(self, device="cuda", hub="ms", engine_type="funasr-nano"):
        if _NANO_DIR not in sys.path:
            sys.path.insert(0, _NANO_DIR)

        # Pre-register FunASRNano model class before AutoModel looks it up
        import model as _nano_model  # noqa: F401
        from funasr import AutoModel

        from livetranslate.modeling.manager import (
            ASR_MODEL_IDS,
            ensure_qwen_weights,
            get_local_model_path,
            neutralize_funasr_requirements,
        )

        model_name = ASR_MODEL_IDS[engine_type]
        local = get_local_model_path(engine_type, hub=hub)
        model = local or model_name

        if local:
            # Safety net; the download flow normally fetches these up-front.
            ensure_qwen_weights(local, hub=hub)
            neutralize_funasr_requirements(local)

        prev_cwd = os.getcwd()
        if local:
            os.chdir(local)
        try:
            self._model = AutoModel(
                model=model,
                trust_remote_code=True,
                device=device,
                hub=hub,
                disable_update=True,
            )
        finally:
            os.chdir(prev_cwd)
        self.language = None
        log.info(f"{engine_type} loaded: {model_name} on {device} (hub={hub})")

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"Fun-ASR-Nano language: {old} -> {self.language}")

    def unload(self):
        if hasattr(self, "_model") and self._model is not None:
            with contextlib.suppress(Exception):
                self._model.model.to("cpu")
            self._model = None

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult | None:
        """Transcribe audio segment (float32, 16kHz mono)."""
        # CWE-377: mktemp is a predictable-name race; mkstemp creates the
        # file with 0600 and a random name (fd closed immediately — wave
        # re-opens by name).
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            audio_16bit = (audio * 32767).astype(np.int16)
            with wave.open(tmp, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_16bit.tobytes())

            kwargs = {"input": [tmp], "batch_size": 1, "disable_pbar": True}
            if self.language:
                kwargs["language"] = self.language

            with torch.inference_mode():
                result = self._model.generate(**kwargs)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)

        if not result or not result[0].get("text"):
            return None

        # "text" keeps punctuation; "text_tn" strips it all via regex
        text = result[0]["text"]

        # Clean special tags
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()

        if not text or text == "sil":
            return None

        detected_lang = self.language or self._guess_language(text)

        log.debug(f"ASR: {len(text)} chars")
        return TranscriptionResult(
            text=text,
            language=detected_lang,
            language_name=detected_lang,
        )

    def _guess_language(self, text: str) -> str:
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        jp = sum(1 for c in text if "\u3040" <= c <= "\u30ff" or "\u31f0" <= c <= "\u31ff")
        ko = sum(1 for c in text if "\uac00" <= c <= "\ud7af")
        total = len(text)
        if total == 0:
            return "auto"
        if jp > 0:
            return "ja"
        if ko > total * 0.3:
            return "ko"
        if cjk > total * 0.3:
            return "zh"
        return "en"
