import logging
import struct

import httpx
import numpy as np

from livetranslate.asr.protocol import TranscriptionResult

log = logging.getLogger("LiveTranslate.ASR.Remote")


class RemoteASRError(Exception):
    """The remote ASR server could not be reached or failed the request.

    Distinct from returning None (which means "no speech detected") so the
    pipeline can route connection failures through the worker-recovery path.
    """


class RemoteASREngine:
    """Speech-to-text via remote faster-whisper server.

    Sends raw PCM audio to the server, receives transcription back.
    Implements the same interface as ASREngine for drop-in replacement.
    """

    def __init__(self, server_url="http://127.0.0.1:8765", timeout=30.0, token: str = ""):
        self._url = server_url.rstrip("/") + "/transcribe"
        self._health_url = server_url.rstrip("/") + "/health"
        # SEC-5: the shared token rides the X-ASR-Token header on every
        # request (health probe included — a token-gated server rejects
        # anonymous health checks).
        self._headers = {"X-ASR-Token": token} if token else {}
        # trust_env=False bypasses the system / HTTP(S)_PROXY: the URL points
        # straight at a LAN/localhost server, and routing localhost through a proxy
        # returns a non-JSON error page instead of connecting. Bound the connect
        # phase so an unreachable host fails fast rather than blocking the load dialog.
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0), trust_env=False)
        self.language = None

        # Verify the server is reachable and speaks our protocol.
        try:
            r = self._client.get(self._health_url, headers=self._headers)
            r.raise_for_status()
            info = r.json()
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach remote ASR server at {server_url}: {e}. "
                f"Make sure asr_server.py is running and the URL is correct."
            ) from e
        log.info(f"Connected to remote ASR server: {info}")

    # --- ASRClient-compatible shim ----------------------------------------
    # The pipeline drives the active ASR backend through the ASRClient interface
    # (a worker-process proxy). RemoteASREngine runs in-process, so it presents the
    # same surface to slot into _switch_asr_engine / _run_asr / _mem_snapshot etc.

    @property
    def status(self) -> str:
        return "ready"

    @property
    def pid(self):
        # No worker process; a None pid makes the memory monitor and worker-recycle
        # logic skip this engine.
        return None

    def shutdown(self):
        self.unload()

    def terminate(self):
        self.unload()

    def set_input_padding(self, pad_seconds):
        pass

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"Remote ASR language: {old} -> {self.language}")

    def unload(self):
        self._client.close()
        log.info("Remote ASR connection closed")

    def transcribe(
        self, audio: np.ndarray, word_timestamps: bool = False, **kwargs
    ) -> TranscriptionResult | None:
        """Send audio to remote server and return transcription."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Build request: [lang_len: uint32][language: bytes][audio: float32 bytes]
        lang_str = (self.language or "").encode("utf-8")
        header = struct.pack("<I", len(lang_str)) + lang_str
        payload = header + audio.tobytes()

        try:
            r = self._client.post(self._url, content=payload, headers=self._headers)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.error(f"Remote ASR request failed: {e}")
            raise RemoteASRError(f"Remote ASR request failed: {e}") from e

        text = data.get("text")
        if not text:
            return None

        elapsed = data.get("elapsed", 0)
        log.debug(f"Remote ASR: {elapsed:.2f}s -> {text[:60]}")

        detected_lang = data.get("language", "unknown")
        return TranscriptionResult(
            text=text,
            language=detected_lang,
            language_name=detected_lang,
        )
