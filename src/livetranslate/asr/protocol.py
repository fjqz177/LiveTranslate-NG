"""Wire protocol shared by the GUI process and the ASR worker process.

Messages are plain dicts exchanged over a multiprocessing Pipe:

    request:  {"id": <hex>, "type": <str>, "payload": <dict>}
    response: {"id": <hex>, "ok": True, "type": <str>, "payload": <any>}
    error:    {"id": <hex>, "ok": False, "type": "error",
               "error": {"message": ..., "traceback": ..., "recoverable": ...}}

Command types: transcribe / set_language / set_input_padding / ping / shutdown.

Also defines the engine contract: TranscriptionResult (the payload of a
transcribe response), EngineCapabilities and the ASREngineBase ABC that all
engine implementations inherit.
"""

import traceback as _traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass

PROTOCOL_VERSION = 1


def ok_response(msg_id, response_type: str, payload=None) -> dict:
    return {
        "id": msg_id,
        "ok": True,
        "type": response_type,
        "payload": payload,
    }


def error_response(msg_id, exc: BaseException, recoverable: bool) -> dict:
    return {
        "id": msg_id,
        "ok": False,
        "type": "error",
        "error": {
            "message": str(exc),
            # Format from the exception object itself: format_exc() depends on
            # an active except context and returns garbage without one.
            "traceback": "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "recoverable": recoverable,
        },
    }


# ── Engine contract ──


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptionResult:
    """Normalized transcription returned by every ASR engine."""

    text: str
    language: str | None  # ISO code, None = not detected
    language_name: str | None = None
    words: tuple[WordTiming, ...] = ()  # optional; empty = not provided


@dataclass(frozen=True)
class EngineCapabilities:
    """Declared abilities of an engine; replaces hasattr/signature probing."""

    word_timestamps: bool = False
    input_padding: bool = False
    # True when the engine runs out-of-process via ASRClient.
    remote: bool = False


class ASREngineBase(ABC):
    """Interface all ASR engine implementations must satisfy.

    Only transcribe() is abstract. set_language / set_input_padding /
    unload have safe defaults so capability checks never need hasattr.
    """

    capabilities: EngineCapabilities = EngineCapabilities()

    @abstractmethod
    def transcribe(self, audio, word_timestamps: bool = False):
        """Return a TranscriptionResult, or None when no speech was detected."""

    def set_language(self, language: str):  # noqa: B027  # intentional no-op hook
        """Set the recognition language; "auto" means detect."""
        pass

    def set_input_padding(self, pad_seconds: float):  # noqa: B027  # intentional no-op hook
        """Adjust the input-padding bucket. No-op unless supported."""
        pass

    def unload(self):  # noqa: B027  # intentional no-op hook
        """Release model resources."""
        pass

    def shutdown(self):
        """Process-manager friendly alias used by the recovery machinery."""
        self.unload()
