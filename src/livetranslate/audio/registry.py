"""Platform dispatch for audio backends and device enumeration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from livetranslate.audio.backend import AudioBackendUnavailable

if TYPE_CHECKING:
    from livetranslate.audio.backend import AudioBackend, DeviceInfo

log = logging.getLogger("LiveTranslate.Audio")


def create_audio_backend() -> AudioBackend:
    """Backend for Windows: WASAPI loopback (pyaudiowpatch).

    Construction failures degrade to the null backend (AUD-1): a missing
    binding must never take the app down before the UI appears.
    """
    from livetranslate.audio.backends.wasapi import WasapiBackend

    try:
        return WasapiBackend()
    except (AudioBackendUnavailable, ImportError, OSError) as e:
        log.error(f"audio backend construction failed, degrading to null: {e}")

    from livetranslate.audio.backends.null import NullAudioBackend

    return NullAudioBackend()


def list_outputs() -> list[DeviceInfo]:
    from livetranslate.audio.backends.wasapi import list_outputs as _list

    return _list()


def list_inputs() -> list[DeviceInfo]:
    from livetranslate.audio.backends.wasapi import list_inputs as _list

    return _list()
