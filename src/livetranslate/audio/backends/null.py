"""Null audio backend: silence for UI startup, tests and headless CI.

Produces correctly-shaped silent chunks so the whole pipeline can run
without any real audio device. read_chunk() is unpaced — the caller's
thread owns the cadence (the real backends pace themselves via stream
reads).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from livetranslate.audio.backend import DeviceInfo


class NullAudioBackend:
    """Silent stream backend (no devices, no errors, no pacing)."""

    name = "null"

    @property
    def device_id(self) -> str | None:
        return self._device_id

    def __init__(self) -> None:
        self._sample_rate = 16000
        self._chunk_ms = 32
        self._running = False
        self._device_id: str | None = None
        self._mic_id: str | None = None

    def list_outputs(self) -> list[DeviceInfo]:
        return []

    def list_inputs(self) -> list[DeviceInfo]:
        return []

    def start(
        self,
        device_id: str | None = None,
        mic_id: str | None = None,
        sample_rate: int = 16000,
        chunk_ms: int = 32,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_ms = chunk_ms
        self._device_id = device_id
        self._mic_id = mic_id
        self._running = True

    def read_chunk(self) -> tuple[np.ndarray, float | None] | None:
        if not self._running:
            return None
        n = int(self._sample_rate * self._chunk_ms / 1000)
        return np.zeros(n, dtype=np.float32), None

    def switch_device(self, device_id: str | None) -> None:
        self._device_id = device_id

    def switch_mic(self, mic_id: str | None) -> None:
        self._mic_id = mic_id

    def stop(self) -> None:
        self._running = False

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "device": self._device_id,
            "rate": self._sample_rate,
            "status": "running" if self._running else "stopped",
            "last_error": None,
        }
