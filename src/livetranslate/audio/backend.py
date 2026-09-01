"""Audio backend protocol: every OS audio stack behind one interface.

Backends produce float32 16 kHz mono chunks (default 32 ms) with an optional
microphone mix. The capture pipeline only sees this contract; concrete
backends live in audio/backends/ and own device enumeration, stream
lifecycle, resampling and hot-switching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    import numpy as np

DeviceKind = Literal["loopback", "input"]


class AudioBackendUnavailable(RuntimeError):
    """A backend cannot be constructed or used on this machine (missing
    binding/driver, unsupported platform variant). Construction layers must
    degrade instead of crashing — the app stays usable and diagnostics tell
    the truth (detect → degrade → guide)."""


@dataclass(frozen=True)
class DeviceInfo:
    id: str
    name: str
    kind: DeviceKind
    channels: int
    default_rate: int
    is_default: bool = False


class AudioBackend(Protocol):
    """System audio capture for one platform.

    Threading contract: a dedicated read thread calls read_chunk() and may
    call switch_*() concurrently; implementations must serialize stream
    open/close internally (the WASAPI backend uses a lock + command queue).
    """

    name: str  # "wasapi" | "coreaudio" | "pipewire" | "null" | ...

    @property
    def device_id(self) -> str | None:
        """Currently captured loopback device (None = following default)."""
        ...

    def list_outputs(self) -> list[DeviceInfo]: ...
    def list_inputs(self) -> list[DeviceInfo]: ...

    def start(
        self,
        device_id: str | None = None,
        mic_id: str | None = None,
        sample_rate: int = 16000,
        chunk_ms: int = 32,
    ) -> None: ...

    def read_chunk(self) -> tuple[np.ndarray, float | None] | None:
        """(mixed mono float32 at sample_rate, mic_rms) or None when idle."""
        ...

    def switch_device(self, device_id: str | None) -> None: ...
    def switch_mic(self, mic_id: str | None) -> None: ...
    def stop(self) -> None: ...

    def diagnostics(self) -> dict[str, object]:
        """{backend, device, rate, status, last_error} for the diagnostics panel."""
        ...
