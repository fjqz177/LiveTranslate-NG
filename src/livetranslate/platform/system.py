"""System integration protocol: files, autostart, single instance and
accelerator discovery behind one interface (Windows-only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class AcceleratorInfo:
    """Detected compute accelerator (torch-free detection)."""

    kind: Literal["cuda", "cpu"]
    device_name: str = ""
    vram_mb: int = 0

    @property
    def display(self) -> str:
        if self.kind == "cuda":
            return self.device_name or "NVIDIA GPU (CUDA)"
        return "CPU"


class SystemIntegration(Protocol):
    """Platform glue that the UI consumes."""

    name: str

    def open_path(self, path: Path) -> None:
        """Open a file or directory with the OS default handler."""
        ...

    def set_autostart(self, enabled: bool) -> None:
        """Enable/disable launch at login (per user)."""
        ...

    def autostart_enabled(self) -> bool: ...

    def try_acquire_single_instance(self, key: str) -> bool:
        """Try to claim the app's single-instance lock; False = another
        instance already runs (the Qt layer then asks it to wake up)."""
        ...

    def release_single_instance(self) -> None: ...

    def accelerator(self) -> AcceleratorInfo:
        """Torch-free accelerator detection (see core/systeminfo.py)."""
        ...
