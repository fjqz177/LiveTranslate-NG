"""Torch-free accelerator detection (Windows-only).

The base install has no torch, so CUDA presence must be probed without
it: nvcuda.dll + nvidia-smi on Windows. Engine extras may later
cross-check with torch.cuda when available, but the pipeline never
depends on that.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess

from livetranslate.platform.system import AcceleratorInfo


def _run(argv: list[str], timeout: float = 5.0) -> str:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return (proc.stdout or "").strip()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return ""


def _cuda_windows() -> tuple[bool, str]:
    try:
        ctypes.WinDLL("nvcuda.dll")  # raises OSError when absent
    except OSError:
        return False, ""
    name = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=5.0)
    return True, name.splitlines()[0] if name else "NVIDIA GPU"


def detect_accelerator() -> AcceleratorInfo:
    """Best-effort accelerator probe; always fast and never raises."""
    ok, name = _cuda_windows()
    return AcceleratorInfo("cuda", name) if ok else AcceleratorInfo("cpu")


def nvidia_smi_available() -> bool:
    """True when the nvidia-smi CLI is on PATH (used for richer probing)."""
    return shutil.which("nvidia-smi") is not None
