"""Torch-free accelerator detection (Windows-only).

The base install has no torch, so CUDA presence must be probed without
it: nvcuda.dll + nvidia-smi on Windows. Engine extras may later
cross-check with torch.cuda when available, but the pipeline never
depends on that.
"""

from __future__ import annotations

import ctypes
import re
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


# Engine-variant driver floors (SelfServe P1-B4). PyTorch wheels bundle the
# CUDA runtime, so only the NVIDIA driver version matters. The only CUDA
# variant shipped is cu126; the floor follows the PyTorch compatibility table
# (cu126 ~ R560). Recalibrate on real hardware during P1 acceptance and keep
# the source in the comment.
_DRIVER_FLOORS: tuple[tuple[str, tuple[int, ...]], ...] = (("cu126", (560,)),)


def _parse_driver_version(raw: str) -> tuple[int, ...]:
    """'566.14' -> (566, 14); leading non-digit noise is skipped."""
    return tuple(int(g) for g in re.findall(r"\d+", raw))


def _variant_for_driver(driver_version: tuple[int, ...]) -> str:
    for variant, floor in _DRIVER_FLOORS:
        if driver_version[: len(floor)] >= floor:
            return variant
    return "cpu"


def _driver_version() -> tuple[int, ...]:
    """Driver version via nvidia-smi; empty tuple when unavailable."""
    try:
        ctypes.WinDLL("nvcuda.dll")
    except OSError:
        return ()
    out = _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        timeout=5.0,
    )
    return _parse_driver_version(out.splitlines()[0]) if out else ()


def detect_variant() -> str:
    """Recommended engine variant: cu126 / cpu.

    Never raises; CUDA-less machines resolve to cpu.
    """
    return _variant_for_driver(_driver_version())
