"""Engine availability: which extras/artifacts are present.

The panel uses engine_status() to render each ENGINE_REGISTRY entry:
available (deps + model artifact present), needs-extras, or
needs-model. Frozen builds get the same signal through the artifact
checks; the actual installation UX (uv extras in dev, engine packs in
Phase 7) keys off these statuses.
"""

from __future__ import annotations

import importlib.util
from typing import Literal

from livetranslate.core.paths import models_dir

EngineStatus = Literal[
    "available",
    "needs-extras",
    "needs-model",
    "unsupported",
    "not-implemented",  # ASR-1: declared in the registry, no worker factory yet
]

# extra -> importable module probe (shared with the panel install button).
# sensevoice-onnx has no extra (fbank is the numpy port, inference is
# onnxruntime in base), so its probe is an always-present base module and
# only the model artifact check gates it.
EXTRAS_PROBE_MAP: dict[str, tuple[str, ...]] = {
    "engine-whisper": ("faster_whisper",),
    "engine-funasr": ("funasr",),
    "engine-sensevoice-onnx": ("numpy",),
}


def extras_installed(module_names: tuple[str, ...]) -> bool:
    """True when every module can be imported (extras presence probe)."""
    return all(importlib.util.find_spec(name) is not None for name in module_names)


def sensevoice_onnx_model_present() -> bool:
    path = models_dir() / "sensevoice" / "sensevoice-small.onnx"
    return path.is_file()


def engine_status(engine_id: str, platform: str) -> EngineStatus:
    """Availability of one ENGINE_REGISTRY entry on this machine."""
    from livetranslate.asr.registry import ENGINE_REGISTRY

    spec = ENGINE_REGISTRY.get(engine_id)
    if spec is None:
        raise KeyError(f"unknown engine id: {engine_id}")
    if platform not in spec.platforms:
        return "unsupported"
    if not spec.worker_ready:
        # ASR-1: never masquerade "not wired up yet" as "deps missing" —
        # installing extras can never make this engine work today.
        return "not-implemented"
    for extra in spec.extras:
        if not extras_installed(EXTRAS_PROBE_MAP.get(extra, (extra,))):
            return "needs-extras"
    if engine_id == "sensevoice-onnx" and not sensevoice_onnx_model_present():
        return "needs-model"
    return "available"
