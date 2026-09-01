"""Engine availability: which artifacts are present.

The panel uses engine_status() to render each ENGINE_REGISTRY entry:
available, needs-model, unsupported, or not-implemented. Engine deps ship
with the base install, so the signal is driven by artifact presence rather
than extras.
"""

from __future__ import annotations

from typing import Literal

from livetranslate.core.paths import models_dir

EngineStatus = Literal[
    "available",
    "needs-model",
    "unsupported",
    "not-implemented",  # ASR-1: declared in the registry, no worker factory yet
]


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
        # extra installs can never make this engine work today.
        return "not-implemented"
    if engine_id == "sensevoice-onnx" and not sensevoice_onnx_model_present():
        return "needs-model"
    return "available"
