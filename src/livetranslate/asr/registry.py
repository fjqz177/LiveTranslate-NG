"""ASR engine registry: capabilities, extras and per-platform defaults.

The GUI/panel reads this table to render engine choices (name, tier,
download size, required extras); recommend_engine() drives the first-run
wizard defaults from the detected accelerator (core/systeminfo.py).

M-MATRIX (2026-08-31): ``EngineSpec.engine_type`` is the single source of truth
for the worker engine type — the ``asr/worker._ENGINE_FACTORIES`` key, the
persisted ``settings['asr_engine']`` value and ``worker_config['engine_type']``.
Nothing else defines that vocabulary; the GUI combo/indices are derived here
(GUI_ENGINE_ORDER), not duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from livetranslate.platform.system import AcceleratorInfo

EngineTier = Literal["recommended", "normal", "advanced", "legacy"]


@dataclass(frozen=True)
class EngineSpec:
    id: str
    display_name: str
    tier: EngineTier
    # Single-source engine type: the asr/worker._ENGINE_FACTORIES key, the
    # persisted settings['asr_engine'] value, and worker_config['engine_type'].
    # REQUIRED (no default); must equal a worker factory key (or 'remote-whisper'
    # for the in-process remote client). Never a separate GUI/combo vocabulary.
    engine_type: str
    extras: tuple[str, ...] = ()  # engine-* groups required at runtime
    platforms: tuple[str, ...] = ()  # Windows-only -> ("win32",)
    download_gb: float = 0.0  # first-run model download estimate
    # ASR-1: whether the ASR worker actually has a factory for this engine
    # (asr/worker._ENGINE_FACTORIES). False = declared but not wired up yet
    # (binding lands with D13) — the UI must say so instead of pretending
    # the extras are missing.
    worker_ready: bool = True


ENGINE_REGISTRY: dict[str, EngineSpec] = {
    "faster-whisper": EngineSpec(
        id="faster-whisper",
        display_name="Whisper (faster-whisper)",
        tier="recommended",
        engine_type="whisper",
        extras=("engine-whisper",),
        platforms=("win32",),
        download_gb=1.5,  # medium; tiny ~0.08, large-v3 ~3
    ),
    "sensevoice-funasr": EngineSpec(
        id="sensevoice-funasr",
        display_name="FunASR (SenseVoice, legacy torch path)",
        tier="legacy",
        engine_type="funasr",
        extras=("engine-funasr",),
        platforms=("win32",),
        download_gb=1.0,
    ),
    "sensevoice-onnx": EngineSpec(
        id="sensevoice-onnx",
        display_name="SenseVoice (ONNX)",
        tier="normal",
        engine_type="sensevoice-onnx",
        extras=("engine-sensevoice-onnx",),
        platforms=("win32",),
        download_gb=0.9,  # onnx export of SenseVoiceSmall
    ),
    "anime-whisper": EngineSpec(
        id="anime-whisper",
        display_name="Anime-Whisper (ja)",
        tier="normal",
        engine_type="anime-whisper",
        extras=("engine-whisper",),
        platforms=("win32",),
        download_gb=3.0,
    ),
    "remote": EngineSpec(
        id="remote",
        display_name="Remote Whisper (LAN GPU server)",
        tier="normal",
        engine_type="remote-whisper",
        platforms=("win32",),
        download_gb=0.0,
    ),
}


# GUI selection order (win32): recommended CPU path first, then the legacy/
# advanced + remote alternatives. Single source of truth for the panel dropdown
# (Task 6 wires it up); do not duplicate elsewhere.
GUI_ENGINE_ORDER: tuple[str, ...] = (
    "faster-whisper",
    "sensevoice-onnx",
    "sensevoice-funasr",
    "anime-whisper",
    "remote",
)


def recommend_engine(_accel: AcceleratorInfo) -> str:
    """First-run default (Windows-only): faster-whisper.

    Whisper (faster-whisper) is the product default: it runs on both CPU and
    CUDA (ctranslate2), has a straightforward hub download path and is the engine
    users reach for. SenseVoice-ONNX stays selectable as a normal option for the
    lighter torch-free path. ``_accel`` is reserved for future per-hardware
    tuning (e.g. a larger model on GPU) and is intentionally unused for now.

    Returns a registry id. Callers resolve ``ENGINE_REGISTRY[id].engine_type`` to
    obtain the worker ``engine_type`` (settings['asr_engine'] /
    worker_config['engine_type']).
    """
    return "faster-whisper"


def engines_for_platform(platform: str) -> list[str]:
    """Registry ids selectable on ``platform`` (registry-definition order)."""
    return [eid for eid, spec in ENGINE_REGISTRY.items() if platform in spec.platforms]


def engine_type_for_engine(engine_id: str) -> str:
    """The worker ``engine_type`` for a registry id (= ``_ENGINE_FACTORIES`` key)."""
    return ENGINE_REGISTRY[engine_id].engine_type


def engine_id_for_type(engine_type: str, platform: str = "win32") -> str | None:
    """Reverse of ``engine_type_for_engine``: registry id for a worker ``engine_type``.

    One-to-one with ``_ENGINE_FACTORIES`` (plus the in-process ``remote-whisper``).
    Round-trips the legacy persisted ``settings['asr_engine']`` values exactly
    (whisper / funasr / anime-whisper / remote-whisper). Returns ``None`` when the
    type is unknown or not selectable on ``platform`` — callers decide the fallback.
    (Chosen over raising so legacy/mismatched persisted values degrade gracefully.)
    """
    by_type: dict[str, str] = {spec.engine_type: eid for eid, spec in ENGINE_REGISTRY.items()}
    engine_id = by_type.get(engine_type)
    if engine_id is None:
        return None
    if engine_id not in engines_for_platform(platform):
        return None
    return engine_id


def engine_display_key(engine_id: str) -> str:
    """The i18n ``t()`` key for an engine's localized dropdown label.

    Registry ids use hyphens (``faster-whisper``, ``sensevoice-onnx``) while the
    i18n keys use underscores (``engine_display_faster_whisper``) — YAML keys
    here are conventionally alphanumeric+underscore. This is the single
    place that maps id -> key, so the panel dropdown (vad_tab.py) and the
    ``tests/asr/test_engine_matrix.py`` matrix guard keep the same vocabulary
    and can never drift apart.
    """
    return f"engine_display_{engine_id.replace('-', '_')}"
