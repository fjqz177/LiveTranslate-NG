"""Filesystem cache detection for downloaded ASR models.

All functions take an explicit `models_dir` (defaulting to ./models) so tests
can reproduce the historical cache-layout edge cases with tmp_path. Pure
filesystem logic — no network, no torch, no Qt.
"""

import importlib.util
import logging
from pathlib import Path

from livetranslate.core.paths import models_dir
from livetranslate.modeling.registry import (
    ASR_MODEL_IDS,
    FUNASR_LEGACY_ENGINE_ALIASES,
    MODEL_SIZE_BYTES,
    WHISPER_SIZES,
    asr_model_id,
    funasr_model_id,
    funasr_profile,
    normalize_funasr_model_key,
    whisper_model_id,
)

log = logging.getLogger("LiveTranslate.ModelCache")

DEFAULT_MODELS_DIR = models_dir()


def _dir(models_dir) -> Path:
    return Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR


def _has_silero_pkg() -> bool:
    """True when the silero-vad PyPI package (model bundled in wheel) is installed."""
    return importlib.util.find_spec("silero_vad") is not None


def is_silero_cached(models_dir=None) -> bool:
    if _has_silero_pkg():
        return True
    torch_hub = _dir(models_dir) / "torch" / "hub"
    return any(torch_hub.glob("snakers4_silero-vad*")) if torch_hub.exists() else False


def _custom_whisper_path(value, models_dir=None):
    if not value or value in WHISPER_SIZES:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _dir(models_dir).parent / path
    return path


def is_faster_whisper_model_dir(path) -> bool:
    """True when path looks like a CTranslate2 faster-whisper model directory."""
    if not path:
        return False
    path = Path(path)
    return path.is_dir() and (path / "model.bin").is_file() and (path / "config.json").is_file()


def resolve_custom_whisper_model(value, models_dir=None) -> str | None:
    path = _custom_whisper_path(value, models_dir)
    if path and is_faster_whisper_model_dir(path):
        return str(path.resolve())
    return None


def _is_builtin_whisper_cache(path: Path) -> bool:
    parts = set(path.parts)
    if any(f"models--Systran--faster-whisper-{s}" in parts for s in WHISPER_SIZES):
        return True
    # ModelScope cache layouts: Systran/faster-whisper-{s} flat dir, or the
    # >=1.38 models/Systran--faster-whisper-{s}/snapshots/{rev} tree.
    if "modelscope" not in parts:
        return False
    return any(
        p == f"faster-whisper-{s}" or p == f"Systran--faster-whisper-{s}"
        for p in parts
        for s in WHISPER_SIZES
    )


def _hf_snapshot_name(path: Path) -> str | None:
    """Return 'org/repo' for .../models--org--repo/snapshots/<hash>."""
    if path.parent.name != "snapshots":
        return None
    repo_dir = path.parent.parent
    if not repo_dir.name.startswith("models--"):
        return None

    encoded = repo_dir.name.removeprefix("models--")
    parts = encoded.split("--", 1)
    if len(parts) != 2 or not all(parts):
        return None
    return f"{parts[0]}/{parts[1]}"


def list_local_faster_whisper_models(models_dir=None) -> list[dict]:
    """Scan models_dir for user-provided faster-whisper model directories."""
    models_dir = _dir(models_dir)
    if not models_dir.exists():
        return []

    entries = []
    name_counts = {}
    seen = set()
    try:
        model_bins = list(models_dir.rglob("model.bin"))
    except (OSError, PermissionError):
        return []

    for model_bin in model_bins:
        model_dir = model_bin.parent
        if _is_builtin_whisper_cache(model_dir):
            continue
        if not is_faster_whisper_model_dir(model_dir):
            continue
        try:
            resolved = str(model_dir.resolve())
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)

        name = _hf_snapshot_name(model_dir) or model_dir.name
        name_counts[name] = name_counts.get(name, 0) + 1
        if name_counts[name] > 1:
            name = f"{name} ({model_dir.name[:8]})"
        entries.append({"name": name, "path": resolved})

    entries.sort(key=lambda item: item["name"].lower())
    return entries


def local_faster_whisper_display_name(path, models_dir=None) -> str | None:
    """Return the same display name used by the local Whisper model selector."""
    resolved = resolve_custom_whisper_model(path, models_dir)
    if not resolved:
        return None
    for item in list_local_faster_whisper_models(models_dir):
        if item["path"] == resolved:
            return item["name"]
    return _hf_snapshot_name(Path(resolved)) or Path(resolved).name


def _ms_model_path(org, name, models_dir=None):
    """Return the first existing ModelScope cache path, or the default.

    Layouts by SDK version: {org}/{name} = <=1.37 with explicit cache_dir;
    models/{org}/{name} = 1.34~1.37 env-default cache, which >=1.38 keeps
    reusing as legacy even when cache_dir is passed (dots in names written
    as ___ by old SDKs); hub trees = older SDKs; models/{org}--{name}/
    snapshots/{revision} = >=1.38 fresh cache.
    """
    ms_root = _dir(models_dir) / "modelscope"
    for sub in (
        ms_root / org / name,
        ms_root / "models" / org / name,
        ms_root / "models" / org / name.replace(".", "___"),
        ms_root / "hub" / "models" / org / name,
        ms_root / "hub" / org / name,
    ):
        if sub.exists():
            return sub
    snap_root = ms_root / "models" / f"{org}--{name}" / "snapshots"
    if snap_root.is_dir():
        snaps = sorted(d for d in snap_root.iterdir() if d.is_dir())
        if snaps:
            # Prefer the revision that actually contains the checkpoint;
            # stray artifact dirs (e.g. an onnx export) must not win.
            complete = [d for d in snaps if (d / "model.pt").is_file()]
            return (complete or snaps)[-1]
    return ms_root / org / name


def _dir_complete(d: Path, min_bytes: int) -> bool:
    """True when a model dir has substantial, readable files (no broken links).

    '.incomplete' blobs from an aborted download are ignored; a dir containing
    only those counts as incomplete.
    """
    total = 0
    for f in d.rglob("*"):
        if f.is_dir() or f.name.endswith(".incomplete"):
            continue
        try:
            total += f.stat().st_size
        except OSError:
            return False
    return total >= min_bytes


def _hf_repo_complete(org: str, name: str, min_bytes: int = 50_000_000, models_dir=None) -> bool:
    """True if a HuggingFace repo cache exists AND finished downloading.

    A killed/aborted download leaves snapshot entries pointing at missing blobs
    (broken symlinks) or '.incomplete' blobs; treating that as cached makes the
    model load hang. Validate a snapshot where every file resolves (stat follows
    symlinks; a broken link raises) and the resolved bytes are substantial. This
    ignores orphan '.incomplete' blobs left behind by an earlier interrupted run.
    """
    snap_root = _dir(models_dir) / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
    if not snap_root.exists():
        return False
    return any(snap.is_dir() and _dir_complete(snap, min_bytes) for snap in snap_root.iterdir())


def _ms_repo_complete(org: str, name: str, min_bytes: int = 50_000_000, models_dir=None) -> bool:
    """True if a ModelScope cache for org/name exists AND finished downloading.

    Mirrors _hf_repo_complete across the historical ModelScope cache layouts
    handled by _ms_model_path: any complete dir wins, an aborted download
    (partial or missing files) does not count.
    """
    ms_root = _dir(models_dir) / "modelscope"
    snap_root = ms_root / "models" / f"{org}--{name}" / "snapshots"
    if snap_root.is_dir():
        for snap in snap_root.iterdir():
            if snap.is_dir() and _dir_complete(snap, min_bytes):
                return True
    for d in (
        ms_root / org / name,
        ms_root / "models" / org / name,
        ms_root / "models" / org / name.replace(".", "___"),
        ms_root / "hub" / "models" / org / name,
        ms_root / "hub" / org / name,
    ):
        if d.is_dir() and _dir_complete(d, min_bytes):
            return True
    return False


def qwen_weights_present(model_dir) -> bool:
    """Whether a nano model's embedded Qwen3-0.6B weights are in place.

    Nano repos ship the Qwen3-0.6B config but not its weights. A variant without
    the subdir needs no Qwen weights, so absence of the subdir counts as present.
    """
    qwen_dir = Path(model_dir) / "Qwen3-0.6B"
    if not qwen_dir.is_dir():
        return True
    return any(f.suffix in (".safetensors", ".bin") for f in qwen_dir.iterdir())


def is_asr_cached(engine_type, model_size="medium", hub="ms", models_dir=None) -> bool:
    if engine_type == "funasr" or engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        model_key = (
            FUNASR_LEGACY_ENGINE_ALIASES[engine_type]
            if engine_type in FUNASR_LEGACY_ENGINE_ALIASES
            else normalize_funasr_model_key(model_size)
        )
        # Accept cache from either hub to avoid redundant downloads; the repo
        # namespace can differ between ModelScope and HuggingFace (SenseVoice).
        ms_org, ms_name = funasr_model_id(model_key, "ms").split("/")
        hf_org, hf_name = funasr_model_id(model_key, "hf").split("/")
        # CORE-1: the ModelScope branch used to be a bare .exists() check —
        # an interrupted download's empty dir counted as cached and the
        # worker then failed at load time. Both hubs now require a complete
        # snapshot (dir_complete + size threshold).
        if not (
            _ms_repo_complete(ms_org, ms_name, models_dir=models_dir)
            or _hf_repo_complete(hf_org, hf_name, models_dir=models_dir)
        ):
            return False
        # Nano's Qwen3-0.6B weights download separately; require them so the
        # download flow (not the deadline-bound worker) pulls them up-front.
        if funasr_profile(model_key)["family"] == "funasr-nano":
            model_dir = get_local_model_path(
                engine_type, hub, funasr_model=model_size, models_dir=models_dir
            )
            if not model_dir or not qwen_weights_present(model_dir):
                return False
        return True
    if engine_type == "anime-whisper":
        # HF-only (not published to ModelScope). Check that snapshots dir actually
        # contains weight files; an .incomplete blob means a prior run aborted
        # mid-download.
        model_id = ASR_MODEL_IDS[engine_type]
        org, name = model_id.split("/")
        snap_root = (
            _dir(models_dir) / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
        )
        if not snap_root.exists():
            return False
        for snap in snap_root.iterdir():
            if not snap.is_dir():
                continue
            has_weights = any(
                (snap / fn).exists() for fn in ("model.safetensors", "pytorch_model.bin")
            )
            has_config = (snap / "config.json").exists()
            if has_weights and has_config:
                return True
        return False
    if engine_type == "whisper":
        if model_size not in WHISPER_SIZES:
            return resolve_custom_whisper_model(model_size, models_dir) is not None
        min_bytes = int(MODEL_SIZE_BYTES.get(f"whisper-{model_size}", 50_000_000) * 0.5)
        # Accept cache from either hub (like funasr) to avoid re-downloading
        # the same weights when the user switches the download source.
        org, name = whisper_model_id(model_size).split("/")
        return _ms_repo_complete(org, name, min_bytes, models_dir=models_dir) or _hf_repo_complete(
            org, name, min_bytes=min_bytes, models_dir=models_dir
        )
    if engine_type == "sensevoice-onnx":
        # M-MATRIX: the ONNX export artifact gates the cached state. This is a
        # base-only torch-free path, so unlike funasr/whisper/anime-whisper there
        # is no torch hub snapshot to validate — the .onnx file is the model.
        return (_dir(models_dir) / "sensevoice" / "sensevoice-small.onnx").is_file()
    return True


def get_local_model_path(engine_type, hub="ms", funasr_model: str | None = None, models_dir=None):
    """Return local snapshot path if model is cached, else None.

    Checks the preferred hub first, then falls back to the other hub.
    """
    models_dir = _dir(models_dir)
    if engine_type == "funasr" or engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        model_key = (
            FUNASR_LEGACY_ENGINE_ALIASES[engine_type]
            if engine_type in FUNASR_LEGACY_ENGINE_ALIASES
            else normalize_funasr_model_key(funasr_model)
        )
        ms_org, ms_name = funasr_model_id(model_key, "ms").split("/")
        hf_org, hf_name = funasr_model_id(model_key, "hf").split("/")
    elif engine_type in ASR_MODEL_IDS:
        ms_org, ms_name = asr_model_id(engine_type, "ms").split("/")
        hf_org, hf_name = asr_model_id(engine_type, "hf").split("/")
    else:
        return None

    def _try_ms():
        local = _ms_model_path(ms_org, ms_name, models_dir)
        return str(local) if local.exists() else None

    def _try_hf():
        snap_dir = models_dir / "huggingface" / "hub" / f"models--{hf_org}--{hf_name}" / "snapshots"
        if snap_dir.exists():
            snaps = sorted(snap_dir.iterdir())
            if snaps:
                return str(snaps[-1])
        return None

    if hub == "ms":
        return _try_ms() or _try_hf()
    return _try_hf() or _try_ms()


def get_whisper_local_path(model_size, hub="ms", models_dir=None) -> str | None:
    """Return a local dir loadable by faster-whisper for a builtin size.

    faster-whisper accepts a local directory directly (no network). ModelScope
    caches are flat model dirs, so they load as-is; for HuggingFace the
    snapshot dir is returned. Prefers the configured hub, falls back to the
    other, and only returns dirs that pass the same completeness check as
    is_asr_cached.
    """
    if model_size not in WHISPER_SIZES:
        return None
    models_dir = _dir(models_dir)
    min_bytes = int(MODEL_SIZE_BYTES.get(f"whisper-{model_size}", 50_000_000) * 0.5)
    org, name = whisper_model_id(model_size).split("/")

    def _try_ms():
        ms_root = models_dir / "modelscope"
        snap_root = ms_root / "models" / f"{org}--{name}" / "snapshots"
        if snap_root.is_dir():
            for snap in sorted(snap_root.iterdir()):
                if snap.is_dir() and _dir_complete(snap, min_bytes):
                    return str(snap)
        for d in (
            ms_root / org / name,
            ms_root / "models" / org / name,
            ms_root / "models" / org / name.replace(".", "___"),
            ms_root / "hub" / "models" / org / name,
            ms_root / "hub" / org / name,
        ):
            if d.is_dir() and _dir_complete(d, min_bytes):
                return str(d)
        return None

    def _try_hf():
        snap_root = models_dir / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
        if not snap_root.exists():
            return None
        for snap in sorted(snap_root.iterdir()):
            if snap.is_dir() and _dir_complete(snap, min_bytes):
                return str(snap)
        return None

    if hub == "ms":
        return _try_ms() or _try_hf()
    return _try_hf() or _try_ms()
