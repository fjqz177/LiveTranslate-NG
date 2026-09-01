"""Engine-runtime area management (SelfServe P1-B1): filesystem + meta.

The engine area lives under the install-tree data root:

    <data>\\engines\
        meta.json                {schema, state, active, variants}
        .staging-<variant>\\      in-flight install (deleted on recovery)
        <variant>\\.venv\\         uv-managed venv per hardware variant

Invariants:
- All mutations are atomic (tmp + os.replace for meta; rename for venvs).
- At most one install in flight; a crash mid-install leaves
  state="installing" and a .staging-* dir, which recover() cleans up.
- A variant becomes active only after its venv is complete (the meta
  `variants[<name>].complete` boolean, set in complete_install — there is
  no filesystem .complete marker; do not probe for one).

Pure filesystem logic: no Qt, no network, no uv subprocess here — the uv
execution lives in engine_runtime's UvRunner (separate module in this file's
sibling scope; see plan §2.3).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from typing import TYPE_CHECKING, Any, Literal

from livetranslate.core import paths as _paths

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger("LiveTranslate.EngineRuntime")

SCHEMA_VERSION = 1
State = Literal["idle", "installing", "ready"]

# Engines whose backends live in the engine venv (not the frozen bundle):
# these need an installed variant + variant_site_packages injection.
VENV_BACKED_ENGINES: tuple[str, ...] = ("whisper", "funasr", "anime-whisper")


class EngineRuntimeError(RuntimeError):
    """Engine-area invariant violated (corrupt meta, unknown variant, ...)."""


def _data_root() -> Path:
    # Resolved per call: tests monkeypatch paths.DATA_ROOT.
    return _paths.data_root()


def engines_dir() -> Path:
    return _data_root() / "engines"


def meta_path() -> Path:
    return engines_dir() / "meta.json"


def staging_dir(variant: str) -> Path:
    return engines_dir() / f".staging-{variant}"


def variant_dir(variant: str) -> Path:
    return engines_dir() / variant


def _valid_variant(variant: str) -> bool:
    return bool(variant) and variant.isidentifier() and not variant.startswith(".")


def load_meta() -> dict[str, Any]:
    """Read meta.json; missing file means a fresh area (empty dict)."""
    path = meta_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise EngineRuntimeError(f"corrupt meta.json: {exc}") from exc
    if not isinstance(data, dict):
        raise EngineRuntimeError("corrupt meta.json: not an object")
    return data


def save_meta(data: dict[str, Any]) -> None:
    """Atomic write (tmp + replace), mirroring core/settings.py discipline."""
    path = meta_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def recover() -> None:
    """Clean a half-finished install: state installing -> idle.

    A previous run died mid-install (power loss / crash). The staging
    directory is garbage; the active variant (if any) is untouched.
    """
    meta = load_meta()
    if meta.get("state") != "installing":
        return
    for staging in engines_dir().glob(".staging-*"):
        shutil.rmtree(staging, ignore_errors=True)
    meta["state"] = "idle"
    save_meta(meta)
    log.info("engine area recovered: stale staging removed, state -> idle")


def begin_install(variant: str, version: str) -> Path:
    """Mark state=installing and return the staging venv dir to fill.

    Raises EngineRuntimeError if another install is in flight or the variant
    name is invalid. Callers run recover() at app startup before any install;
    this function must NOT recover (an in-flight install is not stale here).
    """
    if not _valid_variant(variant):
        raise EngineRuntimeError(f"invalid variant name: {variant!r}")
    meta = load_meta()
    if meta.get("state") == "installing":
        raise EngineRuntimeError("another engine install is in flight")
    if (meta.get("variants") or {}).get(variant, {}).get("complete"):
        raise EngineRuntimeError(f"variant {variant!r} is already installed")
    meta["schema"] = SCHEMA_VERSION
    meta["state"] = "installing"
    meta.setdefault("variants", {})[variant] = {
        "version": version,
        "complete": False,
        "installed_at": None,
    }
    save_meta(meta)
    staging = staging_dir(variant)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def complete_install(variant: str, active: bool = True) -> None:
    """Rename staging -> variant, mark complete, optionally activate."""
    meta = load_meta()
    if meta.get("state") != "installing":
        raise EngineRuntimeError("no install in flight")
    staging = staging_dir(variant)
    target = variant_dir(variant)
    if not (staging / ".venv").exists():
        shutil.rmtree(staging, ignore_errors=True)
        raise EngineRuntimeError(f"staging venv missing for {variant!r}")
    shutil.rmtree(target, ignore_errors=True)
    os.replace(staging, target)
    entry = (meta.setdefault("variants", {})).setdefault(variant, {})
    entry["complete"] = True
    entry["installed_at"] = int(time.time())
    meta["state"] = "ready"
    if active:
        meta["active"] = variant
    save_meta(meta)


def abort_install(variant: str) -> None:
    """Roll a failed install back: remove staging, state -> ready/idle."""
    shutil.rmtree(staging_dir(variant), ignore_errors=True)
    meta = load_meta()
    if meta.get("state") == "installing":
        meta["state"] = "ready" if (meta.get("variants") or {}).get("active") else "idle"
    variants = meta.get("variants", {})
    entry = variants.get(variant)
    if entry is not None and not entry.get("complete"):
        del variants[variant]
    save_meta(meta)


def activate(variant: str) -> None:
    """Switch the active variant (caller must restart the ASR worker)."""
    meta = load_meta()
    entry = (meta.get("variants") or {}).get(variant)
    if entry is None or not entry.get("complete"):
        raise EngineRuntimeError(f"variant {variant!r} is not installed")
    meta["active"] = variant
    save_meta(meta)


def remove_variant(variant: str) -> None:
    """Delete a variant; refusing to delete the active one would strand the
    user mid-config, so deactivate it first and let the caller handle the
    worker restart."""
    meta = load_meta()
    shutil.rmtree(variant_dir(variant), ignore_errors=True)
    variants = meta.get("variants", {})
    variants.pop(variant, None)
    if meta.get("active") == variant:
        meta["active"] = None
    if not variants:
        meta["state"] = "idle"
    save_meta(meta)


def active_variant() -> str | None:
    meta = load_meta()
    active = meta.get("active")
    if not active:
        return None
    return active if (meta.get("variants") or {}).get(active, {}).get("complete") else None


def installed_variants() -> list[str]:
    meta = load_meta()
    return [
        name
        for name, entry in (meta.get("variants") or {}).items()
        if entry.get("complete") and variant_dir(name).exists()
    ]


def variant_site_packages(variant: str) -> Path:
    """site-packages path injected into the ASR worker for this variant."""
    return variant_dir(variant) / ".venv" / "Lib" / "site-packages"


def runtime_prefs(settings: dict[str, Any]) -> tuple[str, str]:
    """(variant, mirror) from settings' engine_runtime section (SelfServe P1-B7).

    Defaults to auto/auto; tolerant of missing keys and legacy settings
    files that predate the section.
    """
    prefs = settings.get("engine_runtime")
    if not isinstance(prefs, dict):
        return "auto", "auto"
    variant = prefs.get("variant", "auto")
    mirror = prefs.get("mirror", "auto")
    return str(variant), str(mirror)
