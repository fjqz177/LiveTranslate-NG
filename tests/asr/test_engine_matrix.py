"""M-MATRIX consistency guard: lock the whole engine matrix.

Pure-import pytest (no Qt, never imports ``asr.engines``; torch only ever as a
*name* in the pyproject checks) that makes ``asr.registry`` the single source
of truth and prevents M1-style drift across:

  1. registry recommendation  ->  GUI_ENGINE_ORDER  ->  ENGINE_REGISTRY
  2. engine_type exact round-trip (engine_type_for_engine <-> engine_id_for_type)
  3. worker_ready exactly (== engine_type in asr.worker._ENGINE_FACTORIES or "remote-whisper")
  4. extras / EXTRAS_PROBE_MAP / pyproject optional-dependencies closure
  5. sensevoice-onnx torch-free + gated by the ONNX artifact
  6. every GUI_ENGINE_ORDER win32 id selectable (engines_for_platform)
  (bonus) every GUI engine id has a localized engine_display_* key (zh/en)

Runs inside ``uv run livetranslate-pr``. If any assertion goes RED on a correct
baseline it is a residual bug to FIX, not an assertion to loosen.
"""

from __future__ import annotations

from pathlib import Path

import tomllib
import yaml

from livetranslate.asr import availability
from livetranslate.asr.registry import (
    ENGINE_REGISTRY,
    GUI_ENGINE_ORDER,
    engine_display_key,
    engine_id_for_type,
    engine_type_for_engine,
    engines_for_platform,
    recommend_engine,
)
from livetranslate.asr.worker import _ENGINE_FACTORIES
from livetranslate.modeling import cache
from livetranslate.platform.system import AcceleratorInfo

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"

# Derive the loadable engine-type vocabulary from asr.worker (never hardcode a
# parallel id->type dict): worker factories + the in-process remote client.
WORKER_LOADABLE = set(_ENGINE_FACTORIES) | {"remote-whisper"}

# Legacy persisted settings['asr_engine'] values -> expected registry id
# (one-to-one with WORKER_LOADABLE / engine_id_for_type).
LEGACY_ENGINE_TYPE_TO_ID = {
    "whisper": "faster-whisper",
    "funasr": "sensevoice-funasr",
    "anime-whisper": "anime-whisper",
    "remote-whisper": "remote",
}


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _project_optional_deps() -> dict[str, list[str]]:
    return _project().get("project", {}).get("optional-dependencies", {})


# ── 1. recommendation is reachable from the GUI ─────────────────────────────


def test_recommendation_is_gui_reachable():
    for accel in (
        AcceleratorInfo(kind="cpu"),
        AcceleratorInfo(kind="cuda", device_name="RTX 4090"),
    ):
        recommended = recommend_engine(accel)
        assert recommended in GUI_ENGINE_ORDER, recommended
        assert recommended in ENGINE_REGISTRY, recommended
    # GUI_ENGINE_ORDER must only reference real registry entries.
    for eid in GUI_ENGINE_ORDER:
        assert eid in ENGINE_REGISTRY


# ── 2. engine_type exact round-trip ─────────────────────────────────────────


def test_engine_type_round_trips_exactly():
    for eid in ENGINE_REGISTRY:
        etype = engine_type_for_engine(eid)
        assert engine_id_for_type(etype, platform="win32") == eid


def test_legacy_engine_type_to_id_round_trips():
    for etype, eid in LEGACY_ENGINE_TYPE_TO_ID.items():
        assert engine_id_for_type(etype, platform="win32") == eid


# ── 3. worker_ready exactly mirrors the worker factories ────────────────────


def test_worker_ready_exactly_matches_factories():
    for eid, spec in ENGINE_REGISTRY.items():
        expected = spec.engine_type in WORKER_LOADABLE
        assert spec.worker_ready == expected, (
            f"{eid}: worker_ready={spec.worker_ready} engine_type={spec.engine_type!r}"
        )


# ── 4. extras / pyproject closure ──────────────────────────────────────────


def test_registry_extras_exist_in_pyproject():
    optional = _project_optional_deps()
    for eid, spec in ENGINE_REGISTRY.items():
        for extra in spec.extras:
            assert extra in optional, f"{eid} declares {extra!r} missing from pyproject"


# ── 5. sensevoice-onnx: torch-free + gated by the ONNX artifact ──────────────


def test_sensevoice_onnx_is_torch_free():
    optional = _project_optional_deps()
    # engine_type is its own type (sensevoice-onnx) — no torch path.
    assert engine_type_for_engine("sensevoice-onnx") == "sensevoice-onnx"
    # registry tier is normal: Whisper (faster-whisper) is the product default,
    # SenseVoice-ONNX stays as a selectable torch-free option.
    assert ENGINE_REGISTRY["sensevoice-onnx"].tier == "normal"
    # base deps must never contain torch.
    base = _project().get("project", {}).get("dependencies", [])
    assert not any(dep.startswith("torch") for dep in base), "torch leaked into base deps"
    # the extra, if declared, must be a torch-free group.
    for extra in ENGINE_REGISTRY["sensevoice-onnx"].extras:
        group = optional.get(extra, [])
        assert not any(member.startswith("torch") for member in group), f"{extra} pulls torch"
        if extra == "engine-sensevoice-onnx":
            assert group == [], "engine-sensevoice-onnx group must stay empty (torch-free)"


def test_sensevoice_onnx_gated_by_onnx_artifact(tmp_path):
    assert callable(availability.sensevoice_onnx_model_present)
    etype = "sensevoice-onnx"
    # is_asr_cached walks the ONNX-file existence branch (base-only, torch-free).
    assert cache.is_asr_cached(etype, models_dir=tmp_path) is False
    artifact = tmp_path / "sensevoice" / "sensevoice-small.onnx"
    artifact.parent.mkdir(parents=True)
    artifact.touch()
    assert cache.is_asr_cached(etype, models_dir=tmp_path) is True


def test_engine_status_sensevoice_onnx_uses_model_gate(monkeypatch):
    monkeypatch.setattr(availability, "sensevoice_onnx_model_present", lambda: False)
    assert availability.engine_status("sensevoice-onnx", "win32") == "needs-model"


# ── 6. top-level selectability on win32 ─────────────────────────────────────


def test_gui_order_win32_selectable():
    selectable = set(engines_for_platform("win32"))
    for eid in GUI_ENGINE_ORDER:
        assert eid in selectable


# ── bonus: every GUI engine id has a localized display key (zh/en) ──────────


def test_every_gui_engine_has_localized_display_key():
    # The key vocabulary comes from registry.engine_display_key (single source),
    # so the dropdown (vad_tab) and this guard can never drift apart.
    zh = yaml.safe_load((ROOT / "i18n" / "zh.yaml").read_text(encoding="utf-8"))
    en = yaml.safe_load((ROOT / "i18n" / "en.yaml").read_text(encoding="utf-8"))
    for eid in GUI_ENGINE_ORDER:
        key = engine_display_key(eid)
        assert key in zh, f"missing zh display key {key!r}"
        assert key in en, f"missing en display key {key!r}"
        assert zh[key] != key, f"zh display key {key!r} untranslated"
        assert en[key] != key, f"en display key {key!r} untranslated"
    assert "engine_tier_recommended" in zh
    assert "engine_tier_recommended" in en
