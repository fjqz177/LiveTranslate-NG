"""Engine-runtime area tests (SelfServe P1-B1): meta state machine.

An autouse fixture points paths.DATA_ROOT at a fresh tmp dir per test;
engine_runtime resolves the area per call, so no module reload is needed.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest

import livetranslate.core.engine_runtime as er
import livetranslate.core.paths as paths

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolated_area(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "data"
    monkeypatch.setattr(paths, "DATA_ROOT", root)
    root.mkdir()


def _make_venv(staging: Path) -> None:
    (staging / ".venv" / "pyvenv.cfg").parent.mkdir(parents=True)
    (staging / ".venv" / "pyvenv.cfg").write_text("home = python\n", encoding="utf-8")


def test_fresh_area_has_empty_meta():
    assert er.load_meta() == {}
    assert er.active_variant() is None
    assert er.installed_variants() == []


def test_install_flow_atomic_and_activates():
    staging = er.begin_install("cpu", "0.2.0")
    assert er.load_meta()["state"] == "installing"
    _make_venv(staging)
    er.complete_install("cpu")
    meta = er.load_meta()
    assert meta["state"] == "ready"
    assert meta["active"] == "cpu"
    assert meta["variants"]["cpu"]["complete"] is True
    assert er.active_variant() == "cpu"
    assert not staging.exists()
    assert (er.variant_dir("cpu") / ".venv" / "pyvenv.cfg").exists()


def test_abort_install_rolls_back():
    staging = er.begin_install("cpu", "0.2.0")
    _make_venv(staging)
    er.abort_install("cpu")
    meta = er.load_meta()
    assert meta["state"] == "idle"
    assert meta.get("variants", {}) == {}
    assert not staging.exists()


def test_recover_cleans_stale_staging():
    staging = er.begin_install("cpu", "0.2.0")
    _make_venv(staging)  # simulate: died before complete_install
    er.recover()
    meta = er.load_meta()
    assert meta["state"] == "idle"
    assert not staging.exists()
    # the incomplete variant entry stays in meta but is not complete, so
    # installed_variants filters it out.
    assert er.installed_variants() == []


def test_double_install_rejected():
    er.begin_install("cpu", "0.2.0")
    with pytest.raises(er.EngineRuntimeError, match="in flight"):
        er.begin_install("cu126", "0.2.0")


def test_activate_and_switch():
    for variant in ("cpu", "cu126"):
        staging = er.begin_install(variant, "0.2.0")
        _make_venv(staging)
        er.complete_install(variant, active=(variant == "cpu"))
    assert er.active_variant() == "cpu"
    er.activate("cu126")
    assert er.active_variant() == "cu126"
    with pytest.raises(er.EngineRuntimeError, match="not installed"):
        er.activate("cu999")


def test_remove_variant_deactivates():
    staging = er.begin_install("cpu", "0.2.0")
    _make_venv(staging)
    er.complete_install("cpu")
    er.remove_variant("cpu")
    assert er.active_variant() is None
    assert er.installed_variants() == []
    assert er.load_meta()["state"] == "idle"
    assert not er.variant_dir("cpu").exists()


def test_invalid_variant_names_rejected():
    for bad in ("", "cuda x", "..", "a/b"):
        with pytest.raises(er.EngineRuntimeError, match="invalid variant"):
            er.begin_install(bad, "0.2.0")


def test_corrupt_meta_raises():
    er.meta_path().parent.mkdir(parents=True, exist_ok=True)
    er.meta_path().write_text("{not json", encoding="utf-8")
    with pytest.raises(er.EngineRuntimeError, match="corrupt meta"):
        er.load_meta()


def test_meta_is_valid_json_after_save():
    er.begin_install("cpu", "0.2.0")
    raw = er.meta_path().read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["schema"] == er.SCHEMA_VERSION
    assert re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", "cpu")


def test_site_packages_path_matches_worker_injection():
    staging = er.begin_install("cpu", "0.2.0")
    _make_venv(staging)
    er.complete_install("cpu")
    assert (
        er.variant_site_packages("cpu") == er.variant_dir("cpu") / ".venv" / "Lib" / "site-packages"
    )


def test_runtime_prefs_defaults_and_legacy_tolerance():
    assert er.runtime_prefs({}) == ("auto", "auto")
    assert er.runtime_prefs({"engine_runtime": None}) == ("auto", "auto")
    assert er.runtime_prefs({"engine_runtime": {"variant": "cu126"}}) == ("cu126", "auto")
    assert er.runtime_prefs({"engine_runtime": {"mirror": "tsinghua"}}) == ("auto", "tsinghua")
    assert er.runtime_prefs({"engine_runtime": {"variant": "cpu", "mirror": "aliyun"}}) == (
        "cpu",
        "aliyun",
    )
