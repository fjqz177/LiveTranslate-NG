"""uv_runner tests (SelfServe P1-B2): command assembly and rollback.

All subprocess calls are mocked — no real uv runs here. The engine area is
isolated per test by pointing paths.DATA_ROOT at a tmp dir.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import livetranslate.core.engine_runtime as er
import livetranslate.core.paths as paths
from livetranslate.core import uv_runner

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "data"
    monkeypatch.setattr(paths, "DATA_ROOT", root)
    root.mkdir()
    req = tmp_path / "reqroot" / "runtime" / "requirements"
    req.mkdir(parents=True)
    (req / "cpu.txt").write_text("torch==2.11.0+cpu\n", encoding="utf-8")
    (req / "cu126.txt").write_text("torch==2.11.0+cu126\n", encoding="utf-8")
    monkeypatch.setattr(uv_runner, "PROJECT_ROOT", tmp_path / "reqroot")


def _fake_py_dir(venv_arg: Path) -> None:
    py_dir = venv_arg.parent / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    py_dir.mkdir(parents=True, exist_ok=True)
    (py_dir / ("python.exe" if os.name == "nt" else "python")).write_text("", encoding="utf-8")


@pytest.fixture()
def fake_uv(monkeypatch) -> Iterator[list[str]]:
    """A fake uv that logs argv and creates a venv python when asked."""
    calls: list[str] = []

    def _run(cmd, capture_output=False, **_kwargs):
        calls.append(" ".join(str(c) for c in cmd))
        if "venv" in cmd and capture_output:
            _fake_py_dir(Path(cmd[2]))  # uv venv <staging>/.venv --python 3.12
        # uv_runner decodes with explicit encoding, so stderr is always str.
        return type("P", (), {"returncode": 0, "stderr": "traceback"})()

    monkeypatch.setattr(uv_runner.subprocess, "run", _run)
    return calls


def test_install_cpu_success(fake_uv: list[str]):
    lines: list[str] = []
    dest = uv_runner.install_variant("cpu", app_version="0.2.0", progress_cb=lines.append)

    assert dest == er.variant_dir("cpu")
    assert er.active_variant() == "cpu"
    joined = " ".join(fake_uv)
    assert "pip install -r" in joined
    assert "--index-url https://pypi.org/simple" in joined
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in joined
    # Regression: per-package index annotations in the pinned requirements
    # dead-end under uv's first-index-wins strategy (certifi on the pytorch
    # mirror) — the install must opt out of it.
    assert "--index-strategy unsafe-best-match" in joined
    assert lines, "progress callbacks fired"


def test_mirror_selection_rewrites_pypi_only(fake_uv: list[str]):
    """P2-C3 measured: Chinese pytorch-wheels mirrors are not PEP 503
    indexes, so mirror selection applies to PyPI only — torch stays on the
    official index for every variant."""
    uv_runner.install_variant("cu126", app_version="0.2.0", pypi_mirror="tsinghua")
    joined = " ".join(fake_uv)
    assert "--index-url https://pypi.tuna.tsinghua.edu.cn/simple" in joined
    assert "--extra-index-url https://download.pytorch.org/whl/cu126" in joined


def test_install_failure_rolls_back(monkeypatch):
    def _fail(cmd, capture_output=False, **_kwargs):
        if "pip" in cmd:
            return type("P", (), {"returncode": 1, "stderr": "boom: no disk"})()
        if capture_output:
            _fake_py_dir(Path(cmd[2]))
        return type("P", (), {"returncode": 0, "stderr": "traceback"})()

    monkeypatch.setattr(uv_runner.subprocess, "run", _fail)

    with pytest.raises(uv_runner.UvRunnerError, match="no disk"):
        uv_runner.install_variant("cpu", app_version="0.2.0")

    assert er.installed_variants() == []
    assert er.load_meta()["state"] == "idle"


def test_fix_pyvenv_home_points_at_real_dir(tmp_path: Path):
    """uv canonicalizes managed interpreters back to the version-alias
    symlink; the rewrite must pin home to the concrete dir (alias junctions
    can be marked untrusted mount points → os error 448)."""
    venv = tmp_path / "venv"
    venv.mkdir()
    cfg = venv / "pyvenv.cfg"
    cfg.write_text(
        "home = C:\\alias\\cpython-3.12-windows-x86_64-none\n"
        "implementation = CPython\n"
        "uv = 0.12.5\n",
        encoding="utf-8",
    )
    real = tmp_path / "real" / "cpython-3.12.13-windows-x86_64-none" / "python.exe"
    real.parent.mkdir(parents=True)
    real.write_text("", encoding="utf-8")

    uv_runner._fix_pyvenv_home(venv, real)

    content = cfg.read_text(encoding="utf-8")
    assert f"home = {real.parent}" in content
    assert "implementation = CPython" in content
    assert "cpython-3.12-windows-x86_64-none" not in content


def test_unknown_variant_rejected():
    with pytest.raises(uv_runner.UvRunnerError, match="unknown variant"):
        uv_runner.requirements_file("cu999")


def test_unknown_mirror_rejected():
    with pytest.raises(uv_runner.UvRunnerError, match="unknown pypi mirror"):
        uv_runner._resolve_mirror("mars")


def test_uv_cache_env_frozen_pins_into_data_root(monkeypatch):
    """Installed/portable builds must leave nothing in the user profile:
    uv's wheel cache is pinned into the install-tree data root."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    env = uv_runner._uv_cache_env()
    assert env is not None
    assert env["UV_CACHE_DIR"] == str(paths.DATA_ROOT / "uv-cache")


def test_uv_cache_env_dev_keeps_uv_default(monkeypatch):
    """Dev runs never touch uv's cache location: the default (or an explicit
    user UV_CACHE_DIR) stays in effect."""
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert uv_runner._uv_cache_env() is None


def test_install_forwards_cache_env_to_uv(monkeypatch):
    sentinel = {"UV_CACHE_DIR": "X:/fake/uv-cache"}
    monkeypatch.setattr(uv_runner, "_uv_cache_env", lambda: sentinel)
    captured: list[dict[str, str] | None] = []

    def _run(cmd, capture_output=False, **_kwargs):
        captured.append(_kwargs.get("env"))
        if "venv" in cmd and capture_output:
            _fake_py_dir(Path(cmd[2]))
        return type("P", (), {"returncode": 0, "stderr": "traceback"})()

    monkeypatch.setattr(uv_runner.subprocess, "run", _run)
    uv_runner.install_variant("cpu", app_version="0.2.0")

    assert captured == [sentinel, sentinel]


def test_pypi_mirrors_include_nju_and_ustc():
    """Domestic PyPI mirrors (nju/ustc) join the candidate set for faster downloads."""
    assert uv_runner.PYPI_MIRRORS["nju"] == "https://mirrors.nju.edu.cn/pypi/simple"
    assert uv_runner.PYPI_MIRRORS["ustc"] == "https://mirrors.ustc.edu.cn/pypi/simple"
    assert uv_runner.PYPI_MIRRORS["official"] == "https://pypi.org/simple"


def test_torch_mirror_rewrites_index():
    """torch index can switch from official to the verified NJU mirror (PEP 503 + {cu})."""
    assert uv_runner._torch_index("cu126", "nju") == "https://mirrors.nju.edu.cn/pytorch/whl/cu126"
    assert uv_runner._torch_index("cpu", "nju") == "https://mirrors.nju.edu.cn/pytorch/whl/cpu"
    # Default stays official (backward compatible with existing callers/tests)
    assert uv_runner._torch_index("cu126") == "https://download.pytorch.org/whl/cu126"


def test_unknown_torch_mirror_rejected():
    with pytest.raises(uv_runner.UvRunnerError, match="unknown torch mirror"):
        uv_runner._torch_index("cpu", "mars")


def test_install_forwards_torch_mirror(fake_uv: list[str]):
    """install_variant forwards torch_mirror to --extra-index-url."""
    uv_runner.install_variant("cu126", app_version="0.2.0", torch_mirror="nju")
    joined = " ".join(fake_uv)
    assert "--extra-index-url https://mirrors.nju.edu.cn/pytorch/whl/cu126" in joined
