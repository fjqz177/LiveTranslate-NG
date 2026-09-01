"""Runtime path resolution tests: frozen bundle anchoring (Phase 7)."""

import sys
from pathlib import Path

from livetranslate.core import paths


def test_dev_project_root_is_repository():
    assert paths._project_root() == paths.PROJECT_ROOT
    assert (paths.PROJECT_ROOT / "pyproject.toml").exists()


def test_frozen_project_root_uses_bundle_dir(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", "C:/Frozen/Bundle", raising=False)
    assert paths._project_root() == Path("C:/Frozen/Bundle")


def test_frozen_project_root_falls_back_to_exe_dir(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(sys, "executable", "C:/Apps/LiveTranslate/LiveTranslate.exe", raising=False)
    assert paths._project_root() == Path("C:/Apps/LiveTranslate")
