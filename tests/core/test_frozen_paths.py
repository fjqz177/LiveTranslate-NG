"""Path semantics (SelfServe P0-A4 + unified data root): data never leaves
the app — frozen builds use the install-tree data\\, dev runs use the
repository root by default (LIVETRANSLATE_PLATFORM_DIRS=1 opts out).

paths.py computes its constants at import time, so each test reloads the
module under the monkeypatched runtime state.
"""

from __future__ import annotations

import importlib
import sys

import livetranslate.core.paths as paths


def _reload_as_frozen(monkeypatch, exe: str) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", exe)
    monkeypatch.delenv("LIVETRANSLATE_PORTABLE_DIR", raising=False)
    monkeypatch.delenv("LIVETRANSLATE_PORTABLE", raising=False)
    monkeypatch.delenv("LIVETRANSLATE_PLATFORM_DIRS", raising=False)
    importlib.reload(paths)


def test_frozen_data_root_is_install_tree_data(monkeypatch):
    _reload_as_frozen(monkeypatch, r"C:\fake\install\app\LiveTranslate.exe")

    assert str(paths.DATA_ROOT) == r"C:\fake\install\data"
    assert str(paths.SETTINGS_FILE) == r"C:\fake\install\data\settings.json"
    assert str(paths.LOG_DIR) == r"C:\fake\install\data\logs"
    assert str(paths.models_dir()) == r"C:\fake\install\data\models"
    assert str(paths.transcripts_dir()) == r"C:\fake\install\data\transcripts"
    assert paths.is_portable() is True


def test_frozen_explicit_override_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\fake\install\app\LiveTranslate.exe")
    monkeypatch.setenv("LIVETRANSLATE_PORTABLE_DIR", str(tmp_path / "smoke"))
    monkeypatch.delenv("LIVETRANSLATE_PORTABLE", raising=False)
    importlib.reload(paths)

    assert tmp_path / "smoke" == paths.DATA_ROOT
    assert tmp_path / "smoke" / "settings.json" == paths.SETTINGS_FILE


def test_dev_portable_env_keeps_repo_root(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setenv("LIVETRANSLATE_PORTABLE", "1")
    monkeypatch.delenv("LIVETRANSLATE_PORTABLE_DIR", raising=False)
    importlib.reload(paths)

    assert paths.DATA_ROOT == paths.PROJECT_ROOT
    assert paths.is_portable() is True


def test_dev_default_keeps_data_in_repo(monkeypatch):
    """The unified rule: dev data lives in the repo root like frozen data
    lives in the install tree — same behavior, no external writes."""
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.delenv("LIVETRANSLATE_PORTABLE", raising=False)
    monkeypatch.delenv("LIVETRANSLATE_PORTABLE_DIR", raising=False)
    monkeypatch.delenv("LIVETRANSLATE_PLATFORM_DIRS", raising=False)
    importlib.reload(paths)

    assert paths.DATA_ROOT == paths.PROJECT_ROOT
    assert paths.is_portable() is True
    assert paths.SETTINGS_FILE == paths.PROJECT_ROOT / "settings.json"
    assert paths.LOG_DIR == paths.PROJECT_ROOT / "logs"
    assert paths.models_dir() == paths.PROJECT_ROOT / "models"
    assert paths.transcripts_dir() == paths.PROJECT_ROOT / "transcripts"
    assert paths.data_root() == paths.PROJECT_ROOT


def test_dev_platform_dirs_opt_out(monkeypatch):
    """LIVETRANSLATE_PLATFORM_DIRS=1 restores the platformdirs layout for
    shared/read-only checkouts (escape hatch, not the default)."""
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setenv("LIVETRANSLATE_PLATFORM_DIRS", "1")
    monkeypatch.delenv("LIVETRANSLATE_PORTABLE", raising=False)
    monkeypatch.delenv("LIVETRANSLATE_PORTABLE_DIR", raising=False)
    importlib.reload(paths)

    assert paths.DATA_ROOT is None
    assert paths.is_portable() is False
    assert "LiveTranslate" in str(paths.CONFIG_DIR)
    assert paths.SETTINGS_FILE == paths.CONFIG_DIR / "settings.json"
    assert paths.data_root() == paths.DATA_DIR
