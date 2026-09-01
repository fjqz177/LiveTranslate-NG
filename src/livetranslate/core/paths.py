"""Runtime path resolution (SelfServe P0-A4: data never leaves the app).

Frozen builds keep ALL runtime data inside the install directory:

    <install root>\app\\     onedir bundle (LiveTranslate.exe + _internal)
    <install root>\\data\\    settings.json / engines / models / transcripts / logs

Nothing is written to %APPDATA% or any other external location. The sidecar
updater swaps only <install root>\app, so data survives updates.

Dev runs follow the same rule by default: all data lives in the repository
root, so dev and frozen behavior stay in sync. LIVETRANSLATE_PLATFORM_DIRS=1
opts back into the platformdirs user directories for shared or read-only
checkouts; there the legacy repo-root models/ fallback still applies, so an
old checkout never forces a multi-GB model move.
"""

import os
import sys
from pathlib import Path

import platformdirs

APP_NAME = "LiveTranslate"


def _project_root() -> Path:
    """Shipped-resource anchor: bundle dir in frozen builds (the spec
    collects i18n/, config.yaml and assets/icons into _MEIPASS), the
    repository root in source runs."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    # src/livetranslate/core/paths.py -> repository root
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _project_root()


def _explicit_data_root() -> Path | None:
    """Where ALL runtime data lives; None means "platform dirs" (dev opt-out).

    Priority:
    1. LIVETRANSLATE_PORTABLE_DIR — explicit override (CI smoke runs).
    2. Frozen: <install root>\\data — data always lives with the app.
    3. Dev: the repository root by default, mirroring the frozen rule;
       LIVETRANSLATE_PLATFORM_DIRS=1 opts out to the platformdirs layout.
    """
    env_dir = os.environ.get("LIVETRANSLATE_PORTABLE_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent.parent / "data"
    env = os.environ.get("LIVETRANSLATE_PORTABLE", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return PROJECT_ROOT
    if (PROJECT_ROOT / "portable.ini").exists():
        return PROJECT_ROOT
    if os.environ.get("LIVETRANSLATE_PLATFORM_DIRS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return None
    return PROJECT_ROOT


DATA_ROOT = _explicit_data_root()

if DATA_ROOT is not None:
    CONFIG_DIR = DATA_ROOT
    DATA_DIR = DATA_ROOT
    LOG_DIR = DATA_ROOT / "logs"
else:
    CONFIG_DIR = Path(platformdirs.user_config_dir(APP_NAME, appauthor=False))
    DATA_DIR = Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))
    LOG_DIR = Path(platformdirs.user_log_dir(APP_NAME, appauthor=False))

SETTINGS_FILE = CONFIG_DIR / "settings.json"


def data_root() -> Path:
    """Effective data root: the in-app root, or the platform data dir when
    opted out (dev only). Resolved per call so tests can repoint it."""
    return DATA_ROOT if DATA_ROOT is not None else DATA_DIR


def models_dir() -> Path:
    """Model cache root: the data root's models/ (frozen install tree or dev
    repo root). When dev opts out to platform dirs, a legacy repo-root
    models/ dir keeps winning so an old checkout never re-downloads."""
    if DATA_ROOT is not None:
        return DATA_ROOT / "models"
    legacy = PROJECT_ROOT / "models"
    if legacy.exists():
        return legacy
    return DATA_DIR / "models"


def transcripts_dir() -> Path:
    return DATA_DIR / "transcripts"


def is_portable() -> bool:
    """True when all data lives alongside the app (the default layout)."""
    return DATA_ROOT is not None
