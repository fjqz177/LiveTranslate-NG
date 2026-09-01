"""Pure settings persistence — no Qt.

SettingsStore (the Qt signal wrapper) lives in
livetranslate.ui.settings_bridge; this module only owns file I/O, the
canonical SETTINGS_FILE location and the migration-aware entry points used
by the app entry point and dialogs.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from livetranslate.core.paths import APP_NAME, PROJECT_ROOT, SETTINGS_FILE
from livetranslate.modeling.registry import migrate_funasr_settings

log = logging.getLogger("LiveTranslate.Settings")

# Pre-rewrite location: settings lived next to the sources as
# user_settings.json. Migrated on first load, kept in place as a backup.
LEGACY_SETTINGS_FILE = PROJECT_ROOT / "user_settings.json"


def load_settings_from(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            # CORE-5: settings.json must be a JSON object. Valid non-object
            # JSON ([]/"x"/42/null) previously slipped past the cast and
            # crashed callers with AttributeError — treat it like corruption.
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                log.warning(
                    "Settings at %s is valid JSON but not an object (%s); using defaults",
                    path,
                    type(data).__name__,
                )
                return None
            return data
    except Exception as e:
        log.warning(f"Failed to load settings from {path}: {e}")
    return None


def save_settings_to(path: Path, settings: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # SEC-2: pid-suffixed temp name — a fixed ".tmp" target is
        # predictable (another local process could race the write), and the
        # file contains plaintext API keys. POSIX gets 0600 before the
        # replace so the final file is never world-readable.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
        if os.name == "posix":
            os.chmod(tmp, 0o600)
        tmp.replace(path)
        log.info(f"Settings saved to {path}")
    except Exception as e:
        log.warning(f"Failed to save settings to {path}: {e}")


def _platform_settings_path() -> Path:
    """Where dev runs kept settings before the unified in-project layout."""
    import platformdirs

    return Path(platformdirs.user_config_dir(APP_NAME, appauthor=False)) / "settings.json"


def _migrate_platform_settings() -> dict[str, Any] | None:
    """One-way copy of the pre-unification platformdirs settings (dev only).

    Before the unified data root, dev runs without portable mode kept
    settings in the platformdirs config dir. When the dev default (repo
    root) has no settings yet, prefer that file over the older legacy
    user_settings.json and copy it; the original stays in place as backup.
    """
    if SETTINGS_FILE != PROJECT_ROOT / "settings.json" or SETTINGS_FILE.exists():
        return None
    old_path = _platform_settings_path()
    if not old_path.exists():
        return None
    data = load_settings_from(old_path)
    if data is None:
        return None
    migrate_funasr_settings(data)
    save_settings_to(SETTINGS_FILE, data)
    log.info(f"Migrated platformdirs settings {old_path} -> {SETTINGS_FILE}")
    return data


def _migrate_legacy_settings() -> dict[str, Any] | None:
    """Copy a pre-rewrite user_settings.json into the canonical settings file.

    One-way copy: the legacy file stays in place as a backup, and further
    edits land in the new location only.
    """
    if SETTINGS_FILE.exists() or not LEGACY_SETTINGS_FILE.exists():
        return None
    data = load_settings_from(LEGACY_SETTINGS_FILE)
    if data is None:
        return None
    migrate_funasr_settings(data)
    save_settings_to(SETTINGS_FILE, data)
    log.info(f"Migrated legacy settings {LEGACY_SETTINGS_FILE} -> {SETTINGS_FILE}")
    return data


def load_user_settings() -> dict[str, Any] | None:
    """Load settings with the unified-layout and legacy migrations.

    Returns None when no settings exist (first run). Entry points use this
    instead of reaching into control_panel helpers.
    """
    migrated = _migrate_platform_settings()
    if migrated is not None:
        return migrated
    migrated = _migrate_legacy_settings()
    data = migrated if migrated is not None else load_settings_from(SETTINGS_FILE)
    if data is not None:
        if migrated is None:
            migrate_funasr_settings(data)
        log.info(f"Loaded saved settings from {SETTINGS_FILE}")
    return data


def save_user_settings(settings: dict[str, Any]) -> None:
    """Persist the user settings dict atomically to the canonical location."""
    save_settings_to(SETTINGS_FILE, settings)
