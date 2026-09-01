"""Qt signal wrapper around the pure settings persistence layer.

SettingsStore owns the runtime settings dict and persists it atomically
(tmp + replace in livetranslate.core.settings). It never hands out its
internal dict: reads return copies and writes replace-merge a fresh dict
onto ``_settings``. Callers that want to edit settings take a copy, mutate
their copy and commit atomically via ``update()`` / ``save()``.

File I/O helpers live in livetranslate.core.settings (no Qt) so non-UI
callers can load/save settings without importing PyQt6.
"""

from pathlib import Path

from PyQt6.QtCore import QObject

from livetranslate.core.settings import load_settings_from, save_settings_to
from livetranslate.modeling.registry import migrate_funasr_settings


class SettingsStore(QObject):
    """Owns the user settings dict and its persistence.

    Reads return copies; writes replace-merge and atomic-save the result.
    The internal dict is never exposed (no ``data`` property), so callers
    cannot mutate committed state in place.
    """

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self._path = Path(path)
        self._settings: dict = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def load(self) -> dict:
        """Load from disk once; returns a copy of the settings."""
        if not self._loaded:
            self._settings = migrate_funasr_settings(load_settings_from(self._path)) or {}
            self._loaded = True
        return dict(self._settings)

    def seed(self, data: dict | None) -> dict:
        """Seed from an already-loaded dict (skips disk I/O)."""
        self._settings = migrate_funasr_settings(data) or {}
        self._loaded = True
        return dict(self._settings)

    def snapshot(self) -> dict:
        return dict(self._settings)

    def get(self, key, default=None):
        return self._settings.get(key, default)

    def update(self, items: dict, save: bool = True) -> bool:
        """Replace-merge several keys; returns True if anything changed.

        Unlike an in-place mutation, the underlying dict is rebound to a
        fresh ``merged`` dict, so a previously held snapshot/copy is never
        affected by a subsequent write. Persists atomically when asked.
        """
        merged = {**self._settings, **items}
        changed = merged != self._settings
        if changed:
            self._settings = merged
            if save:
                save_settings_to(self._path, self._settings)
        return changed

    def save(self):
        save_settings_to(self._path, self._settings)
