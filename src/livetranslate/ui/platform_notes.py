"""Platform degradation notes (§3.7 detect-degrade-guide).

Returns the i18n keys of the live capability degradations so the
settings page can show the same guidance as the diagnostics cards
(no silent failures — every degraded capability has an explanation).
"""

from __future__ import annotations

from PyQt6.QtWidgets import QSystemTrayIcon


def platform_notes() -> list[str]:
    """Detected degradations as i18n keys (empty list = all nominal)."""
    notes: list[str] = []
    if not QSystemTrayIcon.isSystemTrayAvailable():
        notes.append("err_gnome_tray")
    return notes
