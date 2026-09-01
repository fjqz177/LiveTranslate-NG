"""Application icon loading (plan §3.5.7).

Pre-generated assets under assets/icons/ win when present (committed in
the repo; installers bundle them in Phase 7). The QPainter fallback keeps
source runs working even before assets are generated. Tray variants carry
a status dot: run (green) / pause (amber) / error (red).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap

_STATUSES = ("run", "pause", "error")


def asset_dirs() -> list[Path]:
    """Candidate asset roots: frozen bundle first, repo checkout fallback.

    Shared with ui/panel/_chrome.py (spinbox arrow images) so both resolve
    assets the same way in dev and frozen builds.
    """
    here = Path(__file__).resolve()
    dirs = [here.parents[3] / "assets" / "icons"]
    if getattr(sys, "frozen", False):  # PyInstaller bundle (Phase 7)
        dirs.insert(0, Path(sys._MEIPASS) / "assets" / "icons")
    return dirs


def _drawn_icon() -> QIcon:
    """Programmatic fallback: rounded blue square with an "LT" monogram."""
    pix = QPixmap(64, 64)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(60, 130, 240))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 12, 12)
    p.setPen(QColor(255, 255, 255))
    p.setFont(QFont("Consolas", 28, QFont.Weight.Bold))
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "LT")
    p.end()
    return QIcon(pix)


def create_app_icon(status: str | None = None) -> QIcon:
    """App icon; status selects a tray variant (run/pause/error)."""
    filename = f"tray_{status}_64.png" if status in _STATUSES else "app.png"
    for directory in asset_dirs():
        path = directory / filename
        if path.exists():
            return QIcon(str(path))
    return _drawn_icon()
