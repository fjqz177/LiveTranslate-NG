"""Shared base for control panel tabs.

Each tab owns its widgets; shared state (settings dict, store, signals)
lives on the owning ControlPanel and is reached via ``self.panel``.

Also hosts the panel-wide layout helpers (plan §3.5):

- ``page_header``: per-page title + one-line hint so every page explains
  itself (the left nav only names the section).
- ``ScrollPage``: bounded scroll wrapper — pages whose content is taller
  than the window stay fully reachable, while the page's sizeHint stays
  fixed so the panel window never auto-grows past the screen.
"""

from pathlib import Path

from PyQt6.QtCore import QSize, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class TabBase(QWidget):
    """A settings tab with access to the owning ControlPanel."""

    def __init__(self, panel):
        super().__init__()
        self.panel = panel

    @property
    def settings(self) -> dict:
        """Panel-local working copy (draft); committed on save. The only
        authorized in-place mutable copy — the store dict is never handed out."""
        return self.panel._current_settings

    @property
    def config(self) -> dict:
        """The config.yaml-derived configuration dict."""
        return self.panel._config

    def auto_save(self):
        """Request the panel's debounced auto-save."""
        self.panel._auto_save()

    def store_save(self):
        """Persist immediately by committing the draft through the store."""
        self.panel.commit_now()

    def wrap_scroll(self, content: QWidget) -> None:
        """Give this tab a scrolling body (call once, after the content is
        built). The ScrollPage keeps the tab's sizeHint bounded so the
        panel window never auto-grows to fit the tallest page."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(make_scroll_page(content))

    def open_path(self, path: Path) -> None:
        """Open a file/directory with the OS default handler (plan §2:
        never os.startfile — the platform backend owns this)."""
        from livetranslate.platform.registry import create_system_integration

        try:
            create_system_integration().open_path(path)
        except Exception:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def page_header(title: str, hint: str) -> QWidget:
    """Page header block: section title plus a one-line description."""
    block = QWidget()
    layout = QVBoxLayout(block)
    layout.setContentsMargins(0, 0, 0, 4)
    layout.setSpacing(2)
    title_label = QLabel(title)
    title_label.setObjectName("pageTitle")
    hint_label = QLabel(hint)
    hint_label.setObjectName("pageHint")
    hint_label.setWordWrap(True)
    layout.addWidget(title_label)
    layout.addWidget(hint_label)
    return block


def make_scroll_page(content: QWidget, parent=None) -> QWidget:
    """Wrap ``content`` so it scrolls; the page sizeHint stays bounded."""
    return ScrollPage(content, parent=parent)


class ScrollPage(QWidget):
    """Bounded scroll wrapper (plan §3.5): content scrolls, sizeHint sane."""

    def __init__(self, content: QWidget, parent=None):
        super().__init__(parent)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # The viewport keeps its own palette background; leave it unfilled
        # so the themed surface shows through. NOTE: never set an inline
        # stylesheet here — a selector-less rule would match every widget
        # in the subtree (including popups and dialogs opened from it) and
        # paint them transparent/black.
        self._scroll.viewport().setAutoFillBackground(False)
        self._scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll)

    def sizeHint(self) -> QSize:
        """Keep the stacked page height bounded so the panel window never
        auto-grows to fit the tallest page (the pre-redesign behaviour
        pushed the window past 1080px)."""
        return QSize(760, 480)
