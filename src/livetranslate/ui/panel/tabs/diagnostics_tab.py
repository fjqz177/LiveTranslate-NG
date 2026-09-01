"""Diagnostics page (诊断): lazy-embeds the shared DiagnosticsView.

The view needs a reference to the running composition root, which only
exists after app.py wires the panel, so the page builds itself on first
visit instead of at panel construction time.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QVBoxLayout

from livetranslate.core.i18n import t
from livetranslate.ui.panel._tab_base import TabBase, page_header


class DiagnosticsTab(TabBase):
    def __init__(self, panel):
        super().__init__(panel)
        self._view = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 4, 0, 0)
        self._layout.setSpacing(12)
        self._layout.addWidget(page_header(t("nav_diagnostics"), t("page_diagnostics_hint")))
        self._hint = QLabel(t("diag_lazy_hint"))
        self._hint.setWordWrap(True)
        self._layout.addWidget(self._hint)

    @property
    def built(self) -> bool:
        return self._view is not None

    def ensure_built(self, app_ref) -> None:
        """Build the live view on first visit (no-op without an app ref)."""
        if self._view is not None or app_ref is None:
            return
        from livetranslate.ui.diagnostics import DiagnosticsView

        self._hint.hide()
        self._view = DiagnosticsView(app_ref, parent=self)
        self._layout.addWidget(self._view, 1)
