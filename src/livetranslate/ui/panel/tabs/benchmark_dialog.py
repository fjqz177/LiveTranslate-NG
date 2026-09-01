"""Benchmark tool window: opened from the recognition page (plan §3.2
item 3 — the benchmark left the tab bar and became an independent tool).
"""

from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout

from livetranslate.core.i18n import t
from livetranslate.ui.panel.tabs.benchmark_tab import BenchmarkTab


class BenchmarkDialog(QDialog):
    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("benchmark_dialog_title"))
        self.resize(680, 480)

        layout = QVBoxLayout(self)
        self._tab = BenchmarkTab(panel)
        layout.addWidget(self._tab, 1)
        panel._bench_result.connect(self._tab.on_result)

        row = QHBoxLayout()
        row.addStretch(1)
        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.accept)
        row.addWidget(close_btn)
        layout.addLayout(row)
