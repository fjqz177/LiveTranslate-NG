"""Benchmark tab: translation latency benchmark controls and output."""

import contextlib

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from livetranslate.core.benchmark import run_benchmark
from livetranslate.core.i18n import t
from livetranslate.ui.panel._tab_base import TabBase


class BenchmarkTab(TabBase):
    def __init__(self, panel):
        super().__init__(panel)
        layout = QVBoxLayout(self)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel(t("label_source")))
        self._bench_lang = QComboBox()
        self._bench_lang.addItems(["ja", "en", "zh", "ko", "fr", "de"])
        self._bench_lang.setCurrentIndex(0)
        ctrl_row.addWidget(self._bench_lang)
        ctrl_row.addWidget(QLabel(t("target_label")))
        self._bench_target = QComboBox()
        self._bench_target.addItems(["zh", "en", "ja", "ko", "fr", "de", "es", "ru"])
        ctrl_row.addWidget(self._bench_target)
        ctrl_row.addStretch()
        self._bench_btn = QPushButton(t("btn_test_all"))
        self._bench_btn.clicked.connect(self._run_benchmark)
        ctrl_row.addWidget(self._bench_btn)
        layout.addLayout(ctrl_row)

        self._bench_output = QTextEdit()
        self._bench_output.setReadOnly(True)
        self._bench_output.setFont(QFont("Consolas", 9))
        self._bench_output.setStyleSheet(
            "background: #1e1e2e; color: #cdd6f4; border: 1px solid #444;"
        )
        layout.addWidget(self._bench_output)

    def _run_benchmark(self):
        models = self.settings.get("models", [])
        if not models:
            return

        source_lang = self._bench_lang.currentText()
        target_lang = self._bench_target.currentText()
        timeout_s = self.settings.get("timeout", 5)

        self._bench_btn.setEnabled(False)
        self._bench_btn.setText(t("testing"))
        self._bench_output.clear()

        from livetranslate.core.i18n import LANGUAGE_DISPLAY
        from livetranslate.core.translator import DEFAULT_PROMPT

        src = LANGUAGE_DISPLAY.get(source_lang, source_lang)
        tgt = LANGUAGE_DISPLAY.get(target_lang, target_lang)
        prompt = self.settings.get("system_prompt", DEFAULT_PROMPT)
        with contextlib.suppress(KeyError, IndexError):
            prompt = prompt.format(source_lang=src, target_lang=tgt)

        run_benchmark(
            models,
            source_lang,
            target_lang,
            timeout_s,
            prompt,
            self.panel._bench_result.emit,
        )

    def on_result(self, text: str):
        if text == "__DONE__":
            self._bench_btn.setEnabled(True)
            self._bench_btn.setText(t("btn_test_all"))
        else:
            self._bench_output.append(text)
