"""Cache tab: transcript auto-save toggle and model cache management."""

import logging
import threading

from PyQt6 import sip
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from livetranslate.core.i18n import t
from livetranslate.core.paths import transcripts_dir
from livetranslate.core.privacy import redact_text
from livetranslate.modeling.manager import (
    MODELS_DIR,
    dir_size,
    format_size,
    get_cache_entries,
)
from livetranslate.ui.panel._tab_base import TabBase, page_header

log = logging.getLogger("LiveTranslate.Panel")


class _ModelList(QListWidget):
    """Model cache list: clicking empty space clears the selection so the
    highlight never gets stuck on a stale entry."""

    def mousePressEvent(self, event):
        if self.itemAt(event.position().toPoint()) is None:
            self.clearSelection()
        super().mousePressEvent(event)


class CacheTab(TabBase):
    def __init__(self, panel):
        super().__init__(panel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(page_header(t("nav_data"), t("page_data_hint")))
        s = self.settings

        # Transcript auto-save group
        ts_group = QGroupBox(t("group_transcript"))
        ts_layout = QVBoxLayout(ts_group)
        ts_row1 = QHBoxLayout()
        self._auto_save_transcript_cb = QCheckBox(t("label_auto_save_transcript"))
        self._auto_save_transcript_cb.setToolTip(t("auto_save_transcript_tooltip"))
        self._auto_save_transcript_cb.setChecked(s.get("auto_save_transcript", True))
        self._auto_save_transcript_cb.toggled.connect(self.auto_save)
        ts_row1.addWidget(self._auto_save_transcript_cb, 1)
        ts_open_btn = QPushButton(t("btn_open_transcripts"))
        ts_open_btn.clicked.connect(self._open_transcripts_folder)
        ts_row1.addWidget(ts_open_btn)
        ts_layout.addLayout(ts_row1)
        # SEC-1: transcript content stays out of logs unless explicitly
        # opted in (privacy promise); toggling records the setting, the app
        # layer applies it to the pipeline at the next settings sync.
        self._log_transcript_cb = QCheckBox(t("label_log_transcript"))
        self._log_transcript_cb.setToolTip(t("log_transcript_tooltip"))
        self._log_transcript_cb.setChecked(s.get("log_transcript", False))
        self._log_transcript_cb.toggled.connect(self.auto_save)
        ts_layout.addWidget(self._log_transcript_cb)
        layout.addWidget(ts_group)

        top_row = QHBoxLayout()
        self._cache_total = QLabel("")
        self._cache_total.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        top_row.addWidget(self._cache_total, 1)
        open_btn = QPushButton(t("btn_open_folder"))
        open_btn.clicked.connect(self._open_models_folder)
        top_row.addWidget(open_btn)
        delete_all_btn = QPushButton(t("btn_delete_all_exit"))
        delete_all_btn.clicked.connect(self._delete_all_and_exit)
        top_row.addWidget(delete_all_btn)
        layout.addLayout(top_row)

        self._cache_list = _ModelList()
        self._cache_list.setFont(QFont("Consolas", 9))
        self._cache_list.setAlternatingRowColors(True)
        self._cache_list.setMinimumHeight(160)
        layout.addWidget(self._cache_list, 1)

        manage_row = QHBoxLayout()
        self._delete_selected_btn = QPushButton(t("btn_delete_selected"))
        self._delete_selected_btn.setEnabled(False)
        self._delete_selected_btn.clicked.connect(self._delete_selected)
        manage_row.addWidget(self._delete_selected_btn)
        refresh_btn = QPushButton(t("btn_refresh"))
        refresh_btn.clicked.connect(self.refresh)
        manage_row.addWidget(refresh_btn)
        hint = QLabel(t("cache_select_hint"))
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        manage_row.addWidget(hint, 1)
        layout.addLayout(manage_row)

        self._cache_entries = []
        self._cache_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.refresh()

    def collect(self):
        self.settings["auto_save_transcript"] = self._auto_save_transcript_cb.isChecked()
        self.settings["log_transcript"] = self._log_transcript_cb.isChecked()

    def _open_models_folder(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.open_path(MODELS_DIR)

    def _open_transcripts_folder(self):
        ts_dir = transcripts_dir()
        ts_dir.mkdir(parents=True, exist_ok=True)
        self.open_path(ts_dir)

    def refresh(self):
        self._cache_list.clear()
        self._cache_total.setText(t("scanning"))

        def _scan():
            entries = get_cache_entries()
            results = []
            for name, path in entries:
                size = dir_size(path)
                results.append((name, str(path), size))
            # The panel may be gone by the time the scan finishes (test
            # teardown / app shutdown); a destroyed-QObject emit raises.
            try:
                if not sip.isdeleted(self.panel):
                    self.panel._cache_result.emit(results)
            except RuntimeError:
                pass

        threading.Thread(target=_scan, daemon=True).start()

    def on_result(self, results):
        self._cache_list.clear()
        self._cache_entries = results
        total = 0
        for name, _path, size in results:
            total += size
            self._cache_list.addItem(f"{name}  —  {format_size(size)}")
        if not results:
            self._cache_list.addItem(t("no_cached_models"))
        self._cache_total.setText(
            t("cache_total").format(size=format_size(total), count=len(results))
        )

    def _on_selection_changed(self):
        self._delete_selected_btn.setEnabled(bool(self._cache_list.selectedItems()))

    def _delete_selected(self):
        row = self._cache_list.currentRow()
        if row < 0 or row >= len(self._cache_entries):
            return
        name, path, size = self._cache_entries[row]
        ret = QMessageBox.warning(
            self,
            t("delete_selected_confirm_title"),
            t("delete_selected_confirm_msg").format(name=name, size=format_size(size)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        import shutil

        try:
            shutil.rmtree(path)
            log.info(redact_text(f"Deleted: {path}"))
        except OSError as e:
            log.error(redact_text(f"Failed to delete {path}: {e}"))
            QMessageBox.warning(self, t("dialog_delete_title"), str(e))
            return
        self.refresh()

    def _delete_all_and_exit(self):
        if not self._cache_entries:
            return
        import shutil

        def _delete_one(path):
            try:
                shutil.rmtree(path)
                log.info(redact_text(f"Deleted: {path}"))
            except Exception as e:
                log.error(redact_text(f"Failed to delete {path}: {e}"))

        total_size = sum(s for _, _, s in self._cache_entries)
        ret = QMessageBox.warning(
            self,
            t("dialog_delete_title"),
            t("dialog_delete_msg").format(
                count=len(self._cache_entries), size=format_size(total_size)
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        for _name, path, _ in self._cache_entries:
            _delete_one(path)
        QApplication.instance().quit()
