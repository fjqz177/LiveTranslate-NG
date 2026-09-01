"""Whisper-model download for the VAD/ASR tab (识别页 §3.2).

M-SPLIT (2026-08-31): extracted verbatim from ``vad_tab.VadTab`` into a
plain-object mixin. The concrete ``VadTab`` keeps the widget state; this
mixin holds only the whisper-model download / status / size-label methods.
Methods run on a concrete ``VadTab`` instance, so ``self.settings``
(TabBase property) and ``self.panel`` remain reachable.
"""

from pathlib import Path

from PyQt6.QtCore import Qt

from livetranslate.core.i18n import t
from livetranslate.modeling.manager import (
    format_size,
    list_local_faster_whisper_models,
    resolve_custom_whisper_model,
)
from livetranslate.modeling.registry import WHISPER_SIZES


class _WhisperDownloadMixin:
    """Whisper-model download orchestration, mixed into ``VadTab``."""

    def _populate_whisper_models(self, saved_value: str):
        self._whisper_size_combo.clear()
        for size in WHISPER_SIZES:
            self._whisper_size_combo.addItem(size, size)

        local_prefix = t("whisper_local_prefix")
        for item in list_local_faster_whisper_models():
            idx = self._whisper_size_combo.count()
            self._whisper_size_combo.addItem(f"{local_prefix}: {item['name']}", item["path"])
            self._whisper_size_combo.setItemData(idx, item["path"], Qt.ItemDataRole.ToolTipRole)

        selected = resolve_custom_whisper_model(saved_value) or saved_value
        idx = self._whisper_size_combo.findData(selected)
        if idx < 0:
            idx = self._whisper_size_combo.findText(saved_value)
        if idx < 0 and selected:
            label = f"{t('whisper_missing_local')}: {Path(str(selected)).name}"
            idx = self._whisper_size_combo.count()
            self._whisper_size_combo.addItem(label, selected)
            self._whisper_size_combo.setItemData(idx, str(selected), Qt.ItemDataRole.ToolTipRole)
        if idx >= 0:
            self._whisper_size_combo.setCurrentIndex(idx)

    def _set_whisper_status(self, text: str, kind: str) -> None:
        """Status text with a theme-aware color (see _chrome QSS rules)."""
        self._whisper_status.setText(text)
        self._whisper_status.setProperty("status", kind)
        style = self._whisper_status.style()
        style.unpolish(self._whisper_status)
        style.polish(self._whisper_status)

    def _update_whisper_size_label(self):
        from livetranslate.modeling.manager import is_asr_cached
        from livetranslate.modeling.registry import MODEL_SIZE_BYTES

        size = self._selected_whisper_model()
        cached = is_asr_cached("whisper", size, self.settings.get("hub", "ms"))
        if size not in WHISPER_SIZES:
            if cached:
                self._set_whisper_status(t("whisper_local_ready"), "ok")
            else:
                self._set_whisper_status(t("whisper_invalid_local"), "error")
            self._whisper_dl_btn.setEnabled(False)
            return
        if cached:
            self._set_whisper_status(t("whisper_already_cached"), "ok")
            self._whisper_dl_btn.setEnabled(False)
        else:
            est = MODEL_SIZE_BYTES.get(f"whisper-{size}", 0)
            self._set_whisper_status(f"~{format_size(est)}", "none")
            self._whisper_dl_btn.setEnabled(True)

    def _on_whisper_size_changed(self):
        self.settings["whisper_model_size"] = self._selected_whisper_model()
        self._update_whisper_size_label()
        # If already cached, switch engine immediately
        from livetranslate.modeling.manager import is_asr_cached

        size = self._selected_whisper_model()
        if is_asr_cached("whisper", size, self.settings.get("hub", "ms")):
            self.auto_save()

    def _download_whisper(self):
        from livetranslate.modeling.manager import get_missing_models, is_asr_cached

        size = self._selected_whisper_model()
        if size not in WHISPER_SIZES:
            return
        hub = self.settings.get("hub", "ms")
        if is_asr_cached("whisper", size, hub):
            return
        # Ordering (user report, 2026-09-01): don't download a model the
        # runtime can't load. Whisper is venv-backed, so a frozen build with no
        # installed variant must install the engine deps first.
        import sys as _sys

        from PyQt6.QtWidgets import QMessageBox

        from livetranslate.core import engine_runtime

        if (
            "whisper" in engine_runtime.VENV_BACKED_ENGINES
            and getattr(_sys, "frozen", False)
            and engine_runtime.active_variant() is None
        ):
            QMessageBox.warning(self, t("btn_download_whisper"), t("engine_runtime_needed"))
            return
        missing = get_missing_models("whisper", size, hub)
        missing = [m for m in missing if m["type"] != "silero-vad"]
        if not missing:
            return
        from livetranslate.ui.dialogs import ModelDownloadDialog

        dlg = ModelDownloadDialog(missing, hub=hub, parent=self.panel)
        if dlg.exec() == dlg.DialogCode.Accepted:
            self._update_whisper_size_label()
            # Switch to Whisper engine with the downloaded size
            self.auto_save()
