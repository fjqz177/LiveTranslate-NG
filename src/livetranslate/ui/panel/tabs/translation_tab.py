"""Translation tab: model configs, system prompt and network timeout."""

import logging

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from livetranslate.core.i18n import t
from livetranslate.core.translator import DEFAULT_PROMPT, PROMPT_PRESETS
from livetranslate.ui.dialogs import ModelEditDialog
from livetranslate.ui.panel._tab_base import TabBase, page_header

log = logging.getLogger("LiveTranslate.Panel")


class TranslationTab(TabBase):
    def __init__(self, panel):
        super().__init__(panel)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(page_header(t("nav_translation"), t("page_translation_hint")))
        s = self.settings

        models_group = QGroupBox(t("group_model_configs"))
        models_layout = QVBoxLayout(models_group)

        self._model_list = QListWidget()
        self._model_list.setFont(QFont("Consolas", 9))
        self._model_list.itemDoubleClicked.connect(self._on_model_double_clicked)
        self.refresh_model_list()
        models_layout.addWidget(self._model_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton(t("btn_add"))
        add_btn.clicked.connect(self._add_model)
        btn_row.addWidget(add_btn)
        edit_btn = QPushButton(t("btn_edit"))
        edit_btn.clicked.connect(self._edit_model)
        btn_row.addWidget(edit_btn)
        dup_btn = QPushButton(t("btn_duplicate"))
        dup_btn.clicked.connect(self._dup_model)
        btn_row.addWidget(dup_btn)
        del_btn = QPushButton(t("btn_remove"))
        del_btn.clicked.connect(self._remove_model)
        btn_row.addWidget(del_btn)
        models_layout.addLayout(btn_row)
        layout.addWidget(models_group)

        prompt_group = QGroupBox(t("group_system_prompt"))
        prompt_layout = QVBoxLayout(prompt_group)

        # Preset selector
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel(t("label_prompt_preset")))
        self._prompt_preset = QComboBox()
        self._prompt_preset.addItem(t("prompt_daily"), "daily")
        self._prompt_preset.addItem(t("prompt_esports"), "esports")
        self._prompt_preset.addItem(t("prompt_anime"), "anime")
        self._prompt_preset.addItem(t("prompt_webid"), "webid")
        self._prompt_preset.addItem(t("prompt_custom"), "custom")

        current_prompt = s.get("system_prompt", DEFAULT_PROMPT)
        preset_idx = 4  # default to custom
        for i, key in enumerate(["daily", "esports", "anime", "webid"]):
            if current_prompt.strip() == PROMPT_PRESETS[key].strip():
                preset_idx = i
                break
        if current_prompt.strip() == DEFAULT_PROMPT.strip():
            preset_idx = 0
        self._prompt_preset.setCurrentIndex(preset_idx)
        self._prompt_preset.currentIndexChanged.connect(self._on_prompt_preset_changed)
        preset_row.addWidget(self._prompt_preset, 1)
        prompt_layout.addLayout(preset_row)

        # Prompt text editor
        self._prompt_edit = QTextEdit()
        self._prompt_edit.setFont(QFont("Consolas", 9))
        self._prompt_edit.setMaximumHeight(100)
        self._prompt_edit.setPlainText(current_prompt)
        self._prompt_debounce = QTimer()
        self._prompt_debounce.setSingleShot(True)
        self._prompt_debounce.setInterval(600)
        self._prompt_debounce.timeout.connect(self._apply_prompt)
        self._prompt_edit.textChanged.connect(self._prompt_debounce.start)
        prompt_layout.addWidget(self._prompt_edit)
        layout.addWidget(prompt_group)

        net_group = QGroupBox(t("group_network"))
        net_layout = QVBoxLayout(net_group)
        net_row = QHBoxLayout()
        net_row.addWidget(QLabel(t("label_timeout")))
        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(1, 60)
        self._timeout_spin.setValue(s.get("timeout", 5))
        self._timeout_spin.setSuffix(" s")
        self._timeout_spin.valueChanged.connect(lambda v: self.settings.update({"timeout": v}))
        self._timeout_spin.valueChanged.connect(self.auto_save)
        net_row.addWidget(self._timeout_spin)
        net_row.addStretch()
        net_layout.addLayout(net_row)
        layout.addWidget(net_group)

        layout.addStretch()
        self.wrap_scroll(content)

    def collect(self):
        """Write this tab's widget state into the shared settings dict."""
        prompt_text = self._prompt_edit.toPlainText().strip()
        if prompt_text:
            self.settings["system_prompt"] = prompt_text
        self.settings["timeout"] = self._timeout_spin.value()

    # ── Model management ──

    def refresh_model_list(self):
        self._model_list.clear()
        active = self.settings.get("active_model", 0)
        for i, m in enumerate(self.settings.get("models", [])):
            prefix = ">>> " if i == active else "    "
            proxy = m.get("proxy", "none")
            proxy_tag = f"  [proxy: {proxy}]" if proxy != "none" else ""
            text = f"{prefix}{m['name']}{proxy_tag}\n     {m['api_base']}  |  {m['model']}"
            item = QListWidgetItem(text)
            if i == active:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._model_list.addItem(item)

    def _emit_models_list_changed(self):
        models = self.settings.get("models", [])
        active_idx = self.settings.get("active_model", 0)
        self.panel.models_list_changed.emit(models, active_idx)

    def _add_model(self):
        dlg = ModelEditDialog(self.panel)
        if dlg.exec():
            data = dlg.get_data()
            if data["name"] and data["model"]:
                self.settings["models"] = [*self.settings.get("models", []), data]
                self.refresh_model_list()
                self.store_save()
                self._emit_models_list_changed()

    def _edit_model(self):
        row = self._model_list.currentRow()
        models = self.settings.get("models", [])
        if row < 0 or row >= len(models):
            return
        dlg = ModelEditDialog(self.panel, models[row])
        if dlg.exec():
            data = dlg.get_data()
            if data["name"] and data["model"]:
                self.settings["models"] = [data if i == row else m for i, m in enumerate(models)]
                self.refresh_model_list()
                self.store_save()
                self._emit_models_list_changed()
                # Re-apply if editing the active model
                active = self.settings.get("active_model", 0)
                if row == active:
                    self.panel.model_changed.emit(data)

    def _dup_model(self):
        row = self._model_list.currentRow()
        models = self.settings.get("models", [])
        if row < 0 or row >= len(models):
            return
        dup = dict(models[row])
        dup["name"] = dup["name"] + " (copy)"
        self.settings["models"] = [*models, dup]
        self.refresh_model_list()
        self.store_save()
        self._emit_models_list_changed()

    def _remove_model(self):
        row = self._model_list.currentRow()
        models = self.settings.get("models", [])
        if row < 0 or row >= len(models) or len(models) <= 1:
            return
        self.settings["models"] = models[:row] + models[row + 1 :]
        active = self.settings.get("active_model", 0)
        if active >= len(self.settings["models"]):
            self.settings["active_model"] = len(self.settings["models"]) - 1
        self.refresh_model_list()
        self._model_list.setCurrentRow(min(row, len(self.settings["models"]) - 1))
        self.store_save()
        self._emit_models_list_changed()

    def _on_model_double_clicked(self, item):
        row = self._model_list.row(item)
        models = self.settings.get("models", [])
        if 0 <= row < len(models):
            self._model_list.setCurrentRow(row)
            self._edit_model()

    # ── Prompt handling ──

    def _on_prompt_preset_changed(self, index):
        key = self._prompt_preset.itemData(index)
        if key == "custom":
            return
        prompt = PROMPT_PRESETS.get(key, DEFAULT_PROMPT)
        self._prompt_edit.setPlainText(prompt)
        self._apply_prompt()

    def _apply_prompt(self):
        text = self._prompt_edit.toPlainText().strip()
        if text:
            self.settings["system_prompt"] = text
            active = self.panel.get_active_model()
            if active:
                self.panel.model_changed.emit(active)
            self.store_save()
            log.info("System prompt updated")
            # Update preset combo to reflect current state
            self._prompt_preset.blockSignals(True)
            matched = 4  # custom
            for i, key in enumerate(["daily", "esports", "anime", "webid"]):
                if text.strip() == PROMPT_PRESETS[key].strip():
                    matched = i
                    break
            self._prompt_preset.setCurrentIndex(matched)
            self._prompt_preset.blockSignals(False)
