"""General tab (常规): UI language, startup behaviour, reduced motion,
window layout reset and settings import/export (plan §3.2 item 1).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from livetranslate.core.i18n import get_lang, resolve_ui_lang, set_lang, t
from livetranslate.platform.registry import create_system_integration
from livetranslate.platform.window import prefers_reduced_motion
from livetranslate.ui.panel._chrome import DEFAULT_THEME
from livetranslate.ui.panel._tab_base import TabBase, page_header

log = logging.getLogger("LiveTranslate.Panel")


class GeneralTab(TabBase):
    def __init__(self, panel):
        super().__init__(panel)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(page_header(t("nav_general"), t("page_general_hint")))
        s = self.settings

        # -- UI language --
        lang_group = QGroupBox(t("group_ui_language"))
        lang_layout = QVBoxLayout(lang_group)
        self._ui_lang = QComboBox()
        self._ui_lang.addItem(t("lang_system"), "system")
        self._ui_lang.addItem(t("lang_zh"), "zh")
        self._ui_lang.addItem(t("lang_en"), "en")
        stored = s.get("ui_lang")
        if stored not in ("system", "zh", "en"):
            stored = "system"
        idx = self._ui_lang.findData(stored)
        self._ui_lang.setCurrentIndex(max(idx, 0))
        self._ui_lang.currentIndexChanged.connect(self._on_ui_lang_changed)
        lang_layout.addWidget(self._ui_lang)
        hint = QLabel(t("ui_lang_restart_hint"))
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        lang_layout.addWidget(hint)
        layout.addWidget(lang_group)

        # -- Appearance (light/dark chrome) --
        appearance_group = QGroupBox(t("group_appearance"))
        appearance_layout = QHBoxLayout(appearance_group)
        self._theme_dark = QRadioButton(t("theme_dark"))
        self._theme_light = QRadioButton(t("theme_light"))
        self._theme_dark.toggled.connect(lambda checked: checked and self._on_theme_changed("dark"))
        self._theme_light.toggled.connect(
            lambda checked: checked and self._on_theme_changed("light")
        )
        stored_theme = self.settings.get("theme", DEFAULT_THEME)
        if stored_theme == "light":
            self._theme_light.setChecked(True)
        else:
            self._theme_dark.setChecked(True)
        appearance_layout.addWidget(self._theme_dark)
        appearance_layout.addWidget(self._theme_light)
        appearance_layout.addStretch(1)
        layout.addWidget(appearance_group)

        # -- Startup --
        startup_group = QGroupBox(t("group_startup"))
        startup_layout = QVBoxLayout(startup_group)
        self._autostart = QCheckBox(t("label_autostart"))
        self._system = None
        try:
            self._system = create_system_integration()
            self._autostart.setChecked(self._system.autostart_enabled())
        except Exception as e:
            log.warning(f"Autostart control unavailable: {e}")
            self._system = None
            self._autostart.setEnabled(False)
            self._autostart.setToolTip(t("autostart_unavailable"))
        self._autostart.toggled.connect(self._on_autostart_toggled)
        startup_layout.addWidget(self._autostart)
        self._start_hidden = QCheckBox(t("label_start_hidden"))
        self._start_hidden.setChecked(bool(s.get("start_hidden", False)))
        self._start_hidden.toggled.connect(self.auto_save)
        startup_layout.addWidget(self._start_hidden)
        layout.addWidget(startup_group)

        # -- Motion --
        motion_group = QGroupBox(t("group_motion"))
        motion_layout = QVBoxLayout(motion_group)
        self._reduce_motion = QCheckBox(t("label_reduce_motion"))
        motion = s.get("reduce_motion")
        if motion is None:
            motion = prefers_reduced_motion()  # §3.8: follow the OS preference
        self._reduce_motion.setChecked(bool(motion))
        self._reduce_motion.toggled.connect(self.auto_save)
        motion_layout.addWidget(self._reduce_motion)
        layout.addWidget(motion_group)

        # -- Window layout --
        layout_group = QGroupBox(t("group_window_layout"))
        layout_row = QHBoxLayout(layout_group)
        reset_btn = QPushButton(t("btn_reset_positions"))
        reset_btn.clicked.connect(self.panel.reset_positions.emit)
        layout_row.addWidget(reset_btn)
        layout_row.addStretch(1)
        layout.addWidget(layout_group)

        # -- Settings import/export --
        io_group = QGroupBox(t("group_settings_io"))
        io_row = QHBoxLayout(io_group)
        export_btn = QPushButton(t("btn_export_settings"))
        export_btn.clicked.connect(self._export_settings)
        io_row.addWidget(export_btn)
        import_btn = QPushButton(t("btn_import_settings"))
        import_btn.clicked.connect(self._import_settings)
        io_row.addWidget(import_btn)
        io_row.addStretch(1)
        layout.addWidget(io_group)

        # §3.7: live capability degradations, same copy as the diagnostics
        # cards — never fail silently.
        from livetranslate.ui.platform_notes import platform_notes

        notes = platform_notes()
        if notes:
            notes_group = QGroupBox(t("platform_notes_title"))
            notes_layout = QVBoxLayout(notes_group)
            for key in notes:
                note_label = QLabel("• " + t(key))
                note_label.setWordWrap(True)
                note_label.setObjectName("hintLabel")
                notes_layout.addWidget(note_label)
            layout.addWidget(notes_group)

        layout.addStretch(1)
        self.wrap_scroll(content)

    def collect(self):
        self.settings["ui_lang"] = self._ui_lang.currentData()
        self.settings["start_hidden"] = self._start_hidden.isChecked()
        self.settings["reduce_motion"] = self._reduce_motion.isChecked()
        self.settings["theme"] = "light" if self._theme_light.isChecked() else "dark"

    # -- handlers ------------------------------------------------------------

    def _on_theme_changed(self, mode: str):
        self.panel.set_theme_mode(mode)

    def _on_ui_lang_changed(self, _index):
        code = self._ui_lang.currentData()
        resolved = resolve_ui_lang(code)
        # Live-swap so dialogs opened later follow the choice; existing
        # widgets keep their labels until the next launch (hint says so).
        if resolved != get_lang():
            set_lang(resolved)
        self.auto_save()

    def _on_autostart_toggled(self, checked):
        if self._system is None:
            return
        try:
            self._system.set_autostart(bool(checked))
        except Exception as e:
            log.warning(f"Failed to set autostart={checked}: {e}")
            QMessageBox.warning(self, t("autostart_failed_title"), t("autostart_failed_msg"))
            self._autostart.blockSignals(True)
            self._autostart.setChecked(not checked)
            self._autostart.blockSignals(False)

    def _export_settings(self):
        ret = QMessageBox.warning(
            self,
            t("export_warn_title"),
            t("export_warn_msg"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, t("btn_export_settings"), "livetranslate-settings.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            data = self.panel._store.snapshot()
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            QMessageBox.information(
                self, t("export_done_title"), t("export_done_msg").format(path=path)
            )
        except OSError as e:
            QMessageBox.warning(self, t("export_failed_title"), str(e))

    def _import_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, t("btn_import_settings"), "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, t("import_failed_title"), t("import_invalid").format(error=e))
            return
        ret = QMessageBox.warning(
            self,
            t("import_confirm_title"),
            t("import_confirm_msg"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        self.panel._store.seed(data)
        self.panel._store.save()
        # Re-sync the panel draft so a later commit cannot clobber the
        # imported settings with a stale draft (draft/commit split).
        self.panel._current_settings = self.panel._store.snapshot()
        QMessageBox.information(self, t("import_confirm_title"), t("import_done_msg"))
