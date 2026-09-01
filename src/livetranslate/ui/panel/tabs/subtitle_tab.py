"""Subtitle tab: wraps the SubtitleSettingsWidget for OBS capture styling."""

from PyQt6.QtWidgets import QVBoxLayout

from livetranslate.ui.panel._tab_base import TabBase
from livetranslate.ui.subtitle_settings import SubtitleSettingsWidget


class SubtitleTab(TabBase):
    def __init__(self, panel):
        super().__init__(panel)
        subtitle_settings = self.settings.get("subtitle_mode") or {}
        self._widget = SubtitleSettingsWidget(subtitle_settings)
        self._widget.settings_changed.connect(self._on_subtitle_settings_changed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._widget)

    def _on_subtitle_settings_changed(self, s):
        self.settings["subtitle_mode"] = s
        self.auto_save()
        self.panel.subtitle_settings_changed.emit(s)

    def update_settings(self, s):
        self._widget.update_settings(s)

    def sync_click_through(self, checked: bool):
        """Mirror the tray's click-through toggle into the widget."""
        w = self._widget
        w._click_through_check.blockSignals(True)
        w._click_through_check.setChecked(checked)
        w._click_through_check.blockSignals(False)
        w._settings["click_through"] = checked
