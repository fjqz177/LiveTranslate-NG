"""Style tab: overlay style presets, colors, fonts and window opacity."""

from PyQt6.QtGui import QFont, QFontInfo
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFontComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from livetranslate.core.i18n import t
from livetranslate.core.theme import DEFAULT_STYLE, PRESET_NAMES, STYLE_PRESETS
from livetranslate.ui.panel._tab_base import TabBase


class StyleTab(TabBase):
    def __init__(self, panel):
        super().__init__(panel)
        layout = QVBoxLayout(self)
        s = self.settings.get("style", dict(DEFAULT_STYLE))

        # Preset group
        preset_group = QGroupBox(t("group_preset"))
        preset_layout = QHBoxLayout(preset_group)
        self._style_preset = QComboBox()
        # Preset list is derived from the theme registry (single source of
        # truth); the display label comes from a per-preset i18n key.
        preset_names = [(k, t(f"preset_{k}")) for k in PRESET_NAMES] + [
            ("custom", t("preset_custom"))
        ]
        self._preset_keys = [k for k, _ in preset_names]
        for _, label in preset_names:
            self._style_preset.addItem(label)
        current_preset = s.get("preset", "default")
        if current_preset in self._preset_keys:
            self._style_preset.setCurrentIndex(self._preset_keys.index(current_preset))
        else:
            self._style_preset.setCurrentIndex(len(self._preset_keys) - 1)  # custom
        self._style_preset.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self._style_preset, 1)
        reset_btn = QPushButton(t("btn_reset_style"))
        reset_btn.clicked.connect(self._reset_style)
        preset_layout.addWidget(reset_btn)
        layout.addWidget(preset_group)

        # Background group
        bg_group = QGroupBox(t("group_background"))
        bg_layout = QGridLayout(bg_group)
        bg_layout.setColumnStretch(0, 1)
        bg_layout.setColumnMinimumWidth(1, 180)

        bg_layout.addWidget(QLabel(t("label_bg_color")), 0, 0)
        self._bg_color_btn = self._make_color_btn(s.get("bg_color", DEFAULT_STYLE["bg_color"]))
        self._bg_color_btn.clicked.connect(lambda: self._pick_color(self._bg_color_btn))
        bg_layout.addWidget(self._bg_color_btn, 0, 1)

        bg_layout.addWidget(QLabel(t("label_bg_opacity")), 1, 0)
        self._bg_opacity = QSpinBox()
        self._bg_opacity.setRange(0, 100)
        self._bg_opacity.setSuffix("%")
        self._bg_opacity.setValue(
            round(s.get("bg_opacity", DEFAULT_STYLE["bg_opacity"]) / 255 * 100)
        )
        self._bg_opacity.valueChanged.connect(self._on_style_value_changed)
        self._bg_opacity.valueChanged.connect(self.auto_save)
        bg_layout.addWidget(self._bg_opacity, 1, 1)

        bg_layout.addWidget(QLabel(t("label_header_color")), 2, 0)
        self._header_color_btn = self._make_color_btn(
            s.get("header_color", DEFAULT_STYLE["header_color"])
        )
        self._header_color_btn.clicked.connect(lambda: self._pick_color(self._header_color_btn))
        bg_layout.addWidget(self._header_color_btn, 2, 1)

        bg_layout.addWidget(QLabel(t("label_header_opacity")), 3, 0)
        self._header_opacity = QSpinBox()
        self._header_opacity.setRange(0, 100)
        self._header_opacity.setSuffix("%")
        self._header_opacity.setValue(
            round(s.get("header_opacity", DEFAULT_STYLE["header_opacity"]) / 255 * 100)
        )
        self._header_opacity.valueChanged.connect(self._on_style_value_changed)
        self._header_opacity.valueChanged.connect(self.auto_save)
        bg_layout.addWidget(self._header_opacity, 3, 1)

        bg_layout.addWidget(QLabel(t("label_border_radius")), 4, 0)
        self._border_radius = QSpinBox()
        self._border_radius.setRange(0, 30)
        self._border_radius.setValue(s.get("border_radius", DEFAULT_STYLE["border_radius"]))
        self._border_radius.setSuffix(" px")
        self._border_radius.valueChanged.connect(self._on_style_value_changed)
        self._border_radius.valueChanged.connect(self.auto_save)
        bg_layout.addWidget(self._border_radius, 4, 1)

        layout.addWidget(bg_group)

        # Text group
        text_group = QGroupBox(t("group_text"))
        text_layout = QGridLayout(text_group)
        text_layout.setColumnStretch(0, 1)
        text_layout.setColumnMinimumWidth(1, 180)

        text_layout.addWidget(QLabel(t("label_original_font")), 0, 0)
        self._orig_font_combo = QFontComboBox()
        self._orig_font_combo.setCurrentFont(
            QFont(s.get("original_font_family", DEFAULT_STYLE["original_font_family"]))
        )
        self._orig_font_combo.currentFontChanged.connect(self._on_style_value_changed)
        self._orig_font_combo.currentFontChanged.connect(self.auto_save)
        text_layout.addWidget(self._orig_font_combo, 0, 1)
        # §3.5.3: show the family Qt actually resolved through the
        # platform fallback chain, not just the requested name.
        text_layout.addWidget(self._make_resolved_font_label(self._orig_font_combo), 1, 1)

        text_layout.addWidget(QLabel(t("label_original_font_size")), 2, 0)
        self._orig_font_size = QSpinBox()
        self._orig_font_size.setRange(6, 24)
        self._orig_font_size.setValue(
            s.get("original_font_size", DEFAULT_STYLE["original_font_size"])
        )
        self._orig_font_size.setSuffix(" pt")
        self._orig_font_size.valueChanged.connect(self._on_style_value_changed)
        self._orig_font_size.valueChanged.connect(self.auto_save)
        text_layout.addWidget(self._orig_font_size, 2, 1)

        text_layout.addWidget(QLabel(t("label_original_color")), 3, 0)
        self._orig_color_btn = self._make_color_btn(
            s.get("original_color", DEFAULT_STYLE["original_color"])
        )
        self._orig_color_btn.clicked.connect(lambda: self._pick_color(self._orig_color_btn))
        text_layout.addWidget(self._orig_color_btn, 3, 1)

        text_layout.addWidget(QLabel(t("label_translation_font")), 4, 0)
        self._trans_font_combo = QFontComboBox()
        self._trans_font_combo.setCurrentFont(
            QFont(s.get("translation_font_family", DEFAULT_STYLE["translation_font_family"]))
        )
        self._trans_font_combo.currentFontChanged.connect(self._on_style_value_changed)
        self._trans_font_combo.currentFontChanged.connect(self.auto_save)
        text_layout.addWidget(self._trans_font_combo, 4, 1)
        text_layout.addWidget(self._make_resolved_font_label(self._trans_font_combo), 5, 1)

        text_layout.addWidget(QLabel(t("label_translation_font_size")), 6, 0)
        self._trans_font_size = QSpinBox()
        self._trans_font_size.setRange(6, 24)
        self._trans_font_size.setValue(
            s.get("translation_font_size", DEFAULT_STYLE["translation_font_size"])
        )
        self._trans_font_size.setSuffix(" pt")
        self._trans_font_size.valueChanged.connect(self._on_style_value_changed)
        self._trans_font_size.valueChanged.connect(self.auto_save)
        text_layout.addWidget(self._trans_font_size, 6, 1)

        text_layout.addWidget(QLabel(t("label_translation_color")), 7, 0)
        self._trans_color_btn = self._make_color_btn(
            s.get("translation_color", DEFAULT_STYLE["translation_color"])
        )
        self._trans_color_btn.clicked.connect(lambda: self._pick_color(self._trans_color_btn))
        text_layout.addWidget(self._trans_color_btn, 7, 1)

        text_layout.addWidget(QLabel(t("label_timestamp_color")), 8, 0)
        self._ts_color_btn = self._make_color_btn(
            s.get("timestamp_color", DEFAULT_STYLE["timestamp_color"])
        )
        self._ts_color_btn.clicked.connect(lambda: self._pick_color(self._ts_color_btn))
        text_layout.addWidget(self._ts_color_btn, 8, 1)

        layout.addWidget(text_group)

        # Window group
        win_group = QGroupBox(t("group_window"))
        win_layout = QGridLayout(win_group)
        win_layout.setColumnStretch(0, 1)
        win_layout.setColumnMinimumWidth(1, 180)
        win_layout.addWidget(QLabel(t("label_window_opacity")), 0, 0)
        self._window_opacity = QSpinBox()
        self._window_opacity.setRange(30, 100)
        self._window_opacity.setSuffix("%")
        self._window_opacity.setValue(s.get("window_opacity", DEFAULT_STYLE["window_opacity"]))
        self._window_opacity.valueChanged.connect(self._on_style_value_changed)
        self._window_opacity.valueChanged.connect(self.auto_save)
        win_layout.addWidget(self._window_opacity, 0, 1)
        layout.addWidget(win_group)

        layout.addStretch()

    def collect(self):
        self.settings["style"] = self._collect_style()

    def _make_resolved_font_label(self, combo: QFontComboBox) -> QLabel:
        """Secondary label showing the font Qt actually resolved (§3.5.3)."""
        label = QLabel()
        label.setObjectName("hintLabel")

        def _update(font: QFont) -> None:
            info = QFontInfo(font)
            label.setText(t("label_resolved_font").format(family=info.family()))

        combo.currentFontChanged.connect(_update)
        _update(combo.currentFont())
        return label

    def _make_color_btn(self, color: str) -> QPushButton:
        btn = QPushButton()
        btn.setFixedSize(60, 24)
        btn.setProperty("hex_color", color)
        btn.setStyleSheet(f"background-color: {color}; border: 1px solid #888; border-radius: 3px;")
        return btn

    def _pick_color(self, btn: QPushButton):
        from PyQt6.QtGui import QColor as _QColor

        current = _QColor(btn.property("hex_color"))
        color = QColorDialog.getColor(current, self)
        if color.isValid():
            hex_c = color.name()
            btn.setProperty("hex_color", hex_c)
            btn.setStyleSheet(
                f"background-color: {hex_c}; border: 1px solid #888; border-radius: 3px;"
            )
            self._on_style_value_changed()
            self.auto_save()

    def _collect_style(self) -> dict:
        return {
            "preset": self._preset_keys[self._style_preset.currentIndex()],
            "bg_color": self._bg_color_btn.property("hex_color"),
            "bg_opacity": round(self._bg_opacity.value() / 100 * 255),
            "header_color": self._header_color_btn.property("hex_color"),
            "header_opacity": round(self._header_opacity.value() / 100 * 255),
            "border_radius": self._border_radius.value(),
            "original_font_family": self._orig_font_combo.currentFont().family(),
            "translation_font_family": self._trans_font_combo.currentFont().family(),
            "original_font_size": self._orig_font_size.value(),
            "translation_font_size": self._trans_font_size.value(),
            "original_color": self._orig_color_btn.property("hex_color"),
            "translation_color": self._trans_color_btn.property("hex_color"),
            "timestamp_color": self._ts_color_btn.property("hex_color"),
            "window_opacity": self._window_opacity.value(),
        }

    def _apply_style_to_controls(self, s: dict):
        """Update all style controls to match a style dict, without triggering auto-save."""
        self._bg_color_btn.setProperty("hex_color", s["bg_color"])
        self._bg_color_btn.setStyleSheet(
            f"background-color: {s['bg_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._bg_opacity.setValue(round(s["bg_opacity"] / 255 * 100))
        self._header_color_btn.setProperty("hex_color", s["header_color"])
        self._header_color_btn.setStyleSheet(
            f"background-color: {s['header_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._header_opacity.setValue(round(s["header_opacity"] / 255 * 100))
        self._border_radius.setValue(s["border_radius"])
        self._orig_font_combo.setCurrentFont(QFont(s["original_font_family"]))
        self._trans_font_combo.setCurrentFont(QFont(s["translation_font_family"]))
        self._orig_font_size.setValue(s["original_font_size"])
        self._trans_font_size.setValue(s["translation_font_size"])
        self._orig_color_btn.setProperty("hex_color", s["original_color"])
        self._orig_color_btn.setStyleSheet(
            f"background-color: {s['original_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._trans_color_btn.setProperty("hex_color", s["translation_color"])
        self._trans_color_btn.setStyleSheet(
            f"background-color: {s['translation_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._ts_color_btn.setProperty("hex_color", s["timestamp_color"])
        self._ts_color_btn.setStyleSheet(
            f"background-color: {s['timestamp_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._window_opacity.setValue(s["window_opacity"])

    def _on_preset_changed(self, index):
        key = self._preset_keys[index]
        if key == "custom":
            return
        preset = STYLE_PRESETS.get(key)
        if not preset:
            return
        self._block_style_signals(True)
        self._apply_style_to_controls(preset)
        self._block_style_signals(False)
        self.auto_save()

    def _on_style_value_changed(self, *_args):
        """When any style control changes manually, switch preset to Custom."""
        custom_idx = len(self._preset_keys) - 1
        if self._style_preset.currentIndex() != custom_idx:
            self._style_preset.blockSignals(True)
            self._style_preset.setCurrentIndex(custom_idx)
            self._style_preset.blockSignals(False)
        self.auto_save()

    def _reset_style(self):
        self._style_preset.blockSignals(True)
        self._style_preset.setCurrentIndex(0)  # default
        self._style_preset.blockSignals(False)
        self._block_style_signals(True)
        self._apply_style_to_controls(DEFAULT_STYLE)
        self._block_style_signals(False)
        self.auto_save()

    def _block_style_signals(self, block: bool):
        for w in (
            self._bg_opacity,
            self._header_opacity,
            self._border_radius,
            self._orig_font_combo,
            self._trans_font_combo,
            self._orig_font_size,
            self._trans_font_size,
            self._window_opacity,
        ):
            w.blockSignals(block)
