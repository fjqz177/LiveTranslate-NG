"""Panel chrome: the light/dark stylesheet + palette pairs (§3.5 外观).

One Chrome = one QSS string + the safety-net QPalette roles. The QSS wins
wherever it declares colors; the palette covers everything else (plain
labels, scroll viewports, browser panes). Both themes are held to the same
readability bar: body text >= 4.5:1 against its surface, secondary text
>= 4.5:1 on cards, accent used for selection/focus only.

Status-colored labels (VAD engine status) use dynamic `status` properties
with per-theme rules below, so they stay legible in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from livetranslate.ui.icons import asset_dirs

DARK = "dark"
LIGHT = "light"
THEME_MODES = (DARK, LIGHT)
DEFAULT_THEME = DARK

_DISABLED_TEXT_ROLES = (
    QPalette.ColorRole.WindowText,
    QPalette.ColorRole.Text,
    QPalette.ColorRole.ButtonText,
)

# Spinbox arrows ship as PNG assets: Qt stylesheets cannot draw CSS border
# triangles (a `width:0;height:0` + border trick renders as a solid block),
# and once up-button/down-button backgrounds are styled Qt stops drawing its
# default arrows, so an explicit image is required. Placeholders are resolved
# to absolute file URLs at apply time (see _spin_arrow_url).
_SPIN_UP_IMAGE = "{{SPIN_UP_IMAGE}}"
_SPIN_DOWN_IMAGE = "{{SPIN_DOWN_IMAGE}}"

# ---------------------------------------------------------------- dark ----

DARK_PALETTE_ROLES: dict[QPalette.ColorRole, QColor] = {
    QPalette.ColorRole.Window: QColor("#0E1116"),
    QPalette.ColorRole.WindowText: QColor("#F2F4F7"),
    QPalette.ColorRole.Base: QColor("#10141B"),
    QPalette.ColorRole.AlternateBase: QColor("#12171E"),
    QPalette.ColorRole.Text: QColor("#F2F4F7"),
    QPalette.ColorRole.Button: QColor("#161B22"),
    QPalette.ColorRole.ButtonText: QColor("#F2F4F7"),
    QPalette.ColorRole.BrightText: QColor("#FFFFFF"),
    QPalette.ColorRole.Highlight: QColor("#4C8DFF"),
    QPalette.ColorRole.HighlightedText: QColor("#FFFFFF"),
    QPalette.ColorRole.ToolTipBase: QColor("#161B22"),
    QPalette.ColorRole.ToolTipText: QColor("#F2F4F7"),
    QPalette.ColorRole.PlaceholderText: QColor("#9AA3B2"),
    QPalette.ColorRole.Link: QColor("#4C8DFF"),
    QPalette.ColorRole.Light: QColor("#2E3947"),
    QPalette.ColorRole.Midlight: QColor("#1C232E"),
    QPalette.ColorRole.Mid: QColor("#232A35"),
    QPalette.ColorRole.Dark: QColor("#10141B"),
    QPalette.ColorRole.Shadow: QColor("#000000"),
    QPalette.ColorRole.LinkVisited: QColor("#6BA3FF"),
}

DARK_QSS = """
ControlPanel {
    background-color: #0E1116;
}
QStackedWidget#panelPages {
    background: #0E1116;
}
QLabel {
    color: #E6EDF3;
}
QLabel#navBrand {
    color: #F2F4F7;
    font-size: 15px;
    font-weight: 700;
    padding: 2px 20px 0 20px;
}
QLabel#navCaption {
    color: #9AA3B2;
    font-size: 11px;
    padding: 2px 20px 12px 20px;
}
QListWidget#panelNav {
    background: transparent;
    border: none;
    outline: none;
    font-size: 13px;
}
QListWidget#panelNav::item {
    color: #8B949E;
    padding: 10px 16px;
    border-radius: 6px;
    margin: 2px 12px;
}
QListWidget#panelNav::item:hover {
    background: #161B22;
    color: #E6EDF3;
}
QListWidget#panelNav::item:selected {
    background: #1B2635;
    color: #F2F4F7;
}
QLabel#pageTitle {
    color: #F2F4F7;
    font-size: 14pt;
    font-weight: 600;
}
QLabel#pageHint {
    color: #9AA3B2;
    font-size: 9pt;
}
QLabel#hintLabel {
    color: #9AA3B2;
    font-size: 10pt;
}
QScrollArea {
    background: transparent;
    border: none;
}
QGroupBox {
    background: #10141B;
    border: 1px solid #232A35;
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    color: #C9D1D9;
}
QPushButton {
    background: #161B22;
    border: 1px solid #232A35;
    border-radius: 6px;
    padding: 6px 14px;
    color: #F2F4F7;
}
QPushButton:hover {
    background: #1C232E;
}
QPushButton:pressed {
    background: #10141B;
}
QPushButton:focus {
    border-color: #4C8DFF;
}
QPushButton:disabled {
    color: #57606A;
    background: #10141B;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #161B22;
    border: 1px solid #232A35;
    border-radius: 6px;
    padding: 4px 8px;
    color: #F2F4F7;
    selection-background-color: #4C8DFF;
}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border-color: #4C8DFF;
}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {
    color: #57606A;
    background: #10141B;
    border-color: #1C232E;
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
QComboBox QAbstractItemView {
    background: #161B22;
    border: 1px solid #232A35;
    color: #F2F4F7;
    selection-background-color: #4C8DFF;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: #1C232E;
    border: none;
    width: 18px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #232A35;
}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background: #4C8DFF;
}
QSpinBox::up-button:disabled, QDoubleSpinBox::up-button:disabled,
QSpinBox::down-button:disabled, QDoubleSpinBox::down-button:disabled {
    background: #10141B;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: {{SPIN_UP_IMAGE}};
    width: 8px;
    height: 6px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: {{SPIN_DOWN_IMAGE}};
    width: 8px;
    height: 6px;
}
QCheckBox, QRadioButton {
    spacing: 8px;
    color: #E6EDF3;
}
QListWidget {
    background: #10141B;
    border: 1px solid #232A35;
    border-radius: 6px;
    color: #E6EDF3;
    outline: none;
}
QListWidget::item {
    padding: 4px 6px;
}
QListWidget::item:selected {
    background: #1B2635;
    color: #F2F4F7;
}
QTextEdit {
    background: #10141B;
    border: 1px solid #232A35;
    border-radius: 6px;
    color: #E6EDF3;
    padding: 4px 6px;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #232A35;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #4C8DFF;
}
QSlider::handle:horizontal:hover {
    background: #6BA3FF;
}
QToolTip {
    background-color: #161B22;
    color: #F2F4F7;
    border: 1px solid #2E3947;
    padding: 4px 8px;
}
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #232A35;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #2E3947;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #232A35;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QMenu {
    background-color: #161B22;
    color: #F2F4F7;
    border: 1px solid #2E3947;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 20px;
    border-radius: 4px;
}
QMenu::item:selected {
    background: #1B2635;
}
QMenu::item:disabled {
    color: #57606A;
}
QMenu::separator {
    height: 1px;
    background: #232A35;
    margin: 4px 8px;
}
QDialog {
    background-color: #0E1116;
}
QMessageBox {
    background-color: #0E1116;
}
QColorDialog {
    background-color: #0E1116;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #2E3947;
    border-radius: 3px;
    background: #161B22;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #4C8DFF;
}
QCheckBox::indicator:checked {
    background-color: #4C8DFF;
    border-color: #4C8DFF;
}
QRadioButton::indicator {
    border-radius: 7px;
}
QRadioButton::indicator:checked {
    background-color: #4C8DFF;
    border-color: #4C8DFF;
}
QLabel[status="ok"] { color: #3FB950; font-size: 11px; }
QLabel[status="warn"] { color: #DBAB09; font-size: 11px; }
QLabel[status="error"] { color: #F85149; font-size: 11px; }
QLabel[status="none"] { color: #9AA3B2; font-size: 11px; }
"""

# ---------------------------------------------------------------- light ---

LIGHT_PALETTE_ROLES: dict[QPalette.ColorRole, QColor] = {
    QPalette.ColorRole.Window: QColor("#F6F8FA"),
    QPalette.ColorRole.WindowText: QColor("#1F2328"),
    QPalette.ColorRole.Base: QColor("#FFFFFF"),
    QPalette.ColorRole.AlternateBase: QColor("#F0F3F6"),
    QPalette.ColorRole.Text: QColor("#1F2328"),
    QPalette.ColorRole.Button: QColor("#FFFFFF"),
    QPalette.ColorRole.ButtonText: QColor("#1F2328"),
    QPalette.ColorRole.BrightText: QColor("#FFFFFF"),
    QPalette.ColorRole.Highlight: QColor("#4C8DFF"),
    QPalette.ColorRole.HighlightedText: QColor("#FFFFFF"),
    QPalette.ColorRole.ToolTipBase: QColor("#FFFFFF"),
    QPalette.ColorRole.ToolTipText: QColor("#1F2328"),
    QPalette.ColorRole.PlaceholderText: QColor("#6E7781"),
    QPalette.ColorRole.Link: QColor("#0969DA"),
    QPalette.ColorRole.Light: QColor("#FFFFFF"),
    QPalette.ColorRole.Midlight: QColor("#F0F3F6"),
    QPalette.ColorRole.Mid: QColor("#D1D9E0"),
    QPalette.ColorRole.Dark: QColor("#A8B4C0"),
    QPalette.ColorRole.Shadow: QColor("#000000"),
    QPalette.ColorRole.LinkVisited: QColor("#8250DF"),
}

LIGHT_QSS = """
ControlPanel {
    background-color: #F6F8FA;
}
QStackedWidget#panelPages {
    background: #F6F8FA;
}
QLabel {
    color: #1F2328;
}
QLabel#navBrand {
    color: #1F2328;
    font-size: 15px;
    font-weight: 700;
    padding: 2px 20px 0 20px;
}
QLabel#navCaption {
    color: #57606A;
    font-size: 11px;
    padding: 2px 20px 12px 20px;
}
QListWidget#panelNav {
    background: transparent;
    border: none;
    outline: none;
    font-size: 13px;
}
QListWidget#panelNav::item {
    color: #57606A;
    padding: 10px 16px;
    border-radius: 6px;
    margin: 2px 12px;
}
QListWidget#panelNav::item:hover {
    background: #EBEFF3;
    color: #1F2328;
}
QListWidget#panelNav::item:selected {
    background: #D9E6FF;
    color: #0D2C66;
}
QLabel#pageTitle {
    color: #1F2328;
    font-size: 14pt;
    font-weight: 600;
}
QLabel#pageHint {
    color: #57606A;
    font-size: 9pt;
}
QLabel#hintLabel {
    color: #57606A;
    font-size: 10pt;
}
QScrollArea {
    background: transparent;
    border: none;
}
QGroupBox {
    background: #FFFFFF;
    border: 1px solid #D1D9E0;
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    color: #1F2328;
}
QPushButton {
    background: #FFFFFF;
    border: 1px solid #D1D9E0;
    border-radius: 6px;
    padding: 6px 14px;
    color: #1F2328;
}
QPushButton:hover {
    background: #F0F3F6;
}
QPushButton:pressed {
    background: #E5EAF0;
}
QPushButton:focus {
    border-color: #4C8DFF;
}
QPushButton:disabled {
    color: #8C959F;
    background: #F0F3F6;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #FFFFFF;
    border: 1px solid #D1D9E0;
    border-radius: 6px;
    padding: 4px 8px;
    color: #1F2328;
    selection-background-color: #4C8DFF;
}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border-color: #4C8DFF;
}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {
    color: #8C959F;
    background: #F0F3F6;
    border-color: #D8DFE6;
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
QComboBox QAbstractItemView {
    background: #FFFFFF;
    border: 1px solid #D1D9E0;
    color: #1F2328;
    selection-background-color: #4C8DFF;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: #F0F3F6;
    border: none;
    width: 18px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #E5EAF0;
}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background: #4C8DFF;
}
QSpinBox::up-button:disabled, QDoubleSpinBox::up-button:disabled,
QSpinBox::down-button:disabled, QDoubleSpinBox::down-button:disabled {
    background: #F0F3F6;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: {{SPIN_UP_IMAGE}};
    width: 8px;
    height: 6px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: {{SPIN_DOWN_IMAGE}};
    width: 8px;
    height: 6px;
}
QCheckBox, QRadioButton {
    spacing: 8px;
    color: #1F2328;
}
QListWidget {
    background: #FFFFFF;
    border: 1px solid #D1D9E0;
    border-radius: 6px;
    color: #1F2328;
    outline: none;
}
QListWidget::item {
    padding: 4px 6px;
}
QListWidget::item:selected {
    background: #D9E6FF;
    color: #0D2C66;
}
QTextEdit {
    background: #FFFFFF;
    border: 1px solid #D1D9E0;
    border-radius: 6px;
    color: #1F2328;
    padding: 4px 6px;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #D1D9E0;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #4C8DFF;
}
QSlider::handle:horizontal:hover {
    background: #2F6FE0;
}
QToolTip {
    background-color: #FFFFFF;
    color: #1F2328;
    border: 1px solid #D1D9E0;
    padding: 4px 8px;
}
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #C6CFD8;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #A8B4C0;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #C6CFD8;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QMenu {
    background-color: #FFFFFF;
    color: #1F2328;
    border: 1px solid #D1D9E0;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 20px;
    border-radius: 4px;
}
QMenu::item:selected {
    background: #D9E6FF;
}
QMenu::item:disabled {
    color: #8C959F;
}
QMenu::separator {
    height: 1px;
    background: #D1D9E0;
    margin: 4px 8px;
}
QDialog {
    background-color: #F6F8FA;
}
QMessageBox {
    background-color: #F6F8FA;
}
QColorDialog {
    background-color: #F6F8FA;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #D1D9E0;
    border-radius: 3px;
    background: #FFFFFF;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #4C8DFF;
}
QCheckBox::indicator:checked {
    background-color: #4C8DFF;
    border-color: #4C8DFF;
}
QRadioButton::indicator {
    border-radius: 7px;
}
QRadioButton::indicator:checked {
    background-color: #4C8DFF;
    border-color: #4C8DFF;
}
QLabel[status="ok"] { color: #1A7F37; font-size: 11px; }
QLabel[status="warn"] { color: #9A6700; font-size: 11px; }
QLabel[status="error"] { color: #CF222E; font-size: 11px; }
QLabel[status="none"] { color: #57606A; font-size: 11px; }
"""


@dataclass(frozen=True)
class Chrome:
    """One panel theme: QSS + palette roles + disabled text color."""

    qss: str
    palette_roles: dict[QPalette.ColorRole, QColor]
    disabled_text: QColor


CHROMES: dict[str, Chrome] = {
    DARK: Chrome(DARK_QSS, DARK_PALETTE_ROLES, QColor("#57606A")),
    LIGHT: Chrome(LIGHT_QSS, LIGHT_PALETTE_ROLES, QColor("#8C959F")),
}


def chrome(mode: str) -> Chrome:
    """Theme lookup with a safe fallback to the default."""
    return CHROMES.get(mode, CHROMES[DEFAULT_THEME])


def build_palette(ch: Chrome) -> QPalette:
    """Safety-net palette for every widget the QSS does not style.

    Every role is filled in every color group so unstyled surfaces
    (dialog chrome, popup frames, disabled text, inactive windows)
    resolve theme colors instead of leaking the platform light
    defaults into dark mode.
    """
    palette = QPalette()
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        for role, color in ch.palette_roles.items():
            palette.setColor(group, role, color)
    for role, color in ch.palette_roles.items():
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            role,
            ch.disabled_text if role in _DISABLED_TEXT_ROLES else color,
        )
    return palette


def _spin_arrow_url(mode: str, up: bool) -> str:
    """Absolute file URL for a spinbox arrow asset.

    Assets live under assets/icons/spin_{up,down}_{theme}.png in the repo
    checkout and in the frozen _MEIPASS bundle (packaging/livetranslate.spec
    ships the whole icons dir). The URL must be absolute — QSS resolves
    relative urls against the current working directory, which is not stable
    in a frozen build. Falls back to an empty image (arrows simply hidden)
    if the asset is missing, which is strictly better than a broken square.
    """
    filename = f"spin_{'up' if up else 'down'}_{mode}.png"
    for directory in asset_dirs():
        path = directory / filename
        if path.exists():
            return f'url("{path.as_posix()}")'
    return 'url("")'


def apply_app_theme(mode: str) -> None:
    """Apply a chrome at the application level.

    Top-level windows, popups and dialogs (QComboBox dropdowns,
    QColorDialog, QMenu, QMessageBox, the tray diagnostics window)
    resolve their palette from the application palette, so the theme
    must live there — a widget-level stylesheet never reaches them.
    """
    ch = chrome(mode)
    app = QApplication.instance()
    if app is None:
        return
    app.setPalette(build_palette(ch))
    qss = ch.qss.replace(_SPIN_UP_IMAGE, _spin_arrow_url(mode, up=True)).replace(
        _SPIN_DOWN_IMAGE, _spin_arrow_url(mode, up=False)
    )
    app.setStyleSheet(qss)
