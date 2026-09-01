"""Light/dark chrome toggle tests (设置 → 常规 → 外观).

The chrome is applied at the application level (popups, menus and dialogs
resolve palettes from the app palette), so the assertions check the shared
QApplication state rather than the panel's own stylesheet.
"""

import pytest
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QColorDialog, QMenu, QMessageBox

from livetranslate.ui.panel.panel import ControlPanel

CONFIG = {
    "translation": {
        "model": "gpt-test",
        "api_base": "https://example.com/v1",
        "api_key": "sk-test",
        "target_language": "zh",
    },
    "asr": {
        "vad_threshold": 0.5,
        "min_speech_duration": 0.3,
        "max_speech_duration": 20.0,
        "language": "auto",
        "device": "cpu",
    },
}


@pytest.fixture
def panel(qapp, tmp_path, monkeypatch):
    assert qapp is not None
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "settings.json")
    p = ControlPanel(CONFIG)
    yield p
    p.close()


def test_default_theme_is_dark(panel, qapp):
    assert panel._theme_mode == "dark"
    assert panel.get_settings()["theme"] == "dark"
    assert "#0E1116" in qapp.styleSheet()
    assert qapp.palette().color(QPalette.ColorRole.Window).name() == "#0e1116"


def test_light_radio_switches_chrome_and_persists(panel, qapp):
    panel._general_tab._theme_light.setChecked(True)
    assert panel._theme_mode == "light"
    assert panel.get_settings()["theme"] == "light"
    assert "#F6F8FA" in qapp.styleSheet()
    assert qapp.palette().color(QPalette.ColorRole.Window).name() == "#f6f8fa"
    # and back
    panel._general_tab._theme_dark.setChecked(True)
    assert panel._theme_mode == "dark"
    assert "#0E1116" in qapp.styleSheet()
    assert qapp.palette().color(QPalette.ColorRole.Window).name() == "#0e1116"


def test_saved_light_theme_applied_on_construction(qapp, tmp_path, monkeypatch):
    assert qapp is not None
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "s.json")
    p = ControlPanel(CONFIG, saved_settings={"theme": "light"})
    assert p._theme_mode == "light"
    assert "#F6F8FA" in qapp.styleSheet()
    assert p._general_tab._theme_light.isChecked()
    p.close()


def test_invalid_saved_theme_falls_back_to_dark(qapp, tmp_path, monkeypatch):
    assert qapp is not None
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "s.json")
    p = ControlPanel(CONFIG, saved_settings={"theme": "neon"})
    assert p._theme_mode == "dark"
    assert p._general_tab._theme_dark.isChecked()
    p.close()


def test_popups_and_dialogs_resolve_theme_palette(panel):
    """Regression: QColorDialog/QMenu/QMessageBox resolve their palette
    from the application palette — a panel-only stylesheet left them on
    the platform light defaults in dark mode."""
    for mode, window_color, text_color in (
        ("dark", "#0e1116", "#f2f4f7"),
        ("light", "#f6f8fa", "#1f2328"),
    ):
        panel.set_theme_mode(mode)
        for make in (
            lambda: QColorDialog(panel),
            lambda: QMenu(panel),
            lambda: QMessageBox(panel),
        ):
            w = make()
            assert w.palette().color(QPalette.ColorRole.Window).name() == window_color
            assert w.palette().color(QPalette.ColorRole.WindowText).name() == text_color
            w.deleteLater()


def test_popups_from_scroll_subtree_follow_theme(qapp, tmp_path, monkeypatch):
    """Regression: selector-less inline stylesheets on scroll viewports
    used to match every widget in the subtree — including QColorDialog and
    combo popups opened from it — painting them black in light mode. Such
    popups must render with the themed surface."""
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QColorDialog

    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "s.json")
    p = ControlPanel(CONFIG)
    p.resize(920, 660)
    p.show()
    qapp.processEvents()
    p._general_tab._theme_light.setChecked(True)
    qapp.processEvents()
    try:
        cd = QColorDialog(QColor("#ff8800"), p._style_tab)
        cd.show()
        qapp.processEvents()
        img = cd.grab().toImage()
        c = img.pixelColor(8, 8)
        assert c.lightness() > 60, f"color dialog rendered dark in light mode: {c.name()}"
        cd.close()
        # combo popups from the same subtree render the themed surface
        combo = p._style_tab._style_preset
        combo.showPopup()
        qapp.processEvents()
        view = combo.view()
        if view.isVisible():
            img2 = view.window().grab().toImage()
            c2 = img2.pixelColor(img2.width() // 2, img2.height() // 2)
            assert c2.lightness() > 60, f"combo popup rendered dark: {c2.name()}"
        combo.hidePopup()
    finally:
        p.close()


def test_safety_net_palette_fills_disabled_and_inactive_groups(qapp, tmp_path, monkeypatch):
    """Unstyled surfaces (disabled text, inactive windows) must resolve
    theme colors instead of leaking the platform light defaults."""
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "s.json")
    p = ControlPanel(CONFIG)
    try:
        pal = qapp.palette()
        assert pal.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text).name() == "#57606a"
        assert (
            pal.color(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Window).name() == "#0e1116"
        )
    finally:
        p.close()
