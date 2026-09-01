"""Tests for the settings panel's left-nav information architecture
(plan §3.2: seven pages, 200px nav, lazy diagnostics page).
"""

import types

import pytest

from livetranslate.core.i18n import t
from livetranslate.ui.panel.panel import ControlPanel
from livetranslate.ui.panel.tabs.general_tab import GeneralTab

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


def test_seven_pages_in_plan_order(panel):
    assert panel._nav.count() == 7
    assert panel._stack.count() == 7
    assert panel._nav.width() == 200
    expected = [
        t("nav_general"),
        t("nav_translation"),
        t("nav_recognition"),
        t("nav_subtitles"),
        t("nav_data"),
        t("nav_diagnostics"),
        t("nav_about"),
    ]
    labels = [panel._nav.item(i).text() for i in range(panel._nav.count())]
    assert labels == expected
    # First page is the general tab; nav starts on it
    assert panel._stack.currentWidget() is panel._general_tab
    assert isinstance(panel._general_tab, GeneralTab)


def test_benchmark_lives_off_the_recognition_page(panel):
    assert panel._bench_result  # signal still owned by the panel
    # The benchmark dialog is opened on demand; the page only carries a button
    import PyQt6.QtWidgets as w

    page = panel._recognition_page
    buttons = page.findChildren(w.QPushButton)
    assert any(b.text() == t("btn_open_benchmark") for b in buttons)


def test_nav_switches_pages(panel):
    """Regression: clicking the left nav must switch the stacked page.
    (The pre-redesign panel never wired the nav to the stack and stayed
    frozen on page 0.)"""
    for row, page in [
        (0, panel._general_tab),
        (1, panel._translation_tab),
        (2, panel._recognition_page),
        (3, panel._subtitle_page),
        (4, panel._cache_tab),
        (5, panel._diagnostics_tab),
        (6, panel._about_tab),
    ]:
        panel._nav.setCurrentRow(row)
        assert panel._stack.currentWidget() is page
        assert panel._stack.currentIndex() == row


def test_recognition_page_scrolls(panel):
    """Regression: the recognition page carries hotkeys + VAD together, so
    its body must scroll instead of clipping controls off-window."""
    from PyQt6.QtWidgets import QScrollArea

    assert panel._recognition_page.findChild(QScrollArea) is not None


def test_diagnostics_page_builds_lazily_on_attach(panel):
    assert panel._diagnostics_tab.built is False
    panel.attach_app(types.SimpleNamespace())
    assert panel._diagnostics_tab.built is True
    assert panel._diagnostics_tab._view is not None


def test_diagnostics_view_fills_the_page(panel, qapp):
    """Regression: a leftover placeholder stretch pushed the diagnostics
    view into the bottom half of the page — the view must fill everything
    below the header block."""
    panel.resize(920, 660)
    panel._nav.setCurrentRow(panel._diagnostics_index)
    panel.attach_app(types.SimpleNamespace())
    panel.show()
    qapp.processEvents()
    tab = panel._diagnostics_tab
    view = tab._view
    assert view is not None
    header = tab._layout.itemAt(0).widget()
    # starts right below the header (+ 12px layout spacing, small slack)
    assert view.y() <= header.y() + header.height() + 30
    # and reaches the bottom of the page
    assert view.y() + view.height() >= tab.height() - 2
    panel.hide()


def test_diagnostics_page_stays_unbuilt_without_app(panel):
    panel._nav.setCurrentRow(panel._diagnostics_index)
    assert panel._diagnostics_tab.built is False


def test_apply_settings_collects_general_state(panel):
    panel._general_tab._reduce_motion.setChecked(True)
    panel.apply_settings()
    assert panel.get_settings()["reduce_motion"] is True


def test_content_column_capped_and_centered(panel):
    """§3.5.4: the page stack is capped at 960px and centered."""
    assert panel._stack.objectName() == "panelPages"
    assert panel._stack.maximumWidth() == 960


def test_style_tab_shows_resolved_fonts(panel):
    """§3.5.3: each font picker shows the family Qt actually resolved."""
    from PyQt6.QtWidgets import QLabel

    hints = [
        label
        for label in panel._style_tab.findChildren(QLabel)
        if label.objectName() == "hintLabel"
    ]
    assert len(hints) == 2
    assert all(h.text() for h in hints)
