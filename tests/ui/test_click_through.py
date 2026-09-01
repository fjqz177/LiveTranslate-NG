"""Click-through toggle regression tests (platform/window.py).

Regression: QWidget.setWindowFlag() hides visible top-level windows, so
toggling click-through made the overlay/subtitle windows disappear and
the mouse could never get back in — the helpers must keep the window
visible and stable across the polling the overlay/subtitle windows run.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

import livetranslate.platform.window as winutil


def _tool_window() -> QWidget:
    w = QWidget()
    w.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
    w.resize(200, 100)
    return w


def test_set_click_through_keeps_window_visible(qapp):
    w = _tool_window()
    w.show()
    qapp.processEvents()
    assert w.isVisible()
    winutil.set_click_through(w, "all")
    qapp.processEvents()
    assert w.isVisible(), "set_click_through must not hide the window"
    assert winutil.is_click_through(w)


def test_clear_click_through_keeps_window_visible(qapp):
    w = _tool_window()
    w.show()
    qapp.processEvents()
    winutil.set_click_through(w, "all")
    qapp.processEvents()
    winutil.clear_click_through(w)
    qapp.processEvents()
    assert w.isVisible(), "clear_click_through must not hide the window"
    assert not winutil.is_click_through(w)


def test_repeated_toggle_keeps_window_stable(qapp):
    """The overlay polls the toggle every 50ms; alternating calls must
    never hide or flake the window."""
    w = _tool_window()
    w.show()
    qapp.processEvents()
    for i in range(6):
        if i % 2 == 0:
            winutil.set_click_through(w, "all")
        else:
            winutil.clear_click_through(w)
        qapp.processEvents()
        assert w.isVisible(), f"window hidden at poll #{i}"
    assert not winutil.is_click_through(w)
