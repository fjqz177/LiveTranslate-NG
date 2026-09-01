"""Windows window tricks used by the overlay and subtitle window.

Qt-first: WindowTransparentForInput / WindowStaysOnTopHint cover the
simple cases. The Win32 native path exists only for the granularity Qt
lacks — header-interactive click-through (WS_EX_TRANSPARENT toggled by
region) — and is fenced to Windows.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Literal

from PyQt6.QtCore import Qt

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

ClickRegion = Literal["all", "header-interactive"]

_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x20

# SetWindowPos flags: NOMOVE | NOSIZE | NOZORDER | NOACTIVATE | FRAMECHANGED
_SWP_REFRESH = 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020


def _win32_style(win: QWidget) -> int:
    return int(ctypes.windll.user32.GetWindowLongW(int(win.winId()), _GWL_EXSTYLE))


def _win32_set_style(win: QWidget, style: int) -> None:
    ctypes.windll.user32.SetWindowLongW(int(win.winId()), _GWL_EXSTYLE, style)


def _win32_refresh(win: QWidget) -> None:
    """Tell the system to re-evaluate the window after an ex-style change."""
    ctypes.windll.user32.SetWindowPos(int(win.winId()), 0, 0, 0, 0, 0, _SWP_REFRESH)


def set_click_through(win: QWidget, region: ClickRegion = "all") -> None:
    """Make the window click-through.

    "all" is pure Qt everywhere. "header-interactive" keeps the window
    clickable only via Win32 ex-style toggling.

    On Windows the toggle is pure Win32: calling setWindowFlag() on a
    visible top-level window hides it (QWidget::setWindowFlags reshows
    nothing), which made the overlay/subtitle windows disappear the moment
    click-through was turned on. The Win32 bit also survives Qt show/raise
    (callers re-assert it on their timers if Qt ever rebuilds the style).
    """
    if region not in ("all", "header-interactive"):
        region = "all"
    _win32_set_style(win, _win32_style(win) | _WS_EX_TRANSPARENT)
    _win32_refresh(win)


def clear_click_through(win: QWidget) -> None:
    """Restore normal input handling."""
    style = _win32_style(win)
    if style & _WS_EX_TRANSPARENT:
        _win32_set_style(win, style & ~_WS_EX_TRANSPARENT)
        _win32_refresh(win)


def is_click_through(win: QWidget) -> bool:
    if win.windowFlags() & Qt.WindowType.WindowTransparentForInput:
        return True
    return bool(_win32_style(win) & _WS_EX_TRANSPARENT)


def set_topmost(win: QWidget, enabled: bool) -> None:
    """Toggle always-on-top (Qt native)."""
    flags = win.windowFlags()
    if enabled:
        flags |= Qt.WindowType.WindowStaysOnTopHint
    else:
        flags &= ~Qt.WindowType.WindowStaysOnTopHint
    win.setWindowFlags(flags)
    win.show()


def prefers_reduced_motion() -> bool:
    """Read the OS-wide reduced-motion preference (plan §3.8).

    Windows: SPI_GETCLIENTAREAANIMATION. The in-app toggle is authoritative;
    no other platform branch remains.
    """
    try:
        animated = ctypes.c_int()
        ok = ctypes.windll.user32.SystemParametersInfoW(
            0x1042,  # SPI_GETCLIENTAREAANIMATION
            0,
            ctypes.byref(animated),
            0,
        )
        return bool(ok) and not animated.value
    except Exception:
        return False
