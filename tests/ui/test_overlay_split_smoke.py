"""M-SPLIT smoke: overlay.py is a re-export shim over the leaf modules.

Guards the split contract (zero public-symbol change):
- every public symbol keeps its identity when imported through the shim
  vs. its leaf home module (widgets) or core.theme (theme constants);
- SubtitleOverlay still assembles end-to-end from the leaf widgets.
"""

from livetranslate.core import theme
from livetranslate.ui import overlay
from livetranslate.ui.overlay_banner import ErrorBanner
from livetranslate.ui.overlay_controls import DragHandle
from livetranslate.ui.overlay_message import ChatMessage
from livetranslate.ui.overlay_monitor import MonitorBar
from livetranslate.ui.overlay_window import SubtitleOverlay

# Widget symbols are defined in their leaf modules; the shim must re-export
# the same objects, not copies.
_SHIM_WIDGET_SYMBOLS = {
    "SubtitleOverlay": SubtitleOverlay,
    "ChatMessage": ChatMessage,
    "MonitorBar": MonitorBar,
    "DragHandle": DragHandle,
    "ErrorBanner": ErrorBanner,
}

# Theme symbols were already re-exported from core.theme before the split;
# the shim must keep exposing the same objects.
_SHIM_THEME_SYMBOLS = (
    "DEFAULT_STYLE",
    "META_COLOR_ASR",
    "META_COLOR_MENU_BG",
    "META_COLOR_MENU_BORDER",
    "META_COLOR_MENU_FG",
    "META_COLOR_SEPARATOR",
    "META_COLOR_SOURCE_LANG",
    "META_COLOR_TL",
    "hex_to_rgba",
    "migrate_style",
)


def test_shim_widget_symbols_are_the_leaf_objects():
    for name, leaf_sym in _SHIM_WIDGET_SYMBOLS.items():
        assert getattr(overlay, name) is leaf_sym, name


def test_shim_theme_symbols_are_the_core_theme_objects():
    for name in _SHIM_THEME_SYMBOLS:
        assert getattr(overlay, name) is getattr(theme, name), name


def test_subtitle_overlay_assembles_from_leaf_widgets(qapp):
    assert qapp is not None
    win = SubtitleOverlay({})
    qapp.processEvents()
    assert win.message_count() == 0
    # The window still owns a leaf DragHandle / MonitorBar / ErrorBanner.
    assert isinstance(win._handle, DragHandle)
    assert isinstance(win._monitor, MonitorBar)
    assert isinstance(win._error_banner, ErrorBanner)
    win.close()
