"""Backward-compatible re-export shim for the M-SPLIT overlay split.

The UI overlay (previously one ~1441-line module) now lives as flat sibling
leaf modules, each owning a single widget:

- overlay_message  : ChatMessage (+ _escape)
- overlay_monitor  : MonitorBar (+ _BAR_CSS_TPL)
- overlay_controls : _DragArea / DragHandle (+ _BTN_CSS / _COMBO_CSS / _CHECK_CSS)
- overlay_banner   : ErrorBanner
- overlay_window   : SubtitleOverlay

This module keeps the historical public surface importable so existing
importers keep working unchanged. New code should import from the leaves
directly. The private CSS constants stay leaf-local on purpose (importing
``_BAR_CSS_TPL`` / ``_BTN_CSS`` / ... from here used to be an accidental
coupling and now fails loudly, as intended).
"""

from livetranslate.core.theme import (
    DEFAULT_STYLE,
    META_COLOR_ASR,
    META_COLOR_MENU_BG,
    META_COLOR_MENU_BORDER,
    META_COLOR_MENU_FG,
    META_COLOR_SEPARATOR,
    META_COLOR_SOURCE_LANG,
    META_COLOR_TL,
    hex_to_rgba,
    migrate_style,
)
from livetranslate.ui.overlay_banner import ErrorBanner
from livetranslate.ui.overlay_controls import DragHandle
from livetranslate.ui.overlay_message import ChatMessage
from livetranslate.ui.overlay_monitor import MonitorBar
from livetranslate.ui.overlay_window import SubtitleOverlay

__all__ = [
    "DEFAULT_STYLE",
    "META_COLOR_ASR",
    "META_COLOR_MENU_BG",
    "META_COLOR_MENU_BORDER",
    "META_COLOR_MENU_FG",
    "META_COLOR_SEPARATOR",
    "META_COLOR_SOURCE_LANG",
    "META_COLOR_TL",
    "ChatMessage",
    "DragHandle",
    "ErrorBanner",
    "MonitorBar",
    "SubtitleOverlay",
    "hex_to_rgba",
    "migrate_style",
]
