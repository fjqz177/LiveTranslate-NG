from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QLabel,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from livetranslate.core.i18n import t
from livetranslate.core.theme import (
    DEFAULT_STYLE,
    META_COLOR_ASR,
    META_COLOR_MENU_BG,
    META_COLOR_MENU_BORDER,
    META_COLOR_MENU_FG,
    META_COLOR_SOURCE_LANG,
    META_COLOR_TL,
)


class ChatMessage(QWidget):
    """Single chat message widget with original + async translation."""

    def __init__(
        self,
        msg_id: int,
        timestamp: str,
        original: str,
        source_lang: str,
        asr_ms: float,
        style: dict | None = None,
        compact: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.msg_id = msg_id
        self._original = original
        self._translated = ""
        self._timestamp = timestamp
        self._source_lang = source_lang
        self._asr_ms = asr_ms
        self._translate_ms = 0.0
        self._style = dict(style) if style else dict(DEFAULT_STYLE)
        self._compact = compact
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(2)

        s = self._style
        self._header_label = QLabel(self._build_header_html(s))
        self._header_label.setFont(QFont(s["original_font_family"], s["original_font_size"]))
        self._header_label.setTextFormat(Qt.TextFormat.RichText)
        self._header_label.setWordWrap(True)
        self._header_label.setStyleSheet("background: transparent;")
        self._layout.addWidget(self._header_label)

        self._trans_label = QLabel(
            f'<span style="color:#999; font-style:italic;">{t("translating")}</span>'
        )
        self._trans_label.setFont(QFont(s["translation_font_family"], s["translation_font_size"]))
        self._trans_label.setTextFormat(Qt.TextFormat.RichText)
        self._trans_label.setWordWrap(True)
        self._trans_label.setStyleSheet("background: transparent;")
        self._layout.addWidget(self._trans_label)

        self._apply_outline(s)

    def _apply_outline(self, s: dict) -> None:
        """Outline compensation for translucent presets (CORE-9): when the
        style declares outline_compensation >= 2 px, draw a dark halo around
        the subtitle text so readability no longer depends on the desktop
        behind the window (theme.validate_style_contrast enforces the field
        for bg_opacity < 200)."""
        outline = int(s.get("outline_compensation", 0))
        for label in (self._header_label, self._trans_label):
            if outline >= 2:
                effect = QGraphicsDropShadowEffect(label)
                effect.setBlurRadius(outline * 2)
                effect.setOffset(0)
                effect.setColor(QColor(0, 0, 0))
                label.setGraphicsEffect(effect)
            else:
                label.setGraphicsEffect(None)

    def _build_header_html(self, s):
        if self._compact:
            return (
                f'<span style="color:{META_COLOR_SOURCE_LANG};">[{self._source_lang}]</span> '
                f'<span style="color:{s["original_color"]};">{_escape(self._original)}</span>'
            )
        return (
            f'<span style="color:{s["timestamp_color"]};">[{self._timestamp}]</span> '
            f'<span style="color:{META_COLOR_SOURCE_LANG};">[{self._source_lang}]</span> '
            f'<span style="color:{s["original_color"]};">{_escape(self._original)}</span> '
            f'<span style="color:{META_COLOR_ASR}; font-size:9pt;">ASR {self._asr_ms:.0f}ms</span>'
        )

    def update_streaming(self, partial_text: str):
        """Update translation label with partial streaming text (throttled)."""
        self._pending_streaming = partial_text
        if not hasattr(self, "_streaming_timer"):
            # Parented to self so it dies with the widget (no dangling timer
            # firing on a deleted C++ object after clear() / message eviction).
            self._streaming_timer = QTimer(self)
            self._streaming_timer.setSingleShot(True)
            self._streaming_timer.setInterval(50)
            self._streaming_timer.timeout.connect(self._flush_streaming)
        if not self._streaming_timer.isActive():
            self._flush_streaming()
            self._streaming_timer.start()

    def _flush_streaming(self):
        text = getattr(self, "_pending_streaming", None)
        if text is None:
            return
        self._pending_streaming = None
        s = self._style
        self._trans_label.setText(
            f'<span style="color:{s["translation_color"]};">&gt; {_escape(text)}</span>'
        )

    def set_translation(self, translated: str, translate_ms: float):
        self._translated = translated or ""
        self._translate_ms = translate_ms
        # Stop streaming throttle if active
        if hasattr(self, "_streaming_timer"):
            self._streaming_timer.stop()
            self._pending_streaming = None
        s = self._style
        if translated:
            if self._compact:
                self._trans_label.setText(
                    f'<span style="color:{s["translation_color"]};">&gt; {_escape(translated)}</span>'
                )
            else:
                self._trans_label.setText(
                    f'<span style="color:{s["translation_color"]};">&gt; {_escape(translated)}</span> '
                    f'<span style="color:{META_COLOR_TL}; font-size:9pt;">TL {translate_ms:.0f}ms</span>'
                )
        else:
            self._trans_label.setText(
                f'<span style="color:#aaa; font-style:italic;">&gt; {t("same_language")}</span>'
            )

    def apply_style(self, s: dict):
        self._style = dict(s)
        self._header_label.setText(self._build_header_html(s))
        self._header_label.setFont(QFont(s["original_font_family"], s["original_font_size"]))
        self._trans_label.setFont(QFont(s["translation_font_family"], s["translation_font_size"]))
        self._apply_outline(s)
        if self._translated:
            if self._compact:
                self._trans_label.setText(
                    f'<span style="color:{s["translation_color"]};">&gt; {_escape(self._translated)}</span>'
                )
            else:
                self._trans_label.setText(
                    f'<span style="color:{s["translation_color"]};">&gt; {_escape(self._translated)}</span> '
                    f'<span style="color:{META_COLOR_TL}; font-size:9pt;">TL {self._translate_ms:.0f}ms</span>'
                )

    def set_compact(self, compact: bool):
        self._compact = bool(compact)
        self.apply_style(self._style)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(
            f"""
            QMenu {{ background: {META_COLOR_MENU_BG}; color: {META_COLOR_MENU_FG};
                     border: 1px solid {META_COLOR_MENU_BORDER}; }}
            QMenu::item:selected {{ background: #444; }}
            QMenu::separator {{ height: 1px; background: {META_COLOR_MENU_BORDER}; margin: 4px 0; }}
            """
        )
        copy_orig = menu.addAction(t("copy_original"))
        copy_trans = menu.addAction(t("copy_translation"))
        copy_all = menu.addAction(t("copy_all"))
        menu.addSeparator()
        export_menu = menu.addMenu(t("export_menu"))
        export_orig = export_menu.addAction(t("export_original"))
        export_trans = export_menu.addAction(t("export_translation"))
        export_both = export_menu.addAction(t("export_all"))
        menu.addSeparator()
        clear_list = menu.addAction(t("clear_list"))
        action = menu.exec(event.globalPos())
        if action == copy_orig:
            QApplication.clipboard().setText(self._original)
        elif action == copy_trans:
            QApplication.clipboard().setText(self._translated)
        elif action == copy_all:
            QApplication.clipboard().setText(f"{self._original}\n{self._translated}")
        elif action == clear_list:
            overlay = self.window()
            if hasattr(overlay, "_on_clear"):
                overlay._on_clear()
        elif action in (export_orig, export_trans, export_both):
            mode = {export_orig: "original", export_trans: "translation", export_both: "both"}[
                action
            ]
            overlay = self.window()
            if hasattr(overlay, "export_messages"):
                overlay.export_messages(mode, parent=self)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
