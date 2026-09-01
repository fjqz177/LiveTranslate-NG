from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)

from livetranslate.core.i18n import t
from livetranslate.core.theme import DEFAULT_STYLE, hex_to_rgba, migrate_style
from livetranslate.platform import window as winutil
from livetranslate.ui.overlay_banner import ErrorBanner
from livetranslate.ui.overlay_controls import DragHandle
from livetranslate.ui.overlay_message import ChatMessage
from livetranslate.ui.overlay_monitor import MonitorBar


class SubtitleOverlay(QWidget):
    """Chat-style overlay window for displaying live transcription."""

    add_message_signal = pyqtSignal(int, str, str, str, float)
    update_translation_signal = pyqtSignal(int, str, float)
    update_streaming_signal = pyqtSignal(int, str)
    clear_signal = pyqtSignal()
    # Monitor signals (thread-safe)
    update_monitor_signal = pyqtSignal(float, float, object)
    update_stats_signal = pyqtSignal(int, int, int, int, float)
    update_asr_device_signal = pyqtSignal(str)

    settings_requested = pyqtSignal()
    target_language_changed = pyqtSignal(str)
    source_language_changed = pyqtSignal(str)
    model_switch_requested = pyqtSignal(int)
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    hide_requested = pyqtSignal()
    quit_requested = pyqtSignal()
    subtitle_toggled = pyqtSignal()
    mode_changed = pyqtSignal(str)  # "full" or "compact"
    position_changed = pyqtSignal()
    error_banner_signal = pyqtSignal(str)  # thread-safe banner update
    error_banner_clicked = pyqtSignal()
    info_banner_signal = pyqtSignal(str)  # neutral notices (e.g. no tray)
    # Re-exported header checkbox toggles (tray menu subscribes to these
    # instead of reaching into the private DragHandle).
    click_through_toggled = pyqtSignal(bool)
    topmost_toggled = pyqtSignal(bool)
    auto_scroll_toggled = pyqtSignal(bool)
    taskbar_toggled = pyqtSignal(bool)

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._messages = {}
        self._max_messages = 50
        self._click_through = False
        self._height_before_compact = None
        self._mode_anim = None
        self._reduce_motion = False
        # Per-overlay theme state (was ChatMessage class state before)
        self._style = dict(DEFAULT_STYLE)
        self._compact_mode = False
        self._pos_save_timer = QTimer(self)
        self._pos_save_timer.setSingleShot(True)
        self._pos_save_timer.setInterval(500)
        self._pos_save_timer.timeout.connect(lambda: self.position_changed.emit())
        self._last_saved_geo = None
        self._setup_ui()

        self.add_message_signal.connect(self._on_add_message)
        self.update_translation_signal.connect(self._on_update_translation)
        self.update_streaming_signal.connect(self._on_update_streaming)
        self.clear_signal.connect(self._on_clear)
        self.update_monitor_signal.connect(self._on_update_monitor)
        self.update_stats_signal.connect(self._on_update_stats)
        self.update_asr_device_signal.connect(self._on_update_asr_device)

    def _setup_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowTitle("LiveTranslate")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        width = 620
        height = 500
        x = geo.right() - width - 20
        y = geo.bottom() - height - 60
        self.setGeometry(x, y, width, height)
        self.setMinimumSize(480, 200)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._container = QWidget()
        self._container.setStyleSheet(
            "background-color: rgba(15, 15, 25, 200); border-radius: 8px;"
        )

        container_layout = QVBoxLayout(self._container)
        container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.setSpacing(0)

        # Drag handle
        self._handle = DragHandle()
        self._handle.settings_clicked.connect(self.settings_requested.emit)
        self._handle.subtitle_clicked.connect(self.subtitle_toggled.emit)
        self._handle.click_through_toggled.connect(self._set_click_through)
        self._handle.topmost_toggled.connect(self._set_topmost)
        self._handle.taskbar_toggled.connect(self._set_taskbar)
        # Forward header checkbox toggles for external consumers (tray menu).
        self._handle.click_through_toggled.connect(self.click_through_toggled.emit)
        self._handle.topmost_toggled.connect(self.topmost_toggled.emit)
        self._handle.auto_scroll_toggled.connect(self.auto_scroll_toggled.emit)
        self._handle.taskbar_toggled.connect(self.taskbar_toggled.emit)
        self._handle.target_language_changed.connect(self.target_language_changed.emit)
        self._handle.source_language_changed.connect(self.source_language_changed.emit)
        self._handle.model_changed.connect(self.model_switch_requested.emit)
        self._handle.start_clicked.connect(self.start_requested.emit)
        self._handle.stop_clicked.connect(self.stop_requested.emit)
        self._handle.hide_clicked.connect(self.hide_requested.emit)
        self._handle.clear_clicked.connect(self._on_clear)
        self._handle.quit_clicked.connect(self.quit_requested.emit)
        self._handle.mode_changed.connect(self._on_mode_changed)
        self._handle.position_changed.connect(self.position_changed)
        container_layout.addWidget(self._handle)

        # Monitor bar (collapsible)
        self._monitor = MonitorBar()
        container_layout.addWidget(self._monitor)

        # Error banner (plan §3.5.6: top-of-window, not a modal)
        self._error_banner = ErrorBanner()
        self._error_banner.diagnostics_clicked.connect(self.error_banner_clicked.emit)
        self._error_banner.dismissed.connect(self._error_banner.hide)
        container_layout.addWidget(self._error_banner)
        self.error_banner_signal.connect(self._on_error_banner)
        self.info_banner_signal.connect(self._on_info_banner)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                width: 6px; background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,60); border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        self._msg_container = QWidget()
        self._msg_container.setStyleSheet("background: transparent;")
        self._msg_layout = QVBoxLayout(self._msg_container)
        self._msg_layout.setContentsMargins(0, 0, 0, 0)
        self._msg_layout.setSpacing(2)
        self._msg_layout.addStretch()

        # Empty-state guide (plan §3.5.6): branch A = translation not
        # configured, branch B = listening but nothing recognized.
        self._empty_guide = QLabel("")
        self._empty_guide.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_guide.setWordWrap(True)
        self._empty_guide.setStyleSheet("color: #9AA3B2; background: transparent; padding: 12px;")
        self._empty_guide.linkActivated.connect(self.empty_guide_activated)
        self._empty_guide.hide()
        self._msg_layout.insertWidget(0, self._empty_guide)

        self._scroll.setWidget(self._msg_container)
        container_layout.addWidget(self._scroll)

        grip_row = QHBoxLayout()
        grip_row.addStretch()
        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(16, 16)
        self._grip.setStyleSheet("background: transparent;")
        grip_row.addWidget(self._grip)
        container_layout.addLayout(grip_row)

        main_layout.addWidget(self._container)

        # UI-7: the click-through poller only runs while click-through is
        # enabled (50ms SetWindowLongW churn was happening even when off).
        self._ct_timer = QTimer(self)
        self._ct_timer.timeout.connect(self._check_click_through)
        self._ct_timer.setInterval(50)
        self._ct_active = False  # last applied Win32 style state

    def _schedule_pos_save(self):
        if not self.isVisible():
            return
        geo = (self.x(), self.y(), self.width(), self.height())
        if geo != self._last_saved_geo:
            self._last_saved_geo = geo
            self._pos_save_timer.start()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._schedule_pos_save()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_pos_save()

    def set_running(self, running: bool):
        self._handle.set_running(running)

    def _set_topmost(self, enabled: bool):
        flags = self.windowFlags()
        if enabled:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

    def _set_taskbar(self, enabled: bool):
        flags = self.windowFlags()
        if enabled:
            flags &= ~Qt.WindowType.Tool
        else:
            flags |= Qt.WindowType.Tool
        self.setWindowFlags(flags)
        self.show()

    def _set_click_through(self, enabled: bool):
        self._click_through = enabled
        if enabled:
            # Immediate apply + keep polling only while active.
            self._ct_active = False
            self._check_click_through()
            if not self._ct_timer.isActive():
                self._ct_timer.start()
        else:
            self._ct_timer.stop()
            self._ct_active = False
            winutil.clear_click_through(self)

    def _check_click_through(self):
        if not self._click_through:
            return
        cursor = QCursor.pos()
        local = self.mapFromGlobal(cursor)

        scroll_top = self._scroll.mapTo(self, QPoint(0, 0)).y()
        in_header = 0 <= local.x() <= self.width() and 0 <= local.y() < scroll_top

        # Dirty-flag: only touch the native style when the desired state
        # actually changed — SetWindowLongW on every 50ms tick was churn.
        desired = not in_header
        if desired == self._ct_active:
            return
        self._ct_active = desired
        if in_header:
            winutil.clear_click_through(self)
        else:
            winutil.set_click_through(self, "all")

    def _on_mode_changed(self, mode: str):
        compact = mode == "compact"
        self._monitor.setVisible(not compact)
        self._compact_mode = compact
        for msg in self._messages.values():
            msg.set_compact(compact)
        self.mode_changed.emit(mode)

        # Animate window height (instant when reduced motion is on)
        if self._mode_anim and self._mode_anim.state() != QPropertyAnimation.State.Stopped:
            self._mode_anim.stop()

        if self._reduce_motion:
            if compact:
                self._height_before_compact = self.height()
                target_h = self.minimumHeight()
            else:
                target_h = self._height_before_compact or 500
            self.resize(self.width(), target_h)
            return

        if compact:
            self._height_before_compact = self.height()
            target_h = self.minimumHeight()
        else:
            target_h = self._height_before_compact or 500

        actual_h = self.frameGeometry().height()
        if abs(actual_h - target_h) < 10:
            self.resize(self.width(), target_h)
        else:
            anim = QPropertyAnimation(self, b"size")
            anim.setDuration(200)
            anim.setStartValue(QSize(self.width(), actual_h))
            anim.setEndValue(QSize(self.width(), target_h))
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._mode_anim = anim
            anim.start()

    def set_mode(self, mode: str):
        self._handle.set_mode(mode)

    def set_reduce_motion(self, enabled: bool):
        """Skip the compact/full resize transition (plan §3.5: respect
        reduced motion; animations are decoration, never information)."""
        self._reduce_motion = bool(enabled)

    def set_subtitle_checked(self, checked: bool):
        self._handle.set_subtitle_checked(checked)

    # --- Public checkbox setters (tray sync; blockSignals guards echo) ---

    @staticmethod
    def _set_check_quietly(check, value):
        check.blockSignals(True)
        check.setChecked(value)
        check.blockSignals(False)

    def set_click_through_checked(self, value: bool):
        self._set_check_quietly(self._handle._ct_check, value)

    def set_topmost_checked(self, value: bool):
        self._set_check_quietly(self._handle._topmost_check, value)

    def set_auto_scroll_checked(self, value: bool):
        self._set_check_quietly(self._handle._auto_scroll, value)

    def set_taskbar_checked(self, value: bool):
        self._set_check_quietly(self._handle._taskbar_check, value)

    def message_count(self) -> int:
        return len(self._messages)

    @pyqtSlot(float, float, object)
    def _on_update_monitor(self, rms: float, vad_conf: float, mic_rms):
        self._monitor.update_audio(rms, vad_conf, mic_rms)

    @pyqtSlot(int, int, int, int, float)
    def _on_update_stats(self, asr_count, tl_count, prompt_tokens, completion_tokens, cost):
        self._monitor.update_pipeline_stats(
            asr_count, tl_count, prompt_tokens, completion_tokens, cost
        )

    @pyqtSlot(str)
    def _on_update_asr_device(self, device: str):
        self._monitor.update_asr_device(device)

    @pyqtSlot(int, str, str, str, float)
    def _on_add_message(self, msg_id, timestamp, original, source_lang, asr_ms):
        msg = ChatMessage(
            msg_id,
            timestamp,
            original,
            source_lang,
            asr_ms,
            style=self._style,
            compact=self._compact_mode,
        )
        self._messages[msg_id] = msg
        self._msg_layout.addWidget(msg)

        if len(self._messages) > self._max_messages:
            oldest_id = min(self._messages.keys())
            old_msg = self._messages.pop(oldest_id)
            self._msg_layout.removeWidget(old_msg)
            old_msg.deleteLater()

        self._empty_guide.hide()
        QTimer.singleShot(50, self._scroll_to_bottom)

    @pyqtSlot(int, str, float)
    def _on_update_translation(self, msg_id, translated, translate_ms):
        msg = self._messages.get(msg_id)
        if msg:
            msg.set_translation(translated, translate_ms)
            QTimer.singleShot(50, self._scroll_to_bottom)

    def _on_update_streaming(self, msg_id, partial_text):
        msg = self._messages.get(msg_id)
        if msg:
            msg.update_streaming(partial_text)

    @pyqtSlot()
    def _on_clear(self):
        for msg in self._messages.values():
            self._msg_layout.removeWidget(msg)
            msg.deleteLater()
        self._messages.clear()
        # The app re-decides whether an empty guide applies after a clear.

    def _scroll_to_bottom(self):
        if not self._handle.auto_scroll:
            return
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def apply_style(self, style: dict):
        s = migrate_style(style)
        # Container background
        bg_rgba = hex_to_rgba(s["bg_color"], s["bg_opacity"])
        self._container.setStyleSheet(
            f"background-color: {bg_rgba}; border-radius: {s['border_radius']}px;"
        )
        # Header background
        hdr_rgba = hex_to_rgba(s["header_color"], s["header_opacity"])
        self._handle.setStyleSheet(f"background: {hdr_rgba}; border-radius: 4px;")
        # Window opacity
        self.setWindowOpacity(s["window_opacity"] / 100.0)
        # Update all existing messages
        self._style = s
        for msg in self._messages.values():
            msg.apply_style(s)

    def export_messages(self, mode: str, parent=None):
        """Export captured messages to a .txt file.

        mode: "original" | "translation" | "both"
        """
        if not self._messages:
            QMessageBox.information(parent or self, "LiveTranslate", t("export_empty"))
            return

        from datetime import datetime

        suffix = {"original": "original", "translation": "translation", "both": "all"}.get(
            mode, "all"
        )
        default_name = f"livetrans_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}.txt"
        path, _ = QFileDialog.getSaveFileName(
            parent or self,
            t("export_dialog_title"),
            default_name,
            t("export_filter"),
        )
        if not path:
            return

        lines = []
        for msg_id in sorted(self._messages.keys()):
            msg = self._messages[msg_id]
            ts = msg._timestamp
            orig = msg._original or ""
            trans = msg._translated or ""
            if mode == "original":
                lines.append(f"[{ts}] {orig}")
            elif mode == "translation":
                if trans:
                    lines.append(f"[{ts}] {trans}")
            else:
                lines.append(f"[{ts}] {orig}")
                if trans:
                    lines.append(f"  -> {trans}")
                lines.append("")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines).rstrip() + "\n")
        except OSError as e:
            QMessageBox.critical(
                parent or self,
                "LiveTranslate",
                t("export_failed").format(error=str(e)),
            )

    # Thread-safe public API
    def add_message(self, msg_id, timestamp, original, source_lang, asr_ms):
        self.add_message_signal.emit(msg_id, timestamp, original, source_lang, asr_ms)

    def update_translation(self, msg_id, translated, translate_ms):
        self.update_translation_signal.emit(msg_id, translated, translate_ms)

    def show_error(self, text: str):
        """Show a classified error in the banner (thread-safe entry)."""
        self.error_banner_signal.emit(text)

    def show_info(self, text: str):
        """Show a neutral notice in the banner (thread-safe entry)."""
        self.info_banner_signal.emit(text)

    @pyqtSlot(str)
    def _on_error_banner(self, text: str):
        self._error_banner.show_text(text)

    @pyqtSlot(str)
    def _on_info_banner(self, text: str):
        self._error_banner.show_text(text, kind="info")

    def update_streaming(self, msg_id, partial_text):
        self.update_streaming_signal.emit(msg_id, partial_text)

    def update_monitor(self, rms, vad_conf, mic_rms=None):
        self.update_monitor_signal.emit(rms, vad_conf, mic_rms)

    def update_stats(self, asr_count, tl_count, prompt_tokens, completion_tokens, cost=0.0):
        self.update_stats_signal.emit(asr_count, tl_count, prompt_tokens, completion_tokens, cost)

    def update_asr_device(self, device: str):
        self.update_asr_device_signal.emit(device)

    def set_target_language(self, lang: str):
        self._handle.set_target_language(lang)

    def set_source_language(self, lang: str):
        self._handle.set_source_language(lang)

    def set_models(self, models: list, active_index: int = 0):
        self._handle.set_models(models, active_index)

    def clear(self):
        self.clear_signal.emit()

    # -- empty-state guide (plan §3.5.6) ------------------------------------

    def show_empty_guide(self, kind: str) -> None:
        """kind: 'translation' (configure API), 'no-audio' (diagnose) or
        'idle' (normal hint; fades out after 8s per §3.5.6)."""
        if self._messages:
            return
        if kind == "translation":
            text = (
                t("empty_guide_translation")
                + ' <a href="translation">'
                + t("empty_guide_translation_link")
                + "</a>"
            )
        elif kind == "no-audio":
            text = (
                t("empty_guide_no_audio")
                + ' <a href="diagnostics">'
                + t("empty_guide_diagnostics_link")
                + "</a>"
            )
        elif kind == "engine":
            text = (
                t("empty_guide_engine")
                + ' <a href="engine">'
                + t("empty_guide_engine_link")
                + "</a>"
            )
        else:
            text = t("empty_guide_idle")
        self._empty_guide.setText(text)
        self._empty_guide.show()
        if kind == "idle":
            QTimer.singleShot(8000, self._fade_out_empty_guide)

    def _fade_out_empty_guide(self) -> None:
        if not self._empty_guide.isVisible():
            return
        if self._reduce_motion:
            self._empty_guide.hide()
            return
        effect = QGraphicsOpacityEffect(self._empty_guide)
        self._empty_guide.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(300)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)

        def _finish() -> None:
            self._empty_guide.hide()
            self._empty_guide.setGraphicsEffect(None)

        anim.finished.connect(_finish)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def hide_empty_guide(self) -> None:
        self._empty_guide.hide()

    # Emitted when the user clicks a guide link; the shell routes
    # 'translation' -> settings, 'diagnostics' -> diagnostics dialog.
    empty_guide_activated = pyqtSignal(str)
