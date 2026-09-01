"""App shell: tray menu, hotkeys and cross-window wiring.

Moved out of main() so the entrypoint only performs startup sequencing
(setup wizard, model download, window creation) and delegates all the
interactive wiring to build_tray_shell().

The tray menu stays deliberately small (status / pause / overlay /
settings / quit): everything else lives in the overlay header, the
overlay message context menu and the settings panel, and the overlay
is one left-click away from the tray.
"""

import logging
import types

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QCursor, QFontMetrics
from PyQt6.QtWidgets import (
    QApplication,
    QMenu,
    QMessageBox,
    QSystemTrayIcon,
)

from livetranslate.core.i18n import t
from livetranslate.ui.hotkeys import DEFAULT_HOTKEYS, HotkeyManager
from livetranslate.ui.icons import create_app_icon

log = logging.getLogger("LiveTranslate.Shell")


def build_tray_shell(
    app,
    live_trans,
    overlay,
    subwin,
    panel,
    tray,
    subwin_was_enabled,
):
    """Wire the tray menu, hotkeys and the overlay/subtitle/panel sync.

    Returns a namespace with on_start / on_quit callbacks.
    """
    menu = QMenu()

    # --- Status line (read-only, refreshed on every menu open) ---
    # Every QAction is parented to its menu: PyQt garbage-collects
    # parent-less QActions once the shell's local references go away,
    # which silently removed menu entries at runtime.
    status_action = QAction("", menu)
    status_action.setEnabled(False)
    menu.addAction(status_action)
    menu.addSeparator()

    # --- Pause / Resume toggle ---
    pause_action = QAction(t("tray_pause"), menu)
    _is_running = [True]  # mutable for closure

    def on_start():
        try:
            live_trans.start()
            overlay.set_running(True)
            _is_running[0] = True
            pause_action.setText(t("tray_pause"))
            tray.setIcon(create_app_icon("run"))
        except Exception as e:
            log.error(f"Start error: {e}", exc_info=True)

    def on_pause():
        live_trans.pause()
        overlay.set_running(False)
        _is_running[0] = False
        pause_action.setText(t("tray_resume"))
        tray.setIcon(create_app_icon("pause"))

    def on_resume():
        live_trans.resume()
        overlay.set_running(True)
        _is_running[0] = True
        pause_action.setText(t("tray_pause"))
        tray.setIcon(create_app_icon("run"))

    def on_toggle_pause():
        if _is_running[0]:
            on_pause()
        else:
            on_resume()

    pause_action.triggered.connect(on_toggle_pause)
    menu.addAction(pause_action)
    menu.addSeparator()

    # --- Show/hide overlay ---
    overlay_toggle_action = QAction(t("tray_show_overlay"), menu)

    _hide_notified = [False]

    def on_toggle_overlay():
        if overlay.isVisible():
            overlay.hide()
            overlay_toggle_action.setText(t("tray_show_overlay"))
            if not _hide_notified[0]:
                _hide_notified[0] = True
                tray.showMessage(
                    "LiveTranslate",
                    t("hide_tray_hint"),
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )
        else:
            overlay.show()
            overlay.raise_()
            overlay_toggle_action.setText(t("tray_hide_overlay"))

    overlay_toggle_action.triggered.connect(on_toggle_overlay)
    menu.addAction(overlay_toggle_action)
    menu.addSeparator()

    # --- Settings panel / diagnostics (error banner routes here) ---
    panel_action = QAction(t("tray_show_panel"), menu)

    def on_toggle_panel():
        if panel.isVisible():
            panel.hide()
        else:
            panel.show()
            panel.raise_()

    def on_open_diagnostics():
        from livetranslate.ui.diagnostics import DiagnosticsDialog

        dlg = DiagnosticsDialog(live_trans)
        dlg.exec()

    panel_action.triggered.connect(on_toggle_panel)
    menu.addAction(panel_action)
    menu.addSeparator()

    # --- Quit (with confirmation) + hotkeys ---
    quit_action = QAction(t("quit"), menu)

    def _save_overlay_pos():
        pos = overlay.pos()
        size = overlay.size()
        panel.update_settings(
            {
                "overlay_x": pos.x(),
                "overlay_y": pos.y(),
                "overlay_w": size.width(),
                "overlay_h": size.height(),
            }
        )

    overlay.position_changed.connect(_save_overlay_pos)

    def _save_subwin_state():
        sm = dict(panel.get_settings().get("subtitle_mode") or {})
        sm["enabled"] = subwin.isVisible()
        pos = subwin.pos()
        sm["window_x"] = pos.x()
        sm["window_y"] = pos.y()
        panel.update_settings({"subtitle_mode": sm})

    def on_toggle_subwin():
        """Show/hide the OBS subtitle window (driven by the overlay header
        button; the tray menu stays minimal)."""
        if subwin.isVisible():
            subwin.hide()
        else:
            subwin.show()
            subwin.raise_()
            tray.showMessage(
                "LiveTranslate",
                t("subwin_drag_hint"),
                QSystemTrayIcon.MessageIcon.Information,
                3000,
            )
        overlay.set_subtitle_checked(subwin.isVisible())
        _save_subwin_state()

    subwin.position_changed.connect(_save_subwin_state)

    # Sync when the subtitle window is manually closed (e.g. Alt+F4)
    def _on_subwin_closed():
        overlay.set_subtitle_checked(False)
        _save_subwin_state()

    subwin.window_closed.connect(_on_subwin_closed)

    # Restore the subtitle window visibility from the saved state
    if subwin_was_enabled:
        subwin.show()

    # Overlay header subtitle button → subtitle window
    overlay.subtitle_toggled.connect(on_toggle_subwin)

    # Panel subtitle settings → subtitle window
    def _on_panel_subtitle_changed(s):
        subwin.apply_settings(s)

    panel.subtitle_settings_changed.connect(_on_panel_subtitle_changed)

    def _on_reset_positions():
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        subwin.move(100, 100)
        _save_subwin_state()
        ow, oh = overlay.width(), overlay.height()
        overlay.move(geo.right() - ow - 50, geo.bottom() - oh - 100)
        _save_overlay_pos()

    panel.reset_positions.connect(_on_reset_positions)

    # --- Overlay-driven model / language switching (the overlay header
    # combos are the only entry points; the tray menu stays minimal) ---

    def on_overlay_model_switch(index):
        models = panel.get_settings().get("models", [])
        if 0 <= index < len(models):
            panel.update_settings({"active_model": index})
            panel.refresh_model_list()
            live_trans.on_model_changed(models[index])

    def _on_overlay_source_lang(code):
        """Overlay source-language combo → ASR engine + panel + overlay."""
        live_trans.asr_controller.set_language(code)
        panel.set_asr_language(code)
        overlay.set_source_language(code)

    def _on_panel_asr_lang_changed(code):
        """Panel ASR language combo → overlay combo."""
        overlay.set_source_language(code)

    overlay.source_language_changed.connect(_on_overlay_source_lang)
    panel.asr_language_changed.connect(_on_panel_asr_lang_changed)
    overlay.model_switch_requested.connect(on_overlay_model_switch)
    overlay.target_language_changed.connect(live_trans.on_target_language_changed)
    overlay.start_requested.connect(on_resume)
    overlay.stop_requested.connect(on_pause)
    overlay.hide_requested.connect(on_toggle_overlay)

    # --- Status line + tray tooltip, refreshed on every menu open ---
    # The status text is elided to a fixed cap so the menu width adapts to
    # its content (narrow when the status is short) instead of being blown
    # wide by the longest status line; the full text lives in the tooltip.
    _STATUS_MAX_WIDTH = 240

    def _refresh_status():
        settings = panel.get_settings()
        model = panel.get_active_model()
        model_name = (model or {}).get("name") or "?"
        engine = getattr(getattr(live_trans, "_asr_ctl", None), "type", "?") or "?"
        src = settings.get("asr_language", "auto")
        tgt = settings.get("target_language", "zh")
        state = t("tray_state_running") if _is_running[0] else t("tray_state_paused")
        text = t("tray_status_format").format(
            state=state, engine=engine, model=model_name, src=src, tgt=tgt
        )
        metrics = QFontMetrics(menu.font())
        status_action.setText(
            metrics.elidedText(text, Qt.TextElideMode.ElideRight, _STATUS_MAX_WIDTH)
        )
        tray.setToolTip(f"LiveTranslate — {text}")

    menu.aboutToShow.connect(_refresh_status)

    hotkeys = HotkeyManager()
    _active_hotkeys = [dict(DEFAULT_HOTKEYS)]

    def _apply_hotkeys(combos: dict, interactive: bool = False) -> None:
        # Re-register from scratch. OS conflicts revert to the last-good
        # combo; user-initiated changes surface the §3.6 conflict copy.
        for name in list(hotkeys._combos):
            hotkeys.unregister(name)
        applied: dict[str, str] = {}
        failed: list[str] = []
        for name in DEFAULT_HOTKEYS:
            combo = combos.get(name, DEFAULT_HOTKEYS[name])
            status = hotkeys.register(name, combo)
            if status.ok:
                applied[name] = combo
            else:
                failed.append(name)
                fallback = _active_hotkeys[0].get(name, DEFAULT_HOTKEYS[name])
                hotkeys.register(name, fallback)
                applied[name] = fallback
        _active_hotkeys[0] = applied
        if failed:
            if interactive:
                name = failed[0]
                tray.showMessage(
                    "LiveTranslate",
                    t("err_hotkey_conflict").format(combo=combos.get(name, "")),
                    QSystemTrayIcon.MessageIcon.Warning,
                    6000,
                )
                panel.set_hotkey_combo(name, applied[name])
            log.warning(f"Hotkey registration failed for {failed} — kept fallbacks")

    def _on_hotkey(name):
        if name == "pause":
            on_toggle_pause()
        elif name == "overlay":
            on_toggle_overlay()
        elif name == "subtitle":
            on_toggle_subwin()
        elif name == "clear":
            overlay.clear()

    hotkeys.triggered.connect(_on_hotkey)
    panel.hotkeys_changed.connect(lambda c: _apply_hotkeys(c, interactive=True))
    QTimer.singleShot(
        500, lambda: _apply_hotkeys(panel.get_settings().get("hotkeys") or dict(DEFAULT_HOTKEYS))
    )

    def on_quit(confirm: bool = True):
        """Quit the app; programmatic exits (smoke, SIGINT) skip the
        confirmation dialog."""
        if confirm:
            ret = QMessageBox.question(
                panel,
                t("quit_confirm_title"),
                t("quit_confirm_msg"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return
        hotkeys.stop()
        live_trans.stop()
        app.quit()

    quit_action.triggered.connect(on_quit)
    overlay.quit_requested.connect(on_quit)
    menu.addAction(quit_action)

    # --- Connect overlay signals ---
    overlay.settings_requested.connect(on_toggle_panel)
    # §3.5.6: the error banner routes straight to the diagnostics cards.
    overlay.error_banner_clicked.connect(on_open_diagnostics)
    # Translation/engine errors flip the tray icon to the error variant;
    # start/pause/resume restore run/pause above.
    overlay.error_banner_signal.connect(lambda _msg: tray.setIcon(create_app_icon("error")))

    def _show_overlay():
        """Left-click on the tray: bring up the main overlay window."""
        if not overlay.isVisible():
            overlay.show()
            overlay_toggle_action.setText(t("tray_hide_overlay"))
        overlay.raise_()

    def _on_tray_activated(reason):
        if reason == QSystemTrayIcon.ActivationReason.Context:
            menu.popup(QCursor.pos())
        elif reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            _show_overlay()

    # Windows/Linux: pop the QSS-themed QMenu ourselves — the native tray
    # menu can never match the app theme. Left click opens the overlay,
    # right click opens the menu.
    tray.activated.connect(_on_tray_activated)

    # -- empty-state guide (plan §3.5.6): periodic re-evaluation --
    _idle_shown = [False]

    def _engine_is_ready() -> bool:
        import sys as _sys

        from livetranslate.asr.availability import engine_status
        from livetranslate.asr.registry import engine_id_for_type

        settings = live_trans.get_settings() if hasattr(live_trans, "get_settings") else {}
        engine_id = engine_id_for_type(settings.get("asr_engine"), _sys.platform)
        if engine_id is None:
            return True  # unknown / not selectable -> don't nag
        status = engine_status(engine_id, _sys.platform)
        return status in ("available", "unsupported", "not-implemented")

    def _refresh_empty_guide():
        if overlay.message_count() > 0:
            _idle_shown[0] = False
            overlay.hide_empty_guide()
            return
        settings = live_trans.get_settings() if hasattr(live_trans, "get_settings") else {}
        if not settings.get("models"):
            overlay.show_empty_guide("translation")
        elif not _engine_is_ready():
            overlay.show_empty_guide("engine")
        elif getattr(getattr(live_trans, "_pipeline", None), "running", False):
            overlay.show_empty_guide("no-audio")
        elif not _idle_shown[0]:
            _idle_shown[0] = True
            overlay.show_empty_guide("idle")

    def _on_guide_activated(kind: str):
        if kind == "translation":
            on_toggle_panel()
        elif kind == "diagnostics":
            on_open_diagnostics()
        elif kind == "engine":
            on_toggle_panel()
            # Jump straight to the recognition page (nav order: general,
            # translation, recognition, ...) where the one-click fill lives.
            panel._nav.setCurrentRow(2)

    overlay.empty_guide_activated.connect(_on_guide_activated)

    # §3.7 GNOME 无托盘: degrade loudly once (startup banner); the overlay
    # buttons and hotkeys remain the fallback controls.
    if not QSystemTrayIcon.isSystemTrayAvailable():
        log.warning("No system tray available on this desktop")
        QTimer.singleShot(3000, lambda: overlay.show_info(t("err_gnome_tray")))

    empty_state_timer = QTimer()
    empty_state_timer.setInterval(10000)
    empty_state_timer.timeout.connect(_refresh_empty_guide)
    empty_state_timer.start()
    QTimer.singleShot(2000, _refresh_empty_guide)

    def _on_memory_warning(rss_mb: float):
        # Plan §3.2: memory warnings are downgraded from tray popups to a
        # log line + the diagnostics page (a video viewer must never be
        # interrupted mid-watch).
        log.warning(
            "High memory usage: %.0f MB RSS — see the Diagnostics page for details",
            rss_mb,
        )

    live_trans.set_memory_warning_callback(_on_memory_warning)

    return types.SimpleNamespace(on_start=on_start, on_quit=on_quit, menu=menu)
