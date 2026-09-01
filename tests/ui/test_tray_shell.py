"""Tray shell tests: menu structure, self-drawn tray menu and quit
confirmation (ui/app_shell.py)."""

import types

import pytest
from PyQt6.QtWidgets import QMessageBox, QSystemTrayIcon

from livetranslate.core.i18n import t
from livetranslate.ui.app_shell import build_tray_shell


class _FakeHotkeys:
    """No-op hotkey manager so tests never touch real OS hotkeys."""

    def __init__(self):
        self._combos: dict = {}
        self.triggered = _signal()

    def register(self, name, combo):
        return types.SimpleNamespace(ok=True)

    def unregister(self, name):
        pass

    def stop(self):
        pass


def _signal():
    return types.SimpleNamespace(connect=lambda *a, **k: None)


@pytest.fixture
def shell(qapp, monkeypatch):
    assert qapp is not None
    monkeypatch.setattr("livetranslate.ui.app_shell.HotkeyManager", _FakeHotkeys)

    app = types.SimpleNamespace(quit=lambda: None)
    live_trans = types.SimpleNamespace(
        start=lambda: None,
        pause=lambda: None,
        resume=lambda: None,
        stop=lambda: None,
        on_model_changed=lambda m: None,
        on_target_language_changed=lambda c: None,
        asr_controller=types.SimpleNamespace(set_language=lambda c: None),
        set_memory_warning_callback=lambda cb: None,
        get_settings=lambda: {},
    )
    overlay = types.SimpleNamespace(
        set_running=lambda v: None,
        isVisible=lambda: True,
        hide=lambda: None,
        show=lambda: None,
        raise_=lambda: None,
        clear=lambda: None,
        set_subtitle_checked=lambda v: None,
        set_click_through_checked=lambda v: None,
        set_topmost_checked=lambda v: None,
        set_auto_scroll_checked=lambda v: None,
        set_taskbar_checked=lambda v: None,
        set_target_language=lambda c: None,
        set_source_language=lambda c: None,
        set_models=lambda *a: None,
        message_count=lambda: 0,
        show_empty_guide=lambda k: None,
        hide_empty_guide=lambda: None,
        show_info=lambda m: None,
        export_messages=lambda *a, **k: None,
        width=lambda: 100,
        height=lambda: 100,
        x=lambda: 0,
        y=lambda: 0,
        move=lambda *a: None,
        position_changed=_signal(),
        subtitle_toggled=_signal(),
        click_through_toggled=_signal(),
        topmost_toggled=_signal(),
        auto_scroll_toggled=_signal(),
        taskbar_toggled=_signal(),
        settings_requested=_signal(),
        error_banner_clicked=_signal(),
        error_banner_signal=_signal(),
        target_language_changed=_signal(),
        source_language_changed=_signal(),
        model_switch_requested=_signal(),
        start_requested=_signal(),
        stop_requested=_signal(),
        hide_requested=_signal(),
        quit_requested=_signal(),
        empty_guide_activated=_signal(),
    )
    subwin = types.SimpleNamespace(
        isVisible=lambda: False,
        show=lambda: None,
        hide=lambda: None,
        raise_=lambda: None,
        set_click_through=lambda v: None,
        apply_settings=lambda s: None,
        move=lambda *a: None,
        position_changed=_signal(),
        window_closed=_signal(),
    )
    panel = types.SimpleNamespace(
        get_settings=lambda: {
            "models": [{"name": "gpt-test", "model": "gpt"}],
            "active_model": 0,
            "asr_language": "auto",
            "target_language": "zh",
            "subtitle_mode": {},
            "hotkeys": {},
        },
        get_active_model=lambda: {"name": "gpt-test", "model": "gpt"},
        update_settings=lambda *a, **k: None,
        set_subtitle_click_through=lambda v: None,
        set_hotkey_combo=lambda *a: None,
        set_asr_language=lambda c: None,
        refresh_model_list=lambda: None,
        isVisible=lambda: False,
        show=lambda: None,
        hide=lambda: None,
        raise_=lambda: None,
        subtitle_settings_changed=_signal(),
        reset_positions=_signal(),
        hotkeys_changed=_signal(),
        asr_language_changed=_signal(),
    )
    tray = QSystemTrayIcon()
    built = build_tray_shell(
        app, live_trans, overlay, subwin, panel, tray, subwin_was_enabled=False
    )
    yield types.SimpleNamespace(
        shell=built,
        app=app,
        live_trans=live_trans,
        overlay=overlay,
        subwin=subwin,
        panel=panel,
        tray=tray,
    )
    tray.hide()


def _top_level_actions(menu):
    return [a for a in menu.actions() if not a.isSeparator()]


def test_tray_menu_structure(shell):
    """The tray menu stays minimal: status / pause / overlay toggle /
    settings / quit — everything else lives in the overlay header and
    the settings panel."""
    menu = shell.shell.menu
    texts = [a.text() for a in _top_level_actions(menu)]
    assert texts[0] == ""  # status line first
    assert t("tray_pause") in texts
    assert t("tray_show_overlay") in texts or t("tray_hide_overlay") in texts
    assert t("tray_show_panel") in texts
    assert t("quit") in texts
    # quit must come last
    assert texts[-1] == t("quit")
    # the status line is read-only
    assert not menu.actions()[0].isEnabled()
    # deliberately absent: features reachable from the overlay header,
    # the message context menu or the settings panel
    for gone in (
        "tray_clear_subtitles",
        "tray_menu_model",
        "tray_menu_target_lang",
        "tray_menu_asr_lang",
        "tray_menu_overlay",
        "tray_show_log",
        "tray_diagnostics",
        "export_menu",
    ):
        assert t(gone) not in texts, f"{gone} should not be in the tray menu"


def test_tray_menu_survives_garbage_collection(shell):
    """Regression: parent-less QActions/QMenus were garbage-collected once
    the shell's local references went away, silently dropping menu entries
    (model/lang/overlay/export submenus and some actions vanished)."""
    import gc

    gc.collect()
    menu = shell.shell.menu
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    for key in (
        "tray_pause",
        "tray_show_panel",
        "quit",
    ):
        assert t(key) in texts, f"menu entry {key} dropped after GC"
    # the overlay toggle shows its current counter-state
    assert t("tray_show_overlay") in texts or t("tray_hide_overlay") in texts


def test_status_line_elides_to_keep_menu_narrow(shell):
    """The status line must never blow the menu wide: long status text is
    elided to the width cap; the full text stays in the tray tooltip."""
    from PyQt6.QtGui import QFontMetrics

    menu = shell.shell.menu
    menu.aboutToShow.emit()
    status = menu.actions()[0]
    width = QFontMetrics(menu.font()).horizontalAdvance(status.text())
    assert width <= 250, f"status line too wide: {width}px"
    assert t("tray_state_running") in shell.tray.toolTip()


def test_win32_uses_self_drawn_menu(shell):
    """On Windows the tray must not use the native context menu (it cannot
    follow the app theme); the activated signal drives left/right click."""
    assert shell.tray.contextMenu() is None
    assert shell.tray.receivers(shell.tray.activated) > 0


def test_quit_confirmation_cancels(monkeypatch, shell):
    calls: list[str] = []
    shell.app.quit = lambda: calls.append("quit")
    shell.live_trans.stop = lambda: calls.append("stop")
    monkeypatch.setattr(
        "livetranslate.ui.app_shell.QMessageBox.question",
        lambda *a, **k: QMessageBox.StandardButton.No,
    )
    shell.shell.on_quit()
    assert calls == [], "declined quit must not stop the app"


def test_quit_confirmation_accepts(monkeypatch, shell):
    calls: list[str] = []
    shell.app.quit = lambda: calls.append("quit")
    shell.live_trans.stop = lambda: calls.append("stop")
    monkeypatch.setattr(
        "livetranslate.ui.app_shell.QMessageBox.question",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )
    shell.shell.on_quit()
    assert calls == ["stop", "quit"]


def test_programmatic_quit_skips_confirmation(monkeypatch, shell):
    """Smoke/SIGINT exits must never block on a dialog."""
    calls: list[str] = []
    shell.app.quit = lambda: calls.append("quit")
    shell.live_trans.stop = lambda: calls.append("stop")
    monkeypatch.setattr(
        "livetranslate.ui.app_shell.QMessageBox.question",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("confirm must be skipped")),
    )
    shell.shell.on_quit(confirm=False)
    assert calls == ["stop", "quit"]
