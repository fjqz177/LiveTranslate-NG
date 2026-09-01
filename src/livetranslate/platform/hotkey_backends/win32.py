"""Windows global-hotkey backend: Win32 RegisterHotKey via ctypes.

Qt-free by design: a dedicated message-loop thread owns the registration
and delivers WM_HOTKEY to the registered callbacks (the Qt layer marshals
them onto the UI thread). Windows-only.
"""

from __future__ import annotations

import ctypes
import logging
import queue
import threading
import time
from ctypes import wintypes
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from livetranslate.platform.hotkeys import HotkeyCombo, HotkeyStatus

log = logging.getLogger("LiveTranslate.Hotkeys")

# Commands flowing to the message-loop thread: (kind, ...) tuples.
_Command = tuple[Any, ...]

WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000

_MOD_FLAGS = {"ctrl": 0x0002, "alt": 0x0001, "shift": 0x0004, "super": 0x0008}

_VK_KEYS = {
    **{chr(c): c for c in range(ord("A"), ord("Z") + 1)},
    **{str(d): ord(str(d)) for d in range(10)},
    **{f"F{i}": 0x70 + i - 1 for i in range(1, 25)},
    "SPACE": 0x20,
    "ENTER": 0x0D,
    "TAB": 0x09,
    "ESC": 0x1B,
    "BACKSPACE": 0x08,
}

_user32: Any = ctypes.windll.user32


def _mods_to_flags(mods: frozenset[str]) -> int:
    flags = 0
    for name, flag in _MOD_FLAGS.items():
        if name in mods:
            flags |= flag
    return flags


class Win32HotkeyBackend:
    """RegisterHotKey backend; callbacks fire on the backend thread."""

    name = "win32"

    def __init__(self) -> None:
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._replies: queue.Queue[tuple[HotkeyStatus, None]] = queue.Queue()
        self._callbacks: dict[int, tuple[str, Callable[[], None]]] = {}
        self._ids: dict[str, int] = {}
        self._next_id = 1
        self._thread: threading.Thread | None = None
        self._running = False

    # -- HotkeyBackend ------------------------------------------------------

    def capability(self) -> HotkeyStatus:
        return HotkeyStatus(ok=True)

    def register(self, name: str, combo: HotkeyCombo, callback: Callable[[], None]) -> HotkeyStatus:
        if combo.key not in _VK_KEYS:
            return HotkeyStatus(ok=False, reason=f"unsupported key: {combo.key}")
        self.unregister(name)
        self._start_thread()
        self._commands.put(("register", name, combo, callback))
        try:
            status = self._replies.get(timeout=5)[0]
        except queue.Empty:
            return HotkeyStatus(ok=False, reason="hotkey backend not responding")
        return status

    def unregister(self, name: str) -> None:
        if name in self._ids:
            self._commands.put(("unregister", name))

    def stop(self) -> None:
        self._running = False
        self._commands.put(("stop",))
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # -- internals ----------------------------------------------------------

    def _start_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._message_loop, name="hotkeys", daemon=True)
        self._thread.start()

    def _message_loop(self) -> None:
        # RegisterHotKey(NULL, ...) posts WM_HOTKEY to this thread's queue.
        # Poll with PeekMessageW so queued register/unregister commands are
        # serviced within ~10 ms instead of blocking on GetMessageW.
        msg = wintypes.MSG()
        while self._running:
            self._drain_commands()
            while _user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                if msg.message == WM_HOTKEY:
                    self._dispatch(msg.wParam)
                _user32.TranslateMessage(ctypes.byref(msg))
                _user32.DispatchMessageW(ctypes.byref(msg))
            time.sleep(0.01)

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return
            kind = cmd[0]
            if kind == "register":
                _, name, combo, callback = cmd
                hk_id = self._next_id
                self._next_id += 1
                flags = _mods_to_flags(combo.mods) | MOD_NOREPEAT
                ok = bool(_user32.RegisterHotKey(None, hk_id, flags, _VK_KEYS[combo.key]))
                if ok:
                    self._callbacks[hk_id] = (name, callback)
                    self._ids[name] = hk_id
                    log.info(f"Hotkey registered: {name} ({combo})")
                    self._replies.put((HotkeyStatus(ok=True), None))
                else:
                    self._replies.put(
                        (
                            HotkeyStatus(ok=False, reason="already in use by another app"),
                            None,
                        )
                    )
            elif kind == "unregister":
                hk_id = self._ids.pop(cmd[1], -1)
                if hk_id != -1:
                    _user32.UnregisterHotKey(None, hk_id)
                    self._callbacks.pop(hk_id, None)
            elif kind == "stop":
                return

    def _dispatch(self, hk_id: int) -> None:
        entry = self._callbacks.get(hk_id)
        if entry is None:
            return
        name, callback = entry
        log.debug(f"Hotkey pressed: {name}")
        try:
            callback()
        except Exception:
            log.exception(f"Hotkey callback failed: {name}")
