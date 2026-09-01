"""Windows SystemIntegration: open/autostart/single-instance/accelerator."""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from ctypes import wintypes
from typing import TYPE_CHECKING

from livetranslate.core.systeminfo import detect_accelerator

if TYPE_CHECKING:
    from pathlib import Path

    from livetranslate.platform.system import AcceleratorInfo

_AUTOSTART_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_AUTOSTART_NAME = "LiveTranslate"

_MUTEX_PREFIX = "Local\\LiveTranslate-"
_ERROR_ALREADY_EXISTS = 183


def _create_named_mutex(name: str) -> tuple[int, bool]:
    """Create/open a named mutex. Returns (handle, already_existed)."""
    # use_last_error=True pins the Win32 error code across the call —
    # without it, ctypes' own bookkeeping clobbers GetLastError and the
    # ALREADY_EXISTS probe silently misreads.
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    k32.CreateMutexW.restype = wintypes.HANDLE
    handle = k32.CreateMutexW(None, False, name)
    return int(handle), ctypes.get_last_error() == _ERROR_ALREADY_EXISTS


def _close_handle(handle: int) -> None:
    """CloseHandle with explicit argtypes — the default c_int marshalling
    truncates 64-bit handles and silently fails to release the mutex."""
    k32 = ctypes.windll.kernel32
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    k32.CloseHandle(handle)


def _launch_command() -> str:
    """Command registered for login autostart."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    exe = sys.executable
    return f'"{exe}" -m livetranslate'


class Win32SystemIntegration:
    """Windows implementation of the SystemIntegration protocol."""

    name = "win32"

    def __init__(self) -> None:
        # Windows gates single-instance with a named mutex instead of a
        # lock file (see try_acquire_single_instance).
        self._mutex_handle: int = 0

    # -- files ---------------------------------------------------------------

    def open_path(self, path: Path) -> None:
        os.startfile(str(path))  # Windows-only; args fixed

    # -- autostart -----------------------------------------------------------

    def set_autostart(self, enabled: bool) -> None:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _AUTOSTART_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            if enabled:
                winreg.SetValueEx(key, _AUTOSTART_NAME, 0, winreg.REG_SZ, _launch_command())
            else:
                with contextlib.suppress(FileNotFoundError):
                    winreg.DeleteValue(key, _AUTOSTART_NAME)

    def autostart_enabled(self) -> bool:
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, _AUTOSTART_KEY, 0, winreg.KEY_QUERY_VALUE
            ) as key:
                winreg.QueryValueEx(key, _AUTOSTART_NAME)
                return True
        except FileNotFoundError:
            return False

    # -- single instance ------------------------------------------------------

    def try_acquire_single_instance(self, key: str) -> bool:
        """Claim the process-wide named mutex.

        The kernel destroys the mutex when the owning process exits, so a
        crashed or killed instance can never leave a stale lock behind.
        (The previous PID lock file mis-judged liveness under PID reuse:
        a stale lock whose PID had been recycled by another process blocked
        every relaunch.)
        """
        handle, existed = _create_named_mutex(_MUTEX_PREFIX + key)
        if not handle:
            # Unexpected creation failure: fail open — the app still runs,
            # worst case two instances, never a blocked one.
            return True
        if existed:
            # Another instance holds the mutex: close our duplicate handle
            # (leaking it would keep the mutex alive after that instance
            # exits) and report the loss.
            _close_handle(handle)
            return False
        self._mutex_handle = handle
        return True

    def release_single_instance(self) -> None:
        if self._mutex_handle:
            _close_handle(self._mutex_handle)
            self._mutex_handle = 0

    # -- accelerator -----------------------------------------------------------

    def accelerator(self) -> AcceleratorInfo:
        return detect_accelerator()
