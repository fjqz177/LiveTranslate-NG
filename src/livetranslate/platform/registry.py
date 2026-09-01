"""Windows-only platform dispatch for the integration backends.

The Win32 backends are imported directly; no sys.platform dispatch remains.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livetranslate.platform.hotkeys import HotkeyBackend
    from livetranslate.platform.system import SystemIntegration


def create_hotkey_backend() -> HotkeyBackend:
    """Windows-only: return the Win32 backend directly."""
    from livetranslate.platform.hotkey_backends.win32 import Win32HotkeyBackend

    return Win32HotkeyBackend()


def create_system_integration() -> SystemIntegration:
    """Windows-only: return the Win32 system integration directly."""
    from livetranslate.platform.system_backends.win32 import Win32SystemIntegration

    return Win32SystemIntegration()
