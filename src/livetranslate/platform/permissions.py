"""Permission state queries and system-settings navigation (Windows-only).

LiveTranslate targets Windows 10/11, where these user-facing permission
gates (microphone / screen recording / accessibility) do not apply. The
diagnostics panel renders a permission card on every platform via the
neutral ``PermissionStatus`` protocol; Windows supplies
``NullPermissionStatus`` (everything granted, no-op navigation).
"""

from __future__ import annotations

from typing import Literal, Protocol

PermissionState = Literal["granted", "denied", "unknown"]


class PermissionStatus(Protocol):
    """Query and guide OS permission grants."""

    name: str

    def microphone(self) -> PermissionState: ...
    def screen_recording(self) -> PermissionState: ...
    def accessibility(self) -> PermissionState: ...

    def open_system_settings(self, pane: str) -> None:
        """Jump to the OS settings page for a permission ('microphone' /
        'screen-recording' / 'accessibility')."""
        ...


class NullPermissionStatus:
    """Platforms without permission gates: everything is granted, no-op
    navigation."""

    name = "null"

    def microphone(self) -> PermissionState:
        return "granted"

    def screen_recording(self) -> PermissionState:
        return "granted"

    def accessibility(self) -> PermissionState:
        return "granted"

    def open_system_settings(self, pane: str) -> None:
        pass
