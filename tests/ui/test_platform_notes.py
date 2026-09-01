"""§3.7 platform degradation notes tests (detect-degrade-guide)."""

from PyQt6.QtWidgets import QSystemTrayIcon

from livetranslate.ui.platform_notes import platform_notes


def test_no_degradations_on_normal_desktop(qapp, monkeypatch):
    assert qapp is not None
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", staticmethod(lambda: True))
    assert platform_notes() == []


def test_missing_tray_note(qapp, monkeypatch):
    assert qapp is not None
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", staticmethod(lambda: False))
    assert platform_notes() == ["err_gnome_tray"]
