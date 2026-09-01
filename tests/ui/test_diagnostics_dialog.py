"""Offscreen smoke test for the diagnostics dialog."""

from livetranslate.ui.diagnostics import DiagnosticsDialog


class FakeAudio:
    name = "wasapi"

    def diagnostics(self):
        return {
            "backend": "wasapi",
            "device": None,
            "rate": 16000,
            "status": "stopped",
            "last_error": None,
        }


class FakeApp:
    def __init__(self):
        self._audio = FakeAudio()
        self._hotkeys = None
        self._asr_ctl = None
        self._recent_errors = ["err_401 copy"]

    def get_settings(self):
        return {"api_key": "sk-secret-value", "target_language": "zh"}


def test_dialog_builds_all_cards(qapp):
    assert qapp is not None  # pytest-qt fixture (offscreen QApplication)
    dlg = DiagnosticsDialog(FakeApp())
    assert dlg.windowTitle()
    # eight cards (platform/network/audio/hotkeys/permissions/accelerator/
    # storage/logs) + stretch
    assert dlg._cards.count() == 9
    dlg._refresh_summary()
    assert dlg._summary["app"] == "LiveTranslate"
    # settings secrets are masked in the summary
    assert "sk-secret-value" not in str(dlg._summary)
    dlg.close()


def test_network_card_shows_recent_errors(qapp):
    from PyQt6.QtWidgets import QGroupBox, QLabel

    from livetranslate.core.i18n import t

    assert qapp is not None
    dlg = DiagnosticsDialog(FakeApp())
    view = dlg._view
    cards = []
    for i in range(view._cards.count()):
        widget = view._cards.itemAt(i).widget()
        if isinstance(widget, QGroupBox):
            cards.append(widget)
    network = next(c for c in cards if c.title() == t("diag_network"))
    labels = [label.text() for label in network.findChildren(QLabel)]
    assert any("err_401 copy" in text for text in labels)
    dlg.close()


def test_platform_card_reports_tray_unavailable(qapp, monkeypatch):
    from PyQt6.QtWidgets import QGroupBox, QLabel, QSystemTrayIcon

    from livetranslate.core.i18n import t

    assert qapp is not None
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", staticmethod(lambda: False))
    dlg = DiagnosticsDialog(FakeApp())
    view = dlg._view
    cards = [
        view._cards.itemAt(i).widget()
        for i in range(view._cards.count())
        if isinstance(view._cards.itemAt(i).widget(), QGroupBox)
    ]
    platform = next(c for c in cards if c.title() == t("diag_platform"))
    labels = [label.text() for label in platform.findChildren(QLabel)]
    assert t("diag_no") in labels
    dlg.close()
