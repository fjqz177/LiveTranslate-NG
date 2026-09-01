"""Tests for the Qt hotkey adapter (fake platform backend, no real
registration)."""

from livetranslate.platform.hotkeys import HotkeyCombo, HotkeyStatus
from livetranslate.ui.hotkeys import HotkeyManager


class FakeBackend:
    """Records registrations; callbacks fire on demand."""

    name = "fake"

    def __init__(self):
        self.registered: dict[str, HotkeyCombo] = {}
        self.callbacks: dict[str, object] = {}
        self.fail_reason: str | None = None

    def capability(self):
        return HotkeyStatus(ok=True)

    def register(self, name, combo, callback):
        if self.fail_reason:
            return HotkeyStatus(ok=False, reason=self.fail_reason)
        self.registered[name] = combo
        self.callbacks[name] = callback
        return HotkeyStatus(ok=True)

    def unregister(self, name):
        self.registered.pop(name, None)
        self.callbacks.pop(name, None)

    def stop(self):
        self.registered.clear()
        self.callbacks.clear()

    def fire(self, name):
        cb = self.callbacks.get(name)
        if cb is not None:
            cb()


class TestAdapter:
    def test_register_accepts_combo_strings(self):
        backend = FakeBackend()
        mgr = HotkeyManager(backend=backend)
        status = mgr.register("pause", "Ctrl+Alt+P")
        assert status.ok
        assert backend.registered["pause"] == HotkeyCombo.parse("Ctrl+Alt+P")

    def test_register_surfaces_failure_reason(self):
        backend = FakeBackend()
        backend.fail_reason = "already in use by another app"
        mgr = HotkeyManager(backend=backend)
        status = mgr.register("pause", "Ctrl+Alt+P")
        assert not status.ok
        assert status.reason == "already in use by another app"
        assert "pause" not in mgr._combos

    def test_unregister_all_clears_everything(self):
        backend = FakeBackend()
        mgr = HotkeyManager(backend=backend)
        mgr.register("pause", "Ctrl+Alt+P")
        mgr.register("clear", "Ctrl+Alt+C")
        mgr.unregister_all()
        assert backend.registered == {}
        assert mgr._combos == {}

    def test_stop_releases_backend(self):
        backend = FakeBackend()
        mgr = HotkeyManager(backend=backend)
        mgr.register("pause", "Ctrl+Alt+P")
        mgr.stop()
        assert backend.registered == {}
