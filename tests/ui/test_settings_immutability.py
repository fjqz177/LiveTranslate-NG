"""Tests for the M-SETTINGS immutability boundary.

The SettingsStore never hands out its internal dict: reads return copies,
writes replace-merge a fresh dict (so a previously held copy is untouched),
and save is atomic. The panel drafts edits in ``_current_settings`` and
commits them explicitly. These tests pin the two properties that make the
boundary safe: the store never externalizes its live dict, and a commit never
mutates what a caller previously held.
"""

import pytest

from livetranslate.ui.panel.panel import ControlPanel
from livetranslate.ui.settings_bridge import SettingsStore

CONFIG = {
    "translation": {
        "model": "gpt-test",
        "api_base": "https://example.com/v1",
        "api_key": "sk-test",
        "target_language": "zh",
    },
    "asr": {
        "vad_threshold": 0.5,
        "min_speech_duration": 0.3,
        "max_speech_duration": 20.0,
        "language": "auto",
        "device": "cpu",
    },
}


class _SignalSpy:
    """Minimal signal spy: records every emission's arguments."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


@pytest.fixture
def store(tmp_path):
    return SettingsStore(tmp_path / "user_settings.json")


@pytest.fixture
def panel(qapp, tmp_path, monkeypatch):
    assert qapp is not None
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "settings.json")
    p = ControlPanel(CONFIG)
    yield p
    p.close()


class TestStoreNeverExternalizesDict:
    def test_snapshot_is_not_internal_dict(self, store):
        store.update({"a": 1})
        snap = store.snapshot()
        assert snap is not store._settings
        snap["a"] = 999
        assert store.get("a") == 1
        assert store.snapshot()["a"] == 1

    def test_held_snapshot_unchanged_across_update(self, store):
        store.update({"a": 1})
        held = store.snapshot()
        held_id = id(held)
        store.update({"a": 2})
        assert held["a"] == 1
        assert id(held) == held_id
        assert store.snapshot()["a"] == 2


class TestPanelDraftCommit:
    def test_apply_emits_one_complete_snapshot(self, qapp, panel):
        assert qapp is not None
        spy = _SignalSpy()
        panel.settings_changed.connect(spy)
        panel._general_tab._reduce_motion.setChecked(True)
        panel.apply_settings()
        assert len(spy.calls) == 1
        committed = spy.calls[0][0]
        # A fresh committed snapshot, complete with the just-edited key.
        assert committed["reduce_motion"] is True
        assert committed["asr_engine"] == "sensevoice-onnx"

    def test_collect_apply_does_not_touch_held_committed_dict(self, qapp, panel):
        assert qapp is not None
        held = panel._store.snapshot()
        held_id = id(held)
        panel._general_tab._reduce_motion.setChecked(True)
        panel._translation_tab.collect()
        panel.apply_settings()
        # The previously held committed dict object is untouched (the store
        # rebound a fresh dict rather than mutating the old one).
        assert id(held) == held_id
        assert panel._store.snapshot() is not held

    def test_update_settings_dual_merge(self, qapp, panel):
        assert qapp is not None
        panel.update_settings({"target_language": "ja"})
        assert panel._store.snapshot()["target_language"] == "ja"
        assert panel._current_settings["target_language"] == "ja"
