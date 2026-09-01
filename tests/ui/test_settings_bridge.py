"""Tests for livetranslate.ui.settings_bridge — the Qt wrapper of the
pure settings persistence layer."""

import json

import pytest

from livetranslate.ui.settings_bridge import SettingsStore


@pytest.fixture
def store(tmp_path):
    return SettingsStore(tmp_path / "user_settings.json")


class TestStoreBasics:
    def test_load_missing_file_gives_empty(self, store):
        assert store.load() == {}
        assert store.exists() is False

    def test_load_from_disk(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"target_language": "ja"}), encoding="utf-8")
        store = SettingsStore(path)
        data = store.load()
        assert data["target_language"] == "ja"
        # migration normalizes the engine selection on every load
        assert data["asr_engine"] == "funasr"
        assert data["funasr_model"] == "sensevoice-small"

    def test_load_migrates_legacy_engine(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"asr_engine": "sensevoice"}), encoding="utf-8")
        store = SettingsStore(path)
        data = store.load()
        assert data["asr_engine"] == "funasr"
        assert data["funasr_model"] == "sensevoice-small"

    def test_seed_skips_disk(self, store):
        store.seed({"asr_engine": "sensevoice"})
        assert store.snapshot()["asr_engine"] == "funasr"
        assert store.exists() is False  # never persisted

    def test_snapshot_is_a_copy(self, store):
        store.update({"a": 1})
        snap = store.snapshot()
        snap["a"] = 999
        assert store.get("a") == 1

    def test_no_live_dict_or_changed_signal(self, store):
        # M-SETTINGS: the store no longer exposes a live dict nor a vestigial
        # per-key changed signal — it never hands out its internal state.
        assert not hasattr(store, "data")
        assert not hasattr(store, "changed")


class TestStoreBulkCommit:
    def test_update_persists_and_is_bulk(self, store):
        store.update({"a": 1, "b": {"nested": True}})
        snap = store.snapshot()
        assert snap["a"] == 1
        assert snap["b"] == {"nested": True}
        on_disk = json.loads(store.path.read_text(encoding="utf-8"))
        assert on_disk["a"] == 1
        assert on_disk["b"] == {"nested": True}

    def test_update_replaces_not_mutates(self, store):
        store.update({"a": 1})
        held = store.snapshot()
        held_id = id(held)
        store.update({"a": 2})
        # A previously held snapshot is untouched; the store rebound a fresh
        # dict instead of mutating the old one.
        assert held["a"] == 1
        assert id(held) == held_id
        assert store.snapshot()["a"] == 2

    def test_update_empty_returns_false(self, store):
        store.update({"a": 1})
        assert store.update({}) is False

    def test_update_without_save(self, store):
        store.update({"a": 1}, save=False)
        assert store.get("a") == 1
        assert store.path.exists() is False
