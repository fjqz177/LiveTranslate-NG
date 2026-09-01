"""Tests for livetranslate.core.settings — pure persistence, no Qt."""

import json

import livetranslate.core.settings as settings_module
from livetranslate.core.settings import (
    load_settings_from,
    save_settings_to,
)


class TestLoadSaveHelpers:
    def test_load_missing_file_returns_none(self, tmp_path):
        assert load_settings_from(tmp_path / "nope.json") is None

    def test_save_then_load_roundtrip(self, tmp_path):
        path = tmp_path / "s.json"
        save_settings_to(path, {"a": 1})
        assert load_settings_from(path) == {"a": 1}

    def test_save_is_atomic(self, tmp_path):
        path = tmp_path / "s.json"
        save_settings_to(path, {"a": 1})
        assert not path.with_suffix(".tmp").exists()
        assert path.exists()

    def test_load_malformed_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_settings_from(path) is None

    def test_load_valid_non_object_json_returns_none(self, tmp_path):
        """CORE-5: valid-but-not-an-object JSON used to crash callers with
        AttributeError; it must degrade to defaults like syntax corruption."""
        for i, payload in enumerate(("[]", '"x"', "42", "null")):
            path = tmp_path / f"nonobj{i}.json"
            path.write_text(payload, encoding="utf-8")
            assert load_settings_from(path) is None, payload

    def test_save_leaves_no_temp_file_behind(self, tmp_path):
        path = tmp_path / "s.json"
        save_settings_to(path, {"a": 1})
        assert list(tmp_path.glob("*.tmp")) == []
        assert path.exists()

    def test_save_temp_name_is_pid_suffixed(self, tmp_path, monkeypatch):
        """SEC-2: the temp name must not be a predictable fixed target."""
        monkeypatch.setattr(settings_module.os, "getpid", lambda: 12345)
        path = tmp_path / "s.json"
        save_settings_to(path, {"a": 1})
        assert path.exists()
        assert not (tmp_path / "s.json.tmp").exists()

    def test_save_sets_0600_on_posix(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(settings_module.os, "name", "posix")
        monkeypatch.setattr(
            settings_module.os, "chmod", lambda p, mode: calls.append((str(p), mode))
        )
        path = tmp_path / "s.json"
        save_settings_to(path, {"a": 1})
        assert calls and calls[0][1] == 0o600


class TestUserSettingsEntryPoints:
    @staticmethod
    def _isolate(tmp_path, monkeypatch):
        """Point the canonical, legacy and pre-unification platform paths at
        tmp files so the real machine state can never leak into these tests."""
        settings = tmp_path / "settings.json"
        legacy = tmp_path / "user_settings.json"
        monkeypatch.setattr(settings_module, "SETTINGS_FILE", settings)
        monkeypatch.setattr(settings_module, "LEGACY_SETTINGS_FILE", legacy)
        # Anchors the dev-layout migration gate (repo-root settings file).
        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            settings_module,
            "_platform_settings_path",
            lambda: tmp_path / "platform" / "settings.json",
        )
        return settings, legacy

    def test_load_user_settings_migrates_and_returns(self, tmp_path, monkeypatch):
        settings, _ = self._isolate(tmp_path, monkeypatch)
        settings.write_text(json.dumps({"asr_engine": "sensevoice"}), encoding="utf-8")
        data = settings_module.load_user_settings()
        assert data["asr_engine"] == "funasr"
        assert data["funasr_model"] == "sensevoice-small"

    def test_load_user_settings_missing_file_returns_none(self, tmp_path, monkeypatch):
        self._isolate(tmp_path, monkeypatch)
        assert settings_module.load_user_settings() is None

    def test_legacy_file_is_copied_on_first_load(self, tmp_path, monkeypatch):
        settings, legacy = self._isolate(tmp_path, monkeypatch)
        legacy.write_text(json.dumps({"target_language": "ja"}), encoding="utf-8")
        data = settings_module.load_user_settings()
        # migration is one-way: new file created, legacy kept in place
        assert data["target_language"] == "ja"
        assert json.loads(settings.read_text(encoding="utf-8"))["target_language"] == "ja"
        assert legacy.exists()

    def test_legacy_migration_never_overwrites_existing_settings(self, tmp_path, monkeypatch):
        settings, legacy = self._isolate(tmp_path, monkeypatch)
        settings.write_text(json.dumps({"target_language": "en"}), encoding="utf-8")
        legacy.write_text(json.dumps({"target_language": "ja"}), encoding="utf-8")
        data = settings_module.load_user_settings()
        assert data["target_language"] == "en"

    def test_platformdirs_settings_migrated_on_first_load(self, tmp_path, monkeypatch):
        settings, _ = self._isolate(tmp_path, monkeypatch)
        platform = tmp_path / "platform" / "settings.json"
        platform.parent.mkdir()
        platform.write_text(json.dumps({"target_language": "ja"}), encoding="utf-8")
        data = settings_module.load_user_settings()
        # one-way copy: new file created, original kept in place as backup
        assert data["target_language"] == "ja"
        assert json.loads(settings.read_text(encoding="utf-8"))["target_language"] == "ja"
        assert platform.exists()

    def test_platformdirs_migration_prefers_platform_over_legacy(self, tmp_path, monkeypatch):
        settings, legacy = self._isolate(tmp_path, monkeypatch)
        platform = tmp_path / "platform" / "settings.json"
        platform.parent.mkdir()
        platform.write_text(json.dumps({"target_language": "ja"}), encoding="utf-8")
        legacy.write_text(json.dumps({"target_language": "ko"}), encoding="utf-8")
        data = settings_module.load_user_settings()
        # the platformdirs file is fresher than the pre-rewrite legacy file
        assert data["target_language"] == "ja"
        assert json.loads(settings.read_text(encoding="utf-8"))["target_language"] == "ja"

    def test_platformdirs_migration_never_overwrites_existing_settings(self, tmp_path, monkeypatch):
        settings, _ = self._isolate(tmp_path, monkeypatch)
        settings.write_text(json.dumps({"target_language": "en"}), encoding="utf-8")
        platform = tmp_path / "platform" / "settings.json"
        platform.parent.mkdir()
        platform.write_text(json.dumps({"target_language": "ja"}), encoding="utf-8")
        data = settings_module.load_user_settings()
        assert data["target_language"] == "en"

    def test_save_user_settings_roundtrip(self, tmp_path, monkeypatch):
        settings, _ = self._isolate(tmp_path, monkeypatch)
        settings_module.save_user_settings({"target_language": "ja"})
        assert json.loads(settings.read_text(encoding="utf-8"))["target_language"] == "ja"

    def test_settings_paths_are_anchored_to_the_project_root(self, monkeypatch):
        """The unified dev layout keeps both files in the repo root.

        Reloads under a clean environment so a polluted shell (exported
        LIVETRANSLATE_PLATFORM_DIRS/PORTABLE vars) cannot skew the assertion.
        """
        import importlib

        import livetranslate.core.paths as paths

        monkeypatch.delenv("LIVETRANSLATE_PORTABLE_DIR", raising=False)
        monkeypatch.delenv("LIVETRANSLATE_PORTABLE", raising=False)
        monkeypatch.delenv("LIVETRANSLATE_PLATFORM_DIRS", raising=False)
        importlib.reload(paths)
        importlib.reload(settings_module)
        assert settings_module.SETTINGS_FILE == settings_module.PROJECT_ROOT / "settings.json"
        assert (
            settings_module.LEGACY_SETTINGS_FILE
            == settings_module.PROJECT_ROOT / "user_settings.json"
        )
