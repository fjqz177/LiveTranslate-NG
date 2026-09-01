"""Tests for diagnostics collection and redaction."""

from pathlib import Path

from livetranslate.core.diagnostics import (
    collect_summary,
    redact_dict,
    redact_text,
    write_redacted_zip,
)
from livetranslate.core.paths import PROJECT_ROOT


class TestRedactText:
    def test_api_keys_masked(self):
        out = redact_text("key sk-abcdef123456 sent to https://api.example.com")
        assert "sk-abcdef123456" not in out
        assert "***" in out

    def test_windows_user_paths_collapsed(self):
        out = redact_text(r"loaded from C:\Users\zhangsan\models\x")
        assert "zhangsan" not in out
        assert "<user>/..." in out

    def test_mac_user_paths_collapsed(self):
        out = redact_text(r"loaded from /Users/lisi/models/x")
        assert "lisi" not in out
        assert "<user>/..." in out

    def test_plain_text_untouched(self):
        assert redact_text("pipeline started") == "pipeline started"


class TestProxyCredentialMasking:
    """CORE-4: proxy URLs often embed user:pass — they must never reach a
    diagnostics bundle."""

    def test_url_userinfo_masked(self):
        out = redact_text("Download proxy active: http://user:secret123@proxy.local:1080")
        assert "user:secret123" not in out
        assert "http://***@proxy.local:1080" in out

    def test_url_without_credentials_untouched(self):
        out = redact_text("proxy http://proxy.local:1080")
        assert "proxy.local" in out
        assert "***" not in out


class TestRuntimePathFolding:
    """CORE-3: the dev-default data root is the repo itself — the old
    Windows-user regex never matched it, so folding had to become root-aware."""

    def test_linux_home_folded(self):
        out = redact_text("loaded /home/zhangsan/.cache/models/x")
        assert "zhangsan" not in out
        assert "<user>/..." in out

    def test_explicit_data_root_folded(self):
        root = str(PROJECT_ROOT).rstrip("\\/")
        out = redact_text(f"models at {root}\\models\\faster-whisper-small")
        assert "biancheng" not in out
        assert "<data_root>/..." in out

    def test_repo_root_folded_after_data_root(self):
        root = str(PROJECT_ROOT).rstrip("\\/")
        out = redact_text(f"{root}\\settings.json")
        assert "biancheng" not in out
        assert "<data_root>/..." in out or "<repo>/..." in out

    def test_explicit_roots_take_precedence(self):
        out = redact_text(
            r"cache C:\apps\lt\models and repo C:\apps\lt",
            extra_roots=(r"C:\apps\lt",),
        )
        assert "apps" not in out


class TestRedactTranscriptLines:
    """SEC-1 defense in depth: legacy/opt-in log lines with speech or
    translation content must be stripped inside diagnostics bundles even if
    the pipeline call-site gate was bypassed (old logs, debug builds)."""

    def test_legacy_asr_line_stripped(self):
        out = redact_text("2026-08-20 10:00:00 [INFO] LiveTranslate: ASR [en] (123ms): hello world")
        assert "hello world" not in out
        assert "ASR [en] (123ms):" in out
        assert "***" in out

    def test_legacy_interim_line_stripped(self):
        out = redact_text("ASR [ja] (45ms, interim): こんにちは世界")
        assert "こんにちは世界" not in out

    def test_legacy_translate_line_stripped(self):
        out = redact_text("Translate (456ms): 你好世界")
        assert "你好世界" not in out

    def test_legacy_extra_translate_line_stripped(self):
        out = redact_text("Extra translate [ja]: こんにちは")
        assert "こんにちは" not in out

    def test_opt_in_content_lines_stripped(self):
        for line in (
            "ASR content: secret speech",
            "ASR interim content: partial speech",
            "Translate content: secret translation",
            "Extra translate content: more",
        ):
            out = redact_text(line)
            assert "secret speech" not in out
            assert out.endswith("***")

    def test_metadata_lines_are_stripped_too(self):
        # The regex cannot tell "12 chars" (metadata) from real content, so
        # every transcript-shaped line is collapsed — conservative by design.
        out = redact_text("ASR [en] (123ms): 12 chars")
        assert out == "ASR [en] (123ms): ***"


class TestRedactDict:
    def test_secret_keys_masked(self):
        out = redact_dict({"api_key": "sk-1234567890", "name": "deepseek"})
        assert out["api_key"].endswith("***")
        assert out["name"] == "deepseek"

    def test_nested(self):
        out = redact_dict({"models": [{"api_key": "secret", "name": "m"}]})
        assert out["models"][0]["api_key"].endswith("***")

    def test_non_string_values_untouched(self):
        out = redact_dict({"api_key": 42, "n": 1})
        assert out == {"api_key": 42, "n": 1}


class TestSummary:
    def test_minimal_summary(self, monkeypatch):
        monkeypatch.setattr("livetranslate.core.diagnostics.LOG_DIR", Path("/nonexistent"))
        summary = collect_summary(
            platform="win32", accelerator=None, audio_diag=None, hotkey_status=None
        )
        assert summary["app"] == "LiveTranslate"
        assert summary["platform"] == "win32"
        assert summary["engine"] == "unset"


class TestRedactedZip:
    def test_zip_contains_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr("livetranslate.core.diagnostics.LOG_DIR", Path("/nonexistent"))
        out = tmp_path / "diag.zip"
        report = write_redacted_zip(out)
        assert out.exists()
        assert report["files"] == 1  # summary.json only
        assert report["size_bytes"] > 0

    def test_subtitle_included_only_on_opt_in(self, tmp_path, monkeypatch):
        monkeypatch.setattr("livetranslate.core.diagnostics.LOG_DIR", Path("/nonexistent"))
        out = tmp_path / "diag.zip"
        report = write_redacted_zip(out, include_recent_subtitle=True, recent_subtitle="hello")
        assert report["files"] == 2
