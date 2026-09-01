"""Tests for core.privacy redaction (pure stdlib, no third-party imports).

Guard the masking rules that keep API keys, credential-bearing URLs, user
paths and transcript content out of logs and the diagnostics bundle. If any
assertion goes RED on a correct baseline it is a residual leak to fix, not an
assertion to loosen.
"""

from __future__ import annotations

from livetranslate.core.privacy import (
    mask_proxy_url,
    redact_dict,
    redact_settings,
    redact_text,
)


class TestRedactText:
    def test_masks_api_key_value(self):
        assert "sk-abcdefg" not in redact_text("using sk-abcdefg12345xyz")
        assert "***" in redact_text("using sk-abcdefg12345xyz")

    def test_masks_bearer_token(self):
        assert "Bearer abcdef12345" != redact_text("Bearer abcdef12345").replace(
            "***", " abcdef12345"
        )
        out = redact_text("Authorization: Bearer AbC_D-12345678")
        assert "AbC_D-12345678" not in out
        assert "***" in out

    def test_masks_url_userinfo(self):
        out = redact_text("http://user:pw@host:1/seg")
        assert "user:pw" not in out
        assert "***@host:1" in out

    def test_masks_windows_user_path(self):
        out = redact_text("C:\\Users\\someone\\dir\\file.txt")
        assert "someone" not in out
        assert "<user>/..." in out

    def test_masks_mac_and_linux_home(self):
        assert "/Users/someone" not in redact_text("/Users/someone/x")
        assert "/home/someone" not in redact_text("/home/someone/x")

    def test_strips_transcript_content_line(self):
        out = redact_text("ASR content: hello world, this is speech")
        assert "hello world" not in out
        assert out.endswith("***")

    def test_transcript_line_keeps_label_prefix(self):
        out = redact_text("Translate (120ms): bonjour tout le monde")
        assert out.startswith("Translate (120ms): ***")

    def test_keeps_ordinary_text(self):
        text = "Speech segment: 1.5s"
        assert redact_text(text) == text


class TestRedactDict:
    def test_masks_secret_named_values(self):
        out = redact_dict({"api_key": "sk-verysecret123", "model": "whisper"})
        assert "sk-verysecret123" not in out["api_key"]
        assert out["api_key"].startswith("sk-v")
        assert out["model"] == "whisper"

    def test_recurses_into_nested_dicts(self):
        out = redact_dict({"nested": {"token": "tok12345", "ok": "ok"}})
        assert "tok12345" not in out["nested"]["token"]
        assert out["nested"]["ok"] == "ok"

    def test_preserves_non_string_values(self):
        out = redact_dict({"count": 3, "flag": True})
        assert out["count"] == 3
        assert out["flag"] is True


class TestMaskProxyUrl:
    def test_masks_proxy_credentials(self):
        assert mask_proxy_url("http://user:pw@host:1") == "http://***@host:1"

    def test_leaves_plain_proxy(self):
        assert mask_proxy_url("http://host:1") == "http://host:1"


class TestRedactSettings:
    def test_drops_excluded_and_masks_secrets(self):
        snapshot = {
            "model": "whisper",
            "system_prompt": "you are a translator",
            "remote_asr_token": "topsecret",
            "models": {"a": 1},
        }
        out = redact_settings(snapshot, exclude=("models", "system_prompt"))
        assert "system_prompt" not in out
        assert "models" not in out
        assert "topsecret" not in out["remote_asr_token"]
        assert out["model"] == "whisper"
