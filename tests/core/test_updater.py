"""GitHub release check tests (plan §4.9 rewrite): version comparison and
the HTTP result mapping. httpx is monkeypatched — no real network.
"""

from __future__ import annotations

import httpx
import pytest

from livetranslate.core.updater import (
    _normalize_tag,
    check_latest_release,
)


class _FakeResponse:
    def __init__(self, payload: object = None, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        # Match httpx semantics: 4xx/5xx raises an httpx.HTTPError subclass.
        if self.status_code >= 400:
            raise httpx.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _release(**overrides) -> dict:
    base = {
        "tag_name": "v1.2.0",
        "body": "fixes",
        "html_url": "https://github.com/fjqz177/LiveTranslate/releases/tag/v1.2.0",
    }
    base.update(overrides)
    return base


def _fake_get(payload: object, status: int = 200):
    return lambda url, timeout=None, follow_redirects=True: _FakeResponse(payload, status)


class TestNormalizeTag:
    def test_strips_leading_v(self):
        assert _normalize_tag("v1.2.3") == "1.2.3"
        assert _normalize_tag("1.2.3") == "1.2.3"
        assert _normalize_tag("V1.2.3-beta.1") == "1.2.3-beta.1"

    @pytest.mark.parametrize("tag", ["", "not-a-version"])
    def test_non_tag_is_none(self, tag):
        assert _normalize_tag(tag) is None


class TestCheckLatestRelease:
    def test_newer_version(self, monkeypatch):
        monkeypatch.setattr("livetranslate.core.updater.httpx.get", _fake_get(_release()))
        result = check_latest_release("1.0.0")
        assert result.kind == "new"
        assert result.new_version == "1.2.0"
        assert result.url == "https://github.com/fjqz177/LiveTranslate/releases/latest"
        assert result.notes == "fixes"

    def test_up_to_date(self, monkeypatch):
        monkeypatch.setattr(
            "livetranslate.core.updater.httpx.get", _fake_get(_release(tag_name="v1.0.0"))
        )
        assert check_latest_release("1.0.0").kind == "uptodate"

    def test_current_is_noop(self, monkeypatch):
        monkeypatch.setattr(
            "livetranslate.core.updater.httpx.get", _fake_get(_release(tag_name="v1.2.0"))
        )
        assert check_latest_release("1.3.0").kind == "uptodate"

    def test_no_release_404(self, monkeypatch):
        monkeypatch.setattr("livetranslate.core.updater.httpx.get", _fake_get(None, status=404))
        assert check_latest_release("1.0.0").kind == "none"

    def test_http_error(self, monkeypatch):
        monkeypatch.setattr("livetranslate.core.updater.httpx.get", _fake_get(None, status=500))
        result = check_latest_release("1.0.0")
        assert result.kind == "error"
        assert "HTTP" in result.detail

    def test_bad_json(self, monkeypatch):
        monkeypatch.setattr("livetranslate.core.updater.httpx.get", _fake_get(None))
        result = check_latest_release("1.0.0")
        assert result.kind == "error"
        assert "bad JSON" in result.detail

    def test_unversioned_tag(self, monkeypatch):
        monkeypatch.setattr(
            "livetranslate.core.updater.httpx.get", _fake_get(_release(tag_name="nonsense"))
        )
        assert check_latest_release("1.0.0").kind == "error"

    def test_httpx_raises(self, monkeypatch):
        def _boom(*_a, **_k):
            raise httpx.ConnectError("boom")

        monkeypatch.setattr("livetranslate.core.updater.httpx.get", _boom)
        assert check_latest_release("1.0.0").kind == "error"
