"""CORE-6/7/8 regression tests: the Translator's shared state is thread-safe,
the stream_options fallback only retries BadRequest, and json_response uses
the portable json_object mode unless strict mode is explicitly requested."""

import threading

import pytest
from openai import BadRequestError

import livetranslate.core.translator as translator_module
from livetranslate.core.translator import Translator


class _Completions:
    """Attribute-style chain mirroring openai's client.chat.completions."""

    def __init__(self, client):
        self.client = client

    def create(self, **kwargs):
        self.client.calls.append(kwargs)
        return []


class _Chat:
    def __init__(self, client):
        self.completions = _Completions(client)


class _DummyClient:
    def __init__(self):
        self.calls = []
        self.chat = _Chat(self)


def _make_translator(monkeypatch, **kwargs):
    client = _DummyClient()
    monkeypatch.setattr(translator_module, "make_openai_client", lambda *a, **kw: client)
    translator = Translator(
        api_base="https://example.com/v1",
        api_key="test-key",
        model="test-model",
        **kwargs,
    )
    return translator, client


class TestJsonResponseMode:
    """CORE-8: json_object is the portable default; strict json_schema is
    an explicit per-model opt-in."""

    def test_json_response_uses_portable_json_object(self, monkeypatch):
        translator, _ = _make_translator(monkeypatch, json_response=True)
        kwargs = translator._build_request_kwargs("system", "hello")
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_strict_schema_mode_uses_json_schema(self, monkeypatch):
        translator, _ = _make_translator(monkeypatch, json_response=True, json_schema_mode=True)
        kwargs = translator._build_request_kwargs("system", "hello")
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["strict"] is True

    def test_no_json_response_omits_response_format(self, monkeypatch):
        translator, _ = _make_translator(monkeypatch)
        kwargs = translator._build_request_kwargs("system", "hello")
        assert "response_format" not in kwargs

    def test_clone_preserves_schema_mode(self, monkeypatch):
        translator, _ = _make_translator(monkeypatch, json_response=True, json_schema_mode=True)
        clone = translator.with_target_language("ja")
        assert clone._json_schema_mode is True


def _fake_http_response(status: int = 400):
    """Minimum response object openai's exception classes will accept."""
    from types import SimpleNamespace

    return SimpleNamespace(
        request=SimpleNamespace(url="https://example.com/v1", method="POST"),
        status_code=status,
        headers={},
    )


class TestStreamOptionsFallback:
    """CORE-6: only a BadRequest (endpoint rejects stream_options) triggers
    the retry-without; 401/429/network errors propagate untouched."""

    def test_bad_request_retries_without_stream_options(self, monkeypatch):
        translator, client = _make_translator(monkeypatch)

        class _FailingCreate:
            def __init__(self, client):
                self.client = client
                self.count = 0

            def __call__(self, **kwargs):
                self.count += 1
                if "stream_options" in kwargs:
                    raise BadRequestError(
                        "unsupported parameter", response=_fake_http_response(), body=None
                    )
                self.client.calls.append(kwargs)
                return []

        failing = _FailingCreate(client)
        monkeypatch.setattr(translator._client.chat.completions, "create", failing.__call__)
        list(translator.translate_iter("hello", "en"))
        assert failing.count == 2
        # The fallback request deliberately omits stream_options.
        assert "stream_options" not in client.calls[0]

    def test_authentication_error_propagates_without_retry(self, monkeypatch):
        translator, client = _make_translator(monkeypatch)

        from openai import AuthenticationError

        def _auth(**_kwargs):
            raise AuthenticationError("bad key", response=_fake_http_response(401), body=None)

        monkeypatch.setattr(translator._client.chat.completions, "create", _auth)
        with pytest.raises(AuthenticationError):
            list(translator.translate_iter("hello", "en"))
        assert client.calls == []

    def test_sync_path_also_retries_only_bad_request(self, monkeypatch):
        translator, client = _make_translator(monkeypatch, streaming=True)

        def _create(**kwargs):
            if "stream_options" in kwargs:
                raise BadRequestError(
                    "unsupported parameter", response=_fake_http_response(), body=None
                )
            client.calls.append(kwargs)
            return []

        monkeypatch.setattr(translator._client.chat.completions, "create", _create)
        result = translator._translate_streaming("system", "hello")
        assert result == ""
        assert "stream_options" not in client.calls[0]


class TestHistoryThreadSafety:
    """CORE-7: concurrent translate calls must not lose history entries or
    crash the context slice."""

    def test_concurrent_appends_keep_all_entries(self, monkeypatch):
        translator, _ = _make_translator(monkeypatch, streaming=False)
        translator.set_context_turns(5)

        # Fast fake: returns immediately (translate() itself appends to
        # history — the concurrent appends are the thing under test).
        def _fast_translate(_system_prompt, text):
            return "译" + text

        monkeypatch.setattr(translator, "_translate_sync", _fast_translate)

        errors = []

        def _work(i):
            try:
                for _ in range(50):
                    translator.translate(f"hello {i}", "en")
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [threading.Thread(target=_work, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # 8 threads x 50 translations, capped at context_turns + 2 entries.
        assert len(translator._history) <= 5 + 2
        assert translator._history[-1][1] == "译" + translator._history[-1][0]
