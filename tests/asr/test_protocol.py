from livetranslate.asr.protocol import (
    PROTOCOL_VERSION,
    ASREngineBase,
    EngineCapabilities,
    TranscriptionResult,
    WordTiming,
    error_response,
    ok_response,
)


def test_ok_response_shape():
    assert ok_response("abc", "result", {"text": "hi"}) == {
        "id": "abc",
        "ok": True,
        "type": "result",
        "payload": {"text": "hi"},
    }


def test_ok_response_none_payload():
    assert ok_response("abc", "ack")["payload"] is None


def test_ok_response_ready_with_none_id():
    r = ok_response(None, "ready", {"engine_type": "funasr"})
    assert r["id"] is None
    assert r["payload"]["engine_type"] == "funasr"


def test_error_response_shape():
    r = error_response("abc", ValueError("boom"), True)
    assert r["id"] == "abc"
    assert r["ok"] is False
    assert r["type"] == "error"
    assert r["error"]["message"] == "boom"
    assert r["error"]["recoverable"] is True
    assert "ValueError" in r["error"]["traceback"]


def test_error_response_recoverable_flag():
    assert error_response(None, RuntimeError("x"), False)["error"]["recoverable"] is False


def test_protocol_version_is_int():
    assert isinstance(PROTOCOL_VERSION, int)
    assert PROTOCOL_VERSION >= 1


class TestTranscriptionResult:
    def test_defaults(self):
        r = TranscriptionResult(text="hello", language="en")
        assert r.language_name is None
        assert r.words == ()

    def test_words_tuple(self):
        words = (WordTiming("hello", 0.1, 0.5), WordTiming("world", 0.6, 1.0))
        r = TranscriptionResult(
            text="hello world", language="en", language_name="English", words=words
        )
        assert r.words[1].word == "world"
        assert r.words[1].end == 1.0

    def test_frozen(self):
        r = TranscriptionResult(text="x", language=None)
        try:
            r.text = "y"
        except Exception:
            return
        raise AssertionError("TranscriptionResult should be frozen")

    def test_word_timing_frozen(self):
        w = WordTiming("a", 0.0, 1.0)
        try:
            w.end = 2.0
        except Exception:
            return
        raise AssertionError("WordTiming should be frozen")


class TestEngineBase:
    def test_capabilities_defaults(self):
        caps = EngineCapabilities()
        assert caps.word_timestamps is False
        assert caps.input_padding is False
        assert caps.remote is False

    def test_base_defaults_are_safe_noops(self):
        class DummyEngine(ASREngineBase):
            capabilities = EngineCapabilities(word_timestamps=True)

            def transcribe(self, audio, word_timestamps=False):
                return None

        e = DummyEngine()
        assert e.capabilities.word_timestamps is True
        # Base-class defaults must not raise
        e.set_language("ja")
        e.set_input_padding(0.5)
        e.unload()
        e.shutdown()

    def test_transcribe_is_abstract(self):
        class Incomplete(ASREngineBase):
            pass

        try:
            Incomplete()
        except TypeError:
            return
        raise AssertionError("ASREngineBase should require transcribe()")
