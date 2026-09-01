"""SEC-5: the remote ASR client must carry the shared token on every
request (health probe included), and omit the header when unset."""

import numpy as np

from livetranslate.asr.remote import RemoteASREngine


class _FakeResp:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {"status": "ok"}

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def _make_client(monkeypatch, requests):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, headers=None):
            requests.append(("GET", headers))
            return _FakeResp()

        def post(self, url, content=None, headers=None):
            requests.append(("POST", headers))
            return _FakeResp(json_data={"text": "hello", "language": "en", "elapsed": 0.1})

        def close(self):
            pass

    monkeypatch.setattr("livetranslate.asr.remote.httpx.Client", FakeClient)


def test_token_sent_on_health_and_transcribe(monkeypatch):
    requests = []
    _make_client(monkeypatch, requests)
    engine = RemoteASREngine(server_url="http://127.0.0.1:8765", token="s3cret")
    engine.transcribe(np.zeros(16000, dtype=np.float32))

    assert len(requests) == 2
    assert requests[0][0] == "GET"  # health probe
    assert requests[0][1] == {"X-ASR-Token": "s3cret"}
    assert requests[1][0] == "POST"  # transcribe
    assert requests[1][1] == {"X-ASR-Token": "s3cret"}


def test_no_token_omits_header(monkeypatch):
    requests = []
    _make_client(monkeypatch, requests)
    engine = RemoteASREngine(server_url="http://127.0.0.1:8765")
    engine.transcribe(np.zeros(16000, dtype=np.float32))

    for _method, headers in requests:
        assert headers in ({}, None)
