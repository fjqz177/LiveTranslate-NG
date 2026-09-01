"""SEC-5 regression tests for the remote ASR server.

Requires the server deps (fastapi/uvicorn — the workspace's second
package); skipped automatically in the base test environment. The model
load is stubbed so no weights are fetched."""

import struct
import types

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from livetranslate_server import __main__ as server  # noqa: E402

MAX_BODY = server.MAX_BODY_BYTES


@pytest.fixture()
def client(monkeypatch):
    """TestClient with model load + GPU transcription stubbed out."""
    monkeypatch.setattr(server, "load_model", lambda: None)
    monkeypatch.setattr(server, "_run_transcription", lambda audio, language: ("hello world", "en"))
    server.app.state.args = types.SimpleNamespace(model="tiny", device="cpu", compute_type="int8")
    server.app.state.token = None
    with TestClient(server.app) as c:
        yield c


def _payload(text: str = "en") -> bytes:
    lang = text.encode("utf-8")
    return struct.pack("<I", len(lang)) + lang + np.zeros(16000, dtype=np.float32).tobytes()


class TestDefaults:
    def test_health_open_when_no_token(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_transcribe_roundtrip(self, client):
        r = client.post("/transcribe", content=_payload())
        assert r.status_code == 200
        body = r.json()
        assert body["text"] == "hello world"
        assert body["language"] == "en"


class TestTokenAuth:
    def test_health_requires_token_when_configured(self, monkeypatch):
        server.app.state.token = "shared-secret"
        monkeypatch.setattr(server, "load_model", lambda: None)
        server.app.state.args = types.SimpleNamespace(
            model="tiny", device="cpu", compute_type="int8"
        )
        with TestClient(server.app) as c:
            assert c.get("/health").status_code == 401
            assert c.get("/health", headers={"X-ASR-Token": "wrong"}).status_code == 401
            assert c.get("/health", headers={"X-ASR-Token": "shared-secret"}).status_code == 200
            assert c.post("/transcribe", content=_payload()).status_code == 401
            assert (
                c.post(
                    "/transcribe",
                    content=_payload(),
                    headers={"X-ASR-Token": "shared-secret"},
                ).status_code
                == 200
            )


class TestBodyLimit:
    def test_oversized_body_rejected_with_413(self, client):
        payload = b"\x00" * (MAX_BODY + 1)
        r = client.post(
            "/transcribe",
            content=payload,
            headers={"content-length": str(len(payload))},
        )
        assert r.status_code == 413

    def test_oversized_body_without_length_header_rejected(self, client):
        r = client.post("/transcribe", content=b"\x00" * (MAX_BODY + 1))
        assert r.status_code == 413

    def test_invalid_content_length_rejected_with_400(self, client):
        r = client.post("/transcribe", content=_payload(), headers={"content-length": "garbage"})
        assert r.status_code == 400

    def test_malformed_body_rejected_with_400(self, client):
        r = client.post("/transcribe", content=b"\x01\x02")
        assert r.status_code == 400
