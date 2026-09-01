"""Unit tests for modeling/hub_downloader.py (httpx direct REST, no network).

All requests run through httpx.MockTransport; the tests pin the wire formats
(ModelScope /api/v1/models/{id}/repo/files, HuggingFace /api/models/{id}/tree)
that the real endpoints must keep serving, plus the SDK-era cache layouts that
modeling/cache.py depends on.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import httpx
import pytest

from livetranslate.modeling.hub_downloader import (
    DownloadError,
    _safe_dest,
    download_repo,
)

if TYPE_CHECKING:
    from pathlib import Path


def _router(handlers: dict) -> httpx.BaseTransport:
    """Build a MockTransport dispatching on (method, url-path)."""

    def handler(request: httpx.Request) -> httpx.Response:
        for key, fn in handlers.items():
            method, path_prefix = key
            if request.method == method and request.url.path.startswith(path_prefix):
                return fn(request)
        return httpx.Response(404, request=request)

    return httpx.MockTransport(handler)


def _ms_tree_response(files: list[tuple[str, int, str | None]]) -> httpx.Response:
    items = [
        {"Name": p, "Path": p, "Size": size, **({"Sha256": sha} if sha else {})}
        for p, size, sha in files
    ]
    return httpx.Response(200, json={"Data": {"Files": items}})


def _ms_file_response(request: httpx.Request, content: bytes) -> httpx.Response:
    """Serve a ModelScope file request, honoring Range like the real endpoint."""
    start = 0
    header_range = request.headers.get("Range")
    if header_range and header_range.startswith("bytes="):
        start = int(header_range.split("=", 1)[1].rstrip("-") or 0)
    body = content[start:]
    headers = {}
    if start:
        headers["Content-Range"] = f"bytes {start}-{len(content) - 1}/{len(content)}"
        return httpx.Response(206, headers=headers, content=body)
    return httpx.Response(200, content=content)


def test_ms_download_writes_sdk_cache_layout(tmp_path: Path):
    content = b"model-weights" * 1000
    sha = hashlib.sha256(content).hexdigest()

    def file_handler(request: httpx.Request) -> httpx.Response:
        file_path = request.url.params.get("FilePath", "")
        if file_path.endswith("config.yaml"):
            return httpx.Response(200, content=b"cfg-yam")
        return _ms_file_response(request, content)

    transport = _router(
        {
            ("GET", "/api/v1/models/iic/SenseVoiceSmall/repo/files"): lambda r: _ms_tree_response(
                [("model.pt", len(content), sha), ("config.yaml", 7, None)]
            ),
            ("GET", "/api/v1/models/iic/SenseVoiceSmall/repo"): file_handler,
        }
    )

    dest = download_repo("iic/SenseVoiceSmall", "ms", cache_dir=tmp_path, transport=transport)

    assert dest == tmp_path / "iic" / "SenseVoiceSmall"
    assert (dest / "model.pt").read_bytes() == content
    assert (dest / "config.yaml").read_text() == "cfg-yam"
    assert not list(dest.rglob("*.incomplete"))


def test_ms_skips_tree_entries(tmp_path: Path):
    """Directories (Type=tree) must not be fetched — the file endpoint 404s
    on them (real-world failure: SenseVoiceSmall's example/ dir)."""
    content = b"model-weights" * 100
    sha = hashlib.sha256(content).hexdigest()

    tree = httpx.Response(
        200,
        json={
            "Data": {
                "Files": [
                    {"Path": "example", "Type": "tree", "Size": 0},
                    {"Path": "fig", "Type": "tree", "Size": 0},
                    {"Path": "model.pt", "Type": "blob", "Size": len(content), "Sha256": sha},
                ]
            }
        },
    )

    def file_handler(request: httpx.Request) -> httpx.Response:
        file_path = request.url.params.get("FilePath", "")
        if file_path in ("example", "fig"):
            return httpx.Response(404, json={"Message": "not found"}, request=request)
        return _ms_file_response(request, content)

    transport = _router(
        {
            ("GET", "/api/v1/models/iic/SenseVoiceSmall/repo/files"): lambda r: tree,
            ("GET", "/api/v1/models/iic/SenseVoiceSmall/repo"): file_handler,
        }
    )

    dest = download_repo("iic/SenseVoiceSmall", "ms", cache_dir=tmp_path, transport=transport)

    assert (dest / "model.pt").read_bytes() == content
    assert not (dest / "example").exists()
    assert not (dest / "fig").exists()


def test_ms_resume_continues_from_existing_partial(tmp_path: Path):
    content = b"abcdef" * 100
    sha = hashlib.sha256(content).hexdigest()

    def file_handler(request: httpx.Request) -> httpx.Response:
        return _ms_file_response(request, content)

    transport = _router(
        {
            ("GET", "/api/v1/models/iic/SenseVoiceSmall/repo/files"): lambda r: _ms_tree_response(
                [("model.pt", len(content), sha)]
            ),
            ("GET", "/api/v1/models/iic/SenseVoiceSmall/repo"): file_handler,
        }
    )
    partial = tmp_path / "iic" / "SenseVoiceSmall" / "model.pt.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(content[:300])

    dest = download_repo("iic/SenseVoiceSmall", "ms", cache_dir=tmp_path, transport=transport)

    assert (dest / "model.pt").read_bytes() == content


def test_ms_sha256_mismatch_raises(tmp_path: Path):
    content = b"payload"
    transport = _router(
        {
            ("GET", "/api/v1/models/iic/SenseVoiceSmall/repo/files"): lambda r: _ms_tree_response(
                [("model.pt", len(content), "0" * 64)]
            ),
            ("GET", "/api/v1/models/iic/SenseVoiceSmall/repo"): lambda r: _ms_file_response(
                r, content
            ),
        }
    )

    with pytest.raises(DownloadError, match="sha256"):
        download_repo("iic/SenseVoiceSmall", "ms", cache_dir=tmp_path, transport=transport)


def test_hf_download_uses_snapshot_layout_with_real_sha(tmp_path: Path):
    content = b"whisper" * 2000
    sha = hashlib.sha256(content).hexdigest()
    commit = "a" * 40

    transport = _router(
        {
            ("GET", "/api/models/Systran/faster-whisper-small/revision/main"): lambda r: (
                httpx.Response(200, json={"sha": commit})
            ),
            ("GET", "/api/models/Systran/faster-whisper-small/tree/main"): lambda r: httpx.Response(
                200,
                json=[
                    {
                        "path": "model.bin",
                        "size": len(content),
                        "type": "file",
                        "lfs": {"sha256": sha},
                    }
                ],
            ),
            ("GET", "/Systran/faster-whisper-small/resolve/main/model.bin"): lambda r: (
                _ms_file_response(r, content)
            ),
        }
    )

    dest = download_repo(
        "Systran/faster-whisper-small", "hf", cache_dir=tmp_path, transport=transport
    )

    assert dest == tmp_path / "models--Systran--faster-whisper-small" / "snapshots" / commit
    assert (dest / "model.bin").read_bytes() == content


def test_ignore_patterns_skip_files(tmp_path: Path):
    transport = _router(
        {
            ("GET", "/api/v1/models/iic/SenseVoiceSmall/repo/files"): lambda r: _ms_tree_response(
                [("model.pt", 10, None), ("big.gguf", 20, None)]
            ),
            ("GET", "/api/v1/models/iic/SenseVoiceSmall/repo"): lambda r: httpx.Response(
                200, content=b"0123456789"
            ),
        }
    )

    dest = download_repo(
        "iic/SenseVoiceSmall",
        "ms",
        cache_dir=tmp_path,
        ignore_patterns=("*.gguf",),
        transport=transport,
    )

    assert (dest / "model.pt").exists()
    assert not (dest / "big.gguf").exists()


def test_local_dir_mode_writes_flat(tmp_path: Path):
    target = tmp_path / "Qwen3-0.6B"
    transport = _router(
        {
            ("GET", "/api/models/Qwen/Qwen3-0.6B/revision/main"): lambda r: httpx.Response(
                200, json={"sha": "b" * 40}
            ),
            ("GET", "/api/models/Qwen/Qwen3-0.6B/tree/main"): lambda r: httpx.Response(
                200, json=[{"path": "model.safetensors", "size": 6, "type": "file"}]
            ),
            ("GET", "/Qwen/Qwen3-0.6B/resolve/main/model.safetensors"): lambda r: httpx.Response(
                200, content=b"weight"
            ),
        }
    )

    dest = download_repo("Qwen/Qwen3-0.6B", "hf", local_dir=target, transport=transport)

    assert dest == target
    assert (target / "model.safetensors").read_bytes() == b"weight"


def test_missing_both_destinations_raises():
    with pytest.raises(DownloadError, match="cache_dir or local_dir"):
        download_repo("iic/SenseVoiceSmall", "ms", transport=_router({}))


class TestSafeDest:
    """SEC-3: repo-tree entries must never resolve outside the download
    root — a compromised or malicious model repo is an untrusted input."""

    def test_normal_paths_resolve(self, tmp_path: Path):
        root = tmp_path / "cache"
        root.mkdir()
        assert _safe_dest(root, "model.pt") == (root / "model.pt").resolve()
        assert _safe_dest(root, "sub/model.pt") == (root / "sub/model.pt").resolve()

    def test_traversal_rejected(self, tmp_path: Path):
        root = tmp_path / "cache"
        root.mkdir()
        assert _safe_dest(root, "../escape") is None
        assert _safe_dest(root, "a/../../escape") is None
        assert _safe_dest(root, "..") is None

    def test_absolute_paths_rejected(self, tmp_path: Path):
        root = tmp_path / "cache"
        root.mkdir()
        assert _safe_dest(root, "/etc/passwd") is None
        assert _safe_dest(root, "C:\\windows\\evil.txt") is None

    def test_windows_reserved_names_rejected(self, tmp_path: Path):
        root = tmp_path / "cache"
        root.mkdir()
        for name in ("CON", "nul", "com1", "sub/CON", "LPT2.txt"):
            assert _safe_dest(root, name) is None, name


def test_ms_skips_unsafe_tree_entries(tmp_path: Path):
    """Integration: traversal entries are skipped with a warning, safe
    entries still download, nothing lands outside the cache root."""
    content = b"0123456789"
    transport = _router(
        {
            ("GET", "/api/v1/models/evil/evil/repo/files"): lambda r: _ms_tree_response(
                [("model.pt", len(content), None), ("../escape.txt", len(content), None)]
            ),
            ("GET", "/api/v1/models/evil/evil/repo"): lambda r: httpx.Response(
                200, content=content
            ),
        }
    )

    dest = download_repo("evil/evil", "ms", cache_dir=tmp_path, transport=transport)

    assert (dest / "model.pt").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert not (dest / ".." / "escape.txt").exists()
