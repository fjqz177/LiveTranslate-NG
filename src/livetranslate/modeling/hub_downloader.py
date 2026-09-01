"""Lightweight model-repo downloader (SelfServe P0-A2).

Replaces the modelscope / huggingface_hub SDKs in the base install: both are
only needed at model-download time, and the SDKs drag ~130 MB of transitive
dependencies (transformers, jieba, tokenizers, ...) into the frozen bundle.

This module talks to the public REST endpoints directly with httpx:

- ModelScope: GET /api/v1/models/{id}/repo/files (tree) and
  GET /api/v1/models/{id}/repo?FilePath=... (file, supports Range).
- HuggingFace: GET /api/models/{id}/tree/{rev}?recursive=true and
  GET /{id}/resolve/{rev}/{path} (302 to CDN, supports Range).

Cache layouts are byte-compatible with the SDK era (see modeling/cache.py):

- ModelScope: <cache_dir>/<org>/<name>/...          (<=1.37 explicit-cache layout)
- HuggingFace: <cache_dir>/models--<org>--<name>/snapshots/<rev>/...

Downloads write to `<file>.incomplete` and rename on success, so an aborted
run leaves only `.incomplete` blobs — which cache.py's completeness checks
already ignore. Range resumes make a retry continue where it stopped.

Pure httpx + stdlib: no Qt, no torch, no SDK imports.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import httpx

from livetranslate.core.privacy import redact_text

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger("LiveTranslate.HubDownload")

Hub = Literal["ms", "hf"]

_CHUNK = 1024 * 1024
_HEADERS = {"User-Agent": "LiveTranslate/0.2 (model downloader)"}
_PROGRESS_INTERVAL = 2.0  # seconds between log lines

# Windows reserved device names (SEC-3): a repo entry like "CON/model.bin"
# would otherwise be an alternate data stream / device write on Windows.
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class DownloadError(RuntimeError):
    """Repository download failed (network, HTTP, or checksum)."""


@dataclass(frozen=True)
class RepoFile:
    """One file in a repo tree."""

    path: str
    size: int
    sha256: str | None = None


def _ms_tree(client: httpx.Client, repo_id: str, revision: str) -> list[RepoFile]:
    org, name = repo_id.split("/", 1)
    url = f"https://modelscope.cn/api/v1/models/{org}/{name}/repo/files"
    resp = client.get(url, params={"Revision": revision, "Recursive": "true"}, headers=_HEADERS)
    resp.raise_for_status()
    data = resp.json()
    files: list[RepoFile] = []
    for item in data.get("Data", {}).get("Files") or []:
        # The recursive tree lists directories too (Type="tree", e.g.
        # SenseVoiceSmall's example/ and fig/). The file endpoint 404s on
        # them — only blob/file entries are downloadable, and their children
        # arrive as individual blob entries.
        entry_type = str(item.get("Type") or "").lower()
        if entry_type in ("tree", "dir", "directory"):
            continue
        path = item.get("Path") or item.get("Name")
        if not path:
            continue
        size = int(item.get("Size") or 0)
        sha = item.get("Sha256") or None
        files.append(RepoFile(path=path, size=size, sha256=sha.lower() if sha else None))
    return files


def _hf_revision_sha(client: httpx.Client, repo_id: str, revision: str) -> str:
    """Resolve a branch name to the commit sha (snapshots dir is named by sha)."""
    if revision not in ("main", "master"):
        return revision
    org, name = repo_id.split("/", 1)
    url = f"https://huggingface.co/api/models/{org}/{name}/revision/{revision}"
    resp = client.get(url, headers=_HEADERS)
    resp.raise_for_status()
    return resp.json().get("sha") or revision


def _hf_tree(client: httpx.Client, repo_id: str, revision: str) -> tuple[list[RepoFile], str]:
    org, name = repo_id.split("/", 1)
    url = f"https://huggingface.co/api/models/{org}/{name}/tree/{revision}"
    resp = client.get(url, params={"recursive": "true"}, headers=_HEADERS)
    resp.raise_for_status()
    files: list[RepoFile] = []
    for item in resp.json():
        if item.get("type") != "file":
            continue
        size = int(item.get("size") or 0)
        lfs = item.get("lfs") or {}
        sha = lfs.get("sha256") if isinstance(lfs, dict) else None
        files.append(RepoFile(path=item["path"], size=size, sha256=sha))
    commit = _hf_revision_sha(client, repo_id, revision)
    return files, commit


def _file_url(repo_id: str, hub: Hub, revision: str, path: str) -> str:
    if hub == "ms":
        org, name = repo_id.split("/", 1)
        return (
            f"https://modelscope.cn/api/v1/models/{org}/{name}/repo"
            f"?FilePath={path}&Revision={revision}"
        )
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{path}"


def _skip(path: str, ignore_patterns: tuple[str, ...]) -> bool:
    name = path.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(path, p) for p in ignore_patterns)


def _safe_dest(dest_root: Path, rel: str) -> Path | None:
    """Resolve a repo-tree relative path under dest_root, or None when the
    path escapes it (SEC-3). A compromised or malicious model repo must
    never write outside the cache root: absolute paths, '..' segments and
    Windows reserved device names are rejected."""
    p = Path(rel)
    if p.is_absolute() or not p.parts or ".." in p.parts:
        return None
    if any(part.upper().split(".", 1)[0].rstrip(".") in _WINDOWS_RESERVED for part in p.parts):
        return None
    try:
        dest = (dest_root / p).resolve()
    except OSError:
        return None
    root_norm = os.path.normcase(str(dest_root.resolve()))
    dest_norm = os.path.normcase(str(dest))
    if dest_norm == root_norm or dest_norm.startswith(root_norm + os.sep):
        return dest
    return None


def _fetch_file(
    client: httpx.Client,
    repo_id: str,
    hub: Hub,
    revision: str,
    entry: RepoFile,
    dest: Path,
    progress_cb: Callable[[int, int], None] | None,
) -> None:
    """Download one file with Range resume; verify sha256 when the hub provides it."""
    url = _file_url(repo_id, hub, revision, entry.path)
    tmp = dest.with_name(dest.name + ".incomplete")
    done = tmp.stat().st_size if tmp.exists() else 0
    if entry.size and done > entry.size:
        # Stale oversized partial (server file shrank) — restart.
        tmp.unlink(missing_ok=True)
        done = 0
    headers = dict(_HEADERS)
    if done:
        headers["Range"] = f"bytes={done}-"

    with client.stream("GET", url, headers=headers) as resp:
        if resp.status_code == 416:
            # Server considers the range past EOF: the partial is already complete.
            resp.close()
            _verify_and_commit(tmp, dest, entry)
            return
        resp.raise_for_status()
        total = entry.size or 0
        last_log = 0.0
        mode = "ab" if done and resp.status_code == 206 else "wb"
        with tmp.open(mode) as fh:
            for chunk in resp.iter_bytes(chunk_size=_CHUNK):
                fh.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if progress_cb:
                    progress_cb(done, total)
                if now - last_log >= _PROGRESS_INTERVAL:
                    last_log = now
                    pct = f"{done / total * 100:.0f}%" if total else "?"
                    mb = done / 1_048_576
                    log.info("  %s  %s (%.1f MB)", entry.path, pct, mb)

    _verify_and_commit(tmp, dest, entry)


def _verify_and_commit(tmp: Path, dest: Path, entry: RepoFile) -> None:
    if entry.size and tmp.stat().st_size != entry.size:
        raise DownloadError(
            f"size mismatch for {entry.path}: got {tmp.stat().st_size}, want {entry.size}"
        )
    if entry.sha256:
        h = hashlib.sha256()
        with tmp.open("rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK), b""):
                h.update(chunk)
        if h.hexdigest() != entry.sha256:
            raise DownloadError(f"sha256 mismatch for {entry.path}")
    tmp.replace(dest)


def download_repo(
    repo_id: str,
    hub: Hub,
    *,
    cache_dir: Path | None = None,
    local_dir: Path | None = None,
    revision: str | None = None,
    ignore_patterns: tuple[str, ...] = (),
    progress_cb: Callable[[int, int], None] | None = None,
    timeout: float = 120.0,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    """Download a repo; return the resolved model dir.

    Destination (exactly one of the two):
    - cache_dir: SDK-era cache layout — ms: <cache_dir>/<org>/<name>/... ;
      hf: <cache_dir>/models--<org>--<name>/snapshots/<rev>/...
    - local_dir: files land directly under it (no wrapper dirs; used by
      ensure_qwen_weights for the nano model's embedded subdir).

    Raises DownloadError on any failure; partial state is always resumable.
    `transport` is a test seam (httpx.MockTransport); production callers pass None.
    """
    if cache_dir is None and local_dir is None:
        raise DownloadError("download_repo requires cache_dir or local_dir")
    org, name = repo_id.split("/", 1)
    rev = revision or ("master" if hub == "ms" else "main")
    with httpx.Client(
        timeout=httpx.Timeout(timeout, connect=15.0),
        follow_redirects=True,
        transport=transport,
    ) as client:
        if hub == "ms":
            files = _ms_tree(client, repo_id, rev)
            dest_root = local_dir or (cache_dir / org / name if cache_dir else None)
            hf_commit = rev
        else:
            files, hf_commit = _hf_tree(client, repo_id, rev)
            dest_root = local_dir or (
                cache_dir / f"models--{org}--{name}" / "snapshots" / hf_commit
                if cache_dir
                else None
            )
        if dest_root is None:
            raise DownloadError("download_repo requires cache_dir or local_dir")

        wanted = [f for f in files if not _skip(f.path, ignore_patterns)]
        total = sum(f.size for f in wanted)
        downloaded = 0
        log.info(
            "Downloading %s from %s: %d files, %.1f MB",
            repo_id,
            "ModelScope" if hub == "ms" else "HuggingFace",
            len(wanted),
            total / 1_048_576,
        )
        dest_root.mkdir(parents=True, exist_ok=True)
        for entry in wanted:
            dest = _safe_dest(dest_root, entry.path)
            if dest is None:
                log.warning(
                    "Skipping unsafe path in repo tree: %r (%s) — refusing to write "
                    "outside the download root (SEC-3)",
                    entry.path,
                    repo_id,
                )
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            _fetch_file(client, repo_id, hub, rev, entry, dest, progress_cb)
            downloaded += entry.size
        if progress_cb:
            progress_cb(downloaded, total)
    log.info("Downloaded %s -> %s", repo_id, redact_text(str(dest_root)))
    return dest_root
