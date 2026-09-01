"""Diagnostics collection and redacted report packaging (pure, no Qt).

The diagnostics panel renders collect_summary(); the log window offers
write_redacted_zip() which enforces the privacy rules from the plan
(API keys/paths masked, transcript content excluded by default).
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from livetranslate.core.paths import LOG_DIR
from livetranslate.core.version import app_version

_SECRET_KEYS = ("api_key", "apikey", "key", "token", "password", "authorization")
_SECRET_VALUE_RE = re.compile(r"sk-[A-Za-z0-9_-]{6,}|Bearer\s+[A-Za-z0-9._-]{6,}")
# CORE-4: URL userinfo ("http://user:pass@host") — proxies often carry
# credentials in the URL and every log line about them would leak them.
_URL_USERINFO_RE = re.compile(r"(?<=://)[^/@\s]+(?=@)")
_WIN_PATH_RE = re.compile(r'[A-Za-z]:\\(?:Users|Documents and Settings)\\[^"\'\s]*')
_MAC_PATH_RE = re.compile(r"/Users/[^/\"'\s]*")
_LINUX_HOME_RE = re.compile(r"/home/[^/\"'\s]*")
# SEC-1 defense in depth: strip transcript content from log lines, both the
# legacy full-content INFO format and the opt-in "… content:" lines. The
# privacy promise ("transcript content excluded by default") is enforced at
# the pipeline call site; this regex guarantees old/edge logs never leak
# speech or translation text into a diagnostics bundle.
_TRANSCRIPT_LINE_RE = re.compile(
    r"(ASR \[[^\]]*\] \([^)]*(?:, interim)?\):|Translate \(\d+ms\):|"
    r"Extra translate \[[^\]]*\]:|ASR content:|ASR interim content:|"
    r"Translate content:|Extra translate content:) .*"
)


def _runtime_roots() -> tuple[str, ...]:
    """Runtime data/repo roots to collapse in redacted text (CORE-3).

    The dev default data root is the repository itself (e.g.
    D:\\...\\LiveTranslate) and the old Windows-user regex never matched
    it — the redaction promise only held for platformdirs-style installs.
    Fetching the live roots keeps the folding honest for every layout.
    """
    try:
        from livetranslate.core.paths import PROJECT_ROOT, data_root

        roots: list[str] = []
        for r in (data_root(), PROJECT_ROOT):
            s = str(r)
            if s and s not in roots:
                roots.append(s)
        return tuple(roots)
    except Exception:  # never break diagnostics collection
        return ()


def _fold_root(text: str, root: str, placeholder: str) -> str:
    root = root.rstrip("\\/")
    if not root:
        return text
    flags = re.IGNORECASE if os.name == "nt" else 0
    return re.compile(re.escape(root) + r"(?=[\\/]|$)", flags).sub(placeholder, text)


def redact_text(text: str, extra_roots: tuple[str, ...] = ()) -> str:
    """Mask secrets, URL credentials and collapse user/data paths to
    placeholders."""
    masked = _SECRET_VALUE_RE.sub("***", text)
    masked = _URL_USERINFO_RE.sub("***", masked)
    masked = _WIN_PATH_RE.sub("<user>/...", masked)
    masked = _MAC_PATH_RE.sub("<user>/...", masked)
    masked = _LINUX_HOME_RE.sub("<user>/...", masked)
    roots = extra_roots or _runtime_roots()
    if roots:
        masked = _fold_root(masked, roots[0], "<data_root>/...")
        for root in roots[1:]:
            masked = _fold_root(masked, root, "<repo>/...")
    return _TRANSCRIPT_LINE_RE.sub(lambda m: m.group(1) + " ***", masked)


def redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Mask values whose keys look like secrets (recursive)."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if any(secret in key.lower() for secret in _SECRET_KEYS) and isinstance(value, str):
            out[key] = value[:4] + "***" if value else value
        elif isinstance(value, dict):
            out[key] = redact_dict(value)
        elif isinstance(value, list):
            out[key] = [redact_dict(item) if isinstance(item, dict) else item for item in value]
        else:
            out[key] = value
    return out


def collect_summary(
    *,
    platform: str,
    accelerator: Any,
    audio_diag: dict[str, object] | None,
    hotkey_status: dict[str, str] | None,
    permission: Any | None = None,
    engine_id: str = "",
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the diagnostics summary (rendered by the panel, copied
    into issues). All values are display-ready strings.
    """
    summary: dict[str, Any] = {
        "app": "LiveTranslate",
        "version": app_version(),
        "platform": platform,
        "accelerator": accelerator.display if accelerator is not None else "unknown",
        "engine": engine_id or "unset",
        "audio": audio_diag or "unavailable",
        "hotkeys": hotkey_status or {},
    }
    if permission is not None:
        summary["permissions"] = {
            "microphone": permission.microphone(),
            "screen_recording": permission.screen_recording(),
            "accessibility": permission.accessibility(),
        }
    if settings is not None:
        summary["settings"] = redact_dict(settings)
    return summary


def write_redacted_zip(
    out_path: Path,
    *,
    include_recent_subtitle: bool = False,
    recent_subtitle: str | None = None,
) -> dict[str, int]:
    """Package logs (redacted) + summary into out_path.

    Transcript content is excluded unless the user opts in with a single
    recent subtitle line (include_recent_subtitle=True). Returns the file
    count inside the archive.
    """
    summary = collect_summary(
        platform="",
        accelerator=None,
        audio_diag=None,
        hotkey_status=None,
    )
    files = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2),
        )
        files += 1
        log_files = sorted(LOG_DIR.glob("*.log"))[-3:] if LOG_DIR.is_dir() else []
        for log_file in log_files:
            raw = log_file.read_text(encoding="utf-8", errors="replace")
            zf.writestr(f"logs/{log_file.name}", redact_text(raw))
            files += 1
        if include_recent_subtitle and recent_subtitle:
            zf.writestr("recent-subtitle.txt", redact_text(recent_subtitle))
            files += 1
    return {"files": files, "size_bytes": out_path.stat().st_size}


def session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
