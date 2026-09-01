"""Diagnostics collection and redacted report packaging (pure, no Qt).

The diagnostics panel renders collect_summary(); the log window offers
write_redacted_zip() which enforces the privacy rules from the plan
(API keys/paths masked, transcript content excluded by default).

The masking rules themselves live in :mod:`livetranslate.core.privacy` and
are re-exported here so the UI can reach them as ``diagnostics.redact_text``
/ ``diagnostics.redact_dict``; call sites that log live data should redact
at the source instead of relying on the bundle pass.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from livetranslate.core.paths import LOG_DIR
from livetranslate.core.privacy import redact_dict, redact_text
from livetranslate.core.version import app_version


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
