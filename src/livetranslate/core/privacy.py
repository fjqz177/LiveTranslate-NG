"""Centralized privacy redaction for logs and diagnostics (no UI deps).

Single source of truth for masking rules. Every runtime log line that can
carry a secret, a credential-bearing URL, a user/data path, or transcript
text should be run through ``redact_text`` before it is logged; the
diagnostics bundle reuses the same helpers (via ``diagnostics``) so the
packaged report and the live logs stay consistent.

Rules (kept parallel to the "privacy promise" in the UI):
  - CORE-3: collapse user/data/repo roots to ``<data_root>/...`` placeholders,
    including non-platformdirs dev layouts where the repo IS the data root.
  - CORE-4: mask URL userinfo (``http://user:pass@host``) — proxies carry
    credentials in the URL and every log line about them would leak them.
  - SEC-1: strip transcript/speech/translation content from log lines beyond
    the pipeline's opt-in gate, so an old/edge log never leaks speech text.
  - secret values: ``sk-...`` / ``Bearer ...`` and secret-named dict keys.
"""

from __future__ import annotations

import os
import re
from typing import Any

# Dict keys whose values are secrets (matched case-insensitively, substring).
_SECRET_KEYS = ("api_key", "apikey", "key", "token", "password", "authorization")
_SECRET_VALUE_RE = re.compile(r"sk-[A-Za-z0-9_-]{6,}|Bearer\s+[A-Za-z0-9._-]{6,}")
# URL userinfo is the segment before '@' in a scheme://user:pass@host url.
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
    except Exception:  # never break log/diagnostic redaction
        return ()


def _fold_root(text: str, root: str, placeholder: str) -> str:
    root = root.rstrip("\\/")
    if not root:
        return text
    flags = re.IGNORECASE if os.name == "nt" else 0
    return re.compile(re.escape(root) + r"(?=[\\/]|$)", flags).sub(placeholder, text)


def redact_text(text: str, extra_roots: tuple[str, ...] = ()) -> str:
    """Mask secrets, URL credentials and collapse user/data paths to
    placeholders.

    ``extra_roots`` overrides the live-root folding (used by call sites that
    have already resolved a specific root to hide); by default the current
    data root and repo root are folded.
    """
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


def mask_proxy_url(url: str) -> str:
    """Mask the userinfo segment of a proxy URL (CORE-4):

    ``http://user:pw@h:1`` -> ``http://***@h:1``. Proxy credentials must never
    reach logs or the diagnostics bundle.
    """
    return _URL_USERINFO_RE.sub("***", url)


def redact_settings(snapshot: dict[str, Any], exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return a log-safe copy of a settings snapshot.

    Drops the passed-in keys (e.g. the large ``models`` map / ``system_prompt``
    that are not single scalar values), masks the secret-like values with
    ``redact_dict``, then collapses paths + URL userinfo in the remaining
    string values (CORE-3/CORE-4). ``redact_dict`` alone only hides
    secret-Named keys, so a ``download_proxy`` / ``remote_asr_url`` that embeds
    ``user:pass@`` credentials would otherwise reach the log verbatim; doing
    this here keeps every caller safe at the source.
    """
    subset = {k: v for k, v in snapshot.items() if k not in exclude}
    masked = redact_dict(subset)
    return {k: (redact_text(v) if isinstance(v, str) else v) for k, v in masked.items()}
