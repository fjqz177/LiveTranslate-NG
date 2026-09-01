"""GitHub release check (plan §4.9 rewrite).

The self-hosted Ed25519 update bridge is gone. A "check for updates" now
asks GitHub's releases API for the latest release, compares it to the
running version, and points the user at the download page. Pure core: no
Qt, no side effects beyond an injected HTTP fetch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx
from packaging.version import InvalidVersion, Version

# The checkpoint for "latest release". We deliberately do NOT maintain a
# manifest anymore — the version truth lives in the GitHub release tag, so
# asking GitHub directly can never drift from what was actually published.
RELEASES_API_URL = "https://api.github.com/repos/fjqz177/LiveTranslate/releases/latest"
RELEASE_PAGE_URL = "https://github.com/fjqz177/LiveTranslate/releases/latest"

# A tag like "v1.2.3" -> "1.2.3"; tolerate a missing leading v.
_VERSION_RE = re.compile(r"^[vV]?(\d+(?:\.\d+)*.*)$")


@dataclass(frozen=True)
class UpdateCheckResult:
    """Outcome of a GitHub release check."""

    kind: str  # "new" | "uptodate" | "none" | "error"
    new_version: str | None = None
    notes: str | None = None
    url: str | None = None
    detail: str = ""


def _normalize_tag(tag: str) -> str | None:
    """Strip the leading 'v' so Version() can compare against app_version."""
    m = _VERSION_RE.match(tag.strip())
    return m.group(1) if m else None


def _parse_release(payload: dict[str, Any]) -> str | None:
    """Version from a GitHub release payload's tag_name, or None if unusable."""
    tag = payload.get("tag_name")
    if not isinstance(tag, str) or not tag.strip():
        return None
    return _normalize_tag(tag)


def check_latest_release(
    current_version: str,
    timeout: float = 10.0,
) -> UpdateCheckResult:
    """Ask GitHub for the latest release and compare against this install.

    latest -> "new" (with release-page url + notes), same-or-newer ->
    "uptodate", no release published -> "none", any failure -> "error"
    (with detail). Never raises: every outcome is represented in the result.
    """
    try:
        resp = httpx.get(RELEASES_API_URL, timeout=timeout, follow_redirects=True)
        if resp.status_code == 404:
            return UpdateCheckResult(kind="none")  # no release published yet
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError as e:
            return UpdateCheckResult(kind="error", detail=f"bad JSON from GitHub: {e}")
        latest = _parse_release(payload)
        if latest is None:
            return UpdateCheckResult(kind="error", detail="release has no valid version tag")
        try:
            current = Version(current_version)
            upstream = Version(latest)
        except InvalidVersion:
            return UpdateCheckResult(kind="error", detail="unversioned tag")
        if upstream <= current:
            return UpdateCheckResult(kind="uptodate")
        return UpdateCheckResult(
            kind="new",
            new_version=latest,
            notes=payload.get("body") if isinstance(payload.get("body"), str) else None,
            url=RELEASE_PAGE_URL,
        )
    except httpx.HTTPError as e:
        return UpdateCheckResult(kind="error", detail=str(e))
