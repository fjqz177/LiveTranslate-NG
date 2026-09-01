"""Single source of truth for the app version string (UI-2).

Every consumer (about page, engine dependency dialogs, diagnostics
summary, update checks) must read from here instead of hardcoding a
version that drifts from the release pipeline.
"""

from __future__ import annotations

_FALLBACK_VERSION = "0.1.0"


def app_version() -> str:
    """Installed package version, or the pyproject pin as fallback.

    Frozen builds bundle the package metadata; dev runs read the editable
    install. Any failure (missing metadata, broken env) degrades to the
    pinned fallback instead of crashing or lying with an empty string.
    """
    try:
        from importlib import metadata as importlib_metadata

        return importlib_metadata.version("livetranslate")
    except Exception:
        return _FALLBACK_VERSION
