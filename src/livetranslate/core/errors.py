"""Translate-error classification for the §3.6 copy table.

Maps transport/API failures to stable i18n keys so every surface
(inline message, overlay banner, diagnostics card) shows the same copy.
Pure core: no Qt, no engine imports.
"""

from __future__ import annotations


def classify_translate_error(exc: BaseException, using_proxy: bool = False) -> str | None:
    """Return the §3.6 i18n key for a translation failure.

    None means no canonical copy exists (generic failures keep the raw
    exception text). Checks are duck-typed so tests can use light fakes.
    """
    status = getattr(exc, "status_code", None)
    if status == 401:
        return "err_401"
    if status == 429:
        return "err_429"

    cause = exc.__cause__
    if cause is not None and type(cause).__name__ == "ProxyError":
        return "err_proxy"

    # Timeout family: SDK timeout, asyncio TimeoutError, httpx timeout
    for cur in (exc, cause) if cause is not None else (exc,):
        name = type(cur).__name__
        if "Timeout" in name or isinstance(cur, TimeoutError):
            return "err_timeout"

    if using_proxy:
        return "err_proxy"

    name = type(exc).__name__
    if name in ("APIConnectionError", "ConnectError") or isinstance(exc, ConnectionError):
        return "err_conn_refused"
    return None
