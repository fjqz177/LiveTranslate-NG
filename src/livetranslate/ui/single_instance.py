"""Single-instance gate with a local wake channel (plan §3.7 单实例).

The first instance claims the platform lock AND a QLocalServer; a second
instance that loses the lock connects to the server, sends a wake byte and
exits. The UI turns a wake into: show overlay + tray hint (§3.6 copy).
"""

from __future__ import annotations

import logging

from PyQt6.QtNetwork import QLocalServer, QLocalSocket

log = logging.getLogger("LiveTranslate.SingleInstance")

# Wake sockets live until process exit: a destructor in ClosingState would
# drop the queued wake byte, so we pin them here (the caller exits anyway).
_WAKE_SOCKETS: list[QLocalSocket] = []


class _WakeFailed:
    """Sentinel: lock was held but the running instance has no wake channel."""


WAKE_FAILED = _WakeFailed()


def acquire_single_instance(name: str, system) -> tuple[bool, QLocalServer | _WakeFailed | None]:
    """Claim instance lock + wake server. Returns (is_primary, server).

    Secondary callers (lock already held) wake the primary and must exit.
    When the wake connection cannot be established the sentinel WAKE_FAILED
    is returned so the caller can explain the failure instead of exiting
    silently. A primary whose server fails to listen still runs (no wake
    channel).
    """
    if system.try_acquire_single_instance(name):
        QLocalServer.removeServer(name)
        server = QLocalServer()
        if server.listen(name):
            log.info(f"Primary instance: wake server listening on {name}")
            return True, server
        log.warning("Instance lock acquired but wake server failed to listen")
        return True, None
    QLocalServer.removeServer(name)
    sock = QLocalSocket()
    sock.connectToServer(name)
    if sock.waitForConnected(300):
        sock.write(b"wake")
        sock.flush()
        sock.waitForBytesWritten(300)
        # Graceful close flushes any remaining bytes; pin the socket so a
        # ClosingState destructor cannot drop the wake (process exits anyway).
        sock.disconnectFromServer()
        if sock.state() != QLocalSocket.LocalSocketState.UnconnectedState:
            sock.waitForDisconnected(300)
        _WAKE_SOCKETS.append(sock)
        log.info("Notified the running instance")
    else:
        log.warning("Running instance did not accept the wake connection")
        return False, WAKE_FAILED
    return False, None
