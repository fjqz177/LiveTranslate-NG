"""Single-instance gate tests (plan §3.7): primary serves, secondary wakes."""

import pytest
from PyQt6.QtNetwork import QLocalServer

from livetranslate.ui.single_instance import acquire_single_instance


class FakeSystem:
    def __init__(self, acquired: bool):
        self.acquired = acquired
        self.released = False

    def try_acquire_single_instance(self, key: str) -> bool:
        return self.acquired

    def release_single_instance(self) -> None:
        self.released = True


@pytest.fixture(autouse=True)
def _clean_servers():
    QLocalServer.removeServer("lt-single-test")
    yield
    QLocalServer.removeServer("lt-single-test")


def test_primary_acquires_and_serves(qapp):
    assert qapp is not None
    system = FakeSystem(True)
    is_primary, server = acquire_single_instance("lt-single-test", system)
    assert is_primary is True
    assert server is not None
    assert server.isListening()
    server.close()


def test_secondary_wakes_primary_and_exits(qapp):
    assert qapp is not None
    primary_system = FakeSystem(True)
    is_primary, server = acquire_single_instance("lt-single-test", primary_system)
    assert is_primary is True and server is not None

    secondary_system = FakeSystem(False)
    is_primary2, server2 = acquire_single_instance("lt-single-test", secondary_system)
    assert is_primary2 is False
    assert server2 is None

    assert server.waitForNewConnection(2000)  # the wake arrived
    conn = server.nextPendingConnection()
    assert conn is not None
    # The wake byte becomes readable after event-loop processing
    from PyQt6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    conn.readyRead.connect(loop.quit)
    QTimer.singleShot(2000, loop.quit)
    loop.exec()
    assert bytes(conn.readAll()) == b"wake"
    conn.disconnectFromServer()
    server.close()


def test_primary_keeps_running_without_socket(qapp, monkeypatch):
    assert qapp is not None
    monkeypatch.setattr(
        "livetranslate.ui.single_instance.QLocalServer",
        _FailingServer,
    )
    system = FakeSystem(True)
    is_primary, server = acquire_single_instance("lt-single-test", system)
    assert is_primary is True
    assert server is None


def test_secondary_reports_wake_failure(qapp, monkeypatch):
    """Lock held but the wake connection fails: return the WAKE_FAILED
    sentinel so the caller can explain instead of exiting silently."""
    assert qapp is not None
    from livetranslate.ui.single_instance import WAKE_FAILED

    class _FailingSocket:
        def __init__(self) -> None: ...

        def connectToServer(self, name: str) -> None: ...

        def waitForConnected(self, ms: int) -> bool:
            return False

    monkeypatch.setattr("livetranslate.ui.single_instance.QLocalSocket", _FailingSocket)
    is_primary, wake = acquire_single_instance("lt-single-test", FakeSystem(False))
    assert is_primary is False
    assert wake is WAKE_FAILED


class _FailingServer:
    @staticmethod
    def removeServer(name: str) -> None:
        QLocalServer.removeServer(name)

    def __init__(self):
        pass

    def listen(self, name: str) -> bool:
        return False
