"""§3.6 translate-error classification tests (pure core)."""

from livetranslate.core.errors import classify_translate_error


class ProxyError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class APITimeoutError(Exception):
    pass


class ConnectError(Exception):
    pass


class _Status(Exception):
    def __init__(self, status):
        super().__init__(str(status))
        self.status_code = status


def test_http_status_codes():
    assert classify_translate_error(_Status(401)) == "err_401"
    assert classify_translate_error(_Status(429)) == "err_429"
    assert classify_translate_error(_Status(500)) is None


def test_proxy_error_cause():
    exc = APIConnectionError("boom")
    exc.__cause__ = ProxyError("boom")
    assert classify_translate_error(exc) == "err_proxy"


def test_timeout_family():
    assert classify_translate_error(TimeoutError("t")) == "err_timeout"
    assert classify_translate_error(APITimeoutError("t")) == "err_timeout"


def test_connection_family():
    assert classify_translate_error(APIConnectionError("c")) == "err_conn_refused"
    assert classify_translate_error(ConnectError("c")) == "err_conn_refused"
    assert classify_translate_error(ConnectionError("c")) == "err_conn_refused"


def test_proxy_flag_wins_for_generic_errors():
    assert classify_translate_error(APIConnectionError("c"), using_proxy=True) == "err_proxy"
    assert classify_translate_error(_Status(429), using_proxy=True) == "err_429"


def test_generic_errors_have_no_canonical_copy():
    assert classify_translate_error(ValueError("boom")) is None
    assert classify_translate_error(RuntimeError("boom")) is None
