"""SEC-4 supply-chain hardening: the Silero VAD torch.hub fallback must be
pinned to a stable release tag (never the mutable master branch), and the
relaxed-SSL retry must require an explicit opt-in.

The torch module is faked via sys.modules so the tests run without a torch
install (the base test environment never has it).
"""

import sys
import types

import pytest

from livetranslate.modeling import manager


@pytest.fixture()
def fake_torch(monkeypatch):
    """Replace torch.hub.load with a controllable stub."""
    calls = {"n": 0}

    def _install(impl):
        calls["impl"] = impl
        torch = types.ModuleType("torch")
        hub = types.ModuleType("torch.hub")
        hub.load = lambda *_args, **_kwargs: (
            calls.update(n=calls["n"] + 1),
            impl(*_args, **_kwargs),
        )[1]
        torch.hub = hub
        monkeypatch.setitem(sys.modules, "torch", torch)

    return _install, calls


@pytest.fixture()
def no_silero_pkg(monkeypatch):
    monkeypatch.setattr(manager, "_has_silero_pkg", lambda: False)


class TestSileroSupplyChain:
    def test_hub_ref_is_pinned_to_a_tag(self):
        # A mutable branch ("master") would let an upstream repo takeover
        # change what every fresh install downloads.
        ref = manager.SILERO_HUB_REF
        assert ref.startswith("snakers4/silero-vad:")
        assert ref.split(":", 1)[1].startswith("v"), ref
        assert ":master" not in ref and ":main" not in ref


class TestMaskProxyUrl:
    """CORE-4: proxy credentials never reach the logs."""

    def test_userinfo_masked(self):
        assert (
            manager.mask_proxy_url("http://user:secret@proxy.local:1080")
            == "http://***@proxy.local:1080"
        )

    def test_plain_url_untouched(self):
        assert manager.mask_proxy_url("http://proxy.local:1080") == "http://proxy.local:1080"

    def test_no_credentials_untouched(self):
        assert manager.mask_proxy_url("system") == "system"

    def test_ssl_relaxation_requires_opt_in(self, monkeypatch, fake_torch, no_silero_pkg):
        install, calls = fake_torch

        class _CertError(Exception):
            pass

        def _fail(*_args, **_kwargs):
            raise _CertError("CERTIFICATE_VERIFY_FAILED")

        install(_fail)
        monkeypatch.delenv(manager.RELAX_SSL_ENV, raising=False)
        monkeypatch.setattr(manager, "_load_silero_relaxed_ssl", lambda: (None, None))
        with pytest.raises(RuntimeError, match="LIVETRANSLATE_RELAX_SSL"):
            manager.download_silero(proxy="none")
        assert calls["n"] == 1  # strict attempt only, no relaxed retry

    def test_ssl_relaxation_runs_when_opted_in(self, monkeypatch, fake_torch, no_silero_pkg):
        install, calls = fake_torch

        class _CertError(Exception):
            pass

        def _fail(*_args, **_kwargs):
            raise _CertError("CERTIFICATE_VERIFY_FAILED")

        install(_fail)
        monkeypatch.setenv(manager.RELAX_SSL_ENV, "1")
        relaxed_calls = []

        def _relaxed():
            relaxed_calls.append(1)
            return (None, None)

        monkeypatch.setattr(manager, "_load_silero_relaxed_ssl", _relaxed)
        manager.download_silero(proxy="none")
        assert calls["n"] == 1
        assert relaxed_calls == [1]

    def test_non_ssl_errors_never_retry_relaxed(self, monkeypatch, fake_torch, no_silero_pkg):
        install, calls = fake_torch

        def _boom(*_args, **_kwargs):
            raise ConnectionError("no route to host")

        install(_boom)
        monkeypatch.setenv(manager.RELAX_SSL_ENV, "1")
        with pytest.raises(ConnectionError):
            manager.download_silero(proxy="none")
        assert calls["n"] == 1
