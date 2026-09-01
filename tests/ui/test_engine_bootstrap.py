"""EngineBootstrapDialog UI tests (§5.2): variant default, progress, retry.

`QTimer.singleShot` is suppressed so construction doesn't auto-start a real
install; the install phase is driven explicitly with a fake installer injected
through the module-level `_install_variant` seam, exercising the real worker
thread and both __done__/__failed__ outcomes without network or disk.
"""

import pytest
from PyQt6.QtCore import QTimer


@pytest.fixture
def no_autostart(monkeypatch):
    """Stop the __init__ auto-start from launching a real install.

    Returns a truthy sentinel so a test that consumes it reads as a real
    dependency rather than an unused fixture argument.
    """
    monkeypatch.setattr(QTimer, "singleShot", lambda ms, fn: None)
    return 0


def test_variant_combo_defaults_to_detected(qapp):
    import livetranslate.ui.dependency_dialog as dd

    assert qapp is not None
    dlg = dd.EngineBootstrapDialog("cu126")
    # The detected/recommended variant is pre-selected in the switchable combo.
    assert dlg._variant_combo.currentData() == "cu126"
    # Both variants listed, with a size hint.
    assert dlg._variant_combo.count() == 2
    # Progress bar and retry initially hidden; status starts the hint.
    assert dlg._bar.isHidden()
    assert dlg._retry_btn.isHidden()


def test_install_success_accepts(no_autostart, qapp):
    import livetranslate.ui.dependency_dialog as dd

    assert no_autostart is not None
    assert qapp is not None
    dlg = dd.EngineBootstrapDialog("cpu")

    def fake_installer(_variant, *, progress_cb, **kwargs):
        del kwargs  # app_version / pypi_mirror come through here; unused by the fake
        progress_cb("installing cpu wheels")

    dd._install_variant = fake_installer
    try:
        dlg._start_install()
        assert dlg._thread is not None
        dlg._thread.join(timeout=5)
    finally:
        dd._install_variant = None
    assert not dlg._thread.is_alive()
    assert dlg._ok is True


def test_install_failure_offers_retry(no_autostart, qapp):
    import livetranslate.ui.dependency_dialog as dd

    assert no_autostart is not None
    assert qapp is not None
    dlg = dd.EngineBootstrapDialog("cu126")

    def failing_installer(_variant, **kwargs):
        del kwargs
        raise RuntimeError("disk full")

    dd._install_variant = failing_installer
    try:
        dlg._start_install()
        assert dlg._thread is not None
        dlg._thread.join(timeout=5)
        # __failed__ lands on the main thread as a queued signal: flush it.
        from PyQt6.QtWidgets import QApplication

        QApplication.processEvents()
    finally:
        dd._install_variant = None
    assert dlg._ok is False
    assert dlg._error
    # Retry button appears on failure; the quit button remains.
    assert not dlg._retry_btn.isHidden()
