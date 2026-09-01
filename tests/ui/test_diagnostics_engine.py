"""Diagnostics accelerator card tests (当前引擎 display)."""

import types

import pytest

from livetranslate.core.i18n import t
from livetranslate.ui.diagnostics import DiagnosticsView


@pytest.fixture
def view(qapp):
    assert qapp is not None
    app_ref = types.SimpleNamespace(
        _asr_ctl=None,
        _audio=None,
        _hotkeys=None,
        _panel=None,
        get_settings=lambda: {},
    )
    v = DiagnosticsView(app_ref, with_actions=False)
    yield v
    v.deleteLater()


def _field_value(box, label_text):
    form = box.layout()
    for i in range(form.rowCount()):
        label_widget = form.itemAt(i, form.ItemRole.LabelRole).widget()
        if label_widget is not None and label_widget.text() == label_text:
            field = form.itemAt(i, form.ItemRole.FieldRole).widget()
            return field.text() if field is not None else None
    return None


def test_engine_shows_friendly_name_when_active():
    """The active engine must show its registry display name, not the raw
    internal id."""
    app_ref = types.SimpleNamespace(
        _asr_ctl=types.SimpleNamespace(type="sensevoice-onnx"),
        _audio=None,
        _hotkeys=None,
        _panel=None,
        get_settings=lambda: {},
    )
    v = DiagnosticsView(app_ref, with_actions=False)
    box = v._card_accelerator()
    assert _field_value(box, t("diag_engine")) == "SenseVoice (ONNX)"
    assert _field_value(box, t("diag_engine_recommended")) in (
        "SenseVoice (ONNX)",
        "Whisper (faster-whisper)",
    )
    v.deleteLater()


def test_engine_shows_unset_when_not_started():
    """Before the ASR controller activates, the card must not display a raw
    None."""
    app_ref = types.SimpleNamespace(
        _asr_ctl=types.SimpleNamespace(type=None),
        _audio=None,
        _hotkeys=None,
        _panel=None,
        get_settings=lambda: {},
    )
    v = DiagnosticsView(app_ref, with_actions=False)
    box = v._card_accelerator()
    assert _field_value(box, t("diag_engine")) == t("diag_engine_unset")
    v.deleteLater()


def test_engine_shows_unset_without_controller():
    app_ref = types.SimpleNamespace(
        _audio=None,
        _hotkeys=None,
        _panel=None,
        get_settings=lambda: {},
    )
    v = DiagnosticsView(app_ref, with_actions=False)
    box = v._card_accelerator()
    assert _field_value(box, t("diag_engine")) == t("diag_engine_unset")
    v.deleteLater()
