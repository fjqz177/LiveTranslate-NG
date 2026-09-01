"""Cache tab management tests (数据与存储: model list selection + delete)."""

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QMessageBox

from livetranslate.ui.panel.panel import ControlPanel
from livetranslate.ui.panel.tabs.cache_tab import _ModelList

CONFIG = {
    "translation": {
        "model": "gpt-test",
        "api_base": "https://example.com/v1",
        "api_key": "sk-test",
        "target_language": "zh",
    },
    "asr": {
        "vad_threshold": 0.5,
        "min_speech_duration": 0.3,
        "max_speech_duration": 20.0,
        "language": "auto",
        "device": "cpu",
    },
}


@pytest.fixture
def panel(qapp, tmp_path, monkeypatch):
    assert qapp is not None
    monkeypatch.setattr("livetranslate.ui.panel.panel.SETTINGS_FILE", tmp_path / "settings.json")
    p = ControlPanel(CONFIG)
    yield p
    p.close()


def test_delete_button_follows_selection(panel, qtbot):
    tab = panel._cache_tab
    assert not tab._delete_selected_btn.isEnabled()
    tab._cache_entries = [("model-a", "C:/fake/a", 1234)]
    tab._cache_list.addItem("model-a  —  1.2 KB")
    tab._cache_list.setCurrentRow(0)
    # waitUntil polls the condition instead of a single processEvents(), so
    # the selectionChanged -> enable signal can't race on a slow CI runner.
    qtbot.waitUntil(lambda: tab._delete_selected_btn.isEnabled(), timeout=2000)
    tab._cache_list.clearSelection()
    qtbot.waitUntil(lambda: not tab._delete_selected_btn.isEnabled(), timeout=2000)


def test_click_empty_space_clears_selection(qapp):
    w = _ModelList()
    w.addItem("item one")
    w.addItem("item two")
    w.resize(300, 120)
    w.show()
    qapp.processEvents()
    w.setCurrentRow(0)
    assert w.currentRow() == 0
    # click far below the two items (empty space)
    QTest.mouseClick(w.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(150, 100))
    qapp.processEvents()
    assert w.selectedItems() == [], "clicking empty space must clear the selection"
    w.close()


def test_delete_selected_removes_model_dir(tmp_path, monkeypatch, panel):
    tab = panel._cache_tab
    model_dir = tmp_path / "m1"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"x" * 16)
    tab._cache_entries = [("m1", str(model_dir), 16)]
    tab._cache_list.addItem("m1")
    tab._cache_list.setCurrentRow(0)
    monkeypatch.setattr(
        "livetranslate.ui.panel.tabs.cache_tab.QMessageBox.warning",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )
    tab._delete_selected()
    assert not model_dir.exists(), "selected model directory must be deleted"
    # declining keeps the directory
    model_dir2 = tmp_path / "m2"
    model_dir2.mkdir()
    tab._cache_entries = [("m2", str(model_dir2), 1)]
    tab._cache_list.clear()
    tab._cache_list.addItem("m2")
    tab._cache_list.setCurrentRow(0)
    monkeypatch.setattr(
        "livetranslate.ui.panel.tabs.cache_tab.QMessageBox.warning",
        lambda *a, **k: QMessageBox.StandardButton.No,
    )
    tab._delete_selected()
    assert model_dir2.exists(), "declined delete must keep the directory"
