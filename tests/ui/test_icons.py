"""Icon loading tests (plan §3.5.7): committed assets + runtime loader."""

from livetranslate.core.paths import PROJECT_ROOT
from livetranslate.ui.icons import create_app_icon

ASSET_DIR = PROJECT_ROOT / "assets" / "icons"


def test_icon_assets_committed():
    assert (ASSET_DIR / "app.png").exists()
    assert (ASSET_DIR / "app.ico").exists()
    assert (ASSET_DIR / "app.icns").exists()
    for status in ("run", "pause", "error"):
        assert (ASSET_DIR / f"tray_{status}_64.png").exists()


def test_spin_arrow_assets_committed():
    for theme in ("dark", "light"):
        assert (ASSET_DIR / f"spin_up_{theme}.png").exists()
        assert (ASSET_DIR / f"spin_down_{theme}.png").exists()


def test_create_app_icon_loads_assets(qapp):
    assert qapp is not None
    assert not create_app_icon().isNull()
    for status in ("run", "pause", "error"):
        assert not create_app_icon(status).isNull()


def test_unknown_status_falls_back_to_app_icon(qapp):
    assert qapp is not None
    assert not create_app_icon("bogus").isNull()


def test_fallback_icon_drawn_when_assets_missing(qapp, monkeypatch):
    assert qapp is not None
    monkeypatch.setattr("livetranslate.ui.icons.asset_dirs", lambda: [])
    assert not create_app_icon().isNull()
