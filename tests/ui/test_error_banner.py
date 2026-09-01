"""Overlay error banner + idle empty guide tests (§3.5.6)."""

from livetranslate.core.i18n import t
from livetranslate.ui.overlay import SubtitleOverlay

ERR_COPY = t("err_401")  # §3.6 canonical copy in the active language


def test_banner_shows_classified_error(qapp):
    assert qapp is not None
    overlay = SubtitleOverlay({})
    overlay.show_error(ERR_COPY)
    qapp.processEvents()
    banner = overlay._error_banner
    assert not banner.isHidden()
    assert banner._label.text() == ERR_COPY
    banner._auto_hide.stop()
    overlay.close()


def test_banner_diagnostics_click_routes(qapp):
    assert qapp is not None
    overlay = SubtitleOverlay({})
    overlay.show_error(ERR_COPY)
    qapp.processEvents()
    clicked = []
    overlay.error_banner_clicked.connect(lambda: clicked.append(True))
    overlay._error_banner._diag_btn.click()
    assert clicked == [True]
    overlay._error_banner._auto_hide.stop()
    overlay.close()


def test_banner_dismiss_hides(qapp):
    assert qapp is not None
    overlay = SubtitleOverlay({})
    overlay.show_error(ERR_COPY)
    qapp.processEvents()
    overlay._error_banner.dismissed.emit()
    assert overlay._error_banner.isHidden()
    overlay._error_banner._auto_hide.stop()
    overlay.close()


def test_idle_empty_guide_text(qapp):
    assert qapp is not None
    from livetranslate.core.i18n import t

    overlay = SubtitleOverlay({})
    overlay.show_empty_guide("idle")
    assert not overlay._empty_guide.isHidden()
    assert overlay._empty_guide.text() == t("empty_guide_idle")
    overlay.close()


def test_info_banner_uses_neutral_copy(qapp):
    assert qapp is not None
    overlay = SubtitleOverlay({})
    overlay.show_info(t("err_gnome_tray"))
    qapp.processEvents()
    banner = overlay._error_banner
    assert not banner.isHidden()
    assert banner._label.text() == t("err_gnome_tray")
    banner._auto_hide.stop()
    overlay.close()
