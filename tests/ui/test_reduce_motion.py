"""Reduced-motion plumbing tests (plan §3.5.3): the overlay and the
subtitle window must drop their transitions instead of animating.
"""

from livetranslate.ui.overlay import SubtitleOverlay
from livetranslate.ui.subtitle_window import SubtitleWindow, _SubtitleTextWidget


def test_overlay_mode_switch_is_instant(qapp):
    assert qapp is not None
    overlay = SubtitleOverlay({})
    overlay.resize(600, 500)
    overlay.setMinimumHeight(80)
    overlay.set_reduce_motion(True)

    overlay._on_mode_changed("compact")
    assert overlay.height() == 80
    overlay._on_mode_changed("full")
    assert overlay.height() == 500
    overlay.close()


def test_overlay_animates_by_default(qapp):
    assert qapp is not None
    overlay = SubtitleOverlay({})
    assert overlay._reduce_motion is False
    overlay.close()


def test_text_widget_skips_entry_animation(qapp):
    assert qapp is not None
    tw = _SubtitleTextWidget()
    tw.set_config(
        {
            "entry_animation": "fade",
            "exit_animation": "fade",
            "animation_duration": 300,
        }
    )
    tw.set_reduce_motion(True)
    tw._content_opacity_val = 0.0
    tw.animate_in()
    assert tw._content_opacity_val == 1.0
    assert tw._anim_group is None
    tw.close()


def test_text_widget_skips_exit_animation_and_calls_callback(qapp):
    assert qapp is not None
    tw = _SubtitleTextWidget()
    tw.set_config({"entry_animation": "fade", "exit_animation": "fade"})
    tw.set_reduce_motion(True)
    called = []
    tw.animate_out(callback=lambda: called.append(True))
    assert tw._content_opacity_val == 0.0
    assert called == [True]
    tw.close()


def test_subtitle_window_propagates_and_snaps_height(qapp):
    assert qapp is not None
    w = SubtitleWindow({"lines": [{"enabled": True, "font_size": 40}]})
    w.set_reduce_motion(True)
    assert w._reduce_motion is True
    assert all(tw._reduce_motion for tw in w._text_widgets)
    w._fit_height_animated()  # must not create an animation
    assert w._height_anim is None
    w.close()
