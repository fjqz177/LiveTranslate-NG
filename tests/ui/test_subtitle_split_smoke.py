"""M-SPLIT smoke: subtitle_window.py is a re-export shim over the leaf modules.

Guards the split contract (zero public-symbol change):
- every moved symbol keeps its identity when imported through the shim
  vs. its leaf home module (subtitle_config / subtitle_text_widget);
- SubtitleWindow still assembles end-to-end from the leaf widget and
  set_config / set_text / clear keep working.
"""

from livetranslate.ui import subtitle_config, subtitle_text_widget, subtitle_window
from livetranslate.ui.subtitle_window import (
    DEFAULT_SUBTITLE_WIN_SETTINGS,
    SubtitleWindow,
    _merge_settings,
    _resolve_image_path,
    _SubtitleTextWidget,
)


def test_shim_symbols_are_the_leaf_objects():
    assert subtitle_window._SubtitleTextWidget is subtitle_text_widget._SubtitleTextWidget
    assert subtitle_window._resolve_image_path is subtitle_config._resolve_image_path
    assert subtitle_window._merge_settings is subtitle_config._merge_settings
    assert (
        subtitle_window.DEFAULT_SUBTITLE_WIN_SETTINGS
        is subtitle_config.DEFAULT_SUBTITLE_WIN_SETTINGS
    )


def test_shim_direct_imports_hit_the_same_objects():
    # Existing importers use `from ...subtitle_window import ...`; the bound
    # names must be the leaf objects, not copies.
    assert _SubtitleTextWidget is subtitle_text_widget._SubtitleTextWidget
    assert _resolve_image_path is subtitle_config._resolve_image_path
    assert _merge_settings is subtitle_config._merge_settings
    assert DEFAULT_SUBTITLE_WIN_SETTINGS is subtitle_config.DEFAULT_SUBTITLE_WIN_SETTINGS


def test_text_widget_set_config_and_set_text_no_raise(qapp):
    assert qapp is not None
    tw = _SubtitleTextWidget()
    tw.set_config(
        {
            "type": "translation",
            "lang": "zh",
            "font_family": "Microsoft YaHei",
            "font_size": 28,
            "color": "#FFFFFF",
            "opacity": 255,
            "outline_enabled": True,
            "outline_color": "#000000",
            "outline_width": 2,
            "align": "center",
            "entry_animation": "none",
            "exit_animation": "none",
            "animation_duration": 300,
        }
    )
    # Multiline content and pipe-joined translations must render without raising.
    tw.set_text("これは日本語の原文です\n二行目")
    tw.set_text("原文 | 翻译一 | 翻译二")
    qapp.processEvents()
    assert tw._text
    tw.close()


def test_subtitle_window_assembles_and_updates(qapp):
    assert qapp is not None
    win = SubtitleWindow(
        {
            "window_width": 800,
            "lines": [
                {
                    "type": "original",
                    "enabled": True,
                    "font_family": "Microsoft YaHei",
                    "font_size": 24,
                },
                {
                    "type": "translation",
                    "lang": "zh",
                    "enabled": True,
                    "font_family": "Microsoft YaHei",
                    "font_size": 28,
                },
            ],
        }
    )
    qapp.processEvents()
    # The window owns leaf _SubtitleTextWidget instances (shim identity).
    assert all(isinstance(tw, subtitle_text_widget._SubtitleTextWidget) for tw in win._text_widgets)

    win.set_reduce_motion(True)
    # Rapid successive updates must not raise. The second may be queued behind
    # the min-display-time (1.5s) timer, so only the immediate insert is
    # guaranteed; clear() must drop any such pending insertion too.
    win.update_text("これは日本語の原文です", {"zh": "这是中文翻译，含、标点。"})  # noqa: RUF001
    assert len(win._sentences) == 1
    win.update_text("原文带换行\n第二行", {"": "多行翻译\n第二行"})
    qapp.processEvents()
    assert len(win._sentences) >= 1

    win.clear()
    assert win._sentences == []
    qapp.processEvents()
    win.close()
