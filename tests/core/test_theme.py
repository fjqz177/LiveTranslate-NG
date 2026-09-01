"""Tests for theme.py: the single source of truth for overlay styles."""

import pytest

from livetranslate.core.theme import (
    DEFAULT_STYLE,
    MIN_META_CONTRAST,
    MIN_OUTLINE_COMPENSATION,
    MIN_SUBTITLE_CONTRAST,
    PRESET_NAMES,
    STYLE_PRESETS,
    contrast_ratio,
    hex_to_rgba,
    migrate_style,
    relative_luminance,
    validate_style_contrast,
)


class TestHexToRgba:
    def test_basic(self):
        assert hex_to_rgba("#010203", 255) == "rgba(1,2,3,255)"

    def test_no_hash(self):
        assert hex_to_rgba("ff0000", 128) == "rgba(255,0,0,128)"

    def test_zero_opacity(self):
        assert hex_to_rgba("#000000", 0) == "rgba(0,0,0,0)"


class TestMigrateStyle:
    def test_defaults_merged(self):
        s = migrate_style({"preset": "dracula"})
        assert s["bg_color"] == DEFAULT_STYLE["bg_color"]
        assert s["preset"] == "dracula"

    def test_legacy_font_family_split(self):
        s = migrate_style({"font_family": "Consolas"})
        assert s["original_font_family"] == "Consolas"
        assert s["translation_font_family"] == "Consolas"

    def test_split_fields_not_overwritten_by_legacy_key(self):
        s = migrate_style(
            {
                "font_family": "Legacy",
                "original_font_family": "Arial",
                "translation_font_family": "Arial",
            }
        )
        assert s["original_font_family"] == "Arial"
        assert s["translation_font_family"] == "Arial"

    def test_none_input(self):
        assert migrate_style(None)["preset"] == "default"


class TestPresets:
    def test_preset_names_match_registry(self):
        assert list(STYLE_PRESETS) == PRESET_NAMES
        assert len(PRESET_NAMES) == 14  # default + transparent + compact + 11 themes

    def test_every_preset_carries_preset_key(self):
        for key, style in STYLE_PRESETS.items():
            assert style["preset"] == key, key

    def test_every_preset_has_full_field_set(self):
        for key, style in STYLE_PRESETS.items():
            assert set(DEFAULT_STYLE) <= set(style), key


class TestContrastMath:
    def test_luminance_extremes(self):
        assert relative_luminance("#000000") == pytest.approx(0.0)
        assert relative_luminance("#ffffff") == pytest.approx(1.0)

    def test_ratio_extremes(self):
        assert contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0)
        assert contrast_ratio("#777777", "#777777") == pytest.approx(1.0)

    def test_ratio_is_symmetric(self):
        assert contrast_ratio("#cccccc", "#000000") == pytest.approx(
            contrast_ratio("#000000", "#cccccc")
        )

    def test_known_reference_value(self):
        # WCAG reference: #767676 on white is the 4.5:1 AA boundary.
        assert contrast_ratio("#767676", "#ffffff") == pytest.approx(4.54, abs=0.05)


class TestContrastGuard:
    """CORE-9: CLAUDE.md §14 promises >= 7:1 subtitle contrast (or >= 2 px
    outline compensation). This guard makes the promise enforceable — every
    shipped preset must pass, so no low-contrast theme can silently land."""

    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_every_preset_passes_the_guard(self, name):
        problems = validate_style_contrast(STYLE_PRESETS[name])
        assert problems == [], f"{name}: {problems}"

    def test_low_contrast_subtitle_text_is_rejected(self):
        style = {**DEFAULT_STYLE, "translation_color": "#444444", "bg_color": "#000000"}
        problems = validate_style_contrast(style)
        assert any("translation_color" in p for p in problems)

    def test_low_contrast_timestamp_is_rejected(self):
        style = {**DEFAULT_STYLE, "timestamp_color": "#111111", "bg_color": "#000000"}
        problems = validate_style_contrast(style)
        assert any("timestamp_color" in p for p in problems)

    def test_translucent_background_requires_outline(self):
        style = {**DEFAULT_STYLE, "bg_opacity": 120, "outline_compensation": 0}
        problems = validate_style_contrast(style)
        assert any("outline_compensation" in p for p in problems)
        style["outline_compensation"] = MIN_OUTLINE_COMPENSATION
        assert validate_style_contrast(style) == []

    def test_thresholds_are_the_documented_ones(self):
        assert MIN_SUBTITLE_CONTRAST == 7.0
        assert MIN_META_CONTRAST == 3.0
