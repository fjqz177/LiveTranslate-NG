"""Theme / style data for the overlay UI — the single source of truth.

DEFAULT_STYLE, STYLE_PRESETS and their helpers live here so adding a new
preset only touches this one module (previously the preset list, the style
dict and the settings collector each hardcoded their own copies).

Contrast guard (CORE-9): subtitle text must keep >= 7:1 against the
background (WCAG AAA); secondary metadata (timestamps) >= 3:1; presets
with a translucent background (bg_opacity < 200) must declare an outline
compensation of at least 2 px because their effective contrast depends on
the desktop behind the window. validate_style_contrast() enforces this and
tests/core/test_theme.py pins it for every preset.
"""

from typing import Any

# Contrast thresholds (WCAG 2.x ratios).
MIN_SUBTITLE_CONTRAST = 7.0  # original/translation text — the product's core
MIN_META_CONTRAST = 3.0  # timestamps / secondary metadata
OUTLINE_REQUIRED_BELOW_OPACITY = 200  # translucent backgrounds need strokes
MIN_OUTLINE_COMPENSATION = 2  # px

DEFAULT_STYLE = {
    "preset": "default",
    "bg_color": "#000000",
    "bg_opacity": 240,
    "header_color": "#1a1a2e",
    "header_opacity": 230,
    "border_radius": 8,
    "original_font_family": "Microsoft YaHei",
    "translation_font_family": "Microsoft YaHei",
    "original_font_size": 11,
    "translation_font_size": 14,
    "original_color": "#cccccc",
    "translation_color": "#ffffff",
    "timestamp_color": "#888899",
    "window_opacity": 95,
    "outline_compensation": 0,
}

_BASE = DEFAULT_STYLE

STYLE_PRESETS = {
    "default": dict(_BASE),
    "transparent": {
        **_BASE,
        "preset": "transparent",
        "bg_opacity": 120,
        "header_opacity": 120,
        "window_opacity": 70,
        # Translucent background: contrast depends on the desktop behind the
        # window, so the overlay must draw a >= 2 px outline on subtitle text.
        "outline_compensation": 2,
    },
    "compact": {
        **_BASE,
        "preset": "compact",
        "original_font_size": 9,
        "translation_font_size": 11,
    },
    "light": {
        **_BASE,
        "preset": "light",
        "bg_color": "#e8e8f0",
        "bg_opacity": 230,
        "header_color": "#c8c8d8",
        "header_opacity": 220,
        "original_color": "#333333",
        "translation_color": "#111111",
        "timestamp_color": "#666688",
    },
    "dracula": {
        **_BASE,
        "preset": "dracula",
        "bg_color": "#282a36",
        "bg_opacity": 235,
        "header_color": "#44475a",
        "header_opacity": 230,
        "original_color": "#f8f8f2",
        "translation_color": "#f8f8f2",
        "timestamp_color": "#6272a4",
    },
    "nord": {
        **_BASE,
        "preset": "nord",
        "bg_color": "#2e3440",
        "bg_opacity": 235,
        "header_color": "#3b4252",
        "header_opacity": 230,
        "original_color": "#d8dee9",
        "translation_color": "#eceff4",
        # #7b88a1: brightened from nord3 #4c566a (1.69:1) to clear the 3:1
        # meta guard while staying in the nord blue family.
        "timestamp_color": "#7b88a1",
    },
    "monokai": {
        **_BASE,
        "preset": "monokai",
        "bg_color": "#272822",
        "bg_opacity": 235,
        "header_color": "#3e3d32",
        "header_opacity": 230,
        "original_color": "#f8f8f2",
        "translation_color": "#f8f8f2",
        "timestamp_color": "#75715e",
    },
    "solarized": {
        **_BASE,
        "preset": "solarized",
        "bg_color": "#002b36",
        "bg_opacity": 235,
        "header_color": "#073642",
        "header_opacity": 230,
        # #a7b8bd: brightened from canonical base0 #839496 (4.75:1) to meet
        # the 7:1 subtitle guard; #657b83 (base00) for the 3:1 meta guard.
        "original_color": "#a7b8bd",
        "translation_color": "#eee8d5",
        "timestamp_color": "#657b83",
    },
    "gruvbox": {
        **_BASE,
        "preset": "gruvbox",
        "bg_color": "#282828",
        "bg_opacity": 235,
        "header_color": "#3c3836",
        "header_opacity": 230,
        "original_color": "#ebdbb2",
        "translation_color": "#fbf1c7",
        "timestamp_color": "#928374",
    },
    "tokyo_night": {
        **_BASE,
        "preset": "tokyo_night",
        "bg_color": "#1a1b26",
        "bg_opacity": 235,
        "header_color": "#24283b",
        "header_opacity": 230,
        "original_color": "#a9b1d6",
        "translation_color": "#c0caf5",
        # #737aa2: brightened from #565f89 (2.76:1) to clear the 3:1 meta
        # guard (tokyo night "comment light").
        "timestamp_color": "#737aa2",
    },
    "catppuccin": {
        **_BASE,
        "preset": "catppuccin",
        "bg_color": "#1e1e2e",
        "bg_opacity": 235,
        "header_color": "#313244",
        "header_opacity": 230,
        "original_color": "#cdd6f4",
        "translation_color": "#cdd6f4",
        "timestamp_color": "#6c7086",
    },
    "one_dark": {
        **_BASE,
        "preset": "one_dark",
        "bg_color": "#282c34",
        "bg_opacity": 235,
        "header_color": "#3e4452",
        "header_opacity": 230,
        # #c8ccd4: brightened from #abb2bf (6.57:1) to meet the 7:1 subtitle
        # guard; #7d8590 clears the 3:1 meta guard for timestamps.
        "original_color": "#c8ccd4",
        "translation_color": "#e5c07b",
        "timestamp_color": "#7d8590",
    },
    "everforest": {
        **_BASE,
        "preset": "everforest",
        "bg_color": "#2d353b",
        "bg_opacity": 235,
        "header_color": "#343f44",
        "header_opacity": 230,
        "original_color": "#d3c6aa",
        "translation_color": "#d3c6aa",
        "timestamp_color": "#859289",
    },
    "kanagawa": {
        **_BASE,
        "preset": "kanagawa",
        "bg_color": "#1f1f28",
        "bg_opacity": 235,
        "header_color": "#2a2a37",
        "header_opacity": 230,
        "original_color": "#dcd7ba",
        "translation_color": "#dcd7ba",
        # #727189: brightened from #54546d (2.23:1) to clear the 3:1 meta
        # guard while staying in the kanagawa sumi-ink family.
        "timestamp_color": "#727189",
    },
}

PRESET_NAMES = list(STYLE_PRESETS)

# UI-4: overlay meta-chrome accent colors (source-language tag, ASR/TL
# latency, monitor labels, context menu). Single source of truth — the
# overlay used to hardcode a VS Code-ish palette in six places while the
# design tokens (§3.5) lived only in the panel chrome.
META_COLOR_SOURCE_LANG = "#6cf"
META_COLOR_ASR = "#8b8"
META_COLOR_TL = "#db8"
META_COLOR_SEPARATOR = "#555"
META_COLOR_MENU_BG = "#2a2a3a"
META_COLOR_MENU_FG = "#ddd"
META_COLOR_MENU_BORDER = "#555"


def hex_to_rgba(hex_color: str, opacity: int) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


def migrate_style(style: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a saved style dict: merge defaults and migrate the legacy
    single font_family key to the split original/translation fields."""
    style = dict(style or {})
    s = {**DEFAULT_STYLE, **style}
    if "font_family" in s and "original_font_family" not in style:
        s["original_font_family"] = s["font_family"]
        s["translation_font_family"] = s["font_family"]
    return s


# ── Contrast guard (WCAG 2.x) ────────────────────────────────────────────


def relative_luminance(hex_color: str) -> float:
    """WCAG 2.x relative luminance (0 = black, 1 = white)."""
    h = hex_color.lstrip("#")

    def lin(c8: int) -> float:
        c = c8 / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG contrast ratio between two #rrggbb colors (1.0 to 21.0)."""
    l1 = relative_luminance(fg)
    l2 = relative_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def validate_style_contrast(style: dict[str, Any]) -> list[str]:
    """Return human-readable violations of the contrast policy (empty = ok).

    Policy: subtitle text (original/translation) >= 7:1 against the
    background; timestamp metadata >= 3:1; translucent backgrounds
    (bg_opacity < 200) require an outline compensation of >= 2 px because
    the effective contrast then depends on the desktop behind the window.
    """
    s = migrate_style(style)
    problems: list[str] = []
    bg = str(s["bg_color"])
    for key in ("translation_color", "original_color"):
        ratio = contrast_ratio(str(s[key]), bg)
        if ratio < MIN_SUBTITLE_CONTRAST:
            problems.append(
                f"{key} {s[key]} on {bg} is {ratio:.2f}:1 "
                f"(subtitle text needs >= {MIN_SUBTITLE_CONTRAST:.0f}:1)"
            )
    ts_ratio = contrast_ratio(str(s["timestamp_color"]), bg)
    if ts_ratio < MIN_META_CONTRAST:
        problems.append(
            f"timestamp_color {s['timestamp_color']} on {bg} is {ts_ratio:.2f}:1 "
            f"(metadata needs >= {MIN_META_CONTRAST:.0f}:1)"
        )
    if int(s["bg_opacity"]) < OUTLINE_REQUIRED_BELOW_OPACITY:
        outline = int(s.get("outline_compensation", 0))
        if outline < MIN_OUTLINE_COMPENSATION:
            problems.append(
                f"bg_opacity {s['bg_opacity']} < {OUTLINE_REQUIRED_BELOW_OPACITY} "
                f"requires outline_compensation >= {MIN_OUTLINE_COMPENSATION}px (got {outline})"
            )
    return problems
