from pathlib import Path


def _resolve_image_path(path: str) -> str:
    """Resolve image path (relative to project dir or absolute)."""
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute():
        return str(p) if p.exists() else ""
    resolved = Path(__file__).parent / p
    return str(resolved) if resolved.exists() else ""


# Default subtitle window settings
DEFAULT_SUBTITLE_WIN_SETTINGS = {
    "enabled": False,
    "sentences": 1,
    "window_width": 1000,
    "line_spacing": 8,
    "bg_color": "#000000",
    "bg_opacity": 76,
    "bg_image": "",
    "border_radius": 8,
    "auto_hide_timeout": 5,
    "auto_hide_animation": "fade",
    "auto_hide_duration": 300,
    "click_through": False,
    "lines": [
        {
            "type": "original",
            "enabled": True,
            "font_family": "Microsoft YaHei",
            "font_size": 24,
            "color": "#FFFFFF",
            "opacity": 255,
            "outline_enabled": True,
            "outline_color": "#000000",
            "outline_width": 2,
            "align": "center",
            "bg_image": "",
            "entry_animation": "none",
            "exit_animation": "none",
            "animation_duration": 300,
        },
        {
            "type": "translation",
            "lang": "zh",
            "enabled": True,
            "font_family": "Microsoft YaHei",
            "font_size": 28,
            "color": "#FFD700",
            "opacity": 255,
            "outline_enabled": True,
            "outline_color": "#000000",
            "outline_width": 2,
            "align": "center",
            "bg_image": "",
            "entry_animation": "none",
            "exit_animation": "none",
            "animation_duration": 300,
        },
    ],
}


def _merge_settings(base, override):
    result = {**base}
    for k, v in (override or {}).items():
        if k == "lines" and isinstance(v, list):
            result["lines"] = v
        else:
            result[k] = v
    return result
