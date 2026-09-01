"""Generate the LiveTranslate icon family (plan §3.5.7).

Style: flat, two-colour linework — a cold off-black rounded square, a
white speech-bubble outline and a single accent (blue) waveform inside.
No gradients, no glossy "AI" clichés. Tray variants add a small status
dot (run/pause/error) in the §3.5.2 semantic colours.

Also generates the small spinbox up/down arrow images used by the panel
chrome QSS (Qt stylesheets cannot draw CSS border triangles, so the
arrows ship as PNG assets; see ui/panel/_chrome.py).

Run from the repository root:  uv run scripts/generate_icons.py
Output: assets/icons/ (committed; installers pick them up in Phase 7).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 1024
ACCENT = (76, 141, 255, 255)  # #4C8DFF
CANVAS = (16, 20, 27, 255)  # #10141B cold off-black
LINE = (242, 244, 247, 255)  # #F2F4F7
STATUS = {
    "run": (63, 185, 80, 255),  # #3FB950
    "pause": (210, 153, 34, 255),  # #D29922
    "error": (248, 81, 73, 255),  # #F85149
}
# Spinbox arrow colours per chrome theme (must match _chrome.py).
SPIN_ARROW = {
    "dark": (154, 163, 178, 255),  # #9AA3B2
    "light": (87, 96, 106, 255),  # #57606A
}
OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "icons"


def _rounded(draw, box, radius, **kw):
    draw.rounded_rectangle(box, radius=radius, **kw)


def draw_base(canvas: Image.Image) -> ImageDraw.ImageDraw:
    draw = ImageDraw.Draw(canvas)
    s = canvas.width
    # Canvas
    _rounded(draw, (0, 0, s - 1, s - 1), radius=int(s * 0.22), fill=CANVAS)
    # Speech bubble outline (white linework)
    stroke = max(3, int(s * 0.028))
    bx0, by0, bx1, by1 = int(s * 0.22), int(s * 0.26), int(s * 0.78), int(s * 0.64)
    _rounded(draw, (bx0, by0, bx1, by1), radius=int(s * 0.12), outline=LINE, width=stroke)
    # Bubble tail (outline only, two strokes meeting at the tip)
    tip = (int(s * 0.30), int(s * 0.80))
    draw.line((bx0 + int(s * 0.06), by1, tip[0], tip[1]), fill=LINE, width=stroke)
    draw.line((tip[0], tip[1], bx0 + int(s * 0.24), by1), fill=LINE, width=stroke)
    # Waveform bars in the single accent colour, rounded caps
    bar_w = int(s * 0.055)
    centers = [0.36, 0.46, 0.56, 0.66]
    heights = [0.14, 0.26, 0.20, 0.10]
    mid = (by0 + by1) / 2
    for cx, h in zip(centers, heights, strict=True):
        x = int(s * cx)
        y0 = int(mid - s * h / 2)
        y1 = int(mid + s * h / 2)
        _rounded(draw, (x - bar_w // 2, y0, x + bar_w // 2, y1), radius=bar_w // 2, fill=ACCENT)
    return draw


def draw_status(canvas: Image.Image, status: str) -> None:
    """Add the tray status dot (plan §3.5.7: run/pause/error)."""
    s = canvas.width
    draw = ImageDraw.Draw(canvas)
    r = int(s * 0.135)
    cx, cy = int(s * 0.82), int(s * 0.82)
    ring = max(2, int(s * 0.014))
    draw.ellipse(
        (cx - r - ring, cy - r - ring, cx + r + ring, cy + r + ring),
        fill=CANVAS,
    )
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=STATUS[status])


def render(size: int, status: str | None) -> Image.Image:
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw_base(canvas)
    if status:
        draw_status(canvas, status)
    return canvas.resize((size, size), Image.LANCZOS)


def render_spin_arrow(up: bool, color: tuple[int, int, int, int]) -> Image.Image:
    """Small solid triangle for spinbox up/down buttons.

    Rendered at 2x (16x12 canvas, 8x5 triangle centred) so HiDPI screens
    stay crisp; the panel chrome QSS draws it at 8x6 logical pixels.
    """
    w, h = 16, 12  # 2x logical 8x6
    t_h = 10  # 2x triangle height 5, vertically centred
    top = (h - t_h) // 2
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    if up:
        points = [(w / 2, top), (0, top + t_h - 1), (w - 1, top + t_h - 1)]
    else:
        points = [(0, top), (w - 1, top), (w / 2, top + t_h - 1)]
    draw.polygon(points, fill=color)
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_sizes = [16, 24, 32, 48, 64, 128, 256, 512]
    for size in png_sizes:
        render(size, None).save(OUT_DIR / f"app_{size}.png")
        for status in STATUS:
            render(size, status).save(OUT_DIR / f"tray_{status}_{size}.png")
    render(512, None).save(OUT_DIR / "app.png")
    # Spinbox arrows per theme (Qt cannot render CSS border triangles).
    for theme, color in SPIN_ARROW.items():
        render_spin_arrow(True, color).save(OUT_DIR / f"spin_up_{theme}.png")
        render_spin_arrow(False, color).save(OUT_DIR / f"spin_down_{theme}.png")
    # Windows: multi-size .ico; macOS: .icns (Pillow supports both)
    render(256, None).save(
        OUT_DIR / "app.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]
    )
    render(1024, None).save(
        OUT_DIR / "app.icns", sizes=[(s, s) for s in (16, 32, 48, 128, 256, 512, 1024)]
    )
    print(f"icons written to {OUT_DIR}")


if __name__ == "__main__":
    main()
