"""Make the app icons (icons/) from the DKPRP logo: DECK001's solid bar contour waveform.

    python tools/make_icons.py        (needs Pillow)

The logo is five coral bars around a centre line, the same geometry as the sidebar mark in
ui/index.html (viewBox 48x30), on the dark rounded square the icons already used.
"""
from pathlib import Path

from PIL import Image, ImageDraw

ICONS = Path(__file__).resolve().parent.parent / "icons"
BG, BAR = (22, 22, 24, 255), (240, 87, 76, 255)          # --panel (dark) and the solid contour coral
LOGO = [0.42, 0.78, 1.0, 0.62, 0.86]                     # bar heights, as in the sidebar mark


def draw(size: int, margin: float = 0.07, radius: float = 0.22) -> Image.Image:
    """The icon at `size` px, drawn 4x larger and scaled down for smooth edges."""
    s = size * 4
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    m = round(s * margin)
    d.rounded_rectangle([m, m, s - m, s - m], radius=round(s * radius), fill=BG)
    # The bars span 44 of the logo's 48 units; 64 % of the icon wide, centred.
    unit = s * 0.64 / 48
    x0, cy = (s - 44 * unit) / 2, s / 2
    for i, v in enumerate(LOGO):
        x, half = x0 + i * 9.5 * unit, v * 14 * unit
        d.rounded_rectangle([x, cy - half, x + 6 * unit, cy + half], radius=unit * 0.8, fill=BAR)
    return im.resize((size, size), Image.LANCZOS)


def main():
    sizes = {"32x32.png": 32, "64x64.png": 64, "128x128.png": 128, "128x128@2x.png": 256, "icon.png": 512,
             "StoreLogo.png": 50, **{f"Square{n}x{n}Logo.png": n for n in (30, 44, 71, 89, 107, 142, 150, 284, 310)}}
    for name, n in sizes.items():
        draw(n).save(ICONS / name)
    big = draw(1024)
    big.save(ICONS / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)])
    big.save(ICONS / "icon.icns")
    print(f"wrote {len(sizes) + 2} files to {ICONS}")


if __name__ == "__main__":
    main()
