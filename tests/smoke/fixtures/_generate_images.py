"""Render the three smoke-test JPEGs deterministically.

Run once after changing any image layout:

    python -m tests.smoke.fixtures._generate_images

The gold action sequences in `expected.json` reference pixel positions
declared at the top of each `_render_*` function — keep those in sync if
you move buttons around.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CANVAS = (800, 500)
HERE = Path(__file__).parent


def _font(size: int) -> ImageFont.ImageFont:
    # Try a real font if Pillow can find one; fall back to default bitmap font.
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _new(bg: tuple[int, int, int] = (32, 36, 44)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", CANVAS, color=bg)
    return img, ImageDraw.Draw(img)


def _render_click_start_button(out: Path) -> None:
    # Layout: dark background, centered green Start button.
    # Button center -> (400, 280); pointer assumed at canvas centre (400, 250).
    img, d = _new()
    title_font = _font(28)
    btn_font = _font(36)

    d.text((24, 24), "Razer Demo Launcher", fill=(220, 220, 220), font=title_font)
    d.text((24, 70), "Welcome back. Press Start to continue.", fill=(160, 160, 160), font=title_font)

    # Start button rect (315, 240) - (485, 320). Centre = (400, 280).
    d.rounded_rectangle((315, 240, 485, 320), radius=12, fill=(0, 156, 70))
    d.text((360, 254), "Start", fill=(255, 255, 255), font=btn_font)

    img.save(out, format="JPEG", quality=88)


def _render_dismiss_update_popup(out: Path) -> None:
    # Layout: a recording app behind a centered modal "Software Update".
    img, d = _new(bg=(20, 20, 28))
    title_font = _font(24)
    body_font = _font(18)
    btn_font = _font(20)

    # Faint background "REC" indicator so the model sees a recording session.
    d.text((24, 24), "Razer Cortex - Recording (00:14:22)", fill=(120, 120, 120), font=title_font)
    d.ellipse((24, 64, 44, 84), fill=(220, 60, 60))
    d.text((54, 64), "REC", fill=(220, 60, 60), font=title_font)

    # Modal: centered around (250, 140) - (550, 360).
    d.rectangle((230, 140, 570, 360), fill=(48, 50, 60), outline=(120, 120, 130), width=2)
    d.text((250, 156), "Software Update Available", fill=(245, 245, 245), font=title_font)
    d.text(
        (250, 200),
        "A new update for your driver is ready.\nUpdate now to apply changes.",
        fill=(200, 200, 200),
        font=body_font,
    )

    # [Update Now] (red) and [Later] (grey) buttons.
    d.rounded_rectangle((260, 290, 400, 340), radius=8, fill=(170, 50, 50))
    d.text((282, 302), "Update Now", fill=(255, 255, 255), font=btn_font)
    d.rounded_rectangle((420, 290, 540, 340), radius=8, fill=(80, 80, 90))
    d.text((460, 302), "Later", fill=(255, 255, 255), font=btn_font)

    img.save(out, format="JPEG", quality=88)


def _render_pause_then_resume_video(out: Path) -> None:
    # Layout: a video player with a paused preview frame.
    img, d = _new(bg=(0, 0, 0))
    title_font = _font(20)
    overlay_font = _font(60)

    # Fake video content (a dark scene with a moving subject).
    d.rectangle((0, 0, CANVAS[0], 440), fill=(12, 18, 30))
    d.text((24, 24), "Razer Stream - Live", fill=(180, 180, 180), font=title_font)

    # A subject mid-frame — a stylised character silhouette to suggest motion.
    d.ellipse((360, 150, 440, 230), fill=(220, 200, 160))           # head
    d.rectangle((345, 230, 455, 360), fill=(40, 80, 140))            # torso
    d.rectangle((355, 360, 395, 430), fill=(60, 60, 90))             # left leg
    d.rectangle((405, 360, 445, 430), fill=(60, 60, 90))             # right leg
    # Motion blur streaks behind the subject.
    for y in (200, 260, 320):
        d.line((180, y, 330, y), fill=(80, 100, 130), width=4)

    # Bottom timeline with a pause icon (two vertical bars = currently playing).
    d.rectangle((0, 440, CANVAS[0], CANVAS[1]), fill=(20, 20, 28))
    d.rectangle((28, 456, 36, 488), fill=(220, 220, 220))           # pause bar 1
    d.rectangle((44, 456, 52, 488), fill=(220, 220, 220))           # pause bar 2
    d.text((72, 460), "00:42 / 03:18  -  PLAYING", fill=(220, 220, 220), font=title_font)
    # Progress bar.
    d.rectangle((28, 494, CANVAS[0] - 28, 498), fill=(60, 60, 70))
    d.rectangle((28, 494, 28 + int((CANVAS[0] - 56) * 0.21), 498), fill=(0, 156, 70))

    img.save(out, format="JPEG", quality=88)


RENDERERS = {
    "01_click_start_button": _render_click_start_button,
    "02_dismiss_update_popup": _render_dismiss_update_popup,
    "03_pause_then_resume_video": _render_pause_then_resume_video,
}


def main() -> None:
    for name, render in RENDERERS.items():
        out = HERE / name / "image.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        render(out)
        print(f"wrote {out.relative_to(HERE.parent.parent.parent)}")


if __name__ == "__main__":
    main()
