"""
thumbnail_gen.py — High-CTR thumbnail generator (1280x720).

Layout:
  LEFT 40%:  Large emoji + star burst  
  RIGHT 60%: Category badge | Headline (with highlighted word) | Teaser
  TOP RIGHT: Red "TODAY" circle
  BOTTOM:    Dark bar with Telegram link + subscribe bell
"""

import os
import io
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from datetime import datetime
from config import OUTPUT_DIR, ASSETS_DIR

THUMB_W, THUMB_H = 1280, 720

FONT_PATHS = [
    os.path.join(ASSETS_DIR, "fonts", "Montserrat-Bold.ttf"),
    os.path.join(ASSETS_DIR, "fonts", "Roboto-Bold.ttf"),
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/roboto/hinted/Roboto-Bold.ttf",
    "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf",
]
_fcache = {}

def gf(size):
    if size not in _fcache:
        for p in FONT_PATHS:
            if os.path.exists(p):
                try:
                    _fcache[size] = ImageFont.truetype(p, size)
                    break
                except Exception:
                    continue
        if size not in _fcache:
            _fcache[size] = ImageFont.load_default()
    return _fcache[size]

def ts(text, font):
    bb = font.getbbox(text)
    return bb[2] - bb[0], bb[3] - bb[1]


def _load_background(script_json):
    """Load best available background image, apply grading."""
    # Try first chunk's visual if available
    today = datetime.now().strftime("%Y-%m-%d")
    for i in range(1, 5):
        for ext in ["jpg", "png", "mp4"]:
            p = os.path.join(OUTPUT_DIR, f"chunk_{i}_{today}.{ext}")
            if os.path.exists(p) and ext != "mp4":
                try:
                    img = Image.open(p).convert("RGB").resize((THUMB_W, THUMB_H), Image.LANCZOS)
                    # Boost contrast + saturation
                    img = ImageEnhance.Contrast(img).enhance(1.25)
                    img = ImageEnhance.Color(img).enhance(1.30)
                    return img
                except Exception:
                    continue

    # Solid dark fallback
    accent_hex = script_json.get("color_theme", {}).get("accent", "#ff4444").lstrip("#")
    acc = tuple(int(accent_hex[i:i+2], 16) for i in (0, 2, 4))
    img = Image.new("RGB", (THUMB_W, THUMB_H), (15, 15, 20))
    # Subtle gradient from accent on left
    for x in range(THUMB_W // 2):
        alpha = int(60 * (1 - x / (THUMB_W // 2)))
        for y in range(THUMB_H):
            px = img.getpixel((x, y))
            blended = tuple(min(255, px[c] + acc[c] * alpha // 255) for c in range(3))
            img.putpixel((x, y), blended)
    return img


def _vignette(img):
    vign = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(vign)
    w, h = img.size
    for i in range(80):
        alpha = int(180 * (i / 80) ** 1.5)
        draw.rectangle([i, i, w-i, h-i], outline=(0, 0, 0, alpha))
    result = Image.alpha_composite(img.convert("RGBA"), vign)
    return result.convert("RGB")


def _draw_star_burst(draw, cx, cy, r_outer, r_inner, points, color):
    pts = []
    for i in range(points * 2):
        angle = math.pi * i / points - math.pi / 2
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, fill=color)


def generate_thumbnail(script_json):
    today = datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(OUTPUT_DIR, f"thumbnail_{today}.jpg")

    title        = script_json.get("title", "Tech News")
    thumb_headline    = script_json.get("thumbnail_headline", title[:30])
    thumb_highlight   = script_json.get("thumbnail_highlight_word", "")
    thumb_teaser      = script_json.get("thumbnail_teaser", "This changes everything")
    thumb_emoji       = script_json.get("thumbnail_emoji", script_json.get("relevant_emoji", "🤖"))
    sub_category      = script_json.get("sub_category", "AI")
    accent_hex        = script_json.get("color_theme", {}).get("accent", "#ff4444").lstrip("#")
    accent_color      = tuple(int(accent_hex[i:i+2], 16) for i in (0, 2, 4))

    # ── Canvas ────────────────────────────────────────────────────────────────
    bg = _load_background(script_json)
    bg = _vignette(bg)
    canvas = bg.copy()
    draw   = ImageDraw.Draw(canvas)

    # Dark overlay gradient — category-tinted left, neutral right
    overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    for x in range(THUMB_W):
        faction = x / THUMB_W
        # Darker on left (emoji panel), gradually lifting
        base_a  = int(160 - 80 * faction)
        r = int(accent_color[0] * 0.15)
        g = int(accent_color[1] * 0.15)
        b = int(accent_color[2] * 0.15)
        ov_draw.line([(x, 0), (x, THUMB_H)], fill=(r, g, b, base_a))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    # ── LEFT PANEL (40%): Emoji + star burst ─────────────────────────────────
    cx, cy = int(THUMB_W * 0.20), THUMB_H // 2

    # Star burst behind emoji
    _draw_star_burst(draw, cx, cy, 130, 95, 12,
                     (*accent_color, 180))

    # Emoji — large
    try:
        ef = gf(180)
        ew, eh = ts(thumb_emoji, ef)
        draw.text((cx - ew // 2, cy - eh // 2 - 10), thumb_emoji, font=ef,
                  fill=(255, 255, 255, 255))
    except Exception:
        draw.ellipse([cx-80, cy-80, cx+80, cy+80], fill=accent_color)

    # ── RIGHT PANEL (60%): badge, headline, teaser ────────────────────────────
    rx = int(THUMB_W * 0.42)    # right panel left edge
    rw = THUMB_W - rx - 30      # right panel width

    # Category pill badge
    badge_text = f"🤖 {sub_category.upper()}"
    bf = gf(28)
    bw, bh = ts(badge_text, bf)
    bpad_x, bpad_y = 18, 10
    bx1, by1 = rx, 60
    bx2, by2 = bx1 + bw + bpad_x*2, by1 + bh + bpad_y*2
    draw.rounded_rectangle([bx1, by1, bx2, by2], radius=22, fill=(*accent_color, 230))
    draw.text((bx1 + bpad_x, by1 + bpad_y), badge_text, font=bf, fill=(255, 255, 255))

    # Headline — word by word, highlighted word in yellow box
    words = thumb_headline.split()
    hf_size = 74
    hf = gf(hf_size)
    line_x = rx
    line_y = by2 + 28
    for word in words:
        clean = word.strip(".,!?")
        is_highlight = (clean.lower() == thumb_highlight.lower()) if thumb_highlight else False
        ww, wh = ts(word + " ", hf)

        # Wrap if too wide
        if line_x + ww > THUMB_W - 25 and line_x > rx:
            line_x = rx
            line_y += int(wh * 1.35)

        if is_highlight:
            # Yellow box with slight rotation effect (draw as rectangle, no PIL rotate on draw)
            pad = 8
            draw.rounded_rectangle(
                [line_x - pad, line_y - pad//2, line_x + ww - 8 + pad, line_y + wh + pad//2],
                radius=8, fill=(255, 214, 0, 230)
            )
            for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
                draw.text((line_x+dx, line_y+dy), word, font=hf, fill=(0, 0, 0, 200))
            draw.text((line_x, line_y), word, font=hf, fill=(0, 0, 0, 255))
        else:
            for dx, dy in [(-3,0),(3,0),(0,-3),(0,3)]:
                draw.text((line_x+dx, line_y+dy), word, font=hf, fill=(0, 0, 0, 200))
            draw.text((line_x, line_y), word, font=hf, fill=(255, 255, 255))

        line_x += ww

    # Teaser line below headline
    tf = gf(34)
    tw, th = ts(thumb_teaser, tf)
    ty = line_y + hf_size + 38
    draw.text((rx, ty), thumb_teaser, font=tf, fill=(255, 214, 0))

    # ── TOP RIGHT: "TODAY" red circle ─────────────────────────────────────────
    cr, ccx, ccy = 52, THUMB_W - 80, 68
    draw.ellipse([ccx - cr, ccy - cr, ccx + cr, ccy + cr], fill=(220, 0, 0))
    lf = gf(22)
    lw, lh = ts("TODAY", lf)
    draw.text((ccx - lw//2, ccy - lh//2), "TODAY", font=lf, fill=(255, 255, 255))

    # ── BOTTOM BAR (full width) ────────────────────────────────────────────────
    bar_h = 60
    bar_y = THUMB_H - bar_h
    draw.rectangle([0, bar_y, THUMB_W, THUMB_H], fill=(0, 0, 0, 210))

    bf2 = gf(26)
    tg_text = "t.me/technewsbyvj"
    tw2, _ = ts(tg_text, bf2)
    draw.text(((THUMB_W - tw2) // 2, bar_y + 16), tg_text, font=bf2, fill=(255, 255, 255))

    bell_f = gf(30)
    draw.text((THUMB_W - 55, bar_y + 12), "🔔", font=bell_f, fill=(255, 255, 255))

    # Channel watermark (small, bottom left)
    logo_path = os.path.join(ASSETS_DIR, "logo.png")
    if os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA").resize((44, 44), Image.LANCZOS)
            canvas.paste(logo, (12, bar_y + 8), logo)
        except Exception:
            pass

    canvas.save(output_path, "JPEG", quality=95)
    print(f"Thumbnail saved: {output_path}")
    return output_path
