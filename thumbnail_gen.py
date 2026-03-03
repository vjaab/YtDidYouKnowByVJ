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
from google import genai
from google.genai import types
from config import OUTPUT_DIR, ASSETS_DIR, GEMINI_API_KEY

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
    """
    Generates a viral YouTube thumbnail using the Nanobanana AI Image model.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(OUTPUT_DIR, f"thumbnail_{today}.jpg")

    title = script_json.get("title", "Tech News")
    headline = script_json.get("thumbnail_headline", title)
    highlight = script_json.get("thumbnail_highlight_word", "")
    teaser = script_json.get("thumbnail_teaser", "AI Revolution")
    emoji = script_json.get("thumbnail_emoji", "🤖")
    category = script_json.get("sub_category", "AI Research")

    # Construct the NanoBanana Prompt
    # High-CTR formula: Subject on left, Large stylized text on right, vibrant neon colors
    prompt = f"""
    Create a viral high-CTR YouTube Short thumbnail (1280x720).
    STYLE: Nanobanana High-Fidelity. 
    VISUAL: {emoji} themed background with cinematic lighting, depth of field.
    TEXT CONTENT: 
    - MAIN HEADLINE: "{headline.upper()}" (Large, bold, 3D typography)
    - HIGHLIGHT WORD: "{highlight.upper()}" (Yellow background glow)
    - TEASER: "{teaser}" (Bottom third, futuristic font)
    - CATEGORY: "{category}" (Badge in corner)
    
    COMPOSITION: 
    - Subject/Emoji on the left side.
    - Text on the right side.
    - Dark bottom bar with "t.me/technewsbyvj | linkedin.com/in/vijayakumar-j/ | wa.link/vj-wa" in small white font.
    - Vibrant colors, high saturation, 8k quality, sharp focus.
    """

    print(f"Nanobanana → Generating AI Thumbnail for: {headline}...")
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_image(
            model='nanobanana-3-pro-image',
            prompt=prompt,
            config=types.GenerateImageConfig(
                number_of_images=1,
                width=1280,
                height=720,
                output_mime_type="image/jpeg"
            )
        )
        
        # Save the result
        if response.generated_images:
            img_bytes = response.generated_images[0].image_bytes
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            print(f"Nanobanana Thumbnail Saved: {output_path}")
            return output_path
        else:
            raise Exception("No image generated by Nanobanana")

    except Exception as e:
        print(f"Nanobanana failed: {e}. Falling back to legacy Pillow renderer...")
        # Fallback to local Pillow version (we already have it below)
        return _fallback_generate_thumbnail(script_json, output_path)

def _fallback_generate_thumbnail(script_json, output_path):
    title        = script_json.get("title", "Tech News")
    thumb_headline    = script_json.get("thumbnail_headline", title[:30])
    thumb_highlight   = script_json.get("thumbnail_highlight_word", "")
    thumb_teaser      = script_json.get("thumbnail_teaser", "AI Revolution")
    thumb_emoji       = script_json.get("thumbnail_emoji", "🤖")
    sub_category      = script_json.get("sub_category", "AI")
    accent_hex        = script_json.get("color_theme", {}).get("accent", "#ff4444").lstrip("#")
    accent_color      = tuple(int(accent_hex[i:i+2], 16) for i in (0, 2, 4))

    # ── Canvas ────────────────────────────────────────────────────────────────
    bg = _load_background(script_json)
    bg = _vignette(bg)
    canvas = bg.copy()
    draw   = ImageDraw.Draw(canvas)

    # Dark overlay gradient
    overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    for x in range(THUMB_W):
        faction = x / THUMB_W
        base_a  = int(160 - 80 * faction)
        r, g, b = [int(c * 0.15) for c in accent_color]
        ov_draw.line([(x, 0), (x, THUMB_H)], fill=(r, g, b, base_a))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    # ── LEFT PANEL (Emoji) ───────────────────────────────────────────────────
    cx, cy = int(THUMB_W * 0.20), THUMB_H // 2
    _draw_star_burst(draw, cx, cy, 130, 95, 12, (*accent_color, 180))
    try:
        ef = gf(180)
        ew, eh = ts(thumb_emoji, ef)
        draw.text((cx - ew // 2, cy - eh // 2 - 10), thumb_emoji, font=ef, fill=(255, 255, 255, 255))
    except:
        draw.ellipse([cx-80, cy-80, cx+80, cy+80], fill=accent_color)

    # ── RIGHT PANEL (Text) ───────────────────────────────────────────────────
    rx = int(THUMB_W * 0.42)
    # Badge
    badge_text = f"🤖 {sub_category.upper()}"
    bf = gf(28)
    bw, bh = ts(badge_text, bf)
    draw.rounded_rectangle([rx, 60, rx + bw + 36, 60 + bh + 20], radius=22, fill=(*accent_color, 230))
    draw.text((rx + 18, 70), badge_text, font=bf, fill=(255, 255, 255))

    # Headline
    hf = gf(74)
    line_y = 60 + bh + 48
    draw.text((rx, line_y), thumb_headline, font=hf, fill=(255, 255, 255))
    
    # Teaser
    tf = gf(34)
    draw.text((rx, line_y + 100), thumb_teaser, font=tf, fill=(255, 214, 0))

    # ── BOTTOM BAR ───────────────────────────────────────────────────────────
    bar_y = THUMB_H - 60
    draw.rectangle([0, bar_y, THUMB_W, THUMB_H], fill=(0, 0, 0, 210))
    bf2 = gf(24)
    cta = "t.me/technewsbyvj  |  linkedin.com/in/vj  |  wa.link/vj-wa"
    cw, _ = ts(cta, bf2)
    draw.text(((THUMB_W - cw) // 2, bar_y + 18), cta, font=bf2, fill=(255, 255, 255))

    canvas.save(output_path, "JPEG", quality=95)
    return output_path
