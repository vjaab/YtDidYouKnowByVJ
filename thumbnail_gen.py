"""
thumbnail_gen.py — Premium Imagen-3 + Authority Avatar Thumbnail Generator.

Design Philosophy (2026 High-Authority Spec):
  - Generative Backdrop: Unique Imagen-3 tech art for every topic.
  - Personal Authority: Seamlessly integrated avatar (cutout).
  - Premium Typography: Montserrat Black, high contrast yellow/white.
  - Curiosity Gap: Professionally written hooks by Gemini.
"""

import os
import io
import math
import random
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from datetime import datetime
from google import genai
from google.genai import types
from rembg import remove
from config import OUTPUT_DIR, ASSETS_DIR, GEMINI_API_KEY

THUMB_W, THUMB_H = 1280, 720
SHORTS_W, SHORTS_H = 1080, 1920

# ── ASSETS ───────────────────────────────────────────────────────────────────
AVATAR_PATH = os.path.join(ASSETS_DIR, "gemini_img_without_logo.png")
FONT_BLACK = os.path.join(ASSETS_DIR, "fonts", "Montserrat-Black.ttf")
FONT_EXTRABOLD = os.path.join(ASSETS_DIR, "fonts", "Montserrat-ExtraBold.ttf")

FALLBACKS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf"]

_fcache = {}

def _load_font(size, weight="black"):
    key = (size, weight)
    if key not in _fcache:
        candidates = [FONT_BLACK, FONT_EXTRABOLD] + FALLBACKS
        for p in candidates:
            if os.path.exists(p):
                try:
                    _fcache[key] = ImageFont.truetype(p, size)
                    break
                except: continue
        if key not in _fcache:
            _fcache[key] = ImageFont.load_default()
    return _fcache[key]

def _text_size(text, font):
    bb = font.getbbox(text)
    return bb[2] - bb[0], bb[3] - bb[1]

# ── AI AGENTS ─────────────────────────────────────────────────────────────────

def _generate_hook_text(title, client):
    """Generates a high-click-through curiosity gap hook."""
    prompt = f"""You are a viral YouTube thumbnail copywriter. 
Generate a SHORT, punchy curiosity gap hook for this topic: "{title}"
RULES: Max 8 words, emotional, curiosity-gap style, end with "...". 
Use \\n for line breaks (max 3 lines). 
Example: "Google is\\nfinally\\nfinished..."
Return ONLY the text."""
    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        hook = response.text.strip().replace("\\n", "\n")
        return "\n".join(hook.split("\n")[:3])
    except:
        return title[:20] + "..."

def _generate_imagen_background(title, client):
    """Generates a thematic tech background using Imagen-3."""
    print(f"🎨 Generating Premium Imagen background for: {title}")
    prompt = (
        f"A high-authority, cinematic tech background for: {title}. "
        "Dark tech aesthetic, glowing neon accents, cybernetic data lines, 8k resolution. "
        "No text, no humans, clean composition. High contrast."
    )
    try:
        response = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio='16:9',
                add_watermark=False
            )
        )
        img_bytes = response.generated_images[0].image.image_bytes
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print(f"⚠️ Imagen failed: {e}. Using dark fallback.")
        return Image.new("RGB", (THUMB_W, THUMB_H), (10, 10, 15))

# ── IMAGE PROCESSING ─────────────────────────────────────────────────────────

def _process_avatar():
    """Removes background from the user avatar and optimizes it for overlay."""
    if not os.path.exists(AVATAR_PATH):
        print("⚠️ Avatar not found at", AVATAR_PATH)
        return None
    try:
        print("👤 Processing avatar (background removal)...")
        input_img = Image.open(AVATAR_PATH)
        # Remove background using rembg
        output_img = remove(input_img)
        return output_img
    except Exception as e:
        print(f"⚠️ Avatar processing failed: {e}")
        return None

# ── RENDERING ─────────────────────────────────────────────────────────────────

def _render_premium_thumbnail(hook_text, bg_img, avatar_img, accent_color, width, height, is_shorts=False):
    canvas = bg_img.resize((width, height), Image.LANCZOS)
    # Darken background slightly for text readability
    canvas = ImageEnhance.Brightness(canvas).enhance(0.7)
    draw = ImageDraw.Draw(canvas)
    
    # 1. Overlay Avatar (Authority)
    if avatar_img:
        # Scale avatar to fit ~70% of height
        av_h = int(height * 0.85) if not is_shorts else int(height * 0.45)
        scale = av_h / avatar_img.height
        av_res = avatar_img.resize((int(avatar_img.width * scale), av_h), Image.LANCZOS)
        
        # Position: Right side for YT, Center/Bottom for Shorts
        if is_shorts:
            pos = (width - av_res.width, height - av_res.height)
        else:
            pos = (width - av_res.width - 20, height - av_res.height)
            
        # Add subtle glow/shadow behind avatar
        glow = av_res.split()[3].point(lambda x: 255 if x > 0 else 0)
        glow = glow.filter(ImageFilter.GaussianBlur(radius=20))
        glow_img = Image.new("RGBA", av_res.size, (*accent_color, 120))
        canvas.paste(glow_img, pos, mask=glow)
        
        canvas.paste(av_res, pos, mask=av_res)

    # 2. Render Hook Text (Curiosity Gap)
    lines = hook_text.split("\n")
    font_size = 95 if not is_shorts else 110
    font = _load_font(font_size, "black")
    
    # Yellow High-Impact Color
    txt_color = (255, 214, 0) # Premium Yellow
    
    total_h = sum(_text_size(l, font)[1] for l in lines) + 20 * (len(lines)-1)
    y = (height - total_h) // 2 if not is_shorts else height // 2
    x = 60
    
    for line in lines:
        lw, lh = _text_size(line, font)
        # Professional multi-pass drop shadow
        for offset in range(1, 6):
            draw.text((x+offset, y+offset), line, font=font, fill=(0,0,0, 100))
        
        draw.text((x, y), line, font=font, fill=txt_color)
        y += lh + 25

    # 3. Branding Accent
    draw.rectangle([0, height-15, width, height], fill=accent_color)
    
    return canvas

def generate_thumbnail(script_json):
    client = genai.Client(api_key=GEMINI_API_KEY)
    title = script_json.get("title", "AI Breakthrough")
    accent_hex = script_json.get("color_theme", {}).get("accent", "#FFD600").lstrip("#")
    accent_rgb = tuple(int(accent_hex[i:i+2], 16) for i in (0, 2, 4))
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_yt = os.path.join(OUTPUT_DIR, f"thumbnail_{date_str}.jpg")
    out_shorts = os.path.join(OUTPUT_DIR, f"thumbnail_shorts_{date_str}.jpg")

    # Pipeline
    hook = _generate_hook_text(title, client)
    bg = _generate_imagen_background(title, client)
    avatar = _process_avatar()

    # YT (16:9)
    print("🎬 Rendering YouTube Thumbnail...")
    yt = _render_premium_thumbnail(hook, bg, avatar, accent_rgb, THUMB_W, THUMB_H)
    yt.convert("RGB").save(out_yt, "JPEG", quality=95)

    # Shorts (9:16)
    print("🎬 Rendering Shorts Thumbnail...")
    bg_vert = bg.resize((SHORTS_W, SHORTS_H), Image.LANCZOS)
    shorts = _render_premium_thumbnail(hook, bg_vert, avatar, accent_rgb, SHORTS_W, SHORTS_H, is_shorts=True)
    shorts.convert("RGB").save(out_shorts, "JPEG", quality=95)

    print(f"✅ Premium Thumbnails Generated: {out_yt}")
    return out_yt

if __name__ == "__main__":
    # Test run
    test_json = {"title": "OpenAI Search is finally here...", "color_theme": {"accent": "#00E5FF"}}
    generate_thumbnail(test_json)
