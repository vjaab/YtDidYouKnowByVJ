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

def _draw_neon_arrow(draw, start, end, accent_color, width=12):
    """Draws a premium neon arrow pointing from start to end with glow."""
    # 1. Outer neon glow for the line
    for glow_w in range(width + 8, width, -2):
        alpha = int(70 * ((width + 8 - glow_w) / 8))
        draw.line([start, end], fill=(*accent_color, alpha), width=glow_w)
    # Main arrow line (Crimson Red)
    draw.line([start, end], fill=(255, 32, 32, 255), width=width)
    
    # 2. Calculate arrowhead points
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    
    arrow_len = 45
    angle_offset = math.pi / 6 # 30 degrees
    
    p1 = (end[0] - arrow_len * math.cos(angle - angle_offset),
          end[1] - arrow_len * math.sin(angle - angle_offset))
    p2 = (end[0] - arrow_len * math.cos(angle + angle_offset),
          end[1] - arrow_len * math.sin(angle + angle_offset))
          
    # Outer glow for arrowhead
    for glow_w in range(width + 6, width, -2):
        alpha = int(70 * ((width + 6 - glow_w) / 6))
        draw.line([p1, end], fill=(*accent_color, alpha), width=glow_w)
        draw.line([p2, end], fill=(*accent_color, alpha), width=glow_w)
        
    draw.line([p1, end], fill=(255, 32, 32, 255), width=width)
    draw.line([p2, end], fill=(255, 32, 32, 255), width=width)


def _render_compilation_thumbnail(bg_img, avatar_img, accent_color, width, height):
    """Specialized 16:9 thumbnail for 'Did You Know' 5-fact compilations."""
    canvas = bg_img.resize((width, height), Image.LANCZOS)
    canvas = ImageEnhance.Brightness(canvas).enhance(0.6) # Darker for compilation text
    draw = ImageDraw.Draw(canvas)
    
    # 1. Overlay Avatar (Authority) - Right Side
    if avatar_img:
        av_h = int(height * 0.90)
        scale = av_h / avatar_img.height
        av_res = avatar_img.resize((int(avatar_img.width * scale), av_h), Image.LANCZOS)
        
        pos = (width - av_res.width - 20, height - av_res.height)
            
        glow = av_res.split()[3].point(lambda x: 255 if x > 0 else 0)
        glow = glow.filter(ImageFilter.GaussianBlur(radius=25))
        glow_img = Image.new("RGBA", av_res.size, (*accent_color, 140))
        canvas.paste(glow_img, pos, mask=glow)
        
        canvas.paste(av_res, pos, mask=av_res)

    # 2. Render "DID YOU KNOW?" (Massive, Yellow)
    f_main = _load_font(130, "black")
    txt_color = (255, 214, 0)
    text_main = "DID YOU\nKNOW?"
    
    y = 120
    x = 80
    
    for line in text_main.split("\n"):
        lw, lh = _text_size(line, f_main)
        for offset in range(1, 8):
            draw.text((x+offset, y+offset), line, font=f_main, fill=(0,0,0, 150))
        draw.text((x, y), line, font=f_main, fill=txt_color)
        y += lh + 20

    # 3. Render "5 AI FACTS" badge
    f_sub = _load_font(80, "extrabold")
    sub_txt = "5 AI FACTS"
    sub_w, sub_h = _text_size(sub_txt, f_sub)
    
    badge_x = 80
    badge_y = y + 40
    
    # Red/Accent glowing pill
    draw.rounded_rectangle([badge_x - 20, badge_y - 10, badge_x + sub_w + 20, badge_y + sub_h + 30], 
                           radius=25, fill=(220, 20, 60, 240)) # Crimson pill
                           
    for offset in range(1, 4):
        draw.text((badge_x+offset, badge_y+offset), sub_txt, font=f_sub, fill=(0,0,0, 100))
    draw.text((badge_x, badge_y), sub_txt, font=f_sub, fill=(255, 255, 255, 255))

    # 5. Draw CTR-boosting neon-red arrow pointing from center-left toward avatar
    try:
        arrow_start = (width // 2 - 140, height // 2 + 120)
        arrow_end = (width - 340, height // 2 - 10)
        _draw_neon_arrow(draw, arrow_start, arrow_end, accent_color, width=14)
    except Exception as e:
        print(f"⚠️ Arrow rendering failed: {e}")

    # 6. Draw shocked face emoji overlay for high clickability
    try:
        # Load emoji using same font helper
        f_emoji = _load_font(120, "black")
        emoji_x = width // 2 - 190
        emoji_y = height // 2 - 60
        # Multi-pass drop shadow behind emoji
        for offset in range(1, 6):
            draw.text((emoji_x + offset, emoji_y + offset), "😱", font=f_emoji, fill=(0, 0, 0, 180))
        draw.text((emoji_x, emoji_y), "😱", font=f_emoji, fill=(255, 255, 255, 255))
    except Exception as e:
        print(f"⚠️ Emoji rendering failed: {e}")

    # 4. Branding Accent
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
    bg = _generate_imagen_background(title, client)
    avatar = _process_avatar()

    is_compilation = script_json.get("is_longform") and script_json.get("longform_format") == "did_you_know"

    if is_compilation:
        print("🎬 Rendering Long-Form Compilation Thumbnail...")
        yt = _render_compilation_thumbnail(bg, avatar, accent_rgb, THUMB_W, THUMB_H)
        yt.convert("RGB").save(out_yt, "JPEG", quality=95)
        print(f"✅ Premium Compilation Thumbnail Generated: {out_yt}")
        return out_yt
    else:
        hook = _generate_hook_text(title, client)
        
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
