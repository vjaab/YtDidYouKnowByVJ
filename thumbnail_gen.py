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


def wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        # Test width
        test_text = " ".join(current_line)
        bb = font.getbbox(test_text)
        w = bb[2] - bb[0]
        if w > max_width and len(current_line) > 1:
            current_line.pop()
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines

def generate_thumbnail(script_json):
    """
    Generates a 1280x720 YouTube thumbnail and a 1080x1920 Shorts thumbnail
    using specific branding requested by the user.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_path_yt = os.path.join(OUTPUT_DIR, f"thumbnail_{date_str}.jpg")
    output_path_shorts = os.path.join(OUTPUT_DIR, f"thumbnail_shorts_{date_str}.jpg")
    
    # Extract topic from script
    topic = script_json.get("title", "Tech News")
    
    # ── 1. BACKGROUND IMAGE ───────────────────────────────────────────────────
    bg_path = os.path.join(ASSETS_DIR, "gemini_img_without_logo.png")
    if os.path.exists(bg_path):
        base_img = Image.open(bg_path).convert("RGBA")
    else:
        print("Warning: gemini_img_without_logo.png not found, using solid fallback.")
        base_img = Image.new("RGBA", (1080, 1920), (30, 30, 40, 255))
        
    # Create the 1280x720 (16:9) canvas
    # We will crop/resize the background to fill the frame
    img_ratio = base_img.width / base_img.height
    target_ratio = 1280 / 720
    
    if img_ratio > target_ratio:
        # Image is wider than target
        new_w = int(720 * img_ratio)
        resized = base_img.resize((new_w, 720), Image.LANCZOS)
        # Crop center
        left = (new_w - 1280) // 2
        bg_yt = resized.crop((left, 0, left + 1280, 720))
    else:
        # Image is taller than target
        new_h = int(1280 / img_ratio)
        resized = base_img.resize((1280, new_h), Image.LANCZOS)
        # Crop center
        top = (new_h - 720) // 2
        bg_yt = resized.crop((0, top, 1280, top + 720))

    # Create the 1080x1920 (9:16) canvas for Shorts
    shorts_target_ratio = 1080 / 1920
    if img_ratio > shorts_target_ratio:
        new_w = int(1920 * img_ratio)
        resized = base_img.resize((new_w, 1920), Image.LANCZOS)
        left = (new_w - 1080) // 2
        bg_shorts = resized.crop((left, 0, left + 1080, 1920))
    else:
        new_h = int(1080 / img_ratio)
        resized = base_img.resize((1080, new_h), Image.LANCZOS)
        top = (new_h - 1920) // 2
        bg_shorts = resized.crop((0, top, 1080, top + 1920))

    # Helper function to render text on a given canvas
    def apply_branding(canvas_img, width, height):
        # ── 4. SUBTLE DARK GRADIENT ───────────────────────────────────────────
        # Bottom 40% gradient: transparent -> rgba(0,0,0,0.7)
        grad = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        grad_draw = ImageDraw.Draw(grad)
        start_y = int(height * 0.6)
        for y in range(start_y, height):
            alpha = int(255 * 0.7 * ((y - start_y) / (height - start_y)))
            grad_draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
            
        final_img = Image.alpha_composite(canvas_img, grad)
        draw = ImageDraw.Draw(final_img)
        
        # ── 2. & 3. FONT SETUP & SIZING ───────────────────────────────────────
        topic_text = topic.upper()
        font_path = os.path.join(ASSETS_DIR, "fonts", "NanumBarunGothic.ttf")
        if not os.path.exists(font_path):
            font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if not os.path.exists(font_path):
                font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
                
        # Auto-scale font size (between 80 and 100)
        target_size = 100
        max_width = int(width * 0.8)
        
        try:
            font = ImageFont.truetype(font_path, target_size)
        except:
            font = ImageFont.load_default()
            
        # Wrap text into lines
        lines = wrap_text(topic_text, font, max_width)
        
        # If too many lines, shrink font
        while len(lines) > 3 and target_size > 80:
            target_size -= 5
            try:
                font = ImageFont.truetype(font_path, target_size)
            except:
                break
            lines = wrap_text(topic_text, font, max_width)
            
        # Calculate text block height
        line_spacing = 1.3
        line_heights = []
        for line in lines:
            bb = font.getbbox(line)
            line_heights.append(bb[3] - bb[1])
            
        total_text_h = sum(line_heights) + int((len(lines) - 1) * target_size * (line_spacing - 1))
        
        # Position slightly upper half
        start_y = int((height - total_text_h) / 2) - int(height * 0.05)
        
        # ── 4. DRAW TEXT WITH SHADOW & STROKE ─────────────────────────────────
        current_y = start_y
        for i, line in enumerate(lines):
            bb = font.getbbox(line)
            line_w = bb[2] - bb[0]
            line_h = bb[3] - bb[1]
            x = (width - line_w) // 2
            
            # Drop shadow
            shadow_offset = 4
            draw.text(
                (x + shadow_offset, current_y + shadow_offset), 
                line, 
                font=font, 
                fill=(0, 0, 0, int(255 * 0.8)),
                stroke_width=0
            )
            
            # Main bold text with black stroke
            draw.text(
                (x, current_y), 
                line, 
                font=font, 
                fill=(255, 255, 255, 255),
                stroke_width=4,
                stroke_fill=(0, 0, 0, 255)
            )
            
            current_y += line_h + int(target_size * (line_spacing - 1))
            
        return final_img

    # ── 5. EXPORT SPECS ───────────────────────────────────────────────────────
    final_yt = apply_branding(bg_yt, 1280, 720).convert("RGB")
    final_yt.save(output_path_yt, "JPEG", quality=95)
    print(f"✅ Generated YouTube Thumbnail: {output_path_yt}")
    
    final_shorts = apply_branding(bg_shorts, 1080, 1920).convert("RGB")
    final_shorts.save(output_path_shorts, "JPEG", quality=95)
    print(f"✅ Generated Shorts Thumbnail: {output_path_shorts}")
    
    return output_path_yt
