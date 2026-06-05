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
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from datetime import datetime
from google import genai
from google.genai import types
from rembg import remove
from config import OUTPUT_DIR, ASSETS_DIR, GEMINI_API_KEY
import cv2

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
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
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
                aspect_ratio='16:9'
            )
        )
        img_bytes = response.generated_images[0].image.image_bytes
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print(f"⚠️ Imagen failed: {e}. Using dark fallback.")
        return Image.new("RGB", (THUMB_W, THUMB_H), (10, 10, 15))

# ── FIGMA TECH REF UTILITIES ──────────────────────────────────────────────────

def _render_tech_grid(canvas, grid_color=(128, 128, 128, 25), spacing=60, dash_len=5):
    """Renders a beautiful semi-transparent dashed technical grid overlay."""
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = canvas.size
    
    # Vertical lines (dashed)
    for x in range(0, w, spacing):
        for y in range(0, h, dash_len * 2):
            draw.line([(x, y), (x, y + dash_len)], fill=grid_color, width=1)
            
    # Horizontal lines (dashed)
    for y in range(0, h, spacing):
        for x in range(0, w, dash_len * 2):
            draw.line([(x, y), (x + dash_len, y)], fill=grid_color, width=1)
            
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

def _draw_tech_decorations(canvas, accent_color):
    """Draws subtle Figma HUD/UI style decorations (crosshairs, boundary brackets, micro labels)."""
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = canvas.size
    accent_alpha = (*accent_color, 75)
    white_alpha = (255, 255, 255, 55)
    
    # 1. Floating Crosshairs (+) in empty areas
    crosshairs = [
        (w // 4, h // 5),
        (w // 3, h // 2 + 100),
        (w // 2 - 100, h // 4 - 30),
        (w // 2 + 150, h // 2 + 180)
    ]
    cross_size = 8
    for cx, cy in crosshairs:
        draw.line([(cx - cross_size, cy), (cx + cross_size, cy)], fill=white_alpha, width=1)
        draw.line([(cx, cy - cross_size), (cx, cy + cross_size)], fill=white_alpha, width=1)
        
    # 2. Corner right-angle bounding brackets
    margin = 35
    bracket_len = 25
    # Top-Left Bracket
    draw.line([(margin, margin), (margin + bracket_len, margin)], fill=accent_alpha, width=2)
    draw.line([(margin, margin), (margin, margin + bracket_len)], fill=accent_alpha, width=2)
    # Top-Right Bracket
    draw.line([(w - margin, margin), (w - margin - bracket_len, margin)], fill=accent_alpha, width=2)
    draw.line([(w - margin, margin), (w - margin, margin + bracket_len)], fill=accent_alpha, width=2)
    # Bottom-Left Bracket
    draw.line([(margin, h - margin), (margin + bracket_len, h - margin)], fill=accent_alpha, width=2)
    draw.line([(margin, h - margin), (margin, h - margin - bracket_len)], fill=accent_alpha, width=2)
    # Bottom-Right Bracket
    draw.line([(w - margin, h - margin), (w - margin - bracket_len, h - margin)], fill=accent_alpha, width=2)
    draw.line([(w - margin, h - margin), (w - margin, h - margin - bracket_len)], fill=accent_alpha, width=2)
    
    # 3. Monospace dimensional micro-label in top-right
    try:
        f_mono = _load_font(18, "extrabold")
        label_text = f"[ 16:9_HD // {w}x{h} ]"
        draw.text((w - 280, 50), label_text, font=f_mono, fill=(255, 255, 255, 90))
    except Exception as e:
        print("⚠️ Monospace decoration label failed:", e)
        
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

def _get_bezier_points(p0, p1, p2, p3, steps=30):
    points = []
    for t in [i / steps for i in range(steps + 1)]:
        x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points

def _draw_curved_accent(canvas, accent_color):
    """Draws a premium glowing organic bezier curve near bottom-left corner with glowing pips."""
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = canvas.size
    
    # Define Bezier points near the bottom left (away from text and avatar)
    p0 = (60, h - 120)
    p1 = (120, h - 80)
    p2 = (200, h - 160)
    p3 = (260, h - 140)
    
    points = _get_bezier_points(p0, p1, p2, p3)
    
    # Draw glowing shadow for the line
    for gw in range(8, 2, -2):
        draw.line(points, fill=(*accent_color, int(80 / gw)), width=gw)
    # Draw main accent line (White)
    draw.line(points, fill=(255, 255, 255, 200), width=2)
    
    # Draw glowing circular nodes (pips) at key points
    for px, py in [p0, p3]:
        # Outer glow
        draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(*accent_color, 80))
        # Inner node (bright White)
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 255, 255, 240))
        
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

def _draw_multi_tier_glow(canvas, av_res, pos, accent_color):
    """Draws a beautiful, premium multi-tiered glowing aura radiating behind the avatar."""
    glow_canvas = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    # Calculate center of the avatar
    av_center_x = pos[0] + av_res.width // 2
    av_center_y = pos[1] + av_res.height // 2
    
    draw = ImageDraw.Draw(glow_canvas)
    
    # Tier 1: Massive soft backdrop aura (large radius, very low opacity)
    r1 = int(av_res.height * 0.55)
    draw.ellipse([av_center_x - r1, av_center_y - r1, av_center_x + r1, av_center_y + r1], 
                 fill=(*accent_color, 40))
                 
    # Tier 2: Medium backdrop aura (medium radius, medium opacity)
    r2 = int(av_res.height * 0.40)
    draw.ellipse([av_center_x - r2, av_center_y - r2, av_center_x + r2, av_center_y + r2], 
                 fill=(*accent_color, 75))
                 
    # Tier 3: Core intensive backing glow (smaller radius, high opacity)
    r3 = int(av_res.height * 0.22)
    draw.ellipse([av_center_x - r3, av_center_y - r3, av_center_x + r3, av_center_y + r3], 
                 fill=(*accent_color, 130))
                 
    # Apply heavy blur to the radial gradient circle overlay
    glow_canvas = glow_canvas.filter(ImageFilter.GaussianBlur(radius=45))
    
    # Tier 4: Detailed outline body glow matching the avatar's exact shape
    body_mask = av_res.split()[3].point(lambda x: 255 if x > 0 else 0)
    body_glow = body_mask.filter(ImageFilter.GaussianBlur(radius=25))
    body_glow_img = Image.new("RGBA", av_res.size, (*accent_color, 120))
    
    # Compose everything
    canvas_rgba = canvas.convert("RGBA")
    # Paste global radial backdrop glow
    canvas_rgba = Image.alpha_composite(canvas_rgba, glow_canvas)
    # Paste local avatar body glow onto composite
    temp_local = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    temp_local.paste(body_glow_img, pos, mask=body_glow)
    canvas_rgba = Image.alpha_composite(canvas_rgba, temp_local)
    
    # Paste the actual avatar image
    temp_avatar = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    temp_avatar.paste(av_res, pos, mask=av_res)
    canvas_rgba = Image.alpha_composite(canvas_rgba, temp_avatar)
    
    return canvas_rgba.convert("RGB")

# ── IMAGE PROCESSING ─────────────────────────────────────────────────────────

def _process_avatar_still(avatar_path=None, still_time=1.0):
    """
    Extracts an avatar frame (if video) or loads a static image,
    removes the background using rembg with local caching, and returns the cutout PIL Image.
    """
    if not avatar_path:
        avatar_path = AVATAR_PATH
        
    if not os.path.exists(avatar_path):
        print(f"⚠️ Avatar not found at: {avatar_path}. Falling back to default AVATAR_PATH.")
        avatar_path = AVATAR_PATH
        if not os.path.exists(avatar_path):
            print("⚠️ Default avatar not found at", AVATAR_PATH)
            return None

    # Setup cutout cache directory
    cache_dir = os.path.join(OUTPUT_DIR, ".avatar_cutout_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Compute a unique cache key based on path & timestamp
    path_hash = hashlib.md5(f"{os.path.abspath(avatar_path)}_{still_time}".encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, f"cutout_{path_hash}.png")

    if os.path.exists(cache_file):
        try:
            print(f"🎯 Loading cached avatar cutout: {cache_file}")
            return Image.open(cache_file).convert("RGBA")
        except Exception as e:
            print(f"⚠️ Failed to load cached cutout: {e}. Re-processing...")

    print(f"👤 Processing avatar still from: {avatar_path} (time={still_time}s)...")
    try:
        input_img = None
        ext = os.path.splitext(avatar_path)[1].lower()
        
        # If it's a video, extract frame at still_time using cv2
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            cap = cv2.VideoCapture(avatar_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_idx = int(still_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                # Fallback: try reading the first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()
            
            if success:
                # Convert BGR (cv2 default) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_img = Image.fromarray(frame_rgb)
            cap.release()
            
            if input_img is None:
                raise Exception("Failed to extract frame from video.")
        else:
            # It's a static image
            input_img = Image.open(avatar_path).convert("RGBA")
            
        print("🪄 Removing background via rembg...")
        output_img = remove(input_img)
        
        # Save to cache
        output_img.save(cache_file, "PNG")
        return output_img
    except Exception as e:
        print(f"⚠️ Avatar processing failed: {e}")
        # Final fallback, try loading as static image if possible
        try:
            return Image.open(avatar_path).convert("RGBA")
        except:
            return None

def _draw_logo_badges(canvas, script_json, accent_color, is_shorts=False):
    """
    Renders premium glassmorphic HUD logo badges on the canvas.
    """
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = canvas.size
    
    # Gather logos from script_json
    logo_list = []
    
    single_logo = script_json.get("logo_path")
    if single_logo:
        logo_list.append({"path": single_logo, "position": "top_left", "label": "AUTHORITY SYSTEM"})
        
    json_logos = script_json.get("logos")
    if json_logos:
        if isinstance(json_logos, list):
            for l in json_logos:
                if isinstance(l, dict) and "path" in l:
                    logo_list.append({
                        "path": l["path"],
                        "position": l.get("position", "top_left"),
                        "label": l.get("label", "")
                    })
                elif isinstance(l, str):
                    logo_list.append({"path": l, "position": "top_left", "label": ""})
                    
    # Default fallback: if no logos specified, let's auto-overlay assets/logo.png in top_left
    if not logo_list:
        default_logo = os.path.join(ASSETS_DIR, "logo.png")
        if os.path.exists(default_logo):
            logo_list.append({"path": default_logo, "position": "top_left", "label": "GEN NEWS"})

    for logo_spec in logo_list:
        path = logo_spec["path"]
        pos_name = logo_spec["position"]
        label = logo_spec["label"]
        
        if not os.path.exists(path):
            # Check if it's in assets/icons/ or assets/
            for cand in [os.path.join(ASSETS_DIR, "icons", path), os.path.join(ASSETS_DIR, path)]:
                if os.path.exists(cand):
                    path = cand
                    break
            else:
                print(f"⚠️ Logo file not found: {path}")
                continue
                
        try:
            logo_img = Image.open(path).convert("RGBA")
        except Exception as e:
            print(f"⚠️ Failed to load logo {path}: {e}")
            continue
            
        # Draw badge depending on position
        if is_shorts:
            logo_h = 50
            scale = logo_h / logo_img.height
            logo_w = int(logo_img.width * scale)
            logo_res = logo_img.resize((logo_w, logo_h), Image.LANCZOS)
            
            px, py = 50, 60
            draw.rounded_rectangle([px - 15, py - 10, px + logo_w + 15, py + logo_h + 10], radius=8, fill=(10, 10, 15, 180), outline=(*accent_color, 120), width=1)
            overlay.paste(logo_res, (px, py), mask=logo_res)
        else:
            if pos_name == "top_left":
                logo_h = 42
                scale = logo_h / logo_img.height
                logo_w = int(logo_img.width * scale)
                logo_res = logo_img.resize((logo_w, logo_h), Image.LANCZOS)
                
                px = 60
                py = 50
                
                font_label = _load_font(14, "extrabold")
                label_w = 0
                if label:
                    label_w, _ = _text_size(label, font_label)
                    label_w += 20
                    
                badge_w = logo_w + 30 + label_w
                badge_h = logo_h + 20
                
                box_coords = [px - 15, py - 10, px + badge_w - 15, py + badge_h - 10]
                draw.rounded_rectangle(box_coords, radius=10, fill=(10, 10, 15, 210), outline=(*accent_color, 140), width=2)
                
                overlay.paste(logo_res, (px, py), mask=logo_res)
                
                if label:
                    draw.text((px + logo_w + 12, py + 12), label, font=font_label, fill=(255, 255, 255, 230))
                    
            elif pos_name == "bottom_left":
                logo_h = 38
                scale = logo_h / logo_img.height
                logo_w = int(logo_img.width * scale)
                logo_res = logo_img.resize((logo_w, logo_h), Image.LANCZOS)
                
                px = 60
                py = h - 85
                
                box_coords = [px - 12, py - 8, px + logo_w + 12, py + logo_h + 8]
                draw.rounded_rectangle(box_coords, radius=8, fill=(10, 10, 15, 190), outline=(255, 255, 255, 60), width=1)
                overlay.paste(logo_res, (px, py), mask=logo_res)
                
            elif pos_name == "top_right":
                logo_h = 35
                scale = logo_h / logo_img.height
                logo_w = int(logo_img.width * scale)
                logo_res = logo_img.resize((logo_w, logo_h), Image.LANCZOS)
                
                px = w - logo_w - 300
                py = 42
                
                box_coords = [px - 12, py - 8, px + logo_w + 12, py + logo_h + 8]
                draw.rounded_rectangle(box_coords, radius=8, fill=(10, 10, 15, 190), outline=(*accent_color, 100), width=1)
                overlay.paste(logo_res, (px, py), mask=logo_res)

    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

# ── RENDERING ─────────────────────────────────────────────────────────────────

def _render_premium_thumbnail(hook_text, bg_img, avatar_img, accent_color, width, height, script_json=None, is_shorts=False):
    canvas = bg_img.resize((width, height), Image.LANCZOS)
    # Darken background slightly for text readability
    canvas = ImageEnhance.Brightness(canvas).enhance(0.7)
    
    # 1. Apply Figma Tech HUD decorations if it is a 16:9 thumbnail
    if not is_shorts:
        canvas = _render_tech_grid(canvas)
        canvas = _draw_tech_decorations(canvas, accent_color)
        canvas = _draw_curved_accent(canvas, accent_color)
        
    draw = ImageDraw.Draw(canvas)
    
    # 2. Overlay Avatar with Premium Aura Glow
    if avatar_img:
        # Scale avatar to fit ~85% of height for landscape, ~45% for portrait
        av_h = int(height * 0.85) if not is_shorts else int(height * 0.45)
        scale = av_h / avatar_img.height
        av_res = avatar_img.resize((int(avatar_img.width * scale), av_h), Image.LANCZOS)
        
        # Position: Right side for YT, Center/Bottom for Shorts
        if is_shorts:
            pos = (width - av_res.width, height - av_res.height)
        else:
            pos = (width - av_res.width - 20, height - av_res.height)
            
        canvas = _draw_multi_tier_glow(canvas, av_res, pos, accent_color)
        draw = ImageDraw.Draw(canvas) # Re-get draw for subsequent drawing

    # 3. Render Hook Text (Curiosity Gap) with Figma blocks & high-contrast colors
    lines = hook_text.split("\n")
    font_size = 90 if not is_shorts else 110
    font = _load_font(font_size, "black")
    
    # Calculate height considering badge padding
    total_h = sum(_text_size(l, font)[1] for l in lines) + 40 * (len(lines)-1)
    y = (height - total_h) // 2 if not is_shorts else height // 2
    x = 80 if not is_shorts else 60
    
    for idx, line in enumerate(lines):
        lw, lh = _text_size(line, font)
        
        # Multi-color strategy: First line clean white, emphasis lines are the neon accent color
        if idx == 0:
            txt_color = (255, 255, 255)
        else:
            txt_color = accent_color
            
        # Draw translucent tech badge backing box
        box_padding_x = 25
        box_padding_y = 12
        box_coords = [
            x - box_padding_x,
            y - box_padding_y,
            x + lw + box_padding_x,
            y + lh + box_padding_y
        ]
        
        block_overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        block_draw = ImageDraw.Draw(block_overlay)
        # Translucent dark charcoal block (Figma HUD block) with thin accent border
        block_draw.rounded_rectangle(box_coords, radius=12, fill=(10, 10, 15, 195), outline=(*accent_color, 100), width=2)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), block_overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        
        # Professional multi-pass drop shadow behind text
        for offset in range(1, 6):
            draw.text((x+offset, y+offset), line, font=font, fill=(0, 0, 0, 140))
        
        draw.text((x, y), line, font=font, fill=txt_color)
        y += lh + 45
        
    # 4. Branding Accent
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, height-15, width, height], fill=accent_color)
    
    # 5. Render Logos
    if script_json:
        canvas = _draw_logo_badges(canvas, script_json, accent_color, is_shorts=is_shorts)
        
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


def _render_compilation_thumbnail(bg_img, avatar_img, accent_color, width, height, script_json=None):
    """Specialized 16:9 thumbnail for 'Did You Know' 5-fact compilations."""
    canvas = bg_img.resize((width, height), Image.LANCZOS)
    canvas = ImageEnhance.Brightness(canvas).enhance(0.6) # Darker for compilation text
    
    # 1. Apply Figma Tech HUD decorations
    canvas = _render_tech_grid(canvas)
    canvas = _draw_tech_decorations(canvas, accent_color)
    canvas = _draw_curved_accent(canvas, accent_color)
    
    draw = ImageDraw.Draw(canvas)
    
    # 2. Overlay Avatar with Premium Aura Glow - Right Side
    if avatar_img:
        av_h = int(height * 0.90)
        scale = av_h / avatar_img.height
        av_res = avatar_img.resize((int(avatar_img.width * scale), av_h), Image.LANCZOS)
        
        pos = (width - av_res.width - 20, height - av_res.height)
            
        canvas = _draw_multi_tier_glow(canvas, av_res, pos, accent_color)
        draw = ImageDraw.Draw(canvas) # Re-get draw

    # 3. Render "DID YOU KNOW?" with translucent backing boxes & hybrid colors
    f_main = _load_font(130, "black")
    text_main = "DID YOU\nKNOW?"
    
    y = 120
    x = 80
    
    for idx, line in enumerate(text_main.split("\n")):
        lw, lh = _text_size(line, f_main)
        
        # Color: Line 0 White, rest are Premium Yellow
        txt_color = (255, 255, 255) if idx == 0 else (255, 214, 0)
        
        # Translucent tech badge backing box
        box_padding_x = 25
        box_padding_y = 12
        box_coords = [
            x - box_padding_x,
            y - box_padding_y,
            x + lw + box_padding_x,
            y + lh + box_padding_y
        ]
        
        block_overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        block_draw = ImageDraw.Draw(block_overlay)
        block_draw.rounded_rectangle(box_coords, radius=12, fill=(10, 10, 15, 195), outline=(*accent_color, 100), width=2)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), block_overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        
        for offset in range(1, 8):
            draw.text((x+offset, y+offset), line, font=f_main, fill=(0,0,0, 150))
        draw.text((x, y), line, font=f_main, fill=txt_color)
        y += lh + 40

    # 4. Render "N AI FACTS" badge
    f_sub = _load_font(80, "extrabold")
    num_facts = 10
    if script_json:
        num_facts = script_json.get("num_facts", len(script_json.get("fact_scripts", [])) or 10)
    sub_txt = f"{num_facts} AI FACTS"
    sub_w, sub_h = _text_size(sub_txt, f_sub)
    
    badge_x = 80
    badge_y = y + 45
    
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
    
    # 5. Render Logos
    if script_json:
        canvas = _draw_logo_badges(canvas, script_json, accent_color, is_shorts=False)
        
    return canvas

def generate_thumbnail(script_json):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Support varying title or custom hook directly
    custom_hook = script_json.get("custom_hook") or script_json.get("hook_text")
    title = script_json.get("title", "AI Breakthrough")
    
    accent_hex = script_json.get("color_theme", {}).get("accent", "#FFD600").lstrip("#")
    accent_rgb = tuple(int(accent_hex[i:i+2], 16) for i in (0, 2, 4))
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    custom_suffix = script_json.get("output_suffix", "")
    suffix_str = f"_{custom_suffix}" if custom_suffix else f"_{date_str}"
    
    out_yt = os.path.join(OUTPUT_DIR, f"thumbnail{suffix_str}.jpg")
    out_shorts = os.path.join(OUTPUT_DIR, f"thumbnail_shorts{suffix_str}.jpg")

    # Pipeline
    bg = _generate_imagen_background(title, client)
    
    # Dynamic avatar still selection/extraction with frame time
    avatar_still = script_json.get("avatar_still") or script_json.get("avatar_path")
    still_time = float(script_json.get("avatar_still_time", 1.0))
    avatar = _process_avatar_still(avatar_still, still_time)

    is_compilation = script_json.get("is_longform") and script_json.get("longform_format") == "did_you_know"

    if is_compilation:
        print("🎬 Rendering Long-Form Compilation Thumbnail...")
        yt = _render_compilation_thumbnail(bg, avatar, accent_rgb, THUMB_W, THUMB_H, script_json=script_json)
        yt.convert("RGB").save(out_yt, "JPEG", quality=95)
        print(f"✅ Premium Compilation Thumbnail Generated: {out_yt}")
        return out_yt
    else:
        # Determine Hook Text
        if custom_hook:
            print("📝 Using custom hook text from script_json...")
            hook = custom_hook.replace("\\n", "\n")
        else:
            hook = _generate_hook_text(title, client)
            
        # YT (16:9)
        print("🎬 Rendering YouTube Thumbnail...")
        yt = _render_premium_thumbnail(hook, bg, avatar, accent_rgb, THUMB_W, THUMB_H, script_json=script_json)
        yt.convert("RGB").save(out_yt, "JPEG", quality=95)

        # Shorts (9:16)
        print("🎬 Rendering Shorts Thumbnail...")
        bg_vert = bg.resize((SHORTS_W, SHORTS_H), Image.LANCZOS)
        shorts = _render_premium_thumbnail(hook, bg_vert, avatar, accent_rgb, SHORTS_W, SHORTS_H, script_json=script_json, is_shorts=True)
        shorts.convert("RGB").save(out_shorts, "JPEG", quality=95)

        print(f"✅ Premium Thumbnails Generated: {out_yt}")
        return out_yt

if __name__ == "__main__":
    # Test run
    test_json = {"title": "OpenAI Search is finally here...", "color_theme": {"accent": "#00E5FF"}}
    generate_thumbnail(test_json)
