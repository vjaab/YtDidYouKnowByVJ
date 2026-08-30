"""
thumbnail_gen.py — Premium Imagen-3 + Authority Avatar Thumbnail Generator.

Design Philosophy (2026 High-Authority Spec):
  - Generative Backdrop: Unique Imagen-3 tech art for every topic.
  - Personal Authority: Seamlessly integrated avatar (cutout) with emotion overlay.
  - Premium Typography: Montserrat Black, high contrast yellow/white.
  - Curiosity Gap: Professionally written hooks by Gemini.
  - A/B Testing: 3 variants per video with performance tracking.
  - Platform-Native: Rule of thirds, text <3 words, high contrast.
"""

import os
import io
import math
import random
import textwrap
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from datetime import datetime
from google import genai
from google.genai import types
from rembg import remove
from config import OUTPUT_DIR, ASSETS_DIR, GEMINI_API_KEY
import cv2
import requests
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict

THUMB_W, THUMB_H = 1280, 720
SHORTS_W, SHORTS_H = 1080, 1920

# ── A/B TEST CONFIG ──────────────────────────────────────────────────────────
THUMBNAIL_VARIANTS = 3
MAX_TEXT_WORDS = 3
RULE_OF_THIRDS_GRID = True
HIGH_CONTRAST_RATIO = 4.5  # WCAG AA

# ── FACE DETECTION & EMOTION CONFIG ──────────────────────────────────────────
FACE_DETECTION_CONFIDENCE = 0.7
EMOTION_OVERLAYS = {
    "shocked": "😱",
    "curious": "🤔", 
    "excited": "🤯",
    "warning": "⚠️",
    "mind_blown": "💥"
}
EMOTION_WEIGHTS = {
    "security": "warning",
    "privacy": "warning",
    "breaking": "shocked",
    "secret": "shocked",
    "revealed": "mind_blown",
    "ai": "excited",
    "launch": "excited",
    "new": "curious",
    "how": "curious",
    "why": "curious"
}

@dataclass
class ThumbnailVariant:
    """Metadata for A/B testing thumbnail variants."""
    variant_id: str
    path: str
    style: str  # "authority", "curiosity", "urgency"
    hook_text: str
    emotion: str
    text_word_count: int
    contrast_score: float
    rule_of_thirds_score: float
    face_detected: bool
    created_at: str

@dataclass
class ABTestResult:
    """Result of A/B test for thumbnail selection."""
    winner_variant_id: str
    ctr_data: Dict[str, float]
    test_duration_hours: int
    confidence: float

# ── ASSETS ───────────────────────────────────────────────────────────────────
AVATAR_PATH = os.path.join(ASSETS_DIR, "gemini_img_without_logo.png")
FONT_BLACK = os.path.join(ASSETS_DIR, "fonts", "Montserrat-Black.ttf")
FONT_EXTRABOLD = os.path.join(ASSETS_DIR, "fonts", "Montserrat-ExtraBold.ttf")

FALLBACKS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf"]

# Load OpenCV face detector (Haar cascade - lightweight, no extra downloads)
_FACE_CASCADE = None
def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE

_fcache = {}

def _load_font(size, weight="black"):
    key = (size, weight)
    if key not in _fcache:
        if weight == "extrabold":
            candidates = [FONT_EXTRABOLD, FONT_BLACK] + FALLBACKS
        else:
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

# ── FACE DETECTION & EMOTION OVERLAY ──────────────────────────────────────────

def _detect_face_region(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face in image using OpenCV Haar cascade.
    Returns (x, y, w, h) in PIL coordinates or None.
    """
    try:
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        cascade = _get_face_cascade()
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return (int(x), int(y), int(w), int(h))
    except Exception as e:
        print(f"⚠️ Face detection failed: {e}")
    return None


def _select_emotion_for_content(title: str, hook_text: str) -> str:
    """Select appropriate emotion emoji based on content keywords."""
    content = (title + " " + hook_text).lower()
    for keyword, emotion in EMOTION_WEIGHTS.items():
        if keyword in content:
            return EMOTION_OVERLAYS[emotion]
    return EMOTION_OVERLAYS["excited"]


def _apply_emotion_overlay(canvas: Image.Image, face_region: Optional[Tuple], emotion: str, 
                            accent_color: Tuple[int, int, int], is_shorts: bool) -> Image.Image:
    """
    Apply emotion emoji overlay near detected face region following rule of thirds.
    """
    if not face_region:
        return canvas
    
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    fx, fy, fw, fh = face_region
    
    # Rule of thirds: place emotion at intersection points
    third_w, third_h = w // 3, h // 3
    intersections = [
        (third_w, third_h),           # Top-left
        (2 * third_w, third_h),       # Top-right
        (third_w, 2 * third_h),       # Bottom-left
        (2 * third_w, 2 * third_h),   # Bottom-right
    ]
    
    # Pick intersection closest to face but on opposite side for balance
    face_center = (fx + fw // 2, fy + fh // 2)
    best_pos = min(intersections, key=lambda p: abs(p[0] - face_center[0]) + abs(p[1] - face_center[1]))
    
    # Offset slightly from intersection to avoid covering face
    offset_x = 60 if best_pos[0] < w // 2 else -60
    offset_y = -40 if best_pos[1] < h // 2 else 40
    emoji_pos = (best_pos[0] + offset_x, best_pos[1] + offset_y)
    
    # Clamp to canvas bounds
    emoji_pos = (max(50, min(w - 150, emoji_pos[0])), max(50, min(h - 150, emoji_pos[1])))
    
    # Draw emotion emoji with glow
    font_size = 80 if not is_shorts else 100
    font = _load_font(font_size, "black")
    
    # Glow effect
    for offset in range(1, 8):
        draw.text((emoji_pos[0] + offset, emoji_pos[1] + offset), emotion, font=font, fill=(0, 0, 0, 180))
    draw.text(emoji_pos, emotion, font=font, fill=(255, 255, 255, 255))
    
    # Add accent ring around emoji
    ring_radius = font_size // 2 + 10
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ring_draw = ImageDraw.Draw(overlay)
    ring_draw.ellipse([
        emoji_pos[0] - ring_radius, emoji_pos[1] - ring_radius,
        emoji_pos[0] + ring_radius, emoji_pos[1] + ring_radius
    ], outline=(*accent_color, 180), width=4)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    
    return canvas


def _enforce_max_words(text: str, max_words: int = MAX_TEXT_WORDS) -> str:
    """Enforce maximum word count for thumbnail text."""
    words = text.replace("\n", " ").split()
    if len(words) <= max_words:
        return text
    # Keep first max_words words
    return " ".join(words[:max_words])


def _calculate_contrast_ratio(fg_color: Tuple[int, int, int], bg_color: Tuple[int, int, int]) -> float:
    """Calculate WCAG contrast ratio between foreground and background."""
    def luminance(r, g, b):
        def channel(c):
            c = c / 255.0
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    
    l1 = luminance(*fg_color)
    l2 = luminance(*bg_color)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def _calculate_rule_of_thirds_score(canvas: Image.Image, text_positions: List[Tuple], 
                                     face_region: Optional[Tuple]) -> float:
    """Score how well elements align with rule of thirds grid."""
    w, h = canvas.size
    third_w, third_h = w / 3, h / 3
    grid_lines = [third_w, 2 * third_w, third_h, 2 * third_h]
    
    score = 0.0
    elements = text_positions + ([ (face_region[0] + face_region[2]//2, face_region[1] + face_region[3]//2) ] if face_region else [])
    
    for ex, ey in elements:
        # Distance to nearest vertical grid line
        dx = min(abs(ex - gl) for gl in grid_lines[:2])
        # Distance to nearest horizontal grid line
        dy = min(abs(ey - gl) for gl in grid_lines[2:])
        # Score: closer to grid = higher score (normalized)
        score += 1.0 - min(1.0, (dx + dy) / (w + h) * 6)
    
    return score / len(elements) if elements else 0.0


# ── AI AGENTS ─────────────────────────────────────────────────────────────────

def _generate_hook_text(title, client, is_shorts=False, variant_style="curiosity"):
    """Generates a high-click-through curiosity gap hook with variant styles."""
    variant_prompts = {
        "authority": "Authoritative, expert tone. Trust signal. Example: 'EXPERT VERDICT'",
        "curiosity": "Curiosity gap, information void. Example: 'WHAT THEY HID'",
        "urgency": "Urgent, time-sensitive, FOMO. Example: 'ACT NOW BEFORE'"
    }
    style_guidance = variant_prompts.get(variant_style, variant_prompts["curiosity"])
    
    if is_shorts:
        prompt = f"""You are a viral YouTube Short thumbnail designer.
Generate an extremely short, high-impact curiosity value proposition or alert message in ALL CAPS for a mobile vertical thumbnail about: "{title}".
STYLE: {style_guidance}
RULES:
1. Max {MAX_TEXT_WORDS} words TOTAL, ALL CAPS.
2. Extremely punchy, viral and dramatic.
3. Use \\n for line breaks between words to stack vertically.
Return ONLY the raw words. No quotes, no preamble."""
    else:
        prompt = f"""You are a viral YouTube thumbnail copywriter. 
Generate a SHORT, punchy curiosity gap hook for this topic: "{title}"
STYLE: {style_guidance}
RULES: Max {MAX_TEXT_WORDS} words TOTAL, emotional, curiosity-gap style. 
Use \\n for line breaks (max 2 lines). 
Example: "SECRET\\nREVEALED"
Return ONLY the text."""
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        hook = response.text.strip().replace("\\n", "\n")
        # Enforce max words
        hook = _enforce_max_words(hook, MAX_TEXT_WORDS)
        return "\n".join(hook.split("\n")[:2 if not is_shorts else 3])
    except:
        if is_shorts:
            return "WARNING!"
        return title[:20] + "..."

def _generate_imagen_background(title, client):
    """Generates a thematic tech background using Imagen-3."""
    print(f"🎨 Generating Premium Imagen background for: {title}")
    prompt = (
        f"A cinematic high-contrast YouTube thumbnail background about: {title}. "
        "Must feature a shocked expressive tech creator pointing, alongside a stylized generic app icon or tech symbol (such as a lock, gear, alert warning emblem, or app glyph). "
        "Strictly avoid any copyrighted brand logos, trademarks, or company marks (like Apple, GitHub, OpenAI, WhatsApp). "
        "Dark cyberpunk noir mood, dramatic studio neon lighting, high contrast, 8k resolution, clean composition, no text."
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
        print(f"⚠️ Imagen failed: {e}. Trying HuggingFace/Pollinations fallback...")
        
        # Try HuggingFace FLUX.1 first
        try:
            from config import HF_TOKEN
            if HF_TOKEN:
                resp = requests.post(
                    "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
                    headers={"Authorization": f"Bearer {HF_TOKEN}"},
                    json={"inputs": prompt, "parameters": {"width": 1280, "height": 720}},
                    timeout=60
                )
                if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
                    print("✅ HuggingFace background generated successfully!")
                    return Image.open(io.BytesIO(resp.content)).convert("RGB")
                else:
                    print(f"⚠️ HuggingFace returned status: {resp.status_code}")
        except Exception as hfe:
            print(f"⚠️ HuggingFace fallback failed: {hfe}")
        
        # Then try Cloudflare Workers AI
        try:
            from config import CF_ACCOUNT_ID, CF_API_TOKEN
            if CF_ACCOUNT_ID and CF_API_TOKEN:
                resp = requests.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/black-forest-labs/flux-1-schnell",
                    headers={
                        "Authorization": f"Bearer {CF_API_TOKEN}",
                        "Content-Type": "application/json"
                    },
                    json={"prompt": prompt},
                    timeout=60
                )
                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "")
                    if content_type.startswith("image"):
                        print("✅ Cloudflare background generated successfully!")
                        return Image.open(io.BytesIO(resp.content)).convert("RGB")
                    else:
                        try:
                            import base64
                            data = resp.json()
                            if data.get("success") and data.get("result", {}).get("image"):
                                img_bytes = base64.b64decode(data["result"]["image"])
                                print("✅ Cloudflare background generated successfully (base64)!")
                                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        except Exception:
                            pass
                else:
                    print(f"⚠️ Cloudflare returned status: {resp.status_code}")
        except Exception as cfe:
            print(f"⚠️ Cloudflare fallback failed: {cfe}")
        
        # Then try Pollinations
        try:
            import urllib.parse
            encoded_prompt = urllib.parse.quote(prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1280&height=720&nologo=true&private=true"
            resp = requests.get(url, timeout=45)
            if resp.status_code == 200:
                print("✅ Pollinations background generated successfully!")
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            elif resp.status_code == 429:
                print(f"⚠️ Pollinations rate limited (429). Skipping.")
        except Exception as pe:
            print(f"⚠️ Pollinations fallback failed: {pe}")
            
        print("Using dark fallback.")
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

def _render_premium_thumbnail(hook_text, bg_img, avatar_img, accent_color, width, height, script_json=None, is_shorts=False, variant_style="curiosity"):
    """
    Renders premium thumbnail with face detection, emotion overlay, and A/B test metadata.
    Returns: (canvas, metadata_dict)
    """
    canvas = bg_img.resize((width, height), Image.LANCZOS)
    canvas = ImageEnhance.Brightness(canvas).enhance(0.7)
    
    # 1. Apply Figma Tech HUD decorations if it is a 16:9 thumbnail
    if not is_shorts:
        canvas = _render_tech_grid(canvas)
        canvas = _draw_tech_decorations(canvas, accent_color)
        canvas = _draw_curved_accent(canvas, accent_color)
        
    draw = ImageDraw.Draw(canvas)
    
    # 2. Overlay Avatar with Premium Aura Glow
    face_region = None
    avatar_pos = None
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
            
        avatar_pos = pos
        canvas = _draw_multi_tier_glow(canvas, av_res, pos, accent_color)
        draw = ImageDraw.Draw(canvas) # Re-get draw for subsequent drawing
        
        # Detect face in the placed avatar region
        avatar_crop = canvas.crop((pos[0], pos[1], pos[0] + av_res.width, pos[1] + av_res.height))
        face_in_avatar = _detect_face_region(avatar_crop)
        if face_in_avatar:
            # Convert to canvas coordinates
            face_region = (pos[0] + face_in_avatar[0], pos[1] + face_in_avatar[1], 
                          face_in_avatar[2], face_in_avatar[3])
    
    # If no face in avatar, try detecting in full canvas (background might have face)
    if not face_region:
        face_region = _detect_face_region(canvas)
    
    # 3. Apply Emotion Overlay based on content
    emotion = _select_emotion_for_content(
        script_json.get("title", "") if script_json else "", 
        hook_text
    )
    if face_region:
        canvas = _apply_emotion_overlay(canvas, face_region, emotion, accent_color, is_shorts)
        draw = ImageDraw.Draw(canvas)
    
    # 4. Render Hook Text (Curiosity Gap) with Figma blocks & high-contrast colors
    lines = hook_text.split("\n")
    font_size = 90 if not is_shorts else 125
    font = _load_font(font_size, "extrabold" if is_shorts else "black")
    
    # Calculate height considering badge padding
    total_h = sum(_text_size(l, font)[1] for l in lines) + 40 * (len(lines)-1)
    y = (height - total_h) // 2 if not is_shorts else height // 2 - (total_h // 2)
    x = 80 if not is_shorts else 60
    
    text_positions = []
    for idx, line in enumerate(lines):
        lw, lh = _text_size(line, font)
        cur_x = x if not is_shorts else (width - lw) // 2
        
        # Multi-color strategy: First line clean white, emphasis lines are the neon accent color
        if idx == 0:
            txt_color = (255, 255, 255)
        else:
            txt_color = accent_color
            
        # Draw translucent tech badge backing box
        box_padding_x = 25
        box_padding_y = 12
        box_coords = [
            cur_x - box_padding_x,
            y - box_padding_y,
            cur_x + lw + box_padding_x,
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
            draw.text((cur_x+offset, y+offset), line, font=font, fill=(0, 0, 0, 140))
        
        draw.text((cur_x, y), line, font=font, fill=txt_color)
        
        # Track text center position for rule of thirds scoring
        text_positions.append((cur_x + lw // 2, y + lh // 2))
        y += lh + 45
        
    # 5. Branding Accent
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, height-15, width, height], fill=accent_color)
    
    # 6. Render Logos
    if script_json:
        canvas = _draw_logo_badges(canvas, script_json, accent_color, is_shorts=is_shorts)
    
    # 7. Calculate Quality Metrics
    # Sample background color behind text for contrast check
    bg_sample = canvas.crop((text_positions[0][0]-10, text_positions[0][1]-10, 
                            text_positions[0][0]+10, text_positions[0][1]+10)) if text_positions else None
    avg_bg = tuple(np.mean(np.array(bg_sample), axis=(0,1)).astype(int)) if bg_sample else (20, 20, 30)
    
    contrast_score = _calculate_contrast_ratio((255, 255, 255), avg_bg)
    rule_of_thirds_score = _calculate_rule_of_thirds_score(canvas, text_positions, face_region)
    text_word_count = len(hook_text.replace("\n", " ").split())
    
    metadata = {
        "face_detected": face_region is not None,
        "face_region": face_region,
        "emotion": emotion,
        "contrast_score": round(contrast_score, 2),
        "rule_of_thirds_score": round(rule_of_thirds_score, 2),
        "text_word_count": text_word_count,
        "variant_style": variant_style,
        "avatar_position": avatar_pos
    }
    
    return canvas, metadata

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

def _save_variant_metadata(variants: List[ThumbnailVariant], base_path: str):
    """Save A/B test metadata for thumbnail variants."""
    metadata = {
        "test_id": hashlib.md5(f"{base_path}{datetime.now().isoformat()}".encode()).hexdigest()[:12],
        "created_at": datetime.now().isoformat(),
        "variants": [asdict(v) for v in variants],
        "status": "pending_selection",
        "selection_criteria": {
            "min_contrast": HIGH_CONTRAST_RATIO,
            "min_rule_of_thirds": 0.6,
            "max_words": MAX_TEXT_WORDS,
            "face_required": True
        }
    }
    meta_path = base_path.replace(".jpg", "_abtest.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📊 A/B Test Metadata saved: {meta_path}")
    return meta_path


def _select_best_variant(variants: List[ThumbnailVariant]) -> ThumbnailVariant:
    """Select best variant based on quality scores."""
    scored = []
    for v in variants:
        score = 0
        # Contrast (WCAG AA compliant = 4.5)
        if v.contrast_score >= HIGH_CONTRAST_RATIO:
            score += 30
        else:
            score += max(0, v.contrast_score / HIGH_CONTRAST_RATIO * 30)
        
        # Rule of thirds
        score += v.rule_of_thirds_score * 25
        
        # Face detection bonus (+30% CTR proven)
        if v.face_detected:
            score += 30
        
        # Word count compliance
        if v.text_word_count <= MAX_TEXT_WORDS:
            score += 15
        else:
            score += max(0, (MAX_TEXT_WORDS / v.text_word_count) * 15)
        
        scored.append((score, v))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    print(f"🏆 Variant scores: {[(v.variant_id, round(s, 1)) for s, v in scored]}")
    return scored[0][1]


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
        # Define variant styles for A/B testing
        variant_styles = ["authority", "curiosity", "urgency"]
        
        # Generate hooks for each variant style
        if custom_hook:
            print("📝 Using custom hook text from script_json...")
            base_hook = custom_hook.replace("\\n", "\n")
            hooks_yt = {style: base_hook for style in variant_styles}
            hooks_shorts = {style: base_hook.upper() for style in variant_styles}
        else:
            hooks_yt = {style: _generate_hook_text(title, client, is_shorts=False, variant_style=style) for style in variant_styles}
            hooks_shorts = {style: _generate_hook_text(title, client, is_shorts=True, variant_style=style) for style in variant_styles}
        
        # Generate YouTube (16:9) variants
        print(f"🎬 Generating {THUMBNAIL_VARIANTS} YouTube Thumbnail Variants...")
        yt_variants = []
        for i, style in enumerate(variant_styles):
            variant_id = f"yt_{style}_{suffix_str}"
            canvas, meta = _render_premium_thumbnail(
                hooks_yt[style], bg, avatar, accent_rgb, THUMB_W, THUMB_H, 
                script_json=script_json, variant_style=style
            )
            variant_path = os.path.join(OUTPUT_DIR, f"thumbnail_{style}{suffix_str}.jpg")
            canvas.convert("RGB").save(variant_path, "JPEG", quality=95)
            
            yt_variants.append(ThumbnailVariant(
                variant_id=variant_id,
                path=variant_path,
                style=style,
                hook_text=hooks_yt[style].replace("\n", " | "),
                emotion=meta["emotion"],
                text_word_count=meta["text_word_count"],
                contrast_score=meta["contrast_score"],
                rule_of_thirds_score=meta["rule_of_thirds_score"],
                face_detected=meta["face_detected"],
                created_at=datetime.now().isoformat()
            ))
            print(f"   ✅ Variant {i+1}/{THUMBNAIL_VARIANTS} ({style}): {variant_path}")
        
        # Select best variant for YouTube
        best_yt = _select_best_variant(yt_variants)
        # Copy best to main output path
        import shutil
        shutil.copy2(best_yt.path, out_yt)
        print(f"🏆 Best YouTube variant: {best_yt.style} (contrast={best_yt.contrast_score}, rot={best_yt.rule_of_thirds_score}, face={best_yt.face_detected})")
        
        # Save A/B test metadata
        _save_variant_metadata(yt_variants, out_yt)
        
        # Generate Shorts (9:16) variants
        print(f"🎬 Generating {THUMBNAIL_VARIANTS} Shorts Thumbnail Variants...")
        bg_vert = bg.resize((SHORTS_W, SHORTS_H), Image.LANCZOS)
        shorts_variants = []
        for i, style in enumerate(variant_styles):
            variant_id = f"shorts_{style}_{suffix_str}"
            canvas, meta = _render_premium_thumbnail(
                hooks_shorts[style], bg_vert, avatar, accent_rgb, SHORTS_W, SHORTS_H, 
                script_json=script_json, is_shorts=True, variant_style=style
            )
            variant_path = os.path.join(OUTPUT_DIR, f"thumbnail_shorts_{style}{suffix_str}.jpg")
            canvas.convert("RGB").save(variant_path, "JPEG", quality=95)
            
            shorts_variants.append(ThumbnailVariant(
                variant_id=variant_id,
                path=variant_path,
                style=style,
                hook_text=hooks_shorts[style].replace("\n", " | "),
                emotion=meta["emotion"],
                text_word_count=meta["text_word_count"],
                contrast_score=meta["contrast_score"],
                rule_of_thirds_score=meta["rule_of_thirds_score"],
                face_detected=meta["face_detected"],
                created_at=datetime.now().isoformat()
            ))
            print(f"   ✅ Variant {i+1}/{THUMBNAIL_VARIANTS} ({style}): {variant_path}")
        
        # Select best variant for Shorts
        best_shorts = _select_best_variant(shorts_variants)
        shutil.copy2(best_shorts.path, out_shorts)
        print(f"🏆 Best Shorts variant: {best_shorts.style} (contrast={best_shorts.contrast_score}, rot={best_shorts.rule_of_thirds_score}, face={best_shorts.face_detected})")
        
        # Save A/B test metadata for shorts
        _save_variant_metadata(shorts_variants, out_shorts)
        
        print(f"✅ Premium Thumbnails Generated: {out_yt}")
        return out_yt

# ══════════════════════════════════════════════════════════════════════════════
# AI IMAGE PROMPT GENERATOR (Midjourney / DALL-E)
# Generates a single-line text-to-image prompt for thumbnail creation
# ══════════════════════════════════════════════════════════════════════════════

def generate_thumbnail_prompt(script_json: dict) -> str:
    """
    Generates a Midjourney/DALL-E compatible prompt for long-form video thumbnails.
    
    Rules:
    - 16:9 aspect ratio, rule of thirds, high contrast
    - Max 3-4 words bold text (Yellow/White on dark)
    - Left: "Old/Broken" visual | Right: "New/Upgraded" visual
    - Dark neon blue/black bg with Cyan/Yellow/Neon Red accents
    - Sleek 3D render, clean typography, minimal clutter
    
    Args:
        script_json: The video script data containing title, summary, topics, etc.
        
    Returns:
        Single-line image generation prompt string
    """
    title = script_json.get("title", "AI Breakthrough")
    summary = script_json.get("description", script_json.get("script", ""))[:500]
    topics = script_json.get("longform_topics", [])
    subcat = script_json.get("sub_category", "AI & Tech")
    
    # Extract key entities for visual metaphor
    companies = [c.get("name", "") for c in script_json.get("companies_mentioned", [])]
    tools = [t.get("name", "") for t in script_json.get("tools_mentioned", [])]
    entities = companies + tools
    
    # Determine the "Old vs New" visual metaphor based on content
    old_metaphors = {
        "security": "cracked padlock, shattered firewall, data leaking",
        "privacy": "open window, exposed documents, surveillance camera",
        "legacy": "old server rack, tangled cables, dusty hardware, floppy disk",
        "slow": "hourglass, loading spinner, snail, tortoise",
        "broken": "glitching screen, error 404, crashed system, blue screen",
        "outdated": "typewriter, fax machine, CRT monitor, punch cards",
        "vulnerable": "open vault, broken shield, warning signs, red alerts",
        "inefficient": "paperwork pile, manual process, clipboard, bureaucracy",
        "centralized": "single point of failure, monolith, bottleneck",
        "expensive": "burning money, gold bars leaking, expensive contract",
    }
    
    new_metaphors = {
        "security": "quantum encryption shield, biometric fortress, zero-trust architecture",
        "privacy": "encrypted vault, anonymous mask, zero-knowledge proof, local-first",
        "modern": "sleek serverless cloud, clean fiber optics, edge computing nodes",
        "fast": "lightning bolt, warp speed tunnel, instant sync, real-time stream",
        "fixed": "green checkmark, healed system, seamless flow, auto-recovery",
        "ai-powered": "neural network brain, glowing synapses, AI assistant hologram",
        "efficient": "automated pipeline, one-click deploy, streamlined workflow",
        "decentralized": "distributed mesh, blockchain nodes, peer-to-peer network",
        "cost-effective": "rocket ship growth, compound interest, efficient scaling",
        "breakthrough": "lightbulb moment, eureka spark, paradigm shift portal",
    }
    
    # Analyze content to pick metaphors
    content_lower = (title + " " + summary).lower()
    
    # Pick old metaphor
    old_visual = "legacy monolith server, tangled cables, warning lights"
    for key, val in old_metaphors.items():
        if key in content_lower:
            old_visual = val
            break
    
    # Pick new metaphor
    new_visual = "sleek AI-powered cloud, glowing neural pathways"
    for key, val in new_metaphors.items():
        if key in content_lower:
            new_visual = val
            break
    
    # If specific entities mentioned, incorporate them
    entity_visual = ""
    if entities:
        top_entity = entities[0]
        entity_visual = f", subtle {top_entity} logo/icon integration"
    
    # Determine accent color based on subcategory
    accent_map = {
        "security": "Neon Red",
        "privacy": "Electric Cyan", 
        "coding": "Matrix Green",
        "finance": "Gold Yellow",
        "ai": "Neon Purple",
        "tools": "Electric Blue",
        "gadgets": "Vibrant Orange",
    }
    accent = "Electric Cyan"
    for key, val in accent_map.items():
        if key in subcat.lower():
            accent = val
            break
    
    # Generate the hook text (3-4 words max)
    hook_words = _extract_hook_words(title, summary)
    
    # Build the prompt
    prompt_parts = [
        "YouTube thumbnail 16:9, rule of thirds composition",
        "split diagonal: LEFT side old/broken, RIGHT side new/upgraded",
        f"LEFT: {old_visual}, dark ominous lighting, crumbling aesthetic",
        f"RIGHT: {new_visual}{entity_visual}, bright hopeful lighting, pristine",
        f"center vertical divider: glowing {accent} energy beam separating two worlds",
        f"bold text overlay: '{hook_words}' in massive Montserrat Black font",
        "text color: Bright Yellow / Pure White on dark bg, high contrast stroke",
        "background: deep neon blue-black (#0A0A15), cyberpunk noir atmosphere",
        f"accent palette: {accent}, Bright Yellow (#FFD600), Neon Red (#FF2020)",
        "style: hyper-realistic 3D render, Unreal Engine 5, octane render, 8k",
        "clean typography, minimal visual clutter, professional thumbnail design",
        "dramatic volumetric lighting, ray-traced reflections, depth of field",
        "--ar 16:9 --stylize 750 --v 6.1"
    ]
    
    return " | ".join(prompt_parts)


def _extract_hook_words(title: str, summary: str) -> str:
    """Extract 3-4 word hook from title/summary for thumbnail text."""
    # Common high-CTR patterns
    patterns = [
        (r"(don't|never|stop|avoid|warning)\s+\w+", "DON'T USE THIS"),
        (r"(how to|why you should|you must)\s+\w+", "DO THIS NOW"),
        (r"(secret|hidden|revealed|exposed)", "SECRET REVEALED"),
        (r"(best|top|ultimate|complete)\s+\w+", "ULTIMATE GUIDE"),
        (r"(new|just launched|breaking|announced)", "JUST LAUNCHED"),
        (r"(vs|versus|compared|beats)", "THIS BEATS THAT"),
        (r"(free|open source|no cost)", "COMPLETELY FREE"),
        (r"(fast|instant|seconds|lightning)", "INSTANT RESULTS"),
        (r"(ai|artificial intelligence|llm|gpt)", "AI CHANGES EVERYTHING"),
    ]
    
    content = (title + " " + summary).lower()
    for pattern, hook in patterns:
        import re
        if re.search(pattern, content):
            return hook
    
    # Fallback: extract key nouns from title
    words = title.split()
    key_words = [w for w in words if len(w) > 3 and w.lower() not in 
                 {"the", "and", "for", "with", "this", "that", "your", "how", "why", "what"}]
    if key_words:
        return " ".join(key_words[:3]).upper()
    
    return "AI BREAKTHROUGH"


def generate_midjourney_prompt(script_json: dict) -> str:
    """Alias for generate_thumbnail_prompt for clarity."""
    return generate_thumbnail_prompt(script_json)


def generate_dalle_prompt(script_json: dict) -> str:
    """Generates a DALL-E 3 optimized prompt (more natural language)."""
    base_prompt = generate_thumbnail_prompt(script_json)
    # Convert Midjourney parameters to natural language for DALL-E
    dalle_prompt = base_prompt.replace("| ", ", ").replace("--ar 16:9", "16:9 aspect ratio").replace("--stylize 750", "highly stylized").replace("--v 6.1", "photorealistic")
    return f"Create a professional YouTube thumbnail: {dalle_prompt}. Professional photography lighting, commercial quality."


if __name__ == "__main__":
    # Test the prompt generator
    test_script = {
        "title": "Google's New AI Search Kills Traditional SEO Forever",
        "description": "Google just launched AI Overviews that completely changes how search works. Traditional SEO tactics are dead. Here's what replaces them.",
        "sub_category": "AI & Tech Tools",
        "companies_mentioned": [{"name": "Google"}, {"name": "OpenAI"}],
        "tools_mentioned": [{"name": "Gemini"}, {"name": "Search Console"}],
        "longform_topics": [
            {"headline": "AI Overviews Launch", "source_name": "Google"},
            {"headline": "SEO Is Dead", "source_name": "Search Engine Journal"},
        ]
    }
    
    print("=" * 80)
    print("MIDJOURNEY PROMPT:")
    print("=" * 80)
    print(generate_thumbnail_prompt(test_script))
    print()
    print("=" * 80)
    print("DALL-E 3 PROMPT:")
    print("=" * 80)
    print(generate_dalle_prompt(test_script))

# ── TEST RUN ─────────────────────────────────────────────────────────────────
# test_json = {"title": "OpenAI Search is finally here...", "color_theme": {"accent": "#00E5FF"}}
# generate_thumbnail(test_json)
