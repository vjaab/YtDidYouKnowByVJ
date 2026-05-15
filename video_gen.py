"""
video_gen.py — Full 15-layer engagement video.

LAYER 1:  Background chunk clips (Pexels video / Ken Burns)
LAYER 2:  Color grading
LAYER 3:  Dark gradient overlay (bottom 45%)
LAYER 4:  Ambient particles (AI/Space/Tech categories)
LAYER 5:  Hook banner (first 2 seconds, slides from top)
LAYER 6:  Animated logo intro → shrinks to watermark
LAYER 7:  Fact highlight box (at key_stat_timestamp)
LAYER 8:  Reaction emoji burst (at shocking_moment_timestamp)
LAYER 9:  Like reminder (at 50% mark)
LAYER 10: Share prompt (at 80% mark)
LAYER 11: Static title box (192px from bottom, entire video)
LAYER 12: Telegram CTA card (last 6 seconds, slides up)
LAYER 13: Subscribe animation (last 3 seconds, pulse)
LAYER 14: Progress bar (6px at very bottom)
LAYER 15: Background music (vol 0.045)
"""

import os
import sys
import io
import math
import random
import re
import json
import threading
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from datetime import datetime
from google import genai
from google.genai import types
from moviepy import (
    VideoClip, ImageClip, VideoFileClip, AudioFileClip, AudioClip,
    CompositeVideoClip, ColorClip, CompositeAudioClip, concatenate_videoclips, concatenate_audioclips
)
import moviepy.video.fx as vfx
import moviepy.audio.fx as afx
from config import OUTPUT_DIR, ASSETS_DIR, MUSIC_DIR, BGM_VOLUME, LOGS_DIR, BASE_DIR, GEMINI_API_KEY
import imageio_ffmpeg
from pydub import AudioSegment
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

FRAME_W, FRAME_H = 1080, 1920 # Default for Shorts
def set_resolutions(is_longform=False):
    global FRAME_W, FRAME_H
    if is_longform:
        FRAME_W, FRAME_H = 1920, 1080
    else:
        FRAME_W, FRAME_H = 1080, 1920

TITLE_BOTTOM_GAP = 192

import cv2

def apply_tech_grade(frame):
    # Boost contrast and apply a slight cooling (blue/teal) to shadows
    # and warming (orange) to highlights
    frame = frame.astype(np.float32) / 255.0
    frame[:, :, 0] *= 1.1  # Warm up Reds
    frame[:, :, 2] *= 1.2  # Cool down Blues
    return np.clip(frame * 255, 0, 255).astype(np.uint8)

# ── Font ──────────────────────────────────────────────────────────────────────
FONT_PATHS = [
    os.path.join(ASSETS_DIR, "fonts", "Montserrat-Bold.ttf"),
    os.path.join(ASSETS_DIR, "fonts", "Roboto-Bold.ttf"),
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/usr/share/fonts/truetype/roboto/hinted/Roboto-Bold.ttf",
    "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf",  # Linux
]
_fc = {}

def gf(size, bold=False, italic=False):
    """
    Global Font Loader with caching and fallback weights.
    Ensures that 'bold=True' requests actually return a bold font variant.
    """
    key = (size, bold, italic)
    if key not in _fc:
        search_paths = []
        
        # 1. Prioritize specific fonts for bold/italic if available in assets
        if italic:
            search_paths.append(os.path.join(ASSETS_DIR, "fonts", "Montserrat-Italic.ttf"))
        if bold:
            # Try ExtraBold first, then standard Bold
            search_paths.append(os.path.join(ASSETS_DIR, "fonts", "Montserrat-ExtraBold.ttf"))
            search_paths.append(os.path.join(ASSETS_DIR, "fonts", "Montserrat-Bold.ttf"))
        
        # 2. Add the rest of the system/default paths
        for p in FONT_PATHS:
            if p not in search_paths:
                search_paths.append(p)
                
        # 3. Load first available
        for p in search_paths:
            if os.path.exists(p):
                try:
                    _fc[key] = ImageFont.truetype(p, size)
                    break
                except Exception:
                    pass
        
        # 4. Final Fallback
        if key not in _fc:
            _fc[key] = ImageFont.load_default()
            
    return _fc[key]

def ts(text, font):
    bb = font.getbbox(text)
    return bb[2] - bb[0], bb[3] - bb[1]

def _prepare_screenshot_canvas(img, target_w, target_h):
    """
    Creates a premium 'Blurred Backdrop' canvas for wide screenshots.
    Ensures the original image is fully visible (contained) in the center.
    """
    # 1. Create heavily blurred background (scaled to fill)
    bg = ImageOps.fit(img, (target_w, target_h), Image.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=45))
    bg = bg.point(lambda p: p * 0.45) # Darken backdrop
    
    # 2. Prepare foreground (scaled to fit)
    iw, ih = img.size
    scale = min(target_w / iw, target_h / ih) * 0.92 # Slightly smaller for breathing room
    fw, fh = int(iw * scale), int(ih * scale)
    fg = img.resize((fw, fh), Image.LANCZOS).convert("RGBA")
    
    # 3. Add premium drop shadow/glow to foreground
    shadow_pad = 20
    shadow_img = Image.new("RGBA", (fw + shadow_pad*2, fh + shadow_pad*2), (0,0,0,0))
    s_draw = ImageDraw.Draw(shadow_img)
    s_draw.rectangle([shadow_pad, shadow_pad, fw+shadow_pad, fh+shadow_pad], fill=(0,0,0,180))
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=15))
    
    # 4. Composite
    canvas = bg.convert("RGBA")
    canvas.paste(shadow_img, ((target_w - shadow_img.width)//2, (target_h - shadow_img.height)//2), shadow_img)
    canvas.paste(fg, ((target_w - fw)//2, (target_h - fh)//2), fg)
    
    return canvas.convert("RGB")

def _crop_to_circle(img, border_color=(255, 214, 0), border_width=4):
    """Crops an image into a circle with a premium border."""
    img = img.convert("RGBA")
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, w, h), fill=255)
    
    result = Image.new("RGBA", (w, h), (0,0,0,0))
    result.paste(img, (0, 0), mask=mask)
    
    # Add border
    draw_result = ImageDraw.Draw(result)
    draw_result.ellipse((0, 0, w, h), outline=border_color, width=border_width)
    return result

def get_cinematic_font(size, bold=True, italic=False):
    """
    Premium 2026 Spec: High-authority Sans-Serif.
    Defaults to bold=True for impact.
    """
    return gf(size, bold=bold, italic=italic)


# ── Ken Burns ─────────────────────────────────────────────────────────────────
_kb_idx = 0
KB_PATTERNS = ["smooth_zoom", "reveal_zoom", "z_pan_high_energy", "z_pan_subtle"]

def get_ease_factor(t):
    # Standard Ease-In-Out Quadratic
    return 2 * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 2) / 2

def build_ken_burns(img_path, duration, pattern_idx):
    pattern = KB_PATTERNS[pattern_idx % len(KB_PATTERNS)]
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return ColorClip(size=(FRAME_W, FRAME_H), color=(15, 15, 25), duration=duration)
        
    # Heavy padding to ensure we do not hit black borders during diagonal pans/rotations
    pad = int(FRAME_W * 0.2)
    pw, ph = FRAME_W + pad*2, FRAME_H + pad*2
    
    # Pre-scale to the padded resolution once so we aren't doing heavy lifting in the loop
    base_img = ImageOps.fit(img, (pw, ph), Image.LANCZOS)
    
    def make_frame(t):
        progress = min(t / max(duration, 0.01), 1.0)
        eased_t = get_ease_factor(progress)
        
        # Default Centers
        cx, cy = pw // 2, ph // 2
        angle = 0
        
        if pattern == "smooth_zoom":
            # Golden Ratio Zoom In (10-15% max)
            current_scale = 1.0 + (0.15 * eased_t)
        elif pattern == "reveal_zoom":
            # Golden Ratio Zoom Out (1.15 to 1.0)
            current_scale = 1.15 - (0.15 * eased_t)
        elif pattern == "z_pan_high_energy":
            # Diagonal pan (Bottom-Left to Top-Right) + 2% rotation
            current_scale = 1.10 + (0.05 * eased_t)
            angle = -1.0 + (2.0 * eased_t)
            cx = pw // 2 + int(-40 + 80 * eased_t)
            cy = ph // 2 + int(40 - 80 * eased_t)
        else:
            # Subtle Pan (Start slightly off-center left -> zoom toward upper-right third)
            current_scale = 1.0 + (0.12 * eased_t)
            cx = pw // 2 + int(-25 + 50 * eased_t)
            cy = ph // 2 + int(15 - 30 * eased_t)

        cw, ch = int(FRAME_W / current_scale), int(FRAME_H / current_scale)
        
        # Keep viewport strictly within bounds
        cx = max(cw // 2, min(cx, pw - cw // 2))
        cy = max(ch // 2, min(cy, ph - ch // 2))
        
        x1 = cx - cw // 2
        y1 = cy - ch // 2
        
        crop = base_img.crop((x1, y1, x1 + cw, y1 + ch))
        if angle != 0:
            crop = crop.rotate(angle, resample=Image.BICUBIC, expand=False)
            
        out = np.array(crop.resize((FRAME_W, FRAME_H), Image.BICUBIC), dtype=np.float32)
        # Small brightness/contrast multiplier for consistency
        return np.clip(out * 0.88 * 1.12, 0, 255).astype(np.uint8)

    return VideoClip(make_frame, duration=duration)


def build_video_clip(video_path, duration):
    try:
        clip = VideoFileClip(video_path)
        w, h = clip.size
        target_h = int(w * 16 / 9)
        if target_h <= h:
            y1 = (h - target_h) // 2
            clip = clip.cropped(x1=0, y1=y1, x2=w, y2=y1+target_h)
        else:
            target_w = int(h * 9 / 16)
            x1 = (w - target_w) // 2
            clip = clip.cropped(x1=x1, y1=0, x2=x1+target_w, y2=h)
        clip = clip.resized((FRAME_W, FRAME_H))
        if clip.duration > duration:
            s = (clip.duration - duration) / 2
            clip = clip.subclipped(s, s + duration)
        elif clip.duration < duration:
            clip = clip.with_effects([vfx.Loop(duration=duration)])
        clip = clip.with_effects([vfx.LumContrast(lum=0, contrast=0.12), vfx.MultiplyColor(0.88)])
        return clip.without_audio()
    except Exception as e:
        print(f"Video clip failed: {e}")
        return None


# ── PIL clip helper ───────────────────────────────────────────────────────────
def _pil_clip(pil_img, duration, pos=("center", "center"), start=0, opacity=1.0):
    """Wrap a PIL RGBA image into a positioned, masked MoviePy clip."""
    rgb_arr  = np.array(pil_img.convert("RGB"))
    mask_arr = np.array(pil_img.split()[3]).astype(float) / 255.0
    clip = ImageClip(rgb_arr, duration=duration)
    mask = VideoClip(lambda t: mask_arr, is_mask=True, duration=duration)
    clip = clip.with_mask(mask).with_position(pos).with_start(start)
    if opacity < 1.0:
        clip = clip.with_opacity(opacity)
    return clip


# ── LAYER 3: Gradient ─────────────────────────────────────────────────────────
def _gradient_clip(duration):
    # Strong dark gradient covering bottom 45% — blends avatar into B-roll
    h = int(FRAME_H * 0.45)
    arr = np.zeros((h, FRAME_W, 3), dtype=np.uint8)
    mask_arr = np.array(
        [(int(255 * (y/h)**0.5),) * FRAME_W for y in range(h)],
        dtype=float) / 255.0
    clip = ImageClip(arr, duration=duration)
    mask = VideoClip(lambda t: mask_arr, is_mask=True, duration=duration)
    return clip.with_mask(mask).with_position(("center", "bottom"))


# ── LAYER 4: Ambient "Obsidian" particles ──────────────────────────────────────
# ── LAYER 4: Ambient "Obsidian" particles ──────────────────────────────────────
def _ambient_particles(duration, accent_color, particle_style="bokeh"):
    n = 35
    random.seed(42)
    particles = [(random.uniform(0, FRAME_W), random.uniform(0, FRAME_H),
                  random.uniform(0.1, 0.8), random.uniform(0, FRAME_H), random.uniform(8, 25))
                 for _ in range(n)]

    def make_frame(t):
        img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for px, py, speed, offset, p_size in particles:
            y = (py - speed * t * 60 - offset) % FRAME_H
            if particle_style == "digital":
                # Digital blocks/lines
                cv2.rectangle(img, (int(px), int(y)), (int(px+p_size), int(y+2)), accent_color, -1)
            elif particle_style == "stars":
                # Tiny sharp stars
                cv2.circle(img, (int(px), int(y)), 2, (255, 255, 255), -1)
            else:
                # Default bokeh
                cv2.circle(img, (int(px), int(y)), int(p_size), accent_color, -1)
        
        if particle_style != "stars":
            blur_size = 75 if particle_style == "bokeh" else 15
            img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        return img

    def make_mask(t):
        mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        for px, py, speed, offset, p_size in particles:
            y = (py - speed * t * 60 - offset) % FRAME_H
            if particle_style == "digital":
                cv2.rectangle(mask, (int(px), int(y)), (int(px+p_size), int(y+2)), 255, -1)
            else:
                cv2.circle(mask, (int(px), int(y)), int(p_size), 255, -1)
        
        if particle_style != "stars":
            blur_size = 75 if particle_style == "bokeh" else 15
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        return (mask.astype(float) / 255.0) * 0.15 # 15% Opacity digital dust

    clip = VideoClip(make_frame, duration=duration)
    mask = VideoClip(make_mask, is_mask=True, duration=duration)
    return clip.with_mask(mask)

# ── LAYER 1: Dynamic Minimalist Background ────────────────────────────────────
def _dynamic_tech_background(duration, accent_color, bg_base_color=(10, 10, 15)):
    """Generates a high-end, 0-cost local dark tech background (Obsidian style)."""
    # Create base grid and particles
    cols, rows = 12, 22
    spacing_x = FRAME_W // cols
    spacing_y = FRAME_H // rows
    
    def make_frame(t):
        # Dark obsidian base (now randomized)
        frame = np.full((FRAME_H, FRAME_W, 3), bg_base_color, dtype=np.uint8)
        
        # Draw moving grid lines
        grid_alpha = 0.08 + 0.02 * math.sin(t * 0.5)
        grid_color = tuple(int(c * grid_alpha) for c in accent_color)
        
        # Vertical lines with subtle drift
        drift_x = (t * 10) % spacing_x
        for x in range(-spacing_x, FRAME_W + spacing_x, spacing_x):
            cv2.line(frame, (int(x + drift_x), 0), (int(x + drift_x), FRAME_H), grid_color, 1)
            
        # Horizontal lines with subtle drift
        drift_y = (t * 8) % spacing_y
        for y in range(-spacing_y, FRAME_H + spacing_y, spacing_y):
            cv2.line(frame, (0, int(y + drift_y)), (FRAME_W, int(y + drift_y)), grid_color, 1)

        # Pulse glow at center
        pulse = (math.sin(t * 1.2) + 1) / 2
        glow_radius = int(400 + 50 * pulse)
        glow_img = np.zeros_like(frame)
        cv2.circle(glow_img, (FRAME_W//2, FRAME_H//2), glow_radius, accent_color, -1)
        glow_img = cv2.GaussianBlur(glow_img, (151, 151), 0)
        
        # Blend glow with 5% opacity
        frame = cv2.addWeighted(frame, 1.0, glow_img, 0.05, 0)
        
        return frame

    return VideoClip(make_frame, duration=duration)





# ── LAYER 6: Animated logo ─────────────────────────────────────────────────────


# ── LAYER 6: Animated logo ─────────────────────────────────────────────────────
def _animated_logo(duration):
    logo_path = os.path.join(ASSETS_DIR, "logo.png")
    if not os.path.exists(logo_path):
        return None
    try:
        logo_img = Image.open(logo_path).convert("RGBA").resize((110, 110), Image.LANCZOS)
        arr  = np.array(logo_img.convert("RGB"))
        mask = np.array(logo_img.split()[3]).astype(float) / 255.0
        # Simple fade‑in animation for the logo
        def pos(t):
            # Centered for the intro duration
            return (int((FRAME_W - 110) // 2), int(FRAME_H * 0.1))
        clip = VideoClip(lambda t: arr, duration=duration)
        mclip = VideoClip(lambda t: mask, is_mask=True, duration=duration)
        clip = clip.with_mask(mclip).with_position(pos)
        return clip
    except Exception:
        return None


# ── LAYER 7: Fact highlight box ───────────────────────────────────────────────
def _fact_box(key_stat, start_time, accent_color, total_dur):
    if not key_stat or start_time >= total_dur:
        return None
    dur = min(1.5, total_dur - start_time)
    f = gf(110) # Increased from 90 for better readability
    w, h = ts(key_stat, f)
    pad = 20
    img = Image.new("RGBA", (w + pad*2 + 6, h + pad*2 + 6), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, w+pad*2+5, h+pad*2+5], radius=16, fill=(*accent_color, 230))
    draw.rounded_rectangle([3, 3, w+pad*2+2, h+pad*2+2], radius=14, outline=(255,255,255), width=3)
    draw.text((pad, pad), key_stat, font=f, fill=(255,255,255,255))

    def make_frame(t):
        scale = 1.0 + 0.3 * math.exp(-t * 6) * math.sin(t * 20)
        scaled = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
        return np.array(scaled.convert("RGB"))

    def make_mask(t):
        scale = 1.0 + 0.3 * math.exp(-t * 6) * math.sin(t * 20)
        scaled = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
        return np.array(scaled.split()[3]).astype(float) / 255.0

    clip = VideoClip(make_frame, duration=dur).with_start(start_time)
    mask = VideoClip(make_mask, is_mask=True, duration=dur).with_start(start_time)
    clip = clip.with_mask(mask).with_position(("center", int(FRAME_H * 0.38)))
    return clip


# ── LAYER 8: Reaction emoji burst ─────────────────────────────────────────────
def _emoji_burst(start_time, total_dur):
    if start_time >= total_dur:
        return []
    dur = min(0.8, total_dur - start_time)
    burst_emojis = ["😱", "🤯", "🔥", "💥", "⚡"]
    clips = []
    for i, emoji in enumerate(burst_emojis[:5]):
        angle = (i / 5) * math.pi * 2
        tx = int(FRAME_W//2 + 220 * math.cos(angle))
        ty = int(FRAME_H * 0.38 + 300 * math.sin(angle)) # Center burst higher up
        f = gf(80)
        img = Image.new("RGBA", (120, 120), (0,0,0,0))
        ImageDraw.Draw(img).text((10, 10), emoji, font=f)

        def make_frame(t, _tx=tx, _ty=ty, _img=img):
            progress = t / dur
            scale = progress * 1.5 if progress < 0.6 else 1.5 * (1 - (progress - 0.6) / 0.4)
            alpha = int(255 * max(0, 1 - progress * 1.3))
            scaled = _img.resize((max(1, int(120*scale)), max(1, int(120*scale))), Image.LANCZOS)
            result = Image.new("RGBA", (FRAME_W, FRAME_H), (0,0,0,0))
            px = _tx - scaled.width//2
            py = _ty - scaled.height//2 - int(60 * progress)
            try:
                result.paste(scaled, (px, py), scaled)
            except Exception:
                pass
            # Apply fade
            arr = np.array(result.convert("RGB"))
            return arr

        def make_mask(t, _tx=tx, _ty=ty, _img=img):
            progress = t / dur
            scale = progress * 1.5 if progress < 0.6 else 1.5 * (1 - (progress - 0.6) / 0.4)
            alpha_v = max(0.0, 1.0 - progress * 1.3)
            scaled = _img.resize((max(1, int(120*scale)), max(1, int(120*scale))), Image.LANCZOS)
            result = Image.new("L", (FRAME_W, FRAME_H), 0)
            m = Image.fromarray((np.array(scaled.split()[3]) * alpha_v).astype(np.uint8))
            px = _tx - scaled.width//2
            py = _ty - scaled.height//2 - int(60 * progress)
            try:
                result.paste(m, (px, py))
            except Exception:
                pass
            return np.array(result).astype(float) / 255.0

        clip = VideoClip(make_frame, duration=dur).with_start(start_time)
        mask = VideoClip(make_mask, is_mask=True, duration=dur).with_start(start_time)
        clips.append(clip.with_mask(mask))
    return clips


# ── LAYER 9 & 10: Like / Share reminders ──────────────────────────────────────
def _pill_reminder(text, start_time, total_dur, hold=2.0):
    if start_time >= total_dur:
        return None
    dur = min(hold + 0.6, total_dur - start_time)
    f = gf(28)
    w, h = ts(text, f)
    pad_x, pad_y = 20, 12
    img = Image.new("RGBA", (w + pad_x*2, h + pad_y*2), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0,0,w+pad_x*2,h+pad_y*2], radius=24, fill=(0,0,0,179))
    draw.text((pad_x, pad_y), text, font=f, fill=(255,255,255,255))
    arr  = np.array(img.convert("RGB"))
    mask = np.array(img.split()[3]).astype(float) / 255.0

    def opacity_fn(t):
        fade_in  = 0.3
        fade_out = 0.3
        rel_end  = dur - fade_out
        if t < fade_in:
            return t / fade_in
        elif t > rel_end:
            return max(0, (dur - t) / fade_out)
        return 1.0

    clip = VideoClip(lambda t: arr, duration=dur)
    mclip = VideoClip(lambda t: mask * opacity_fn(t), is_mask=True, duration=dur)
    return clip.with_mask(mclip).with_position((30, 110)).with_start(start_time)


# ── LAYER 11: Static title ────────────────────────────────────────────────────
def _title_clip(title, duration):
    f = gf(58)
    max_w = 900
    words = title.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if ts(test, f)[0] > max_w and cur:
            lines.append(" ".join(cur)); cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    lines = lines[:2]

    lh     = ts("Ag", f)[1]
    lsp    = int(lh * 1.3)
    bl_h   = lh + (len(lines)-1) * lsp
    lwidths = [ts(l, f)[0] for l in lines]
    box_w  = min(max(lwidths) + 60, FRAME_W - 20)
    box_h  = bl_h + 40
    canvas = Image.new("RGBA", (FRAME_W, box_h + 10), (0,0,0,0))
    draw   = ImageDraw.Draw(canvas)
    bx1 = (FRAME_W - box_w) // 2
    draw.rounded_rectangle([bx1, 5, bx1+box_w, box_h+5], radius=12, fill=(0,0,0,140))
    for i, line in enumerate(lines):
        lw, _ = ts(line, f)
        tx = (FRAME_W - lw) // 2
        ty = 5 + 20 + i * lsp
        for dx, dy in [(-3,0),(3,0),(0,-3),(0,3)]:
            draw.text((tx+dx, ty+dy), line, font=f, fill=(0,0,0,220))
        draw.text((tx, ty), line, font=f, fill=(255,255,255,255))

    arr  = np.array(canvas.convert("RGB"))
    mask = np.array(canvas.split()[3]).astype(float) / 255.0
    clip = ImageClip(arr, duration=duration)
    mclip = VideoClip(lambda t: mask, is_mask=True, duration=duration)
    y_pos = FRAME_H - TITLE_BOTTOM_GAP - box_h
    return clip.with_mask(mclip).with_position(("center", y_pos))


# ── LAYER 12: Telegram CTA card ───────────────────────────────────────────────
def _telegram_cta_overlay(total_dur):
    """Shows the two Telegram screenshots sequentially in the last 4 seconds."""
    cta_dur = 4.0
    if total_dur < cta_dur + 1:
        return None
    start_t = total_dur - cta_dur

    try:
        p1 = os.path.join(ASSETS_DIR, "branding", "tele_brand1.jpg")
        p2 = os.path.join(ASSETS_DIR, "branding", "tele_brand2.jpg")
        
        # Use .png fallback if .jpg doesn't exist
        if not os.path.exists(p1): p1 = p1.replace(".jpg", ".png")
        if not os.path.exists(p2): p2 = p2.replace(".jpg", ".png")
        
        if not os.path.exists(p1) or not os.path.exists(p2):
            return None

        def create_simple_clip(path, duration, start_offset):
            img = Image.open(path).convert("RGB")
            # Scale to fit width, maintaining aspect ratio
            w = int(FRAME_W * 0.9)
            ratio = w / float(img.width)
            h = int(img.height * ratio)
            img = img.resize((w, h), Image.Resampling.LANCZOS)
            
            arr = np.array(img)
            clip = ImageClip(arr, duration=duration)
            return clip.with_position("center").with_start(start_t + start_offset)

        # Show brand1 for 2s, then brand2 for 2s (Total 4s Blitz)
        c1 = create_simple_clip(p1, 2.0, 0)
        c2 = create_simple_clip(p2, 2.0, 2.0)
        
        # Add "Source Code & Guide" overlay text in the top half
        from moviepy.video.VideoClip import TextClip
        f_cta = get_cinematic_font(48, bold=True)
        txt = "Get the Full Resource Guide 📥\nLink in Bio"
        
        # We'll use a manual Text Overlay on a small canvas to avoid ImageMagick dependencies if possible,
        # but since we already have ImageDraw logic elsewhere, let's stick to it.
        def make_txt_overlay(t):
            overlay = Image.new("RGBA", (FRAME_W, 200), (0,0,0,0))
            d = ImageDraw.Draw(overlay)
            tw, th = ts(txt, f_cta)
            # Semi-transparent backing
            d.rounded_rectangle([(FRAME_W-tw)//2-20, 20, (FRAME_W+tw)//2+20, 180], radius=20, fill=(0,0,0,180))
            d.text(((FRAME_W-tw)//2, 50), txt, font=f_cta, fill=(255,215,0,255), align="center")
            return np.array(overlay)

        t_overlay = VideoClip(make_txt_overlay, duration=cta_dur).with_position(("center", 150)).with_start(start_t)
        
        return [c1, c2, t_overlay]
        
    except Exception as e:
        print("Sequential CTA Error:", e)
        return None
        
    except Exception as e:
        print("Dual Card CTA Error:", e)
        return None# ══════════════════════════════════════════════════════════════════════════════
# ── LAYER E2: Article Evidence Scan (Social Proof) ─────────────────────────────
def _article_scan_overlay(image_path, start_t, duration=2.5):
    """Slides an article snippet in from the side in the top 50% zone."""
    try:
        img = Image.open(image_path).convert("RGBA")
        
        # We want a "Snippet" feel, so we take the top-middle part of the article
        w, h = img.size
        # Crop a nice horizontal strip
        snippet = img.crop((0, int(h*0.05), w, int(h*0.45)))
        
        # Scale to fit width
        target_w = int(FRAME_W * 0.9)
        ratio = target_w / float(snippet.width)
        target_h = int(snippet.height * ratio)
        snippet = snippet.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # Add rounded corners and a glowy border
        mask = Image.new("L", (target_w, target_h), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, target_w, target_h], radius=30, fill=255)
        snippet.putalpha(mask)
        
        # Add "EVIDENCE" badge
        draw = ImageDraw.Draw(snippet)
        f_badge = get_cinematic_font(32, bold=True)
        draw.rounded_rectangle([20, 20, 220, 70], radius=10, fill=(255, 0, 0, 230))
        draw.text((45, 25), "EVIDENCE", font=f_badge, fill=(255,255,255,255))
        
        arr = np.array(snippet.convert("RGB"))
        alpha = np.array(snippet.split()[3]).astype(float) / 255.0
        
        clip = ImageClip(arr, duration=duration)
        mclip = VideoClip(lambda t: alpha, is_mask=True, duration=duration)
        
        def pos_fn(t):
            # Slide in from right, pause, slide out to left
            if t < 0.5:
                # Slide in
                x = FRAME_W - (FRAME_W - (FRAME_W - target_w)//2) * (t/0.5)
            elif t > duration - 0.5:
                # Slide out
                x = (FRAME_W - target_w)//2 - (target_w + 100) * ((t - (duration-0.5))/0.5)
            else:
                x = (FRAME_W - target_w)//2
            return (int(x), 280) # Top 50% zone
            
        return clip.with_mask(mclip).with_position(pos_fn).with_start(start_t)
    except Exception as e:
        print(f"Evidence Scan Error: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# ENGAGEMENT LAYERS (Retention Boosters)
# ══════════════════════════════════════════════════════════════════════════════

# ── LAYER E1: Pattern Interrupt Flash (Stop the Scroll) ───────────────────────
def _pattern_interrupt_flash(accent_color, total_dur):
    """A 0.3s color flash at the very start to pattern-interrupt the feed scroll."""
    dur = min(0.3, total_dur)
    def make_frame(t):
        # Bright flash that fades to black in 0.3s
        intensity = max(0, 1.0 - (t / dur))
        frame = np.full((FRAME_H, FRAME_W, 3), 0, dtype=np.uint8)
        # Mix accent color with white for a punchy flash
        flash_color = tuple(min(255, int(c + (255 - c) * intensity * 0.7)) for c in accent_color)
        frame[:, :] = flash_color
        return frame

    def make_mask(t):
        intensity = max(0, 1.0 - (t / dur)) * 0.85
        return np.full((FRAME_H, FRAME_W), intensity, dtype=np.float64)

    clip = VideoClip(make_frame, duration=dur)
    mask = VideoClip(make_mask, is_mask=True, duration=dur)
    return clip.with_mask(mask).with_start(0)


# ── LAYER E2: Giant Hook Text (First 1.5s) ────────────────────────────────────
def _hook_text_overlay(hook_text, accent_color, total_dur):
    """Displays giant hook text — clean white serif, no background box (reference style)."""
    if not hook_text:
        return None
    dur = min(2.5, total_dur)
    f = get_cinematic_font(68, italic=True)
    max_w = FRAME_W - 120

    # Word-wrap the hook text
    words = hook_text.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if ts(test, f)[0] > max_w and cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    lines = lines[:3]

    lh = ts("Ag", f)[1]
    lsp = int(lh * 1.5)
    total_h = lh + (len(lines) - 1) * lsp
    canvas_h = total_h + 60
    canvas = Image.new("RGBA", (FRAME_W, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for i, line in enumerate(lines):
        lw, _ = ts(line, f)
        tx = (FRAME_W - lw) // 2
        ty = 30 + i * lsp
        for dx, dy in [(-3, -3), (3, -3), (-3, 3), (3, 3), (-2, 0), (2, 0), (0, -2), (0, 2)]:
            draw.text((tx + dx, ty + dy), line, font=f, fill=(0, 0, 0, 200))
        draw.text((tx, ty), line, font=f, fill=(255, 255, 255, 255))

    arr = np.array(canvas.convert("RGB"))
    mask = np.array(canvas.split()[3]).astype(float) / 255.0

    def opacity_fn(t):
        if t < 0.2:
            return t / 0.2
        elif t > dur - 0.5:
            return max(0, (dur - t) / 0.5)
        return 1.0

    clip = VideoClip(lambda t: arr, duration=dur)
    mclip = VideoClip(lambda t: mask * opacity_fn(t), is_mask=True, duration=dur)
    y_pos = int(FRAME_H * 0.38)
    return clip.with_mask(mclip).with_position(("center", y_pos)).with_start(0)


# ── LAYER E3: Micro-Cliffhanger Captions ──────────────────────────────────────
def _micro_cliffhanger_overlay(cliffhangers, accent_color, total_dur):
    """Animated teaser text overlays that appear every ~10 seconds."""
    clips = []
    if not cliffhangers:
        return clips

    for ch in cliffhangers:
        ch_ts = float(ch.get("timestamp", 0))
        ch_text = ch.get("text", "")
        if not ch_text or ch_ts >= total_dur - 1:
            continue

        dur = min(2.0, total_dur - ch_ts)
        f = gf(32)
        w, h = ts(ch_text, f)
        pad_x, pad_y = 24, 14
        img = Image.new("RGBA", (w + pad_x * 2, h + pad_y * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Accent-colored pill with glow
        draw.rounded_rectangle([0, 0, w + pad_x * 2, h + pad_y * 2], radius=20, fill=(*accent_color, 200))
        draw.text((pad_x, pad_y), ch_text, font=f, fill=(255, 255, 255, 255))
        arr = np.array(img.convert("RGB"))
        mask_arr = np.array(img.split()[3]).astype(float) / 255.0

        def make_opacity(t, _dur=dur):
            if t < 0.2:
                return t / 0.2
            elif t > _dur - 0.3:
                return max(0, (_dur - t) / 0.3)
            return 1.0

        # Slide in from right
        def make_pos(t, _dur=dur):
            slide = min(t / 0.3, 1.0)
            x = int(FRAME_W - (FRAME_W * 0.9) * slide)
            return (min(x, FRAME_W - 50), int(FRAME_H * 0.45))

        clip = VideoClip(lambda t: arr, duration=dur)
        mclip = VideoClip(lambda t, _dur=dur: mask_arr * make_opacity(t, _dur), is_mask=True, duration=dur)
        clips.append(clip.with_mask(mclip).with_position(make_pos).with_start(ch_ts))

    return clips


# ── LAYER E4: Interactive Challenge Banner ────────────────────────────────────
def _interactive_challenge_overlay(challenge_data, accent_color, total_dur):
    """A comment/challenge prompt with pulsing border."""
    if not challenge_data:
        return None
    ch_ts = float(challenge_data.get("timestamp", 0))
    ch_text = challenge_data.get("text", "")
    if not ch_text or ch_ts >= total_dur - 2:
        return None

    dur = min(3.0, total_dur - ch_ts)
    f = gf(36)
    w, h = ts(ch_text, f)
    pad_x, pad_y = 30, 20
    box_w = w + pad_x * 2 + 8
    box_h = h + pad_y * 2 + 8

    def make_frame(t):
        img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Pulsing border
        pulse = 1.0 + 0.15 * math.sin(t * 8)
        border_w = max(2, int(3 * pulse))
        draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=16, fill=(0, 0, 0, 200), outline=(*accent_color, 255), width=border_w)
        # Icon
        icon_f = gf(30)
        draw.text((pad_x - 10, pad_y - 2), "💬", font=icon_f)
        draw.text((pad_x + 30, pad_y), ch_text, font=f, fill=(255, 255, 255, 255))
        return np.array(img.convert("RGB"))

    def make_mask(t):
        img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        pulse = 1.0 + 0.15 * math.sin(t * 8)
        border_w = max(2, int(3 * pulse))
        draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=16, fill=(255, 255, 255, 220), outline=(255, 255, 255, 255), width=border_w)
        opacity = 1.0
        if t < 0.3:
            opacity = t / 0.3
        elif t > dur - 0.5:
            opacity = max(0, (dur - t) / 0.5)
        return np.array(img.split()[3]).astype(float) / 255.0 * opacity

    clip = VideoClip(make_frame, duration=dur)
    mclip = VideoClip(make_mask, is_mask=True, duration=dur)
    x_pos = (FRAME_W - box_w) // 2
    y_pos = int(FRAME_H * 0.28) # Moved up to clear the new lower-third subtitles
    return clip.with_mask(mclip).with_position((x_pos, y_pos)).with_start(ch_ts)


# ── LAYER E5: Identity CTA Card (Last 5s) ────────────────────────────────────
def _identity_cta_overlay(identity_text, accent_color, total_dur):
    """Identity-based CTA for the final moments of the video."""
    if not identity_text:
        return None
    dur = 2.0
    start = total_dur - dur  # Exactly in the last 4 seconds
    if start < 0:
        return None

    f = gf(34)
    max_w = FRAME_W - 120
    words = identity_text.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if ts(test, f)[0] > max_w and cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))

    lh = ts("Ag", f)[1]
    lsp = int(lh * 1.3)
    total_h = lh + (len(lines) - 1) * lsp
    box_h = total_h + 50
    box_w = FRAME_W - 80

    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Glassmorphism card
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=18, fill=(0, 0, 0, 160))
    # Accent top bar
    draw.rectangle([0, 0, box_w, 4], fill=(*accent_color, 255))

    for i, line in enumerate(lines):
        lw, _ = ts(line, f)
        tx = (box_w - lw) // 2
        ty = 25 + i * lsp
        draw.text((tx, ty), line, font=f, fill=(255, 255, 255, 230))

    arr = np.array(img.convert("RGB"))
    mask_arr = np.array(img.split()[3]).astype(float) / 255.0

    def opacity_fn(t):
        if t < 0.4:
            return t / 0.4
        elif t > dur - 0.5:
            return max(0, (dur - t) / 0.5)
        return 1.0

    clip = VideoClip(lambda t: arr, duration=dur)
    mclip = VideoClip(lambda t: mask_arr * opacity_fn(t), is_mask=True, duration=dur)
    x_pos = 40
    y_pos = int(FRAME_H - TITLE_BOTTOM_GAP - box_h - 280)
    return clip.with_mask(mclip).with_position((x_pos, y_pos)).with_start(start)


# ── VISUAL UNDERSTANDING LAYER: Infographic Cards ─────────────────────────────

def _render_definition_card(term, definition, accent_color, width=900):
    """Glassmorphism definition card: TERM + one-line explanation."""
    f_term = gf(52)
    f_def  = gf(34)

    # Word-wrap definition
    def_words = definition.split()
    def_lines, cur = [], []
    for w in def_words:
        test = " ".join(cur + [w])
        if ts(test, f_def)[0] > width - 80 and cur:
            def_lines.append(" ".join(cur)); cur = [w]
        else:
            cur.append(w)
    if cur: def_lines.append(" ".join(cur))

    term_h  = ts("Ag", f_term)[1]
    def_lh  = ts("Ag", f_def)[1]
    total_h = 30 + term_h + 16 + len(def_lines) * int(def_lh * 1.4) + 30

    img  = Image.new("RGBA", (width, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Card background
    draw.rounded_rectangle([0, 0, width, total_h], radius=22,
                            fill=(12, 12, 20, 220))
    # Accent left bar
    draw.rectangle([0, 0, 6, total_h], fill=(*accent_color, 255))
    # Top accent label strip
    draw.rounded_rectangle([12, 10, 160, 38], radius=10,
                            fill=(*accent_color, 180))
    draw.text((22, 12), "DEFINITION", font=gf(20), fill=(255, 255, 255, 255))

    # Term (bold, accent)
    draw.text((20, 44), term.upper(), font=f_term,
              fill=(*accent_color, 255))

    # Divider
    div_y = 44 + term_h + 8
    draw.line([(20, div_y), (width - 20, div_y)],
              fill=(*accent_color, 60), width=1)

    # Definition lines
    for i, line in enumerate(def_lines):
        draw.text((20, div_y + 10 + i * int(def_lh * 1.4)),
                  line, font=f_def, fill=(220, 220, 220, 255))

    return img


def _render_comparison_card(left_label, left_val, right_label, right_val,
                            accent_color, width=960):
    """Side-by-side comparison card: X vs Y."""
    f_label = gf(30)
    f_val   = gf(52)

    half = width // 2
    h    = 180
    img  = Image.new("RGBA", (width, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Left panel (dark)
    draw.rounded_rectangle([0, 0, half - 6, h], radius=18,
                            fill=(20, 20, 30, 220))
    # Right panel (accent tinted)
    draw.rounded_rectangle([half + 6, 0, width, h], radius=18,
                            fill=(*[min(255, c // 4) for c in accent_color], 220))

    # VS divider
    draw.ellipse([half - 28, h // 2 - 28, half + 28, h // 2 + 28],
                 fill=(*accent_color, 255))
    draw.text((half, h // 2), "VS", font=gf(26),
              fill=(255, 255, 255, 255), anchor="mm")

    # Left content
    lw_l, _ = ts(left_label, f_label)
    draw.text(((half - 6 - lw_l) // 2, 20), left_label,
              font=f_label, fill=(160, 160, 180, 255))
    lw_v, lh_v = ts(left_val, f_val)
    draw.text(((half - 6 - lw_v) // 2, 70), left_val,
              font=f_val, fill=(255, 255, 255, 255))

    # Right content
    rw_l, _ = ts(right_label, f_label)
    draw.text((half + 6 + (half - 6 - rw_l) // 2, 20), right_label,
              font=f_label, fill=(200, 200, 220, 255))
    rw_v, _ = ts(right_val, f_val)
    draw.text((half + 6 + (half - 6 - rw_v) // 2, 70), right_val,
              font=f_val, fill=(255, 255, 255, 255))

    return img


def _render_process_steps(steps, accent_color, width=960, active_step=None):
    """Numbered flow steps (up to 4) shown as a horizontal pill row."""
    n     = min(len(steps), 4)
    f     = gf(28)
    pad   = 18
    h     = 90
    img   = Image.new("RGBA", (width, h), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(img)

    col_w = width // n
    for i, step in enumerate(steps[:n]):
        if active_step is not None and i > active_step:
            break
            
        x0 = i * col_w + 8
        x1 = (i + 1) * col_w - 8
        # Highlight current step darker
        fill = (*accent_color, 200) if i == 0 else (25, 25, 40, 200)
        draw.rounded_rectangle([x0, 0, x1, h], radius=16, fill=fill)

        # Step number circle
        cx = x0 + 30
        draw.ellipse([cx - 18, h // 2 - 18, cx + 18, h // 2 + 18],
                     fill=(255, 255, 255, 255))
        draw.text((cx, h // 2), str(i + 1), font=gf(22),
                  fill=(*accent_color, 255), anchor="mm")

        # Step text
        sw, _ = ts(step[:22], f)
        draw.text((cx + 26, (h - ts("Ag", f)[1]) // 2),
                  step[:22], font=f, fill=(255, 255, 255, 230))

        # Arrow connector
        if i < n - 1:
            ax = x1 + 4
            draw.polygon([(ax, h // 2 - 8), (ax + 8, h // 2),
                           (ax, h // 2 + 8)],
                          fill=(*accent_color, 180))

    return img


def _render_stat_card(stat_value, stat_label, accent_color, width=600):
    """Standalone stat highlight card (bigger than fact box, more context)."""
    f_val   = gf(100)
    f_label = gf(32)

    vw, vh = ts(stat_value, f_val)
    lw, lh = ts(stat_label, f_label)
    card_w  = max(vw, lw) + 80
    card_h  = vh + lh + 60

    img  = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle([0, 0, card_w, card_h], radius=24,
                            fill=(10, 10, 18, 230))
    draw.rectangle([0, 0, card_w, 5], fill=(*accent_color, 255))

    # Value — centered
    draw.text(((card_w - vw) // 2, 20), stat_value,
              font=f_val, fill=(*accent_color, 255))
    # Label — centered, dimmer
    draw.text(((card_w - lw) // 2, 28 + vh), stat_label,
              font=f_label, fill=(200, 200, 210, 200))

def _render_flowchart_card(steps, accent_color, width=980, active_step=None):
    """Vertical architectural flowchart styled like a cloud diagram."""
    n = min(len(steps), 4)
    if n == 0: return Image.new("RGBA", (width, 100), (0, 0, 0, 0))
    
    # Image style parameters
    bg_color = (250, 250, 250, 245) # White glass
    box_outline = (0, 0, 0, 255)
    text_color = (0, 0, 0, 255)
    arrow_color = (0, 0, 0, 255)
    
    step_h = 130
    gap = 70
    total_h = n * step_h + (n-1) * gap + 80
    
    img = Image.new("RGBA", (width, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    f_step = gf(40) # slightly larger
    f_step_sm = gf(32) # smaller fallback for long labels
    
    # Background glass
    draw.rounded_rectangle([0, 0, width, total_h], radius=30, fill=bg_color, outline=accent_color, width=4)
    
    for i, step in enumerate(steps[:n]):
        if active_step is not None and i > active_step:
            break
            
        y0 = 40 + i * (step_h + gap)
        y1 = y0 + step_h
        
        box_w = width - 160
        bx0 = 80
        bx1 = bx0 + box_w
        
        step_lower = step.lower()
        
        # Decide shape and color based on keywords
        is_db = any(k in step_lower for k in ["db", "database", "storage", "cache", "data", "meta"])
        is_network = any(k in step_lower for k in ["cdn", "network", "internet", "gateway", "balancer"])
        is_client = any(k in step_lower for k in ["viewer", "creator", "user", "client", "browser", "camera"])
        
        box_fill = (218, 213, 255, 255) # Default purple (Services)
        if is_db:
            box_fill = (14, 161, 255, 255) # Blue
        elif is_network:
            box_fill = (255, 170, 0, 255) # Orange
        elif is_client:
            box_fill = (200, 200, 200, 255) # Grayish
        
        # Drop shadow
        draw.rounded_rectangle([bx0 + 5, y0 + 5, bx1 + 5, y1 + 5], radius=20, fill=(0, 0, 0, 40))
        
        if is_db:
            # Draw cylinder
            curve = 20
            # main body
            draw.rectangle([bx0, y0 + curve, bx1, y1 - curve], fill=box_fill)
            # body lines
            draw.line([(bx0, y0 + curve), (bx0, y1 - curve)], fill=box_outline, width=4)
            draw.line([(bx1, y0 + curve), (bx1, y1 - curve)], fill=box_outline, width=4)
            # bottom ellipse
            draw.ellipse([bx0, y1 - 2*curve, bx1, y1], fill=box_fill, outline=box_outline, width=4)
            # top ellipse
            draw.ellipse([bx0, y0, bx1, y0 + 2*curve], fill=box_fill, outline=box_outline, width=4)
        elif is_network:
            # Pill shape
            draw.rounded_rectangle([bx0, y0, bx1, y1], radius=step_h//2, outline=box_outline, width=4, fill=box_fill)
        else:
            # Rounded rectangle
            draw.rounded_rectangle([bx0, y0, bx1, y1], radius=20, outline=box_outline, width=4, fill=box_fill)
        
        # Step text (Centered) — auto-shrink and truncate for long labels
        t_color = text_color
        max_text_w = box_w - 40  # 20px padding on each side
        display_text = step
        font_to_use = f_step
        tw, th = ts(display_text, font_to_use)
        
        # If text overflows, try smaller font
        if tw > max_text_w:
            font_to_use = f_step_sm
            tw, th = ts(display_text, font_to_use)
        
        # If still overflows, truncate with ellipsis
        if tw > max_text_w:
            while tw > max_text_w and len(display_text) > 5:
                display_text = display_text[:-2]
                tw, th = ts(display_text + "…", font_to_use)
            display_text = display_text + "…"
            tw, th = ts(display_text, font_to_use)
        
        draw.text((bx0 + box_w//2 - tw//2, y0 + step_h//2 - th//2 - 5), display_text, font=font_to_use, fill=t_color)
        
        # Connector Arrow (Curve or straight down)
        if i < n - 1:
            ay_start = y1 + 4
            ay_end = y1 + gap - 10
            ax = bx0 + box_w // 2
            
            # Draw line
            draw.line([(ax, ay_start), (ax, ay_end)], fill=arrow_color, width=4)
            # Arrow head
            draw.polygon([(ax - 12, ay_end - 15), (ax + 12, ay_end - 15), (ax, ay_end)], fill=arrow_color)
            
    return img

def _render_slide_card(title, bullets, accent_color, is_longform=False, active_step=None):
    import math
    
    h = 700 if not is_longform else 800
    w = 980 if not is_longform else 1500
    
    # Create the base background image
    bg_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(bg_img)
    
    # Base light 'tech-glass' background
    cdraw.rectangle([0, 0, w, h], fill=(240, 242, 248, 255))
    
    # Glowing subtle blobs
    cdraw.ellipse([w*0.3, h*0.1, w*1.2, h*1.5], fill=(200, 230, 255, 100))
    cdraw.ellipse([w*0.5, h*0.3, w*1.0, h*1.0], fill=(220, 240, 255, 130))
    
    # Cyber sweeping arcs (Subtle gray-blue)
    cdraw.arc([-w*0.2, -h*0.2, w*0.8, h*1.2], 270, 360, fill=(180, 200, 220, 80), width=6)
    cdraw.arc([-w*0.1, -h*0.1, w*0.6, h*1.0], 270, 360, fill=(180, 200, 220, 50), width=3)
    
    # Hex grid on the left (Subtle)
    hex_size = 30 if not is_longform else 45
    for row in range(int(h / (hex_size*1.5)) + 1):
        for col in range(7):
            cx = 40 + col * hex_size * 1.5
            cy = 40 + row * hex_size * math.sqrt(3)
            if col % 2 == 1:
                cy += hex_size * math.sqrt(3) / 2
                
            points = []
            for i in range(6):
                angle_rad = math.pi / 3 * i
                points.append((cx + hex_size * math.cos(angle_rad), cy + hex_size * math.sin(angle_rad)))
            cdraw.polygon(points, outline=(180, 200, 255, 40), width=3)

    # Gradient for text readability (Light to darker-light)
    for x in range(w):
        alpha = int(40 * (x / w))
        cdraw.line([(x, 0), (x, h)], fill=(200, 210, 230, alpha))
            
    # Apply rounded corner mask to the entire generated background
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w, h], 40, fill=255)
    
    pil = Image.new("RGBA", (w, h), (0,0,0,0))
    pil.paste(bg_img, (0, 0), mask)
    
    # Draw border
    d = ImageDraw.Draw(pil)
    d.rounded_rectangle([0,0,w,h], 40, outline=accent_color, width=8)

    f_title = gf(65 if not is_longform else 80)
    
    # Title
    ttw, tth = ts(title, f_title)
    
    # Tech style layout (Right aligned text)
    title_x = w - ttw - 50
    title_y = 50
    # Draw text
    d.text((title_x, title_y), title, fill=(0, 0, 0, 255), font=f_title)
    
    # Divider line
    d.line([(w // 2, 140), (w - 50, 140)], fill=(*accent_color, 255), width=4)
    
    # Bullets
    f_bullet = gf(42 if not is_longform else 55)
    start_y = 180
    by = start_y
    for i, bullet in enumerate(bullets):
        if active_step is not None and i > active_step:
            break
            
        b_lines = wrap_text_to_lines(str(bullet).split(), [ts(wd, f_bullet)[0] for wd in str(bullet).split()], w // 2 - 20, f_bullet)
        for line_words in b_lines:
            line_text = " ".join(line_words)
            lw, lh = ts(line_text, f_bullet)
            line_x = w - lw - 50
            
            # Draw dot
            if line_words == b_lines[0]:
                dot_y = by + (25 if not is_longform else 35)
                d.ellipse([line_x - 30, dot_y - 10, line_x - 10, dot_y + 10], fill=(*accent_color, 255))
            
            d.text((line_x, by), line_text, font=f_bullet, fill=(5, 5, 5, 255))
            by += 55 if not is_longform else 75
        by += 25
            
    return pil

class InfographicAuditEngine:
    """Gemini Vision UI/UX Auditor. Critiques generated infographics and text layout."""
    def __init__(self, api_key):
        self.api_key = api_key
        
    def audit_infographic(self, pil_image, expected_data, infographic_type):
        if not self.api_key: return {"score": 10, "needs_refinement": False, "refined_data": expected_data}
        print(f"👁️ [INFO LOOP] Gemini Vision is auditing the '{infographic_type}' layout...")
        try:
            from google import genai
            import io, json
            
            client = genai.Client(api_key=self.api_key)
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            prompt = (
                f"You are a Senior UX/UI Designer. An automated pipeline generated this '{infographic_type}' slide.\n"
                f"Original Data Sent: {json.dumps(expected_data)}\n"
                "CRITICAL TASK:\n"
                "1. Does the text physically overflow its visual bounding box, run off-screen, or overlap other elements poorly?\n"
                "2. Is the text so dense that it is illegible in a fast-paced 40-second video format?\n"
                "If YES to either: set 'needs_refinement': true, and drastically abbreviate/summarize the text in 'refined_data' so it will fit beautifully in the next render pass.\n"
                "If PERFECT: set 'needs_refinement': false and return the original data.\n\n"
                "Return EXACTLY this JSON:\n"
                "{\n"
                "  \"score\": 1-10,\n"
                "  \"issues\": \"Short diagnosis of visual overlap/density\",\n"
                "  \"needs_refinement\": true,\n"
                "  \"refined_data\": { ... shortened data structured EXACTLY like Original Data Sent ... }\n"
                "}"
            )
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type='image/png'),
                    prompt
                ]
            )
            raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(raw)
        except Exception as e:
            print(f"⚠️ Infographic Audit Failed: {e}")
            return {"score": 10, "needs_refinement": False, "refined_data": expected_data}

def _infographic_card_clip(infographic_type, infographic_data,
                           accent_color, start_time, duration, audio_duration):
    """
    Dispatcher: reads infographic_type + infographic_data dict from a subtitle_chunk
    and returns a positioned, animated MoviePy clip (or None).

    Expected infographic_data shapes:
      definition  → {"term": "RAG", "definition": "Retrieval-Augmented Generation..."}
      stat        → {"value": "$4.6B", "label": "OpenAI 2025 Revenue"}
      comparison  → {"left_label": "GPT-4", "left_val": "128K ctx",
                     "right_label": "GPT-5", "right_val": "1M ctx"}
      process     → {"steps": ["Fetch", "Embed", "Retrieve", "Generate"]}
      flowchart   → {"steps": ["Step A", "Step B", "Step C"]}
    """
    if not infographic_data or start_time >= audio_duration:
        return None

    dur = min(duration + 0.5, audio_duration - start_time)
    itype = (infographic_type or "").lower()

    try:
        current_data = infographic_data
        max_iters = 1
        pil = None
        
        for i in range(max_iters + 1):
            if itype == "definition":
                pil = _render_definition_card(
                    current_data.get("term", ""),
                    current_data.get("definition", ""),
                    accent_color
                )
            elif itype == "comparison":
                pil = _render_comparison_card(
                    current_data.get("left_label", "A"),
                    current_data.get("left_val", "—"),
                    current_data.get("right_label", "B"),
                    current_data.get("right_val", "—"),
                    accent_color
                )
            elif itype == "process":
                pil = _render_process_steps(
                    current_data.get("steps", []),
                    accent_color
                )
            elif itype == "stat":
                pil = _render_stat_card(
                    current_data.get("value", ""),
                    current_data.get("label", ""),
                    accent_color
                )
            elif itype == "flowchart":
                pil = _render_flowchart_card(
                    current_data.get("steps", []),
                    accent_color
                )
            elif itype == "slide":
                is_longform = FRAME_W == 1920
                pil = _render_slide_card(
                    current_data.get("title", "Architecture"),
                    current_data.get("bullet_points", []),
                    accent_color,
                    is_longform=is_longform
                )
            else:
                return None
                
            # Act Loop: Evaluate with Auditor
            if isinstance(pil, Image.Image):
                auditor = InfographicAuditEngine(GEMINI_API_KEY)
                feedback = auditor.audit_infographic(pil, current_data, itype)
                
                if feedback.get("needs_refinement", False) and i < max_iters:
                    print(f"🔄 [INFO LOOP] Refining {itype}: {feedback.get('issues')}")
                    current_data = feedback.get("refined_data", current_data)
                    continue
                else:
                    if "score" in feedback:
                        print(f"⭐ [INFO LOOP] {itype} approved (Score: {feedback.get('score', 10)})")
                    break
            else:
                return None
                
    except Exception as e:
        print(f"Infographic render error ({itype}): {e}")
        return None

    num_steps = 1
    if itype == "slide":
        num_steps = len(current_data.get("bullet_points", []))
    elif itype == "process":
        num_steps = min(len(current_data.get("steps", [])), 4)
    elif itype == "flowchart":
        num_steps = min(len(current_data.get("steps", [])), 4)

    # Prevent ANY infographic from staying on screen for too long (applies to both shortform and longform)
    # We allocate ~3.5 seconds per step, but cap it at a reasonable maximum so it doesn't just sit there.
    max_dur = max(4.5, num_steps * 3.5)
    dur = min(dur, max_dur)

    def get_arr_mask(step_idx=None):
        if step_idx is None:
            return np.array(pil.convert("RGB")), np.array(pil.split()[3]).astype(float) / 255.0
            
        if itype == "slide":
            step_pil = _render_slide_card(current_data.get("title", ""), current_data.get("bullet_points", []), accent_color, is_longform=FRAME_W==1920, active_step=step_idx)
        elif itype == "process":
            step_pil = _render_process_steps(current_data.get("steps", []), accent_color, active_step=step_idx)
        elif itype == "flowchart":
            step_pil = _render_flowchart_card(current_data.get("steps", []), accent_color, active_step=step_idx)
        else:
            step_pil = pil
        return np.array(step_pil.convert("RGB")), np.array(step_pil.split()[3]).astype(float) / 255.0

    iw, ih = pil.size
    x_pos   = (FRAME_W - iw) // 2

    def opacity_fn(t):
        if t < 0.25: return t / 0.25
        if t > dur - 0.35: return max(0, (dur - t) / 0.35)
        return 1.0

    def y_pos_fn(t):
        slide = min(t / 0.25, 1.0)
        eased = 1 - (1 - slide) ** 2
        base_y = int(FRAME_H * 0.15) if FRAME_H > FRAME_W else int(FRAME_H * 0.10)
        return int(base_y + 40 * (1 - eased))

    if num_steps <= 1:
        arr, mask_arr = get_arr_mask()
        clip  = VideoClip(lambda t: arr, duration=dur)
        mclip = VideoClip(lambda t: mask_arr * opacity_fn(t), is_mask=True, duration=dur)
    else:
        step_dur = dur / max(num_steps, 1)
        step_data = [get_arr_mask(i) for i in range(num_steps)]
        
        def make_frame(t):
            idx = min(int(t / step_dur), num_steps - 1)
            return step_data[idx][0]
            
        def make_mask(t):
            idx = min(int(t / step_dur), num_steps - 1)
            return step_data[idx][1] * opacity_fn(t)
            
        clip = VideoClip(make_frame, duration=dur)
        mclip = VideoClip(make_mask, is_mask=True, duration=dur)

    return (clip.with_mask(mclip)
                .with_position(lambda t: (x_pos, y_pos_fn(t)))
                .with_start(start_time)
                .with_effects([vfx.CrossFadeOut(0.3)]))


# ── SERIES IDENTITY: Persistent Badge ──────────────────────────────────────────

def _series_badge(series_name, accent_color, duration):
    """Top-left persistent series identifier."""
    f = gf(26)
    prefix = "▶ "
    text = prefix + series_name
    tw, th = ts(text, f)
    pad_x, pad_y = 18, 10
    img = Image.new("RGBA", (tw + pad_x*2, th + pad_y*2), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, img.width, img.height],
                            radius=12, fill=(*accent_color, 220))
    draw.text((pad_x, pad_y), text, font=f, fill=(255,255,255,255))
    arr  = np.array(img.convert("RGB"))
    mask = np.array(img.split()[3]).astype(float) / 255.0
    clip = ImageClip(arr, duration=duration)
    mclip = VideoClip(lambda t: mask, is_mask=True, duration=duration)
    return clip.with_mask(mclip).with_position((24, 24))


def _next_video_tease(tease_text, accent_color, total_dur):
    """Last 3s teaser card for the next video."""
    if not tease_text or total_dur < 4:
        return None
    dur = 3.0
    start = total_dur - dur
    f_label = gf(22)
    f_tease = gf(36)
    
    lw, lh = ts("NEXT UP →", f_label)
    # Clip text to fit
    tease_text = tease_text[:40]
    tw, th = ts(tease_text, f_tease)
    box_w = max(lw, tw) + 60
    box_h = lh + th + 50

    img = Image.new("RGBA", (box_w, box_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, box_w, box_h], radius=18,
                            fill=(0, 0, 0, 200))
    draw.rectangle([0, 0, box_w, 4], fill=(*accent_color, 255))
    draw.text((30, 14), "NEXT UP →", font=f_label,
              fill=(*accent_color, 255))
    draw.text((30, 14 + lh + 10), tease_text, font=f_tease,
              fill=(255, 255, 255, 230))

    arr  = np.array(img.convert("RGB"))
    mask = np.array(img.split()[3]).astype(float) / 255.0

    def opacity(t):
        if t < 0.4: return t / 0.4
        if t > dur - 0.4: return max(0, (dur - t) / 0.4)
        return 1.0

    clip  = ImageClip(arr, duration=dur)
    mclip = VideoClip(lambda t: mask * opacity(t), is_mask=True, duration=dur)
    x_pos = (FRAME_W - box_w) // 2
    y_pos = int(FRAME_H - TITLE_BOTTOM_GAP - box_h - 20)
    return clip.with_mask(mclip).with_position((x_pos, y_pos)).with_start(start)


# ── LAYER E6: Curiosity Timer ("Wait for it...") ─────────────────────────────
def _curiosity_timer(total_dur):
    """A 'Wait for it...' countdown in the first 5-8 seconds to keep early viewers."""
    start = 2.0
    dur = min(4.0, total_dur - start - 1)
    if dur <= 0:
        return None

    f = gf(24)
    text = "⏳ Wait for it..."
    w, h = ts(text, f)
    pad_x, pad_y = 16, 8
    img = Image.new("RGBA", (w + pad_x * 2, h + pad_y * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, w + pad_x * 2, h + pad_y * 2], radius=14, fill=(255, 255, 255, 30))
    draw.text((pad_x, pad_y), text, font=f, fill=(255, 255, 255, 200))
    arr = np.array(img.convert("RGB"))
    mask_arr = np.array(img.split()[3]).astype(float) / 255.0

    def opacity_fn(t):
        # Pulsing opacity
        base = 0.7 + 0.3 * math.sin(t * 3)
        if t < 0.3:
            return base * (t / 0.3)
        elif t > dur - 0.5:
            return base * max(0, (dur - t) / 0.5)
        return base

    clip = VideoClip(lambda t: arr, duration=dur)
    mclip = VideoClip(lambda t: mask_arr * opacity_fn(t), is_mask=True, duration=dur)
    return clip.with_mask(mclip).with_position((FRAME_W - w - pad_x * 2 - 20, 170)).with_start(start)


# ── Sync checks ───────────────────────────────────────────────────────────────
def _sync_checks(chunks, audio_duration):
    issues = []
    total = sum(c["duration"] for c in chunks)
    if abs(total - audio_duration) > 0.3:
        issues.append(f"CHECK1 chunk_total={total:.2f} audio={audio_duration:.2f}")
        if chunks:
            diff = audio_duration - total
            chunks[-1]["duration"] = max(0.1, chunks[-1]["duration"] + diff)
            chunks[-1]["end"] += diff
    if TITLE_BOTTOM_GAP < 0 or TITLE_BOTTOM_GAP > FRAME_H:
        issues.append("CHECK4 title margin invalid")
    if issues:
        log_path = os.path.join(OUTPUT_DIR, "sync_log.txt")
        with open(log_path, "a") as f:
            for iss in issues:
                f.write(f"[{datetime.now()}] {iss}\n")
    return chunks


# ── MINIMALIST TECH UI COMPONENTS ─────────────────────────────────────────────

def render_header_bar(title, category, accent_color, frame_width=1080):
    """Cinematic elegant title with dark gradient backing for contrast."""
    img = Image.new('RGBA', (frame_width, FRAME_H), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    # Typography: Cinematic Serif (Increased size for visibility)
    f_title = get_cinematic_font(100, bold=True, italic=True)
    
    # Check width
    tw = draw.textlength(title, font=f_title)
    if tw > 850:
        f_title = get_cinematic_font(80, bold=True, italic=True)
        tw = draw.textlength(title, font=f_title)
        if tw > 850:
            title = title[:45] + "..."
            tw = draw.textlength(title, font=f_title)
        
    start_x = (frame_width - tw) // 2
    start_y = int(FRAME_H * 0.50) # Positioned just above the subtitles, bridging the B-Roll and Avatar
    
    # Dark gradient backing behind header text for contrast (Increased alpha)
    band_top = start_y - 40
    band_bot = start_y + 140
    for y in range(band_top, band_bot):
        dist_top = y - band_top
        dist_bot = band_bot - y
        fade = min(dist_top, dist_bot) / 40.0
        alpha = int(min(1.0, fade) * 190)
        draw.line([(0, y), (frame_width, y)], fill=(0, 0, 0, alpha))
    
    # Text shadow/glow
    for dx, dy in [(-2,0), (2,0), (0,-2), (0,2)]:
        draw.text((start_x + dx, start_y + dy), title, font=f_title, fill=(0, 0, 0, 180))
        
    draw.text((start_x, start_y), title, font=f_title, fill=(255, 255, 255, 255))
    
    return img

def render_entity_tags(entities, accent_color, frame_width=1080, on_right=False):
    """Renders small floating tags for various entities (Models, Clouds, Companies, etc.) on the side."""
    tag_h = 600
    img = Image.new('RGBA', (frame_width, tag_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    # Font (Increased from 32 for better readability)
    f_val = ImageFont.truetype('assets/fonts/Montserrat-Bold.ttf', 44)
    
    if not entities:
        return img
        
    curr_y = 10
    
    # Limit to top 6 entities to avoid cluttering the whole screen
    for ent in entities[:6]:
        val = ent.get("name", "Unknown")
        logo_path = ent.get("local_logo_path") or ent.get("local_hq_path")
        
        # Measure text
        val_w = draw.textlength(val, font=f_val)
        box_w = val_w + 40
        
        # Load logo if available
        logo_img = None
        logo_w, logo_h = 0, 0
        if logo_path and os.path.exists(logo_path):
            try:
                raw_logo = Image.open(logo_path).convert("RGBA")
                # Scale logo to fit nicely within height of 40 (box is 60)
                aspect = raw_logo.width / raw_logo.height
                logo_h = 40
                logo_w = int(logo_h * aspect)
                logo_img = raw_logo.resize((logo_w, logo_h), Image.LANCZOS)
                box_w += logo_w + 10 # Add space for logo + padding
            except Exception as e:
                print(f"Failed to load logo {logo_path}: {e}")
                
        box_h = 80 # Increased from 60 to accommodate larger font
        
        if on_right:
            start_x = frame_width - box_w - 40
        else:
            start_x = 40 # Left side padding
        
        # High-visibility Rounded box (Light/White theme)
        draw.rounded_rectangle([start_x, curr_y, start_x + box_w, curr_y + box_h], radius=12, fill=(240, 240, 240, 255), outline=(255, 255, 255, 255), width=2)
        
        # Accent line position
        acc_x = (start_x + box_w - 6) if on_right else start_x
        draw.rectangle([acc_x, curr_y + 12, acc_x + 6, curr_y + box_h - 12], fill=accent_color)
        
        # Calculate content positions
        content_x = start_x + (15 if on_right else 22)
        
        # Paste Logo
        if logo_img:
            # Calculate where to paste the logo
            logo_x = int(content_x)
            logo_y = int(curr_y + (box_h - logo_h) // 2)
            img.paste(logo_img, (logo_x, logo_y), logo_img)
            # Shift text over
            content_x += logo_w + 10
            
        # Text - High Contrast DARK on LIGHT
        draw.text((content_x, curr_y + 11), val, font=f_val, fill=(5, 5, 5, 255))
        
        curr_y += box_h + 10
        
    return img

def render_emoji_popup(emoji, frame_width=1080):
    """Renders a single large 3D-styled emoji for a popup."""
    img = Image.new('RGBA', (400, 400), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    # Using a large font for the emoji
    try:
        f = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', 200)
    except:
        f = ImageFont.load_default()
        
    # Draw centered emoji
    draw.text((200, 200), emoji, font=f, fill=(255,255,255,255), anchor="mm")
    return img

def insert_easter_egg(frame_width=1080, frame_height=1920):
    """Creates a 1-frame high-contrast tech glitch for retention hacking."""
    img = Image.new('RGBA', (frame_width, frame_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    # Neon glitch style
    draw.rectangle([0, 0, frame_width, frame_height], fill=(0, 255, 100, 40))
    f = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', 60)
    draw.text((frame_width//2, frame_height//2), "ALGORITHM DETECTED", font=f, fill=(255,255,255,255), anchor="mm")
    return img

def _apply_handheld_shake(clip):
    """Adds a slow, subtle random drift to a clip."""
    def shake(t):
        # 1-2 pixel drift to simulate handheld camera
        off_x = math.sin(t * 1.5) * 2 + math.cos(t * 0.8) * 1.5
        off_y = math.cos(t * 1.2) * 2 + math.sin(t * 0.9) * 1.5
        return (int(off_x), int(off_y))
    return clip.with_position(shake)

def _apply_intensive_glitch(frame, intensity=0.1):
    """Procedurally shifts color channels for a digital glitch effect."""
    if random.random() > intensity:
        return frame
    
    h, w, c = frame.shape
    shift = random.randint(5, 15)
    new_frame = frame.copy()
    # Shift R channel
    new_frame[:, shift:, 0] = frame[:, :w-shift, 0]
    # Shift B channel
    new_frame[:, :w-shift, 2] = frame[:, shift:, 2]
    return new_frame


def _generate_film_grain(duration, frame_width=1080, frame_height=1920):
    """Creates a procedural film grain overlay."""
    def make_frame(t):
        # Create a tiny noise texture
        noise = np.random.randint(0, 30, (frame_height//4, frame_width//4, 3), dtype='uint8')
        # Upscale it to get a 'gritty' feel
        img = Image.fromarray(noise, 'RGB').resize((frame_width, frame_height), Image.NEAREST)
        return np.array(img.convert('RGBA'))
    
    return VideoClip(make_frame, duration=duration).with_opacity(0.04)

def _generate_lens_flare(duration, frame_width=1080):
    """Procedural lens flare that drifts across the screen."""
    def make_frame(t):
        img = Image.new('RGBA', (frame_width, 800), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        # Slow drift
        x_pos = (t * 100) % (frame_width + 400) - 200
        # Draw a soft glowing gradient circle
        draw.ellipse([x_pos-150, 200, x_pos+150, 500], fill=(255, 255, 255, 30))
        return np.array(img)
    return VideoClip(make_frame, duration=duration).with_opacity(0.12).with_position(("center", 400))

def _generate_room_tone(duration):
    """Synthesizes a low-freq atmospheric 'room tone'."""
    duration = max(0.1, duration)  # Guard against zero/negative duration
    samples = np.random.normal(0, 0.01, int(44100 * duration))
    # Low pass filter (moving average)
    samples = np.convolve(samples, np.ones(100)/100, mode='same')
    
    # Save to temp and load or use a library
    temp_path = "/tmp/room_tone.wav"
    import soundfile as sf
    sf.write(temp_path, samples, 44100)
    clip = AudioFileClip(temp_path)
    # Clamp to requested duration to avoid out-of-bounds audio reads
    if clip.duration and clip.duration > duration:
        clip = clip.subclipped(0, duration)
    return clip.with_effects([afx.MultiplyVolume(0.1)])

def _sweep_clip(duration, accent_color, frame_width=1080):
    """Creates a moving highlights 'sweep' for entity tags."""
    sweep_w = 150
    def make_frame(t):
        # Moving diagonal gradient
        img = Image.new('RGBA', (frame_width, 400), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        progress = (t / duration) * (frame_width + sweep_w*2)
        x_pos = progress - sweep_w
        
        # Draw a slanted semi-transparent white beam
        draw.polygon([
            (x_pos, 0), (x_pos + sweep_w, 0), 
            (x_pos + sweep_w - 50, 400), (x_pos - 50, 400)
        ], fill=(255, 255, 255, 30))
        return np.array(img)
    
    return VideoClip(make_frame, duration=duration).with_opacity(0.4)



# ── LAYER 16: Article Screenshot (New Layer) ──────────────────────────────────
def easeInOutQuad(t):
    t = max(0.0, min(1.0, t))
    return 2*t*t if t < 0.5 else 1 - pow(-2*t + 2, 2) / 2

def _article_screenshot_clip(screenshot_path, duration):
    """
    Shows the source article as a full-screen backdrop with a premium Ken Burns effect.
    Ensures visibility every 10s for 7s throughout the video.
    """
    print(f"🔍 DEBUG: _article_screenshot_clip called with path: {screenshot_path}, dur: {duration}")
    if not screenshot_path or not os.path.exists(screenshot_path):
        print(f"⚠️ DEBUG: Screenshot path missing or invalid: {screenshot_path}")
        return []
    try:
        # Load and prepare the canvas once
        img = Image.open(screenshot_path).convert("RGB")
        target_h, target_w = FRAME_H, FRAME_W
        canvas = _prepare_screenshot_canvas(img, target_w, target_h)
        
        # Prepare RGB and Mask arrays for explicit alpha handling in MoviePy
        arr_rgba = np.array(canvas.convert("RGBA"))
        arr_rgb = arr_rgba[:, :, :3]
        arr_mask = (arr_rgba[:, :, 3] / 255.0).astype(float)
        
        clips = []
        interval = 10.0
        display_dur = 7.0
        
        # Start at 0.0 to ensure it appears immediately
        current_start = 0.0
        while current_start < duration:
            # Clamp duration for the last clip if it would exceed total video length
            current_dur = min(display_dur, duration - current_start)
            if current_dur < 1.0:
                break
            
            # Create a static ImageClip from the prepared canvas array
            # ImageClip(arr_rgb) is faster than loading from disk every time in a loop
            clip = ImageClip(arr_rgb, duration=current_dur)
            
            # --- PREMIUM KEN BURNS EFFECT (Zoom + Pan) ---
            zoom_amt = 0.35 # Balanced zoom
            clip = clip.resized(lambda t, cd=current_dur: 1.0 + zoom_amt * easeInOutQuad(t / cd))
            
            def pan_fn(t, cd=current_dur):
                prog = easeInOutQuad(t / cd)
                off_x = -int(90 * prog) 
                off_y = -int(60 * prog)
                return (off_x, off_y)

            clip = clip.with_position(pan_fn).with_start(current_start)
            clip = clip.with_effects([vfx.CrossFadeIn(0.6), vfx.CrossFadeOut(0.6)])
            clips.append(clip)
            
            current_start += interval
            
        print(f"🎬 Generated {len(clips)} article screenshot intervals for path {os.path.basename(screenshot_path)}.")
        return clips
    except Exception as e:
        print(f"Article screenshot clip error: {e}")
        import traceback
        traceback.print_exc()
        return []

def _evidence_screenshot_clip(evidence_path, duration):
    """
    Shows a secondary 'Evidence' or 'Use Case' screenshot during the analytical section.
    """
    if not evidence_path or not os.path.exists(evidence_path):
        return []
    try:
        img = Image.open(evidence_path).convert("RGB")
        target_h, target_w = FRAME_H, FRAME_W
        canvas = _prepare_screenshot_canvas(img, target_w, target_h)
        
        arr_rgba = np.array(canvas.convert("RGBA"))
        arr_rgb = arr_rgba[:, :, :3]
        arr_mask = (arr_rgba[:, :, 3] / 255.0).astype(float)
        
        start = 28.0 
        dur = min(6.0, duration - start - 5.0)
        
        if dur > 1.0:
            clip = ImageClip(arr_rgb, duration=dur)
            mclip = VideoClip(lambda t: arr_mask, is_mask=True, duration=dur)
            clip = clip.with_mask(mclip)
            
            # Subtle zoom only for evidence
            clip = clip.resized(lambda t, d=dur: 1.0 + 0.12 * easeInOutQuad(t / d))
            clip = clip.with_position("center").with_start(start)
            clip = clip.with_effects([vfx.CrossFadeIn(0.6), vfx.CrossFadeOut(0.6)])
            return [clip]
            
        return []
    except Exception as e:
        print(f"Evidence screenshot clip error: {e}")
        return []


def _ai_disclosure_overlay(duration):
    """
    Subtle 2026 AI disclosure overlay. 
    Moved to a top-left unobtrusive position to avoid clashing with captions.
    """
    w, h = 420, 45 # Significantly smaller than previous version
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Minimalist translucent capsule
    draw.rounded_rectangle([0, 0, w, h], radius=12, fill=(0, 0, 0, 100))
    
    txt = "AI-ENHANCED VISUALS/AUDIO"
    font = gf(18) # Tiny, discreet font
    draw.text((w//2, h//2), txt, font=font, fill=(255, 255, 255, 120), anchor="mm")
    
    arr = np.array(img.convert("RGB"))
    mask = np.array(img.split()[3]).astype(float) / 255.0
    
    clip_dur = 4.0 # Brief appearance
    clip = ImageClip(arr, duration=clip_dur)
    mclip = VideoClip(lambda t: mask, is_mask=True, duration=clip_dur)
    
    # Positioned top-left, away from captions and logo
    return clip.with_mask(mclip).with_position((40, 40)).with_start(0.5).with_effects([vfx.CrossFadeIn(0.4), vfx.CrossFadeOut(0.6)])


def _brand_watermark(duration):
    """
    Fixed Brand Identity to prevent 'Reused Content' flags.
    """
    w, h = 180, 80
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Neon Box for VJ
    draw.rectangle([0, 0, w, h], outline=(255, 64, 64, 120), width=4)
    draw.text((w//2, h//2), "VJ AI NEWS", font=gf(22), fill=(255, 255, 255, 120), anchor="mm")
    
    arr = np.array(img.convert("RGB"))
    mask = np.array(img.split()[3]).astype(float) / 255.0
    
    clip = ImageClip(arr, duration=duration)
    mclip = VideoClip(lambda t: mask, is_mask=True, duration=duration)
    
    return clip.with_mask(mclip).with_position((FRAME_W - w - 40, FRAME_H - h - 300)).with_start(0).with_opacity(0.6)

def _intro_clip(duration, accent_color):
    """Create a brief intro segment showing the animated logo on a dark background."""
    logo = _animated_logo(duration)
    bg = ColorClip(size=(FRAME_W, FRAME_H), color=(0, 0, 0), duration=duration)
    if logo is None:
        return bg
    # place logo near top-center
    logo = logo.with_position(("center", int(FRAME_H * 0.15)))
    return CompositeVideoClip([bg, logo], size=(FRAME_W, FRAME_H)).with_duration(duration)

def _outro_clip(duration, accent_color):
    """Create a brief outro segment showing the telegram CTA."""
    bg = ColorClip(size=(FRAME_W, FRAME_H), color=(10, 10, 15), duration=duration)
    return bg.with_duration(duration)

def apply_pattern_interrupts(frame_np, t, cues):
    """Applies visual disruption effects based on retention cues (glitches, zooms, shakes)."""
    if not cues:
        return frame_np
        
    h, w, c = frame_np.shape
    for cue in cues:
        start_t = float(cue.get("timestamp", 0))
        duration = 0.4 # Fast interrupt for high retention
        
        if start_t <= t <= (start_t + duration):
            effect = cue.get("effect", "").lower()
            
            if "glitch" in effect:
                # RGB Channel Splitting / Displacement
                shift = 12
                # Green channel shift
                frame_np[:, shift:, 1] = frame_np[:, :-shift, 1]
                # Red channel shift
                frame_np[shift:, :, 0] = frame_np[:-shift, :, 0]
                
            elif "zoom" in effect:
                # 10% Zoom focus
                zoom_factor = 1.10
                nh, nw = int(h / zoom_factor), int(w / zoom_factor)
                dy, dx = (h - nh)//2, (w - nw)//2
                cropped = frame_np[dy:dy+nh, dx:dx+nw]
                frame_np = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
                
            elif "shake" in effect:
                # Rapid displacement
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                frame_np = cv2.warpAffine(frame_np, M, (w, h))
                
            break # Apply only one effect at a time
            
    return frame_np

# ── RETENTION BOOSTER OVERLAYS ────────────────────────────────────────────────

def _render_hook_overlay(hook_text, width, height, timestamp):
    """Bold hook text that flashes in the first 1.5 seconds with fade-in."""
    try:
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Fade in from 0-0.3s, hold 0.3-1.0s, fade out 1.0-1.5s
        if timestamp < 0.3:
            alpha = int(255 * (timestamp / 0.3))
        elif timestamp < 1.0:
            alpha = 255
        else:
            alpha = int(255 * (1.0 - (timestamp - 1.0) / 0.5))
        alpha = max(0, min(255, alpha))
        
        # Semi-transparent dark backdrop
        backdrop_alpha = int(alpha * 0.6)
        draw.rectangle([(0, height // 2 - 120), (width, height // 2 + 120)], 
                       fill=(0, 0, 0, backdrop_alpha))
        
        # Massive Bold hook text
        try:
            font = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', 110)
        except:
            font = ImageFont.load_default()
        
        # Truncate to fit
        text = hook_text.upper()[:40]
        bb = draw.textbbox((0, 0), text, font=font)
        tw = bb[2] - bb[0]
        x = (width - tw) // 2
        y = height // 2 - 50
        
        # Neon Accent Glow for Hook
        glow_draw = ImageDraw.Draw(img)
        glow_draw.text((x, y), text, font=font, fill=(0, 229, 255, 120), stroke_width=12, stroke_fill=(0, 229, 255, 60))

        # Main white text
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255),
                  stroke_width=5, stroke_fill=(0, 0, 0, 255))
        
        return img
    except Exception as e:
        print(f"Hook overlay error: {e}")
        return None


def _render_comment_bait(comment_text, width, height):
    """Comment engagement prompt at the bottom in the last 3 seconds."""
    try:
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 2026 Glassmorphism Bubble
        bubble_y = height - 280
        bubble_h = 100
        padding = 60
        
        try:
            font = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', 38)
        except:
            font = ImageFont.load_default()
        
        text = f"💬 {comment_text}"
        bb = draw.textbbox((0, 0), text, font=font)
        tw = bb[2] - bb[0]
        
        # Bubble Background
        rect = [(width - tw) // 2 - padding, bubble_y, (width + tw) // 2 + padding, bubble_y + bubble_h]
        draw.rounded_rectangle(rect, radius=20, fill=(0, 0, 0, 220), outline=(0, 229, 255, 180), width=3)
        
        # Text
        draw.text(((width - tw) // 2, bubble_y + 25), text, font=font, fill=(0, 229, 255, 255))
        
        return img
    except Exception as e:
        print(f"Comment bait error: {e}")
        return None

def _render_animated_stat(stat_text, width, height, progress_ratio, accent_color):
    """Animated stat counting up (0.0 to 1.0 ratio)."""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Parse number from stat_text (e.g. "$4.6B" or "98%")
    # Simple heuristic: find numbers and symbols
    parts = re.findall(r'(\$|[\d\.]+|[BMK%]+)', stat_text)
    
    display_text = ""
    for p in parts:
        if re.match(r'[\d\.]+', p):
            val = float(p)
            # Count up number
            cur_val = val * progress_ratio
            if "." in p:
                display_text += f"{cur_val:.1f}"
            else:
                display_text += f"{int(cur_val)}"
        else:
            display_text += p
            
    try:
        font = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', 160)
    except:
        font = ImageFont.load_default()
    
    bb = draw.textbbox((0, 0), display_text, font=font)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    
    x = (width - tw) // 2
    y = (height - th) // 2 - 100
    
    # Shadow
    draw.text((x+10, y+10), display_text, font=font, fill=(0,0,0,150))
    # Main with neon accent
    draw.text((x, y), display_text, font=font, fill=accent_color, stroke_width=6, stroke_fill=(255,255,255,255))
    
    return img

def build_transparency_watermark(width, height):
    """Creates a subtle, high-end transparency watermark for 2026 compliance."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    
    # Text: "AI HUMAN-IN-THE-LOOP PRODUCTION"
    text = "AI HUMAN-IN-THE-LOOP PRODUCTION"
    font = gf(24) # Small, elite typography
    tw, th = ts(text, font)
    
    # Position: Very Top Right corner
    x, y = width - tw - 40, 40
    
    # Glassmorphism backing
    rect = [x - 15, y - 8, x + tw + 15, y + th + 8]
    d.rounded_rectangle(rect, radius=8, fill=(0, 0, 0, 80), outline=(255, 255, 255, 40), width=1)
    
    # Semi-transparent text
    d.text((x, y), text, font=font, fill=(255, 255, 255, 140))
    
    return img

def composite_frame(background_frame, timestamp, header_img, subtitle_img, transparency_img=None):
    """Clean talking-head composite: header + subtitles only."""
    frame = Image.fromarray(background_frame).convert('RGBA')
    
    # 1. Header at top
    frame.alpha_composite(header_img, dest=(0, 0))
    
    # 2. Transparency Watermark (2026 Compliance)
    if transparency_img is not None:
        frame.alpha_composite(transparency_img, dest=(0, 0))
    
    # 3. Subtitles
    if subtitle_img is not None:
        frame.alpha_composite(subtitle_img, dest=(0, 0))
    
    return np.array(frame.convert('RGB'))

def verify_text_visibility(frame_array, zone_name, y_start, y_end):
    """Validates that text is readable (high contrast)."""
    region = frame_array[y_start:y_end, 50:1030]
    # Check for presence of bright pixels (text) and dark pixels (obsidian bg)
    bright = np.sum(region > 160)
    dark = np.sum(region < 50)
    
    if bright < 200:
        print(f"⚠️ {zone_name}: WARNING - Low text density detected.")
    if dark < 200:
        print(f"⚠️ {zone_name}: WARNING - Low background contrast.")
def wrap_text_to_lines(words, word_widths, max_width, font):
    lines = []
    current_line = []
    current_w = 0
    space_w = 18
    for word, w in zip(words, word_widths):
        # Standard wrapping based on width
        if not current_line or (current_w + w <= max_width):
            current_line.append(word)
            current_w += w + space_w
        else:
            lines.append(current_line)
            current_line = [word]
            current_w = w + space_w
    if current_line:
        lines.append(current_line)
    return lines

def render_subtitle_frame(word_data, bg_frame=None, accent_color=(255,214,0), frame_width=1080, frame_height=1920, y_shift=0):
    """Viral 'High Energy' captions: Large tilted words with pop sounds."""
    img = Image.new('RGBA', (frame_width, frame_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    scale_ratio = frame_width / 1080.0 if frame_width < frame_height else frame_width / 1920.0
    base_size = int(58 * scale_ratio) # Reverted to original as requested
    
    f_main = gf(base_size, bold=True)
    
    words = [wd["word"] for wd in word_data]
    word_widths = []
    fake_draw = ImageDraw.Draw(Image.new("RGBA", (1,1)))
    
    for i, wd in enumerate(word_data):
        word_widths.append(fake_draw.textbbox((0,0), words[i], font=f_main)[2] - fake_draw.textbbox((0,0), words[i], font=f_main)[0])
    
    max_sub_width = int(frame_width * 0.80) # More narrow for punchy center focus
    lines = wrap_text_to_lines(words, word_widths, max_sub_width, f_main)
    
    line_h = int(90 * scale_ratio) # Reverted to original
    
    # Position: LOWER THIRD (60% Height)
    start_y = int(frame_height * 0.60) - (len(lines) * line_h // 2) + y_shift
    
    # Calculate dimensions for the unified background block
    max_line_w = 0
    temp_idx = 0
    for line in lines:
        line_w = sum(word_widths[temp_idx:temp_idx+len(line)]) + 22 * (len(line)-1)
        if line_w > max_line_w:
            max_line_w = line_w
        temp_idx += len(line)
    
    # Tightened Obsidian Background Block (Improved Visibility for B-Roll)
    bg_pad_x, bg_pad_y = 30, 18
    block_x1 = (frame_width - max_line_w) // 2 - bg_pad_x
    block_x2 = (frame_width + max_line_w) // 2 + bg_pad_x
    block_y1 = start_y - bg_pad_y
    block_y2 = start_y + len(lines) * line_h - (line_h - base_size) + bg_pad_y
    
    draw.rounded_rectangle(
        [block_x1, block_y1, block_x2, block_y2], 
        radius=12, 
        fill=(0, 0, 0, 215) # Increased opacity from 160 for guaranteed contrast
    )

    word_idx = 0
    for i, line in enumerate(lines):
        line_y = start_y + i * line_h
        line_w = sum(word_widths[word_idx:word_idx+len(line)]) + 22 * (len(line)-1)
        cur_x = (frame_width - line_w) // 2

        for word_text in line:
            wd = word_data[word_idx]
            is_active = wd["is_active"]
            
            if is_active:
                c_fill = (204, 255, 0, 255) # Electric Yellow
                f_word = gf(int(base_size * 1.4), bold=True)
                w_w, w_h = ts(word_text, f_word)
                word_img = Image.new("RGBA", (w_w + 60, w_h + 60), (0,0,0,0))
                word_draw = ImageDraw.Draw(word_img)
                
                # High-Contrast Edge: 5px black stroke simulation (Stronger for visibility)
                stroke = 5
                for dx in range(-stroke, stroke+1):
                    for dy in range(-stroke, stroke+1):
                        if dx*dx + dy*dy <= stroke*stroke:
                            word_draw.text((30+dx, 30+dy), word_text, font=f_word, fill=(0,0,0,255))
                
                # 3D Depth: 4px offset shadow
                word_draw.text((34, 34), word_text, font=f_word, fill=(0,0,0,180))
                # Main Text
                word_draw.text((30, 30), word_text, font=f_word, fill=c_fill)
                
                tilt = 0 # Reverted to no tilt as per reference
                rotated = word_img.rotate(tilt, resample=Image.BICUBIC, expand=True)
                
                orig_w = word_widths[word_idx]
                target_x = cur_x - (rotated.width - orig_w)//2
                target_y = line_y - (rotated.height - base_size)//2 + 2
                img.alpha_composite(rotated, (target_x, target_y))
            else:
                c_fill = (255, 255, 255, 255)
                # Inactive words get a 3px stroke and offset shadow for readability
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        draw.text((cur_x+dx, line_y+2+dy), word_text, font=f_main, fill=(0,0,0,255))
                draw.text((cur_x+3, line_y+5), word_text, font=f_main, fill=(0,0,0,160))
                draw.text((cur_x, line_y + 2), word_text, font=f_main, fill=c_fill)
            
            cur_x += word_widths[word_idx] + 22
            word_idx += 1
            
    return img

def _generate_lipsync_video(audio_path):

    from lip_sync import generate_lip_sync, get_available_engine

    face_path = os.path.join(ASSETS_DIR, "Firefly_video_final.mp4")
    if not os.path.exists(face_path):
        print("Firefly_video_final.mp4 not found in assets. Skipping lip sync.")
        return None

    output_path = os.path.join(OUTPUT_DIR, "temp_lipsync.mp4")
    
    # If Kaggle was enabled but failed to return a lipsync (e.g., crashed), do NOT fall back to local MPS/CPU 
    # to avoid extremely long 30+ min processing times.
    has_kaggle = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
    use_local_only = os.environ.get("USE_LOCAL_ONLY") == "true"
    
    if has_kaggle and not use_local_only:
        print("⚠️ Kaggle GPU was enabled but no lip-sync received. Skipping slow local fallback.")
        return None

    engine = get_available_engine()
    print(f"🎭 Lip-sync engine: {engine or 'None available'}")

    result = generate_lip_sync(
        face_path=face_path,
        audio_path=audio_path,
        output_path=output_path,
    )

    if result and os.path.exists(result):
        print(f"🎭 Lip-sync successful: {result}")
        return result

    print("🎭 Lip-sync generation failed or unavailable.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# VISUAL AUDIT ENGINE — Agentic Observe & Critique for Video Rendering
# ══════════════════════════════════════════════════════════════════════════════

class VisualAuditEngine:
    """Gemini Vision auditor for rendered videos.
    
    Extracts sample frames → sends to Gemini Vision → returns structured
    refinement commands that feed back into the render loop.
    
    Agentic Loop: Plan → Act → **Observe → Critique** → Refine
    """
    
    # Refinable parameters and their safe ranges
    PARAM_RANGES = {
        "avatar_scale_mult": (0.6, 1.4),
        "subtitle_y_shift": (-80, 80),
    }

    def __init__(self, api_key):
        self.api_key = api_key

    def _extract_frames(self, video_path, num_frames=4):
        """Extract sample frames at key timestamps from the rendered video."""
        frames = []
        try:
            clip = VideoFileClip(video_path)
            dur = clip.duration
            if dur is None or dur <= 0:
                return frames
            
            # Sample at 10%, 30%, 60%, 90% of the video
            fractions = [0.10, 0.30, 0.60, 0.90]
            for frac in fractions[:num_frames]:
                t = dur * frac
                frame = clip.get_frame(t)
                pil_img = Image.fromarray(frame)
                frames.append((frac, pil_img))
            clip.close()
        except Exception as e:
            print(f"⚠️ [AUDIT] Frame extraction failed: {e}")
        return frames

    def _frames_to_bytes(self, frames):
        """Convert PIL frames to PNG bytes for Gemini Vision."""
        parts = []
        for frac, pil_img in frames:
            # Downscale for API efficiency (max 720px wide)
            w, h = pil_img.size
            if w > 720:
                scale = 720 / w
                pil_img = pil_img.resize((720, int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG', quality=80)
            parts.append({
                'mime_type': 'image/jpeg',
                'data': buf.getvalue()
            })
        return parts

    def _clamp_refinements(self, refinements):
        """Clamp refinement values to safe ranges so Gemini can't break the render."""
        clamped = {}
        for key, (lo, hi) in self.PARAM_RANGES.items():
            if key in refinements:
                try:
                    val = float(refinements[key])
                    clamped[key] = max(lo, min(hi, val))
                except (ValueError, TypeError):
                    pass
        return clamped

    def audit(self, video_path, script_text=""):
        """Run Gemini Vision audit on the rendered video.
        
        Returns:
            dict with keys: score (float), issues (str), refinement_commands (dict)
            Returns None on failure.
        """
        if not self.api_key:
            print("⚠️ [AUDIT] No API key, skipping visual audit.")
            return None

        frames = self._extract_frames(video_path)
        if not frames:
            print("⚠️ [AUDIT] No frames extracted, skipping audit.")
            return None

        print(f"👁️ [AUDIT] Sending {len(frames)} sample frames to Gemini Vision...")

        try:
            from google import genai

            client = genai.Client(api_key=self.api_key)
            image_parts = self._frames_to_bytes(frames)

            param_desc = json.dumps({k: f"range {v}" for k, v in self.PARAM_RANGES.items()})

            prompt = (
                "You are a Senior Video Production QA Engineer reviewing a YouTube Shorts / tech news video.\n"
                f"Script Summary (for context): {script_text[:300]}...\n\n"
                f"You are shown {len(frames)} frames sampled at 10%, 30%, 60%, 90% of the video.\n\n"
                "CRITICAL EVALUATION CRITERIA:\n"
                "1. SUBTITLE READABILITY: Are subtitles clearly visible, properly positioned, not overlapping with the avatar or other UI elements?\n"
                "2. AVATAR POSITIONING: Is the avatar (talking head PiP) appropriately sized and positioned without blocking important content?\n"
                "3. TEXT OVERLAP: Do any text elements (title, header, captions, CTA cards) overlap each other or get cut off?\n"
                "4. VISUAL HIERARCHY: Is the overall layout clean, professional, and suitable for a fast-paced tech news short?\n"
                "5. CONTRAST & LEGIBILITY: Can all text be read against the background at a glance?\n\n"
                f"AVAILABLE REFINEMENT PARAMETERS: {param_desc}\n"
                "- avatar_scale_mult: Scale multiplier for the avatar PiP (1.0 = current size, 0.8 = smaller, 1.2 = bigger)\n"
                "- subtitle_y_shift: Vertical pixel shift for subtitles (negative = move up, positive = move down)\n\n"
                "Return EXACTLY this JSON (no markdown fencing):\n"
                '{\n'
                '  "score": <1.0 to 10.0>,\n'
                '  "issues": "<concise 1-2 sentence diagnosis>",\n'
                '  "refinement_commands": {\n'
                '    "avatar_scale_mult": <float or omit if fine>,\n'
                '    "subtitle_y_shift": <int or omit if fine>\n'
                '  }\n'
                '}\n'
                'If everything looks great (score >= 8.5), return an empty refinement_commands: {}'
            )

            # Build content: images first, then prompt text
            contents = image_parts + [prompt]

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents
            )

            raw = response.text.strip()
            # Strip markdown code fence if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]  # Remove first line (```json)
                raw = raw.rsplit("```", 1)[0]  # Remove trailing ```
            raw = raw.strip()

            result = json.loads(raw)

            # Sanitize: clamp refinement values to safe ranges
            if "refinement_commands" in result:
                result["refinement_commands"] = self._clamp_refinements(result["refinement_commands"])

            score = result.get("score", "?")
            issues = result.get("issues", "No issues reported")
            print(f"👁️ [AUDIT] Score: {score}/10 | {issues}")
            return result

        except json.JSONDecodeError as e:
            print(f"⚠️ [AUDIT] Failed to parse Gemini response: {e}")
            return None
        except Exception as e:
            print(f"⚠️ [AUDIT] Gemini Vision audit failed: {e}")
            return None


def create_video(audio_path, script_json, chunks, output_path=None):
    """
    Agentic Loop for Video: Plan -> Act -> Observe -> Critique -> Refine
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    best_video_path = None
    
    # 0. PLAN: Inherit style from script_json
    iterations = 0
    max_iters = 1
    
    # Dynamic refinement parameters
    dynamic_params = {
        "avatar_scale_mult": 1.0,
        "subtitle_y_shift": 0
    }
    
    while iterations <= max_iters:
        print(f"🎬 [VIDEO LOOP] Act: Rendering iteration {iterations}...")
        
        # Original create_video logic (condensed for the loop)
        video_path = _create_video_internal(audio_path, script_json, chunks, output_path, dynamic_params)
        
        if not video_path: break
        
        # 1. OBSERVE & CRITIQUE
        if api_key and iterations < max_iters:
            try:
                auditor = VisualAuditEngine(api_key)
                feedback = auditor.audit(video_path, script_json.get("script", ""))
                
                if feedback and feedback.get("score", 0) < 8.5:
                    print(f"🔄 [VIDEO LOOP] Quality: {feedback.get('score')}/10. Issues: {feedback.get('issues')}")
                    # 2. REFINE
                    refinements = feedback.get("refinement_commands", {})
                    if refinements:
                        dynamic_params.update(refinements)
                        iterations += 1
                        continue
                else:
                    print(f"⭐ [VIDEO LOOP] Visual Quality Score: {feedback.get('score', 'N/A')}/10. Approved.")
            except Exception as e:
                print(f"⚠️ [VIDEO LOOP] Visual audit failed (non-fatal): {e}")
        
        best_video_path = video_path
        break
        
    return best_video_path

def _create_video_internal(audio_path, script_json, chunks, output_path=None, dynamic_params=None):
    """The original heavy-lifting render logic."""
    if dynamic_params is None: dynamic_params = {}
    
    avatar_scale_mult = dynamic_params.get("avatar_scale_mult", 1.0)
    subtitle_y_shift = dynamic_params.get("subtitle_y_shift", 0)

    is_longform = "Slot C" in script_json.get("slot", "")
    set_resolutions(is_longform)
    
    today = datetime.now().strftime("%Y-%m-%d")
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, f"video_{today}.mp4")

    # ── AUDIO VALIDATION ─────────────────────────────────────────────────
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        print(f"ERROR: Audio file is missing or empty: {audio_path}")
        return None

    audio          = AudioFileClip(audio_path)
    audio_duration = audio.duration

    if audio_duration is None or audio_duration <= 0:
        print(f"ERROR: Audio clip has no valid duration: {audio_path}")
        return None
    print(f"Audio validated: {audio_duration:.2f}s from {os.path.basename(audio_path)}")

    if not chunks:
        print("ERROR: no chunks")
        return None

    chunks = _sync_checks(chunks, audio_duration)

    # Meta
    title          = script_json.get("title", "Tech News")
    color_theme    = script_json.get("color_theme", {})
    accent_hex     = color_theme.get("accent", "#FFD700").lstrip("#")
    accent_color   = tuple(int(accent_hex[i:i+2], 16) for i in (0, 2, 4))
    sub_category   = script_json.get("sub_category", "AI")
    emoji          = script_json.get("relevant_emoji", "")
    key_stat       = script_json.get("key_stat", "")
    key_stat_ts    = float(script_json.get("key_stat_timestamp", 0))
    shock_ts       = float(script_json.get("shocking_moment_timestamp", 0))

    companies = script_json.get("companies_mentioned", [])
    tools = script_json.get("tools_mentioned", [])
    key_entities = script_json.get("key_entities", [])
    if not key_entities:
        for c in companies: key_entities.append({"name": c, "type": "COMPANY"})
        for t in tools: key_entities.append({"name": t, "type": "TOOL"})

    # ── FULL SCREEN BACKGROUND ───────────────────────────────────────────
    print("Preparing full-screen background...")
    visual_paths = []
    for chunk in chunks:
        vpath = chunk.get("visual_path")
        if vpath and os.path.exists(vpath) and vpath not in visual_paths:
            visual_paths.append(vpath)

    bg_layer_clips = []
    particle_clips = []
    logo_clips = []
    fact_clips = []
    burst_clips = []
    reminder_clips = []
    
    if not visual_paths:
        bg_layer_clips.append(ColorClip(size=(FRAME_W, FRAME_H), color=(10, 10, 15), duration=audio_duration))
    else:
        crossfade = 0.4 # Increased for smoother motion transitions
        num_clips = len(visual_paths)
        clip_dur = (audio_duration + (num_clips - 1) * crossfade) / num_clips
        current_start = 0.0
        
        for i, vp in enumerate(visual_paths):
            try:
                if vp.endswith(".mp4"):
                    c_clip = VideoFileClip(vp)
                    if c_clip.duration < clip_dur:
                        c_clip = c_clip.with_effects([vfx.Loop(duration=clip_dur)])
                    else:
                        c_clip = c_clip.subclipped(0, clip_dur)
                else:
                    c_clip = ImageClip(vp).with_duration(clip_dur)
                
                # Standard Resize & Crop to 9:16
                w, h = c_clip.size
                target_h = int(w * 16 / 9)
                if target_h <= h:
                    y1 = (h - target_h) // 2
                    c_clip = c_clip.cropped(x1=0, y1=y1, x2=w, y2=y1 + target_h)
                else:
                    target_w = int(h * 9 / 16)
                    x1 = (w - target_w) // 2
                    c_clip = c_clip.cropped(x1=x1, y1=0, x2=w, y2=h)

                c_clip = c_clip.resized((FRAME_W, FRAME_H))
                
                # ── CREATIVE TRANSITIONS ──
                if i > 0:
                    trans_type = random.choice(["zoom", "slide_r", "slide_l", "slide_t", "glitch"])
                    
                    if trans_type == "zoom":
                        # Dramatic Zoom In
                        c_clip = c_clip.with_effects([vfx.CrossFadeIn(crossfade)])
                        c_clip = c_clip.resized(lambda t: 1.3 - (0.3 * min(1, t / crossfade)) if t < crossfade else 1.0)
                    
                    elif "slide" in trans_type:
                        # Smooth Directional Slide
                        c_clip = c_clip.with_effects([vfx.CrossFadeIn(crossfade * 0.5)])
                        def slide_pos(t):
                            if t > crossfade: return ("center", "center")
                            prog = t / crossfade
                            # Exponential ease out for 'smooth' feel
                            prog = 1 - (1 - prog)**3 
                            if trans_type == "slide_r": return (int(FRAME_W * (1 - prog)), "center")
                            if trans_type == "slide_l": return (int(-FRAME_W * (1 - prog)), "center")
                            if trans_type == "slide_t": return ("center", int(-FRAME_H * (1 - prog)))
                            return ("center", "center")
                        c_clip = c_clip.with_position(slide_pos)
                    
                    elif trans_type == "glitch":
                        # Fast Glitch Cut
                        c_clip = c_clip.with_effects([vfx.CrossFadeIn(0.1)])
                        # (Glitch logic is handled per-frame in make_final_frame based on start time)
                        pass

                    # Impact Flash Sync
                    flash = ColorClip(size=(FRAME_W, FRAME_H), color=(255, 255, 255), duration=0.2).with_opacity(0.6)
                    flash = flash.with_start(current_start).with_effects([vfx.CrossFadeOut(0.15)])
                    logo_clips.append(flash)

                # ── CONTINUOUS MOTION ──
                scale_factor = 1.0 + random.uniform(0.18, 0.25)
                # Capture scale_factor and clip_dur by value to avoid lambda-in-loop bugs
                c_clip = c_clip.resized(lambda t, sf=scale_factor, cd=clip_dur: 1.0 + (sf - 1.0) * (t / cd))
                c_clip = _apply_handheld_shake(c_clip)
                c_clip = c_clip.with_start(current_start)
                
                bg_layer_clips.append(c_clip)
                current_start += (clip_dur - crossfade)
            except Exception as e:
                print(f"Failed to load background img {vp}: {e}")

    # ── AVATAR VIDEO PiP ──────────────────────────────────────────────────
    print("Preparing Dimension Avatar PiP...")
    lipsync_path = script_json.get("kaggle_lipsync_path")
    if not lipsync_path or not os.path.exists(lipsync_path):
        lipsync_path = _generate_lipsync_video(audio_path)
        
    firefly_path = os.path.join(ASSETS_DIR, "Firefly_video_final.mp4")
    avatar_video_path = lipsync_path if lipsync_path else firefly_path

    avatar_pip = None
    if os.path.exists(avatar_video_path):
        vid_clip = VideoFileClip(avatar_video_path)
        if vid_clip.duration < audio_duration:
            vid_clip = vid_clip.with_effects([vfx.Loop(duration=audio_duration)])
        else:
            vid_clip = vid_clip.subclipped(0, audio_duration)

        w, h = vid_clip.size
        
        # Crop to portrait aspect ratio (9:16) to focus on the character and fit naturally in Shorts
        target_aspect = 9 / 16
        if w/h > target_aspect:
            new_w = int(h * target_aspect)
            x1 = (w - new_w) // 2
            vid_clip = vid_clip.cropped(x1=x1, y1=0, x2=x1+new_w, y2=h)
        else:
            new_h = int(w / target_aspect)
            y1 = (h - new_h) // 2
            vid_clip = vid_clip.cropped(x1=0, y1=y1, x2=w, y2=y1+new_h)
            
        w, h = vid_clip.size
        
        # Make the avatar occupy exactly the bottom 40% of the screen
        height_pip = int(FRAME_H * 0.40)
        width_pip = int(height_pip * (w / h))
        
        # Apply refinements from dynamic_params
        cur_w = max(1, int(width_pip * avatar_scale_mult))
        cur_h = max(1, int(height_pip * avatar_scale_mult))
        avatar_clip = vid_clip.resized((cur_w, cur_h)).without_audio()
        
        try:
            from rembg import remove, new_session
            print("Using rembg for clean AI background removal...")
            # u2net_human_seg is optimized for human boundaries (head and body extraction)
            _rembg_session = new_session("u2net_human_seg")
            _rembg_cache = {"t": -1.0, "rgba": None}
            
            def get_rgba_frame(t, get_frame):
                if abs(t - _rembg_cache["t"]) < 0.001 and _rembg_cache["rgba"] is not None:
                    return _rembg_cache["rgba"]
                frame_rgb = get_frame(t).copy()
                rgba = remove(frame_rgb, session=_rembg_session).copy()
                
                alpha = rgba[..., 3].astype(np.float32)
                
                # Step 3: Local Color Decontamination
                # Neutralize fringes by pulling local neighborhood colors from the avatar interior
                edge_mask = (alpha > 5) & (alpha < 245)
                if np.any(edge_mask):
                    # Use a dilated solid core to find the "safe" interior colors
                    core_mask = (alpha > 250)
                    if np.any(core_mask):
                        # Simple local neighborhood pull: use the mean of the solid core
                        # For high-performance, we use the global core mean but biased 
                        # to remove the specific blue/white background contamination
                        avg_subject_color = np.mean(rgba[..., :3][core_mask], axis=0)
                        
                        # Apply local correction to edge pixels
                        rgba[..., :3][edge_mask] = rgba[..., :3][edge_mask] * 0.1 + avg_subject_color * 0.9
                
                rgba[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
                
                _rembg_cache["t"] = t
                _rembg_cache["rgba"] = rgba
                return rgba

            _avatar_get_frame = avatar_clip.get_frame
            avatar_rgb = VideoClip(
                lambda t: get_rgba_frame(t, _avatar_get_frame)[..., :3],
                duration=audio_duration
            ).resized((cur_w, cur_h))
            mclip = VideoClip(
                lambda t: (get_rgba_frame(t, _avatar_get_frame)[..., 3] / 255.0).astype(np.float32),
                is_mask=True, duration=audio_duration
            )
            avatar_clip = avatar_rgb.with_mask(mclip)
            
        except Exception as e:
            print(f"rembg background removal failed: {e}")
            Y, X = np.ogrid[:cur_h, :cur_w]
            fade_thickness = int(min(cur_w, cur_h) * 0.12)
            dist_edge = np.minimum(np.minimum(X, cur_w - 1 - X), np.minimum(Y, cur_h - 1 - Y))
            a_mask_np = np.clip(dist_edge / fade_thickness, 0.0, 1.0).astype(np.float32)
            mclip = VideoClip(lambda t: a_mask_np, is_mask=True, duration=audio_duration)
            avatar_clip = avatar_clip.with_mask(mclip)

        def pip_position(t):
            scaled_w = int(cur_w * 1.0)
            scaled_h = int(cur_h * 1.0)
            base_x = (FRAME_W - scaled_w) // 2
            base_y = FRAME_H - scaled_h
            return (base_x, base_y)

        avatar_pip = avatar_clip.with_position(pip_position).with_start(0)

    # ── LAYERS ───────────────────────────────────────────────────────────
    screenshot_path = script_json.get("screenshot_path")
    screenshot_clips = _article_screenshot_clip(screenshot_path, audio_duration)
    gradient = _gradient_clip(audio_duration)

    # ── HUMAN REALISM OVERLAYS ───────────────────────────────────────────────
    grain_layer = _generate_film_grain(audio_duration, FRAME_W, FRAME_H)
    flare_layer = _generate_lens_flare(audio_duration, FRAME_W)
    
    # ── COMPLIANCE & BRANDING ────────────────────────────────────────────────
    disclosure = _ai_disclosure_overlay(audio_duration)
    
    # ── ENGAGEMENT LAYERS (Retention Boosters) ────────────────────────────────
    engagement_clips = []
    
    # Hook overlay removed to declutter as requested
    # hook_overlay = _hook_text_overlay(hook_text, accent_color, audio_duration)
    # if hook_overlay:
    #     engagement_clips.append(hook_overlay)

    # Phased scans removed in favor of full-screen loops as requested
    pass
    # ── LAYER 12: Telegram CTA card (Last 6 seconds) ──────────────────────────
    # Disabled full-screen sequential CTA as requested; keeping only the 3s cropped outro.
    # telegram_cta = _telegram_cta_overlay(audio_duration)
    # if telegram_cta:
    #     if isinstance(telegram_cta, list):
    #         engagement_clips.extend(telegram_cta)
    #     else:
    #         engagement_clips.append(telegram_cta)

    # ── COMPOSITING ──
    # Collect infographics logic — DISABLED (user requested removal)
    infographic_clips = []
    # for chunk in chunks:
    #     if chunk.get("has_infographic") and chunk.get("infographic_type"):
    #         iclip = _infographic_card_clip(...)
    #         if iclip: infographic_clips.append(iclip)

    # Stack background, then screenshot clips on top of background
    base_layers = bg_layer_clips + screenshot_clips
    
    # Add overlays
    base_layers.append(gradient)
    base_layers.extend(logo_clips)
    
    if flare_layer: base_layers.append(flare_layer)
    if grain_layer: base_layers.append(grain_layer)
    if avatar_pip: base_layers.append(avatar_pip)
    # ── LOGO BRANDING OVERLAY STACK ────────────────────────────────────
    # Place multiple logos/photos in the top-right corner
    branding_entities = []
    # Collect up to 4 entities to avoid overcrowding
    for ent_list_key in ["people", "companies", "key_entities"]:
        for ent in script_json.get(ent_list_key, []):
            if len(branding_entities) >= 4: break
            
            lp = ent.get("local_logo_path") or ent.get("local_image_path")
            if lp and os.path.exists(lp):
                branding_entities.append((ent.get("name", "Entity"), lp, ent_list_key == "people"))
    
    card_size = 130 # Slightly smaller for stack
    margin = 30
    current_y = 80
    
    for i, (name, path, is_person) in enumerate(branding_entities):
        try:
            img = Image.open(path).convert("RGBA")
            if is_person:
                # Circular crop for people
                img = ImageOps.fit(img, (card_size, card_size), Image.LANCZOS)
                img = _crop_to_circle(img, border_color=accent_color)
            else:
                # Rounded card for companies/models
                lw, lh = img.size
                scale = (card_size - 40) / max(lw, lh)
                img = img.resize((int(lw * scale), int(lh * scale)), Image.LANCZOS)
                canvas = Image.new("RGBA", (card_size, card_size), (0,0,0,0))
                draw = ImageDraw.Draw(canvas)
                draw.rounded_rectangle([0, 0, card_size, card_size], radius=25, fill=(255,255,255,230))
                canvas.paste(img, ((card_size - img.width)//2, (card_size - img.height)//2), img if img.mode == 'RGBA' else None)
                img = canvas
            
            b_clip = ImageClip(np.array(img)).with_duration(audio_duration)
            # Staggered entry (0.5s apart)
            entry_delay = 0.5 + (i * 0.5)
            b_clip = b_clip.with_position((FRAME_W - card_size - 50, current_y)).with_start(entry_delay)
            b_clip = b_clip.with_effects([vfx.CrossFadeIn(0.5)])
            base_layers.append(b_clip)
            
            current_y += card_size + margin
        except Exception as e:
            print(f"Branding failed for {name}: {e}")

    base_layers.append(disclosure)
    base_layers.extend(engagement_clips)

    # ... rest of the original create_video logic continues ...

    # ── PROGRESS BAR ────────────────────────────────────────────────────────
    def get_progress_color(t):
        if audio_duration - t <= 10.0:
            # Shift from accent to neon red in the last 10 seconds
            ratio = max(0, min(1, (10 - (audio_duration - t)) / 10.0))
            red_target = (255, 32, 32)
            c = tuple(int(accent_color[i] + (red_target[i] - accent_color[i]) * ratio) for i in range(3))
            return c
        return accent_color

    # Applying dynamic color using simple VideoClip
    def make_progress_frame(t):
        color = get_progress_color(t)
        base_img = np.zeros((6, FRAME_W, 3), dtype=np.uint8)
        base_img[:, :] = color
        return base_img
        
    progress = VideoClip(make_progress_frame, duration=audio_duration)
    progress = progress.with_position(lambda t: (int((t / max(audio_duration, 0.01)) * FRAME_W) - FRAME_W, FRAME_H - 6))
    base_layers.append(progress)
    
    # ── INFINITE LOOP VISUAL SYNC ─────────────────────────────────────────────
    # If the script is a loop, we force the last 0.5s to crossfade into the first frame
    if script_json.get("loop_score", 0) >= 8:
        print("Enabling Infinite Loop Visual Sync...")
        first_frame = bg_layer_clips[0].get_frame(0)
        finish_img = ImageClip(first_frame).with_duration(0.5).with_start(audio_duration - 0.5).with_effects([vfx.CrossFadeIn(0.4)])
        base_layers.append(finish_img)

    # ── EMOJI POPUPS ─────────────────────────────────────────────────────────
    emoji_popups = script_json.get("emoji_popups", [])
    for ep in emoji_popups:
        ep_ts = float(ep.get("timestamp", 0))
        if ep_ts < audio_duration:
            e_img = render_emoji_popup(ep.get("emoji", "🚀"))
            e_clip = ImageClip(np.array(e_img)).with_duration(1.0).with_start(ep_ts)
            # Center-ish position with a little random offset
            pos = (FRAME_W//2 - 200 + random.randint(-50, 50), FRAME_H//2 - 400)
            e_clip = e_clip.with_position(pos).with_effects([vfx.CrossFadeIn(0.2)])
            base_layers.append(e_clip)

    # ── EASTER EGG FRAME (0.05s) ─────────────────────────────────────────────
    egg_ts = random.uniform(audio_duration * 0.4, audio_duration * 0.7)
    egg_img = insert_easter_egg()
    egg_clip = ImageClip(np.array(egg_img)).with_duration(0.05).with_start(egg_ts)
    base_layers.append(egg_clip)

    # ── UNIVERSAL LAYOUT: HUD OVERLAYS ──
    try:
        from efficiency_engine import render_value_header, render_efficiency_scale, render_evidence_card
        
        # 1. Zone A (0%-30%): Value Headers (Every chunk/sentence) - DISABLED as requested
        pass
        
        # 2. Zone B (30%-50%): Evidence & Efficiency - DISABLED as requested
        # for i in range(int(audio_duration // 15)):
        #     ...
        # for chunk in chunks:
        #     ...
        pass
                
    except Exception as e:
        print(f"HUD Overlay failed: {e}")

    base_comp = CompositeVideoClip(base_layers, size=(FRAME_W, FRAME_H)).with_duration(audio_duration)

    # ── KINETIC SFX DESIGN ───────────────────────────────────────────────────
    final_audio_layers = [audio]
    
    # ── HUMAN REALISM: Environmental Room Tone (0.01 vol) ──────────────────
    try:
        room_tone = _generate_room_tone(audio_duration)
        final_audio_layers.append(room_tone)
    except Exception as e:
        print(f"Room tone synthesis failed (non-fatal): {e}")

    sfx_cues = script_json.get("sfx_cues", [])
    for cue in sfx_cues:
        ctype = cue.get("type", "woosh").lower()
        cue_ts = float(cue.get("timestamp", 0))
        sfx_path = os.path.join(ASSETS_DIR, "sfx", f"{ctype}.wav")
        if os.path.exists(sfx_path) and os.path.getsize(sfx_path) > 0 and cue_ts < audio_duration:
            try:
                sfx_clip = AudioFileClip(sfx_path)
                max_sfx_dur = audio_duration - cue_ts
                if sfx_clip.duration and sfx_clip.duration > max_sfx_dur:
                    sfx_clip = sfx_clip.subclipped(0, max_sfx_dur)
                sfx_clip = sfx_clip.with_start(cue_ts).with_effects([afx.MultiplyVolume(0.4)])
                final_audio_layers.append(sfx_clip)
            except Exception as e:
                print(f"SFX load failed for {ctype} (non-fatal): {e}")
                
    # Auto-inject SFX for subtitle transitions (new lines) - DISABLED as requested
    # for chunk in chunks:
    #     # 1. Woosh on every chunk (sentence) start
    #     cue_ts = chunk["start"]
    #     sfx_path_woosh = os.path.join(ASSETS_DIR, "sfx", "woosh.wav")
    #     if os.path.exists(sfx_path_woosh) and cue_ts < audio_duration:
    #         try:
    #             sfx_clip = AudioFileClip(sfx_path_woosh).with_start(cue_ts).with_effects([afx.MultiplyVolume(0.3)])
    #             final_audio_layers.append(sfx_clip)
    #         except: pass
    
    # Background Music Selection (Topic-Aware: unique music per headline)
    music_files = sorted([f for f in os.listdir(MUSIC_DIR) if f.endswith(('.mp3', '.wav', '.m4a'))])
    if music_files:
        # Use a hash of the original news headline to select a track
        # This ensures every unique topic gets a specific music track assigned to it
        headline = script_json.get("original_news_headline", "")
        import hashlib
        music_hash = int(hashlib.md5(headline.encode()).hexdigest(), 16)
        music_idx = music_hash % len(music_files)
        
        bgm_filename = music_files[music_idx]
        bgm_path = os.path.join(MUSIC_DIR, bgm_filename)
        print(f"🎵 Topic-Aware BGM Selection: {bgm_filename} (Hash-based index: {music_idx+1}/{len(music_files)})")
    else:
        bgm_path = os.path.join(MUSIC_DIR, "modern_tech.mp3")

    if os.path.exists(bgm_path) and os.path.getsize(bgm_path) > 0:
        try:
            bgm = AudioFileClip(bgm_path)
            if bgm.duration is None or bgm.duration <= 0:
                print(f"WARNING: BGM has no valid duration, skipping.")
            else:
                if bgm.duration < audio_duration:
                    # Use afx.AudioLoop (NOT vfx.Loop which is video-only and causes
                    # out-of-bounds audio reads — the root cause of the IndexError crash)
                    bgm = bgm.with_effects([afx.AudioLoop(duration=audio_duration)])
                else:
                    bgm = bgm.subclipped(0, audio_duration)
                
                def ducking_volume(get_frame, t):
                    if isinstance(t, np.ndarray):
                        vols = []
                        for time_t in t:
                            is_speak = any(c["start"] - 0.1 <= time_t <= c["end"] + 0.1 for c in chunks)
                            # Duck when speaking, boost when silent for 'energy'
                            vol = BGM_VOLUME * 0.2 if is_speak else BGM_VOLUME * 1.3
                            vols.append(vol)
                        multiplier = np.array(vols).reshape(-1, 1)
                        return get_frame(t) * multiplier
                    else:
                        is_speak = any(c["start"] - 0.1 <= t <= c["end"] + 0.1 for c in chunks)
                        vol = BGM_VOLUME * 0.2 if is_speak else BGM_VOLUME * 1.3
                        return get_frame(t) * vol
                    
                bgm = bgm.transform(ducking_volume).with_effects([afx.AudioFadeOut(2)])
                
                final_audio_layers.append(bgm)
        except Exception as e:
            print(f"BGM loading failed (non-fatal): {e}")

    final_audio = CompositeAudioClip(final_audio_layers).with_duration(audio_duration)
    
    # Header bar DISABLED — reference style has no persistent title
    header_img = Image.new('RGBA', (FRAME_W, FRAME_H), (0, 0, 0, 0))
    
    # Pre-render 2026 Compliance Watermark
    transparency_img = build_transparency_watermark(FRAME_W, FRAME_H)

    def make_final_frame(t):
        bg_frame = base_comp.get_frame(t)
        
        # Subtitle logic map for the exact current timestamp
        subtitle_img = None
        
        # Find active chunk or hold the last one if t is past the end
        active_chunk = None
        for chunk in chunks:
            if chunk["start"] - 0.1 <= t <= chunk["end"] + 0.1:
                active_chunk = chunk
                break
                
        if not active_chunk and chunks and t > chunks[-1]["end"]:
            active_chunk = chunks[-1]
            
        if active_chunk:
            word_status_list = []
            for w in active_chunk.get("words", []):
                scale = 1.0
                is_active = w["start"] - 0.05 <= t <= w["end"] + 0.05
                if is_active:
                    word_dur = w["end"] - w["start"]
                    p = (t - w["start"]) / max(word_dur, 0.01)
                    if 0 <= p <= 0.2:
                        scale = 1.0 + (0.25 * (p / 0.2))
                    elif 0.2 < p <= 0.4:
                        scale = 1.25 - (0.25 * ((p-0.2)/0.2))
                    else:
                        scale = 1.0

                word_status_list.append({
                    "word": w["word"],
                    "is_active": is_active,
                    "is_spoken": t > w["end"],
                    "scale": scale
                })

            if word_status_list:
                subtitle_img = render_subtitle_frame(
                    word_status_list, bg_frame=bg_frame, 
                    accent_color=accent_color, frame_width=FRAME_W, frame_height=FRAME_H,
                    y_shift=subtitle_y_shift
                )
                
                # ── SENTENCE POP ANIMATION ──
                # Scale up briefly (1.05) when a new subtitle block starts
                chunk_start = active_chunk["start"]
                if 0 <= (t - chunk_start) <= 0.4:
                    pop_p = (t - chunk_start) / 0.4
                    # Scale curve: 1.0 -> 1.05 -> 1.0
                    sub_scale = 1.0 + 0.05 * math.sin(pop_p * math.pi)
                    
                    sub_img_pil = Image.fromarray(np.array(subtitle_img))
                    sw, sh = sub_img_pil.size
                    new_w, new_h = int(sw * sub_scale), int(sh * sub_scale)
                    sub_img_pil = sub_img_pil.resize((new_w, new_h), Image.LANCZOS)
                    
                    # Canvas to keep original size but centered scaled text
                    canvas = Image.new("RGBA", (sw, sh), (0,0,0,0))
                    offset_x = (sw - new_w) // 2
                    offset_y = (sh - new_h) // 2
                    canvas.paste(sub_img_pil, (offset_x, offset_y))
                    subtitle_img = canvas

        # ── HOOK TRANSITION BURST ──────────────────────────────────────────
        # Inject high-impact transition exactly when the hook text fades (usually around 4s)
        hook_end_time = 4.2 
        if abs(t - hook_end_time) < 0.25:
            # 1. Intense Glitch
            bg_frame = _apply_intensive_glitch(bg_frame, intensity=1.2)
            # 2. Flash Burst
            bg_frame = bg_frame.astype(np.float32)
            bg_frame = np.clip(bg_frame + 60, 0, 255).astype(np.uint8)
            
        # ── RETENTION LAYERS: Script-Driven Engagement Cues ──────────────────
        retention_hooks = script_json.get("retention_cues", [])
        for cue in retention_hooks:
            cue_t = float(cue.get("timestamp", 0))
            effect = cue.get("effect", "")
            
            # Use a slightly longer window for cues (0.4s)
            if abs(t - cue_t) < 0.2:
                # 1. SNAP ZOOM (Aggressive punch in)
                if effect == "zoom_snap":
                    h, w = bg_frame.shape[:2]
                    zoom = 1.15 # 15% punch in
                    nh, nw = int(h / zoom), int(w / zoom)
                    top, left = (h - nh) // 2, (w - nw) // 2
                    bg_frame = cv2.resize(bg_frame[top:top+nh, left:left+nw], (w, h))
                
                # 2. EPIC SHAKE (Attention reset)
                elif effect == "shake_epic":
                    shift_x = random.randint(-15, 15)
                    shift_y = random.randint(-15, 15)
                    bg_frame = np.roll(bg_frame, (shift_x, shift_y), axis=(0, 1))
                
                # 3. DIGITAL GLITCH (Technical reveal)
                elif effect == "glitch_digital":
                    bg_frame = _apply_intensive_glitch(bg_frame, intensity=0.9)
                
                # 4. ACCENT FLASH (Breaking/Important)
                elif effect == "flash_accent":
                    bg_frame = bg_frame.astype(np.float32)
                    for i in range(3):
                        bg_frame[:, :, i] = np.clip(bg_frame[:, :, i] + accent_color[i] * 0.6, 0, 255)
                    bg_frame = bg_frame.astype(np.uint8)

        # ── VIRAL FX: SFX REACTIVITY ─────────────────────────────────────────

        # 2. SFX Digital Glitch
        for cue in sfx_cues:
            if cue.get("type") == "glitch" and abs(t - float(cue.get("timestamp", 0))) < 0.25:
                bg_frame = _apply_intensive_glitch(bg_frame, intensity=0.8)
                # Random camera shake during glitch
                shift_x = random.randint(-8, 8)
                shift_y = random.randint(-8, 8)
                bg_frame = np.roll(bg_frame, (shift_x, shift_y), axis=(0, 1))

        # 3. Energy Vibration on Woosh
        for cue in sfx_cues:
            if cue.get("type") == "woosh" and abs(t - float(cue.get("timestamp", 0))) < 0.2:
                # Zoom in slightly (simulated)
                h, w = bg_frame.shape[:2]
                zoom = 1.05
                nh, nw = int(h / zoom), int(w / zoom)
                top, left = (h - nh) // 2, (w - nw) // 2
                cropped = bg_frame[top:top+nh, left:left+nw]
                bg_frame = cv2.resize(cropped, (w, h))

        # Intro flash DISABLED — reference style
        pass

        return composite_frame(bg_frame, t, header_img, subtitle_img, transparency_img)


    final = VideoClip(make_final_frame, duration=audio_duration)
    final = final.with_audio(final_audio)

    # ── TELEGRAM CTA V2 (Last 3s) ─────────────────────────────────────────────
    cta_duration = 3.0
    cta_img = Image.new("RGBA", (FRAME_W, FRAME_H), (10, 10, 15, 255))
    cta_d = ImageDraw.Draw(cta_img)
    
    # 1. Topic-Sync Headline
    topic = script_json.get("topic", "AI")
    cta_txt = f"Full {topic} guide + source code"
    cta_d.text((FRAME_W//2, 180), cta_txt, fill=(255, 255, 255, 255), font=gf(54, bold=True), anchor="mm")
    
    # 2. Telegram Channel Screenshot (Top 50% Crop)
    try:
        brand_path = os.path.join(ASSETS_DIR, "branding", "tele_brand2.jpg")
        if os.path.exists(brand_path):
            brand_img = Image.open(brand_path).convert("RGBA")
            # Crop top 50%
            bw, bh = brand_img.size
            brand_img = brand_img.crop((0, 0, bw, bh // 2))
            ratio = (FRAME_W - 100) / float(brand_img.width)
            brand_img = brand_img.resize((int(brand_img.width * ratio), int(brand_img.height * ratio)), Image.LANCZOS)
            cta_img.alpha_composite(brand_img, (50, 320))
    except: pass
    
    # 3. Link in Bio (Bottom 50%)
    pill_y = FRAME_H - 450
    cta_d.rounded_rectangle([200, pill_y, FRAME_W - 200, pill_y + 120], radius=60, fill=(204, 255, 0, 255))
    cta_d.text((FRAME_W//2, pill_y + 60), "Link in Bio", fill=(0, 0, 0, 255), font=gf(50, bold=True), anchor="mm")
    
    # Description
    cta_d.text((FRAME_W//2, pill_y + 180), "Join the community 🚀", fill=(200, 200, 200, 255), font=gf(34), anchor="mm")

    cta_clip = ImageClip(np.array(cta_img.convert("RGB"))).with_duration(cta_duration)
    
    # SFX at CTA
    sfx_path = os.path.join(ASSETS_DIR, "sfx", "pop.wav")
    if os.path.exists(sfx_path):
        cta_audio = AudioFileClip(sfx_path).with_effects([afx.MultiplyVolume(0.5)])
        silence = AudioClip(lambda t: [0,0], duration=max(0.1, cta_duration - cta_audio.duration))
        cta_audio = concatenate_audioclips([cta_audio, silence])
        cta_clip = cta_clip.with_audio(cta_audio)
        
    final = concatenate_videoclips([final, cta_clip], method="compose")

    print(f"Exporting {audio_duration:.1f}s → {output_path}")
    
    # ── TEXT VISIBILITY EXPORT CHECK ──
    print("Extracting test frames for text visibility check...")
    try:
        log_file = os.path.join(LOGS_DIR, f"visibility_{datetime.today().strftime('%Y-%m-%d')}.txt")
        with open(log_file, "a") as f:
            f.write(f"\n--- Checking text visibility for {os.path.basename(output_path)} ---\n")
            fractions = [0.1, 0.3, 0.6, 0.9]
            timestamps = [audio_duration * p for p in fractions]
            for i, (t, p) in enumerate(zip(timestamps, fractions)):
                test_frame = final.get_frame(t + 2.0)
                img = Image.fromarray(test_frame)
                
                print(f"Validating text rendering at {t:.1f}s...")
                verify_text_visibility(test_frame, f"SUBTITLE {p}", 1450, 1800)
                verify_text_visibility(test_frame, f"HEADER {p}", 0, 240)
                
                test_path = output_path.replace(".mp4", f"_test_{int(p*100)}pct.jpg")
                img.save(test_path)
    except Exception as e:
        print(f"Visibility frames failed: {e}")

    final.write_videofile(
        output_path, fps=30, codec="libx264", audio_codec="aac",
        threads=4, preset="ultrafast", ffmpeg_params=["-pix_fmt", "yuv420p"]
    )
    return output_path
