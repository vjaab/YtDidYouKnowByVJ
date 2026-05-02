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
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datetime import datetime
from google import genai
from google.genai import types
from moviepy import (
    VideoClip, ImageClip, VideoFileClip, AudioFileClip,
    CompositeVideoClip, ColorClip, CompositeAudioClip, concatenate_videoclips
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

def gf(size):
    if size not in _fc:
        for p in FONT_PATHS:
            if os.path.exists(p):
                try:
                    _fc[size] = ImageFont.truetype(p, size)
                    break
                except Exception:
                    pass
        if size not in _fc:
            _fc[size] = ImageFont.load_default()
    return _fc[size]

def ts(text, font):
    bb = font.getbbox(text)
    return bb[2] - bb[0], bb[3] - bb[1]


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
    base_img = img.resize((pw, ph), Image.LANCZOS)
    
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
    h = int(FRAME_H * 0.50)
    arr = np.zeros((h, FRAME_W, 3), dtype=np.uint8)
    mask_arr = np.array(
        [(int(200 * (y/h)**0.7),) * FRAME_W for y in range(h)],
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
    """Displays giant hook text in the first 1.5 seconds to stop the scroll."""
    if not hook_text:
        return None
    dur = min(1.8, total_dur)
    f = gf(72)
    max_w = FRAME_W - 100

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
    lines = lines[:3]  # Max 3 lines

    lh = ts("Ag", f)[1]
    lsp = int(lh * 1.4)
    total_h = lh + (len(lines) - 1) * lsp
    canvas_h = total_h + 80
    canvas = Image.new("RGBA", (FRAME_W, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    # Semi-transparent dark backdrop
    draw.rounded_rectangle([30, 10, FRAME_W - 30, canvas_h - 10], radius=20, fill=(0, 0, 0, 180))

    for i, line in enumerate(lines):
        lw, _ = ts(line, f)
        tx = (FRAME_W - lw) // 2
        ty = 40 + i * lsp
        # Text shadow
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3)]:
            draw.text((tx + dx, ty + dy), line, font=f, fill=(0, 0, 0, 200))
        draw.text((tx, ty), line, font=f, fill=(*accent_color, 255))

    arr = np.array(canvas.convert("RGB"))
    mask = np.array(canvas.split()[3]).astype(float) / 255.0

    def opacity_fn(t):
        if t < 0.15:
            return t / 0.15  # Fade in
        elif t > dur - 0.4:
            return max(0, (dur - t) / 0.4)  # Fade out
        return 1.0

    clip = VideoClip(lambda t: arr, duration=dur)
    mclip = VideoClip(lambda t: mask * opacity_fn(t), is_mask=True, duration=dur)
    y_pos = int(FRAME_H * 0.32) # Moved up slightly
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
    y_pos = int(FRAME_H * 0.72) # Moved significantly down to clear subtitles
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


def _render_process_steps(steps, accent_color, width=960):
    """Numbered flow steps (up to 4) shown as a horizontal pill row."""
    n     = min(len(steps), 4)
    f     = gf(28)
    pad   = 18
    h     = 90
    img   = Image.new("RGBA", (width, h), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(img)

    col_w = width // n
    for i, step in enumerate(steps[:n]):
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

def _render_flowchart_card(steps, accent_color, width=900):
    """Vertical architectural flowchart: Step 1 -> Step 2 -> Step 3."""
    n = min(len(steps), 4)
    if n == 0: return Image.new("RGBA", (width, 100), (0, 0, 0, 0))
    
    step_h = 100
    gap = 40
    total_h = n * step_h + (n-1) * gap + 60
    
    img = Image.new("RGBA", (width, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    f_step = gf(32)
    f_num = gf(24)
    
    # Background glass
    draw.rounded_rectangle([0, 0, width, total_h], radius=25, fill=(10, 10, 15, 200))
    
    for i, step in enumerate(steps[:n]):
        y0 = 30 + i * (step_h + gap)
        y1 = y0 + step_h
        
        # Step box
        box_w = width - 100
        bx0 = 50
        bx1 = bx0 + box_w
        
        draw.rounded_rectangle([bx0, y0, bx1, y1], radius=15, outline=(*accent_color, 150), width=2, fill=(20, 20, 30, 240))
        
        # Number badge
        draw.ellipse([bx0 - 20, y0 + step_h//2 - 20, bx0 + 20, y0 + step_h//2 + 20], fill=(*accent_color, 255))
        draw.text((bx0, y0 + step_h//2), str(i+1), font=f_num, fill=(255,255,255), anchor="mm")
        
        # Step text (Monospace feel)
        draw.text((bx0 + 40, y0 + step_h//2), step.upper(), font=f_step, fill=(255, 255, 255, 255), anchor="lm")
        
        # Connector Arrow
        if i < n - 1:
            ay_start = y1 + 5
            ay_end = y1 + gap - 5
            ax = bx0 + box_w // 2
            draw.line([(ax, ay_start), (ax, ay_end)], fill=(*accent_color, 120), width=3)
            # Arrow head
            draw.polygon([(ax - 10, ay_end - 10), (ax + 10, ay_end - 10), (ax, ay_end)], fill=(*accent_color, 120))
            
    return img

def _render_slide_card(title, bullets, accent_color, is_longform=False):
    h = 600 if not is_longform else 800
    w = 900 if not is_longform else 1500
    pil = Image.new("RGBA", (w, h), (0,0,0,0))
    d = ImageDraw.Draw(pil)
    bg = (20,20,30,235)
    d.rounded_rectangle([0,0,w,h], 40, fill=bg, outline=accent_color, width=4)
    
    cx = w // 2
    f_title = gf(50 if not is_longform else 70)
    
    # Title
    ttw, tth = ts(title, f_title)
    d.text((cx - ttw//2, 40), title, fill=(*accent_color, 255), font=f_title)
    
    # Divider
    d.line([(40, 130), (w - 40, 130)], fill=(*accent_color, 200), width=3)
    
    # Bullets
    f_bullet = gf(35 if not is_longform else 50)
    start_y = 180
    by = start_y
    for i, bullet in enumerate(bullets):
        dot_y = by + (20 if not is_longform else 30)
        d.ellipse([60, dot_y - 8, 76, dot_y + 8], fill=(*accent_color, 255))
        
        b_lines = wrap_text_to_lines(str(bullet).split(), [ts(wd, f_bullet)[0] for wd in str(bullet).split()], w - 160, f_bullet)
        for line_words in b_lines:
            d.text((100, by), " ".join(line_words), font=f_bullet, fill=(255,255,255,255))
            by += 45 if not is_longform else 65
        by += 20
            
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
                contents=[{'mime_type': 'image/png', 'data': img_bytes}, prompt]
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

    arr      = np.array(pil.convert("RGB"))
    mask_arr = np.array(pil.split()[3]).astype(float) / 255.0

    # Slide-up entrance + fade out
    iw, ih = pil.size
    x_pos   = (FRAME_W - iw) // 2

    def opacity_fn(t):
        if t < 0.25: return t / 0.25
        if t > dur - 0.35: return max(0, (dur - t) / 0.35)
        return 1.0

    def y_pos_fn(t):
        slide = min(t / 0.25, 1.0)
        # Ease-out: starts 40px below final position
        eased = 1 - (1 - slide) ** 2
        return int(FRAME_H * 0.62 + 40 * (1 - eased))

    clip  = VideoClip(lambda t: arr, duration=dur)
    mclip = VideoClip(lambda t: mask_arr * opacity_fn(t),
                      is_mask=True, duration=dur)

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
    """Sleek minimalist header with floating badge."""
    header_h = 240
    img = Image.new('RGBA', (frame_width, header_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    # Typography: Roboto-Bold (similar to YouTube Sans) per user spec
    f_title = ImageFont.truetype('assets/fonts/Roboto-Bold.ttf', 55)
    
    # Check width
    tw = draw.textlength(title, font=f_title)
    if tw > 800: title = title[:40] + "..."
    tw = draw.textlength(title, font=f_title)
    
    # Center text
    padding_x = 40
    padding_y = 20
    box_w = tw + padding_x * 2
    box_h = 55 + padding_y * 2
    start_x = (frame_width - box_w) // 2
    
    # Move title little bit up: Y=50
    start_y = 50
    text_y_offset = -8
    
    # 1. High-contrast pill for header
    draw.rounded_rectangle([start_x, start_y, start_x+box_w, start_y+box_h], radius=15, fill=(245, 245, 245, 255))
    
    # 2. Wordmark lockup in Dark Grey for readability
    draw.text((start_x + padding_x, start_y + padding_y + text_y_offset), title, font=f_title, fill=(10, 10, 10, 255))
    
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
def _article_screenshot_clip(screenshot_path, duration):
    """
    Transformative Fullscreen logic: Shows the source article as the main visual.
    Anchored to (0,0) to ensure top/left edges don't go outside during zoom.
    Displayed twice for maximum impact.
    """
    if not screenshot_path or not os.path.exists(screenshot_path):
        return []
    try:
        # Load and verify image
        img = Image.open(screenshot_path).convert("RGBA")
        
        # Fullscreen Resize: 1080x1920 (Force fullscreen fit)
        img = img.resize((FRAME_W, FRAME_H), Image.LANCZOS)
        
        arr = np.array(img.convert("RGB"))
        mask = np.array(img.split()[3]).astype(float) / 255.0
        
        clips = []
        
        # --- FIRST APPEARANCE: Evidence/Hook Phase ---
        # Show briefly early on to establish source credibility
        start1 = 3.5
        dur1 = 2.5
        if duration > start1 + dur1:
            clip1 = ImageClip(arr, duration=dur1)
            mclip1 = VideoClip(lambda t: mask, is_mask=True, duration=dur1)
            # Subtle zoom: 1.0 to 1.10
            # Anchor at (0,0) so top and left stay fixed
            clip1 = clip1.resized(lambda t: 1.0 + 0.10 * (t / dur1))
            clip1 = clip1.with_mask(mclip1).with_position((0, 0)).with_start(start1)
            clip1 = clip1.with_effects([vfx.CrossFadeIn(0.4), vfx.CrossFadeOut(0.4)])
            clips.append(clip1)

        # --- SECOND APPEARANCE: Deep Dive Phase ---
        # Full Headline Zoom for detailed commentary phase
        start2 = 12.0
        dur2 = min(36.0, duration - start2)
        if dur2 > 0:
            clip2 = ImageClip(arr, duration=dur2)
            mclip2 = VideoClip(lambda t: mask, is_mask=True, duration=dur2)
            # Deep Zoom: 1.0 to 1.30
            # Anchor at (0,0) to keep headline and logo in view
            clip2 = clip2.resized(lambda t: 1.0 + 0.30 * (t / dur2))
            clip2 = clip2.with_mask(mclip2).with_position((0, 0)).with_start(start2)
            clip2 = clip2.with_effects([vfx.CrossFadeIn(0.5), vfx.CrossFadeOut(0.5)])
            clips.append(clip2)
            
        return clips
        
    except Exception as e:
        print(f"Article screenshot clip error: {e}")
        return []

def _evidence_screenshot_clip(evidence_path, duration):
    """
    Shows a secondary 'Evidence' or 'Use Case' screenshot during the analytical section.
    Appears typically between 25-35 seconds mark.
    """
    if not evidence_path or not os.path.exists(evidence_path):
        return []
    try:
        img = Image.open(evidence_path).convert("RGBA")
        img = img.resize((FRAME_W, FRAME_H), Image.LANCZOS)
        arr = np.array(img.convert("RGB"))
        mask = np.array(img.split()[3]).astype(float) / 255.0
        
        # Appearance window: Typically during the 'Deep-Dive' or 'Analytical Commentary'
        start = 28.0 
        dur = min(6.0, duration - start - 5.0)
        
        if dur > 1.0:
            clip = ImageClip(arr, duration=dur)
            mclip = VideoClip(lambda t: mask, is_mask=True, duration=dur)
            # Subtle upward slide + zoom
            clip = clip.resized(lambda t: 1.05 + 0.15 * (t / dur))
            clip = clip.with_mask(mclip).with_position((0, 0)).with_start(start)
            clip = clip.with_effects([vfx.CrossFadeIn(0.6), vfx.CrossFadeOut(0.6)])
            
            # Add a 'PROVEN EVIDENCE' label overlay
            f = gf(40)
            txt = "REAL-WORLD EVIDENCE"
            tw, th = ts(txt, f)
            label = Image.new("RGBA", (tw + 40, th + 20), (0,0,0,0))
            draw = ImageDraw.Draw(label)
            draw.rounded_rectangle([0, 0, label.width, label.height], radius=15, fill=(255, 214, 0, 230))
            draw.text((20, 10), txt, font=f, fill=(0,0,0,255))
            
            l_clip = _pil_clip(label, dur, start=start).with_position(("center", 450))
            l_clip = l_clip.with_effects([vfx.CrossFadeIn(0.8)])
            
            return [clip, l_clip]
            
    except Exception as e:
        print(f"Evidence screenshot clip error: {e}")
    return []


def _ai_disclosure_overlay(duration):
    """
    Mandatory AI Disclosure for Monetization Compliance (Realistic Synthetic content).
    """
    h, w = 60, 800
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Translucent black capsule
    draw.rounded_rectangle([0, 0, w, h], radius=30, fill=(0, 0, 0, 160))
    
    txt = "Visual/Audio generated with AI to enhance reporting."
    font = gf(28)
    draw.text((w//2, h//2), txt, font=font, fill=(200, 200, 200, 255), anchor="mm")
    
    arr = np.array(img.convert("RGB"))
    mask = np.array(img.split()[3]).astype(float) / 255.0
    
    clip_dur = 5.0 # Show for 5 seconds
    clip = ImageClip(arr, duration=clip_dur)
    mclip = VideoClip(lambda t: mask, is_mask=True, duration=clip_dur)
    
    return clip.with_mask(mclip).with_position(("center", 40)).with_start(1.0).with_effects([vfx.CrossFadeIn(0.5), vfx.CrossFadeOut(0.5)])


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
    
    # Position: Top Right, slightly below the header bar (which is at y=50 usually)
    x, y = width - tw - 40, 160
    
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
    space_w = 12
    for word, w in zip(words, word_widths):
        if not current_line or current_w + w <= max_width:
            current_line.append(word)
            current_w += w + space_w
        else:
            lines.append(current_line)
            current_line = [word]
            current_w = w + space_w
    if current_line:
        lines.append(current_line)
    return lines

def render_subtitle_frame(word_data, bg_frame=None, accent_color=(255,214,0), frame_width=1080, frame_height=1920):
    """Modern Hormozi-style minimalist captions with Kinetic Word Pop."""
    img = Image.new('RGBA', (frame_width, frame_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    # Base fonts (Relative scaling for robustness)
    scale_ratio = frame_width / 1080.0 if frame_width < frame_height else frame_width / 1920.0
    base_size = int(85 * scale_ratio)
    pop_size = int(95 * scale_ratio)
    f_main = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', base_size)
    
    # word_data is a list of {"word": str, "is_active": bool, "is_spoken": bool, "scale": float}
    words = [wd["word"] for wd in word_data]
    
    word_widths = []
    fake_draw = ImageDraw.Draw(Image.new("RGBA", (1,1)))
    
    for i, wd in enumerate(word_data):
        # Dynamically calculate font size for this word based on its scale
        s = pop_size if wd["is_active"] else base_size
        s = int(s * wd["scale"])
        f_current = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', s)
        bbox = fake_draw.textbbox((0,0), words[i], font=f_current)
        word_widths.append(bbox[2]-bbox[0])
    
    max_sub_width = int(frame_width * 0.85)
    lines = wrap_text_to_lines(words, word_widths, max_sub_width, f_main)
    
    line_h = int(130 * scale_ratio)
    total_h = len(lines) * line_h
    
    # Position: Lower-middle for vertical, bottom for horizontal
    if frame_width < frame_height:
        start_y = int(frame_height * 0.5) # Middle for vertical Shorts
    else:
        start_y = int(frame_height * 0.75) # Near bottom for longform

    # ── READABILITY BOOST: Dark vignette behind subtitles ──
    # Draw a soft dark glow to ensure text pops against white/busy backgrounds
    sub_bg_h = total_h + 100
    sub_bg = Image.new('RGBA', (frame_width, sub_bg_h), (0,0,0,0))
    sub_draw = ImageDraw.Draw(sub_bg)
    # Gradient fade or semi-trans box
    sub_draw.rectangle([0, 0, frame_width, sub_bg_h], fill=(0, 0, 0, 140))
    # Paste with a mask if you want it feathered, but a simple alpha box works well
    img.alpha_composite(sub_bg, dest=(0, start_y - 40))
    
    word_idx = 0
    for i, line in enumerate(lines):
        line_y = start_y + i * line_h
        line_w = sum(word_widths[word_idx:word_idx+len(line)]) + 12 * (len(line)-1)
        cur_x = (frame_width - line_w) // 2
        
        for word_text in line:
            wd = word_data[word_idx]
            is_shouting = word_text.isupper() and len(word_text) > 1
            is_active = wd["is_active"]
            
            # Recalculate font with scale
            s = pop_size if is_active else base_size
            s = int(s * wd["scale"])
            f = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', s)
            
            # Colors
            c_fill = (255, 255, 255, 255)
            if is_active:
                c_fill = (*accent_color, 255)
            elif is_shouting:
                c_fill = (*accent_color, 180) 
            elif wd["is_spoken"]:
                c_fill = (180, 180, 180, 255)
            
            # Adjust Y for scaling (keep baseline consistent or center it)
            # Center it vertically within the line_h
            bbox = draw.textbbox((cur_x, line_y), word_text, font=f)
            th = bbox[3] - bbox[1]
            y_offset = (line_h - th) // 2 - 10
            
            # Main text with Glow for active word
            if is_active:
                # Neon Glow
                glow_img = Image.new('RGBA', (word_widths[word_idx] + 20, line_h), (0,0,0,0))
                glow_draw = ImageDraw.Draw(glow_img)
                glow_draw.text((10, y_offset), word_text, font=f, fill=(*accent_color, 80), stroke_width=9, stroke_fill=(*accent_color, 40))
                img.alpha_composite(glow_img, dest=(cur_x - 10, line_y))

            draw.text(
                (cur_x, line_y + y_offset), 
                word_text, font=f, fill=c_fill,
                stroke_width=3, stroke_fill=(0, 0, 0, 255)
            )
            
            cur_x += word_widths[word_idx] + 12
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
    accent_hex     = color_theme.get("accent", "#ff4444").lstrip("#")
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
        crossfade = 0.1
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
                
                w, h = c_clip.size
                target_h = int(w * 16 / 9)
                if target_h <= h:
                    y1 = (h - target_h) // 2
                    c_clip = c_clip.cropped(x1=0, y1=y1, x2=w, y2=y1 + target_h)
                else:
                    target_w = int(h * 9 / 16)
                    x1 = (w - target_w) // 2
                    c_clip = c_clip.cropped(x1=x1, y1=0, x2=x1 + target_w, y2=h)

                c_clip = c_clip.resized((FRAME_W, FRAME_H))
                
                if script_json.get("breaking_news_level", 0) >= 8 and random.random() < 0.45:
                    split_h = FRAME_H // 2
                    top_half = c_clip.cropped(y2=split_h)
                    bottom_half = c_clip.cropped(y1=split_h).with_effects([vfx.MirrorX()])
                    c_clip = CompositeVideoClip([
                        top_half.with_position(("center", "top")),
                        bottom_half.with_position(("center", "bottom"))
                    ], size=(FRAME_W, FRAME_H))

                scale_factor = 1.0 + random.uniform(0.05, 0.12)
                c_clip = c_clip.resized(lambda t: 1.0 + (scale_factor - 1.0) * (t / clip_dur))
                c_clip = _apply_handheld_shake(c_clip)
                c_clip = c_clip.with_start(current_start)
                
                if i > 0:
                    flash = ColorClip(size=(FRAME_W, FRAME_H), color=(255, 255, 255), duration=0.15).with_opacity(0.7)
                    flash = flash.with_start(current_start).with_effects([vfx.CrossFadeOut(0.1)])
                    logo_clips.append(flash)
                
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

        width_pip, height_pip = 450, 680
        w, h = vid_clip.size
        # Crop to target aspect ratio
        target_aspect = width_pip / height_pip
        if w/h > target_aspect:
            new_w = int(h * target_aspect)
            x1 = (w - new_w) // 2
            vid_clip = vid_clip.cropped(x1=x1, y1=0, x2=x1+new_w, y2=h)
        else:
            new_h = int(w / target_aspect)
            y1 = (h - new_h) // 2
            vid_clip = vid_clip.cropped(x1=0, y1=y1, x2=w, y2=y1+new_h)
        
        # Apply refinements from dynamic_params
        cur_w = int(width_pip * avatar_scale_mult)
        cur_h = int(height_pip * avatar_scale_mult)
        avatar_clip = vid_clip.resized((cur_w, cur_h)).without_audio()
        
        # Rounded & Feathered Mask
        a_mask_np = np.zeros((cur_h, cur_w), dtype=np.uint8)
        radius = int(32 * avatar_scale_mult)
        cv2.circle(a_mask_np, (radius, radius), radius, 255, -1)
        cv2.circle(a_mask_np, (cur_w-radius, radius), radius, 255, -1)
        cv2.circle(a_mask_np, (radius, cur_h-radius), radius, 255, -1)
        cv2.circle(a_mask_np, (cur_w-radius, cur_h-radius), radius, 255, -1)
        cv2.rectangle(a_mask_np, (radius, 0), (cur_w-radius, cur_h), 255, -1)
        cv2.rectangle(a_mask_np, (0, radius), (cur_w, cur_h-radius), 255, -1)
        
        mask_img = Image.fromarray(a_mask_np)
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=5))
        a_mask_feathered = np.array(mask_img).astype(float) / 255.0
        
        mclip = VideoClip(lambda t: a_mask_feathered, is_mask=True, duration=audio_duration)
        avatar_clip = avatar_clip.with_mask(mclip)

        _screenshot_path_check = script_json.get("screenshot_path")
        _has_screenshot = bool(_screenshot_path_check and os.path.exists(_screenshot_path_check))

        def pip_position(t):
            base_x, base_y = FRAME_W - cur_w - 20, 260
            base_y += subtitle_y_shift # Apply feedback shift
            
            sway_y = math.sin(t * 0.3 * 2 * math.pi) * 3 + math.sin(t * 1.2 * 2 * math.pi) * 1.5
            sway_x = math.sin(t * 0.2 * 2 * math.pi) * 2 + math.cos(t * 0.7 * 2 * math.pi) * 1
            breathing = math.sin(t * 0.25 * 2 * math.pi) * 1
            
            e_x, e_y = 0, 0
            if shock_ts > 0 and abs(t - shock_ts) < 1.5:
                p = 1.0 - abs(t - shock_ts) / 1.5
                e_x, e_y = int(-15 * p), int(-8 * p)
            elif key_stat_ts > 0 and abs(t - key_stat_ts) < 1.0:
                p = 1.0 - abs(t - key_stat_ts) / 1.0
                e_y = int(5 * math.sin(p * math.pi))
            
            if not _has_screenshot:
                return (base_x + int(sway_x) + e_x, base_y + int(sway_y + breathing) + e_y)

            if t < 6.0: return (base_x + int(sway_x) + e_x, base_y + int(sway_y + breathing) + e_y)
            cycle = (t - 6.0) % 11.0
            if cycle < 4.0: return (FRAME_W + 1000, base_y)
            return (base_x + int(sway_x) + e_x, base_y + int(sway_y + breathing) + e_y)

        def avatar_scale(t):
            base = 1.0
            if shock_ts > 0 and abs(t - shock_ts) < 1.0:
                p = 1.0 - abs(t - shock_ts) / 1.0
                base = 1.0 + 0.12 * math.sin(p * math.pi)
            elif key_stat_ts > 0 and abs(t - key_stat_ts) < 0.8:
                p = 1.0 - abs(t - key_stat_ts) / 0.8
                base = 1.0 + 0.08 * math.sin(p * math.pi)
            return base

        avatar_clip = avatar_clip.with_effects([vfx.Resize(avatar_scale)])

        def _hud_pane(duration, accent):
            pw, ph = cur_w + 40, cur_h + 40
            img = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle([0, 0, pw, ph], radius=40, fill=(20, 20, 30, 160))
            return _pil_clip(img, duration)

        pane = _hud_pane(audio_duration, accent_color)

        avatar_pip = CompositeVideoClip([
            pane.with_position((-20, -20)),
            avatar_clip.with_position((0, 0))
        ], size=(cur_w + 60, cur_h + 60), is_mask=False).with_position(pip_position).with_start(0)

        # ── Avatar-Relative Callouts ──
        top_kws = sorted(script_json.get("keywords", []), key=len, reverse=True)[:3]
        for i, kw in enumerate(top_kws):
            t_kw = (i + 1) * (audio_duration / 4)
            if t_kw > audio_duration - 2: continue
            skw = Image.new("RGBA", (ts(kw, gf(28))[0] + 30, 50), (0, 0, 0, 0))
            ImageDraw.Draw(skw).rounded_rectangle([0, 0, skw.width, 50], radius=15, fill=(*accent_color, 220))
            ImageDraw.Draw(skw).text((15, 10), kw, font=gf(28), fill=(255, 255, 255))
            def kpos(t, _ts=t_kw):
                rel = t - _ts
                if rel < 0 or rel > 2.0: return (FRAME_W + 100, 0)
                bx, by = pip_position(t)
                return (bx - 40 + int(rel*40), by - 60 - int(rel*40))
            logo_clips.append(_pil_clip(skw, 2.0, start=t_kw).with_position(kpos))

    # ── LAYERS ───────────────────────────────────────────────────────────
    screenshot_path = script_json.get("screenshot_path")
    screenshot_clips = _article_screenshot_clip(screenshot_path, audio_duration)
    tint = ColorClip(size=(FRAME_W, FRAME_H), color=accent_color, duration=audio_duration).with_opacity(0.02)
    gradient = _gradient_clip(audio_duration)
    
    sub_cat_low = sub_category.lower()
    p_style = "digital" if any(x in sub_cat_low for x in ["tool", "hands", "code"]) else \
              "stars" if any(x in sub_cat_low for x in ["research", "news", "predict"]) else "bokeh"
    
    bg_h = color_theme.get("background", "#0A0A0F").lstrip("#")
    bg_base = tuple(int(bg_h[i:i+2], 16) for i in (0, 2, 4))
    
    ambient = _ambient_particles(audio_duration, accent_color, particle_style=p_style)
    particle_clips.append(ambient)
    tech_bg = _dynamic_tech_background(audio_duration, accent_color, bg_base_color=bg_base)
    bg_layer_clips.insert(0, tech_bg)
    
    tags_img = render_entity_tags(key_entities, accent_color, FRAME_W, on_right=False)
    def tag_pos(t):
        if t < 0.5: return (-500 + int(500 * (t/0.5)), 320)
        return (0, 320)
    tags_clip = ImageClip(np.array(tags_img)).with_duration(audio_duration).with_position(tag_pos)
    sweep = _sweep_clip(5.0, accent_color, FRAME_W).with_effects([vfx.Loop(duration=audio_duration)]).with_position((0, 320))
    logo_clips.extend([tags_clip, sweep])

    # ── HUMAN REALISM OVERLAYS ───────────────────────────────────────────────
    grain_layer = _generate_film_grain(audio_duration, FRAME_W, FRAME_H)
    flare_layer = _generate_lens_flare(audio_duration, FRAME_W)
    
    # ── COMPLIANCE & BRANDING ────────────────────────────────────────────────
    disclosure = _ai_disclosure_overlay(audio_duration)
    watermark = _brand_watermark(audio_duration)
    
    # ── ENGAGEMENT LAYERS (Retention Boosters) ────────────────────────────────
    engagement_clips = []
    
    # E1: Pattern Interrupt Flash (0.3s accent flash at start)
    pi_flash = _pattern_interrupt_flash(accent_color, audio_duration)
    if pi_flash:
        engagement_clips.append(pi_flash)
    
    # E2: Giant Hook Text (first 1.5s)
    hook_text = script_json.get("hook_text", "")
    hook_overlay = _hook_text_overlay(hook_text, accent_color, audio_duration)
    if hook_overlay:
        engagement_clips.append(hook_overlay)

    # ── COMPOSITING ──
    # Collect infographics logic
    infographic_clips = []
    for chunk in chunks:
        if chunk.get("has_infographic") and chunk.get("infographic_type"):
            iclip = _infographic_card_clip(
                chunk.get("infographic_type"),
                chunk.get("infographic_data", {}),
                accent_color,
                chunk.get("start", 0),
                chunk.get("duration", 2),
                audio_duration
            )
            if iclip:
                infographic_clips.append(iclip)

    base_layers = bg_layer_clips + screenshot_clips + [tint, gradient] + particle_clips + logo_clips + infographic_clips
    if flare_layer: base_layers.append(flare_layer)
    if grain_layer: base_layers.append(grain_layer)
    if avatar_pip: base_layers.append(avatar_pip)
    base_layers.append(disclosure)
    base_layers.append(watermark)
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
                # Clamp SFX so it doesn't extend past audio_duration
                max_sfx_dur = audio_duration - cue_ts
                if sfx_clip.duration and sfx_clip.duration > max_sfx_dur:
                    sfx_clip = sfx_clip.subclipped(0, max_sfx_dur)
                sfx_clip = sfx_clip.with_start(cue_ts).with_effects([afx.MultiplyVolume(0.4)])
                final_audio_layers.append(sfx_clip)
            except Exception as e:
                print(f"SFX load failed for {ctype} (non-fatal): {e}")
    
    # Background Music with Auto-Ducking
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
    
    # Pre-render header (only persistent overlay)
    header_img = render_header_bar(title, sub_category, accent_color, FRAME_W)
    
    # Pre-render 2026 Compliance Watermark
    transparency_img = build_transparency_watermark(FRAME_W, FRAME_H)

    def make_final_frame(t):
        bg_frame = base_comp.get_frame(t)
        
        # Subtitle logic map for the exact current timestamp
        subtitle_img = None
        for chunk in chunks:
            if chunk["start"] - 0.1 <= t <= chunk["end"] + 0.1:
                # REQUEST: Subtitles should disappear for the final feedback CTA
                text_low = chunk.get("text", "").lower()
                cta_keywords = ["follow", "updates", "suggestions", "feedback", "whatsapp", "telegram", "bio"]
                if any(kw in text_low for kw in cta_keywords) and t > audio_duration - 2.5:
                    break
                    
                word_status_list = []
                for w in chunk.get("words", []):
                    # Kinetic Pop Logic: Scale up 1.25x for first 150ms of word
                    scale = 1.0
                    is_active = w["start"] - 0.05 <= t <= w["end"] + 0.05
                    if is_active:
                        # Pop peaks in first 20% of duration (or 0.15s) and settles
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
                        accent_color=accent_color, frame_width=FRAME_W, frame_height=FRAME_H
                    )
                break
                
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

        # ── EMERGENCY: 0.1s VISUAL SHOCK (Intro Flash) ────────────────────
        if t < 0.2:
            # 200ms white flash intro to stop the scroll
            bg_frame = bg_frame.astype(np.float32)
            bg_frame += 100 # Brighten the whole frame
            bg_frame = np.clip(bg_frame, 0, 255).astype(np.uint8)

        return composite_frame(bg_frame, t, header_img, subtitle_img, transparency_img)


    final = VideoClip(make_final_frame, duration=audio_duration)
    final = final.with_audio(final_audio)

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
