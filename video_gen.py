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

TITLE_BOTTOM_GAP = 192  # Default; overridden per-video by LayoutProfile

import hashlib as _hashlib

def _generate_layout_profile(headline):
    """
    YPP Compliance: Generate a deterministic-but-unique visual layout
    for each video based on its headline hash. This breaks the
    'template fingerprint' that YouTube's Inauthentic Content policy flags.
    """
    seed = int(_hashlib.md5(headline.encode()).hexdigest(), 16)
    rng = random.Random(seed)

    # Gradient
    gradient_height_pct = rng.uniform(0.40, 0.50)          # 40-50% (was fixed 45%)
    gradient_position = rng.choice(["bottom", "top"])       # was always bottom

    # Title box
    title_bottom_gap = rng.randint(165, 220)                # was fixed 192px

    # Particles
    particle_style = rng.choice(["bokeh", "digital", "stars", "digital_rain", "lens_dust"])

    # Progress bar
    progress_bar_height = rng.randint(4, 8)                 # was fixed 6px
    progress_bar_position = rng.choice(["bottom", "top"])    # was always bottom

    # Hook transition
    hook_transition_time = rng.uniform(3.5, 5.0)            # was fixed 4.2s

    # Avatar horizontal offset
    avatar_x_offset = rng.randint(-60, 60)                  # was always centered

    # Subtitle Y jitter
    subtitle_y_jitter = rng.randint(-30, 30)                # was fixed 0

    # CTA end card style
    cta_variant = rng.randint(0, 3)                         # 4 CTA styles
    cta_pill_colors = [
        (204, 255, 0),    # Neon green (original)
        (0, 200, 255),    # Cyan
        (255, 100, 100),  # Coral
        (180, 130, 255),  # Lavender
    ]
    cta_pill_color = cta_pill_colors[cta_variant]
    cta_headlines = [
        "Full {topic} guide + source code",
        "Get the complete {topic} breakdown",
        "{topic} implementation playbook",
        "Deep dive: {topic} explained",
    ]
    cta_headline_template = cta_headlines[cta_variant]
    cta_descriptions = [
        "Join the community 🚀",
        "Free access — link in bio 📥",
        "Grab it before it's gone ⚡",
        "Level up your stack 🔧",
    ]
    cta_description = cta_descriptions[cta_variant]

    profile = {
        "gradient_height_pct": gradient_height_pct,
        "gradient_position": gradient_position,
        "title_bottom_gap": title_bottom_gap,
        "particle_style": particle_style,
        "progress_bar_height": progress_bar_height,
        "progress_bar_position": progress_bar_position,
        "hook_transition_time": hook_transition_time,
        "avatar_x_offset": avatar_x_offset,
        "subtitle_y_jitter": subtitle_y_jitter,
        "cta_pill_color": cta_pill_color,
        "cta_headline_template": cta_headline_template,
        "cta_description": cta_description,
    }
    print(f"🎲 Layout Profile: gradient={gradient_position}@{gradient_height_pct:.0%}, "
          f"particles={particle_style}, title_gap={title_bottom_gap}px, "
          f"progress={progress_bar_position}@{progress_bar_height}px, "
          f"avatar_offset={avatar_x_offset}px, cta_variant={cta_variant}")
    return profile

import cv2

def apply_tech_grade(frame):
    """
    Applies a premium cinematic color grading to the background image:
    1. Contrast enhancement via an S-curve.
    2. Split-toning: cool teal/blue in shadows, warm orange/gold in highlights.
    """
    # Convert to float32 in [0, 1]
    arr = frame.astype(np.float32) / 255.0
    
    # 1. S-curve contrast boost: f(x) = 3x^2 - 2x^3
    arr = 3 * (arr ** 2) - 2 * (arr ** 3)
    
    # 2. Split toning based on luminance
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    lum = np.expand_dims(lum, axis=2) # Shape: (H, W, 1)
    
    shadow_mask = np.clip(1.0 - lum, 0, 1)
    highlight_mask = np.clip(lum, 0, 1)
    
    # Cool shadows: slight boost to Blue, minor boost to Green
    arr[:, :, 2] += shadow_mask[:, :, 0] * 0.04  # Blue
    arr[:, :, 1] += shadow_mask[:, :, 0] * 0.01  # Green
    
    # Warm highlights: slight boost to Red, drop Blue
    arr[:, :, 0] += highlight_mask[:, :, 0] * 0.05  # Red
    arr[:, :, 1] += highlight_mask[:, :, 0] * 0.02  # Green
    arr[:, :, 2] -= highlight_mask[:, :, 0] * 0.02  # Reduce Blue
    
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8).astype(np.uint8)

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

def _prepare_screenshot_canvas(img, target_w, target_h, url=None):
    """
    Creates a premium 'Blurred Backdrop' canvas for wide screenshots.
    - Longform (16:9): Clean frosted glass card with rounded corners + drop shadow (no browser mockup)
    - Shorts (9:16): Elegant macOS dark-mode browser mockup with controls, outline, and domain name.
    Also automatically crops the top 80px of captured screenshots for longform to remove any browser chrome.
    """
    is_longform = target_w > target_h

    # 1. Automatic chrome removal (crop top 80px of the screenshot) for longform
    if is_longform and img.height > 160:
        img = img.crop((0, 80, img.width, img.height))

    # Create heavily blurred background (scaled to fill)
    bg = ImageOps.fit(img, (target_w, target_h), Image.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=45))
    bg = bg.point(lambda p: p * 0.45) # Darken backdrop
    
    # 2. Fit screenshot image
    iw, ih = img.size
    bar_h = 0 if is_longform else 50
    
    # Calculate scale so that the screenshot PLUS the top bar fits nicely in the safe area
    scale = min(target_w * 0.86 / iw, (target_h * 0.86 - bar_h) / ih)
    fw, fh = int(iw * scale), int(ih * scale)
    
    # Resize original screenshot
    fg_resized = img.resize((fw, fh), Image.LANCZOS).convert("RGBA")
    
    # Overall window dimensions
    win_w = fw
    win_h = fh + bar_h
    
    # 3. Create the window image
    browser_win = Image.new("RGBA", (win_w, win_h), (0, 0, 0, 0))
    b_draw = ImageDraw.Draw(browser_win)
    
    if not is_longform:
        # Draw dark browser top bar background (Shorts only)
        bar_color = (30, 31, 33, 255) # Dark charcoal
        b_draw.rounded_rectangle([0, 0, win_w, bar_h + 20], radius=16, fill=bar_color)
        
        # Draw macOS window controls
        dot_radius = 5
        dot_y = bar_h // 2
        b_draw.ellipse([20 - dot_radius, dot_y - dot_radius, 20 + dot_radius, dot_y + dot_radius], fill=(255, 95, 87, 255))
        b_draw.ellipse([36 - dot_radius, dot_y - dot_radius, 36 + dot_radius, dot_y + dot_radius], fill=(254, 188, 46, 255))
        b_draw.ellipse([52 - dot_radius, dot_y - dot_radius, 52 + dot_radius, dot_y + dot_radius], fill=(40, 200, 64, 255))
        
        # Draw address bar
        addr_w = min(400, int(win_w * 0.65))
        addr_x1 = (win_w - addr_w) // 2
        addr_x2 = addr_x1 + addr_w
        addr_y1 = 10
        addr_y2 = bar_h - 10
        b_draw.rounded_rectangle([addr_x1, addr_y1, addr_x2, addr_y2], radius=8, fill=(45, 46, 49, 255))
        
        # Parse domain name from URL
        from urllib.parse import urlparse
        domain = "techcrunch.com"
        if url:
            try:
                domain = urlparse(url).netloc
                if domain.startswith("www."):
                    domain = domain[4:]
            except Exception:
                pass
        if not domain:
            domain = "techcrunch.com"
            
        # Draw lock icon and domain text in address bar
        addr_font = gf(14, bold=False)
        domain_text = f"🔒  {domain}"
        b_draw.text((win_w // 2, bar_h // 2), domain_text, font=addr_font, fill=(180, 180, 182, 255), anchor="mm")
        
        # Paste resized screenshot onto browser window below the bar
        browser_win.paste(fg_resized, (0, bar_h), fg_resized)
    else:
        # Longform: no browser controls, just paste the screenshot directly
        browser_win.paste(fg_resized, (0, 0), fg_resized)
    
    # 4. Crop the entire window to rounded rectangle corners
    mask = Image.new("L", (win_w, win_h), 0)
    m_draw = ImageDraw.Draw(mask)
    m_draw.rounded_rectangle([0, 0, win_w, win_h], radius=16, fill=255)
    
    rounded_browser = Image.new("RGBA", (win_w, win_h), (0,0,0,0))
    rounded_browser.paste(browser_win, (0,0), mask=mask)
    
    # 5. Draw a sleek outline around the rounded window
    r_draw = ImageDraw.Draw(rounded_browser)
    if is_longform:
        # Subtle frosted glass border outline for longform card
        r_draw.rounded_rectangle([0, 0, win_w, win_h], radius=16, outline=(255, 255, 255, 40), width=3)
    else:
        # Electric Cyan outline for Shorts browser mockup
        r_draw.rounded_rectangle([0, 0, win_w, win_h], radius=16, outline=(0, 240, 255, 120), width=3)
    
    # 6. Add beautiful drop shadow to the rounded window
    shadow_pad = 30
    shadow_img = Image.new("RGBA", (win_w + shadow_pad*2, win_h + shadow_pad*2), (0,0,0,0))
    s_draw = ImageDraw.Draw(shadow_img)
    s_draw.rounded_rectangle([shadow_pad, shadow_pad, win_w+shadow_pad, win_h+shadow_pad], radius=16, fill=(0,0,0,200))
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=20))
    
    # 7. Composite
    canvas = bg.convert("RGBA")
    canvas.paste(shadow_img, ((target_w - shadow_img.width)//2, (target_h - shadow_img.height)//2), shadow_img)
    canvas.paste(rounded_browser, ((target_w - win_w)//2, (target_h - win_h)//2), rounded_browser)
    
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
# Phase 3: Added crash_zoom and drift_parallax for higher visual energy
KB_PATTERNS = ["smooth_zoom", "reveal_zoom", "z_pan_high_energy", "z_pan_subtle", "crash_zoom", "drift_parallax"]

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
        elif pattern == "crash_zoom":
            # Phase 3: Fast 1.3x zoom-in with slight tilt (HIGH ENERGY)
            current_scale = 1.0 + (0.30 * eased_t)
            angle = -0.5 + (1.0 * eased_t)
        elif pattern == "drift_parallax":
            # Phase 3: Slow lateral drift with gentle zoom (CINEMATIC)
            current_scale = 1.05 + (0.08 * eased_t)
            cx = pw // 2 + int(-60 + 120 * eased_t)
            cy = ph // 2 + int(10 * math.sin(eased_t * math.pi))
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


# ── PHASE 3: CINEMATIC TRANSITIONS ─────────────────────────────────────────
def _create_transition_clip(transition_type, duration=0.2):
    """
    Creates a brief transition clip between visual chunks for retention.
    Types: whip_pan, zoom_punch, flash_cut, glitch
    """
    from config import ENABLE_CINEMATIC_TRANSITIONS
    if not ENABLE_CINEMATIC_TRANSITIONS:
        return None
    
    trans_dur = min(duration, 0.3)  # Max 300ms
    
    if transition_type == "flash_cut":
        # 2-frame white flash
        def make_frame(t):
            progress = t / max(trans_dur, 0.01)
            brightness = int(255 * (1.0 - progress))  # Flash then fade
            return np.full((FRAME_H, FRAME_W, 3), brightness, dtype=np.uint8)
        return VideoClip(make_frame, duration=trans_dur)
    
    elif transition_type == "glitch":
        # RGB channel offset effect
        def make_frame(t):
            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            progress = t / max(trans_dur, 0.01)
            offset = int(15 * (1.0 - progress))
            # Red channel shifted right, Blue shifted left
            frame[offset:, :, 0] = 80  # Red shifted down
            frame[:FRAME_H-offset, :, 2] = 80  # Blue shifted up
            # Random scan lines
            for y in range(0, FRAME_H, random.randint(20, 60)):
                frame[y:y+2, :, :] = 200
            return frame
        return VideoClip(make_frame, duration=trans_dur)
    
    elif transition_type == "zoom_punch":
        # Quick 1.1x zoom burst
        def make_frame(t):
            progress = t / max(trans_dur, 0.01)
            # Dark frame with a bright center expanding
            frame = np.full((FRAME_H, FRAME_W, 3), 5, dtype=np.uint8)
            radius = int(FRAME_W * 0.3 * progress)
            cv2.circle(frame, (FRAME_W//2, FRAME_H//2), max(1, radius), (30, 30, 40), -1)
            return frame
        return VideoClip(make_frame, duration=trans_dur)
    
    elif transition_type == "whip_pan":
        # Horizontal motion blur
        def make_frame(t):
            progress = t / max(trans_dur, 0.01)
            frame = np.full((FRAME_H, FRAME_W, 3), 10, dtype=np.uint8)
            # Horizontal streaks
            for y in range(0, FRAME_H, 3):
                brightness = int(40 * (1.0 - abs(progress - 0.5) * 2))
                frame[y:y+1, :, :] = brightness
            return frame
        return VideoClip(make_frame, duration=trans_dur)
    
    return None


def get_transition_type_for_chunk(chunk_idx, retention_map, total_chunks):
    """
    Phase 3: Selects the appropriate transition type based on the retention_map.
    Maps pattern interrupt types to visual transitions.
    """
    from config import ENABLE_CINEMATIC_TRANSITIONS
    if not ENABLE_CINEMATIC_TRANSITIONS or not retention_map:
        return "crossfade"  # Default
    
    pattern_interrupts = retention_map.get("pattern_interrupts", [])
    
    for pi in pattern_interrupts:
        pi_word = pi.get("at_word", 0)
        pi_type = pi.get("type", "")
        estimated_chunk = pi_word // max(1, 170 // total_chunks)
        
        if abs(chunk_idx - estimated_chunk) <= 1:
            # Map retention event type to visual transition
            if pi_type in ["contradiction", "emotional_pivot"]:
                return "flash_cut"
            elif pi_type in ["stat_bomb", "number"]:
                return "zoom_punch"
            elif pi_type in ["rhetorical_question", "direct_address"]:
                return "whip_pan"
            else:
                return "glitch"
    
    # Default: alternate between crossfade and subtle transitions
    if chunk_idx % 5 == 0:
        return "zoom_punch"
    elif chunk_idx % 7 == 0:
        return "flash_cut"
    
    return "crossfade"


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
def _dual_directional_gradient_clip(duration, top_pct=0.32, bottom_pct=0.32):
    """
    Creates a full-screen dual gradient clip: dark at top and bottom,
    fading to transparent in the middle, framing the main visual area.
    """
    arr = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    mask_arr = np.zeros((FRAME_H, FRAME_W), dtype=float)
    
    top_h = int(FRAME_H * top_pct)
    bottom_h = int(FRAME_H * bottom_pct)
    
    for y in range(FRAME_H):
        if y < top_h:
            opacity = ((top_h - y) / top_h) ** 1.2
        elif y > FRAME_H - bottom_h:
            dist_from_bottom = y - (FRAME_H - bottom_h)
            opacity = (dist_from_bottom / bottom_h) ** 1.2
        else:
            opacity = 0.0
        mask_arr[y, :] = opacity
        
    clip = ImageClip(arr, duration=duration)
    mask = VideoClip(lambda t: mask_arr, is_mask=True, duration=duration)
    return clip.with_mask(mask)


def _gradient_clip(duration, height_pct=0.45, position="bottom", is_longform=False):
    if not is_longform:
        # Dual-directional gradient for vertical Shorts framing
        return _dual_directional_gradient_clip(duration)
    
    # Fallback to original single-sided gradient for landscape long-form
    h = int(FRAME_H * height_pct)
    arr = np.zeros((h, FRAME_W, 3), dtype=np.uint8)
    if position == "top":
        mask_arr = np.array(
            [(int(255 * ((h - y)/h)**0.5),) * FRAME_W for y in range(h)],
            dtype=float) / 255.0
    else:
        mask_arr = np.array(
            [(int(255 * (y/h)**0.5),) * FRAME_W for y in range(h)],
            dtype=float) / 255.0
    clip = ImageClip(arr, duration=duration)
    mask = VideoClip(lambda t: mask_arr, is_mask=True, duration=duration)
    return clip.with_mask(mask).with_position(("center", position))


# ── LAYER 4: Ambient "Obsidian" particles ──────────────────────────────────────
# ── LAYER 4: Ambient "Obsidian" particles ──────────────────────────────────────
def _ambient_particles(duration, accent_color, particle_style="bokeh"):
    n = 35
    random.seed(42)
    
    # Pre-generate particle properties: (x, y, speed, offset, size, streak_len)
    particles = []
    for _ in range(n):
        px = random.uniform(0, FRAME_W)
        py = random.uniform(0, FRAME_H)
        speed = random.uniform(0.15, 0.55) # falling speed
        offset = random.uniform(0, FRAME_H)
        p_size = random.uniform(4, 15) if particle_style in ["bokeh", "lens_dust"] else random.uniform(8, 25)
        streak_len = random.uniform(15, 40)
        particles.append((px, py, speed, offset, p_size, streak_len))

    def make_frame(t):
        scale_down = 4 if particle_style in ["bokeh", "lens_dust"] else 2
        sm_w, sm_h = FRAME_W // scale_down, FRAME_H // scale_down
        img = np.zeros((sm_h, sm_w, 3), dtype=np.uint8)
        
        for px, py, speed, offset, p_size, streak_len in particles:
            # Fall downward: add speed instead of subtract
            y = (py + speed * t * 45 + offset) % FRAME_H
            
            # Horizontal drift for lens_dust
            if particle_style == "lens_dust":
                x = (px + math.sin(t * 0.4 + offset) * 15) % FRAME_W
            else:
                x = px
                
            sm_px = int(x / scale_down)
            sm_py = int(y / scale_down)
            sm_size = max(1, int(p_size / scale_down))
            
            if particle_style == "digital":
                cv2.rectangle(img, (sm_px, sm_py), (sm_px+sm_size, sm_py+max(1, 2//scale_down)), accent_color, -1)
            elif particle_style == "digital_rain":
                sm_streak = max(2, int(streak_len / scale_down))
                cv2.line(img, (sm_px, sm_py), (sm_px, sm_py + sm_streak), accent_color, max(1, 2//scale_down))
            elif particle_style == "stars":
                cv2.circle(img, (sm_px, sm_py), max(1, 2//scale_down), (255, 255, 255), -1)
            else: # bokeh, lens_dust
                cv2.circle(img, (sm_px, sm_py), sm_size, accent_color, -1)
        
        if particle_style not in ["stars", "digital_rain"]:
            blur_size = 19 if particle_style in ["bokeh", "lens_dust"] else 7
            img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
            
        return cv2.resize(img, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)

    def make_mask(t):
        scale_down = 4 if particle_style in ["bokeh", "lens_dust"] else 2
        sm_w, sm_h = FRAME_W // scale_down, FRAME_H // scale_down
        mask = np.zeros((sm_h, sm_w), dtype=np.uint8)
        
        for px, py, speed, offset, p_size, streak_len in particles:
            y = (py + speed * t * 45 + offset) % FRAME_H
            
            if particle_style == "lens_dust":
                x = (px + math.sin(t * 0.4 + offset) * 15) % FRAME_W
            else:
                x = px
                
            sm_px = int(x / scale_down)
            sm_py = int(y / scale_down)
            sm_size = max(1, int(p_size / scale_down))
            
            if particle_style == "digital":
                cv2.rectangle(mask, (sm_px, sm_py), (sm_px+sm_size, sm_py+max(1, 2//scale_down)), 255, -1)
            elif particle_style == "digital_rain":
                sm_streak = max(2, int(streak_len / scale_down))
                cv2.line(mask, (sm_px, sm_py), (sm_px, sm_py + sm_streak), 255, max(1, 2//scale_down))
            elif particle_style == "stars":
                cv2.circle(mask, (sm_px, sm_py), max(1, 2//scale_down), 255, -1)
            else:
                cv2.circle(mask, (sm_px, sm_py), sm_size, 255, -1)
        
        if particle_style not in ["stars", "digital_rain"]:
            blur_size = 19 if particle_style in ["bokeh", "lens_dust"] else 7
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            
        mask_full = cv2.resize(mask, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)
        
        # Bolder opacity for digital rain/digital structures
        max_opacity = 0.25 if particle_style in ["digital_rain", "digital"] else 0.15
        return (mask_full.astype(float) / 255.0) * max_opacity

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
    
    # Pre-render the massive blur for the pulse glow to save CPU time
    pre_glow = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    cv2.circle(pre_glow, (FRAME_W//2, FRAME_H//2), 450, accent_color, -1)
    
    # Scale down, blur, scale up to approximate large Gaussian blur very quickly
    scale = 4
    pre_glow_sm = cv2.resize(pre_glow, (FRAME_W//scale, FRAME_H//scale))
    pre_glow_sm = cv2.GaussianBlur(pre_glow_sm, (39, 39), 0)
    base_glow = cv2.resize(pre_glow_sm, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)
    
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

        # Pulse glow at center (alpha blended)
        pulse = (math.sin(t * 1.2) + 1) / 2
        opacity = 0.02 + 0.03 * pulse # Max 5% opacity
        frame = cv2.addWeighted(frame, 1.0, base_glow, opacity, 0)
        
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
def _title_clip(title, duration, bottom_gap=None):
    if bottom_gap is None:
        bottom_gap = TITLE_BOTTOM_GAP
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
    y_pos = FRAME_H - bottom_gap - box_h
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


# ══════════════════════════════════════════════════════════════════════════════
# LONGFORM RETENTION UPGRADE: New Visual Layers (2026 Premium Spec)
# ══════════════════════════════════════════════════════════════════════════════

# ── IMPROVEMENT #2: Kinetic Metric Pop-Up ─────────────────────────────────────
def _kinetic_metric_popup(metric_text, start_time, accent_color, audio_duration, hold=2.0):
    """
    Punches a key metric/stat onto the screen with an elastic scale-in animation.
    Reference: Vaibhav Sisinty '15x Faster', '3.8 Cr', '90% Cheaper' pop-ups.
    Renders a fullscreen-width horizontal bar at 25% height (16:9) or 35% (9:16).
    """
    if not metric_text or start_time >= audio_duration:
        return None
    dur = min(hold + 0.5, audio_duration - start_time)
    if dur < 0.5:
        return None

    is_landscape = FRAME_W > FRAME_H
    bar_h = 130 if is_landscape else 160
    bar_w = int(FRAME_W * 0.85)

    # Render the metric card
    img = Image.new("RGBA", (bar_w, bar_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Frosted glass background
    draw.rounded_rectangle([0, 0, bar_w, bar_h], radius=20, fill=(0, 0, 0, 200))
    # Accent top/bottom edge lines
    draw.rectangle([0, 0, bar_w, 4], fill=(*accent_color, 255))
    draw.rectangle([0, bar_h - 4, bar_w, bar_h], fill=(*accent_color, 255))

    # Metric text — large, bold, accent colored
    f_metric = gf(80 if is_landscape else 70, bold=True)
    tw, th = ts(metric_text, f_metric)

    # If text is too wide, shrink font
    if tw > bar_w - 80:
        f_metric = gf(60 if is_landscape else 55, bold=True)
        tw, th = ts(metric_text, f_metric)

    tx = (bar_w - tw) // 2
    ty = (bar_h - th) // 2

    # Glow effect behind text
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1)]:
        draw.text((tx + dx, ty + dy), metric_text, font=f_metric, fill=(*accent_color, 100))
    draw.text((tx, ty), metric_text, font=f_metric, fill=(255, 255, 255, 255))

    arr = np.array(img.convert("RGB"))
    mask_arr = np.array(img.split()[3]).astype(float) / 255.0

    def opacity_fn(t):
        # Elastic scale-in: 0→1 in 0.15s, hold, fade out 0.3s
        if t < 0.15:
            return t / 0.15
        elif t > dur - 0.3:
            return max(0, (dur - t) / 0.3)
        return 1.0

    def scale_fn(t):
        if t < 0.15:
            # Elastic bounce: overshoot to 1.15x then settle to 1.0x
            progress = t / 0.15
            return 0.5 + 0.65 * progress - 0.15 * math.sin(progress * math.pi)
        elif t < 0.35:
            # Settle bounce
            settle = (t - 0.15) / 0.2
            return 1.0 + 0.08 * math.exp(-settle * 5) * math.sin(settle * 12)
        return 1.0

    def make_frame(t):
        s = scale_fn(t)
        if abs(s - 1.0) < 0.01:
            return arr
        sw, sh = max(1, int(bar_w * s)), max(1, int(bar_h * s))
        scaled = Image.fromarray(arr).resize((sw, sh), Image.LANCZOS)
        # Center crop back to original size
        cx, cy = (sw - bar_w) // 2, (sh - bar_h) // 2
        cx, cy = max(0, cx), max(0, cy)
        cropped = np.array(scaled)[cy:cy + bar_h, cx:cx + bar_w]
        if cropped.shape[0] < bar_h or cropped.shape[1] < bar_w:
            result = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
            result[:cropped.shape[0], :cropped.shape[1]] = cropped
            return result
        return cropped

    def make_mask(t):
        s = scale_fn(t)
        o = opacity_fn(t)
        if abs(s - 1.0) < 0.01:
            return mask_arr * o
        sw, sh = max(1, int(bar_w * s)), max(1, int(bar_h * s))
        scaled = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize((sw, sh), Image.LANCZOS)
        cx, cy = max(0, (sw - bar_w) // 2), max(0, (sh - bar_h) // 2)
        cropped = np.array(scaled)[cy:cy + bar_h, cx:cx + bar_w].astype(float) / 255.0
        if cropped.shape[0] < bar_h or cropped.shape[1] < bar_w:
            result = np.zeros((bar_h, bar_w), dtype=float)
            result[:cropped.shape[0], :cropped.shape[1]] = cropped
            return result * o
        return cropped * o

    clip = VideoClip(make_frame, duration=dur)
    mclip = VideoClip(make_mask, is_mask=True, duration=dur)

    y_pos = int(FRAME_H * 0.22) if is_landscape else int(FRAME_H * 0.32)
    x_pos = (FRAME_W - bar_w) // 2
    return clip.with_mask(mclip).with_position((x_pos, y_pos)).with_start(start_time)


# ── IMPROVEMENT #3: Visual Progress Tracker (Dot Navigator) ──────────────────
def _longform_progress_dots(fact_timestamps, accent_color, audio_duration):
    """
    Apple keynote-style dot navigator showing which fact is currently active.
    5 dots at the top-center. Active = large accent, upcoming = small gray,
    completed = medium dimmed accent.
    """
    if not fact_timestamps or audio_duration <= 0:
        return []

    total_facts = len(fact_timestamps)
    dot_spacing = 36
    dot_r_active = 8
    dot_r_inactive = 5
    dot_r_done = 6
    total_w = (total_facts - 1) * dot_spacing + dot_r_active * 2
    canvas_w = total_w + 60
    canvas_h = 40

    clips = []

    for i, ft in enumerate(fact_timestamps):
        active_fact = i  # This fact is active during this segment
        start_s = float(ft.get("approx_start_seconds", 0))

        # Duration until next fact or end
        if i + 1 < len(fact_timestamps):
            end_s = float(fact_timestamps[i + 1].get("approx_start_seconds", audio_duration))
        else:
            end_s = audio_duration
        fact_dur = max(0.5, end_s - start_s)

        # Render dot strip for this fact segment
        img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Semi-transparent backing pill
        draw.rounded_rectangle([0, 0, canvas_w, canvas_h], radius=canvas_h // 2, fill=(0, 0, 0, 120))

        for j in range(total_facts):
            cx = 30 + j * dot_spacing
            cy = canvas_h // 2

            if j < active_fact:
                # Completed: medium dimmed accent
                r = dot_r_done
                color = (*accent_color, 140)
            elif j == active_fact:
                # Active: large, bright accent with glow
                r = dot_r_active
                color = (*accent_color, 255)
                # Glow ring
                draw.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3],
                             fill=(*accent_color, 60))
            else:
                # Upcoming: small gray
                r = dot_r_inactive
                color = (120, 120, 130, 180)

            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

        arr = np.array(img.convert("RGB"))
        mask_arr = np.array(img.split()[3]).astype(float) / 255.0

        def make_opacity(t, _dur=fact_dur):
            if t < 0.3:
                return t / 0.3
            elif t > _dur - 0.3:
                return max(0, (_dur - t) / 0.3)
            return 1.0

        clip = VideoClip(lambda t, _a=arr: _a, duration=fact_dur)
        mclip = VideoClip(lambda t, _m=mask_arr, _d=fact_dur: _m * make_opacity(t, _d),
                          is_mask=True, duration=fact_dur)

        x_pos = (FRAME_W - canvas_w) // 2
        y_pos = 20  # Top of screen
        clip = clip.with_mask(mclip).with_position((x_pos, y_pos)).with_start(start_s)
        clips.append(clip)

    return clips


# ── IMPROVEMENT #1: Floating Circular Face-Cam Frame ─────────────────────────
def _apply_circular_facecam_frame(avatar_clip, cur_w, cur_h, accent_color, audio_duration, is_longform=False):
    """
    Wraps the avatar PiP in a premium circular frame with neon glow ring
    and drop shadow. Returns a new composite clip.
    Reference: Vaibhav Sisinty floating circular face-cam.
    """
    # Determine circle diameter (use the smaller dimension)
    diameter = min(cur_w, cur_h)

    # Create circular mask
    circle_mask = np.zeros((cur_h, cur_w), dtype=np.float32)
    cy, cx = cur_h // 2, cur_w // 2
    Y, X = np.ogrid[:cur_h, :cur_w]
    r = diameter // 2 - 4  # Slight inset for border
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    circle_mask[dist <= r] = 1.0
    # Smooth edge (anti-alias)
    edge_band = 3.0
    circle_mask[(dist > r) & (dist <= r + edge_band)] = 1.0 - (dist[(dist > r) & (dist <= r + edge_band)] - r) / edge_band

    # Apply circular mask to the avatar
    mask_clip = VideoClip(lambda t: circle_mask, is_mask=True, duration=audio_duration)
    avatar_clip = avatar_clip.with_mask(mask_clip)

    # Create glow ring overlay
    ring_size = diameter + 16  # Slightly larger than avatar
    ring_img = Image.new("RGBA", (ring_size, ring_size), (0, 0, 0, 0))
    ring_draw = ImageDraw.Draw(ring_img)

    # Outer glow (soft, large)
    for glow_r in range(8, 0, -1):
        alpha = int(25 * (glow_r / 8))
        ring_draw.ellipse([8 - glow_r, 8 - glow_r, ring_size - 8 + glow_r, ring_size - 8 + glow_r],
                          outline=(*accent_color, alpha), width=2)

    # Main border ring
    ring_draw.ellipse([4, 4, ring_size - 4, ring_size - 4],
                      outline=(*accent_color, 200), width=3)
    # Inner subtle white ring
    ring_draw.ellipse([7, 7, ring_size - 7, ring_size - 7],
                      outline=(255, 255, 255, 60), width=1)

    ring_arr = np.array(ring_img.convert("RGB"))
    ring_mask = np.array(ring_img.split()[3]).astype(float) / 255.0

    # Pulsing glow opacity
    def ring_opacity(t):
        pulse = 0.7 + 0.3 * math.sin(t * 2.0)
        return ring_mask * pulse

    ring_clip = VideoClip(lambda t: ring_arr, duration=audio_duration)
    ring_mclip = VideoClip(ring_opacity, is_mask=True, duration=audio_duration)
    ring_clip = ring_clip.with_mask(ring_mclip)

    return avatar_clip, ring_clip, ring_size


# ── IMPROVEMENT #7: Mid-Video Subscribe CTA ──────────────────────────────────
def _mid_video_subscribe_prompt(accent_color, audio_duration):
    """
    Visual-only subscribe prompt that appears at 75% of the video.
    Glassmorphic card with pulsing subscribe button icon.
    Duration: 4 seconds. No spoken CTA — purely visual overlay.
    """
    dur = 4.0
    start_time = audio_duration * 0.75
    if start_time + dur > audio_duration:
        start_time = max(0, audio_duration - dur - 2)

    is_landscape = FRAME_W > FRAME_H
    card_w = int(FRAME_W * 0.55) if is_landscape else int(FRAME_W * 0.85)
    card_h = 160 if is_landscape else 200

    def make_frame(t):
        img = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Glassmorphic background
        draw.rounded_rectangle([0, 0, card_w, card_h], radius=24, fill=(10, 10, 18, 220))
        # Accent border with pulse
        pulse = 1.0 + 0.12 * math.sin(t * 6)
        border_w = max(2, int(3 * pulse))
        draw.rounded_rectangle([0, 0, card_w, card_h], radius=24,
                                outline=(*accent_color, 220), width=border_w)

        # Subscribe icon (red pill button)
        btn_w, btn_h = 220, 50
        btn_x = (card_w - btn_w) // 2
        btn_y = 20
        # Pulse the button
        btn_scale = 1.0 + 0.06 * math.sin(t * 4)
        scaled_w = int(btn_w * btn_scale)
        scaled_h = int(btn_h * btn_scale)
        btn_x_s = (card_w - scaled_w) // 2
        btn_y_s = btn_y - int((scaled_h - btn_h) / 2)

        draw.rounded_rectangle([btn_x_s, btn_y_s, btn_x_s + scaled_w, btn_y_s + scaled_h],
                                radius=scaled_h // 2, fill=(220, 20, 60, 255))
        f_btn = gf(26, bold=True)
        draw.text((card_w // 2, btn_y_s + scaled_h // 2), "🔔 SUBSCRIBE", font=f_btn,
                  fill=(255, 255, 255, 255), anchor="mm")

        # Supporting text
        f_text = gf(24)
        msg = "70% of viewers aren't subscribed yet"
        draw.text((card_w // 2, btn_y_s + scaled_h + 30), msg, font=f_text,
                  fill=(200, 200, 210, 200), anchor="mm")

        f_small = gf(20)
        draw.text((card_w // 2, btn_y_s + scaled_h + 60), "Hit subscribe — you'll thank me later",
                  font=f_small, fill=(160, 160, 170, 180), anchor="mm")

        return np.array(img.convert("RGB"))

    def make_mask(t):
        img = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([0, 0, card_w, card_h], radius=24, fill=(255, 255, 255, 230))

        opacity = 1.0
        if t < 0.4:
            opacity = t / 0.4
        elif t > dur - 0.5:
            opacity = max(0, (dur - t) / 0.5)
        return np.array(img.split()[3]).astype(float) / 255.0 * opacity

    clip = VideoClip(make_frame, duration=dur)
    mclip = VideoClip(make_mask, is_mask=True, duration=dur)

    x_pos = (FRAME_W - card_w) // 2
    y_pos = int(FRAME_H * 0.10) if is_landscape else int(FRAME_H * 0.25)
    return clip.with_mask(mclip).with_position((x_pos, y_pos)).with_start(start_time)


# ── IMPROVEMENT #6: Value Loop Montage (Intro Teaser) ────────────────────────
def _value_loop_montage_clips(script_json, accent_color, audio_duration):
    """
    Generates quick-flash teaser metric cards for the first 8-12 seconds.
    Shows the most shocking stat from each upcoming fact as a rapid montage.
    These overlay ON TOP of the normal intro visuals.
    """
    metric_popups = script_json.get("metric_popups", [])
    if not metric_popups:
        # Fallback: use fact_timestamps topics as teasers
        fact_timestamps = script_json.get("fact_timestamps", [])
        for ft in fact_timestamps[1:4]:  # Facts 2-4 as teasers
            topic = ft.get("topic", "")
            if topic:
                metric_popups.append({
                    "text": topic[:30],
                    "timestamp": 0,
                    "fact_number": ft.get("fact_number", 0)
                })

    if not metric_popups:
        return []

    clips = []
    teaser_dur = 2.5  # Each teaser flash lasts 2.5s
    current_t = 1.0  # Start after 1 second

    for i, mp in enumerate(metric_popups[:4]):
        text = mp.get("text", "")
        if not text or current_t >= audio_duration - 5:
            continue

        # Render teaser card — giant metric text on dark overlay
        card_w, card_h = FRAME_W, int(FRAME_H * 0.35)
        img = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Dark backdrop
        draw.rectangle([0, 0, card_w, card_h], fill=(0, 0, 0, 180))

        # Giant metric text
        f_metric = gf(72 if FRAME_W > FRAME_H else 60, bold=True)
        tw, th = ts(text, f_metric)
        if tw > card_w - 80:
            f_metric = gf(55 if FRAME_W > FRAME_H else 48, bold=True)
            tw, th = ts(text, f_metric)

        tx = (card_w - tw) // 2
        ty = (card_h - th) // 2

        # Accent color glow
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            draw.text((tx + dx, ty + dy), text, font=f_metric, fill=(*accent_color, 120))
        draw.text((tx, ty), text, font=f_metric, fill=(255, 255, 255, 255))

        # Fact number badge
        f_badge = gf(22, bold=True)
        badge_txt = f"COMING UP: FACT {mp.get('fact_number', i + 2)}"
        btw, bth = ts(badge_txt, f_badge)
        draw.rounded_rectangle([tx - 10, ty - bth - 20, tx + btw + 10, ty - 8],
                                radius=8, fill=(*accent_color, 200))
        draw.text((tx, ty - bth - 14), badge_txt, font=f_badge, fill=(255, 255, 255, 255))

        arr = np.array(img.convert("RGB"))
        mask_arr = np.array(img.split()[3]).astype(float) / 255.0

        def opacity_fn(t, _dur=teaser_dur):
            if t < 0.12:
                return t / 0.12  # Snap in
            elif t > _dur - 0.2:
                return max(0, (_dur - t) / 0.2)
            return 1.0

        clip = VideoClip(lambda t, _a=arr: _a, duration=teaser_dur)
        mclip = VideoClip(lambda t, _m=mask_arr, _d=teaser_dur: _m * opacity_fn(t, _d),
                          is_mask=True, duration=teaser_dur)

        y_pos = int(FRAME_H * 0.30)
        clip = clip.with_mask(mclip).with_position(("center", y_pos)).with_start(current_t)
        clips.append(clip)

        current_t += teaser_dur + 0.1  # Small gap between flashes

    return clips


# ── IMPROVEMENT #4: Pattern Interrupt Helpers ────────────────────────────────
def _generate_snap_zoom_interrupts(chunks, audio_duration, interval=5.0):
    """
    Generates snap-zoom timestamps every `interval` seconds.
    These are applied per-frame in make_final_frame.
    Returns a list of timestamps where snap-zooms should occur.
    """
    timestamps = []
    t = interval
    while t < audio_duration - 1.0:
        # Avoid zooming during the very start or end
        timestamps.append(t)
        t += interval
    return timestamps

def _fact_boundary_darkener(fact_timestamps, accent_color, audio_duration):
    """
    Creates brief 0.6s 'chapter boundary' dark overlays at the start of each fact.
    Dims everything to 30% then reveals the new fact badge — creates visual breathing room.
    """
    clips = []
    for i, ft in enumerate(fact_timestamps):
        if i == 0:
            continue  # Skip first fact — intro handles this
        start_s = float(ft.get("approx_start_seconds", 0))
        if start_s >= audio_duration - 2:
            continue
        dur = 0.6

        def make_frame(t):
            return np.full((FRAME_H, FRAME_W, 3), 0, dtype=np.uint8)

        def make_mask(t, _dur=dur):
            # Quick darken then reveal
            if t < 0.15:
                return np.full((FRAME_H, FRAME_W), 0.7 * (t / 0.15), dtype=float)
            elif t > _dur - 0.25:
                return np.full((FRAME_H, FRAME_W), 0.7 * max(0, (_dur - t) / 0.25), dtype=float)
            return np.full((FRAME_H, FRAME_W), 0.7, dtype=float)

        clip = VideoClip(make_frame, duration=dur)
        mclip = VideoClip(make_mask, is_mask=True, duration=dur)
        clips.append(clip.with_mask(mclip).with_start(start_s))

    return clips


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

def render_shorts_header_bar(title, accent_color=(255, 255, 255), frame_width=1080):
    """Renders a solid black top bar with white and accent colored title text for Shorts."""
    font = gf(54, bold=True)
    draw_temp = ImageDraw.Draw(Image.new('RGBA', (frame_width, 200)))
    
    # Wrap text to fit inside the bar (with 60px padding on each side)
    max_w = frame_width - 120
    words = title.split()
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        w = draw_temp.textlength(test_line, font=font)
        if w > max_w and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        lines.append(" ".join(current_line))
        
    lines = lines[:2] # Max 2 lines to keep header compact
    
    bbox = font.getbbox("Ag")
    line_height = bbox[3] - bbox[1]
    line_spacing = int(line_height * 0.2)
    
    padding_y = 35
    bar_height = padding_y * 2 + len(lines) * line_height + (len(lines) - 1) * line_spacing
    
    img = Image.new('RGBA', (frame_width, FRAME_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Shift down by 3 cm (113 pixels at 96 DPI) to avoid system HUD/notch overlap
    offset_y = 113
    
    # Draw solid black background bar shifted down
    draw.rectangle([0, offset_y, frame_width, offset_y + bar_height], fill=(0, 0, 0, 255))
    
    # Draw centered text with color variant
    for i, line in enumerate(lines):
        words_in_line = line.split()
        if not words_in_line:
            continue
            
        highlight_mask = [False] * len(words_in_line)
        if len(lines) == 1:
            # Highlight the last 1 or 2 words (approx last 30%)
            num_highlight = max(1, len(words_in_line) // 3)
            for idx in range(len(words_in_line) - num_highlight, len(words_in_line)):
                highlight_mask[idx] = True
        else:
            # Highlight the entire second line
            if i == 1:
                highlight_mask = [True] * len(words_in_line)
                
        space_w = draw.textlength(" ", font=font)
        word_widths = [draw.textlength(w, font=font) for w in words_in_line]
        total_line_w = sum(word_widths) + space_w * (len(words_in_line) - 1)
        
        cur_x = (frame_width - total_line_w) // 2
        ty = offset_y + padding_y + i * (line_height + line_spacing)
        
        for idx, word in enumerate(words_in_line):
            w_w = word_widths[idx]
            color = accent_color if highlight_mask[idx] else (255, 255, 255, 255)
            
            # Draw shadow for maximum contrast
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((cur_x + dx, ty + dy), word, font=font, fill=(0, 0, 0, 150))
                
            draw.text((cur_x, ty), word, font=font, fill=color)
            cur_x += w_w + space_w
            
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

def render_dynamic_entity_tags(entities, accent_color, t, audio_duration, frame_width=1080, frame_height=1920, screenshot_intervals=None):
    """Renders small floating tags for various entities dynamically with fade effects."""
    img = Image.new('RGBA', (frame_width, frame_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    scale = frame_width / 1080.0
    
    # Scale parameters
    box_h = int(450 * scale)
    spacing = int(16 * scale)
    start_y = int(293 * scale)
    start_x = int(40 * scale)
    
    font_size_name = int(72 * scale)
    font_size_desc = int(44 * scale)
    
    # Fonts
    try:
        f_name = ImageFont.truetype('assets/fonts/Roboto-Regular.ttf', font_size_name)
    except Exception as e:
        print(f"Error loading name font: {e}")
        f_name = ImageFont.load_default()
        
    try:
        f_desc = ImageFont.truetype('assets/fonts/Roboto-Regular.ttf', font_size_desc)
    except Exception as e:
        print(f"Error loading desc font: {e}")
        f_desc = ImageFont.load_default()
    
    num_entities = len(entities)
    if num_entities == 0:
        return img
        
    duration_per_entity = audio_duration / num_entities
    
    screenshot_opacity = 1.0
    if screenshot_intervals:
        for s_start, s_end in screenshot_intervals:
            if s_start <= t <= s_end:
                screenshot_opacity = 0.0
                break
            elif s_start - 0.3 <= t < s_start:
                screenshot_opacity = min(screenshot_opacity, (s_start - t) / 0.3)
            elif s_end < t <= s_end + 0.3:
                screenshot_opacity = min(screenshot_opacity, (t - s_end) / 0.3)
                
    for i, ent in enumerate(entities):
        opacity = 0.0
        # Show each label one by one in sequence based on duration_per_entity
        start = i * duration_per_entity
        end = (i + 1) * duration_per_entity
        
        if start <= t <= end:
            fade_in = min(0.3, duration_per_entity * 0.1)
            fade_out = min(0.5, duration_per_entity * 0.1)
            
            if t - start < fade_in:
                opacity = (t - start) / fade_in
            elif end - t < fade_out:
                opacity = (end - t) / fade_out
            else:
                opacity = 1.0
                
        # Fade out during screenshots to not overlay on the article screenshot
        opacity *= screenshot_opacity
        
        if opacity <= 0.0:
            continue
            
        val = ent.get("name", "Unknown")
        desc = ent.get("description", "")
        logo_img = ent.get("pil_logo")
        
        # Calculate dynamic box width
        name_w = draw.textlength(val, font=f_name)
        desc_w = draw.textlength(desc, font=f_desc) if desc else 0
        max_text_w = max(name_w, desc_w)
        
        logo_w, logo_h = 0, 0
        if logo_img:
            # Scale logo to fit nicely within height of 350
            aspect = logo_img.width / logo_img.height
            logo_h = int(350 * scale)
            logo_w = int(logo_h * aspect)
            
        content_x_start = int(40 * scale)
        padding_after_text = int(40 * scale)
        logo_space = (logo_w + int(32 * scale)) if logo_img else 0
        
        # Bare minimum box width required
        min_box_w = content_x_start + logo_space + max_text_w + padding_after_text
        
        # Max allowed box width spanning till the right side
        max_box_w = frame_width - (start_x * 2)
        
        box_w = min(min_box_w, max_box_w)
            
        curr_y = start_y
        
        # Create a temp surface for the individual tag to apply opacity
        tag_img = Image.new('RGBA', (int(box_w + 10), int(box_h + 10)), (0,0,0,0))
        tag_draw = ImageDraw.Draw(tag_img)
        
        content_x = int(40 * scale)
        
        # Paste Logo if available
        if logo_img:
            scaled_logo = logo_img.resize((logo_w, logo_h), Image.LANCZOS)
            logo_x = int(content_x)
            logo_y = int((box_h - logo_h) // 2)
            tag_img.paste(scaled_logo, (logo_x, logo_y), scaled_logo)
            content_x += logo_w + int(32 * scale)
            
        # Draw Text (Floating without a box, so we add a stroke for contrast)
        stroke_w = int(2 * scale)
        stroke_c = (0, 0, 0, 200)
        
        if desc:
            name_y = box_h // 2 - int(35 * scale)
            desc_y = box_h // 2 + int(35 * scale)
            tag_draw.text((content_x, name_y), val, font=f_name, fill=(255, 255, 255, 255), stroke_width=stroke_w, stroke_fill=stroke_c, anchor="lm")
            tag_draw.text((content_x, desc_y), desc, font=f_desc, fill=(220, 220, 220, 255), stroke_width=stroke_w, stroke_fill=stroke_c, anchor="lm")
        else:
            tag_draw.text((content_x, box_h // 2), val, font=f_name, fill=(255, 255, 255, 255), stroke_width=stroke_w, stroke_fill=stroke_c, anchor="lm")
            
        # Apply opacity to the tag image
        if opacity < 1.0:
            r, g, b, a = tag_img.split()
            a = a.point(lambda p: int(p * opacity))
            tag_img = Image.merge('RGBA', (r, g, b, a))
            
        # Paste the tag onto the main frame image
        img.alpha_composite(tag_img, (start_x - 2, curr_y - 2))
        
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

def _mix_and_master_audio(voice_path, bgm_path, sfx_cues, chunks, retention_hooks, output_duration, bgm_volume_config, output_path, fact_timestamps=None, retention_map=None):
    """
    Composes and masters the entire video soundtrack using Pydub:
    1. Import Voiceover, apply professional leveler.
    2. Import BGM, loop, and apply smooth sidechain-compression (ducking) and dramatic silence beats.
    3. Synthesize environmental room tone (low atmospheric hum).
    4. Inject and mix SFX cues (woosh, pop, glitch) precisely.
    5. Master the final composite output (limit peaks, normalize to -1dB).
    """
    import math
    import numpy as np
    from pydub import AudioSegment
    
    print("🎙️ Starting Pydub Audio Mastering Engine...")
    
    # 1. Load Voiceover
    voice = AudioSegment.from_file(voice_path).set_frame_rate(44100).set_channels(2)
    target_duration_ms = int(output_duration * 1000)
    
    # Copy sfx_cues to prevent mutation and dynamically inject fact transitions
    sfx_cues = list(sfx_cues) if sfx_cues is not None else []
    if fact_timestamps:
        for ft in fact_timestamps:
            start_s = float(ft.get("approx_start_seconds", 0))
            if start_s > 2.0 and start_s < output_duration:
                sfx_cues.append({
                    "type": "glitch",
                    "timestamp": start_s
                })
                
    # ── PHASE 4: STRATEGIC SFX LAYER (Auto-inject at pattern interrupts) ──
    from config import ENABLE_STRATEGIC_SFX
    if ENABLE_STRATEGIC_SFX and retention_map:
        pi_timestamps = retention_map.get("pattern_interrupts", [])
        for pi in pi_timestamps:
            pi_word = pi.get("at_word", 0)
            pi_type = pi.get("type", "contradiction")
            # Estimate timestamp from word position
            pi_time_s = pi_word / 3.0
            
            # Map pattern interrupt type to a sound effect
            if pi_type in ["contradiction", "stat_bomb"]:
                sfx_type = "glitch"
            elif pi_type in ["rhetorical_question", "direct_address"]:
                sfx_type = "pop"
            else:
                sfx_type = "woosh"
                
            if pi_time_s < output_duration:
                sfx_cues.append({
                    "type": sfx_type,
                    "timestamp": pi_time_s
                })
    
    # 2. Load and Prepare BGM
    ducked_bgm = AudioSegment.silent(duration=target_duration_ms, frame_rate=44100).set_channels(2)
    if bgm_path and os.path.exists(bgm_path) and os.path.getsize(bgm_path) > 0:
        try:
            bgm = AudioSegment.from_file(bgm_path).set_frame_rate(44100).set_channels(2)
            # Loop BGM to cover the entire target duration
            looped_bgm = AudioSegment.empty()
            while len(looped_bgm) < target_duration_ms:
                looped_bgm += bgm
            looped_bgm = looped_bgm[:target_duration_ms]
            
            # --- Sidechain Compression & Hook Silencing ---
            step_ms = 20
            n_steps = target_duration_ms // step_ms
            
            # Identify when speaking (speech activity envelope)
            speech_active = np.zeros(n_steps)
            for c in chunks:
                start_step = max(0, int(c["start"] * 1000 / step_ms))
                end_step = min(n_steps - 1, int(c["end"] * 1000 / step_ms))
                speech_active[start_step:end_step + 1] = 1.0
                
            # Identify dramatic silence beats (from retention cues and fact boundaries)
            silence_active = np.ones(n_steps)
            for cue in retention_hooks:
                if cue.get("effect") in ("zoom_snap", "flash_accent"):
                    cue_t = float(cue.get("timestamp", 0))
                    # 0.4 seconds of silence around cue
                    start_sil = max(0, int((cue_t - 0.2) * 1000 / step_ms))
                    end_sil = min(n_steps - 1, int((cue_t + 0.2) * 1000 / step_ms))
                    silence_active[start_sil:end_sil + 1] = 0.0
                    
            if fact_timestamps:
                for ft in fact_timestamps:
                    start_s = float(ft.get("approx_start_seconds", 0))
                    if start_s > 2.0:
                        # 0.3s silence before the fact starts
                        start_sil = max(0, int((start_s - 0.3) * 1000 / step_ms))
                        end_sil = min(n_steps - 1, int(start_s * 1000 / step_ms))
                        silence_active[start_sil:end_sil + 1] = 0.0
                    
            # Compute attack/release curves for sidechain compression
            duck_envelope = np.zeros(n_steps)
            alpha_attack = 1.0 - math.exp(-step_ms / 80.0)    # 80ms fast attack to duck music
            alpha_release = 1.0 - math.exp(-step_ms / 500.0)  # 500ms release to bring music back
            
            current_duck = 0.0
            for i in range(n_steps):
                target_duck = speech_active[i]
                if target_duck > current_duck:
                    current_duck += (target_duck - current_duck) * alpha_attack
                else:
                    current_duck += (target_duck - current_duck) * alpha_release
                duck_envelope[i] = current_duck
                
            # Compute smooth attack/release for dramatic silence beats to avoid clicks/pops
            smooth_silence = np.ones(n_steps)
            alpha_sil_attack = 1.0 - math.exp(-step_ms / 30.0)   # 30ms sudden cut
            alpha_sil_release = 1.0 - math.exp(-step_ms / 120.0) # 120ms restore
            
            current_sil = 1.0
            for i in range(n_steps):
                target_sil = silence_active[i]
                if target_sil < current_sil:
                    current_sil += (target_sil - current_sil) * alpha_sil_attack
                else:
                    current_sil += (target_sil - current_sil) * alpha_sil_release
                    
                smooth_silence[i] = current_sil
                
            # Apply ducking envelope in 20ms steps
            bgm_chunks = []
            base_bgm_gain_db = 20 * math.log10(bgm_volume_config) if bgm_volume_config > 0 else -100.0
            
            for i in range(n_steps):
                chunk_start = i * step_ms
                chunk_end = (i + 1) * step_ms
                bgm_chunk = looped_bgm[chunk_start:chunk_end]
                
                duck_factor = duck_envelope[i]
                sil_factor = smooth_silence[i]
                
                # Interpolate volume multiplier: 1.2x (unducked) to 0.25x (ducked)
                vol_multiplier = (1.2 * (1.0 - duck_factor) + 0.25 * duck_factor) * sil_factor
                
                # ── PHASE 4: DYNAMIC BGM ENERGY CURVE ────────────────────────
                # BGM follows Hook→Body→Payoff→CTA arc instead of flat volume
                from config import ENABLE_DYNAMIC_BGM_CURVE
                progress_ratio = i / max(1, n_steps)
                
                if ENABLE_DYNAMIC_BGM_CURVE:
                    if progress_ratio < 0.05:
                        # Hook zone (0-5%): Higher energy to match opening
                        energy_mult = 1.4
                    elif progress_ratio < 0.80:
                        # Body zone (5-80%): Lower, voice-focused
                        energy_mult = 0.85
                    elif progress_ratio < 0.92:
                        # Payoff zone (80-92%): Build back up for climax
                        ramp_prog = (progress_ratio - 0.80) / 0.12
                        energy_mult = 0.85 + (0.40 * ramp_prog)  # 0.85 → 1.25
                    else:
                        # CTA zone (92-100%): Drop for authority
                        energy_mult = 0.6
                    vol_multiplier *= energy_mult
                else:
                    # --- LEGACY: INTENSITY RAMP FOR LONGFORM (Last 30%) ---
                    if progress_ratio > 0.7:
                        ramp_factor = 1.0 + 0.45 * ((progress_ratio - 0.7) / 0.3)
                        vol_multiplier *= ramp_factor
                
                # ── PHASE 4: PATTERN INTERRUPT DUCKING ────────────────────────
                # Duck BGM at retention pattern interrupt timestamps for impact
                from config import ENABLE_STRATEGIC_SFX
                if ENABLE_STRATEGIC_SFX and retention_map:
                    # retention_map contains pattern_interrupt info
                    pi_timestamps = retention_map.get("pattern_interrupts", [])
                    for pi in pi_timestamps:
                        pi_word = pi.get("at_word", 0)
                        # Estimate timestamp from word position (≈3 words/sec in 170-word/58s script)
                        pi_time_ms = int((pi_word / 3.0) * 1000)
                        # Duck window: 250ms before to 250ms after
                        if pi_time_ms - 250 <= chunk_start <= pi_time_ms + 250:
                            vol_multiplier *= 0.15  # Deep duck for impact
                            break
                
                if vol_multiplier < 0.0001 or base_bgm_gain_db < -90.0:
                    gain_db = -100.0
                else:
                    gain_db = base_bgm_gain_db + 20 * math.log10(vol_multiplier)
                    
                bgm_chunks.append(bgm_chunk.apply_gain(gain_db))
                
            ducked_bgm = bgm_chunks[0]
            for c in bgm_chunks[1:]:
                ducked_bgm += c
                
            # Append remaining milliseconds if any
            rem_ms = target_duration_ms % step_ms
            if rem_ms > 0:
                rem_chunk = looped_bgm[target_duration_ms - rem_ms:]
                duck_factor = duck_envelope[-1]
                sil_factor = smooth_silence[-1]
                vol_multiplier = (1.2 * (1.0 - duck_factor) + 0.25 * duck_factor) * sil_factor
                
                progress_ratio = 1.0
                ramp_factor = 1.45
                vol_multiplier *= ramp_factor
                
                gain_db = base_bgm_gain_db + 20 * math.log10(max(0.0001, vol_multiplier))
                ducked_bgm += rem_chunk.apply_gain(gain_db)
                
            # Fade out BGM during last 2 seconds
            ducked_bgm = ducked_bgm.fade_out(2000)
            print("   🎵 Topic-Aware BGM looped and processed with sidechain compression + fact boundary ducking.")
        except Exception as e:
            print(f"   ⚠️ BGM processing failed: {e}")
            
    # 3. Synthesize Room Tone (atmospheric background noise)
    room_tone = AudioSegment.silent(duration=target_duration_ms, frame_rate=44100).set_channels(2)
    try:
        duration_s = target_duration_ms / 1000.0
        samples = np.random.normal(0, 0.01, int(44100 * duration_s))
        samples = np.convolve(samples, np.ones(120)/120, mode='same')  # Moving average low-pass filter
        samples_int16 = (samples * 32767).astype(np.int16)
        
        # Load into AudioSegment
        raw_room_tone = AudioSegment(
            samples_int16.tobytes(),
            frame_rate=44100,
            sample_width=2,
            channels=1
        ).set_channels(2)
        
        # Reduce volume to -36dB for a very subtle room tone
        room_tone = raw_room_tone - 36
        print("   🏠 Environmental Room Tone synthesized.")
    except Exception as e:
        print(f"   ⚠️ Room Tone synthesis failed: {e}")
        
    # Combine layers: Room Tone + Ducked BGM + Voiceover
    composite = room_tone.overlay(ducked_bgm)
    composite = composite.overlay(voice, position=0)
    
    # 4. Mix SFX cues
    sfx_count = 0
    for cue in sfx_cues:
        ctype = cue.get("type", "woosh").lower()
        cue_ts = float(cue.get("timestamp", 0))
        sfx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sfx", f"{ctype}.wav")
        if os.path.exists(sfx_path) and os.path.getsize(sfx_path) > 0 and cue_ts < output_duration:
            try:
                sfx = AudioSegment.from_file(sfx_path).set_frame_rate(44100).set_channels(2)
                # SFX volume mapping: Woosh is subtle, pops are crispy, glitches are sharp
                if ctype == "woosh":
                    sfx = sfx - 10  # Moderate woosh volume
                elif ctype == "pop":
                    sfx = sfx - 6   # Crispy pop volume
                else:
                    sfx = sfx - 8   # Default volume
                
                pos_ms = int(cue_ts * 1000)
                composite = composite.overlay(sfx, position=pos_ms)
                sfx_count += 1
            except Exception as e:
                print(f"   ⚠️ Failed to load SFX {ctype}: {e}")
                
    # Auto-inject transition Woosh SFX for subtitle transitions (every 3rd sentence)
    auto_sfx_count = 0
    for i, chunk in enumerate(chunks):
        if i % 3 == 0 or i == 0:
            cue_ts = chunk["start"]
            sfx_path_woosh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sfx", "woosh.wav")
            if os.path.exists(sfx_path_woosh) and cue_ts < output_duration:
                try:
                    sfx = AudioSegment.from_file(sfx_path_woosh).set_frame_rate(44100).set_channels(2)
                    # Very subtle woosh for transitions
                    sfx = sfx - 18
                    pos_ms = int(cue_ts * 1000)
                    composite = composite.overlay(sfx, position=pos_ms)
                    auto_sfx_count += 1
                except:
                    pass
                    
    print(f"   🔊 Mixed {sfx_count} explicit SFX cues + {auto_sfx_count} auto-transition wooshes.")
    
    # 5. Master Output (Normalize to -1.0dB headroom to prevent clipping)
    from pydub.effects import normalize
    mastered = normalize(composite, headroom=1.0)
    
    # Export final master wav
    mastered.export(output_path, format="wav")
    print(f"⭐ Audio mastering complete! Output saved to: {output_path}")

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
    Shows the screenshot from second 4.0 to 12.0 (8 seconds duration) with fade-in and fade-out.
    """
    if not screenshot_path or not os.path.exists(screenshot_path):
        return []
        
    try:
        raw_img = Image.open(screenshot_path)
        canvas_img = _prepare_screenshot_canvas(raw_img, FRAME_W, FRAME_H)
        canvas_arr = np.array(canvas_img.convert("RGB"))
        
        # Show from 4.0 to 12.0 (8s duration)
        start_time = 4.0
        # Cap end_time at duration - 2.0 to avoid overlapping with outro/CTA card
        end_time = min(12.0, duration - 2.0)
        clip_dur = end_time - start_time
        
        if clip_dur < 1.0:
            return []
            
        screenshot_clip = ImageClip(canvas_arr).with_duration(clip_dur).with_start(start_time)
        
        # Subtle Ken Burns effect (zoom in slightly over the duration)
        screenshot_clip = screenshot_clip.resized(lambda t, cd=clip_dur: 1.0 + 0.08 * (t / cd))
        
        # Smooth fade-in and fade-out (0.3s)
        screenshot_clip = screenshot_clip.with_effects([
            vfx.CrossFadeIn(0.3),
            vfx.CrossFadeOut(0.3)
        ])
        
        return [screenshot_clip]
    except Exception as e:
        print(f"⚠️ Error creating article screenshot clip: {e}")
        return []

def _longform_article_screenshot_clips(script_json, audio_duration):
    """
    For long-form videos: Maps the article screenshot of EACH topic to its
    approximate active time window. Shows it at the start of the topic, and
    optionally halfway through, with premium Ken Burns zoom and pan transitions.
    """
    print("🎬 Generating topic-aligned screenshots for long-form compilation... (Disabled repeating overlay)")
    return []


def _longform_topic_transition_clips(script_json, audio_duration):
    """
    Creates stunning animated fullscreen topic transition title cards 
    for each fact segment in longform compilation videos.
    """
    print("🎬 Generating longform topic transition cards...")
    fact_timestamps = script_json.get("fact_timestamps", [])
    topics = script_json.get("longform_topics", [])
    if not fact_timestamps:
        return []
        
    clips = []
    
    for i, ft in enumerate(fact_timestamps):
        fact_num = ft.get("fact_number", i + 1)
        start_s = float(ft.get("approx_start_seconds", 0))
        
        # Don't create transition card for Fact 1 if it's right at the start of the video
        # since we have the main intro hook clip.
        if i == 0 and start_s < 2.0:
            continue
            
        # Get topic headline
        topic_idx = i
        if topic_idx >= len(topics):
            topic_idx = len(topics) - 1
        topic_headline = ""
        if topics:
            topic_headline = topics[topic_idx].get("headline", ft.get("topic", ""))
        else:
            topic_headline = ft.get("topic", "")
            
        if not topic_headline:
            continue
            
        dur = 3.0 # Show card for exactly 3 seconds
        
        try:
            # Create a Pillow image for the card
            img = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # 1. Dark semi-transparent card backdrop
            card_w, card_h = 1200, 450
            card_x1 = (FRAME_W - card_w) // 2
            card_y1 = (FRAME_H - card_h) // 2
            card_x2 = card_x1 + card_w
            card_y2 = card_y1 + card_h
            
            # Semi-transparent glassmorphic background
            draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=24, fill=(10, 12, 18, 225))
            
            # Neon border highlight (Electric Cyan)
            draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=24, outline=(0, 240, 255, 180), width=4)
            
            # 2. Draw "FACT X OF N" tracker
            total_f = script_json.get("num_facts", len(topics) if topics else 10)
            label_text = f"FACT #{fact_num} OF {total_f}"
            label_font = gf(28, bold=True)
            draw.text((FRAME_W // 2, card_y1 + 60), label_text, font=label_font, fill=(0, 240, 255, 255), anchor="mm")
            
            # 3. Draw Headline text
            headline_font = gf(44, bold=True)
            # Wrap headline to fit in the card
            words = topic_headline.split()
            lines = []
            current_line = []
            for w in words:
                test_line = " ".join(current_line + [w])
                bbox = draw.textbbox((0, 0), test_line, font=headline_font)
                if bbox[2] - bbox[0] < card_w - 120:
                    current_line.append(w)
                else:
                    lines.append(" ".join(current_line))
                    current_line = [w]
            if current_line:
                lines.append(" ".join(current_line))
                
            # Limit to 2 lines
            lines = lines[:2]
            
            y_offset = card_y1 + 180
            for line in lines:
                draw.text((FRAME_W // 2, y_offset), line, font=headline_font, fill=(255, 255, 255, 255), anchor="mm")
                y_offset += 75
                
            # Convert to numpy arrays
            arr_rgba = np.array(img.convert("RGBA"))
            arr_rgb = arr_rgba[:, :, :3]
            arr_mask = (arr_rgba[:, :, 3] / 255.0).astype(float)
            
            # Create video clip
            card_clip = ImageClip(arr_rgb, duration=dur)
            mask_clip = VideoClip(lambda t: arr_mask, is_mask=True, duration=dur)
            card_clip = card_clip.with_mask(mask_clip)
            
            # Dynamic entrance animation (zoom-in ease effect)
            card_clip = card_clip.resized(lambda t, d=dur: 0.9 + 0.1 * easeInOutQuad(t / d))
            card_clip = card_clip.with_position("center").with_start(start_s)
            card_clip = card_clip.with_effects([vfx.CrossFadeIn(0.4), vfx.CrossFadeOut(0.4)])
            
            clips.append(card_clip)
            print(f"  🎬 Generated Topic {fact_num} transition card at {start_s}s: {topic_headline[:30]}...")
            
        except Exception as e:
            print(f"⚠️ Error creating transition card for Topic {fact_num}: {e}")
            import traceback
            traceback.print_exc()
            
    return clips

def _evidence_screenshot_clip(evidence_path, duration):
    """
    Shows a secondary 'Evidence' or 'Use Case' screenshot during the analytical section.
    """
    if not evidence_path or not os.path.exists(evidence_path):
        return []
    try:
        img = Image.open(evidence_path).convert("RGB")
        target_h, target_w = FRAME_H, FRAME_W
        canvas = _prepare_screenshot_canvas(img, target_w, target_h, evidence_path)
        
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
    
    # Position: Very Top Right corner (above the shifted title bar for Shorts)
    y = 40
    x = width - tw - 40
    
    # Glassmorphism backing
    rect = [x - 15, y - 8, x + tw + 15, y + th + 8]
    d.rounded_rectangle(rect, radius=8, fill=(0, 0, 0, 80), outline=(255, 255, 255, 40), width=1)
    
    # Semi-transparent text
    d.text((x, y), text, font=font, fill=(255, 255, 255, 140))
    
    return img

def composite_frame(background_frame, timestamp, header_img, subtitle_img, transparency_img=None, entity_tags_img=None):
    """Clean talking-head composite: header + subtitles + entity tags."""
    frame = Image.fromarray(background_frame).convert('RGBA')
    
    # 1. Header at top
    frame.alpha_composite(header_img, dest=(0, 0))
    
    # 2. Transparency Watermark (2026 Compliance)
    if transparency_img is not None:
        frame.alpha_composite(transparency_img, dest=(0, 0))
        
    # 3. Dynamic Entity Tags (Shorts spoken topics)
    if entity_tags_img is not None:
        frame.alpha_composite(entity_tags_img, dest=(0, 0))
    
    # 4. Subtitles
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
    
    is_landscape = frame_width > frame_height
    if is_landscape:
        base_size = int(72 * scale_ratio) # Larger base size for landscape/longform readability
    else:
        base_size = int(58 * scale_ratio) # Reverted to original as requested
    
    f_main = gf(base_size, bold=True)
    
    words = [wd["word"] for wd in word_data]
    word_widths = []
    fake_draw = ImageDraw.Draw(Image.new("RGBA", (1,1)))
    
    for i, wd in enumerate(word_data):
        word_widths.append(fake_draw.textbbox((0,0), words[i], font=f_main)[2] - fake_draw.textbbox((0,0), words[i], font=f_main)[0])
    
    if is_landscape:
        max_sub_width = int(frame_width * 0.65) # Narrower center focus in 16:9
    else:
        max_sub_width = int(frame_width * 0.80) # More narrow for punchy center focus
    lines = wrap_text_to_lines(words, word_widths, max_sub_width, f_main)
    
    line_h = int(90 * scale_ratio) # Reverted to original
    
    # Position: LOWER THIRD
    # 60% Height for Shorts, 80% Height for Landscape
    is_landscape = frame_width > frame_height
    y_pos_pct = 0.80 if is_landscape else 0.60
    start_y = int(frame_height * y_pos_pct) - (len(lines) * line_h // 2) + y_shift
    
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
                f_word = gf(int(base_size * 1.12), bold=True) # Boost size by 12% for premium Pop zoom effect
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
                target_x = int(cur_x - (rotated.width - orig_w)//2)
                target_y = int(line_y - (rotated.height - base_size)//2 + 2)
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

def _generate_lipsync_video(audio_path, face_path=None):
    if face_path is None:
        face_path = os.path.join(ASSETS_DIR, "video", "Firefly_video_final.mp4")
    if not os.path.exists(face_path):
        print(f"{os.path.basename(face_path)} not found in assets. Skipping lip sync.")
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
        """Convert PIL frames to PNG bytes for Gemini Vision using new SDK Part objects."""
        from google.genai import types
        parts = []
        for frac, pil_img in frames:
            # Downscale for API efficiency (max 720px wide)
            w, h = pil_img.size
            if w > 720:
                scale = 720 / w
                pil_img = pil_img.resize((720, int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG', quality=80)
            
            # Use the SDK's direct method for building a Part from bytes
            parts.append(types.Part.from_bytes(
                data=buf.getvalue(),
                mime_type='image/jpeg'
            ))
        return parts

    def _clamp_refinements(self, refinements):
        """Clamp refinement values to safe ranges so Gemini can't break the render."""
        clamped = {}
        for key, (lo, hi) in self.PARAM_RANGES.items():
            if key in refinements:
                try:
                    val = float(refinements[key])
                    clamped_val = max(lo, min(hi, val))
                    if key == "subtitle_y_shift":
                        clamped[key] = int(clamped_val)
                    else:
                        clamped[key] = clamped_val
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
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            image_parts = self._frames_to_bytes(frames)

            param_desc = json.dumps({k: f"range {v}" for k, v in self.PARAM_RANGES.items()})
            
            # Use explicit Part for the text as well
            text_part = types.Part.from_text(text=(
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
            ))

            # Build contents as a list of Part objects
            contents = image_parts + [text_part]

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
        if api_key and iterations < max_iters and script_json:
            try:
                auditor = VisualAuditEngine(api_key)
                feedback = auditor.audit(video_path, script_json.get("script", ""))
                
                if feedback and feedback.get("score", 0) < 8.5:
                    score = feedback.get("score")
                    issues = feedback.get("issues")
                    print(f"🔄 [VIDEO LOOP] Quality: {score}/10. Issues: {issues}")
                    # 2. REFINE
                    refinements = feedback.get("refinement_commands", {})
                    if refinements:
                        dynamic_params.update(refinements)
                        iterations += 1
                        continue
                elif feedback:
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

    # ── YPP COMPLIANCE: Per-Video Layout Randomization ────────────────────
    headline = script_json.get("original_news_headline", script_json.get("title", "Tech News"))
    layout = _generate_layout_profile(headline)
    # Merge layout jitter into subtitle shift
    subtitle_y_shift += layout["subtitle_y_jitter"]

    slot_str = script_json.get("slot", "")
    is_longform = "Slot C" in slot_str or "Slot L" in slot_str or script_json.get("is_longform", False)
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

    # Let's collect all screenshot active intervals early in the main scope
    screenshot_intervals = []
    if is_longform:
        fact_timestamps = script_json.get("fact_timestamps", [])
        topics = script_json.get("longform_topics", [])
        if topics and fact_timestamps:
            for i, ft in enumerate(fact_timestamps):
                start_s = float(ft.get("approx_start_seconds", 0))
                if i + 1 < len(fact_timestamps):
                    end_s = float(fact_timestamps[i + 1].get("approx_start_seconds", audio_duration))
                else:
                    end_s = audio_duration
                fact_dur = max(1.0, end_s - start_s)
                
                first_dur = min(6.0, fact_dur)
                if first_dur >= 1.0:
                    screenshot_intervals.append((start_s, start_s + first_dur))
                    
                if fact_dur > 18.0:
                    sec_start = start_s + 14.0
                    sec_dur = min(5.0, end_s - sec_start)
                    if sec_dur >= 1.0:
                        screenshot_intervals.append((sec_start, sec_start + sec_dur))
    else:
        # For Shorts: screenshot is active from 4.0s to min(12.0s, audio_duration - 2.0)
        start_s = 4.0
        end_s = min(12.0, audio_duration - 2.0)
        if end_s - start_s >= 1.0:
            screenshot_intervals.append((start_s, end_s))

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
        crossfade = 0.4 if not is_longform else 0.6 # Longer crossfade for longform smoothness
        
        # --- LONGFORM 2.5s PACING PATTERN INTERRUPTS ---
        if is_longform:
            clip_dur = 2.5 + crossfade
            num_clips_needed = int(audio_duration // 2.5) + 1
            expanded_visual_paths = []
            while len(expanded_visual_paths) < num_clips_needed:
                expanded_visual_paths.extend(visual_paths)
            expanded_visual_paths = expanded_visual_paths[:num_clips_needed]
            
            current_start = 0.0
            clip_cache = {}
            
            for i, vp in enumerate(expanded_visual_paths):
                try:
                    if vp.endswith(".mp4"):
                        if vp in clip_cache:
                            c_clip = clip_cache[vp].copy()
                        else:
                            c_clip = VideoFileClip(vp).without_audio()
                            clip_cache[vp] = c_clip
                        
                        if c_clip.duration < clip_dur:
                            c_clip = c_clip.with_effects([vfx.Loop(duration=clip_dur)])
                        else:
                            c_clip = c_clip.subclipped(0, clip_dur)
                        
                        # Standard Resize & Crop (adapts to 16:9 based on longform)
                        w, h = c_clip.size
                        # 16:9 landscape crop
                        target_w_crop = int(h * 16 / 9)
                        if target_w_crop <= w:
                            x1 = (w - target_w_crop) // 2
                            c_clip = c_clip.cropped(x1=x1, y1=0, x2=x1 + target_w_crop, y2=h)
                        else:
                            target_h_crop = int(w * 9 / 16)
                            y1 = (h - target_h_crop) // 2
                            c_clip = c_clip.cropped(x1=0, y1=y1, x2=w, y2=y1 + target_h_crop)
                        c_clip = c_clip.resized((FRAME_W, FRAME_H))
                    else:
                        if vp.endswith(".png"):
                            if vp in clip_cache:
                                c_clip = clip_cache[vp].copy()
                            else:
                                try:
                                    raw_img = Image.open(vp)
                                    canvas_img = _prepare_screenshot_canvas(raw_img, FRAME_W, FRAME_H)
                                    canvas_arr = np.array(canvas_img.convert("RGB"))
                                    c_clip = ImageClip(canvas_arr)
                                    clip_cache[vp] = c_clip
                                except Exception as e:
                                    print(f"⚠️ Error preparing screenshot canvas in longform for {vp}: {e}")
                                    c_clip = ImageClip(vp)
                                    clip_cache[vp] = c_clip
                            c_clip = c_clip.with_duration(clip_dur)
                        else:
                            if vp in clip_cache:
                                c_clip = clip_cache[vp].copy()
                            else:
                                c_clip = ImageClip(vp)
                                clip_cache[vp] = c_clip
                            c_clip = c_clip.with_duration(clip_dur)
                            
                            # Standard Resize & Crop (adapts to 16:9 based on longform)
                            w, h = c_clip.size
                            # 16:9 landscape crop
                            target_w_crop = int(h * 16 / 9)
                            if target_w_crop <= w:
                                x1 = (w - target_w_crop) // 2
                                c_clip = c_clip.cropped(x1=x1, y1=0, x2=x1 + target_w_crop, y2=h)
                            else:
                                target_h_crop = int(w * 9 / 16)
                                y1 = (h - target_h_crop) // 2
                                c_clip = c_clip.cropped(x1=0, y1=y1, x2=w, y2=y1 + target_h_crop)
                            c_clip = c_clip.resized((FRAME_W, FRAME_H))
                    
                    if i > 0:
                        retention_map = script_json.get("retention_map", {})
                        if retention_map:
                            trans_type = get_transition_type_for_chunk(i, retention_map, len(expanded_visual_paths))
                            if trans_type == "flash_cut": trans_type = "glitch"
                            elif trans_type == "zoom_punch": trans_type = "zoom"
                            elif trans_type == "whip_pan": trans_type = random.choice(["slide_r", "slide_l"])
                            else: trans_type = random.choice(["zoom", "slide_r", "slide_l", "slide_t", "glitch"])
                        else:
                            trans_type = random.choice(["zoom", "slide_r", "slide_l", "slide_t", "glitch"])
                        
                        if trans_type == "zoom":
                            c_clip = c_clip.with_effects([vfx.CrossFadeIn(crossfade)])
                            c_clip = c_clip.resized(lambda t: 1.3 - (0.3 * min(1, t / crossfade)) if t < crossfade else 1.0)
                        elif "slide" in trans_type:
                            c_clip = c_clip.with_effects([vfx.CrossFadeIn(crossfade * 0.5)])
                            def slide_pos(t):
                                if t > crossfade: return ("center", "center")
                                prog = t / crossfade
                                prog = 1 - (1 - prog)**3 
                                if trans_type == "slide_r": return (int(FRAME_W * (1 - prog)), "center")
                                if trans_type == "slide_l": return (int(-FRAME_W * (1 - prog)), "center")
                                if trans_type == "slide_t": return ("center", int(-FRAME_H * (1 - prog)))
                                return ("center", "center")
                            c_clip = c_clip.with_position(slide_pos)
                        elif trans_type == "glitch":
                            c_clip = c_clip.with_effects([vfx.CrossFadeIn(0.1)])
                            trans_clip = _create_transition_clip("glitch", duration=0.15)
                            if trans_clip:
                                trans_clip = trans_clip.with_start(current_start)
                                logo_clips.append(trans_clip)

                        flash = ColorClip(size=(FRAME_W, FRAME_H), color=(255, 255, 255), duration=0.2).with_opacity(0.6)
                        flash = flash.with_start(current_start).with_effects([vfx.CrossFadeOut(0.15)])
                        logo_clips.append(flash)

                    scale_factor = 1.0 + random.uniform(0.18, 0.25)
                    c_clip = c_clip.resized(lambda t, sf=scale_factor, cd=clip_dur: 1.0 + (sf - 1.0) * (t / cd))
                    c_clip = _apply_handheld_shake(c_clip)
                    
                    is_warm = (i % 2 == 0)
                    def tint_frame(frame, is_w=is_warm):
                        frame_f = frame.astype(np.float32)
                        if is_w:
                            frame_f[:, :, 0] = np.clip(frame_f[:, :, 0] * 1.04 + 3, 0, 255)
                            frame_f[:, :, 1] = np.clip(frame_f[:, :, 1] * 1.01, 0, 255)
                            frame_f[:, :, 2] = np.clip(frame_f[:, :, 2] * 0.96 - 3, 0, 255)
                        else:
                            frame_f[:, :, 0] = np.clip(frame_f[:, :, 0] * 0.96 - 3, 0, 255)
                            frame_f[:, :, 1] = np.clip(frame_f[:, :, 1] * 1.01, 0, 255)
                            frame_f[:, :, 2] = np.clip(frame_f[:, :, 2] * 1.04 + 3, 0, 255)
                        return frame_f.astype(np.uint8)
                    c_clip = c_clip.image_transform(tint_frame)
                    
                    c_clip = c_clip.with_start(current_start)
                    bg_layer_clips.append(c_clip)
                    current_start += (clip_dur - crossfade)
                except Exception as e:
                    print(f"Failed to load background img {vp}: {e}")
        else:
            # --- SHORTS PACING SYNCHRONIZED TO CHUNKS ---
            clip_cache = {}
            for i, chunk in enumerate(chunks):
                vp = chunk.get("visual_path")
                if not vp or not os.path.exists(vp):
                    continue
                
                start_t = chunk["start"]
                end_t = chunk["end"]
                
                is_last = (i == len(chunks) - 1)
                this_crossfade = crossfade if not is_last else 0.0
                clip_dur = (end_t - start_t) + this_crossfade
                
                if clip_dur <= 0.01:
                    continue
                
                try:
                    if vp.endswith(".mp4"):
                        if vp in clip_cache:
                            c_clip = clip_cache[vp].copy()
                        else:
                            c_clip = VideoFileClip(vp).without_audio()
                            clip_cache[vp] = c_clip
                        
                        if c_clip.duration < clip_dur:
                            c_clip = c_clip.with_effects([vfx.Loop(duration=clip_dur)])
                        else:
                            c_clip = c_clip.subclipped(0, clip_dur)
                        
                        # 9:16 portrait crop (Shorts)
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
                    else:
                        if vp.endswith(".png"):
                            if vp in clip_cache:
                                c_clip = clip_cache[vp].copy()
                            else:
                                try:
                                    raw_img = Image.open(vp)
                                    canvas_img = _prepare_screenshot_canvas(raw_img, FRAME_W, FRAME_H)
                                    canvas_arr = np.array(canvas_img.convert("RGB"))
                                    c_clip = ImageClip(canvas_arr)
                                    clip_cache[vp] = c_clip
                                except Exception as e:
                                    print(f"⚠️ Error preparing screenshot canvas for {vp}: {e}")
                                    c_clip = ImageClip(vp)
                                    clip_cache[vp] = c_clip
                            c_clip = c_clip.with_duration(clip_dur)
                        else:
                            if vp in clip_cache:
                                c_clip = clip_cache[vp].copy()
                            else:
                                c_clip = ImageClip(vp)
                                clip_cache[vp] = c_clip
                            c_clip = c_clip.with_duration(clip_dur)
                            
                            # 9:16 portrait crop (Shorts)
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
                    
                    # Apply premium cinematic color grade
                    c_clip = c_clip.image_transform(apply_tech_grade)
                    
                    if i > 0:
                        retention_map = script_json.get("retention_map", {})
                        if retention_map:
                            trans_type = get_transition_type_for_chunk(i, retention_map, len(chunks))
                            if trans_type == "flash_cut": trans_type = "glitch"
                            elif trans_type == "zoom_punch": trans_type = "zoom"
                            elif trans_type == "whip_pan": trans_type = random.choice(["slide_r", "slide_l"])
                            else: trans_type = random.choice(["zoom", "slide_r", "slide_l", "slide_t", "glitch", "morph"])
                        else:
                            trans_type = random.choice(["zoom", "slide_r", "slide_l", "slide_t", "glitch", "morph"])
                        
                        if trans_type == "zoom":
                            c_clip = c_clip.with_effects([vfx.CrossFadeIn(this_crossfade)])
                            c_clip = c_clip.resized(lambda t, cf=this_crossfade: 1.25 - (0.25 * min(1, t / cf)) if t < cf else 1.0)
                        elif trans_type == "morph":
                            c_clip = c_clip.with_effects([vfx.CrossFadeIn(this_crossfade)])
                            c_clip = c_clip.resized(lambda t, cf=this_crossfade: 1.2 - (0.2 * min(1, t / cf)) if t < cf else 1.0)
                            
                            def morph_blur(frame, t, cf=this_crossfade):
                                if t >= cf:
                                    return frame
                                prog = t / cf
                                blur_radius = int(19 * (1.0 - prog))
                                if blur_radius % 2 == 0:
                                    blur_radius += 1
                                if blur_radius > 1:
                                    return cv2.GaussianBlur(frame, (blur_radius, blur_radius), 0)
                                return frame
                            c_clip = c_clip.transform(lambda gf, t: morph_blur(gf(t), t))
                        elif "slide" in trans_type:
                            c_clip = c_clip.with_effects([vfx.CrossFadeIn(this_crossfade * 0.5)])
                            def slide_pos(t, cf=this_crossfade, tt=trans_type):
                                if t > cf: return ("center", "center")
                                prog = t / cf
                                prog = 1 - (1 - prog)**3
                                if tt == "slide_r": return (int(FRAME_W * (1 - prog)), "center")
                                if tt == "slide_l": return (int(-FRAME_W * (1 - prog)), "center")
                                if tt == "slide_t": return ("center", int(-FRAME_H * (1 - prog)))
                                return ("center", "center")
                            c_clip = c_clip.with_position(slide_pos)
                        elif trans_type == "glitch":
                            c_clip = c_clip.with_effects([vfx.CrossFadeIn(0.1)])
                            trans_clip = _create_transition_clip("glitch", duration=0.15)
                            if trans_clip:
                                trans_clip = trans_clip.with_start(start_t)
                                logo_clips.append(trans_clip)
                        
                        # Premium Luminance Dip
                        dip = ColorClip(size=(FRAME_W, FRAME_H), color=(0, 0, 0), duration=0.2).with_opacity(0.35)
                        dip = dip.with_start(start_t).with_effects([vfx.CrossFadeIn(0.1), vfx.CrossFadeOut(0.1)])
                        logo_clips.append(dip)
                    
                    scale_factor = 1.0 + random.uniform(0.15, 0.22)
                    c_clip = c_clip.resized(lambda t, sf=scale_factor, cd=clip_dur: 1.0 + (sf - 1.0) * (t / cd))
                    c_clip = _apply_handheld_shake(c_clip)
                    
                    c_clip = c_clip.with_start(start_t)
                    bg_layer_clips.append(c_clip)
                except Exception as e:
                    print(f"Failed to load background img {vp} for chunk {i}: {e}")

        # ── B-ROLL BURSTS AT FACT BOUNDARIES ──────────────────────────────────────
        if is_longform and script_json.get("longform_format") == "did_you_know":
            fact_timestamps_lf = script_json.get("fact_timestamps", [])
            for ft in fact_timestamps_lf:
                start_s = float(ft.get("approx_start_seconds", 0))
                if start_s > 5.0 and len(expanded_visual_paths) >= 3:
                    burst_start = start_s - 0.25
                    try:
                        burst_images = random.sample(expanded_visual_paths, 3)
                        for idx, b_vp in enumerate(burst_images):
                            b_clip = ImageClip(b_vp).with_duration(0.17).resized((FRAME_W, FRAME_H))
                            b_clip = b_clip.with_start(burst_start + idx * 0.17)
                            if idx == 0:
                                b_clip = b_clip.with_effects([vfx.CrossFadeIn(0.05)])
                            burst_clips.append(b_clip)
                    except:
                        pass

    # ── AVATAR VIDEO PiP ──────────────────────────────────────────────────
    # Skip avatar entirely when Kaggle GPU fallback was used (no lip-sync available)
    skip_avatar = script_json.get("skip_avatar", False)
    avatar_pip = None
    ring_clip = None
    ring_size = 0
    
    if skip_avatar:
        print("⏭️ Skipping Avatar PiP (Kaggle GPU fallback — no lip-sync available).")
        lipsync_path = None
    else:
        print("Preparing Dimension Avatar PiP...")
        lipsync_path = script_json.get("kaggle_lipsync_path")
        face_template = script_json.get("lipsync_face_path") or os.path.join(ASSETS_DIR, "video", "Firefly_video_final.mp4")
    
        if not lipsync_path or not os.path.exists(lipsync_path):
            lipsync_path = _generate_lipsync_video(audio_path, face_template)
        
        avatar_video_path = lipsync_path if lipsync_path else face_template

    if not skip_avatar and os.path.exists(avatar_video_path):
        vid_clip = VideoFileClip(avatar_video_path)
        if vid_clip.duration < audio_duration:
            vid_clip = vid_clip.with_effects([vfx.Loop(duration=audio_duration)])
        else:
            vid_clip = vid_clip.subclipped(0, audio_duration)

        w, h = vid_clip.size
        
        # Crop avatar based on video format
        if is_longform:
            # For longform 16:9: crop to square (1:1) for bottom-right PiP bubble
            target_aspect = 1.0
        else:
            # For Shorts 9:16: crop to portrait to focus on character
            target_aspect = 9 / 16
        
        if w/h > target_aspect:
            new_w = int(h * target_aspect)
            x1 = (w - new_w) // 2
            vid_clip = vid_clip.cropped(x1=x1, y1=0, x2=x1+new_w, y2=h)
        else:
            new_h = int(w / target_aspect)
            # Shift the vertical crop window upwards to ensure face and hair are fully visible
            # in the circular PiP instead of being cut off at the top
            y1 = int((h - new_h) * 0.12) if h > new_h else 0
            vid_clip = vid_clip.cropped(x1=0, y1=y1, x2=w, y2=y1+new_h)
            
        w, h = vid_clip.size
        
        # Avatar size: 60% for longform (premium centered presenter), 40% for Shorts
        avatar_height_pct = 0.60 if is_longform else 0.40
        height_pip = int(FRAME_H * avatar_height_pct)
        width_pip = int(height_pip * (w / h))
        
        # Apply refinements from dynamic_params
        cur_w = max(1, int(width_pip * avatar_scale_mult))
        cur_h = max(1, int(height_pip * avatar_scale_mult))
        avatar_clip = vid_clip.resized((cur_w, cur_h)).without_audio()

        # ── AI BACKGROUND REMOVAL (Premium Mode - Highly Optimized & Dynamic) ───────
        try:
            from rembg import remove, new_session
            print("👤 Initializing AI Background Removal (Dynamic Mode)...")
            
            # Use u2net_human_seg for faster and highly precise human segmentation
            rembg_session = new_session(model_name="u2net_human_seg")
            
            # Keep a reference to the unmasked avatar clip for frame extraction
            unmasked_avatar = avatar_clip
            
            # Memoize computed masks to avoid redundant rembg processing
            mask_cache = {}
            fps = getattr(vid_clip, "fps", 30.0) or 30.0
            
            def make_mask_frame(t):
                # cache by integer frame index
                frame_idx = int(round(t * fps))
                if frame_idx in mask_cache:
                    return mask_cache[frame_idx]
                
                # Get the unmasked frame
                frame = unmasked_avatar.get_frame(t)
                
                # Perform dynamic background removal
                rgba = remove(
                    frame,
                    session=rembg_session,
                    alpha_matting=False,
                    post_process_mask=True
                )
                mask = (rgba[:, :, 3] / 255.0).astype(np.float32)
                
                # Erase the bottom 12% of the mask to completely hide the Gemini/Veo watermark logo
                h_mask, w_mask = mask.shape
                watermark_height = int(h_mask * 0.12)
                mask[-watermark_height:, :] = 0.0
                
                mask_cache[frame_idx] = mask
                return mask
            
            mclip = VideoClip(make_mask_frame, is_mask=True, duration=audio_duration)
            avatar_clip = avatar_clip.with_mask(mclip)
            print("   ✅ Dynamic AI background removal mask applied successfully (frame-by-frame).")
            
        except Exception as e:
            print(f"⚠️ rembg failed: {e}. Falling back to Rounded Authority Card.")
            # Fallback: Clean Rounded Card instead of a messy vignette
            Y, X = np.ogrid[:cur_h, :cur_w]
            rad = int(min(cur_w, cur_h) * 0.15)
            mask = np.ones((cur_h, cur_w), dtype=np.float32)
            for y, x in [(rad, rad), (rad, cur_w-rad), (cur_h-rad, rad), (cur_h-rad, cur_w-rad)]:
                dist = np.sqrt((Y-y)**2 + (X-x)**2)
                corner_mask = (dist > rad) & ( ( (Y<rad) if y==rad else (Y>cur_h-rad) ) & ( (X<rad) if x==rad else (X>cur_w-rad) ) )
                mask[corner_mask] = 0
            
            mclip = VideoClip(lambda t: mask, is_mask=True, duration=audio_duration)
            avatar_clip = avatar_clip.with_mask(mclip)

        # ── REFINED "ALIVE" MOTION (Head-Bob & Breathing) ────────────────
        # Apply motion AFTER masking so the mask follows the head movement
        import math
        # Calculate a slow zoom that increases scale by 10% over the full video
        zoom_speed = 0.10 / max(audio_duration, 1.0)
        
        # Combined dynamic scale: shrink/glide for Shorts intro + continuous micro-breathing/slow zoom
        def avatar_resize_fn(t):
            base_scale = 1.0 + zoom_speed * t + 0.006 * math.sin(t * 1.8)
            if is_longform:
                return base_scale
                
            # Glide/shrink logic for Shorts
            glide_dur = 3.5
            scale_start = 1080.0 / cur_w
            scale_end = 1.0
            
            if t < glide_dur:
                p = t / glide_dur
                p = 1.0 - (1.0 - p)**3 # cubic ease-out
                intro_scale = scale_start + (scale_end - scale_start) * p
            else:
                intro_scale = scale_end
                
            return intro_scale * base_scale

        avatar_clip = avatar_clip.with_effects([
            # Continuous dynamic zoom-in + Micro-Breathing + intro glide resize
            vfx.Resize(avatar_resize_fn), 
            # Natural Head Tilt: Very subtle +/- 0.5 degree swing
            vfx.Rotate(lambda t: 0.6 * math.sin(t * 1.4 + 0.5))
        ])

        # ── IMPROVEMENT #1: Circular Face-Cam Frame (Premium PiP) ─────────
        ring_clip = None
        ring_size = 0
        # Disabled circular facecam for longform to create a centered talking-head presenter
        # standing in front of background visuals, matching the explainer style of the reference video.
        if is_longform and False:
            try:
                avatar_clip, ring_clip, ring_size = _apply_circular_facecam_frame(
                    avatar_clip, cur_w, cur_h, accent_color, audio_duration, is_longform=True
                )
                print("   ✅ Circular face-cam frame with glow ring applied.")
            except Exception as e:
                print(f"   ⚠️ Circular frame failed (non-fatal): {e}")

        def get_scaled_dims(t):
            scale = avatar_resize_fn(t)
            return int(cur_w * scale), int(cur_h * scale)

        def pip_position(t):
            scaled_w, scaled_h = get_scaled_dims(t)
            if is_longform:
                # Center horizontally for longform 16:9 like a news anchor / central explainer host
                base_x = (FRAME_W - scaled_w) // 2
                base_y = FRAME_H - scaled_h
                return (base_x, base_y)
            else:
                # Glide/shrink position interpolation for Shorts
                glide_dur = 3.5
                
                # Start position (centered just below the header bar)
                x_start = 0.0
                y_start = 350.0
                
                # Target bottom-center PIP position
                x_end = (FRAME_W - scaled_w) / 2.0 + layout["avatar_x_offset"]
                y_end = FRAME_H - scaled_h - 30.0
                
                if t < glide_dur:
                    p = t / glide_dur
                    p = 1.0 - (1.0 - p)**3 # cubic ease-out
                    x_pos = x_start + (x_end - x_start) * p
                    y_pos = y_start + (y_end - y_start) * p
                    return (int(x_pos), int(y_pos))
                else:
                    return (int(x_end), int(y_end))

        if not is_longform:
            def hide_avatar_during_screenshots(t):
                # Fade out/in slightly (0.3s) at boundaries
                for s_start, s_end in screenshot_intervals:
                    if s_start <= t <= s_end:
                        fade_win = 0.3
                        if t < s_start + fade_win:
                            return max(0.0, (s_start + fade_win - t) / fade_win)
                        elif t > s_end - fade_win:
                            return max(0.0, (t - (s_end - fade_win)) / fade_win)
                        return 0.0
                return 1.0

            # Wrap the avatar's mask to apply the hiding factor
            orig_mask = avatar_clip.mask
            if orig_mask is not None:
                def hide_mask_frame(t):
                    return orig_mask.get_frame(t) * hide_avatar_during_screenshots(t)
                avatar_clip = avatar_clip.with_mask(VideoClip(hide_mask_frame, is_mask=True, duration=audio_duration))
                
            # Wrap the ring's mask to apply the hiding factor
            if ring_clip is not None and ring_clip.mask is not None:
                orig_ring_mask = ring_clip.mask
                def hide_ring_frame(t):
                    return orig_ring_mask.get_frame(t) * hide_avatar_during_screenshots(t)
                ring_clip = ring_clip.with_mask(VideoClip(hide_ring_frame, is_mask=True, duration=audio_duration))

        # Position the glow ring to track the avatar
        def ring_position(t):
            ax, ay = pip_position(t)
            # Center the ring around the avatar (ring is slightly larger)
            offset = (ring_size - min(cur_w, cur_h)) // 2
            return (ax - offset, ay - offset)

        avatar_pip = avatar_clip.with_position(pip_position).with_start(0)

    # ── LAYERS ───────────────────────────────────────────────────────────
    if is_longform:
        screenshot_clips = _longform_article_screenshot_clips(script_json, audio_duration)
    else:
        screenshot_path = script_json.get("screenshot_path")
        screenshot_clips = _article_screenshot_clip(screenshot_path, audio_duration)
    gradient = _gradient_clip(audio_duration, height_pct=layout["gradient_height_pct"], position=layout["gradient_position"], is_longform=is_longform)
    
    # Ambient Particles
    particle_layer = _ambient_particles(audio_duration, accent_color, particle_style=layout["particle_style"])

    # ── HUMAN REALISM OVERLAYS ───────────────────────────────────────────────
    grain_layer = _generate_film_grain(audio_duration, FRAME_W, FRAME_H)
    flare_layer = None  # Disabled lens flare circle drifting left-to-right as requested
    
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

    # ── LONGFORM: "FACT X/N" BADGE OVERLAYS ──────────────────────────────────
    longform_badge_clips = []
    if is_longform and script_json.get("longform_format") == "did_you_know":
        fact_timestamps = script_json.get("fact_timestamps", [])
        # Use num_facts from script or count only numeric fact entries
        total_facts = script_json.get("num_facts", 10)
        
        for i, ft in enumerate(fact_timestamps):
            fact_num = ft.get("fact_number", i + 1)
            # Skip non-numeric entries (recaps, outro, cold open)
            if not isinstance(fact_num, int) or fact_num <= 0:
                continue
            start_s = float(ft.get("approx_start_seconds", 0))
            
            # Duration until next fact or end of audio
            if i + 1 < len(fact_timestamps):
                end_s = float(fact_timestamps[i + 1].get("approx_start_seconds", audio_duration))
            else:
                end_s = audio_duration
            fact_dur = max(1.0, end_s - start_s)
            
            # Render badge image — larger font for 16:9 readability
            badge_text = f"FACT {fact_num}/{total_facts}"
            badge_f = gf(34, bold=True)
            bw, bh = ts(badge_text, badge_f)
            pad_x, pad_y = 20, 12
            badge_img = Image.new("RGBA", (bw + pad_x * 2, bh + pad_y * 2), (0, 0, 0, 0))
            badge_draw = ImageDraw.Draw(badge_img)
            badge_draw.rounded_rectangle(
                [0, 0, bw + pad_x * 2 - 1, bh + pad_y * 2 - 1],
                radius=14, fill=(*accent_color, 220)
            )
            badge_draw.text((pad_x, pad_y), badge_text, font=badge_f, fill=(255, 255, 255, 255))
            
            badge_arr = np.array(badge_img.convert("RGB"))
            badge_mask = np.array(badge_img.split()[3]).astype(float) / 255.0
            
            def make_badge_opacity(t, _dur=fact_dur):
                if t < 0.4: return t / 0.4
                elif t > _dur - 0.4: return max(0, (_dur - t) / 0.4)
                return 1.0
            
            b_clip = VideoClip(lambda t, _arr=badge_arr: _arr, duration=fact_dur)
            b_mask = VideoClip(lambda t, _m=badge_mask, _dur=fact_dur: _m * make_badge_opacity(t, _dur), is_mask=True, duration=fact_dur)
            b_clip = b_clip.with_mask(b_mask).with_position((40, 40)).with_start(start_s)
            b_clip = b_clip.with_effects([vfx.CrossFadeIn(0.3)])
            longform_badge_clips.append(b_clip)

    # Default constant for badge generation when fact_timestamps is missing
    LONGFORM_NUM_TOPICS_DEFAULT = 10

    # Stack background, then screenshot clips on top of background
    topic_transition_clips = []
    if is_longform:
        topic_transition_clips = _longform_topic_transition_clips(script_json, audio_duration)
        
    base_layers = bg_layer_clips + burst_clips + [particle_layer] + screenshot_clips + topic_transition_clips
    
    # Add overlays
    base_layers.append(gradient)
    base_layers.extend(logo_clips)
    
    if flare_layer: base_layers.append(flare_layer)
    if grain_layer: base_layers.append(grain_layer)
    if avatar_pip: base_layers.append(avatar_pip)
    base_layers.extend(longform_badge_clips)
    # ── LOGO BRANDING OVERLAY STACK ────────────────────────────────────
    # Render entity tags on the left side below the title for Shorts, or use the right-side logo stack for longform
    entities_list = []
    for ent_list_key in ["companies", "people", "key_entities"]:
        for ent in script_json.get(ent_list_key, []):
            if not any(e.get("name") == ent.get("name") for e in entities_list):
                # Ensure local_logo_path is set
                logo_path = ent.get("local_logo_path") or ent.get("local_hq_path") or ent.get("local_image_path")
                if logo_path:
                    ent["local_logo_path"] = logo_path
                entities_list.append(ent)
                
    if not is_longform:
        # Filter entities to only those that have a name, a description, and a logo
        entities_list = [
            e for e in entities_list
            if e.get("name") and e.get("description") and e.get("local_logo_path") and os.path.exists(e.get("local_logo_path"))
        ]
                
    if not is_longform and entities_list:
        # Pre-calculate active time intervals for each entity based on chunk mentions
        for ent in entities_list:
            name = ent.get("name", "").lower()
            intervals = []
            for chunk in chunks:
                chunk_text = chunk.get("text", "").lower()
                if name in chunk_text:
                    intervals.append((chunk["start"], chunk["end"] + 3.0))
            
            # Merge overlapping/adjacent intervals
            merged = []
            if intervals:
                intervals.sort(key=lambda x: x[0])
                curr_start, curr_end = intervals[0]
                for s, e in intervals[1:]:
                    if s <= curr_end:
                        curr_end = max(curr_end, e)
                    else:
                        merged.append((curr_start, curr_end))
                        curr_start, curr_end = s, e
                merged.append((curr_start, curr_end))
            
            # If the entity is never mentioned in the script text, default to showing it throughout
            if not merged:
                merged.append((0.0, audio_duration))
                
            ent["active_intervals"] = merged

        # Pre-load PIL images for entity logos to avoid disk I/O in the frame loop
        for ent in entities_list:
            logo_path = ent.get("local_logo_path")
            if logo_path and os.path.exists(logo_path):
                try:
                    ent["pil_logo"] = Image.open(logo_path).convert("RGBA")
                except Exception as e:
                    print(f"Failed to pre-load logo {logo_path}: {e}")
                    ent["pil_logo"] = None
            else:
                ent["pil_logo"] = None
        
    if is_longform:
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

    # ══════════════════════════════════════════════════════════════════════
    # LONGFORM RETENTION UPGRADE: Wire in new premium visual layers
    # ══════════════════════════════════════════════════════════════════════
    snap_zoom_timestamps = []  # Used per-frame in make_final_frame
    
    if is_longform and script_json.get("longform_format") == "did_you_know":
        fact_timestamps_lf = script_json.get("fact_timestamps", [])
        
        # IMPROVEMENT #1: Glow ring behind avatar PiP
        if ring_clip is not None:
            positioned_ring = ring_clip.with_position(ring_position).with_start(0)
            # Insert ring BEFORE avatar in the layer stack (so it renders behind)
            # Find avatar_pip index and insert ring before it
            # Find index using identity check to avoid MoviePy's buggy Clip.__eq__
            idx = next((i for i, clip in enumerate(base_layers) if clip is avatar_pip), -1)
            if idx != -1:
                base_layers.insert(idx, positioned_ring)
            else:
                base_layers.append(positioned_ring)
            print("   🔴 Glow ring layer added behind avatar.")

        # IMPROVEMENT #3: Progress Dot Navigator (top-center)
        try:
            progress_dot_clips = _longform_progress_dots(fact_timestamps_lf, accent_color, audio_duration)
            base_layers.extend(progress_dot_clips)
            print(f"   ⏺ Progress dot navigator added ({len(progress_dot_clips)} segments).")
        except Exception as e:
            print(f"   ⚠️ Progress dots failed (non-fatal): {e}")

        # IMPROVEMENT #2: Kinetic Metric Pop-Ups
        try:
            metric_popups = script_json.get("metric_popups", [])
            # Fallback: extract key_stat from each fact script if no metric_popups
            if not metric_popups:
                fact_scripts = script_json.get("fact_scripts", [])
                for fs in fact_scripts:
                    key_stat = fs.get("key_stat", "")
                    # Estimate timestamp from fact number
                    fact_num = fs.get("fact_number", 0)
                    est_ts = 5.0 + (fact_num - 1) * 30.0  # Rough estimate
                    if key_stat and est_ts < audio_duration:
                        metric_popups.append({"text": key_stat, "timestamp": est_ts})
            
            metric_popup_clips = []
            for mp in metric_popups:
                mp_clip = _kinetic_metric_popup(
                    mp.get("text", ""), float(mp.get("timestamp", 0)),
                    accent_color, audio_duration
                )
                if mp_clip:
                    metric_popup_clips.append(mp_clip)
            base_layers.extend(metric_popup_clips)
            print(f"   📊 Kinetic metric pop-ups added ({len(metric_popup_clips)} metrics).")
        except Exception as e:
            print(f"   ⚠️ Metric pop-ups failed (non-fatal): {e}")

        # IMPROVEMENT #6: Value Loop Montage (Intro Teasers)
        try:
            value_loop_clips = _value_loop_montage_clips(script_json, accent_color, audio_duration)
            base_layers.extend(value_loop_clips)
            print(f"   🎬 Value loop montage added ({len(value_loop_clips)} teasers).")
        except Exception as e:
            print(f"   ⚠️ Value loop montage failed (non-fatal): {e}")

        # IMPROVEMENT #7: Mid-Video Subscribe CTA
        try:
            subscribe_cta = _mid_video_subscribe_prompt(accent_color, audio_duration)
            if subscribe_cta:
                base_layers.append(subscribe_cta)
                print("   🔔 Mid-video subscribe CTA added at 75% mark.")
        except Exception as e:
            print(f"   ⚠️ Subscribe CTA failed (non-fatal): {e}")

        # IMPROVEMENT #4: Fact Boundary Darkeners (chapter breaks)
        try:
            boundary_clips = _fact_boundary_darkener(fact_timestamps_lf, accent_color, audio_duration)
            base_layers.extend(boundary_clips)
            print(f"   🌑 Fact boundary darkeners added ({len(boundary_clips)} boundaries).")
        except Exception as e:
            print(f"   ⚠️ Boundary darkeners failed (non-fatal): {e}")

        # IMPROVEMENT #4: Generate snap-zoom timestamps for per-frame processing
        snap_zoom_timestamps = _generate_snap_zoom_interrupts(chunks, audio_duration, interval=5.0)
        print(f"   ⚡ Snap-zoom interrupts scheduled ({len(snap_zoom_timestamps)} zooms every 5s).")



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
    pb_h = layout["progress_bar_height"]
    pb_pos = layout["progress_bar_position"]
    def make_progress_frame(t):
        color = get_progress_color(t)
        base_img = np.zeros((pb_h, FRAME_W, 3), dtype=np.uint8)
        base_img[:, :] = color
        return base_img
        
    progress = VideoClip(make_progress_frame, duration=audio_duration)
    if pb_pos == "top":
        progress = progress.with_position(lambda t: (int((t / max(audio_duration, 0.01)) * FRAME_W) - FRAME_W, 0))
    else:
        progress = progress.with_position(lambda t: (int((t / max(audio_duration, 0.01)) * FRAME_W) - FRAME_W, FRAME_H - pb_h))
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

    # ── KINETIC SFX & BGM MASTERING ENGINE (Pydub-driven) ────────────────────
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

    # Generate a single high-fidelity mastered soundtrack using pure Pydub
    mastered_audio_path = os.path.join(OUTPUT_DIR, f"master_soundtrack_{today}.wav")
    sfx_cues = script_json.get("sfx_cues", [])
    
    _mix_and_master_audio(
        voice_path=audio_path,
        bgm_path=bgm_path,
        sfx_cues=sfx_cues,
        chunks=chunks,
        retention_hooks=script_json.get("retention_cues", []),
        output_duration=audio_duration,
        bgm_volume_config=BGM_VOLUME,
        output_path=mastered_audio_path,
        fact_timestamps=script_json.get("fact_timestamps", []),
        retention_map=script_json.get("retention_map", {})
    )
    
    final_audio = AudioFileClip(mastered_audio_path)
    
    # Header bar: solid black top bar with white text for Shorts, disabled for Longform
    if not is_longform:
        header_img = render_shorts_header_bar(title, accent_color, FRAME_W)
    else:
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
                is_active = w["start"] - 0.05 <= t <= w["end"] + 0.05
                word_status_list.append({
                    "word": w["word"],
                    "is_active": is_active,
                    "is_spoken": t > w["end"],
                    "scale": 1.0 # No zoom
                })

            if word_status_list:
                subtitle_img = render_subtitle_frame(
                    word_status_list, bg_frame=bg_frame, 
                    accent_color=accent_color, frame_width=FRAME_W, frame_height=FRAME_H,
                    y_shift=subtitle_y_shift
                )
                
        # SENTENCE POP ANIMATION DISABLED (No zoom)
        pass

        # ── HOOK TRANSITION BURST ──────────────────────────────────────────
        # Inject high-impact transition exactly when the hook text fades
        hook_end_time = layout["hook_transition_time"]
        if abs(t - hook_end_time) < 0.25:
            # 1. Intense Glitch
            bg_frame = _apply_intensive_glitch(bg_frame, intensity=1.2)
            # 2. Flash Burst
            bg_frame = bg_frame.astype(np.float32)
            bg_frame = np.clip(bg_frame + 60, 0, 255).astype(np.uint8)

        # ── IMPROVEMENT #4: Snap-Zoom Pattern Interrupts (5-second rule) ──
        # Quick 1.08x zoom pulse every 5 seconds to re-engage viewers
        for sz_t in snap_zoom_timestamps:
            delta = t - sz_t
            if 0 <= delta < 0.3:
                # Quick zoom in over 0.3s
                progress = delta / 0.3
                zoom = 1.0 + 0.08 * (1.0 - abs(progress - 0.5) * 2)  # Peak at 0.15s
                h_f, w_f = bg_frame.shape[:2]
                nh, nw = int(h_f / zoom), int(w_f / zoom)
                dy, dx = (h_f - nh) // 2, (w_f - nw) // 2
                if nh > 0 and nw > 0 and dy >= 0 and dx >= 0:
                    cropped = bg_frame[dy:dy + nh, dx:dx + nw]
                    bg_frame = cv2.resize(cropped, (w_f, h_f), interpolation=cv2.INTER_LINEAR)
                break  # Only one snap-zoom at a time

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

        entity_tags_img = None
        if not is_longform and entities_list:
            entity_tags_img = render_dynamic_entity_tags(entities_list, accent_color, t, audio_duration, FRAME_W, FRAME_H, screenshot_intervals=screenshot_intervals)

        return composite_frame(bg_frame, t, header_img, subtitle_img, transparency_img, entity_tags_img)


    final = VideoClip(make_final_frame, duration=audio_duration)
    final = final.with_audio(final_audio)

    # ── IMPROVEMENT #9: Seamless CTA (Longform) vs. Static CTA (Shorts) ───────
    if is_longform:
        # For longform: overlay CTA as a floating card during the last 5s
        # instead of hard-cutting to a static dark card
        cta_overlay_dur = 5.0
        cta_start = max(0, audio_duration - cta_overlay_dur)
        
        cta_card_w = int(FRAME_W * 0.45)
        cta_card_h = 280
        cta_overlay_img = Image.new("RGBA", (cta_card_w, cta_card_h), (0, 0, 0, 0))
        cta_d = ImageDraw.Draw(cta_overlay_img)
        
        # Glassmorphic floating CTA card
        cta_d.rounded_rectangle([0, 0, cta_card_w, cta_card_h], radius=20, fill=(10, 10, 18, 220))
        cta_d.rounded_rectangle([0, 0, cta_card_w, cta_card_h], radius=20,
                                 outline=(*accent_color, 180), width=2)
        
        # Topic-Sync Headline
        topic = script_json.get("topic", "AI")
        cta_txt = layout["cta_headline_template"].format(topic=topic)
        cta_d.text((cta_card_w // 2, 40), cta_txt, fill=(255, 255, 255, 255),
                   font=gf(30, bold=True), anchor="mm")
        
        # Telegram brand image (mini)
        try:
            brand_path = os.path.join(ASSETS_DIR, "branding", "tele_brand2.jpg")
            if os.path.exists(brand_path):
                brand_img = Image.open(brand_path).convert("RGBA")
                bw, bh = brand_img.size
                brand_img = brand_img.crop((0, 0, bw, bh // 2))
                ratio = (cta_card_w - 40) / float(brand_img.width)
                brand_img = brand_img.resize((int(brand_img.width * ratio), int(brand_img.height * ratio)), Image.LANCZOS)
                cta_overlay_img.alpha_composite(brand_img, (20, 65))
        except:
            pass
        
        # Link in Bio pill
        pill_y = cta_card_h - 70
        pill_color = layout["cta_pill_color"]
        cta_d.rounded_rectangle([30, pill_y, cta_card_w - 30, pill_y + 50], radius=25,
                                 fill=(*pill_color, 255))
        cta_d.text((cta_card_w // 2, pill_y + 25), "Link in Bio",
                   fill=(0, 0, 0, 255), font=gf(28, bold=True), anchor="mm")
        
        cta_arr = np.array(cta_overlay_img.convert("RGB"))
        cta_mask = np.array(cta_overlay_img.split()[3]).astype(float) / 255.0
        
        def cta_opacity(t):
            if t < 0.5:
                return t / 0.5
            return 1.0
        
        cta_clip = VideoClip(lambda t, _a=cta_arr: _a, duration=cta_overlay_dur)
        cta_mclip = VideoClip(lambda t, _m=cta_mask: _m * cta_opacity(t),
                              is_mask=True, duration=cta_overlay_dur)
        cta_clip = cta_clip.with_mask(cta_mclip)
        
        # Position: bottom-left (leaving bottom-right free for YouTube end screen)
        cta_x = 40
        cta_y = FRAME_H - cta_card_h - 40
        cta_clip = cta_clip.with_position((cta_x, cta_y)).with_start(cta_start)
        
        # Compose the CTA overlay on top of the main video
        final = CompositeVideoClip([final, cta_clip], size=(FRAME_W, FRAME_H)).with_duration(audio_duration)
        final = final.with_audio(final_audio)
        
        # Activate the existing _next_video_tease function
        tease_text = script_json.get("next_video_tease", "")
        if tease_text:
            tease_clip = _next_video_tease(tease_text, accent_color, audio_duration)
            if tease_clip:
                final = CompositeVideoClip([final, tease_clip], size=(FRAME_W, FRAME_H)).with_duration(audio_duration)
                final = final.with_audio(final_audio)
        
        print("   ✅ Seamless CTA overlay applied (last 5s, bottom-left).")
    else:
        # Shorts: keep existing static CTA card behavior (appended at end)
        cta_duration = 3.0
        cta_img = Image.new("RGBA", (FRAME_W, FRAME_H), (10, 10, 15, 255))
        cta_d = ImageDraw.Draw(cta_img)
        
        topic = script_json.get("topic", "AI")
        cta_txt = layout["cta_headline_template"].format(topic=topic)
        cta_d.text((FRAME_W//2, 180), cta_txt, fill=(255, 255, 255, 255), font=gf(54, bold=True), anchor="mm")
        
        try:
            brand_path = os.path.join(ASSETS_DIR, "branding", "tele_brand2.jpg")
            if os.path.exists(brand_path):
                brand_img = Image.open(brand_path).convert("RGBA")
                bw, bh = brand_img.size
                brand_img = brand_img.crop((0, 0, bw, bh // 2))
                ratio = (FRAME_W - 100) / float(brand_img.width)
                brand_img = brand_img.resize((int(brand_img.width * ratio), int(brand_img.height * ratio)), Image.LANCZOS)
                cta_img.alpha_composite(brand_img, (50, 320))
        except:
            pass
        
        pill_y = FRAME_H - 450
        pill_color = layout["cta_pill_color"]
        cta_d.rounded_rectangle([200, pill_y, FRAME_W - 200, pill_y + 120], radius=60, fill=(*pill_color, 255))
        cta_d.text((FRAME_W//2, pill_y + 60), "Link in Bio", fill=(0, 0, 0, 255), font=gf(50, bold=True), anchor="mm")
        
        cta_d.text((FRAME_W//2, pill_y + 180), layout["cta_description"], fill=(200, 200, 200, 255), font=gf(34), anchor="mm")

        cta_clip = ImageClip(np.array(cta_img.convert("RGB"))).with_duration(cta_duration)
        
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
                if not is_longform:
                    verify_text_visibility(test_frame, f"HEADER {p}", 113, 353)
                else:
                    verify_text_visibility(test_frame, f"HEADER {p}", 0, 240)
                
                test_path = output_path.replace(".mp4", f"_test_{int(p*100)}pct.jpg")
                img.save(test_path)
    except Exception as e:
        print(f"Visibility frames failed: {e}")

    final.write_videofile(
        output_path, fps=30, codec="libx264", audio_codec="aac",
        threads=4, preset="ultrafast", ffmpeg_params=["-pix_fmt", "yuv420p"]
    )
    
    try:
        final.close()
    except Exception as e:
        print(f"Cleanup warning: {e}")
        
    return output_path
