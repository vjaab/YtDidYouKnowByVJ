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
import gc
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

CI_LITE = os.environ.get("CI_LITE", "0") == "1"

FRAME_W, FRAME_H = 1080, 1920 # Default for Shorts
IS_LONGFORM_ACTIVE = False
def set_resolutions(is_longform=False):
    global FRAME_W, FRAME_H
    if is_longform:
        FRAME_W, FRAME_H = 1920, 1080
    else:
        FRAME_W, FRAME_H = 1080, 1920

TITLE_BOTTOM_GAP = 192  # Default; overridden per-video by LayoutProfile

import hashlib as _hashlib

# ══════════════════════════════════════════════════════════════════════════════
# SAFE ZONE CALCULATOR — Prevents overlay collisions (badges, captions, branding)
# ══════════════════════════════════════════════════════════════════════════════
class SafeZoneCalculator:
    """
    Calculates and manages safe zones for all overlay elements to prevent collisions.
    Divides the frame into reserved zones and available zones for each element type.
    """
    
    def __init__(self, frame_w, frame_h, layout=None, is_longform=False):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.layout = layout or {}
        self.is_longform = is_longform
        self.reserved_zones = []  # List of (x1, y1, x2, y2, name) tuples
        self.element_positions = {}  # element_type -> (x, y, w, h)
        
        # Initialize default reserved zones
        self._init_default_zones()
    
    def _init_default_zones(self):
        """Initialize zones that are always reserved based on layout type."""
        # YouTube UI safe zones - STRICT per feedback
        # Top 15% for platform headers/notch (192px on 1080p, 162px on 1920p)
        top_ui_h = int(self.frame_h * 0.15)
        self.reserve_zone(0, 0, self.frame_w, top_ui_h, "youtube_top_ui")
        # Bottom 20% for player controls, captions, engagement buttons (384px on 1080p, 216px on 1920p)
        bottom_ui_h = int(self.frame_h * 0.20)
        self.reserve_zone(0, self.frame_h - bottom_ui_h, self.frame_w, self.frame_h, "youtube_bottom_ui")
        
        # Title area (bottom) - positioned above bottom UI zone
        title_gap = self.layout.get("title_bottom_gap", 192)
        title_y_start = self.frame_h - title_gap - 200
        title_y_end = self.frame_h - title_gap
        # Ensure title doesn't overlap with bottom UI
        if title_y_end > self.frame_h - bottom_ui_h:
            title_y_end = self.frame_h - bottom_ui_h
            title_y_start = title_y_end - 200
        self.reserve_zone(
            0, title_y_start, 
            self.frame_w, title_y_end, 
            "title_area"
        )
        
        # Define STRICT safe caption zone - center 60% vertical viewport (20%-80%)
        # This ensures captions stay in the middle area, away from top/bottom UI
        caption_top = int(self.frame_h * 0.20)
        caption_bottom = int(self.frame_h * 0.80)
        self.reserve_zone(0, caption_top, self.frame_w, caption_bottom, "caption_safe_zone")
        
        # Define UPPER-MIDDLE safe zone for kinetic captions (30%-55% of frame)
        # This is the "lower third" equivalent for vertical video - above presenter, below top UI
        upper_middle_top = int(self.frame_h * 0.30)
        upper_middle_bottom = int(self.frame_h * 0.55)
        self.reserve_zone(0, upper_middle_top, self.frame_w, upper_middle_bottom, "upper_middle_safe_zone")
        
        # Define LOWER-MIDDLE safe zone for stat callouts (55%-75% of frame)
        lower_middle_top = int(self.frame_h * 0.55)
        lower_middle_bottom = int(self.frame_h * 0.75)
        self.reserve_zone(0, lower_middle_top, self.frame_w, lower_middle_bottom, "lower_middle_safe_zone")
        
        # Avatar/talking head zone - dedicated corner
        if self.is_longform:
            # Longform: center-bottom presenter area
            self.reserve_zone(
                self.frame_w // 2 - 300, self.frame_h - 950,
                self.frame_w // 2 + 300, self.frame_h - 50,
                "longform_presenter"
            )
        else:
            # Shorts: dedicated corner (bottom-right by default)
            layout_type = self.layout.get("layout_type", "asymmetric")
            if layout_type == "split_screen":
                # Right side, upper area
                self.reserve_zone(
                    self.frame_w - 320, 100,
                    self.frame_w - 20, 420,
                    "avatar_corner"
                )
            elif layout_type == "hero_center":
                # Left side, lower area (changed from right side)
                self.reserve_zone(
                    20, self.frame_h - 420,
                    340, self.frame_h - 100,
                    "avatar_corner"
                )
            elif layout_type == "side_strip":
                # Left strip
                self.reserve_zone(20, self.frame_h // 2 - 160, 340, self.frame_h // 2 + 160, "avatar_corner")
            elif layout_type == "top_center":
                # Left middle (changed from top center)
                self.reserve_zone(20, self.frame_h // 2 - 160, 340, self.frame_h // 2 + 160, "avatar_corner")
            elif layout_type == "corner_cycling":
                # Will be set per-video based on corner_index
                corner_idx = self.layout.get("corner_index", 0)
                corners = [
                    (20, 100, 340, 420),  # top-left
                    (self.frame_w - 340, 100, self.frame_w - 20, 420),  # top-right
                    (20, self.frame_h - 420, 340, self.frame_h - 100),  # bottom-left
                    (20, self.frame_h - 420, 340, self.frame_h - 100),  # bottom-left (changed from bottom-right)
                ]
                x1, y1, x2, y2 = corners[corner_idx % 4]
                self.reserve_zone(x1, y1, x2, y2, "avatar_corner")
            else:  # asymmetric - full screen presenter bottom center
                self.reserve_zone(
                    self.frame_w // 2 - 160, self.frame_h - 420,
                    self.frame_w // 2 + 160, self.frame_h - 100,
                    "avatar_corner"
                )
    
    def reserve_zone(self, x1, y1, x2, y2, name):
        """Reserve a rectangular zone so no other element can overlap it."""
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.frame_w, x2), min(self.frame_h, y2)
        if x2 > x1 and y2 > y1:
            self.reserved_zones.append((x1, y1, x2, y2, name))
            # Track element position for get_position()
            self.element_positions[name] = (x1, y1, x2 - x1, y2 - y1)
    
    def check_overlap(self, x1, y1, x2, y2, margin=20):
        """Check if a proposed zone overlaps any reserved zone."""
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(self.frame_w, x2 + margin), min(self.frame_h, y2 + margin)
        
        for rz_x1, rz_y1, rz_x2, rz_y2, name in self.reserved_zones:
            if not (x2 <= rz_x1 or x1 >= rz_x2 or y2 <= rz_y1 or y1 >= rz_y2):
                return True, name
        return False, None
    
    def find_safe_position(self, element_w, element_h, preferred_positions=None, element_type="overlay"):
        """
        Find a safe position for an element of given dimensions.
        Returns (x, y) coordinates or None if no safe position found.
        """
        if preferred_positions is None:
            preferred_positions = self._get_default_preferred_positions(element_type)
        
        for pos in preferred_positions:
            if isinstance(pos, str):
                # Named position
                x, y = self._resolve_named_position(pos, element_w, element_h)
            elif isinstance(pos, (list, tuple)) and len(pos) == 2:
                if isinstance(pos[0], str):
                    # Named position with options: ("top_left", {"margin": 40})
                    name, options = pos
                    margin = options.get("margin", 0) if isinstance(options, dict) else 0
                    x, y = self._resolve_named_position(name, element_w, element_h)
                    if margin:
                        if "left" in name:
                            x += margin
                        elif "right" in name:
                            x -= margin
                        if "top" in name:
                            y += margin
                        elif "bottom" in name:
                            y -= margin
                else:
                    # Coordinate tuple: (x, y)
                    x, y = pos
            else:
                continue
            
            x2, y2 = x + element_w, y + element_h
            overlaps, zone_name = self.check_overlap(x, y, x2, y2)
            
            if not overlaps:
                self.reserve_zone(x, y, x2, y2, element_type)
                self.element_positions[element_type] = (x, y, element_w, element_h)
                return (x, y)
        
        # Fallback: try grid search
        return self._grid_search_fallback(element_w, element_h, element_type)
    
    def _get_default_preferred_positions(self, element_type):
        """Get default preferred positions for each element type."""
        positions = {
            "badge": [
                ("top_left", {"margin": 40}),
                ("top_right", {"margin": 40}),
                ("bottom_left", {"margin": 40}),
            ],
            "caption": [
                ("lower_third_center", {}),
                ("upper_middle_center", {}),
            ],
            "stat_callout": [
                ("center_left", {"margin": 100}),
                ("center_right", {"margin": 100}),
                ("upper_center", {"margin": 100}),
            ],
            "chapter_card": [
                ("center", {}),
            ],
            "code_snippet": [
                ("center_left", {"margin": 60}),
                ("center_right", {"margin": 60}),
            ],
            "diagram": [
                ("center", {}),
            ],
            "cta_card": [
                ("bottom_left", {"margin": 40}),
                ("bottom_right", {"margin": 40}),
            ],
        }
        return positions.get(element_type, [("center", {})])
    
    def _resolve_named_position(self, name, w, h):
        """Resolve named position to coordinates."""
        positions = {
            "top_left": (40, 100),
            "top_right": (self.frame_w - w - 40, 100),
            "bottom_left": (40, self.frame_h - h - 180),
            "bottom_right": (self.frame_w - w - 40, self.frame_h - h - 180),
            "center": ((self.frame_w - w) // 2, (self.frame_h - h) // 2),
            "center_left": (60, (self.frame_h - h) // 2),
            "center_right": (self.frame_w - w - 60, (self.frame_h - h) // 2),
            "upper_center": ((self.frame_w - w) // 2, 150),
            "lower_third_center": ((self.frame_w - w) // 2, int(self.frame_h * 0.65)),
            "upper_middle_center": ((self.frame_w - w) // 2, int(self.frame_h * 0.35)),
        }
        return positions.get(name, ((self.frame_w - w) // 2, (self.frame_h - h) // 2))
    
    def _grid_search_fallback(self, w, h, element_type):
        """Last resort: grid search for any available position."""
        step = 50
        for y in range(100, self.frame_h - h - 100, step):
            for x in range(50, self.frame_w - w - 50, step):
                overlaps, _ = self.check_overlap(x, y, x + w, y + h)
                if not overlaps:
                    self.reserve_zone(x, y, x + w, y + h, element_type)
                    self.element_positions[element_type] = (x, y, w, h)
                    return (x, y)
        return None
    
    def get_position(self, element_type):
        """Get the reserved position for an element type."""
        return self.element_positions.get(element_type)
    
    def clear_element(self, element_type):
        """Clear an element's reservation (for dynamic elements)."""
        if element_type in self.element_positions:
            x, y, w, h = self.element_positions[element_type]
            # Remove from reserved zones
            self.reserved_zones = [z for z in self.reserved_zones if z[4] != element_type]
            del self.element_positions[element_type]


def get_vibrant_dominant_color(img_path):
    """
    Analyzes an image cheaply by resizing it to 32x32, converting to HSV, and
    filtering for vibrant colors (Saturation > 0.22, Value between 0.2 and 0.9).
    Returns the dominant vibrant (R, G, B) tuple, or None if no vibrant color exists.
    """
    if not img_path or not os.path.exists(img_path):
        return None
    try:
        with Image.open(img_path) as img:
            small_img = img.resize((32, 32))
            colors = small_img.getcolors(32 * 32)
            if not colors:
                return None
            
            vibrant_colors = []
            for count, color in colors:
                r, g, b = color[:3]
                # RGB to HSV calculation
                mx = max(r, g, b)
                mn = min(r, g, b)
                df = mx - mn
                sat = 0.0 if mx == 0 else (df / mx)
                val = mx / 255.0
                
                # Check for vibrance (discard grayscale, near-black, near-white)
                if sat > 0.22 and 0.2 < val < 0.9:
                    vibrant_colors.append((count, (r, g, b)))
            
            if vibrant_colors:
                # Sort by count/frequency
                vibrant_colors.sort(key=lambda x: x[0], reverse=True)
                return vibrant_colors[0][1]
            return None
    except Exception as e:
        print(f"⚠️ Error extracting dominant color: {e}")
        return None

# Layout profile generation now handled by layout_engine.py
from layout_engine import generate_layout_profile

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
    key = (size, bold, italic, IS_LONGFORM_ACTIVE)
    if key not in _fc:
        search_paths = []
        
        # 1. Prioritize specific fonts for bold/italic if available in assets
        if IS_LONGFORM_ACTIVE:
            # Roboto for Longform
            if bold:
                search_paths.append(os.path.join(ASSETS_DIR, "fonts", "Roboto-Bold.ttf"))
            else:
                search_paths.append(os.path.join(ASSETS_DIR, "fonts", "Roboto-Regular.ttf"))
        else:
            # Montserrat for Shorts
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

def _prepare_screenshot_canvas(img, target_w, target_h, url=None, apply_vignette=False):
    """
    Creates a premium 'Blurred Backdrop' canvas for wide screenshots.
    - Longform (16:9): Clean frosted glass card with rounded corners + drop shadow (no browser mockup)
    - Shorts (9:16): Elegant macOS dark-mode browser mockup with controls, outline, and domain name.
    Also automatically crops the top 80px of captured screenshots for longform to remove any browser chrome.
    
    Args:
        apply_vignette: If True, applies a 35% opacity dark vignette overlay on the screenshot content
                       to reduce visual clutter and keep focus on presenter/captions.
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
    
    # ── OPTIONAL: Apply vignette/dimming overlay on screenshot content ─────────
    # Reduces visual clutter from dense web pages (small text, busy UI)
    if apply_vignette:
        # Create a dark overlay with 35% opacity
        vignette = Image.new("RGBA", (fw, fh), (0, 0, 0, int(255 * 0.35)))
        # Apply vignette with gradient - darker at edges, lighter in center
        vignette_arr = np.array(vignette)
        h_v, w_v = vignette_arr.shape[:2]
        Y, X = np.ogrid[:h_v, :w_v]
        center_y, center_x = h_v // 2, w_v // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        # Normalize distance and invert (center = 1.0, edges = 0.0)
        vignette_alpha = 1.0 - (dist / max_dist)
        # Apply 35% opacity at edges, 15% at center
        vignette_arr[:, :, 3] = (vignette_arr[:, :, 3] * (0.15 + 0.20 * vignette_alpha)).astype(np.uint8)
        vignette = Image.fromarray(vignette_arr, "RGBA")
        fg_resized = Image.alpha_composite(fg_resized, vignette)
    
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


def _build_layout_bg_clip(vp, clip_dur, layout, chunk_idx):
    """
    Builds a composite background layer clip for a specific chunk.
    Applies the layout crop bounds, overscan calculation, and a zero-resize drift pan.
    """
    layout_type = layout.get("layout_type", "asymmetric")
    
    # Target visual asset dimensions
    if layout_type == "split_screen":
        target_w = FRAME_W
        target_h = int(FRAME_H * 0.58)
        pos = (0, int(FRAME_H * 0.42))
    elif layout_type == "hero_center":
        target_w = FRAME_W
        target_h = int(FRAME_H * 0.60)
        pos = (0, int(FRAME_H * 0.20))
    else:  # asymmetric / full screen
        target_w = FRAME_W
        target_h = FRAME_H
        pos = (0, 0)

    # 15% overscan relative to the target dimensions
    overscan_w = int(target_w * 1.15)
    overscan_h = int(target_h * 1.15)
    
    is_video = vp.endswith(".mp4")
    
    # Load primary asset clip
    if is_video:
        c_clip = VideoFileClip(vp).without_audio()
        if c_clip.duration < clip_dur:
            c_clip = c_clip.with_effects([vfx.Loop(duration=clip_dur)])
        else:
            c_clip = c_clip.subclipped(0, clip_dur)
    else:
        # We will load the image inside make_frame using PIL to pan dynamically,
        # so we don't need a heavy ImageClip structure.
        c_clip = None

    # Calculate crop pan coordinates (drift pan over duration)
    # Drift pan is a moving window crop of size (target_w, target_h)
    # inside the (overscan_w, overscan_h) pre-scaled image.
    dw = overscan_w - target_w
    dh = overscan_h - target_h
    
    # Seed a deterministic pan style per chunk
    rng = random.Random(chunk_idx)
    pan_style = rng.choice(["pan_left_right", "pan_right_left", "pan_top_bottom", "pan_bottom_top", "drift_diagonal"])
    
    def get_crop_coords(t):
        progress = min(t / max(clip_dur, 0.01), 1.0)
        # Slow ease pan
        eased = 2 * progress * progress if progress < 0.5 else 1 - pow(-2 * progress + 2, 2) / 2
        
        if pan_style == "pan_left_right":
            x = int(dw * eased)
            y = dh // 2
        elif pan_style == "pan_right_left":
            x = int(dw * (1.0 - eased))
            y = dh // 2
        elif pan_style == "pan_top_bottom":
            x = dw // 2
            y = int(dh * eased)
        elif pan_style == "pan_bottom_top":
            x = dw // 2
            y = int(dh * (1.0 - eased))
        else:  # drift_diagonal
            x = int(dw * eased)
            y = int(dh * eased)
            
        # Ensure we don't overshoot boundaries
        x = max(0, min(x, dw))
        y = max(0, min(y, dh))
        return x, y, x + target_w, y + target_h

    # Target aspect ratios for crop preprocessing
    target_aspect = target_w / target_h

    if is_video:
        # For video: use transform filter to crop each frame dynamically
        # Since it's a video file, it's already a clip
        w, h = c_clip.size
        current_aspect = w / h
        if current_aspect > target_aspect:
            crop_w = int(h * target_aspect)
            x1 = (w - crop_w) // 2
            c_clip = c_clip.cropped(x1=x1, y1=0, x2=x1 + crop_w, y2=h)
        else:
            crop_h = int(w / target_aspect)
            y1 = (h - crop_h) // 2
            c_clip = c_clip.cropped(x1=0, y1=y1, x2=w, y2=y1 + crop_h)
            
        c_clip = c_clip.resized((overscan_w, overscan_h))
        
        def crop_filter(get_frame, t):
            frame = get_frame(t)
            x1, y1, x2, y2 = get_crop_coords(t)
            cropped_frame = frame[y1:y2, x1:x2]
            # Ensure exact target dimensions
            if cropped_frame.shape[0] != target_h or cropped_frame.shape[1] != target_w:
                cropped_frame = cv2.resize(cropped_frame, (target_w, target_h))
            return cropped_frame
            
        c_clip = c_clip.transform(crop_filter)
    else:
        # For image: pre-scale once using PIL, then create a VideoClip from make_frame
        try:
            pil_img = Image.open(vp).convert("RGB")
            # crop to target aspect ratio first
            w, h = pil_img.size
            current_aspect = w / h
            if current_aspect > target_aspect:
                crop_w = int(h * target_aspect)
                x1 = (w - crop_w) // 2
                pil_img = pil_img.crop((x1, 0, x1 + crop_w, h))
            else:
                crop_h = int(w / target_aspect)
                y1 = (h - crop_h) // 2
                pil_img = pil_img.crop((0, y1, w, y1 + crop_h))
            base_img = ImageOps.fit(pil_img, (overscan_w, overscan_h), Image.LANCZOS)
        except Exception as e:
            print(f"⚠️ Error preparing image for crop: {e}")
            base_img = Image.new("RGB", (overscan_w, overscan_h), (10, 10, 15))
            
        def make_frame(t):
            x1, y1, x2, y2 = get_crop_coords(t)
            cropped_pil = base_img.crop((x1, y1, x2, y2))
            frame_arr = np.array(cropped_pil)
            # Ensure exact target dimensions
            if frame_arr.shape[0] != target_h or frame_arr.shape[1] != target_w:
                frame_arr = cv2.resize(frame_arr, (target_w, target_h))
            return frame_arr
            
        c_clip = VideoClip(make_frame, duration=clip_dur)

    # Apply color grading tint/grade
    c_clip = c_clip.image_transform(apply_tech_grade)

    # Construct the backing/blurred backdrop layers
    if layout_type in ["split_screen", "hero_center"]:
        try:
            if is_video:
                bg_blur = ColorClip(size=(FRAME_W, FRAME_H), color=(10, 10, 15), duration=clip_dur)
            else:
                raw_img = Image.open(vp).convert("RGB")
                bg_img = ImageOps.fit(raw_img, (FRAME_W, FRAME_H), Image.LANCZOS)
                bg_arr = np.array(bg_img)
                bg_arr = cv2.GaussianBlur(bg_arr, (51, 51), 0)
                bg_blur = ImageClip(bg_arr).with_duration(clip_dur).with_opacity(0.35)
        except Exception as e:
            print(f"⚠️ Error constructing backdrop blur: {e}")
            bg_blur = ColorClip(size=(FRAME_W, FRAME_H), color=(10, 10, 15), duration=clip_dur)
            
        c_clip = c_clip.with_position(pos)
        comp = CompositeVideoClip([bg_blur, c_clip], size=(FRAME_W, FRAME_H)).with_duration(clip_dur)
        return comp
    else:
        # Asymmetric layout: heavily blurred full-screen background
        try:
            if is_video:
                bg_blur = ColorClip(size=(FRAME_W, FRAME_H), color=(10, 10, 15), duration=clip_dur)
            else:
                raw_img = Image.open(vp).convert("RGB")
                bg_img = ImageOps.fit(raw_img, (FRAME_W, FRAME_H), Image.LANCZOS)
                bg_arr = np.array(bg_img)
                bg_arr = cv2.GaussianBlur(bg_arr, (71, 71), 0)
                bg_blur = ImageClip(bg_arr).with_duration(clip_dur)
        except Exception as e:
            bg_blur = ColorClip(size=(FRAME_W, FRAME_H), color=(10, 10, 15), duration=clip_dur)
        
        # Scale to overscan once and crop pan the blurred background as well (cinematic drift)
        if not is_video and not isinstance(bg_blur, ColorClip):
            overscan_w_bg = int(FRAME_W * 1.15)
            overscan_h_bg = int(FRAME_H * 1.15)
            bg_blur_pil = Image.fromarray(bg_blur.img)
            base_bg_blur = ImageOps.fit(bg_blur_pil, (overscan_w_bg, overscan_h_bg), Image.LANCZOS)
            dw_bg = overscan_w_bg - FRAME_W
            dh_bg = overscan_h_bg - FRAME_H
            
            def make_bg_frame(t):
                x1, y1, x2, y2 = get_crop_coords(t)
                x1_bg = int(x1 * (dw_bg / max(1, dw)))
                y1_bg = int(y1 * (dh_bg / max(1, dh)))
                cropped_bg = base_bg_blur.crop((x1_bg, y1_bg, x1_bg + FRAME_W, y1_bg + FRAME_H))
                return np.array(cropped_bg)
                
            bg_blur = VideoClip(make_bg_frame, duration=clip_dur)
        return bg_blur

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


# ─── Themed Background Images per Category ──────────────────────────────────
THEMED_BACKGROUNDS = {
    "Quiz & Trivia": "assets/backgrounds/quiz_bg.jpg",
    "AI & Tech Tools": "assets/backgrounds/ai_tools_bg.jpg",
    "Tech Gadgets & Inventions": "assets/backgrounds/gadgets_bg.jpg",
    "Finance & Tech Economy": "assets/backgrounds/finance_bg.jpg",
    "Agentic AI Facts": "assets/backgrounds/agentic_bg.jpg",
    "Facts & Trivia": "assets/backgrounds/facts_bg.jpg",
    "Coding & Development Hacks": "assets/backgrounds/coding_bg.jpg",
}

def _themed_background_clip(duration, category, accent_color, opacity=0.15):
    """
    Creates a themed background image clip with subtle animation and color overlay.
    Falls back to gradient if image not found.
    """
    import os
    bg_path = THEMED_BACKGROUNDS.get(category)
    if not bg_path or not os.path.exists(bg_path):
        return _dual_directional_gradient_clip(duration)
    
    try:
        raw_img = Image.open(bg_path).convert("RGB")
        # Resize to cover frame (Ken Burns style)
        img_w, img_h = raw_img.size
        target_ratio = FRAME_W / FRAME_H
        img_ratio = img_w / img_h
        
        if img_ratio > target_ratio:
            # Image wider - crop sides
            new_w = int(img_h * target_ratio)
            x1 = (img_w - new_w) // 2
            raw_img = raw_img.crop((x1, 0, x1 + new_w, img_h))
        else:
            # Image taller - crop top/bottom
            new_h = int(img_w / target_ratio)
            y1 = (img_h - new_h) // 2
            raw_img = raw_img.crop((0, y1, img_w, y1 + new_h))
        
        raw_img = raw_img.resize((FRAME_W, FRAME_H), Image.Resampling.LANCZOS)
        canvas_arr = np.array(raw_img)
        
        # Apply color overlay (accent color at low opacity)
        accent_rgb = tuple(int(accent_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        overlay = np.full_like(canvas_arr, accent_rgb, dtype=np.uint8)
        canvas_arr = cv2.addWeighted(canvas_arr, 1 - opacity, overlay, opacity, 0)
        
        clip = ImageClip(canvas_arr, duration=duration)
        
        # Subtle Ken Burns zoom (1.0 to 1.05)
        clip = clip.resized(lambda t, d=duration: 1.0 + 0.05 * (t / d))
        
        return clip
    except Exception as e:
        print(f"⚠️ Themed background failed: {e}. Using gradient fallback.")
        return _dual_directional_gradient_clip(duration)


# Particle styles per category for thematic variety
PARTICLE_STYLES = {
    "Quiz & Trivia": "sparkles",        # Golden sparkles for quiz excitement
    "AI & Tech Tools": "digital_rain",   # Matrix-style code rain
    "Tech Gadgets & Inventions": "circuit",  # Circuit board traces
    "Finance & Tech Economy": "coins",    # Floating coins/gold particles
    "Agentic AI Facts": "neural",        # Neural network nodes
    "Facts & Trivia": "stars",           # Twinkling stars
    "Coding & Development Hacks": "brackets",  # Code brackets/symbols
}

def _ambient_particles(duration, accent_color, particle_style="bokeh"):
    n = 35
    random.seed(42)
    
    # Pre-generate particle properties: (x, y, speed, offset, size, streak_len, angle, rotation_speed)
    particles = []
    for _ in range(n):
        px = random.uniform(0, FRAME_W)
        py = random.uniform(0, FRAME_H)
        speed = random.uniform(0.15, 0.55)
        offset = random.uniform(0, FRAME_H)
        p_size = random.uniform(4, 15) if particle_style in ["bokeh", "lens_dust", "sparkles", "stars", "coins"] else random.uniform(8, 25)
        streak_len = random.uniform(15, 40)
        angle = random.uniform(0, 2 * math.pi)
        rotation_speed = random.uniform(-0.5, 0.5)
        particles.append((px, py, speed, offset, p_size, streak_len, angle, rotation_speed))

    def make_frame(t):
        scale_down = 4 if particle_style in ["bokeh", "lens_dust", "sparkles", "stars", "coins"] else 2
        sm_w, sm_h = FRAME_W // scale_down, FRAME_H // scale_down
        img = np.zeros((sm_h, sm_w, 3), dtype=np.uint8)
        
        for px, py, speed, offset, p_size, streak_len, angle, rot_speed in particles:
            # Fall downward with slight horizontal drift
            y = (py + speed * t * 45 + offset) % FRAME_H
            x = (px + math.sin(t * 0.3 + offset) * 20) % FRAME_W
            
            sm_px = int(x / scale_down)
            sm_py = int(y / scale_down)
            sm_size = max(1, int(p_size / scale_down))
            curr_angle = angle + rot_speed * t
            
            if particle_style == "digital_rain":
                # Matrix-style falling characters
                sm_streak = max(2, int(streak_len / scale_down))
                cv2.line(img, (sm_px, sm_py), (sm_px, sm_py + sm_streak), accent_color, max(1, 1//scale_down))
            elif particle_style == "sparkles":
                # Golden sparkles with twinkle
                alpha = 0.5 + 0.5 * math.sin(t * 3 + offset)
                color = tuple(int(c * alpha) for c in accent_color)
                cv2.circle(img, (sm_px, sm_py), sm_size, color, -1)
                # Cross sparkle
                cv2.line(img, (sm_px - sm_size*2, sm_py), (sm_px + sm_size*2, sm_py), color, 1)
                cv2.line(img, (sm_px, sm_py - sm_size*2), (sm_px, sm_py + sm_size*2), color, 1)
            elif particle_style == "circuit":
                # Circuit board traces - horizontal/vertical lines with nodes
                cv2.line(img, (sm_px, sm_py), (sm_px + sm_size*3, sm_py), accent_color, 1)
                cv2.circle(img, (sm_px, sm_py), 2, accent_color, -1)
                cv2.circle(img, (sm_px + sm_size*3, sm_py), 2, accent_color, -1)
            elif particle_style == "coins":
                # Floating gold coins
                coin_color = (0, 215, 255)  # Gold in BGR
                cv2.circle(img, (sm_px, sm_py), sm_size, coin_color, -1)
                cv2.circle(img, (sm_px, sm_py), sm_size, accent_color, 2)
                # Dollar sign
                cv2.putText(img, "$", (sm_px - 3, sm_py + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            elif particle_style == "gears":
                # Rotating gear shapes
                cv2.circle(img, (sm_px, sm_py), sm_size, accent_color, 1)
                cv2.circle(img, (sm_px, sm_py), sm_size - 3, accent_color, 1)
                # Gear teeth
                for i in range(8):
                    a = curr_angle + i * math.pi / 4
                    x1 = int(sm_px + (sm_size - 2) * math.cos(a))
                    y1 = int(sm_py + (sm_size - 2) * math.sin(a))
                    x2 = int(sm_px + sm_size * math.cos(a))
                    y2 = int(sm_py + sm_size * math.sin(a))
                    cv2.line(img, (x1, y1), (x2, y2), accent_color, 1)
            elif particle_style == "neural":
                # Neural network nodes with connections
                cv2.circle(img, (sm_px, sm_py), max(1, sm_size//2), accent_color, -1)
                # Draw connection to next particle (simplified)
            elif particle_style == "brackets":
                # Code brackets and symbols
                symbols = ["{", "}", "[", "]", "()", "=>", "//", "/*", "*/"]
                sym = symbols[random.randint(0, len(symbols)-1)]
                cv2.putText(img, sym, (sm_px, sm_py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, accent_color, 1)
            elif particle_style == "stars":
                # Twinkling stars
                alpha = 0.3 + 0.7 * (math.sin(t * 2 + offset) + 1) / 2
                color = tuple(int(c * alpha) for c in (255, 255, 255))
                cv2.circle(img, (sm_px, sm_py), max(1, 2//scale_down), color, -1)
            else:  # bokeh, lens_dust
                cv2.circle(img, (sm_px, sm_py), sm_size, accent_color, -1)
        
        if particle_style not in ["stars", "digital_rain", "sparkles", "circuit", "coins", "gears", "neural", "brackets"]:
            blur_size = 19 if particle_style in ["bokeh", "lens_dust"] else 7
            img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
            
        return cv2.resize(img, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)

    def make_mask(t):
        scale_down = 4 if particle_style in ["bokeh", "lens_dust"] else 2
        sm_w, sm_h = FRAME_W // scale_down, FRAME_H // scale_down
        mask = np.zeros((sm_h, sm_w), dtype=np.uint8)
        
        for px, py, speed, offset, p_size, streak_len, angle, rotation_speed in particles:
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
    """Displays giant hook text. Redesigned to use centered bold uppercase sans-serif text,
    obsidian card backing, and neon accent border.
    """
    enable_hook = os.environ.get("ENABLE_HOOK_OVERLAY", "1") == "1"
    if not enable_hook or not hook_text:
        return None
        
    dur = min(3.0, total_dur)
    f = gf(68, bold=True)
    max_w = FRAME_W - 120

    # Word-wrap the hook text in ALL CAPS
    words = hook_text.upper().split()
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
    lsp = int(lh * 1.3)
    total_h = lh + (len(lines) - 1) * lsp
    
    # Calculate dimensions for the backdrop block
    max_line_w = max(ts(line, f)[0] for line in lines)
    
    bg_pad_x, bg_pad_y = 40, 25
    block_w = max_line_w + bg_pad_x * 2
    block_h = total_h + bg_pad_y * 2
    
    canvas_h = block_h + 60
    canvas = Image.new("RGBA", (FRAME_W, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    
    # Position of block in canvas
    bx1 = (FRAME_W - block_w) // 2
    by1 = 30
    bx2 = bx1 + block_w
    by2 = by1 + block_h
    
    # Draw glassmorphic background box with neon border outline
    draw.rounded_rectangle(
        [bx1, by1, bx2, by2],
        radius=20,
        fill=(10, 10, 15, 230),
        outline=accent_color,
        width=3
    )

    for i, line in enumerate(lines):
        lw, _ = ts(line, f)
        tx = (FRAME_W - lw) // 2
        ty = by1 + bg_pad_y + i * lsp
        
        # High contrast drop shadow
        for dx, dy in [(-3, -3), (3, -3), (-3, 3), (3, 3), (-2, 0), (2, 0), (0, -2), (0, 2)]:
            draw.text((tx + dx, ty + dy), line, font=f, fill=(0, 0, 0, 220))
            
        # Alternate line colors: Line 1 is the accent color, others are white
        txt_fill = (255, 255, 255, 255)
        if len(lines) > 1 and i == 1:
            txt_fill = (*accent_color, 255) if len(accent_color) == 3 else (204, 255, 0, 255)
        elif len(lines) == 1:
            txt_fill = (*accent_color, 255) if len(accent_color) == 3 else (204, 255, 0, 255)
            
        draw.text((tx, ty), line, font=f, fill=txt_fill)

    arr = np.array(canvas.convert("RGB"))
    mask = np.array(canvas.split()[3]).astype(float) / 255.0

    def opacity_fn(t):
        if t < 0.2:
            return t / 0.2
        elif t > dur - 0.4:
            return max(0, (dur - t) / 0.4)
        return 1.0

    clip = VideoClip(lambda t: arr, duration=dur)
    mclip = VideoClip(lambda t: mask * opacity_fn(t), is_mask=True, duration=dur)
    
    # Center-middle position (moved down to clear entity logo/name/description)
    y_pos = int(FRAME_H * 0.48) - (canvas_h // 2)
    return clip.with_mask(mclip).with_position(("center", y_pos)).with_start(0)


TOGGLE_FLIP_OFFSET_SEC = 1.2


def _create_settings_mockup_clip(text, start_time, duration, accent_color, audio_duration):
    """
    Creates an animated, pre-rendered glassmorphic settings card overlay clip.
    Toggles off, draws a tapping hand cursor, ripple, and a neon outline highlight.
    Precomputed and cached to avoid per-frame draw overhead during main video render.
    """
    enable_mockup = os.environ.get("ENABLE_SETTINGS_MOCKUP", "1") == "1"
    if not enable_mockup:
        return None

    # Calculate actual visual duration
    dur = min(duration, audio_duration - start_time)
    if dur < 0.5:
        return None

    # Animation FPS
    fps = 12
    total_frames = int(dur * fps) + 1
    
    rgb_frames = []
    alpha_masks = []
    
    # Determine settings title/description based on text keywords
    text_l = text.lower()
    if "track" in text_l or "tracking" in text_l or "privacy" in text_l:
        title = "Allow Apps to Request to Track"
        desc = "Let apps ask to track activity across other apps."
    elif "location" in text_l or "gps" in text_l or "map" in text_l:
        title = "Location Services"
        desc = "Share your location with apps in the background."
    elif "ad" in text_l or "ads" in text_l or "personalized" in text_l:
        title = "Personalized Ads"
        desc = "Apple advertising privacy settings."
    elif "analytics" in text_l or "data" in text_l or "share" in text_l:
        title = "Share iPhone Analytics"
        desc = "Send diagnostic data to Apple daily."
    else:
        title = "Background App Refresh"
        desc = "Allow apps to refresh content in background."

    # Dimensions
    card_w, card_h = 920, 200
    card_x1 = (FRAME_W - card_w) // 2
    # Use lower-middle safe zone (55%-75%)
    card_y1 = int(FRAME_H * 0.65)
    card_x2 = card_x1 + card_w
    card_y2 = card_y1 + card_h

    toggle_w, toggle_h = 100, 52
    toggle_x1 = card_x2 - toggle_w - 40
    toggle_y1 = card_y1 + (card_h - toggle_h) // 2
    toggle_x2 = toggle_x1 + toggle_w
    toggle_y2 = toggle_y1 + toggle_h

    # Switch flips OFF at TOGGLE_FLIP_OFFSET_SEC into the chunk
    flip_offset = TOGGLE_FLIP_OFFSET_SEC

    for frame_idx in range(total_frames):
        t_offset = frame_idx / float(fps)
        
        # Transparent canvas
        canvas = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        # 1. Glassmorphic card background
        draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=24, fill=(18, 19, 21, 235), outline=(255, 255, 255, 30), width=2)
        
        f_title = gf(32, bold=True)
        f_desc = gf(20, bold=False)
        
        draw.text((card_x1 + 40, card_y1 + 45), title, font=f_title, fill=(255, 255, 255, 255))
        draw.text((card_x1 + 40, card_y1 + 105), desc, font=f_desc, fill=(160, 160, 165, 255))
        
        # 2. Toggle Switch State
        is_on = t_offset < flip_offset
        if is_on:
            draw.rounded_rectangle([toggle_x1, toggle_y1, toggle_x2, toggle_y2], radius=toggle_h//2, fill=(48, 209, 88, 255))
            knob_x = toggle_x2 - toggle_h // 2
        else:
            draw.rounded_rectangle([toggle_x1, toggle_y1, toggle_x2, toggle_y2], radius=toggle_h//2, fill=(120, 120, 128, 255))
            knob_x = toggle_x1 + toggle_h // 2
            
        knob_y = toggle_y1 + toggle_h // 2
        knob_r = 22
        draw.ellipse([knob_x - knob_r, knob_y - knob_r, knob_x + knob_r, knob_y + knob_r], fill=(255, 255, 255, 255))
        
        # 3. Tapping ripple animation (lasts 0.4 seconds after flip)
        if not is_on and (0 <= t_offset - flip_offset < 0.4):
            ripple_progress = (t_offset - flip_offset) / 0.4
            ripple_r = int(22 + 45 * ripple_progress)
            ripple_opacity = int(200 * (1.0 - ripple_progress))
            
            # Draw ripple in RGBA using separate image overlay
            ripple_img = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
            r_draw = ImageDraw.Draw(ripple_img)
            r_draw.ellipse([knob_x - ripple_r, knob_y - ripple_r, knob_x + ripple_r, knob_y + ripple_r], outline=(*accent_color, ripple_opacity), width=4)
            canvas.alpha_composite(ripple_img)
            draw = ImageDraw.Draw(canvas) # reset draw
            
        # 4. Neon bounding box highlight (lasts 0.6 seconds after flip)
        if not is_on and (0 <= t_offset - flip_offset < 0.6):
            box_img = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
            box_draw = ImageDraw.Draw(box_img)
            box_draw.rounded_rectangle([card_x1 - 4, card_y1 - 4, card_x2 + 4, card_y2 + 4], radius=28, outline=(255, 40, 150, 255), width=5)
            canvas.alpha_composite(box_img)
            draw = ImageDraw.Draw(canvas) # reset draw
            
        # 5. Hand cursor sliding in to click
        if t_offset < flip_offset + 0.5:
            if t_offset < flip_offset:
                # Sliding in from bottom right
                p = t_offset / flip_offset
                hx = int(FRAME_W + (toggle_x2 - FRAME_W) * p)
                hy = int(FRAME_H + (knob_y - FRAME_H) * p)
            else:
                # Sliding out
                p = (t_offset - flip_offset) / 0.5
                hx = int(knob_x + (FRAME_W - knob_x) * p)
                hy = int(knob_y + (FRAME_H - knob_y) * p)
                
            pointer_img = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
            p_draw = ImageDraw.Draw(pointer_img)
            p_draw.polygon([(hx, hy), (hx + 20, hy + 40), (hx + 40, hy + 20)], fill=(255, 214, 0, 255), outline=(0,0,0,255), width=2)
            canvas.alpha_composite(pointer_img)
            draw = ImageDraw.Draw(canvas) # reset draw
            
        # Store frame arrays
        rgb_frames.append(np.array(canvas.convert("RGB")))
        alpha_masks.append(np.array(canvas.split()[3]).astype(float) / 255.0)

    # Frame retrieval functions
    def make_rgb_frame(t):
        idx = min(int(t * fps), len(rgb_frames) - 1)
        return rgb_frames[idx]
        
    def make_mask_frame(t):
        idx = min(int(t * fps), len(alpha_masks) - 1)
        return alpha_masks[idx]

    clip = VideoClip(make_rgb_frame, duration=dur)
    mclip = VideoClip(make_mask_frame, is_mask=True, duration=dur)
    
    # Smooth fade-in/fade-out
    clip = clip.with_mask(mclip).with_effects([vfx.CrossFadeIn(0.25), vfx.CrossFadeOut(0.25)])
    return clip.with_start(start_time)


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
            # Use upper-middle safe zone
            return (min(x, FRAME_W - 50), int(FRAME_H * 0.425))

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
    # Use upper-middle safe zone (30%-55%)
    y_pos = int(FRAME_H * 0.425)
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


# ── LAYER QUIZ: Quiz CTA Overlay (Last 5s) ─────────────────────────────────────
def _quiz_cta_overlay(comment_hook, incentive_cta_type, digital_asset_offer, accent_color, total_dur):
    """Quiz-specific CTA for the final moments - trivia-focused engagement language."""
    dur = 4.0
    start = total_dur - dur
    if start < 0:
        return None

    f_main = gf(38, bold=True)
    f_sub = gf(28)
    
    # Build CTA text - trivia-focused engagement (per feedback)
    cta_lines = [
        "Did you get it right? Comment your answer below! 👇",
        "What trivia question should we do next?"
    ]
    
    # Keep incentive-specific variants as backup
    if incentive_cta_type == "benchmark_challenge":
        cta_lines = [
            "Think you know tech? Comment your score! 🏆",
            "Beat the high score below"
        ]
    elif incentive_cta_type == "digital_vault":
        cta_lines = [
            f"Get {digital_asset_offer or '50 tech quizzes'} free!",
            "🔗 Link in bio →"
        ]
    elif incentive_cta_type == "community_audit":
        cta_lines = [
            "Sub & comment your score for monthly giveaway!",
            "💰 $100 API credits up for grabs"
        ]
    
    max_w = FRAME_W - 120
    lines = []
    for line in cta_lines:
        words = line.split()
        cur = []
        for w in words:
            test = " ".join(cur + [w])
            if ts(test, f_main if len(lines) == 0 else f_sub)[0] > max_w and cur:
                lines.append(" ".join(cur))
                cur = [w]
            else:
                cur.append(w)
        if cur:
            lines.append(" ".join(cur))
    
    # Calculate dimensions
    lh_main = ts("Ag", f_main)[1]
    lh_sub = ts("Ag", f_sub)[1]
    lsp_main = int(lh_main * 1.3)
    lsp_sub = int(lh_sub * 1.3)
    
    total_h = 0
    for i, line in enumerate(lines):
        if i == 0:
            total_h += lh_main
        else:
            total_h += lsp_sub
    total_h += 20 * (len(lines) - 1)  # Extra gap
    
    box_h = total_h + 60
    box_w = FRAME_W - 80
    
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Glassmorphism card with accent border
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=20, fill=(10, 15, 25, 220))
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=20, outline=(*accent_color, 200), width=3)
    # Accent top bar
    draw.rectangle([0, 0, box_w, 4], fill=(*accent_color, 255))
    
    y = 30
    for i, line in enumerate(lines):
        font = f_main if i == 0 else f_sub
        lw, _ = ts(line, font)
        tx = (box_w - lw) // 2
        color = (*accent_color, 255) if i == 0 else (255, 255, 255, 230)
        draw.text((tx, y), line, font=font, fill=color)
        y += lsp_main if i == 0 else lsp_sub
        y += 15  # Gap between lines
    
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
    y_pos = int(FRAME_H - TITLE_BOTTOM_GAP - box_h - 100)
    return clip.with_mask(mclip).with_position((x_pos, y_pos)).with_start(start)


# ── ANIMATED COMMENT CTA GRAPHIC ───────────────────────────────────────────────
def _animated_comment_cta(comment_keyword, accent_color, total_dur):
    """
    Creates an animated callout graphic showing a chat box with the keyword highlighted.
    Appears in the last 5 seconds of the video.
    Reference: Feedback recommendation for visual "Comment 'ROBOT'" cue.
    """
    dur = 5.0
    start = total_dur - dur
    if start < 0:
        return None
    
    f_main = gf(42, bold=True)
    f_keyword = gf(48, bold=True)
    f_sub = gf(28)
    
    # Chat bubble dimensions
    bubble_w = int(FRAME_W * 0.85)
    bubble_h = 180
    
    # Create animated frames
    fps = 15
    total_frames = int(dur * fps)
    
    rgb_frames = []
    alpha_masks = []
    
    for frame_idx in range(total_frames):
        t = frame_idx / fps
        progress = t / dur
        
        img = Image.new("RGBA", (bubble_w, bubble_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Glassmorphic chat bubble background
        draw.rounded_rectangle([0, 0, bubble_w - 1, bubble_h - 1], radius=24, 
                               fill=(10, 15, 25, 230), outline=(*accent_color, 200), width=3)
        
        # Animated pulse border
        pulse = 1.0 + 0.1 * math.sin(t * 6)
        border_w = max(2, int(3 * pulse))
        draw.rounded_rectangle([0, 0, bubble_w - 1, bubble_h - 1], radius=24, 
                               outline=(*accent_color, int(255 * pulse)), width=border_w)
        
        # Chat icon
        icon_size = 40
        icon_x = 30
        icon_y = (bubble_h - icon_size) // 2
        draw.ellipse([icon_x, icon_y, icon_x + icon_size, icon_y + icon_size], 
                     fill=(*accent_color, 200))
        draw.text((icon_x + icon_size // 2, icon_y + icon_size // 2), "💬", 
                  font=gf(24), fill=(255, 255, 255, 255), anchor="mm")
        
        # "Comment:" label
        label_x = icon_x + icon_size + 20
        label_y = 25
        draw.text((label_x, label_y), "Comment:", font=f_sub, fill=(180, 180, 200, 255))
        
        # Keyword with highlight animation
        keyword_x = label_x
        keyword_y = label_y + 35
        
        # Animate keyword: scale up slightly, color pulse
        keyword_scale = 1.0 + 0.05 * math.sin(t * 8)
        keyword_color = (*accent_color, 255)
        
        # Draw keyword with highlight background
        kw_text = f"'{comment_keyword}'"
        kw_w = draw.textbbox((0, 0), kw_text, font=f_keyword)[2]
        kw_h = draw.textbbox((0, 0), kw_text, font=f_keyword)[3]
        
        # Highlight pill behind keyword
        pill_pad = 12
        pill_x1 = keyword_x - pill_pad
        pill_y1 = keyword_y - 4
        pill_x2 = keyword_x + kw_w + pill_pad
        pill_y2 = keyword_y + kw_h + 4
        
        # Pulsing highlight
        highlight_alpha = int(180 + 75 * math.sin(t * 8))
        draw.rounded_rectangle([pill_x1, pill_y1, pill_x2, pill_y2], radius=12, 
                               fill=(*accent_color, highlight_alpha))
        
        draw.text((keyword_x, keyword_y), kw_text, font=f_keyword, fill=(255, 255, 255, 255))
        
        # Sub-text - trivia-focused engagement (per feedback)
        sub_text = "Did you get it right?"
        draw.text((keyword_x, keyword_y + kw_h + 15), sub_text, font=f_sub, fill=(200, 200, 220, 255))
        
        # Animated arrow pointing down
        arrow_y = bubble_h - 35 + int(8 * math.sin(t * 4))
        draw.text((bubble_w // 2, arrow_y), "▼", font=gf(28), fill=(*accent_color, 255), anchor="mm")
        
        # Store frame
        rgb_frames.append(np.array(img.convert("RGB")))
        alpha_masks.append(np.array(img.split()[3]).astype(float) / 255.0)
    
    def make_frame(t):
        idx = min(int(t * fps), len(rgb_frames) - 1)
        return rgb_frames[idx]
    
    def make_mask(t):
        idx = min(int(t * fps), len(alpha_masks) - 1)
        # Fade in/out
        opacity = 1.0
        if t < 0.4:
            opacity = t / 0.4
        elif t > dur - 0.5:
            opacity = max(0, (dur - t) / 0.5)
        return alpha_masks[idx] * opacity
    
    clip = VideoClip(make_frame, duration=dur)
    mclip = VideoClip(make_mask, is_mask=True, duration=dur)
    
    # Position in lower-middle safe zone (above bottom UI, below presenter)
    x_pos = (FRAME_W - bubble_w) // 2
    y_pos = int(FRAME_H * 0.68)  # Lower-middle safe zone
    
    return clip.with_mask(mclip).with_position((x_pos, y_pos)).with_start(start)


# ── SAVE/SHARE VISUAL CUES ────────────────────────────────────────────────────
# Animated on-screen prompts for Save and Share CTAs (appears at CTA moment)

def _save_cta_overlay(accent_color, total_dur, cta_text="Save this for later!"):
    """
    Creates an animated 'Save' icon with text that appears at the CTA moment.
    Uses a bookmark/flag icon with pulse animation.
    """
    dur = 4.0
    start = total_dur - dur
    if start < 0:
        return None
    
    f_main = gf(42, bold=True)
    f_sub = gf(28)
    
    # Card dimensions
    card_w = int(FRAME_W * 0.8)
    card_h = 140
    
    fps = 15
    total_frames = int(dur * fps)
    
    rgb_frames = []
    alpha_masks = []
    
    for frame_idx in range(total_frames):
        t = frame_idx / fps
        progress = t / dur
        
        img = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Glassmorphic card background
        draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=20, 
                               fill=(10, 15, 25, 230), outline=(*accent_color, 200), width=3)
        
        # Animated pulse border
        pulse = 1.0 + 0.12 * math.sin(t * 5)
        border_w = max(2, int(3 * pulse))
        draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=20, 
                               outline=(*accent_color, int(255 * pulse)), width=border_w)
        
        # Bookmark/Save icon
        icon_size = 50
        icon_x = 30
        icon_y = (card_h - icon_size) // 2
        # Draw bookmark shape
        draw.rounded_rectangle([icon_x, icon_y, icon_x + icon_size, icon_y + icon_size], radius=8, 
                               fill=(*accent_color, 200))
        # Bookmark cutout
        draw.polygon([
            (icon_x + 15, icon_y + 8),
            (icon_x + 35, icon_y + 8),
            (icon_x + 25, icon_y + 20),
            (icon_x + 35, icon_y + 32),
            (icon_x + 15, icon_y + 32),
        ], fill=(255, 255, 255, 255))
        
        # Main text
        text_x = icon_x + icon_size + 20
        text_y = 20
        draw.text((text_x, text_y), "📌 SAVE THIS", font=f_main, fill=(255, 255, 255, 255))
        
        # Sub text
        draw.text((text_x, text_y + 55), cta_text, font=f_sub, fill=(200, 200, 220, 255))
        
        # Animated arrow/indicator
        arrow_x = card_w - 60
        arrow_y = card_h // 2 + int(10 * math.sin(t * 4))
        draw.text((arrow_x, arrow_y), "▶", font=gf(32), fill=(*accent_color, 255), anchor="mm")
        
        # Sparkle particles
        for i in range(3):
            sparkle_x = card_w - 100 + i * 25 + int(5 * math.sin(t * 3 + i))
            sparkle_y = 30 + int(5 * math.cos(t * 3 + i))
            sparkle_alpha = int(200 + 55 * math.sin(t * 6 + i))
            draw.text((sparkle_x, sparkle_y), "✨", font=gf(20), fill=(255, 255, 200, sparkle_alpha), anchor="mm")
        
        # Store frame
        rgb_frames.append(np.array(img.convert("RGB")))
        alpha_masks.append(np.array(img.split()[3]).astype(float) / 255.0)
    
    def make_frame(t):
        idx = min(int(t * fps), len(rgb_frames) - 1)
        return rgb_frames[idx]
    
    def make_mask(t):
        idx = min(int(t * fps), len(alpha_masks) - 1)
        opacity = 1.0
        if t < 0.3:
            opacity = t / 0.3
        elif t > dur - 0.4:
            opacity = max(0, (dur - t) / 0.4)
        return alpha_masks[idx] * opacity
    
    clip = VideoClip(make_frame, duration=dur)
    mclip = VideoClip(make_mask, is_mask=True, duration=dur)
    
    # Position in lower-middle safe zone
    x_pos = (FRAME_W - card_w) // 2
    y_pos = int(FRAME_H * 0.65)
    
    return clip.with_mask(mclip).with_position((x_pos, y_pos)).with_start(start)


def _share_cta_overlay(accent_color, total_dur, cta_text="Send this to a developer!"):
    """
    Creates an animated 'Share' icon with text that appears at the CTA moment.
    Uses a share arrow icon with pulse animation.
    """
    dur = 4.0
    start = total_dur - dur
    if start < 0:
        return None
    
    f_main = gf(42, bold=True)
    f_sub = gf(28)
    
    # Card dimensions
    card_w = int(FRAME_W * 0.8)
    card_h = 140
    
    fps = 15
    total_frames = int(dur * fps)
    
    rgb_frames = []
    alpha_masks = []
    
    for frame_idx in range(total_frames):
        t = frame_idx / fps
        progress = t / dur
        
        img = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Glassmorphic card background
        draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=20, 
                               fill=(10, 15, 25, 230), outline=(*accent_color, 200), width=3)
        
        # Animated pulse border - different color for share
        share_color = (255, 100, 100)  # Coral/pink for share
        pulse = 1.0 + 0.12 * math.sin(t * 5)
        border_w = max(2, int(3 * pulse))
        draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=20, 
                               outline=(*share_color, int(255 * pulse)), width=border_w)
        
        # Share icon (three dots connected)
        icon_size = 50
        icon_x = 30
        icon_y = (card_h - icon_size) // 2
        # Draw three circles with connecting lines
        for i in range(3):
            cx = icon_x + 10 + i * 18
            cy = icon_y + 25
            draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(*share_color, 200))
            if i < 2:
                draw.line([cx + 8, cy, cx + 10, cy], fill=(*share_color, 200), width=3)
        
        # Main text
        text_x = icon_x + icon_size + 20
        text_y = 20
        draw.text((text_x, text_y), "📤 SHARE THIS", font=f_main, fill=(255, 255, 255, 255))
        
        # Sub text
        draw.text((text_x, text_y + 55), cta_text, font=f_sub, fill=(200, 200, 220, 255))
        
        # Animated share arrow
        arrow_x = card_w - 60
        arrow_y = card_h // 2 + int(10 * math.sin(t * 4))
        draw.text((arrow_x, arrow_y), "➤", font=gf(32), fill=(*share_color, 255), anchor="mm")
        
        # Sparkle particles
        for i in range(3):
            sparkle_x = card_w - 100 + i * 25 + int(5 * math.sin(t * 3 + i))
            sparkle_y = 30 + int(5 * math.cos(t * 3 + i))
            sparkle_alpha = int(200 + 55 * math.sin(t * 6 + i))
            draw.text((sparkle_x, sparkle_y), "✨", font=gf(20), fill=(255, 200, 200, sparkle_alpha), anchor="mm")
        
        # Store frame
        rgb_frames.append(np.array(img.convert("RGB")))
        alpha_masks.append(np.array(img.split()[3]).astype(float) / 255.0)
    
    def make_frame(t):
        idx = min(int(t * fps), len(rgb_frames) - 1)
        return rgb_frames[idx]
    
    def make_mask(t):
        idx = min(int(t * fps), len(alpha_masks) - 1)
        opacity = 1.0
        if t < 0.3:
            opacity = t / 0.3
        elif t > dur - 0.4:
            opacity = max(0, (dur - t) / 0.4)
        return alpha_masks[idx] * opacity
    
    clip = VideoClip(make_frame, duration=dur)
    mclip = VideoClip(make_mask, is_mask=True, duration=dur)
    
    # Position in lower-middle safe zone
    x_pos = (FRAME_W - card_w) // 2
    y_pos = int(FRAME_H * 0.65)
    
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


def _render_quiz_options_card(option_a, option_b, option_c, accent_color, width=960):
    """Render three quiz options (A, B, C) as vertical cards for mobile viewing."""
    f_label = gf(28, bold=True)
    f_text = gf(32)
    
    card_w = width - 80
    card_h = 110
    gap = 20
    total_h = card_h * 3 + gap * 2
    
    img = Image.new("RGBA", (width, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    options = [
        ("A", option_a, (255, 215, 0)),  # Gold for A
        ("B", option_b, (100, 200, 255)),  # Blue for B
        ("C", option_c, (255, 120, 120)),  # Red/pink for C
    ]
    
    for i, (letter, text, label_color) in enumerate(options):
        y = i * (card_h + gap)
        # Card background
        draw.rounded_rectangle([40, y, width - 40, y + card_h], radius=16,
                               fill=(15, 15, 25, 230), outline=(*label_color, 180), width=3)
        
        # Option letter circle
        circle_x = 80
        circle_y = y + card_h // 2
        draw.ellipse([circle_x - 28, circle_y - 28, circle_x + 28, circle_y + 28],
                     fill=(*label_color, 255))
        draw.text((circle_x, circle_y), letter, font=gf(36, bold=True),
                  fill=(15, 15, 25, 255), anchor="mm")
        
        # Option text
        # Wrap text if too long
        max_text_w = card_w - 100
        words = text.split()
        lines = []
        current_line = []
        current_len = 0
        for w in words:
            test_line = " ".join(current_line + [w])
            tw, _ = ts(test_line, f_text)
            if tw > max_text_w and current_line:
                lines.append(" ".join(current_line))
                current_line = [w]
            else:
                current_line.append(w)
        if current_line:
            lines.append(" ".join(current_line))
        
        line_h = 38
        text_y_start = y + (card_h - len(lines) * line_h) // 2
        for li, line in enumerate(lines):
            draw.text((circle_x + 50, text_y_start + li * line_h), line,
                      font=f_text, fill=(240, 240, 250, 255))
    
    return img


def _render_quiz_reveal_card(correct_letter, correct_text, accent_color, width=960):
    """Render the answer reveal with correct option highlighted in green."""
    f_title = gf(36, bold=True)
    f_text = gf(40, bold=True)
    f_desc = gf(26)
    
    card_w = width - 80
    card_h = 160
    
    img = Image.new("RGBA", (width, card_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Green highlight for correct answer
    green = (48, 209, 88)
    
    # Background card
    draw.rounded_rectangle([40, 0, width - 40, card_h], radius=18,
                           fill=(10, 25, 15, 240), outline=(*green, 255), width=4)
    
    # "CORRECT ANSWER" label
    draw.text((width // 2, 30), "✅  CORRECT ANSWER", font=f_title,
              fill=(*green, 255), anchor="mm")
    
    # Option letter in large circle
    circle_x = width // 2 - 100
    circle_y = card_h // 2 + 10
    draw.ellipse([circle_x - 40, circle_y - 40, circle_x + 40, circle_y + 40],
                 fill=(*green, 255))
    draw.text((circle_x, circle_y), correct_letter, font=gf(56, bold=True),
              fill=(255, 255, 255, 255), anchor="mm")
    
    # Correct option text
    max_text_w = card_w - 160
    words = correct_text.split()
    lines = []
    current_line = []
    for w in words:
        test_line = " ".join(current_line + [w])
        tw, _ = ts(test_line, f_text)
        if tw > max_text_w and current_line:
            lines.append(" ".join(current_line))
            current_line = [w]
        else:
            current_line.append(w)
    if current_line:
        lines.append(" ".join(current_line))
    
    line_h = 48
    text_y_start = card_h // 2 - (len(lines) * line_h) // 2 + 10
    for li, line in enumerate(lines):
        draw.text((circle_x + 60, text_y_start + li * line_h), line,
                  font=f_text, fill=(255, 255, 255, 255))
    
    return img


def _render_quiz_countdown(number, accent_color, width=960, with_progress=False, progress_pct=0.0):
    """Render a large countdown number (4, 3, 2, 1) for the quiz pause with optional progress bar."""
    f_num = gf(180, bold=True)
    
    # Increased height for progress bar
    img_height = 380 if with_progress else 300
    img = Image.new("RGBA", (width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Glowing background circle
    center_x = width // 2
    center_y = 150
    radius = 100
    
    # Outer glow
    for r in range(radius + 20, radius, -2):
        alpha = int(30 * (radius + 20 - r) / 20)
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r],
                     outline=(*accent_color, alpha), width=2)
    
    # Main circle
    draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                 fill=(15, 15, 25, 240), outline=(*accent_color, 255), width=6)
    
    # Countdown number
    draw.text((center_x, center_y), str(number), font=f_num,
              fill=(255, 255, 255, 255), anchor="mm")
    
    # "THINK..." label below circle
    f_label = gf(36, bold=True)
    draw.text((center_x, center_y + radius + 30), "THINK...", font=f_label,
              fill=(*accent_color, 255), anchor="mm")
    
    # Visual progress bar at bottom (NEW)
    if with_progress:
        bar_y = center_y + radius + 80
        bar_width = int(width * 0.6)
        bar_height = 12
        bar_x = (width - bar_width) // 2
        
        # Background track
        draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height], 
                              radius=6, fill=(30, 30, 40, 200))
        
        # Progress fill
        fill_width = int(bar_width * progress_pct)
        if fill_width > 0:
            draw.rounded_rectangle([bar_x, bar_y, bar_x + fill_width, bar_y + bar_height], 
                                  radius=6, fill=(*accent_color, 255))
        
        # Progress label
        f_progress = gf(24, bold=True)
        draw.text((center_x, bar_y + bar_height + 15), f"{int(progress_pct * 100)}% READY", 
                  font=f_progress, fill=(*accent_color, 200), anchor="mm")
    
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
      definition       → {"term": "RAG", "definition": "Retrieval-Augmented Generation..."}
      stat             → {"value": "$4.6B", "label": "OpenAI 2025 Revenue"}
      comparison       → {"left_label": "GPT-4", "left_val": "128K ctx",
                         "right_label": "GPT-5", "right_val": "1M ctx"}
      process          → {"steps": ["Fetch", "Embed", "Retrieve", "Generate"]}
      flowchart        → {"steps": ["Step A", "Step B", "Step C"]}
      quiz_options     → {"option_a": "Moth in relay", "option_b": "Software error", "option_c": "Power surge"}
      quiz_reveal      → {"correct_letter": "A", "correct_text": "Moth in relay"}
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
            elif itype == "quiz_options":
                pil = _render_quiz_options_card(
                    current_data.get("option_a", "Option A"),
                    current_data.get("option_b", "Option B"),
                    current_data.get("option_c", "Option C"),
                    accent_color
                )
            elif itype == "quiz_reveal":
                pil = _render_quiz_reveal_card(
                    current_data.get("correct_letter", "A"),
                    current_data.get("correct_text", "Correct Answer"),
                    accent_color
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


# ── CATEGORY-SPECIFIC AVATAR STYLE ─────────────────────────────────────────────
def _get_category_avatar_style(category: str, is_shorts: bool = False) -> dict:
    """
    Returns category-specific visual style for avatar PiP.
    Adds visual variety while keeping layout-based positioning.
    """
    category = category.lower().strip()
    
    styles = {
        "quiz & trivia": {
            "scale_mult": 1.0,
            "entrance_style": "pop_in",      # Quick pop with bounce
            "glow_style": "pulse_fast",      # Fast pulse for energy
            "accent_tint": (255, 215, 0),    # Gold accent
            "border_style": "double_ring",   # Double ring for quiz vibe
        },
        "ai & tech tools": {
            "scale_mult": 1.0,
            "entrance_style": "slide_right", # Tech slide-in
            "glow_style": "pulse_medium",
            "accent_tint": (0, 230, 255),
            "border_style": "neon_cyan",     # Cyan neon for tech
        },
        "tech gadgets & inventions": {
            "scale_mult": 1.05,
            "entrance_style": "zoom_reveal", # Zoom from center
            "glow_style": "pulse_slow",
            "accent_tint": (255, 110, 0),    # Orange for excitement
            "border_style": "gradient_ring", # Colorful gradient
        },
        "finance & tech economy": {
            "scale_mult": 0.95,
            "entrance_style": "slide_up",    # Professional slide up
            "glow_style": "pulse_medium",
            "accent_tint": (255, 215, 0),
            "border_style": "gold_ring",     # Gold for finance
        },
        "facts & trivia": {
            "scale_mult": 1.0,
            "entrance_style": "fade_in",     # Clean fade
            "glow_style": "steady",          # Steady glow for facts
            "accent_tint": (86, 217, 160),   # Green for knowledge
            "border_style": "single_ring",
        },
        "coding & development hacks": {
            "scale_mult": 1.0,
            "entrance_style": "typewriter",  # Type-on effect
            "glow_style": "pulse_medium",
            "accent_tint": (50, 255, 50),
            "border_style": "terminal_green", # Terminal aesthetic
        },
        "interview questions": {
            "scale_mult": 1.0,
            "entrance_style": "slide_left",  # Question from left
            "glow_style": "pulse_fast",
            "accent_tint": (197, 121, 230),  # Purple for questions
            "border_style": "double_ring",
        },
        "programming language origins": {
            "scale_mult": 1.0,
            "entrance_style": "zoom_reveal",
            "glow_style": "pulse_slow",
            "accent_tint": (95, 158, 255),   # Cornflower blue
            "border_style": "gradient_ring",
        },
        "tech company founding stories": {
            "scale_mult": 1.0,
            "entrance_style": "fade_in",
            "glow_style": "steady",
            "accent_tint": (255, 182, 193),  # Light pink for stories
            "border_style": "single_ring",
        },
        "famous bugs & glitches": {
            "scale_mult": 1.0,
            "entrance_style": "glitch_in",   # Glitch effect for bugs
            "glow_style": "pulse_fast",
            "accent_tint": (255, 80, 80),
            "border_style": "red_ring",      # Red for bugs
        },
        "agentic ai facts": {
            "scale_mult": 1.05,
            "entrance_style": "slide_right",
            "glow_style": "pulse_medium",
            "accent_tint": (0, 230, 255),
            "border_style": "neon_cyan",
        },
    }
    
    # Default fallback
    default = {
        "scale_mult": 1.0,
        "entrance_style": "fade_in",
        "glow_style": "pulse_medium",
        "accent_tint": None,  # Use layout accent
        "border_style": "single_ring",
    }
    
    return styles.get(category, default)


# ── Avatar frame with entrance animations (no circular mask, no border, no ring) ────────────
def _apply_circular_facecam_frame(avatar_clip, cur_w, cur_h, accent_color, audio_duration, is_longform=False, cat_style=None):
    """
    Returns avatar clip with entrance animation only (no circular mask, no border, no ring).
    Kept for API compatibility with calling code.
    """
    entrance_style = cat_style.get("entrance_style", "fade_in") if cat_style else "fade_in"
    
    def entrance_transform(t):
        """Returns (scale, opacity, x_offset, y_offset) for entrance animation"""
        progress = min(t / 0.8, 1.0)  # 800ms entrance duration
        
        if entrance_style == "pop_in":
            # Quick pop with bounce (cubic ease-out + overshoot)
            p = 1.0 - (1.0 - progress) ** 3
            scale = 0.3 + 0.7 * p + 0.15 * math.sin(progress * 4 * math.pi) * (1 - progress)
            return scale, progress, 0, 0
            
        elif entrance_style == "slide_right":
            # Slide in from right
            p = 1.0 - (1.0 - progress) ** 2
            scale = 1.0
            x_offset = (cur_w * 0.5) * (1 - p)
            return scale, p, x_offset, 0
            
        elif entrance_style == "zoom_reveal":
            # Zoom from center
            p = progress ** 2
            scale = 0.1 + 0.9 * p
            return scale, p, 0, 0
            
        elif entrance_style == "slide_up":
            # Professional slide up
            p = 1.0 - (1.0 - progress) ** 2
            y_offset = (cur_h * 0.3) * (1 - p)
            return 1.0, p, 0, y_offset
            
        elif entrance_style == "slide_left":
            # Slide in from left
            p = 1.0 - (1.0 - progress) ** 2
            x_offset = (-cur_w * 0.5) * (1 - p)
            return 1.0, p, x_offset, 0
            
        elif entrance_style == "glitch_in":
            # Glitch effect - rapid position jitter
            glitch_intensity = (1 - progress) * 0.15
            jitter_x = glitch_intensity * cur_w * math.sin(t * 50) * (1 - progress)
            jitter_y = glitch_intensity * cur_h * math.cos(t * 47) * (1 - progress)
            return 1.0, progress, jitter_x, jitter_y
            
        elif entrance_style == "typewriter":
            # Typewriter - character by character reveal
            p = progress
            return 1.0, p, 0, 0
            
        else:  # fade_in default
            return 1.0, progress, 0, 0
    
    # Apply entrance transform to avatar clip using Resize effect
    def entrance_scale_fn(t):
        scale, _, _, _ = entrance_transform(t)
        return scale
    
    def entrance_opacity_fn(t):
        _, opacity, _, _ = entrance_transform(t)
        return opacity
    
    avatar_clip = avatar_clip.with_effects([
        vfx.Resize(entrance_scale_fn),
    ])
    
    # Apply opacity animation via a simple fade-in mask (full frame, not circular)
    entrance_mask = VideoClip(lambda t: np.full((cur_h, cur_w), entrance_opacity_fn(t)), is_mask=True, duration=audio_duration)
    avatar_clip = avatar_clip.with_mask(entrance_mask)

    # No border, no ring, no glow - just entrance animation
    return avatar_clip, None, 0
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
    f_val = gf(44, bold=True)
    
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
                logo_h = 20
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
            logo_h = int(175 * scale)
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
        f = gf(200, bold=True)
    except:
        f = ImageFont.load_default()
        
    # Draw centered emoji
    draw.text((200, 200), emoji, font=f, fill=(255,255,255,255), anchor="mm")
    return img


# ─── DYNAMIC LAYOUT VISUAL ELEMENTS ───────────────────────────────────────────
def _create_layout_visual_clip(layout_type, visual_type, chunk, accent_color, audio_duration):
    """
    Creates layout-aware visual elements for a chunk based on its dynamic layout.
    Returns a VideoClip that adds layout-specific visual treatment.
    """
    start_time = chunk.get("start", 0)
    duration = chunk.get("duration", 2.0)
    
    if start_time >= audio_duration or duration <= 0:
        return None
    
    # Clamp duration
    end_time = min(start_time + duration, audio_duration)
    clip_duration = end_time - start_time
    
    if clip_duration < 0.5:
        return None
    
    # Layout-specific visual treatments
    if layout_type == "split_screen":
        # Split screen: Add vertical divider accent
        return _create_split_screen_accent(accent_color, clip_duration).with_start(start_time)
    
    elif layout_type == "hero_center":
        # Hero center: Add center glow/pulse
        return _create_hero_center_glow(accent_color, clip_duration).with_start(start_time)
    
    elif layout_type == "side_strip":
        # Side strip: Add vertical strip indicator
        return _create_side_strip_indicator(accent_color, clip_duration).with_start(start_time)
    
    elif layout_type == "top_center":
        # Top center: Add top banner accent
        return _create_top_center_banner(accent_color, clip_duration).with_start(start_time)
    
    elif layout_type == "asymmetric":
        # Asymmetric: Add corner accent
        return _create_asymmetric_corner(accent_color, clip_duration).with_start(start_time)
    
    return None


def _create_split_screen_accent(accent_color, duration):
    """Vertical divider line for split_screen layout."""
    from moviepy import VideoClip
    import numpy as np
    
    line_w = 4
    line_h = FRAME_H
    line_x = FRAME_W // 2
    
    fps = 15
    total_frames = int(duration * fps)
    
    def make_frame(t):
        img = np.zeros((FRAME_H, FRAME_W, 4), dtype=np.uint8)
        # Animated vertical line
        pulse = 1.0 + 0.3 * np.sin(t * 4)
        alpha = int(180 * pulse)
        color = (*accent_color, alpha)
        img[:, line_x - line_w//2:line_x + line_w//2] = color
        return img
    
    def make_mask(t):
        img = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        img[:, line_x - line_w//2:line_x + line_w//2] = 1.0
        return img
    
    clip = VideoClip(make_frame, duration=duration)
    mclip = VideoClip(make_mask, is_mask=True, duration=duration)
    return clip.with_mask(mclip)


def _create_hero_center_glow(accent_color, duration):
    """Center glow pulse for hero_center layout."""
    from moviepy import VideoClip
    import numpy as np
    
    center_x = FRAME_W // 2
    center_y = FRAME_H // 2
    max_radius = min(FRAME_W, FRAME_H) // 2
    
    def make_frame(t):
        img = np.zeros((FRAME_H, FRAME_W, 4), dtype=np.uint8)
        # Pulsing radial gradient
        pulse = 0.5 + 0.5 * np.sin(t * 3)
        radius = int(max_radius * (0.3 + 0.4 * pulse))
        
        y, x = np.ogrid[:FRAME_H, :FRAME_W]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = dist < radius
        alpha = (1.0 - dist[mask] / radius) * 60
        img[mask] = (*accent_color, alpha.astype(np.uint8))
        return img
    
    def make_mask(t):
        img = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        pulse = 0.5 + 0.5 * np.sin(t * 3)
        radius = int(max_radius * (0.3 + 0.4 * pulse))
        y, x = np.ogrid[:FRAME_H, :FRAME_W]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        img[dist < radius] = (1.0 - dist[dist < radius] / radius) * 0.3
        return img
    
    clip = VideoClip(make_frame, duration=duration)
    mclip = VideoClip(make_mask, is_mask=True, duration=duration)
    return clip.with_mask(mclip)


def _create_side_strip_indicator(accent_color, duration):
    """Vertical strip on left side for side_strip layout."""
    from moviepy import VideoClip
    import numpy as np
    
    strip_w = 80
    
    def make_frame(t):
        img = np.zeros((FRAME_H, FRAME_W, 4), dtype=np.uint8)
        # Animated vertical strip
        pulse = 1.0 + 0.2 * np.sin(t * 5)
        alpha = int(150 * pulse)
        color = (*accent_color, alpha)
        img[:, :strip_w] = color
        
        # Add subtle pattern
        for y in range(0, FRAME_H, 40):
            accent_y = y + int(20 * np.sin(t * 4 + y * 0.1))
            if 0 <= accent_y < FRAME_H:
                img[max(0, accent_y-5):accent_y+5, :strip_w] = (*accent_color, min(255, alpha + 50))
        return img
    
    def make_mask(t):
        img = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        img[:, :strip_w] = 0.4
        return img
    
    clip = VideoClip(make_frame, duration=duration)
    mclip = VideoClip(make_mask, is_mask=True, duration=duration)
    return clip.with_mask(mclip)


def _create_top_center_banner(accent_color, duration):
    """Top banner accent for top_center layout."""
    from moviepy import VideoClip
    import numpy as np
    
    banner_h = 120
    
    def make_frame(t):
        img = np.zeros((FRAME_H, FRAME_W, 4), dtype=np.uint8)
        # Top banner with animated gradient
        for y in range(banner_h):
            grad = y / banner_h
            alpha = int(180 * (1.0 - grad))
            pulse = 1.0 + 0.3 * np.sin(t * 3 + grad * 10)
            alpha = int(alpha * pulse)
            img[y, :] = (*accent_color, alpha)
        return img
    
    def make_mask(t):
        img = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        for y in range(banner_h):
            grad = y / banner_h
            img[y, :] = 0.5 * (1.0 - grad)
        return img
    
    clip = VideoClip(make_frame, duration=duration)
    mclip = VideoClip(make_mask, is_mask=True, duration=duration)
    return clip.with_mask(mclip)


def _create_asymmetric_corner(accent_color, duration):
    """Corner accent for asymmetric layout."""
    from moviepy import VideoClip
    import numpy as np
    
    corner_size = 100
    
    def make_frame(t):
        img = np.zeros((FRAME_H, FRAME_W, 4), dtype=np.uint8)
        # Four corners with pulsing accents
        corners = [
            (0, 0),                           # Top-left
            (FRAME_W - corner_size, 0),       # Top-right
            (0, FRAME_H - corner_size),       # Bottom-left
            (FRAME_W - corner_size, FRAME_H - corner_size)  # Bottom-right
        ]
        
        for i, (cx, cy) in enumerate(corners):
            pulse = 1.0 + 0.4 * np.sin(t * 4 + i * 1.5)
            alpha = int(120 * pulse)
            
            for y in range(corner_size):
                for x in range(corner_size):
                    if x + y < corner_size:  # Triangle corner
                        grad = (x + y) / corner_size
                        a = int(alpha * (1.0 - grad))
                        if cy + y < FRAME_H and cx + x < FRAME_W:
                            img[cy + y, cx + x] = (*accent_color, a)
        return img
    
    def make_mask(t):
        img = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        for cx, cy in [(0, 0), (FRAME_W - corner_size, 0), (0, FRAME_H - corner_size), (FRAME_W - corner_size, FRAME_H - corner_size)]:
            for y in range(corner_size):
                for x in range(corner_size):
                    if x + y < corner_size:
                        grad = (x + y) / corner_size
                        if cy + y < FRAME_H and cx + x < FRAME_W:
                            img[cy + y, cx + x] = 0.3 * (1.0 - grad)
        return img
    
    clip = VideoClip(make_frame, duration=duration)
    mclip = VideoClip(make_mask, is_mask=True, duration=duration)
    return clip.with_mask(mclip)


# ─── ENTITY LOGO PAIRING NEAR AVATAR ─────────────────────────────────────────
def _create_entity_logo_pip_clips(script_json, avatar_pip_func, audio_duration, cur_w, cur_h, accent_color):
    """
    Creates logo clips that appear near the avatar when entities are mentioned.
    Returns list of (clip, start_time, end_time) for compositing.
    """
    import os
    from moviepy import VideoClip, ImageClip
    
    # Collect all entities with logos
    entities = []
    for key in ["companies_mentioned", "tools_mentioned", "key_entities", "people"]:
        for ent in script_json.get(key, []):
            name = ent.get("name") if isinstance(ent, dict) else ent
            logo_path = ent.get("local_logo_path") or ent.get("local_hq_path") or ent.get("local_image_path")
            if name and logo_path and os.path.exists(logo_path):
                entities.append({"name": name, "logo_path": logo_path, "type": key})
    
    if not entities:
        return []
    
    # Get word timestamps to find when entities are mentioned
    script_text = script_json.get("script", "").lower()
    word_timestamps = script_json.get("word_timestamps", [])
    
    logo_clips = []
    logo_size = int(min(cur_w, cur_h) * 0.6)  # 60% of avatar size
    
    for ent in entities:
        name = ent["name"].lower()
        # Find mentions in script
        idx = 0
        while True:
            idx = script_text.find(name, idx)
            if idx == -1:
                break
            
            word_idx = len(script_text[:idx].split())
            if word_idx < len(word_timestamps):
                mention_time = word_timestamps[word_idx][0] if word_timestamps[word_idx] else 0
                start_time = max(0, mention_time - 0.5)
                end_time = min(audio_duration, mention_time + 3.0)
                
                # Load and prepare logo
                try:
                    logo_img = Image.open(ent["logo_path"]).convert("RGBA")
                    # Make square
                    lw, lh = logo_img.size
                    if lw != lh:
                        size = max(lw, lh)
                        new_img = Image.new("RGBA", (size, size), (0,0,0,0))
                        new_img.paste(logo_img, ((size-lw)//2, (size-lh)//2))
                        logo_img = new_img
                    logo_img = logo_img.resize((logo_size, logo_size), Image.LANCZOS)
                    logo_arr = np.array(logo_img)
                    
                    # Create clip that follows avatar position
                    def make_logo_frame(t, start=start_time, end=end_time, avatar_func=avatar_pip_func):
                        if start <= t < end:
                            ax, ay = avatar_func(t)
                            # Position logo to the right of avatar
                            return logo_arr
                        return np.zeros((logo_size, logo_size, 4), dtype=np.uint8)
                    
                    def make_logo_mask(t, start=start_time, end=end_time):
                        if start <= t < end:
                            # Circular mask with fade
                            progress = (t - start) / (end - start)
                            fade = 1.0
                            if progress < 0.2:
                                fade = progress / 0.2
                            elif progress > 0.8:
                                fade = (1.0 - progress) / 0.2
                            mask = np.ones((logo_size, logo_size), dtype=np.float32) * fade
                            # Circular mask
                            y, x = np.ogrid[:logo_size, :logo_size]
                            center = logo_size // 2
                            dist = np.sqrt((x - center)**2 + (y - center)**2)
                            mask[dist > center] = 0
                            return mask
                        return np.zeros((logo_size, logo_size), dtype=np.float32)
                    
                    logo_clip = VideoClip(make_logo_frame, duration=audio_duration)
                    logo_clip = logo_clip.with_mask(VideoClip(make_logo_mask, is_mask=True, duration=audio_duration))
                    
                    # Position function - to the right of avatar
                    def logo_position(t, avatar_func=avatar_pip_func):
                        ax, ay = avatar_func(t)
                        return (ax + cur_w + 20, ay + (cur_h - logo_size) // 2)
                    
                    logo_clips.append((logo_clip.with_position(logo_position).with_start(0), start_time, end_time))
                    
                except Exception as e:
                    print(f"⚠️ Failed to create logo clip for {ent['name']}: {e}")
            
            idx += len(name)
    
    return logo_clips


def insert_easter_egg(frame_width=1080, frame_height=1920):
    """Creates a 1-frame high-contrast tech glitch for retention hacking."""
    img = Image.new('RGBA', (frame_width, frame_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    # Neon glitch style
    draw.rectangle([0, 0, frame_width, frame_height], fill=(0, 255, 100, 40))
    f = gf(60, bold=True)
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
                vol_multiplier = (1.5 * (1.0 - duck_factor) + 0.20 * duck_factor) * sil_factor
                
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
                
                # ── PHASE 5: FACT BOUNDARY DUCKING (Enhanced) ─────────────────────
                # Additional duck at fact transitions for clear mental separation
                if fact_timestamps:
                    for ft in fact_timestamps:
                        start_s = float(ft.get("approx_start_seconds", 0))
                        if start_s > 2.0:
                            fact_start_ms = int(start_s * 1000)
                            # 300ms before fact start
                            if fact_start_ms - 300 <= chunk_start <= fact_start_ms:
                                vol_multiplier *= 0.2  # Moderate duck for mental separation
                                break
                
                # ── PHASE 6: HOOK ZONE ENHANCEMENT ──────────────────────────────
                # First 3 seconds: Keep BGM higher energy, less ducked for impact
                if chunk_start < 3000:
                    # Reduce ducking in hook zone for more energy
                    vol_multiplier = max(vol_multiplier, 0.6)
                
                # ── PHASE 7: CTA ZONE ───────────────────────────────────────────
                # Last 5 seconds: Drop BGM significantly for CTA authority
                if chunk_start > target_duration_ms - 5000:
                    vol_multiplier *= 0.3
                
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
                vol_multiplier = (1.5 * (1.0 - duck_factor) + 0.20 * duck_factor) * sil_factor
                
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
                elif ctype == "glitch":
                    sfx = sfx - 4   # Glitch needs to cut through
                elif ctype == "bass":
                    sfx = sfx - 2   # Bass hit for impact moments
                else:
                    sfx = sfx - 8   # Default volume
                
                pos_ms = int(cue_ts * 1000)
                composite = composite.overlay(sfx, position=pos_ms)
                sfx_count += 1
            except Exception as e:
                print(f"   ⚠️ Failed to load SFX {ctype}: {e}")
                
    # ── ENHANCED: Auto-inject SFX at pattern interrupts (retention_map driven) ──
    from config import ENABLE_STRATEGIC_SFX
    if ENABLE_STRATEGIC_SFX and retention_map and not CI_LITE:
        pi_timestamps = retention_map.get("pattern_interrupts", [])
        for pi in pi_timestamps:
            pi_word = pi.get("at_word", 0)
            pi_type = pi.get("type", "contradiction")
            pi_time_s = pi_word / 3.0
            
            if pi_time_s < output_duration:
                # Map pattern interrupt type to SFX
                if pi_type in ["contradiction", "stat_bomb"]:
                    sfx_type = "glitch"  # Sharp impact
                elif pi_type in ["rhetorical_question", "direct_address"]:
                    sfx_type = "pop"     # Crisp attention grabber
                elif pi_type in ["emotional_pivot", "curiosity_gap"]:
                    sfx_type = "woosh"   # Smooth transition
                else:
                    sfx_type = "woosh"
                
                sfx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sfx", f"{sfx_type}.wav")
                if os.path.exists(sfx_path) and os.path.getsize(sfx_path) > 0:
                    try:
                        sfx = AudioSegment.from_file(sfx_path).set_frame_rate(44100).set_channels(2)
                        if sfx_type == "woosh":
                            sfx = sfx - 12
                        elif sfx_type == "pop":
                            sfx = sfx - 6
                        elif sfx_type == "glitch":
                            sfx = sfx - 4
                        else:
                            sfx = sfx - 8
                        pos_ms = int(pi_time_s * 1000)
                        composite = composite.overlay(sfx, position=pos_ms)
                        sfx_count += 1
                    except Exception as e:
                        print(f"   ⚠️ Failed to auto-inject SFX {sfx_type}: {e}")
    
    # Auto-inject transition Woosh SFX for subtitle transitions - REMOVED to prevent visual change sound distraction
    auto_sfx_count = 0
                    
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
        # Apply vignette to reduce visual clutter from dense web page screenshots
        canvas_img = _prepare_screenshot_canvas(raw_img, FRAME_W, FRAME_H, apply_vignette=True)
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
        # Apply vignette to evidence screenshots too
        canvas = _prepare_screenshot_canvas(img, target_w, target_h, evidence_path, apply_vignette=True)
        
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

# ── MID-VIDEO RE-ENGAGEMENT HOOKS ─────────────────────────────────────────
def _summary_slide_clip(summary_text, accent_color, audio_duration, start_time, duration=4.0):
    """
    Creates a summary slide overlay that appears at key transition points
    to reset viewer attention and reinforce key takeaways.
    """
    try:
        f = gf(36, bold=True)
        # Wrap text to fit width
        max_chars = 35
        words = summary_text.split()
        lines = []
        current_line = []
        current_len = 0
        for w in words:
            if current_len + len(w) + 1 > max_chars:
                lines.append(" ".join(current_line))
                current_line = [w]
                current_len = len(w)
            else:
                current_line.append(w)
                current_len += len(w) + 1
        if current_line:
            lines.append(" ".join(current_line))
        
        line_height = 50
        padding_x, padding_y = 40, 30
        text_w = max(f.getbbox(l)[2] for l in lines)
        text_h = len(lines) * line_height
        img_w = min(text_w + padding_x * 2, FRAME_W - 100)
        img_h = text_h + padding_y * 2
        
        img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Background with accent border
        draw.rounded_rectangle([0, 0, img_w - 1, img_h - 1], radius=16, 
                              fill=(10, 10, 15, 230), outline=(*accent_color, 200), width=3)
        
        # "KEY TAKEAWAY" label
        label_f = gf(20, bold=True)
        draw.text((img_w // 2, padding_y), "KEY TAKEAWAY", font=label_f, fill=(*accent_color, 255), anchor="mm")
        
        # Summary text lines
        y_start = padding_y + 35
        for i, line in enumerate(lines):
            draw.text((padding_x, y_start + i * line_height), line, font=f, fill=(255, 255, 255, 255))
        
        arr = np.array(img.convert("RGB"))
        mask = np.array(img.split()[3]).astype(float) / 255.0
        
        clip = ImageClip(arr, duration=duration)
        mclip = VideoClip(lambda t: mask, is_mask=True, duration=duration)
        
        # Position center screen, slightly above center
        return clip.with_mask(mclip).with_position(("center", int(FRAME_H * 0.35))).with_start(start_time).with_effects([vfx.CrossFadeIn(0.5), vfx.CrossFadeOut(0.5)])
    except Exception as e:
        print(f"⚠️ Summary slide error: {e}")
        return None


def _quiz_prompt_clip(question, accent_color, audio_duration, start_time, duration=3.0):
    """
    Creates an interactive quiz/poll overlay to re-engage viewers at mid-video points.
    """
    try:
        f = gf(32, bold=True)
        max_chars = 40
        words = question.split()
        lines = []
        current_line = []
        current_len = 0
        for w in words:
            if current_len + len(w) + 1 > max_chars:
                lines.append(" ".join(current_line))
                current_line = [w]
                current_len = len(w)
            else:
                current_line.append(w)
                current_len += len(w) + 1
        if current_line:
            lines.append(" ".join(current_line))
        
        line_height = 45
        padding_x, padding_y = 40, 30
        text_w = max(f.getbbox(l)[2] for l in lines)
        text_h = len(lines) * line_height
        img_w = min(text_w + padding_x * 2, FRAME_W - 100)
        img_h = text_h + padding_y * 2 + 60  # Extra space for "💬 Comment below"
        
        img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Background with pulsing accent border
        draw.rounded_rectangle([0, 0, img_w - 1, img_h - 1], radius=16,
                              fill=(15, 15, 25, 240), outline=(*accent_color, 255), width=4)
        
        # Question lines
        y_start = padding_y
        for i, line in enumerate(lines):
            draw.text((padding_x, y_start + i * line_height), line, font=f, fill=(255, 255, 255, 255))
        
        # CTA at bottom
        cta_f = gf(22, bold=True)
        draw.text((img_w // 2, img_h - padding_y - 25), "💬 Drop your answer in comments!", font=cta_f, fill=(*accent_color, 255), anchor="mm")
        
        arr = np.array(img.convert("RGB"))
        mask = np.array(img.split()[3]).astype(float) / 255.0
        
        clip = ImageClip(arr, duration=duration)
        mclip = VideoClip(lambda t: mask, is_mask=True, duration=duration)
        
        # Position lower third
        return clip.with_mask(mclip).with_position(("center", int(FRAME_H * 0.65))).with_start(start_time).with_effects([vfx.CrossFadeIn(0.3)])
    except Exception as e:
        print(f"⚠️ Quiz prompt error: {e}")
        return None


def _pattern_interrupt_marker(accent_color, audio_duration, start_time, marker_type="shift", duration=1.5):
    """
    Creates a brief visual marker at pattern interrupt points (topic transitions, etc.)
    Types: 'shift' (topic change), 'payoff' (key insight coming), 'recap' (summary incoming)
    """
    try:
        markers = {
            "shift": ("⚡ TOPIC SHIFT", "NEXT: "),
            "payoff": ("🎯 KEY INSIGHT", "PAY ATTENTION: "),
            "recap": ("📋 QUICK RECAP", "SO FAR: "),
            "loop": ("🔄 LOOPING BACK", "REMEMBER: ")
        }
        label, prefix = markers.get(marker_type, markers["shift"])
        
        f = gf(40, bold=True)
        padding_x, padding_y = 50, 25
        text_w = f.getbbox(label)[2]
        img_w = text_w + padding_x * 2
        img_h = 80
        
        img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Animated background
        draw.rounded_rectangle([0, 0, img_w - 1, img_h - 1], radius=12,
                              fill=(*accent_color, 200), outline=(255, 255, 255, 100), width=2)
        
        draw.text((img_w // 2, img_h // 2), label, font=f, fill=(255, 255, 255, 255), anchor="mm")
        
        arr = np.array(img.convert("RGB"))
        mask = np.array(img.split()[3]).astype(float) / 255.0
        
        clip = ImageClip(arr, duration=duration)
        mclip = VideoClip(lambda t: mask, is_mask=True, duration=duration)
        
        # Top center, animate in/out
        pos = ("center", 80)
        return clip.with_mask(mclip).with_position(pos).with_start(start_time).with_effects([vfx.CrossFadeIn(0.2), vfx.CrossFadeOut(0.3)])
    except Exception as e:
        print(f"⚠️ Pattern interrupt marker error: {e}")
        return None


def _quiz_countdown_overlay(accent_color, start_time, audio_duration, duration=5.0):
    """
    Creates an extended 4-5 second countdown overlay for quiz answer pause.
    Each number appears for ~1.3-1.5 seconds with visual progress bar and sound cue.
    Extended from 3s to give viewers time to read options and form an answer.
    """
    clips = []
    try:
        # Extended countdown: 4-5 seconds total
        countdown_numbers = [4, 3, 2, 1] if duration >= 4.5 else [3, 2, 1]
        num_count = len(countdown_numbers)
        seconds_per_num = duration / num_count
        
        for i, num in enumerate(countdown_numbers):
            num_start = start_time + i * seconds_per_num
            if num_start >= audio_duration:
                break
                
            pil = _render_quiz_countdown(num, accent_color, with_progress=True, progress_pct=(i / num_count))
            arr = np.array(pil.convert("RGB"))
            mask_arr = np.array(pil.split()[3]).astype(float) / 255.0
            
            clip_dur = seconds_per_num
            clip = VideoClip(lambda t, _arr=arr: _arr, duration=clip_dur)
            mclip = VideoClip(lambda t, _m=mask_arr: _m, is_mask=True, duration=clip_dur)
            
            # Pulse animation - slower for extended duration
            def make_pulse(t, _dur=clip_dur):
                progress = t / _dur
                scale = 1.0 + 0.08 * math.sin(progress * 2 * math.pi * 1.5)
                return scale
            
            clip = clip.with_mask(mclip).with_position("center").with_start(num_start)
            clips.append(clip)
        
        # Add "REVEAL!" flash at the end
        reveal_start = start_time + duration
        if reveal_start < audio_duration:
            reveal_pil = _render_quiz_countdown("✓", accent_color, with_progress=False)
            reveal_arr = np.array(reveal_pil.convert("RGB"))
            reveal_mask = np.array(reveal_pil.split()[3]).astype(float) / 255.0
            reveal_clip = VideoClip(lambda t, _arr=reveal_arr: _arr, duration=0.5)
            reveal_mclip = VideoClip(lambda t, _m=reveal_mask: _m, is_mask=True, duration=0.5)
            reveal_clip = reveal_clip.with_mask(reveal_mclip).with_position("center").with_start(reveal_start).with_effects([vfx.CrossFadeIn(0.1)])
            clips.append(reveal_clip)
        
        return clips
    except Exception as e:
        print(f"⚠️ Quiz countdown error: {e}")
        return []


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
            font = gf(110, bold=True)
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
            font = gf(38, bold=True)
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
        font = gf(160, bold=True)
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
    space_w = 22
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

# ══════════════════════════════════════════════════════════════════════════════
# KINETIC CAPTIONS (Hormozi Style) — Word-by-word active highlighting
# ══════════════════════════════════════════════════════════════════════════════

def _render_kinetic_caption(word_data, frame_width, frame_height, accent_color, y_shift=0):
    """
    Renders Hormozi-style kinetic captions:
    - Single line locked in safe zone (lower third for landscape, upper-middle for portrait)
    - Active word: enlarged, bold, accent color, pop animation
    - Spoken words: dimmed white
    - Future words: normal white
    - Black stroke for contrast on any background
    """
    img = Image.new('RGBA', (frame_width, frame_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    scale_ratio = frame_width / 1080.0 if frame_width < frame_height else frame_width / 1920.0
    is_landscape = frame_width > frame_height
    base_size = int(72 * scale_ratio) if is_landscape else int(58 * scale_ratio)

    f_main = gf(base_size, bold=True)
    f_active = gf(int(base_size * 1.18), bold=True)  # 18% larger for active word

    words = [wd["word"] for wd in word_data]
    if not words:
        return img

    # Calculate word widths
    fake_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    word_widths_main = [fake_draw.textbbox((0, 0), w, font=f_main)[2] - fake_draw.textbbox((0, 0), w, font=f_main)[0] for w in words]
    word_widths_active = [fake_draw.textbbox((0, 0), w, font=f_active)[2] - fake_draw.textbbox((0, 0), w, font=f_active)[0] for w in words]

    max_sub_width = int(frame_width * 0.75) if is_landscape else int(frame_width * 0.85)
    
    # Wrap to lines using main font widths
    lines = wrap_text_to_lines(words, word_widths_main, max_sub_width, f_main)
    lines = lines[:1]  # Kinetic style: single line only
    
    if not lines:
        return img
    
    target_line = lines[0]
    line_word_count = len(target_line)
    
    # Find active word index
    active_idx = -1
    for idx, wd in enumerate(word_data):
        if wd.get("is_active", False):
            active_idx = idx
            break
    if active_idx == -1:
        for idx, wd in enumerate(word_data):
            if not wd.get("is_spoken", False):
                active_idx = idx
                break
    if active_idx == -1:
        active_idx = len(word_data) - 1

    # Find which word in the target line is active
    word_counter = 0
    active_in_line = -1
    for i, wd in enumerate(word_data):
        if word_counter <= i < word_counter + line_word_count:
            if i == active_idx:
                active_in_line = i - word_counter
                break
        if i >= word_counter + line_word_count:
            break

    # Calculate total line width with active word enlarged
    line_w = 0
    for i, word in enumerate(target_line):
        if i == active_in_line:
            line_w += word_widths_active[word_counter + i]
        else:
            line_w += word_widths_main[word_counter + i]
        if i < line_word_count - 1:
            line_w += 22  # space

    # Safe Zone: center 60% vertical viewport (20%-80%)
    # For Shorts (portrait): use upper-middle zone (30%-55%) - avoids top UI and presenter area
    # For Longform (landscape): use lower-middle zone (55%-75%) - avoids bottom UI
    if is_landscape:
        # Landscape: lower third area
        y_pos_pct = 0.70
    else:
        # Portrait: upper-middle zone (center of safe 60%)
        y_pos_pct = 0.425  # Middle of 30%-55% range
    line_h = int(90 * scale_ratio)
    start_y = int(frame_height * y_pos_pct) - (line_h // 2) + y_shift

    # CLAMP: Ensure captions stay in upper-middle safe zone (30%-55%) to avoid YouTube auto-caption overlap
    # YouTube auto-captions appear in bottom 20% of screen, so keep our captions above 55%
    min_y = int(frame_height * 0.30) - (line_h // 2)  # Top of safe zone
    max_y = int(frame_height * 0.55) - (line_h // 2)  # Bottom of safe zone (above YouTube captions)
    start_y = max(min_y, min(start_y, max_y))

    # Background block (obsidian with rounded corners)
    bg_pad_x, bg_pad_y = 35, 20
    block_x1 = (frame_width - line_w) // 2 - bg_pad_x
    block_x2 = (frame_width + line_w) // 2 + bg_pad_x
    block_y1 = start_y - bg_pad_y
    block_y2 = start_y + line_h - (line_h - base_size) + bg_pad_y

    draw.rounded_rectangle(
        [block_x1, block_y1, block_x2, block_y2],
        radius=16,
        fill=(0, 0, 0, 220)
    )

    # Accent top edge line
    draw.rectangle([block_x1, block_y1, block_x2, block_y1 + 3], fill=(*accent_color, 255))

    # Render words
    cur_x = (frame_width - line_w) // 2
    for offset, word_text in enumerate(target_line):
        global_idx = word_counter + offset
        wd = word_data[global_idx]
        is_active = (offset == active_in_line)
        is_spoken = wd.get("is_spoken", False)

        if is_active:
            # Active word: accent color, enlarged, pop effect
            c_fill = (*accent_color, 255)
            f_word = f_active
            w_w = word_widths_active[global_idx]
            
            # Create word image for pop animation
            w_h = fake_draw.textbbox((0, 0), word_text, font=f_word)[3] - fake_draw.textbbox((0, 0), word_text, font=f_word)[1]
            word_img = Image.new("RGBA", (w_w + 80, w_h + 80), (0, 0, 0, 0))
            word_draw = ImageDraw.Draw(word_img)
            
            # Thick black stroke (6px) for maximum contrast
            stroke = 6
            for dx in range(-stroke, stroke + 1):
                for dy in range(-stroke, stroke + 1):
                    if dx * dx + dy * dy <= stroke * stroke:
                        word_draw.text((40 + dx, 40 + dy), word_text, font=f_word, fill=(0, 0, 0, 255))
            
            # Accent glow behind
            word_draw.text((42, 42), word_text, font=f_word, fill=(*accent_color, 100))
            # Main text
            word_draw.text((40, 40), word_text, font=f_word, fill=c_fill)
            
            # Subtle rotation for energy (±2 degrees based on word index)
            angle = 2.0 * math.sin(active_idx * 0.5) if active_idx >= 0 else 0
            rotated = word_img.rotate(angle, resample=Image.BICUBIC, expand=True)
            
            target_x = int(cur_x - (rotated.width - w_w) // 2)
            target_y = int(start_y - (rotated.height - base_size) // 2 + 3)
            img.alpha_composite(rotated, (target_x, target_y))
        else:
            # Inactive words: dimmed if spoken, bright white if future
            opacity = 140 if is_spoken else 255
            c_fill = (255, 255, 255, opacity)
            f_word = f_main
            w_w = word_widths_main[global_idx]
            w_h = fake_draw.textbbox((0, 0), word_text, font=f_word)[3] - fake_draw.textbbox((0, 0), word_text, font=f_word)[1]
            
            # Black stroke
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx * dx + dy * dy <= 16:
                        draw.text((cur_x + dx, start_y + 3 + dy), word_text, font=f_word, fill=(0, 0, 0, opacity))
            draw.text((cur_x, start_y + 3), word_text, font=f_word, fill=c_fill)

        # Advance cursor
        if offset == active_in_line:
            cur_x += word_widths_active[global_idx] + 22
        else:
            cur_x += word_widths_main[global_idx] + 22

    return img


_WRAP_CACHE = {}

# ══════════════════════════════════════════════════════════════════════════════
# STAT CALLOUT GRAPHICS — Metrics display (%, latency, throughput, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_stat_callout(stat_text, stat_label, accent_color, frame_width, frame_height):
    """
    Renders a metric callout card for displaying statistics like:
    - "99.9%" latency
    - "45ms" p99 latency  
    - "10K req/s" throughput
    - "3.2x" speedup
    """
    img = Image.new('RGBA', (frame_width, frame_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    is_landscape = frame_width > frame_height
    scale = frame_width / 1920.0 if is_landscape else frame_width / 1080.0
    
    # Stat value - large and prominent
    f_stat = gf(int(96 * scale), bold=True)
    f_label = gf(int(32 * scale), bold=True)
    
    stat_w, stat_h = draw.textbbox((0, 0), stat_text, font=f_stat)[2:4]
    label_w, label_h = draw.textbbox((0, 0), stat_label, font=f_label)[2:4]
    
    pad_x, pad_y = int(60 * scale), int(30 * scale)
    card_w = max(stat_w, label_w) + pad_x * 2
    card_h = stat_h + label_h + pad_y * 2 + int(20 * scale)
    
    # Center horizontally, position in upper-middle safe zone
    x1 = (frame_width - card_w) // 2
    y1 = int(frame_height * (0.25 if is_landscape else 0.30)) - card_h // 2
    x2 = x1 + card_w
    y2 = y1 + card_h
    
    # Glassmorphic background with accent border
    draw.rounded_rectangle([x1, y1, x2, y2], radius=int(20 * scale), 
                           fill=(10, 10, 18, 230), outline=(*accent_color, 200), width=3)
    # Accent top bar
    draw.rectangle([x1, y1, x2, y1 + int(4 * scale)], fill=(*accent_color, 255))
    
    # Stat value - accent color
    stat_x = (frame_width - stat_w) // 2
    stat_y = y1 + pad_y
    # Glow effect
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        draw.text((stat_x + dx, stat_y + dy), stat_text, font=f_stat, fill=(*accent_color, 100))
    draw.text((stat_x, stat_y), stat_text, font=f_stat, fill=(255, 255, 255, 255))
    
    # Label - dimmed white
    label_x = (frame_width - label_w) // 2
    label_y = stat_y + stat_h + int(15 * scale)
    draw.text((label_x, label_y), stat_label, font=f_label, fill=(200, 200, 210, 220))
    
    return img


def _stat_callout_clip(stat_text, stat_label, start_time, accent_color, audio_duration, hold=2.5):
    """Creates an animated stat callout clip with pop-in animation."""
    if start_time >= audio_duration:
        return None
    dur = min(hold + 0.4, audio_duration - start_time)
    if dur < 0.5:
        return None
    
    img = _render_stat_callout(stat_text, stat_label, accent_color, FRAME_W, FRAME_H)
    arr = np.array(img.convert("RGB"))
    mask_arr = np.array(img.split()[3]).astype(float) / 255.0
    
    def opacity_fn(t):
        if t < 0.2:
            return t / 0.2
        elif t > dur - 0.3:
            return max(0, (dur - t) / 0.3)
        return 1.0
    
    def scale_fn(t):
        if t < 0.15:
            progress = t / 0.15
            return 0.5 + 0.5 * progress - 0.1 * math.sin(progress * math.pi)
        elif t < 0.3:
            settle = (t - 0.15) / 0.15
            return 1.0 + 0.05 * math.exp(-settle * 5) * math.sin(settle * 10)
        return 1.0
    
    def make_frame(t):
        s = scale_fn(t)
        if abs(s - 1.0) < 0.01:
            return arr
        iw, ih = img.size
        sw, sh = max(1, int(iw * s)), max(1, int(ih * s))
        scaled = Image.fromarray(arr).resize((sw, sh), Image.LANCZOS)
        cx, cy = (sw - iw) // 2, (sh - ih) // 2
        cropped = np.array(scaled)[cy:cy + ih, cx:cx + iw]
        if cropped.shape[0] < ih or cropped.shape[1] < iw:
            result = np.zeros((ih, iw, 3), dtype=np.uint8)
            result[:cropped.shape[0], :cropped.shape[1]] = cropped
            return result
        return cropped
    
    def make_mask(t):
        s = scale_fn(t)
        o = opacity_fn(t)
        if abs(s - 1.0) < 0.01:
            return mask_arr * o
        iw, ih = img.size
        sw, sh = max(1, int(iw * s)), max(1, int(ih * s))
        scaled = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize((sw, sh), Image.LANCZOS)
        cx, cy = max(0, (sw - iw) // 2), max(0, (sh - ih) // 2)
        cropped = np.array(scaled)[cy:cy + ih, cx:cx + iw].astype(float) / 255.0
        if cropped.shape[0] < ih or cropped.shape[1] < iw:
            result = np.zeros((ih, iw), dtype=float)
            result[:cropped.shape[0], :cropped.shape[1]] = cropped
            return result * o
        return cropped * o
    
    clip = VideoClip(make_frame, duration=dur)
    mclip = VideoClip(make_mask, is_mask=True, duration=dur)
    return clip.with_mask(mclip).with_position("center").with_start(start_time).with_effects([vfx.CrossFadeOut(0.2)])


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER TRANSITIONS / TITLE CARDS — Topic shift markers
# ═══════════════════════════════════════════════════════════════════════════════

def _chapter_transition_card(chapter_title, chapter_num, total_chapters, accent_color, start_time, audio_duration, hold=3.0):
    """Creates a cinematic chapter transition title card."""
    if start_time >= audio_duration:
        return None
    dur = min(hold, audio_duration - start_time)
    if dur < 1.0:
        return None
    
    is_landscape = FRAME_W > FRAME_H
    scale = FRAME_W / 1920.0 if is_landscape else FRAME_W / 1080.0
    
    img = Image.new('RGBA', (FRAME_W, FRAME_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Full-screen dark overlay
    draw.rectangle([0, 0, FRAME_W, FRAME_H], fill=(0, 0, 0, 240))
    
    # Chapter number indicator
    f_num = gf(int(48 * scale), bold=True)
    num_text = f"CHAPTER {chapter_num} / {total_chapters}"
    num_w, num_h = draw.textbbox((0, 0), num_text, font=f_num)[2:4]
    draw.text(((FRAME_W - num_w) // 2, int(FRAME_H * 0.25)), num_text, font=f_num, fill=(*accent_color, 255))
    
    # Divider line
    div_y = int(FRAME_H * 0.25) + num_h + int(30 * scale)
    draw.line([(FRAME_W // 4, div_y), (3 * FRAME_W // 4, div_y)], fill=(*accent_color, 200), width=3)
    
    # Chapter title - large, centered
    f_title = gf(int(72 * scale), bold=True)
    # Wrap title
    max_w = FRAME_W - int(120 * scale)
    words = chapter_title.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        tw = draw.textbbox((0, 0), test, font=f_title)[2]
        if tw > max_w and cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    lines = lines[:2]
    
    lh = draw.textbbox((0, 0), "Ag", font=f_title)[3]
    lsp = int(lh * 1.3)
    start_y = div_y + int(60 * scale)
    
    for i, line in enumerate(lines):
        lw = draw.textbbox((0, 0), line, font=f_title)[2]
        lx = (FRAME_W - lw) // 2
        ly = start_y + i * lsp
        # Shadow
        for dx, dy in [(-3, -3), (3, -3), (-3, 3), (3, 3)]:
            draw.text((lx + dx, ly + dy), line, font=f_title, fill=(0, 0, 0, 255))
        draw.text((lx, ly), line, font=f_title, fill=(255, 255, 255, 255))
    
    arr = np.array(img.convert("RGB"))
    mask_arr = np.array(img.split()[3]).astype(float) / 255.0
    
    def opacity_fn(t):
        if t < 0.3:
            return t / 0.3
        elif t > dur - 0.4:
            return max(0, (dur - t) / 0.4)
        return 1.0
    
    def scale_fn(t):
        if t < 0.2:
            progress = t / 0.2
            return 0.95 + 0.05 * progress
        return 1.0
    
    def make_frame(t):
        s = scale_fn(t)
        if abs(s - 1.0) < 0.01:
            return arr
        iw, ih = img.size
        sw, sh = max(1, int(iw * s)), max(1, int(ih * s))
        scaled = Image.fromarray(arr).resize((sw, sh), Image.LANCZOS)
        cx, cy = (sw - iw) // 2, (sh - ih) // 2
        cropped = np.array(scaled)[cy:cy + ih, cx:cx + iw]
        if cropped.shape[0] < ih or cropped.shape[1] < iw:
            result = np.zeros((ih, iw, 3), dtype=np.uint8)
            result[:cropped.shape[0], :cropped.shape[1]] = cropped
            return result
        return cropped
    
    def make_mask(t):
        s = scale_fn(t)
        o = opacity_fn(t)
        if abs(s - 1.0) < 0.01:
            return mask_arr * o
        iw, ih = img.size
        sw, sh = max(1, int(iw * s)), max(1, int(ih * s))
        scaled = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize((sw, sh), Image.LANCZOS)
        cx, cy = max(0, (sw - iw) // 2), max(0, (sh - ih) // 2)
        cropped = np.array(scaled)[cy:cy + ih, cx:cx + iw].astype(float) / 255.0
        if cropped.shape[0] < ih or cropped.shape[1] < iw:
            result = np.zeros((ih, iw), dtype=float)
            result[:cropped.shape[0], :cropped.shape[1]] = cropped
            return result * o
        return cropped * o
    
    clip = VideoClip(make_frame, duration=dur)
    mclip = VideoClip(make_mask, is_mask=True, duration=dur)
    return clip.with_mask(mclip).with_position("center").with_start(start_time).with_effects([vfx.CrossFadeIn(0.4), vfx.CrossFadeOut(0.4)])


# ═══════════════════════════════════════════════════════════════════════════════
# CODE SNIPPET / DIAGRAM DISPLAY — Technical content visualization
# ═══════════════════════════════════════════════════════════════════════════════

def _render_code_snippet(code_text, language, accent_color, frame_width, frame_height):
    """Renders a code snippet with syntax highlighting simulation."""
    img = Image.new('RGBA', (frame_width, frame_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    is_landscape = frame_width > frame_height
    scale = frame_width / 1920.0 if is_landscape else frame_width / 1080.0
    
    # Monospace font for code
    f_code = gf(int(28 * scale))
    
    # Split into lines
    lines = code_text.split('\n')
    lines = lines[:20]  # Max 20 lines
    
    # Calculate dimensions
    line_heights = [draw.textbbox((0, 0), line, font=f_code)[3] for line in lines]
    lh = max(line_heights) if line_heights else draw.textbbox((0, 0), "Ag", font=f_code)[3]
    lsp = int(lh * 1.4)
    code_h = len(lines) * lsp + int(40 * scale)
    code_w = max([draw.textbbox((0, 0), line, font=f_code)[2] for line in lines] + [0])
    code_w = min(code_w + int(80 * scale), frame_width - int(80 * scale))
    
    pad_x, pad_y = int(40 * scale), int(30 * scale)
    card_w = code_w + pad_x * 2
    card_h = code_h + pad_y * 2
    
    x1 = (frame_width - card_w) // 2
    y1 = int(frame_height * (0.35 if is_landscape else 0.30)) - card_h // 2
    x2 = x1 + card_w
    y2 = y1 + card_h
    
    # Dark editor background
    draw.rounded_rectangle([x1, y1, x2, y2], radius=int(12 * scale), fill=(20, 20, 30, 240))
    
    # Top bar (editor chrome)
    draw.rounded_rectangle([x1, y1, x2, y1 + int(40 * scale)], radius=int(12 * scale), fill=(35, 35, 45, 255))
    # Window dots
    dot_colors = [(255, 95, 87), (254, 188, 46), (40, 200, 64)]
    for i, dc in enumerate(dot_colors):
        draw.ellipse([x1 + int(20 * scale) + i * int(25 * scale), y1 + int(12 * scale),
                      x1 + int(32 * scale) + i * int(25 * scale), y1 + int(24 * scale)], fill=(*dc, 255))
    # Language label
    draw.text((x1 + int(100 * scale), y1 + int(8 * scale)), language.upper(), font=gf(int(14 * scale)), fill=(150, 150, 160, 255))
    
    # Code lines with line numbers
    line_num_color = (100, 100, 120, 255)
    for i, line in enumerate(lines):
        ly = y1 + int(50 * scale) + i * lsp
        # Line number
        draw.text((x1 + int(20 * scale), ly), str(i + 1), font=f_code, fill=line_num_color)
        # Code text - simple highlighting simulation
        draw.text((x1 + int(70 * scale), ly), line, font=f_code, fill=(220, 220, 230, 255))
    
    return img


def _code_snippet_clip(code_text, language, start_time, accent_color, audio_duration, hold=4.0):
    """Creates an animated code snippet display clip."""
    if start_time >= audio_duration:
        return None
    dur = min(hold, audio_duration - start_time)
    if dur < 1.0:
        return None
    
    img = _render_code_snippet(code_text, language, accent_color, FRAME_W, FRAME_H)
    arr = np.array(img.convert("RGB"))
    mask_arr = np.array(img.split()[3]).astype(float) / 255.0
    
    def opacity_fn(t):
        if t < 0.3:
            return t / 0.3
        elif t > dur - 0.4:
            return max(0, (dur - t) / 0.4)
        return 1.0
    
    def make_frame(t):
        return arr
    
    def make_mask(t):
        return mask_arr * opacity_fn(t)
    
    clip = VideoClip(make_frame, duration=dur)
    mclip = VideoClip(make_mask, is_mask=True, duration=dur)
    return clip.with_mask(mclip).with_position("center").with_start(start_time).with_effects([vfx.CrossFadeIn(0.5), vfx.CrossFadeOut(0.5)])


def _render_architecture_diagram(components, connections, accent_color, frame_width, frame_height):
    """Renders a simple architecture diagram with boxes and arrows."""
    img = Image.new('RGBA', (frame_width, frame_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    is_landscape = frame_width > frame_height
    scale = frame_width / 1920.0 if is_landscape else frame_width / 1080.0
    
    f_box = gf(int(32 * scale), bold=True)
    f_label = gf(int(20 * scale))
    
    # Layout components in a horizontal flow
    n = len(components)
    if n == 0:
        return img
    
    box_w = int(200 * scale)
    box_h = int(80 * scale)
    gap = int(60 * scale)
    total_w = n * box_w + (n - 1) * gap
    start_x = (frame_width - total_w) // 2
    start_y = int(frame_height * 0.40)
    
    # Draw connections (arrows between boxes)
    for i in range(n - 1):
        x1 = start_x + i * (box_w + gap) + box_w
        y1 = start_y + box_h // 2
        x2 = start_x + (i + 1) * (box_w + gap)
        y2 = start_y + box_h // 2
        # Arrow line
        draw.line([(x1, y1), (x2, y2)], fill=(*accent_color, 180), width=3)
        # Arrow head
        head_size = int(12 * scale)
        draw.polygon([(x2 - head_size, y2 - head_size), (x2, y2), (x2 - head_size, y2 + head_size)], fill=(*accent_color, 180))
    
    # Draw component boxes
    for i, comp in enumerate(components):
        x = start_x + i * (box_w + gap)
        y = start_y
        
        # Box type determines style
        comp_type = comp.get("type", "service")
        if comp_type == "database":
            # Cylinder shape
            draw.ellipse([x, y, x + box_w, y + int(20 * scale)], fill=(30, 144, 255, 200), outline=(*accent_color, 200), width=2)
            draw.rectangle([x, y + int(10 * scale), x + box_w, y + box_h - int(10 * scale)], fill=(30, 144, 255, 200), outline=(*accent_color, 200), width=2)
            draw.ellipse([x, y + box_h - int(20 * scale), x + box_w, y + box_h], fill=(30, 144, 255, 200), outline=(*accent_color, 200), width=2)
        elif comp_type == "queue":
            # Rounded rectangle with vertical lines
            draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=int(10 * scale), fill=(255, 165, 0, 200), outline=(*accent_color, 200), width=2)
            for j in range(3):
                lx = x + box_w // 4 + j * box_w // 4
                draw.line([(lx, y + int(10 * scale)), (lx, y + box_h - int(10 * scale))], fill=(255, 255, 255, 100), width=1)
        else:
            # Standard service box
            draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=int(12 * scale), fill=(40, 40, 60, 230), outline=(*accent_color, 200), width=2)
        
        # Component name
        name = comp.get("name", f"Service {i+1}")
        tw = draw.textbbox((0, 0), name, font=f_box)[2]
        draw.text((x + (box_w - tw) // 2, y + box_h // 2 - int(10 * scale)), name, font=f_box, fill=(255, 255, 255, 255))
        
        # Tech label
        tech = comp.get("tech", "")
        if tech:
            lw = draw.textbbox((0, 0), tech, font=f_label)[2]
            draw.text((x + (box_w - lw) // 2, y + box_h // 2 + int(20 * scale)), tech, font=f_label, fill=(180, 180, 200, 255))
    
    return img


def _architecture_diagram_clip(components, connections, start_time, accent_color, audio_duration, hold=5.0):
    """Creates an animated architecture diagram clip."""
    if start_time >= audio_duration:
        return None
    dur = min(hold, audio_duration - start_time)
    if dur < 1.0:
        return None
    
    img = _render_architecture_diagram(components, connections, accent_color, FRAME_W, FRAME_H)
    arr = np.array(img.convert("RGB"))
    mask_arr = np.array(img.split()[3]).astype(float) / 255.0
    
    def opacity_fn(t):
        if t < 0.4:
            return t / 0.4
        elif t > dur - 0.5:
            return max(0, (dur - t) / 0.5)
        return 1.0
    
    def make_frame(t):
        return arr
    
    def make_mask(t):
        return mask_arr * opacity_fn(t)
    
    clip = VideoClip(make_frame, duration=dur)
    mclip = VideoClip(make_mask, is_mask=True, duration=dur)
    return clip.with_mask(mclip).with_position("center").with_start(start_time).with_effects([vfx.CrossFadeIn(0.5), vfx.CrossFadeOut(0.5)])


def render_subtitle_frame(word_data, bg_frame=None, accent_color=(255,214,0), frame_width=1080, frame_height=1920, y_shift=0):
    """Viral 'High Energy' captions: Large tilted words with pop sounds.
    
    Revised: Locked static line rendering (kinetic text), safe-zone positioning,
    dimmed past words, and precomputed wrap caching for high rendering performance.
    """
    # Use kinetic captions if enabled
    enable_kinetic = os.environ.get("ENABLE_KINETIC_CAPTIONS", "1") == "1"
    if enable_kinetic:
        return _render_kinetic_caption(word_data, frame_width, frame_height, accent_color, y_shift)

    img = Image.new('RGBA', (frame_width, frame_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)

    scale_ratio = frame_width / 1080.0 if frame_width < frame_height else frame_width / 1920.0
    is_landscape = frame_width > frame_height
    if is_landscape:
        base_size = int(72 * scale_ratio)
    else:
        base_size = int(58 * scale_ratio)

    f_main = gf(base_size, bold=True)

    active_word_data = [wd for wd in word_data if not wd.get("is_spoken", False)]
    if not active_word_data:
        return img

    words = [wd["word"] for wd in active_word_data]
    word_widths = []
    fake_draw = ImageDraw.Draw(Image.new("RGBA", (1,1)))
    for i, wd in enumerate(active_word_data):
        word_widths.append(fake_draw.textbbox((0,0), words[i], font=f_main)[2] - fake_draw.textbbox((0,0), words[i], font=f_main)[0])

    if is_landscape:
        max_sub_width = int(frame_width * 0.65)
    else:
        max_sub_width = int(frame_width * 0.80)
    lines = wrap_text_to_lines(words, word_widths, max_sub_width, f_main)
    lines = lines[:1]
    line_h = int(90 * scale_ratio)
    
    # Safe Zone: center 60% vertical viewport
    if is_landscape:
        y_pos_pct = 0.70
    else:
        y_pos_pct = 0.425  # Upper-middle zone
    start_y = int(frame_height * y_pos_pct) - (len(lines) * line_h // 2) + y_shift

    # CLAMP: Ensure captions stay in upper-middle safe zone (30%-55%) to avoid YouTube auto-caption overlap
    min_y = int(frame_height * 0.30) - (len(lines) * line_h // 2)
    max_y = int(frame_height * 0.55) - (len(lines) * line_h // 2)
    start_y = max(min_y, min(start_y, max_y))
    
    max_line_w = 0
    temp_idx = 0
    for line in lines:
        line_w = sum(word_widths[temp_idx:temp_idx+len(line)]) + 22 * (len(line)-1)
        if line_w > max_line_w:
            max_line_w = line_w
        temp_idx += len(line)
    
    bg_pad_x, bg_pad_y = 30, 18
    block_x1 = (frame_width - max_line_w) // 2 - bg_pad_x
    block_x2 = (frame_width + max_line_w) // 2 + bg_pad_x
    block_y1 = start_y - bg_pad_y
    block_y2 = start_y + len(lines) * line_h - (line_h - base_size) + bg_pad_y

    draw.rounded_rectangle(
        [block_x1, block_y1, block_x2, block_y2],
        radius=12,
        fill=(0, 0, 0, 215)
    )

    word_idx = 0
    for i, line in enumerate(lines):
        line_y = start_y + i * line_h
        line_w = sum(word_widths[word_idx:word_idx+len(line)]) + 22 * (len(line)-1)
        cur_x = (frame_width - line_w) // 2

        for word_text in line:
            wd = active_word_data[word_idx]
            is_active = wd["is_active"]

            if is_active:
                c_fill = (204, 255, 0, 255) # Electric Yellow
                f_word = gf(int(base_size * 1.12), bold=True)
                w_w, w_h = ts(word_text, f_word)
                word_img = Image.new("RGBA", (w_w + 60, w_h + 60), (0,0,0,0))
                word_draw = ImageDraw.Draw(word_img)

                stroke = 5
                for dx in range(-stroke, stroke+1):
                    for dy in range(-stroke, stroke+1):
                        if dx*dx + dy*dy <= stroke*stroke:
                            word_draw.text((30+dx, 30+dy), word_text, font=f_word, fill=(0,0,0,255))

                word_draw.text((34, 34), word_text, font=f_word, fill=(0,0,0,180))
                word_draw.text((30, 30), word_text, font=f_word, fill=c_fill)

                rotated = word_img.rotate(0, resample=Image.BICUBIC, expand=True)
                orig_w = word_widths[word_idx]
                target_x = int(cur_x - (rotated.width - orig_w)//2)
                target_y = int(line_y - (rotated.height - base_size)//2 + 2)
                img.alpha_composite(rotated, (target_x, target_y))
            else:
                c_fill = (255, 255, 255, 255)
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

# These would be pre-recorded clips: pointing, surprised, thinking, nodding, etc.
AVATAR_EXPRESSIONS = {
    "pointing": ["look at this", "check this out", "see here", "notice"],
    "surprised": ["wow", "amazing", "incredible", "unbelievable", "shocking", "mind blown"],
    "thinking": ["think about", "consider", "imagine", "what if", "puzzle"],
    "nodding": ["exactly", "right", "correct", "yes", "absolutely", "precisely"],
    "shrugging": ["who knows", "maybe", "perhaps", "unclear", "uncertain"],
}

# Expression video paths (would need to be created - using main avatar as fallback)
EXPRESSION_VIDEO_MAP = {
    "pointing": "assets/video/expressions/pointing.mp4",
    "surprised": "assets/video/expressions/surprised.mp4",
    "thinking": "assets/video/expressions/thinking.mp4",
    "nodding": "assets/video/expressions/nodding.mp4",
    "shrugging": "assets/video/expressions/shrugging.mp4",
}

# Dual Avatar / Interview Mode
DUAL_AVATAR_CONFIG = {
    "enabled": False,
    "host_avatar": "assets/video/host.mp4",      # Primary presenter
    "guest_avatar": "assets/video/guest.mp4",    # Interview guest
    "host_position": "left",                     # left/right
    "switch_on_speaker": True,                   # Auto-switch based on script markers
}



def _get_avatar_expression_segments(script_text, audio_duration, word_timestamps):
    """
    Analyzes script for trigger words and returns time segments where
    specific expressions should play. Returns list of (start, end, expression_type).
    """
    segments = []
    script_lower = script_text.lower()
    
    # Find trigger words and their approximate timestamps
    for expr_type, triggers in AVATAR_EXPRESSIONS.items():
        for trigger in triggers:
            # Find all occurrences in script
            idx = 0
            while True:
                idx = script_lower.find(trigger, idx)
                if idx == -1:
                    break
                
                # Estimate timestamp from word position
                word_idx = len(script_lower[:idx].split())
                if word_idx < len(word_timestamps):
                    start_time = word_timestamps[word_idx][0] if word_timestamps[word_idx] else 0
                    end_time = min(start_time + 3.0, audio_duration)  # 3 second expression
                    segments.append((start_time, end_time, expr_type))
                
                idx += len(trigger)
    
    # Sort by start time and merge overlapping
    segments.sort(key=lambda x: x[0])
    merged = []
    for seg in segments:
        if not merged or seg[0] > merged[-1][1]:
            merged.append(list(seg))
        else:
            merged[-1][1] = max(merged[-1][1], seg[1])
            # Priority: surprised > pointing > thinking > nodding > shrugging
            priority = {"surprised": 5, "pointing": 4, "thinking": 3, "nodding": 2, "shrugging": 1}
            if priority.get(seg[2], 0) > priority.get(merged[-1][2], 0):
                merged[-1][2] = seg[2]
    
    return [(s, e, t) for s, e, t in merged]



def _create_video_internal(audio_path, script_json, chunks, output_path=None, dynamic_params=None):
    """The original heavy-lifting render logic."""
    if dynamic_params is None: dynamic_params = {}
    
    # ── CI-LITE MODE: Simplified rendering for GitHub Actions (low resources) ────────
    if CI_LITE:
        print("🔧 CI-LITE mode enabled: simplified rendering for CI environment")
    
    avatar_scale_mult = dynamic_params.get("avatar_scale_mult", 1.0)
    subtitle_y_shift = dynamic_params.get("subtitle_y_shift", 0)

    # ── YPP COMPLIANCE: Per-Video Layout Randomization ────────────────────
    headline = script_json.get("original_news_headline", script_json.get("title", "Tech News"))
    
    # Include editorial perspective in layout seed for additional variation
    editorial_perspective = script_json.get("editorial_perspective", "")
    fingerprint = script_json.get("content_fingerprint", "")
    layout_seed = f"{headline}|{editorial_perspective}|{fingerprint}"
    
    # Extract dominant color from first valid visual path
    first_visual_path = None
    for chunk in chunks:
        vp = chunk.get("visual_path")
        if vp and os.path.exists(vp):
            first_visual_path = vp
            break
            
    dominant_color = None
    if first_visual_path:
        dominant_color = get_vibrant_dominant_color(first_visual_path)
        if dominant_color:
            print(f"🎨 Dominant color extracted from first asset: {dominant_color}")
        else:
            print(f"🎨 No vibrant dominant color found in {first_visual_path}, using default accents.")

    layout = generate_layout_profile(layout_seed, dominant_color=dominant_color)
    # Merge layout jitter into subtitle shift
    subtitle_y_shift += layout["subtitle_y_jitter"]

    slot_str = script_json.get("slot", "")
    is_longform = "Slot C" in slot_str or "Slot L" in slot_str or script_json.get("is_longform", False)
    global IS_LONGFORM_ACTIVE
    IS_LONGFORM_ACTIVE = is_longform
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

    # ── SAFE ZONE CALCULATOR ──────────────────────────────────────────────────
    # Initialize early so all overlay systems can use it
    safe_zones = SafeZoneCalculator(FRAME_W, FRAME_H, layout, is_longform)
    print(f"🛡️ SafeZoneCalculator initialized: {len(safe_zones.reserved_zones)} reserved zones")

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
        for c in companies:
            c_name = c.get("name") if isinstance(c, dict) else c
            if c_name:
                key_entities.append({"name": c_name, "type": "COMPANY"})
        for t in tools:
            t_name = t.get("name") if isinstance(t, dict) else t
            if t_name:
                key_entities.append({"name": t_name, "type": "TOOL"})

    # ── FULL SCREEN BACKGROUND ───────────────────────────────────────────
    print("Preparing full-screen background...")
    visual_paths = []
    for chunk in chunks:
        vpath = chunk.get("visual_path")
        if vpath and os.path.exists(vpath) and vpath not in visual_paths:
            visual_paths.append(vpath)
        elif vpath and not os.path.exists(vpath):
            print(f"⚠️ Visual path missing on disk: {vpath}")

    bg_layer_clips = []
    particle_clips = []
    logo_clips = []
    fact_clips = []
    burst_clips = []
    reminder_clips = []
    
    if not visual_paths:
        print("⚠️ No valid visual paths found! Using solid color background.")
        bg_layer_clips.append(ColorClip(size=(FRAME_W, FRAME_H), color=(10, 10, 15), duration=audio_duration))
    else:
        print(f"✅ Using {len(visual_paths)} unique visual assets for background")
        print(f"DEBUG: is_longform={is_longform}, about to check if is_longform")
        crossfade = 0.4 if not is_longform else 0.6 # Longer crossfade for longform smoothness
        
        # --- CI-LITE: Skip complex background rendering ---
        if CI_LITE and is_longform:
            print("🔧 CI-LITE: Using simplified static background for longform")
            # Just use a single static background
            vp = visual_paths[0]
            # Define expanded_visual_paths for B-roll bursts later
            expanded_visual_paths = visual_paths[:]
            try:
                if vp.endswith(".mp4"):
                    c_clip = VideoFileClip(vp).without_audio()
                    if c_clip.duration < audio_duration:
                        c_clip = c_clip.with_effects([vfx.Loop(duration=audio_duration)])
                    else:
                        c_clip = c_clip.subclipped(0, audio_duration)
                    w, h = c_clip.size
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
                    raw_img = Image.open(vp).convert("RGB")
                    bg_img = ImageOps.fit(raw_img, (FRAME_W, FRAME_H), Image.LANCZOS)
                    bg_arr = np.array(bg_img)
                    bg_arr = cv2.GaussianBlur(bg_arr, (71, 71), 0)
                    c_clip = ImageClip(bg_arr, duration=audio_duration)
                bg_layer_clips.append(c_clip)
            except Exception as e:
                print(f"⚠️ CI-LITE background failed: {e}")
                bg_layer_clips.append(ColorClip(size=(FRAME_W, FRAME_H), color=(10, 10, 15), duration=audio_duration))
        
        # --- LONGFORM 2.5s PACING PATTERN INTERRUPTS ---
        elif is_longform:
            print("DEBUG: Entering is_longform=True branch")
            if os.environ.get("USE_LEGACY_LONGFORM_BG", "0") == "1":
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
                                        canvas_img = _prepare_screenshot_canvas(raw_img, FRAME_W, FRAME_H, apply_vignette=True)
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
                from collections import OrderedDict
                import psutil
                
                clip_dur = 2.5 + crossfade
                num_clips_needed = int(audio_duration // 2.5) + 1
                expanded_visual_paths = []
                while len(expanded_visual_paths) < num_clips_needed:
                    expanded_visual_paths.extend(visual_paths)
                expanded_visual_paths = expanded_visual_paths[:num_clips_needed]
                
                # Headroom canvas dimensions (1.05x zoom padding)
                CANVAS_W = int(FRAME_W * 1.05)
                CANVAS_H = int(FRAME_H * 1.05)
                
                def tint_numpy(arr, is_w):
                    frame_f = arr.astype(np.float32)
                    if is_w:
                        frame_f[:, :, 0] = np.clip(frame_f[:, :, 0] * 1.04 + 3, 0, 255)
                        frame_f[:, :, 1] = np.clip(frame_f[:, :, 1] * 1.01, 0, 255)
                        frame_f[:, :, 2] = np.clip(frame_f[:, :, 2] * 0.96 - 3, 0, 255)
                    else:
                        frame_f[:, :, 0] = np.clip(frame_f[:, :, 0] * 0.96 - 3, 0, 255)
                        frame_f[:, :, 1] = np.clip(frame_f[:, :, 1] * 1.01, 0, 255)
                        frame_f[:, :, 2] = np.clip(frame_f[:, :, 2] * 1.04 + 3, 0, 255)
                    return frame_f.astype(np.uint8)

                # OrderedDict LRU cache (capacity 4)
                # Cache holds post-crop, post-resized assets at CANVAS size:
                # - np.ndarray (for images)
                # - VideoFileClip (for videos)
                clip_cache = OrderedDict()
                CACHE_CAPACITY = 4
                
                def get_processed_asset(vp, is_warm):
                    cache_key = (vp, is_warm)
                    if cache_key in clip_cache:
                        clip_cache.move_to_end(cache_key)
                        return clip_cache[cache_key]
                        
                    if len(clip_cache) >= CACHE_CAPACITY:
                        evict_key, evict_val = clip_cache.popitem(last=False)
                        if isinstance(evict_val, VideoFileClip):
                            try:
                                evict_val.close()
                            except Exception as ex:
                                print(f"⚠️ Error closing evicted VideoFileClip: {ex}")
                        del evict_val
                        
                    if vp.endswith(".mp4"):
                        v_clip = VideoFileClip(vp).without_audio()
                        w, h = v_clip.size
                        target_w_crop = int(h * 16 / 9)
                        if target_w_crop <= w:
                            x1 = (w - target_w_crop) // 2
                            v_clip = v_clip.cropped(x1=x1, y1=0, x2=x1 + target_w_crop, y2=h)
                        else:
                            target_h_crop = int(w * 9 / 16)
                            y1 = (h - target_h_crop) // 2
                            v_clip = v_clip.cropped(x1=0, y1=y1, x2=w, y2=y1 + target_h_crop)
                        v_clip = v_clip.resized((CANVAS_W, CANVAS_H))
                        clip_cache[cache_key] = v_clip
                        return v_clip
                    else:
                        raw_img = Image.open(vp)
                        w, h = raw_img.size
                        target_w_crop = int(h * 16 / 9)
                        if target_w_crop <= w:
                            x1 = (w - target_w_crop) // 2
                            canvas_img = raw_img.crop((x1, 0, x1 + target_w_crop, h))
                        else:
                            target_h_crop = int(w * 9 / 16)
                            y1 = (h - target_h_crop) // 2
                            canvas_img = raw_img.crop((0, y1, w, y1 + target_h_crop))
                        canvas_img = canvas_img.resize((CANVAS_W, CANVAS_H))
                        
                        canvas_arr = np.array(canvas_img.convert("RGB"))
                        tinted_arr = tint_numpy(canvas_arr, is_warm)
                        clip_cache[cache_key] = tinted_arr
                        return tinted_arr
                
                # Pre-calculate shake offsets curves (150 frames max)
                shake_offsets = []
                for f in range(150):
                    t = f / 30.0
                    off_x = math.sin(t * 1.5) * 2 + math.cos(t * 0.8) * 1.5
                    off_y = math.cos(t * 1.2) * 2 + math.sin(t * 0.9) * 1.5
                    shake_offsets.append((int(off_x), int(off_y)))

                # Pre-calculate starts, scale factors, and transition types
                seg_starts = []
                seg_scale_factors = []
                seg_trans_types = []
                
                current_start = 0.0
                for i, vp in enumerate(expanded_visual_paths):
                    seg_starts.append(current_start)
                    seg_scale_factors.append(1.0 + random.uniform(0.18, 0.25))
                    
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
                        seg_trans_types.append(trans_type)
                        
                        if trans_type == "glitch":
                            trans_clip = _create_transition_clip("glitch", duration=0.15)
                            if trans_clip:
                                trans_clip = trans_clip.with_start(current_start)
                                logo_clips.append(trans_clip)

                        flash = ColorClip(size=(FRAME_W, FRAME_H), color=(255, 255, 255), duration=0.2).with_opacity(0.6)
                        flash = flash.with_start(current_start).with_effects([vfx.CrossFadeOut(0.15)])
                        logo_clips.append(flash)
                    else:
                        seg_trans_types.append(None)
                        
                    current_start += (clip_dur - crossfade)

                process = psutil.Process(os.getpid())
                frame_counter = [0]
                
                def make_bg_frame(t):
                    frame_counter[0] += 1
                    if frame_counter[0] % 300 == 0:
                        rss = process.memory_info().rss / (1024 * 1024)
                        print(f"🎬 [BG RENDER] Frame {frame_counter[0]} | Video time: {t:.2f}s | Memory RSS: {rss:.2f} MB")
                        
                    seg_idx = int(t // 2.5)
                    t_rel = t % 2.5
                    
                    if seg_idx >= len(expanded_visual_paths):
                        seg_idx = len(expanded_visual_paths) - 1
                        
                    vp = expanded_visual_paths[seg_idx]
                    is_warm = (seg_idx % 2 == 0)
                    
                    # 1. Fetch current frame
                    asset = get_processed_asset(vp, is_warm)
                    if isinstance(asset, np.ndarray):
                        curr_canvas = asset
                    else:
                        t_v = min(t_rel, asset.duration - 0.01)
                        raw_frame = asset.get_frame(t_v)
                        curr_canvas = tint_numpy(raw_frame, is_warm)
                        
                    # 2. Zoom & Shake with Headroom
                    sf = seg_scale_factors[seg_idx]
                    scale = 1.0 + (sf - 1.0) * (t_rel / clip_dur)
                    
                    trans_type = seg_trans_types[seg_idx]
                    if trans_type == "zoom" and t_rel < crossfade:
                        scale *= (1.3 - (0.3 * (t_rel / crossfade)))
                        
                    crop_w, crop_h = int(CANVAS_W / scale), int(CANVAS_H / scale)
                    dx, dy = (CANVAS_W - crop_w) // 2, (CANVAS_H - crop_h) // 2
                    frame = cv2.resize(curr_canvas[dy:dy+crop_h, dx:dx+crop_w], (CANVAS_W, CANVAS_H))
                    
                    # Translation via warpAffine
                    frame_idx = min(149, int(t_rel * 30.0))
                    off_x, off_y = shake_offsets[frame_idx]
                    if off_x != 0 or off_y != 0:
                        T = np.float32([[1, 0, off_x], [0, 1, off_y]])
                        frame = cv2.warpAffine(frame, T, (CANVAS_W, CANVAS_H), borderMode=cv2.BORDER_REPLICATE)
                        
                    # 3. Handle transition boundaries
                    if seg_idx > 0 and t_rel < crossfade:
                        prev_vp = expanded_visual_paths[seg_idx - 1]
                        prev_is_warm = ((seg_idx - 1) % 2 == 0)
                        prev_asset = get_processed_asset(prev_vp, prev_is_warm)
                        
                        if isinstance(prev_asset, np.ndarray):
                            prev_canvas = prev_asset
                        else:
                            t_prev_rel = 2.5 + t_rel
                            t_v = min(t_prev_rel, prev_asset.duration - 0.01)
                            raw_frame = prev_asset.get_frame(t_v)
                            prev_canvas = tint_numpy(raw_frame, prev_is_warm)
                            
                        prev_sf = seg_scale_factors[seg_idx - 1]
                        prev_scale = 1.0 + (prev_sf - 1.0) * ((2.5 + t_rel) / clip_dur)
                        
                        pcw, pch = int(CANVAS_W / prev_scale), int(CANVAS_H / prev_scale)
                        pdx, pdy = (CANVAS_W - pcw) // 2, (CANVAS_H - pch) // 2
                        prev_frame = cv2.resize(prev_canvas[pdy:pdy+pch, pdx:pdx+pcw], (CANVAS_W, CANVAS_H))
                        
                        po_x, po_y = shake_offsets[min(149, int((2.5 + t_rel) * 30.0))]
                        if po_x != 0 or po_y != 0:
                            T_p = np.float32([[1, 0, po_x], [0, 1, po_y]])
                            prev_frame = cv2.warpAffine(prev_frame, T_p, (CANVAS_W, CANVAS_H), borderMode=cv2.BORDER_REPLICATE)
                            
                        if trans_type and "slide" in trans_type:
                            prog = t_rel / crossfade
                            prog = 1 - (1 - prog)**3
                            combined = prev_frame.copy()
                            if trans_type == "slide_r":
                                x = int(CANVAS_W * (1 - prog))
                                combined[:, x:] = frame[:, :CANVAS_W-x]
                            elif trans_type == "slide_l":
                                x = int(CANVAS_W * (1 - prog))
                                combined[:, :CANVAS_W-x] = frame[:, x:]
                            elif trans_type == "slide_t":
                                y = int(CANVAS_H * (1 - prog))
                                combined[:CANVAS_H-y, :] = frame[y:, :]
                            frame = combined
                        else:
                            cf_dur = 0.1 if trans_type == "glitch" else crossfade
                            if t_rel < cf_dur:
                                alpha = t_rel / cf_dur
                                frame = cv2.addWeighted(frame, alpha, prev_frame, 1.0 - alpha, 0)
                                
                    # 4. Final crop to production dimensions
                    margin_x = (CANVAS_W - FRAME_W) // 2
                    margin_y = (CANVAS_H - FRAME_H) // 2
                    return frame[margin_y:margin_y+FRAME_H, margin_x:margin_x+FRAME_W]
                    
                bg_clip = VideoClip(make_bg_frame, duration=audio_duration)
                bg_layer_clips.append(bg_clip)
                # Reclaim any garbage from the asset processing phase
                gc.collect()
        else:
            print("DEBUG: Entering is_longform=False (SHORTS) branch")
            # --- SHORTS PACING SYNCHRONIZED TO CHUNKS ---
            print("DEBUG: Starting SHORTS chunk loop")
            clip_cache = {}
            for i, chunk in enumerate(chunks):
                print(f"DEBUG: Processing chunk {i}")
                vp = chunk.get("visual_path")
                if not vp or not os.path.exists(vp):
                    print(f"DEBUG: Chunk {i} has no valid vp, skipping")
                    continue
                
                start_t = chunk["start"]
                end_t = chunk["end"]
                
                is_last = (i == len(chunks) - 1)
                this_crossfade = crossfade if not is_last else 0.0
                clip_dur = (end_t - start_t) + this_crossfade
                
                if clip_dur <= 0.01:
                    continue
                
                try:
                    layout_variation_enabled = os.environ.get("ENABLE_LAYOUT_VARIATION", "0") == "1"
                    if layout_variation_enabled:
                        c_clip = _build_layout_bg_clip(vp, clip_dur, layout, i)
                    else:
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
                                        canvas_img = _prepare_screenshot_canvas(raw_img, FRAME_W, FRAME_H, apply_vignette=True)
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
                        
                        scale_factor = 1.0 + random.uniform(0.15, 0.22)
                        c_clip = c_clip.resized(lambda t, sf=scale_factor, cd=clip_dur: 1.0 + (sf - 1.0) * (t / cd))
                    
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
                    
                    c_clip = _apply_handheld_shake(c_clip)
                    
                    c_clip = c_clip.with_start(start_t)
                    bg_layer_clips.append(c_clip)
                except Exception as e:
                    print(f"Failed to load background img {vp} for chunk {i}: {e}")

        # ── B-ROLL BURSTS AT FACT BOUNDARIES ──────────────────────────────────────
        if is_longform and script_json.get("longform_format") in ["did_you_know", "vaibhav", "chaptered"]:
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

    # ── AVATAR EXPRESSION/GESTURE TRIGGERS ──────────────────────────────────────
# Maps trigger words to avatar expression video segments
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
        
        # Check if circular avatar is enabled based on layout archetype
        layout_variation_enabled = os.environ.get("ENABLE_LAYOUT_VARIATION", "0") == "1"
        enable_circular_avatar = (
            layout_variation_enabled 
            and not is_longform 
            and layout.get("layout_type") in ["split_screen", "hero_center"]
        )

        # Crop avatar based on video format
        if enable_circular_avatar:
            # For circular facecam: crop to square (1:1) for PiP bubble
            target_aspect = 1.0
        else:
            # For Shorts asymmetric or longform presenter: crop to portrait 9:16
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
        
        # Avatar size based on layout type
        if is_longform:
            avatar_height_pct = 0.60
        else:
            if layout_variation_enabled:
                if layout.get("layout_type") == "split_screen":
                    avatar_height_pct = 0.22
                elif layout.get("layout_type") == "hero_center":
                    avatar_height_pct = 0.25
                else: # asymmetric (full-screen presenter)
                    avatar_height_pct = 0.85
            else:
                avatar_height_pct = 0.40
                
        height_pip = int(FRAME_H * avatar_height_pct)
        width_pip = int(height_pip * (w / h))
        
        # Apply refinements from dynamic_params
        cur_w = max(1, int(width_pip * avatar_scale_mult))
        cur_h = max(1, int(height_pip * avatar_scale_mult))

        if is_longform:
            # 1. Full-screen portrait version: height 918, width 516 (aspect 9/16)
            fs_w, fs_h = 516, 918
            avatar_fs_clip = vid_clip.resized((fs_w, fs_h)).without_audio()
            
            # 2. Windowed square version: height 320, width 320 (cropped to 1.0 aspect)
            w_raw, h_raw = vid_clip.size
            crop_size = min(w_raw, h_raw)
            x_crop = (w_raw - crop_size) // 2
            y_crop = (h_raw - crop_size) // 2
            win_cropped = vid_clip.cropped(x1=x_crop, y1=y_crop, x2=x_crop+crop_size, y2=y_crop+crop_size)
            avatar_win_clip = win_cropped.resized((320, 320)).without_audio()
            
            # Initialize rembg for full-screen presenter
            from rembg import remove, new_session
            rembg_session_fs = new_session(model_name="u2net_human_seg")
            mask_cache_fs = {}
            mask_cache_win = {}
            
            # Pre-generate card mask for windowed layout once
            card_mask_img = Image.new("L", (320, 320), 0)
            card_mask_draw = ImageDraw.Draw(card_mask_img)
            card_mask_draw.rounded_rectangle([0, 0, 320, 320], radius=24, fill=255)
            card_mask_arr = np.array(card_mask_img).astype(np.float32) / 255.0
            
            avatar_pip = None
            ring_clip = None
        else:
            # ── CATEGORY-SPECIFIC AVATAR STYLE ────────────────────────────────────
            # Get category-specific visual style for visual variety
            category = script_json.get("sub_category", "").lower()
            is_shorts = not is_longform
            cat_style = _get_category_avatar_style(category, is_shorts)
            
            # Apply category-specific scale multiplier
            avatar_scale_mult *= cat_style.get("scale_mult", 1.0)
            
            # Recalculate dimensions with category scale
            cur_w = max(1, int(width_pip * avatar_scale_mult))
            cur_h = max(1, int(height_pip * avatar_scale_mult))

            avatar_clip = vid_clip.resized((cur_w, cur_h)).without_audio()
            
            # ── SAFE ZONE: Dedicated corner for talking head with clean backdrop ────────
            # Reserve the avatar corner zone in SafeZoneCalculator
            avatar_zone = safe_zones.get_position("avatar_corner")
            if avatar_zone:
                av_x, av_y, av_w, av_h = avatar_zone
                print(f"🎯 Talking head assigned to dedicated corner: ({av_x}, {av_y}) size {av_w}x{av_h}")
            
            # ── AVATAR EXPRESSION TRIGGERS ──────────────────────────────────
            # Detect trigger words in script and prepare expression overlay clips
            expression_segments = []
            if not is_longform and script_json.get("enable_expressions", True):
                script_text = script_json.get("script", "")
                word_timestamps = script_json.get("word_timestamps", [])
                expression_segments = _get_avatar_expression_segments(script_text, audio_duration, word_timestamps)
                if expression_segments:
                    print(f"🎭 Avatar expressions triggered: {len(expression_segments)} segments")
            
            # Pre-load expression video clips (fallback to main avatar if not found)
            expression_clips = {}
            for expr_type, expr_path in EXPRESSION_VIDEO_MAP.items():
                if os.path.exists(expr_path):
                    try:
                        expr_clip = VideoFileClip(expr_path)
                        if expr_clip.duration < audio_duration:
                            expr_clip = expr_clip.with_effects([vfx.Loop(duration=audio_duration)])
                        else:
                            expr_clip = expr_clip.subclipped(0, audio_duration)
                        # Apply same crop/resize as main avatar
                        ew, eh = expr_clip.size
                        if ew/eh > target_aspect:
                            new_w = int(eh * target_aspect)
                            x1 = (ew - new_w) // 2
                            expr_clip = expr_clip.cropped(x1=x1, y1=0, x2=x1+new_w, y2=eh)
                        else:
                            new_h = int(ew / target_aspect)
                            y1 = int((eh - new_h) * 0.12) if eh > new_h else 0
                            expr_clip = expr_clip.cropped(x1=0, y1=y1, x2=ew, y2=y1+new_h)
                        expr_clip = expr_clip.resized((cur_w, cur_h)).without_audio()
                        expression_clips[expr_type] = expr_clip
                    except Exception as e:
                        print(f"⚠️ Failed to load expression clip {expr_type}: {e}")
            
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
                shadow_cache = {}
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
                    
                    # ── EDGE FEATHERING: Eliminate haloing around shoulders/head ──────
                    # Erode mask slightly to remove fringe, then feather edges
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask = cv2.erode(mask, kernel, iterations=1)
                    # Gaussian blur for soft edge transition (feathering)
                    mask = cv2.GaussianBlur(mask, (5, 5), 0)
                    
                    mask_cache[frame_idx] = mask
                    return mask
                
                def make_shadow_frame(t):
                    """Generate a drop shadow for the presenter to ground them against background."""
                    frame_idx = int(round(t * fps))
                    if frame_idx in shadow_cache:
                        return shadow_cache[frame_idx]
                    
                    # Get the base mask
                    frame = unmasked_avatar.get_frame(t)
                    rgba = remove(
                        frame,
                        session=rembg_session,
                        alpha_matting=False,
                        post_process_mask=True
                    )
                    mask = (rgba[:, :, 3] / 255.0).astype(np.float32)
                    
                    # Erase watermark area
                    h_mask, w_mask = mask.shape
                    watermark_height = int(h_mask * 0.12)
                    mask[-watermark_height:, :] = 0.0
                    
                    # Create shadow: offset down-right, blur heavily, reduce opacity
                    shadow = cv2.GaussianBlur(mask, (21, 21), 0)
                    # Offset shadow (5px down, 3px right)
                    M = np.float32([[1, 0, 3], [0, 1, 5]])
                    shadow = cv2.warpAffine(shadow, M, (w_mask, h_mask), borderMode=cv2.BORDER_CONSTANT)
                    # Reduce opacity to 30%
                    shadow = shadow * 0.3
                    
                    shadow_cache[frame_idx] = shadow
                    return shadow
                
                mclip = VideoClip(make_mask_frame, is_mask=True, duration=audio_duration)
                avatar_clip = avatar_clip.with_mask(mclip)
                
                # Create shadow clip (rendered behind avatar)
                shadow_clip = VideoClip(lambda t: np.zeros((cur_h, cur_w, 3), dtype=np.uint8), duration=audio_duration)
                shadow_mask_clip = VideoClip(make_shadow_frame, is_mask=True, duration=audio_duration)
                shadow_clip = shadow_clip.with_mask(shadow_mask_clip)
                
                # Store shadow clip for later compositing (will be added to base_layers)
                # We'll return it via a closure or attach to avatar_clip
                avatar_clip.shadow_clip = shadow_clip
                
                print("   ✅ Dynamic AI background removal with edge feathering & drop shadow applied.")
                
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
            # Calculate a slow zoom that increases scale by 10% over the full video
            zoom_speed = 0.10 / max(audio_duration, 1.0)
            
            # Combined dynamic scale: shrink/glide for Shorts intro + continuous micro-breathing/slow zoom
            def avatar_resize_fn(t):
                base_scale = 1.0 + zoom_speed * t + 0.006 * math.sin(t * 1.8)
                if is_longform:
                    return base_scale
                    
                # Zoom-reveal: start full-screen, zoom to PiP at hook (2.5s)
                # Only for asymmetric layout (full-screen presenter)
                layout_type = layout.get("layout_type", "asymmetric")
                if layout_type == "asymmetric" and layout.get("zoom_reveal", True):
                    reveal_time = 2.5
                    if t < reveal_time:
                        # Start at full-screen scale (cover 9:16 frame), ease to normal
                        fullscreen_scale = max(FRAME_W / cur_w, FRAME_H / cur_h) * 1.1
                        p = t / reveal_time
                        p = 1.0 - (1.0 - p)**4  # quartic ease-out for dramatic slowdown
                        intro_scale = fullscreen_scale + (1.0 - fullscreen_scale) * p
                    else:
                        intro_scale = 1.0
                else:
                    # Standard glide/shrink logic for Shorts
                    glide_dur = 3.5
                    scale_start = 1080.0 / cur_w
                    scale_end = 1.0
                    
                    if t < glide_dur:
                        p = t / glide_dur
                        p = 1.0 - (1.0 - p)**3  # cubic ease-out
                        intro_scale = scale_start + (scale_end - scale_start) * p
                    else:
                        intro_scale = scale_end
                        
                return intro_scale * base_scale

            avatar_clip = avatar_clip.with_effects([
                # Continuous dynamic zoom-in + Micro-Breathing + intro glide resize
                vfx.Resize(avatar_resize_fn), 
                # Natural Head Tilt: Very subtle +/- 0.5 degree swing
                vfx.Rotate(lambda t: 0.6 * math.sin(t * 1.4 + 0.5)),
            ])
            # Color Matching: Disabled to keep avatar look intact (lip sync)

            # Recalculate dimensions
            cur_w = max(1, int(width_pip * avatar_scale_mult))
            cur_h = max(1, int(height_pip * avatar_scale_mult))

            # ── IMPROVEMENT #1: Circular Face-Cam Frame ─────────
            ring_clip = None
            ring_size = 0
            # Enable circular face-cam for shorts
            enable_circular_for_shorts = (
                not is_longform 
                and layout_variation_enabled 
                and layout.get("layout_type") in ["split_screen", "hero_center", "corner_cycling", "top_center", "side_strip"]
            )
            if enable_circular_for_shorts:
                try:
                    avatar_clip, ring_clip, ring_size = _apply_circular_facecam_frame(
                        avatar_clip, cur_w, cur_h, accent_color, audio_duration, 
                        is_longform=False, cat_style=None
                    )
                    print(f"   ✅ Circular face-cam frame applied")
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
                    # Use SafeZoneCalculator for dedicated talking head corner
                    avatar_pos = safe_zones.get_position("avatar_corner")
                    if avatar_pos:
                        av_x, av_y, av_w, av_h = avatar_pos
                        # Position avatar in its reserved zone
                        home_x = av_x + (av_w - scaled_w) // 2
                        home_y = av_y + (av_h - scaled_h) // 2
                    else:
                        # Fallback to layout-based positioning
                        layout_type = layout.get("layout_type", "asymmetric")
                        # Corner cycling for extra variety (uses daily tracker)
                        corner_index = layout.get("corner_index", 0)
                        corners = [
                            (40.0, 120.0),                                    # top-left
                            (FRAME_W - scaled_w - 40.0, 120.0),              # top-right
                            (40.0, FRAME_H - scaled_h - 180.0),              # bottom-left
                            (40.0, FRAME_H - scaled_h - 180.0),              # bottom-left (changed from bottom-right)
                        ]
                        
                        # Home position (where avatar returns when not in transition)
                        if layout_variation_enabled:
                            if layout_type == "split_screen":
                                home_x = FRAME_W - scaled_w - 40.0
                                home_y = 120.0
                            elif layout_type == "hero_center":
                                home_x = 40.0
                                home_y = FRAME_H - scaled_h - 180.0
                            elif layout_type == "side_strip":
                                home_x = 20.0
                                home_y = (FRAME_H - scaled_h) / 2.0
                            elif layout_type == "top_center":
                                # Left middle (changed from top center)
                                home_x = 20.0
                                home_y = (FRAME_H - scaled_h) / 2.0
                            elif layout_type == "corner_cycling":
                                home_x, home_y = corners[corner_index % 4]
                            else:  # asymmetric
                                home_x = (FRAME_W - scaled_w) / 2.0
                                home_y = FRAME_H - scaled_h
                        else:
                            home_x = (FRAME_W - scaled_w) / 2.0 + layout["avatar_x_offset"]
                            home_y = FRAME_H - scaled_h - 30.0
                    
                    # Screenshot-aware positioning: glide to top-right corner before screenshot, fade during, return after
                    # Screenshot typically shows 4.0-12.0s
                    ss_start = 4.0
                    ss_end = min(12.0, audio_duration - 2.0)
                    transition_dur = 1.5  # Time to glide to/from corner
                    
                    # Check if we're in screenshot transition period
                    if ss_start - transition_dur <= t < ss_start:
                        # Glide from home to top-right corner before screenshot
                        corner_x = FRAME_W - scaled_w - 40.0
                        corner_y = 120.0
                        p = (t - (ss_start - transition_dur)) / transition_dur
                        p = 1.0 - (1.0 - p)**3  # ease-out
                        x_pos = home_x + (corner_x - home_x) * p
                        y_pos = home_y + (corner_y - home_y) * p
                        return (int(x_pos), int(y_pos))
                    elif ss_start <= t < ss_end:
                        # During screenshot: stay at corner (will be faded out by mask)
                        return (int(FRAME_W - scaled_w - 40.0), int(120.0))
                    elif ss_end <= t < ss_end + transition_dur:
                        # Glide back from corner to home after screenshot
                        corner_x = FRAME_W - scaled_w - 40.0
                        corner_y = 120.0
                        p = (t - ss_end) / transition_dur
                        p = p * p * (3 - 2 * p)  # ease-in-out (smoothstep)
                        x_pos = corner_x + (home_x - corner_x) * p
                        y_pos = corner_y + (home_y - corner_y) * p
                        return (int(x_pos), int(y_pos))
                    
                    # Normal positioning (outside screenshot transitions)
                    if layout_variation_enabled:
                        if layout_type == "asymmetric":
                            return (int(home_x), int(home_y))
                    else:
                        # Legacy default with glide from bottom-left
                        glide_dur = 3.5
                        x_start = 0.0
                        y_start = 350.0
                        x_end = home_x
                        y_end = home_y
                        
                        if t < glide_dur:
                            p = t / glide_dur
                            p = 1.0 - (1.0 - p)**3
                            x_pos = x_start + (x_end - x_start) * p
                            y_pos = y_start + (y_end - y_start) * p
                            return (int(x_pos), int(y_pos))
                        return (int(x_end), int(y_end))
                    
                    # Layout variations with glide from bottom-left
                    glide_dur = 3.5
                    x_start = 0.0
                    y_start = 350.0
                    
                    if t < glide_dur:
                        p = t / glide_dur
                        p = 1.0 - (1.0 - p)**3
                        x_pos = x_start + (home_x - x_start) * p
                        y_pos = y_start + (home_y - y_start) * p
                        return (int(x_pos), int(y_pos))
                    return (int(home_x), int(home_y))

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

            # ── EXPRESSION COMPOSITING ─────────────────────────────────────────
            # Composite expression clips over base avatar at trigger times
            if expression_segments and expression_clips:
                def make_expression_frame(t):
                    # Check if any expression should be active at time t
                    for start, end, expr_type in expression_segments:
                        if start <= t < end and expr_type in expression_clips:
                            return expression_clips[expr_type].get_frame(t)
                    return avatar_clip.get_frame(t)
                
                def make_expression_mask(t):
                    # Use avatar's mask for expressions too (same face shape)
                    if avatar_clip.mask:
                        return avatar_clip.mask.get_frame(t)
                    return np.ones((cur_h, cur_w), dtype=np.float32)
                
                expr_clip = VideoClip(make_expression_frame, duration=audio_duration)
                expr_clip = expr_clip.with_mask(VideoClip(make_expression_mask, is_mask=True, duration=audio_duration))
                avatar_clip = expr_clip
                print(f"   ✅ Expression compositing applied: {len(expression_segments)} segments")

            avatar_pip = avatar_clip.with_position(pip_position).with_start(0)

    # ── LAYERS ───────────────────────────────────────────────────────────
    if is_longform:
        screenshot_clips = _longform_article_screenshot_clips(script_json, audio_duration)
    else:
        screenshot_path = script_json.get("screenshot_path")
        screenshot_clips = _article_screenshot_clip(screenshot_path, audio_duration)
    gradient = _gradient_clip(audio_duration, height_pct=layout["gradient_height_pct"], position=layout["gradient_position"], is_longform=is_longform)
    
    # ── CI-LITE: Skip heavy overlay layers in CI environment ───────────────────
    if not CI_LITE:
        # Ambient Particles
        bg_accent = layout["theme"]["accent"] if (layout.get("theme") and os.environ.get("ENABLE_LAYOUT_VARIATION", "0") == "1") else accent_color
        particle_layer = _ambient_particles(audio_duration, bg_accent, particle_style=layout["particle_style"])

        # ── HUMAN REALISM OVERLAYS ───────────────────────────────────────────────
        grain_layer = _generate_film_grain(audio_duration, FRAME_W, FRAME_H)
        flare_layer = None  # Disabled lens flare circle drifting left-to-right as requested
    else:
        particle_layer = None
        grain_layer = None
        flare_layer = None
    
    # ── COMPLIANCE & BRANDING ────────────────────────────────────────────────
    disclosure = _ai_disclosure_overlay(audio_duration)
    
    # ── ENGAGEMENT LAYERS (Retention Boosters) ────────────────────────────────
    engagement_clips = []
    
    # Re-enabled hook overlay stack
    hook_text = script_json.get("hook_text") or script_json.get("hook") or script_json.get("title", "Tech News")
    hook_overlay = _hook_text_overlay(hook_text, accent_color, audio_duration)
    if hook_overlay:
        engagement_clips.append(hook_overlay)

    # ── MID-VIDEO RE-ENGAGEMENT HOOKS ──────────────────────────────────────
    # Add summary slides, quiz prompts, and pattern interrupt markers at strategic points
    if is_longform and not CI_LITE:
        # Summary slides at 30%, 50%, 70% of video
        summary_points = [
            (0.30, "Core concept explained — now diving into implementation"),
            (0.50, "Key insight: the architecture scales differently than expected"),
            (0.70, "Putting it all together — here's the complete workflow"),
        ]
        for pct, summary_text in summary_points:
            start_time = audio_duration * pct
            if start_time < audio_duration - 5:
                slide = _summary_slide_clip(summary_text, accent_color, audio_duration, start_time)
                if slide:
                    engagement_clips.append(slide)
        
        # Quiz prompts at natural pause points (topic transitions)
        chapters = script_json.get("chapters", [])
        if chapters:
            for i, ch in enumerate(chapters):
                if i > 0 and i < len(chapters) - 1:
                    start_s = float(ch.get("approx_start_seconds", 0))
                    if start_s > 10 and start_s < audio_duration - 10:
                        question = f"What's your take on {ch.get('chapter_title', 'this section')[:30]}?"
                        quiz = _quiz_prompt_clip(question, accent_color, audio_duration, start_s + 2.0)
                        if quiz:
                            engagement_clips.append(quiz)
                        
                        # Pattern interrupt marker
                        marker = _pattern_interrupt_marker(accent_color, audio_duration, start_s, "shift")
                        if marker:
                            engagement_clips.append(marker)
        
        # Payoff zone marker at 70%
        payoff_start = audio_duration * 0.70
        if payoff_start < audio_duration - 10:
            marker = _pattern_interrupt_marker(accent_color, audio_duration, payoff_start, "payoff")
            if marker:
                engagement_clips.append(marker)
            
            # Final recap before outro
            recap_start = audio_duration * 0.85
            if recap_start < audio_duration - 8:
                marker = _pattern_interrupt_marker(accent_color, audio_duration, recap_start, "recap")
                if marker:
                    engagement_clips.append(marker)
    else:
        # For Shorts: single mid-point engagement
        mid_point = audio_duration * 0.50
        if mid_point < audio_duration - 3:
            # Quick summary
            slide = _summary_slide_clip("Here's the key takeaway...", accent_color, audio_duration, mid_point, duration=2.5)
            if slide:
                engagement_clips.append(slide)
            
            # Pattern interrupt
            marker = _pattern_interrupt_marker(accent_color, audio_duration, mid_point, "shift", duration=1.0)
            if marker:
                engagement_clips.append(marker)

    # Phased scans removed in favor of full-screen loops as requested
    pass
    
    # ── CHAPTER TRANSITIONS / TITLE CARDS ──────────────────────────────────────
    # Add cinematic chapter transitions at topic shifts (for longform)
    if is_longform:
        chapters = script_json.get("chapters", [])
        if chapters:
            for i, ch in enumerate(chapters):
                if i > 0:  # Skip first chapter (has intro)
                    start_s = float(ch.get("approx_start_seconds", 0))
                    if start_s > 5 and start_s < audio_duration - 5:
                        chapter_title = ch.get("chapter_title", f"Chapter {i+1}")
                        trans_clip = _chapter_transition_card(
                            chapter_title, i + 1, len(chapters),
                            accent_color, start_s, audio_duration, hold=3.0
                        )
                        if trans_clip:
                            engagement_clips.append(trans_clip)
                            print(f"   📖 Chapter transition added: {chapter_title} at {start_s:.1f}s")
    
    # ── CI-LITE: Skip heavy overlay layers in CI environment ───────────────────
    if not CI_LITE:
        # ── STAT CALLOUT GRAPHICS ──────────────────────────────────────────────────
        # Add metric callouts from script data (%, latency, throughput, etc.)
        metric_popups = script_json.get("metric_popups", [])
        # Also extract from fact_scripts and key_stats
        fact_scripts = script_json.get("fact_scripts", [])
        for fs in fact_scripts:
            key_stat = fs.get("key_stat", "")
            fact_num = fs.get("fact_number", 0)
            if key_stat and fact_num > 0:
                # Parse stat value and label
                import re
                match = re.match(r'([\d\.]+[%BMKx]?)\s*(.*)', key_stat.strip())
                if match:
                    stat_val, stat_lbl = match.groups()
                    metric_popups.append({
                        "value": stat_val,
                        "label": stat_lbl.strip() or "Metric",
                        "timestamp": float(fs.get("approx_start_seconds", 5.0 + fact_num * 30))
                    })
        
        for mp in metric_popups:
            ts_val = mp.get("timestamp", 0)
            if ts_val > 3 and ts_val < audio_duration - 3:
                callout = _stat_callout_clip(
                    mp.get("value", ""), mp.get("label", "Metric"),
                    ts_val, accent_color, audio_duration, hold=2.5
                )
                if callout:
                    engagement_clips.append(callout)
                    print(f"   📊 Stat callout added: {mp.get('value')} {mp.get('label')} at {ts_val:.1f}s")
        
        # ── CODE SNIPPET / DIAGRAM DISPLAY ────────────────────────────────────────
        # Add code snippets and architecture diagrams for technical content
        code_snippets = script_json.get("code_snippets", [])
        for cs in code_snippets:
            ts_val = cs.get("timestamp", 0)
            if ts_val > 5 and ts_val < audio_duration - 5:
                code_clip = _code_snippet_clip(
                    cs.get("code", ""), cs.get("language", "python"),
                    ts_val, accent_color, audio_duration, hold=cs.get("duration", 4.0)
                )
                if code_clip:
                    engagement_clips.append(code_clip)
                    print(f"   💻 Code snippet added at {ts_val:.1f}s")
        
        arch_diagrams = script_json.get("architecture_diagrams", [])
        for ad in arch_diagrams:
            ts_val = ad.get("timestamp", 0)
            if ts_val > 5 and ts_val < audio_duration - 5:
                arch_clip = _architecture_diagram_clip(
                    ad.get("components", []), ad.get("connections", []),
                    ts_val, accent_color, audio_duration, hold=ad.get("duration", 5.0)
                )
                if arch_clip:
                    engagement_clips.append(arch_clip)
                    print(f"   🏗️ Architecture diagram added at {ts_val:.1f}s")
        
        # ── AUDIO-DRIVEN PATTERN INTERRUPTS (Every 4-6 seconds) ──────────────────
        # Generate visual cuts synchronized to audio beats/pace
        interrupt_interval = 5.0  # Every 5 seconds
        if is_longform:
            interrupt_interval = 6.0  # Slightly longer for longform
        
        pattern_interrupt_times = []
        t = interrupt_interval
        while t < audio_duration - 5:
            # Avoid chapter boundaries
            at_boundary = False
            for ch in script_json.get("chapters", []):
                ch_start = float(ch.get("approx_start_seconds", 0))
                if abs(t - ch_start) < 3.0:
                    at_boundary = True
                    break
            if not at_boundary:
                pattern_interrupt_times.append(t)
            t += interrupt_interval
        
        # Add snap-zoom and flash effects at these intervals
        for pi_t in pattern_interrupt_times:
            # Snap-zoom punch
            from config import ENABLE_CINEMATIC_TRANSITIONS
            if ENABLE_CINEMATIC_TRANSITIONS:
                trans_clip = _create_transition_clip("zoom_punch", duration=0.2)
                if trans_clip:
                    trans_clip = trans_clip.with_start(pi_t)
                    engagement_clips.append(trans_clip)
                
                # Accent flash
                flash = ColorClip(size=(FRAME_W, FRAME_H), color=accent_color, duration=0.15).with_opacity(0.4)
                flash = flash.with_start(pi_t).with_effects([vfx.CrossFadeOut(0.1)])
                engagement_clips.append(flash)
        
        print(f"   ⚡ Pattern interrupts added: {len(pattern_interrupt_times)} cuts every {interrupt_interval}s")
    else:
        print("   🔧 CI-LITE: Skipping stat callouts, code snippets, diagrams, pattern interrupts")
    
    # ── LAYER QUIZ: Quiz Countdown & CTA ──────────────────────────────────────
    is_quiz = sub_category.lower() in ["quiz", "quiz & trivia", "trivia"]
    if is_quiz and not CI_LITE:
        print("🎯 QUIZ MODE: Adding quiz-specific visual layers")
        
        # Find the quiz reveal chunk to time the countdown
        # The countdown should appear ~3 seconds before the answer reveal
        reveal_chunk = None
        options_chunk = None
        for chunk in chunks:
            if chunk.get("infographic_type") == "quiz_reveal":
                reveal_chunk = chunk
            elif chunk.get("infographic_type") == "quiz_options":
                options_chunk = chunk
        
        # Add countdown before reveal (extended to 5s for better retention)
        if reveal_chunk:
            reveal_start = reveal_chunk.get("start", 0)
            countdown_duration = 5.0  # Extended from 3s to 5s for better engagement
            countdown_start = max(0, reveal_start - countdown_duration - 0.5)  # 0.5s buffer
            countdown_clips = _quiz_countdown_overlay(accent_color, countdown_start, audio_duration, duration=countdown_duration)
            if countdown_clips:
                engagement_clips.extend(countdown_clips)
        
        # Add quiz-specific CTA at the end
        comment_hook = script_json.get("comment_hook", "QUIZ")
        incentive_cta_type = script_json.get("incentive_cta_type", "comment_trigger")
        digital_asset_offer = script_json.get("digital_asset_offer", "the quiz pack")
        quiz_cta = _quiz_cta_overlay(comment_hook, incentive_cta_type, digital_asset_offer, accent_color, audio_duration)
        if quiz_cta:
            engagement_clips.append(quiz_cta)
        
        # Add animated "Comment 'KEYWORD'" CTA graphic (visual callout)
        animated_cta = _animated_comment_cta(comment_hook, accent_color, audio_duration)
        if animated_cta:
            engagement_clips.append(animated_cta)
    elif is_quiz and CI_LITE:
        print("   🔧 CI-LITE: Skipping quiz CTA layers")
    else:
        # Standard CTA for non-quiz videos (identity-based)
        identity_text = script_json.get("identity_cta", "")
        identity_cta = _identity_cta_overlay(identity_text, accent_color, audio_duration)
        if identity_cta:
            engagement_clips.append(identity_cta)
    
    # ── SAVE/SHARE VISUAL CUES ─────────────────────────────────────────────────
    # Add animated Save/Share overlays based on incentive_cta_type
    incentive_cta_type = script_json.get("incentive_cta_type", "")
    if incentive_cta_type == "save_trigger":
        save_cta = _save_cta_overlay(accent_color, audio_duration, "Save this for later!")
        if save_cta:
            engagement_clips.append(save_cta)
            print("   ✨ Added Save CTA visual cue")
    elif incentive_cta_type == "share_trigger":
        share_cta = _share_cta_overlay(accent_color, audio_duration, "Send this to a developer!")
        if share_cta:
            engagement_clips.append(share_cta)
            print("   ✨ Added Share CTA visual cue")

    # ── LAYER 12: Telegram CTA card (Last 6 seconds) ──────────────────────────
    # Disabled full-screen sequential CTA as requested; keeping only the 3s cropped outro.
    # telegram_cta = _telegram_cta_overlay(audio_duration)
    # if telegram_cta:
    #     if isinstance(telegram_cta, list):
    #         engagement_clips.extend(telegram_cta)
    #     else:
    #         engagement_clips.append(telegram_cta)

    # ── COMPOSITING ──
    # Collect infographics logic — ENABLED if feature flag is active
    infographic_clips = []
    enable_infographics = os.environ.get("ENABLE_INFOGRAPHICS", "1") == "1"
    if enable_infographics and not CI_LITE:
        for chunk in chunks:
            if chunk.get("has_infographic") and chunk.get("infographic_type"):
                iclip = _infographic_card_clip(
                    chunk.get("infographic_type"),
                    chunk.get("infographic_data"),
                    accent_color,
                    chunk["start"],
                    chunk["duration"],
                    audio_duration
                )
                if iclip:
                    infographic_clips.append(iclip)
    elif enable_infographics and CI_LITE:
        print("   🔧 CI-LITE: Skipping infographics")

    # Collect settings mockup clips
    settings_mockup_clips = []
    enable_mockup = os.environ.get("ENABLE_SETTINGS_MOCKUP", "1") == "1"
    if enable_mockup and not CI_LITE:
        for chunk in chunks:
            if chunk.get("is_setting_chunk"):
                sclip = _create_settings_mockup_clip(
                    chunk.get("text", ""),
                    chunk["start"],
                    chunk["duration"],
                    accent_color,
                    audio_duration
                )
                if sclip:
                    settings_mockup_clips.append(sclip)

    # ── DYNAMIC LAYOUT VISUAL ELEMENTS ─────────────────────────────────────────
    # Add layout-aware visual treatments per chunk based on dynamic layout
    layout_visual_clips = []
    enable_dynamic_layout = os.environ.get("ENABLE_DYNAMIC_LAYOUT", "1") == "1"
    if enable_dynamic_layout and not CI_LITE:
        try:
            from ecosystem_logic import get_layout_for_chunk
            category = script_json.get("sub_category", "AI & Tech Tools")
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                layout_type = get_layout_for_chunk(chunk, category, i, total_chunks)
                visual_type = chunk.get("visual_type", "")
                
                # Add layout-specific visual element
                layout_clip = _create_layout_visual_clip(
                    layout_type, visual_type, chunk, accent_color, audio_duration
                )
                if layout_clip:
                    layout_visual_clips.append(layout_clip)
                    
            print(f"   🎨 Dynamic Layout Visuals: {len(layout_visual_clips)} clips added")
        except Exception as e:
            print(f"   ⚠️ Dynamic layout visuals failed: {e}")
    elif enable_mockup and CI_LITE:
        print("   🔧 CI-LITE: Skipping settings mockup")

    # ── LONGFORM: "FACT X/N" BADGE OVERLAYS ──────────────────────────────────
    longform_badge_clips = []
    if is_longform and not CI_LITE and script_json.get("longform_format") in ["did_you_know", "vaibhav", "chaptered"]:
        fact_timestamps = script_json.get("fact_timestamps", [])
        total_facts = script_json.get("num_facts", 10)
        is_vaibhav = script_json.get("longform_format") == "vaibhav"
        
        # Reserve badge zone in safe zones
        badge_pos = safe_zones.find_safe_position(300, 80, element_type="badge")
        if badge_pos:
            badge_x, badge_y = badge_pos
        else:
            badge_x, badge_y = 40, 40  # fallback
        
        for i, ft in enumerate(fact_timestamps):
            fact_num = ft.get("fact_number", i + 1)
            
            # Skip non-numeric entries only for did_you_know
            if not is_vaibhav:
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
            if is_vaibhav:
                badge_text = ft.get("section", f"SECTION {fact_num}").upper()
            else:
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
            # Use safe zone position for badge
            b_clip = b_clip.with_mask(b_mask).with_position((badge_x, badge_y)).with_start(start_s)
            b_clip = b_clip.with_effects([vfx.CrossFadeIn(0.3)])
            longform_badge_clips.append(b_clip)

    # Default constant for badge generation when fact_timestamps is missing
    LONGFORM_NUM_TOPICS_DEFAULT = 10

    # Stack background, then screenshot clips on top of background
    topic_transition_clips = []
    if is_longform:
        topic_transition_clips = _longform_topic_transition_clips(script_json, audio_duration)
        
    base_layers = bg_layer_clips + burst_clips + ([particle_layer] if particle_layer else []) + screenshot_clips + topic_transition_clips + infographic_clips + settings_mockup_clips
    
    # Add overlays
    base_layers.append(gradient)
    base_layers.extend(logo_clips)
    
    if flare_layer: base_layers.append(flare_layer)
    if grain_layer: base_layers.append(grain_layer)
    # Add presenter drop shadow BEFORE avatar (renders behind)
    if avatar_pip and hasattr(avatar_pip, 'clip') and hasattr(avatar_pip.clip, 'shadow_clip'):
        base_layers.append(avatar_pip.clip.shadow_clip.with_position(pip_position).with_start(0))
    if avatar_pip: base_layers.append(avatar_pip)
    base_layers.extend(longform_badge_clips)
    
    # Filter out any None values that may have been added
    base_layers = [clip for clip in base_layers if clip is not None]
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
        
        # Use safe zone for branding stack position
        branding_pos = safe_zones.find_safe_position(80, 300, element_type="branding_stack")
        if branding_pos:
            brand_x, brand_y = branding_pos
        else:
            brand_x, brand_y = FRAME_W - 80, 40  # fallback top-right
        
        card_size = 65 # Slightly smaller for stack
        margin = 15
        current_y = brand_y
        
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
                    draw.rounded_rectangle([0, 0, card_size, card_size], radius=12, fill=(255,255,255,230))
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

    # ── ENTITY LOGO PIP (near avatar) ──────────────────────────────────────
    if not is_longform and avatar_pip:
        try:
            entity_logo_clips = _create_entity_logo_pip_clips(
                script_json, pip_position, audio_duration, cur_w, cur_h, accent_color
            )
            for logo_clip, start_t, end_t in entity_logo_clips:
                # Trim clip to active interval
                logo_clip = logo_clip.subclipped(start_t, end_t).with_start(start_t)
                base_layers.append(logo_clip)
            if entity_logo_clips:
                print(f"   🏷️ Entity logo PIP clips added: {len(entity_logo_clips)}")
        except Exception as e:
            print(f"   ⚠️ Entity logo PIP failed (non-fatal): {e}")

# ═════════════════════════════════════════════════════════════════════
    # LONGFORM RETENTION UPGRADE: Wire in new premium visual layers
    # ════════════════════════════════════════════════════════════════════
    if not CI_LITE:
        snap_zoom_timestamps = []  # Used per-frame in make_final_frame
        
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

        if is_longform and script_json.get("longform_format") in ["did_you_know", "vaibhav", "chaptered"]:
            fact_timestamps_lf = script_json.get("fact_timestamps", [])

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
    else:
        snap_zoom_timestamps = []
        print("   🔧 CI-LITE: Skipping LONGFORM RETENTION UPGRADE layers")



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
    
    # Use safe zone to position progress bar above YouTube bottom UI (120px from bottom)
    yt_bottom_ui_start = FRAME_H - 120
    safe_bottom_y = yt_bottom_ui_start - pb_h - 10  # 10px margin above YouTube UI
    
    def make_progress_frame(t):
        color = get_progress_color(t)
        base_img = np.zeros((pb_h, FRAME_W, 3), dtype=np.uint8)
        base_img[:, :] = color
        return base_img
        
    progress = VideoClip(make_progress_frame, duration=audio_duration)
    if pb_pos == "top":
        # Top position: below YouTube top UI (80px from top)
        progress = progress.with_position(lambda t: (int((t / max(audio_duration, 0.01)) * FRAME_W) - FRAME_W, 90))
    else:
        # Bottom position: above YouTube bottom UI
        progress = progress.with_position(lambda t: (int((t / max(audio_duration, 0.01)) * FRAME_W) - FRAME_W, safe_bottom_y))
    base_layers.append(progress)
    
    # ── CI-LITE: Skip remaining optional layers in CI environment ────────────
    if not CI_LITE:
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
                # Use safe zone for emoji position (center but avoiding avatar/badge zones)
                emoji_pos = safe_zones.find_safe_position(400, 400, element_type="emoji")
                if emoji_pos:
                    pos = emoji_pos
                else:
                    pos = (FRAME_W//2 - 200, FRAME_H//2 - 400)
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
    else:
        print("   🔧 CI-LITE: Skipping infinite loop, emoji popups, easter egg, HUD overlays")

    # Free memory before compositing all layers
    gc.collect()
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
    sfx_cues = list(script_json.get("sfx_cues", []))
    
    # Auto-generate SFX cues to match visual pacing and transitions
    last_whoosh = -5.0
    for idx, c in enumerate(chunks):
        start_t = c.get("start", 0.0)
        # 1. Slide transition whoosh SFX at chunk boundaries (at least 2.0s apart)
        if start_t - last_whoosh >= 2.0 and start_t < audio_duration:
            sfx_cues.append({
                "type": "woosh",
                "timestamp": start_t
            })
            last_whoosh = start_t
            
        # 2. Pop click SFX exactly when setting toggle flips
        if c.get("is_setting_chunk"):
            toggle_t = start_t + TOGGLE_FLIP_OFFSET_SEC
            if toggle_t < audio_duration:
                sfx_cues.append({
                    "type": "pop",
                    "timestamp": toggle_t
                })
                
        # 3. Pop entrance SFX when infographics appear
        if c.get("has_infographic") and c.get("infographic_type"):
            if start_t < audio_duration:
                sfx_cues.append({
                    "type": "pop",
                    "timestamp": start_t
                })
    
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
        
        # Zero-gap subtitle timing: a chunk remains active until the next one starts
        active_chunk = None
        for i, chunk in enumerate(chunks):
            chunk_start = chunk["start"]
            next_start = chunks[i+1]["start"] if i + 1 < len(chunks) else audio_duration
            if chunk_start - 0.1 <= t < next_start:
                active_chunk = chunk
                break
            
        if active_chunk and active_chunk.get("is_setting_chunk"):
            enable_mockup = os.environ.get("ENABLE_SETTINGS_MOCKUP", "1") == "1"
            if enable_mockup:
                dur = active_chunk.get("duration", 1.0)
                if dur > 0.1:
                    t_rel = max(0.0, min(t - active_chunk["start"], dur))
                    # Sine wave easing from 1.0 to 1.22x
                    zoom_factor = 1.0 + 0.22 * math.sin(math.pi * t_rel / dur)
                    try:
                        h, w, c = bg_frame.shape
                        pil_img = Image.fromarray(bg_frame)
                        new_w = int(w / zoom_factor)
                        new_h = int(h / zoom_factor)
                        x1 = (w - new_w) // 2
                        y1 = (h - new_h) // 2
                        cropped = pil_img.crop((x1, y1, x1 + new_w, y1 + new_h))
                        resized = cropped.resize((w, h), Image.BILINEAR)
                        bg_frame = np.array(resized)
                    except Exception as e:
                        pass
            
        if active_chunk:
            enable_hook = os.environ.get("ENABLE_HOOK_OVERLAY", "1") == "1"
            if not is_longform and enable_hook and t < 3.0:
                # Timing collision fix: skip subtitles while hook overlay is showing
                pass
            else:
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

        # Minimize Static Branding: Only pass transparency_img in the first 5 seconds for longform
        this_transparency_img = transparency_img if (not is_longform or t < 5.0) else None
        
        # ── LONGFORM TALING-HEAD OVERLAY SYSTEM (Transitions & Styled Cards) ──
        if is_longform and 'avatar_fs_clip' in locals():
            # Get active fact timestamp
            is_fs = False
            active_headline = ""
            for idx_ft, ft in enumerate(fact_timestamps):
                start_s = float(ft.get("approx_start_seconds", 0))
                next_s = float(fact_timestamps[idx_ft+1].get("approx_start_seconds", 9999)) if idx_ft + 1 < len(fact_timestamps) else 9999
                if start_s <= t < next_s:
                    # First 3 seconds of each fact is a full-screen transition
                    is_fs = (t - start_s) < 3.0
                    active_headline = ft.get("topic", "")
                    break
            
            # Force full-screen presenter during the first 5 seconds of the video (Intro Hook)
            if t < 5.0:
                is_fs = True
                active_headline = script_json.get("title", "")
                
            bg_accent = layout["theme"]["accent"] if layout.get("theme") else accent_color
            
            def apply_color_grade(frame, accent_rgb):
                # Subtle 12% tint matching the theme accent color
                tint_arr = np.full(frame.shape, accent_rgb, dtype=np.uint8)
                return cv2.addWeighted(frame, 0.88, tint_arr, 0.12, 0)

            if is_fs:
                # 1. Blur backdrop completely for full-screen presenter
                bg_frame = cv2.GaussianBlur(bg_frame, (51, 51), 0)
                
                # 2. Get full-screen frame
                fs_frame = avatar_fs_clip.get_frame(t)
                
                # 3. Retrieve or compute mask from rembg
                frame_idx = int(round(t * 30.0))
                if frame_idx in mask_cache_fs:
                    mask_fs = mask_cache_fs[frame_idx]
                else:
                    try:
                        rgba = remove(fs_frame, session=rembg_session_fs, alpha_matting=False, post_process_mask=True)
                        mask_fs = (rgba[:, :, 3] / 255.0).astype(np.float32)
                        
                        # Erase bottom 12% watermark
                        h_m, w_m = mask_fs.shape
                        wm_h = int(h_m * 0.12)
                        mask_fs[-wm_h:, :] = 0.0
                    except Exception as e:
                        mask_fs = np.ones((fs_frame.shape[0], fs_frame.shape[1]), dtype=np.float32)
                    mask_cache_fs[frame_idx] = mask_fs
                
                # 4. Apply color grading
                fs_frame_graded = apply_color_grade(fs_frame, bg_accent)
                
                # 5. Composite full-screen presenter (516x918 centered at bottom)
                # Position: x = (1920 - 516) // 2 = 702, y = 1080 - 918 = 162
                mask_3d = np.expand_dims(mask_fs, axis=2)
                dest_area = bg_frame[162:162+918, 702:702+516]
                blended = fs_frame_graded * mask_3d + dest_area * (1.0 - mask_3d)
                bg_frame[162:162+918, 702:702+516] = blended.astype(np.uint8)
                
                # 6. Render elegant fact title text at top center
                if active_headline:
                    pil_frame = Image.fromarray(bg_frame).convert("RGBA")
                    draw_t = ImageDraw.Draw(pil_frame)
                    title_font = gf(40, bold=True)
                    title_text = active_headline.upper()
                    tb = draw_t.textbbox((0, 0), title_text, font=title_font)
                    tw = tb[2] - tb[0]
                    th = tb[3] - tb[1]
                    rect_x1 = (1920 - tw) // 2 - 40
                    rect_y1 = 40
                    rect_x2 = (1920 + tw) // 2 + 40
                    rect_y2 = 40 + th + 30
                    draw_t.rounded_rectangle([rect_x1, rect_y1, rect_x2, rect_y2], radius=15, fill=(0, 0, 0, 180), outline=(*bg_accent, 180), width=2)
                    draw_t.text(((1920 - tw) // 2, 52), title_text, font=title_font, fill=(255, 255, 255, 255))
                    bg_frame = np.array(pil_frame.convert("RGB"))
            else:
                # Windowed card mode (bottom-right: 320x320)
                win_frame = avatar_win_clip.get_frame(t)
                win_frame_graded = apply_color_grade(win_frame, bg_accent)
                
                # AI background removal for windowed avatar
                win_frame_idx = int(round(t * 30.0))
                if win_frame_idx in mask_cache_win:
                    mask_win = mask_cache_win[win_frame_idx]
                else:
                    try:
                        rgba_win = remove(win_frame, session=rembg_session_fs, alpha_matting=False, post_process_mask=True)
                        mask_win = (rgba_win[:, :, 3] / 255.0).astype(np.float32)
                        # Erase bottom 12% watermark
                        h_wm, w_wm = mask_win.shape
                        wm_h = int(h_wm * 0.12)
                        mask_win[-wm_h:, :] = 0.0
                    except Exception:
                        mask_win = np.ones((win_frame.shape[0], win_frame.shape[1]), dtype=np.float32)
                    mask_cache_win[win_frame_idx] = mask_win
                
                # Combine rembg mask with rounded card mask
                combined_mask = mask_win * card_mask_arr
                
                card_img = Image.new("RGBA", (320, 320), (0, 0, 0, 0))
                c_draw = ImageDraw.Draw(card_img)
                # Glassmorphic rounded rectangle border window
                c_draw.rounded_rectangle([0, 0, 320, 320], radius=24, fill=(15, 15, 20, 180), outline=(*bg_accent, 180), width=2)
                
                # Apply combined mask (bg-removed + rounded corners) to avatar
                win_pil = Image.fromarray(win_frame_graded).convert("RGBA")
                mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8), mode="L")
                win_pil.putalpha(mask_img)
                card_img.alpha_composite(win_pil)
                
                # Paste card in bottom-right corner: x = 1550, y = 610
                main_pil = Image.fromarray(bg_frame).convert("RGBA")
                main_pil.alpha_composite(card_img, dest=(1550, 610))
                bg_frame = np.array(main_pil.convert("RGB"))

        return composite_frame(bg_frame, t, header_img, subtitle_img, this_transparency_img, entity_tags_img)


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

    # Free memory before the heavy write operation
    gc.collect()

    # CI/CD environment detection for memory-optimized rendering
    is_ci = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"
    
    if is_ci:
        # Aggressive memory savings for CI runners
        thread_count = 1
        preset = "ultrafast"
        # Reduce resolution for CI if needed (can be enabled via env var)
        reduce_resolution = os.environ.get("REDUCE_CI_RESOLUTION") == "true"
        if reduce_resolution and not is_longform:
            print("⚠️ CI mode: Reducing resolution to 720x1280 for memory")
            # This would require resizing the final clip, skip for now
    else:
        # For longform, use lower thread count to reduce FFmpeg memory pressure
        thread_count = 2 if is_longform else 4
        preset = "ultrafast"
    
    print(f"🎬 Rendering video: threads={thread_count}, preset={preset}, CI={is_ci}")
    
    # Add timeout and error handling for write_videofile
    try:
        final.write_videofile(
            output_path, fps=30, codec="libx264", audio_codec="aac",
            threads=thread_count, preset=preset, ffmpeg_params=["-pix_fmt", "yuv420p"]
        )
    except Exception as e:
        print(f"❌ Video write failed: {e}")
        # Try with even lower settings as last resort
        if thread_count > 1:
            print("🔄 Retrying with single thread...")
            gc.collect()
            final.write_videofile(
                output_path, fps=30, codec="libx264", audio_codec="aac",
                threads=1, preset="ultrafast", ffmpeg_params=["-pix_fmt", "yuv420p"]
            )
        else:
            raise
    
    try:
        final.close()
    except Exception as e:
        print(f"Cleanup warning: {e}")
        
    return output_path
