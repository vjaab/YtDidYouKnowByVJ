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
LAYER 15: Background music (vol 0.06)
"""

import os
import sys
import io
import math
import random
import threading
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datetime import datetime
from moviepy import (
    VideoClip, ImageClip, VideoFileClip, AudioFileClip,
    CompositeVideoClip, ColorClip, CompositeAudioClip, concatenate_videoclips
)
import moviepy.video.fx as vfx
import moviepy.audio.fx as afx
from config import OUTPUT_DIR, ASSETS_DIR, MUSIC_DIR, BGM_VOLUME, LOGS_DIR, BASE_DIR
import imageio_ffmpeg
from pydub import AudioSegment
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

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
def _ambient_particles(duration, accent_color):
    n = 35
    random.seed(42)
    particles = [(random.uniform(0, FRAME_W), random.uniform(0, FRAME_H),
                  random.uniform(0.1, 0.8), random.uniform(0, FRAME_H), random.uniform(8, 25))
                 for _ in range(n)]

    def make_frame(t):
        img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for px, py, speed, offset, p_size in particles:
            y = (py - speed * t * 60 - offset) % FRAME_H
            cv2.circle(img, (int(px), int(y)), int(p_size), accent_color, -1)
        img = cv2.GaussianBlur(img, (75, 75), 0)
        return img

    def make_mask(t):
        mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        for px, py, speed, offset, p_size in particles:
            y = (py - speed * t * 60 - offset) % FRAME_H
            cv2.circle(mask, (int(px), int(y)), int(p_size), 255, -1)
        mask = cv2.GaussianBlur(mask, (75, 75), 0)
        return (mask.astype(float) / 255.0) * 0.15 # 15% Opacity digital dust

    clip = VideoClip(make_frame, duration=duration)
    mask = VideoClip(make_mask, is_mask=True, duration=duration)
    return clip.with_mask(mask)

# ── LAYER 1: Dynamic Minimalist Background ────────────────────────────────────
def _dynamic_tech_background(duration, accent_color):
    """Generates a high-end, 0-cost local dark tech background (Obsidian style)."""
    # Create base grid and particles
    cols, rows = 12, 22
    spacing_x = FRAME_W // cols
    spacing_y = FRAME_H // rows
    
    def make_frame(t):
        # Dark obsidian base
        frame = np.full((FRAME_H, FRAME_W, 3), (10, 10, 15), dtype=np.uint8)
        
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

# ── PROFILE PICTURE LAYER ─────────────────────────────────────────────────────
def _dynamic_avatar_clip(duration, audio_path, accent_color):
    """
    Attempts to generate an avatar video using Wav2Lip if the directory exists.
    If it fails or is absent, falls back to an audio visualizer centered 
    around the avatar image.
    """
    import numpy as np
    from pydub import AudioSegment
    import math
    import subprocess
    # Primary preference: Video (for motion) -> YouTube Pic -> Original Profile
    avatar_face_path = os.path.join(ASSETS_DIR, "Firefly_video_2.mp4")
    if not os.path.exists(avatar_face_path):
        avatar_face_path = os.path.join(ASSETS_DIR, "youtube_pic.png")
    if not os.path.exists(avatar_face_path):
        avatar_face_path = os.path.join(ASSETS_DIR, "vj_profile.jpg")
        
    output_temp_avatar = os.path.join(OUTPUT_DIR, "temp_avatar.mp4")
    
    if not os.path.exists(avatar_face_path):
        print(f"No avatar source (video or image) found in assets. Skipping avatar generation.")
        return None

    success = False
    wav2lip_dir = os.path.join(BASE_DIR, "Wav2Lip")

    # 1. Try Wav2Lip (Primary AI Engine)
    if os.path.exists(wav2lip_dir):
        print(f"Wav2Lip detected at {wav2lip_dir}. Attempting avatar generation...")
        
        # Build a CLEAN environment for the Wav2Lip subprocess.
        # The 'Fatal Python error: config_init_hash_seed' crash happens when
        # PYTHONHASHSEED is missing, empty, or has a non-integer value.
        # We ALWAYS force it to '0' to guarantee a valid state.
        w2l_env = os.environ.copy()
        w2l_env["PYTHONHASHSEED"] = "0"
        print(f"Wav2Lip env PYTHONHASHSEED forced to: '{w2l_env.get('PYTHONHASHSEED')}'")

        cmd = [
            sys.executable, "inference.py",
            "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
            "--face", avatar_face_path,
            "--audio", audio_path,
            "--outfile", output_temp_avatar,
            "--pads", "0", "20", "0", "0",
            "--face_det_batch_size", "2",
            "--wav2lip_batch_size", "16"
        ]
        print(f"Wav2Lip CMD: {' '.join(str(c) for c in cmd)}")
        try:
            result = subprocess.run(cmd, cwd=wav2lip_dir, capture_output=True, text=True, env=w2l_env, timeout=1800)
            print(f"Wav2Lip STDOUT: {result.stdout[-500:] if result.stdout else '(empty)'}")
            print(f"Wav2Lip STDERR: {result.stderr[-500:] if result.stderr else '(empty)'}")
            if result.returncode != 0:
                raise Exception(f"Wav2Lip process returned {result.returncode}")
                
            if os.path.exists(output_temp_avatar):
                success = True
                print(f"Wav2Lip output file created: {output_temp_avatar}")
        except subprocess.TimeoutExpired:
            print("Wav2Lip generation timed out after 600s.")
        except Exception as e:
            print(f"Wav2Lip generation failed: {e}")

    # 3. Handle successful deep learning avatar
    if success and os.path.exists(output_temp_avatar):
        print(f"Avatar successfully generated via AI. Path: {output_temp_avatar}")
        try:
            av_clip = VideoFileClip(output_temp_avatar).subclipped(0, duration)
            
            # Upscale if the output is too small for modern displays (e.g. 96x96 -> 500x500)
            target_h = 500
            if av_clip.h < target_h:
                print(f"Upscaling generated avatar from {av_clip.h}px to {target_h}px for clarity.")
                av_clip = av_clip.resized(height=target_h)

            vw, vh = av_clip.size
            size = min(vw, vh)
            
            # Create a High-Quality Soft Mask (Feathered Edge)
            mask_img = np.zeros((vh, vw), dtype=np.uint8)
            cv2.circle(mask_img, (vw//2, vh//2), size//2 - 2, 255, -1)
            # Apply Gaussian Blur to the binary mask for soft alpha transitions (premium look)
            mask_img = cv2.GaussianBlur(mask_img, (15, 15), 0)
            
            mask_clip = VideoClip(lambda t: mask_img.astype(float) / 255.0, is_mask=True, duration=duration)
            av_clip = av_clip.with_mask(mask_clip)

            # Create a Glowing Border Ring behind the avatar
            border_size = size + 20
            def make_border_frame(t):
                # Extra small pulse to make the avatar feel alive
                pulse = 1.0 + 0.02 * math.sin(t * 2.0)
                cur_r = int((border_size // 2) * pulse)
                
                img = np.zeros((border_size + 40, border_size + 40, 3), dtype=np.uint8)
                center = (img.shape[1]//2, img.shape[0]//2)
                # Outer glow
                cv2.circle(img, center, cur_r, accent_color, 4)
                img = cv2.GaussianBlur(img, (11, 11), 0)
                return img

            border_clip = VideoClip(make_border_frame, duration=duration).with_position(("center", 800))
            
            # Position avatar slightly above center (800px from top) to avoid overlapping with captions at 1200px
            return CompositeVideoClip([border_clip, av_clip.with_position(("center", 800))], size=(FRAME_W, FRAME_H))
        except Exception as e:
            print(f"Failed to load generated avatar video: {e}")
            success = False # Proceed to fallback if loading fails
    
    # 4. Fallback: Audio Visualizer
    print("Falling back to Auto-Visualizer for Avatar...")
    try:
        audio_seg = AudioSegment.from_file(audio_path).set_channels(1).set_frame_rate(20)
        samples = np.array(audio_seg.get_array_of_samples())
        samples = np.abs(samples)
        max_amp = np.max(samples) if np.max(samples) > 0 else 1
        samples = samples / max_amp
    except Exception as e:
        print(f"Audio for visualizer failed: {e}")
        samples = np.zeros(int(duration * 20))
        
    try:
        img = Image.open(avatar_img_path).convert("RGBA")
        size = 300
        img = img.resize((size, size), Image.LANCZOS)
        
        # Circular mask
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        img.putalpha(mask)
        
        def make_frame(t):
            frame_bg = Image.new("RGBA", (500, 500), (0,0,0,0))
            d_circle = ImageDraw.Draw(frame_bg)
            
            idx = int(t * 20)
            if idx >= len(samples):
                idx = len(samples) - 1
            amp = samples[idx] if idx >= 0 else 0
            
            # Draw visualizer ring
            radius = (size // 2) + 15 + int(60 * amp)
            center = (250, 250)
            
            # Glow effect layer
            d_circle.ellipse((center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius), 
                             outline=(*accent_color, int(150 + 105 * amp)), width=12)
                             
            # Paste avatar
            frame_bg.paste(img, (100, 100), img)
            
            return np.array(frame_bg)
            
        return VideoClip(make_frame, duration=duration).with_position(("center", 800))
    except Exception as e:
        print(f"Visualizer fallback failed: {e}")
        return None



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
    f = gf(90)
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
        ty = int(FRAME_H//2 + 300 * math.sin(angle))
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
def _telegram_cta(accent_color, total_dur):
    cta_dur = min(6.0, total_dur * 0.18)
    start   = total_dur - cta_dur
    card_h  = int(FRAME_H * 0.38)
    dest_y  = FRAME_H - card_h

    img = Image.new("RGBA", (FRAME_W, card_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    # Dark gradient fill
    for y in range(card_h):
        a = int(217 * (y / card_h) ** 0.5)
        draw.line([(0,y),(FRAME_W,y)], fill=(0,0,0,a))
    # Top accent line
    draw.rectangle([0,0,FRAME_W,4], fill=(*accent_color, 255))
    
    f1, f2, f3 = gf(36), gf(44), gf(26)

    l1 = "📲 Join for daily tech intel"
    l2_t = "t.me/technewsbyvj"
    l2_l = "linkedin.com/in/vijayakumar-j/"
    l3 = "Free  •  Daily  •  Exclusive"
    
    # Render line 1
    w1, h1 = ts(l1, f1)
    draw.text(((FRAME_W-w1)//2, 30), l1, font=f1, fill=(255,255,255,255))
    
    # Load and Render Icons for line 2
    try:
        tg_icon = Image.open(os.path.join(ASSETS_DIR, "icons", "telegram_logo.png")).convert("RGBA").resize((60, 60), Image.LANCZOS)
        li_icon = Image.open(os.path.join(ASSETS_DIR, "icons", "linkedin_logo.png")).convert("RGBA").resize((60, 60), Image.LANCZOS)
        
        # Telegram Row
        tw_t, th_t = ts(l2_t, f2)
        tx_t = (FRAME_W - (tw_t + 70)) // 2
        img.paste(tg_icon, (tx_t, 110), tg_icon)
        draw.text((tx_t + 75, 110), l2_t, font=f2, fill=(*accent_color, 255))
        
        # LinkedIn Row
        tw_l, th_l = ts(l2_l, f2)
        tx_l = (FRAME_W - (tw_l + 70)) // 2
        img.paste(li_icon, (tx_l, 195), li_icon)
        draw.text((tx_l + 75, 195), l2_l, font=f2, fill=(*accent_color, 255))
        
        # Line 3 (Shifted down)
        w3, h3 = ts(l3, f3)
        draw.text(((FRAME_W-w3)//2, 285), l3, font=f3, fill=(180,180,180,255))
    except Exception as e:
        print(f"Icon render failed: {e}")
        # Simple text fallback if icons missing
        l2 = f"{l2_t} | {l2_l}"
        w, h = ts(l2, f2)
        draw.text(((FRAME_W-w)//2, 120), l2, font=f2, fill=(*accent_color, 255))
        w3, h3 = ts(l3, f3)
        draw.text(((FRAME_W-w3)//2, 210), l3, font=f3, fill=(180,180,180,255))

    arr  = np.array(img.convert("RGB"))
    mask = np.array(img.split()[3]).astype(float) / 255.0

    def pos(t):
        slide = min(t / 0.4, 1.0)
        slide = slide * slide * (3 - 2 * slide)
        y = int(FRAME_H - card_h * slide)
        return (0, y)

    clip = VideoClip(lambda t: arr, duration=cta_dur)
    mclip = VideoClip(lambda t: mask, is_mask=True, duration=cta_dur)
    return clip.with_mask(mclip).with_position(pos).with_start(start)


# ── LAYER 13: Subscribe animation ─────────────────────────────────────────────
def _subscribe_clip(total_dur):
    dur   = min(3.0, total_dur * 0.1)
    start = total_dur - dur
    f = gf(44)
    text = "SUBSCRIBE 🔔"
    w, h = ts(text, f)
    pad_x, pad_y = 30, 16
    img = Image.new("RGBA", (w + pad_x*2, h + pad_y*2), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0,0,w+pad_x*2,h+pad_y*2], radius=30, fill=(220,0,0,230))
    draw.text((pad_x, pad_y), text, font=f, fill=(255,255,255,255))

    arr  = np.array(img.convert("RGB"))
    mask = np.array(img.split()[3]).astype(float) / 255.0
    bpos = int(FRAME_H - TITLE_BOTTOM_GAP - (h+pad_y*2) - 230)

    def scale_fn(t):
        return 1.0 + 0.06 * math.sin(t * 10)

    def make_frame(t):
        s = scale_fn(t)
        scaled = Image.fromarray(arr).resize((int(arr.shape[1]*s), int(arr.shape[0]*s)), Image.LANCZOS)
        return np.array(scaled)

    def make_mask(t):
        s = scale_fn(t)
        orig_mask = Image.fromarray((mask * 255).astype(np.uint8))
        scaled = orig_mask.resize((int(orig_mask.width*s), int(orig_mask.height*s)), Image.LANCZOS)
        return np.array(scaled).astype(float) / 255.0

    clip = VideoClip(make_frame, duration=dur)
    mclip = VideoClip(make_mask, is_mask=True, duration=dur)
    clip = clip.with_mask(mclip)
    clip = clip.with_position(lambda t: ("center", bpos))
    return clip.with_start(start)


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
    header_h = 180
    img = Image.new('RGBA', (frame_width, header_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    # Solid dark strip behind title for contrast
    draw.rectangle([0, 0, frame_width, header_h], fill=(0, 0, 0, 180))
    
    # Main Title – positioned higher for better visibility
    f_title = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', 44)
    tw = draw.textlength(title, font=f_title)
    if tw > 800: title = title[:40] + "..."
    tw = draw.textlength(title, font=f_title)
    draw.text(((frame_width-tw)//2, 80), title, font=f_title, fill=(255,255,255,255))
    
    return img

def render_telegram_cta(accent_color, frame_width=1080):
    """Minimalist Telegram, LinkedIn & WhatsApp CTA banner."""
    card_h = 420
    img = Image.new("RGBA", (frame_width, card_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    # Translucent dark base
    draw.rectangle([0,0,frame_width,card_h], fill=(10,10,15,230))
    draw.line([(0,0),(frame_width,0)], fill=(*accent_color,255), width=4)
    
    f1 = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', 48)
    t1 = "t.me/technewsbyvj"
    t2 = "linkedin.com/in/vijayakumar-j/"
    t3 = "WhatsApp: t.ly/vj-wa" # Shorter display link
    
    try:
        tg_path = os.path.join(ASSETS_DIR, "icons", "telegram_logo.png")
        li_path = os.path.join(ASSETS_DIR, "icons", "linkedin_logo.png")
        wa_path = os.path.join(ASSETS_DIR, "icons", "whatsapp_logo.png")
        
        # Robust icon loading
        def load_icon(path, size=(60,60)):
            if os.path.exists(path):
                return Image.open(path).convert("RGBA").resize(size, Image.LANCZOS)
            return None

        tg_icon = load_icon(tg_path)
        li_icon = load_icon(li_path)
        wa_icon = load_icon(wa_path)
        
        # TG Row
        w1 = draw.textlength(t1, font=f1)
        x1 = (frame_width - (w1 + 80)) // 2
        if tg_icon: img.paste(tg_icon, (int(x1), 60), tg_icon)
        draw.text((x1 + 80, 65), t1, font=f1, fill=(*accent_color, 255))
        
        # LI Row
        w2 = draw.textlength(t2, font=f1)
        x2 = (frame_width - (w2 + 80)) // 2
        if li_icon: img.paste(li_icon, (int(x2), 160), li_icon)
        draw.text((x2 + 80, 165), t2, font=f1, fill=(*accent_color, 255))

        # WA Row
        w3 = draw.textlength(t3, font=f1)
        x3 = (frame_width - (w3 + 80)) // 2
        if wa_icon: img.paste(wa_icon, (int(x3), 260), wa_icon)
        draw.text((x3 + 80, 265), t3, font=f1, fill=(*accent_color, 255))

    except Exception as e:
        print(f"CTA Render Icons Error: {e}")
        x1 = (frame_width - draw.textlength(t1, font=f1))//2
        draw.text((x1, 75), t1, font=f1, fill=(*accent_color,255))
    
    return img

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
    cta_img = render_telegram_cta(accent_color, FRAME_W)
    cta_clip = ImageClip(np.array(cta_img)).with_duration(duration).with_position(("center", "center"))
    return CompositeVideoClip([bg, cta_clip], size=(FRAME_W, FRAME_H)).with_duration(duration)

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

def composite_frame(background_frame, timestamp, header_img, subtitle_img, cta_img, video_duration, cues=None):
    """Final assembly of minimalist layers with Pattern Interrupts."""
    # Apply visual effects to the base background/avatar frame first
    if cues:
        background_frame = apply_pattern_interrupts(background_frame, timestamp, cues)
        
    frame = Image.fromarray(background_frame).convert('RGBA')
    
    # 1. Header is always at top
    frame.alpha_composite(header_img, dest=(0,0))
    
    # 2. Subtitles on top
    if subtitle_img is not None:
        frame.alpha_composite(subtitle_img, dest=(0,0))
        
    # 3. CTA slides up at the end
    if timestamp >= video_duration - 6:
        progress = min((timestamp-(video_duration-6))/0.4, 1.0)
        # Power easing
        progress = 1-(1-progress)**3
        cta_y = int(FRAME_H - (240 * progress))
        frame.alpha_composite(cta_img, dest=(0, cta_y))
        
    return np.array(frame.convert('RGB'))

def verify_text_visibility(frame_array, zone_name, y_start, y_end):
    """Validates that text is readable (high contrast)."""
    region = frame_array[y_start:y_end, 50:1030]
    # Check for presence of bright pixels (text) and dark pixels (obsidian bg)
    bright = np.sum(region > 200)
    dark = np.sum(region < 40)
    
    if bright < 500:
        print(f"⚠️ {zone_name}: WARNING - Low text density detected.")
    if dark < 1000:
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

def render_subtitle_frame(text, current_words, bg_frame=None, accent_color=(255,214,0), frame_width=1080, frame_height=1920):
    """Modern Hormozi-style minimalist captions."""
    img = Image.new('RGBA', (frame_width, frame_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    f_main = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', 78)
    f_pop = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', 88)
    
    words = text.split()
    current_word_list = current_words.get('current', [])
    
    word_widths = []
    fake_draw = ImageDraw.Draw(Image.new("RGBA", (1,1)))
    for word in words:
        f = f_pop if word in current_word_list else f_main
        bbox = fake_draw.textbbox((0,0), word, font=f)
        word_widths.append(bbox[2]-bbox[0])
    
    lines = wrap_text_to_lines(words, word_widths, 900, f_main)
    
    line_h = 110
    total_h = len(lines) * line_h
    start_y = 1200 # Higher up to avoid avatar
    
    word_idx = 0
    for i, line in enumerate(lines):
        line_y = start_y + i * line_h
        line_w = sum(word_widths[word_idx:word_idx+len(line)]) + 12 * (len(line)-1)
        cur_x = (frame_width - line_w) // 2
        
        for word in line:
            is_active = word in current_word_list
            f = f_pop if is_active else f_main
            
            # Colors
            if is_active:
                color = (255, 255, 0, 255)
            elif word in current_words.get('spoken', []):
                color = (180, 180, 180, 255)
            else:
                color = (255, 255, 255, 255)
                
            # Heavy stroke
            s_w = 8 if is_active else 5
            for dx in range(-s_w, s_w+1, 2):
                for dy in range(-s_w, s_w+1, 2):
                    draw.text((cur_x+dx, line_y+dy), word, font=f, fill=(0,0,0,255))
            
            draw.text((cur_x, line_y), word, font=f, fill=color)
            cur_x += word_widths[word_idx] + 12
            word_idx += 1
            
    return img

def create_video(audio_path, script_json, chunks, output_path=None):
    global _kb_idx
    today = datetime.now().strftime("%Y-%m-%d")
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, f"video_{today}.mp4")

    audio          = AudioFileClip(audio_path)
    audio_duration = audio.duration

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
    # ── Hook text removed ──────────────────────────────────────────────────────
    key_stat       = script_json.get("key_stat", "")
    key_stat_ts    = float(script_json.get("key_stat_timestamp", 0))
    shock_ts       = float(script_json.get("shocking_moment_timestamp", 0))

    # ── LAYER 1: Dynamic Background ───────────────────────────────────────────
    print("Building dynamic minimalist background...")
    base = _dynamic_tech_background(audio_duration, accent_color)

    # ── LAYER 2: Avatar & Identity ────────────────────────────────────────────
    avatar = _dynamic_avatar_clip(audio_duration, audio_path, accent_color)
    
    # ── LAYER 3: Tint ─────────────────────────────────────────────────────────
    tint = ColorClip(size=(FRAME_W, FRAME_H), color=accent_color, duration=audio_duration).with_opacity(0.02)

    # ── LAYER 4: Gradient ─────────────────────────────────────────────────────
    gradient = _gradient_clip(audio_duration)

    # ── LAYER 5: Particles (Subtle) ───────────────────────────────────────────
    particle_clips = [_ambient_particles(audio_duration, accent_color)]

    # ── Hook banner removed ──────────────────────────────────────────────────

    # ── LAYER 7: Animated logo (Removed VJ Branding) ──────────────────────────
    logo_clips = []

    # ── LAYER 8: Fact highlight - REMOVED for minimalist style ────────────────
    fact_clips = []

    # ── LAYER 9: Emoji burst - REMOVED for minimalist style ───────────────────
    burst_clips = []

    # ── LAYER 10: Like reminder - REMOVED for minimalist style ────────────────
    reminder_clips = []

    # ── LAYER 11: Main Composite Base ─────────────────────────────────────────
    base_layers = [base, tint, gradient] + particle_clips + logo_clips + fact_clips + burst_clips + reminder_clips
    if avatar:
        base_layers.append(avatar)
    
    progress = ColorClip(size=(FRAME_W, 6), color=accent_color, duration=audio_duration)
    progress = progress.with_position(lambda t: (int((t / max(audio_duration, 0.01)) * FRAME_W) - FRAME_W, FRAME_H - 6))
    base_layers.append(progress)
    
    base_comp = CompositeVideoClip(base_layers, size=(FRAME_W, FRAME_H)).with_duration(audio_duration)

    # Pre-render custom composite assets
    header_img = render_header_bar(title, sub_category, accent_color, FRAME_W)
    cta_img = render_telegram_cta(accent_color, FRAME_W)

    def make_final_frame(t):
        bg_frame = base_comp.get_frame(t)
        
        # Subtitle logic map for the exact current timestamp
        subtitle_img = None
        for chunk in chunks:
            if chunk["start"] - 0.1 <= t <= chunk["end"] + 0.1:
                curr_wds = {"current": [], "spoken": []}
                for w in chunk.get("words", []):
                    if w["start"] - 0.1 <= t <= w["end"] + 0.05:
                        curr_wds["current"].append(w["word"])
                    elif t > w["end"]:
                        curr_wds["spoken"].append(w["word"])
                ctext = " ".join([w["word"] for w in chunk.get("words", [])])
                if ctext:
                    subtitle_img = render_subtitle_frame(
                        ctext, curr_wds, bg_frame=bg_frame, 
                        accent_color=accent_color, frame_width=FRAME_W, frame_height=FRAME_H
                    )
                break
                
        return composite_frame(bg_frame, t, header_img, subtitle_img, cta_img, audio_duration, cues=script_json.get("retention_cues", []))

    final = VideoClip(make_final_frame, duration=audio_duration)
    final = final.with_audio(audio)

    # ── INTRO / OUTRO ────────────────────────────────────────────────────────
    # 2026 Strategy: Infinite Loops need SEAMLESS transition (no intro/outro)
    is_infinite_loop = script_json.get("loop_score", 0) >= 8
    
    if is_infinite_loop:
        print("INFINITE LOOP DETECTED: Skipping intro/outro for seamless scaling.")
        final_video = final
    else:
        intro = _intro_clip(2.0, accent_color)
        outro = _outro_clip(3.0, accent_color)
        final_video = concatenate_videoclips([intro, final, outro])
    
    final_duration = final_video.duration

    # ── LAYER 15: BGM ─────────────────────────────────────────────────────────
    music_files = [f for f in os.listdir(MUSIC_DIR) if f.endswith(".mp3")]
    if music_files:
        bg_music = random.choice(music_files)
        bgm = AudioFileClip(os.path.join(MUSIC_DIR, bg_music)).with_volume_scaled(BGM_VOLUME)
        if bgm.duration < final_duration:
            bgm = bgm.with_effects([afx.AudioLoop(duration=final_duration)])
        else:
            bgm = bgm.subclipped(0, final_duration)
            
        # If it's a loop, we might want to fade the BGM or keep it constant
        if is_infinite_loop:
            # Subtle volume dip at the very end to avoid pop
            bgm = bgm.with_effects([afx.AudioFadeOut(0.1)])
            
        final_video = final_video.with_audio(CompositeAudioClip([final_video.audio, bgm]))

    final = final_video

    print(f"Exporting {final_duration:.1f}s → {output_path}")
    
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
                verify_text_visibility(test_frame, f"SUBTITLE {p}", 1250, 1480)
                verify_text_visibility(test_frame, f"HEADER {p}", 0, 200)
                
                test_path = output_path.replace(".mp4", f"_test_{int(p*100)}pct.jpg")
                img.save(test_path)
    except Exception as e:
        print(f"Visibility frames failed: {e}")

    final.write_videofile(
        output_path, fps=30, codec="libx264", audio_codec="aac",
        threads=4, preset="ultrafast", ffmpeg_params=["-pix_fmt", "yuv420p"]
    )
    return output_path
