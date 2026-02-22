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
import io
import math
import random
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datetime import datetime
from moviepy import (
    VideoClip, ImageClip, VideoFileClip, AudioFileClip,
    CompositeVideoClip, ColorClip, CompositeAudioClip,
)
import moviepy.video.fx as vfx
import moviepy.audio.fx as afx
from config import OUTPUT_DIR, ASSETS_DIR, MUSIC_DIR, BGM_VOLUME

FRAME_W, FRAME_H = 1080, 1920
TITLE_BOTTOM_GAP = 192

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
KB_PATTERNS = ["zoom_in", "zoom_out", "pan_right"]

def _ease(t):
    return t * t * (3 - 2 * t)

def build_ken_burns(img_path, duration, pattern_idx):
    pattern = KB_PATTERNS[pattern_idx % len(KB_PATTERNS)]
    try:
        img = Image.open(img_path).convert("RGB").resize((FRAME_W, FRAME_H), Image.LANCZOS)
    except Exception:
        return ColorClip(size=(FRAME_W, FRAME_H), color=(15, 15, 25), duration=duration)
    pad  = int(FRAME_W * 0.15)
    pw, ph = FRAME_W + pad*2, FRAME_H + pad*2
    padded = np.array(Image.fromarray(np.array(img)).resize((pw, ph), Image.LANCZOS))

    def make_frame(t):
        p = _ease(min(t / max(duration, 0.01), 1.0))
        if pattern == "zoom_in":
            scale = 1.0 + 0.12 * p; cx, cy = pw//2, ph//2
        elif pattern == "zoom_out":
            scale = 1.12 - 0.12 * p; cx, cy = pw//2, ph//2
        else:
            scale = 1.08; cx = int(pw*(0.45+0.10*p)); cy = ph//2
        cw, ch = int(FRAME_W/scale), int(FRAME_H/scale)
        x1 = max(0, min(cx-cw//2, pw-cw)); y1 = max(0, min(cy-ch//2, ph-ch))
        crop = padded[y1:y1+ch, x1:x1+cw]
        out = np.array(Image.fromarray(crop).resize((FRAME_W, FRAME_H), Image.BICUBIC), dtype=np.float32)
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


# ── LAYER 4: Ambient particles ────────────────────────────────────────────────
def _ambient_particles(duration, accent_color):
    n = 25
    random.seed(42)
    particles = [(random.uniform(0, FRAME_W), random.uniform(0, FRAME_H),
                  random.uniform(0.3, 1.2), random.uniform(0, FRAME_H))
                 for _ in range(n)]

    def make_frame(t):
        img = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for px, py, speed, offset in particles:
            y = (py - speed * t * 60 - offset) % FRAME_H
            draw.ellipse([px-3, y-3, px+3, y+3], fill=(*accent_color, 38))
        return np.array(img.convert("RGB"))

    def make_mask(t):
        img = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for px, py, speed, offset in particles:
            y = (py - speed * t * 60 - offset) % FRAME_H
            draw.ellipse([px-3, y-3, px+3, y+3], fill=(0, 0, 0, 38))
        return np.array(img.split()[3]).astype(float) / 255.0

    clip = VideoClip(make_frame, duration=duration)
    mask = VideoClip(make_mask, is_mask=True, duration=duration)
    return clip.with_mask(mask)


# ── LAYER 5: Hook banner ──────────────────────────────────────────────────────
def _hook_banner(hook_text, accent_color, total_dur):
    banner_h, dur = 120, 2.2

    def _render():
        img = Image.new("RGBA", (FRAME_W, banner_h), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0,0,FRAME_W,banner_h], fill=(*accent_color, 204))
        f = gf(52)
        w, h = ts(hook_text[:60], f)
        draw.text(((FRAME_W-w)//2, (banner_h-h)//2), hook_text[:60], font=f, fill=(255,255,255,255))
        return img

    img_arr  = np.array(_render().convert("RGB"))
    mask_arr = np.array(_render().split()[3]).astype(float) / 255.0

    def banner_y(t):
        if t < 0.25:
            return int(-banner_h + (banner_h + 60) * (t/0.25))
        elif t < 1.7:
            return 60
        else:
            return int(60 - (banner_h + 60) * ((t - 1.7) / 0.5))

    clip = VideoClip(lambda t: img_arr, duration=dur)
    mask = VideoClip(lambda t: mask_arr, is_mask=True, duration=dur)
    clip = clip.with_mask(mask)
    clip = clip.with_position(lambda t: (0, banner_y(t)))
    return clip


# ── LAYER 6: Animated logo ─────────────────────────────────────────────────────
def _animated_logo(duration):
    logo_path = os.path.join(ASSETS_DIR, "logo.png")
    if not os.path.exists(logo_path):
        return None
    try:
        logo_img = Image.open(logo_path).convert("RGBA").resize((110, 110), Image.LANCZOS)
        arr  = np.array(logo_img.convert("RGB"))
        mask = np.array(logo_img.split()[3]).astype(float) / 255.0

        def pos(t):
            if t < 0.6:
                scale = min(t / 0.5, 1.0)
                x = int((FRAME_W - 110*scale) // 2)
                y = int(FRAME_H * 0.08)
                return (x, y)
            else:
                return (FRAME_W - 130, 75)

        def opacity_fn(t):
            return 1.0 if t < 0.6 else 0.28

        clip = VideoClip(lambda t: arr, duration=duration)
        mclip = VideoClip(lambda t: mask, is_mask=True, duration=duration)
        clip = clip.with_mask(mclip)
        clip = clip.with_position(pos)
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
    l2 = "t.me/technewsbyvj"
    l3 = "Free  •  Daily  •  Exclusive"
    for i, (txt, font, col) in enumerate([(l1,f1,(255,255,255,255)),(l2,f2,(*accent_color,255)),(l3,f3,(180,180,180,255))]):
        w, h = ts(txt, font)
        draw.text(((FRAME_W-w)//2, 30 + i*90), txt, font=font, fill=col)

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


# ── MAIN ──────────────────────────────────────────────────────────────────────
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
    hook_text      = script_json.get("hook_banner_text", script_json.get("hook", ""))
    key_stat       = script_json.get("key_stat", "")
    key_stat_ts    = float(script_json.get("key_stat_timestamp", 0))
    shock_ts       = float(script_json.get("shocking_moment_timestamp", 0))

    # ── LAYER 1: Background chunks ────────────────────────────────────────────
    print("Building background chunk clips...")
    chunk_clips = []
    for idx, chunk in enumerate(chunks):
        dur   = chunk["duration"]
        vtype = chunk.get("visual_type", "photo")
        vpath = chunk.get("visual_path")
        bg = None
        if vpath and os.path.exists(vpath):
            if vtype == "video":
                bg = build_video_clip(vpath, dur)
            else:
                bg = build_ken_burns(vpath, dur, _kb_idx); _kb_idx += 1
        if bg is None:
            bg = ColorClip(size=(FRAME_W, FRAME_H), color=(15, 15, 25), duration=dur)
        if idx > 0 and dur >= 3.0:
            bg = bg.with_effects([vfx.CrossFadeIn(0.25 if dur < 5 else 0.40)])
        chunk_clips.append(bg.with_start(chunk["start"]).with_duration(dur))

    base = CompositeVideoClip(chunk_clips, size=(FRAME_W, FRAME_H)).with_duration(audio_duration)

    # ── LAYER 2: Tint ─────────────────────────────────────────────────────────
    tint = ColorClip(size=(FRAME_W, FRAME_H), color=accent_color, duration=audio_duration).with_opacity(0.04)

    # ── LAYER 3: Gradient ─────────────────────────────────────────────────────
    gradient = _gradient_clip(audio_duration)

    # ── LAYER 4: Particles ────────────────────────────────────────────────────
    particle_clips = []
    if sub_category in ("AI", "Space", "Technology", "Cybersecurity", "Robotics"):
        particle_clips.append(_ambient_particles(audio_duration, accent_color))

    # ── LAYER 5: Hook banner ──────────────────────────────────────────────────
    hook_clips = []
    if hook_text:
        hook_clips.append(_hook_banner(hook_text, accent_color, audio_duration))

    # ── LAYER 6: Animated logo ────────────────────────────────────────────────
    logo_clips = []
    logo = _animated_logo(audio_duration)
    if logo:
        logo_clips.append(logo)

    # ── LAYER 7: Fact highlight ───────────────────────────────────────────────
    fact_clips = []
    if key_stat and key_stat_ts < audio_duration:
        fb = _fact_box(key_stat, key_stat_ts, accent_color, audio_duration)
        if fb:
            fact_clips.append(fb)

    # ── LAYER 8: Emoji burst ──────────────────────────────────────────────────
    burst_clips = _emoji_burst(shock_ts, audio_duration)

    # ── LAYER 9: Like reminder (50%) ──────────────────────────────────────────
    reminder_clips = []
    like_t = audio_duration * 0.50
    lr = _pill_reminder("👍 Tap Like if this surprised you!", like_t, audio_duration)
    if lr:
        reminder_clips.append(lr)

    # ── LAYER 10: Share prompt (80%) ──────────────────────────────────────────
    share_t = audio_duration * 0.80
    sr = _pill_reminder("📤 Share this with someone who needs to know!", share_t, audio_duration)
    if sr:
        reminder_clips.append(sr)

    # ── LAYER 11: Title ───────────────────────────────────────────────────────
    title_clip = _title_clip(title, audio_duration)

    # ── LAYER 12: Telegram CTA ────────────────────────────────────────────────
    cta_clip = _telegram_cta(accent_color, audio_duration)

    # ── LAYER 13: Subscribe ───────────────────────────────────────────────────
    sub_clip = _subscribe_clip(audio_duration)

    # ── LAYER 14: Progress bar ────────────────────────────────────────────────
    progress = ColorClip(size=(FRAME_W, 6), color=accent_color, duration=audio_duration)
    progress = progress.with_position(
        lambda t: (int((t / max(audio_duration, 0.01)) * FRAME_W) - FRAME_W, FRAME_H - 6)
    )

    # ── COMPOSITE ─────────────────────────────────────────────────────────────
    layers = (
        [base, tint, gradient]
        + particle_clips
        + hook_clips
        + logo_clips
        + fact_clips
        + burst_clips
        + reminder_clips
        + [title_clip, cta_clip, sub_clip, progress]
    )
    final = CompositeVideoClip(layers, size=(FRAME_W, FRAME_H))

    # ── LAYER 15: BGM ─────────────────────────────────────────────────────────
    music_files = [f for f in os.listdir(MUSIC_DIR) if f.endswith(".mp3")]
    if music_files:
        bgm = AudioFileClip(os.path.join(MUSIC_DIR, random.choice(music_files))).with_volume_scaled(BGM_VOLUME)
        if bgm.duration < audio_duration:
            bgm = bgm.with_effects([afx.AudioLoop(duration=audio_duration)])
        else:
            bgm = bgm.subclipped(0, audio_duration)
        final = final.with_audio(CompositeAudioClip([audio, bgm]))
    else:
        final = final.with_audio(audio)

    print(f"Exporting {audio_duration:.1f}s → {output_path}")
    final.write_videofile(
        output_path, fps=30, codec="libx264", audio_codec="aac",
        threads=4, preset="ultrafast", ffmpeg_params=["-pix_fmt", "yuv420p"]
    )
    return output_path
