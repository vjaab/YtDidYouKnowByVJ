"""
thumbnail_gen.py — High-Authority Split-Screen Thumbnail Generator.

Design Philosophy (Inspired by premium creator thumbnails):
  - Split-screen: LEFT = Bold curiosity-gap hook text on dark bg
  - RIGHT = High-quality topic-relevant image
  - Zero clutter: No emojis, arrows, or busy graphics
  - Premium typography: Montserrat ExtraBold, high contrast
  - Curiosity gap: Headline ends with "..." to force clicks
  
Generates:
  - 1280x720 (16:9) YouTube thumbnail
  - 1080x1920 (9:16) Shorts thumbnail
"""

import os
import io
import math
import textwrap
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from datetime import datetime
from google import genai
from google.genai import types
from config import OUTPUT_DIR, ASSETS_DIR, GEMINI_API_KEY

THUMB_W, THUMB_H = 1280, 720
SHORTS_W, SHORTS_H = 1080, 1920

# ─────────────────────────────────────────────────────────────────────────────
# FONT LOADING
# ─────────────────────────────────────────────────────────────────────────────
FONT_EXTRABOLD = os.path.join(ASSETS_DIR, "fonts", "Montserrat-ExtraBold.ttf")
FONT_BOLD = os.path.join(ASSETS_DIR, "fonts", "Montserrat-Bold.ttf")
FONT_BLACK = os.path.join(ASSETS_DIR, "fonts", "Montserrat-Black.ttf")

FALLBACKS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/roboto/hinted/Roboto-Bold.ttf",
    "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf",
]

_fcache = {}

def _load_font(size, weight="extrabold"):
    """Load font with fallback chain."""
    key = (size, weight)
    if key not in _fcache:
        paths = {
            "extrabold": [FONT_EXTRABOLD, FONT_BLACK, FONT_BOLD],
            "bold": [FONT_BOLD, FONT_EXTRABOLD],
            "black": [FONT_BLACK, FONT_EXTRABOLD, FONT_BOLD],
        }
        candidates = paths.get(weight, [FONT_BOLD]) + FALLBACKS
        for p in candidates:
            if os.path.exists(p):
                try:
                    _fcache[key] = ImageFont.truetype(p, size)
                    break
                except Exception:
                    continue
        if key not in _fcache:
            _fcache[key] = ImageFont.load_default()
    return _fcache[key]


def _text_size(text, font):
    bb = font.getbbox(text)
    return bb[2] - bb[0], bb[3] - bb[1]


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI: GENERATE CURIOSITY-GAP HOOK
# ─────────────────────────────────────────────────────────────────────────────
def _generate_hook_text(title):
    """Use Gemini to generate a short, punchy curiosity-gap hook for the thumbnail."""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""You are a YouTube thumbnail copywriter. Generate a SHORT, punchy hook for this video topic.

TOPIC: "{title}"

RULES:
1. Maximum 8 words total
2. Use a "curiosity gap" — make the viewer NEED to click to find out the rest
3. Use simple, emotional language (not technical jargon)
4. End with "..." to create incompleteness
5. Use line breaks for visual impact (each phrase on its own line, max 3 lines)
6. Style: "You're losing because..." or "This changes everything..." or "Nobody is talking about..."

EXAMPLES:
- "You're losing\\nbecause the system\\nwas never..."
- "This is why\\nOpenAI is\\nterrified..."
- "Nobody will\\ntell you\\nthis..."
- "The real reason\\nGoogle is\\npanicking..."

Return ONLY the hook text with \\n for line breaks. No quotes, no explanation."""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(temperature=0.8)
        )
        hook = response.text.strip().strip('"').strip("'")
        # Clean up and enforce line break format
        if "\\n" in hook:
            hook = hook.replace("\\n", "\n")
        # Enforce max 3 lines
        lines = hook.split("\n")[:3]
        return "\n".join(lines)
    except Exception as e:
        print(f"  ⚠️ Hook generation failed: {e}")
        # Fallback: use first 6 words of title
        words = title.split()[:6]
        return " ".join(words[:3]) + "\n" + " ".join(words[3:]) + "..."


# ─────────────────────────────────────────────────────────────────────────────
# LOAD TOPIC-RELEVANT IMAGE (right side of split)
# ─────────────────────────────────────────────────────────────────────────────
def _load_topic_image(script_json):
    """Load the best available topic-relevant image for the right panel."""
    today = datetime.now().strftime("%Y-%m-%d")

    # Priority 1: Screenshot (article evidence)
    screenshot = script_json.get("screenshot_path")
    if screenshot and os.path.exists(screenshot):
        try:
            return Image.open(screenshot).convert("RGB")
        except Exception:
            pass

    # Priority 2: Imagen-generated chunk visual
    for i in range(1, 6):
        for ext in ["jpg", "png"]:
            p = os.path.join(OUTPUT_DIR, f"chunk_{i}_{today}.{ext}")
            if os.path.exists(p):
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    continue

    # Priority 3: Lipsync avatar frame
    lipsync = script_json.get("kaggle_lipsync_path")
    if lipsync and os.path.exists(lipsync):
        try:
            from moviepy import VideoFileClip
            clip = VideoFileClip(lipsync)
            frame = clip.get_frame(1.0)  # Get frame at 1 second
            clip.close()
            return Image.fromarray(frame)
        except Exception:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: SPLIT-SCREEN THUMBNAIL (16:9)
# ─────────────────────────────────────────────────────────────────────────────
def _render_split_thumbnail(hook_text, topic_img, accent_color, width=1280, height=720):
    """
    Renders the High-Authority split-screen thumbnail.
    LEFT: Dark background + bold hook text
    RIGHT: Topic-relevant image
    """
    canvas = Image.new("RGB", (width, height), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)

    split_x = int(width * 0.42)  # 42% left, 58% right

    # ── LEFT PANEL: Dark textured background with hook text ──────────────
    # Subtle noise texture on the dark side
    for y in range(height):
        for x in range(0, split_x, 4):
            noise = int(12 + (hash((x, y)) % 8))
            draw.point((x, y), fill=(noise, noise, noise + 2))

    # Subtle accent glow from the right edge of the left panel
    glow_width = 80
    for x in range(split_x - glow_width, split_x):
        progress = (x - (split_x - glow_width)) / glow_width
        alpha = int(25 * progress)
        r = min(255, accent_color[0] + alpha)
        g = min(255, accent_color[1] + alpha)
        b = min(255, accent_color[2] + alpha)
        for y in range(height):
            base = canvas.getpixel((x, y))
            blended = (
                min(255, base[0] + int(r * progress * 0.15)),
                min(255, base[1] + int(g * progress * 0.15)),
                min(255, base[2] + int(b * progress * 0.15)),
            )
            canvas.putpixel((x, y), blended)

    # ── RIGHT PANEL: Topic image ─────────────────────────────────────────
    right_w = width - split_x
    if topic_img:
        img_w, img_h = topic_img.size
        # Crop to fill the right panel
        target_ratio = right_w / height
        img_ratio = img_w / img_h

        if img_ratio > target_ratio:
            new_w = int(img_h * target_ratio)
            left = (img_w - new_w) // 2
            cropped = topic_img.crop((left, 0, left + new_w, img_h))
        else:
            new_h = int(img_w / target_ratio)
            top = (img_h - new_h) // 2
            cropped = topic_img.crop((0, top, img_w, top + new_h))

        resized = cropped.resize((right_w, height), Image.LANCZOS)

        # Boost contrast slightly
        resized = ImageEnhance.Contrast(resized).enhance(1.15)
        resized = ImageEnhance.Color(resized).enhance(1.1)

        canvas.paste(resized, (split_x, 0))

        # Gradient blend from left panel into right panel
        blend_w = 60
        for x in range(blend_w):
            alpha = 1.0 - (x / blend_w)
            for y in range(height):
                px = canvas.getpixel((split_x + x, y))
                dark = (12, 12, 16)
                blended = tuple(int(dark[c] * alpha + px[c] * (1 - alpha)) for c in range(3))
                canvas.putpixel((split_x + x, y), blended)
    else:
        # Fallback: dark gradient on right side too
        for x in range(split_x, width):
            progress = (x - split_x) / right_w
            shade = int(15 + 10 * progress)
            for y in range(height):
                canvas.putpixel((x, y), (shade, shade, shade + 3))

    # ── HOOK TEXT (Left Panel) ───────────────────────────────────────────
    lines = hook_text.split("\n")
    max_text_w = split_x - 80  # 40px padding on each side

    # Auto-scale font size
    font_size = 78
    font = _load_font(font_size, "extrabold")

    # Check if text fits, scale down if needed
    while font_size > 40:
        all_fit = True
        for line in lines:
            lw, _ = _text_size(line, font)
            if lw > max_text_w:
                all_fit = False
                break
        if all_fit:
            break
        font_size -= 4
        font = _load_font(font_size, "extrabold")

    # Calculate total text block height
    line_gap = int(font_size * 0.3)
    line_heights = []
    for line in lines:
        _, lh = _text_size(line, font)
        line_heights.append(lh)
    total_h = sum(line_heights) + line_gap * (len(lines) - 1)

    # Center vertically in left panel
    start_y = (height - total_h) // 2
    text_x = 40  # Left-aligned with padding

    for i, line in enumerate(lines):
        lw, lh = _text_size(line, font)
        y = start_y + sum(line_heights[:i]) + line_gap * i

        # Strong drop shadow
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                draw.text((text_x + dx, y + dy), line, font=font, fill=(0, 0, 0))

        # Main text: pure white
        draw.text((text_x, y), line, font=font, fill=(255, 255, 255))

    # ── SUBTLE ACCENT LINE (vertical divider) ────────────────────────────
    line_y1 = int(height * 0.15)
    line_y2 = int(height * 0.85)
    draw.line([(split_x - 2, line_y1), (split_x - 2, line_y2)],
              fill=accent_color, width=3)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: SHORTS THUMBNAIL (9:16)
# ─────────────────────────────────────────────────────────────────────────────
def _render_shorts_thumbnail(hook_text, topic_img, accent_color, width=1080, height=1920):
    """
    Renders a vertical Shorts thumbnail.
    TOP 55%: Topic-relevant image
    BOTTOM 45%: Dark panel with bold hook text
    """
    canvas = Image.new("RGB", (width, height), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)

    split_y = int(height * 0.55)

    # ── TOP: Topic image ─────────────────────────────────────────────────
    if topic_img:
        img_w, img_h = topic_img.size
        top_h = split_y
        target_ratio = width / top_h
        img_ratio = img_w / img_h

        if img_ratio > target_ratio:
            new_w = int(img_h * target_ratio)
            left = (img_w - new_w) // 2
            cropped = topic_img.crop((left, 0, left + new_w, img_h))
        else:
            new_h = int(img_w / target_ratio)
            top = (img_h - new_h) // 2
            cropped = topic_img.crop((0, top, img_w, top + new_h))

        resized = cropped.resize((width, top_h), Image.LANCZOS)
        resized = ImageEnhance.Contrast(resized).enhance(1.15)
        canvas.paste(resized, (0, 0))

        # Gradient blend from image into dark panel
        blend_h = 80
        for y in range(blend_h):
            alpha = y / blend_h
            for x in range(width):
                px = canvas.getpixel((x, split_y - blend_h + y))
                dark = (12, 12, 16)
                blended = tuple(int(px[c] * (1 - alpha) + dark[c] * alpha) for c in range(3))
                canvas.putpixel((x, split_y - blend_h + y), blended)

    # ── BOTTOM: Hook text ────────────────────────────────────────────────
    lines = hook_text.split("\n")
    max_text_w = width - 100

    font_size = 90
    font = _load_font(font_size, "extrabold")

    while font_size > 48:
        all_fit = True
        for line in lines:
            lw, _ = _text_size(line, font)
            if lw > max_text_w:
                all_fit = False
                break
        if all_fit:
            break
        font_size -= 4
        font = _load_font(font_size, "extrabold")

    line_gap = int(font_size * 0.35)
    line_heights = []
    for line in lines:
        _, lh = _text_size(line, font)
        line_heights.append(lh)
    total_h = sum(line_heights) + line_gap * (len(lines) - 1)

    bottom_h = height - split_y
    start_y = split_y + (bottom_h - total_h) // 2
    text_x = 50

    for i, line in enumerate(lines):
        lw, lh = _text_size(line, font)
        y = start_y + sum(line_heights[:i]) + line_gap * i

        # Drop shadow
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                draw.text((text_x + dx, y + dy), line, font=font, fill=(0, 0, 0))

        draw.text((text_x, y), line, font=font, fill=(255, 255, 255))

    # Horizontal accent line divider
    line_x1 = 50
    line_x2 = width - 50
    draw.line([(line_x1, split_y + 10), (line_x2, split_y + 10)],
              fill=accent_color, width=3)

    # Accent border (bottom and sides only)
    border_t = 6
    draw.rectangle([0, height - border_t, width, height], fill=accent_color)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def generate_thumbnail(script_json):
    """
    Generates both 1280x720 YouTube and 1080x1920 Shorts thumbnails
    using the High-Authority Split-Screen design.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_path_yt = os.path.join(OUTPUT_DIR, f"thumbnail_{date_str}.jpg")
    output_path_shorts = os.path.join(OUTPUT_DIR, f"thumbnail_shorts_{date_str}.jpg")

    title = script_json.get("title", "Tech News")
    accent_hex = script_json.get("color_theme", {}).get("accent", "#FFD600").lstrip("#")
    accent_color = tuple(int(accent_hex[i:i+2], 16) for i in (0, 2, 4))

    # 1. Generate curiosity-gap hook text
    print("🎨 Generating thumbnail hook text...")
    hook_text = _generate_hook_text(title)
    print(f"   Hook: {repr(hook_text)}")

    # 2. Load topic-relevant image
    print("🖼️  Loading topic image for thumbnail...")
    topic_img = _load_topic_image(script_json)
    if topic_img:
        print(f"   ✅ Topic image loaded: {topic_img.size}")
    else:
        print("   ⚠️ No topic image found, using dark fallback.")

    # 3. Render 16:9 YouTube thumbnail
    yt_thumb = _render_split_thumbnail(hook_text, topic_img, accent_color, THUMB_W, THUMB_H)
    yt_thumb.save(output_path_yt, "JPEG", quality=95)
    print(f"✅ Generated YouTube Thumbnail: {output_path_yt}")

    # 4. Render 9:16 Shorts thumbnail
    shorts_thumb = _render_shorts_thumbnail(hook_text, topic_img, accent_color, SHORTS_W, SHORTS_H)
    shorts_thumb.save(output_path_shorts, "JPEG", quality=95)
    print(f"✅ Generated Shorts Thumbnail: {output_path_shorts}")

    return output_path_yt
