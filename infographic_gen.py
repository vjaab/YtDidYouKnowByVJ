"""
infographic_gen.py — Programmatic motion-graphic infographic cards.

6 card types rendered with Pillow, animated with MoviePy:
  TYPE 1: Stat Card        (numbers, funding, valuations)
  TYPE 2: Comparison Card  (X vs Y)
  TYPE 3: Timeline Card    (dates, launches, history)
  TYPE 4: Definition Card  (explanations, "means")
  TYPE 5: Ranking Card     (top lists, number 1)
  TYPE 6: Growth Card      (percentage changes)
"""

import os
import math
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoClip, ImageClip
from config import ASSETS_DIR

FRAME_W, FRAME_H = 1080, 1920
VISUAL_CENTER_Y = 650  # Center of visual zone for vertical (200-1100)
VISUAL_CENTER_Y_LONGFORM = 540 # True center for 1080p

def get_dimensions(is_longform):
    if is_longform:
        return 1920, 1080, VISUAL_CENTER_Y_LONGFORM
    return 1080, 1920, VISUAL_CENTER_Y

# ── Font helpers ──────────────────────────────────────────────────────────────
_FONT_EXTRA_BOLD = os.path.join(ASSETS_DIR, "fonts", "Montserrat-ExtraBold.ttf")
_FONT_BOLD = os.path.join(ASSETS_DIR, "fonts", "Montserrat-Bold.ttf")
_FONT_REGULAR = os.path.join(ASSETS_DIR, "fonts", "Roboto-Regular.ttf")

_FALLBACKS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf",
]

_font_cache = {}


def _load_font(path, size):
    key = (path, size)
    if key not in _font_cache:
        for p in [path] + _FALLBACKS:
            if os.path.exists(p):
                try:
                    _font_cache[key] = ImageFont.truetype(p, size)
                    break
                except Exception:
                    pass
        if key not in _font_cache:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def _eb(size):
    return _load_font(_FONT_EXTRA_BOLD, size)


def _bold(size):
    return _load_font(_FONT_BOLD, size)


def _reg(size):
    return _load_font(_FONT_REGULAR, size)


def _ts(text, font):
    bb = font.getbbox(text)
    return bb[2] - bb[0], bb[3] - bb[1]


def _center_text(draw, text, font, y, color, card_x, card_w):
    tw, _ = _ts(text, font)
    x = card_x + (card_w - tw) // 2
    draw.text((x, y), text, font=font, fill=color)


# ── Common card frame ─────────────────────────────────────────────────────────
def _draw_card_bg(draw, cx, cy, cw, ch, accent_color, border=2, radius=24):
    """Draws the dark card background with accent border and shadow."""
    # Shadow
    draw.rounded_rectangle(
        [cx + 6, cy + 12, cx + cw + 6, cy + ch + 12],
        radius=radius, fill=(0, 0, 0, 128)
    )
    # Border
    draw.rounded_rectangle(
        [cx - border, cy - border, cx + cw + border, cy + ch + border],
        radius=radius, fill=(*accent_color, 255)
    )
    # Inner fill
    draw.rounded_rectangle(
        [cx, cy, cx + cw, cy + ch],
        radius=radius, fill=(15, 15, 15, 242)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 1 — STAT CARD
# ═══════════════════════════════════════════════════════════════════════════════
def _render_stat_card(data, accent_color, progress=1.0):
    """
    progress: 0.0→1.0 controls count-up animation on the headline number.
    """
    canvas = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    cw, ch = 920, 480
    cx = (FRAME_W - cw) // 2
    cy = VISUAL_CENTER_Y - ch // 2

    _draw_card_bg(draw, cx, cy, cw, ch, accent_color)

    icon = data.get("icon", "📊")
    headline = data.get("headline", "")
    subtext = data.get("subtext", "")
    context = data.get("context", "")
    source = data.get("source", "")

    # Count-up animation for numeric headlines
    if data.get("count_up") and progress < 1.0:
        count_to = float(data.get("count_to", 0))
        count_from = float(data.get("count_from", 0))
        suffix = data.get("count_suffix", "")
        prefix = data.get("count_prefix", "$")
        # Ease out
        p = 1.0 - (1.0 - progress) ** 3
        current = count_from + (count_to - count_from) * p
        if count_to == int(count_to) and count_from == int(count_from):
            headline = f"{prefix}{int(current)}{suffix}"
        else:
            headline = f"{prefix}{current:.1f}{suffix}"

    # Icon
    fi = _eb(80)
    _center_text(draw, icon, fi, cy + 35, (255, 255, 255, 255), cx, cw)

    # Headline (Increased from 96 for mobile)
    fh = _eb(116)
    _center_text(draw, headline, fh, cy + 140, (*accent_color, 255), cx, cw)

    # Subtext (Increased from 42 for clarity)
    fs = _bold(52)
    _center_text(draw, subtext, fs, cy + 270, (255, 255, 255, 255), cx, cw)

    # Context
    fc = _reg(30)
    _center_text(draw, context, fc, cy + 340, (170, 170, 170, 255), cx, cw)

    # Source
    fsr = _reg(24)
    _center_text(draw, source, fsr, cy + ch - 50, (102, 102, 102, 255), cx, cw)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 2 — COMPARISON CARD
# ═══════════════════════════════════════════════════════════════════════════════
def _render_comparison_card(data, accent_color, progress=1.0):
    canvas = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    cw, ch = 920, 520
    cx = (FRAME_W - cw) // 2
    cy = VISUAL_CENTER_Y - ch // 2

    _draw_card_bg(draw, cx, cy, cw, ch, accent_color)

    left = data.get("left", {})
    right = data.get("right", {})
    winner = data.get("winner", "left")

    col_w = (cw - 100) // 2  # Each column width

    # Left column
    lx = cx + 20
    fn = _bold(36)
    fv = _eb(72) # Increased from 52 for impact
    fl = _reg(28)

    l_name = left.get("name", "A")
    l_value = left.get("value", "—")
    l_label = left.get("label", "")
    l_icon = left.get("icon", "🔹")

    _center_text(draw, l_icon, _eb(56), cy + 40, (255, 255, 255, 255), lx, col_w)
    _center_text(draw, l_name, fn, cy + 120, (255, 255, 255, 255), lx, col_w)
    l_color = (*accent_color, 255) if winner == "left" else (180, 180, 180, 255)
    _center_text(draw, l_value, fv, cy + 180, l_color, lx, col_w)
    _center_text(draw, l_label, fl, cy + 250, (170, 170, 170, 255), lx, col_w)

    if winner == "left":
        _center_text(draw, "🏆 Winner", _bold(26), cy + 310, (*accent_color, 255), lx, col_w)

    # VS badge
    vs_x = cx + cw // 2 - 35
    vs_y = cy + ch // 2 - 35
    draw.ellipse([vs_x, vs_y, vs_x + 70, vs_y + 70], fill=(*accent_color, 255))
    _center_text(draw, "VS", _bold(28), vs_y + 18, (255, 255, 255, 255), vs_x, 70)

    # Right column
    rx = cx + cw - col_w - 20
    r_name = right.get("name", "B")
    r_value = right.get("value", "—")
    r_label = right.get("label", "")
    r_icon = right.get("icon", "🔸")

    _center_text(draw, r_icon, _eb(56), cy + 40, (255, 255, 255, 255), rx, col_w)
    _center_text(draw, r_name, fn, cy + 120, (255, 255, 255, 255), rx, col_w)
    r_color = (*accent_color, 255) if winner == "right" else (180, 180, 180, 255)
    _center_text(draw, r_value, fv, cy + 180, r_color, rx, col_w)
    _center_text(draw, r_label, fl, cy + 250, (170, 170, 170, 255), rx, col_w)

    if winner == "right":
        _center_text(draw, "🏆 Winner", _bold(26), cy + 310, (*accent_color, 255), rx, col_w)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 3 — TIMELINE CARD
# ═══════════════════════════════════════════════════════════════════════════════
def _render_timeline_card(data, accent_color, progress=1.0):
    canvas = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    events = data.get("events", [])
    current_idx = data.get("current_index", len(events) - 1)

    cw, ch = 880, 120 * max(len(events), 2) + 80
    ch = min(ch, 500)  # Cap at 500px
    cx = (FRAME_W - cw) // 2
    cy = VISUAL_CENTER_Y - ch // 2

    _draw_card_bg(draw, cx, cy, cw, ch, accent_color)

    fd = _bold(36) # Increased from 28
    ft = _reg(38) # Increased from 32
    line_x = cx + 60

    # Vertical timeline line
    draw.line(
        [(line_x, cy + 50), (line_x, cy + ch - 40)],
        fill=(*accent_color, 120), width=3
    )

    for i, event in enumerate(events):
        ey = cy + 55 + i * 110
        is_current = (i == current_idx)

        # Reveal animation: events appear sequentially based on progress
        event_progress = min(1.0, progress * len(events) - i)
        if event_progress <= 0:
            continue

        alpha = int(255 * min(1.0, event_progress))

        # Dot
        dot_r = 12 if is_current else 8
        dot_color = (*accent_color, alpha) if is_current else (120, 120, 120, alpha)
        draw.ellipse(
            [line_x - dot_r, ey - dot_r, line_x + dot_r, ey + dot_r],
            fill=dot_color
        )
        if is_current:
            # Outer pulse ring
            draw.ellipse(
                [line_x - 18, ey - 18, line_x + 18, ey + 18],
                outline=(*accent_color, int(alpha * 0.4)), width=2
            )

        # Date
        date_text = event.get("date", "")
        text_color = (*accent_color, alpha) if is_current else (200, 200, 200, alpha)
        draw.text((line_x + 30, ey - 22), date_text, font=fd, fill=text_color)

        # Event text
        evt_text = event.get("text", "")
        evt_color = (255, 255, 255, alpha) if is_current else (150, 150, 150, alpha)
        draw.text((line_x + 30, ey + 8), evt_text, font=ft, fill=evt_color)

        # Arrow for current
        if is_current:
            draw.text((cx + cw - 60, ey - 8), "←", font=_bold(32), fill=(*accent_color, alpha))

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 4 — DEFINITION CARD
# ═══════════════════════════════════════════════════════════════════════════════
def _render_definition_card(data, accent_color, progress=1.0):
    canvas = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    cw, ch = 880, 440
    cx = (FRAME_W - cw) // 2
    cy = VISUAL_CENTER_Y - ch // 2

    _draw_card_bg(draw, cx, cy, cw, ch, accent_color)

    term = data.get("term", "")
    icon = data.get("icon", "📖")
    definition = data.get("definition", "")
    example = data.get("example", "")

    # Term + icon (Increased from 52 to 68)
    ft = _eb(68)
    chars_to_show = max(1, int(len(term) * min(1.0, progress * 2)))
    display_term = term[:chars_to_show]
    term_line = f"{display_term}  {icon}"
    _center_text(draw, term_line, ft, cy + 40, (*accent_color, 255), cx, cw)

    # Divider line (draws left to right)
    if progress > 0.25:
        line_progress = min(1.0, (progress - 0.25) * 4)
        line_w = int((cw - 80) * line_progress)
        draw.line(
            [(cx + 40, cy + 120), (cx + 40 + line_w, cy + 120)],
            fill=(*accent_color, 200), width=2
        )

    # Definition (fade in)
    if progress > 0.4:
        fd = _reg(34)
        # Word wrap definition
        words = definition.split()
        lines = []
        cur = []
        for w in words:
            test = " ".join(cur + [w])
            if _ts(test, fd)[0] > cw - 80 and cur:
                lines.append(" ".join(cur))
                cur = [w]
            else:
                cur.append(w)
        if cur:
            lines.append(" ".join(cur))

        def_alpha = int(255 * min(1.0, (progress - 0.4) * 3))
        for i, line in enumerate(lines[:3]):
            _center_text(draw, line, fd, cy + 150 + i * 48, (255, 255, 255, def_alpha), cx, cw)

    # Example (fade in later)
    if progress > 0.6 and example:
        fe = _reg(28)
        ex_alpha = int(255 * min(1.0, (progress - 0.6) * 3))
        _center_text(draw, f"e.g. {example}", fe, cy + ch - 70, (150, 150, 150, ex_alpha), cx, cw)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 5 — RANKING CARD
# ═══════════════════════════════════════════════════════════════════════════════
def _render_ranking_card(data, accent_color, progress=1.0):
    canvas = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    items = data.get("items", [])
    highlight_idx = data.get("highlight_index", 0)

    cw = 880
    row_h = 80
    ch = min(len(items) * row_h + 60, 520)
    cx = (FRAME_W - cw) // 2
    cy = VISUAL_CENTER_Y - ch // 2

    _draw_card_bg(draw, cx, cy, cw, ch, accent_color)

    fn = _bold(36)
    fv = _reg(30)

    # Items reveal from bottom to top
    for i, item in enumerate(items):
        # Reversed reveal: last item first
        reveal_order = len(items) - 1 - i
        item_progress = min(1.0, progress * (len(items) + 1) - reveal_order)
        if item_progress <= 0:
            continue

        alpha = int(255 * min(1.0, item_progress))
        ry = cy + 30 + i * row_h
        is_highlight = (i == highlight_idx)

        # Highlighted row background
        if is_highlight:
            draw.rounded_rectangle(
                [cx + 10, ry, cx + cw - 10, ry + row_h - 10],
                radius=12, fill=(*accent_color, int(40 * min(1.0, item_progress)))
            )

        # Rank number (Increased from 40 to 50)
        rank = item.get("rank", i + 1)
        rank_text = f"#{rank}"
        rank_color = (*accent_color, alpha) if is_highlight else (180, 180, 180, alpha)
        draw.text((cx + 30, ry + 20), rank_text, font=_eb(50), fill=rank_color)

        # Name
        name = item.get("name", "")
        name_color = (255, 255, 255, alpha)
        draw.text((cx + 120, ry + 15), name, font=fn, fill=name_color)

        # Metric
        metric = item.get("metric", "")
        draw.text((cx + 120, ry + 52), metric, font=fv, fill=(150, 150, 150, alpha))

        # Trophy for #1
        if rank == 1 and item_progress > 0.5:
            draw.text((cx + cw - 80, ry + 18), "🏆", font=_eb(40), fill=(255, 255, 255, alpha))

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 6 — GROWTH CARD
# ═══════════════════════════════════════════════════════════════════════════════
def _render_growth_card(data, accent_color, progress=1.0):
    canvas = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    cw, ch = 880, 460
    cx = (FRAME_W - cw) // 2
    cy = VISUAL_CENTER_Y - ch // 2

    _draw_card_bg(draw, cx, cy, cw, ch, accent_color)

    percentage = str(data.get("percentage", "+0%"))
    is_growth = not percentage.startswith("-")
    before = data.get("before", "")
    after = data.get("after", "")
    label = data.get("label", "")
    period = data.get("period", "")

    # Arrow (animated draw)
    arrow_color = (76, 175, 80, 255) if is_growth else (244, 67, 54, 255)
    arrow_symbol = "↑" if is_growth else "↓"
    arrow_alpha = int(255 * min(1.0, progress * 3))

    fa = _eb(100)
    _center_text(draw, arrow_symbol, fa, cy + 20, (*arrow_color[:3], arrow_alpha), cx, cw)

    # Percentage (Increased from 96 to 116)
    fp = _eb(116)
    # Extract number from percentage string
    num_match = re.search(r'[\d.]+', percentage)
    if num_match and progress < 0.8:
        num_val = float(num_match.group())
        p = 1.0 - (1.0 - min(1.0, progress * 1.5)) ** 3
        sign = "+" if is_growth else "-"
        display_pct = f"{sign}{num_val * p:.0f}%"
    else:
        display_pct = percentage

    pct_color = (*accent_color, 255) if is_growth else (244, 67, 54, 255)
    _center_text(draw, display_pct, fp, cy + 135, pct_color, cx, cw)

    # Label
    if label:
        fl = _bold(36)
        _center_text(draw, label, fl, cy + 260, (255, 255, 255, 255), cx, cw)

    # Before → After
    if before and after:
        fb = _reg(30)
        ba_text = f"{before} → {after}"
        _center_text(draw, ba_text, fb, cy + 320, (180, 180, 180, 255), cx, cw)

    # Period
    if period:
        fpr = _reg(24)
        _center_text(draw, period, fpr, cy + ch - 55, (102, 102, 102, 255), cx, cw)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# DISPATCHER — Routes to correct card type
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 7 — SLIDE CARD
# ═══════════════════════════════════════════════════════════════════════════════
def _render_slide_card(data, accent_color, progress=1.0, is_longform=False):
    import math
    
    fw, fh, fcy = get_dimensions(is_longform)
    canvas = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    
    cw, ch = (1400, 700) if is_longform else (920, 600)
    cx = (fw - cw) // 2
    cy = fcy - ch // 2

    # Create the base background image
    bg_img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(bg_img)
    
    # Base dark blue gradient simulation
    cdraw.rectangle([0, 0, cw, ch], fill=(0, 40, 120, 255))
    
    # Glowing cyan blobs for the light effect
    cdraw.ellipse([cw*0.3, ch*0.1, cw*1.2, ch*1.5], fill=(0, 150, 255, 90))
    cdraw.ellipse([cw*0.5, ch*0.3, cw*1.0, ch*1.0], fill=(0, 200, 255, 120))
    
    # Cyber sweeping arcs
    cdraw.arc([-cw*0.2, -ch*0.2, cw*0.8, ch*1.2], 270, 360, fill=(150, 220, 255, 100), width=6)
    cdraw.arc([-cw*0.1, -ch*0.1, cw*0.6, ch*1.0], 270, 360, fill=(150, 220, 255, 60), width=3)
    
    # Hex grid on the left
    hex_size = 30 if not is_longform else 45
    for row in range(int(ch / (hex_size*1.5)) + 1):
        for col in range(7):
            hcx = 40 + col * hex_size * 1.5
            hcy = 40 + row * hex_size * math.sqrt(3)
            if col % 2 == 1:
                hcy += hex_size * math.sqrt(3) / 2
                
            points = []
            for i in range(6):
                angle_rad = math.pi / 3 * i
                points.append((hcx + hex_size * math.cos(angle_rad), hcy + hex_size * math.sin(angle_rad)))
            cdraw.polygon(points, outline=(0, 200, 255, 50), width=3)

    # Dark gradient on right for text readability
    for x in range(cw):
        alpha = int(160 * (x / cw))
        cdraw.line([(x, 0), (x, ch)], fill=(0, 10, 40, alpha))
            
    # Apply rounded corner mask to the entire generated background
    mask = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, cw, ch], 40, fill=255)
    
    card_layer = Image.new("RGBA", (fw, fh), (0,0,0,0))
    card_layer.paste(bg_img, (cx, cy), mask)
    
    ImageDraw.Draw(card_layer).rounded_rectangle([cx, cy, cx+cw, cy+ch], 40, outline=accent_color, width=8)
    canvas = Image.alpha_composite(canvas, card_layer)

    draw = ImageDraw.Draw(canvas)
    
    title = data.get("title", "Technical Architecture")
    bullets = data.get("bullet_points", [])

    ft = _eb(60 if is_longform else 50)
    ttw, tth = _ts(title, ft)
    
    title_x = cx + cw - ttw - 50
    title_y = cy + 50
    draw.text((title_x+3, title_y+3), title, fill=(0,0,0,200), font=ft)
    draw.text((title_x, title_y), title, fill=(255, 255, 255, 255), font=ft)
    
    line_w = int((cw // 2 - 40) * progress)
    if line_w > 0:
        draw.line([(cx + cw // 2 + 20, cy + 140), (cx + cw // 2 + 20 + line_w, cy + 140)], fill=(*accent_color, 255), width=4)
        
    fb = _bold(40 if is_longform else 34)
    start_y = cy + 180
    for i, bullet in enumerate(bullets):
        bullet_progress = min(1.0, progress * (len(bullets) + 1) - i)
        if bullet_progress <= 0:
            continue
            
        alpha = int(255 * min(1.0, bullet_progress * 2))
        
        max_bw = cw // 2 - 40
        words = str(bullet).split()
        lines = []
        cur = []
        for w_word in words:
            test = " ".join(cur + [w_word])
            if _ts(test, fb)[0] > max_bw and cur:
                lines.append(" ".join(cur))
                cur = [w_word]
            else:
                cur.append(w_word)
        if cur:
            lines.append(" ".join(cur))
            
        by = start_y + i * 80
        for j, b_line in enumerate(lines):
            lw, lh = _ts(b_line, fb)
            line_x = cx + cw - lw - 50
            
            if j == 0:
                dot_y = by + (25 if is_longform else 20)
                draw.ellipse([line_x - 30, dot_y - 8, line_x - 14, dot_y + 8], fill=(*accent_color, alpha))
            
            draw.text((line_x+2, by+2), b_line, font=fb, fill=(0,0,0,int(150*(alpha/255))))
            draw.text((line_x, by), b_line, font=fb, fill=(230, 240, 255, alpha))
            by += 45

    return canvas

_TYPE_MAP = {
    "stat": _render_stat_card,
    "funding_stat": _render_stat_card,
    "comparison": _render_comparison_card,
    "timeline": _render_timeline_card,
    "definition": _render_definition_card,
    "ranking": _render_ranking_card,
    "growth": _render_growth_card,
    "percentage": _render_growth_card,
    "slide": _render_slide_card,
}


def render_infographic(infographic_type, infographic_data, accent_color, progress=1.0, is_longform=False):
    """
    Renders an infographic card frame.
    Returns a PIL RGBA Image.
    Now includes dynamic parsing for unstructured data from Gemini.
    """
    # STEP 2.2: Dynamic Data Injection Logic
    if isinstance(infographic_data, str):
        # Attempt to parse a string-based data point from the LLM
        # Format: "Term: Definition | Example" or "Stat: 100M | Label"
        try:
            parts = infographic_data.split("|")
            main = parts[0].split(":")
            if infographic_type == "definition":
                infographic_data = {
                    "term": main[0].strip(),
                    "definition": main[1].strip() if len(main)>1 else "",
                    "example": parts[1].strip() if len(parts)>1 else ""
                }
            elif infographic_type == "stat":
                infographic_data = {
                    "headline": main[0].strip(),
                    "subtext": main[1].strip() if len(main)>1 else "",
                    "context": parts[1].strip() if len(parts)>1 else ""
                }
        except Exception:
            pass # Fall back to using the string as a headline if parsing fails

    renderer = _TYPE_MAP.get(infographic_type, _render_stat_card)
    
    # We pass is_longform to slide renderer only for now, or adapt others later.
    import inspect
    sig = inspect.signature(renderer)
    if "is_longform" in sig.parameters:
        return renderer(infographic_data, accent_color, progress, is_longform=is_longform)
    else:
        # Fallback to overriding global for older renderers inside their scope
        # It's better to just pass it in or accept they render at 1080x1920 center.
        return renderer(infographic_data, accent_color, progress)


# ═══════════════════════════════════════════════════════════════════════════════
# MoviePy clip builder — builds animated clip for a chunk
# ═══════════════════════════════════════════════════════════════════════════════
def build_infographic_clip(chunk, accent_color, is_longform=False):
    """
    Builds a MoviePy clip for an infographic card with:
      - Entry: scale 0.85→1.0 + fade in (0.3s)
      - Hold: full visibility
      - Exit: scale→0.95 + fade out (0.2s)
    Returns (card_clip, overlay_clip) or (None, None).
    """
    info = chunk.get("infographic_data", {})
    info_type = chunk.get("infographic_type", "stat")
    dur = chunk["duration"]
    start = chunk["start"]

    if dur < 0.2:
        return None, None

    fw, fh, fcy = get_dimensions(is_longform)
    fade_in = 0.3
    fade_out = 0.2
    count_dur = min(1.5, dur * 0.6)  # Count-up animation duration

    def make_frame(t):
        # Progress for count-up animation
        progress = min(1.0, t / count_dur) if count_dur > 0 else 1.0
        img = render_infographic(info_type, info, accent_color, progress, is_longform=is_longform)
        return np.array(img.convert("RGB"))

    def make_mask(t):
        progress = min(1.0, t / count_dur) if count_dur > 0 else 1.0
        img = render_infographic(info_type, info, accent_color, progress, is_longform=is_longform)
        mask_arr = np.array(img.split()[3]).astype(float) / 255.0

        # Fade in
        if t < fade_in:
            mask_arr *= t / fade_in
        # Fade out
        if dur - t < fade_out:
            mask_arr *= max(0, (dur - t) / fade_out)

        return mask_arr

    card_clip = VideoClip(make_frame, duration=dur)
    card_mask = VideoClip(make_mask, is_mask=True, duration=dur)
    card_clip = card_clip.with_mask(card_mask).with_start(start)

    # Dark overlay behind card (0.75 opacity)
    overlay_arr = np.zeros((fh, fw, 3), dtype=np.uint8)

    def overlay_mask(t):
        base = 0.75
        if t < fade_in:
            return np.full((fh, fw), base * (t / fade_in))
        if dur - t < fade_out:
            return np.full((fh, fw), base * max(0, (dur - t) / fade_out))
        return np.full((fh, fw), base)

    overlay_clip = VideoClip(lambda t: overlay_arr, duration=dur)
    overlay_mask_clip = VideoClip(overlay_mask, is_mask=True, duration=dur)
    overlay_clip = overlay_clip.with_mask(overlay_mask_clip).with_start(start)

    return card_clip, overlay_clip
