#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
carousel_renderer.py — Pillow-based renderer for AI News carousel slides.
Generates 6 slides at 1080x1350 with consistent branding.
Uses available TTF fonts (Montserrat, Roboto) from assets/fonts.
"""

import os
import json
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Any, Tuple, Optional

CANVAS_W, CANVAS_H = 1080, 1350

BRAND_COLORS = {
    "bg_dark": "#0B0E14",
    "bg_card": "#12161F",
    "border": "#232A38",
    "text_primary": "#F1F3F8",
    "text_secondary": "#8A93A8",
    "accent_purple": "#C879E6",
    "accent_blue": "#5FB3F0",
    "accent_green": "#86D9A0",
    "accent_yellow": "#E8B85C",
    "accent_cyan": "#58C7D6",
    "accent_red": "#F07178",
    "brand_gradient_start": "#C879E6",
    "brand_gradient_end": "#5FB3F0",
}

SLIDE_COLORS = {
    "hook": BRAND_COLORS["accent_purple"],
    "what_happened": BRAND_COLORS["accent_blue"],
    "whats_new": BRAND_COLORS["accent_green"],
    "why_matters": BRAND_COLORS["accent_yellow"],
    "real_world_example": BRAND_COLORS["accent_cyan"],
    "takeaway_cta": BRAND_COLORS["accent_red"],
}

FONTS_DIR = Path(__file__).parent / "assets" / "fonts"

FONT_CACHE = {}

def get_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Get cached font."""
    key = (name, size)
    if key not in FONT_CACHE:
        font_path = FONTS_DIR / name
        if font_path.exists():
            FONT_CACHE[key] = ImageFont.truetype(str(font_path), size)
        else:
            FONT_CACHE[key] = ImageFont.load_default()
    return FONT_CACHE[key]

def get_fonts():
    """Get standard fonts for rendering using available TTF fonts."""
    return {
        "title": get_font("Montserrat-ExtraBold.ttf", 58),
        "title_small": get_font("Montserrat-Bold.ttf", 48),
        "subtitle": get_font("Montserrat-Medium.ttf", 24),
        "body": get_font("Roboto-Regular.ttf", 22),
        "body_large": get_font("Roboto-Regular.ttf", 26),
        "body_bold": get_font("Roboto-Bold.ttf", 22),
        "caption": get_font("Roboto-Regular.ttf", 18),
        "mono": get_font("Roboto-Mono.ttf", 16) if (FONTS_DIR / "Roboto-Mono.ttf").exists() else get_font("Roboto-Regular.ttf", 16),
        "mono_large": get_font("Roboto-Mono.ttf", 20) if (FONTS_DIR / "Roboto-Mono.ttf").exists() else get_font("Roboto-Bold.ttf", 20),
        "badge": get_font("Roboto-Regular.ttf", 14),
        "page_num": get_font("Montserrat-Bold.ttf", 28),
        "footer": get_font("Montserrat-Medium.ttf", 16),
    }

def draw_rounded_rect(draw: ImageDraw.Draw, xy: Tuple[int, int, int, int], radius: int, fill: str, outline: str = None, width: int = 0):
    """Draw rounded rectangle."""
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

def draw_gradient_bg(img: Image.Image, start_color: str, end_color: str):
    """Draw vertical gradient background."""
    draw = ImageDraw.Draw(img)
    for y in range(CANVAS_H):
        ratio = y / CANVAS_H
        r = int(int(start_color[1:3], 16) * (1 - ratio) + int(end_color[1:3], 16) * ratio)
        g = int(int(start_color[3:5], 16) * (1 - ratio) + int(end_color[3:5], 16) * ratio)
        b = int(int(start_color[5:7], 16) * (1 - ratio) + int(end_color[5:7], 16) * ratio)
        draw.line([(0, y), (CANVAS_W, y)], fill=(r, g, b))

def draw_dots_pattern(img: Image.Image, opacity: float = 0.03):
    """Draw subtle dot pattern."""
    draw = ImageDraw.Draw(img, "RGBA")
    dot_color = (255, 255, 255, int(255 * opacity))
    spacing = 28
    for x in range(-10, CANVAS_W + 10, spacing):
        for y in range(-10, CANVAS_H + 10, spacing):
            draw.ellipse([x, y, x + 2, y + 2], fill=dot_color)

def draw_glow_orb(img: Image.Image, color: str, x: int, y: int, radius: int, opacity: float = 0.08):
    """Draw glow orb."""
    overlay = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    for i in range(radius, 0, -1):
        alpha = int(255 * opacity * (i / radius) * 0.3)
        draw.ellipse([x - i, y - i, x + i, y + i], fill=(r, g, b, alpha))
    img.alpha_composite(overlay)

def draw_chrome_bar(draw: ImageDraw.Draw, fonts: Dict, slide_num: int, total_slides: int, category: str = "AI NEWS"):
    """Draw top chrome bar with filename, category, and slide indicator."""
    # Chrome bar background
    draw_rounded_rect(draw, (0, 0, CANVAS_W, 72), 0, BRAND_COLORS["bg_card"], BRAND_COLORS["border"], 1)
    
    # Window dots
    dots = [(24, 24), (48, 24), (72, 24)]
    dot_colors = ["#FF5F56", "#FFBD2E", "#27C93F"]
    for (x, y), color in zip(dots, dot_colors):
        draw.ellipse([x, y, x + 14, y + 14], fill=color)
    
    # Category/filename
    draw.text((100, 22), category, font=fonts["mono"], fill=BRAND_COLORS["text_secondary"])
    
    # Right side: slide indicator
    slide_text = f"{slide_num:02d} / {total_slides:02d}"
    bbox = draw.textbbox((0, 0), slide_text, font=fonts["page_num"])
    text_w = bbox[2] - bbox[0]
    draw.text((CANVAS_W - text_w - 24, 18), slide_text, font=fonts["page_num"], fill=BRAND_COLORS["text_secondary"])

def draw_footer(draw: ImageDraw.Draw, fonts: Dict, accent_color: str):
    """Draw bottom footer with brand."""
    y_start = CANVAS_H - 88
    draw_rounded_rect(draw, (0, y_start, CANVAS_W, CANVAS_H), 0, BRAND_COLORS["bg_card"], BRAND_COLORS["border"], 1)
    
    # Brand avatar
    avatar_x, avatar_y = 48, y_start + 23
    draw_rounded_rect(draw, (avatar_x, avatar_y, avatar_x + 42, avatar_y + 42), 12, 
                      None, None, 0)
    # Gradient avatar
    for i in range(21):
        ratio = i / 20
        r = int(int(BRAND_COLORS["brand_gradient_start"][1:3], 16) * (1 - ratio) + 
                int(BRAND_COLORS["brand_gradient_end"][1:3], 16) * ratio)
        g = int(int(BRAND_COLORS["brand_gradient_start"][3:5], 16) * (1 - ratio) + 
                int(BRAND_COLORS["brand_gradient_end"][3:5], 16) * ratio)
        b = int(int(BRAND_COLORS["brand_gradient_start"][5:7], 16) * (1 - ratio) + 
                int(BRAND_COLORS["brand_gradient_end"][5:7], 16) * ratio)
        draw.ellipse([avatar_x + i, avatar_y + i, avatar_x + 42 - i, avatar_y + 42 - i], 
                     outline=(r, g, b), width=1)
    
    draw.text((avatar_x + 10, avatar_y + 8), "VJ", font=fonts["footer"], fill="#0B0E14")
    
    # Brand name and handle
    draw.text((avatar_x + 60, y_start + 18), "VJ AI NEWS", font=fonts["footer"], fill=BRAND_COLORS["text_primary"])
    draw.text((avatar_x + 60, y_start + 42), "@vijayakumarj_ai", font=fonts["badge"], fill=BRAND_COLORS["text_secondary"])

def render_hook_slide(draw: ImageDraw.Draw, fonts: Dict, slide: Dict, accent: str, category: str, date: str):
    """Render Slide 1: HOOK"""
    # Large hook title
    title = slide.get("title", "")
    body = slide.get("body", "")
    
    # Title - large, centered
    wrapped = textwrap.wrap(title, width=18)
    y_start = 280
    for i, line in enumerate(wrapped):
        bbox = draw.textbbox((0, 0), line, font=fonts["title"])
        text_w = bbox[2] - bbox[0]
        x = (CANVAS_W - text_w) // 2
        y = y_start + i * 70
        # Shadow
        draw.text((x + 3, y + 3), line, font=fonts["title"], fill="#000000")
        draw.text((x, y), line, font=fonts["title"], fill=BRAND_COLORS["text_primary"])
    
    # Body text
    if body:
        wrapped_body = textwrap.wrap(body, width=42)
        y_body = y_start + len(wrapped) * 70 + 40
        for line in wrapped_body[:2]:
            bbox = draw.textbbox((0, 0), line, font=fonts["body_large"])
            text_w = bbox[2] - bbox[0]
            x = (CANVAS_W - text_w) // 2
            draw.text((x, y_body), line, font=fonts["body_large"], fill=BRAND_COLORS["text_secondary"])
            y_body += 40
    
    # Visual placeholder area
    visual_y = CANVAS_H - 400
    draw_rounded_rect(draw, (120, visual_y, CANVAS_W - 120, visual_y + 200), 24, 
                      BRAND_COLORS["bg_card"], accent, 2)
    draw.text((CANVAS_W // 2, visual_y + 90), "[ AI VISUAL ]", font=fonts["mono"], 
              fill=BRAND_COLORS["text_secondary"], anchor="mm")
    
    # Swipe indicator
    draw.text((CANVAS_W // 2, CANVAS_H - 140), "Swipe →", font=fonts["caption"], 
              fill=accent, anchor="mm")

def render_what_happened_slide(draw: ImageDraw.Draw, fonts: Dict, slide: Dict, accent: str, category: str, date: str):
    """Render Slide 2: WHAT HAPPENED?"""
    title = slide.get("title", "WHAT HAPPENED?")
    body = slide.get("body", [])
    
    # Section title
    draw.text((60, 120), title, font=fonts["title_small"], fill=accent)
    
    # Date/Source badge
    badge_text = f"📅 {date}  |  📰 {slide.get('source', 'Source')}"
    draw.text((60, 180), badge_text, font=fonts["badge"], fill=BRAND_COLORS["text_secondary"])
    
    # Bullets
    y = 260
    for i, bullet in enumerate(body[:3]):
        # Bullet circle
        bx, by = 80, y + 8
        draw.ellipse([bx, by, bx + 12, by + 12], fill=accent)
        
        # Bullet text
        wrapped = textwrap.wrap(bullet, width=55)
        for j, line in enumerate(wrapped):
            draw.text((110, y + j * 36), line, font=fonts["body"], fill=BRAND_COLORS["text_primary"])
        y += len(wrapped) * 36 + 24
    
    # Visual hint area
    visual_y = y + 40
    draw_rounded_rect(draw, (80, visual_y, CANVAS_W - 80, visual_y + 280), 20, 
                      BRAND_COLORS["bg_card"], BRAND_COLORS["border"], 1)
    draw.text((CANVAS_W // 2, visual_y + 130), "[ DIAGRAM / ICONS ]", font=fonts["mono"], 
              fill=BRAND_COLORS["text_secondary"], anchor="mm")

def render_whats_new_slide(draw: ImageDraw.Draw, fonts: Dict, slide: Dict, accent: str, category: str, date: str):
    """Render Slide 3: WHAT'S NEW?"""
    title = slide.get("title", "WHAT'S NEW?")
    body = slide.get("body", "")
    
    draw.text((60, 120), title, font=fonts["title_small"], fill=accent)
    
    # Key technical change
    wrapped = textwrap.wrap(body, width=50)
    y = 200
    for line in wrapped[:6]:
        draw.text((60, y), line, font=fonts["body"], fill=BRAND_COLORS["text_primary"])
        y += 38
    
    # Before/After visual
    visual_y = y + 40
    card_w = (CANVAS_W - 180) // 2
    
    # BEFORE
    draw_rounded_rect(draw, (60, visual_y, 60 + card_w, visual_y + 220), 16, 
                      BRAND_COLORS["bg_card"], BRAND_COLORS["accent_red"], 2)
    draw.text((60 + card_w // 2, visual_y + 30), "BEFORE", font=fonts["badge"], 
              fill=BRAND_COLORS["accent_red"], anchor="mm")
    draw.text((60 + card_w // 2, visual_y + 110), "Manual setup\nComplex config\nSlow iteration", 
              font=fonts["body"], fill=BRAND_COLORS["text_secondary"], anchor="mm", align="center")
    
    # AFTER
    draw_rounded_rect(draw, (CANVAS_W - 60 - card_w, visual_y, CANVAS_W - 60, visual_y + 220), 16, 
                      BRAND_COLORS["bg_card"], BRAND_COLORS["accent_green"], 2)
    draw.text((CANVAS_W - 60 - card_w // 2, visual_y + 30), "AFTER", font=fonts["badge"], 
              fill=BRAND_COLORS["accent_green"], anchor="mm")
    draw.text((CANVAS_W - 60 - card_w // 2, visual_y + 110), "One-click deploy\nAuto config\nInstant feedback", 
              font=fonts["body"], fill=BRAND_COLORS["text_secondary"], anchor="mm", align="center")

def render_why_matters_slide(draw: ImageDraw.Draw, fonts: Dict, slide: Dict, accent: str, category: str, date: str):
    """Render Slide 4: WHY DOES IT MATTER?"""
    title = slide.get("title", "WHY DOES IT MATTER?")
    body = slide.get("body", "")
    
    draw.text((60, 120), title, font=fonts["title_small"], fill=accent)
    
    # Impact text
    wrapped = textwrap.wrap(body, width=50)
    y = 200
    for line in wrapped[:5]:
        draw.text((60, y), line, font=fonts["body"], fill=BRAND_COLORS["text_primary"])
        y += 38
    
    # Before → After comparison
    y += 40
    draw.text((60, y), "IMPACT", font=fonts["badge"], fill=accent)
    y += 40
    
    impacts = [
        ("Developers", "Hours saved per week", "2h → 15min"),
        ("Business", "Time to production", "Weeks → Days"),
        ("Users", "Experience quality", "Basic → Pro"),
    ]
    
    for label, metric, change in impacts:
        draw.text((80, y), label, font=fonts["mono_large"], fill=BRAND_COLORS["text_secondary"])
        draw.text((320, y), metric, font=fonts["body"], fill=BRAND_COLORS["text_primary"])
        draw.text((680, y), change, font=fonts["mono_large"], fill=BRAND_COLORS["accent_green"])
        y += 50

def render_real_world_example_slide(draw: ImageDraw.Draw, fonts: Dict, slide: Dict, accent: str, category: str, date: str):
    """Render Slide 5: REAL-WORLD EXAMPLE"""
    title = slide.get("title", "REAL-WORLD EXAMPLE")
    body = slide.get("body", "")
    
    draw.text((60, 120), title, font=fonts["title_small"], fill=accent)
    
    # Code/Workflow block
    y = 200
    draw_rounded_rect(draw, (60, y, CANVAS_W - 60, y + 420), 16, 
                      "#0D1117", BRAND_COLORS["border"], 1)
    
    # Code header
    draw_rounded_rect(draw, (60, y, CANVAS_W - 60, y + 48), 16, 
                      BRAND_COLORS["bg_card"], BRAND_COLORS["border"], 1)
    draw.text((80, y + 10), "python", font=fonts["badge"], fill=BRAND_COLORS["accent_cyan"])
    draw.text((CANVAS_W - 80, y + 10), "Copy", font=fonts["badge"], fill=BRAND_COLORS["text_secondary"], anchor="ra")
    
    # Code content
    code_lines = body.split("\n")
    code_y = y + 70
    for line in code_lines[:12]:
        if line.strip():
            draw.text((90, code_y), line, font=fonts["mono"], fill=BRAND_COLORS["text_primary"])
        code_y += 28

def render_takeaway_cta_slide(draw: ImageDraw.Draw, fonts: Dict, slide: Dict, accent: str, category: str, date: str):
    """Render Slide 6: TAKEAWAY + CTA"""
    title = slide.get("title", "KEY TAKEAWAYS")
    body = slide.get("body", [])
    
    draw.text((60, 120), title, font=fonts["title_small"], fill=accent)
    
    # Three key takeaways
    y = 220
    for i, takeaway in enumerate(body[:3]):
        # Number circle
        num_x, num_y = 80, y
        draw_rounded_rect(draw, (num_x, num_y, num_x + 44, num_y + 44), 22, accent, None, 0)
        draw.text((num_x + 22, num_y + 22), str(i + 1), font=fonts["mono_large"], 
                  fill="#0B0E14", anchor="mm")
        
        # Takeaway text
        wrapped = textwrap.wrap(takeaway, width=48)
        for j, line in enumerate(wrapped):
            draw.text((150, y + j * 34), line, font=fonts["body"], fill=BRAND_COLORS["text_primary"])
        y += max(len(wrapped) * 34, 50) + 20
    
    # CTA Section
    y += 40
    # Save this post
    cta_y = y
    draw_rounded_rect(draw, (60, cta_y, CANVAS_W - 60, cta_y + 80), 16, 
                      BRAND_COLORS["accent_purple"], None, 0)
    draw.text((CANVAS_W // 2, cta_y + 40), "💾  Save this post for later", 
              font=fonts["subtitle"], fill="#0B0E14", anchor="mm")
    
    # Follow
    follow_y = cta_y + 110
    draw.text((CANVAS_W // 2, follow_y), "Follow for daily AI updates", 
              font=fonts["body"], fill=BRAND_COLORS["text_secondary"], anchor="mm")
    draw.text((CANVAS_W // 2, follow_y + 40), "@vijayakumarj_ai", 
              font=fonts["title_small"], fill=BRAND_COLORS["accent_purple"], anchor="mm")

SLIDE_RENDERERS = {
    "hook": render_hook_slide,
    "what_happened": render_what_happened_slide,
    "whats_new": render_whats_new_slide,
    "why_matters": render_why_matters_slide,
    "real_world_example": render_real_world_example_slide,
    "takeaway_cta": render_takeaway_cta_slide,
}

def render_carousel(carousel: Dict, output_dir: Path) -> List[Path]:
    """Render all 6 slides of the carousel."""
    slides = carousel.get("slides", [])
    if len(slides) != 6:
        raise ValueError(f"Expected 6 slides, got {len(slides)}")
    
    fonts = get_fonts()
    category = carousel.get("category", "AI NEWS")
    date = carousel.get("date", "")
    source = carousel.get("source", "")
    
    output_paths = []
    
    for i, slide in enumerate(slides):
        slide_num = i + 1
        slide_type = slide.get("type", "hook")
        accent = SLIDE_COLORS.get(slide_type, BRAND_COLORS["accent_purple"])
        
        # Create image
        img = Image.new("RGBA", (CANVAS_W, CANVAS_H), BRAND_COLORS["bg_dark"])
        
        # Background effects
        draw_gradient_bg(img, BRAND_COLORS["bg_dark"], BRAND_COLORS["bg_card"])
        draw_dots_pattern(img)
        draw_glow_orb(img, accent, CANVAS_W - 100, 100, 300)
        
        draw = ImageDraw.Draw(img)
        
        # Chrome bar
        draw_chrome_bar(draw, fonts, slide_num, 6, category)
        
        # Render slide content
        renderer = SLIDE_RENDERERS.get(slide_type, render_hook_slide)
        renderer(draw, fonts, slide, accent, category, date)
        
        # Footer
        draw_footer(draw, fonts, accent)
        
        # Save
        safe_title = "".join(c for c in carousel.get("headline", "carousel") if c.isalnum() or c in " -_").strip()[:30]
        safe_title = safe_title.replace(" ", "_")
        output_path = output_dir / f"carousel_{safe_title}_{slide_num:02d}.jpg"
        img.convert("RGB").save(output_path, "JPEG", quality=95)
        output_paths.append(output_path)
        print(f"✅ Rendered slide {slide_num}/6: {output_path.name}")
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Render AI News Carousel")
    parser.add_argument("--carousel-json", required=True, help="Path to carousel JSON")
    parser.add_argument("--output-dir", default="output/social_images", help="Output directory")
    args = parser.parse_args()
    
    carousel_path = Path(args.carousel_json)
    if not carousel_path.exists():
        print(f"❌ Carousel JSON not found: {carousel_path}")
        sys.exit(1)
    
    with open(carousel_path) as f:
        carousel = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        paths = render_carousel(carousel, output_dir)
        print(f"\n✅ Carousel rendered: {len(paths)} slides")
        for p in paths:
            print(f"   {p}")
    except Exception as e:
        print(f"❌ Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()