#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
carousel_renderer_html.py — HTML/CSS + Playwright renderer for AI News carousel slides.
Generates 6 slides at 1080x1350 with light, card-based editorial aesthetic.
"""

import os
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Optional
from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

TEMPLATE_DIR = Path(__file__).parent / "carousel_templates"
LAYOUTS_DIR = TEMPLATE_DIR / "layouts"
PARTIALS_DIR = TEMPLATE_DIR / "partials"
CSS_FILE = TEMPLATE_DIR / "css" / "styles.css"

CANVAS_W, CANVAS_H = 1080, 1350

# Accent color mapping per slide type
SLIDE_ACCENTS = {
    "hook": {
        "color": "#EC4899",
        "light": "#FDF2F8",
    },
    "what_happened": {
        "color": "#0066FF",
        "light": "#DBEAFE",
    },
    "whats_new": {
        "color": "#059669",
        "light": "#D1FAE5",
    },
    "why_matters": {
        "color": "#D97706",
        "light": "#FEF3C7",
    },
    "real_world_example": {
        "color": "#FF4B4B",
        "light": "#FEF2F2",
    },
    "takeaway_cta": {
        "color": "#FF2E93",
        "light": "#FDF4FF",
    },
}

# Default metrics for why_matters slide
DEFAULT_METRICS = [
    {"label": "Developers", "value": "Hours saved", "delta": "2h → 15min"},
    {"label": "Business", "value": "Time to prod", "delta": "Weeks → Days"},
    {"label": "Users", "value": "Experience", "delta": "Basic → Pro"},
]

# Default before/after text for whats_new
DEFAULT_BEFORE = "Manual setup\nComplex configuration\nSlow iteration"
DEFAULT_AFTER = "One-click deploy\nAuto-configuration\nInstant feedback"


def load_css() -> str:
    """Load the CSS content."""
    return CSS_FILE.read_text(encoding="utf-8")


def get_accent_colors(slide_type: str) -> Dict[str, str]:
    """Get accent colors for a slide type."""
    return SLIDE_ACCENTS.get(slide_type, SLIDE_ACCENTS["hook"])


def parse_code_lines(body: str) -> List[str]:
    """Parse code body into highlighted lines."""
    lines = body.strip().split("\n")
    highlighted = []
    for line in lines:
        # Simple syntax highlighting hints (will be styled via CSS classes)
        highlighted.append(line)
    return highlighted[:15]  # Limit lines


def build_slide_context(carousel: Dict, slide: Dict, slide_num: int, total_slides: int = 6) -> Dict[str, Any]:
    """Build rendering context for a single slide."""
    slide_type = slide.get("type", "hook")
    accent = get_accent_colors(slide_type)

    context = {
        "slide_data": slide,
        "slide_current": slide_num,
        "slide_total": total_slides,
        "category": carousel.get("category", "AI NEWS"),
        "date": carousel.get("date", ""),
        "source": carousel.get("source", ""),
        "accent_color": accent["color"],
        "accent_color_light": accent["light"],
        "css_content": load_css(),
    }

    # Slide-specific additions
    if slide_type == "why_matters":
        context["metrics"] = DEFAULT_METRICS
    elif slide_type == "whats_new":
        context["before_text"] = DEFAULT_BEFORE
        context["after_text"] = DEFAULT_AFTER
    elif slide_type == "real_world_example":
        context["code_lines"] = parse_code_lines(slide.get("body", ""))

    return context


def render_slide(env: Environment, template_name: str, context: Dict, output_path: Path, slide_num: int) -> Path:
    """Render a single slide to PNG via Playwright."""
    template = env.get_template(template_name)
    html = template.render(**context)

    # Write HTML for debugging
    html_path = output_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": CANVAS_W, "height": CANVAS_H},
            device_scale_factor=2,
        )
        page.goto(f"file://{html_path.absolute()}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(300)  # Let fonts/animations settle
        page.screenshot(path=str(output_path), type="png")
        browser.close()

    print(f"✅ Rendered slide {slide_num}/6: {output_path.name}")
    return output_path


def render_carousel(carousel: Dict, output_dir: Path) -> List[Path]:
    """Render all 6 slides of the carousel using HTML/CSS + Playwright."""
    slides = carousel.get("slides", [])
    if len(slides) != 6:
        raise ValueError(f"Expected 6 slides, got {len(slides)}")

    # Setup Jinja2 environment with partials
    env = Environment(
        loader=FileSystemLoader([str(TEMPLATE_DIR), str(LAYOUTS_DIR), str(PARTIALS_DIR)]),
        autoescape=True,
    )

    output_paths = []
    safe_headline = "".join(c for c in carousel.get("headline", "carousel") if c.isalnum() or c in " -_").strip()[:30]
    safe_headline = safe_headline.replace(" ", "_")

    for i, slide in enumerate(slides):
        slide_num = i + 1
        slide_type = slide.get("type", "hook")

        # Determine partial template
        partial_map = {
            "hook": "hook.html",
            "what_happened": "what_happened.html",
            "whats_new": "whats_new.html",
            "why_matters": "why_matters.html",
            "real_world_example": "real_world_example.html",
            "takeaway_cta": "takeaway_cta.html",
        }
        partial = partial_map.get(slide_type, "hook.html")

        context = build_slide_context(carousel, slide, slide_num)

        output_path = output_dir / f"carousel_{safe_headline}_{slide_num:02d}.jpg"
        render_slide(env, partial, context, output_path, slide_num)
        output_paths.append(output_path)

    return output_paths


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Render AI News Carousel (HTML/CSS)")
    parser.add_argument("--carousel-json", required=True, help="Path to carousel JSON")
    parser.add_argument("--output-dir", default="output/social_images", help="Output directory")
    args = parser.parse_args()

    carousel_path = Path(args.carousel_json)
    if not carousel_path.exists():
        print(f"❌ Carousel JSON not found: {carousel_path}")
        exit(1)

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
        exit(1)


if __name__ == "__main__":
    main()