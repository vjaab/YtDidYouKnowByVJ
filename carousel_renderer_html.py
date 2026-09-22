#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
carousel_renderer_html.py — Dynamic HTML/CSS + Playwright renderer for AI News & Engineering carousels.
Supports 7 visual themes, 18+ visual layouts, syntax highlighting, and variable slide counts (5-8 slides).
Uses a persistent Playwright browser session for fast rendering.
"""

import os
import sys
import json
import re
import html
from pathlib import Path
from typing import Dict, List, Any, Optional
from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

TEMPLATE_DIR = Path(__file__).parent / "carousel_templates"
LAYOUTS_DIR = TEMPLATE_DIR / "layouts"
PARTIALS_DIR = TEMPLATE_DIR / "partials"
CSS_FILE = TEMPLATE_DIR / "css" / "styles.css"

DEFAULT_CANVAS_W, DEFAULT_CANVAS_H = 1080, 1350

# Layout to Partial template mapping
LAYOUT_TEMPLATE_MAP = {
    # Opening / Hooks
    "hero_hook": "hero_hook.html",
    "big_number": "big_number.html",
    "scenario_question": "scenario_question.html",
    "hook": "hook.html",
    
    # Systems & Flows
    "architecture_diagram": "architecture_diagram.html",
    "process_flow": "process_flow.html",
    "input_output": "input_output.html",
    "what_happened": "what_happened.html",
    
    # Comparisons
    "before_after": "before_after.html",
    "common_mistake": "common_mistake.html",
    "side_by_side": "side_by_side.html",
    "whats_new": "whats_new.html",
    
    # Code & Practical Examples
    "code_block": "code_block.html",
    "code_breakdown": "code_block.html",
    "real_world_scenario": "real_world_scenario.html",
    "real_world_example": "real_world_example.html",
    
    # Metrics & Value
    "metrics_cards": "metrics_cards.html",
    "why_matters": "why_matters.html",
    "checklist": "checklist.html",
    
    # Quizzes & Interactive
    "quiz_choice": "quiz_choice.html",
    "quiz_predict_output": "quiz_predict_output.html",
    
    # Takeaways & CTA
    "takeaway": "takeaway.html",
    "takeaway_cta": "takeaway_cta.html",
}


def load_css(canvas_width: int = DEFAULT_CANVAS_W, canvas_height: int = DEFAULT_CANVAS_H) -> str:
    """Load the master CSS styles with dynamic canvas dimensions."""
    if CSS_FILE.exists():
        css = CSS_FILE.read_text(encoding="utf-8")
        # Replace hardcoded canvas dimensions with dynamic values
        css = css.replace("--canvas-w: 1080px;", f"--canvas-w: {canvas_width}px;")
        css = css.replace("--canvas-h: 1350px;", f"--canvas-h: {canvas_height}px;")
        css = css.replace("width: 1080px;\n  height: 1350px;", f"width: {canvas_width}px;\n  height: {canvas_height}px;")
        css = css.replace("width: 1080px;\n  height: 1350px;", f"width: {canvas_width}px;\n  height: {canvas_height}px;")
        css = css.replace(".carousel-canvas {\n  position: relative;\n  width: 1080px;\n  height: 1350px;", f".carousel-canvas {{\n  position: relative;\n  width: {canvas_width}px;\n  height: {canvas_height}px;")
        return css
    return ""


def highlight_code_syntax(code_text: str) -> str:
    """Lightweight single-pass syntax highlighter converting code into styled HTML spans."""
    if not code_text:
        return ""
    
    # Token regexes
    tok_comment = r'(?P<comment>#[^\n]*|//[^\n]*)'
    tok_string = r'(?P<string>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')'
    keywords = [
        'import', 'from', 'as', 'def', 'class', 'return',
        'async', 'await', 'if', 'else', 'elif', 'try',
        'except', 'finally', 'with', 'for', 'in', 'while',
        'const', 'let', 'var', 'function', 'true', 'false',
        'None', 'True', 'False'
    ]
    tok_keyword = r'(?P<keyword>\b(?:' + '|'.join(keywords) + r')\b)'
    tok_function = r'(?P<function>\b[a-zA-Z_][a-zA-Z0-9_]*(?=\())'
    tok_number = r'(?P<number>\b\d+(?:\.\d+)?\b)'

    master_regex = re.compile('|'.join([tok_comment, tok_string, tok_keyword, tok_function, tok_number]))

    result = []
    last_idx = 0
    for match in master_regex.finditer(code_text):
        start, end = match.span()
        if start > last_idx:
            result.append(html.escape(code_text[last_idx:start]))
        kind = match.lastgroup
        val = match.group(kind)
        escaped_val = html.escape(val)
        if kind == 'comment':
            result.append(f'<span class="com">{escaped_val}</span>')
        elif kind == 'string':
            result.append(f'<span class="str">{escaped_val}</span>')
        elif kind == 'keyword':
            result.append(f'<span class="kw">{escaped_val}</span>')
        elif kind == 'function':
            result.append(f'<span class="fn">{escaped_val}</span>')
        elif kind == 'number':
            result.append(f'<span class="num">{escaped_val}</span>')
        else:
            result.append(escaped_val)
        last_idx = end

    if last_idx < len(code_text):
        result.append(html.escape(code_text[last_idx:]))

    return ''.join(result)


def build_slide_context(
    carousel: Dict,
    slide: Dict,
    slide_num: int,
    total_slides: int,
    strategy: Optional[Dict] = None,
    css_content: str = "",
    canvas_width: int = DEFAULT_CANVAS_W,
    canvas_height: int = DEFAULT_CANVAS_H,
) -> Dict[str, Any]:
    """Build the Jinja2 template context for a single slide."""
    layout_type = slide.get("layout_type") or slide.get("type") or "hero_hook"
    
    theme_css_class = "theme--tech-dark"
    theme_colors = None
    brand = {"handle": "@vijayakumarj_ai", "name": "Vijayakumar J", "short_name": "VJ"}
    
    if strategy:
        theme_css_class = strategy.get("theme_css_class", theme_css_class)
        theme_colors = strategy.get("theme_colors")
        if "brand" in strategy:
            brand = strategy["brand"]
            
    # Process code if present
    highlighted_code = None
    if slide.get("code"):
        highlighted_code = highlight_code_syntax(slide["code"])
        
    context = {
        "slide_data": slide,
        "slide_current": slide_num,
        "slide_total": total_slides,
        "category": slide.get("eyebrow") or carousel.get("category", "AI ENGINEERING"),
        "date": carousel.get("date", ""),
        "source": carousel.get("source", ""),
        "theme_css_class": theme_css_class,
        "theme_colors": theme_colors,
        "brand": brand,
        "css_content": css_content,
        "highlighted_code": highlighted_code,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "deep_dive": slide.get("deep_dive"),
        "key_fact": slide.get("key_fact"),
        "why_it_matters": slide.get("why_it_matters"),
        "source_evidence": slide.get("source_evidence"),
    }
    
    # Compatibility mapping for legacy & dynamic templates
    if layout_type in ["why_matters", "metrics_cards"]:
        context["metrics"] = slide.get("metrics")
    if layout_type in ["whats_new", "before_after"]:
        context["before_text"] = slide.get("before_text")
        context["after_text"] = slide.get("after_text")
    if layout_type in ["real_world_example", "code_block"]:
        context["code_lines"] = slide.get("body", "").split("\n") if isinstance(slide.get("body"), str) else []
    if layout_type == "checklist":
        items = slide.get("items") or slide.get("checklist_items")
        if items:
            slide["checklist_items"] = items
            context["checklist_items"] = items
    if layout_type in ["takeaway", "takeaway_cta"]:
        takeaways = slide.get("takeaways") or slide.get("items")
        if takeaways:
            slide["takeaways"] = takeaways
            context["takeaways"] = takeaways
        
    return context


def get_template_for_layout(layout_type: str, env: Environment):
    """Find the best matching template for a given layout type."""
    template_name = LAYOUT_TEMPLATE_MAP.get(layout_type, "hero_hook.html")
    try:
        return env.get_template(template_name)
    except Exception:
        # Fallback to hook.html or hero_hook.html
        try:
            return env.get_template("hero_hook.html")
        except Exception:
            return env.get_template("hook.html")


def render_carousel(
    carousel: Dict,
    output_dir: Path,
    strategy: Optional[Dict] = None,
    canvas_width: int = DEFAULT_CANVAS_W,
    canvas_height: int = DEFAULT_CANVAS_H,
) -> List[Path]:
    """
    Render all slides of the carousel using HTML/CSS and a persistent Playwright browser.
    Supports dynamic 5-8 slides and visual strategy instructions.
    """
    slides = carousel.get("slides", [])
    if not slides:
        raise ValueError("Carousel contains no slides to render")

    total_slides = len(slides)

    # If no visual strategy passed, create one dynamically
    if not strategy:
        try:
            from visual_strategy import create_visual_strategy
            strategy = create_visual_strategy(carousel)
        except Exception as e:
            print(f"⚠️ Could not build visual strategy: {e}, using default styling")

    # If strategy provides slide layout hints, align slides
    if strategy and "slides" in strategy and len(strategy["slides"]) == total_slides:
        for i, s in enumerate(slides):
            strat_slide = strategy["slides"][i]
            if "layout_type" not in s:
                s["layout_type"] = strat_slide.get("layout_type")

    # Setup Jinja2 environment
    env = Environment(
        loader=FileSystemLoader([str(TEMPLATE_DIR), str(LAYOUTS_DIR), str(PARTIALS_DIR)]),
        autoescape=True,
    )

    css_content = load_css(canvas_width, canvas_height)
    output_paths = []
    safe_headline = "".join(c for c in carousel.get("headline", "carousel") if c.isalnum() or c in " -_").strip()[:30]
    safe_headline = safe_headline.replace(" ", "_")

    # Include canvas dimensions in filename to support multiple formats
    size_suffix = f"_{canvas_width}x{canvas_height}"

    # Prepare rendered HTML files for each slide
    slide_html_items = []
    for i, slide in enumerate(slides):
        slide_num = i + 1
        layout_type = slide.get("layout_type") or slide.get("type", "hero_hook")
        
        template = get_template_for_layout(layout_type, env)
        context = build_slide_context(carousel, slide, slide_num, total_slides, strategy, css_content, canvas_width, canvas_height)
        
        html_rendered = template.render(**context)
        html_path = output_dir / f"carousel_{safe_headline}_{slide_num:02d}{size_suffix}.html"
        html_path.write_text(html_rendered, encoding="utf-8")
        
        img_path = output_dir / f"carousel_{safe_headline}_{slide_num:02d}{size_suffix}.jpg"
        slide_html_items.append((slide_num, layout_type, html_path, img_path))

    # High-performance batch screenshot via single Playwright browser instance
    print(f"🚀 Launching Playwright browser to render {total_slides} slides at {canvas_width}x{canvas_height}...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": canvas_width, "height": canvas_height},
            device_scale_factor=2,
        )

        for slide_num, layout_type, html_path, img_path in slide_html_items:
            page.goto(f"file://{html_path.absolute()}")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(250)  # Font & rendering stabilization
            
            # Save as high-quality JPEG
            page.screenshot(path=str(img_path), type="jpeg", quality=95)
            output_paths.append(img_path)
            print(f"  ✅ Slide {slide_num}/{total_slides} [{layout_type}]: {img_path.name}")

        browser.close()

    # Persist carousel history to prevent repeating the same pattern soon
    if strategy:
        try:
            from carousel_history import record_carousel
            record_carousel(
                theme=strategy.get("visual_theme", "TECH_DARK"),
                layout_pattern=[s.get("layout_type") for s in strategy.get("slides", [])],
                topic=carousel.get("headline", ""),
                slide_count=total_slides,
            )
        except Exception as e:
            print(f"⚠️ Could not update carousel history: {e}")

    return output_paths


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Render AI News Carousel with Dynamic Themes & Layouts")
    parser.add_argument("--carousel-json", required=True, help="Path to carousel JSON")
    parser.add_argument("--output-dir", default="output/social_images", help="Output directory")
    parser.add_argument("--canvas-width", type=int, default=DEFAULT_CANVAS_W, help="Canvas width")
    parser.add_argument("--canvas-height", type=int, default=DEFAULT_CANVAS_H, help="Canvas height")
    args = parser.parse_args()

    carousel_path = Path(args.carousel_json)
    if not carousel_path.exists():
        print(f"❌ Carousel JSON not found: {carousel_path}")
        sys.exit(1)

    with open(carousel_path, "r", encoding="utf-8") as f:
        carousel = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        paths = render_carousel(carousel, output_dir, canvas_width=args.canvas_width, canvas_height=args.canvas_height)
        print(f"\n🎉 Successfully rendered {len(paths)} slides in {output_dir} at {args.canvas_width}x{args.canvas_height}")
        for p in paths:
            print(f"   {p}")
    except Exception as e:
        print(f"❌ Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()