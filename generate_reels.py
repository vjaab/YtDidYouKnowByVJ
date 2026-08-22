#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_reels.py — Generate Instagram Reels/Stories from decorator flow content.
"""

import os
import sys
import json
import argparse
import base64
import subprocess
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

TEMPLATE_ROOT = Path(__file__).parent / "decorator_flow_template"
TEMPLATES_DIR = TEMPLATE_ROOT / "templates"
ASSETS_DIR = TEMPLATE_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
ICONS_DIR = ASSETS_DIR / "icons"

# Reels: 9:16 (1080x1920)
REELS_W, REELS_H = 1080, 1920
# Stories: 9:16 (1080x1920) with safe zones
STORIES_W, STORIES_H = 1080, 1920

def font_data_uri(filename: str) -> str:
    path = FONTS_DIR / filename
    if path.exists():
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:font/woff2;base64,{b64}"
    return ""

def icon_svg(name: str) -> str:
    path = ICONS_DIR / f"{name}.svg"
    if path.exists():
        raw = path.read_text()
        lines = [l for l in raw.splitlines() if not l.strip().startswith("<!--")]
        svg = "\n".join(lines)
        svg = svg.replace(f'class="lucide lucide-{name}"', "")
        return svg
    return ""

def get_fonts_dict():
    fonts = {}
    for font_file in FONTS_DIR.glob("*.woff2"):
        key = font_file.stem.replace("-latin", "").replace("-normal", "").replace("-", "")
        fonts[key] = font_data_uri(font_file.name)
    return fonts

def render_reel(topic_data: dict, output_dir: Path, slide_index: int = 0) -> Path:
    """Generate a single Reel frame (9:16)."""
    fonts = get_fonts_dict()
    
    for step in topic_data["steps"]:
        step["icon_svg"] = icon_svg(step["icon"])
    
    arrow_svg = icon_svg("arrow-down")
    check_svg = icon_svg("check-circle-2")
    
    context = {
        "canvas_w": REELS_W,
        "canvas_h": REELS_H,
        "fonts": fonts,
        "filename": topic_data["filename"],
        "topic": "PYTHON",
        "slide_current": str(slide_index + 1),
        "slide_total": str(len(topic_data["steps"])),
        "eyebrow": topic_data["eyebrow"],
        "title": topic_data["title"],
        "subtitle_pre": topic_data["subtitle_pre"],
        "subtitle_bold": topic_data["subtitle_bold"],
        "subtitle_post": topic_data["subtitle_post"],
        "steps": topic_data["steps"],
        "arrow_svg": arrow_svg,
        "check_svg": check_svg,
        "closing_line": topic_data["closing_line"],
        "is_reel": True,
    }
    
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    tpl = env.get_template("decorator_flow.html.j2")
    html = tpl.render(**context)
    
    html_path = output_dir / f"reel_{topic_data['topic'].lower().replace(' ', '_')}_{slide_index}.html"
    html_path.write_text(html)
    
    output_path = output_dir / f"reel_{topic_data['topic'].lower().replace(' ', '_')}_{slide_index}.png"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": REELS_W, "height": REELS_H}, device_scale_factor=2)
        page.goto(f"file://{html_path.absolute()}")
        page.wait_for_timeout(300)
        page.screenshot(path=str(output_path), type="png")
        browser.close()
    
    print(f"✅ Reel frame rendered: {output_path}")
    return output_path

def render_story(topic_data: dict, output_dir: Path, step_index: int = 0) -> Path:
    """Generate a single Story frame (9:16 with safe zones)."""
    fonts = get_fonts_dict()
    
    step = topic_data["steps"][step_index]
    step["icon_svg"] = icon_svg(step["icon"])
    
    arrow_svg = icon_svg("arrow-down")
    check_svg = icon_svg("check-circle-2")
    
    context = {
        "canvas_w": STORIES_W,
        "canvas_h": STORIES_H,
        "fonts": fonts,
        "filename": topic_data["filename"],
        "topic": "PYTHON",
        "slide_current": str(step_index + 1),
        "slide_total": str(len(topic_data["steps"])),
        "eyebrow": topic_data["eyebrow"],
        "title": step["title"],
        "subtitle_pre": "",
        "subtitle_bold": "",
        "subtitle_post": step["desc"],
        "steps": [step],
        "arrow_svg": arrow_svg,
        "check_svg": check_svg,
        "closing_line": topic_data["closing_line"] if step_index == len(topic_data["steps"]) - 1 else "",
        "is_story": True,
        "step_number": step_index + 1,
        "total_steps": len(topic_data["steps"]),
    }
    
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    tpl = env.get_template("decorator_flow.html.j2")
    html = tpl.render(**context)
    
    html_path = output_dir / f"story_{topic_data['topic'].lower().replace(' ', '_')}_{step_index}.html"
    html_path.write_text(html)
    
    output_path = output_dir / f"story_{topic_data['topic'].lower().replace(' ', '_')}_{step_index}.png"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": STORIES_W, "height": STORIES_H}, device_scale_factor=2)
        page.goto(f"file://{html_path.absolute()}")
        page.wait_for_timeout(300)
        page.screenshot(path=str(output_path), type="png")
        browser.close()
    
    print(f"✅ Story frame rendered: {output_path}")
    return output_path

def generate_reel_video(topic_data: dict, output_dir: Path) -> Path:
    """Generate a Reel video (MP4) from frames using ffmpeg."""
    frames = []
    for i, step in enumerate(topic_data["steps"]):
        frame = render_reel(topic_data, output_dir, i)
        frames.append(frame)
    
    # Add title frame
    title_frame = render_reel(topic_data, output_dir, -1)
    frames.insert(0, title_frame)
    
    # Add closing frame
    closing_frame = render_reel(topic_data, output_dir, len(topic_data["steps"]))
    frames.append(closing_frame)
    
    output_video = output_dir / f"reel_{topic_data['topic'].lower().replace(' ', '_')}.mp4"
    
    list_file = output_dir / "frame_list.txt"
    with open(list_file, "w") as f:
        for frame in frames:
            f.write(f"file '{frame.absolute()}'\nduration 3\n")
        f.write(f"file '{frames[-1].absolute()}'\n")
    
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-vf", "fps=30,format=yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        str(output_video)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        print(f"✅ Reel video generated: {output_video}")
        return output_video
    except subprocess.CalledProcessError as e:
        print(f"⚠️ FFmpeg failed: {e.stderr.decode() if e.stderr else e}")
        return None
    except FileNotFoundError:
        print("⚠️ FFmpeg not installed. Skipping video generation.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate Reels/Stories")
    parser.add_argument("--topic", required=True, help="Topic name")
    parser.add_argument("--format", choices=["reel", "story", "reel-video", "all"], default="all")
    parser.add_argument("--output-dir", default="output/social_images", help="Output directory")
    args = parser.parse_args()
    
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_decorator_images import DEFAULT_TOPICS
    
    topic_data = next((t for t in DEFAULT_TOPICS if t["topic"].lower() == args.topic.lower()), None)
    if not topic_data:
        print(f"❌ Topic not found: {args.topic}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.format in ["reel", "all"]:
            print(f"🎬 Generating Reel frames for: {topic_data['topic']}")
            for i in range(len(topic_data["steps"])):
                render_reel(topic_data, output_dir, i)
        
        if args.format in ["story", "all"]:
            print(f"📱 Generating Story frames for: {topic_data['topic']}")
            for i in range(len(topic_data["steps"])):
                render_story(topic_data, output_dir, i)
        
        if args.format in ["reel-video", "all"]:
            print(f"🎥 Generating Reel video for: {topic_data['topic']}")
            generate_reel_video(topic_data, output_dir)
        
        print("\n✅ Generation complete!")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()