#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_decorator_images.py — Generate educational images using decorator_flow_template
for Instagram (4:5), Facebook (16:9), and send to Telegram for review.
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright
import requests

# ── Config ──────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else ""

TEMPLATE_ROOT = Path(__file__).parent / "decorator_flow_template"
TEMPLATES_DIR = TEMPLATE_ROOT / "templates"
ASSETS_DIR = TEMPLATE_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
ICONS_DIR = ASSETS_DIR / "icons"

# Canvas sizes
INSTAGRAM_W, INSTAGRAM_H = 1080, 1350  # 4:5
FACEBOOK_W, FACEBOOK_H = 1200, 628     # 1.91:1 (FB link post)

# Topics to rotate through
DEFAULT_TOPICS = [
    {
        "topic": "Python Decorators",
        "filename": "decorators.py",
        "eyebrow": "CONCEPT WALKTHROUGH",
        "title": "How Python Decorators Work",
        "subtitle_pre": "",
        "subtitle_bold": "@decorator",
        "subtitle_post": " wraps a function without touching its source",
        "steps": [
            {"accent": "#C879E6", "icon": "function-square", "title": "Define Decorator Function", "desc": "Takes a function as an argument."},
            {"accent": "#5FB3F0", "icon": "layers", "title": "Define Wrapper Function (Inner)", "desc": "This function will replace the original function."},
            {"accent": "#86D9A0", "icon": "zap", "title": "Add Logic Before/After Call", "desc": "Wrapper executes additional code, then calls original function."},
            {"accent": "#E8B85C", "icon": "corner-down-right", "title": "Return Wrapper Function", "desc": "Decorator returns the newly created wrapper."},
            {"accent": "#58C7D6", "icon": "at-sign", "title": "Apply Decorator (@syntax)", "desc": "Place @decorator_name above the target function definition."},
            {"accent": "#F07178", "icon": "repeat-2", "title": "Original Function is Replaced", "desc": "The decorated function now refers to the wrapper."},
        ],
        "closing_line": "decorator chain complete — ready to use",
    },
    {
        "topic": "AWS Lambda",
        "filename": "lambda_handler.py",
        "eyebrow": "SERVERLESS ARCHITECTURE",
        "title": "AWS Lambda Execution Model",
        "subtitle_pre": "Event → ",
        "subtitle_bold": "Lambda Handler",
        "subtitle_post": " → Response",
        "steps": [
            {"accent": "#FF9900", "icon": "cloud", "title": "Event Trigger", "desc": "API Gateway, S3, DynamoDB, or EventBridge invokes Lambda."},
            {"accent": "#232F3E", "icon": "cpu", "title": "Cold Start / Warm Start", "desc": "New container spins up or reuses existing execution environment."},
            {"accent": "#00A8E8", "icon": "code", "title": "Handler Execution", "desc": "Your code runs: event, context → response (or error)."},
            {"accent": "#D13212", "icon": "database", "title": "Service Integration", "desc": "SDK calls to DynamoDB, S3, SNS, SQS, etc."},
            {"accent": "#7AA116", "icon": "check-circle", "title": "Return Response", "desc": "JSON serializable response sent back to invoker."},
        ],
        "closing_line": "serverless function ready — scales to zero",
    },
    {
        "topic": "RAG Pipeline",
        "filename": "rag_pipeline.py",
        "eyebrow": "AI ARCHITECTURE",
        "title": "Retrieval-Augmented Generation Pipeline",
        "subtitle_pre": "Query → ",
        "subtitle_bold": "Embed → Retrieve → Generate",
        "subtitle_post": " → Answer",
        "steps": [
            {"accent": "#C879E6", "icon": "search", "title": "Query Embedding", "desc": "User question embedded via text-embedding model."},
            {"accent": "#5FB3F0", "icon": "database", "title": "Vector Search", "desc": "Top-k similar chunks retrieved from vector DB."},
            {"accent": "#86D9A0", "icon": "file-text", "title": "Context Assembly", "desc": "Retrieved docs formatted into prompt context window."},
            {"accent": "#E8B85C", "icon": "bot", "title": "LLM Generation", "desc": "Generator model produces grounded answer with citations."},
            {"accent": "#58C7D6", "icon": "shield-check", "title": "Guardrails & Eval", "desc": "Fact-check, hallucination detection, safety filters."},
        ],
        "closing_line": "RAG pipeline complete — grounded & citable",
    },
    {
        "topic": "Kubernetes Pods",
        "filename": "pod.yaml",
        "eyebrow": "K8S FUNDAMENTALS",
        "title": "Kubernetes Pod Lifecycle",
        "subtitle_pre": "Pending → ",
        "subtitle_bold": "Running",
        "subtitle_post": " → Succeeded/Failed",
        "steps": [
            {"accent": "#326CE5", "icon": "circle", "title": "Pending", "desc": "Pod accepted, waiting for scheduler & image pull."},
            {"accent": "#009688", "icon": "play-circle", "title": "Container Creating", "desc": "Runtime pulls image, creates container, runs init."},
            {"accent": "#4CAF50", "icon": "check-circle-2", "title": "Running", "desc": "All containers healthy, serving traffic."},
            {"accent": "#FF9800", "icon": "alert-triangle", "title": "Terminating", "desc": "SIGTERM sent, grace period, then SIGKILL."},
            {"accent": "#F44336", "icon": "x-circle", "title": "Failed / Completed", "desc": "Non-zero exit = Failed, zero = Succeeded."},
        ],
        "closing_line": "pod lifecycle mastered — ready for deployments",
    },
    {
        "topic": "Docker Multi-stage",
        "filename": "Dockerfile",
        "eyebrow": "CONTAINER OPTIMIZATION",
        "title": "Docker Multi-Stage Builds",
        "subtitle_pre": "Builder → ",
        "subtitle_bold": "Runtime",
        "subtitle_post": " = Tiny Image",
        "steps": [
            {"accent": "#2496ED", "icon": "hammer", "title": "Build Stage", "desc": "Full SDK image: compile, test, install deps."},
            {"accent": "#0DB7ED", "icon": "package", "title": "Copy Artifacts", "desc": "COPY --from=builder /app/dist /app — only outputs."},
            {"accent": "#00D9A5", "icon": "box", "title": "Runtime Stage", "desc": "Minimal base (distroless/alpine): copy + run."},
            {"accent": "#7B68EE", "icon": "shield", "title": "Security Hardening", "desc": "Non-root user, read-only fs, drop capabilities."},
            {"accent": "#FF6B6B", "icon": "trending-up", "title": "Result", "desc": "90%+ size reduction, faster deploy, smaller attack surface."},
        ],
        "closing_line": "multi-stage build ready — ship less, run faster",
    },
]


def font_data_uri(filename: str) -> str:
    """Encode font as base64 data URI."""
    path = FONTS_DIR / filename
    if path.exists():
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:font/woff2;base64,{b64}"
    # Fallback to Google Fonts CDN
    return ""


def icon_svg(name: str) -> str:
    """Read lucide icon SVG."""
    path = ICONS_DIR / f"{name}.svg"
    if path.exists():
        raw = path.read_text()
        lines = [l for l in raw.splitlines() if not l.strip().startswith("<!--")]
        svg = "\n".join(lines)
        svg = svg.replace(f'class="lucide lucide-{name}"', "")
        return svg
    return ""


def get_fonts_dict():
    """Get all font data URIs."""
    fonts = {}
    for font_file in FONTS_DIR.glob("*.woff2"):
        key = font_file.stem.replace("-latin", "").replace("-normal", "").replace("-", "")
        fonts[key] = font_data_uri(font_file.name)
    return fonts


def render_image(template_name: str, context: dict, output_path: Path, width: int, height: int):
    """Render HTML template to PNG using Playwright."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    tpl = env.get_template(template_name)
    html = tpl.render(**context)
    
    # Write HTML for debugging
    html_path = output_path.with_suffix(".html")
    html_path.write_text(html)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
        page.goto(f"file://{html_path.absolute()}")
        page.wait_for_timeout(200)
        page.screenshot(path=str(output_path), type="png")
        browser.close()
    
    print(f"✅ Rendered: {output_path}")
    return output_path


def generate_instagram_post(topic_data: dict, output_dir: Path) -> Path:
    """Generate Instagram 4:5 image."""
    fonts = get_fonts_dict()
    
    # Add icon SVGs to steps
    for i, step in enumerate(topic_data["steps"]):
        step["icon_svg"] = icon_svg(step["icon"])
    
    arrow_svg = icon_svg("arrow-down")
    check_svg = icon_svg("check-circle-2")
    
    context = {
        "canvas_w": INSTAGRAM_W,
        "canvas_h": INSTAGRAM_H,
        "fonts": fonts,
        "filename": topic_data["filename"],
        "topic": "PYTHON",
        "slide_current": "01",
        "slide_total": "01",
        "eyebrow": topic_data["eyebrow"],
        "title": topic_data["title"],
        "subtitle_pre": topic_data["subtitle_pre"],
        "subtitle_bold": topic_data["subtitle_bold"],
        "subtitle_post": topic_data["subtitle_post"],
        "steps": topic_data["steps"],
        "arrow_svg": arrow_svg,
        "check_svg": check_svg,
        "closing_line": topic_data["closing_line"],
    }
    
    output_path = output_dir / f"instagram_{topic_data['topic'].lower().replace(' ', '_')}.png"
    return render_image("decorator_flow.html.j2", context, output_path, INSTAGRAM_W, INSTAGRAM_H)


def generate_facebook_post(topic_data: dict, output_dir: Path) -> Path:
    """Generate Facebook 1.91:1 image."""
    fonts = get_fonts_dict()
    
    # Add icon SVGs to steps
    for i, step in enumerate(topic_data["steps"]):
        step["icon_svg"] = icon_svg(step["icon"])
    
    arrow_svg = icon_svg("arrow-down")
    check_svg = icon_svg("check-circle-2")
    
    # Facebook uses slightly different layout (wider)
    context = {
        "canvas_w": FACEBOOK_W,
        "canvas_h": FACEBOOK_H,
        "fonts": fonts,
        "filename": topic_data["filename"],
        "topic": "PYTHON",
        "slide_current": "01",
        "slide_total": "01",
        "eyebrow": topic_data["eyebrow"],
        "title": topic_data["title"],
        "subtitle_pre": topic_data["subtitle_pre"],
        "subtitle_bold": topic_data["subtitle_bold"],
        "subtitle_post": topic_data["subtitle_post"],
        "steps": topic_data["steps"],
        "arrow_svg": arrow_svg,
        "check_svg": check_svg,
        "closing_line": topic_data["closing_line"],
    }
    
    output_path = output_dir / f"facebook_{topic_data['topic'].lower().replace(' ', '_')}.png"
    return render_image("decorator_flow.html.j2", context, output_path, FACEBOOK_W, FACEBOOK_H)


def send_image_to_telegram(image_path: Path, caption: str = "") -> bool:
    """Send an image to Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured — skipping")
        return False

    try:
        with open(image_path, "rb") as f:
            files = {"photo": f}
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption[:1024],
                "parse_mode": "HTML",
            }
            resp = requests.post(f"{TELEGRAM_BASE_URL}/sendPhoto", data=data, files=files, timeout=30)
            resp.raise_for_status()
            print(f"📱 Telegram image sent: {image_path.name}")
            return True
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")
        return False


def send_telegram_message(message: str, emoji: str = "ℹ️"):
    """Send a plain notification to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"{TELEGRAM_BASE_URL}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": f"{emoji} {message}", "parse_mode": "HTML"},
            timeout=15
        )
    except Exception as e:
        print(f"Telegram notify failed: {e}")


def load_topic_index() -> int:
    """Load current topic index."""
    state_file = Path(__file__).parent / ".topic_index.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                return json.load(f).get("index", 0)
        except Exception:
            pass
    return 0


def save_topic_index(index: int):
    """Save topic index."""
    state_file = Path(__file__).parent / ".topic_index.json"
    try:
        with open(state_file, "w") as f:
            json.dump({"index": index}, f)
    except Exception as e:
        print(f"⚠️ Failed to save topic index: {e}")


def get_next_topic() -> dict:
    """Get next topic in round-robin."""
    index = load_topic_index()
    topic = DEFAULT_TOPICS[index % len(DEFAULT_TOPICS)]
    save_topic_index((index + 1) % len(DEFAULT_TOPICS))
    return topic


def main():
    parser = argparse.ArgumentParser(description="Generate decorator flow images for social media")
    parser.add_argument("--now", action="store_true", help="Run immediately")
    parser.add_argument("--dry-run", action="store_true", help="Preview without posting to Telegram")
    parser.add_argument("--topic", type=str, help="Specific topic to generate")
    args = parser.parse_args()

    if not args.now and not args.dry_run:
        print("Usage: python generate_decorator_images.py --now       # Generate and send to Telegram")
        print("       python generate_decorator_images.py --dry-run   # Generate only")
        print("       python generate_decorator_images.py --now --topic 'Python Decorators'")
        sys.exit(1)

    # Get topic
    if args.topic:
        topic_data = next((t for t in DEFAULT_TOPICS if t["topic"].lower() == args.topic.lower()), None)
        if not topic_data:
            print(f"❌ Topic not found: {args.topic}")
            sys.exit(1)
    else:
        topic_data = get_next_topic()

    print(f"🎨 Generating images for: {topic_data['topic']}")

    output_dir = Path(__file__).parent / "output" / "social_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate Instagram image (4:5)
        ig_path = generate_instagram_post(topic_data, output_dir)
        
        # Generate Facebook image (1.91:1)
        fb_path = generate_facebook_post(topic_data, output_dir)

        if args.dry_run:
            print(f"\n✅ DRY RUN COMPLETE")
            print(f"   Instagram: {ig_path}")
            print(f"   Facebook:  {fb_path}")
            return

        # Send to Telegram for review
        caption = f"📚 <b>{topic_data['topic']}</b>\n\nReady for review. Approve for posting?\n\nFollow @Vijayakumarj_ai for more!"
        
        send_image_to_telegram(ig_path, caption)
        send_image_to_telegram(fb_path, caption)
        
        send_telegram_message(
            f"✅ Images generated for: {topic_data['topic']}\n"
            f"Instagram (4:5): {ig_path.name}\n"
            f"Facebook (1.91:1): {fb_path.name}\n\nReply to approve for posting.",
            emoji="🤖"
        )

        print("\n✅ Generation complete. Images sent to Telegram for review.")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()