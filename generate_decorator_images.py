#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_decorator_images.py — Generate educational images using decorator_flow_template
for Instagram (4:5), Facebook (1.91:1), and send to Telegram for review.
Single variant: modern
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
import datetime
import hashlib
import io
from PIL import Image

# Add parent directory to path for trending_engine import
sys.path.insert(0, str(Path(__file__).parent))

try:
    from trending_engine import fetch_all_trending_signals, compute_engagement_score
    TRENDING_ENGINE_AVAILABLE = True
except ImportError:
    TRENDING_ENGINE_AVAILABLE = False
    print("⚠️ trending_engine not available, using fallback topics")

# AI Image Generation configs
AI_IMAGE_PROVIDER = os.getenv("AI_IMAGE_PROVIDER", "template")  # template, dall-e-3, stable-diffusion, recraft
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
STABLE_DIFFUSION_API_URL = os.getenv("STABLE_DIFFUSION_API_URL", "")  # e.g., local SD WebUI or Replicate
RECRAFT_API_KEY = os.getenv("RECRAFT_API_KEY", "")

# ── GitHub Actions Output Helper ───────────────────────────────────────────────
def set_gha_output(key: str, value: str):
    """Set GitHub Actions output."""
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{key}={value}\n")
    print(f"{key}={value}")  # Also print for visibility

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

# Single variant configuration (modern)
VARIANT = {
    "name": "Modern Gradient",
    "bg_primary": "#0B0E14",
    "bg_secondary": "#12161F",
    "accent_glow": True,
    "dots_opacity": 0.05,
    "watermark_opacity": 0.025,
}

# Topics to rotate through (fallback when trending engine unavailable)
DEFAULT_TOPICS = [
    {
        "topic": "Python Decorators",
        "category": "python",
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
        "category": "aws",
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
        "category": "ai_ml",
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
        "category": "kubernetes",
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
        "category": "devops",
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


def get_variant_styles() -> dict:
    """Get CSS variables for the modern variant."""
    return VARIANT


def render_image(template_name: str, context: dict, output_path: Path, width: int, height: int):
    """Render HTML template to PNG using Playwright."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    tpl = env.get_template(template_name)
    html = tpl.render(**context)
    
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


def generate_ai_image_dalle3(prompt: str, output_path: Path, width: int, height: int) -> Path:
    """Generate image using OpenAI DALL-E 3."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    # DALL-E 3 supports specific sizes
    size_map = {
        (1080, 1350): "1024x1024",  # Instagram 4:5 - closest
        (1200, 628): "1792x1024",   # Facebook 1.91:1 - closest
    }
    size = size_map.get((width, height), "1024x1024")
    
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality="hd",
        n=1,
    )
    
    image_url = response.data[0].url
    img_response = requests.get(image_url, timeout=60)
    img_response.raise_for_status()
    
    img = Image.open(io.BytesIO(img_response.content))
    img = img.resize((width, height), Image.LANCZOS)
    img.save(output_path)
    
    print(f"✅ DALL-E 3 generated: {output_path}")
    return output_path


def generate_ai_image_stable_diffusion(prompt: str, output_path: Path, width: int, height: int) -> Path:
    """Generate image using Stable Diffusion (local WebUI API or Replicate)."""
    if not STABLE_DIFFUSION_API_URL:
        raise ValueError("STABLE_DIFFUSION_API_URL not set")
    
    payload = {
        "prompt": prompt,
        "negative_prompt": "text, watermark, signature, blurry, low quality, distorted",
        "width": width,
        "height": height,
        "steps": 30,
        "cfg_scale": 7,
        "sampler_name": "DPM++ 2M Karras",
    }
    
    response = requests.post(f"{STABLE_DIFFUSION_API_URL}/sdapi/v1/txt2img", json=payload, timeout=120)
    response.raise_for_status()
    
    import base64
    result = response.json()
    img_data = base64.b64decode(result["images"][0])
    img = Image.open(io.BytesIO(img_data))
    img.save(output_path)
    
    print(f"✅ Stable Diffusion generated: {output_path}")
    return output_path


def generate_ai_image_recraft(prompt: str, output_path: Path, width: int, height: int) -> Path:
    """Generate image using Recraft API."""
    if not RECRAFT_API_KEY:
        raise ValueError("RECRAFT_API_KEY not set")
    
    # Recraft supports specific aspect ratios
    aspect_ratio = f"{width}:{height}"
    
    headers = {
        "Authorization": f"Bearer {RECRAFT_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "style": "digital_illustration",
        "aspect_ratio": aspect_ratio,
    }
    
    response = requests.post("https://external.api.recraft.ai/v1/images/generations", 
                           json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    
    result = response.json()
    image_url = result["data"][0]["url"]
    
    img_response = requests.get(image_url, timeout=60)
    img_response.raise_for_status()
    
    img = Image.open(io.BytesIO(img_response.content))
    img = img.resize((width, height), Image.LANCZOS)
    img.save(output_path)
    
    print(f"✅ Recraft generated: {output_path}")
    return output_path


def build_ai_prompt(topic_data: dict, platform: str) -> str:
    """Build AI image generation prompt from topic data."""
    title = topic_data.get("title", "")
    category = topic_data.get("category", "")
    steps = topic_data.get("steps", [])
    closing = topic_data.get("closing_line", "")
    
    step_descriptions = " → ".join([s.get("desc", "") for s in steps[:3]])
    
    style_guides = {
        "instagram": "Instagram educational carousel style, 4:5 aspect, clean modern design, tech aesthetic, vibrant gradient backgrounds, minimal text space, professional typography",
        "facebook": "Facebook link post image, 1.91:1 aspect, eye-catching thumbnail, tech blog style, bold colors, clear visual hierarchy",
    }
    
    style = style_guides.get(platform, style_guides["instagram"])
    
    prompt = f"""Educational tech illustration for social media: "{title}"
Category: {category}
Key concepts: {step_descriptions}
Takeaway: {closing}
Style: {style}
No text, no watermarks, no logos, clean background, high quality, 4K"""
    
    return prompt


def generate_image_with_provider(topic_data: dict, output_path: Path, width: int, height: int, platform: str) -> Path:
    """Generate image using configured AI provider, fallback to template."""
    provider = AI_IMAGE_PROVIDER.lower()
    
    if provider == "template":
        return render_image("decorator_flow.html.j2", build_template_context(topic_data, width, height), output_path, width, height)
    
    prompt = build_ai_prompt(topic_data, platform)
    
    try:
        if provider == "dall-e-3":
            return generate_ai_image_dalle3(prompt, output_path, width, height)
        elif provider == "stable-diffusion":
            return generate_ai_image_stable_diffusion(prompt, output_path, width, height)
        elif provider == "recraft":
            return generate_ai_image_recraft(prompt, output_path, width, height)
        else:
            print(f"⚠️ Unknown AI_IMAGE_PROVIDER: {provider}, falling back to template")
            return render_image("decorator_flow.html.j2", build_template_context(topic_data, width, height), output_path, width, height)
    except Exception as e:
        print(f"⚠️ AI image generation failed ({provider}): {e}, falling back to template")
        return render_image("decorator_flow.html.j2", build_template_context(topic_data, width, height), output_path, width, height)


def build_template_context(topic_data: dict, width: int, height: int) -> dict:
    """Build context for template rendering (extracted from generate_images)."""
    fonts = get_fonts_dict()
    variant_styles = get_variant_styles()
    
    for step in topic_data["steps"]:
        step["icon_svg"] = icon_svg(step["icon"])
    
    arrow_svg = icon_svg("arrow-down")
    check_svg = icon_svg("check-circle-2")
    
    return {
        "fonts": fonts,
        "filename": topic_data["filename"],
        "topic": topic_data["category"].upper(),
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
        "variant": "modern",
        "variant_styles": variant_styles,
        "canvas_w": width,
        "canvas_h": height,
    }


def generate_images(topic_data: dict, output_dir: Path) -> tuple:
    """Generate Instagram and Facebook images using configured provider (template or AI)."""
    print(f"🎨 Generating images with provider: {AI_IMAGE_PROVIDER}")
    
    # Instagram 4:5
    ig_path = output_dir / f"instagram_{topic_data['topic'].lower().replace(' ', '_')}.png"
    generate_image_with_provider(topic_data, ig_path, INSTAGRAM_W, INSTAGRAM_H, "instagram")
    
    # Facebook 1.91:1
    fb_path = output_dir / f"facebook_{topic_data['topic'].lower().replace(' ', '_')}.png"
    generate_image_with_provider(topic_data, fb_path, FACEBOOK_W, FACEBOOK_H, "facebook")
    
    return ig_path, fb_path


def generate_caption(topic_data: dict, hashtags: str) -> str:
    """Generate caption for social media posts."""
    return f"""📚 {topic_data['topic']}

{topic_data['closing_line'].capitalize()}.

Follow @Vijayakumarj_ai for more tech concepts explained visually!

{hashtags}"""


def generate_poll_data(topic_data: dict, output_dir: Path) -> Path:
    """Generate poll/quiz data for Instagram Stories."""
    topic_key = topic_data["topic"].lower().replace(" ", "_").replace("-", "_")
    
    poll_templates = {
        "python_decorators": {
            "quiz": {"question": "What does @decorator do?", "options": ["Wraps a function", "Deletes a function", "Creates a class", "Imports a module"], "correct": 0},
            "poll": {"question": "How often do you use decorators?", "options": ["Daily", "Weekly", "Rarely", "Never heard of them"]},
        },
        "aws_lambda": {
            "quiz": {"question": "What triggers AWS Lambda?", "options": ["API Gateway, S3, DynamoDB", "Only manual invocation", "Only CloudWatch events", "Only SNS topics"], "correct": 0},
            "poll": {"question": "What's your Lambda experience?", "options": ["Production use", "Learning", "Tried once", "Never used"]},
        },
        "rag_pipeline": {
            "quiz": {"question": "What does RAG stand for?", "options": ["Retrieval-Augmented Generation", "Random Access Generation", "Recursive Algorithm Generation", "Real-time AI Generation"], "correct": 0},
            "poll": {"question": "Have you built a RAG system?", "options": ["Yes, in production", "Experimenting", "Planning to", "No"]},
        },
        "kubernetes_pods": {
            "quiz": {"question": "What comes after 'Pending' in pod lifecycle?", "options": ["ContainerCreating → Running", "Running → Terminated", "Failed → Succeeded", "Succeeded → Failed"], "correct": 0},
            "poll": {"question": "How do you debug pod issues?", "options": ["kubectl logs", "kubectl describe", "Events", "All of the above"]},
        },
        "docker_multistage": {
            "quiz": {"question": "What's the main benefit of multi-stage builds?", "options": ["Smaller final image size", "Faster build time", "More layers", "Larger attack surface"], "correct": 0},
            "poll": {"question": "What's your Docker image size?", "options": ["<100MB", "100-500MB", "500MB-1GB", ">1GB"]},
        },
    }
    
    template = poll_templates.get(topic_key, poll_templates["python_decorators"])
    
    poll_data = {
        "topic": topic_data["topic"],
        "quiz_poll": template["quiz"],
        "opinion_poll": template["poll"],
    }
    
    poll_path = output_dir / f"poll_{topic_data['topic'].lower().replace(' ', '_')}.json"
    with open(poll_path, "w") as f:
        json.dump(poll_data, f, indent=2)
    
    print(f"✅ Poll data saved: {poll_path}")
    return poll_path


def save_metadata(output_dir: Path, topic_data: dict, variants: list, ig_paths: list, fb_paths: list, caption: str, hashtags: str, poll_path: Path):
    """Save metadata JSON for workflow consumption."""
    meta = {
        "topic": topic_data["topic"],
        "category": topic_data["category"],
        "variant": "modern",
        "instagram_images": [str(p) for p in ig_paths],
        "facebook_images": [str(p) for p in fb_paths],
        "caption": caption,
        "hashtags": hashtags,
        "poll_file": str(poll_path),
        "generated_at": __import__("datetime").datetime.now(datetime.timezone.utc).isoformat(),
    }
    
    meta_path = output_dir / f"metadata_{topic_data['topic'].lower().replace(' ', '_')}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"✅ Metadata saved: {meta_path}")
    return meta_path


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
    state_file = Path(__file__).parent / ".topic_index.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                return json.load(f).get("index", 0)
        except Exception:
            pass
    return 0


def save_topic_index(index: int):
    state_file = Path(__file__).parent / ".topic_index.json"
    try:
        with open(state_file, "w") as f:
            json.dump({"index": index}, f)
    except Exception as e:
        print(f"⚠️ Failed to save topic index: {e}")


def get_trending_topics(category="AI & Tech Tools", max_topics=10):
    """Fetch trending topics from trending engine and convert to image generator format."""
    if not TRENDING_ENGINE_AVAILABLE:
        return None
    
    try:
        print(f"🔍 Fetching trending topics for category: {category}...")
        signals = fetch_all_trending_signals(target_country="US", category=category)
        
        if not signals:
            print("⚠️ No trending signals found")
            return None
        
        # Convert trending signals to topic format
        topics = []
        for signal in signals[:max_topics]:
            title = signal.get("title", "Untitled")
            # Clean title for use as topic
            clean_title = title[:60].strip()
            
            # Map signal types to categories
            signal_type = signal.get("type", "")
            category_map = {
                "github_trending": "github",
                "reddit_trending": "reddit",
                "hacker_news": "hackernews",
                "huggingface_trending": "huggingface",
                "arxiv_papers": "research",
                "newsletter_ai": "newsletter",
                "google_trends": "trends",
                "youtube_most_popular": "youtube",
                "youtube_outliers": "youtube",
                "youtube_trending": "youtube",
            }
            mapped_category = category_map.get(signal_type, "tech")
            
            # Generate steps based on signal content
            description = signal.get("description", "")
            steps = generate_steps_from_signal(signal)
            
            topic_data = {
                "topic": clean_title,
                "category": mapped_category,
                "filename": f"{clean_title.lower().replace(' ', '_').replace('/', '_')[:40]}.txt",
                "eyebrow": f"TRENDING: {signal_type.upper().replace('_', ' ')}",
                "title": clean_title,
                "subtitle_pre": "",
                "subtitle_bold": "📈 Trending Now",
                "subtitle_post": "",
                "steps": steps,
                "closing_line": f"trending signal from {signal.get('source', {}).get('name', 'unknown')} — explore more",
                "_engagement": signal.get("_engagement", {}),
                "_source_url": signal.get("url", ""),
            }
            topics.append(topic_data)
        
        print(f"✅ Generated {len(topics)} dynamic topics from trending signals")
        return topics
        
    except Exception as e:
        print(f"⚠️ Failed to fetch trending topics: {e}")
        return None


def generate_steps_from_signal(signal: dict) -> list:
    """Generate step data from a trending signal for the image template."""
    # Default step templates based on signal type
    signal_type = signal.get("type", "")
    description = signal.get("description", "")
    title = signal.get("title", "")
    tags = signal.get("_engagement", {}).get("tags", []) if "_engagement" in signal else []
    views = signal.get("_engagement", {}).get("views", 0) if "_engagement" in signal else 0
    
    # Color palette for steps
    colors = ["#C879E6", "#5FB3F0", "#86D9A0", "#E8B85C", "#58C7D6", "#F07178"]
    icons = ["search", "zap", "layers", "code", "database", "shield-check", "bot", "cloud", "cpu", "git-branch"]
    
    steps = []
    
    # Step 1: What is it
    steps.append({
        "accent": colors[0],
        "icon": icons[0],
        "title": "What's Trending",
        "desc": title[:80],
    })
    
    # Step 2: Source
    source_name = signal.get("source", {}).get("name", "Unknown")
    steps.append({
        "accent": colors[1 % len(colors)],
        "icon": icons[1 % len(icons)],
        "title": "Signal Source",
        "desc": source_name[:80],
    })
    
    # Step 3: Engagement metrics
    if views > 0:
        steps.append({
            "accent": colors[2 % len(colors)],
            "icon": icons[2 % len(icons)],
            "title": "Engagement",
            "desc": f"{views:,} views + trending signals",
        })
    
    # Step 4: Key topics/tags
    if tags:
        top_tags = ", ".join(tags[:3])
        steps.append({
            "accent": colors[3 % len(colors)],
            "icon": icons[3 % len(icons)],
            "title": "Key Topics",
            "desc": top_tags[:80],
        })
    
    # Step 5: Description snippet
    if description:
        steps.append({
            "accent": colors[4 % len(colors)],
            "icon": icons[4 % len(icons)],
            "title": "Why It Matters",
            "desc": description[:100],
        })
    
    # Step 6: Action
    steps.append({
        "accent": colors[5 % len(colors)],
        "icon": icons[5 % len(icons)],
        "title": "Explore Further",
        "desc": "Click source link for full context",
    })
    
    return steps


def get_dynamic_topic(category="AI & Tech Tools") -> dict:
    """Get next dynamic topic from trending signals, fallback to rotation."""
    # Try to fetch trending topics
    trending_topics = get_trending_topics(category)
    
    if trending_topics:
        # Use a hash of current date + category to pick consistently within a run
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        seed = hashlib.md5(f"{today}-{category}".encode()).hexdigest()
        index = int(seed, 16) % len(trending_topics)
        return trending_topics[index]
    
    # Fallback to rotating default topics
    return get_next_topic()


def load_hashtags(hashtags_file: str) -> str:
    """Load hashtags from file."""
    if Path(hashtags_file).exists():
        with open(hashtags_file) as f:
            return f.read().strip()
    return ""


def main():
    parser = argparse.ArgumentParser(description="Generate decorator flow images for social media")
    parser.add_argument("--now", action="store_true", help="Run immediately")
    parser.add_argument("--dry-run", action="store_true", help="Preview without posting to Telegram")
    parser.add_argument("--topic", type=str, help="Specific topic to generate")
    parser.add_argument("--hashtags-file", type=str, help="Path to hashtags file")
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
        # Use dynamic trending topic based on category
        category = os.getenv("TRENDING_CATEGORY", "AI & Tech Tools")
        topic_data = get_dynamic_topic(category)

    # Load hashtags
    hashtags = ""
    if args.hashtags_file:
        hashtags = load_hashtags(args.hashtags_file)

    print(f"🎨 Generating images for: {topic_data['topic']}")

    output_dir = Path(__file__).parent / "output" / "social_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate single variant images
        print(f"\n🎨 Generating modern variant...")
        ig_path, fb_path = generate_images(topic_data, output_dir)
        all_ig_paths = [ig_path]
        all_fb_paths = [fb_path]
        
        # Generate caption
        caption = generate_caption(topic_data, hashtags)
        caption_path = output_dir / f"caption_{topic_data['topic'].lower().replace(' ', '_')}.txt"
        caption_path.write_text(caption)
        print(f"✅ Caption saved: {caption_path}")
        
        # Generate poll data
        poll_path = generate_poll_data(topic_data, output_dir)
        
        # Save metadata
        meta_path = save_metadata(
            output_dir, topic_data, ["modern"], 
            all_ig_paths, all_fb_paths, 
            caption, hashtags, poll_path
        )

        if args.dry_run:
            print(f"\n✅ DRY RUN COMPLETE")
            print(f"   Instagram: {', '.join(str(p.name) for p in all_ig_paths)}")
            print(f"   Facebook:  {', '.join(str(p.name) for p in all_fb_paths)}")
            print(f"   Caption: {caption_path}")
            print(f"   Poll: {poll_path}")
            print(f"   Metadata: {meta_path}")
            
            # Output for GitHub Actions
            set_gha_output("topic", topic_data["topic"])
            set_gha_output("ig_images", ','.join(str(p) for p in all_ig_paths))
            set_gha_output("fb_images", ','.join(str(p) for p in all_fb_paths))
            set_gha_output("caption_file", str(caption_path))
            set_gha_output("poll_file", str(poll_path))
            set_gha_output("hashtags", hashtags)
            return

        # Send to Telegram for review
        caption_text = f"📚 <b>{topic_data['topic']}</b> (modern)\n\nReady for review. Approve for posting?\n\nFollow @Vijayakumarj_ai for more!"
        
        send_image_to_telegram(all_ig_paths[0], caption_text)
        send_image_to_telegram(all_fb_paths[0], caption_text)
        
        send_telegram_message(
            f"✅ Images generated for: {topic_data['topic']}\n"
            f"Instagram (4:5): {len(all_ig_paths)} image\n"
            f"Facebook (1.91:1): {len(all_fb_paths)} image\n\nReply to approve for posting.",
            emoji="🤖"
        )

        # Output for GitHub Actions
        set_gha_output("topic", topic_data["topic"])
        set_gha_output("ig_images", ','.join(str(p) for p in all_ig_paths))
        set_gha_output("fb_images", ','.join(str(p) for p in all_fb_paths))
        set_gha_output("caption_file", str(caption_path))
        set_gha_output("poll_file", str(poll_path))
        set_gha_output("hashtags", hashtags)
        
        print("\n✅ Generation complete. Images sent to Telegram for review.")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()