#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_decorator_images.py — Generate AI News Carousel images for Instagram (4:5),
Facebook (1.91:1), and send to Telegram for review.
Now uses LLM-generated carousel content + Pillow renderer for consistent branding.
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
import requests
import datetime
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from trending_engine import fetch_all_trending_signals, compute_engagement_score
    TRENDING_ENGINE_AVAILABLE = True
except ImportError:
    TRENDING_ENGINE_AVAILABLE = False
    print("⚠️ trending_engine not available, using fallback topics")

# ── GitHub Actions Output Helper ───────────────────────────────────────────────
def set_gha_output(key: str, value: str):
    """Set GitHub Actions output."""
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{key}={value}\n")
    print(f"{key}={value}")

# ── Config ──────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else ""

# Canvas sizes
INSTAGRAM_W, INSTAGRAM_H = 1080, 1350  # 4:5
FACEBOOK_W, FACEBOOK_H = 1200, 628     # 1.91:1 (FB link post)

# Carousel settings
CAROUSEL_SLIDES = 6

def sanitize_filename(name: str, max_len: int = 100) -> str:
    """Sanitize a string for use as a filename."""
    name = name.replace('—', '-').replace('–', '-')
    name = name.replace(':', '-').replace(';', '-')
    name = name.replace('/', '-').replace('\\', '-')
    name = re.sub(r'[<>|"*?]', '-', name)
    name = re.sub(r'[\s\-]+', '_', name)
    name = name.strip('_.')
    if len(name) > max_len:
        name = name[:max_len].rstrip('_.')
    return name.lower()

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
            timeout=15,
        )
    except Exception as e:
        print(f"Telegram notify failed: {e}")

def send_carousel_to_telegram(image_paths: list, caption: str = "") -> bool:
    """Send carousel images as a media group to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured — skipping")
        return False

    try:
        # Send as media group (album)
        media = []
        for i, img_path in enumerate(image_paths):
            with open(img_path, "rb") as f:
                media.append({
                    "type": "photo",
                    "media": f"attach://photo{i}",
                    "caption": caption[:1024] if i == 0 else "",
                    "parse_mode": "HTML",
                })
        
        files = {}
        for i, img_path in enumerate(image_paths):
            files[f"photo{i}"] = open(img_path, "rb")
        
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "media": json.dumps(media),
        }
        
        resp = requests.post(f"{TELEGRAM_BASE_URL}/sendMediaGroup", data=data, files=files, timeout=60)
        
        for f in files.values():
            f.close()
        
        resp.raise_for_status()
        print(f"📱 Telegram carousel sent: {len(image_paths)} images")
        return True
    except Exception as e:
        print(f"⚠️ Telegram carousel send failed: {e}")
        return False

def load_hashtags(hashtags_file: str) -> str:
    """Load hashtags from file."""
    if Path(hashtags_file).exists():
        with open(hashtags_file) as f:
            return f.read().strip()
    return ""

def generate_caption(carousel: dict, hashtags: str) -> str:
    """Generate caption for social media posts from carousel data."""
    headline = carousel.get("headline", "AI News Update")
    summary = carousel.get("summary", "")
    source = carousel.get("source", "Unknown")
    source_url = carousel.get("source_url", "")
    date = carousel.get("date", datetime.datetime.now().strftime("%d %b %y"))
    
    return f"""📰 {headline}

{summary}

📅 {date} | Source: {source}
🔗 {source_url}

Follow @vijayakumarj_ai for daily AI updates!

{hashtags}"""

def generate_poll_data(carousel: dict, output_dir: Path) -> Path:
    """Generate poll/quiz data for Instagram Stories based on carousel."""
    headline = carousel.get("headline", "AI News")
    safe_topic = sanitize_filename(headline)
    
    poll_data = {
        "topic": headline,
        "quiz_poll": {
            "question": f"What's the key takeaway from: {headline}?",
            "options": [
                "Major capability improvement",
                "New developer tools released",
                "Significant cost reduction",
                "Research breakthrough"
            ],
            "correct": 0
        },
        "opinion_poll": {
            "question": "How will this impact your workflow?",
            "options": [
                "Will adopt immediately",
                "Evaluating for future use",
                "Not relevant to my work",
                "Need more info"
            ]
        },
    }
    
    poll_path = output_dir / f"poll_{safe_topic}.json"
    with open(poll_path, "w") as f:
        json.dump(poll_data, f, indent=2)
    
    print(f"✅ Poll data saved: {poll_path}")
    return poll_path

def save_metadata(output_dir: Path, carousel: dict, ig_paths: list, fb_paths: list, caption: str, hashtags: str, poll_path: Path) -> Path:
    """Save metadata JSON for workflow consumption."""
    headline = carousel.get("headline", "AI News")
    safe_topic = sanitize_filename(headline)
    
    meta = {
        "topic": headline,
        "category": "AI News",
        "type": "carousel",
        "slides": len(ig_paths),
        "instagram_images": [str(p) for p in ig_paths],
        "facebook_images": [str(p) for p in fb_paths],
        "caption": caption,
        "hashtags": hashtags,
        "poll_file": str(poll_path),
        "carousel_json": str(output_dir / f"carousel_{safe_topic}.json"),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    
    meta_path = output_dir / f"metadata_{safe_topic}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"✅ Metadata saved: {meta_path}")
    return meta_path

def generate_facebook_images_from_carousel(carousel: dict, ig_paths: list, output_dir: Path) -> list:
    """Generate Facebook-format images from carousel (first slide as link post, or all as album)."""
    # For Facebook, we'll use the first slide as the main link post image
    # and create a Facebook-optimized version (1.91:1)
    from PIL import Image
    
    fb_paths = []
    if ig_paths:
        # Resize first slide to Facebook 1.91:1
        ig_first = Image.open(ig_paths[0])
        fb_img = ig_first.resize((FACEBOOK_W, FACEBOOK_H), Image.LANCZOS)
        
        safe_topic = sanitize_filename(carousel.get("headline", "carousel"))
        fb_path = output_dir / f"facebook_{safe_topic}.jpg"
        fb_img.convert("RGB").save(fb_path, "JPEG", quality=95)
        fb_paths.append(fb_path)
        print(f"✅ Facebook image generated: {fb_path.name}")
    
    return fb_paths

def main():
    parser = argparse.ArgumentParser(description="Generate AI News Carousel images for social media")
    parser.add_argument("--now", action="store_true", help="Run immediately")
    parser.add_argument("--dry-run", action="store_true", help="Preview without posting to Telegram")
    parser.add_argument("--topic", type=str, help="Specific topic to generate")
    parser.add_argument("--hashtags-file", type=str, help="Path to hashtags file")
    parser.add_argument("--platform", choices=["both", "instagram", "facebook"], default="both", help="Target platform(s)")
    args = parser.parse_args()

    if not args.now and not args.dry_run:
        print("Usage: python generate_decorator_images.py --now       # Generate and send to Telegram")
        print("       python generate_decorator_images.py --dry-run   # Generate only")
        print("       python generate_decorator_images.py --now --topic 'OpenAI GPT-5'")
        sys.exit(1)

    # Load hashtags
    hashtags = ""
    if args.hashtags_file:
        hashtags = load_hashtags(args.hashtags_file)

    output_dir = Path(__file__).parent / "output" / "social_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Import carousel modules
        from ai_news_carousel import fetch_ai_news_stories, select_best_story, generate_carousel_json
        from carousel_renderer_html import render_carousel
        from visual_strategy import create_visual_strategy
        
        # Get story
        if args.topic:
            stories = fetch_ai_news_stories()
            story = next((s for s in stories if args.topic.lower() in s.get("title", "").lower()), None)
            if not story:
                print(f"❌ Topic not found: {args.topic}")
                sys.exit(1)
        else:
            stories = fetch_ai_news_stories()
            story = select_best_story(stories)
            if not story:
                print("❌ No stories meet quality threshold")
                sys.exit(1)

        print(f"📰 Selected story: {story.get('title')}")
        print(f"   Score: {story.get('_score', 'N/A')}")

        # Generate carousel JSON
        carousel = generate_carousel_json(story)
        safe_title = sanitize_filename(carousel.get("headline", "ai_news"))

        # Create Visual Strategy (Theme, Layout Sequence, Styling)
        strategy = create_visual_strategy(carousel, story=story)
        strategy_path = output_dir / f"strategy_{safe_title}.json"
        with open(strategy_path, "w", encoding="utf-8") as f:
            json.dump(strategy, f, indent=2)
        print(f"🎨 Visual Strategy created & saved: {strategy_path}")
        print(f"   Theme: {strategy.get('visual_theme')} | Domain: {strategy.get('domain')} | Slides: {len(carousel.get('slides', []))}")
        
        # Save carousel JSON
        carousel_path = output_dir / f"carousel_{safe_title}.json"
        with open(carousel_path, "w", encoding="utf-8") as f:
            json.dump(carousel, f, indent=2)
        print(f"✅ Carousel JSON saved: {carousel_path}")

        # Render carousel slides
        slide_count = len(carousel.get("slides", []))
        print(f"\n🎨 Rendering {slide_count} carousel slides with theme {strategy.get('visual_theme')}...")
        ig_paths = render_carousel(carousel, output_dir, strategy=strategy)
        
        # Generate Facebook images only if needed
        fb_paths = []
        if args.platform in ["both", "facebook"]:
            fb_paths = generate_facebook_images_from_carousel(carousel, ig_paths, output_dir)
        
        # Generate caption
        caption = generate_caption(carousel, hashtags)
        caption_path = output_dir / f"caption_{safe_title}.txt"
        caption_path.write_text(caption)
        print(f"✅ Caption saved: {caption_path}")
        
        # Generate poll data
        poll_path = generate_poll_data(carousel, output_dir)
        
        # Save metadata
        meta_path = save_metadata(
            output_dir, carousel, ig_paths, fb_paths,
            caption, hashtags, poll_path
        )

        if args.dry_run:
            print(f"\n✅ DRY RUN COMPLETE")
            print(f"   Carousel JSON: {carousel_path}")
            print(f"   Instagram slides: {len(ig_paths)}")
            for p in ig_paths:
                print(f"      {p.name}")
            if fb_paths:
                print(f"   Facebook: {len(fb_paths)}")
                for p in fb_paths:
                    print(f"      {p.name}")
            print(f"   Caption: {caption_path}")
            print(f"   Poll: {poll_path}")
            print(f"   Metadata: {meta_path}")

            # Output for GitHub Actions
            set_gha_output("topic", carousel.get("headline", "AI News"))
            set_gha_output("ig_images", ','.join(str(p) for p in ig_paths))
            set_gha_output("fb_images", ','.join(str(p) for p in fb_paths))
            set_gha_output("caption_file", str(caption_path))
            set_gha_output("poll_file", str(poll_path))
            set_gha_output("hashtags", hashtags)
            set_gha_output("carousel_json", str(carousel_path))
            set_gha_output("is_carousel", "true")
            return

        # Send to Telegram for review
        caption_text = f"📰 <b>{carousel.get('headline', 'AI News')}</b>\n\n{carousel.get('summary', '')[:200]}...\n\nCarousel: {len(ig_paths)} slides\n\nReady for review. Approve for posting?"
        
        # Send Instagram carousel
        if args.platform in ["both", "instagram"]:
            send_carousel_to_telegram(ig_paths, caption_text)
        
        # Send Facebook version
        if args.platform in ["both", "facebook"] and fb_paths:
            fb_caption = f"📘 <b>Facebook Version</b>\n\n{caption_text}"
            send_image_to_telegram(fb_paths[0], fb_caption)
        
        send_telegram_message(
            f"✅ Carousel generated for: {carousel.get('headline', 'AI News')}\n"
            f"Instagram Carousel (4:5): {len(ig_paths)} slides\n"
            f"{'Facebook (1.91:1): ' + str(len(fb_paths)) + ' image' if fb_paths else ''}\nReply to approve for posting.",
            emoji="🤖"
        )

        # Output for GitHub Actions
        set_gha_output("topic", carousel.get("headline", "AI News"))
        set_gha_output("ig_images", ','.join(str(p) for p in ig_paths))
        set_gha_output("fb_images", ','.join(str(p) for p in fb_paths))
        set_gha_output("caption_file", str(caption_path))
        set_gha_output("poll_file", str(poll_path))
        set_gha_output("hashtags", hashtags)
        set_gha_output("carousel_json", str(carousel_path))
        set_gha_output("is_carousel", "true")
        
        print("\n✅ Generation complete. Carousel sent to Telegram for review.")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()