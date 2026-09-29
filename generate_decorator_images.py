#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_decorator_images.py — Generate AI News Carousel images for Instagram (4:5),
Facebook (9:16 Stories/Reels), and send to Telegram for review.
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
FACEBOOK_STORY_W, FACEBOOK_STORY_H = 1080, 1920  # 9:16 (FB Stories/Reels)

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

def generate_facebook_images_from_carousel(carousel: dict, ig_paths: list, output_dir: Path, strategy: dict = None) -> list:
    """Generate Facebook 9:16 format images from carousel (for Stories/Reels)."""
    # Render carousel slides directly in 9:16 format using HTML renderer
    fb_paths = []
    if ig_paths:
        from carousel_renderer_html import render_carousel, DEFAULT_CANVAS_W, DEFAULT_CANVAS_H
        
        safe_topic = sanitize_filename(carousel.get("headline", "carousel"))
        
        # Render all slides in 9:16 format (1080x1920)
        fb_slide_paths = render_carousel(
            carousel,
            output_dir,
            strategy=strategy,
            canvas_width=FACEBOOK_STORY_W,
            canvas_height=FACEBOOK_STORY_H,
        )
        
        # Rename to facebook_ prefix for clarity
        for i, slide_path in enumerate(fb_slide_paths):
            fb_path = output_dir / f"facebook_{safe_topic}_{i+1:02d}.jpg"
            if slide_path != fb_path:
                slide_path.rename(fb_path)
            fb_paths.append(fb_path)
            print(f"✅ Facebook 9:16 image generated: {fb_path.name}")
    
    return fb_paths

def main():
    parser = argparse.ArgumentParser(description="Generate AI News Carousel images for social media")
    parser.add_argument("--now", action="store_true", help="Run immediately")
    parser.add_argument("--dry-run", action="store_true", help="Preview without posting to Telegram")
    parser.add_argument("--topic", type=str, help="Specific topic to generate")
    parser.add_argument("--hashtags-file", type=str, help="Path to hashtags file")
    parser.add_argument("--style", choices=["cartoon_dialogue", "code_editor"], default="cartoon_dialogue", help="Carousel visual style")
    parser.add_argument("--mode", choices=["auto", "concept", "news", "did_you_know"], default="did_you_know", help="Content mode: did_you_know, concept, news, auto")
    parser.add_argument("--ai-backgrounds", action="store_true", help="Optionally generate AI backgrounds per slide")
    parser.add_argument("--platform", choices=["both", "instagram", "facebook", "threads"], default="both", help="Target platform(s)")
    args = parser.parse_args()

    if not args.now and not args.dry_run:
        print("Usage: python generate_decorator_images.py --now       # Generate and send to Telegram")
        print("       python generate_decorator_images.py --dry-run   # Generate only")
        print("       python generate_decorator_images.py --now --topic 'OpenAI GPT-5' --style cartoon_dialogue")
        sys.exit(1)

    # Load hashtags
    hashtags = ""
    if args.hashtags_file:
        hashtags = load_hashtags(args.hashtags_file)

    output_dir = Path(__file__).parent / "output" / "social_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ai_news_carousel import fetch_ai_news_stories, select_best_story

        # ── Style: cartoon_dialogue (Fixed Mascots + HTML Speech Bubbles) ───
        if args.style == "cartoon_dialogue":
            from cartoon_dialogue_engine import (
                generate_cartoon_dialogue_json,
                render_cartoon_dialogue_carousel,
                fetch_or_select_did_you_know_fact
            )

            story = None
            mode = args.mode
            if mode == "auto":
                mode = "did_you_know"

            topic_to_use = args.topic

            if mode in ["did_you_know", "dyk"]:
                print("🧠 Mode: 'did_you_know' — Selecting high-attraction tech fact...")
                story = fetch_or_select_did_you_know_fact(topic=args.topic)
                topic_to_use = story.get("title", args.topic or "Did You Know Tech Fact")
            elif mode == "news":
                try:
                    stories = fetch_ai_news_stories()
                    if args.topic:
                        story = next((s for s in stories if args.topic.lower() in s.get("title", "").lower()), None)
                        if not story and args.mode == "news":
                            story = {
                                "title": args.topic,
                                "source": "Verified Tech News",
                                "date": datetime.datetime.now().strftime("%d %b %Y"),
                                "description": args.topic,
                            }
                    else:
                        story = select_best_story(stories)
                except Exception as e:
                    print(f"⚠️ Note on fetching stories: {e}")
                topic_to_use = args.topic or (story.get("title") if story else "AI News")
            else:
                topic_to_use = args.topic or "Tech Concept"

            print(f"🎭 Generating Mascot Cartoon Dialogue ({mode} mode: '{topic_to_use}')...")
            dialogue = generate_cartoon_dialogue_json(topic=topic_to_use, story=story, mode=mode)
            safe_title = sanitize_filename(dialogue.get("headline", dialogue.get("hook", "dyk_update")))

            # Save dialogue JSON
            carousel_path = output_dir / f"carousel_{safe_title}.json"
            with open(carousel_path, "w", encoding="utf-8") as f:
                json.dump(dialogue, f, indent=2)
            print(f"✅ Cartoon Dialogue JSON saved: {carousel_path}")

            # Optional AI backgrounds
            bg_images = None
            if args.ai_backgrounds:
                print("🌌 Generating optional AI scene backgrounds...")
                try:
                    from image_gen import _generate_cloudflare_image, _generate_huggingface_image
                    bg_images = []
                    slides = dialogue.get("slides", [])
                    hook = dialogue.get("hook", "artificial intelligence datacenter")
                    for s_idx in range(len(slides)):
                        bg_file = output_dir / f"bg_{safe_title}_{s_idx+1:02d}.jpg"
                        bg_prompt = f"Abstract subtle minimalist digital technology background, soft atmospheric neon cyan and purple ambient glow, cinematic wide shot, no text, no words, no characters, clean studio background, depth of field"
                        out_bg = _generate_cloudflare_image(bg_prompt, str(bg_file), aspect_ratio="9:16") or _generate_huggingface_image(bg_prompt, str(bg_file), aspect_ratio="9:16")
                        bg_images.append(str(out_bg) if out_bg else None)
                except Exception as bg_err:
                    print(f"⚠️ AI background generation skipped: {bg_err}")
                    bg_images = None

            # Render Instagram 4:5 slides
            print(f"🎨 Rendering {len(dialogue.get('slides', []))} slides for Instagram (4:5)...")
            ig_paths = render_cartoon_dialogue_carousel(dialogue, output_dir, canvas_width=INSTAGRAM_W, canvas_height=INSTAGRAM_H, prefix="carousel_", bg_images=bg_images)

            # Render Facebook 9:16 format if needed
            fb_paths = []
            if args.platform in ["both", "facebook"]:
                print(f"📘 Rendering Facebook 9:16 stories...")
                fb_paths = render_cartoon_dialogue_carousel(dialogue, output_dir, canvas_width=FACEBOOK_STORY_W, canvas_height=FACEBOOK_STORY_H, prefix="facebook_", bg_images=bg_images)

            # Caption
            caption = dialogue.get("caption", "")
            if not caption:
                caption = f"🧠 {dialogue.get('hook')}\n\n💡 {dialogue.get('takeaway')}\n\nFollow @vijayakumarj_ai for daily mind-blowing tech facts!"
            if hashtags and hashtags not in caption:
                caption = f"{caption}\n\n{hashtags}"

            caption_path = output_dir / f"caption_{safe_title}.txt"
            caption_path.write_text(caption, encoding="utf-8")
            print(f"✅ Caption saved: {caption_path}")

            # Poll data
            carousel = {"headline": dialogue.get("headline", dialogue.get("hook", "Did You Know?")), "summary": dialogue.get("takeaway", "")}
            poll_path = generate_poll_data(carousel, output_dir)

            # Save metadata
            meta_path = save_metadata(
                output_dir, carousel, ig_paths, fb_paths,
                caption, hashtags, poll_path
            )

        # ── Style: code_editor (Existing Tech/Code Cards) ──────────────
        else:
            from ai_news_carousel import generate_carousel_json
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
                run_context = f"{datetime.datetime.now().strftime('%Y%m%d')}-{os.getenv('GITHUB_RUN_NUMBER', '0')}-{os.getenv('GITHUB_RUN_ATTEMPT', '1')}"
                stories = fetch_ai_news_stories(run_context=run_context)
                story = select_best_story(stories, run_context=run_context)
                if not story:
                    print("❌ No stories meet quality threshold")
                    sys.exit(1)

            print(f"📰 Selected story: {story.get('title')}")
            carousel = generate_carousel_json(story)
            safe_title = sanitize_filename(carousel.get("headline", "ai_news"))

            run_context = f"{datetime.datetime.now().strftime('%Y%m%d')}-{os.getenv('GITHUB_RUN_NUMBER', '0')}-{os.getenv('GITHUB_RUN_ATTEMPT', '1')}"
            strategy = create_visual_strategy(carousel, story=story, run_context=run_context)
            strategy_path = output_dir / f"strategy_{safe_title}.json"
            with open(strategy_path, "w", encoding="utf-8") as f:
                json.dump(strategy, f, indent=2)

            carousel_path = output_dir / f"carousel_{safe_title}.json"
            with open(carousel_path, "w", encoding="utf-8") as f:
                json.dump(carousel, f, indent=2)

            slide_count = len(carousel.get("slides", []))
            print(f"\n🎨 Rendering {slide_count} carousel slides for Instagram (4:5) with theme {strategy.get('visual_theme')}...")
            ig_paths = render_carousel(carousel, output_dir, strategy=strategy, canvas_width=INSTAGRAM_W, canvas_height=INSTAGRAM_H)

            fb_paths = []
            if args.platform in ["both", "facebook"]:
                fb_paths = generate_facebook_images_from_carousel(carousel, ig_paths, output_dir, strategy=strategy)

            caption = generate_caption(carousel, hashtags)
            caption_path = output_dir / f"caption_{safe_title}.txt"
            caption_path.write_text(caption, encoding="utf-8")

            poll_path = generate_poll_data(carousel, output_dir)
            meta_path = save_metadata(
                output_dir, carousel, ig_paths, fb_paths,
                caption, hashtags, poll_path
            )
            headline_for_output = carousel.get("headline", "AI News")

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
            set_gha_output("threads_images", ','.join(str(p) for p in ig_paths))
            set_gha_output("caption_file", str(caption_path))
            set_gha_output("poll_file", str(poll_path))
            set_gha_output("hashtags", hashtags)
            set_gha_output("carousel_json", str(carousel_path))
            set_gha_output("is_carousel", "true")
            return

        # Send to Telegram preview if configured
        caption_text = f"🧠 <b>{carousel.get('headline', 'Did You Know?')}</b>\n\n{carousel.get('summary', '')[:200]}...\n\nCarousel: {len(ig_paths)} slides"
        
        # Send Instagram carousel preview
        if args.platform in ["both", "instagram"]:
            send_carousel_to_telegram(ig_paths, caption_text)
        
        # Send Threads carousel preview
        if args.platform in ["both", "threads"]:
            send_carousel_to_telegram(ig_paths, f"🧵 <b>Threads Version</b>\n\n{caption_text}")

        # Send Facebook version preview
        if args.platform in ["both", "facebook"] and fb_paths:
            fb_caption = f"📘 <b>Facebook Version</b>\n\n{caption_text}"
            send_image_to_telegram(fb_paths[0], fb_caption)
        
        platform_summary = []
        if args.platform in ["both", "instagram"]:
            platform_summary.append(f"Instagram Carousel (4:5): {len(ig_paths)} slides")
        if args.platform in ["both", "threads"]:
            platform_summary.append(f"Threads Carousel: {len(ig_paths)} slides")
        if args.platform in ["both", "facebook"] and fb_paths:
            platform_summary.append(f"Facebook (1.91:1): {len(fb_paths)} image")
            
        send_telegram_message(
            f"✅ Content generated for: {carousel.get('headline', 'Did You Know?')}\n"
            + "\n".join(platform_summary),
            emoji="🤖"
        )

        # Output for GitHub Actions
        set_gha_output("topic", carousel.get("headline", "AI News"))
        set_gha_output("ig_images", ','.join(str(p) for p in ig_paths))
        set_gha_output("fb_images", ','.join(str(p) for p in fb_paths))
        set_gha_output("threads_images", ','.join(str(p) for p in ig_paths))
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