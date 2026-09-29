#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_to_threads.py — Post images/carousels to Threads via Graph API.
Supports single image posts and multi-image carousels (2-20 items).

Threads API standards:
  - Uses graph.threads.net (NOT graph.facebook.com)
  - Text limit: 2200 characters
  - Image limit: 8 MB per image (JPEG/PNG)
  - Carousel: 2-20 items
  - Rate limit: 250 posts per 24h per user
  - Links in follow-up replies (main post stays clean to avoid reach suppression)
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent to path for threads_upload module
sys.path.insert(0, str(Path(__file__).parent))

from threads_upload import (
    upload_images_to_threads,
    _check_credentials,
)


def post_to_threads(image_path: str, caption: str) -> str:
    """Post single image or carousel to Threads.

    If image_path contains comma-separated paths, treats as carousel.

    Args:
        image_path: Single image path or comma-separated paths for carousel.
        caption: Text caption for the post (max 2200 chars).

    Returns:
        Post ID on success.

    Raises:
        RuntimeError: If posting fails.
    """
    paths = [p.strip() for p in image_path.split(",") if p.strip()]

    if len(paths) > 1:
        print(f"🧵 Posting CAROUSEL to Threads: {len(paths)} images")
    else:
        print(f"🧵 Posting single image to Threads: {paths[0] if paths else 'none'}")

    if not paths:
        raise RuntimeError("No image paths provided")

    success, result = upload_images_to_threads(paths, caption)

    if not success:
        raise RuntimeError(result)

    # Optional informational Telegram notification
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat = os.getenv("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat:
        try:
            import requests
            post_type = f"Carousel ({len(paths)} slides)" if len(paths) > 1 else "Single Image"
            msg = f"🧵 <b>Posted to Threads!</b>\n\n📌 <b>Type:</b> {post_type}\n🆔 <b>Post ID:</b> <code>{result}</code>"
            requests.post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                json={"chat_id": tg_chat, "text": msg, "parse_mode": "HTML"},
                timeout=10
            )
        except Exception as e:
            print(f"ℹ️ Telegram notice note: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Post to Threads (image or carousel)")
    parser.add_argument("--image", required=True, help="Image file path(s) - comma separated for carousel")
    parser.add_argument("--caption", default="", help="Post caption (max 2200 chars)")
    parser.add_argument("--caption-file", default="", help="Path to text file containing post caption")
    parser.add_argument("--topic", default="", help="Topic title for tracking logs")
    args = parser.parse_args()

    if not _check_credentials():
        print("❌ Threads credentials not configured (need THREADS_USER_ID, THREADS_ACCESS_TOKEN)")
        sys.exit(1)

    caption = args.caption
    if args.caption_file and Path(args.caption_file).exists():
        caption = Path(args.caption_file).read_text(encoding="utf-8")

    if not caption:
        print("❌ No caption provided (--caption or --caption-file required)")
        sys.exit(1)

    try:
        post_id = post_to_threads(args.image, caption)
        print(f"✅ Threads post published: {post_id}")
        print(f"POST_ID={post_id}")

        # Record in trackers to prevent duplication
        if args.topic:
            try:
                from telegram_approval_handler import record_topic_in_tracker
                record_topic_in_tracker(args.topic, subcategory="Threads Post")
            except Exception as e:
                print(f"⚠️ Note recording topic: {e}")
            try:
                from ai_news_carousel import record_carousel_topic
                record_carousel_topic(args.topic, "", ["threads", "tech"], "Threads Post")
            except Exception as e:
                print(f"⚠️ Note recording carousel tracker: {e}")

    except Exception as e:
        print(f"❌ Threads post failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
