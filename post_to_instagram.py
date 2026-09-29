#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_to_instagram.py — Post images/carousels to Instagram via Graph API.
Supports both single image posts and multi-image carousels.
"""

import os
import sys
import argparse
import requests
from pathlib import Path

# Add parent to path for instagram_upload module
sys.path.insert(0, str(Path(__file__).parent))

from instagram_upload import (
    upload_reel_to_instagram,
    upload_carousel_to_instagram,
    _check_credentials,
)

def upload_single_image_to_instagram(image_path: str, caption: str):
    """Upload single static image (JPG/PNG) to Instagram using Graph API."""
    from instagram_upload import (
        _check_credentials, _check_rate_limit, _load_token, _refresh_token_if_needed,
        upload_video_to_github_releases, create_image_container, wait_for_container,
        publish_container, _increment_rate_limit
    )
    if not _check_credentials():
        return False, "Skipped: Instagram credentials not configured (IG_USER_ID, IG_ACCESS_TOKEN)"
    
    allowed, current_count = _check_rate_limit()
    if not allowed:
        return False, f"Skipped: Instagram rate limit reached ({current_count} posts today)"
    
    token, expiry = _load_token()
    token = _refresh_token_if_needed(token, expiry)
    if token != os.getenv("IG_ACCESS_TOKEN"):
        os.environ["IG_ACCESS_TOKEN"] = token

    public_url, upload_key = upload_video_to_github_releases(image_path)
    if not public_url:
        return False, f"Failed to host image publicly: {upload_key}"

    print(f"📡 [Instagram Image] Creating image container...")
    container_id = create_image_container(public_url, caption=caption, is_carousel_item=False)
    print(f"✔ Container created: {container_id}")

    print(f"📡 [Instagram Image] Waiting for container processing...")
    wait_for_container(container_id)
    print(f"✔ Container processing finished")

    print(f"📡 [Instagram Image] Publishing post...")
    published_id = publish_container(container_id)
    print(f"🎉 Instagram Image published! ID: {published_id}")

    _increment_rate_limit()
    return True, published_id


def post_to_instagram(image_path: str, caption: str) -> str:
    """Post single image, video reel, or multi-image carousel to Instagram.
    
    If image_path contains comma-separated paths, treats as carousel.
    """
    paths = [p.strip() for p in image_path.split(",") if p.strip()]
    if not paths:
        raise ValueError("No images or videos provided for Instagram post")
    
    if len(paths) > 1:
        print(f"📸 Posting CAROUSEL to Instagram: {len(paths)} images")
        success, result = upload_carousel_to_instagram(paths, caption)
    else:
        single_path = paths[0]
        ext = Path(single_path).suffix.lower()
        if ext in [".mp4", ".mov", ".m4v"]:
            print(f"📹 Posting REEL to Instagram: {single_path}")
            success, result = upload_reel_to_instagram(single_path, caption)
        else:
            print(f"📸 Posting single IMAGE to Instagram: {single_path}")
            success, result = upload_single_image_to_instagram(single_path, caption)
    
    if not success:
        raise RuntimeError(result)

    # Optional informational Telegram notification
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat = os.getenv("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat:
        try:
            post_type = f"Carousel ({len(paths)} slides)" if len(paths) > 1 else "Single Image"
            msg = f"🎉 <b>Posted to Instagram!</b>\n\n📌 <b>Type:</b> {post_type}\n🆔 <b>Post ID:</b> <code>{result}</code>"
            requests.post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                json={"chat_id": tg_chat, "text": msg, "parse_mode": "HTML"},
                timeout=10
            )
        except Exception as e:
            print(f"ℹ️ Telegram notice note: {e}")

    return result

def main():
    parser = argparse.ArgumentParser(description="Post to Instagram (image, reel, or carousel)")
    parser.add_argument("--image", required=True, help="Image file path(s) - comma separated for carousel")
    parser.add_argument("--caption", default="", help="Post caption string")
    parser.add_argument("--caption-file", default="", help="Path to text file containing post caption")
    args = parser.parse_args()
    
    if not _check_credentials():
        print("❌ Instagram credentials not configured")
        sys.exit(1)

    caption = args.caption
    if args.caption_file and Path(args.caption_file).exists():
        caption = Path(args.caption_file).read_text(encoding="utf-8")
    
    try:
        post_id = post_to_instagram(args.image, caption)
        print(f"✅ Instagram post published: {post_id}")
        print(f"POST_ID={post_id}")
    except Exception as e:
        print(f"❌ Instagram post failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()