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
    return result


def main():
    parser = argparse.ArgumentParser(description="Post to Threads (image or carousel)")
    parser.add_argument("--image", required=True, help="Image file path(s) - comma separated for carousel")
    parser.add_argument("--caption", required=True, help="Post caption (max 2200 chars)")
    args = parser.parse_args()

    if not _check_credentials():
        print("❌ Threads credentials not configured (need THREADS_USER_ID, THREADS_ACCESS_TOKEN)")
        sys.exit(1)

    try:
        post_id = post_to_threads(args.image, args.caption)
        print(f"✅ Threads post published: {post_id}")
        print(f"POST_ID={post_id}")
    except Exception as e:
        print(f"❌ Threads post failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
