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

def post_to_instagram(image_path: str, caption: str) -> str:
    """Post single image or carousel to Instagram.
    
    If image_path contains comma-separated paths, treats as carousel.
    """
    paths = [p.strip() for p in image_path.split(",") if p.strip()]
    
    if len(paths) > 1:
        print(f"📸 Posting CAROUSEL to Instagram: {len(paths)} images")
        success, result = upload_carousel_to_instagram(paths, caption)
    else:
        print(f"📸 Posting single image to Instagram: {paths[0] if paths else 'none'}")
        success, result = upload_reel_to_instagram(paths[0], caption) if paths else (False, "No image provided")
    
    if not success:
        raise RuntimeError(result)
    return result

def main():
    parser = argparse.ArgumentParser(description="Post to Instagram (image or carousel)")
    parser.add_argument("--image", required=True, help="Image file path(s) - comma separated for carousel")
    parser.add_argument("--caption", required=True, help="Post caption")
    args = parser.parse_args()
    
    if not _check_credentials():
        print("❌ Instagram credentials not configured")
        sys.exit(1)
    
    try:
        post_id = post_to_instagram(args.image, args.caption)
        print(f"✅ Instagram post published: {post_id}")
        print(f"POST_ID={post_id}")
    except Exception as e:
        print(f"❌ Instagram post failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()