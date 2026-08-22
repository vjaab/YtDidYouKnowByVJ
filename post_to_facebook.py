#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_to_facebook.py — Post images to Facebook Page via Graph API.
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path

FB_PAGE_ID = os.getenv("FB_PAGE_ID", "")
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "")
GRAPH_API_BASE = "https://graph.facebook.com/v20.0"

def post_to_facebook(image_path: str, caption: str) -> str:
    """Post image to Facebook Page."""
    if not FB_PAGE_ID or not FB_PAGE_ACCESS_TOKEN:
        raise ValueError("Facebook credentials not configured")
    
    print(f"📘 Posting to Facebook: {image_path}")
    print(f"   Caption: {caption[:100]}...")
    
    # Facebook Graph API: post photo to page feed
    url = f"{GRAPH_API_BASE}/{FB_PAGE_ID}/photos"
    
    # For local files, we need to upload as multipart
    # For URLs, we can use the url parameter
    if image_path.startswith("http"):
        data = {
            "url": image_path,
            "caption": caption,
            "access_token": FB_PAGE_ACCESS_TOKEN,
        }
        resp = requests.post(url, data=data, timeout=30)
    else:
        # Upload local file
        with open(image_path, "rb") as f:
            files = {"source": f}
            data = {
                "caption": caption,
                "access_token": FB_PAGE_ACCESS_TOKEN,
            }
            resp = requests.post(url, data=data, files=files, timeout=60)
    
    resp.raise_for_status()
    result = resp.json()
    post_id = result.get("post_id") or result.get("id")
    return post_id

def main():
    parser = argparse.ArgumentParser(description="Post to Facebook Page")
    parser.add_argument("--image", required=True, help="Image file path or URL")
    parser.add_argument("--caption", required=True, help="Post caption")
    args = parser.parse_args()
    
    try:
        post_id = post_to_facebook(args.image, args.caption)
        print(f"✅ Facebook post published: {post_id}")
        print(f"POST_ID={post_id}")
    except Exception as e:
        print(f"❌ Facebook post failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()