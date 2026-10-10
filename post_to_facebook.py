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

def _post_fb_with_ai_fallback(url: str, data: dict, files: dict = None, timeout: int = 30) -> requests.Response:
    """Post to Facebook Graph API with is_ai_generated flag if enabled, with automatic fallback."""
    ai_flag = os.getenv("AI_FLAG", "true").lower() in ("true", "1", "yes")
    if ai_flag:
        data["is_ai_generated"] = "true"

    resp = requests.post(url, data=data, files=files, timeout=timeout)
    if resp.status_code == 400 and "is_ai_generated" in data:
        err_msg = resp.text.lower()
        if "is_ai_generated" in err_msg or "param" in err_msg:
            print("⚠️ Note: Facebook Graph API rejected is_ai_generated parameter on this endpoint. Retrying without it...")
            data_retry = {k: v for k, v in data.items() if k != "is_ai_generated"}
            if files:
                for f in files.values():
                    if hasattr(f, "seek"):
                        f.seek(0)
            resp = requests.post(url, data=data_retry, files=files, timeout=timeout)
    return resp

def post_to_facebook(image_paths, caption: str) -> str:
    """Post single image or carousel/multi-photo post to Facebook Page feed.

    Args:
        image_paths: Path to image, comma-separated string of paths, or list of paths.
        caption: Caption / message for the post.

    Returns:
        Post ID from Facebook Graph API.
    """
    if not FB_PAGE_ID or not FB_PAGE_ACCESS_TOKEN:
        raise ValueError("Facebook credentials not configured")
    
    if isinstance(image_paths, str):
        if "," in image_paths:
            paths = [p.strip() for p in image_paths.split(",") if p.strip()]
        else:
            paths = [image_paths.strip()]
    elif isinstance(image_paths, (list, tuple)):
        paths = list(image_paths)
    else:
        raise ValueError(f"Invalid image_paths type: {type(image_paths)}")

    if not paths:
        raise ValueError("No image paths provided")

    # If only 1 image, post directly as a single photo post
    if len(paths) == 1:
        img = paths[0]
        print(f"📘 Posting single photo to Facebook: {img}")
        print(f"   Caption: {caption[:100]}...")
        url = f"{GRAPH_API_BASE}/{FB_PAGE_ID}/photos"
        if img.startswith("http"):
            data = {
                "url": img,
                "caption": caption,
                "access_token": FB_PAGE_ACCESS_TOKEN,
            }
            resp = _post_fb_with_ai_fallback(url, data=data, timeout=30)
        else:
            with open(img, "rb") as f:
                files = {"source": f}
                data = {
                    "caption": caption,
                    "access_token": FB_PAGE_ACCESS_TOKEN,
                }
                resp = _post_fb_with_ai_fallback(url, data=data, files=files, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        return result.get("post_id") or result.get("id")

    # Multi-photo / carousel post:
    print(f"📘 Posting {len(paths)} photos as multi-photo carousel to Facebook...")
    print(f"   Caption: {caption[:100]}...")

    photo_ids = []
    for idx, img in enumerate(paths):
        print(f"   📤 Uploading unpublished photo {idx+1}/{len(paths)}: {img}")
        url = f"{GRAPH_API_BASE}/{FB_PAGE_ID}/photos"
        if img.startswith("http"):
            data = {
                "url": img,
                "published": "false",
                "access_token": FB_PAGE_ACCESS_TOKEN,
            }
            resp = requests.post(url, data=data, timeout=30)
        else:
            with open(img, "rb") as f:
                files = {"source": f}
                data = {
                    "published": "false",
                    "access_token": FB_PAGE_ACCESS_TOKEN,
                }
                resp = requests.post(url, data=data, files=files, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        photo_id = result.get("id")
        if not photo_id:
            raise RuntimeError(f"Failed to get photo ID for {img}: {result}")
        photo_ids.append(photo_id)
        print(f"   ✔ Photo {idx+1} uploaded (ID: {photo_id})")

    # Publish to feed attaching all uploaded photos
    print(f"📡 Publishing feed post attaching {len(photo_ids)} photos...")
    feed_url = f"{GRAPH_API_BASE}/{FB_PAGE_ID}/feed"
    attached_media = [{"media_fbid": pid} for pid in photo_ids]
    data = {
        "message": caption,
        "attached_media": json.dumps(attached_media),
        "access_token": FB_PAGE_ACCESS_TOKEN,
    }
    resp = _post_fb_with_ai_fallback(feed_url, data=data, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    post_id = result.get("id")
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