#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_to_instagram.py — Post images to Instagram via Graph API.
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path

IG_USER_ID = os.getenv("IG_USER_ID", "")
IG_ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN", "")
GRAPH_API_BASE = "https://graph.facebook.com/v20.0"

def create_media_container(image_path: str, caption: str) -> str:
    """Create a media container for Instagram post."""
    if not IG_USER_ID or not IG_ACCESS_TOKEN:
        raise ValueError("IG credentials not configured")
    
    # Upload image to a temporary hosting (using Facebook's upload)
    # For simplicity, we'll use the image URL approach - but Instagram requires
    # the image to be publicly accessible
    
    # First, we need to upload the image to get a media container ID
    # Instagram Graph API requires the image to be at a public URL
    # We'll use a simple approach: upload to a temporary location or use existing hosting
    
    # For now, we'll use the image directly if it's already hosted
    # In production, you'd upload to a CDN/S3 first
    
    url = f"{GRAPH_API_BASE}/{IG_USER_ID}/media"
    data = {
        "image_url": image_path,  # Must be publicly accessible URL
        "caption": caption,
        "access_token": IG_ACCESS_TOKEN,
    }
    
    resp = requests.post(url, data=data, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    return result.get("id")

def publish_media(container_id: str) -> str:
    """Publish the media container."""
    url = f"{GRAPH_API_BASE}/{IG_USER_ID}/media_publish"
    data = {
        "creation_id": container_id,
        "access_token": IG_ACCESS_TOKEN,
    }
    resp = requests.post(url, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json().get("id")

def post_to_instagram(image_path: str, caption: str) -> str:
    """Post image to Instagram."""
    # Note: Instagram Graph API requires image_url to be a public HTTPS URL
    # This is a simplified version - in production you'd upload to a CDN first
    print(f"📸 Posting to Instagram: {image_path}")
    print(f"   Caption: {caption[:100]}...")
    
    # For local files, you'd need to upload to a public URL first
    # This is a placeholder for the actual implementation
    container_id = create_media_container(image_path, caption)
    post_id = publish_media(container_id)
    return post_id

def main():
    parser = argparse.ArgumentParser(description="Post to Instagram")
    parser.add_argument("--image", required=True, help="Image file path or URL")
    parser.add_argument("--caption", required=True, help="Post caption")
    args = parser.parse_args()
    
    try:
        post_id = post_to_instagram(args.image, args.caption)
        print(f"✅ Instagram post published: {post_id}")
        print(f"POST_ID={post_id}")
    except Exception as e:
        print(f"❌ Instagram post failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()