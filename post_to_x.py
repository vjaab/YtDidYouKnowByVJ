#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_to_x.py — Post images to X (Twitter) via API v2.
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from requests_oauthlib import OAuth1Session

X_API_KEY = os.getenv("X_API_KEY", "")
X_API_SECRET = os.getenv("X_API_SECRET", "")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_SECRET = os.getenv("X_ACCESS_SECRET", "")

def upload_media(image_path: str) -> str:
    """Upload media to X and return media_id."""
    if not all([X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET]):
        raise ValueError("X credentials not configured")
    
    oauth = OAuth1Session(
        X_API_KEY,
        client_secret=X_API_SECRET,
        resource_owner_key=X_ACCESS_TOKEN,
        resource_owner_secret=X_ACCESS_SECRET,
    )
    
    with open(image_path, "rb") as f:
        files = {"media": f}
        resp = oauth.post("https://upload.twitter.com/1.1/media/upload.json", files=files, timeout=60)
    
    resp.raise_for_status()
    return resp.json()["media_id_string"]

def post_tweet(text: str, media_ids: list) -> str:
    """Post tweet with media."""
    oauth = OAuth1Session(
        X_API_KEY,
        client_secret=X_API_SECRET,
        resource_owner_key=X_ACCESS_TOKEN,
        resource_owner_secret=X_ACCESS_SECRET,
    )
    
    payload = {
        "text": text,
        "media": {"media_ids": media_ids},
    }
    resp = oauth.post("https://api.twitter.com/2/tweets", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]["id"]

def post_to_x(image_path: str, text: str) -> str:
    """Post image to X (Twitter)."""
    print(f"🐦 Posting to X: {image_path}")
    print(f"   Text: {text[:100]}...")
    
    media_id = upload_media(image_path)
    tweet_id = post_tweet(text, [media_id])
    return tweet_id

def main():
    parser = argparse.ArgumentParser(description="Post to X (Twitter)")
    parser.add_argument("--image", required=True, help="Image file path")
    parser.add_argument("--text", required=True, help="Tweet text")
    args = parser.parse_args()
    
    try:
        tweet_id = post_to_x(args.image, args.text)
        print(f"✅ X post published: {tweet_id}")
        print(f"TWEET_ID={tweet_id}")
    except Exception as e:
        print(f"❌ X post failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()