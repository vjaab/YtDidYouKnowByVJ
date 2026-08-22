#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_to_linkedin.py — Post images to LinkedIn via API.
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path

LINKEDIN_ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN", "")
LINKEDIN_API_BASE = "https://api.linkedin.com/v2"

def register_upload() -> tuple:
    """Register an upload with LinkedIn and return upload URL and asset URN."""
    if not LINKEDIN_ACCESS_TOKEN:
        raise ValueError("LinkedIn access token not configured")
    
    headers = {
        "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    
    payload = {
        "registerUploadRequest": {
            "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
            "owner": "urn:li:person:YOUR_PERSON_URN",  # Need to get this from /me
            "serviceRelationships": [
                {
                    "relationshipType": "OWNER",
                    "identifier": "urn:li:userGeneratedContent",
                }
            ],
        }
    }
    
    resp = requests.post(
        f"{LINKEDIN_API_BASE}/assets?action=registerUpload",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()["value"]
    upload_url = data["uploadMechanism"]["com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"]["uploadUrl"]
    asset_urn = data["asset"]
    return upload_url, asset_urn

def upload_image(upload_url: str, image_path: str):
    """Upload image to LinkedIn's upload URL."""
    with open(image_path, "rb") as f:
        headers = {"Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}"}
        resp = requests.post(upload_url, headers=headers, data=f, timeout=60)
    resp.raise_for_status()

def create_post(text: str, asset_urn: str) -> str:
    """Create a LinkedIn post with the uploaded image."""
    headers = {
        "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    
    # Get person URN
    me_resp = requests.get(
        f"{LINKEDIN_API_BASE}/me",
        headers=headers,
        timeout=30,
    )
    me_resp.raise_for_status()
    person_urn = me_resp.json()["id"]
    
    payload = {
        "author": f"urn:li:person:{person_urn}",
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text},
                "shareMediaCategory": "IMAGE",
                "media": [{"status": "READY", "media": asset_urn}],
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }
    
    resp = requests.post(
        f"{LINKEDIN_API_BASE}/ugcPosts",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]

def post_to_linkedin(image_path: str, text: str) -> str:
    """Post image to LinkedIn."""
    print(f"💼 Posting to LinkedIn: {image_path}")
    print(f"   Text: {text[:100]}...")
    
    upload_url, asset_urn = register_upload()
    upload_image(upload_url, image_path)
    post_id = create_post(text, asset_urn)
    return post_id

def main():
    parser = argparse.ArgumentParser(description="Post to LinkedIn")
    parser.add_argument("--image", required=True, help="Image file path")
    parser.add_argument("--text", required=True, help="Post text")
    args = parser.parse_args()
    
    try:
        post_id = post_to_linkedin(args.image, args.text)
        print(f"✅ LinkedIn post published: {post_id}")
        print(f"POST_ID={post_id}")
    except Exception as e:
        print(f"❌ LinkedIn post failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()