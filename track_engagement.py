#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_engagement.py — Track engagement metrics for posted content.
"""

import os
import sys
import json
import argparse
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta

# API credentials
IG_USER_ID = os.getenv("IG_USER_ID", "")
IG_ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN", "")
FB_PAGE_ID = os.getenv("FB_PAGE_ID", "")
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "")
X_API_KEY = os.getenv("X_API_KEY", "")
X_API_SECRET = os.getenv("X_API_SECRET", "")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_SECRET = os.getenv("X_ACCESS_SECRET", "")
LINKEDIN_ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN", "")

GRAPH_API_BASE = "https://graph.facebook.com/v20.0"
X_API_BASE = "https://api.twitter.com/2"
LINKEDIN_API_BASE = "https://api.linkedin.com/v2"

METRICS_FILE = Path(__file__).parent / "logs" / "engagement_metrics.json"

def load_metrics() -> dict:
    """Load existing metrics."""
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"posts": [], "daily_totals": {}}

def save_metrics(metrics: dict):
    """Save metrics to file."""
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

def get_instagram_metrics(post_id: str) -> dict:
    """Get Instagram post metrics."""
    if not IG_ACCESS_TOKEN:
        return {}
    
    url = f"{GRAPH_API_BASE}/{post_id}/insights"
    params = {
        "metric": "impressions,reach,likes,comments,shares,saves,engagement",
        "access_token": IG_ACCESS_TOKEN,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 200:
        data = resp.json().get("data", [])
        return {item["name"]: item["values"][0]["value"] for item in data}
    return {}

def get_facebook_metrics(post_id: str) -> dict:
    """Get Facebook post metrics."""
    if not FB_PAGE_ACCESS_TOKEN:
        return {}
    
    # Extract post ID from post_id (format: PAGE_ID_POST_ID)
    if "_" in post_id:
        actual_post_id = post_id.split("_")[1]
    else:
        actual_post_id = post_id
    
    url = f"{GRAPH_API_BASE}/{actual_post_id}/insights"
    params = {
        "metric": "post_impressions,post_reach,post_engaged_users,post_reactions_like_total,post_comments,post_shares",
        "access_token": FB_PAGE_ACCESS_TOKEN,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 200:
        data = resp.json().get("data", [])
        return {item["name"]: item["values"][0]["value"] for item in data}
    return {}

def get_x_metrics(tweet_id: str) -> dict:
    """Get X/Twitter post metrics."""
    if not all([X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET]):
        return {}
    
    from requests_oauthlib import OAuth1Session
    oauth = OAuth1Session(
        X_API_KEY,
        client_secret=X_API_SECRET,
        resource_owner_key=X_ACCESS_TOKEN,
        resource_owner_secret=X_ACCESS_SECRET,
    )
    
    url = f"{X_API_BASE}/tweets/{tweet_id}"
    params = {
        "tweet.fields": "public_metrics,organic_metrics,promoted_metrics",
    }
    resp = oauth.get(url, params=params, timeout=30)
    if resp.status_code == 200:
        metrics = resp.json().get("data", {}).get("public_metrics", {})
        return {
            "impressions": metrics.get("impression_count", 0),
            "likes": metrics.get("like_count", 0),
            "retweets": metrics.get("retweet_count", 0),
            "replies": metrics.get("reply_count", 0),
            "quotes": metrics.get("quote_count", 0),
            "engagement": metrics.get("like_count", 0) + metrics.get("retweet_count", 0) + metrics.get("reply_count", 0),
        }
    return {}

def get_linkedin_metrics(post_id: str) -> dict:
    """Get LinkedIn post metrics."""
    if not LINKEDIN_ACCESS_TOKEN:
        return {}
    
    headers = {
        "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    
    # LinkedIn uses different endpoint for metrics
    url = f"{LINKEDIN_API_BASE}/socialActions/{post_id}"
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        return {
            "likes": data.get("likesSummary", {}).get("totalLikes", 0),
            "comments": data.get("commentsSummary", {}).get("totalComments", 0),
            "shares": data.get("sharesSummary", {}).get("totalShares", 0),
        }
    return {}

def track_post_engagement(platform: str, post_id: str, topic: str) -> dict:
    """Track engagement for a single post."""
    print(f"📊 Tracking {platform} engagement for post: {post_id}")
    
    metrics = {}
    if platform == "instagram":
        metrics = get_instagram_metrics(post_id)
    elif platform == "facebook":
        metrics = get_facebook_metrics(post_id)
    elif platform == "x" or platform == "twitter":
        metrics = get_x_metrics(post_id)
    elif platform == "linkedin":
        metrics = get_linkedin_metrics(post_id)
    
    # Calculate engagement rate
    impressions = metrics.get("impressions", metrics.get("post_impressions", 1))
    total_engagement = (
        metrics.get("likes", 0) + 
        metrics.get("comments", 0) + 
        metrics.get("shares", 0) + 
        metrics.get("saves", 0) +
        metrics.get("retweets", 0) +
        metrics.get("replies", 0)
    )
    engagement_rate = (total_engagement / impressions * 100) if impressions > 0 else 0
    
    return {
        "platform": platform,
        "post_id": post_id,
        "topic": topic,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "total_engagement": total_engagement,
        "engagement_rate": round(engagement_rate, 2),
    }

def main():
    parser = argparse.ArgumentParser(description="Track engagement metrics")
    parser.add_argument("--topic", required=True, help="Topic name")
    parser.add_argument("--platform", choices=["instagram", "facebook", "x", "twitter", "linkedin", "all"], default="all")
    parser.add_argument("--post-id", help="Specific post ID to track")
    parser.add_argument("--delay", type=int, default=0, help="Delay before tracking (seconds)")
    args = parser.parse_args()
    
    if args.delay > 0:
        print(f"⏳ Waiting {args.delay}s before tracking...")
        time.sleep(args.delay)
    
    metrics_data = load_metrics()
    
    # If post-id provided, track that specific post
    if args.post_id:
        platforms = [args.platform] if args.platform != "all" else ["instagram", "facebook", "x", "linkedin"]
        for platform in platforms:
            result = track_post_engagement(platform, args.post_id, args.topic)
            metrics_data["posts"].append(result)
            print(f"   {platform}: {result['total_engagement']} engagement ({result['engagement_rate']}% rate)")
    else:
        # Track all recent posts for this topic
        today = datetime.utcnow().date().isoformat()
        if today not in metrics_data["daily_totals"]:
            metrics_data["daily_totals"][today] = {"total_engagement": 0, "posts": 0}
        
        # This would need a post registry - for now just track what we can
        print("ℹ️ No specific post ID provided. Use --post-id to track a specific post.")
    
    save_metrics(metrics_data)
    print(f"✅ Engagement tracking complete. Metrics saved to {METRICS_FILE}")

if __name__ == "__main__":
    main()