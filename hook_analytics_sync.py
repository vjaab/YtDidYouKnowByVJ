#!/usr/bin/env python3
"""
hook_analytics_sync.py — Sync YouTube Analytics with Hook Performance
Run periodically (e.g., daily via cron) to update hook analytics with real view/retention data.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hook_analytics import (
    _load_analytics,
    _save_analytics,
    get_category_leaderboard,
    print_analytics_summary
)

try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    print("⚠️ googleapiclient not installed. Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

# YouTube Analytics API scopes
SCOPES = ['https://www.googleapis.com/auth/yt-analytics.readonly', 'https://www.googleapis.com/auth/youtube.readonly']

TOKEN_FILE = Path(__file__).parent / "token_youtube_analytics.json"
CREDENTIALS_FILE = Path(__file__).parent / "credentials_youtube.json"


def get_youtube_analytics_service():
    """Authenticate and return YouTube Analytics service."""
    if not YOUTUBE_API_AVAILABLE:
        return None
    
    creds = None
    
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                print(f"❌ Credentials file not found: {CREDENTIALS_FILE}")
                print("   Download from Google Cloud Console > APIs & Services > Credentials")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    
    return build('youtubeAnalytics', 'v2', credentials=creds)


def fetch_video_analytics(service, video_id, start_date: str, end_date: str) -> dict:
    """
    Fetch analytics for a specific video.
    Returns dict with views, avgViewDuration, engagement metrics.
    """
    try:
        response = service.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics="views,estimatedMinutesWatched,averageViewDuration,subscribersGained,likes,comments,shares",
            dimensions="video",
            filters=f"video=={video_id}",
            sort="-views"
        ).execute()
        
        rows = response.get("rows", [])
        if rows:
            row = rows[0]
            return {
                "views": row[0],
                "estimated_minutes_watched": row[1],
                "avg_view_duration_sec": row[2],
                "subscribers_gained": row[3],
                "likes": row[4],
                "comments": row[5],
                "shares": row[6]
            }
    except Exception as e:
        print(f"⚠️ Failed to fetch analytics for {video_id}: {e}")
    
    return {}


def fetch_recent_videos_analytics(days_back: int = 7) -> dict:
    """
    Fetch analytics for all videos published in the last N days.
    Returns dict mapping video_id -> analytics dict.
    """
    if not YOUTUBE_API_AVAILABLE:
        return {}
    
    service = get_youtube_analytics_service()
    if not service:
        return {}
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    # First get list of recent videos
    try:
        yt_service = build('youtube', 'v3', credentials=service._credentials)
        response = yt_service.search().list(
            part="id",
            channelId="UC_x5XG1OV2P6uZZ5FSM9Ttw",  # Replace with actual channel ID
            order="date",
            publishedAfter=(datetime.now() - timedelta(days=days_back)).isoformat("T") + "Z",
            maxResults=50,
            type="video"
        ).execute()
        
        video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
        print(f"📹 Found {len(video_ids)} recent videos")
        
        # Fetch analytics for each video
        results = {}
        for vid in video_ids:
            analytics = fetch_video_analytics(service, vid, start_date, end_date)
            if analytics:
                results[vid] = analytics
        
        return results
        
    except Exception as e:
        print(f"⚠️ Failed to fetch recent videos: {e}")
        return {}


def sync_hook_analytics_with_youtube(days_back: int = 7):
    """
    Main sync function: fetch YouTube analytics and update hook analytics.
    """
    print(f"🔄 Syncing hook analytics with YouTube (last {days_back} days)...")
    
    # Load current hook analytics
    analytics_data = _load_analytics()
    
    # Try to fetch YouTube analytics
    if YOUTUBE_API_AVAILABLE:
        yt_analytics = fetch_recent_videos_analytics(days_back)
        
        if yt_analytics:
            print(f"✅ Fetched analytics for {len(yt_analytics)} videos")
            
            # Update hook analytics with real data
            # This requires matching video_id to hook_pattern/variant
            # For now, we'd need a mapping from video_id -> hook info
            # which would come from the upload logs or a separate tracking DB
            
            print("📝 Note: To fully sync, implement video_id -> hook_pattern mapping")
            print("   This requires storing hook info at upload time with video_id")
        else:
            print("⚠️ No YouTube analytics data fetched")
    else:
        print("⚠️ YouTube Analytics API not available - skipping live sync")
    
    # Print current analytics summary
    print_analytics_summary()
    
    # Print category leaderboards
    categories = analytics_data.get("categories", {})
    for category in categories:
        print(f"\n🏆 Top hooks for {category}:")
        leaderboard = get_category_leaderboard(category, top_n=3)
        for i, entry in enumerate(leaderboard, 1):
            print(f"  {i}. {entry['pattern']}/{entry['variant']}: {entry['views']} views, {entry['retention']:.1%} ret, {entry['engagement']:.1%} eng")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sync hook analytics with YouTube")
    parser.add_argument("--days", type=int, default=7, help="Days back to fetch analytics")
    parser.add_argument("--print-only", action="store_true", help="Only print current analytics")
    args = parser.parse_args()
    
    if args.print_only:
        print_analytics_summary()
    else:
        sync_hook_analytics_with_youtube(args.days)


if __name__ == "__main__":
    main()