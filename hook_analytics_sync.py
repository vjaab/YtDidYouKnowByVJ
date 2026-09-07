#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hook_analytics_sync.py -- Sync YouTube Analytics with Hook Performance

Run periodically (daily via the hook-analytics-sync.yml workflow, and once per
upload via scripts/ci_analytics_check.py) to pull real view/retention data for
recently-uploaded Shorts and feed it back into hook_analytics.json so that
select_hook_patterns_for_category() actually learns from real audience behavior.

How the loop closes:
  1. main.py writes logs/hook_video_map.json at upload time:
         video_id -> {hook_pattern, hook_variant, category, title, uploaded_at, synced}
  2. This script finds entries that are old enough to have stable retention
     data (default: 48h+) and haven't been synced yet.
  3. For each, it pulls lifetime-to-date YouTube Analytics (views, average
     view duration, likes/comments/shares) plus the video's actual duration
     from the Data API, computes retention_rate and engagement_rate, and
     calls record_hook_performance() so the real numbers land in
     hook_analytics.json.
  4. The entry is marked "synced" (with the raw metrics snapshot attached)
     so it's only ever recorded once -- re-running this script is always
     safe and won't double-count views.
"""

import os
import re
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import LOGS_DIR
from hook_analytics import (
    _load_analytics,
    record_hook_performance,
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
    print("[WARNING] googleapiclient not installed. Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

# YouTube Analytics API scopes
SCOPES = ['https://www.googleapis.com/auth/yt-analytics.readonly', 'https://www.googleapis.com/auth/youtube.readonly']

TOKEN_FILE = Path(__file__).parent / "token_youtube_analytics.json"
CREDENTIALS_FILE = Path(__file__).parent / "client_secret.json"

HOOK_VIDEO_MAP_FILE = Path(LOGS_DIR) / "hook_video_map.json"

# Don't trust retention numbers until a Short has had time to find its
# audience. 48h comfortably covers the first-day and second-day traffic bump.
DEFAULT_MIN_AGE_HOURS = 48


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
                print(f"[ERROR] Credentials file not found: {CREDENTIALS_FILE}")
                print("   Download from Google Cloud Console > APIs & Services > Credentials")
                return None
            if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
                print("[ERROR] YouTube Analytics token missing or invalid in CI environment. Run locally to authenticate.")
                print("   (This means YOUTUBE_ANALYTICS_TOKEN_JSON is not set as a repo secret yet,")
                print("    or the token has expired and needs to be regenerated locally.)")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            # Use fixed port matching authorized redirect URI in Google Cloud Console
            creds = flow.run_local_server(port=54630)

        if creds:
            with open(TOKEN_FILE, "w") as token:
                token.write(creds.to_json())

    return creds


def _parse_iso8601_duration(duration: str) -> float:
    """Parse an ISO 8601 duration string like 'PT45S' or 'PT1M5S' into seconds."""
    if not duration:
        return 0.0
    match = re.match(
        r'PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?',
        duration
    )
    if not match:
        return 0.0
    parts = match.groupdict()
    hours = float(parts.get('hours') or 0)
    minutes = float(parts.get('minutes') or 0)
    seconds = float(parts.get('seconds') or 0)
    return hours * 3600 + minutes * 60 + seconds


def fetch_video_analytics(service, video_id, start_date: str, end_date: str) -> dict:
    """
    Fetch lifetime-to-date analytics for a specific video.
    Returns dict with views, avgViewDuration, engagement metrics, AND swipe-away proxy (averageViewPercentage).
    """
    try:
        response = service.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics="views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained,likes,comments,shares",
            dimensions="video",
            filters=f"video=={video_id}",
            sort="-views"
        ).execute()

        rows = response.get("rows", [])
        if rows:
            row = rows[0]
            avg_view_pct = row[3] if len(row) > 3 else 0
            swipe_away_rate = max(0.0, 1.0 - (avg_view_pct / 100.0)) if avg_view_pct else 1.0
            return {
                "views": row[0],
                "estimated_minutes_watched": row[1],
                "avg_view_duration_sec": row[2],
                "avg_view_percentage": avg_view_pct,
                "swipe_away_rate": swipe_away_rate,
                "subscribers_gained": row[4] if len(row) > 4 else 0,
                "likes": row[5] if len(row) > 5 else 0,
                "comments": row[6] if len(row) > 6 else 0,
                "shares": row[7] if len(row) > 7 else 0
            }
    except Exception as e:
        print(f"[WARNING] Failed to fetch analytics for {video_id}: {e}")
    
    return {}


def fetch_video_duration(yt_service, video_id: str) -> float:
    """Fetch a video's actual duration (seconds) via the Data API, for normalizing retention."""
    try:
        response = yt_service.videos().list(part="contentDetails", id=video_id).execute()
        items = response.get("items", [])
        if items:
            return _parse_iso8601_duration(items[0].get("contentDetails", {}).get("duration", ""))
    except Exception as e:
        print(f"[WARNING] Failed to fetch duration for {video_id}: {e}")
    return 0.0


def _load_hook_video_map() -> dict:
    """Load the video_id -> hook mapping written by main.py at upload time."""
    if HOOK_VIDEO_MAP_FILE.exists():
        try:
            with open(HOOK_VIDEO_MAP_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to read {HOOK_VIDEO_MAP_FILE}: {e}")
    return {}


def _save_hook_video_map(data: dict):
    HOOK_VIDEO_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HOOK_VIDEO_MAP_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _get_pending_videos(hook_video_map: dict, min_age_hours: int, max_age_days: int) -> dict:
    """
    Videos that are: not yet synced, old enough for retention to have
    stabilized, and not so old that we've given up on ever syncing them.
    """
    pending = {}
    now = datetime.now()
    for video_id, entry in hook_video_map.items():
        if entry.get("synced"):
            continue
        uploaded_at_str = entry.get("uploaded_at")
        if not uploaded_at_str:
            continue
        try:
            uploaded_at = datetime.fromisoformat(uploaded_at_str)
        except ValueError:
            continue
        age = now - uploaded_at
        if timedelta(hours=min_age_hours) <= age <= timedelta(days=max_age_days):
            pending[video_id] = entry
    return pending


def sync_hook_analytics_with_youtube(days_back: int = 7, min_age_hours: int = DEFAULT_MIN_AGE_HOURS):
    """
    Main sync function: find uploads that are old enough to trust, pull real
    YouTube Analytics for each, and record the result against the hook
    pattern/variant that produced it.

    `days_back` doubles as "give up trying to sync videos older than this
    many days" so existing callers (ci_analytics_check.py, main()) keep
    working unchanged.
    """
    print(f"[SYNC] Syncing hook analytics with YouTube (videos {min_age_hours}h-{days_back}d old)...")

    hook_video_map = _load_hook_video_map()
    if not hook_video_map:
        print(f"[WARNING] No {HOOK_VIDEO_MAP_FILE} found yet -- nothing to sync. "
              "It's written automatically the next time a Short uploads.")
        print_analytics_summary()
        return

    pending = _get_pending_videos(hook_video_map, min_age_hours, days_back)
    already_synced = sum(1 for e in hook_video_map.values() if e.get("synced"))
    print(f"[VIDEO] {len(pending)} video(s) pending sync · {already_synced} already synced · "
          f"{len(hook_video_map)} tracked total")

    if not pending:
        print_analytics_summary()
        return

    if not YOUTUBE_API_AVAILABLE:
        print("[WARNING] YouTube Analytics API not available - skipping live sync")
        return

    creds = get_youtube_analytics_service()
    if not creds:
        print("[WARNING] Could not get valid YouTube Analytics credentials - skipping live sync this run")
        return

    analytics_service = build('youtubeAnalytics', 'v2', credentials=creds)
    yt_service = build('youtube', 'v3', credentials=creds)

    synced_count = 0
    for video_id, entry in pending.items():
        category = entry.get("category")
        pattern_id = entry.get("hook_pattern")
        variant_id = entry.get("hook_variant")
        if not (category and pattern_id and variant_id):
            print(f"[WARNING] Skipping {video_id} -- hook mapping entry is incomplete")
            continue

        start_date = entry["uploaded_at"][:10]
        end_date = datetime.now().strftime("%Y-%m-%d")

        analytics = fetch_video_analytics(analytics_service, video_id, start_date, end_date)
        if not analytics or not analytics.get("views"):
            print(f"[WARNING] No analytics yet for {video_id} ({entry.get('title', '')[:40]!r}) -- will retry next run")
            continue

        duration_sec = fetch_video_duration(yt_service, video_id)
        views = int(analytics.get("views", 0) or 0)
        avg_view_duration = float(analytics.get("avg_view_duration_sec", 0) or 0)
        avg_view_percentage = float(analytics.get("avg_view_percentage", 0) or 0)
        swipe_away_rate = float(analytics.get("swipe_away_rate", 1.0) or 1.0)
        likes = int(analytics.get("likes", 0) or 0)
        comments = int(analytics.get("comments", 0) or 0)
        shares = int(analytics.get("shares", 0) or 0)

        retention_rate = min(avg_view_duration / duration_sec, 1.0) if duration_sec > 0 else 0.0
        engagement_rate = (likes + comments + shares) / views if views > 0 else 0.0

        record_hook_performance(
            category=category,
            pattern_id=pattern_id,
            variant_id=variant_id,
            views=views,
            retention_rate=retention_rate,
            engagement_rate=engagement_rate,
            swipe_away_rate=swipe_away_rate,
            avg_view_percentage=avg_view_percentage
        )

        entry["synced"] = True
        entry["synced_at"] = datetime.now().isoformat()
        entry["last_metrics"] = {
            "views": views,
            "avg_view_duration_sec": avg_view_duration,
            "avg_view_percentage": round(avg_view_percentage, 2),
            "swipe_away_rate": round(swipe_away_rate, 4),
            "duration_sec": duration_sec,
            "retention_rate": round(retention_rate, 4),
            "engagement_rate": round(engagement_rate, 4),
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "subscribers_gained": analytics.get("subscribers_gained", 0),
        }
        hook_video_map[video_id] = entry
        synced_count += 1

        print(f"[OK] Synced {video_id} [{category}/{pattern_id}/{variant_id}]: "
              f"{views} views, {retention_rate:.1%} retention, {engagement_rate:.1%} engagement, "
              f"swipe_away: {swipe_away_rate:.1%}")

    if synced_count:
        _save_hook_video_map(hook_video_map)
        print(f"[SAVE] Recorded {synced_count} new video(s) into hook_analytics.json "
              f"and marked them synced in {HOOK_VIDEO_MAP_FILE.name}")
    else:
        print("[INFO] Nothing had analytics data available to sync this run.")

    # Print current analytics summary + leaderboards
    print_analytics_summary()
    analytics_data = _load_analytics()
    categories = analytics_data.get("categories", {})
    for category in categories:
        print(f"\n[TROPHY] Top hooks for {category}:")
        leaderboard = get_category_leaderboard(category, top_n=3)
        for i, item in enumerate(leaderboard, 1):
            swipe = item.get('swipe_away_rate', 0)
            print(f"  {i}. {item['pattern']}/{item['variant']}: {item['views']} views, "
                  f"{item['retention']:.1%} ret, {item['engagement']:.1%} eng, "
                  f"swipe_away: {swipe:.1%}")


def get_underperforming_hooks(min_views: int = 50, max_swipe_away: float = 0.7) -> list:
    """
    Identify hooks with high swipe-away rates (>70%) that need rewriting.
    Returns list of dicts with category, pattern, variant, hook_text, swipe_away_rate, views.
    """
    from hook_analytics import get_hook_analytics
    data = get_hook_analytics()
    
    underperforming = []
    for category, cat_data in data.get("categories", {}).items():
        for pattern_id, pattern_data in cat_data.items():
            for variant_id, variant_data in pattern_data.get("variants", {}).items():
                views = variant_data.get("views", 0)
                swipe_away = variant_data.get("swipe_away_rate", 0)
                retention = variant_data.get("retention", 0)
                
                if views >= min_views and (swipe_away > max_swipe_away or retention < 0.3):
                    hook_texts = variant_data.get("hook_texts", [])
                    latest_hook = hook_texts[-1].get("text", "") if hook_texts else ""
                    underperforming.append({
                        "category": category,
                        "pattern": pattern_id,
                        "variant": variant_id,
                        "hook_text": latest_hook,
                        "swipe_away_rate": swipe_away,
                        "retention": retention,
                        "views": views,
                        "hook_texts": [h.get("text", "") for h in hook_texts]
                    })
    
    underperforming.sort(key=lambda x: x["swipe_away_rate"], reverse=True)
    return underperforming


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sync hook analytics with YouTube")
    parser.add_argument("--days", type=int, default=7,
                         help="Give up trying to sync videos older than this many days")
    parser.add_argument("--min-age-hours", type=int, default=DEFAULT_MIN_AGE_HOURS,
                         help="Only sync videos at least this many hours old (lets retention stabilize)")
    parser.add_argument("--print-only", action="store_true", help="Only print current analytics")
    parser.add_argument("--find-underperforming", action="store_true", 
                         help="Print hooks with >70% swipe-away rate that need rewriting")
    args = parser.parse_args()

    if args.print_only:
        print_analytics_summary()
    elif args.find_underperforming:
        print("[SEARCH] Finding underperforming hooks (swipe-away > 70%)...")
        under = get_underperforming_hooks()
        if under:
            for u in under:
                print(f"\n[WARNING] {u['category']} / {u['pattern']} / {u['variant']}")
                print(f"   Swipe-away: {u['swipe_away_rate']:.1%} | Retention: {u['retention']:.1%} | Views: {u['views']}")
                print(f"   Latest hook: \"{u['hook_text']}\"")
                print(f"   All variants: {u['hook_texts']}")
        else:
            print("[OK] No underperforming hooks found.")
    else:
        sync_hook_analytics_with_youtube(args.days, args.min_age_hours)


if __name__ == "__main__":
    main()