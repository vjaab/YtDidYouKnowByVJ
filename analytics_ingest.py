"""
analytics_ingest.py — YouTube Analytics ingestion for performance feedback loop.
Fetches video metrics and updates topic scoring for better content selection.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from config import (
    YOUTUBE_ANALYTICS_API_KEY,
    YOUTUBE_CHANNEL_ID,
    YOUTUBE_CLIENT_SECRET_FILE,
    LOGS_DIR,
    BASE_DIR
)


# Scopes for YouTube Analytics API
SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly"
]

# Metrics to fetch
VIDEO_METRICS = [
    "views",
    "estimatedMinutesWatched",
    "averageViewDuration",
    "averageViewPercentage",
    "subscribersGained",
    "likes",
    "comments",
    "shares",
    "annotationClickThroughRate",
    "cardClickThroughRate",
    "endScreenClickThroughRate"
]

DIMENSIONS = ["video", "day"]


@dataclass
class VideoMetrics:
    """Video performance metrics from YouTube Analytics."""
    video_id: str
    video_title: str
    date: str
    views: int
    estimated_minutes_watched: int
    average_view_duration: float  # seconds
    average_view_percentage: float
    subscribers_gained: int
    likes: int
    comments: int
    shares: int
    ctr_annotation: float
    ctr_card: float
    ctr_end_screen: float
    
    # Computed fields
    retention_rate: float = 0.0
    engagement_rate: float = 0.0
    
    def __post_init__(self):
        if self.average_view_duration > 0:
            # Need video duration for retention rate - placeholder
            pass
        if self.views > 0:
            self.engagement_rate = (self.likes + self.comments + self.shares) / self.views


class YouTubeAnalyticsClient:
    """Client for YouTube Analytics API."""
    
    def __init__(self):
        self.analytics = None
        self.youtube = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with YouTube Analytics API."""
        creds = None
        token_path = Path(BASE_DIR) / "token_analytics.json"
        
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not Path(YOUTUBE_CLIENT_SECRET_FILE).exists():
                    raise FileNotFoundError(
                        f"Client secret file not found: {YOUTUBE_CLIENT_SECRET_FILE}"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    YOUTUBE_CLIENT_SECRET_FILE, SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            with open(token_path, "w") as token:
                token.write(creds.to_json())
        
        self.analytics = build("youtubeAnalytics", "v2", credentials=creds)
        self.youtube = build("youtube", "v3", credentials=creds)
    
    def get_video_metrics(
        self,
        video_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[VideoMetrics]:
        """Fetch metrics for specific videos."""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        all_metrics = []
        
        for video_id in video_ids:
            try:
                # Get video title
                video_response = self.youtube.videos().list(
                    part="snippet",
                    id=video_id
                ).execute()
                
                if not video_response.get("items"):
                    continue
                
                video_title = video_response["items"][0]["snippet"]["title"]
                
                # Get analytics
                response = self.analytics.reports().query(
                    ids=f"channel=={YOUTUBE_CHANNEL_ID}",
                    startDate=start_date,
                    endDate=end_date,
                    metrics=",".join(VIDEO_METRICS),
                    dimensions="day",
                    filters=f"video=={video_id}",
                    sort="day"
                ).execute()
                
                rows = response.get("rows", [])
                for row in rows:
                    metrics = VideoMetrics(
                        video_id=video_id,
                        video_title=video_title,
                        date=row[0],
                        views=row[1],
                        estimated_minutes_watched=row[2],
                        average_view_duration=row[3],
                        average_view_percentage=row[4],
                        subscribers_gained=row[5],
                        likes=row[6],
                        comments=row[7],
                        shares=row[8],
                        ctr_annotation=row[9],
                        ctr_card=row[10],
                        ctr_end_screen=row[11]
                    )
                    all_metrics.append(metrics)
                    
            except HttpError as e:
                print(f"⚠️ Error fetching metrics for {video_id}: {e}")
                continue
        
        return all_metrics
    
    def get_channel_performance(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get aggregate channel performance."""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            response = self.analytics.reports().query(
                ids=f"channel=={YOUTUBE_CHANNEL_ID}",
                startDate=start_date,
                endDate=end_date,
                metrics="views,estimatedMinutesWatched,averageViewDuration,subscribersGained,likes,comments,shares",
                dimensions="day",
                sort="day"
            ).execute()
            
            rows = response.get("rows", [])
            total_views = sum(r[1] for r in rows)
            total_minutes = sum(r[2] for r in rows)
            avg_duration = sum(r[3] for r in rows) / len(rows) if rows else 0
            total_subs = sum(r[4] for r in rows)
            total_likes = sum(r[5] for r in rows)
            total_comments = sum(r[6] for r in rows)
            total_shares = sum(r[7] for r in rows)
            
            return {
                "period": f"{start_date} to {end_date}",
                "total_views": total_views,
                "total_watch_time_minutes": total_minutes,
                "average_view_duration_seconds": avg_duration,
                "subscribers_gained": total_subs,
                "total_likes": total_likes,
                "total_comments": total_comments,
                "total_shares": total_shares,
                "daily_breakdown": rows
            }
        except HttpError as e:
            print(f"⚠️ Error fetching channel performance: {e}")
            return {}
    
    def get_top_videos(
        self,
        max_results: int = 20,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """Get top performing videos by views."""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            response = self.analytics.reports().query(
                ids=f"channel=={YOUTUBE_CHANNEL_ID}",
                startDate=start_date,
                endDate=end_date,
                metrics="views,averageViewDuration,averageViewPercentage,likes,comments,shares,subscribersGained",
                dimensions="video",
                sort="-views",
                maxResults=max_results
            ).execute()
            
            rows = response.get("rows", [])
            video_ids = [r[0] for r in rows]
            
            # Batch fetch video titles
            titles = {}
            if video_ids:
                for i in range(0, len(video_ids), 50):
                    batch = video_ids[i:i+50]
                    vid_response = self.youtube.videos().list(
                        part="snippet",
                        id=",".join(batch)
                    ).execute()
                    for item in vid_response.get("items", []):
                        titles[item["id"]] = item["snippet"]["title"]
            
            results = []
            for row in rows:
                results.append({
                    "video_id": row[0],
                    "title": titles.get(row[0], "Unknown"),
                    "views": row[1],
                    "avg_view_duration": row[2],
                    "avg_view_percentage": row[3],
                    "likes": row[4],
                    "comments": row[5],
                    "shares": row[6],
                    "subscribers_gained": row[7]
                })
            
            return results
            
        except HttpError as e:
            print(f"⚠️ Error fetching top videos: {e}")
            return []


# ─── Feedback Loop Integration ───

METRICS_CACHE_FILE = Path(LOGS_DIR) / "youtube_metrics_cache.json"
TOPIC_PERFORMANCE_FILE = Path(LOGS_DIR) / "topic_performance.json"


def load_metrics_cache() -> Dict:
    """Load cached metrics."""
    if METRICS_CACHE_FILE.exists():
        try:
            with open(METRICS_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_metrics_cache(cache: Dict):
    """Save metrics cache."""
    try:
        with open(METRICS_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save metrics cache: {e}")


def load_topic_performance() -> Dict:
    """Load topic performance tracking."""
    if TOPIC_PERFORMANCE_FILE.exists():
        try:
            with open(TOPIC_PERFORMANCE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_topic_performance(performance: Dict):
    """Save topic performance."""
    try:
        with open(TOPIC_PERFORMANCE_FILE, "w") as f:
            json.dump(performance, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save topic performance: {e}")


def update_topic_performance_from_metrics(metrics: List[VideoMetrics]):
    """Update topic performance scores based on video metrics."""
    performance = load_topic_performance()
    cache = load_metrics_cache()
    
    for m in metrics:
        video_id = m.video_id
        
        # Store raw metrics
        cache[video_id] = {
            "last_updated": datetime.now().isoformat(),
            "title": m.video_title,
            "metrics": asdict(m)
        }
        
        # Extract topic/category from title (simple heuristic)
        title_lower = m.video_title.lower()
        
        # Map to categories
        category = "Unknown"
        if any(k in title_lower for k in ["github", "repo", "repository"]):
            category = "GitHub Repos"
        elif any(k in title_lower for k in ["tool", "alternative", "free"]):
            category = "AI Tools"
        elif any(k in title_lower for k in ["interview", "question"]):
            category = "Interview Prep"
        elif any(k in title_lower for k in ["quiz", "trivia", "fact"]):
            category = "Quiz & Trivia"
        elif any(k in title_lower for k in ["bug", "glitch", "fail"]):
            category = "Bugs & Glitches"
        elif any(k in title_lower for k in ["found", "story", "history"]):
            category = "Founding Stories"
        elif any(k in title_lower for k in ["hack", "tip", "trick", "shortcut"]):
            category = "Dev Hacks"
        
        # Update performance score
        if category not in performance:
            performance[category] = {
                "total_views": 0,
                "total_videos": 0,
                "avg_retention": 0.0,
                "avg_engagement": 0.0,
                "total_subs_gained": 0,
                "videos": []
            }
        
        perf = performance[category]
        perf["total_views"] += m.views
        perf["total_videos"] += 1
        perf["total_subs_gained"] += m.subscribers_gained
        perf["avg_retention"] = (
            (perf["avg_retention"] * (perf["total_videos"] - 1) + m.average_view_percentage) 
            / perf["total_videos"]
        )
        perf["avg_engagement"] = (
            (perf["avg_engagement"] * (perf["total_videos"] - 1) + m.engagement_rate) 
            / perf["total_videos"]
        )
        perf["videos"].append({
            "video_id": video_id,
            "title": m.video_title,
            "views": m.views,
            "retention": m.average_view_percentage,
            "engagement": m.engagement_rate,
            "date": m.date
        })
        
        # Keep only last 20 videos per category
        perf["videos"] = perf["videos"][-20:]
    
    save_metrics_cache(cache)
    save_topic_performance(performance)
    print(f"✅ Updated topic performance from {len(metrics)} video metrics")


def get_topic_performance_scores() -> Dict[str, float]:
    """Get normalized performance scores for topic selection."""
    performance = load_topic_performance()
    if not performance:
        return {}
    
    # Calculate composite scores
    scores = {}
    max_views = max((p["total_views"] for p in performance.values()), default=1)
    max_retention = max((p["avg_retention"] for p in performance.values()), default=1)
    max_engagement = max((p["avg_engagement"] for p in performance.values()), default=1)
    max_subs = max((p["total_subs_gained"] for p in performance.values()), default=1)
    
    for category, data in performance.items():
        if data["total_videos"] < 2:
            continue  # Need minimum data
        
        # Weighted composite score
        score = (
            (data["total_views"] / max_views) * 0.3 +
            (data["avg_retention"] / max_retention) * 0.4 +
            (data["avg_engagement"] / max_engagement) * 0.2 +
            (data["total_subs_gained"] / max_subs) * 0.1
        )
        scores[category] = round(score, 4)
    
    return scores


def fetch_and_update_analytics(days_back: int = 30):
    """Main entry point: fetch analytics and update performance tracking."""
    print(f"📊 Fetching YouTube Analytics (last {days_back} days)...")
    
    client = YouTubeAnalyticsClient()
    
    # Get channel performance
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    channel_perf = client.get_channel_performance(start_date, end_date)
    print(f"   Channel: {channel_perf.get('total_views', 0):,} views, "
          f"{channel_perf.get('subscribers_gained', 0)} subs gained")
    
    # Get top videos
    top_videos = client.get_top_videos(max_results=50, start_date=start_date, end_date=end_date)
    print(f"   Top {len(top_videos)} videos fetched")
    
    # Get detailed metrics for top videos
    video_ids = [v["video_id"] for v in top_videos]
    metrics = client.get_video_metrics(video_ids, start_date, end_date)
    
    # Update topic performance
    update_topic_performance_from_metrics(metrics)
    
    # Get scores
    scores = get_topic_performance_scores()
    print(f"   Topic scores: {scores}")
    
    return {
        "channel_performance": channel_perf,
        "top_videos": top_videos,
        "topic_scores": scores
    }


# ─── Standalone Test ───

if __name__ == "__main__":
    print("=" * 60)
    print("YOUTUBE ANALYTICS INGESTION TEST")
    print("=" * 60)
    
    # Check config
    print(f"YOUTUBE_ANALYTICS_API_KEY: {'✅ Set' if YOUTUBE_ANALYTICS_API_KEY else '❌ Missing'}")
    print(f"YOUTUBE_CHANNEL_ID: {'✅ Set' if YOUTUBE_CHANNEL_ID else '❌ Missing'}")
    print(f"YOUTUBE_CLIENT_SECRET_FILE: {'✅ Exists' if Path(YOUTUBE_CLIENT_SECRET_FILE).exists() else '❌ Missing'}")
    
    if not YOUTUBE_ANALYTICS_API_KEY or not YOUTUBE_CHANNEL_ID:
        print("\n⚠️ Skipping live test - configure credentials first")
        print("Required environment variables:")
        print("  YOUTUBE_ANALYTICS_API_KEY")
        print("  YOUTUBE_CHANNEL_ID")
        print("  YOUTUBE_CLIENT_SECRET_FILE (path to client_secret.json)")
    else:
        print("\n🔄 Running live analytics fetch...")
        try:
            result = fetch_and_update_analytics(days_back=7)
            print(f"\n✅ Analytics fetch complete!")
            print(f"   Topic scores: {result['topic_scores']}")
        except Exception as e:
            print(f"❌ Error: {e}")