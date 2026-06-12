"""
trending_engine.py — Unified Trending Signal Aggregator (Phase 1)

Fetches real-time engagement signals from platforms where AI content goes viral:
1. YouTube Trending Analysis (YouTube Data API v3)
2. Reddit Hot Posts (r/MachineLearning, r/LocalLLaMA, etc.)
3. GitHub Trending Repos (No API key needed)
4. Gemini Deep Trend Cross-Analysis

Returns a unified list of trending topics with virality scores.
"""

import os
import re
import json
import time
import requests
from datetime import datetime, timedelta, timezone
from config import (
    GEMINI_API_KEY, YOUTUBE_DATA_API_KEY, REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET, TRENDING_NICHE_BIAS
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. YOUTUBE TRENDING SHORTS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_youtube_trending_shorts():
    """
    Uses YouTube Data API v3 to find recently uploaded AI Shorts
    with high view counts — topics proven to get views in Shorts format.
    """
    if not YOUTUBE_DATA_API_KEY:
        print("⚠️ YouTube Data API key missing. Skipping YouTube trending fetch.")
        return []

    print("📺 Fetching trending AI Shorts from YouTube Data API...")
    
    search_queries = [
        "tech tips hidden features",
        "iPhone Android tricks 2026",
        "AI tools free productivity",
        "privacy security phone settings",
        "tech myths debunked",
        "free app alternative paid",
        "Windows shortcuts tricks",
    ]
    
    all_results = []
    
    for query in search_queries:
        try:
            # Search for recent Shorts (< 60s) with high view counts
            published_after = (datetime.now(timezone.utc) - timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%SZ")
            
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "videoDuration": "short",  # < 4 minutes (Shorts)
                "order": "viewCount",
                "publishedAfter": published_after,
                "maxResults": 5,
                "key": YOUTUBE_DATA_API_KEY,
                "relevanceLanguage": "en"
            }
            
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                print(f"  ⚠️ YouTube API Error ({r.status_code}): {r.text[:200]}")
                continue
                
            data = r.json()
            video_ids = [item["id"]["videoId"] for item in data.get("items", []) if "videoId" in item.get("id", {})]
            
            if not video_ids:
                continue
            
            # Get video statistics for engagement signals
            stats_url = "https://www.googleapis.com/youtube/v3/videos"
            stats_params = {
                "part": "statistics,snippet",
                "id": ",".join(video_ids),
                "key": YOUTUBE_DATA_API_KEY
            }
            
            stats_r = requests.get(stats_url, params=stats_params, timeout=15)
            if stats_r.status_code != 200:
                continue
                
            stats_data = stats_r.json()
            
            for item in stats_data.get("items", []):
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                
                views = int(stats.get("viewCount", 0))
                likes = int(stats.get("likeCount", 0))
                comments = int(stats.get("commentCount", 0))
                
                # Only include videos with meaningful engagement
                if views < 1000:
                    continue
                
                all_results.append({
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", "")[:300],
                    "source": {"name": f"YouTube ({snippet.get('channelTitle', 'Unknown')})"},
                    "url": f"https://youtube.com/shorts/{item['id']}",
                    "urlToImage": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                    "publishedAt": snippet.get("publishedAt", ""),
                    "type": "youtube_trending",
                    "_engagement": {
                        "views": views,
                        "likes": likes,
                        "comments": comments,
                        "like_ratio": (likes / max(views, 1)) * 100
                    }
                })
            
            time.sleep(0.5)  # Rate limit courtesy
            
        except Exception as e:
            print(f"  ⚠️ YouTube search failed for '{query}': {e}")
    
    # Deduplicate by title similarity
    seen = set()
    unique = []
    for r in all_results:
        title_key = re.sub(r'[^a-z0-9]', '', r["title"].lower())[:40]
        if title_key not in seen:
            seen.add(title_key)
            unique.append(r)
    
    print(f"✅ YouTube Trending: Found {len(unique)} high-performing AI Shorts.")
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# 2. REDDIT HOT POSTS
# ─────────────────────────────────────────────────────────────────────────────
def _get_reddit_token():
    """Get OAuth token for Reddit API."""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        return None
    try:
        auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        data = {"grant_type": "client_credentials"}
        headers = {"User-Agent": "VJTechNews/1.0"}
        r = requests.post("https://www.reddit.com/api/v1/access_token",
                          auth=auth, data=data, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json().get("access_token")
    except Exception as e:
        print(f"  ⚠️ Reddit OAuth failed: {e}")
    return None


def fetch_reddit_hot_ai():
    """
    Fetches hot posts from AI subreddits. Uses OAuth if credentials available,
    falls back to public JSON API otherwise.
    """
    print("🔴 Fetching hot AI posts from Reddit...")
    
    subreddits = [
        "LifeProTips",
        "techsupport",
        "Android",
        "iphone",
        "privacy",
        "technology",
        "gadgets",
        "MachineLearning",
        "ChatGPT",
        "LocalLLaMA",
    ]
    
    token = _get_reddit_token()
    headers = {"User-Agent": "VJTechNews/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        base_url = "https://oauth.reddit.com"
    else:
        base_url = "https://www.reddit.com"
    
    all_posts = []
    
    for sub in subreddits:
        try:
            if token:
                url = f"{base_url}/r/{sub}/hot.json?limit=10"
            else:
                url = f"{base_url}/r/{sub}/hot.json?limit=10"
            
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200:
                print(f"  ⚠️ Reddit r/{sub} failed ({r.status_code})")
                continue
            
            data = r.json()
            posts = data.get("data", {}).get("children", [])
            
            for post in posts:
                p = post.get("data", {})
                
                # Skip pinned/stickied and non-text posts
                if p.get("stickied") or p.get("is_video"):
                    continue
                
                ups = p.get("ups", 0)
                num_comments = p.get("num_comments", 0)
                upvote_ratio = p.get("upvote_ratio", 0.5)
                
                # Only include posts with meaningful engagement
                if ups < 50:
                    continue
                
                # Calculate age in hours for velocity scoring
                created_utc = p.get("created_utc", 0)
                age_hours = max(1, (time.time() - created_utc) / 3600)
                upvote_velocity = ups / age_hours  # Upvotes per hour
                
                all_posts.append({
                    "title": p.get("title", ""),
                    "description": (p.get("selftext", "") or p.get("title", ""))[:400],
                    "source": {"name": f"Reddit r/{sub}"},
                    "url": f"https://reddit.com{p.get('permalink', '')}",
                    "urlToImage": "",
                    "publishedAt": datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat() if created_utc else "",
                    "type": "reddit_trending",
                    "_engagement": {
                        "upvotes": ups,
                        "comments": num_comments,
                        "upvote_ratio": upvote_ratio,
                        "upvote_velocity": round(upvote_velocity, 1),
                        "age_hours": round(age_hours, 1)
                    }
                })
                
            time.sleep(1)  # Reddit rate limit: 1 req/sec
            
        except Exception as e:
            print(f"  ⚠️ Reddit r/{sub} fetch failed: {e}")
    
    # Sort by upvote velocity (fastest-rising posts first)
    all_posts.sort(key=lambda x: x.get("_engagement", {}).get("upvote_velocity", 0), reverse=True)
    
    print(f"✅ Reddit: Found {len(all_posts)} trending AI posts.")
    return all_posts[:20]  # Top 20


# ─────────────────────────────────────────────────────────────────────────────
# 3. GITHUB TRENDING REPOS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_github_trending_ai():
    """
    Scrapes GitHub Trending page for AI/ML repos.
    Stars velocity = early signal for tool content before mainstream coverage.
    No API key needed.
    """
    print("🐙 Fetching trending AI repos from GitHub...")
    
    try:
        url = "https://api.github.com/search/repositories"
        # Repos created or pushed in the last 7 days with AI/ML topics
        since_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        
        params = {
            "q": f"(\"tech tips\" OR \"hidden features\" OR \"life hack\" OR \"AI tool\" OR \"free alternative\" OR \"privacy\") pushed:>{since_date}",
            "sort": "stars",
            "order": "desc",
            "per_page": 15
        }
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "VJTechNews/1.0"
        }
        
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            print(f"  ⚠️ GitHub API Error ({r.status_code})")
            return []
        
        data = r.json()
        repos = data.get("items", [])
        
        results = []
        for repo in repos:
            stars = repo.get("stargazers_count", 0)
            if stars < 100:
                continue
            
            # Calculate approximate stars velocity
            created_at = repo.get("created_at", "")
            age_days = 1
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    age_days = max(1, (datetime.now(timezone.utc) - created_dt).days)
                except:
                    pass
            
            stars_per_day = stars / age_days
            
            results.append({
                "title": f"GitHub Trending: {repo.get('full_name', '')} — {repo.get('description', '')[:100]}",
                "description": f"⭐ {stars} stars ({stars_per_day:.0f}/day) | {repo.get('language', 'Unknown')} | {repo.get('description', '')}",
                "source": {"name": f"GitHub ({repo.get('full_name', '')})"},
                "url": repo.get("html_url", ""),
                "urlToImage": repo.get("owner", {}).get("avatar_url", ""),
                "publishedAt": created_at,
                "type": "github_trending",
                "_engagement": {
                    "stars": stars,
                    "stars_per_day": round(stars_per_day, 1),
                    "forks": repo.get("forks_count", 0),
                    "watchers": repo.get("watchers_count", 0)
                }
            })
        
        results.sort(key=lambda x: x["_engagement"]["stars_per_day"], reverse=True)
        print(f"✅ GitHub: Found {len(results)} trending AI repos.")
        return results
        
    except Exception as e:
        print(f"  ⚠️ GitHub trending fetch failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 4. UNIFIED TRENDING AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────
def compute_engagement_score(article):
    """
    Computes a unified engagement score (0-100) from platform-specific signals.
    This replaces the old keyword-only viral scoring.
    """
    eng = article.get("_engagement", {})
    art_type = article.get("type", "")
    
    score = 0.0
    
    if art_type == "youtube_trending":
        views = eng.get("views", 0)
        likes = eng.get("likes", 0)
        like_ratio = eng.get("like_ratio", 0)
        
        # View count tiers
        if views >= 500000: score += 40
        elif views >= 100000: score += 30
        elif views >= 50000: score += 25
        elif views >= 10000: score += 18
        elif views >= 1000: score += 10
        
        # Like ratio bonus (high engagement = quality content)
        if like_ratio > 5: score += 15
        elif like_ratio > 3: score += 10
        
        # Comments = discussion = viral potential
        comments = eng.get("comments", 0)
        if comments > 500: score += 15
        elif comments > 100: score += 10
        elif comments > 20: score += 5
    
    elif art_type == "reddit_trending":
        ups = eng.get("upvotes", 0)
        velocity = eng.get("upvote_velocity", 0)
        ratio = eng.get("upvote_ratio", 0.5)
        comments = eng.get("comments", 0)
        
        # Upvote velocity is the STRONGEST signal (how fast is it rising?)
        if velocity > 200: score += 35
        elif velocity > 100: score += 28
        elif velocity > 50: score += 20
        elif velocity > 20: score += 12
        elif velocity > 5: score += 6
        
        # Absolute upvote tiers
        if ups > 5000: score += 20
        elif ups > 1000: score += 15
        elif ups > 500: score += 10
        elif ups > 100: score += 5
        
        # High upvote ratio = consensus (not controversial spam)
        if ratio > 0.95: score += 10
        elif ratio > 0.90: score += 7
        
        # Discussion depth
        if comments > 200: score += 10
        elif comments > 50: score += 5
    
    elif art_type == "github_trending":
        stars_pd = eng.get("stars_per_day", 0)
        stars = eng.get("stars", 0)
        
        # Stars velocity (early breakout signal)
        if stars_pd > 500: score += 35
        elif stars_pd > 100: score += 28
        elif stars_pd > 50: score += 20
        elif stars_pd > 10: score += 12
        
        # Absolute stars
        if stars > 10000: score += 20
        elif stars > 5000: score += 15
        elif stars > 1000: score += 10
    
    elif art_type == "trending":
        # NewsAPI trending articles (legacy)
        score += 15
    
    else:
        # RSS articles (lowest signal strength)
        score += 5
    
    # Niche bias: Prefer topics from specialized sources
    niche_sources = ["reddit_trending", "github_trending"]
    if art_type in niche_sources:
        score += TRENDING_NICHE_BIAS * 15  # Up to +10.5 bonus for niche
    
    return min(100, score)


def fetch_all_trending_signals():
    """
    Master aggregator: fetches from all trending sources and returns
    a unified, scored article list ready for the pipeline.
    """
    print("\n🔥 === TRENDING ENGINE: Fetching Multi-Platform Signals === 🔥")
    
    all_articles = []
    
    # 1. YouTube Trending Shorts
    try:
        yt_articles = fetch_youtube_trending_shorts()
        all_articles.extend(yt_articles)
    except Exception as e:
        print(f"⚠️ YouTube trending failed: {e}")
    
    # 2. Reddit Hot Posts
    try:
        reddit_articles = fetch_reddit_hot_ai()
        all_articles.extend(reddit_articles)
    except Exception as e:
        print(f"⚠️ Reddit trending failed: {e}")
    
    # 3. GitHub Trending Repos
    try:
        github_articles = fetch_github_trending_ai()
        all_articles.extend(github_articles)
    except Exception as e:
        print(f"⚠️ GitHub trending failed: {e}")
    
    # 4. Compute unified engagement scores
    for art in all_articles:
        art["_engagement_score"] = compute_engagement_score(art)
    
    # Sort by engagement score
    all_articles.sort(key=lambda x: x.get("_engagement_score", 0), reverse=True)
    
    # Summary
    yt_count = sum(1 for a in all_articles if a.get("type") == "youtube_trending")
    reddit_count = sum(1 for a in all_articles if a.get("type") == "reddit_trending")
    github_count = sum(1 for a in all_articles if a.get("type") == "github_trending")
    
    print(f"\n📊 Trending Engine Summary: {len(all_articles)} total signals")
    print(f"   YouTube: {yt_count} | Reddit: {reddit_count} | GitHub: {github_count}")
    if all_articles:
        top = all_articles[0]
        print(f"   🏆 Top Signal: '{top['title'][:60]}...' (Score: {top.get('_engagement_score', 0)})")
    
    return all_articles
