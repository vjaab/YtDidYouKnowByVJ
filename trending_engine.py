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
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from config import (
    GEMINI_API_KEY, YOUTUBE_DATA_API_KEY, REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET, TRENDING_NICHE_BIAS
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. YOUTUBE TRENDING SHORTS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_youtube_trending_shorts(target_country="US"):
    """
    Uses YouTube Data API v3 to find recently uploaded AI Shorts
    with high view counts — topics proven to get views in Shorts format.
    """
    if not YOUTUBE_DATA_API_KEY:
        print("⚠️ YouTube Data API key missing. Skipping YouTube trending fetch.")
        return []

    print(f"📺 Fetching trending AI Shorts from YouTube Data API for region={target_country}...")
    
    search_queries = [
        "tech tips hidden features",
        "iPhone Android tricks 2026",
        "AI tools free productivity",
        "privacy security phone settings",
        "tech myths debunked",
        "free app alternative paid",
        "Windows shortcuts tricks",
        "online scam alerts identity theft",
        "smartphone battery saving tricks",
        "everyday life tech hacks",
        "photo video editing tricks viral",
        "shopping hacks budget apps",
        "smart home lifestyle organization",
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
                "relevanceLanguage": "en",
                "regionCode": target_country
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
        print("  ⚠️ Reddit OAuth credentials missing (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET).")
        print("    → Reddit blocks unauthenticated requests from cloud IPs (GitHub Actions).")
        print("    → Create a Reddit app at https://www.reddit.com/prefs/apps and add secrets.")
        return None
    try:
        auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        data = {"grant_type": "client_credentials"}
        headers = {"User-Agent": "VJTechNews/1.0 (by /u/vjaab)"}
        r = requests.post("https://www.reddit.com/api/v1/access_token",
                          auth=auth, data=data, headers=headers, timeout=10)
        if r.status_code == 200:
            token = r.json().get("access_token")
            if token:
                print("  ✅ Reddit OAuth token acquired successfully.")
            return token
        else:
            print(f"  ⚠️ Reddit OAuth failed (HTTP {r.status_code}): {r.text[:200]}")
    except Exception as e:
        print(f"  ⚠️ Reddit OAuth failed: {e}")
    return None


def fetch_reddit_via_google_news(sub):
    """
    Fallback: Query Google News RSS for hot posts in a specific subreddit
    since GHA IPs are blocked by Reddit directly.
    Engagement values are estimated baselines (not real), flagged accordingly.
    """
    import urllib.parse
    import xml.etree.ElementTree as ET
    
    print(f"  🔍 Reddit Fallback: Querying Google News RSS for r/{sub}...")
    articles = []
    
    try:
        query = f"site:reddit.com/r/{sub}"
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        items = root.findall('.//item')
        
        for item in items[:5]: # Limit to top 5 recent posts
            title = item.find('title').text or ""
            # Strip source suffix E.g. "Title - Subreddit - Source"
            title_clean = re.sub(r'\s+-\s+.*$', '', title).strip()
            link = item.find('link').text or ""
            pub_date = item.find('pubDate').text or ""
            
            # Low baseline engagement — flagged as estimated so scoring can penalize
            articles.append({
                "title": title_clean,
                "description": f"Google News indexed post from r/{sub}: {title_clean}",
                "source": {"name": f"Reddit r/{sub} (via News)"},
                "url": link,
                "urlToImage": "",
                "publishedAt": pub_date,
                "type": "reddit_trending",
                "_engagement": {
                    "upvotes": 10,
                    "comments": 2,
                    "upvote_ratio": 0.7,
                    "upvote_velocity": 1.0,
                    "age_hours": 24.0,
                    "engagement_estimated": True
                }
            })
    except Exception as e:
        print(f"  ⚠️ Reddit Fallback for r/{sub} failed: {e}")
        
    return articles


def fetch_reddit_hot_ai():
    """
    Fetches hot posts from AI subreddits. Uses OAuth if credentials available,
    falls back to old.reddit.com public JSON API otherwise.
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
    headers = {"User-Agent": "VJTechNews/1.0 (by /u/vjaab)"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        base_url = "https://oauth.reddit.com"
    else:
        # old.reddit.com is more permissive than www.reddit.com for unauthenticated access
        base_url = "https://old.reddit.com"
    
    all_posts = []
    consecutive_failures = 0  # Early-abort counter
    
    for sub in subreddits:
        # Early abort: if 3+ consecutive subreddits fail, Reddit is blocking us entirely
        if consecutive_failures >= 3:
            remaining = len(subreddits) - subreddits.index(sub)
            print(f"  🛑 Reddit: {consecutive_failures} consecutive failures. Aborting remaining {remaining} subreddits native calls.")
            if not token:
                print("    → Fix: Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET as GitHub Secrets.")
            
            # Fall back to Google News for the remaining subreddits
            for fallback_sub in subreddits[subreddits.index(sub):]:
                all_posts.extend(fetch_reddit_via_google_news(fallback_sub))
            break
        
        try:
            url = f"{base_url}/r/{sub}/hot.json?limit=10&raw_json=1"
            
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200:
                print(f"  ⚠️ Reddit r/{sub} failed ({r.status_code})")
                consecutive_failures += 1
                all_posts.extend(fetch_reddit_via_google_news(sub))
                continue
            
            # Success — reset consecutive failure counter
            consecutive_failures = 0
            
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
            consecutive_failures += 1
            all_posts.extend(fetch_reddit_via_google_news(sub))
    
    # Sort by upvote velocity (fastest-rising posts first)
    all_posts.sort(key=lambda x: x.get("_engagement", {}).get("upvote_velocity", 0), reverse=True)
    
    print(f"✅ Reddit: Found {len(all_posts)} trending AI posts.")
    return all_posts[:20]  # Top 20


# ─────────────────────────────────────────────────────────────────────────────
# 3. GITHUB TRENDING REPOS
# ─────────────────────────────────────────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0"
]

GITHUB_TRENDING_CACHE_FILE = os.path.join("logs", "github_trending_cache.json")

def scrape_github_trending(language=None, since="daily"):
    """
    Scrapes github.com/trending directly using BeautifulSoup.
    No API keys/tokens are required.
    """
    import random
    from bs4 import BeautifulSoup
    
    url = "https://github.com/trending"
    if language:
        url += f"/{language}"
    url += f"?since={since}"
    
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"  ⚠️ GitHub Scraper HTTP error: {resp.status_code}")
            return []
            
        soup = BeautifulSoup(resp.text, "html.parser")
        repos = []
        for article in soup.select("article.Box-row"):
            # Repo Slug / Name
            a_tag = article.select_one("h2 a")
            if not a_tag or not a_tag.get("href"):
                continue
            slug = a_tag["href"].strip("/")
            
            # Description
            desc_tag = article.select_one("p")
            desc = desc_tag.get_text(strip=True) if desc_tag else ""
            
            # Language
            lang_tag = article.select_one("[itemprop='programmingLanguage']")
            lang = lang_tag.get_text(strip=True) if lang_tag else "Unknown"
            
            # Total Stars
            stars_tag = article.select_one("a[href$='/stargazers']")
            stars = 0
            if stars_tag:
                try:
                    stars_str = stars_tag.get_text(strip=True).replace(",", "")
                    stars = int(stars_str)
                except:
                    pass
                    
            # Total Forks
            forks_tag = article.select_one("a[href$='/forks']")
            forks = 0
            if forks_tag:
                try:
                    forks_str = forks_tag.get_text(strip=True).replace(",", "")
                    forks = int(forks_str)
                except:
                    pass
            
            # Stars Period (daily/weekly/monthly)
            stars_period_tag = article.select_one("span.d-inline-block.float-sm-right") or article.select_one("span.float-sm-right")
            stars_period = 0
            if stars_period_tag:
                try:
                    text = stars_period_tag.get_text(strip=True)
                    digits = "".join([c for c in text if c.isdigit()])
                    if digits:
                        stars_period = int(digits)
                except:
                    pass
                    
            repos.append({
                "repo": slug,
                "full_name": slug,
                "description": desc,
                "language": lang,
                "stargazers_count": stars,
                "forks_count": forks,
                "stars_in_period": stars_period,
                "url": f"https://github.com/{slug}",
                "html_url": f"https://github.com/{slug}"
            })
        return repos
    except Exception as e:
        print(f"  ⚠️ GitHub Scraper Exception: {e}")
        return []

def load_cached_github_trending():
    """Loads cached GitHub trending results, warning if they are >48 hours stale."""
    if not os.path.exists(GITHUB_TRENDING_CACHE_FILE):
        return []
    try:
        mtime = os.path.getmtime(GITHUB_TRENDING_CACHE_FILE)
        age_hours = (time.time() - mtime) / 3600.0
        if age_hours > 48:
            print(f"\n⚠️ CRITICAL: GitHub Trending cache is stale (>48 hours)! Age: {age_hours:.1f} hours.\n")
        else:
            print(f"📋 Loaded GitHub Trending cache (Age: {age_hours:.1f} hours).")
        with open(GITHUB_TRENDING_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load github trending cache: {e}")
        return []

def save_github_trending_cache(repos):
    """Saves the successfully scraped/fetched GitHub trending results to cache."""
    if not repos:
        return
    try:
        os.makedirs(os.path.dirname(GITHUB_TRENDING_CACHE_FILE), exist_ok=True)
        with open(GITHUB_TRENDING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(repos, f, indent=2)
        print("💾 GitHub Trending cache updated successfully.")
    except Exception as e:
        print(f"⚠️ Failed to save github trending cache: {e}")

def _parse_github_repos(repos, min_stars=50):
    """Shared parser: converts GitHub API/scraped repo objects into trending articles."""
    results = []
    for repo in repos:
        stars = repo.get("stargazers_count", 0) or repo.get("stars", 0)
        # Scraped repos might have lower total stars initially, but we want to count them if they are trending today
        if stars < min_stars and repo.get("stars_in_period", 0) < 10:
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
        
        # Normalize stars velocity according to period
        stars_in_period = repo.get("stars_in_period", 0)
        if stars_in_period > 0:
            stars_per_day = stars_in_period
        else:
            stars_per_day = stars / age_days
        
        desc = repo.get('description', '') or ''
        
        # Topical relevance boost
        title_lower = repo.get('full_name', '').lower()
        desc_lower = desc.lower()
        
        # Broad technical and developer topics list:
        keywords = [
            'llm', 'gpt', 'llama', 'agent', 'ai', 'transformer', 'stable-diffusion', 'deepseek', 
            'compiler', 'terminal', 'database', 'api', 'editor', 'linux', 'rust', 'python', 'go', 
            'security', 'hack', 'exploit', 'performance', 'git', 'open-source', 'productivity', 
            'machine-learning', 'dataset', 'nlp', 'vision', 'neural', 'weights', 'inference'
        ]
        
        relevance_score = 0
        if any(kw in title_lower or kw in desc_lower for kw in keywords):
            relevance_score = 30
        
        results.append({
            "title": f"GitHub Trending: {repo.get('full_name', '')} — {desc[:100]}",
            "description": f"⭐ {stars} stars ({stars_per_day:.0f}/day) | {repo.get('language', 'Unknown')} | {desc}",
            "source": {"name": f"GitHub ({repo.get('full_name', '')})"},
            "url": repo.get("html_url", ""),
            "urlToImage": repo.get("owner", {}).get("avatar_url", "") if isinstance(repo.get("owner"), dict) else "",
            "publishedAt": created_at or datetime.now(timezone.utc).isoformat(),
            "type": "github_trending",
            "_engagement": {
                "stars": stars,
                "stars_per_day": round(stars_per_day, 1),
                "forks": repo.get("forks_count", 0),
                "watchers": repo.get("watchers_count", 0) or repo.get("stargazers_count", 0)
            },
            "_relevance_score": relevance_score
        })
    return results

def fetch_github_trending_ai():
    """
    Fetches trending AI/ML/Developer repos from GitHub.
    Tries scraping first, falls back to Search API or local cache if blocked/empty.
    Deduplicates results and ranks by stars velocity.
    """
    import random
    print("🐙 Fetching trending repos from GitHub...")
    all_repos = []
    
    # ── Strategy 1: BeautifulSoup scraping of github.com/trending (Option 1) ──
    scraped_any = False
    for since_period in ["daily", "weekly"]:
        for lang in [None, "python", "typescript"]:
            try:
                scraped = scrape_github_trending(language=lang, since=since_period)
                if scraped:
                    all_repos.extend(scraped)
                    scraped_any = True
                time.sleep(random.uniform(0.5, 1.2))
            except Exception as e:
                print(f"  ⚠️ Error scraping language '{lang}' for since '{since_period}': {e}")
            
    if not scraped_any:
        print("⚠️ Scraping returned 0 results. Checking cache fallback...")
        cached = load_cached_github_trending()
        if cached:
            all_repos.extend(cached)
            
    # ── Strategy 2: GitHub Search API (Option 2) ──
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": random.choice(USER_AGENTS)
    }
    since_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    search_queries = [
        f"topic:machine-learning stars:>50 pushed:>{since_date}",
        f"topic:llm stars:>50 pushed:>{since_date}",
        f"topic:artificial-intelligence stars:>50 pushed:>{since_date}",
        f"(AI OR LLM OR GPT OR \"open source\") stars:>100 pushed:>{since_date}",
    ]
    
    api_repos = []
    for query in search_queries:
        try:
            url = "https://api.github.com/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": 10
            }
            token = os.getenv("GITHUB_TOKEN")
            if token:
                headers["Authorization"] = f"token {token}"
                
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                api_repos.extend(data.get("items", []))
            else:
                print(f"  ⚠️ GitHub API Error ({r.status_code}) for query: {query[:50]}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  ⚠️ GitHub Search API query failed: {e}")
            
    # Merge and deduplicate by lowercased slug (full_name)
    merged_repos = []
    seen_lower_slugs = set()
    
    # 1. Scraped / Cached
    for repo in all_repos:
        slug = repo.get("full_name", "") or repo.get("repo", "")
        if not slug:
            continue
        slug_lower = slug.lower()
        if slug_lower not in seen_lower_slugs:
            seen_lower_slugs.add(slug_lower)
            merged_repos.append(repo)
            
    # 2. Search API
    for repo in api_repos:
        slug = repo.get("full_name", "")
        if not slug:
            continue
        slug_lower = slug.lower()
        if slug_lower not in seen_lower_slugs:
            seen_lower_slugs.add(slug_lower)
            merged_repos.append({
                "repo": slug,
                "full_name": slug,
                "description": repo.get("description", "") or "",
                "language": repo.get("language", "Unknown") or "Unknown",
                "stargazers_count": repo.get("stargazers_count", 0),
                "forks_count": repo.get("forks_count", 0),
                "stars_in_period": 0,
                "url": repo.get("html_url", ""),
                "html_url": repo.get("html_url", ""),
                "created_at": repo.get("created_at", "")
            })
            
    if scraped_any:
        # Cache only the live scraped results (which are our raw source)
        save_github_trending_cache(all_repos)
        
    parsed = _parse_github_repos(merged_repos, min_stars=50)
    parsed.sort(key=lambda x: x["_engagement"]["stars_per_day"], reverse=True)
    print(f"✅ GitHub: Found {len(parsed)} trending repos.")
    return parsed

# ─────────────────────────────────────────────────────────────────────────────
# 4. PROGRAMMATIC GOOGLE TRENDS TECH MINER (Stream A)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_google_trending_tech(target_country="US"):
    """
    Fetches the active Google Trends RSS feed for the target region and filters
    terms against a whitelist of tech trigger words.
    """
    print(f"📈 Fetching daily trends from Google Trends RSS for geo={target_country}...")
    url = f"https://trends.google.com/trending/rss?geo={target_country}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0'}
    
    whitelist = {'ai', 'open-source', 'github', 'ios', 'android', 'nvidia', 'code', 'tool', 'software', 'chatgpt', 'dev', 'leak', 'hack'}
    tech_trends = []
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        items = root.findall('.//item')
        
        for item in items:
            title = item.find('title').text or ""
            desc = ""
            desc_elem = item.find('description')
            if desc_elem is not None:
                desc = desc_elem.text or ""
                
            approx_traffic = item.find('{https://trends.google.com/trends/trendingsearches/daily}approx_traffic')
            traffic_str = approx_traffic.text if approx_traffic is not None else "N/A"
            
            title_lower = title.lower()
            desc_lower = desc.lower()
            
            # Whitelist match
            is_tech = False
            for word in whitelist:
                if word in title_lower or word in desc_lower:
                    is_tech = True
                    break
                    
            if is_tech:
                tech_trends.append({
                    "title": f"Google Trend: {title}",
                    "description": f"Breakout Google search trend with traffic {traffic_str}. Context: {desc}",
                    "source": {"name": f"Google Trends ({target_country})"},
                    "url": f"https://trends.google.com/trends/explore?geo={target_country}&q={urllib.parse.quote(title)}",
                    "urlToImage": "",
                    "publishedAt": datetime.now(timezone.utc).isoformat(),
                    "type": "google_trends",
                    "_engagement": {
                        "traffic": traffic_str,
                        "query": title
                    }
                })
        
        print(f"✅ Google Trends ({target_country}): Found {len(tech_trends)} tech-related trending terms.")
    except Exception as e:
        print(f"⚠️ Google Trends RSS fetch failed: {e}")
        
    return tech_trends


# ─────────────────────────────────────────────────────────────────────────────
# 5. YOUTUBE OUTLIER HUNTER (Stream B)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_youtube_outlier_trends(target_country="US", outlier_threshold=3.0):
    """
    YouTube Data API (The Outlier Hunter).
    Queries search.list endpoint daily using broad tech keywords,
    then uses channels.list to pull subscriber counts and computes view-to-sub ratio.
    """
    if not YOUTUBE_DATA_API_KEY:
        print("⚠️ YouTube Data API key missing. Skipping YouTube Outlier Hunter.")
        return []

    print(f"📺 Running YouTube Outlier Hunter for region={target_country}...")
    keywords = ["new AI tool", "developer update", "github open source", "coding hack"]
    published_after = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    video_candidates = {}
    
    for kw in keywords:
        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": kw,
                "type": "video",
                "publishedAfter": published_after,
                "maxResults": 25,
                "relevanceLanguage": "en",
                "key": YOUTUBE_DATA_API_KEY,
                "regionCode": target_country
            }
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                print(f"  ⚠️ YouTube Outlier search failed for '{kw}' (region={target_country}): {r.text[:200]}")
                continue
                
            data = r.json()
            for item in data.get("items", []):
                vid = item.get("id", {}).get("videoId")
                if vid:
                    snippet = item.get("snippet", {})
                    video_candidates[vid] = {
                        "title": snippet.get("title", ""),
                        "description": snippet.get("description", ""),
                        "channelId": snippet.get("channelId", ""),
                        "channelTitle": snippet.get("channelTitle", ""),
                        "publishedAt": snippet.get("publishedAt", ""),
                        "videoId": vid
                    }
            time.sleep(0.5)
        except Exception as e:
            print(f"  ⚠️ YouTube Outlier search query '{kw}' failed: {e}")
            
    if not video_candidates:
        return []
        
    print(f"  🔍 Found {len(video_candidates)} candidate videos. Fetching stats and channel subscriber counts...")
    
    # Batch get statistics and tags for all candidate videos
    vid_list = list(video_candidates.keys())
    outliers = []
    
    for i in range(0, len(vid_list), 50):
        batch_vids = vid_list[i:i+50]
        try:
            url = "https://www.googleapis.com/youtube/v3/videos"
            params = {
                "part": "statistics,snippet",
                "id": ",".join(batch_vids),
                "key": YOUTUBE_DATA_API_KEY
            }
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                continue
            data = r.json()
            for item in data.get("items", []):
                vid = item.get("id")
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})
                views = int(stats.get("viewCount", 0))
                likes = int(stats.get("likeCount", 0))
                comments = int(stats.get("commentCount", 0))
                tags = snippet.get("tags", [])
                
                if vid in video_candidates:
                    video_candidates[vid].update({
                        "views": views,
                        "likes": likes,
                        "comments": comments,
                        "tags": tags
                    })
        except Exception as e:
            print(f"  ⚠️ Error fetching video stats: {e}")
            
    # Batch get channel subscriber counts
    channel_ids = list(set(v["channelId"] for v in video_candidates.values() if v.get("channelId")))
    channel_subs = {}
    
    for i in range(0, len(channel_ids), 50):
        batch_channels = channel_ids[i:i+50]
        try:
            url = "https://www.googleapis.com/youtube/v3/channels"
            params = {
                "part": "statistics",
                "id": ",".join(batch_channels),
                "key": YOUTUBE_DATA_API_KEY
            }
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                for item in data.get("items", []):
                    cid = item.get("id")
                    stats = item.get("statistics", {})
                    subs = int(stats.get("subscriberCount", 0))
                    channel_subs[cid] = subs
        except Exception as e:
            print(f"  ⚠️ Error fetching channel statistics: {e}")
            
    # Calculate outlier scores and filter
    for vid, v in video_candidates.items():
        if "views" not in v:
            continue
            
        cid = v["channelId"]
        subs = channel_subs.get(cid, 0)
        views = v["views"]
        
        outlier_score = views / max(subs, 1)
        
        if outlier_score > outlier_threshold:
            outliers.append({
                "title": v["title"],
                "description": f"Outlier Score: {outlier_score:.2f} (Views: {views} | Subscribers: {subs}). Description: {v['description']}",
                "source": {"name": f"YouTube Outlier ({v['channelTitle']})"},
                "url": f"https://youtube.com/watch?v={vid}",
                "urlToImage": "",
                "publishedAt": v["publishedAt"],
                "type": "youtube_outliers",
                "_engagement": {
                    "views": views,
                    "likes": v["likes"],
                    "comments": v["comments"],
                    "subscribers": subs,
                    "outlier_score": outlier_score,
                    "tags": v["tags"]
                }
            })
            
    outliers.sort(key=lambda x: x["_engagement"]["outlier_score"], reverse=True)
    print(f"✅ YouTube Outlier Hunter: Found {len(outliers)} viral outlier tech videos.")
    return outliers


# ─────────────────────────────────────────────────────────────────────────────
# 6. UNIFIED TRENDING AGGREGATOR
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
        
        # Like ratio bonus
        if like_ratio > 5: score += 15
        elif like_ratio > 3: score += 10
        
        # Comments
        comments = eng.get("comments", 0)
        if comments > 500: score += 15
        elif comments > 100: score += 10
        elif comments > 20: score += 5
    
    elif art_type == "reddit_trending":
        ups = eng.get("upvotes", 0)
        velocity = eng.get("upvote_velocity", 0)
        ratio = eng.get("upvote_ratio", 0.5)
        comments = eng.get("comments", 0)
        
        if velocity > 200: score += 35
        elif velocity > 100: score += 28
        elif velocity > 50: score += 20
        elif velocity > 20: score += 12
        elif velocity > 5: score += 6
        
        if ups > 5000: score += 20
        elif ups > 1000: score += 15
        elif ups > 500: score += 10
        elif ups > 100: score += 5
        
        if ratio > 0.95: score += 10
        elif ratio > 0.90: score += 7
        
        if comments > 200: score += 10
        elif comments > 50: score += 5
    
    elif art_type == "github_trending":
        stars_pd = eng.get("stars_per_day", 0)
        stars = eng.get("stars", 0)
        
        if stars_pd > 500: score += 35
        elif stars_pd > 100: score += 28
        elif stars_pd > 50: score += 20
        elif stars_pd > 10: score += 12
        
        if stars > 10000: score += 20
        elif stars > 5000: score += 15
        elif stars > 1000: score += 10

    elif art_type == "google_trends":
        # Google Trends represents massive search volume/velocity.
        score += 35
        # Add traffic bonus
        traffic = eng.get("traffic", "N/A").lower()
        if "m" in traffic:
            score += 25
        elif "k" in traffic:
            try:
                num = int(traffic.replace("k+", "").replace(",", "").strip())
                if num >= 100: score += 20
                elif num >= 50: score += 15
                else: score += 10
            except:
                score += 10

    elif art_type == "youtube_outliers":
        outlier = eng.get("outlier_score", 0.0)
        if outlier >= 20.0: score += 45
        elif outlier >= 10.0: score += 35
        elif outlier >= 5.0: score += 25
        else: score += 15
        
        views = eng.get("views", 0)
        if views >= 100000: score += 20
        elif views >= 50000: score += 15
        elif views >= 10000: score += 10
    
    elif art_type == "trending":
        score += 15
    
    else:
        score += 5
    
    niche_sources = ["reddit_trending", "github_trending", "youtube_outliers"]
    if art_type in niche_sources:
        score += TRENDING_NICHE_BIAS * 15
    
    # Penalize estimated engagement (e.g. Google News RSS fallback for Reddit)
    if eng.get("engagement_estimated"):
        score = int(score * 0.4)
    
    return min(100, score)


def fetch_all_trending_signals(target_country="US"):
    """
    Master aggregator: fetches from all trending sources and returns
    a unified, scored article list ready for the pipeline.
    """
    print(f"\n🔥 === TRENDING ENGINE: Fetching Multi-Platform Signals for region={target_country} === 🔥")
    
    all_articles = []
    
    # 1. YouTube Trending Shorts
    try:
        yt_articles = fetch_youtube_trending_shorts(target_country)
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

    # 4. Google Trends (Stream A)
    try:
        gt_articles = fetch_google_trending_tech(target_country)
        all_articles.extend(gt_articles)
    except Exception as e:
        print(f"⚠️ Google Trends fetch failed: {e}")

    # 5. YouTube Outlier Hunter (Stream B)
    try:
        yo_articles = fetch_youtube_outlier_trends(target_country)
        all_articles.extend(yo_articles)
    except Exception as e:
        print(f"⚠️ YouTube Outlier Hunter fetch failed: {e}")
    
    # 6. Compute unified engagement scores
    for art in all_articles:
        art["_engagement_score"] = compute_engagement_score(art)
    
    # Sort by engagement score
    all_articles.sort(key=lambda x: x.get("_engagement_score", 0), reverse=True)
    
    # Summary
    yt_count = sum(1 for a in all_articles if a.get("type") == "youtube_trending")
    reddit_count = sum(1 for a in all_articles if a.get("type") == "reddit_trending")
    reddit_estimated = sum(1 for a in all_articles if a.get("type") == "reddit_trending" and a.get("_engagement", {}).get("engagement_estimated"))
    github_count = sum(1 for a in all_articles if a.get("type") == "github_trending")
    gt_count = sum(1 for a in all_articles if a.get("type") == "google_trends")
    yo_count = sum(1 for a in all_articles if a.get("type") == "youtube_outliers")
    
    print(f"\n📊 Trending Engine Summary: {len(all_articles)} total signals")
    print(f"   YouTube Trending: {yt_count} | Reddit: {reddit_count} | GitHub: {github_count}")
    print(f"   Google Trends: {gt_count} | YouTube Outliers: {yo_count}")
    if all_articles:
        top = all_articles[0]
        print(f"   🏆 Top Signal: '{top['title'][:60]}...' (Score: {top.get('_engagement_score', 0)})")
    
    # ── Data Source Health Dashboard ──────────────────────────────────────
    yt_status = "✅ Active" if yt_count > 0 else "❌ Offline (YOUTUBE_DATA_API_KEY missing?)"
    reddit_native = reddit_count - reddit_estimated
    if reddit_native > 0:
        reddit_status = "✅ Active"
    elif reddit_estimated > 0:
        reddit_status = f"⚠️ Degraded (fallback only, {reddit_estimated} estimated)"
    else:
        reddit_status = "❌ Offline (REDDIT_CLIENT_ID/SECRET missing?)"
    github_status = "✅ Active" if github_count > 0 else "⚠️ No results"
    gt_status = "✅ Active" if gt_count > 0 else "⚠️ No results"
    yo_status = "✅ Active" if yo_count > 0 else "⚠️ No results"
    
    active_count = sum(1 for s in [yt_status, reddit_status, github_status, gt_status, yo_status] if s.startswith("✅"))
    print(f"\n🏥 Data Source Health: {active_count}/5 sources fully active")
    print(f"   YouTube Trending : {yt_status}")
    print(f"   Reddit           : {reddit_status}")
    print(f"   GitHub            : {github_status}")
    print(f"   Google Trends     : {gt_status}")
    print(f"   YouTube Outliers  : {yo_status}")
    
    return all_articles
