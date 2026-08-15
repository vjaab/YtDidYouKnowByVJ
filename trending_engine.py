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
# CATEGORY → SIGNAL MAPPINGS
# Maps each WEEKLY_SCHEDULE category to source-specific keywords, subreddits, queries
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_SIGNALS = {
    "AI & Tech Tools": {
        "youtube_queries": [
            "AI tools free productivity 2026",
            "best free AI tools for students",
            "ChatGPT alternatives free",
            "AI productivity hacks",
            "local LLM tools Ollama LM Studio",
        ],
        "reddit_subs": ["LocalLLaMA", "MachineLearning", "OpenAI", "ClaudeAI", "ChatGPT", "AI_Agents"],
        "github_topics": ["llm", "ai-agent", "mcp", "rag", "llm-inference", "evals", "local-llm", "ollama"],
        "github_languages": ["python", "typescript", "rust"],
        "hn_keywords": ['llm', 'gpt', 'llama', 'claude', 'deepseek', 'agent', 'mcp', 'rag', 'ollama', 'vllm', 'inference', 'local'],
        "hf_tasks": ["text-generation", "conversational", "text2text-generation", "summarization"],
        "arxiv_cats": ["cs.AI", "cs.CL", "cs.LG"],
        "google_trends_whitelist": {'ai', 'tool', 'chatgpt', 'llm', 'ollama', 'local', 'free', 'productivity'},
        "youtube_outlier_keywords": ["new AI tool", "free AI", "local LLM", "AI productivity"],
    },
    "Tech Gadgets & Inventions": {
        "youtube_queries": [
            "new tech gadgets 2026",
            "smart glasses AR VR review",
            "wearable tech review",
            "smart home devices 2026",
            "cool gadgets you didn't know existed",
        ],
        "reddit_subs": ["gadgets", "hardware", "technology", "smartHome", "wearables", "VR"],
        "github_topics": ["iot", "home-assistant", "esp32", "raspberry-pi", "arduino", "embedded", "robotics"],
        "github_languages": ["cpp", "rust", "python", "c"],
        "hn_keywords": ['gadget', 'hardware', 'wearable', 'ar', 'vr', 'smart-glasses', 'iot', 'raspberry-pi', 'esp32', 'robotics'],
        "hf_tasks": ["computer-vision", "audio-classification", "object-detection"],
        "arxiv_cats": ["cs.HC", "cs.RO", "cs.CV", "eess.SP"],
        "google_trends_whitelist": {'gadget', 'smart', 'glasses', 'wearable', 'ar', 'vr', 'home', 'robot'},
        "youtube_outlier_keywords": ["new gadget", "smart glasses", "wearable", "robot", "smart home"],
    },
    "Finance & Tech Economy": {
        "youtube_queries": [
            "AI finance tools budgeting",
            "fintech apps save money",
            "crypto trading AI bot",
            "stock analysis AI tools",
            "personal finance automation",
        ],
        "reddit_subs": ["personalfinance", "financialindependence", "CryptoCurrency", "stocks", "fintech", "algotrading"],
        "github_topics": ["fintech", "trading-bot", "quantitative-finance", "portfolio-optimization", "crypto-trading"],
        "github_languages": ["python", "rust", "typescript"],
        "hn_keywords": ['fintech', 'crypto', 'trading', 'finance', 'portfolio', 'quantitative', 'yield', 'defi'],
        "hf_tasks": ["tabular-classification", "time-series-forecasting"],
        "arxiv_cats": ["q-fin", "econ.GN", "cs.CY"],
        "google_trends_whitelist": {'finance', 'crypto', 'trading', 'stock', 'money', 'invest', 'fintech', 'budget'},
        "youtube_outlier_keywords": ["finance AI", "crypto bot", "trading AI", "budget app", "fintech"],
    },
    "Facts & Trivia": {
        "youtube_queries": [
            "tech facts you didn't know",
            "computer history trivia",
            "AI facts mind blown",
            "tech myths debunked",
            "weird tech history",
        ],
        "reddit_subs": ["todayilearned", "technology", "computerscience", "programminghorror", "retrobattlestations"],
        "github_topics": ["awesome", "computer-history", "tech-trivia", "fun-facts"],
        "github_languages": None,
        "hn_keywords": ['history', 'trivia', 'fact', 'origin', 'first', 'invented', 'moth', 'grace-hopper', 'backrub'],
        "hf_tasks": [],
        "arxiv_cats": ["cs.HC", "physics.hist-ph"],
        "google_trends_whitelist": {'fact', 'trivia', 'history', 'myth', 'did-you-know', 'origin', 'invented'},
        "youtube_outlier_keywords": ["tech fact", "history", "trivia", "myth busted", "did you know"],
    },
    "Coding & Development Hacks": {
        "youtube_queries": [
            "coding productivity tips 2026",
            "VS Code extensions must have",
            "Python tricks you didn't know",
            "GitHub Copilot tips",
            "developer workflow automation",
        ],
        "reddit_subs": ["programming", "learnprogramming", "cscareerquestions", "webdev", "rust", "python", "golang"],
        "github_topics": ["developer-tools", "productivity", "vscode-extension", "cli", "automation", "code-generation", "refactoring"],
        "github_languages": ["python", "typescript", "rust", "go", "javascript"],
        "hn_keywords": ['coding', 'programming', 'developer', 'vscode', 'github', 'copilot', 'refactor', 'debug', 'cli', 'terminal'],
        "hf_tasks": ["code-generation", "fill-mask"],
        "arxiv_cats": ["cs.SE", "cs.PL", "cs.MS"],
        "google_trends_whitelist": {'code', 'coding', 'programming', 'developer', 'vscode', 'github', 'python', 'rust', 'trick', 'hack'},
        "youtube_outlier_keywords": ["coding hack", "developer tip", "VS Code", "GitHub Copilot", "Python trick"],
    },
    "Quiz & Trivia": {
        "youtube_queries": [
            "tech quiz questions",
            "programming trivia challenge",
            "computer science quiz",
            "AI history quiz",
            "guess the tech company",
        ],
        "reddit_subs": ["trivia", "quiz", "computerscience", "programminghorror", "todayilearned"],
        "github_topics": ["quiz", "trivia", "interview-prep", "coding-challenges"],
        "github_languages": None,
        "hn_keywords": ['quiz', 'trivia', 'interview', 'question', 'challenge', 'puzzle'],
        "hf_tasks": [],
        "arxiv_cats": None,
        "google_trends_whitelist": {'quiz', 'trivia', 'question', 'challenge', 'test', 'guess'},
        "youtube_outlier_keywords": ["tech quiz", "trivia", "interview question", "can you guess"],
    },
    "Interview Questions": {
        "youtube_queries": [
            "software engineering interview questions 2026",
            "system design interview",
            "Java interview questions",
            "Python interview questions",
            "Kubernetes Docker interview",
        ],
        "reddit_subs": ["cscareerquestions", "programming", "leetcode", "systemdesign", "java", "python", "kubernetes"],
        "github_topics": ["interview-prep", "system-design", "leetcode", "cracking-the-coding-interview", "algorithms"],
        "github_languages": ["java", "python", "javascript", "go", "cpp"],
        "hn_keywords": ['interview', 'leetcode', 'system-design', 'algorithm', 'data-structure', 'coding-interview', 'spring-boot', 'kubernetes', 'docker'],
        "hf_tasks": [],
        "arxiv_cats": None,
        "google_trends_whitelist": {'interview', 'leetcode', 'system-design', 'algorithm', 'spring', 'kubernetes', 'docker', 'java', 'python'},
        "youtube_outlier_keywords": ["interview question", "system design", "LeetCode", "coding interview", "Spring Boot", "Kubernetes"],
    },
    "Programming Language Origins": {
        "youtube_queries": [
            "programming language history",
            "why was Python created",
            "history of JavaScript",
            "Rust language origin",
            "programming language design",
        ],
        "reddit_subs": ["programming", "computerscience", "rust", "python", "golang", "javascript", "java"],
        "github_topics": ["language-design", "compiler", "programming-language", "history"],
        "github_languages": ["rust", "go", "python", "javascript", "java", "cpp"],
        "hn_keywords": ['language', 'origin', 'history', 'design', 'creator', 'guido', 'brendan-eich', 'graydon-hoare'],
        "hf_tasks": [],
        "arxiv_cats": ["cs.PL"],
        "google_trends_whitelist": {'language', 'origin', 'history', 'python', 'javascript', 'rust', 'java', 'go', 'creator'},
        "youtube_outlier_keywords": ["language origin", "history of", "creator of", "why was", "designed"],
    },
    "Tech Company Founding Stories": {
        "youtube_queries": [
            "how Google started",
            "Apple founding story",
            "Microsoft early days",
            "startup founder story",
            "tech company history",
        ],
        "reddit_subs": ["technology", "startups", "entrepreneur", "business", "YCombinator"],
        "github_topics": ["startup", "business", "entrepreneurship", "founder"],
        "github_languages": None,
        "hn_keywords": ['founder', 'startup', 'ycombinator', 'google', 'apple', 'microsoft', 'amazon', 'meta', 'nvidia', 'history'],
        "hf_tasks": [],
        "arxiv_cats": ["cs.CY", "econ.GN"],
        "google_trends_whitelist": {'founder', 'startup', 'google', 'apple', 'microsoft', 'amazon', 'nvidia', 'history', 'billion'},
        "youtube_outlier_keywords": ["founder story", "how started", "early days", "Y Combinator", "billion dollar"],
    },
    "Famous Bugs & Glitches": {
        "youtube_queries": [
            "famous software bugs",
            "worst programming mistakes",
            "Mars Climate Orbiter bug",
            "Ariane 5 explosion software",
            "biggest tech failures",
        ],
        "reddit_subs": ["programminghorror", "computerscience", "softwareengineering", "devops", "sysadmin"],
        "github_topics": ["bug", "postmortem", "incident", "failure", "debugging", "outage"],
        "github_languages": None,
        "hn_keywords": ['bug', 'glitch', 'failure', 'outage', 'postmortem', 'incident', 'mars', 'ariane', 'therac', 'overflow'],
        "hf_tasks": [],
        "arxiv_cats": ["cs.SE", "cs.CR"],
        "google_trends_whitelist": {'bug', 'glitch', 'failure', 'outage', 'crash', 'error', 'mistake', 'disaster'},
        "youtube_outlier_keywords": ["famous bug", "software disaster", "worst bug", "Mars Orbiter", "Ariane 5"],
    },
    "Agentic AI Facts": {
        "youtube_queries": [
            "AI agents explained",
            "autonomous AI agents 2026",
            "AutoGen CrewAI LangGraph",
            "AI agent workflow",
            "multi-agent systems",
        ],
        "reddit_subs": ["AI_Agents", "MachineLearning", "LocalLLaMA", "OpenAI", "LangChain"],
        "github_topics": ["agent", "autogen", "crewai", "langgraph", "multi-agent", "agentic", "autogpt", "babyagi"],
        "github_languages": ["python", "typescript"],
        "hn_keywords": ['agent', 'autogen', 'crewai', 'langgraph', 'autogpt', 'babyagi', 'metaGPT', 'swarm', 'reflexion'],
        "hf_tasks": ["text-generation", "conversational"],
        "arxiv_cats": ["cs.AI", "cs.MA", "cs.LG"],
        "google_trends_whitelist": {'agent', 'autogen', 'crewai', 'langgraph', 'autonomous', 'multi-agent', 'reflexion'},
        "youtube_outlier_keywords": ["AI agent", "AutoGen", "CrewAI", "LangGraph", "autonomous agent", "multi-agent"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. YOUTUBE TRENDING SHORTS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_youtube_trending_shorts(target_country="US", category="AI & Tech Tools"):
    """
    Uses YouTube Data API v3 to find recently uploaded Shorts
    with high view counts for the given category.
    """
    if not YOUTUBE_DATA_API_KEY:
        print("⚠️ YouTube Data API key missing. Skipping YouTube trending fetch.")
        return []

    signals = CATEGORY_SIGNALS.get(category, CATEGORY_SIGNALS["AI & Tech Tools"])
    search_queries = signals.get("youtube_queries", [])
    
    cat_display = category
    print(f"📺 Fetching trending Shorts from YouTube Data API for category='{cat_display}', region={target_country}...")
    
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


def fetch_reddit_via_google_news(sub, category="AI & Tech Tools"):
    """
    Fallback: Query Google News RSS for hot posts in a specific subreddit
    since GHA IPs are blocked by Reddit directly.
    Filters by category-specific keywords.
    Engagement values are estimated baselines (not real), flagged accordingly.
    """
    signals = CATEGORY_SIGNALS.get(category, CATEGORY_SIGNALS["AI & Tech Tools"])
    filter_keywords = signals.get("hn_keywords", [])
    
    import urllib.parse
    import xml.etree.ElementTree as ET
    
    print(f"  🔍 Reddit Fallback: Querying Google News RSS for r/{sub} (category='{category}')...")
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
            
            # Filter by category keywords
            title_lower = title_clean.lower()
            if filter_keywords:
                if not any(kw in title_lower for kw in filter_keywords):
                    continue
            
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


def fetch_reddit_hot_ai(category="AI & Tech Tools"):
    """
    Fetches hot posts from AI subreddits. Uses OAuth if credentials available,
    falls back to old.reddit.com public JSON API otherwise.
    Filters by category-specific subreddits.
    """
    signals = CATEGORY_SIGNALS.get(category, CATEGORY_SIGNALS["AI & Tech Tools"])
    subreddits = signals.get("reddit_subs", ["MachineLearning", "LocalLLaMA", "AI_Agents", "OpenAI", "ClaudeAI", "singularity", "ChatGPT", "technology", "privacy", "gadgets"])
    
    print(f"🔴 Fetching hot posts from Reddit for category='{category}'...")
    
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
                all_posts.extend(fetch_reddit_via_google_news(fallback_sub, category))
            break
        
        try:
            url = f"{base_url}/r/{sub}/hot.json?limit=10&raw_json=1"
            
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200:
                print(f"  ⚠️ Reddit r/{sub} failed ({r.status_code})")
                consecutive_failures += 1
                all_posts.extend(fetch_reddit_via_google_news(sub, category))
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

def _parse_github_repos(repos, min_stars=50, keywords=None):
    """Shared parser: converts GitHub API/scraped repo objects into trending articles."""
    if keywords is None:
        keywords = [
            'llm', 'gpt', 'llama', 'agent', 'ai', 'transformer', 'stable-diffusion', 'deepseek', 
            'compiler', 'terminal', 'database', 'api', 'editor', 'linux', 'rust', 'python', 'go', 
            'security', 'hack', 'exploit', 'performance', 'git', 'open-source', 'productivity', 
            'machine-learning', 'dataset', 'nlp', 'vision', 'neural', 'weights', 'inference'
        ]
    
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

def fetch_github_trending_ai(category="AI & Tech Tools"):
    """
    Fetches trending AI/ML/Developer repos from GitHub.
    Filters by category-specific topics and languages.
    Tries scraping first, falls back to Search API or local cache if blocked/empty.
    Deduplicates results and ranks by stars velocity.
    """
    import random
    signals = CATEGORY_SIGNALS.get(category, CATEGORY_SIGNALS["AI & Tech Tools"])
    github_topics = signals.get("github_topics", ['llm', 'gpt', 'llama', 'agent', 'ai', 'transformer', 'stable-diffusion', 'deepseek', 
        'compiler', 'terminal', 'database', 'api', 'editor', 'linux', 'rust', 'python', 'go', 
        'security', 'hack', 'exploit', 'performance', 'git', 'open-source', 'productivity', 
        'machine-learning', 'dataset', 'nlp', 'vision', 'neural', 'weights', 'inference'])
    github_languages = signals.get("github_languages", [None, "python", "typescript", "rust", "cpp"])
    keywords = signals.get("hn_keywords", ['llm', 'gpt', 'llama', 'claude', 'deepseek', 'agent', 'mcp', 'transformer', 'quantization', 'gpu', 'inference', 'vllm', 'compiler', 'rust', 'python', 'c++', 'benchmark', 'model', 'dataset', 'arxiv'])
    
    print(f"🐙 Fetching trending repos from GitHub for category='{category}'...")
    all_repos = []
    
    # ── Strategy 1: BeautifulSoup scraping of github.com/trending (Option 1) ──
    scraped_any = False
    for since_period in ["daily", "weekly"]:
        for lang in github_languages:
            try:
                scraped = scrape_github_trending(language=lang, since=since_period)
                if scraped:
                    all_repos.extend(scraped)
                    scraped_any = True
                time.sleep(random.uniform(0.5, 1.2))
            except Exception as e:
                print(f"  ⚠️ Error scraping language '{lang}' for since '{since_period}': {e}")
            
    if not scraped_any or len(all_repos) < 10:
        print(f"⚠️ Scraping returned too few results ({len(all_repos)} repos). Checking cache fallback...")
        cached = load_cached_github_trending()
        if cached:
            all_repos = list(cached)
            scraped_any = False
            
    # ── Strategy 2: GitHub Search API (Option 2) ──
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": random.choice(USER_AGENTS)
    }
    since_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    search_queries = [
        f"topic:{topic} stars:>10 pushed:>{since_date}" for topic in github_topics[:8]
    ]
    # Add broader queries
    search_queries.extend([
        f"(AI OR LLM OR GPT OR \"open source\") stars:>100 pushed:>{since_date}",
        f"awesome-{github_topics[0]}-servers OR awesome-ai-tools OR awesome-agentic-ai stars:>20 pushed:>{since_date}",
    ])
    
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
        
    parsed = _parse_github_repos(merged_repos, min_stars=50, keywords=keywords)
    parsed.sort(key=lambda x: x["_engagement"]["stars_per_day"], reverse=True)
    print(f"✅ GitHub: Found {len(parsed)} trending repos for category='{category}'.")
    return parsed

# ─────────────────────────────────────────────────────────────────────────────
# 4. PROGRAMMATIC GOOGLE TRENDS TECH MINER (Stream A)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_google_trending_tech(target_country="US", category="AI & Tech Tools"):
    """
    Fetches the active Google Trends RSS feed for the target region and filters
    terms against a whitelist of tech trigger words.
    Filters by category-specific whitelist.
    """
    signals = CATEGORY_SIGNALS.get(category, CATEGORY_SIGNALS["AI & Tech Tools"])
    whitelist = signals.get("google_trends_whitelist", {'ai', 'open-source', 'github', 'ios', 'android', 'nvidia', 'code', 'tool', 'software', 'chatgpt', 'dev', 'leak', 'hack'})
    
    print(f"📈 Fetching daily trends from Google Trends RSS for geo={target_country}, category='{category}'...")
    url = f"https://trends.google.com/trending/rss?geo={target_country}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0'}
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
def fetch_youtube_outlier_trends(target_country="US", category="AI & Tech Tools", outlier_threshold=3.0):
    """
    YouTube Data API (The Outlier Hunter).
    Queries search.list endpoint daily using category-specific keywords,
    then uses channels.list to pull subscriber counts and computes view-to-sub ratio.
    """
    if not YOUTUBE_DATA_API_KEY:
        print("⚠️ YouTube Data API key missing. Skipping YouTube Outlier Hunter.")
        return []

    signals = CATEGORY_SIGNALS.get(category, CATEGORY_SIGNALS["AI & Tech Tools"])
    keywords = signals.get("youtube_outlier_keywords", ["new AI tool", "developer update", "github open source", "coding hack"])
    
    print(f"📺 Running YouTube Outlier Hunter for region={target_country}, category='{category}'...")
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
# 6. HACKER NEWS TRENDING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def fetch_hacker_news_trending(category="AI & Tech Tools"):
    """
    Fetches top tech/AI discussions from Hacker News.
    Uses Official HN Firebase REST API and RSS fallback.
    Filters by category-specific keywords.
    """
    signals = CATEGORY_SIGNALS.get(category, CATEGORY_SIGNALS["AI & Tech Tools"])
    ai_keywords = signals.get("hn_keywords", ['ai', 'llm', 'gpt', 'llama', 'claude', 'deepseek', 'agent', 'mcp',
        'transformer', 'quantization', 'gpu', 'inference', 'vllm', 'compiler',
        'rust', 'python', 'c++', 'benchmark', 'model', 'dataset', 'arxiv'])
    
    print(f"🧡 Fetching top discussions from Hacker News for category='{category}'...")
    articles = []
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    
    try:
        top_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        r = requests.get(top_url, headers=headers, timeout=10)
        if r.status_code == 200:
            item_ids = r.json()[:35]
            for item_id in item_ids:
                try:
                    item_url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
                    ir = requests.get(item_url, headers=headers, timeout=5)
                    if ir.status_code == 200:
                        data = ir.json() or {}
                        title = data.get("title", "")
                        url = data.get("url") or f"https://news.ycombinator.com/item?id={item_id}"
                        score = data.get("score", 0)
                        comments = data.get("descendants", 0)
                        
                        title_lower = title.lower()
                        if any(kw in title_lower for kw in ai_keywords):
                            articles.append({
                                "title": f"Hacker News: {title}",
                                "description": f"🧡 {score} points | 💬 {comments} comments | HN URL: https://news.ycombinator.com/item?id={item_id}",
                                "source": {"name": "Hacker News"},
                                "url": url,
                                "urlToImage": "",
                                "publishedAt": datetime.fromtimestamp(data.get("time", time.time()), tz=timezone.utc).isoformat(),
                                "type": "hacker_news",
                                "_engagement": {
                                    "points": score,
                                    "comments": comments,
                                    "hn_id": item_id
                                }
                            })
                except Exception:
                    pass
                time.sleep(0.05)
    except Exception as e:
        print(f"  ⚠️ HN Firebase API error: {e}")
        
    # RSS Fallback if API returned < 3 articles
    if len(articles) < 3:
        try:
            feed_url = "https://news.ycombinator.com/rss"
            req = urllib.request.Request(feed_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            for item in root.findall('.//item')[:20]:
                title = item.find('title').text or ""
                link = item.find('link').text or ""
                if any(kw in title.lower() for kw in ai_keywords):
                    articles.append({
                        "title": f"Hacker News RSS: {title}",
                        "description": f"Top HN Discussion: {title}",
                        "source": {"name": "Hacker News (RSS)"},
                        "url": link,
                        "urlToImage": "",
                        "publishedAt": datetime.now(timezone.utc).isoformat(),
                        "type": "hacker_news",
                        "_engagement": {"points": 50, "comments": 15}
                    })
        except Exception as ex:
            print(f"  ⚠️ HN RSS Fallback error: {ex}")
            
    print(f"✅ Hacker News: Found {len(articles)} AI/engineering discussions.")
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# 7. HUGGING FACE TRENDING PAPERS & MODELS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def fetch_huggingface_trending(category="AI & Tech Tools"):
    """
    Fetches trending daily papers & open-weight model releases from Hugging Face.
    Uses Hugging Face Daily Papers API (huggingface.co/api/daily_papers).
    Filters by category-relevant tasks.
    """
    signals = CATEGORY_SIGNALS.get(category, CATEGORY_SIGNALS["AI & Tech Tools"])
    hf_tasks = signals.get("hf_tasks", ["text-generation", "conversational", "text2text-generation", "summarization"])
    
    print(f"🤗 Fetching daily trending papers & models from Hugging Face for category='{category}'...")
    articles = []
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    
    try:
        url = "https://huggingface.co/api/daily_papers"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for item in data[:15]:
                paper = item.get("paper", {})
                title = paper.get("title", "")
                summary = paper.get("summary", "") or ""
                paper_id = paper.get("id", "")
                num_upvotes = item.get("publishedAt", "")
                num_upvotes = item.get("upvotes", 0) or paper.get("upvotes", 0) or 10
                
                # Filter by task relevance if task info available
                paper_tags = paper.get("tags", [])
                if hf_tasks and paper_tags:
                    if not any(task in paper_tags for task in hf_tasks):
                        continue
                
                paper_url = f"https://huggingface.co/papers/{paper_id}" if paper_id else "https://huggingface.co/papers"
                
                articles.append({
                    "title": f"Hugging Face Paper: {title}",
                    "description": f"🤗 HF Trending Paper ({num_upvotes} upvotes) | {summary[:280]}",
                    "source": {"name": "Hugging Face Daily Papers"},
                    "url": paper_url,
                    "urlToImage": f"https://arxiv.org/static/browse/0.3.4/images/arxiv-logo-fb.png",
                    "publishedAt": paper.get("publishedAt") or datetime.now(timezone.utc).isoformat(),
                    "type": "huggingface_trending",
                    "_engagement": {
                        "upvotes": num_upvotes,
                        "paper_id": paper_id
                    }
                })
    except Exception as e:
        print(f"  ⚠️ Hugging Face API error: {e}")
        
    print(f"✅ Hugging Face: Found {len(articles)} daily papers/models for category='{category}'.")
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# 8. ARXIV AI RESEARCH PAPERS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def fetch_arxiv_ai_papers(category="AI & Tech Tools"):
    """
    Fetches latest AI research papers directly from ArXiv API
    filtered by category-specific categories.
    """
    signals = CATEGORY_SIGNALS.get(category, CATEGORY_SIGNALS["AI & Tech Tools"])
    arxiv_cats = signals.get("arxiv_cats", ["cs.AI", "cs.CL", "cs.LG"])
    
    if not arxiv_cats:
        print(f"⚠️ No ArXiv categories defined for '{category}', skipping.")
        return []
    
    cat_query = "+OR+".join([f"cat:{c}" for c in arxiv_cats])
    print(f"📄 Fetching latest research papers from ArXiv ({', '.join(arxiv_cats)}) for category='{category}'...")
    articles = []
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    
    try:
        url = f"http://export.arxiv.org/api/query?search_query={cat_query}&sortBy=submittedDate&sortOrder=descending&max_results=12"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=12) as resp:
            xml_data = resp.read()
            
        root = ET.fromstring(xml_data)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', namespace)
        
        for entry in entries:
            title = entry.find('atom:title', namespace).text or ""
            title_clean = " ".join(title.split())
            summary = entry.find('atom:summary', namespace).text or ""
            summary_clean = " ".join(summary.split())
            link = entry.find('atom:id', namespace).text or ""
            published = entry.find('atom:published', namespace).text or ""
            
            authors = [a.find('atom:name', namespace).text for a in entry.findall('atom:author', namespace) if a.find('atom:name', namespace) is not None]
            authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            
            articles.append({
                "title": f"ArXiv Research: {title_clean}",
                "description": f"Authors: {authors_str} | Abstract: {summary_clean[:300]}...",
                "source": {"name": f"ArXiv ({', '.join(arxiv_cats)})"},
                "url": link,
                "urlToImage": "",
                "publishedAt": published,
                "type": "arxiv_papers",
                "_engagement": {
                    "citations_estimated": 10,
                    "relevance_boost": 25
                }
            })
    except Exception as e:
        print(f"  ⚠️ ArXiv API error: {e}")
        
    print(f"✅ ArXiv: Found {len(articles)} research papers for category='{category}'.")
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# 9. TLDR AI & TECH NEWSLETTERS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def fetch_tldr_ai_newsletters(category="AI & Tech Tools"):
    """
    Fetches summarized open-source AI announcements from TLDR AI and curated newsletter RSS.
    Filters by category-relevant keywords.
    """
    signals = CATEGORY_SIGNALS.get(category, CATEGORY_SIGNALS["AI & Tech Tools"])
    filter_keywords = signals.get("hn_keywords", [])  # Reuse HN keywords as filter
    
    print(f"📰 Fetching daily announcements from TLDR AI & Newsletters for category='{category}'...")
    articles = []
    feeds = [
        ("TLDR Tech", "https://tldr.tech/tech/rss"),
        ("Import AI", "https://importai.substack.com/feed"),
        ("ByteByteGo", "https://blog.bytebytego.com/feed")
    ]
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    for name, feed_url in feeds:
        try:
            req = urllib.request.Request(feed_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            items = root.findall('.//item')
            for item in items[:4]:
                title = item.find('title').text or ""
                link = item.find('link').text or ""
                desc_elem = item.find('description')
                desc = desc_elem.text if desc_elem is not None else ""
                clean_desc = re.sub('<[^<]+?>', '', desc)[:300].strip()
                
                # Filter by category keywords
                title_lower = title.lower()
                desc_lower = clean_desc.lower()
                if filter_keywords:
                    if not any(kw in title_lower or kw in desc_lower for kw in filter_keywords):
                        continue
                
                articles.append({
                    "title": f"{name}: {title}",
                    "description": clean_desc if clean_desc else title,
                    "source": {"name": name},
                    "url": link,
                    "urlToImage": "",
                    "publishedAt": datetime.now(timezone.utc).isoformat(),
                    "type": "newsletter_ai",
                    "_engagement": {"curated_score": 30}
                })
        except Exception as e:
            print(f"  ⚠️ Newsletter RSS fetch error ({name}): {e}")
            
    print(f"✅ TLDR & Newsletters: Found {len(articles)} curated announcements for category='{category}'.")
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# 10. UNIFIED TRENDING AGGREGATOR
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

    elif art_type == "hacker_news":
        pts = eng.get("points", 0)
        comments = eng.get("comments", 0)
        if pts >= 300: score += 40
        elif pts >= 150: score += 30
        elif pts >= 50: score += 20
        else: score += 10
        
        if comments >= 100: score += 15
        elif comments >= 30: score += 10

    elif art_type == "huggingface_trending":
        upvotes = eng.get("upvotes", 0)
        score += 35
        if upvotes >= 50: score += 15
        elif upvotes >= 20: score += 10

    elif art_type == "arxiv_papers":
        score += 30
        title_desc = (article.get("title", "") + " " + article.get("description", "")).lower()
        if any(kw in title_desc for kw in ["quantization", "vram", "memory", "inference", "agent", "benchmark", "cost", "speed"]):
            score += 15

    elif art_type == "newsletter_ai":
        score += 25

    elif art_type == "google_trends":
        score += 35
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
    
    niche_sources = ["reddit_trending", "github_trending", "youtube_outliers", "hacker_news", "huggingface_trending", "arxiv_papers"]
    if art_type in niche_sources:
        score += TRENDING_NICHE_BIAS * 15
    
    if eng.get("engagement_estimated"):
        score = int(score * 0.4)
    
    return min(100, score)


def fetch_all_trending_signals(target_country="US", category="AI & Tech Tools"):
    """
    Master aggregator: fetches from all trending sources and returns
    a unified, scored article list ready for the pipeline.
    Filters all sources by the given category.
    """
    print(f"\n🔥 === TRENDING ENGINE: Fetching Multi-Platform Signals for region={target_country}, category='{category}' === 🔥")
    
    all_articles = []
    
    # 1. YouTube Trending Shorts
    try:
        yt_articles = fetch_youtube_trending_shorts(target_country, category)
        all_articles.extend(yt_articles)
    except Exception as e:
        print(f"⚠️ YouTube trending failed: {e}")
    
    # 2. Reddit Hot Posts
    try:
        reddit_articles = fetch_reddit_hot_ai(category)
        all_articles.extend(reddit_articles)
    except Exception as e:
        print(f"⚠️ Reddit trending failed: {e}")
    
    # 3. GitHub Trending Repos & Topics
    try:
        github_articles = fetch_github_trending_ai(category)
        all_articles.extend(github_articles)
    except Exception as e:
        print(f"⚠️ GitHub trending failed: {e}")
    
    # 4. Hacker News Top Discussions
    try:
        hn_articles = fetch_hacker_news_trending(category)
        all_articles.extend(hn_articles)
    except Exception as e:
        print(f"⚠️ Hacker News fetch failed: {e}")
    
    # 5. Hugging Face Trending Papers & Models
    try:
        hf_articles = fetch_huggingface_trending(category)
        all_articles.extend(hf_articles)
    except Exception as e:
        print(f"⚠️ Hugging Face fetch failed: {e}")
    
    # 6. ArXiv AI Research Papers
    try:
        arxiv_articles = fetch_arxiv_ai_papers(category)
        all_articles.extend(arxiv_articles)
    except Exception as e:
        print(f"⚠️ ArXiv AI fetch failed: {e}")
    
    # 7. TLDR AI & Newsletters
    try:
        tldr_articles = fetch_tldr_ai_newsletters(category)
        all_articles.extend(tldr_articles)
    except Exception as e:
        print(f"⚠️ TLDR AI fetch failed: {e}")
    
    # 8. Google Trends (Stream A)
    try:
        gt_articles = fetch_google_trending_tech(target_country, category)
        all_articles.extend(gt_articles)
    except Exception as e:
        print(f"⚠️ Google Trends fetch failed: {e}")
    
    # 9. YouTube Outlier Hunter (Stream B)
    try:
        yo_articles = fetch_youtube_outlier_trends(target_country, category)
        all_articles.extend(yo_articles)
    except Exception as e:
        print(f"⚠️ YouTube Outlier Hunter fetch failed: {e}")
    
    # Compute unified engagement scores
    for art in all_articles:
        art["_engagement_score"] = compute_engagement_score(art)
    
    # Sort by engagement score
    all_articles.sort(key=lambda x: x.get("_engagement_score", 0), reverse=True)
    
    # Summary
    yt_count = sum(1 for a in all_articles if a.get("type") == "youtube_trending")
    reddit_count = sum(1 for a in all_articles if a.get("type") == "reddit_trending")
    reddit_estimated = sum(1 for a in all_articles if a.get("type") == "reddit_trending" and a.get("_engagement", {}).get("engagement_estimated"))
    github_count = sum(1 for a in all_articles if a.get("type") == "github_trending")
    hn_count = sum(1 for a in all_articles if a.get("type") == "hacker_news")
    hf_count = sum(1 for a in all_articles if a.get("type") == "huggingface_trending")
    arxiv_count = sum(1 for a in all_articles if a.get("type") == "arxiv_papers")
    tldr_count = sum(1 for a in all_articles if a.get("type") == "newsletter_ai")
    gt_count = sum(1 for a in all_articles if a.get("type") == "google_trends")
    yo_count = sum(1 for a in all_articles if a.get("type") == "youtube_outliers")
    
    print(f"\n📊 Trending Engine Summary: {len(all_articles)} total signals")
    print(f"   YouTube: {yt_count} | Reddit: {reddit_count} | GitHub: {github_count} | Hacker News: {hn_count}")
    print(f"   HuggingFace: {hf_count} | ArXiv: {arxiv_count} | TLDR AI: {tldr_count} | Google Trends: {gt_count} | YouTube Outliers: {yo_count}")
    if all_articles:
        top = all_articles[0]
        print(f"   🏆 Top Signal: '{top['title'][:60]}...' (Score: {top.get('_engagement_score', 0)})")
    
    # ── Data Source Health Dashboard ──────────────────────────────────────
    yt_status = "✅ Active" if yt_count > 0 else "❌ Offline (YOUTUBE_DATA_API_KEY missing?)"
    reddit_native = reddit_count - reddit_estimated
    reddit_status = "✅ Active" if reddit_native > 0 else ("⚠️ Degraded" if reddit_estimated > 0 else "❌ Offline")
    github_status = "✅ Active" if github_count > 0 else "⚠️ No results"
    hn_status = "✅ Active" if hn_count > 0 else "⚠️ No results"
    hf_status = "✅ Active" if hf_count > 0 else "⚠️ No results"
    arxiv_status = "✅ Active" if arxiv_count > 0 else "⚠️ No results"
    tldr_status = "✅ Active" if tldr_count > 0 else "⚠️ No results"
    gt_status = "✅ Active" if gt_count > 0 else "⚠️ No results"
    yo_status = "✅ Active" if yo_count > 0 else "⚠️ No results"
    
    active_count = sum(1 for s in [yt_status, reddit_status, github_status, hn_status, hf_status, arxiv_status, tldr_status, gt_status, yo_status] if s.startswith("✅"))
    print(f"\n🏥 Data Source Health: {active_count}/9 sources fully active")
    print(f"   YouTube Trending : {yt_status}")
    print(f"   Reddit           : {reddit_status}")
    print(f"   GitHub           : {github_status}")
    print(f"   Hacker News      : {hn_status}")
    print(f"   Hugging Face     : {hf_status}")
    print(f"   ArXiv AI         : {arxiv_status}")
    print(f"   TLDR AI          : {tldr_status}")
    print(f"   Google Trends    : {gt_status}")
    print(f"   YouTube Outliers : {yo_status}")
    
    return all_articles
