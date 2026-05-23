import re
import requests
import feedparser
from datetime import datetime, timedelta, timezone
from config import NEWS_API_KEY

# ── RESEARCH SOURCES ──────────────────────────────────────────────────────────
RSS_FEEDS = [
    "https://openai.com/blog/rss/",
    "https://research.google/blog/rss/", 
    "https://www.anthropic.com/rss",
    "https://huggingface.co/blog/feed.xml",
    "https://aws.amazon.com/blogs/machine-learning/feed/",
    "https://arstechnica.com/tag/ai/feed/",
    "https://decrypt.co/news/technology/ai/rss",
    "https://www.wired.com/feed/tag/ai/latest/rss",
    "https://www.technologyreview.com/feed/",
    "https://www.engadget.com/tag/artificial-intelligence/rss/",
]

TOOL_RSS_FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
    "https://venturebeat.com/category/ai/feed/",
    "https://news.ycombinator.com/rss", 
    "https://mashable.com/category/artificial-intelligence/rss",
    "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "https://www.reuters.com/technology/rss",
]

def fetch_trending_from_newsapi():
    """Fetches global trending AI news using NewsAPI.org sorted by popularity."""
    if not NEWS_API_KEY:
        print("⚠️ NewsAPI key missing. Skipping global trend fetch.")
        return []
    
    print("📡 Fetching global trending AI news from NewsAPI...")
    try:
        # Query for AI breakthroughs, model releases, and leaks in the last 48h
        from_date = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q=(AI OR \"Artificial Intelligence\" OR LLM OR \"Generative AI\" OR \"OpenAI\" OR \"DeepMind\")&"
            f"from={from_date}&"
            f"sortBy=popularity&"
            f"language=en&"
            f"pageSize=20&"
            f"apiKey={NEWS_API_KEY}"
        )
        response = requests.get(url)
        data = response.json()
        
        if data.get("status") != "ok":
            print(f"⚠️ NewsAPI Error: {data.get('message')}")
            return []
            
        articles = data.get("articles", [])
        print(f"✅ NewsAPI: Found {len(articles)} trending articles.")
        
        # Reformat to match internal structure
        return [{
            "title": a.get("title"),
            "description": a.get("description"),
            "source": {"name": a.get("source", {}).get("name")},
            "url": a.get("url"),
            "urlToImage": a.get("urlToImage"),
            "publishedAt": a.get("publishedAt"),
            "type": "trending"
        } for a in articles]
        
    except Exception as e:
        print(f"⚠️ NewsAPI Fetch failed: {e}")
        return []

def _fetch_rss(feed_urls, feed_type="research"):
    print(f"Fetching from {feed_type} blogs...")
    all_articles = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    for feed_url in feed_urls:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:20]:
                title = getattr(entry, 'title', '')
                desc  = getattr(entry, 'summary', '')
                
                # Parse date
                pub_dt = None
                for field in ('published_parsed', 'updated_parsed'):
                    t = getattr(entry, field, None)
                    if t:
                        try:
                            pub_dt = datetime(*t[:6], tzinfo=timezone.utc)
                            break
                        except:
                            pass
                
                # Parse image
                image_url = ""
                if hasattr(entry, 'media_content') and len(entry.media_content) > 0:
                    image_url = entry.media_content[0].get('url', '')
                elif hasattr(entry, 'media_thumbnail') and len(entry.media_thumbnail) > 0:
                    image_url = entry.media_thumbnail[0].get('url', '')
                if not image_url and hasattr(entry, 'content') and len(entry.content) > 0:
                    m = re.search(r'img.*?src=[\'\"](.*?)[\'\"]', entry.content[0].value)
                    if m:
                        image_url = m.group(1)

                all_articles.append({
                    "title": title,
                    "description": desc,
                    "source": {"name": getattr(feed.feed, 'title', 'Research Blog')},
                    "url": entry.link,
                    "urlToImage": image_url,
                    "publishedAt": pub_dt.isoformat() if pub_dt else "",
                    "type": feed_type,
                    "_pub_dt": pub_dt or datetime(1970, 1, 1, tzinfo=timezone.utc)
                })
        except Exception as e:
            print(f"Feed failed {feed_url}: {e}")
            
    # Sort by date descending
    all_articles.sort(key=lambda x: x["_pub_dt"], reverse=True)
    
    # STRICT RECENCY FILTER: Only keep articles from the last 7 days.
    # We never want to fallback to 2-year-old articles.
    absolute_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    valid_articles = [a for a in all_articles if a["_pub_dt"] >= absolute_cutoff]
    
    fresh_articles = [a for a in valid_articles if a["_pub_dt"] >= cutoff]
    
    # Remove the internal temporary _pub_dt field
    for a in all_articles:
        if "_pub_dt" in a: del a["_pub_dt"]

    if fresh_articles:
        articles = fresh_articles
        print(f"{feed_type.capitalize()} Blogs: Found {len(articles)} fresh articles in the last 24h.")
    elif valid_articles:
        articles = valid_articles[:10]  # Take top 10 most recent if no fresh ones (but within 7 days)
        print(f"{feed_type.capitalize()} Blogs: Found 0 fresh articles. Falling back to {len(articles)} recent ones from this week.")
    else:
        # STEP 1.2: Historical Fallback from log check (ONLY if they are recent enough)
        print(f"{feed_type.capitalize()} Blogs: RSS FEEDS EMPTY or STALE. Checking historical facts_log.json...")
        from config import TRACKER_FILE
        import json
        import os
        if os.path.exists(TRACKER_FILE):
             with open(TRACKER_FILE, 'r') as f:
                 tracker = json.load(f)
                 history = tracker.get("history", [])
                 if history:
                     # Filter by type if possible, or just take random recent ones
                     backup = [h for h in history if h.get("sub_category") == feed_type or feed_type == "research"]
                     articles = backup[-10:] if backup else history[-10:]
                     print(f"✅ Loaded {len(articles)} historical articles from tracker as fallback.")
                     # Reformat to match RSS article structure
                     articles = [{
                         "title": a.get("news_headline", a.get("title")),
                         "description": "Historical coverage fallback.",
                         "source": {"name": "Historical Cache"},
                         "url": a.get("news_source_url", ""),
                         "urlToImage": "",
                         "publishedAt": a.get("date", ""),
                         "type": feed_type
                     } for a in articles]
        
        if not articles:
            print("⚠️ Critical: No RSS or Historical data found.")

    return articles


def fetch_tech_news():
    print("=== FETCH LATEST AI RESEARCH & ENGINEERING NEWS ===")
    return _fetch_rss(RSS_FEEDS, "research")

def fetch_ai_tools():
    print("=== FETCH LATEST AI TOOLS & PRODUCTS ===")
    return _fetch_rss(TOOL_RSS_FEEDS, "tools")

def fetch_x_trending_ai_topics():
    """
    Fetches trending AI topics from X.com (Twitter) using the search recent API v2.
    Sorts by engagement to return the most popular/viral tweets on AI.
    """
    from config import X_BEARER_TOKEN
    if not X_BEARER_TOKEN or "XXX" in X_BEARER_TOKEN or not X_BEARER_TOKEN.strip():
        print("⚠️ X.com Bearer Token missing. Skipping X trending fetch.")
        return []
        
    print("📡 Fetching viral trending AI topics from X.com...")
    url = "https://api.twitter.com/2/tweets/search/recent"
    
    # Query for popular AI tweets in English from last 7 days
    query = "(#AI OR #LLM OR #GenerativeAI OR OpenAI OR DeepMind OR Claude3 OR Gemini1.5) -is:retweet lang:en"
    
    params = {
        "query": query,
        "max_results": 15,
        "tweet.fields": "created_at,public_metrics,author_id",
        "expansions": "author_id",
        "user.fields": "username,name"
    }
    
    headers = {
        "Authorization": f"Bearer {X_BEARER_TOKEN}"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=25)
        if response.status_code != 200:
            print(f"⚠️ X.com API Error (HTTP {response.status_code}): {response.text}")
            return []
            
        data = response.json()
        tweets = data.get("data", [])
        users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
        
        articles = []
        for t in tweets:
            author_id = t.get("author_id")
            user_info = users.get(author_id, {})
            username = user_info.get("username", "XUser")
            
            text = t.get("text", "")
            # Clean text for title
            clean_text = re.sub(r"https://t.co/\S+", "", text).strip()
            clean_text = clean_text.replace("\n", " ")
            title = f"X.com Alert: @{username} on AI"
            
            # Format public metrics
            metrics = t.get("public_metrics", {})
            likes = metrics.get("like_count", 0)
            rts = metrics.get("retweet_count", 0)
            engagement_desc = f"📈 Engagement: {likes} Likes, {rts} Retweets. \n\nTweet: {text}"
            
            tweet_url = f"https://x.com/{username}/status/{t.get('id')}"
            
            articles.append({
                "title": title,
                "description": engagement_desc,
                "source": {"name": f"X.com (@{username})"},
                "url": tweet_url,
                "urlToImage": "",
                "publishedAt": t.get("created_at"),
                "type": "trending"
            })
            
        print(f"✅ X.com Search: Retrieved {len(articles)} viral AI topics.")
        return articles
    except Exception as e:
        print(f"⚠️ X.com Search fetch failed: {e}")
        return []
