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
]

TOOL_RSS_FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
    "https://venturebeat.com/category/ai/feed/",
    "https://news.ycombinator.com/rss", 
]

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
    
    fresh_articles = [a for a in all_articles if a["_pub_dt"] >= cutoff]
    
    # Remove the internal temporary _pub_dt field
    for a in all_articles:
        del a["_pub_dt"]

    if fresh_articles:
        articles = fresh_articles
        print(f"{feed_type.capitalize()} Blogs: Found {len(articles)} fresh articles in the last 24h.")
    elif all_articles:
        articles = all_articles[:10]  # Take top 10 most recent if no fresh ones
        print(f"{feed_type.capitalize()} Blogs: Found 0 fresh articles. Falling back to the {len(articles)} most recent ones from RSS.")
    else:
        # STEP 1.2: Historical Fallback from log check
        print(f"{feed_type.capitalize()} Blogs: RSS FEEDS EMPTY. Checking historical facts_log.json...")
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
