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
    articles = []
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
                
                if pub_dt and pub_dt < cutoff:
                    continue

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

                articles.append({
                    "title": title,
                    "description": desc,
                    "source": {"name": getattr(feed.feed, 'title', 'Research Blog')},
                    "url": entry.link,
                    "urlToImage": image_url,
                    "publishedAt": pub_dt.isoformat() if pub_dt else "",
                    "type": feed_type
                })
        except Exception as e:
            print(f"Feed failed {feed_url}: {e}")
            
    print(f"{feed_type.capitalize()} Blogs: Found {len(articles)} fresh articles.")
    return articles


def fetch_tech_news():
    print("=== FETCH LATEST AI RESEARCH & ENGINEERING NEWS ===")
    return _fetch_rss(RSS_FEEDS, "research")

def fetch_ai_tools():
    print("=== FETCH LATEST AI TOOLS & PRODUCTS ===")
    return _fetch_rss(TOOL_RSS_FEEDS, "tools")
