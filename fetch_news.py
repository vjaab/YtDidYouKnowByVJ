import re
import requests
import feedparser
from datetime import datetime, timedelta, timezone
from config import TWITTER_BEARER_TOKEN, NEWS_API_KEY

# ── APPROACH 1: Twitter API v2 (Best) ─────────────────────────────────────────
def _fetch_from_twitter():
    if not TWITTER_BEARER_TOKEN:
        print("Twitter Bearer Token not found. Skipping Twitter API.")
        return []

    print("Fetching trending topics from Twitter API v2...")
    # Keywords
    keywords = "AI OR \"artificial intelligence\" OR ChatGPT OR OpenAI OR Google OR Meta OR Apple OR Tesla OR Gemini OR GPT OR Claude OR tech OR startup"
    query = f"({keywords}) -is:retweet -is:reply min_retweets:500 min_faves:1000 lang:en"
    
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": 50,
        "sort_order": "relevancy",
        "tweet.fields": "public_metrics,entities,author_id,created_at",
        "expansions": "author_id",
        "user.fields": "name,username,profile_image_url"
    }
    headers = {
        "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            tweets = data.get("data", [])
            includes = data.get("includes", {}).get("users", [])
            user_map = {u["id"]: u for u in includes}
            
            articles = []
            for t in tweets:
                author_id = t.get("author_id")
                user = user_map.get(author_id, {})
                
                # We format this as an "article" so the rest of the pipeline handles it exactly the same
                articles.append({
                    "title": t.get("text", "")[:100] + "...", 
                    "description": t.get("text", ""),
                    "source": {"name": f"Twitter: @{user.get('username', 'unknown')}"},
                    "url": f"https://x.com/{user.get('username', 'unknown')}/status/{t.get('id')}",
                    "urlToImage": user.get("profile_image_url", ""),
                    "publishedAt": t.get("created_at", ""),
                    "metrics": t.get("public_metrics", {})
                })
            print(f"Twitter API: Found {len(articles)} viral tweets.")
            return articles
        else:
            print(f"Twitter API error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"Twitter API failed: {e}")
    return []

# ── APPROACH 2: NewsAPI Fallback ──────────────────────────────────────────────
DOMAINS = "techcrunch.com,theverge.com,wired.com,arstechnica.com"

def _fetch_from_newsapi():
    if not NEWS_API_KEY:
        print("NewsAPI key not found. Skipping NewsAPI.")
        return []

    print("Falling back to NewsAPI...")
    from_date = (datetime.now(timezone.utc) - timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%S')
    
    try:
        params = {
            "q": "AI OR tech OR startup OR OpenAI OR Google",
            "language": "en",
            "sortBy": "popularity",
            "pageSize": 50,
            "from": from_date,
            "domains": DOMAINS,
            "apiKey": NEWS_API_KEY
        }
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            articles = data.get("articles", [])
            print(f"NewsAPI: Found {len(articles)} popular articles in last 6 hours.")
            return articles
        else:
            print(f"NewsAPI error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"NewsAPI failed: {e}")
    return []

# ── APPROACH 3: RSS Fallback ──────────────────────────────────────────────────
RSS_FEEDS = [
    "https://feeds.feedburner.com/TechCrunch",
    "https://www.theverge.com/rss/index.xml",
    "https://hnrss.org/frontpage"
]

def _fetch_from_rss():
    print("Falling back to RSS feeds...")
    articles = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24) # Relaxed time for RSS

    for feed_url in RSS_FEEDS:
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
                    "source": {"name": getattr(entry, 'source', {}).get("title", "RSS Feed")},
                    "url": entry.link,
                    "urlToImage": image_url,
                    "publishedAt": pub_dt.isoformat() if pub_dt else "",
                })
        except Exception as e:
            print(f"Feed failed {feed_url}: {e}")
            
    print(f"RSS: Found {len(articles)} fresh articles.")
    return articles[:50]


def fetch_tech_news():
    print("=== FETCH TRENDING TOPICS ===")
    
    # Priority 1: Twitter API v2
    articles = _fetch_from_twitter()
    if len(articles) >= 5:
        return articles
        
    # Priority 2: NewsAPI fallback
    articles = _fetch_from_newsapi()
    if len(articles) >= 5:
        return articles
        
    # Priority 3: RSS feeds fallback
    articles = _fetch_from_rss()
    return articles
