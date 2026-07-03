import re
import random
import time
import requests
import feedparser
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from config import NEWS_API_KEY

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/114.0'
]

# ── CONTENT SOURCES (Mass-Appeal Tech & AI Research) ─────────────────────────────────────────
RSS_FEEDS = [
    "https://www.theverge.com/rss/index.xml",
    "https://arstechnica.com/gadgets/feed/",
    "https://www.wired.com/feed/category/gear/latest/rss",
    "https://www.tomsguide.com/feeds/all",
    "https://lifehacker.com/feed/rss",
    "https://mashable.com/category/tech/rss",
    "https://www.cnet.com/rss/news/",
    "https://arstechnica.com/tag/ai/feed/",
    "https://www.engadget.com/rss/",
    "https://www.technologyreview.com/feed/",
    
    # AI Research & Engineering (from Tech News by VJ bot.py)
    "https://openai.com/blog/rss/",
    "https://research.google/blog/rss/", 
    "https://www.anthropic.com/rss",
    "https://huggingface.co/blog/feed.xml",
    "https://aws.amazon.com/blogs/machine-learning/feed/",
]

TOOL_RSS_FEEDS = [
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/apps/rss/index.xml",
    "https://venturebeat.com/feed/",
    "https://news.ycombinator.com/rss",
    "https://9to5mac.com/feed/",
    "https://9to5google.com/feed/",
    "https://www.xda-developers.com/feed/",
    "https://www.makeuseof.com/feed/",
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
            f"q=(\"tech tips\" OR \"hidden features\" OR \"AI tool\" OR \"iPhone\" OR \"Android\" OR \"privacy\")&"
            f"from={from_date}&"
            f"sortBy=popularity&"
            f"language=en&"
            f"pageSize=20&"
            f"apiKey={NEWS_API_KEY}"
        )
        response = requests.get(url, timeout=25)
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

def fetch_deep_article_content(url):
    """Visits the actual URL to grab real paragraph text for a better LLM summary."""
    try:
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, 'html.parser')
            # Extract text from paragraphs
            paragraphs = soup.find_all('p')
            text = " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
            return text[:1000] # Return the first 1000 characters
    except Exception:
        pass
    return ""

def _fetch_rss(feed_urls, feed_type="research", hours=24):
    print(f"Fetching from {feed_type} blogs (within last {hours} hours)...")
    all_articles = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

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

                # Clean HTML from RSS description
                clean_desc = ""
                if desc:
                    try:
                        soup = BeautifulSoup(desc, "html.parser")
                        clean_desc = soup.get_text()[:400].strip()
                    except Exception:
                        clean_desc = desc[:400].strip()

                # Fetch deeper content if description is short or teaser
                deep_content = ""
                if len(clean_desc) < 150:
                    deep_content = fetch_deep_article_content(entry.link)

                final_desc = deep_content if (deep_content and len(deep_content) > len(clean_desc)) else clean_desc

                all_articles.append({
                    "title": title,
                    "description": final_desc,
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
        print(f"{feed_type.capitalize()} Blogs: Found {len(articles)} fresh articles in the last {hours}h.")
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


def fetch_tech_news(hours=24):
    print("=== FETCH LATEST AI RESEARCH & ENGINEERING NEWS ===")
    return _fetch_rss(RSS_FEEDS, "research", hours=hours)

def fetch_ai_tools(hours=24):
    print("=== FETCH LATEST AI TOOLS & PRODUCTS ===")
    return _fetch_rss(TOOL_RSS_FEEDS, "tools", hours=hours)

def fetch_reddit_news(hours=24):
    """Fetches trending tech/AI news from specified subreddits."""
    import random
    from bs4 import BeautifulSoup
    
    REDDIT_SUBREDDITS = [
        "MachineLearning",
        "artificial",
        "LocalLLaMA", 
        "technology",
        "singularity" 
    ]
    
    def is_within_hours(published_date_str, hrs):
        if not published_date_str:
             return True
        try:
            from dateutil import parser as date_parser
            import pytz
            pub_date = date_parser.parse(published_date_str)
            if pub_date.tzinfo is None:
                 pub_date = pytz.utc.localize(pub_date)
            now = datetime.now(pytz.utc)
            return (now - pub_date) < timedelta(hours=hrs)
        except Exception:
            return True

    news_items = []
    print("👽 Fetching Reddit top posts...")
    
    for sub in REDDIT_SUBREDDITS:
        # Use t=week to fetch up to 7 days, then filter locally by `hours`
        url = f"https://www.reddit.com/r/{sub}/top/.rss?t=week&limit=15"
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                for entry in feed.entries:
                    published = getattr(entry, 'updated', None)
                    if published and not is_within_hours(published, hours):
                        continue
                        
                    post_url = entry.link
                    
                    # Extract image url if any
                    image_url = ""
                    if hasattr(entry, 'content') and len(entry.content) > 0:
                        m = re.search(r'img.*?src=[\'\"](.*?)[\'\"]', entry.content[0].value)
                        if m:
                            image_url = m.group(1)
                    
                    deep_content = fetch_deep_article_content(post_url)
                    # Clean Reddit summaries/HTML if not fetching deep content
                    reddit_summary = ""
                    if hasattr(entry, 'summary'):
                        soup_sum = BeautifulSoup(entry.summary, "html.parser")
                        reddit_summary = soup_sum.get_text()[:400].strip() + "..."
                    
                    news_items.append({
                        "title": entry.title,
                        "description": deep_content if deep_content else reddit_summary,
                        "source": {"name": f"r/{sub}"},
                        "url": post_url,
                        "urlToImage": image_url,
                        "publishedAt": published or datetime.now().isoformat(),
                        "type": "research" if sub in ["MachineLearning", "LocalLLaMA", "singularity"] else "trending"
                    })
            else:
                print(f"⚠️ Reddit Error {response.status_code} for r/{sub}")
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ Error fetching r/{sub}: {e}")
            
    print(f"✅ Reddit Search: Retrieved {len(news_items)} posts from the last {hours} hours.")
    return news_items

def _fetch_xcom_via_google_news():
    """
    Fallback: Query Google News RSS for viral AI tweets from X.com
    when the X.com API is unavailable (credits depleted, rate limited, etc.)
    """
    print("  🔍 X.com Fallback: Querying Google News RSS for viral AI tweets...")
    articles = []
    
    queries = [
        "site:x.com AI artificial intelligence",
        "site:x.com LLM machine learning viral",
        "site:x.com OpenAI Google DeepMind",
    ]
    
    for query in queries:
        try:
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en&gl=US&ceid=US:en"
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()
                
            root = ET.fromstring(xml_data)
            items = root.findall('.//item')
            
            for item in items[:3]:  # Top 3 per query
                title = item.find('title').text or ""
                title_clean = re.sub(r'\s+-\s+.*$', '', title).strip()
                link = item.find('link').text or ""
                pub_date = item.find('pubDate').text or ""
                
                articles.append({
                    "title": title_clean,
                    "description": f"Google News indexed tweet: {title_clean}",
                    "source": {"name": "X.com (via News)"},
                    "url": link,
                    "urlToImage": "",
                    "publishedAt": pub_date,
                    "type": "trending"
                })
        except Exception as e:
            print(f"  ⚠️ X.com News fallback query failed: {e}")
    
    if articles:
        print(f"  ✅ X.com Fallback: Retrieved {len(articles)} topics via Google News.")
    else:
        print(f"  ⚠️ X.com Fallback: No results from Google News.")
    return articles


def fetch_x_trending_ai_topics():
    """
    Fetches trending AI topics from X.com (Twitter) using the search recent API v2.
    Sorts by engagement to return the most popular/viral tweets on AI.
    Falls back to Google News RSS if API credits are depleted.
    """
    from config import X_BEARER_TOKEN
    if not X_BEARER_TOKEN or "XXX" in X_BEARER_TOKEN or not X_BEARER_TOKEN.strip():
        print("⚠️ X.com Bearer Token missing. Trying Google News fallback...")
        return _fetch_xcom_via_google_news()
        
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
            if response.status_code == 402:
                print("⚠️ X.com API Error (HTTP 402): Credits Depleted. Falling back to Google News RSS...")
            elif response.status_code == 429:
                print("⚠️ X.com API Error (HTTP 429): Rate Limited. Falling back to Google News RSS...")
            else:
                print(f"⚠️ X.com API Error (HTTP {response.status_code}): {response.text[:200]}. Falling back to Google News RSS...")
            return _fetch_xcom_via_google_news()
            
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
        print(f"⚠️ X.com Search fetch failed: {e}. Trying Google News fallback...")
        return _fetch_xcom_via_google_news()

