import re
import requests
import feedparser
from datetime import datetime, timedelta, timezone
from config import NEWS_API_KEY

# ── NewsAPI config ────────────────────────────────────────────────────────────
DOMAINS = (
    "techcrunch.com,theverge.com,wired.com,arstechnica.com,"
    "engadget.com,thenextweb.com,venturebeat.com,9to5google.com,"
    "9to5mac.com,zdnet.com,tomsguide.com,androidauthority.com"
)

# ── AI & Tech only RSS feeds (category-specific URLs) ────────────────────────
RSS_FEEDS = [
    # TechCrunch — AI category
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    # The Verge — AI section
    "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
    # Ars Technica — AI section
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    # MIT Technology Review — AI
    "https://www.technologyreview.com/feed/",
    # VentureBeat — AI
    "https://venturebeat.com/category/ai/feed/",
    # The Next Web — Neural (AI)
    "https://thenextweb.com/neural/feed/",
    # Wired — Backchannel (Science/Tech)
    "https://www.wired.com/feed/tag/ai/rss",
    # TechCrunch general tech as backup
    "https://feeds.feedburner.com/TechCrunch",
]

FRESHNESS_HOURS = 24

# ── AI/Tech keyword filter ────────────────────────────────────────────────────
TECH_KEYWORDS = {
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "neural network", "chatgpt", "openai", "gemini", "claude", "llm",
    "robot", "robotics", "automation", "algorithm", "semiconductor",
    "chip", "gpu", "nvidia", "apple", "google", "microsoft", "meta",
    "amazon", "tesla", "spacex", "smartphone", "iphone", "android",
    "software", "app", "startup", "tech", "cyber", "hack", "privacy",
    "data", "cloud", "quantum", "5g", "ev", "electric vehicle",
    "drone", "ar", "vr", "augmented reality", "virtual reality",
    "gpt", "model", "dataset", "training", "inference", "api",
    "launch", "release", "product", "gadget", "device", "wearable",
    "browser", "search", "social media", "regulation", "ban", "policy"
}

EXCLUDE_KEYWORDS = {
    "recipe", "workout", "diet", "horoscope", "sports", "nfl", "nba",
    "cricket", "soccer", "football", "baseball", "fashion", "celebrity",
    "movie review", "box office", "music", "concert", "tourism", "travel",
    "astrology", "supplement", "toothbrush", "olympics", "winter games",
    "creatine", "sti test", "fill power", "lunar eclipse guide"
}


def _is_tech_related(title, description=""):
    """Return True if the article is clearly AI/tech related."""
    combined = (title + " " + description).lower()

    # Exclude obvious non-tech content first
    for kw in EXCLUDE_KEYWORDS:
        if kw in combined:
            return False

    # Must contain at least one tech keyword
    for kw in TECH_KEYWORDS:
        if kw in combined:
            return True

    return False


def _parse_entry_date(entry):
    for field in ('published_parsed', 'updated_parsed'):
        t = getattr(entry, field, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def _is_fresh(entry, hours=FRESHNESS_HOURS):
    pub_dt = _parse_entry_date(entry)
    if pub_dt is None:
        return True     # No date → keep optimistically
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return pub_dt >= cutoff


def _entry_to_article(entry):
    image_url = ""
    if hasattr(entry, 'media_content') and len(entry.media_content) > 0:
        image_url = entry.media_content[0].get('url', '')
    elif hasattr(entry, 'media_thumbnail') and len(entry.media_thumbnail) > 0:
        image_url = entry.media_thumbnail[0].get('url', '')
    if not image_url and hasattr(entry, 'content') and len(entry.content) > 0:
        m = re.search(r'img.*?src=[\'\"](.*?)[\'\"]', entry.content[0].value)
        if m:
            image_url = m.group(1)

    pub_dt = _parse_entry_date(entry)
    return {
        "title":       entry.title,
        "description": getattr(entry, 'summary', ''),
        "source":      {"name": getattr(entry, 'source', {}).get("title", "RSS Feed")},
        "url":         entry.link,
        "urlToImage":  image_url,
        "publishedAt": pub_dt.isoformat() if pub_dt else "",
    }


def fetch_tech_news():
    # ── 1. NewsAPI with AI/tech query ─────────────────────────────────────────
    if NEWS_API_KEY:
        from_date = (datetime.now(timezone.utc) - timedelta(hours=FRESHNESS_HOURS)).strftime('%Y-%m-%dT%H:%M:%S')
        ai_query = (
            "artificial intelligence OR AI OR ChatGPT OR OpenAI OR Gemini "
            "OR robot OR machine learning OR tech OR Apple OR Google OR Microsoft "
            "OR startup OR gadget OR smartphone OR privacy OR cybersecurity"
        )
        try:
            params = {
                "q":        ai_query,
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": 30,
                "from":     from_date,
                "domains":  DOMAINS,
                "apiKey":   NEWS_API_KEY
            }
            r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
            data = r.json()
            if data.get("status") == "ok" and data.get("articles"):
                articles = [a for a in data["articles"] if _is_tech_related(a.get("title",""), a.get("description",""))]
                if articles:
                    print(f"NewsAPI: {len(articles)} AI/tech articles (last 24h).")
                    return articles[:20]
        except Exception as e:
            print(f"NewsAPI failed: {e}")

    # ── 2. RSS fallback — AI/tech category feeds + keyword filter ─────────────
    print("Falling back to AI/tech RSS feeds (last 24h filter)...")
    fresh, stale = [], []

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:20]:
                title = getattr(entry, 'title', '')
                desc  = getattr(entry, 'summary', '')
                if not _is_tech_related(title, desc):
                    continue
                article = _entry_to_article(entry)
                if _is_fresh(entry, hours=FRESHNESS_HOURS):
                    fresh.append(article)
                else:
                    stale.append(article)
        except Exception as e:
            print(f"Feed failed {feed_url}: {e}")

    fresh.sort(key=lambda a: a.get("publishedAt", ""), reverse=True)

    if len(fresh) >= 5:
        print(f"RSS: {len(fresh)} fresh AI/tech articles (< {FRESHNESS_HOURS}h).")
        return fresh[:20]

    # Relax to 72h if not enough
    print(f"Only {len(fresh)} fresh articles. Expanding to 72h window...")
    fresh = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:20]:
                if _is_tech_related(getattr(entry, 'title', ''), getattr(entry, 'summary', '')):
                    if _is_fresh(entry, hours=72):
                        fresh.append(_entry_to_article(entry))
        except Exception:
            pass

    fresh.sort(key=lambda a: a.get("publishedAt", ""), reverse=True)
    if fresh:
        print(f"RSS (72h): {len(fresh)} AI/tech articles.")
        return fresh[:20]

    print("WARNING: Falling back to stale articles.")
    return stale[:20]
