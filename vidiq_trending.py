"""
vidiq_trending.py — Core module to fetch trending AI/technology topics
and keyword opportunities using the YouTube Data API and Google Autocomplete Suggest API.
"""
import os
import re
import requests
from datetime import datetime

def get_trending_videos(api_key=None, region="IN"):
    """
    Fetches live trending tech videos from YouTube Data API v3 (Category 28: Science & Technology).
    """
    yt_key = os.getenv("YOUTUBE_DATA_API_KEY")
    if yt_key:
        print(f"📡 Querying YouTube Data API for trending Science & Tech videos (region={region})...")
        try:
            url = "https://www.googleapis.com/youtube/v3/videos"
            params = {
                "part": "snippet,statistics",
                "chart": "mostPopular",
                "videoCategoryId": "28", # Science & Technology
                "maxResults": 10,
                "regionCode": region,
                "key": yt_key
            }
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                videos = []
                for item in r.json().get("items", []):
                    snippet = item.get("snippet", {})
                    vid_id = item.get("id")
                    videos.append({
                        "title": snippet.get("title", ""),
                        "id": vid_id,
                        "url": f"https://youtube.com/watch?v={vid_id}"
                    })
                if videos:
                    return videos
            else:
                print(f"⚠️ YouTube trending API returned {r.status_code}. Using fallbacks.")
        except Exception as e:
            print(f"⚠️ YouTube trending API failed: {e}. Using fallbacks.")
            
    return _get_fallback_trending_videos()

def get_keyword_research(seed_term, api_key=None):
    """
    Fetches search suggestions for a seed keyword from YouTube Suggest API
    to compute a mock but high-signal search score.
    """
    import urllib.parse
    try:
        encoded_query = urllib.parse.quote(seed_term)
        url = f"http://suggestqueries.google.com/complete/search?client=firefox&ds=yt&q={encoded_query}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            suggestions = data[1] if len(data) > 1 else []
            
            # Calculate a search score based on number and lengths of suggestions
            num_sug = len(suggestions)
            score = min(98, 45 + num_sug * 6)  # 0 suggestions = 45, 9 suggestions = 98
            volume = 1000 + num_sug * 1200
            competition = max(10, 85 - num_sug * 5) # more suggestions = less niche/easier?
            
            return {
                "keyword": seed_term,
                "search_volume": volume,
                "competition": competition,
                "score": score,
                "suggestions": suggestions[:5]
            }
    except Exception as e:
        print(f"⚠️ YouTube keyword suggest API failed: {e}")
        
    return {"keyword": seed_term, "search_volume": 6200, "competition": 32, "score": 78}

def get_breakout_channels(api_key=None, category="AI"):
    """
    Identifies competitor channels and crawls their latest videos if key is present.
    """
    yt_key = os.getenv("YOUTUBE_DATA_API_KEY")
    fallbacks = _get_fallback_breakout_channels()
    
    if not yt_key:
        return fallbacks
        
    print(f"📡 Querying YouTube API for breakout channels in category: {category}...")
    for ch in fallbacks:
        try:
            # Search for 2 recent videos from the channel name
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": ch["name"],
                "type": "video",
                "maxResults": 2,
                "order": "date",
                "key": yt_key
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                videos = []
                for item in r.json().get("items", []):
                    snippet = item.get("snippet", {})
                    vid_id = item.get("id", {}).get("videoId")
                    if vid_id:
                        videos.append({
                            "title": snippet.get("title", ""),
                            "url": f"https://youtube.com/watch?v={vid_id}"
                        })
                if videos:
                    ch["recent_videos"] = videos
        except Exception as e:
            print(f"⚠️ Failed to update breakout channel {ch['name']}: {e}")
            
    return fallbacks

def get_pipeline_topics(category="AI & Tech Tools"):
    """
    Combines trending videos, breakout competitor insights, and keyword opportunities
    into a structured, ranked topic list for the script generator.
    """
    api_key = os.getenv("VIDIQ_API_KEY")
    
    print(f"📡 Querying YouTube and Autocomplete Pipeline for category: {category}...")
    trending_vids = get_trending_videos(api_key, region="IN")
    competitor_channels = get_breakout_channels(api_key, category=category)
    
    ranked_topics = []
    
    # 1. Process local trending inputs
    for video in trending_vids:
        title = video.get("title", "")
        clean_topic = _extract_topic(title)
        if not clean_topic or len(clean_topic) < 10:
            continue
            
        research = get_keyword_research(clean_topic, api_key)
        ranked_topics.append({
            "title": clean_topic,
            "original_title": title,
            "score": research.get("score", 60),
            "search_volume": research.get("search_volume", 5000),
            "competition": research.get("competition", 40),
            "source": "vidiq_trending_video",
            "url": video.get("url", f"https://youtube.com/watch?v={video.get('id', '')}")
        })
        
    # 2. Process breakout channels
    for ch in competitor_channels:
        ch_name = ch.get("name", "Competitor")
        for video in ch.get("recent_videos", []):
            title = video.get("title", "")
            clean_topic = _extract_topic(title)
            if not clean_topic or len(clean_topic) < 10:
                continue
                
            research = get_keyword_research(clean_topic, api_key)
            ranked_topics.append({
                "title": clean_topic,
                "original_title": title,
                "score": research.get("score", 55) + 5,  # competitor bias boost
                "search_volume": research.get("search_volume", 4000),
                "competition": research.get("competition", 30),
                "source": f"vidiq_competitor_{ch_name}",
                "url": video.get("url", "")
            })
            
    # Rank by overall search/competition opportunity score
    ranked_topics.sort(key=lambda x: x["score"], reverse=True)
    return ranked_topics

def _extract_topic(title):
    """
    Strips clickbait hooks and emojis from YouTube titles to extract the core topic.
    """
    if not title:
        return ""
        
    # Remove common emojis
    text = re.sub(r'[^\w\s\-\.\,\:\'\?\!\/\&]', '', title)
    
    # Remove common clickbait prefixes and filler phrases
    fillers = [
        r"(?i)\byou won't believe\b",
        r"(?i)\bthis changes everything\b",
        r"(?i)\bsecret settings?\b",
        r"(?i)\bsecret features?\b",
        r"(?i)\bhow i\b",
        r"(?i)\bshocking truth\b",
        r"(?i)\bterrifying truth\b",
        r"(?i)\bwarning\b",
        r"(?i)\bhidden hack\b",
        r"(?i)\bsecrets? revealed\b",
        r"(?i)\b99% of people don't know\b",
        r"(?i)\bdon't do this\b"
    ]
    
    for pat in fillers:
        text = re.sub(pat, "", text)
        
    # Clean whitespace and excess punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[:\-\s\.\,]+', '', text).strip()
    text = re.sub(r'[:\-\s\.\,]+$', '', text).strip()
    
    return text

# ── FALLBACK DATABASES ──

def _get_fallback_trending_videos():
    return [
        {
            "title": "Google Veo vs OpenAI Sora: The Ultimate AI Video Generator Test",
            "id": "mock_veo_sora",
            "url": "https://gizmodo.com/google-veo-video-generator-details-2026"
        },
        {
            "title": "NVIDIA Blackwell B200 GPU Architecture Deep Dive",
            "id": "mock_nvidia_blackwell",
            "url": "https://blogs.nvidia.com/blog/2024/03/18/blackwell-architecture/"
        },
        {
            "title": "Claude 3.5 Sonnet: The Ultimate Programming AI Assistant",
            "id": "mock_claude_sonnet",
            "url": "https://www.anthropic.com/news/claude-3-5-sonnet"
        },
        {
            "title": "How to Build Your Own Agentic AI Workflow from Scratch",
            "id": "mock_agentic_workflow",
            "url": "https://python.langchain.com/docs/get_started/introduction"
        }
    ]

def _get_fallback_breakout_channels():
    return [
        {
            "name": "TechSecrets",
            "recent_videos": [
                {"title": "Stop Using ChatGPT! Use This Insane AI Hack Instead", "url": "https://techcrunch.com/apps"},
                {"title": "The Hidden iPhone Settings You Must Turn Off Now", "url": "https://wired.com/gear"}
            ]
        },
        {
            "name": "DevInsights",
            "recent_videos": [
                {"title": "Meta Llama 3 400B Open Source AI Changes Coding Forever", "url": "https://venturebeat.com"},
                {"title": "How I Automated My ENTIRE Job with Python AI Agents", "url": "https://lifehacker.com"}
            ]
        }
    ]
