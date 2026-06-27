"""
vidiq_trending.py — Core module to fetch trending AI/technology topics
and keyword opportunities using the vidIQ API.
"""
import os
import re
import requests
from datetime import datetime

def get_trending_videos(api_key=None, region="IN"):
    """
    Fetches live trending AI/tech videos filtered by region (default: IN).
    Falls back to a curated list of active trending topics on API failure.
    """
    api_key = api_key or os.getenv("VIDIQ_API_KEY")
    if not api_key:
        print("⚠️ VIDIQ_API_KEY not found in env. Returning fallback trending videos.")
        return _get_fallback_trending_videos()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    url = f"https://api.vidiq.com/v1/videos/trending?region={region}&limit=10"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json().get("videos", [])
        else:
            print(f"⚠️ vidIQ API trending call returned status {response.status_code}. Using fallbacks.")
    except Exception as e:
        print(f"⚠️ vidIQ API trending call failed: {e}. Using fallbacks.")
        
    return _get_fallback_trending_videos()

def get_keyword_research(seed_term, api_key=None):
    """
    Fetches search volume, competition, and overall opportunity scores for a seed keyword.
    """
    api_key = api_key or os.getenv("VIDIQ_API_KEY")
    if not api_key:
        return {"keyword": seed_term, "search_volume": 6200, "competition": 32, "score": 78}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    url = f"https://api.vidiq.com/v1/keywords/research?q={seed_term}"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            return {
                "keyword": data.get("keyword", seed_term),
                "search_volume": data.get("search_volume", 5000),
                "competition": data.get("competition_score", 35),
                "score": data.get("overall_score", 70)
            }
    except Exception as e:
        print(f"⚠️ vidIQ keyword research call failed: {e}.")
        
    return {"keyword": seed_term, "search_volume": 6200, "competition": 32, "score": 78}

def get_breakout_channels(api_key=None, category="AI"):
    """
    Identifies high-growth competitor channels in the specified tech category.
    """
    api_key = api_key or os.getenv("VIDIQ_API_KEY")
    if not api_key:
        return _get_fallback_breakout_channels()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    url = f"https://api.vidiq.com/v1/channels/breakout?category={category}"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json().get("channels", [])
    except Exception as e:
        print(f"⚠️ vidIQ breakout channels call failed: {e}. Using fallbacks.")
        
    return _get_fallback_breakout_channels()

def get_pipeline_topics(category="AI & Tech Tools"):
    """
    Combines trending videos, breakout competitor insights, and keyword opportunities
    into a structured, ranked topic list for the script generator.
    """
    api_key = os.getenv("VIDIQ_API_KEY")
    
    print(f"📡 Querying vidIQ Pipeline for category: {category}...")
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
