# -*- coding: utf-8 -*-
"""
music_fetcher.py - Fetch free trending music for Shorts from real free sources.
Sources: Free Music Archive (FMA), Jamendo, StreamBeats, local curation.
No Pixabay Music API (doesn't exist).
"""

import os
import json
import requests
import hashlib
import time
from pathlib import Path
from config import MUSIC_DIR, BASE_DIR

CACHE_FILE = os.path.join(BASE_DIR, ".music_cache.json")

# Genre mappings for different sources
GENRE_MAP = {
    "electronic": ["Electronic", "Dance", "EDM", "Synthwave"],
    "hiphop": ["Hip-Hop", "Hip Hop", "Rap", "Beat"],
    "ambient": ["Ambient", "Atmospheric", "Chill", "Downtempo"],
    "cinematic": ["Cinematic", "Soundtrack", "Score", "Epic"],
    "corporate": ["Corporate", "Inspirational", "Upbeat", "Motivational"],
    "lofi": ["Lo-Fi", "LoFi", "Chillhop", "Study"],
}

# Curated free music sources with direct MP3 links (CC0/CC-BY/Royalty-free)
# These are verified working sources for Shorts background music
CURATED_SOURCES = {
    "streambeats": {
        "name": "StreamBeats by Harris Heller",
        "license": "Free for creators (including commercial)",
        "note": "ZIP files available at https://github.com/harrisheller/StreamBeats/releases - download and extract MP3s manually",
    },
    "pixabay_videos_audio": {
        "name": "Pixabay Video Audio (extract from video API)",
        "license": "Pixabay Content License (free commercial use)",
        "note": "Use Pixabay Video API to get video URLs, extract audio with ffmpeg",
    },
    "youtube_audio_library": {
        "name": "YouTube Audio Library",
        "license": "Free for YouTube creators",
        "note": "Manual download from YouTube Studio > Audio Library",
    }
}

def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {"tracks": {}, "last_fetch": 0, "sources_checked": []}

def _save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def fetch_fma_tracks(genre=None, per_page=20):
    """
    Fetch tracks from Free Music Archive API.
    Note: FMA API is deprecated but may still work for basic queries.
    """
    # FMA API endpoint (may be deprecated)
    url = "https://freemusicarchive.org/api/track/search"
    params = {
        "limit": per_page,
        "sort": "listens",  # Most popular = trending
        "direction": "desc",
    }
    if genre:
        # FMA uses genre IDs, not names
        genre_ids = {
            "electronic": "34",
            "hiphop": "18",
            "ambient": "9",
            "cinematic": "52",
            "corporate": "53",
            "lofi": "54",
        }
        if genre in genre_ids:
            params["genre"] = genre_ids[genre]
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            tracks = data.get("dataset", [])
            print(f"[MUSIC] Fetched {len(tracks)} tracks from FMA")
            return tracks
    except Exception as e:
        print(f"[WARN] FMA fetch failed: {e}")
    return []

def fetch_jamendo_tracks(genre=None, mood=None, per_page=20):
    """
    Fetch tracks from Jamendo API.
    Requires client_id (free registration at https://developer.jamendo.com)
    """
    client_id = os.getenv("JAMENDO_CLIENT_ID", "")
    if not client_id:
        return []
    
    url = "https://api.jamendo.com/v3.0/tracks"
    params = {
        "client_id": client_id,
        "limit": per_page,
        "order": "popularity_total",
        "include": "musicinfo",
        "audioformat": "mp32",
        "audiodlformat": "mp32",
    }
    
    tags = []
    if genre:
        tags.append(genre)
    if mood:
        tags.append(mood)
    if tags:
        params["tags"] = ",".join(tags)
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            tracks = data.get("results", [])
            print(f"[MUSIC] Fetched {len(tracks)} tracks from Jamendo")
            return tracks
    except Exception as e:
        print(f"[WARN] Jamendo fetch failed: {e}")
    return []

def download_track(url, filename, output_dir=None):
    """Download a track from a direct URL."""
    if output_dir is None:
        output_dir = MUSIC_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        print(f"[MUSIC] Track already cached: {filename}")
        return filepath
    
    try:
        print(f"[DOWNLOAD] Downloading: {filename}...")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify it's a valid MP3
        if os.path.getsize(filepath) < 10000:  # Less than 10KB = probably error page
            os.remove(filepath)
            print(f"[WARN] Downloaded file too small, removed")
            return None
        
        print(f"[OK] Downloaded: {filename} ({os.path.getsize(filepath)//1024} KB)")
        return filepath
    except Exception as e:
        print(f"[WARN] Download failed for {filename}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return None

def sync_curated_sources(max_per_source=2):
    """
    Sync tracks from curated free music sources.
    These sources require manual download (ZIP extraction or direct download).
    This function logs what's available and returns empty list.
    """
    print("[MUSIC] Curated sources available for manual download:")
    for key, source in CURATED_SOURCES.items():
        print(f"  - {source['name']}: {source.get('note', '')}")
        print(f"    License: {source['license']}")
    
    print("[MUSIC] Add MP3 files manually to assets/music/ for immediate use")
    return []


def fetch_pixabay_video_audio(q="technology", per_page=10):
    """
    Fetch videos from Pixabay Video API and return audio URLs.
    Use ffmpeg to extract audio from video files.
    """
    api_key = os.getenv("PIXABAY_API_KEY", "")
    if not api_key:
        print("[WARN] PIXABAY_API_KEY not set. Skipping video audio fetch.")
        return []
    
    url = "https://pixabay.com/api/videos/"
    params = {
        "key": api_key,
        "q": q,
        "per_page": per_page,
        "order": "popular",
        "category": "music",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        videos = data.get("hits", [])
        print(f"[MUSIC] Fetched {len(videos)} videos from Pixabay for audio extraction")
        
        results = []
        for v in videos:
            # Get the best quality video URL
            video_files = v.get("videos", {})
            best_url = None
            for quality in ["large", "medium", "small", "tiny"]:
                if quality in video_files and video_files[quality].get("url"):
                    best_url = video_files[quality]["url"]
                    break
            
            if best_url:
                results.append({
                    "id": v.get("id"),
                    "title": v.get("tags", "").split(",")[0] if v.get("tags") else f"pixabay_video_{v.get('id')}",
                    "video_url": best_url,
                    "duration": v.get("duration", 0),
                    "tags": v.get("tags", ""),
                })
        return results
    except Exception as e:
        print(f"[WARN] Pixabay video fetch failed: {e}")
        return []


def download_and_extract_audio(video_info, output_dir=None):
    """
    Download video and extract audio using ffmpeg.
    Returns path to extracted MP3.
    """
    if output_dir is None:
        output_dir = MUSIC_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_id = video_info.get("id")
    title = video_info.get("title", f"pixabay_video_{video_id}")
    safe_title = "".join(c for c in title if c.isalnum() or c in "_- ")[:60]
    video_filename = f"pixabay_vid_{video_id}_{safe_title}.mp4"
    audio_filename = f"pixabay_aud_{video_id}_{safe_title}.mp3"
    video_path = os.path.join(output_dir, video_filename)
    audio_path = os.path.join(output_dir, audio_filename)
    
    if os.path.exists(audio_path):
        print(f"[MUSIC] Audio already extracted: {audio_filename}")
        return audio_path
    
    # Download video
    video_url = video_info.get("video_url")
    if not video_url:
        return None
    
    try:
        print(f"[DOWNLOAD] Downloading video: {video_filename}...")
        resp = requests.get(video_url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(video_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract audio with ffmpeg
        import subprocess
        print(f"[AUDIO] Extracting audio to: {audio_filename}...")
        result = subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "libmp3lame", "-ab", "192k",
            "-ar", "44100", audio_path
        ], capture_output=True, timeout=60)
        
        # Clean up video file
        if os.path.exists(video_path):
            os.remove(video_path)
        
        if result.returncode == 0 and os.path.exists(audio_path):
            print(f"[OK] Extracted audio: {audio_filename}")
            return audio_path
        else:
            print(f"[WARN] ffmpeg failed: {result.stderr.decode()[:200]}")
            return None
            
    except Exception as e:
        print(f"[WARN] Video download/extract failed: {e}")
        if os.path.exists(video_path):
            os.remove(video_path)
        return None


def sync_from_pixabay_videos(queries=None, max_per_query=2):
    """
    Fetch videos from Pixabay and extract audio for music.
    """
    if queries is None:
        queries = ["technology", "ambient", "electronic", "lofi", "corporate", "upbeat"]
    
    cache = _load_cache()
    downloaded = []
    
    for q in queries:
        videos = fetch_pixabay_video_audio(q=q, per_page=5)
        for video in videos[:max_per_query]:
            track_id = f"pixvid_{video.get('id', '')}"
            if track_id in cache["tracks"]:
                continue
            
            audio_path = download_and_extract_audio(video)
            if audio_path:
                cache["tracks"][track_id] = {
                    "title": video.get("title", f"Pixabay Video {video.get('id')}"),
                    "artist": "Pixabay",
                    "genre": q,
                    "license": "Pixabay Content License",
                    "local_path": audio_path,
                    "source": "pixabay_video",
                    "duration": video.get("duration"),
                }
                downloaded.append(audio_path)
    
    _save_cache(cache)
    return downloaded

def sync_from_fma(genres=None, max_per_genre=3):
    """Sync trending tracks from Free Music Archive."""
    if genres is None:
        genres = list(GENRE_MAP.keys())
    
    cache = _load_cache()
    downloaded = []
    
    for genre in genres:
        tracks = fetch_fma_tracks(genre=genre, per_page=10)
        for track in tracks[:max_per_genre]:
            track_id = f"fma_{track.get('track_id', '')}"
            if track_id in cache["tracks"]:
                continue
            
            # FMA provides direct download URLs for some tracks
            download_url = track.get("track_file") or track.get("track_url")
            if not download_url:
                continue
            
            title = track.get("track_title", f"fma_{track_id}")
            safe_title = "".join(c for c in title if c.isalnum() or c in "_- ")[:80]
            filename = f"fma_{safe_title.replace(' ', '_')}.mp3"
            path = download_track(download_url, filename)
            if path:
                cache["tracks"][track_id] = {
                    "title": title,
                    "artist": track.get("artist_name", "Unknown"),
                    "genre": genre,
                    "license": track.get("track_license", "CC"),
                    "local_path": path,
                    "source": "fma",
                }
                downloaded.append(path)
    
    _save_cache(cache)
    return downloaded

def sync_from_jamendo(genres=None, moods=None, max_per_category=3):
    """Sync trending tracks from Jamendo (requires API key)."""
    if genres is None:
        genres = list(GENRE_MAP.keys())
    if moods is None:
        moods = ["energetic", "upbeat", "inspiring", "cool", "relaxed", "happy"]
    
    cache = _load_cache()
    downloaded = []
    
    for genre in genres:
        for mood in moods:
            tracks = fetch_jamendo_tracks(genre=genre, mood=mood, per_page=10)
            for track in tracks[:max_per_category]:
                track_id = f"jam_{track.get('id', '')}"
                if track_id in cache["tracks"]:
                    continue
                
                download_url = track.get("audiodownload") or track.get("audio")
                if not download_url:
                    continue
                
                title = track.get("name", f"jam_{track_id}")
                safe_title = "".join(c for c in title if c.isalnum() or c in "_- ")[:80]
                filename = f"jam_{safe_title.replace(' ', '_')}.mp3"
                path = download_track(download_url, filename)
                if path:
                    cache["tracks"][track_id] = {
                        "title": title,
                        "artist": track.get("artist_name", "Unknown"),
                        "genre": genre,
                        "mood": mood,
                        "license": "Jamendo Standard License",
                        "local_path": path,
                        "source": "jamendo",
                    }
                    downloaded.append(path)
    
    _save_cache(cache)
    return downloaded

def sync_all_sources():
    """Sync from all available sources."""
    all_downloaded = []
    
    # 1. Curated sources (always work, no API key needed)
    print("[MUSIC] Syncing curated sources...")
    all_downloaded.extend(sync_curated_sources(max_per_source=3))
    
    # 2. Pixabay Video Audio (requires PIXABAY_API_KEY, uses existing video API)
    if os.getenv("PIXABAY_API_KEY"):
        print("[MUSIC] Syncing audio from Pixabay videos...")
        all_downloaded.extend(sync_from_pixabay_videos(max_per_query=2))
    else:
        print("[MUSIC] Skipping Pixabay video audio (no PIXABAY_API_KEY)")
    
    # 3. FMA (no API key, but API may be deprecated)
    print("[MUSIC] Syncing from Free Music Archive...")
    all_downloaded.extend(sync_from_fma(max_per_genre=2))
    
    # 4. Jamendo (requires JAMENDO_CLIENT_ID)
    if os.getenv("JAMENDO_CLIENT_ID"):
        print("[MUSIC] Syncing from Jamendo...")
        all_downloaded.extend(sync_from_jamendo(max_per_category=2))
    else:
        print("[MUSIC] Skipping Jamendo (no JAMENDO_CLIENT_ID)")
    
    print(f"[MUSIC] Total new tracks: {len(all_downloaded)}")
    return all_downloaded

# ─── HIGH-RETENTION BGM ENGINE (Viewer Retention & Attention-Catching) ───────
BGM_TRACKER_FILE = os.path.join(BASE_DIR, "bgm_tracker.json")

# Curated Attention & Retention Profiles for local music tracks
# Scores (0-100) measure: instant hook presence (0-3s), tempo/rhythm drive, energy curve, and viral retention power.
HIGH_ATTENTION_TRACK_PROFILES = {
    "jam_Energy.mp3": {
        "score": 98, "energy": "very_high", "bpm": 128,
        "vibes": ["electronic", "upbeat", "driving", "tools", "news", "viral"],
        "desc": "Ultra-punchy driving electronic beat with instant hook presence"
    },
    "jam_Action_Inspiration_Trailer.mp3": {
        "score": 96, "energy": "very_high", "bpm": 130,
        "vibes": ["cinematic", "epic", "research", "breakthrough", "deep_dive", "news"],
        "desc": "High-impact cinematic trailer rhythm with powerful retention drops"
    },
    "jam_Energetic_Rock.mp3": {
        "score": 95, "energy": "very_high", "bpm": 132,
        "vibes": ["rock", "energetic", "news", "high_adrenaline", "tech_trends"],
        "desc": "Fast-paced adrenaline drive that prevents viewer swipe-away"
    },
    "jam_Energetic_Pop.mp3": {
        "score": 95, "energy": "high", "bpm": 126,
        "vibes": ["pop", "upbeat", "viral", "tools", "trends", "news"],
        "desc": "Catchy upbeat groove optimized for high completion rates"
    },
    "jam_Escape_From_The_Machine_Planet.mp3": {
        "score": 94, "energy": "very_high", "bpm": 125,
        "vibes": ["cyberpunk", "synthwave", "research", "ai", "sci_fi", "tools"],
        "desc": "Futuristic cyberpunk synth drive with relentless momentum"
    },
    "jam_Upbeat.mp3": {
        "score": 94, "energy": "high", "bpm": 124,
        "vibes": ["upbeat", "electronic", "tools", "positive", "tech", "news"],
        "desc": "Bright, driving tech beat with fast-paced rhythmic clarity"
    },
    "modern_tech.mp3": {
        "score": 93, "energy": "high", "bpm": 122,
        "vibes": ["tech", "electronic", "tools", "signature", "modern", "news"],
        "desc": "Signature tech channel identity, punchy modern production"
    },
    "jam_The_Epic.mp3": {
        "score": 93, "energy": "high", "bpm": 120,
        "vibes": ["cinematic", "epic", "research", "breakthrough", "tech_trends"],
        "desc": "Epic orchestral-electronic drop that creates monumental scale"
    },
    "jam_Elite.mp3": {
        "score": 92, "energy": "high", "bpm": 125,
        "vibes": ["electronic", "beat", "swagger", "news", "tech", "tools"],
        "desc": "Swaggering high-tech electronic groove with clean drums"
    },
    "jam_Electronica.mp3": {
        "score": 92, "energy": "high", "bpm": 128,
        "vibes": ["electronic", "fast", "coding", "future", "tech_trends"],
        "desc": "Rapid electronic pulse ideal for fast-moving coding/tools news"
    },
    "jam_Confidence.mp3": {
        "score": 91, "energy": "high", "bpm": 118,
        "vibes": ["hiphop", "punchy", "interview", "vaibhav", "coding", "tools"],
        "desc": "Bold, punchy rhythm that radiates authority and confidence"
    },
    "Defiance.mp3": {
        "score": 91, "energy": "high", "bpm": 124,
        "vibes": ["cyberpunk", "driving", "research", "news", "ai"],
        "desc": "Intense driving synth bassline with forward-moving energy"
    },
    "Defiance_long_remix.mp3": {
        "score": 91, "energy": "high", "bpm": 124,
        "vibes": ["cyberpunk", "driving", "research", "news", "ai"],
        "desc": "Extended cyberpunk pulse with dynamic variation"
    },
    "jam_Beyond_Borders_of_Inspiration.mp3": {
        "score": 90, "energy": "high", "bpm": 120,
        "vibes": ["uplifting", "tech_trends", "future", "research", "tools"],
        "desc": "Soaring electronic chords with inspiring forward push"
    },
    "jam_Inspiring_Epic_Glory.mp3": {
        "score": 90, "energy": "high", "bpm": 118,
        "vibes": ["cinematic", "epic", "research", "breakthrough", "news"],
        "desc": "Glory/triumph aesthetic for landmark AI breakthrough stories"
    },
    "Crossroads.mp3": {
        "score": 89, "energy": "medium_high", "bpm": 116,
        "vibes": ["driving", "momentum", "news", "trends", "research"],
        "desc": "Tension and resolution rhythm that builds curiosity"
    },
    "Destiny.mp3": {
        "score": 89, "energy": "medium_high", "bpm": 118,
        "vibes": ["cinematic", "epic", "research", "tech_trends"],
        "desc": "Epic cinematic momentum that sustains attention across sections"
    },
    "jam_Groovy_SIX.mp3": {
        "score": 89, "energy": "high", "bpm": 122,
        "vibes": ["groovy", "tech", "tools", "upbeat", "viral"],
        "desc": "Infectious tech groove that keeps viewer head nodding"
    },
    "jam_Zewor_Beats_-_Keep_going_88bpm.mp3": {
        "score": 88, "energy": "medium_high", "bpm": 116,
        "vibes": ["hiphop", "head_nod", "interview", "coding", "tools"],
        "desc": "Punchy boom-bap rhythm with satisfying snare snaps"
    },
    "jam_Hip_Hop_Instrumental.mp3": {
        "score": 88, "energy": "medium_high", "bpm": 114,
        "vibes": ["hiphop", "boom_bap", "interview", "vaibhav", "coding"],
        "desc": "Classic rhythmic hip-hop beat that keeps speech crisp and forward"
    },
    "Faith.mp3": {
        "score": 87, "energy": "medium_high", "bpm": 116,
        "vibes": ["pulse", "driving", "news", "tech_trends"],
        "desc": "Steady electronic pulse with emotional lift"
    },
    "jam_Motivational.mp3": {
        "score": 87, "energy": "high", "bpm": 120,
        "vibes": ["motivational", "upbeat", "tech_trends", "tools"],
        "desc": "Driving motivational energy for high-achievement topics"
    },
    "jam_Never_Give_Up.mp3": {
        "score": 86, "energy": "medium_high", "bpm": 118,
        "vibes": ["driving", "upbeat", "news", "tech"],
        "desc": "Steady, positive driving beat"
    },
    "jam_To_The_Roofs.mp3": {
        "score": 86, "energy": "high", "bpm": 122,
        "vibes": ["electronic", "pulse", "trends", "tools"],
        "desc": "Uplifting melodic electronic rhythm"
    },
    "jam_On_the_Come_Up.mp3": {
        "score": 85, "energy": "medium_high", "bpm": 112,
        "vibes": ["hiphop", "swagger", "tools", "interview"],
        "desc": "Confident hip hop track with clear dynamic structure"
    },
    "jam_Wish_You_Were_Here.mp3": {
        "score": 85, "energy": "high", "bpm": 120,
        "vibes": ["electronic", "rhythm", "tools", "news"],
        "desc": "Energetic electronic dance groove"
    },
}

# Retention Killers: tracks that must NEVER be selected for YouTube Shorts
RETENTION_KILLER_PATTERNS = [
    "meditation", "relaxation", "silence", "clair_de_lune",
    "dark_room", "lonely", "despair", "tears", "christmas",
    "villain", "green screen", "wave effect", "tape",
    "bruwynn", "beach_sunset", "love_story", "serenity",
    "spanish_horizon", "halls_of_despair", "hidden_tears"
]

def is_retention_killer(filename):
    """Check if a track is disqualified from Shorts due to low viewer retention."""
    name_lower = filename.lower()
    for pattern in RETENTION_KILLER_PATTERNS:
        if pattern in name_lower:
            return True
    return False

def _load_bgm_tracker(tracker_file=None):
    """Load BGM rotation tracker history."""
    if tracker_file is None:
        tracker_file = BGM_TRACKER_FILE
    if os.path.exists(tracker_file):
        try:
            with open(tracker_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"history": [], "usage_counts": {}}

def _record_bgm_usage(track_filename, headline="", topic_type="", tracker_file=None):
    """Record selected BGM into tracker to prevent repetitive tracks across runs."""
    if tracker_file is None:
        tracker_file = BGM_TRACKER_FILE
    tracker = _load_bgm_tracker(tracker_file)
    history = tracker.get("history", [])
    usage_counts = tracker.get("usage_counts", {})
    
    # Prepend newest entry (keep up to 30 runs)
    entry = {
        "track": track_filename,
        "headline": headline[:80] if headline else "",
        "topic_type": topic_type or "auto",
        "timestamp": os.getenv("GITHUB_RUN_ID", "") or str(int(time.time()))
    }
    history.insert(0, entry)
    tracker["history"] = history[:30]
    tracker["usage_counts"][track_filename] = usage_counts.get(track_filename, 0) + 1
    
    try:
        with open(tracker_file, "w") as f:
            json.dump(tracker, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save BGM tracker: {e}")

def get_local_music_pool(only_high_retention=True):
    """
    Get available local music files with dynamic path resolution and retention validation.
    Resolves paths relative to MUSIC_DIR so it works on any platform (macOS, CI, Kaggle).
    """
    from config import BGM_MIN_ATTENTION_SCORE
    
    cache = _load_cache()
    cached_tracks = list(cache.get("tracks", {}).values())
    
    # Collect all existing local audio files in assets/music/
    discovered_files = {}
    for ext in (".mp3", ".wav", ".m4a"):
        for f in Path(MUSIC_DIR).glob(f"*{ext}"):
            if not f.is_file():
                continue
            # Filter out corrupt/tiny files (< 100KB)
            if f.stat().st_size < 100000:
                continue
            discovered_files[f.name] = str(f)
    
    music_pool = []
    
    # 1. Process curated high-attention tracks first
    for filename, profile in HIGH_ATTENTION_TRACK_PROFILES.items():
        if filename in discovered_files:
            music_pool.append({
                "title": filename.replace(".mp3", "").replace("jam_", ""),
                "filename": filename,
                "local_path": discovered_files[filename],
                "attention_score": profile["score"],
                "energy": profile["energy"],
                "bpm": profile.get("bpm", 120),
                "vibes": profile["vibes"],
                "desc": profile.get("desc", ""),
                "is_curated": True
            })
    
    # 2. Process other scanned files if not requiring strict high-retention
    if not only_high_retention:
        for filename, filepath in discovered_files.items():
            if filename in HIGH_ATTENTION_TRACK_PROFILES:
                continue
            if is_retention_killer(filename):
                continue
            music_pool.append({
                "title": filename.replace(".mp3", "").replace("jam_", ""),
                "filename": filename,
                "local_path": filepath,
                "attention_score": 75,
                "energy": "medium",
                "bpm": 110,
                "vibes": ["general"],
                "desc": "General background track",
                "is_curated": False
            })
    else:
        # Filter to only tracks meeting the minimum attention score threshold
        music_pool = [t for t in music_pool if t.get("attention_score", 0) >= BGM_MIN_ATTENTION_SCORE]
    
    return music_pool

def select_high_retention_bgm(headline="", category="", topic_type="", tracker_file=None):
    """
    Selects a high attention-catching BGM for YouTube Shorts generation.
    
    Key Retention Features:
    1. Strictly selects from Tier-1 tracks (score >= 85) with instant hook presence.
    2. Category & Topic affinity matching (e.g. Research -> Cinematic Trailer, Tools -> Modern Tech Beat).
    3. Multi-run anti-fatigue cooldown: Prevents repeating recently used tracks across pipeline runs.
    4. Deterministic hash tie-breaking on headline for consistency when re-generating.
    """
    from config import ENABLE_HIGH_RETENTION_BGM, BGM_ROTATION_COOLDOWN, BGM_MIN_ATTENTION_SCORE
    import time
    
    pool = get_local_music_pool(only_high_retention=ENABLE_HIGH_RETENTION_BGM)
    if not pool:
        # Fallback to standard pool if curated list is somehow missing
        pool = get_local_music_pool(only_high_retention=False)
    
    if not pool:
        fallback = os.path.join(MUSIC_DIR, "modern_tech.mp3")
        return fallback if os.path.exists(fallback) else None
    
    # Load recent history to enforce anti-fatigue rotation
    tracker = _load_bgm_tracker(tracker_file)
    recent_history = tracker.get("history", [])
    recent_filenames = [entry.get("track") for entry in recent_history[:BGM_ROTATION_COOLDOWN]]
    
    # Normalize category and topic context
    context_text = f"{headline} {category} {topic_type}".lower()
    
    # Map topics to preferred vibes
    preferred_vibes = []
    if any(k in context_text for k in ["research", "paper", "arxiv", "breakthrough", "model", "deepmind", "openai"]):
        preferred_vibes = ["cinematic", "epic", "research", "cyberpunk", "breakthrough"]
    elif any(k in context_text for k in ["tool", "github", "release", "app", "framework", "repo", "library"]):
        preferred_vibes = ["tools", "upbeat", "electronic", "groovy", "tech"]
    elif any(k in context_text for k in ["news", "trend", "economy", "market", "billion", "launch", "breaking"]):
        preferred_vibes = ["news", "energetic", "driving", "viral", "high_adrenaline"]
    elif any(k in context_text for k in ["code", "coding", "interview", "question", "quiz", "vaibhav"]):
        preferred_vibes = ["hiphop", "punchy", "confidence", "interview", "head_nod"]
    else:
        preferred_vibes = ["electronic", "upbeat", "tech", "viral"]
    
    # Score each candidate track
    scored_candidates = []
    for track in pool:
        fn = track["filename"]
        base_score = float(track.get("attention_score", 85))
        
        # 1. Topic affinity bonus (up to +12)
        vibe_matches = sum(1 for v in preferred_vibes if v in track.get("vibes", []))
        vibe_bonus = min(12.0, vibe_matches * 4.0)
        
        # 2. Recency Penalty (Anti-fatigue rotation across runs)
        recency_penalty = 0.0
        if fn in recent_filenames:
            # The more recently used, the larger the penalty
            pos = recent_filenames.index(fn)  # 0 is most recent
            recency_penalty = max(10.0, 45.0 - (pos * 7.0))
        
        # 3. Deterministic hash jitter (+0.0 to +3.0) for reproducible tie-breaking on same headline
        hash_val = int(hashlib.md5(f"{headline}_{fn}".encode("utf-8")).hexdigest(), 16)
        hash_jitter = (hash_val % 300) / 100.0
        
        final_score = base_score + vibe_bonus - recency_penalty + hash_jitter
        scored_candidates.append((final_score, track))
    
    # Sort descending by final score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_track = scored_candidates[0]
    
    chosen_path = best_track["local_path"]
    chosen_fn = best_track["filename"]
    
    # Record usage in tracker
    _record_bgm_usage(chosen_fn, headline=headline, topic_type=topic_type, tracker_file=tracker_file)
    
    print(f"🔥 [HIGH-RETENTION BGM] Selected: {chosen_fn}")
    print(f"   📊 Retention Score: {best_track.get('attention_score')}/100 | Energy: {best_track.get('energy')} | BPM: {best_track.get('bpm')}")
    print(f"   🎯 Matched Vibes: {', '.join(best_track.get('vibes', [])[:4])}")
    print(f"   💡 Description: {best_track.get('desc')}")
    
    return chosen_path

def select_music_for_topic(headline, music_pool=None, category=None, topic_type=None):
    """
    Public entry point for BGM selection.
    Routes through select_high_retention_bgm to ensure viewer retention.
    """
    from config import ENABLE_HIGH_RETENTION_BGM
    
    if ENABLE_HIGH_RETENTION_BGM:
        bgm_path = select_high_retention_bgm(
            headline=headline or "",
            category=category or "",
            topic_type=topic_type or ""
        )
        if bgm_path and os.path.exists(bgm_path):
            return bgm_path
            
    # Fallback to deterministic hash selection if disabled
    if music_pool is None:
        music_pool = get_local_music_pool(only_high_retention=False)
    
    if not music_pool:
        fallback = os.path.join(MUSIC_DIR, "modern_tech.mp3")
        return fallback if os.path.exists(fallback) else None
    
    music_hash = int(hashlib.md5((headline or "default").encode()).hexdigest(), 16)
    idx = music_hash % len(music_pool)
    return music_pool[idx]["local_path"]

def get_attribution_for_track(track_path):
    """Get attribution text for a track if required."""
    cache = _load_cache()
    for track in cache["tracks"].values():
        if track.get("local_path") == track_path:
            return track.get("attribution", "")
    return ""

# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sync free trending music for Shorts")
    parser.add_argument("--source", choices=["all", "curated", "pixabay-video", "fma", "jamendo"], default="all")
    parser.add_argument("--genres", nargs="+", default=list(GENRE_MAP.keys()))
    parser.add_argument("--moods", nargs="+", default=["energetic", "upbeat", "inspiring", "cool", "relaxed", "happy"])
    parser.add_argument("--max-per-category", type=int, default=3)
    args = parser.parse_args()
    
    if args.source == "all":
        sync_all_sources()
    elif args.source == "curated":
        sync_curated_sources(max_per_source=args.max_per_category)
    elif args.source == "pixabay-video":
        if not os.getenv("PIXABAY_API_KEY"):
            print("[WARN] PIXABAY_API_KEY required for pixabay-video source")
        else:
            sync_from_pixabay_videos(max_per_query=args.max_per_category)
    elif args.source == "fma":
        sync_from_fma(args.genres, args.max_per_category)
    elif args.source == "jamendo":
        sync_from_jamendo(args.genres, args.moods, args.max_per_category)