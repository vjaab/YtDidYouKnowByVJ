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

def get_local_music_pool():
    """Get all available local music files (cached + manual)."""
    cache = _load_cache()
    cached_tracks = list(cache["tracks"].values())
    
    # Also scan for manually added files
    manual_files = []
    for ext in (".mp3", ".wav", ".m4a"):
        for f in Path(MUSIC_DIR).glob(f"*{ext}"):
            if not f.name.startswith(("pixabay_", "fma_", "jam_", "km_")):
                manual_files.append({
                    "title": f.stem,
                    "local_path": str(f),
                    "source": "manual",
                    "license": "Unknown - verify before use"
                })
    
    return cached_tracks + manual_files

def select_music_for_topic(headline, music_pool=None):
    """
    Deterministically select a music track for a topic using hash.
    Ensures same topic always gets same track.
    """
    if music_pool is None:
        music_pool = get_local_music_pool()
    
    if not music_pool:
        fallback = os.path.join(MUSIC_DIR, "modern_tech.mp3")
        return fallback if os.path.exists(fallback) else None
    
    music_hash = int(hashlib.md5(headline.encode()).hexdigest(), 16)
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