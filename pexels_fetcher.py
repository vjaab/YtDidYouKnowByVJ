"""
pexels_fetcher.py — Per-chunk Pexels video/photo search using Gemini for search terms.

Priority per chunk:
  1. Pexels VIDEO  (portrait, matching duration)
  2. Pexels PHOTO  (portrait, Ken Burns applied by video_gen)
  3. Imagen 4 AI   (9:16, generated)
"""

import os
import io
import json
import time
import threading
import requests
from PIL import Image
from datetime import datetime
from google import genai
from config import GEMINI_API_KEY, OUTPUT_DIR

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
TODAY = datetime.now().strftime("%Y-%m-%d")

_download_lock = threading.Lock()
_used_media = set()          # Track used Pexels IDs to prevent duplicates


# ─────────────────────────────────────────────────────────────────────────────
# STEP A: Ask Gemini for best Pexels search term per chunk
# ─────────────────────────────────────────────────────────────────────────────
def get_search_terms(chunks, topic_context=""):
    """Single Gemini call to assign primary + fallback search term to each chunk."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    chunks_json = json.dumps([
        {"chunk_id": c["chunk_id"], "text": c["text"], "duration": round(c["duration"], 2)}
        for c in chunks
    ], indent=2)

    prompt = f"""You are a visual media selector for a tech news YouTube Shorts video.
Topic context: {topic_context}

For each subtitle chunk below, generate the best Pexels search term to find a relevant video clip or photo.

Rules:
- Use 2-3 concrete, visual words that Pexels will have results for
- Match the spoken content with a real-world visual
- Avoid abstract words: 'shocking', 'incredible', 'amazing'
- Prefer: specific nouns + setting (e.g. 'smartphone screen app' not 'technology')
- Prefer portrait/vertical orientation subjects
- 'pexels_type' should be 'video' for action/motion concepts, 'photo' for still concepts

Chunks:
{chunks_json}

Respond ONLY with a valid JSON array, no markdown:
[
  {{
    "chunk_id": 1,
    "pexels_primary": "two word search",
    "pexels_fallback": "alternative search",
    "pexels_type": "video"
  }}
]"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw = resp.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        terms = json.loads(raw)
        # Map back to chunks
        term_map = {t["chunk_id"]: t for t in terms}
        for c in chunks:
            info = term_map.get(c["chunk_id"], {})
            c["pexels_primary"]  = info.get("pexels_primary", c["text"][:20])
            c["pexels_fallback"] = info.get("pexels_fallback", "technology news")
            c["pexels_type"]     = info.get("pexels_type", "photo")
        print(f"Gemini assigned search terms for {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Gemini search term generation failed: {e}. Using text as search query.")
        for c in chunks:
            c["pexels_primary"]  = " ".join(c["text"].split()[:3])
            c["pexels_fallback"] = "technology innovation"
            c["pexels_type"]     = "photo"
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP B: Download Pexels VIDEO
# ─────────────────────────────────────────────────────────────────────────────
def _search_pexels_video(query, chunk_duration):
    if not PEXELS_API_KEY:
        return None
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": 10, "orientation": "portrait"},
            timeout=15
        )
        if r.status_code != 200:
            return None
        videos = r.json().get("videos", [])
        min_dur = max(1.0, chunk_duration - 1.0)
        max_dur = chunk_duration + 5.0

        for v in videos:
            vid_id = v.get("id")
            if vid_id in _used_media:
                continue
            dur = v.get("duration", 0)
            if min_dur <= dur <= max_dur:
                # Pick best portrait file
                files = v.get("video_files", [])
                portrait_files = [
                    f for f in files
                    if f.get("width", 0) < f.get("height", 1)
                    and f.get("height", 0) >= 720
                ]
                if not portrait_files:
                    # Accept any file if no portrait
                    portrait_files = sorted(files, key=lambda f: f.get("height", 0), reverse=True)
                if portrait_files:
                    chosen = portrait_files[0]
                    return {"id": vid_id, "url": chosen["link"], "duration": dur, "type": "video"}
    except Exception as e:
        print(f"Pexels video search error: {e}")
    return None


def _download_pexels_video(url, output_path):
    try:
        r = requests.get(url, timeout=60, stream=True)
        if r.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            return output_path
    except Exception as e:
        print(f"Pexels video download failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP C: Download Pexels PHOTO
# ─────────────────────────────────────────────────────────────────────────────
def _search_pexels_photo(query):
    if not PEXELS_API_KEY:
        return None
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": 10, "orientation": "portrait"},
            timeout=15
        )
        if r.status_code != 200:
            return None
        photos = r.json().get("photos", [])
        for p in photos:
            pid = p.get("id")
            if pid in _used_media:
                continue
            url = p.get("src", {}).get("large2x") or p.get("src", {}).get("large")
            if url:
                return {"id": pid, "url": url, "type": "photo"}
    except Exception as e:
        print(f"Pexels photo search error: {e}")
    return None


def _download_save_photo(url, output_path):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if r.status_code == 200:
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            # Crop to 9:16
            w, h = img.size
            target_h = int(w * 16 / 9)
            if target_h <= h:
                top = (h - target_h) // 2
                img = img.crop((0, top, w, top + target_h))
            else:
                target_w = int(h * 9 / 16)
                left = (w - target_w) // 2
                img = img.crop((left, 0, left + target_w, h))
            img = img.resize((1080, 1920), Image.LANCZOS)
            img.save(output_path, "JPEG", quality=90)
            return output_path
    except Exception as e:
        print(f"Photo download failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP D: Imagen 4 fallback
# ─────────────────────────────────────────────────────────────────────────────
def _generate_imagen(prompt, output_path):
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    style = ", photorealistic, cinematic, dramatic lighting, 9:16 vertical, no text, no watermarks"
    try:
        result = client.models.generate_images(
            model="imagen-4.0-fast-generate-001",
            prompt=prompt + style,
            config=types.GenerateImagesConfig(
                number_of_images=1, aspect_ratio="9:16", output_mime_type="image/jpeg"
            )
        )
        for gi in result.generated_images:
            with open(output_path, "wb") as f:
                f.write(gi.image.image_bytes)
            return output_path
    except Exception as e:
        print(f"Imagen failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Fetch visual asset for a single chunk
# ─────────────────────────────────────────────────────────────────────────────
def fetch_chunk_visual(chunk):
    """
    Fetches the best visual for a given chunk.
    Adds 'visual_path' and 'visual_type' ('video'|'photo'|'imagen') to chunk dict.
    Returns the updated chunk.
    """
    cid     = chunk["chunk_id"]
    dur     = chunk["duration"]
    primary = chunk.get("pexels_primary", "technology")
    fallback = chunk.get("pexels_fallback", "innovation")
    preferred_type = chunk.get("pexels_type", "photo")

    video_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.mp4")
    photo_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.jpg")

    # ── Priority 1: Pexels VIDEO ──────────────────────────────────────────────
    if PEXELS_API_KEY and preferred_type == "video":
        for q in [primary, fallback]:
            result = _search_pexels_video(q, dur)
            if result:
                path = _download_pexels_video(result["url"], video_out)
                if path:
                    with _download_lock:
                        _used_media.add(result["id"])
                    chunk["visual_path"] = path
                    chunk["visual_type"] = "video"
                    print(f"  Chunk {cid}: Pexels VIDEO ✓ ({q})")
                    return chunk

    # ── Priority 2: Pexels PHOTO ─────────────────────────────────────────────
    if PEXELS_API_KEY:
        for q in [primary, fallback]:
            result = _search_pexels_photo(q)
            if result:
                path = _download_save_photo(result["url"], photo_out)
                if path:
                    with _download_lock:
                        _used_media.add(result["id"])
                    chunk["visual_path"] = path
                    chunk["visual_type"] = "photo"
                    print(f"  Chunk {cid}: Pexels PHOTO ✓ ({q})")
                    return chunk

    # ── Priority 3: Imagen 4 ─────────────────────────────────────────────────
    print(f"  Chunk {cid}: Falling back to Imagen4 ({primary})")
    path = _generate_imagen(primary, photo_out)
    if path:
        chunk["visual_path"] = path
        chunk["visual_type"] = "photo"
        return chunk

    chunk["visual_path"] = None
    chunk["visual_type"] = None
    return chunk


def fetch_all_chunk_visuals(chunks, topic_context=""):
    """
    Main entry point. Calls Gemini once for all search terms,
    then downloads visuals for each chunk in parallel threads.
    """
    print(f"Getting Pexels search terms for {len(chunks)} chunks...")
    chunks = get_search_terms(chunks, topic_context)

    print(f"Downloading chunk visuals ({len(chunks)} chunks)...")
    threads = []
    for chunk in chunks:
        t = threading.Thread(target=fetch_chunk_visual, args=(chunk,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Report summary
    videos = sum(1 for c in chunks if c.get("visual_type") == "video")
    photos = sum(1 for c in chunks if c.get("visual_type") == "photo")
    imagen = sum(1 for c in chunks if c.get("visual_type") == "imagen")
    failed = sum(1 for c in chunks if not c.get("visual_path"))
    print(f"Visuals: {videos} videos, {photos} photos, {imagen} Imagen4, {failed} failed")

    # Fill any failed chunks with the previous chunk's visual
    last_path = None
    last_type = "photo"
    for c in chunks:
        if c.get("visual_path"):
            last_path = c["visual_path"]
            last_type = c["visual_type"]
        elif last_path:
            c["visual_path"] = last_path
            c["visual_type"] = last_type

    return chunks
