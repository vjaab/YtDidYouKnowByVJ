"""
pexels_fetcher.py — Visual Fetching Decision Tree per chunk, with Gemini Relevance Scoring.
"""

import os
import io
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
_used_media = set()

client = genai.Client(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# RELEVANCE SCORING
# ─────────────────────────────────────────────────────────────────────────────
def score_relevance(chunk_text, visual_desc):
    """
    Calls Gemini to rate relevance 0-10.
    """
    import re
    attempts = 0
    while attempts < 3:
        try:
            target_model = "gemini-2.5-pro" if attempts < 2 else "gemini-2.5-flash"
            prompt = f"""Rate relevance 0-10 between technical text and visual description.
Chunk text: '{chunk_text}'
Visual description: '{visual_desc}'
Return ONLY the raw integer (0-10)."""
            
            response = client.models.generate_content(
                model=target_model, 
                contents=prompt,
                config=genai.types.GenerateContentConfig(temperature=0.0)
            )
            score_text = response.text.strip()
            match = re.search(r'\d+', score_text)
            if match:
                 score = int(match.group())
                 return min(10, max(0, score))
            return 5 # Default if no number found
        except Exception as e:
            wait_time = (2 ** attempts) + 2
            print(f"Gemini scoring failed (att {attempts+1}): {e}. Retrying in {wait_time}s...")
            attempts += 1
            time.sleep(wait_time)
            
    return 5  # Return middle score on total failure


# ─────────────────────────────────────────────────────────────────────────────
# PEXELS SEARCH AND PARSE
# ─────────────────────────────────────────────────────────────────────────────
def _search_pexels_videos(query, chunk_duration):
    if not PEXELS_API_KEY:
        return []
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": 5, "orientation": "portrait"},
            timeout=15
        )
        if r.status_code != 200:
            return []
        
        results = []
        for v in r.json().get("videos", []):
            vid_id = v.get("id")
            if vid_id in _used_media:
                continue
                
            dur = v.get("duration", 0)
            if dur < max(1.0, chunk_duration - 1.0):
                continue
                
            # Extract title/description from URL
            url_slug = v.get("url", "").split("/")[-2] if "pexels.com/video/" in v.get("url", "") else query
            title = url_slug.replace("-", " ")
            
            # Find portrait link
            files = v.get("video_files", [])
            portrait_files = [f for f in files if f.get("width", 0) < f.get("height", 1) and f.get("height", 0) >= 720]
            if not portrait_files:
                portrait_files = sorted(files, key=lambda f: f.get("height", 0), reverse=True)
                
            if portrait_files:
                results.append({
                    "id": vid_id, 
                    "link": portrait_files[0]["link"], 
                    "desc": title, 
                    "type": "video"
                })
        return results
    except Exception as e:
        print(f"Pexels video search error: {e}")
    return []

def _search_pexels_photos(query):
    if not PEXELS_API_KEY:
        return []
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": 5, "orientation": "portrait"},
            timeout=15
        )
        if r.status_code != 200:
            return []
            
        results = []
        for p in r.json().get("photos", []):
            pid = p.get("id")
            if pid in _used_media:
                continue
            
            alt = p.get("alt", query)
            url = p.get("src", {}).get("large2x") or p.get("src", {}).get("large")
            if url:
                results.append({
                    "id": pid,
                    "link": url,
                    "desc": alt,
                    "type": "photo"
                })
        return results
    except Exception as e:
        print(f"Pexels photo search error: {e}")
    return []

def _download_video(url, output_path):
    try:
        r = requests.get(url, timeout=60, stream=True)
        if r.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            return output_path
        return None
    except Exception as e:
        print(f"Video download err: {e}")
        return None

def _download_photo(url, output_path):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if r.status_code == 200:
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
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
        return None
    except Exception as e:
        print(f"Photo download err: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# STEP E: Imagen 3
# ─────────────────────────────────────────────────────────────────────────────
def _generate_imagen3(chunk_text, output_path):
    # 1. Ask Gemini to craft the perfect Imagen prompt
    prompt_builder = f"""Create a detailed Imagen 3 prompt for this text:
'{chunk_text}'
Requirements:
- Photorealistic, cinematic, 9:16 vertical
- No text, no watermarks, no faces of real people
- Directly shows what the text describes
- High detail, dramatic lighting"""
    
    best_prompt = chunk_text # Default
    attempts = 0
    while attempts < 3:
        try:
            target_model = "gemini-2.5-pro" if attempts < 2 else "gemini-2.5-flash"
            resp = client.models.generate_content(model=target_model, contents=prompt_builder)
            best_prompt = resp.text.strip()
            break
        except Exception as e:
            print(f"Imagen prompt gen failed (att {attempts+1}): {e}")
            attempts += 1
            time.sleep(2)
            
    print(f"  -> Generated Imagen prompt: {best_prompt[:60]}...")
        
    attempts = 0
    while attempts < 3:
        try:
            result = client.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt=best_prompt,
                config=genai.types.GenerateImagesConfig(
                    number_of_images=1, aspect_ratio="9:16", output_mime_type="image/jpeg"
                )
            )
            for gi in result.generated_images:
                with open(output_path, "wb") as f:
                    f.write(gi.image.image_bytes)
                return output_path
        except Exception as e:
            wait_time = (2 ** attempts) + 3
            print(f"Imagen3 generation failed (att {attempts+1}): {e}. Retrying in {wait_time}s...")
            attempts += 1
            time.sleep(wait_time)
            
    return None

def fetch_chunk_visual(chunk, script_data):
    """
    Executes the Visual Fetching Decision Tree (A -> B -> C -> D -> E)
    """
    from entity_fetcher import fetch_person_photo, fetch_company_logo
    cid = chunk["chunk_id"]
    text = chunk["text"]
    dur = chunk["duration"]
    
    primary_q = chunk.get("pexels_primary", "technology")
    fallback_q = chunk.get("pexels_fallback", "innovation")
    
    video_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.mp4")
    photo_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.jpg")
    
    # ── STEP A: Determine if chunk mentions a PERSON ─────────────────────
    if chunk.get("has_person"):
        # Match person from script_data
        person_list = script_data.get("people", [])
        if person_list:
            # For simplicity, use first person or matching name
            p = person_list[0] 
            path = fetch_person_photo(p)
            if path:
                chunk["visual_path"] = path
                chunk["visual_type"] = "photo"
                chunk["relevance_score"] = 10
                chunk["source"] = "Step A: Person"
                print(f"Chunk {cid} -> STEP A (Person Match): 10/10")
                return chunk

    # ── STEP B: Determine if chunk mentions a COMPANY ────────────────────
    if chunk.get("has_company"):
        company_list = script_data.get("companies", [])
        if company_list:
            cname = chunk.get("company_name", company_list[0].get("name"))
            c = next((comp for comp in company_list if comp["name"] == cname), company_list[0])
            path = fetch_company_logo(c)
            if path:
                if path.endswith(".png"):
                    # We might need to layer this on a background, but for Layer 1 just use it directly (it'll get ken burns)
                    pass
                chunk["visual_path"] = path
                chunk["visual_type"] = "photo"
                chunk["relevance_score"] = 10
                chunk["source"] = "Step B: Company"
                print(f"Chunk {cid} -> STEP B (Company Match): 10/10")
                return chunk

    # ── STEP C: Nanobanana (Imagen 3) ────────────────────────────────────
    print(f"Chunk {cid} -> STEP C: Nanobanana (Imagen 3) generation")
    path = _generate_imagen3(text, photo_out)
    if path:
        chunk["visual_path"] = path
        chunk["visual_type"] = "photo"
        chunk["relevance_score"] = 10
        chunk["source"] = "Step C: Nanobanana (Imagen 3)"
        print(f"Chunk {cid} -> Nanobanana Image Generated: 10/10")
        return chunk

    # ── STEP D: Pexels VIDEOS (Fallback) ─────────────────────────────────
    for query in [primary_q, fallback_q]:
        videos = _search_pexels_videos(query, dur)
        best_vid = None
        best_score = -1
        
        for v in videos[:3]:  # Limit to top 3 to reduce API calls
            score = score_relevance(text, v["desc"])
            if score >= 5:  # Lowered threshold to avoid excessive retries
                if score > best_score:
                    best_score = score
                    best_vid = v
        
        if best_vid:
            path = _download_video(best_vid["link"], video_out)
            if path:
                with _download_lock:
                    _used_media.add(best_vid["id"])
                chunk["visual_path"] = path
                chunk["visual_type"] = "video"
                chunk["relevance_score"] = best_score
                chunk["source"] = f"Step D: Pexels Video ({query})"
                print(f"Chunk {cid} -> STEP D (Pexels Video): {best_score}/10")
                return chunk

    # ── STEP E: Pexels PHOTOS (Ultimate Fallback) ────────────────────────
    for query in [primary_q, fallback_q]:
        photos = _search_pexels_photos(query)
        best_photo = None
        best_score = -1
        
        for p in photos[:3]:  # Limit to top 3 to reduce API calls
            score = score_relevance(text, p["desc"])
            if score >= 5:  # Lowered threshold
                if score > best_score:
                    best_score = score
                    best_photo = p
                    
        if best_photo:
            path = _download_photo(best_photo["link"], photo_out)
            if path:
                with _download_lock:
                    _used_media.add(best_photo["id"])
                chunk["visual_path"] = path
                chunk["visual_type"] = "photo"
                chunk["relevance_score"] = best_score
                chunk["source"] = f"Step E: Pexels Photo ({query})"
                print(f"Chunk {cid} -> STEP E (Pexels Photo): {best_score}/10")
                return chunk
        
    chunk["visual_path"] = None
    chunk["visual_type"] = None
    chunk["relevance_score"] = 0
    chunk["source"] = "Failed"
    return chunk


def fetch_all_chunk_visuals(chunks, topic_context="", script_data=None):
    if script_data is None:
        script_data = {}
        
    print(f"Running Decision Tree for {len(chunks)} chunks (sequential with delay)...")
    
    for i, chunk in enumerate(chunks):
        # Pexels fallback/primary might not be there if we skipped the older Gemini step
        if "pexels_primary" not in chunk:
            chunk["pexels_primary"] = " ".join(chunk["text"].split()[:3])
            chunk["pexels_fallback"] = "technology"

        print(f"  Processing chunk {i+1}/{len(chunks)}...")
        try:
            fetch_chunk_visual(chunk, script_data)
        except Exception as e:
            print(f"  Chunk {chunk.get('chunk_id')} failed: {e}")
            chunk["visual_path"] = None
            chunk["visual_type"] = None
            chunk["relevance_score"] = 0
            chunk["source"] = "Failed"
        
        # Small delay between chunks to avoid API rate limiting
        if i < len(chunks) - 1:
            time.sleep(1)

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
