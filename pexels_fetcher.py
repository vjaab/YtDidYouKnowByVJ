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
import random

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
            # Use 2.0-flash for high speed and stability
            target_model = "gemini-2.0-flash" 
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
            return 5
        except Exception as e:
            err_str = str(e).lower()
            # If rate limited, wait longer (60s) for the minute to reset
            if "429" in err_str or "resource_exhausted" in err_str:
                wait_time = 60
                print(f"⚠️ Gemini Rate Limit Hit (429). Waiting {wait_time}s...")
            else:
                wait_time = (2 ** attempts) + 5
                print(f"Gemini scoring failed (att {attempts+1}): {e}. Retrying in {wait_time}s...")
            
            attempts += 1
            time.sleep(wait_time)
            
    return 5  # Return middle score on total failure


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL VISUAL CONTINUITY
# ─────────────────────────────────────────────────────────────────────────────
def generate_visual_style_guide(headline):
    """
    Asks Gemini to define a consistent visual 'vibe' for the whole video.
    This ensures all AI-generated images share a palette and lighting style.
    """
    print("🎨 Designing Global Visual Style Guide...")
    try:
        target_model = "gemini-2.0-flash"
        prompt = f"""Based on this news headline: '{headline}', define a cohesive visual style for a 9:16 vertical cinematic video.
Return a short string (max 40 words) describing the lighting, color palette, and camera style.
Example Output: 'Dark tech-noir aesthetic, cyan and deep purple neon lighting, shot on 35mm lens, high contrast, anamorphic lens flares, unreal engine 5 style.'
Return ONLY the description."""
        
        response = client.models.generate_content(
            model=target_model, 
            contents=prompt,
            config=genai.types.GenerateContentConfig(temperature=0.7)
        )
        style = response.text.strip()
        print(f"   ✨ Global Vibe: {style}")
        return style
    except Exception as e:
        print(f"   ⚠️ Style guide failed: {e}. Using default.")
        return "Cinematic lighting, professional photography, high detail, photorealistic, 8k."


# ─────────────────────────────────────────────────────────────────────────────
# TOPIC DETECTION & IMAGEN TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
TOPIC_KEYWORDS = {
    "ai_company": ["openai", "anthropic", "google", "meta", "microsoft", "nvidia", "apple", "amazon", "startup", "funding", "ipo", "acquisition"],
    "semiconductor": ["chip", "gpu", "tpu", "semiconductor", "tsmc", "intel", "amd", "qualcomm", "arm", "wafer", "foundry"],
    "robotics": ["robot", "humanoid", "boston dynamics", "figure", "tesla bot", "automation", "warehouse"],
    "neural_network": ["neural", "llm", "model", "gpt", "claude", "gemini", "training", "parameters", "transformer"],
    "data_center": ["data center", "server", "infrastructure", "cloud", "cooling", "energy", "power"],
    "autonomous_vehicle": ["self-driving", "autonomous", "tesla", "waymo", "cruise", "lidar", "ev"],
    "ai_policy": ["regulation", "law", "ban", "congress", "eu ai act", "government", "policy", "safety"],
    "consumer_tech": ["smartphone", "wearable", "assistant", "alexa", "siri", "device", "product launch"]
}

TOPIC_PROMPT_TEMPLATES = {
    "ai_company": [
        "Futuristic {name} headquarters, glowing logo on glass skyscraper, night cityscape, cinematic",
        "Silicon Valley style modern office interior for {name} with neural network digital art, bright and airy, minimalist",
        "A sleek, minimalist {name} logo glowing on a black polished marble surface, luxury tech aesthetic, cinematic"
    ],
    "semiconductor": [
        "Extreme close-up of semiconductor chip, microscopic glowing circuit traces, dark background, dramatic studio lighting",
        "A silicon wafer being processed in a high-tech cleanroom, orange neon accents, reflection on surface",
        "Advanced microchip architecture visualization, complex 3D nanostructures, glowing electricity flowing through paths"
    ],
    "robotics": [
        "Humanoid robot in modern factory, white silver design, blue ambient lighting, cinematic",
        "Close-up on a robotic eye with glowing blue sensor, high precision mechanical parts, metallic finish",
        "A dedicated robot arm working on a circuit board, sparks flying, high contrast, industrial cyberpunk style"
    ],
    "neural_network": [
        "Abstract neural network visualization, glowing nodes and connections, deep blue purple palette",
        "A digital brain composed of light particles and binary code, cosmic background, high energy",
        "Complex web of synaptic connections glowing in the dark, representing artificial intelligence, ethereal glow"
    ],
    "data_center": [
        "Massive AI data center, glowing server racks, cool blue lighting, foggy atmosphere, wide shot",
        "Inside a server room with endless rows of blinking lights, symmetrical perspective, futuristic cloud infrastructure",
        "A digital rendering of a global network hub connecting to a data center, glowing fiber optics, dark obsidian palette"
    ],
    "autonomous_vehicle": [
        "Self-driving car on futuristic highway at night, sensor beams, neon city reflections",
        "A sleek autonomous electric vehicle interior, no steering wheel, holographic dashboard display, luxury",
        "Close-up on a LiDAR sensor unit on top of a car, emitting purple laser beams into the surrounding environment"
    ],
    "ai_policy": [
        "Government building with holographic AI symbols, dramatic political atmosphere, editorial style",
        "A gavel resting on a digital circuit board, representing AI regulation and law, high contrast lighting",
        "A futuristic holographic bill of rights or legal document being reviewed by AI, serious tone, blue and gold"
    ],
    "consumer_tech": [
        "Person using AI holographic smartphone interface, modern home, soft natural lighting",
        "A wearable AI device on a person's wrist or ear, glowing softly, high-end lifestyle photography",
        "A minimalist AI assistant speaker on a marble table, emitting a subtle blue pulse of light, clean home interior"
    ]
}

def detect_topic(headline):
    if not headline: return None
    headline = headline.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in headline:
                return topic
    return None

def _search_pexels_videos(query, chunk_duration, dynamic_params=None):
    if dynamic_params is None: dynamic_params = {}
    if not PEXELS_API_KEY:
        return []
    try:
        orientation = dynamic_params.get("orientation", "portrait")
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": 5, "orientation": orientation},
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

def _search_pexels_photos(query, orientation="portrait"):
    if not PEXELS_API_KEY:
        return []
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": 5, "orientation": orientation},
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

def _download_photo(url, output_path, is_longform=False):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if r.status_code == 200:
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            w, h = img.size
            if is_longform:
                # Target 16:9 landscape
                target_w = int(h * 16 / 9)
                if target_w <= w:
                    left = (w - target_w) // 2
                    img = img.crop((left, 0, left + target_w, h))
                else:
                    target_h = int(w * 9 / 16)
                    top = (h - target_h) // 2
                    img = img.crop((0, top, w, top + target_h))
                img = img.resized((1920, 1080))
            else:
                # Target 9:16 portrait
                target_h = int(w * 16 / 9)
                if target_h <= h:
                    top = (h - target_h) // 2
                    img = img.crop((0, top, w, top + target_h))
                else:
                    target_w = int(h * 9 / 16)
                    left = (w - target_w) // 2
                    img = img.crop((left, 0, left + target_w, h))
                img = img.resized((1080, 1920))
            img.save(output_path, "JPEG", quality=90)
            return output_path
        return None
    except Exception as e:
        print(f"Photo download err: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# STEP E: Imagen 3
# ─────────────────────────────────────────────────────────────────────────────
def _generate_imagen3(chunk_text, output_path, topic_context="", global_style_guide=""):
    topic = detect_topic(topic_context)
    
    style_suffix = f", {global_style_guide}" if global_style_guide else ", cinematic lighting, professional photography, 8k, photorealistic"
    
    if topic:
        template = random.choice(TOPIC_PROMPT_TEMPLATES[topic])
        # Try to find a name to inject
        entity_name = topic_context or "AI Company"
        # If it's a list or more complex, detect_topic might have been simple, 
        # but let's try to extract the main subject.
        best_prompt = f"{template.format(name=entity_name)}, {style_suffix}, news editorial style, 9:16 vertical, ultra realistic, no text"
        print(f"  -> Detected Topic: {topic}. Using themed prompt for {entity_name}.")
    else:
        # 1. Ask Gemini to craft the perfect Imagen prompt
        topic_prompt = f"Topic Context: {topic_context}. The image MUST be highly relevant to this topic.\n" if topic_context else ""
        prompt_builder = f"""Create a detailed Imagen 3 prompt for this text:
'{chunk_text}'
{topic_prompt}Requirements:
- Photorealistic, cinematic, 9:16 vertical
- Style guide to follow: {global_style_guide}
- No text, no watermarks, no faces of real people
- Directly shows what the text describes, highly precise to the overall topic above
- High detail, dramatic lighting
- RETURN ONLY the prompt text. No introductory sentence."""
        
        best_prompt = chunk_text # Default
        attempts = 0
        while attempts < 3:
            try:
                target_model = "gemini-2.0-flash"
                resp = client.models.generate_content(model=target_model, contents=prompt_builder)
                best_prompt = resp.text.strip()
                # Clean up any lingering intro text
                if best_prompt.lower().startswith("here is") or "prompt:" in best_prompt.lower()[:20]:
                    best_prompt = best_prompt.split("\n")[-1]
                break
            except Exception as e:
                print(f"Imagen prompt gen failed (att {attempts+1}): {e}")
                attempts += 1
                time.sleep(2)
            
    print(f"  -> Generated Imagen prompt: {best_prompt[:80]}...")
        
    # Early exit if we already know Imagen is exhausted for this run
    if os.environ.get("IMAGEN_QUOTA_EXHAUSTED"):
         return None

    # Updated to 4.0 models as 3.0 is missing from the API in this envirodef fetch_chunk_visual(chunk, script_data, topic_context="", global_style_guide="", is_longform=False):
    """
    Executes the Visual Fetching Decision Tree (A -> B -> C -> D -> E)
    """
    from entity_fetcher import fetch_person_photo, fetch_company_logo
    cid = chunk["chunk_id"]
    text = chunk["text"]
    dur = chunk["duration"]
    
    orientation = "landscape" if is_longform else "portrait"
    
    primary_q = chunk.get("pexels_primary", "technology")
    fallback_q = chunk.get("pexels_fallback", "innovation")
    
    video_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.mp4")
    photo_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.jpg")
    
    # ── STEP A/B omitted for brevity, logic remains same ──
    # ...
    
    # ── STEP C: Nanobanana (Imagen 3) ────────────────────────────────────
    print(f"Chunk {cid} -> STEP C: Nanobanana (Imagen 3) generation")
    path = _generate_imagen3(text, photo_out, topic_context, global_style_guide)
    if path:
        chunk["visual_path"] = path
        chunk["visual_type"] = "photo"
        chunk["relevance_score"] = 10
        chunk["source"] = "Step C: Nanobanana (Imagen 3)"
        print(f"Chunk {cid} -> Nanobanana Image Generated: 10/10")
        return chunk

    # ── STEP D: Pexels VIDEOS (Fallback) ─────────────────────────────────
    for query in [primary_q, fallback_q]:
        videos = _search_pexels_videos(query, dur, dynamic_params={"orientation": orientation})
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
        photos = _search_pexels_photos(query, orientation=orientation)
        best_photo = None
        best_score = -1
        
        for p in photos[:3]:  # Limit to top 3 to reduce API calls
            score = score_relevance(text, p["desc"])
            if score >= 5:  # Lowered threshold
                if score > best_score:
                    best_score = score
                    best_photo = p
                    
        if best_photo:
            path = _download_photo(best_photo["link"], photo_out, is_longform=is_longform)
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


def fetch_all_chunk_visuals(chunks, topic_context="", script_data=None, is_longform=False):
    if script_data is None:
        script_data = {}
        
    # 1. Generate Global Style Guide for visual continuity
    global_style = generate_visual_style_guide(topic_context)
        
    print(f"Running Decision Tree for {len(chunks)} chunks (with Smart Throttling)...")
    
    for i, chunk in enumerate(chunks):
        if "pexels_primary" not in chunk:
            chunk["pexels_primary"] = " ".join(chunk["text"].split()[:3])
            chunk["pexels_fallback"] = "technology"

        print(f"  Processing chunk {i+1}/{len(chunks)}...")
        try:
            fetch_chunk_visual(chunk, script_data, topic_context, global_style, is_longform=is_longform)
        except Exception as e:
            print(f"  Chunk {chunk.get('chunk_id')} failed: {e}")
            chunk["visual_path"] = None
            chunk["visual_type"] = None
            chunk["relevance_score"] = 0
            chunk["source"] = "Failed"
        
        # ── SMART THROTTLING ──────────────────────────────────────────────────
        # Only sleep if we used an API limited source (Steps C, D, or E)
        api_sources = [
            "Step C: Nanobanana (Imagen 3)", 
            "Step D: Pexels Video", 
            "Step E: Pexels Photo"
        ]
        
        if i < len(chunks) - 1:
            if chunk.get("source") in api_sources:
                print(f"  -> AI/Search used. Cooling down for 10s to respect API Rate Limits...")
                time.sleep(10)
            else:
                print(f"  -> Local/Cached asset. Skipping cooldown.")

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
