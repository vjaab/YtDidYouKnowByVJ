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
        "Futuristic AI company headquarters, glowing logo on glass skyscraper, night cityscape, cinematic",
        "Silicon Valley style modern office interior with neural network digital art, bright and airy, minimalist",
        "A sleek, minimalist brand logo glowing on a black polished marble surface, luxury tech aesthetic, cinematic"
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
def _generate_imagen3(chunk_text, output_path, topic_context=""):
    topic = detect_topic(topic_context)
    if topic:
        variation = random.choice(TOPIC_PROMPT_TEMPLATES[topic])
        best_prompt = f"{variation}, news editorial style, photorealistic, 8K, 9:16 vertical, cinematic lighting, ultra realistic"
        print(f"  -> Detected Topic: {topic}. Using themed prompt.")
    else:
        # 1. Ask Gemini to craft the perfect Imagen prompt
        topic_prompt = f"Topic Context: {topic_context}. The image MUST be highly relevant to this topic.\n" if topic_context else ""
        prompt_builder = f"""Create a detailed Imagen 3 prompt for this text:
'{chunk_text}'
{topic_prompt}Requirements:
- Photorealistic, cinematic, 9:16 vertical
- No text, no watermarks, no faces of real people
- Directly shows what the text describes, highly precise to the overall topic above
- High detail, dramatic lighting
- RETURN ONLY the prompt text. No introductory sentence like "Here is a prompt" or "Of course"."""
        
        best_prompt = chunk_text # Default
        attempts = 0
        while attempts < 3:
            try:
                target_model = "gemini-2.0-flash"
                resp = client.models.generate_content(model=target_model, contents=prompt_builder)
                best_prompt = resp.text.strip()
                # Clean up any lingering intro text if Gemini ignores instructions
                if best_prompt.lower().startswith("here is") or "prompt:" in best_prompt.lower()[:20]:
                    best_prompt = best_prompt.split("\n")[-1]
                break
            except Exception as e:
                print(f"Imagen prompt gen failed (att {attempts+1}): {e}")
                attempts += 1
                time.sleep(2)
            
    print(f"  -> Generated Imagen prompt: {best_prompt[:60]}...")
        
    # Early exit if we already know Imagen is exhausted for this run
    if os.environ.get("IMAGEN_QUOTA_EXHAUSTED"):
         return None

    # Updated to 4.0 models as 3.0 is missing from the API in this environment
    models_to_try = [
        "imagen-4.0-fast-generate-001",
        "imagen-4.0-generate-001", 
        "imagen-4.0-ultra-generate-001"
    ]
    for model_name in models_to_try:
        attempts = 0
        while attempts < 2: 
            try:
                result = client.models.generate_images(
                    model=model_name,
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
                err_str = str(e).lower()
                wait_time = (2 ** attempts) + 3
                print(f"Imagen generation failed ({model_name}, att {attempts+1}): {e}")
                
                if "429" in err_str and ("quota" in err_str or "exhausted" in err_str):
                    print(f"Quota exceeded for {model_name}. Marking Global Quota Exhausted.")
                    os.environ["IMAGEN_QUOTA_EXHAUSTED"] = "1"
                    break  # Break out of the attempts loop to try the next model
                    
                attempts += 1
                if attempts < 3:
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            
    return None

def fetch_chunk_visual(chunk, script_data, topic_context=""):
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
    path = _generate_imagen3(text, photo_out, topic_context)
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
            fetch_chunk_visual(chunk, script_data, topic_context)
        except Exception as e:
            print(f"  Chunk {chunk.get('chunk_id')} failed: {e}")
            chunk["visual_path"] = None
            chunk["visual_type"] = None
            chunk["relevance_score"] = 0
            chunk["source"] = "Failed"
        
        # Substantial delay between chunks to stay under 10 RPM (Images) / 15 RPM (Gemini)
        if i < len(chunks) - 1:
            print(f"  -> Cooling down for 10s to respect API Rate Limits...")
            time.sleep(10)

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
