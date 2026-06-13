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
from config import GEMINI_API_KEY, OUTPUT_DIR, VEO_MODEL_ID
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
            target_model = "gemini-2.5-flash" 
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
    Asks Gemini to define a consistent, HEADLINE-SPECIFIC visual 'vibe' for the whole video.
    This ensures all AI-generated images share a palette, lighting style, and are relevant to the story.
    """
    print("🎨 Designing Global Visual Style Guide...")
    try:
        target_model = "gemini-2.5-flash"
        prompt = f"""Based on this news headline: '{headline}', define a cohesive visual style for a 9:16 vertical cinematic video.

CRITICAL: The style MUST be tailored to this specific story. Include:
1. Color palette that matches the mood/entities (e.g., OpenAI = green/white, Google = blue/red/yellow/green, cybersecurity = dark/neon)
2. Lighting style that fits the story tone (e.g., lawsuit = dramatic/contrasty, product launch = bright/clean)
3. Visual motifs related to the entities mentioned (e.g., tech company logos as glowing elements, relevant product imagery)

Return a short string (max 50 words) describing the lighting, color palette, camera style, AND relevant visual motifs.
Example: 'Dark courtroom drama aesthetic, OpenAI green and Tesla silver palette, dramatic split-lighting, gavel imagery, digital contract visuals, shot on 35mm lens, high contrast, editorial news photography style.'
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

# ─────────────────────────────────────────────────────────────────────────────
# 30-DAY VISUAL AESTHETIC MATRIX (Nano & Veo Variety)
# ─────────────────────────────────────────────────────────────────────────────
DAILY_AESTHETIC_MATRIX = [
    {"lighting": "Soft teal and orange", "camera": "Subtle track-in", "mood": "futuristic tech-docu", "motif": "glowing neural networks"}, # Day 0 (Padding)
    {"lighting": "Teal and Orange", "camera": "Slow Pan", "mood": "Futuristic Docu", "motif": "Neural Networks"},
    {"lighting": "Golden Hour", "camera": "Drone Sweep", "mood": "Cinematic Tech", "motif": "Solar Arrays"},
    {"lighting": "Neon Cyberpunk", "camera": "Static Glitch", "mood": "High-Energy", "motif": "Binary Code Streams"},
    {"lighting": "Soft Minimalist", "camera": "Macro Focus", "mood": "Clean Lab", "motif": "Silicon Wafers"},
    {"lighting": "Dramatic Chiaroscuro", "camera": "Low Angle", "mood": "Serious Corporate", "motif": "Glass Skyscrapers"},
    {"lighting": "Ethereal Glow", "camera": "Dreamy Blur", "mood": "Conceptual AI", "motif": "Floating Light Particles"},
    {"lighting": "Industrial Cold", "camera": "Handheld Shake", "mood": "Gritty Tech", "motif": "Robotic Limbs"},
    {"lighting": "Hyper-Realistic", "camera": "Top Down", "mood": "Editorial News", "motif": "Printed Circuit Boards"},
    {"lighting": "Midnight Blue", "camera": "Tracking Shot", "mood": "Deep Sea Tech", "motif": "Optical Fibers"},
    {"lighting": "Sunset Warmth", "camera": "Rack Focus", "mood": "Human-Centric", "motif": "Biometric Sensors"},
    {"lighting": "Monochrome Stark", "camera": "Zoom In", "mood": "Investigative", "motif": "Data Center Rows"},
    {"lighting": "Pastel Digital", "camera": "Floating Cam", "mood": "Soft Software", "motif": "UI/UX Holograms"},
    {"lighting": "Electric Purple", "camera": "High Speed", "mood": "Cybernetic", "motif": "Plasma Energy Gates"},
    {"lighting": "Natural Sunlight", "camera": "Window Reflection", "mood": "Startup Vibe", "motif": "Modern Office Plants"},
    {"lighting": "Infrared Vision", "camera": "Thermal Scan", "mood": "Surveillance", "motif": "Heat Map Visuals"},
    {"lighting": "Retro CRT", "camera": "Interlaced Scan", "mood": "Nostalgic Tech", "motif": "Floppy Disks & Terminals"},
    {"lighting": "Obsidian Black", "camera": "Reflection", "mood": "Luxury Tech", "motif": "Polished Carbon Fiber"},
    {"lighting": "Forest Green", "camera": "Organic Motion", "mood": "Sustainable Tech", "motif": "Bio-engineered Leaves"},
    {"lighting": "Volumetric Light", "camera": "God Rays", "mood": "Heroic Launch", "motif": "Satellite Dishes"},
    {"lighting": "Microscopic", "camera": "Slow Drift", "mood": "Nanotech", "motif": "DNA Double Helix"},
    {"lighting": "Blueprint Blue", "camera": "Line Art", "mood": "Engineering", "motif": "Technical Schematics"},
    {"lighting": "Crimson Warning", "camera": "Fast Cuts", "mood": "Crisis Mode", "motif": "Server Overload Sparks"},
    {"lighting": "Quartz White", "camera": "Crystal Clear", "mood": "Pure Science", "motif": "Prisms & Refractions"},
    {"lighting": "Steampunk Brass", "camera": "Gear Motion", "mood": "Alt-History Tech", "motif": "Clockwork AI"},
    {"lighting": "Arctic White", "camera": "Frosty Lens", "mood": "Cold Storage", "motif": "Liquid Nitrogen Cooling"},
    {"lighting": "Holographic Rainbow", "camera": "Iridescent", "mood": "Web3 / NFT", "motif": "Shimmering Mesh Nets"},
    {"lighting": "Deep Crimson", "camera": "Underlighting", "mood": "Secretive", "motif": "Underground Bunker Tech"},
    {"lighting": "Morning Mist", "camera": "Soft Diffuse", "mood": "New Dawn", "motif": "Emerging AI Entities"},
    {"lighting": "Cosmic Nebula", "camera": "Starfield", "mood": "Space Tech", "motif": "Orbital Station Views"},
    {"lighting": "Graphite Grey", "camera": "Matte Finish", "mood": "Industrial Design", "motif": "Brushed Metal Textures"},
    {"lighting": "Rainbow Spectrum", "camera": "Vibrant", "mood": "Inclusive Tech", "motif": "Diverse Humanoid Robots"}
]

def detect_topic(headline):
    if not headline: return None
    headline = headline.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in headline:
                return topic
    return None

def _generate_veo_video(prompt, output_path, aspect_ratio="9:16"):
    """Generates a video using Google Veo (async polling)."""
    print(f"🎬 Generating Veo Video: {prompt[:80]}...")
    try:
        operation = client.models.generate_videos(
            model=VEO_MODEL_ID,
            prompt=prompt,
            config=genai.types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
            )
        )
        
        attempts = 0
        while not operation.done and attempts < 120: # Max 20 mins
            time.sleep(10)
            operation = client.operations.get(operation)
            attempts += 1
            if attempts % 6 == 0:
                print(f"   ...still generating Veo video ({attempts*10}s)")

        if operation.error:
            print(f"❌ Veo Operation Failed: {operation.error}")
            return None

        if operation.result and hasattr(operation.result, 'generated_videos') and operation.result.generated_videos:
            gen_video = operation.result.generated_videos[0]
            video = gen_video.video if hasattr(gen_video, 'video') else gen_video
            
            if hasattr(video, 'video_bytes') and video.video_bytes:
                with open(output_path, "wb") as f:
                    f.write(video.video_bytes)
                return output_path
            elif hasattr(video, 'uri') and video.uri:
                # If it's a URI, we need to download it with the API key
                r = requests.get(video.uri, headers={"x-goog-api-key": GEMINI_API_KEY}, timeout=60)
                if r.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(r.content)
                    return output_path
        print(f"⚠️ Veo operation finished but no video found. Result: {operation.result}")
    except Exception as e:
        print(f"⚠️ Veo generation failed: {e}")
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
            
            # Find portrait/landscape link depending on orientation
            files = v.get("video_files", [])
            if orientation == "portrait":
                target_files = [f for f in files if f.get("width", 0) < f.get("height", 1) and f.get("height", 0) >= 720]
            else:
                target_files = [f for f in files if f.get("width", 0) > f.get("height", 1) and f.get("width", 0) >= 1280]
                
            if not target_files:
                target_files = sorted(files, key=lambda f: f.get("height", 0), reverse=True)
                
            if target_files:
                results.append({
                    "id": vid_id, 
                    "link": target_files[0]["link"], 
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
        from PIL import Image, ImageOps
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
                img = img.resize((1920, 1080))
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
                img = img.resize((1080, 1920))
            img.save(output_path, "JPEG", quality=90)
            return output_path
        return None
    except Exception as e:
        print(f"Photo download err: {e}")
        return None

def _fetch_pexels_fallback(chunk, duration, is_video=False, is_longform=False):
    cid = chunk.get("chunk_id")
    primary_q = chunk.get("pexels_primary", "technology")
    fallback_q = chunk.get("pexels_fallback", "innovation")
    video_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.mp4")
    photo_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.jpg")
    
    orientation = "landscape" if is_longform else "portrait"
    
    if is_video:
        # Try videos first
        for query in [primary_q, fallback_q]:
            print(f"   Searching Pexels video for '{query}'...")
            videos = _search_pexels_videos(query, duration, {"orientation": orientation})
            for v in videos:
                path = _download_video(v["link"], video_out)
                if path:
                    with _download_lock:
                        _used_media.add(v["id"])
                    return path, f"Video ({query})", "video"
        
        # Fallback to photos if no videos found
        for query in [primary_q, fallback_q]:
            print(f"   No video found. Searching Pexels photo for '{query}'...")
            photos = _search_pexels_photos(query, orientation=orientation)
            for p in photos:
                path = _download_photo(p["link"], photo_out, is_longform=is_longform)
                if path:
                    with _download_lock:
                        _used_media.add(p["id"])
                    return path, f"Photo ({query})", "photo"
    else:
        # Try photos first
        for query in [primary_q, fallback_q]:
            print(f"   Searching Pexels photo for '{query}'...")
            photos = _search_pexels_photos(query, orientation=orientation)
            for p in photos:
                path = _download_photo(p["link"], photo_out, is_longform=is_longform)
                if path:
                    with _download_lock:
                        _used_media.add(p["id"])
                    return path, f"Photo ({query})", "photo"
                    
        # Fallback to videos if no photos found
        for query in [primary_q, fallback_q]:
            print(f"   No photo found. Searching Pexels video for '{query}'...")
            videos = _search_pexels_videos(query, duration, {"orientation": orientation})
            for v in videos:
                path = _download_video(v["link"], video_out)
                if path:
                    with _download_lock:
                        _used_media.add(v["id"])
                    return path, f"Video ({query})", "video"
                    
    return None, None, None

# ─────────────────────────────────────────────────────────────────────────────
# STEP E: Imagen 3
# ─────────────────────────────────────────────────────────────────────────────
_subject_cache = {}

def _extract_visual_subject(headline):
    """Use Gemini to extract the primary visual subject/entity from a headline. Cached per headline."""
    if not headline: return "AI Technology"
    if headline in _subject_cache:
        return _subject_cache[headline]

    attempts = 0
    while attempts < 3:
        try:
            target_model = "gemini-2.5-flash"
            prompt = f"""From this news headline, extract the PRIMARY visual subject that should appear in a background image.
Return ONLY the short subject name (1-5 words). Examples:
- "Elon Musk sues OpenAI" → "OpenAI vs Elon Musk"
- "Google launches Gemini 2.0" → "Google Gemini"
- "NVIDIA stock hits record high" → "NVIDIA"
- "New self-driving car regulation passed" → "Autonomous Vehicles"

Headline: "{headline}"
Return ONLY the subject:"""
            
            resp = client.models.generate_content(
                model=target_model, contents=prompt,
                config=genai.types.GenerateContentConfig(temperature=0.0)
            )
            subject = resp.text.strip().strip('"').strip("'")
            if subject and len(subject) < 60:
                _subject_cache[headline] = subject
                return subject
            break # Exit on invalid subject
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str:
                wait_time = 60
                print(f"  ⚠️ Entity extraction rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ⚠️ Entity extraction failed (att {attempts+1}): {e}")
                time.sleep(2)
            attempts += 1
    
    # Fallback: use first 4 meaningful words from headline
    words = [w for w in (headline or "").split() if len(w) > 2][:4]
    subject = " ".join(words) if words else "AI Technology"
    _subject_cache[headline] = subject
    return subject


def generate_premium_prompt_via_gemini(chunk_text, topic_context, global_style_guide, original_visual_prompt=None, aspect_ratio="9:16", is_video=False):
    """
    Calls Gemini to generate a highly specific, cinematic, visual prompt for Imagen or Veo.
    """
    import re
    target_model = "gemini-2.5-flash"
    orientation = "16:9 landscape format" if aspect_ratio == "16:9" else "9:16 vertical format for mobile"
    media_type = "video clip with slow, fluid cinematic motion" if is_video else "photorealistic high-end image"
    
    # Clean up chunk text and original visual prompt
    chunk_text_clean = re.sub(r'\s+', ' ', chunk_text).strip()
    original_visual_prompt_clean = re.sub(r'\s+', ' ', original_visual_prompt).strip() if original_visual_prompt else ""
    
    prompt = f"""You are a senior Hollywood director and AI visual prompt designer.
Generate an elite prompt for a fullscreen background {media_type} to be used in a high-production-value YouTube Short about tech news.

CONTEXT:
- Video Topic/Headline: "{topic_context}"
- Current sentence being spoken: "{chunk_text_clean}"
- Global Visual Style/Aesthetic: "{global_style_guide}"
"""
    if original_visual_prompt_clean:
        prompt += f'- Original concept suggestion: "{original_visual_prompt_clean}"\n'
        
    prompt += f"""
RULES for the generated prompt:
1. It MUST be highly relevant, visually representing the core concept or subject of: "{chunk_text_clean}".
2. Do NOT put any text, typography, subtitles, labels, logos, or watermarks in the prompt. Keep it purely visual.
3. Do NOT include faces of real people (e.g. Sam Altman, Sundar Pichai, Elon Musk). Instead, use generic descriptions (e.g., "a visionary CEO in a dark tech-noir room", "a researcher looking at a glowing holographic screen").
4. Describe the camera shot, angle, lighting, and lens details to make it look premium (e.g. "shot on 35mm lens, cinematic split lighting, shallow depth of field, subtle hand-held camera shake, volumetric dust particles").
5. Use vibrant, harmonious, modern color grading (e.g., cyber-cyan/amber contrast, deep emerald greens and dark charcoal, moody cyberpunk neon).
6. Format must be {orientation}.
7. Do NOT include any introductory or concluding text (e.g. "Here is your prompt:"). Return ONLY the raw prompt.

Output the visual prompt:"""

    try:
        response = client.models.generate_content(
            model=target_model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(temperature=0.8)
        )
        res = response.text.strip().strip('"').strip("'")
        print(f"   [GEMINI PROMPT GEN] Created prompt: {res[:100]}...")
        return res
    except Exception as e:
        print(f"⚠️ Gemini prompt generation failed: {e}. Using fallback.")
        vibe = global_style_guide if global_style_guide else "cinematic lighting, photorealistic, 8k"
        subj = original_visual_prompt_clean if original_visual_prompt_clean else chunk_text_clean
        return f"{subj}, {vibe}, {orientation}, highly detailed, photorealistic, no text"


def _generate_imagen3(prompt, output_path, topic_context="", global_style_guide="", visual_subject=None, aspect_ratio="9:16"):
    """
    Generates an image via Imagen 4.0 using the fully engineered prompt.
    Keeps signature compatibility for other modules like entity_fetcher.
    """
    print(f"🎨 Generating Imagen Image with prompt: {prompt[:80]}...")
    
    # Early exit if we already know Imagen is exhausted for this run
    if os.environ.get("IMAGEN_QUOTA_EXHAUSTED"):
         return None

    # Actually generate the image using Imagen 4.0
    models_to_try = [
        "imagen-4.0-fast-generate-001",
        "imagen-4.0-generate-001",
        "imagen-4.0-ultra-generate-001"
    ]
    
    for model_name in models_to_try:
        try:
            result = client.models.generate_images(
                model=model_name,
                prompt=prompt,
                config=genai.types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio=aspect_ratio,
                    output_mime_type="image/jpeg",
                )
            )
            for gen_img in result.generated_images:
                with open(output_path, "wb") as f:
                    f.write(gen_img.image.image_bytes)
                return output_path
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str and ("quota" in err_str or "exhausted" in err_str):
                print(f"  ⚠️ Imagen quota exhausted on {model_name}. Trying next...")
                continue
            print(f"  ⚠️ Imagen call failed on {model_name}: {e}")
            break
            
    return None


def fetch_chunk_visual(chunk, script_data, topic_context="", global_style_guide="", is_longform=False, visual_mode="veo_concept", visual_subject=None):
    """
    Executes the Visual Fetching Decision Tree with Switch-Back Logic
    """
    from entity_fetcher import fetch_person_photo, fetch_company_logo
    cid = chunk["chunk_id"]
    text = chunk["text"]
    dur = chunk["duration"]
    
    orientation = "16:9" if is_longform else "9:16"
    
    video_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.mp4")
    photo_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.jpg")
    
    if not visual_subject:
        visual_subject = _extract_visual_subject(topic_context)
    
    headline = topic_context if topic_context else "Scientific Research"
    
    day_idx = datetime.now().day
    style = DAILY_AESTHETIC_MATRIX[day_idx % len(DAILY_AESTHETIC_MATRIX)]
    print(f"  📅 Day {day_idx} Aesthetic: {style['mood']} ({style['lighting']})")

    # Get original visual prompt generated during script planning (if any)
    original_prompt = chunk.get("nano_visual_prompt", "")

    if visual_mode == "nano_hook" or visual_mode == "nano_concept":
        print(f"Chunk {cid} -> MODE: {visual_mode}")
        # Generate custom prompt via Gemini
        custom_prompt = generate_premium_prompt_via_gemini(
            chunk_text=text,
            topic_context=topic_context,
            global_style_guide=global_style_guide,
            original_visual_prompt=original_prompt,
            aspect_ratio=orientation,
            is_video=False
        )
        path = _generate_imagen3(custom_prompt, photo_out, aspect_ratio=orientation)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "photo"
            chunk["relevance_score"] = 10
            chunk["source"] = f"Imagen ({visual_mode})"
            return chunk
            
        # Pexels fallback for photo mode!
        print(f"Chunk {cid} -> Imagen failed, falling back to Pexels search...")
        path, source_desc, v_type = _fetch_pexels_fallback(chunk, dur, is_video=False, is_longform=is_longform)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = v_type
            chunk["relevance_score"] = 7
            chunk["source"] = f"Pexels fallback ({source_desc})"
            return chunk

    elif visual_mode == "nano_evidence":
        selected_screenshot = None
        screenshot_source = None
        
        if is_longform:
            fact_num = chunk.get("fact_number")
            topics = script_data.get("longform_topics", [])
            if isinstance(fact_num, int) and 1 <= fact_num <= len(topics):
                topic = topics[fact_num - 1]
                screenshot_path = topic.get("screenshot_path")
                if screenshot_path and os.path.exists(screenshot_path):
                    selected_screenshot = screenshot_path
                    screenshot_source = f"Fact {fact_num} Article Screenshot"
            
            if not selected_screenshot:
                main_screenshot = script_data.get("screenshot_path")
                if main_screenshot and os.path.exists(main_screenshot):
                    selected_screenshot = main_screenshot
                    screenshot_source = "Main Article Screenshot"
        else:
            main_screenshot = script_data.get("screenshot_path")
            evidence_screenshot = script_data.get("evidence_screenshot_path")
            
            if cid == 2:
                if main_screenshot and os.path.exists(main_screenshot):
                    selected_screenshot = main_screenshot
                    screenshot_source = "Real Article Screenshot"
                elif evidence_screenshot and os.path.exists(evidence_screenshot):
                    selected_screenshot = evidence_screenshot
                    screenshot_source = "Evidence Screenshot"
            else:
                if evidence_screenshot and os.path.exists(evidence_screenshot):
                    selected_screenshot = evidence_screenshot
                    screenshot_source = "Evidence Screenshot"
                elif main_screenshot and os.path.exists(main_screenshot):
                    selected_screenshot = main_screenshot
                    screenshot_source = "Real Article Screenshot"
                    
        if selected_screenshot:
            print(f"Chunk {cid} -> MODE: nano_evidence (Using {screenshot_source}: {selected_screenshot})")
            chunk["visual_path"] = selected_screenshot
            chunk["visual_type"] = "photo"
            chunk["relevance_score"] = 10
            chunk["source"] = screenshot_source
            return chunk

        print(f"Chunk {cid} -> MODE: nano_evidence (Using AI Macro Fallback)")
        evidence_concept = f"A professional macro photograph of scientific research paper, technical charts, code editor or document titled '{headline}'."
        custom_prompt = generate_premium_prompt_via_gemini(
            chunk_text=text,
            topic_context=topic_context,
            global_style_guide=global_style_guide,
            original_visual_prompt=evidence_concept,
            aspect_ratio=orientation,
            is_video=False
        )
        path = _generate_imagen3(custom_prompt, photo_out, aspect_ratio=orientation)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "photo"
            chunk["relevance_score"] = 10
            chunk["source"] = "Imagen (nano_evidence)"
            return chunk
            
        # Pexels fallback for evidence mode!
        print(f"Chunk {cid} -> Imagen evidence failed, falling back to Pexels search...")
        path, source_desc, v_type = _fetch_pexels_fallback(chunk, dur, is_video=False, is_longform=is_longform)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = v_type
            chunk["relevance_score"] = 7
            chunk["source"] = f"Pexels evidence fallback ({source_desc})"
            return chunk

    elif visual_mode == "veo_concept" or visual_mode == "veo_cta":
        print(f"Chunk {cid} -> MODE: {visual_mode}")
        custom_video_prompt = generate_premium_prompt_via_gemini(
            chunk_text=text,
            topic_context=topic_context,
            global_style_guide=global_style_guide,
            original_visual_prompt=original_prompt,
            aspect_ratio=orientation,
            is_video=True
        )
        path = _generate_veo_video(custom_video_prompt, video_out, aspect_ratio=orientation)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "video"
            chunk["relevance_score"] = 10
            chunk["source"] = f"Veo ({visual_mode})"
            return chunk
            
        # Fallback to Imagen if Veo fails
        print(f"Chunk {cid} -> Veo failed, falling back to Imagen")
        custom_img_prompt = generate_premium_prompt_via_gemini(
            chunk_text=text,
            topic_context=topic_context,
            global_style_guide=global_style_guide,
            original_visual_prompt=original_prompt,
            aspect_ratio=orientation,
            is_video=False
        )
        path = _generate_imagen3(custom_img_prompt, photo_out, aspect_ratio=orientation)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "photo"
            chunk["relevance_score"] = 8
            chunk["source"] = "Imagen (fallback from veo)"
            return chunk
            
        # Pexels fallback for video mode!
        print(f"Chunk {cid} -> Veo/Imagen failed, falling back to Pexels search...")
        path, source_desc, v_type = _fetch_pexels_fallback(chunk, dur, is_video=True, is_longform=is_longform)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = v_type
            chunk["relevance_score"] = 7
            chunk["source"] = f"Pexels video fallback ({source_desc})"
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
    
    # 2. Extract Visual Subject once (to avoid 429 rate limits on redundant calls)
    vis_subject = _extract_visual_subject(topic_context)
        
    print(f"Running Decision Tree for {len(chunks)} chunks (with Smart Throttling)...")
    
    current_fact = None
    fact_offset = 0

    for i, chunk in enumerate(chunks):
        if "pexels_primary" not in chunk:
            chunk["pexels_primary"] = " ".join(chunk["text"].split()[:3])
            chunk["pexels_fallback"] = "technology"

        # Determine visual mode based on index and is_longform
        total_chunks = len(chunks)
        
        if is_longform:
            fact_num = chunk.get("fact_number")
            
            # If the fact number changes, reset the offset
            if fact_num != current_fact:
                current_fact = fact_num
                fact_offset = 0
            else:
                fact_offset += 1
                
            # Determine mode within the current fact
            if fact_num == 0:  # Cold Open
                if fact_offset == 0:
                    v_mode = "nano_hook"
                else:
                    v_mode = "veo_concept" if fact_offset % 2 == 0 else "nano_concept"
            elif fact_num == "outro":
                v_mode = "veo_cta"
            elif isinstance(fact_num, str) and "recap" in fact_num:
                v_mode = "nano_concept"
            else:
                # Standard fact structure
                if fact_offset == 0:
                    v_mode = "nano_hook"
                elif fact_offset == 1:
                    v_mode = "nano_evidence"  # Show topic-aligned screenshot right after hook
                elif i == total_chunks - 1:
                    v_mode = "veo_cta"
                else:
                    # Alternate between Veo video and Imagen image
                    v_mode = "veo_concept" if fact_offset % 2 == 0 else "nano_concept"
        else:
            # Shorts logic
            if i == 0:
                v_mode = "nano_hook"
            elif i == 1:
                v_mode = "nano_evidence" # The "Evidence Flash"
            elif i == total_chunks - 1:
                v_mode = "veo_cta"
            elif i % 4 == 1:
                # Every 4th chunk (starting from 5), switch back to Evidence
                v_mode = "nano_evidence"
            else:
                # Alternate concept loop for the rest
                v_mode = "veo_concept" if i % 2 == 0 else "nano_concept"
        
        # ── PHASE 3: RETENTION-DRIVEN VISUAL OVERRIDES ────────────────────
        # Use the retention_map from Phase 2 to force visual changes at risk zones
        retention_map = script_data.get("retention_map", {})
        if retention_map and not is_longform:
            # Get pattern interrupt positions (convert word positions to chunk indices)
            pattern_interrupts = retention_map.get("pattern_interrupts", [])
            risk_zones = retention_map.get("retention_risk_zones", [])
            
            # Check if this chunk aligns with a pattern interrupt
            chunk_text_words = chunk.get("text", "").split()
            for pi in pattern_interrupts:
                pi_word = pi.get("at_word", 0)
                pi_type = pi.get("type", "")
                # Rough mapping: each chunk is ~5-8 words, so chunk index ≈ word_pos / 6
                estimated_chunk = pi_word // max(1, 170 // total_chunks)
                if abs(i - estimated_chunk) <= 1:
                    # Override visual mode for pattern interrupt effect
                    if pi_type in ["contradiction", "stat_bomb"]:
                        v_mode = "nano_hook"  # High-impact imagery
                    elif pi_type in ["rhetorical_question", "direct_address"]:
                        v_mode = "nano_evidence"  # Ground with proof
                    break
            
            # Force visual change at retention risk zones
            for rz in risk_zones:
                rz_word = rz.get("at_word", 0)
                estimated_chunk = rz_word // max(1, 170 // total_chunks)
                if abs(i - estimated_chunk) <= 1 and v_mode not in ["nano_hook", "nano_evidence"]:
                    v_mode = "veo_concept"  # Motion video to recapture attention
                    break

        print(f"  Processing chunk {i+1}/{len(chunks)} [{v_mode}]...")
        try:
            fetch_chunk_visual(chunk, script_data, topic_context, global_style, is_longform=is_longform, visual_mode=v_mode)
        except Exception as e:
            print(f"  Chunk {chunk.get('chunk_id')} failed: {e}")
            chunk["visual_path"] = None
            chunk["visual_type"] = None
            chunk["relevance_score"] = 0
            chunk["source"] = "Failed"
        
        # ── SMART THROTTLING ──────────────────────────────────────────────────
        api_sources = [
            "Imagen (nano_hook)", "Imagen (nano_concept)", "Imagen (nano_evidence)",
            "Veo (veo_concept)", "Veo (veo_cta)", "Imagen (fallback from veo)",
            "Real Article Screenshot"
        ]
        
        if i < len(chunks) - 1:
            if chunk.get("source") in api_sources:
                print(f"  -> AI generated asset used. Cooling down for 10s to respect API Rate Limits...")
                time.sleep(10)
            else:
                print(f"  -> Local/Cached asset. Skipping cooldown.")

    # Robust two-pass visual gap filler
    first_path = None
    first_type = "photo"
    for c in chunks:
        if c.get("visual_path") and os.path.exists(c["visual_path"]):
            first_path = c["visual_path"]
            first_type = c.get("visual_type", "photo")
            break

    if not first_path:
        # Absolute fallback if all generations failed and no screenshots exist
        first_path = "dummy_screenshot.png"
        first_type = "photo"

    last_path = first_path
    last_type = first_type
    for c in chunks:
        if c.get("visual_path") and os.path.exists(c["visual_path"]):
            last_path = c["visual_path"]
            last_type = c.get("visual_type", "photo")
        else:
            c["visual_path"] = last_path
            c["visual_type"] = last_type
            c["source"] = c.get("source") or "Gap-filled"

    return chunks
