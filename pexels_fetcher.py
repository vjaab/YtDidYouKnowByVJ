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
def calculate_local_keyword_score(chunk_text, visual_desc):
    """
    Calculates a quick keyword overlap relevance score (0-10) locally.
    """
    import re
    # Clean and tokenize
    words_chunk = set(re.findall(r'[a-z0-9]+', chunk_text.lower()))
    words_desc = set(re.findall(r'[a-z0-9]+', visual_desc.lower()))
    
    # Remove common stopwords and visual noise words
    stopwords = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at", 
        "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant", "cannot", "could", 
        "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for", "from", 
        "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes", "her", "here", 
        "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in", 
        "into", "is", "isnt", "it", "its", "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor", 
        "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
        "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such", "than", "that", "thats", 
        "the", "their", "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", 
        "theyre", "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", 
        "wed", "well", "were", "weve", "werent", "what", "whats", "when", "whens", "where", "wheres", "which", "while", 
        "who", "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", 
        "your", "yours", "yourself", "yourselves", "of", "in", "the", "on", "close", "closeup", "up", "showing", "shown"
    }
    
    clean_chunk = words_chunk - stopwords
    clean_desc = words_desc - stopwords
    
    if not clean_chunk:
        return 5
        
    overlap = clean_chunk.intersection(clean_desc)
    
    if not overlap:
        return 2
        
    # Calculate score based on ratio of overlap
    ratio = len(overlap) / len(clean_chunk)
    # Scale to 0-10, with a base of 6 for any matching technical keywords
    score = int(6 + ratio * 4)
    return min(10, score)


def score_relevance(chunk_text, visual_desc):
    """
    Rates the relevance between technical text and visual description (0-10).
    Uses a fast local keyword heuristic as primary search and rate-limit fallback.
    """
    import re
    
    # 1. Run local keyword check first
    local_score = calculate_local_keyword_score(chunk_text, visual_desc)
    if local_score >= 8:
        print(f"   [RELEVANCE] Local strong match found (score: {local_score}) for '{visual_desc[:40]}'")
        return local_score

    # 2. Call Gemini for nuanced semantic understanding
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
            break
        except Exception as e:
            err_str = str(e).lower()
            # If rate limited, wait longer (60s) for the minute to reset
            if "429" in err_str or "resource_exhausted" in err_str:
                wait_time = 15 * (attempts + 1)  # Progressive backing off: 15s, 30s...
                print(f"⚠️ Gemini Rate Limit Hit (429). Waiting {wait_time}s...")
            else:
                wait_time = (2 ** attempts) + 5
                print(f"Gemini scoring failed (att {attempts+1}): {e}. Retrying in {wait_time}s...")
            
            attempts += 1
            time.sleep(wait_time)
            
    # Fallback to local score on total API failure to keep rendering robust
    print(f"   [RELEVANCE] API failed. Falling back to local score: {local_score} for '{visual_desc[:40]}'")
    return local_score


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL VISUAL CONTINUITY
# ─────────────────────────────────────────────────────────────────────────────
def generate_visual_style_guide(headline, is_longform=False):
    """
    Asks Gemini to define a consistent, HEADLINE-SPECIFIC visual 'vibe' for the whole video.
    This ensures all AI-generated images share a palette, lighting style, and are relevant to the story.
    """
    print("🎨 Designing Global Visual Style Guide...")
    try:
        target_model = "gemini-2.5-flash"
        orientation = "16:9 landscape cinematic video" if is_longform else "9:16 vertical cinematic video"
        
        prompt = f"""Based on this news headline: '{headline}', define a cohesive visual style for a {orientation}.

CRITICAL: The style MUST be tailored to this specific story. Include:
1. Color palette that matches the mood/entities (e.g., OpenAI = green/white, Google = blue/red/yellow/green, cybersecurity = dark/neon)
2. Lighting style that fits the story tone (e.g., lawsuit = dramatic/contrasty, product launch = bright/clean)
3. Visual motifs related to the entities mentioned (e.g., tech company logos as glowing elements, relevant product imagery)
"""
        if is_longform:
            prompt += """4. CONTEXTUAL NO-B-ROLL RULE: Avoid all generic stock-style abstract tech loops (like digital brains, glowing neural networks, or particle vortexes). Instead, prioritize direct, real-world visuals: screen recordings of code editor interfaces (e.g. VS Code, Jupyter), terminal inputs/outputs, system architecture blueprints, vector diagrams, API request flows, or simple high-contrast typography cards for data/metrics.
"""
        prompt += """
Return a short string (max 60 words) describing the lighting, color palette, camera style, AND relevant visual motifs.
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
    if os.environ.get("VEO_QUOTA_EXHAUSTED") == "true":
        print("   ⏭️ Veo quota previously exhausted. Skipping Veo generation call.")
        return None

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
            err_msg = str(operation.error).upper()
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "QUOTA" in err_msg:
                print("🚨 Veo API Quota Exhausted (429/Resource Exhausted). Setting fast-fallback flag.")
                os.environ["VEO_QUOTA_EXHAUSTED"] = "true"
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
        err_msg = str(e).upper()
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "QUOTA" in err_msg:
            print("🚨 Veo API Quota Exhausted (429/Resource Exhausted). Setting fast-fallback flag.")
            os.environ["VEO_QUOTA_EXHAUSTED"] = "true"
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
def _extract_clean_search_query(chunk, topic_context=""):
    """
    Formulates a clean, high-relevance search query for Pexels based on
    visual directions (nano_visual_prompt) or chunk text.
    """
    import re
    
    # 1. Try to extract from visual direction prompt
    prompt = chunk.get("nano_visual_prompt", "")
    if prompt:
        # Lowercase for uniform processing
        q = prompt.strip().lower()
        
        # Remove common introductory filler phrases/prefixes
        fillers = [
            r"^what should be shown\s*:\s*",
            r"^what to show\s*:\s*",
            r"^visual direction\s*:\s*",
            r"^b-roll of\s+(a\s+|an\s+|the\s+)?",
            r"^b-roll showing\s+(a\s+|an\s+|the\s+)?",
            r"^b-roll\s+(a\s+|an\s+|the\s+)?",
            r"^close-up of\s+(a\s+|an\s+|the\s+)?",
            r"^close up of\s+(a\s+|an\s+|the\s+)?",
            r"^extreme close-up of\s+(a\s+|an\s+|the\s+)?",
            r"^extreme close up of\s+(a\s+|an\s+|the\s+)?",
            r"^macro photograph of\s+(a\s+|an\s+|the\s+)?",
            r"^macro photo of\s+(a\s+|an\s+|the\s+)?",
            r"^screenshot of\s+(a\s+|an\s+|the\s+)?",
            r"^diagram showing\s+(a\s+|an\s+|the\s+)?",
            r"^diagram of\s+(a\s+|an\s+|the\s+)?",
            r"^sketch of\s+(a\s+|an\s+|the\s+)?",
            r"^a simple whiteboard marker sketch of\s+(a\s+|an\s+|the\s+)?",
            r"^whiteboard sketch of\s+(a\s+|an\s+|the\s+)?",
            r"^a drawing of\s+(a\s+|an\s+|the\s+)?",
            r"^animated ui mockup showing\s+(a\s+|an\s+|the\s+)?",
            r"^ui mockup showing\s+(a\s+|an\s+|the\s+)?",
            r"^mockup of\s+(a\s+|an\s+|the\s+)?",
            r"^photo of\s+(a\s+|an\s+|the\s+)?",
            r"^image of\s+(a\s+|an\s+|the\s+)?",
            r"^video of\s+(a\s+|an\s+|the\s+)?",
            r"^clip showing\s+(a\s+|an\s+|the\s+)?",
            r"^show\s+(a\s+|an\s+|the\s+)?",
            r"^illustration of\s+(a\s+|an\s+|the\s+)?",
            r"^concept of\s+(a\s+|an\s+|the\s+)?",
        ]
        for f in fillers:
            q = re.sub(f, "", q)
            
        # Remove trailing visual style guides/clauses (split on common keywords)
        style_fillers = [
            ", cinematic", ", photorealistic", ", realistic", ", futuristic",
            ", cyberpunk", ", luxury tech", ", minimalist", ", high detail",
            ", shot on", " with dramatic", " with glowing", " glowing on",
            " emitting ", " in a dark", " in a modern"
        ]
        for sf in style_fillers:
            if sf in q:
                q = q.split(sf)[0]
                
        q = q.strip().strip(".,:;-")
        # Keep it reasonably short (max 6 words for Pexels search query efficiency)
        words = q.split()
        if 1 <= len(words) <= 7:
            return q
        elif len(words) > 7:
            # Take first 5 words if too long
            return " ".join(words[:5])

    # 2. Extract from chunk text using keyword matching for tech nouns
    text = chunk.get("text", "")
    if text:
        tech_keywords = [
            "gpu", "tpu", "semiconductor", "microchip", "silicon", "cleanroom",
            "server", "data center", "robot", "humanoid", "database", "cybersecurity",
            "hacker", "firewall", "encryption", "quantum", "neural network",
            "machine learning", "deep learning", "algorithm", "source code",
            "terminal", "coding", "programmer", "software", "api", "cloud",
            "fiber optic", "satellite", "dna", "microscope", "telecom", "supercomputer"
        ]
        text_lower = text.lower()
        found_keywords = []
        for kw in tech_keywords:
            if kw in text_lower:
                found_keywords.append(kw)
        if found_keywords:
            # Use found keywords (deduplicated)
            unique_kws = list(dict.fromkeys(found_keywords))
            return " ".join(unique_kws[:2])

    # 3. Fallback to topic context or base tech query
    if topic_context:
        # Clean topic context slightly
        clean_topic = topic_context.lower()
        # Remove common news prefixes
        clean_topic = re.sub(r"^(deep dive\s*:\s*|news\s*:\s*)", "", clean_topic)
        # Use first 3 words of topic context + 'technology'
        words = clean_topic.split()
        if words:
            return " ".join(words[:3]) + " technology"

    return "technology"


def _filter_and_sort_candidates_by_relevance(chunk_text, candidates):
    """
    Scores the relevance of a list of candidates against the chunk text,
    sorting them by highest score. Returns a list of (score, candidate) tuples.
    """
    scored_candidates = []
    for c in candidates:
        desc = c.get("desc", "")
        if not desc:
            score = 5 # Default middle score if no description
        else:
            score = score_relevance(chunk_text, desc)
        scored_candidates.append((score, c))
    
    # Sort by score in descending order
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return scored_candidates


def _fetch_pexels_fallback(chunk, duration, is_video=False, is_longform=False, topic_context=""):
    cid = chunk.get("chunk_id")
    
    # Formulate search queries
    smart_query = _extract_clean_search_query(chunk, topic_context)
    primary_q = smart_query
    fallback_q = chunk.get("pexels_fallback", "technology")
    
    if primary_q == fallback_q:
        queries = [primary_q]
    else:
        queries = [primary_q, fallback_q]
        
    video_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.mp4")
    photo_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.jpg")
    
    orientation = "landscape" if is_longform else "portrait"
    
    if is_video:
        # 1. Search videos for all queries
        all_videos = []
        for query in queries:
            print(f"   Searching Pexels video for '{query}'...")
            videos = _search_pexels_videos(query, duration, {"orientation": orientation})
            all_videos.extend(videos)
            
        # Score and rank videos
        if all_videos:
            scored_videos = _filter_and_sort_candidates_by_relevance(chunk.get("text", ""), all_videos)
            for score, v in scored_videos:
                print(f"      -> Best video candidate: {v['desc']} (Score: {score}/10)")
                path = _download_video(v["link"], video_out)
                if path:
                    with _download_lock:
                        _used_media.add(v["id"])
                    return path, f"Video ({v['desc']}) [Score: {score}]", "video"
        
        # 2. Fallback to photos if no videos found
        all_photos = []
        for query in queries:
            print(f"   No video found. Searching Pexels photo for '{query}'...")
            photos = _search_pexels_photos(query, orientation=orientation)
            all_photos.extend(photos)
            
        if all_photos:
            scored_photos = _filter_and_sort_candidates_by_relevance(chunk.get("text", ""), all_photos)
            for score, p in scored_photos:
                print(f"      -> Best photo candidate (video fallback): {p['desc']} (Score: {score}/10)")
                path = _download_photo(p["link"], photo_out, is_longform=is_longform)
                if path:
                    with _download_lock:
                        _used_media.add(p["id"])
                    return path, f"Photo ({p['desc']}) [Score: {score}]", "photo"
    else:
        # 1. Search photos for all queries
        all_photos = []
        for query in queries:
            print(f"   Searching Pexels photo for '{query}'...")
            photos = _search_pexels_photos(query, orientation=orientation)
            all_photos.extend(photos)
            
        # Score and rank photos
        if all_photos:
            scored_photos = _filter_and_sort_candidates_by_relevance(chunk.get("text", ""), all_photos)
            for score, p in scored_photos:
                print(f"      -> Best photo candidate: {p['desc']} (Score: {score}/10)")
                path = _download_photo(p["link"], photo_out, is_longform=is_longform)
                if path:
                    with _download_lock:
                        _used_media.add(p["id"])
                    return path, f"Photo ({p['desc']}) [Score: {score}]", "photo"
                    
        # 2. Fallback to videos if no photos found
        all_videos = []
        for query in queries:
            print(f"   No photo found. Searching Pexels video for '{query}'...")
            videos = _search_pexels_videos(query, duration, {"orientation": orientation})
            all_videos.extend(videos)
            
        if all_videos:
            scored_videos = _filter_and_sort_candidates_by_relevance(chunk.get("text", ""), all_videos)
            for score, v in scored_videos:
                print(f"      -> Best video candidate (photo fallback): {v['desc']} (Score: {score}/10)")
                path = _download_video(v["link"], video_out)
                if path:
                    with _download_lock:
                        _used_media.add(v["id"])
                    return path, f"Video ({v['desc']}) [Score: {score}]", "video"
                    
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


def generate_premium_prompt_via_gemini(chunk_text, topic_context, global_style_guide, original_visual_prompt=None, aspect_ratio="9:16", is_video=False, is_hook=False):
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
        
    if is_hook:
        prompt += (
            "\nCRITICAL INSTRUCTION FOR THE HOOK VISUAL (First 3 seconds):\n"
            "This is the opening hook of the video, which must capture viewer attention instantly.\n"
            "- For privacy/security topics: Describe a macro close-up of a physical phone screen displaying a stylized 'Tracking Active' alert or warning popup, or a fast-paced zoom-in on a stylized security app icon.\n"
            "- For performance/utility topics: Describe a close-up of a device screen showcasing a dramatic peak value or a highly optimized interface.\n"
            "- Ensure the scene has dramatic dark cyber-neon lighting, vibrant contrast, cybernetic textures, and looks incredibly mysterious and high-intrigue. No generic stock photos.\n"
        )
        
    prompt += f"""
RULES for the generated prompt:
1. It MUST be highly relevant, visually representing the core concept or subject of: "{chunk_text_clean}".
2. Do NOT put any text, typography, subtitles, labels, logos, or watermarks in the prompt. Keep it purely visual.
3. Do NOT include faces of real people (e.g. Sam Altman, Sundar Pichai, Elon Musk). Instead, use generic descriptions (e.g., "a visionary CEO in a dark tech-noir room", "a researcher looking at a glowing holographic screen").
4. Describe the camera shot, angle, lighting, and lens details to make it look premium (e.g. "shot on 35mm lens, cinematic split lighting, shallow depth of field, subtle hand-held camera shake, volumetric dust particles").
5. Use vibrant, harmonious, modern color grading (e.g., cyber-cyan/amber contrast, deep emerald greens and dark charcoal, moody cyberpunk neon).
6. Format must be {orientation}.
7. For technical, cloud, coding, hardware, or software development concepts, prioritize concrete and realistic visual assets (e.g., clean code editor windows showing syntax-highlighted code, terminal windows with command line outputs, server rack units inside modern blue/orange data centers, system architecture block diagrams, database table relationships) rather than generic abstract stock concepts like glowing digital brains or neon particle vortexes.
8. Do NOT include any introductory or concluding text (e.g. "Here is your prompt:"). Return ONLY the raw prompt.

Output the visual prompt:"""

    # Try primary model, then fallback model on 429/rate-limit
    models_to_try = [target_model, "gemini-2.0-flash-lite"]
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(temperature=0.8)
            )
            res = response.text.strip().strip('"').strip("'")
            print(f"   [GEMINI PROMPT GEN] Created prompt ({model_name}): {res[:100]}...")
            return res
        except Exception as e:
            err_str = str(e).upper()
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"⚠️ Gemini prompt generation failed: 429 RESOURCE_EXHAUSTED. Trying fallback model...")
                continue  # Try the next model in the list
            print(f"⚠️ Gemini prompt generation failed: {e}. Using hardcoded fallback.")
            break
    # All models exhausted — use a hardcoded prompt
    vibe = global_style_guide if global_style_guide else "cinematic lighting, photorealistic, 8k"
    subj = original_visual_prompt_clean if original_visual_prompt_clean else chunk_text_clean
    return f"{subj}, {vibe}, {orientation}, highly detailed, photorealistic, no text"



def _generate_huggingface_image(prompt, output_path, aspect_ratio="9:16"):
    """Generate an image using Hugging Face FLUX.1 Schnell (free tier, needs HF_TOKEN)."""
    from config import HF_TOKEN
    if not HF_TOKEN:
        return None
    
    if aspect_ratio == "9:16":
        width, height = (1080, 1920)
    elif aspect_ratio == "16:9":
        width, height = (1920, 1080)
    else:
        width, height = (1024, 1024)
    
    try:
        print(f"     → Attempting Hugging Face FLUX.1 Schnell fallback...")
        resp = requests.post(
            "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt, "parameters": {"width": width, "height": height}},
            timeout=60
        )
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
            with open(output_path, "wb") as f:
                f.write(resp.content)
            print(f"  ✅ [huggingface] FLUX.1 Schnell generated successfully!")
            return output_path
        elif resp.status_code == 503:
            print(f"  ⚠️ [huggingface] Model loading (503). Skipping.")
        else:
            print(f"  ⚠️ [huggingface] Returned status: {resp.status_code}")
    except Exception as e:
        print(f"  ⚠️ [huggingface] Failed: {e}")
    return None


def _generate_pollinations_image(prompt, output_path, aspect_ratio="9:16"):
    """Free, no-key AI image generation fallback if Imagen, HuggingFace and Veo fail."""
    if aspect_ratio == "9:16":
        width, height = (1080, 1920)
    elif aspect_ratio == "16:9":
        width, height = (1920, 1080)
    else:
        width, height = (1024, 1024)
    import urllib.parse
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&private=true"
    
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"     → Attempting Pollinations AI fallback (attempt {attempt}/{max_attempts})...")
            resp = requests.get(url, timeout=45)
            if resp.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                return output_path
            elif resp.status_code == 429:
                wait = 15 * attempt
                print(f"  ⚠️ [pollinations] Rate limited (429). Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"  ⚠️ [pollinations] Attempt {attempt} returned status: {resp.status_code}")
        except Exception as e:
            print(f"  ⚠️ [pollinations] Attempt {attempt} failed: {e}")
        
        if attempt < max_attempts:
            time.sleep(10 * attempt)  # 10s, 20s backoff between retries
            
    return None


def _generate_cloudflare_image(prompt, output_path, aspect_ratio="9:16"):
    """Generate an image using Cloudflare Workers AI FLUX.1 Schnell (free tier, needs CF credentials)."""
    from config import CF_ACCOUNT_ID, CF_API_TOKEN
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return None
    
    try:
        print(f"     → Attempting Cloudflare Workers AI FLUX.1 Schnell fallback...")
        resp = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/black-forest-labs/flux-1-schnell",
            headers={
                "Authorization": f"Bearer {CF_API_TOKEN}",
                "Content-Type": "application/json"
			},
            json={"prompt": prompt},
            timeout=60
        )
        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "")
            if content_type.startswith("image"):
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                print(f"  ✅ [cloudflare] FLUX.1 Schnell generated successfully!")
                return output_path
            else:
                # Cloudflare may return JSON with base64 image
                try:
                    import base64
                    data = resp.json()
                    if data.get("success") and data.get("result", {}).get("image"):
                        img_bytes = base64.b64decode(data["result"]["image"])
                        with open(output_path, "wb") as f:
                            f.write(img_bytes)
                        print(f"  ✅ [cloudflare] FLUX.1 Schnell generated successfully (base64)!")
                        return output_path
                except Exception:
                    pass
                print(f"  ⚠️ [cloudflare] Unexpected response format: {content_type}")
        elif resp.status_code == 429:
            print(f"  ⚠️ [cloudflare] Rate limited (429). Skipping.")
        else:
            print(f"  ⚠️ [cloudflare] Returned status: {resp.status_code}")
    except Exception as e:
        print(f"  ⚠️ [cloudflare] Failed: {e}")
    return None


def _generate_fallback_image(prompt, output_path, aspect_ratio="9:16"):
    """Unified fallback: tries HuggingFace FLUX → Cloudflare FLUX → Pollinations."""
    path = _generate_huggingface_image(prompt, output_path, aspect_ratio=aspect_ratio)
    if path:
        return path, "HuggingFace (FLUX.1)"
    path = _generate_cloudflare_image(prompt, output_path, aspect_ratio=aspect_ratio)
    if path:
        return path, "Cloudflare (FLUX.1)"
    path = _generate_pollinations_image(prompt, output_path, aspect_ratio=aspect_ratio)
    if path:
        return path, "Pollinations"
    return None, None


def _generate_imagen3(prompt, output_path, topic_context="", global_style_guide="", visual_subject=None, aspect_ratio="9:16"):
    """
    Generates an image via Imagen 4.0 using the fully engineered prompt.
    Keeps signature compatibility for other modules like entity_fetcher.
    """
    print(f"🎨 Generating Imagen Image with prompt: {prompt[:80]}...")
    
    # Early exit if we already know Imagen is exhausted/unavailable for this run
    if os.environ.get("IMAGEN_QUOTA_EXHAUSTED"):
        print("  ⏭️ Imagen unavailable (paid plan required or quota exhausted). Skipping.")
        path, fb_source = _generate_fallback_image(prompt, output_path, aspect_ratio=aspect_ratio)
        if path:
            print(f"  ✅ Fallback generator ({fb_source}) succeeded.")
            return path
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
            # Detect "paid plans only" / INVALID_ARGUMENT errors and fast-fail all future calls
            if "paid plan" in err_str or ("invalid_argument" in err_str and "imagen" in err_str):
                print(f"  ⚠️ Imagen call failed on {model_name}: {e}")
                print("  🚨 Imagen requires a paid plan. Setting fast-fail flag for remaining chunks.")
                os.environ["IMAGEN_QUOTA_EXHAUSTED"] = "true"
                break
            print(f"  ⚠️ Imagen call failed on {model_name}: {e}")
            break
    # Fallback to HuggingFace/Cloudflare/Pollinations if Imagen fails
    print("  ⚠️ Imagen failed completely. Trying fallback generators...")
    path, fb_source = _generate_fallback_image(prompt, output_path, aspect_ratio=aspect_ratio)
    if path:
        print(f"  ✅ Fallback generator ({fb_source}) succeeded.")
        return path
        
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
            is_video=False,
            is_hook=(visual_mode == "nano_hook")
        )
        path = _generate_imagen3(custom_prompt, photo_out, aspect_ratio=orientation)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "photo"
            chunk["relevance_score"] = 10
            chunk["source"] = f"Imagen ({visual_mode})"
            return chunk
            
        # Fallback to HuggingFace FLUX / Pollinations AI
        print(f"Chunk {cid} -> Imagen failed, trying HuggingFace/Pollinations fallback...")
        path, fb_source = _generate_fallback_image(custom_prompt, photo_out, aspect_ratio=orientation)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "photo"
            chunk["relevance_score"] = 9
            chunk["source"] = f"{fb_source} ({visual_mode})"
            return chunk
            
        # Pexels fallback for photo mode!
        print(f"Chunk {cid} -> Imagen/Pollinations failed, falling back to Pexels search...")
        path, source_desc, v_type = _fetch_pexels_fallback(chunk, dur, is_video=False, is_longform=is_longform, topic_context=topic_context)
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
            
        # Fallback to HuggingFace FLUX / Pollinations AI
        print(f"Chunk {cid} -> Imagen evidence failed, trying HuggingFace/Pollinations fallback...")
        path, fb_source = _generate_fallback_image(custom_prompt, photo_out, aspect_ratio=orientation)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "photo"
            chunk["relevance_score"] = 9
            chunk["source"] = f"{fb_source} (nano_evidence)"
            return chunk

        # Pexels fallback for evidence mode!
        print(f"Chunk {cid} -> Imagen/Pollinations evidence failed, falling back to Pexels search...")
        path, source_desc, v_type = _fetch_pexels_fallback(chunk, dur, is_video=False, is_longform=is_longform, topic_context=topic_context)
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
            
        # Fallback to HuggingFace FLUX / Pollinations AI
        print(f"Chunk {cid} -> Imagen fallback failed, trying HuggingFace/Pollinations fallback...")
        path, fb_source = _generate_fallback_image(custom_img_prompt, photo_out, aspect_ratio=orientation)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "photo"
            chunk["relevance_score"] = 8
            chunk["source"] = f"{fb_source} (fallback from veo)"
            return chunk

        # Pexels fallback for video mode!
        print(f"Chunk {cid} -> Veo/Imagen/Pollinations failed, falling back to Pexels search...")
        path, source_desc, v_type = _fetch_pexels_fallback(chunk, dur, is_video=True, is_longform=is_longform, topic_context=topic_context)
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
    global_style = generate_visual_style_guide(topic_context, is_longform=is_longform)
    
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
        # Cooldown between chunks to respect API rate limits (Imagen, Veo, HuggingFace, Pollinations)
        api_source_prefixes = [
            "Imagen", "Veo", "HuggingFace", "Cloudflare", "Pollinations",
            "Real Article Screenshot"
        ]
        
        if i < len(chunks) - 1:
            chunk_source = chunk.get("source", "")
            if any(chunk_source.startswith(prefix) for prefix in api_source_prefixes):
                cooldown = 15 if chunk_source.startswith("Pollinations") else 10
                print(f"  -> AI generated asset ({chunk_source}). Cooling down for {cooldown}s...")
                time.sleep(cooldown)
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
