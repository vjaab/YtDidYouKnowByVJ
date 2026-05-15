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
    Asks Gemini to define a consistent, HEADLINE-SPECIFIC visual 'vibe' for the whole video.
    This ensures all AI-generated images share a palette, lighting style, and are relevant to the story.
    """
    print("🎨 Designing Global Visual Style Guide...")
    try:
        target_model = "gemini-2.0-flash"
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
                r = requests.get(video.uri, headers={"x-goog-api-key": GEMINI_API_KEY})
                if r.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(r.content)
                    return output_path
        print(f"⚠️ Veo operation finished but no video found. Result: {operation.result}")
    except Exception as e:
        print(f"⚠️ Veo generation failed: {e}")
    return None

# Pexels Stock Footage functions removed in favor of Generative AI.

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
            target_model = "gemini-2.0-flash"
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


def _generate_imagen3(chunk_text, output_path, topic_context="", global_style_guide="", visual_subject=None):
    topic = detect_topic(topic_context)
    
    # ── DAILY AESTHETIC ROTATION ──
    day_idx = datetime.now().day
    daily_style = DAILY_AESTHETIC_MATRIX[day_idx % len(DAILY_AESTHETIC_MATRIX)]
    
    style_suffix = f", {global_style_guide}" if global_style_guide else f", {daily_style['lighting']} lighting, {daily_style['mood']} style, 8k, photorealistic"
    
    # Use passed subject or extract if missing
    if not visual_subject:
        visual_subject = _extract_visual_subject(topic_context)
    
    print(f"  -> Visual Subject: '{visual_subject}'")
    
    if topic:
        template = random.choice(TOPIC_PROMPT_TEMPLATES[topic])
        best_prompt = f"{template.format(name=visual_subject)}, {style_suffix}, {daily_style['motif']} elements, news editorial style, 9:16 vertical, ultra realistic, no text"
        print(f"  -> Detected Topic: {topic}. Using themed prompt for {visual_subject} with Day {day_idx} style.")
    else:
        # Ask Gemini to craft a headline-specific Imagen prompt
        topic_prompt = f"Topic Context: {topic_context}. The image MUST be highly relevant to this specific story.\n" if topic_context else ""
        prompt_builder = f"""Create a detailed Imagen 3 prompt for a YouTube Shorts background image.

NEWS HEADLINE: '{topic_context}'
CURRENT CHUNK TEXT: '{chunk_text}'

{topic_prompt}Requirements:
- Photorealistic, cinematic, 9:16 vertical
- Style guide to follow: {global_style_guide}
- The image MUST visually represent the specific topic/entities in the headline above
- Include relevant logos, products, or symbolic imagery that viewers will immediately associate with the story
- No text, no watermarks, no faces of real people
- High detail, dramatic lighting, news editorial style
- RETURN ONLY the prompt text. No introductory sentence."""
        
        best_prompt = chunk_text  # Default
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
                prompt=best_prompt,
                config=genai.types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="9:16",
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
                continue
            break
    return None

    # Updated to 4.0 models as 3.0 is missing from the API in this environment
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
    
    # ── SWITCH-BACK LOGIC ──────────────────────────────────────────────────
    if not visual_subject:
        visual_subject = _extract_visual_subject(topic_context)
    
    headline = topic_context if topic_context else "Scientific Research"
    
    # ── DAILY AESTHETIC ROTATION ──
    day_idx = datetime.now().day
    style = DAILY_AESTHETIC_MATRIX[day_idx % len(DAILY_AESTHETIC_MATRIX)]
    print(f"  📅 Day {day_idx} Aesthetic: {style['mood']} ({style['lighting']})")

    veo_broll_prompt = (
        f"Cinematic, high-definition 4k video in a {style['mood']} style. "
        f"Subject: {visual_subject}. Visuals should feature {style['motif']}, with {style['lighting']} lighting. "
        f"Motion: {style['camera']}. Style: Minimalist, clean. No text in video. Video language: English."
    )
    
    nano_evidence_prompt = (
        f"A professional macro photograph of a scientific research paper titled '{headline}'. "
        f"Style: {style['mood']}. Focus on a specific complex diagram or a paragraph of mathematical equations. "
        f"Realistic paper texture with {style['lighting']} lighting. Format: Vertical 9:16 for mobile."
    )
    
    nano_concept_prompt = (
        f"Cinematic, high-definition 9:16 vertical image in a {style['mood']} style. "
        f"Subject: {visual_subject}. Visuals should feature {style['motif']}, with {style['lighting']} lighting. "
        f"Style: Minimalist, clean. No text."
    )

    if visual_mode == "nano_hook" or visual_mode == "nano_concept":
        print(f"Chunk {cid} -> MODE: {visual_mode}")
        prompt = nano_concept_prompt if visual_mode == "nano_concept" else nano_concept_prompt + " Dramatic hero shot."
        path = _generate_imagen3(prompt, photo_out, topic_context, global_style_guide, visual_subject=visual_subject)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "photo"
            chunk["relevance_score"] = 10
            chunk["source"] = f"Imagen ({visual_mode})"
            return chunk

    elif visual_mode == "nano_evidence":
        # Prefer REAL article screenshot if available in script_data
        real_screenshot = script_data.get("screenshot_path")
        if real_screenshot and os.path.exists(real_screenshot):
            print(f"Chunk {cid} -> MODE: nano_evidence (Using REAL screenshot)")
            chunk["visual_path"] = real_screenshot
            chunk["visual_type"] = "photo"
            chunk["relevance_score"] = 10
            chunk["source"] = "Real Article Screenshot"
            return chunk

        print(f"Chunk {cid} -> MODE: nano_evidence (Using AI Macro Fallback)")
        path = _generate_imagen3(nano_evidence_prompt, photo_out, topic_context, global_style_guide)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "photo"
            chunk["relevance_score"] = 10
            chunk["source"] = "Imagen (nano_evidence)"
            return chunk

    elif visual_mode == "veo_concept" or visual_mode == "veo_cta":
        print(f"Chunk {cid} -> MODE: {visual_mode}")
        path = _generate_veo_video(veo_broll_prompt, video_out, aspect_ratio=orientation)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "video"
            chunk["relevance_score"] = 10
            chunk["source"] = f"Veo ({visual_mode})"
            return chunk
            
        # Fallback to Imagen if Veo fails
        print(f"Chunk {cid} -> Veo failed, falling back to Imagen")
        path = _generate_imagen3(nano_concept_prompt, photo_out, topic_context, global_style_guide, visual_subject=visual_subject)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "photo"
            chunk["relevance_score"] = 8
            chunk["source"] = "Imagen (fallback from veo)"
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
    
    for i, chunk in enumerate(chunks):
        if "pexels_primary" not in chunk:
            chunk["pexels_primary"] = " ".join(chunk["text"].split()[:3])
            chunk["pexels_fallback"] = "technology"

        # Determine visual mode based on index
        total_chunks = len(chunks)
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
