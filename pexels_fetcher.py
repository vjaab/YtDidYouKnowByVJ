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
def _extract_visual_subject(headline):
    """Use Gemini to extract the primary visual subject/entity from a headline."""
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
            return subject
    except Exception as e:
        print(f"  ⚠️ Entity extraction failed: {e}")
    
    # Fallback: use first 4 meaningful words from headline
    words = [w for w in (headline or "").split() if len(w) > 2][:4]
    return " ".join(words) if words else "AI Technology"


def _generate_imagen3(chunk_text, output_path, topic_context="", global_style_guide=""):
    topic = detect_topic(topic_context)
    
    style_suffix = f", {global_style_guide}" if global_style_guide else ", cinematic lighting, professional photography, 8k, photorealistic"
    
    # Extract the actual visual subject from the headline (not the full headline)
    visual_subject = _extract_visual_subject(topic_context)
    print(f"  -> Visual Subject Extracted: '{visual_subject}'")
    
    if topic:
        template = random.choice(TOPIC_PROMPT_TEMPLATES[topic])
        best_prompt = f"{template.format(name=visual_subject)}, {style_suffix}, news editorial style, 9:16 vertical, ultra realistic, no text"
        print(f"  -> Detected Topic: {topic}. Using themed prompt for {visual_subject}.")
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
def fetch_chunk_visual(chunk, script_data, topic_context="", global_style_guide="", is_longform=False, visual_mode="veo_concept"):
    """
    Executes the Visual Fetching Decision Tree with Switch-Back Logic
    """
    from entity_fetcher import fetch_person_photo, fetch_company_logo
    cid = chunk["chunk_id"]
    text = chunk["text"]
    dur = chunk["duration"]
    
    orientation = "landscape" if is_longform else "portrait"
    
    video_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.mp4")
    photo_out = os.path.join(OUTPUT_DIR, f"chunk_{cid}_{TODAY}.jpg")
    
    # ── SWITCH-BACK LOGIC ──────────────────────────────────────────────────
    visual_subject = _extract_visual_subject(topic_context)
    headline = topic_context if topic_context else "Scientific Research"

    veo_broll_prompt = f"Cinematic, high-definition 4k video in a futuristic tech-docu style. Subject: {visual_subject}. Visuals should feature glowing neural networks, sleek 3D data visualizations, or professional laboratory settings with soft teal and orange lighting. Motion: Subtle camera track-in or slow pan. Style: Minimalist, clean, tech-evangelist aesthetic. No text in video. Video language: English."
    
    nano_evidence_prompt = f"A professional macro photograph of a scientific research paper titled '{headline}'. Shallow depth of field, focusing on a specific complex diagram or a paragraph of mathematical equations. Realistic paper texture with subtle digital overlays of scanning lines and data points. Lighting: Cool office morning light. Format: Vertical 9:16 for mobile."
    
    nano_concept_prompt = f"Cinematic, high-definition 9:16 vertical image in a futuristic tech-docu style. Subject: {visual_subject}. Visuals should feature glowing neural networks, sleek 3D data visualizations, or professional laboratory settings with soft teal and orange lighting. Style: Minimalist, clean, tech-evangelist aesthetic. No text."

    if visual_mode == "nano_hook" or visual_mode == "nano_concept":
        print(f"Chunk {cid} -> MODE: {visual_mode}")
        prompt = nano_concept_prompt if visual_mode == "nano_concept" else nano_concept_prompt + " Dramatic hero shot."
        path = _generate_imagen3(prompt, photo_out, topic_context, global_style_guide)
        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "photo"
            chunk["relevance_score"] = 10
            chunk["source"] = f"Imagen ({visual_mode})"
            return chunk

    elif visual_mode == "nano_evidence":
        print(f"Chunk {cid} -> MODE: nano_evidence")
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
        path = _generate_imagen3(nano_concept_prompt, photo_out, topic_context, global_style_guide)
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
            v_mode = "nano_evidence"
        elif i == total_chunks - 1:
            v_mode = "veo_cta"
        else:
            # Alternate concept loop
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
            "Veo (veo_concept)", "Veo (veo_cta)", "Imagen (fallback from veo)"
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
