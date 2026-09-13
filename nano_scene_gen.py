"""
nano_scene_gen.py — Per-Sentence "Nano-Scene" Visual Generation Engine.

Generates one Imagen 4.0 background image per subtitle chunk (sentence),
creating the @vaibhavsisinty-style visual sync where backgrounds change
every 2-3 seconds to match exactly what's being spoken.

Pipeline:
  1. For each chunk that has a `nano_visual_prompt`, first try Pexels/Pixabay with relevance review
  2. If no relevant stock image found, generate via Imagen 4.0
  3. If Imagen quota exhausts, fallback chain: HuggingFace → Cloudflare → Pollinations
  4. If all AI generation fails, reuse last successful image (graceful degradation)
"""

import os
import io
import json
import time
import random
import requests
import shutil
import tempfile
from datetime import datetime
from PIL import Image, ImageOps
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, OUTPUT_DIR, HF_TOKEN, CF_ACCOUNT_ID, CF_API_TOKEN, HAS_CF_FALLBACK

TODAY = datetime.now().strftime("%Y-%m-%d")

client = genai.Client(api_key=GEMINI_API_KEY)

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")

_NANO_VISUAL_CACHE = {}


def _extract_chunk_keywords(chunk_text, topic_context=""):
    """Extract relevant search keywords from chunk text for Pexels search."""
    import re
    
    tech_keywords = [
        "artificial intelligence", "machine learning", "deep learning", "neural network",
        "llm", "gpt", "claude", "gemini", "transformer", "generative ai",
        "python", "javascript", "typescript", "rust", "go", "programming",
        "github", "open source", "repository", "code", "software development",
        "api", "microservices", "cloud", "aws", "gcp", "azure", "kubernetes",
        "docker", "container", "devops", "ci/cd", "deployment",
        "database", "sql", "postgresql", "mongodb", "redis",
        "frontend", "react", "vue", "nextjs", "tailwind",
        "backend", "nodejs", "fastapi", "django", "flask",
        "mobile", "ios", "android", "flutter", "react native",
        "cybersecurity", "encryption", "authentication", "oauth",
        "blockchain", "crypto", "web3", "smart contract",
        "data science", "pandas", "numpy", "visualization",
        "computer vision", "nlp", "natural language processing",
        "gpu", "tpu", "semiconductor", "chip", "nvidia", "intel", "amd",
        "robot", "humanoid", "automation", "autonomous",
        "server", "data center", "infrastructure", "cooling",
        "quantum", "algorithm", "terminal", "coding", "programmer",
        "fiber optic", "satellite", "dna", "microscope", "supercomputer"
    ]
    
    text_lower = (chunk_text + " " + topic_context).lower()
    found_keywords = []
    
    for kw in tech_keywords:
        if kw in text_lower:
            found_keywords.append(kw)
    
    if not found_keywords:
        words = re.findall(r'\b[A-Z][a-z]+\b|\b[A-Z]{2,}\b', chunk_text)
        found_keywords = list(set(words))[:3]
    
    return found_keywords[:3]


def _search_pexels_photos(query, orientation="portrait", per_page=5):
    """Search Pexels for photos matching the query."""
    if not PEXELS_API_KEY:
        return []
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": per_page, "orientation": orientation},
            timeout=15
        )
        if r.status_code != 200:
            return []
            
        results = []
        for p in r.json().get("photos", []):
            url = p.get("src", {}).get("large2x") or p.get("src", {}).get("large")
            if url:
                results.append({
                    "id": f"pexels_{p.get('id')}",
                    "link": url,
                    "desc": p.get("alt", query),
                    "type": "photo"
                })
        return results
    except Exception as e:
        print(f"  ⚠️ Pexels search failed for '{query}': {e}")
        return []


def _search_pixabay_photos(query, orientation="portrait", per_page=5):
    """Search Pixabay for photos matching the query."""
    if not PIXABAY_API_KEY:
        return []
    try:
        r = requests.get(
            "https://pixabay.com/api/",
            params={
                "key": PIXABAY_API_KEY,
                "q": query,
                "per_page": per_page,
                "orientation": orientation,
                "image_type": "photo",
                "order": "popular",
                "category": "computer"
            },
            timeout=15
        )
        if r.status_code != 200:
            return []
            
        results = []
        for p in r.json().get("hits", []):
            url = p.get("largeImageURL") or p.get("webformatURL")
            if url:
                results.append({
                    "id": f"pixabay_{p.get('id')}",
                    "link": url,
                    "desc": p.get("tags", query),
                    "type": "photo"
                })
        return results
    except Exception as e:
        print(f"  ⚠️ Pixabay search failed for '{query}': {e}")
        return []


def _download_image(url, output_path):
    """Download image from URL to local path."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=30, stream=True)
        if r.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"  ⚠️ Failed to download image: {e}")
    return False


def _crop_to_9_16(image_path, output_path):
    """Crop image to 9:16 aspect ratio (1080x1920)."""
    try:
        img = Image.open(image_path).convert("RGB")
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
        return True
    except Exception as e:
        print(f"  ⚠️ Failed to crop image: {e}")
        return False


def _review_image_relevance_gemini(image_path, chunk_text, topic_context, gemini_api_key=None):
    """
    Use Gemini Vision to review if the downloaded image is relevant to the chunk topic.
    Returns (is_relevant: bool, confidence: float, reason: str)
    """
    if not gemini_api_key:
        gemini_api_key = GEMINI_API_KEY
    if not gemini_api_key:
        return True, 0.5, "No API key for review"
    
    try:
        review_client = genai.Client(api_key=gemini_api_key)
        
        img = Image.open(image_path)
        img_w, img_h = img.size
        
        max_dim = 2048
        if max(img_w, img_h) > max_dim:
            scale = max_dim / max(img_w, img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_bytes = buf.getvalue()
        
        prompt = f"""Analyze this image and determine if it's visually relevant to the spoken content.

Spoken text: "{chunk_text}"
Topic context: "{topic_context}"

Return ONLY a JSON object:
{{
  "is_relevant": true/false,
  "confidence": 0.0-1.0,
  "reason": "Brief explanation of why the image is or isn't relevant"
}}

Consider:
- Does the image show concepts, tools, logos, or visuals related to the spoken text?
- Is it generic stock photo filler, or does it have specific relevance to the technical topic?
- Would a viewer understand the connection between this image and what's being said?"""
        
        response = review_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        raw = response.text.strip()
        if "{" in raw and "}" in raw:
            raw = raw[raw.find("{"):raw.rfind("}") + 1]
        result = json.loads(raw)
        
        is_relevant = result.get("is_relevant", False)
        confidence = result.get("confidence", 0.5)
        reason = result.get("reason", "No reason provided")
        
        return is_relevant, confidence, reason
        
    except Exception as e:
        err_str = str(e)
        # If quota exceeded, skip review and accept the image (assume it's relevant)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            print(f"  ⚠️ Gemini Vision quota exceeded, skipping review and accepting image")
            return True, 0.8, "Quota exceeded - auto-accepted"
        print(f"  ⚠️ Gemini Vision review failed: {e}")
        return True, 0.5, f"Review failed: {e}"


def _fetch_and_review_pexels_candidates(chunk, topic_context, max_attempts=3):
    """
    Fetch candidate images from Pexels/Pixabay, review with Gemini Vision,
    and return the first relevant image path. Returns None if all fail.
    """
    import json
    
    cache_key = f"{chunk.get('chunk_id')}_{topic_context[:50]}"
    if cache_key in _NANO_VISUAL_CACHE:
        return _NANO_VISUAL_CACHE[cache_key]
    
    keywords = _extract_chunk_keywords(chunk.get("text", ""), topic_context)
    if not keywords:
        keywords = ["technology", "AI", "software development"]
    
    all_candidates = []
    for keyword in keywords:
        pexels_results = _search_pexels_photos(keyword, orientation="portrait", per_page=3)
        all_candidates.extend(pexels_results)
        
        if len(all_candidates) < max_attempts:
            pixabay_results = _search_pixabay_photos(keyword, orientation="portrait", per_page=3)
            all_candidates.extend(pixabay_results)
        
        if len(all_candidates) >= max_attempts * 2:
            break
    
    seen = set()
    unique_candidates = []
    for c in all_candidates:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique_candidates.append(c)
    
    if not unique_candidates:
        return None
    
    for attempt, candidate in enumerate(unique_candidates[:max_attempts * 2]):
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
            
            if not _download_image(candidate["link"], temp_path):
                continue
            
            if not _crop_to_9_16(temp_path, temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
                continue
            
            is_relevant, confidence, reason = _review_image_relevance_gemini(
                temp_path, chunk.get("text", ""), topic_context
            )
            
            print(f"  🔍 Pexels review (attempt {attempt+1}): {'RELEVANT' if is_relevant else 'NOT RELEVANT'} (confidence: {confidence:.2f}) - {reason[:80]}")
            
            if is_relevant and confidence >= 0.6:
                output_path = os.path.join(OUTPUT_DIR, f"nano_pexels_{chunk.get('chunk_id')}_{TODAY}.jpg")
                import shutil
                shutil.move(temp_path, output_path)
                
                _NANO_VISUAL_CACHE[cache_key] = output_path
                return output_path
            
            try:
                os.unlink(temp_path)
            except:
                pass
                
        except Exception as e:
            print(f"  ⚠️ Error processing candidate image: {e}")
            continue
    
    return None


def _generate_missing_prompts(chunks, headline, style_guide, aspect_ratio="9:16"):
    """
    For chunks that don't have a nano_visual_prompt (older schema or alignment fallback),
    use Gemini Flash to batch-generate visual prompts for all of them in one call.
    """
    missing = [c for c in chunks if not c.get("nano_visual_prompt")]
    if not missing:
        return chunks

    print(f"  🎨 Generating nano-scene prompts for {len(missing)} chunks without prompts...")

    # Build batch context
    chunk_list = "\n".join([
        f"[{c.get('chunk_id', i+1)}] \"{c.get('text', '')}\""
        for i, c in enumerate(missing)
    ])

    format_desc = "16:9 landscape format" if aspect_ratio == "16:9" else "9:16 vertical format"
    prompt = f"""You are a senior Hollywood director and AI visual prompt designer acting as the expert Visual Director.
HEADLINE: {headline}
VISUAL STYLE/VIBE: {style_guide}

For each sentence below, analyze the spoken technical concept and select the most suitable visual representation. You must generate detailed metadata to teach the concept visually, passing the "Muted Viewer Test" (a mobile viewer must understand the key idea even if audio is muted).

VISUAL SELECTION LOGIC:
1. Video: Use when explaining real-world scenarios, future technology concepts, AI agents, robots, data centers, autonomous systems, human-AI interaction, cybersecurity attacks, or software development workflows.
2. AI Image: Use when explaining conceptual ideas, architectural components, hardware designs, side-by-side comparisons, or historical timelines.
3. Whiteboard: Use when explaining algorithms, system design, software engineering logic, network routing, database replication, or mathematical equations.
4. Infographic: Use when comparing statistics, listing feature tables, showing performance benchmarks, key-value configurations, or percentage changes.
5. Diagram: Use when illustrating complex system architectures, server-client interactions, database failovers, or data streaming pipes.
6. Animated UI Mockup: Use when showing settings menus, app navigation, toggle switches, console logs, or command line commands.

TECHNOLOGY VISUALIZATION RULES:
- Avoid generic stock-style visuals (e.g. generic glowing brains, random gears, standard robots with blue eyes).
- Programming/Coding -> Realistic code editors (e.g. VS Code screen with syntax highlighted Python/TypeScript code and terminal showing logs).
- Cybersecurity -> Shield overlays, lock icons, firewall block diagrams, threat maps, simulated terminal attacks, decrypting animation.
- Databases/Storage -> Tabular structures, query flow arrows, database nodes, disk read/write animations.
- Cloud/Infrastructure -> Connected server racks, optical fiber routes, cloud icons with api endpoints.
- Network -> Routers, packets (pulses of light), routing tables, network map.
- AI/ML -> Neural networks, training dataset matrices, weights/biases graphs, training loop animations.

PROMPT RULES:
1. The nano_visual_prompt MUST be directly relevant to the sentence content.
2. NO text, typography, logos, or watermarks in any generated prompts.
3. NO faces of real people (e.g. Sam Altman, Elon Musk). Use generic descriptions (e.g. "a tech executive looking at a futuristic interface").
4. Include premium details: camera shot, angle, lens (e.g. "close-up, 35mm lens"), dramatic lighting (e.g. "cinematic split lighting"), color grading (e.g. "vibrant cyber-cyan/amber contrast"), and textures (e.g. "volumetric dust particles in light beams").
5. The format must be {format_desc}.
6. Keep each prompt under 80 words.

SENTENCES:
{chunk_list}

Return ONLY a JSON array of objects, one per sentence, in order:
[
  {{
    "chunk_id": 1,
    "scene_objective": "What technical concept must be understood here",
    "visual_type": "Video|AI Image|Whiteboard|Infographic|Diagram|Animated UI Mockup|Code Snippet|Screen Recording|Flowchart|Terminal Output|GitHub UI|Side-by-side Comparison|Architecture Diagram",
    "nano_visual_prompt": "Cinematic close-up of...",
    "on_screen_elements": ["labels", "arrows", "highlights", "icons", "charts", "code snippets"],
    "camera_motion": "Slow zoom|Dolly-in|Orbit|Pan|Tracking shot|None",
    "transition": "Match cut|Zoom transition|Morph|Swipe|Data stream transition|Neural network transition"
  }},
  ...
]"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(temperature=0.7)
        )
        raw = response.text.strip()
        # Extract JSON array
        if "[" in raw and "]" in raw:
            raw = raw[raw.find("["):raw.rfind("]") + 1]

        import json
        prompts = json.loads(raw)

        # Apply prompts back to chunks
        prompt_map = {p.get("chunk_id", i + 1): p for i, p in enumerate(prompts)}
        for c in missing:
            cid = c.get("chunk_id", 0)
            if cid in prompt_map and prompt_map[cid]:
                p_data = prompt_map[cid]
                c["scene_objective"] = p_data.get("scene_objective", "Understand spoken concept")
                c["visual_type"] = p_data.get("visual_type", "AI Image")
                c["nano_visual_prompt"] = p_data.get("nano_visual_prompt") or p_data.get("prompt") or ""
                
                # Ensure on_screen_elements is a list of strings
                c["on_screen_elements"] = p_data.get("on_screen_elements", [])
                if isinstance(c["on_screen_elements"], str):
                    c["on_screen_elements"] = [c["on_screen_elements"]]
                elif not isinstance(c["on_screen_elements"], list):
                    c["on_screen_elements"] = []

                c["camera_motion"] = p_data.get("camera_motion", "None")
                c["transition"] = p_data.get("transition", "Match cut")
            else:
                # Fallback: use the chunk text itself as a basic prompt
                c["scene_objective"] = "Understand spoken concept"
                c["visual_type"] = "AI Image"
                c["nano_visual_prompt"] = (
                    f"Cinematic visualization of: {c.get('text', 'technology')[:60]}. "
                    f"Photorealistic, {aspect_ratio} format, {style_guide}, no text, no faces."
                )
                c["on_screen_elements"] = []
                c["camera_motion"] = "None"
                c["transition"] = "Match cut"
        print(f"  ✅ Generated {len(prompts)} nano-scene prompts via Gemini Flash.")
    except Exception as e:
        print(f"  ⚠️ Batch prompt generation failed: {e}. Using fallback prompts.")
        for c in missing:
            c["scene_objective"] = "Understand spoken concept"
            c["visual_type"] = "AI Image"
            c["nano_visual_prompt"] = (
                f"Cinematic visualization of: {c.get('text', 'technology')[:60]}. "
                f"Photorealistic, {aspect_ratio} format, {style_guide}, no text, no faces."
            )
            c["on_screen_elements"] = []
            c["camera_motion"] = "None"
            c["transition"] = "Match cut"

    return chunks



def _generate_huggingface_image(prompt, output_path, aspect_ratio="9:16"):
    """Generate an image using Hugging Face FLUX.1 Schnell (free tier, needs HF_TOKEN)."""
    from config import HF_TOKEN
    if not HF_TOKEN:
        return None
    
    import requests
    width, height = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)
    
    try:
        print(f"     → Attempting Hugging Face FLUX.1 Schnell fallback...")
        resp = requests.post(
            "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
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
    width, height = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)
    import requests
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
    from config import HAS_CF_FALLBACK, CF_ACCOUNT_ID, CF_API_TOKEN
    if not HAS_CF_FALLBACK:
        return None
    
    import requests
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

def _generate_imagen_image(prompt, output_path, aspect_ratio="9:16"):
    """Generate a single image via Imagen 4.0. Returns path on success, None on failure."""

    # Early exit if quota is exhausted for this run
    if os.environ.get("IMAGEN_QUOTA_EXHAUSTED"):
        return None

    models_to_try = [
        "imagen-4.0-fast-generate-001",
        "imagen-4.0-generate-001",
        "imagen-4.0-ultra-generate-001",
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
                ),
            )
            for gen_img in result.generated_images:
                with open(output_path, "wb") as f:
                    f.write(gen_img.image.image_bytes)
                return output_path
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str and ("quota" in err_str or "exhausted" in err_str):
                print(f"  ⚠️ Imagen quota exhausted on {model_name}. Trying next model...")
                continue
            elif "429" in err_str:
                # Rate limited but not quota-exhausted — wait and retry
                print(f"  ⏳ Imagen rate limited. Waiting 15s...")
                time.sleep(15)
                continue
            else:
                print(f"  ⚠️ Imagen failed ({model_name}): {e}")
                break

    return None


def generate_nano_scene_visuals(chunks, headline, style_guide="", aspect_ratio="9:16"):
    """
    Main entry point: generates one background image per chunk (2-3 second visual sync).
    
    Priority order:
    1. Pexels/Pixabay stock photos with Gemini Vision relevance review
    2. Imagen 4.0 AI generation
    3. HuggingFace FLUX.1 Schnell
    4. Cloudflare Workers AI FLUX.1 Schnell
    5. Pollinations AI (free, no-key)
    6. Reuse last successful image (graceful degradation)

    Args:
        chunks: List of chunk dicts with 'chunk_id', 'text', and optionally 'nano_visual_prompt'
        headline: The news headline (used for visual context)
        style_guide: Global visual style guide string

    Returns:
        chunks: The same list, updated with 'visual_path', 'visual_type', and 'source' fields
    """
    if not chunks:
        return chunks

    total = len(chunks)
    print(f"\n🎬 NANO-SCENE ENGINE: Generating {total} per-sentence backgrounds (Pexels + AI fallback)...")

    # Step 1: Ensure all chunks have nano_visual_prompts
    chunks = _generate_missing_prompts(chunks, headline, style_guide, aspect_ratio=aspect_ratio)

    # Step 2: Generate images with Pexels-first strategy
    last_successful_path = None
    generated_count = 0
    reused_count = 0
    pexels_success_count = 0

    for i, chunk in enumerate(chunks):
        cid = chunk.get("chunk_id", i + 1)
        prompt = chunk.get("nano_visual_prompt", "")
        chunk_text = chunk.get("text", "")

        if not prompt:
            # No prompt available — reuse last image
            if last_successful_path:
                chunk["visual_path"] = last_successful_path
                chunk["visual_type"] = "photo"
                chunk["source"] = "Nano-Scene (reused)"
                reused_count += 1
            continue

        output_path = os.path.join(OUTPUT_DIR, f"nano_scene_{cid}_{TODAY}.jpg")

        print(f"  [{i + 1}/{total}] Processing: {chunk_text[:60]}...")

        # ── Strategy 1: Pexels/Pixabay with Gemini Vision Review ────────────────
        print(f"     → Trying Pexels/Pixabay stock photos with relevance review...")
        pexels_path = _fetch_and_review_pexels_candidates(chunk, headline, max_attempts=3)
        
        if pexels_path:
            chunk["visual_path"] = pexels_path
            chunk["visual_type"] = chunk.get("visual_type", "photo")
            chunk["source"] = "Nano-Scene (Pexels/Pixabay + Reviewed)"
            chunk["relevance_score"] = 10
            last_successful_path = pexels_path
            generated_count += 1
            pexels_success_count += 1
            print(f"     ✅ Pexels image approved and saved!")
        else:
            # ── Strategy 2: AI Generation Fallback Chain ─────────────────────────
            print(f"     → Pexels unavailable/rejected, trying AI generation...")
            
            path = _generate_imagen_image(prompt, output_path, aspect_ratio=aspect_ratio)

            if not path:
                # Fallback chain: HuggingFace FLUX → Cloudflare FLUX → Pollinations AI
                print(f"     → Imagen failed, trying HuggingFace/Cloudflare/Pollinations fallback...")
                path = _generate_huggingface_image(prompt, output_path, aspect_ratio=aspect_ratio)
                if path:
                    source_name = "Nano-Scene (HuggingFace FLUX.1)"
                    relevance = 9
                else:
                    path = _generate_cloudflare_image(prompt, output_path, aspect_ratio=aspect_ratio)
                    if path:
                        source_name = "Nano-Scene (Cloudflare FLUX.1)"
                        relevance = 9
                    else:
                        path = _generate_pollinations_image(prompt, output_path, aspect_ratio=aspect_ratio)
                        source_name = "Nano-Scene (Pollinations AI)"
                        relevance = 9
            else:
                source_name = "Nano-Scene (Imagen 4.0)"
                relevance = 10

            if path:
                chunk["visual_path"] = path
                chunk["visual_type"] = chunk.get("visual_type", "photo")
                chunk["source"] = source_name
                chunk["relevance_score"] = relevance
                last_successful_path = path
                generated_count += 1
            elif last_successful_path:
                # All generation failed — gracefully reuse last successful image
                chunk["visual_path"] = last_successful_path
                chunk["visual_type"] = "photo"
                chunk["source"] = "Nano-Scene (reused)"
                chunk["relevance_score"] = 7
                reused_count += 1
            else:
                # No images generated at all yet — mark as failed
                chunk["visual_path"] = None
                chunk["visual_type"] = None
                chunk["source"] = "Failed"
                chunk["relevance_score"] = 0

        # Smart throttling: 5s between generation calls to avoid rate limits
        if i < total - 1 and chunk.get("visual_path"):
            time.sleep(5)

    print(f"\n  ✅ Nano-Scene Generation Complete:")
    print(f"     - Pexels/Pixabay (reviewed): {pexels_success_count}")
    print(f"     - AI Generated: {generated_count - pexels_success_count}")
    print(f"     - Reused: {reused_count}")
    print(f"     - Failed: {total - generated_count - reused_count}")

    # Fill any remaining gaps (chunks that failed and had no predecessor)
    _fill_visual_gaps(chunks)
    
    # Ensure visual type diversity
    _ensure_visual_type_diversity(chunks)
    
    # Apply visual type specific styling metadata
    _apply_visual_type_styling(chunks)

    return chunks


def _fill_visual_gaps(chunks):
    """Robust two-pass visual gap filler."""
    first_path = None
    first_type = "photo"
    for c in chunks:
        if c.get("visual_path") and os.path.exists(c["visual_path"]):
            first_path = c["visual_path"]
            first_type = c.get("visual_type", "photo")
            break

    if not first_path:
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
            c["source"] = c.get("source") or "Nano-Scene (gap-filled)"


def _ensure_visual_type_diversity(chunks):
    """
    Post-process chunks to ensure visual type diversity.
    Forces a mix of visual types to prevent monotony.
    """
    # Visual types that should be distributed throughout
    visual_types = [
        "Video", "AI Image", "Whiteboard", "Infographic", 
        "Diagram", "Animated UI Mockup", "Code Snippet", 
        "Screen Recording", "Flowchart", "Terminal Output", 
        "GitHub UI", "Side-by-side Comparison", "Architecture Diagram"
    ]
    
    # Track used types
    used_types = set()
    type_counts = {}
    
    for i, chunk in enumerate(chunks):
        current_type = chunk.get("visual_type", "photo")
        used_types.add(current_type)
        type_counts[current_type] = type_counts.get(current_type, 0) + 1
    
    # If we only have 1-2 types, diversify
    if len(used_types) <= 2 and len(chunks) > 3:
        print(f"   🔄 Diversifying visual types (currently: {used_types})")
        
        # Assign types in a round-robin fashion for better distribution
        target_types = visual_types[:min(len(chunks), len(visual_types))]
        random.shuffle(target_types)
        
        for i, chunk in enumerate(chunks):
            if i < len(target_types):
                chunk["visual_type"] = target_types[i]
                chunk["source"] = chunk.get("source", "").replace("(photo)", f"({target_types[i]})")
    
    # Log diversity stats
    final_types = [c.get("visual_type", "photo") for c in chunks]
    type_dist = {}
    for t in final_types:
        type_dist[t] = type_dist.get(t, 0) + 1
    print(f"   📊 Visual Type Distribution: {type_dist}")


def _apply_visual_type_styling(chunks):
    """
    Apply visual type specific metadata for downstream rendering.
    This helps video_gen.py apply appropriate styling per visual type.
    """
    type_styles = {
        "Screen Recording": {
            "render_style": "screen_recording",
            "overlay_elements": ["cursor", "window_chrome", "highlight_region"],
            "camera_motion": "Pan"
        },
        "Terminal Output": {
            "render_style": "terminal",
            "overlay_elements": ["prompt", "command", "output", "cursor_blink"],
            "camera_motion": "None"
        },
        "Code Snippet": {
            "render_style": "code_editor",
            "overlay_elements": ["syntax_highlight", "line_numbers", "highlight_line"],
            "camera_motion": "Slow zoom"
        },
        "Terminal Output": {
            "render_style": "terminal",
            "overlay_elements": ["prompt", "command", "output"],
            "camera_motion": "None"
        },
        "Whiteboard": {
            "render_style": "whiteboard",
            "overlay_elements": ["hand_drawn", "arrows", "annotations"],
            "camera_motion": "Orbit"
        },
        "Diagram": {
            "render_style": "diagram",
            "overlay_elements": ["labels", "connections", "highlight_path"],
            "camera_motion": "Dolly-in"
        },
        "Architecture Diagram": {
            "render_style": "architecture",
            "overlay_elements": ["service_boxes", "data_flow", "legend"],
            "camera_motion": "Orbit"
        },
        "Flowchart": {
            "render_style": "flowchart",
            "overlay_elements": ["decision_diamonds", "process_boxes", "flow_arrows"],
            "camera_motion": "Pan"
        },
        "Screen Recording": {
            "render_style": "screen_recording",
            "overlay_elements": ["cursor", "click_highlight", "window_chrome"],
            "camera_motion": "Tracking shot"
        },
        "GitHub UI": {
            "render_style": "github_ui",
            "overlay_elements": ["repo_header", "file_tree", "code_view"],
            "camera_motion": "Slow zoom"
        },
        "Side-by-side Comparison": {
            "render_style": "comparison",
            "overlay_elements": ["vs_divider", "left_labels", "right_labels"],
            "camera_motion": "None"
        },
        "Animated UI Mockup": {
            "render_style": "ui_mockup",
            "overlay_elements": ["tap_indicators", "screen_transitions"],
            "camera_motion": "Match cut"
        },
        "Infographic": {
            "render_style": "infographic",
            "overlay_elements": ["charts", "icons", "stat_highlights"],
            "camera_motion": "Dolly-in"
        }
    }
    
    for chunk in chunks:
        vtype = chunk.get("visual_type", "photo")
        if vtype in type_styles:
            chunk["render_style"] = type_styles[vtype]["render_style"]
            chunk["overlay_elements"] = type_styles[vtype]["overlay_elements"]
            chunk["camera_motion"] = type_styles[vtype]["camera_motion"]
