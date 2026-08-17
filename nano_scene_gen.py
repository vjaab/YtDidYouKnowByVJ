"""
nano_scene_gen.py — Per-Sentence "Nano-Scene" Visual Generation Engine.

Generates one Imagen 4.0 background image per subtitle chunk (sentence),
creating the @vaibhavsisinty-style visual sync where backgrounds change
every 2-3 seconds to match exactly what's being spoken.

Pipeline:
  1. For each chunk that has a `nano_visual_prompt`, generate an Imagen image
  2. If a chunk lacks a prompt, use Gemini Flash to generate one on-the-fly
  3. If Imagen quota exhausts, reuse the last successful image (graceful degradation)
"""

import os
import time
import random
from datetime import datetime
from google import genai
from config import GEMINI_API_KEY, OUTPUT_DIR

TODAY = datetime.now().strftime("%Y-%m-%d")

client = genai.Client(api_key=GEMINI_API_KEY)


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
    Main entry point: generates one Imagen 4.0 background image per chunk.

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
    print(f"\n🎬 NANO-SCENE ENGINE: Generating {total} per-sentence backgrounds...")

    # Step 1: Ensure all chunks have nano_visual_prompts
    chunks = _generate_missing_prompts(chunks, headline, style_guide, aspect_ratio=aspect_ratio)

    # Step 2: Generate images
    last_successful_path = None
    generated_count = 0
    reused_count = 0

    for i, chunk in enumerate(chunks):
        cid = chunk.get("chunk_id", i + 1)
        prompt = chunk.get("nano_visual_prompt", "")

        if not prompt:
            # No prompt available — reuse last image
            if last_successful_path:
                chunk["visual_path"] = last_successful_path
                chunk["visual_type"] = "photo"
                chunk["source"] = "Nano-Scene (reused)"
                reused_count += 1
            continue

        output_path = os.path.join(OUTPUT_DIR, f"nano_scene_{cid}_{TODAY}.jpg")

        print(f"  [{i + 1}/{total}] Generating: {prompt[:70]}...")

        path = _generate_imagen_image(prompt, output_path, aspect_ratio=aspect_ratio)

        if not path:
            # Fallback chain: HuggingFace FLUX → Cloudflare FLUX → Pollinations AI
            print(f"  [{i + 1}/{total}] Imagen failed, trying HuggingFace/Cloudflare/Pollinations fallback...")
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
            # Use the visual_type from the AI prompt, fallback to "photo"
            chunk["visual_type"] = chunk.get("visual_type", "photo")
            chunk["source"] = source_name
            chunk["relevance_score"] = relevance
            last_successful_path = path
            generated_count += 1
        elif last_successful_path:
            # Imagen failed — gracefully reuse last successful image
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

        # Smart throttling: 5s between Imagen calls to avoid rate limits
        if i < total - 1 and path:
            time.sleep(5)

    print(f"\n  ✅ Nano-Scene Generation Complete: {generated_count} generated, {reused_count} reused, "
          f"{total - generated_count - reused_count} failed")

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


# Call diversity enforcement after generation
def generate_nano_scene_visuals(chunks, headline, style_guide="", aspect_ratio="9:16"):
    # ... existing code ...
    
    # At the end of the function, before returning:
    _ensure_visual_type_diversity(chunks)
    _apply_visual_type_styling(chunks)
    
    return chunks
