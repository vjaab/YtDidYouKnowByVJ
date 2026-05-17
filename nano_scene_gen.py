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


def _generate_missing_prompts(chunks, headline, style_guide):
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

    prompt = f"""You are a cinematic visual director for YouTube Shorts.

HEADLINE: {headline}
VISUAL STYLE: {style_guide}

For each sentence below, generate a specific, cinematic IMAGE PROMPT that visually represents
EXACTLY what is being spoken in that sentence. The image will be used as a fullscreen 9:16 
background behind a talking head.

RULES:
- Each prompt must be SPECIFIC to the sentence content (not generic)
- NO text in images. NO faces of real people. NO watermarks.
- Photorealistic, cinematic lighting, 9:16 vertical format, 8K quality
- Include relevant objects, environments, or symbolic imagery
- Keep each prompt under 80 words

SENTENCES:
{chunk_list}

Return ONLY a JSON array of objects, one per sentence, in order:
[
  {{"chunk_id": 1, "prompt": "Cinematic close-up of..."}},
  ...
]"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
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
        prompt_map = {p.get("chunk_id", i + 1): p.get("prompt", "") for i, p in enumerate(prompts)}
        for c in missing:
            cid = c.get("chunk_id", 0)
            if cid in prompt_map and prompt_map[cid]:
                c["nano_visual_prompt"] = prompt_map[cid]
            else:
                # Fallback: use the chunk text itself as a basic prompt
                c["nano_visual_prompt"] = (
                    f"Cinematic visualization of: {c.get('text', 'technology')[:60]}. "
                    f"Photorealistic, 9:16 vertical, {style_guide}, no text, no faces."
                )
        print(f"  ✅ Generated {len(prompts)} nano-scene prompts via Gemini Flash.")
    except Exception as e:
        print(f"  ⚠️ Batch prompt generation failed: {e}. Using fallback prompts.")
        for c in missing:
            c["nano_visual_prompt"] = (
                f"Cinematic visualization of: {c.get('text', 'technology')[:60]}. "
                f"Photorealistic, 9:16 vertical, {style_guide}, no text, no faces."
            )

    return chunks


def _generate_imagen_image(prompt, output_path):
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
                    aspect_ratio="9:16",
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


def generate_nano_scene_visuals(chunks, headline, style_guide=""):
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
    chunks = _generate_missing_prompts(chunks, headline, style_guide)

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

        path = _generate_imagen_image(prompt, output_path)

        if path:
            chunk["visual_path"] = path
            chunk["visual_type"] = "photo"
            chunk["source"] = "Nano-Scene (Imagen 4.0)"
            chunk["relevance_score"] = 10
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

    return chunks


def _fill_visual_gaps(chunks):
    """Forward-fill: propagate the last successful visual to any gap chunks."""
    last_path = None
    last_type = "photo"
    for c in chunks:
        if c.get("visual_path"):
            last_path = c["visual_path"]
            last_type = c.get("visual_type", "photo")
        elif last_path:
            c["visual_path"] = last_path
            c["visual_type"] = last_type
            c["source"] = c.get("source", "Nano-Scene (gap-filled)")
