import os
import io
import requests
from PIL import Image
from datetime import datetime
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, OUTPUT_DIR

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
TODAY = datetime.now().strftime("%Y-%m-%d")


def _save_image_from_url(url, output_path, target_ratio=9/16):
    """Download an image URL, crop to 9:16, save to disk. Returns path or None."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")

        # Centre-crop to 9:16
        w, h = img.size
        target_h = int(w * 16 / 9)
        if target_h <= h:
            top  = (h - target_h) // 2
            img  = img.crop((0, top, w, top + target_h))
        else:
            target_w = int(h * 9 / 16)
            left = (w - target_w) // 2
            img  = img.crop((left, 0, left + target_w, h))

        img = img.resize((1080, 1920), Image.LANCZOS)
        img.save(output_path, "JPEG", quality=90)
        return output_path
    except Exception as e:
        print(f"Image download failed ({url[:60]}): {e}")
        return None


def _search_pexels(query, count=4):
    """Search Pexels for `count` landscape/portrait images matching the query."""
    if not PEXELS_API_KEY:
        return []
    try:
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": PEXELS_API_KEY}
        params  = {"query": query, "per_page": count * 2, "orientation": "portrait"}
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if r.status_code != 200:
            print(f"Pexels error {r.status_code}: {r.text[:100]}")
            return []
        photos = r.json().get("photos", [])
        # Prefer large portrait photos
        return [p["src"].get("large2x", p["src"]["large"]) for p in photos[:count]]
    except Exception as e:
        print(f"Pexels search failed: {e}")
        return []


def _search_pexels_multi(keywords, total=4):
    """
    Use multiple keyword queries to get varied images.
    Rotates through the keywords to get 1 image per query variation.
    """
    urls = []
    search_terms = keywords[:4] if keywords else ["technology", "artificial intelligence"]

    for i, kw in enumerate(search_terms):
        found = _search_pexels(kw, count=2)
        if found:
            # Pick the one not already used to keep variety
            for url in found:
                if url not in urls:
                    urls.append(url)
                    break
        if len(urls) >= total:
            break

    # If still short, general fallback search
    if len(urls) < total:
        fallback_terms = ["technology", "artificial intelligence", "digital", "innovation"]
        for term in fallback_terms:
            if len(urls) >= total:
                break
            found = _search_pexels(term, count=2)
            for url in found:
                if url not in urls:
                    urls.append(url)
                    break

    return urls[:total]


def generate_images(prompts, image_url=None, keywords=None):
    """
    Image strategy (priority order):
    1. Direct news article image URL (already downloaded previously)
    2. Pexels search using script keywords → 4 varied images
    3. Imagen 4.0 AI generation (fallback)
    """

    # ── 1. Original news article image ──────────────────────────────────────
    if image_url and image_url.startswith("http"):
        try:
            out = os.path.join(OUTPUT_DIR, f"news_bg_{TODAY}.jpg")
            path = _save_image_from_url(image_url, out)
            if path:
                print(f"Using original news article image: {image_url[:60]}")
                return [path] * 4
        except Exception as e:
            print(f"News image failed: {e}")

    # ── 2. Pexels keyword search ─────────────────────────────────────────────
    if PEXELS_API_KEY and keywords:
        print(f"Searching Pexels for keywords: {keywords[:4]}")
        pexels_urls = _search_pexels_multi(keywords, total=4)
        if pexels_urls:
            paths = []
            for i, url in enumerate(pexels_urls):
                out  = os.path.join(OUTPUT_DIR, f"pexels_{i+1}_{TODAY}.jpg")
                path = _save_image_from_url(url, out)
                if path:
                    paths.append(path)
            if paths:
                # Pad to 4 frames if fewer downloaded
                while len(paths) < 4:
                    paths.append(paths[-1])
                print(f"Pexels: {len(set(paths))} unique images downloaded.")
                return paths[:4]

    # ── 3. Imagen 4.0 fallback ───────────────────────────────────────────────
    print("Falling back to Imagen 4.0 image generation...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    style_suffix = (
        ", photorealistic, cinematic lighting, dramatic atmosphere, "
        "hyper detailed, professional photography, no text, no watermarks, "
        "no logos, no people's faces. RETURN ONLY the prompt text, no intro."
    )

    generated_paths = []
    previous_path   = None

    # Updated to 4.0 models as 3.0 is missing from the API in this environment
    models_to_try = [
        "imagen-4.0-fast-generate-001",
        "imagen-4.0-generate-001",
        "imagen-4.0-ultra-generate-001"
    ]
    
    for i, prompt in enumerate(prompts):
        out  = os.path.join(OUTPUT_DIR, f"frame_{i+1}_{TODAY}.png")
        success = False
        
        for model_name in models_to_try:
            try:
                print(f"Generating image {i+1}/4 using {model_name}...")
                result = client.models.generate_images(
                    model=model_name,
                    prompt=prompt + style_suffix,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="9:16",
                        output_mime_type="image/jpeg",
                    )
                )
                for gen_img in result.generated_images:
                    with open(out, "wb") as f:
                        f.write(gen_img.image.image_bytes)
                    break
                    
                generated_paths.append(out)
                previous_path = out
                success = True
                break  # Break out of the models loop on success
                
            except Exception as e:
                err_str = str(e).lower()
                print(f"Imagen image {i+1} failed using {model_name}: {e}")
                if "429" in err_str and ("quota" in err_str or "exhausted" in err_str):
                    print(f"Quota exceeded for {model_name}. Switching to fallback model.")
                    continue  # Try next model
                
                # If it's a different error, stop trying models and fall back to previous image
                break
                
        if not success:
            if previous_path:
                generated_paths.append(previous_path)
            else:
                return None

    return generated_paths
