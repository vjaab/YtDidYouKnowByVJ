import os
import io
import time
import requests
from PIL import Image
from datetime import datetime
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, OUTPUT_DIR

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
TODAY = datetime.now().strftime("%Y-%m-%d")



def _generate_huggingface_image(prompt, output_path, aspect_ratio="9:16"):
    """Generate an image using Hugging Face FLUX.1 Schnell (free tier, needs HF_TOKEN)."""
    from config import HF_TOKEN
    if not HF_TOKEN:
        return None
    
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
                print(f"Using original news article image for first slots: {image_url[:60]}")
                # We return it twice, but still try to get variety for the other 2 frames
                article_paths = [path] * 2
                
                # ── 2a. Pexels fallback for remaining frames ──────────────────────
                if PEXELS_API_KEY and keywords:
                    pexels_urls = _search_pexels_multi(keywords, total=2)
                    for i, p_url in enumerate(pexels_urls):
                        out_p = os.path.join(OUTPUT_DIR, f"pexels_mixed_{i+1}_{TODAY}.jpg")
                        p_path = _save_image_from_url(p_url, out_p)
                        if p_path:
                            article_paths.append(p_path)
                
                # Pad to 4
                while len(article_paths) < 4:
                    article_paths.append(path)
                return article_paths[:4]
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
            # Fallback chain: HuggingFace FLUX → Cloudflare FLUX → Pollinations AI
            print(f"Imagen failed for image {i+1}, trying HuggingFace/Cloudflare/Pollinations fallback...")
            path = _generate_huggingface_image(prompt + style_suffix, out, aspect_ratio="9:16")
            if not path:
                path = _generate_cloudflare_image(prompt + style_suffix, out, aspect_ratio="9:16")
            if not path:
                path = _generate_pollinations_image(prompt + style_suffix, out, aspect_ratio="9:16")
            if path:
                generated_paths.append(path)
                previous_path = path
                success = True
            elif previous_path:
                generated_paths.append(previous_path)
            else:
                return None

    return generated_paths
