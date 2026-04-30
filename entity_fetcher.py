import os
import requests
from io import BytesIO
from PIL import Image
from config import OUTPUT_DIR
from pexels_fetcher import _search_pexels_photos, _download_photo

def _save_image_from_url(url, output_path, is_logo=False):
    """Downloads and saves an image, ensuring proper formatting."""
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content)).convert("RGBA")
            
            if is_logo:
                # If logo, keep transparency or put on dark background? User asked for 2 options, we just save it as PNG
                img.save(output_path, "PNG")
            else:
                img = img.convert("RGB")
                img.save(output_path, "JPEG", quality=90)
            return output_path
    except Exception as e:
        print(f"Failed to download image from {url}: {e}")
    return None

def fetch_person_photo(person):
    name = person.get("name")
    twitter_handle = person.get("twitter_handle")
    wiki_slug = person.get("wikipedia_slug") or name.replace(" ", "_")
    
    print(f"Fetching photo for person: {name}...")
    output_path = os.path.join(OUTPUT_DIR, f"person_{name.replace(' ', '_')}.jpg")
    if os.path.exists(output_path):
        return output_path


    # PRIORITY 2: Wikipedia API
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_slug}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if "thumbnail" in data and "source" in data["thumbnail"]:
                img_url = data["thumbnail"]["source"]
                # Try to get larger image if possible by removing size limit from URL
                img_url = img_url.replace(data["thumbnail"]["width"], "400") # rough hack
                path = _save_image_from_url(img_url, output_path)
                if path:
                    print(f"  -> Found Wikipedia photo for {name}")
                    return path
    except Exception as e:
        print(f"  Wikipedia API fetch failed: {e}")

    # PRIORITY 4: Pexels fallback
    print(f"  -> Falling back to Pexels for {name}...")
    result = _search_pexels_photos(f"{name} portrait face")
    if result:
        path = _download_photo(result[0]["link"], output_path)
        if path:
            return path

    print(f"  -> Could not find photo for {name}")
    return None

def fetch_company_logo(company):
    name = company.get("name")
    domain = company.get("domain") or company.get("company_domain")
    if not domain:
        domain = f"{name.lower().replace(' ', '')}.com"
    
    print(f"Fetching logo for company: {name} (domain: {domain})...")
    output_path = os.path.join(OUTPUT_DIR, f"company_{name.replace(' ', '_')}.png")
    if os.path.exists(output_path):
        return output_path

    # PRIORITY 1: Clearbit Logo API
    try:
        url = f"https://logo.clearbit.com/{domain}?size=600"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            path = _save_image_from_url(url, output_path, is_logo=True)
            if path:
                print(f"  -> Found Clearbit logo for {name}")
                return path
    except Exception as e:
        print(f"  Clearbit fetch failed: {e}")

    # PRIORITY 2: Wikipedia API
    try:
        wiki_slug = name.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_slug}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if "thumbnail" in data and "source" in data["thumbnail"]:
                img_url = data["thumbnail"]["source"]
                path = _save_image_from_url(img_url, output_path, is_logo=True)
                if path:
                    print(f"  -> Found Wikipedia logo for {name}")
                    return path
    except Exception as e:
        print(f"  Wikipedia fetch failed: {e}")

    # PRIORITY 3: Pexels fallback (as photo)
    print(f"  -> Falling back to Pexels for {name}...")
    pexels_out = os.path.join(OUTPUT_DIR, f"company_{name.replace(' ', '_')}_office.jpg")
    search_query = company.get("hq_pexels_search") or f"{name} office headquarters"
    result = _search_pexels_photos(search_query)
    if result:
        path = _download_photo(result[0]["link"], pexels_out)
        if path:
            return path

    print(f"  -> Could not find logo/hq for {name}")
    return None

def fetch_all_entities(script_data):
    """
    Downloads all person photos and company logos.
    Updates the script_data object with local paths in place.
    """
    for person in script_data.get("people", []):
        path = fetch_person_photo(person)
        if path:
            person["local_image_path"] = path

    for company in script_data.get("companies", []):
        path = fetch_company_logo(company)
        if path:
            if path.endswith(".png"):
                company["local_logo_path"] = path
            else:
                company["local_hq_path"] = path

    for entity in script_data.get("key_entities", []):
        # We can try to fetch logo for anything, clearbit is domain-based and cheap
        path = fetch_company_logo(entity)
        if path:
            if path.endswith(".png"):
                entity["local_logo_path"] = path
            else:
                entity["local_hq_path"] = path

    return script_data

def get_retention_layers_config():
    """
    Coordination point for kinetic layers and pacing.
    Called by main.py to ensure engagement triggers are active.
    """
    from config import ENABLE_KINETIC_CAPTIONS, ENABLE_AUDIO_DUCKING, ENABLE_PERIODIC_CUTS
    
    return {
        "kinetic_captions": ENABLE_KINETIC_CAPTIONS,
        "audio_ducking": ENABLE_AUDIO_DUCKING,
        "camera_cuts": ENABLE_PERIODIC_CUTS,
        "pacing_mode": "kinetic_high_energy"
    }
