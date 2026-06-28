import os
import requests
import hashlib
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from config import OUTPUT_DIR, BASE_DIR
from pexels_fetcher import _generate_imagen3

def _generate_local_fallback_logo(name, output_path, is_logo=True):
    """
    Generates a beautiful local placeholder image using PIL when all downloads and AI generations fail.
    This guarantees the file exists on disk and the entity validation passes.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        size = (400, 400)
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Color palette for placeholder background based on name hash
        name_hash = int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16)
        colors = [
            (255, 68, 68),   # Red
            (0, 229, 255),   # Cyan
            (0, 255, 127),   # Green
            (255, 215, 0),   # Gold
            (224, 170, 255), # Purple
            (255, 105, 180), # Pink
            (30, 144, 255)   # Dodger Blue
        ]
        bg_color = colors[name_hash % len(colors)]
        
        # Draw rounded rectangle background card
        draw.rounded_rectangle([10, 10, 390, 390], radius=50, fill=bg_color)
        draw.rounded_rectangle([10, 10, 390, 390], radius=50, outline=(255, 255, 255, 200), width=8)
        
        # Get initials (up to 2 characters)
        initials = "".join([part[0].upper() for part in name.split() if part])[:2]
        if not initials:
            initials = "AI"
            
        font = None
        font_paths = [
            os.path.join(BASE_DIR, "assets", "fonts", "Montserrat-ExtraBold.ttf"),
            os.path.join(BASE_DIR, "assets", "fonts", "Montserrat-Bold.ttf"),
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, 180)
                    break
                except:
                    pass
        if not font:
            try:
                font = ImageFont.load_default()
            except:
                pass
                
        if font:
            bb = draw.textbbox((0, 0), initials, font=font)
            tw = bb[2] - bb[0]
            th = bb[3] - bb[1]
            tx = (400 - tw) // 2 - bb[0]
            ty = (400 - th) // 2 - bb[1]
            draw.text((tx, ty), initials, font=font, fill=(255, 255, 255, 255))
            
        if is_logo:
            img.save(output_path, "PNG")
        else:
            img = img.convert("RGB")
            img.save(output_path, "JPEG", quality=90)
            
        print(f"  🎨 Generated local fallback logo/placeholder for {name} -> {output_path}")
        return output_path
    except Exception as e:
        print(f"  ⚠️ Failed to generate local fallback logo for {name}: {e}")
        return None


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

def _get_wikipedia_slug(name):
    """Queries Wikipedia Search API to find the most relevant title/slug for an entity name."""
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": name,
            "format": "json",
            "srlimit": 1
        }
        r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            search_results = data.get("query", {}).get("search", [])
            if search_results:
                return search_results[0]["title"].replace(" ", "_")
    except Exception as e:
        print(f"  Wikipedia Search API lookup failed for {name}: {e}")
    return name.replace(" ", "_")

def _fetch_ddg_image(name, output_path, is_logo=False):
    """Fallback: DuckDuckGo Instant Answer API for entity image from internet."""
    try:
        ddg_url = f"https://api.duckduckgo.com/?q={name}&format=json"
        r = requests.get(ddg_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            ddg_img = data.get("Image")
            if ddg_img:
                if not ddg_img.startswith("http"):
                    ddg_img = "https://duckduckgo.com" + ddg_img
                path = _save_image_from_url(ddg_img, output_path, is_logo=is_logo)
                if path:
                    print(f"  -> Found DuckDuckGo image/logo from internet for {name}")
                    return path
    except Exception as e:
        print(f"  DuckDuckGo API lookup failed for {name}: {e}")
    return None

def fetch_person_photo(person):
    name = person.get("name")
    twitter_handle = person.get("twitter_handle")
    wiki_slug = person.get("wikipedia_slug") or _get_wikipedia_slug(name)
    
    print(f"Fetching photo for person: {name} (slug: {wiki_slug})...")
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
                img_url = img_url.replace(str(data["thumbnail"]["width"]), "400") # rough hack
                path = _save_image_from_url(img_url, output_path)
                if path:
                    print(f"  -> Found Wikipedia photo for {name}")
                    return path
    except Exception as e:
        print(f"  Wikipedia API fetch failed: {e}")

    # PRIORITY 3: DuckDuckGo Fallback
    path = _fetch_ddg_image(name, output_path, is_logo=False)
    if path:
        return path

    # PRIORITY 4: Generative AI fallback (was Pexels)
    print(f"  -> Falling back to Generative AI for {name}...")
    path = _generate_imagen3(f"Professional portrait photo of {name}", output_path, topic_context=name)
    if path:
        return path

    # PRIORITY 5: Local beautiful placeholder fallback
    path = _generate_local_fallback_logo(name, output_path, is_logo=False)
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

    # PRIORITY 1: Hunter.io Logo API (Clearbit successor)
    try:
        url = f"https://logos.hunter.io/{domain}"
        path = _save_image_from_url(url, output_path, is_logo=True)
        if path:
            print(f"  -> Found Hunter.io logo for {name}")
            return path
    except Exception as e:
        print(f"  Hunter.io fetch failed: {e}")

    # PRIORITY 2: Wikipedia API
    try:
        wiki_slug = company.get("wikipedia_slug") or _get_wikipedia_slug(name)
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

    # PRIORITY 3: DuckDuckGo Fallback
    path = _fetch_ddg_image(name, output_path, is_logo=True)
    if path:
        return path

    # PRIORITY 4: Generative AI fallback (was Pexels)
    print(f"  -> Falling back to Generative AI for {name}...")
    imagen_out = os.path.join(OUTPUT_DIR, f"company_{name.replace(' ', '_')}_office.jpg")
    search_query = company.get("hq_pexels_search") or f"{name} office headquarters"
    path = _generate_imagen3(f"Professional corporate office headquarters for {search_query}", imagen_out, topic_context=name)
    if path:
        return path

    # PRIORITY 5: Local beautiful placeholder fallback
    path = _generate_local_fallback_logo(name, output_path, is_logo=True)
    if path:
        return path

    print(f"  -> Could not find logo/hq for {name}")
    return None

def fetch_all_entities(script_data):
    """
    Downloads all person photos and company logos.
    Updates the script_data object with local paths in place.
    """
    # ── GATHER ALL POTENTIAL ENTITIES ─────────────────────────────────────────
    # If there are companies_mentioned or tools_mentioned that are not in key_entities/companies/people,
    # add them to key_entities to ensure their logos are fetched and they are rendered.
    companies_mentioned = script_data.get("companies_mentioned", [])
    tools_mentioned = script_data.get("tools_mentioned", [])
    
    # Check if we have any entity fields populated at all
    has_any = (
        len(script_data.get("companies", [])) > 0 or
        len(script_data.get("people", [])) > 0 or
        len(script_data.get("key_entities", [])) > 0 or
        len(companies_mentioned) > 0 or
        len(tools_mentioned) > 0
    )
    
    if not has_any:
        # Pre-emptively scan the script text or headline for known tech companies to add
        script_text = (script_data.get("script") or "").lower()
        headline_text = (script_data.get("original_news_headline") or "").lower()
        
        known_tech = [
            {"name": "Apple", "domain": "apple.com", "description": "Tech Company"},
            {"name": "Google", "domain": "google.com", "description": "Tech Company"},
            {"name": "Samsung", "domain": "samsung.com", "description": "Tech Company"},
            {"name": "OpenAI", "domain": "openai.com", "description": "AI Company"},
            {"name": "ChatGPT", "domain": "openai.com", "description": "AI Tool"},
            {"name": "Microsoft", "domain": "microsoft.com", "description": "Tech Company"},
            {"name": "Meta", "domain": "meta.com", "description": "Tech Company"},
            {"name": "Nvidia", "domain": "nvidia.com", "description": "Tech Company"},
            {"name": "Claude", "domain": "anthropic.com", "description": "AI Tool"},
            {"name": "Anthropic", "domain": "anthropic.com", "description": "AI Company"},
            {"name": "Tesla", "domain": "tesla.com", "description": "Tech Company"},
            {"name": "Amazon", "domain": "amazon.com", "description": "Tech Company"},
        ]
        
        found_any = False
        for tech in known_tech:
            if tech["name"].lower() in script_text or tech["name"].lower() in headline_text:
                if "companies" not in script_data or not isinstance(script_data["companies"], list):
                    script_data["companies"] = []
                script_data["companies"].append(tech)
                found_any = True
                break
                
        if not found_any:
            # Absolute fallback to our channel brand
            if "companies" not in script_data or not isinstance(script_data["companies"], list):
                script_data["companies"] = []
            script_data["companies"].append({
                "name": "VJ AI News",
                "domain": "youtube.com",
                "description": "Daily Tech News"
            })
            print("  ⚠️ No entities found in script_data. Added VJ AI News as fallback entity.")

    key_entities = script_data.get("key_entities", [])
    if not isinstance(key_entities, list):
        key_entities = []
        script_data["key_entities"] = key_entities

    existing_names = {ent.get("name", "").lower() for ent in key_entities if isinstance(ent, dict)}
    for p in script_data.get("people", []):
        if isinstance(p, dict):
            existing_names.add(p.get("name", "").lower())
    for c in script_data.get("companies", []):
        if isinstance(c, dict):
            existing_names.add(c.get("name", "").lower())

    for c in companies_mentioned:
        if c and isinstance(c, str) and c.lower() not in existing_names:
            key_entities.append({"name": c, "type": "COMPANY", "description": "Tech Company"})
            existing_names.add(c.lower())

    for t in tools_mentioned:
        if t and isinstance(t, str) and t.lower() not in existing_names:
            key_entities.append({"name": t, "type": "TOOL", "description": "AI Tool"})
            existing_names.add(t.lower())

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
