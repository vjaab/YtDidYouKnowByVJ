import os
import requests
import hashlib
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from config import OUTPUT_DIR, BASE_DIR
from pexels_fetcher import _generate_imagen3

TECH_ENTITY_MAPPING = {
    # OpenAI
    "openai": {
        "domain": "openai.com",
        "wiki_slug": "OpenAI",
        "keywords": "OpenAI logo",
        "description": "AI Research Company",
        "type": "COMPANY"
    },
    "chatgpt": {
        "domain": "openai.com",
        "wiki_slug": "ChatGPT",
        "keywords": "ChatGPT logo",
        "description": "AI Chatbot Tool",
        "type": "TOOL"
    },
    "gpt-4": {
        "domain": "openai.com",
        "wiki_slug": "GPT-4",
        "keywords": "OpenAI GPT-4 logo",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    "gpt-4o": {
        "domain": "openai.com",
        "wiki_slug": "GPT-4o",
        "keywords": "OpenAI GPT-4o logo",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    "sora": {
        "domain": "openai.com",
        "wiki_slug": "Sora_(text-to-video_model)",
        "keywords": "OpenAI Sora logo",
        "description": "AI Video Generator",
        "type": "TOOL"
    },
    "dall-e": {
        "domain": "openai.com",
        "wiki_slug": "DALL-E",
        "keywords": "OpenAI DALL-E logo",
        "description": "AI Image Generator",
        "type": "TOOL"
    },
    "dall-e 3": {
        "domain": "openai.com",
        "wiki_slug": "DALL-E",
        "keywords": "OpenAI DALL-E logo",
        "description": "AI Image Generator",
        "type": "TOOL"
    },
    # Google
    "google": {
        "domain": "google.com",
        "wiki_slug": "Google",
        "keywords": "Google logo",
        "description": "Tech Giant",
        "type": "COMPANY"
    },
    "gemini": {
        "domain": "google.com",
        "wiki_slug": "Gemini_(chatbot)",
        "keywords": "Google Gemini logo",
        "description": "AI Chatbot Tool",
        "type": "TOOL"
    },
    "gemini 1.5": {
        "domain": "google.com",
        "wiki_slug": "Gemini_(chatbot)",
        "keywords": "Google Gemini logo",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    "gemini 1.5 pro": {
        "domain": "google.com",
        "wiki_slug": "Gemini_(chatbot)",
        "keywords": "Google Gemini logo",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    "gemini 1.5 flash": {
        "domain": "google.com",
        "wiki_slug": "Gemini_(chatbot)",
        "keywords": "Google Gemini logo",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    "imagen": {
        "domain": "google.com",
        "wiki_slug": "Google_Imagen",
        "keywords": "Google Imagen logo",
        "description": "AI Image Generator",
        "type": "TOOL"
    },
    "veo": {
        "domain": "google.com",
        "wiki_slug": "Google",
        "keywords": "Google Veo logo",
        "description": "AI Video Generator",
        "type": "TOOL"
    },
    "google colab": {
        "domain": "google.com",
        "wiki_slug": "Google_Colab",
        "keywords": "Google Colab logo",
        "description": "Hosted Notebook Service",
        "type": "TOOL"
    },
    # Anthropic
    "anthropic": {
        "domain": "anthropic.com",
        "wiki_slug": "Anthropic",
        "keywords": "Anthropic logo",
        "description": "AI Research Company",
        "type": "COMPANY"
    },
    "claude": {
        "domain": "anthropic.com",
        "wiki_slug": "Claude_(chatbot)",
        "keywords": "Anthropic Claude logo",
        "description": "AI Assistant Tool",
        "type": "TOOL"
    },
    "claude 3": {
        "domain": "anthropic.com",
        "wiki_slug": "Claude_(chatbot)",
        "keywords": "Anthropic Claude logo",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    "claude 3.5": {
        "domain": "anthropic.com",
        "wiki_slug": "Claude_(chatbot)",
        "keywords": "Anthropic Claude logo",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    "claude 3.5 sonnet": {
        "domain": "anthropic.com",
        "wiki_slug": "Claude_(chatbot)",
        "keywords": "Anthropic Claude logo",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    # Meta / Facebook
    "meta": {
        "domain": "meta.com",
        "wiki_slug": "Meta_Platforms",
        "keywords": "Meta logo",
        "description": "Tech Giant",
        "type": "COMPANY"
    },
    "llama": {
        "domain": "meta.com",
        "wiki_slug": "Llama_(language_model)",
        "keywords": "Meta Llama logo",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    "llama 2": {
        "domain": "meta.com",
        "wiki_slug": "Llama_(language_model)",
        "keywords": "Meta Llama logo",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    "llama 3": {
        "domain": "meta.com",
        "wiki_slug": "Llama_(language_model)",
        "keywords": "Meta Llama logo",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    "llama 3.1": {
        "domain": "meta.com",
        "wiki_slug": "Llama_(language_model)",
        "keywords": "Meta Llama logo",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    "llama 3.2": {
        "domain": "meta.com",
        "wiki_slug": "Llama_(language_model)",
        "keywords": "Meta Llama logo",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    # Microsoft
    "microsoft": {
        "domain": "microsoft.com",
        "wiki_slug": "Microsoft",
        "keywords": "Microsoft logo",
        "description": "Tech Giant",
        "type": "COMPANY"
    },
    "copilot": {
        "domain": "microsoft.com",
        "wiki_slug": "Microsoft_Copilot",
        "keywords": "Microsoft Copilot logo",
        "description": "AI Assistant Tool",
        "type": "TOOL"
    },
    "microsoft copilot": {
        "domain": "microsoft.com",
        "wiki_slug": "Microsoft_Copilot",
        "keywords": "Microsoft Copilot logo",
        "description": "AI Assistant Tool",
        "type": "TOOL"
    },
    "github copilot": {
        "domain": "github.com",
        "wiki_slug": "GitHub_Copilot",
        "keywords": "GitHub Copilot logo",
        "description": "AI Coding Assistant",
        "type": "TOOL"
    },
    # Apple
    "apple": {
        "domain": "apple.com",
        "wiki_slug": "Apple_Inc.",
        "keywords": "Apple logo",
        "description": "Tech Giant",
        "type": "COMPANY"
    },
    "apple intelligence": {
        "domain": "apple.com",
        "wiki_slug": "Apple_Intelligence",
        "keywords": "Apple Intelligence logo",
        "description": "AI Feature Suite",
        "type": "TOOL"
    },
    # GitHub
    "github": {
        "domain": "github.com",
        "wiki_slug": "GitHub",
        "keywords": "GitHub logo",
        "description": "Code Hosting Platform",
        "type": "COMPANY"
    },
    # Python
    "python": {
        "domain": "python.org",
        "wiki_slug": "Python_(programming_language)",
        "keywords": "Python language logo",
        "description": "Programming Language",
        "type": "TOOL"
    },
    # Ollama
    "ollama": {
        "domain": "ollama.com",
        "wiki_slug": "Ollama",
        "keywords": "Ollama logo",
        "description": "Local LLM Runner",
        "type": "TOOL"
    },
    # Generic Tech Acronyms / Terms
    "llm": {
        "domain": "wikipedia.org",
        "wiki_slug": "Large_language_model",
        "keywords": "artificial intelligence brain network icon",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    "large language model": {
        "domain": "wikipedia.org",
        "wiki_slug": "Large_language_model",
        "keywords": "artificial intelligence brain network icon",
        "description": "Large Language Model",
        "type": "TOOL"
    },
    "gpu": {
        "domain": "nvidia.com",
        "wiki_slug": "Graphics_processing_unit",
        "keywords": "NVIDIA GPU graphics card microchip",
        "description": "Graphics Processing Unit",
        "type": "TOOL"
    },
    "t4 gpu": {
        "domain": "nvidia.com",
        "wiki_slug": "Tesla_T4",
        "keywords": "NVIDIA Tesla T4 GPU chip graphics card",
        "description": "NVIDIA GPU Accelerator",
        "type": "TOOL"
    },
    "tesla t4": {
        "domain": "nvidia.com",
        "wiki_slug": "Tesla_T4",
        "keywords": "NVIDIA Tesla T4 GPU chip graphics card",
        "description": "NVIDIA GPU Accelerator",
        "type": "TOOL"
    },
    # Other AI Companies / Tools
    "midjourney": {
        "domain": "midjourney.com",
        "wiki_slug": "Midjourney",
        "keywords": "Midjourney logo",
        "description": "AI Image Generator",
        "type": "TOOL"
    },
    "stable diffusion": {
        "domain": "stability.ai",
        "wiki_slug": "Stable_Diffusion",
        "keywords": "Stable Diffusion logo",
        "description": "AI Image Generator",
        "type": "TOOL"
    },
    "stability ai": {
        "domain": "stability.ai",
        "wiki_slug": "Stability_AI",
        "keywords": "Stability AI logo",
        "description": "AI Research Company",
        "type": "COMPANY"
    },
    "perplexity": {
        "domain": "perplexity.ai",
        "wiki_slug": "Perplexity_AI",
        "keywords": "Perplexity AI logo",
        "description": "AI Search Engine",
        "type": "TOOL"
    },
    "groq": {
        "domain": "groq.com",
        "wiki_slug": "Groq",
        "keywords": "Groq logo",
        "description": "AI Inference Chipmaker",
        "type": "COMPANY"
    },
    "mistral": {
        "domain": "mistral.ai",
        "wiki_slug": "Mistral_AI",
        "keywords": "Mistral AI logo",
        "description": "AI Research Company",
        "type": "COMPANY"
    },
    "deepseek": {
        "domain": "deepseek.com",
        "wiki_slug": "DeepSeek",
        "keywords": "DeepSeek logo",
        "description": "AI Research Company",
        "type": "COMPANY"
    },
    "hugging face": {
        "domain": "huggingface.co",
        "wiki_slug": "Hugging_Face",
        "keywords": "Hugging Face logo",
        "description": "AI Model Repository",
        "type": "COMPANY"
    },
    "huggingface": {
        "domain": "huggingface.co",
        "wiki_slug": "Hugging_Face",
        "keywords": "Hugging Face logo",
        "description": "AI Model Repository",
        "type": "COMPANY"
    },
    "nvidia": {
        "domain": "nvidia.com",
        "wiki_slug": "Nvidia",
        "keywords": "NVIDIA logo",
        "description": "AI Computing Company",
        "type": "COMPANY"
    },
    "x.ai": {
        "domain": "x.ai",
        "wiki_slug": "X.AI_(company)",
        "keywords": "xAI logo",
        "description": "AI Startup",
        "type": "COMPANY"
    },
    "xai": {
        "domain": "x.ai",
        "wiki_slug": "X.AI_(company)",
        "keywords": "xAI logo",
        "description": "AI Startup",
        "type": "COMPANY"
    },
    "grok": {
        "domain": "x.ai",
        "wiki_slug": "Grok_(chatbot)",
        "keywords": "Grok logo",
        "description": "AI Chatbot Tool",
        "type": "TOOL"
    },
    "runway": {
        "domain": "runwayml.com",
        "wiki_slug": "Runway_(company)",
        "keywords": "Runway AI logo",
        "description": "AI Creative Platform",
        "type": "COMPANY"
    },
    "pika": {
        "domain": "pika.art",
        "wiki_slug": "Pika_Labs",
        "keywords": "Pika AI logo",
        "description": "AI Video Generator",
        "type": "TOOL"
    },
    "elevenlabs": {
        "domain": "elevenlabs.io",
        "wiki_slug": "ElevenLabs",
        "keywords": "ElevenLabs logo",
        "description": "AI Voice Generator",
        "type": "TOOL"
    },
    "suno": {
        "domain": "suno.com",
        "wiki_slug": "Suno_AI",
        "keywords": "Suno AI logo",
        "description": "AI Music Platform",
        "type": "TOOL"
    },
    "udio": {
        "domain": "udio.com",
        "wiki_slug": "Udio",
        "keywords": "Udio logo",
        "description": "AI Music Platform",
        "type": "TOOL"
    },
    # People
    "sam altman": {
        "wiki_slug": "Sam_Altman",
        "keywords": "Sam Altman portrait",
        "description": "CEO of OpenAI",
        "type": "PEOPLE"
    },
    "sundar pichai": {
        "wiki_slug": "Sundar_Pichai",
        "keywords": "Sundar Pichai portrait",
        "description": "CEO of Google",
        "type": "PEOPLE"
    },
    "elon musk": {
        "wiki_slug": "Elon_Musk",
        "keywords": "Elon Musk portrait",
        "description": "Tech Visionary & Founder",
        "type": "PEOPLE"
    },
    "mark zuckerberg": {
        "wiki_slug": "Mark_Zuckerberg",
        "keywords": "Mark Zuckerberg portrait",
        "description": "CEO of Meta",
        "type": "PEOPLE"
    },
    "satya nadella": {
        "wiki_slug": "Satya_Nadella",
        "keywords": "Satya Nadella portrait",
        "description": "CEO of Microsoft",
        "type": "PEOPLE"
    },
    "jensen huang": {
        "wiki_slug": "Jensen_Huang",
        "keywords": "Jensen Huang portrait",
        "description": "CEO of NVIDIA",
        "type": "PEOPLE"
    },
    "tim cook": {
        "wiki_slug": "Tim_Cook",
        "keywords": "Tim Cook portrait",
        "description": "CEO of Apple",
        "type": "PEOPLE"
    },
    "jeff bezos": {
        "wiki_slug": "Jeff_Bezos",
        "keywords": "Jeff Bezos portrait",
        "description": "Founder of Amazon",
        "type": "PEOPLE"
    },
    "demis hassabis": {
        "wiki_slug": "Demis_Hassabis",
        "keywords": "Demis Hassabis portrait",
        "description": "CEO of Google DeepMind",
        "type": "PEOPLE"
    },
    "yann lecun": {
        "wiki_slug": "Yann_LeCun",
        "keywords": "Yann LeCun portrait",
        "description": "Chief AI Scientist at Meta",
        "type": "PEOPLE"
    },
    "andrej karpathy": {
        "wiki_slug": "Andrej_Karpathy",
        "keywords": "Andrej Karpathy portrait",
        "description": "AI Researcher & Educator",
        "type": "PEOPLE"
    },
    "ilya sutskever": {
        "wiki_slug": "Ilya_Sutskever",
        "keywords": "Ilya Sutskever portrait",
        "description": "Co-founder of Safe Superintelligence",
        "type": "PEOPLE"
    },
    "dario amodei": {
        "wiki_slug": "Dario_Amodei",
        "keywords": "Dario Amodei portrait",
        "description": "CEO of Anthropic",
        "type": "PEOPLE"
    },
    "mira murati": {
        "wiki_slug": "Mira_Murati",
        "keywords": "Mira Murati portrait",
        "description": "Former CTO of OpenAI",
        "type": "PEOPLE"
    },
    "greg brockman": {
        "wiki_slug": "Greg_Brockman",
        "keywords": "Greg Brockman portrait",
        "description": "President of OpenAI",
        "type": "PEOPLE"
    }
}

def resolve_tech_entity(name):
    if not name:
        return None
    normalized = name.lower().strip()
    # Check exact match first
    if normalized in TECH_ENTITY_MAPPING:
        return TECH_ENTITY_MAPPING[normalized]
    # Check substring matches (longest key first)
    for key in sorted(TECH_ENTITY_MAPPING.keys(), key=len, reverse=True):
        if key in normalized:
            return TECH_ENTITY_MAPPING[key]
    return None

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
    resolved = resolve_tech_entity(name)
    twitter_handle = person.get("twitter_handle")
    
    wiki_slug = person.get("wikipedia_slug")
    if not wiki_slug:
        if resolved and "wiki_slug" in resolved:
            wiki_slug = resolved["wiki_slug"]
        else:
            wiki_slug = _get_wikipedia_slug(name)
    
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
    search_name = name
    if resolved and "keywords" in resolved:
        search_name = resolved["keywords"]
    path = _fetch_ddg_image(search_name, output_path, is_logo=False)
    if path:
        return path

    # PRIORITY 4: Generative AI fallback (was Pexels)
    print(f"  -> Falling back to Generative AI for {name}...")
    desc = person.get("description") or "visionary tech industry leader portrait"
    prompt = f"Professional portrait photo headshot of a {desc}, corporate studio lighting, clean office background, highly detailed, photorealistic, premium corporate headshot"
    path = _generate_imagen3(prompt, output_path, topic_context=name, aspect_ratio="9:16")
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
    resolved = resolve_tech_entity(name)
    
    domain = company.get("domain") or company.get("company_domain")
    if not domain and resolved and "domain" in resolved:
        domain = resolved["domain"]
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
        wiki_slug = company.get("wikipedia_slug")
        if not wiki_slug:
            if resolved and "wiki_slug" in resolved:
                wiki_slug = resolved["wiki_slug"]
            else:
                wiki_slug = _get_wikipedia_slug(name)
                
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
    search_name = name
    if resolved and "keywords" in resolved:
        search_name = resolved["keywords"]
    path = _fetch_ddg_image(search_name, output_path, is_logo=True)
    if path:
        return path

    # PRIORITY 4: Generative AI fallback
    print(f"  -> Falling back to Generative AI logo for {name}...")
    prompt = (
        f"A clean, modern, minimalist vector-style logo for '{name}', "
        f"professional corporate brand identity, digital vector graphic icon, "
        f"high contrast, centered composition, white background, no photorealistic backgrounds, "
        f"no text, no clutter"
    )
    # Generate as 1:1 aspect ratio square image
    path = _generate_imagen3(prompt, output_path, topic_context=name, aspect_ratio="1:1")
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
    # Override descriptions and types of known tech entities to clean up categorizations
    for ent_list_key in ["companies", "people", "key_entities"]:
        for ent in script_data.get(ent_list_key, []):
            if isinstance(ent, dict) and ent.get("name"):
                resolved = resolve_tech_entity(ent["name"])
                if resolved:
                    if "description" in resolved:
                        ent["description"] = resolved["description"]
                    if "type" in resolved:
                        ent["type"] = resolved["type"]

    # ── GATHER ALL POTENTIAL ENTITIES ─────────────────────────────────────────
    # If there are companies_mentioned or tools_mentioned that are not in key_entities/companies/people,
    # add them to key_entities to ensure their logos are fetched and they are rendered.
    companies_mentioned = script_data.get("companies_mentioned", [])
    tools_mentioned = script_data.get("tools_mentioned", [])
    
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
        ent_type = str(entity.get("type", "")).upper()
        if ent_type in ["PEOPLE", "PERSON"]:
            path = fetch_person_photo(entity)
            if path:
                entity["local_image_path"] = path
        else:
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
