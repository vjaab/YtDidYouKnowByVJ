import os
import subprocess
import time
import requests
from config import ASSETS_DIR, LOGS_DIR, FONTS_DIR, GEMINI_API_KEY, GEMINI_FLASH_MODEL


def generate_fallback_screenshot(url, output_path, desktop=False, headline=None):
    """
    Generates a high-quality mockup screenshot card using PIL.
    Used as a fallback when Playwright fails or times out.
    """
    from PIL import Image, ImageDraw, ImageFont
    from urllib.parse import urlparse
    
    # 1. Canvas Dimensions
    width, height = (1920, 1080) if desktop else (1080, 1920)
    
    # 2. Extract Domain Name
    domain = "Unknown Source"
    if url:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path
            if domain.startswith("www."):
                domain = domain[4:]
        except Exception:
            pass
            
    # 3. Determine Headline
    if not headline:
        if url:
            try:
                path = urlparse(url).path
                seg = [s for s in path.split("/") if s][-1]
                headline = seg.replace("-", " ").replace("_", " ").title()
            except Exception:
                headline = "Tech News Update"
        else:
            headline = "Tech News Update"

    headline = str(headline)
    
    # 4. Create Background (Vertical Gradient)
    img = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(img)
    
    color_start = (10, 10, 20, 255)
    color_end = (20, 20, 40, 255)
    
    for y in range(height):
        ratio = y / height
        r = int(color_start[0] + ratio * (color_end[0] - color_start[0]))
        g = int(color_start[1] + ratio * (color_end[1] - color_start[1]))
        b = int(color_start[2] + ratio * (color_end[2] - color_start[2]))
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))
        
    # Draw subtle background grid
    grid_spacing = 80
    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 6), width=1)
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill=(255, 255, 255, 6), width=1)

    # 5. Dimensions
    if desktop:
        card_w, card_h = 1200, 600
        padding = 80
        title_font_size = 56
        meta_font_size = 28
    else:
        card_w, card_h = 900, 1300
        padding = 70
        title_font_size = 64
        meta_font_size = 32
        
    card_x1 = (width - card_w) // 2
    card_y1 = (height - card_h) // 2
    card_x2 = card_x1 + card_w
    card_y2 = card_y1 + card_h
    
    # 6. Load Fonts
    def get_font(name, size):
        font_path = os.path.join(FONTS_DIR, name)
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass
        return ImageFont.load_default()
        
    bold_font = get_font("Montserrat-Bold.ttf", title_font_size)
    reg_font = get_font("Roboto-Regular.ttf", meta_font_size)
    
    # 7. Draw Card Base
    draw.rounded_rectangle(
        [card_x1 + 4, card_y1 + 4, card_x2 + 4, card_y2 + 4],
        radius=30,
        fill=(0, 0, 0, 100)
    )
    draw.rounded_rectangle(
        [card_x1, card_y1, card_x2, card_y2],
        radius=30,
        fill=(10, 12, 20, 235),
        outline=(0, 229, 255, 180),
        width=4
    )
    
    # 8. Badge (Solid cyan background with dark text)
    badge_text = "ARTICLE PREVIEW"
    badge_font = get_font("Montserrat-Bold.ttf", meta_font_size - 4)
    badge_bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
    badge_w = badge_bbox[2] - badge_bbox[0]
    badge_h = badge_bbox[3] - badge_bbox[1]
    
    badge_padding_x = 24
    badge_padding_y = 12
    badge_x1 = card_x1 + padding
    badge_y1 = card_y1 + padding
    badge_x2 = badge_x1 + badge_w + 2 * badge_padding_x
    badge_y2 = badge_y1 + badge_h + 2 * badge_padding_y
    
    draw.rounded_rectangle(
        [badge_x1, badge_y1, badge_x2, badge_y2],
        radius=10,
        fill=(0, 229, 255, 255),
        outline=(0, 229, 255, 255),
        width=2
    )
    draw.text(
        (badge_x1 + badge_padding_x, badge_y1 + badge_padding_y - 2),
        badge_text,
        font=badge_font,
        fill=(10, 12, 20, 255)
    )
    
    # 9. Domain name
    domain_text = domain.upper()
    domain_bbox = draw.textbbox((0, 0), domain_text, font=reg_font)
    domain_w = domain_bbox[2] - domain_bbox[0]
    draw.text(
        (card_x2 - padding - domain_w, badge_y1 + 4),
        domain_text,
        font=reg_font,
        fill=(255, 255, 255, 160)
    )
    
    # 10. Wrap headline
    words = headline.split()
    lines = []
    current_line = []
    max_text_w = card_w - 2 * padding
    
    for w in words:
        test_line = " ".join(current_line + [w])
        test_bbox = draw.textbbox((0, 0), test_line, font=bold_font)
        test_w = test_bbox[2] - test_bbox[0]
        if test_w <= max_text_w:
            current_line.append(w)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [w]
            else:
                lines.append(w)
                current_line = []
    if current_line:
        lines.append(" ".join(current_line))
        
    max_lines = 4 if desktop else 8
    lines = lines[:max_lines]
    
    line_spacing = title_font_size + 15
    y_text_start = badge_y2 + 60
    
    for i, line in enumerate(lines):
        y_pos = y_text_start + i * line_spacing
        draw.text((card_x1 + padding + 2, y_pos + 2), line, font=bold_font, fill=(0, 0, 0, 150))
        draw.text((card_x1 + padding, y_pos), line, font=bold_font, fill=(255, 255, 255, 255))
        
    # 11. Divider
    divider_y = card_y2 - padding - 60
    draw.line(
        [(card_x1 + padding, divider_y), (card_x2 - padding, divider_y)],
        fill=(255, 255, 255, 40),
        width=2
    )
    
    # 12. CTA (no emojis to prevent missing glyph squares)
    cta_text = "FULL STORY & SOURCE ON TELEGRAM @technewsbyvj"
    cta_font = get_font("Montserrat-Bold.ttf", meta_font_size - 2)
    draw.text(
        (card_x1 + padding, divider_y + 25),
        cta_text,
        font=cta_font,
        fill=(0, 229, 255, 220)
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_rgb = img.convert("RGB")
    img_rgb.save(output_path, "PNG")
    print(f"🎨 Fallback screenshot card created: {output_path} (Headline: '{headline[:40]}...')")
    return output_path


def check_screenshot_validity(image_path):
    """
    Checks if the screenshot contains a human verification page or error.
    Returns True if valid, False if it is a captcha/cloudflare/etc.
    """
    import os
    # Get list of API keys
    api_keys_env = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
    api_keys = [k.strip() for k in api_keys_env.split(",") if k.strip()]
    if not api_keys:
        if GEMINI_API_KEY:
            api_keys = [GEMINI_API_KEY]
        else:
            print("⚠️ GEMINI_API_KEY not found in config. Skipping screenshot validation.")
            return True

    # Initialize models to try
    models_to_try = [GEMINI_FLASH_MODEL or "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5-flash"]
    
    from google import genai
    from PIL import Image
    
    img = None
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"⚠️ Failed to open image {image_path}: {e}")
        return True # Can't read, skip blocking it

    prompt = (
        "Analyze this screenshot. Does it display a human verification screen, "
        "CAPTCHA, Cloudflare 'Checking your browser' or 'Verify you are human' page, "
        "Turnstile, hCaptcha, DDoS protection page, access denied error, "
        "or generic browser connection/network error page? "
        "Answer with exactly 'YES' or 'NO' (no punctuation or explanation)."
    )

    # Rotate through models and keys
    model_idx = 0
    key_idx = 0
    attempts = 0
    max_attempts = len(models_to_try) * len(api_keys)

    while attempts < max_attempts:
        current_model = models_to_try[model_idx % len(models_to_try)]
        current_key = api_keys[key_idx % len(api_keys)]
        
        try:
            client = genai.Client(api_key=current_key)
            response = client.models.generate_content(
                model=current_model,
                contents=[img, prompt]
            )
            answer = response.text.strip().upper()
            print(f"🔍 Gemini ({current_model}) verification scan result: {answer}")
            
            if "YES" in answer:
                print(f"⚠️ Screenshot {image_path} detected as human verification/block screen.")
                return False
            return True
        except Exception as e:
            print(f"⚠️ API call failed with model {current_model} using key index {key_idx % len(api_keys)}: {e}")
            attempts += 1
            # Rotate key on every failure, and rotate model after trying all keys
            key_idx += 1
            if key_idx % len(api_keys) == 0:
                model_idx += 1
                
    print("⚠️ All keys and models exhausted for screenshot verification. Assuming screenshot is valid.")
    return True


def capture_article_screenshot(url, output_filename, desktop=False, headline=None):
    """
    Captures a screenshot of the article URL using playwright.
    Falls back to generating a beautiful mock card if Playwright fails.
    
    Args:
        url: Article URL to screenshot.
        output_filename: Filename to save the screenshot as.
        desktop: If True, use desktop viewport (1920x1080) for 16:9 longform videos.
                 If False (default), use mobile viewport (1080x1920) for 9:16 Shorts.
        headline: Optional text headline to draw on fallback card.
    """
    if not url:
        return None
    
    # ── Pre-flight Check ────────────────────────────────────────────────────────
    try:
        # Use a real browser user agent to avoid being blocked by simple scrapers
        resp = requests.head(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10, allow_redirects=True)
        if resp.status_code in (404, 410):
            print(f"⚠️ URL definitely dead or 404: {url} (Status: {resp.status_code})")
            return None
        elif resp.status_code >= 400:
            print(f"⚠️ URL pre-flight check returned status {resp.status_code} for {url}. Proceeding to Playwright anyway...")
    except Exception as e:
        print(f"⚠️ URL pre-flight check failed for {url}: {e}")
        # Continue anyway, as HEAD might be blocked but GET might work via Playwright
        pass

    output_path = os.path.join(ASSETS_DIR, "screenshots", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Desktop viewport for longform 16:9, mobile viewport for Shorts 9:16
    viewport = "1920,1080" if desktop else "1080,1920"
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    
    try:
        print(f"📸 Capturing screenshot via Playwright Python API: {url} -> {output_path}")
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            width, height = map(int, viewport.split(','))
            context = browser.new_context(
                user_agent=user_agent,
                viewport={"width": width, "height": height}
            )
            page = context.new_page()
            page.set_default_timeout(20000)
            
            try:
                page.goto(url, wait_until="domcontentloaded")
            except Exception as e:
                print(f"⚠️ Playwright page.goto warning: {e}")
                
            # Wait for content to settle
            page.wait_for_timeout(3000)
            
            # Zoom in the page to make text highly readable
            zoom_factor = 1.2 if desktop else 1.5
            print(f"🔍 Zooming page by {zoom_factor}x for enhanced readability")
            try:
                page.evaluate(f"document.body.style.zoom = '{zoom_factor}';")
                page.wait_for_timeout(500)
            except Exception as zoom_err:
                print(f"⚠️ Failed to apply page zoom: {zoom_err}")
                
            page.screenshot(path=output_path)
            browser.close()
            
        if os.path.exists(output_path):
            if check_screenshot_validity(output_path):
                return output_path
            else:
                print(f"🗑️ Deleting invalid screenshot (contains human verification/error).")
                try:
                    os.remove(output_path)
                except Exception:
                    pass
    except Exception as e:
        print(f"⚠️ Playwright screenshot failed/timed out for {url}: {e}")
        
    # Generate premium fallback card if Playwright screenshot fails
    try:
        print(f"🎨 Generating beautiful mockup fallback card...")
        return generate_fallback_screenshot(url, output_path, desktop=desktop, headline=headline)
    except Exception as ex:
        print(f"❌ Fallback mockup generation failed: {ex}")
        
    return None

