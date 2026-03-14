import os
import subprocess
import time
import requests
from config import ASSETS_DIR, LOGS_DIR

def capture_article_screenshot(url, output_filename):
    """
    Captures a screenshot of the article URL using playwright via npx.
    """
    if not url:
        return None
    
    # ── Pre-flight Check: Ensure URL is valid and not a 404 ───────────────────
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        # Try HEAD first (faster), allowing redirects
        response = requests.head(url, headers=headers, timeout=12, allow_redirects=True)
        
        # If HEAD not supported (405) or returns error (>=400), try GET
        if response.status_code >= 400:
            # Minimal download using stream=True
            response = requests.get(url, headers=headers, timeout=12, stream=True, allow_redirects=True)
            
        if response.status_code >= 400:
            print(f"⚠️ Skipping screenshot: URL returned {response.status_code} ({url})")
            return None
            
    except Exception as e:
        print(f"🔍 Pre-flight check warning for {url}: {e}")
        # We continue to Playwright if it's just a timeout/connection issue, 
        # as the headless browser might have more persistence or better headers.
        pass

    output_path = os.path.join(ASSETS_DIR, "screenshots", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # We'll use a vertical viewport for mobile-friendly look
    # viewport: 1080x1920
    # networkidle: waits until there are no network connections for at least 500ms
    cmd = [
        "npx", "-y", "playwright", "screenshot",
        "--viewport-size=1080,1920",
        "--wait-for-timeout=5000",
        url,
        output_path
    ]
    
    # Note: If we had a custom js script, we could use --wait-for-load-state=networkidle.
    # The 'playwright screenshot' CLI doesn't support --wait-for-load-state directly.
    # However, it does support --wait-for-timeout which we've increased to 5000ms.
    
    try:
        print(f"📸 Capturing screenshot: {url} -> {output_path}")
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(output_path):
            return output_path
    except Exception as e:
        print(f"❌ Screenshot capture failed for {url}: {e}")
        
    return None
