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
    
    # ── Pre-flight Check ────────────────────────────────────────────────────────
    try:
        # Use a real browser user agent to avoid being blocked by simple scrapers
        resp = requests.head(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10, allow_redirects=True)
        if resp.status_code >= 400:
            print(f"⚠️ URL unreachable or 404: {url} (Status: {resp.status_code})")
            return None
    except Exception as e:
        print(f"⚠️ URL pre-flight check failed for {url}: {e}")
        # Continue anyway, as HEAD might be blocked but GET might work via Playwright
        pass

    output_path = os.path.join(ASSETS_DIR, "screenshots", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Increase timeout and use wait-until=networkidle via custom script if possible, 
    # but for CLI we use --wait-for-timeout.
    cmd = [
        "npx", "-y", "playwright", "screenshot",
        "--viewport-size=1080,1920",
        "--wait-for-timeout=8000", # Increased for slower enterprise sites
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
