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
    
    # ── Pre-flight Check Removed ────────────────────────────────────────────────
    # Wait for playwright to do the job; requests is often blocked by Cloudflare.
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
