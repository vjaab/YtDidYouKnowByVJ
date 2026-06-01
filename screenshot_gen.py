import os
import subprocess
import time
import requests
from config import ASSETS_DIR, LOGS_DIR

def capture_article_screenshot(url, output_filename, desktop=False):
    """
    Captures a screenshot of the article URL using playwright via npx.
    
    Args:
        url: Article URL to screenshot.
        output_filename: Filename to save the screenshot as.
        desktop: If True, use desktop viewport (1920x1080) for 16:9 longform videos.
                 If False (default), use mobile viewport (1080x1920) for 9:16 Shorts.
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
    
    # Increase timeout and use wait-until=networkidle via custom script if possible, 
    # but for CLI we use --wait-for-timeout.
    cmd = [
        "npx", "-y", "playwright", "screenshot",
        f"--viewport-size={viewport}",
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
