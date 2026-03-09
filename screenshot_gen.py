import os
import subprocess
import time
from config import ASSETS_DIR, LOGS_DIR

def capture_article_screenshot(url, output_filename):
    """
    Captures a screenshot of the article URL using playwright via npx.
    """
    if not url:
        return None
    
    output_path = os.path.join(ASSETS_DIR, "screenshots", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # We'll use a vertical viewport for mobile-friendly look
    # viewport: 1080x1920
    # We use npx to avoid dependency issues if playwright isn't in venv
    cmd = [
        "npx", "-y", "playwright", "screenshot",
        "--viewport-size=1080,1920",
        url,
        output_path
    ]
    
    try:
        print(f"📸 Capturing screenshot: {url} -> {output_path}")
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(output_path):
            return output_path
    except Exception as e:
        print(f"❌ Screenshot capture failed: {e}")
        
    return None
