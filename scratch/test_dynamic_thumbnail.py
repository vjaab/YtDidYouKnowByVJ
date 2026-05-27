import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thumbnail_gen import generate_thumbnail
from config import ASSETS_DIR, OUTPUT_DIR

def run_tests():
    print("🚀 Starting Dynamic & Logo-branded Premium Thumbnail Tests...")
    
    # 1. Tech Authority (Landscape)
    test_tech = {
        "title": "Artificial Intelligence is Here",
        "custom_hook": "AGI IS\nALREADY\nHERE...",
        "color_theme": {"accent": "#00E5FF"},
        "avatar_still": os.path.join(ASSETS_DIR, "video", "black_blazer.mp4"),
        "avatar_still_time": 1.2,
        "logo_path": os.path.join(ASSETS_DIR, "logo.png"),
        "logos": [
            {"path": os.path.join(ASSETS_DIR, "logo.png"), "position": "top_left", "label": "GEN NEWS"}
        ],
        "output_suffix": "test_thumb_tech"
    }

    # 2. Cyber Hack (Landscape)
    test_cyber = {
        "title": "Leaked Hacker Bot",
        "custom_hook": "LEAKED\nCODING\nROBOT...",
        "color_theme": {"accent": "#00FF7F"},
        "avatar_still": os.path.join(ASSETS_DIR, "video", "leather_jacket.mp4"),
        "avatar_still_time": 1.0,
        "logos": [
            {"path": os.path.join(ASSETS_DIR, "icons", "telegram_logo.png"), "position": "top_left", "label": "SECURE TELEGRAM"},
            {"path": os.path.join(ASSETS_DIR, "logo.png"), "position": "bottom_left"}
        ],
        "output_suffix": "test_thumb_cyber"
    }

    # 3. Viral Creator (Landscape / Shorts)
    test_viral = {
        "title": "The Coding Apocalypse 2026",
        "custom_hook": "2026\nCODING\nAPOCALYPSE",
        "color_theme": {"accent": "#FF4444"},
        "avatar_still": os.path.join(ASSETS_DIR, "video", "pink_full_sleeve_t_shirt.mp4"),
        "avatar_still_time": 0.8,
        "logos": [
            {"path": os.path.join(ASSETS_DIR, "icons", "linkedin_logo.png"), "position": "top_left", "label": "EXCLUSIVE"}
        ],
        "output_suffix": "test_thumb_viral"
    }

    test_cases = [
        ("Tech Authority", test_tech),
        ("Cyber Hack", test_cyber),
        ("Viral Creator", test_viral)
    ]

    for name, script_json in test_cases:
        print(f"\n🎨 Generating dynamic thumbnail variation: {name}...")
        try:
            out_path = generate_thumbnail(script_json)
            print(f"✅ Generated {name} thumbnail successfully at: {out_path}")
        except Exception as e:
            print(f"❌ Failed to generate {name} thumbnail: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_tests()
