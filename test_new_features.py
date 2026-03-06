import os
import sys
import numpy as np
from PIL import Image

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_gen import clean_tts_text
from video_gen import render_entity_tags

print("--- Testing Audio cleaning ---")
dirty_text = "Hello world... [pause] This is — actually amazing. [INTENSE MUSIC] (silence) Goodbye."
cleaned = clean_tts_text(dirty_text)
print(f"Original: {dirty_text}")
print(f"Cleaned:  {cleaned}")

# Verify it doesn't remove important pauses like ... or —
if "..." in cleaned and "—" in cleaned:
    print("✅ Punctuation pauses preserved.")
else:
    print("❌ Punctuation pauses LOST!")

if "pause" in cleaned.lower() or "music" in cleaned.lower():
    print("❌ Meta instructions REMAIN!")
else:
    print("✅ Meta instructions STRIPPED.")

print("\n--- Testing Entity Tags Rendering ---")
companies = ["OpenAI", "Microsoft"]
tools = ["ChatGPT", "DALL-E 3"]
accent_color = (0, 229, 255) # Cyan

tags_img = render_entity_tags(companies, tools, accent_color)
tags_img.save("output/test_entity_tags.png")
print("✅ Entity tags image saved to output/test_entity_tags.png")

# Check if anything was actually drawn (not just empty image)
bbox = tags_img.getbbox()
if bbox:
    print(f"✅ Image is not empty. BBox: {bbox}")
else:
    print("❌ Image is EMPTY!")
