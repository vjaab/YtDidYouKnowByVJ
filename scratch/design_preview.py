import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

# Add parent dir to path
sys.path.append(os.getcwd())

from video_gen import render_subtitle_frame, gf, ts, _prepare_screenshot_canvas

FRAME_W, FRAME_H = 1080, 1920

def create_preview():
    # 1. Full Screen Article Screenshot Backdrop (Aggressive Zoom Simulation)
    article_img = Image.new("RGB", (1200, 1600), (240, 240, 240))
    d = ImageDraw.Draw(article_img)
    for i in range(20):
        d.rectangle([100, 100 + i*60, 1100, 130 + i*60], fill=(200, 200, 200))
    d.text((150, 150), "ARTICLE EVIDENCE: 10X PERFORMANCE GAIN", fill=(50, 50, 50), font=gf(40, bold=True))
    
    bg = _prepare_screenshot_canvas(article_img, FRAME_W, FRAME_H)
    # Ensure RGBA
    bg = bg.convert("RGBA")
    
    # 2. Simulate the 35% zoom
    zoom_factor = 1.35
    zw, zh = int(FRAME_W * zoom_factor), int(FRAME_H * zoom_factor)
    bg = bg.resize((zw, zh), Image.Resampling.LANCZOS)
    bg = bg.crop(((zw - FRAME_W)//2, (zh - FRAME_H)//2, (zw + FRAME_W)//2, (zh + FRAME_H)//2))
    
    # 3. Captions
    word_data = [
        {"word": "This", "is_active": False, "start": 0, "end": 0.5},
        {"word": "Security", "is_active": False, "start": 0.5, "end": 1.0},
        {"word": "Expert", "is_active": True, "start": 1.0, "end": 1.5},
        {"word": "Lives", "is_active": True, "start": 1.5, "end": 2.0},
    ]
    
    sub_img = render_subtitle_frame(word_data, frame_width=FRAME_W, frame_height=FRAME_H)
    sub_img = sub_img.convert("RGBA")
    
    bg.alpha_composite(sub_img)
    
    # 4. Save preview
    out_path = "design_preview_final.png"
    bg.convert("RGB").save(out_path)
    print(f"Preview saved to {out_path}")

if __name__ == "__main__":
    create_preview()
