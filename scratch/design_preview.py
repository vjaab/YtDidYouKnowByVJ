import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import random

# Add parent dir to path
sys.path.append(os.getcwd())

from video_gen import render_subtitle_frame, gf, ts
from efficiency_engine import render_value_header, render_efficiency_scale, render_evidence_card

FRAME_W, FRAME_H = 1080, 1920

def create_preview():
    # 1. Base Background (Zone C)
    bg = Image.new("RGBA", (FRAME_W, FRAME_H), (15, 15, 25, 255))
    draw = ImageDraw.Draw(bg)
    # Mock subject
    draw.ellipse([400, 1000, 680, 1280], fill=(50, 50, 100, 255))
    
    # 2. Zone A: Value Header
    h_img = render_value_header("SQLite vs FST: The Speed Test", FRAME_W, FRAME_H, gf(48, bold=True))
    bg.alpha_composite(h_img)
    
    # 3. Zone B: Efficiency Scale
    p_path = "assets/problem_weight_icon.png"
    s_path = "assets/solution_bolt_icon.png"
    # Render at t=0.5 (mid-animation)
    scale_img = render_efficiency_scale(0.5, 1.0, FRAME_W, FRAME_H, p_path, s_path)
    bg.alpha_composite(scale_img)
    
    # 4. Zone B: Evidence Card
    ev_data = {"title": "Benchmark Results", "value": "10x Faster Processing"}
    ev_img = render_evidence_card(ev_data, FRAME_W, FRAME_H, gf(32), gf(64, bold=True))
    bg.alpha_composite(ev_img)
    
    # 5. Captions (Centered)
    word_data = [
        {"word": "FST", "is_active": False},
        {"word": "is", "is_active": False},
        {"word": "10x", "is_active": True},
        {"word": "Faster", "is_active": True},
    ]
    sub_img = render_subtitle_frame(word_data, frame_width=FRAME_W, frame_height=FRAME_H)
    bg.alpha_composite(sub_img)
    
    # Save preview
    out_path = "design_preview_universal.png"
    bg.convert("RGB").save(out_path)
    print(f"Preview saved to {out_path}")

if __name__ == "__main__":
    create_preview()
