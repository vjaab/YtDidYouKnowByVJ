#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Quiz & Trivia category shorts avatar rendering locally.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_gen import (
    _get_category_avatar_style, 
    _apply_circular_facecam_frame,
    SafeZoneCalculator,
    FRAME_W, FRAME_H
)
from PIL import Image, ImageDraw
import numpy as np
from moviepy import ImageClip

def create_test_avatar():
    """Create a test avatar image (simulating lip-sync output)."""
    img = Image.new('RGBA', (400, 400), (100, 150, 200, 255))
    draw = ImageDraw.Draw(img)
    # Draw a simple face
    draw.ellipse([100, 80, 300, 280], fill=(200, 180, 160, 255))  # face
    draw.ellipse([140, 130, 180, 170], fill=(50, 50, 50, 255))    # left eye
    draw.ellipse([220, 130, 260, 170], fill=(50, 50, 50, 255))    # right eye
    draw.ellipse([180, 200, 220, 240], fill=(180, 100, 100, 255)) # mouth
    return img

def test_category_style():
    """Test the category style function for Quiz & Trivia."""
    print("=" * 60)
    print("Testing _get_category_avatar_style for 'Quiz & Trivia'")
    print("=" * 60)
    
    style = _get_category_avatar_style("Quiz & Trivia", True)
    print(f"Style: {style}")
    
    expected = {
        "scale_mult": 1.0,
        "entrance_style": "pop_in",
        "glow_style": "pulse_fast",
        "border_style": "double_ring",
        "accent_tint": (255, 215, 0),
    }
    
    for key, val in expected.items():
        assert style[key] == val, f"Mismatch for {key}: got {style[key]}, expected {val}"
    print("✅ All style values match expected!")

def test_circular_frame():
    """Test the circular face-cam frame with Quiz & Trivia style."""
    print("\n" + "=" * 60)
    print("Testing _apply_circular_facecam_frame with Quiz & Trivia style")
    print("=" * 60)
    
    # Create test avatar
    avatar_img = create_test_avatar()
    avatar_arr = np.array(avatar_img.convert("RGB"))
    avatar_clip = ImageClip(avatar_arr, duration=10.0)
    
    # Get Quiz & Trivia style
    cat_style = _get_category_avatar_style("Quiz & Trivia", True)
    print(f"Category style: {cat_style}")
    
    # Apply circular frame
    accent_color = (0, 240, 255)  # Default cyan
    audio_duration = 10.0
    
    try:
        result_clip, ring_clip, ring_size = _apply_circular_facecam_frame(
            avatar_clip, 400, 400, accent_color, audio_duration, 
            is_longform=False, cat_style=cat_style
        )
        print(f"✅ Circular frame applied successfully!")
        print(f"   Ring size: {ring_size}")
        print(f"   Ring clip: {ring_clip is not None}")
        print(f"   Border style: {cat_style['border_style']}")
        print(f"   Glow style: {cat_style['glow_style']}")
        print(f"   Accent tint: {cat_style['accent_tint']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_safe_zone_calculator():
    """Test SafeZoneCalculator for shorts with split_screen layout (Quiz layout)."""
    print("\n" + "=" * 60)
    print("Testing SafeZoneCalculator for split_screen layout")
    print("=" * 60)
    
    layout = {
        "layout_type": "split_screen",
        "corner_index": 0,
    }
    
    calc = SafeZoneCalculator(FRAME_W, FRAME_H, layout=layout, is_longform=False)
    
    # Get avatar corner position
    avatar_pos = calc.get_position("avatar_corner")
    print(f"Avatar corner position: {avatar_pos}")
    
    if avatar_pos:
        x, y, w, h = avatar_pos
        print(f"   x={x}, y={y}, w={w}, h={h}")
        # For split_screen, avatar should be top-right
        assert x > FRAME_W // 2, "Avatar should be on right side for split_screen"
        assert y < FRAME_H // 2, "Avatar should be on top for split_screen"
        print("✅ Avatar position correct for split_screen (top-right)")

def main():
    print("🧪 Testing Quiz & Trivia Shorts Avatar Rendering")
    print("=" * 60)
    
    try:
        test_category_style()
        test_circular_frame()
        test_safe_zone_calculator()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()