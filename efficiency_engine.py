import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import ImageClip
import moviepy.video.fx as vfx

def render_value_header(text, frame_w, frame_h, font):
    """Zone A (0%-30%): Dynamic Headlines."""
    img = Image.new("RGBA", (frame_w, frame_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    # Check for 'vs' to do comparison layout
    if " vs " in text.lower():
        parts = text.split(" vs ")
        # Render A (Left) and B (Right)
        draw.text((frame_w//4, 150), parts[0], font=font, fill=(255,255,255,255), anchor="mm")
        draw.text((3*frame_w//4, 150), parts[1], font=font, fill=(204, 255, 0, 255), anchor="mm")
    else:
        draw.text((frame_w//2, 150), text, font=font, fill=(255,255,255,255), anchor="mm")
    
    return img

def render_efficiency_scale(t, duration, frame_w, frame_h, problem_icon_path, solution_icon_path):
    """Zone B (30%-50%): Problem vs Solution Animation."""
    img = Image.new("RGBA", (frame_w, frame_h), (0,0,0,0))
    
    prog = t / duration
    
    # Problem: Sinks on Y-axis
    if os.path.exists(problem_icon_path):
        p_icon = Image.open(problem_icon_path).convert("RGBA").resize((250, 250), Image.LANCZOS)
        # Apply red tint
        red = Image.new("RGBA", p_icon.size, (255, 0, 0, 100))
        p_icon = Image.alpha_composite(p_icon, red)
        
        y_pos = int(700 + (200 * prog)) # Sinks from 700 to 900
        img.paste(p_icon, (frame_w//4 - 125, y_pos), p_icon)
        
    # Solution: Rises on Y-axis
    if os.path.exists(solution_icon_path):
        s_icon = Image.open(solution_icon_path).convert("RGBA").resize((180, 180), Image.LANCZOS)
        # Apply green tint
        green = Image.new("RGBA", s_icon.size, (0, 255, 0, 100))
        s_icon = Image.alpha_composite(s_icon, green)
        
        y_pos = int(900 - (200 * prog)) # Rises from 900 to 700
        img.paste(s_icon, (3*frame_w//4 - 90, y_pos), s_icon)
        
    return img

def render_evidence_card(data, frame_w, frame_h, font_small, font_bold):
    """Zone B: Clean Card Evidence Pop."""
    card_w, card_h = 700, 250
    img = Image.new("RGBA", (frame_w, frame_h), (0,0,0,0))
    
    # Subtle drop shadow
    shadow = Image.new("RGBA", (card_w + 20, card_h + 20), (0,0,0,60))
    img.paste(shadow, ((frame_w - card_w)//2 + 10, 500 + 10))
    
    # Main Card
    card = Image.new("RGBA", (card_w, card_h), (255,255,255,255))
    cd = ImageDraw.Draw(card)
    cd.rounded_rectangle([0, 0, card_w, card_h], radius=5, fill=(255,255,255,255))
    
    # Content
    cd.text((40, 40), data.get("title", "Research Evidence"), font=font_small, fill=(100,100,100,255))
    cd.text((40, 100), data.get("value", "99.9% Efficiency"), font=font_bold, fill=(0,0,0,255))
    
    img.paste(card, ((frame_w - card_w)//2, 500), card)
    return img
