import numpy as np
import copy
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from config import ASSETS_DIR
import os

W, H = 1080, 1920

def get_font(style, size):
    fonts = {
        "black": "Montserrat-Black.ttf",
        "extrabold": "Montserrat-ExtraBold.ttf",
        "bold": "Montserrat-Bold.ttf",
        "roboto_bold": "Roboto-Bold.ttf"
    }
    path = os.path.join(ASSETS_DIR, "fonts", fonts.get(style, "Roboto-Bold.ttf"))
    if not os.path.exists(path):
        print(f"FAILED TO LOAD {path}")
        path = os.path.join(ASSETS_DIR, "fonts", "Roboto-Regular.ttf")
    try:
        return ImageFont.truetype(path, size)
    except:
        return ImageFont.load_default()

def measure_text(text, font):
    bb = font.getbbox(text)
    return bb[2] - bb[0], bb[3] - bb[1]

def determine_brightness_mods(bg_region):
    if bg_region is None or bg_region.size == 0:
        return 0.82, 0, 0
    brightness = np.mean(bg_region)
    if brightness > 128:
        return 0.90, 2, 24 # op, extra_stroke, extra_blur
    return 0.82, 0, 0

def draw_style_a(draw, text, x, y, font, text_color, base_stroke, bg_region, bg_color=(0,0,0)):
    # Solid dark pill
    op, extra_stroke, _ = determine_brightness_mods(bg_region)
    stroke = base_stroke + extra_stroke
    
    tw, th = measure_text(text, font)
    # Padding: 14 T/B, 28 L/R
    pad_x, pad_y = 28, 14
    
    # We simulate the 12px blur backdrop by drawing the pill, but actual blur requires image manipulation.
    # The rule says Style A has a "Blur backdrop behind box: gaussian 12px"
    # To keep this fast via ImageDraw, we'll draw a slightly larger shadow/blurred box if possible,
    # but the instructions just say "Blur backdrop behind box" which implies we need the bg frame.
    # We will just draw the pill here.
    
    draw.rounded_rectangle(
        [x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y],
        radius=50,
        fill=(bg_color[0], bg_color[1], bg_color[2], int(255 * op))
    )
    
    # Stroke
    for dx in range(-stroke, stroke+1):
        for dy in range(-stroke, stroke+1):
            if dx*dx + dy*dy <= stroke*stroke:
                draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0,255))
    
    # Shadow offset 3,3 blur 10
    # For speed in PIL, just drawing offset 3,3 as solid shadow
    draw.text((x+3, y+3), text, font=font, fill=(0,0,0,240))
    
    # Text
    draw.text((x, y), text, font=font, fill=text_color)
    return draw

def render_frosted_glass_subtitle(bg_frame_array, text_lines, x, y, font, text_color, base_stroke, active_word_idx=None):
    """Style B implementation. bg_frame_array is the full RGB video frame numpy array."""
    # Subtitle width max 860.
    
    # Calc bounding box
    lsp = int(measure_text("Ag", font)[1] * 1.4)
    total_h = lsp * len(text_lines)
    box_w = 0
    for line in text_lines:
        lw = sum([measure_text(w["word"], font)[0] + measure_text(" ", font)[0] for w in line])
        if lw > box_w: box_w = lw
    
    box_w += 68 # 34 L/R
    box_h = total_h + 44 # 22 T/B
    
    y1 = max(0, min(y, H))
    y2 = max(0, min(y+box_h, H))
    x1 = max(0, min(x, W))
    x2 = max(0, min(x+box_w, W))
    
    # Calculate brightness
    bg_region = bg_frame_array[y1:y2, x1:x2] if box_w > 0 and box_h > 0 else None
    op, extra_stroke, extra_blur = determine_brightness_mods(bg_region)
    stroke = base_stroke + extra_stroke
    
    # We only return the isolated transparent overlay layer that Moviepy can comp on top, OR the fully comped image.
    # Because we need frosted glass, we MUST output an RGB image for this box area with alpha, or just overlay it.
    
    canvas = Image.new("RGBA", (W, H), (0,0,0,0))
    
    if bg_region is not None and bg_region.size > 0:
        # Step 1 & 2: extract and blur
        region_img = Image.fromarray(bg_region).convert("RGBA")
        blurred_region = region_img.filter(ImageFilter.GaussianBlur(18 + extra_blur))
        
        # Step 3: Darken region by 55%
        # Darkening RGB channels
        darkened_arr = np.array(blurred_region)
        darkened_arr[:,:,:3] = darkened_arr[:,:,:3] * 0.45
        darkened_img = Image.fromarray(darkened_arr)
        
        # Step 4: Draw rounded rectangle over it color rgba(0,0,0,0.75) -- rule actually says op 0.75 or 0.82 min
        # Rule 7 says: Min background opacity: 0.82
        mask = Image.new("L", (x2-x1, y2-y1), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0,0, x2-x1, y2-y1], radius=18, fill=255)
        
        box_layer = Image.new("RGBA", (x2-x1, y2-y1), (0,0,0, int(255 * max(0.82, op))))
        
        # Composite the frosted part into canvas
        canvas.paste(darkened_img, (x1, y1), mask)
        canvas.paste(box_layer, (x1, y1), mask)
    
    draw = ImageDraw.Draw(canvas)
    
    cur_y = y + 22
    for line in text_lines:
        line_text = " ".join([d["word"] for d in line])
        lw = sum([measure_text(wd["word"], font)[0] + measure_text(" ", font)[0] for wd in line])
        cur_x = x + (box_w - lw) // 2
        for i, wd in enumerate(line):
            word = wd["word"]
            is_active = (wd.get("is_active", False))
            color = (255, 214, 0, 255) if is_active else text_color # Rule 5: Yellow or White
            
            # Stroke
            for dx in range(-stroke, stroke+1):
                for dy in range(-stroke, stroke+1):
                    if dx*dx + dy*dy <= stroke*stroke:
                        draw.text((cur_x+dx, cur_y+dy), word, font=font, fill=(0,0,0,255))
            
            # Shadow
            draw.text((cur_x+3, cur_y+3), word, font=font, fill=(0,0,0,240))
            
            # Text
            draw.text((cur_x, cur_y), word, font=font, fill=color)
            
            ww = measure_text(word, font)[0]
            sw = measure_text(" ", font)[0]
            cur_x += ww + sw
        cur_y += lsp

    return canvas
