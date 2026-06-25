import os
import random
import io
from PIL import Image, ImageDraw, ImageFont, ImageChops
from datetime import datetime
from google import genai
from config import GEMINI_API_KEY

TODAY = datetime.now().strftime("%Y-%m-%d")

def generate_whiteboard_doodle_ai(text, topic_context=""):
    """
    Calls Gemini to generate a conceptual metaphor sketch description,
    then calls Imagen to generate a whiteboard-style sketch of that metaphor.
    """
    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY not found in config. Skipping AI whiteboard doodle generation.")
        return None
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # 1. Ask Gemini to define a visual metaphor doodle
        prompt = f"""You are an expert AI Whiteboard Animation Director.
For the spoken phrase: "{text}"
(Context: {topic_context})

Define a single visual metaphor, sketch, or literal drawing to draw on a whiteboard.
Guidelines:
- Do NOT just write out words or text.
- Create a visual metaphor. E.g. "GitHub" -> "Octocat logo or code tree", "privacy/reputation at risk" -> "shield cracking".
- Keep it simple, clear, and easy to draw as a line sketch.
- Return ONLY the description of the sketch (max 15 words).
Example: "A hand cursor clicking a large button"
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        doodle_concept = response.text.strip().strip('"').strip("'")
        print(f"🎨 [WHITEBOARD AI DOODLE] Concept: '{doodle_concept}' for text: '{text[:40]}...'")
        
        # 2. Call Imagen to generate a whiteboard drawing of this concept
        imagen_prompt = f"A simple whiteboard marker sketch of {doodle_concept}, black marker hand-drawn outline on a clean white whiteboard background, minimalist doodle, professional sketch, no text."
        
        # Try fast generate model first
        models = ["imagen-4.0-fast-generate-001", "imagen-4.0-generate-001"]
        for model in models:
            try:
                print(f"🎨 Generating whiteboard doodle using {model}...")
                result = client.models.generate_images(
                    model=model,
                    prompt=imagen_prompt,
                    config=genai.types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="1:1",  # square doodle
                        output_mime_type="image/jpeg",
                    )
                )
                for gen_img in result.generated_images:
                    return Image.open(io.BytesIO(gen_img.image.image_bytes))
            except Exception as e:
                print(f"⚠️ Imagen whiteboard doodle failed on {model}: {e}")
                continue

        # Try Pollinations fallback
        print("🎨 Imagen whiteboard doodle failed. Trying Pollinations AI fallback...")
        try:
            import urllib.parse
            import requests
            encoded_prompt = urllib.parse.quote(imagen_prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&private=true"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                print("✅ Pollinations whiteboard doodle generated successfully!")
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as pe:
            print(f"⚠️ Pollinations whiteboard doodle failed: {pe}")
    except Exception as e:
        print(f"⚠️ Failed to generate AI whiteboard doodle: {e}")
    return None

def generate_whiteboard_frame(text, output_path, headline="", is_longform=False):
    width, height = (1920, 1080) if is_longform else (1080, 1920)
    
    # Creamy Whiteboard background
    bg_color = (252, 251, 249)
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw Dot Grid for notebook look
    dot_color = (225, 225, 230)
    dot_spacing = 40
    for x in range(dot_spacing, width, dot_spacing):
        for y in range(dot_spacing, height, dot_spacing):
            draw.ellipse([x-1, y-1, x+1, y+1], fill=dot_color)
            
    # Load fonts
    font_dir = "assets/fonts"
    title_font_path = os.path.join(font_dir, "Montserrat-ExtraBold.ttf")
    body_font_path = os.path.join(font_dir, "Montserrat-Bold.ttf")
    
    if not os.path.exists(title_font_path): title_font_path = "DejaVuSans-Bold.ttf"
    if not os.path.exists(body_font_path): body_font_path = "DejaVuSans-Bold.ttf"
    
    try:
        title_font = ImageFont.truetype(title_font_path, 40 if is_longform else 32)
        body_font = ImageFont.truetype(body_font_path, 54 if is_longform else 48)
    except:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        
    # Draw Title/Headline box at top
    draw_title = (headline or "TECH TIPS").upper()
    try:
        title_w = draw.textlength(draw_title, font=title_font)
        title_h = 40
    except:
        title_w = len(draw_title) * 20
        title_h = 40
        
    title_y = 70
    title_box_x1 = (width - title_w) // 2 - 30
    title_box_x2 = (width + title_w) // 2 + 30
    title_box_y1 = title_y - 15
    title_box_y2 = title_y + title_h + 15
    
    # Draw sketchy box
    def draw_sketchy_rect(x1, y1, x2, y2, color=(30, 100, 200), width_line=4):
        # Top line
        draw.line([x1 - random.randint(2,8), y1 + random.randint(-2,2), x2 + random.randint(2,8), y1 + random.randint(-2,2)], fill=color, width=width_line)
        # Bottom line
        draw.line([x1 - random.randint(2,8), y2 + random.randint(-2,2), x2 + random.randint(2,8), y2 + random.randint(-2,2)], fill=color, width=width_line)
        # Left line
        draw.line([x1 + random.randint(-2,2), y1 - random.randint(2,8), x1 + random.randint(-2,2), y2 + random.randint(2,8)], fill=color, width=width_line)
        # Right line
        draw.line([x2 + random.randint(-2,2), y1 - random.randint(2,8), x2 + random.randint(-2,2), y2 + random.randint(2,8)], fill=color, width=width_line)
        
    draw_sketchy_rect(title_box_x1, title_box_y1, title_box_x2, title_box_y2, color=(30, 100, 200), width_line=4)
    draw.text(((width - title_w) // 2, title_y), draw_title, fill=(30, 100, 200), font=title_font)
    
    # Wrap text for the main body
    max_char_width = 45 if is_longform else 28
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_char_width:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    if current_line:
        lines.append(" ".join(current_line))
        
    # Draw body text in the center
    charcoal = (40, 40, 45)
    line_h = 75 if is_longform else 65
    total_text_h = len(lines) * line_h
    start_y = (height - total_text_h) // 2 - 50
    
    for i, line in enumerate(lines):
        try:
            line_w = draw.textlength(line, font=body_font)
        except:
            line_w = len(line) * 24
        draw.text(((width - line_w) // 2, start_y + i * line_h), line, fill=charcoal, font=body_font)
        
    # Draw sketchy divider line below text
    divider_y = start_y + total_text_h + 30
    draw.line([width//2 - 200 + random.randint(-10,10), divider_y, width//2 + 200 + random.randint(-10,10), divider_y + random.randint(-2,2)], fill=(220, 50, 50), width=5)
    
    # Generate and draw AI whiteboard doodle if possible
    ai_doodle = generate_whiteboard_doodle_ai(text, headline)
    if ai_doodle:
        doodle_size = 300 if is_longform else 400
        ai_doodle = ai_doodle.resize((doodle_size, doodle_size), Image.Resampling.LANCZOS).convert("RGB")
        doodle_x = (width - doodle_size) // 2
        doodle_y = divider_y + 40
        
        # Multiply blend with the creamy background dot grid
        box = (doodle_x, doodle_y, doodle_x + doodle_size, doodle_y + doodle_size)
        bg_region = img.crop(box)
        blended_region = ImageChops.multiply(bg_region, ai_doodle)
        img.paste(blended_region, box)
    else:
        # Draw Doodle Graphics depending on keywords
        doodle_color = (220, 50, 50)
        text_lower = text.lower()
        
        if any(k in text_lower for k in ["scam", "hack", "safe", "privacy", "lock", "secure"]):
            # Draw shield/lock
            doodle_x = width // 2
            doodle_y = divider_y + 120
            draw.line([doodle_x - 40, doodle_y - 45, doodle_x + 40, doodle_y - 45], fill=doodle_color, width=4)
            draw.line([doodle_x - 40, doodle_y - 45, doodle_x - 40, doodle_y], fill=doodle_color, width=4)
            draw.line([doodle_x + 40, doodle_y - 45, doodle_x + 40, doodle_y], fill=doodle_color, width=4)
            draw.line([doodle_x - 40, doodle_y, doodle_x, doodle_y + 50], fill=doodle_color, width=4)
            draw.line([doodle_x + 40, doodle_y, doodle_x, doodle_y + 50], fill=doodle_color, width=4)
            draw.line([doodle_x - 15, doodle_y, doodle_x - 3, doodle_y + 15], fill=doodle_color, width=4)
            draw.line([doodle_x - 3, doodle_y + 15, doodle_x + 20, doodle_y - 15], fill=doodle_color, width=4)
        elif any(k in text_lower for k in ["phone", "iphone", "android", "device", "mobile"]):
            # Phone silhouette
            doodle_x = width // 2
            doodle_y = divider_y + 100
            draw_sketchy_rect(doodle_x - 35, doodle_y - 60, doodle_x + 35, doodle_y + 60, color=doodle_color, width_line=4)
            draw.line([doodle_x - 30, doodle_y - 45, doodle_x + 30, doodle_y - 45], fill=doodle_color, width=2)
            draw.line([doodle_x - 30, doodle_y + 45, doodle_x + 30, doodle_y + 45], fill=doodle_color, width=2)
            draw.ellipse([doodle_x - 6, doodle_y + 48, doodle_x + 6, doodle_y + 60], outline=doodle_color, width=2)
        elif any(k in text_lower for k in ["money", "free", "save", "price", "cost", "dollar"]):
            # Dollar sign
            doodle_x = width // 2
            doodle_y = divider_y + 110
            draw.line([doodle_x, doodle_y - 50, doodle_x, doodle_y + 50], fill=doodle_color, width=5)
            draw.arc([doodle_x - 25, doodle_y - 40, doodle_x + 25, doodle_y], start=180, end=360, fill=doodle_color, width=5)
            draw.arc([doodle_x - 25, doodle_y - 40, doodle_x + 25, doodle_y], start=0, end=90, fill=doodle_color, width=5)
            draw.line([doodle_x - 25, doodle_y, doodle_x + 25, doodle_y], fill=doodle_color, width=5)
            draw.arc([doodle_x - 25, doodle_y, doodle_x + 25, doodle_y + 40], start=0, end=180, fill=doodle_color, width=5)
            draw.arc([doodle_x - 25, doodle_y, doodle_x + 25, doodle_y + 40], start=270, end=360, fill=doodle_color, width=5)
        else:
            # Lightbulb
            doodle_x = width // 2
            doodle_y = divider_y + 110
            draw.ellipse([doodle_x - 35, doodle_y - 45, doodle_x + 35, doodle_y + 25], outline=doodle_color, width=4)
            draw.line([doodle_x - 20, doodle_y + 25, doodle_x + 20, doodle_y + 25], fill=doodle_color, width=4)
            draw.line([doodle_x - 15, doodle_y + 33, doodle_x + 15, doodle_y + 33], fill=doodle_color, width=4)
            draw.line([doodle_x - 8, doodle_y + 41, doodle_x + 8, doodle_y + 41], fill=doodle_color, width=4)
            draw.line([doodle_x - 10, doodle_y + 10, doodle_x - 10, doodle_y - 15], fill=doodle_color, width=3)
            draw.line([doodle_x + 10, doodle_y + 10, doodle_x + 10, doodle_y - 15], fill=doodle_color, width=3)
            draw.line([doodle_x - 10, doodle_y - 15, doodle_x + 10, doodle_y - 15], fill=doodle_color, width=3)
            draw.line([doodle_x - 55, doodle_y - 30, doodle_x - 43, doodle_y - 20], fill=doodle_color, width=3)
            draw.line([doodle_x + 55, doodle_y - 30, doodle_x + 43, doodle_y - 20], fill=doodle_color, width=3)
            draw.line([doodle_x, doodle_y - 65, doodle_x, doodle_y - 52], fill=doodle_color, width=3)

    img.save(output_path, "JPEG", quality=95)
    return output_path

def generate_whiteboard_visuals(chunks, headline, is_longform=False):
    print(f"🎨 Generating whiteboard animation visuals for all {len(chunks)} chunks...")
    os.makedirs("output", exist_ok=True)
    for i, chunk in enumerate(chunks):
        cid = chunk.get("chunk_id", i + 1)
        text = chunk.get("text", "")
        output_path = os.path.join("output", f"whiteboard_{cid}_{TODAY}.jpg")
        path = generate_whiteboard_frame(text, output_path, headline=headline, is_longform=is_longform)
        chunk["visual_path"] = path
        chunk["visual_type"] = "photo"
        chunk["source"] = "Whiteboard Animation Frame"
        chunk["relevance_score"] = 10
    return chunks
