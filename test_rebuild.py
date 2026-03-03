from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- REBUILD TEXT VISIBILITY FIXES ---

def wrap_text_to_lines(words, word_widths, max_width):
    lines = []
    current_line = []
    current_w = 0
    font_black = ImageFont.truetype('assets/fonts/Montserrat-Black.ttf', 72)
    fake_img = Image.new("RGBA", (1,1))
    draw = ImageDraw.Draw(fake_img)
    space_w = draw.textlength(" ", font=font_black)
    
    for word, w in zip(words, word_widths):
        if not current_line or current_w + w <= max_width:
            current_line.append(word)
            current_w += w + space_w
        else:
            lines.append(current_line)
            current_line = [word]
            current_w = w + space_w
    if current_line:
        lines.append(current_line)
    return lines

def render_subtitle_frame(text, current_words, accent_color=(255,214,0), frame_width=1080, frame_height=1920):
    img = Image.new('RGBA', (frame_width, frame_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    font_black = ImageFont.truetype('assets/fonts/Montserrat-Black.ttf', 72)
    
    words = text.split()
    word_widths = []
    total_width = 0
    for word in words:
        bbox = draw.textbbox((0,0), word+" ", font=font_black)
        w = bbox[2] - bbox[0]
        word_widths.append(w)
        total_width += w
        
    max_width = 860
    if total_width > max_width:
        lines = wrap_text_to_lines(words, word_widths, max_width)
    else:
        lines = [words]
        
    line_height = 90
    total_text_height = len(lines) * line_height
    box_padding_x = 40
    box_padding_y = 28
    box_width = min(total_width + box_padding_x*2, max_width + box_padding_x*2)
    box_height = total_text_height + box_padding_y*2
    
    box_x = (frame_width - box_width) // 2
    box_y = 1250 + (230 - box_height) // 2
    
    overlay = Image.new('RGBA', (frame_width, frame_height), (0,0,0,0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    overlay_draw.rounded_rectangle(
        [box_x - box_padding_x, box_y - box_padding_y, box_x + box_width + box_padding_x, box_y + box_height + box_padding_y],
        radius=20, fill=(0, 0, 0, 215)
    )
    overlay_draw.rounded_rectangle(
        [box_x - box_padding_x - 2, box_y - box_padding_y - 2, box_x + box_width + box_padding_x + 2, box_y + box_height + box_padding_y + 2],
        radius=22, outline=(*accent_color, 255), width=2
    ) 
    
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    
    word_index = 0
    for line_num, line_words in enumerate(lines):
        line_y = box_y + line_num * line_height
        line_width = sum(word_widths[word_index:word_index+len(line_words)])
        line_x = (frame_width - line_width) // 2
        
        current_x = line_x
        for word in line_words:
            w = word_widths[word_index]
            if word in current_words.get('current', []):
                color = (255, 214, 0, 255)
            elif word in current_words.get('spoken', []):
                color = (187, 187, 187, 255)
            else:
                color = (255, 255, 255, 255)
                
            stroke_w = 8
            for dx in range(-stroke_w, stroke_w+1, 2):
                for dy in range(-stroke_w, stroke_w+1, 2):
                    if dx*dx + dy*dy <= stroke_w*stroke_w*1.2:
                        draw.text((current_x+dx, line_y+dy), word, font=font_black, fill=(0,0,0,255))
            draw.text((current_x+4, line_y+4), word, font=font_black, fill=(0,0,0,230))
            draw.text((current_x, line_y), word, font=font_black, fill=color)
            
            current_x += w
            word_index += 1
            
    return img

def render_header_bar(title, category, accent_color, frame_width=1080):
    header_height = 200
    img = Image.new('RGBA', (frame_width, header_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    for y in range(header_height):
        alpha = int(235 * (1 - (y / header_height)**0.7))
        draw.line([(0,y),(frame_width,y)], fill=(0,0,0,alpha))
        
    draw.line([(0, header_height-3), (frame_width, header_height-3)], fill=(*accent_color, 180), width=3)
    
    font_bold = ImageFont.truetype('assets/fonts/Montserrat-Bold.ttf', 30)
    badge_text = f"{category}"
    badge_bbox = draw.textbbox((0,0), badge_text, font=font_bold)
    badge_w = badge_bbox[2] - badge_bbox[0]
    badge_h = badge_bbox[3] - badge_bbox[1]
    
    badge_x = 40
    badge_y = 28
    badge_pad_x = 22
    badge_pad_y = 12
    
    draw.rounded_rectangle([badge_x, badge_y, badge_x + badge_w + badge_pad_x*2, badge_y + badge_h + badge_pad_y*2], radius=20, fill=(*accent_color, 255))
    for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
        draw.text((badge_x+badge_pad_x+dx, badge_y+badge_pad_y+dy), badge_text, font=font_bold, fill=(0,0,0,255))
    draw.text((badge_x+badge_pad_x, badge_y+badge_pad_y), badge_text, font=font_bold, fill=(255,255,255,255))
    
    font_handle = ImageFont.truetype('assets/fonts/Roboto-Bold.ttf', 28)
    handle_text = "t.me/technewsbyvj"
    handle_bbox = draw.textbbox((0,0), handle_text, font=font_handle)
    handle_w = handle_bbox[2] - handle_bbox[0]
    handle_x = frame_width - handle_w - 40
    handle_y = 38
    
    pad = 14
    draw.rounded_rectangle([handle_x-pad, handle_y-pad, handle_x+handle_w+pad, handle_y+28+pad], radius=16, fill=(0,0,0,200))
    for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
        draw.text((handle_x+dx, handle_y+dy), handle_text, font=font_handle, fill=(0,0,0,255))
    draw.text((handle_x, handle_y), handle_text, font=font_handle, fill=(204,204,204,255))
    
    font_title = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', 40)
    max_title_w = 900
    while True:
        t_bbox = draw.textbbox((0,0), title, font=font_title)
        t_w = t_bbox[2] - t_bbox[0]
        if t_w <= max_title_w:
            break
        title = title[:-4] + "..."
    title_x = (frame_width - t_w) // 2
    title_y = 110
    
    stroke_w = 7
    for dx in range(-stroke_w, stroke_w+1, 2):
        for dy in range(-stroke_w, stroke_w+1, 2):
            if dx*dx + dy*dy <= stroke_w*stroke_w*1.2:
                draw.text((title_x+dx, title_y+dy), title, font=font_title, fill=(0,0,0,255))
    draw.text((title_x+4, title_y+4), title, font=font_title, fill=(0,0,0,220))
    draw.text((title_x, title_y), title, font=font_title, fill=(255,255,255,255))
    draw.line([(title_x, title_y+52), (title_x+t_w, title_y+52)], fill=(*accent_color, 200), width=3)
    
    return img

def render_telegram_cta(accent_color, frame_width=1080):
    """Minimalist Telegram, LinkedIn & WhatsApp CTA banner for test."""
    card_height = 420
    img = Image.new('RGBA', (frame_width, card_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([0,0,frame_width,card_height], fill=(8,8,8,250))
    draw.line([(0,0),(frame_width,0)], fill=(*accent_color,255), width=4)
    
    f1_cta = ImageFont.truetype('assets/fonts/Montserrat-ExtraBold.ttf', 48)
    t1 = "t.me/technewsbyvj"
    t2 = "linkedin.com/in/vj"
    t3 = "WhatsApp: t.ly/vj-wa"
    
    try:
        tg_icon = Image.open('assets/icons/telegram_logo.png').convert("RGBA").resize((60, 60), Image.LANCZOS)
        li_icon = Image.open('assets/icons/linkedin_logo.png').convert("RGBA").resize((60, 60), Image.LANCZOS)
        wa_icon = Image.open('assets/icons/whatsapp_logo.png').convert("RGBA").resize((60, 60), Image.LANCZOS)
        
        # TG Row
        w1 = draw.textlength(t1, font=f1_cta)
        x1 = (frame_width - (w1 + 80)) // 2
        img.paste(tg_icon, (int(x1), 60), tg_icon)
        draw.text((x1 + 80, 65), t1, font=f1_cta, fill=(*accent_color, 255))
        
        # LI Row
        w2 = draw.textlength(t2, font=f1_cta)
        x2 = (frame_width - (w2 + 80)) // 2
        img.paste(li_icon, (int(x2), 160), li_icon)
        draw.text((x2 + 80, 165), t2, font=f1_cta, fill=(*accent_color, 255))

        # WA Row
        w3 = draw.textlength(t3, font=f1_cta)
        x3 = (frame_width - (w3 + 80)) // 2
        img.paste(wa_icon, (int(x3), 260), wa_icon)
        draw.text((x3 + 80, 265), t3, font=f1_cta, fill=(*accent_color, 255))
    except:
        x1 = (frame_width - draw.textlength(t1, font=f1_cta))//2
        draw.text((x1, 90), t1, font=f1_cta, fill=(*accent_color,255))
    
    return img

def composite_frame(background_frame, timestamp, header_img, subtitle_img, cta_img, video_duration):
    frame = Image.fromarray(background_frame).convert('RGBA')
    frame.alpha_composite(header_img, dest=(0,0))
    if subtitle_img is not None:
        frame.alpha_composite(subtitle_img, dest=(0,0))
    if timestamp >= video_duration - 6:
        progress = (timestamp-(video_duration-6))/0.4
        progress = min(progress, 1.0)
        progress = 1-(1-progress)**3
        cta_y = int(1580 + (1-progress)*340)
        frame.alpha_composite(cta_img, dest=(0, cta_y))
    return np.array(frame.convert('RGB'))

def verify_text_visibility(frame_array, zone_name, y_start, y_end):
    region = frame_array[y_start:y_end, 90:990]
    white_pixels = np.sum((region[:,:,0] > 200) & (region[:,:,1] > 200) & (region[:,:,2] > 200))
    dark_pixels = np.sum((region[:,:,0] < 50) & (region[:,:,1] < 50) & (region[:,:,2] < 50))
    has_text = white_pixels > 500
    has_bg = dark_pixels > 2000
    if not has_text:
        print(f"❌ {zone_name}: NO TEXT VISIBLE")
    if not has_bg:
        print(f"❌ {zone_name}: NO DARK BACKGROUND")
    if has_text and has_bg:
        print(f"✅ {zone_name}: Text visible correctly")
    return has_text and has_bg
