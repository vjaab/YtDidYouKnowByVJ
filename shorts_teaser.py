"""
shorts_teaser.py — Automatically extracts the single best fact from a completed 
long-form compilation, renders it as a 9:16 vertical Short, and uploads it to YouTube 
as a teaser with cross-promotion linking back to the full video.
"""
import os
import sys
import random
import traceback
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip, VideoClip
import moviepy.video.fx as vfx
from config import OUTPUT_DIR, ASSETS_DIR
from youtube_upload import upload_video

# Asset Font Loader Helper
def _load_teaser_font(size):
    font_paths = [
        os.path.join(ASSETS_DIR, "fonts", "Montserrat-ExtraBold.ttf"),
        os.path.join(ASSETS_DIR, "fonts", "Montserrat-Bold.ttf"),
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Arial.ttf"
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except:
                pass
    return ImageFont.load_default()

def draw_teaser_overlay(fact_num, hook_text, accent_color=(0, 240, 255)):
    """
    Creates a transparent 1080x1920 overlay with a premium top header and 
    bottom call-to-action banner for maximum viewer conversion.
    """
    img = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # ── TOP BANNER: "🤯 INSANE AI FACT X/10" ──────────────────────────────
    f_title = _load_teaser_font(52)
    title_text = f"🤯 INSANE AI FACT {fact_num}/10"
    
    # Text sizing fallback for PIL
    try:
        tw = draw.textbbox((0, 0), title_text, font=f_title)[2]
    except:
        tw = len(title_text) * 32
        
    pad_x, pad_y = 35, 18
    tx = (1080 - tw) // 2
    ty = 130
    
    # Neon background pill
    draw.rounded_rectangle(
        [tx - pad_x, ty - pad_y, tx + tw + pad_x, ty + 65 + pad_y], 
        radius=30, fill=(15, 15, 22, 230), outline=accent_color, width=4
    )
    
    # Shadow and text
    draw.text((tx + 3, ty + 3), title_text, font=f_title, fill=(0, 0, 0, 160))
    draw.text((tx, ty), title_text, font=f_title, fill=(255, 255, 255, 255))
    
    # ── BOTTOM BANNER: "👇 FULL VIDEO LINKED BELOW!" ─────────────────────
    f_cta = _load_teaser_font(48)
    cta_text = "👇 WATCH FULL VIDEO BELOW!"
    try:
        cw = draw.textbbox((0, 0), cta_text, font=f_cta)[2]
    except:
        cw = len(cta_text) * 28
        
    cx = (1080 - cw) // 2
    cy = 1680
    
    # Crimson pulse background pill
    draw.rounded_rectangle(
        [cx - pad_x, cy - pad_y, cx + cw + pad_x, cy + 60 + pad_y], 
        radius=30, fill=(220, 20, 60, 230), outline=(255, 255, 255, 180), width=3
    )
    
    draw.text((cx + 3, cy + 3), cta_text, font=f_cta, fill=(0, 0, 0, 160))
    draw.text((cx, cy), cta_text, font=f_cta, fill=(255, 255, 255, 255))
    
    return img

def generate_and_upload_shorts_teaser(script_json, longform_video_id, dry_run=False):
    """
    Extracts the best performing fact from a rendered longform video, 
    crops it to vertical 9:16, applies high-energy overlay assets, 
    and uploads it to YouTube as a viral Shorts teaser funneling back to the longform.
    """
    print("⚡ Starting Shorts Teaser Generation Engine...")
    
    fact_timestamps = script_json.get("fact_timestamps", [])
    if not fact_timestamps:
        print("⚠️ No fact timestamps available. Skipping Shorts Teaser.")
        return False
        
    # 1. Identify the Best Fact for Shorts
    best_fact_info = script_json.get("best_fact_for_shorts", {})
    best_fact_num = best_fact_info.get("fact_number", 1)
    
    print(f"🎯 Target teaser fact: Fact #{best_fact_num} (Reason: {best_fact_info.get('reason', 'Default')})")
    
    # Find timestamps for the chosen fact
    target_start = 0.0
    target_end = 0.0
    found = False
    
    for i, ft in enumerate(fact_timestamps):
        f_num = ft.get("fact_number", i + 1)
        if f_num == best_fact_num:
            target_start = float(ft.get("approx_start_seconds", 0))
            if i + 1 < len(fact_timestamps):
                target_end = float(fact_timestamps[i + 1].get("approx_start_seconds", target_start + 45))
            else:
                # Default duration to 50 seconds if last fact
                target_end = target_start + 50.0
            found = True
            break
            
    if not found:
        print("⚠️ Fact number not found in timestamps. Using Fact #1 as default.")
        target_start = float(fact_timestamps[0].get("approx_start_seconds", 0))
        if len(fact_timestamps) > 1:
            target_end = float(fact_timestamps[1].get("approx_start_seconds", 45))
        else:
            target_end = target_start + 45.0
            
    # Clip duration check (Shorts must be < 60s)
    duration = target_end - target_start
    if duration > 58.0:
        print(f"⚠️ Teaser duration ({duration:.1f}s) is too long for Shorts. Trimming to 58s.")
        target_end = target_start + 58.0
        duration = 58.0
        
    print(f"🎬 Slice Range: {target_start:.2f}s ➔ {target_end:.2f}s (Duration: {duration:.1f}s)")
    
    # 2. Slice and Smart Crop Longform Video
    # Locate longform output video file
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    longform_suffix = script_json.get("output_suffix", "")
    suffix_str = f"_{longform_suffix}" if longform_suffix else f"_{today}"
    
    longform_filename = f"video_longform{suffix_str}.mp4"
    longform_path = os.path.join(OUTPUT_DIR, longform_filename)
    
    if not os.path.exists(longform_path):
        # Fallback to scanning the output directory for any MP4
        import glob
        mp4_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*longform*.mp4")))
        if mp4_files:
            longform_path = mp4_files[-1]
        else:
            print(f"❌ Long-form video file not found at: {longform_path}. Skipping teaser.")
            return False
            
    print(f"📹 Slicing source longform video: {longform_path}...")
    
    try:
        lf_clip = VideoFileClip(longform_path)
        teaser_slice = lf_clip.subclipped(target_start, target_end)
        
        # Crop the center 9:16 segment (horizontal center is identical in explainer layouts)
        # 16:9 is 1920x1080 -> center 9:16 segment width is 1080 * 9 / 16 = 607.5px (use 608px for alignment)
        crop_w = 608
        x1 = (1920 - crop_w) // 2
        
        print("✂️ Cropping 16:9 to 9:16 vertical widescreen segment...")
        cropped_vertical = teaser_slice.cropped(x1=x1, y1=0, x2=x1 + crop_w, y2=1080)
        resized_vertical = cropped_vertical.resized((1080, 1920))
        
        # 3. Apply Premium Engagement Overlay
        accent_hex = script_json.get("color_theme", {}).get("accent", "#00E5FF").lstrip("#")
        accent_rgb = tuple(int(accent_hex[i:i+2], 16) for i in (0, 2, 4))
        
        overlay_img = draw_teaser_overlay(best_fact_num, best_fact_info.get("hook_for_shorts", ""), accent_rgb)
        overlay_arr = np.array(overlay_img)
        overlay_rgb = overlay_arr[:, :, :3]
        overlay_mask = (overlay_arr[:, :, 3] / 255.0).astype(float)
        
        overlay_clip = ImageClip(overlay_rgb, duration=duration)
        overlay_mask_clip = VideoClip(lambda t: overlay_mask, is_mask=True, duration=duration)
        overlay_clip = overlay_clip.with_mask(overlay_mask_clip)
        
        final_teaser = CompositeVideoClip([resized_vertical, overlay_clip], size=(1080, 1920)).with_duration(duration)
        
        # Export vertical teaser
        teaser_filename = f"shorts_teaser{suffix_str}.mp4"
        teaser_output_path = os.path.join(OUTPUT_DIR, teaser_filename)
        
        print(f"💾 Rendering vertical Shorts Teaser video to: {teaser_output_path}...")
        final_teaser.write_videofile(
            teaser_output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            threads=4,
            logger=None
        )
        print("✅ Shorts Teaser rendered successfully!")
        
        # 4. Upload Shorts Teaser to YouTube
        if dry_run:
            print("🏁 [DRY RUN] Teaser generated successfully. Skipping YouTube upload.")
            return True
            
        print("🚀 Uploading Shorts Teaser to YouTube...")
        longform_title = script_json.get("title", "10 AI Facts")
        teaser_title = f"This 1 fact changes everything... 🤯 #Shorts #AIFacts"
        
        teaser_description = (
            f"This is just 1 of {len(fact_timestamps)} insane AI facts. \n"
            f"Watch the FULL video here: https://youtu.be/{longform_video_id}\n\n"
            f"Daily cutting-edge tech intelligence by VJ.\n\n"
            f"#Shorts #TechNews #ArtificialIntelligence #VJTech"
        )
        
        tags = ["Shorts", "AIFacts", "TechNews", "VJTech", "AI", "Explainer"]
        
        uploaded, teaser_id = upload_video(
            video_path=teaser_output_path,
            title=teaser_title,
            description=teaser_description,
            tags=tags,
            thumbnail_path=None
        )
        
        if uploaded:
            print(f"🎉 Shorts Teaser live on YouTube: https://youtu.be/{teaser_id}")
            return True
        else:
            print(f"❌ YouTube Shorts Teaser upload failed: {teaser_id}")
            return False
            
    except Exception as e:
        print(f"❌ Shorts Teaser failed: {e}")
        traceback.print_exc()
        return False
