"""
shorts_teaser.py — High-Energy Non-Technical Shorts Generator for YouTube.

Transforms deep-dive longform tech topics into viral, layperson-accessible 9:16 Shorts (45-55s):
  - Primary Media: High-energy Pexels portrait B-roll, UI animations, and product screenshots
  - Captions: Center-screen dynamic kinetic subtitles (1-3 words, neon colors, emojis)
  - Pacing: Rapid 1.5–2.5 seconds per visual cut + smooth zoom/punch-in
  - Graphics: Big bold callout badges ("WATCH THIS", "BEFORE vs AFTER", "SECRET LEAK")
  - Audio: Dedicated ELI5 voiceover narration + upbeat BGM
  - CTA: Clear funnel directing viewers to the full longform video
"""

import os
import sys
import random
import glob
import math
import traceback
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from moviepy import (
    VideoFileClip, ImageClip, CompositeVideoClip, VideoClip,
    AudioFileClip, CompositeAudioClip, concatenate_videoclips
)
import moviepy.video.fx as vfx

from config import OUTPUT_DIR, ASSETS_DIR, MUSIC_DIR
from config_longform import (
    LONGFORM_SHORTS_TEASER_DURATION,
    SHORTS_VISUAL_CUT_DURATION,
    SHORTS_CAPTION_Y_POS,
    SHORTS_MAX_WORDS_PER_CHUNK
)
from youtube_upload import upload_video
from tags_helper import get_optimized_metadata


# ─────────────────────────────────────────────────────────────────────────────
# FONT HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _load_teaser_font(size, bold=True):
    font_paths = [
        os.path.join(ASSETS_DIR, "fonts", "Montserrat-ExtraBold.ttf"),
        os.path.join(ASSETS_DIR, "fonts", "Montserrat-Bold.ttf"),
        os.path.join(ASSETS_DIR, "fonts", "Roboto-Bold.ttf"),
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHICS & OVERLAY RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def draw_callout_badge(text, emoji_icon="", accent_color=(255, 230, 0), width=1080):
    """
    Renders a high-energy alert badge overlay (e.g. 'WATCH THIS', 'BEFORE vs AFTER').
    Positioned in upper center (y ~ 300) safe from platform UI.
    """
    img = Image.new("RGBA", (width, 140), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    f_badge = _load_teaser_font(44, bold=True)
    
    full_text = f"{emoji_icon} {text.upper()}".strip() if emoji_icon else text.upper()
    try:
        tw = draw.textbbox((0, 0), full_text, font=f_badge)[2]
    except Exception:
        tw = len(full_text) * 26
        
    pad_x, pad_y = 36, 16
    cx = (width - tw) // 2
    cy = 30
    
    # Outer glowing pill
    draw.rounded_rectangle(
        [cx - pad_x, cy - pad_y, cx + tw + pad_x, cy + 56 + pad_y],
        radius=26, fill=(18, 18, 28, 235), outline=accent_color, width=4
    )
    # Drop shadow text
    draw.text((cx + 3, cy + 3), full_text, font=f_badge, fill=(0, 0, 0, 240))
    # Vibrant foreground text
    draw.text((cx, cy), full_text, font=f_badge, fill=(255, 255, 255, 255))
    return img


def draw_full_video_cta(accent_color=(0, 240, 255), width=1080):
    """
    Renders the high-contrast CTA banner directing viewers to the full deep dive.
    """
    img = Image.new("RGBA", (width, 180), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    f_cta_main = _load_teaser_font(42, bold=True)
    f_cta_sub = _load_teaser_font(28, bold=False)
    
    main_text = "👇 WATCH THE FULL DEEP DIVE"
    sub_text = "LINK IN DESCRIPTION & PINNED COMMENT"
    
    try:
        mw = draw.textbbox((0, 0), main_text, font=f_cta_main)[2]
        sw = draw.textbbox((0, 0), sub_text, font=f_cta_sub)[2]
    except Exception:
        mw, sw = len(main_text) * 26, len(sub_text) * 16
        
    pad_x, pad_y = 40, 16
    box_w = max(mw, sw) + pad_x * 2
    bx = (width - box_w) // 2
    by = 20
    
    draw.rounded_rectangle(
        [bx, by, bx + box_w, by + 130],
        radius=28, fill=(220, 20, 60, 240), outline=(255, 255, 255, 220), width=4
    )
    
    # Text lines
    mx = (width - mw) // 2
    sx = (width - sw) // 2
    draw.text((mx + 2, by + 18), main_text, font=f_cta_main, fill=(0, 0, 0, 180))
    draw.text((mx, by + 16), main_text, font=f_cta_main, fill=(255, 255, 255, 255))
    draw.text((sx + 1, by + 74), sub_text, font=f_cta_sub, fill=(0, 0, 0, 160))
    draw.text((sx, by + 73), sub_text, font=f_cta_sub, fill=(255, 230, 0, 255))
    return img


def draw_center_kinetic_caption(words_list, active_idx=0, active_color=(255, 230, 0), width=1080, height=1920):
    """
    Renders 1-3 words kinetic caption in the CENTER-SCREEN safe zone (y ~ 50-54%).
    Active word is scaled and highlighted with high-contrast outline.
    """
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    font_main = _load_teaser_font(72, bold=True)
    font_active = _load_teaser_font(78, bold=True)
    
    # Measure line total width
    word_widths = []
    for i, w in enumerate(words_list):
        f = font_active if i == active_idx else font_main
        try:
            bbox = draw.textbbox((0, 0), w, font=f)
            ww = bbox[2] - bbox[0]
        except Exception:
            ww = len(w) * 44
        word_widths.append(ww)
        
    space_w = 26
    total_w = sum(word_widths) + space_w * max(0, len(words_list) - 1)
    
    # Center-screen coordinates (y = 52% height)
    center_y = int(height * SHORTS_CAPTION_Y_POS)
    start_x = (width - total_w) // 2
    
    # Background capsule for readability over fast b-roll
    pad_x, pad_y = 36, 18
    draw.rounded_rectangle(
        [start_x - pad_x, center_y - pad_y, start_x + total_w + pad_x, center_y + 88 + pad_y],
        radius=24, fill=(10, 10, 18, 190), outline=(255, 255, 255, 90), width=2
    )
    
    # Draw each word
    cur_x = start_x
    for i, w in enumerate(words_list):
        f = font_active if i == active_idx else font_main
        color = active_color if i == active_idx else (255, 255, 255, 255)
        # Heavy drop shadow / outline
        for dx, dy in [(-3, -3), (3, -3), (-3, 3), (3, 3), (0, 4), (0, -4), (4, 0), (-4, 0)]:
            draw.text((cur_x + dx, center_y + dy), w, font=f, fill=(0, 0, 0, 240))
        draw.text((cur_x, center_y), w, font=f, fill=color)
        cur_x += word_widths[i] + space_w
        
    return img


# ─────────────────────────────────────────────────────────────────────────────
# B-ROLL CLIP BUILDER (FAST PACING & ZOOM PUNCH-IN)
# ─────────────────────────────────────────────────────────────────────────────
def _create_punch_in_clip_from_image(img_path_or_obj, duration, target_w=1080, target_h=1920):
    """Creates a 1.5-2.5s clip with a smooth punch-in zoom effect from an image."""
    if isinstance(img_path_or_obj, str):
        base_img = Image.open(img_path_or_obj).convert("RGB")
    else:
        base_img = img_path_or_obj.convert("RGB")
        
    # Resize / crop image to fill 1080x1920
    bw, bh = base_img.size
    scale = max(target_w / bw, target_h / bh) * 1.15  # 15% extra headroom for zoom
    nw, nh = int(bw * scale), int(bh * scale)
    scaled_base = base_img.resize((nw, nh), Image.LANCZOS)
    arr_base = np.array(scaled_base)
    
    def make_frame(t):
        progress = min(max(t / max(duration, 0.01), 0.0), 1.0)
        # Smooth ease-in zoom: scale 1.0 -> 1.10
        current_zoom = 1.0 + 0.10 * (progress * progress)
        cw = int(target_w * current_zoom)
        ch = int(target_h * current_zoom)
        
        # Center crop
        cx, cy = nw // 2, nh // 2
        x1 = max(0, cx - cw // 2)
        y1 = max(0, cy - ch // 2)
        crop = arr_base[y1:y1 + ch, x1:x1 + cw]
        
        # Resize cropped frame back to target dimensions
        crop_img = Image.fromarray(crop).resize((target_w, target_h), Image.BILINEAR)
        return np.array(crop_img)
        
    return VideoClip(make_frame, duration=duration)


def _create_punch_in_clip_from_video(video_path, duration, target_w=1080, target_h=1920):
    """Trims, center-crops to 9:16 vertical, and scales video with smooth punch-in."""
    v_clip = VideoFileClip(video_path).without_audio()
    if v_clip.duration < duration:
        # Loop if video is shorter than beat duration
        v_clip = v_clip.with_effects([vfx.Loop(duration=duration)])
    else:
        v_clip = v_clip.subclipped(0, duration)
        
    vw, vh = v_clip.w, v_clip.h
    # Scale to fill 1080x1920
    scale = max(target_w / vw, target_h / vh)
    new_w, new_h = int(vw * scale), int(vh * scale)
    v_clip = v_clip.resized((new_w, new_h))
    
    # Center crop to 1080x1920
    x1 = (new_w - target_w) // 2
    y1 = (new_h - target_h) // 2
    v_clip = v_clip.cropped(x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)
    return v_clip.with_duration(duration)


def _fetch_beat_media(beat_query, beat_dur, beat_idx, output_dir):
    """
    Fetches high-energy vertical 9:16 media from Pexels or Pixabay.
    Falls back to high-res tech photos or gradient visuals.
    """
    from pexels_fetcher import (
        _search_pexels_videos, _search_pexels_photos,
        _search_pixabay_videos, _search_pixabay_photos,
        _download_video, _download_photo
    )
    
    clean_query = beat_query.replace("-", " ").strip()
    
    # 1. Try Pexels vertical video
    try:
        v_results = _search_pexels_videos(clean_query, beat_dur, {"orientation": "portrait"})
        if v_results:
            cand = random.choice(v_results[:3])
            out_path = os.path.join(output_dir, f"short_beat_{beat_idx}_vid.mp4")
            if _download_video(cand["link"], out_path):
                return ("video", out_path)
    except Exception as e:
        print(f"   ⚠️ Pexels video fetch failed for '{clean_query}': {e}")

    # 2. Try Pexels vertical photo
    try:
        p_results = _search_pexels_photos(clean_query, orientation="portrait")
        if p_results:
            cand = random.choice(p_results[:3])
            out_path = os.path.join(output_dir, f"short_beat_{beat_idx}_img.jpg")
            if _download_photo(cand["link"], out_path, is_longform=False):
                return ("image", out_path)
    except Exception as e:
        print(f"   ⚠️ Pexels photo fetch failed for '{clean_query}': {e}")

    # 3. Dynamic tech visual fallback
    img = Image.new("RGB", (1080, 1920), (12, 16, 28))
    draw = ImageDraw.Draw(img)
    # Subtle abstract neon grid
    for gy in range(0, 1920, 80):
        alpha = int(30 + 15 * math.sin(gy / 100.0))
        draw.line([(0, gy), (1080, gy)], fill=(0, 200, 255, alpha), width=1)
    for gx in range(0, 1080, 80):
        draw.line([(gx, 0), (gx, 1920)], fill=(0, 200, 255, 30), width=1)
    out_path = os.path.join(output_dir, f"short_beat_{beat_idx}_fallback.png")
    img.save(out_path)
    return ("image", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR & UPLOADER
# ─────────────────────────────────────────────────────────────────────────────
def generate_and_upload_shorts_teaser(script_json, longform_video_id, dry_run=False):
    """
    Renders and uploads a dedicated Non-Technical Short (45-55s):
    - ELI5 analogy narrative
    - High-energy B-roll (Pexels) with rapid 1.5-2.5s pacing & punch-in zooms
    - Center-screen kinetic captions (1-3 words, bright colors, emojis)
    - Bold callout badges + end-screen longform CTA
    """
    print("\n⚡ [SHORTS ENGINE] Generating High-Energy Non-Technical Short...")
    
    try:
        layman_data = script_json.get("layman_short") or {}
        headline = script_json.get("original_news_headline") or script_json.get("title", "AI Breakthrough")
        
        # ── 1. NARRATION AUDIO & SCRIPT ──────────────────────────────────────────
        layman_script = layman_data.get("script", "")
        if not layman_script or len(layman_script.split()) < 30:
            print("⚠️ Dedicated layman script not found. Creating conversational ELI5 script...")
            layman_script = (
                f"Your phone is about to change and almost nobody noticed. "
                f"{headline}. Think of this like rush-hour highway traffic: when everyone tries to use AI, "
                f"everything bottlenecks. But this new update acts like opening express lanes for your device. "
                f"That means faster answers, zero lag, and much lower battery drain. "
                f"We just released the full deep-dive video with the complete evidence. "
                f"Tap the link below or check the pinned comment to watch the full story!"
            )
            
        print(f"📖 Shorts Script ({len(layman_script.split())} words):\n'{layman_script}'")
        
        # Generate voiceover for this short
        from audio_gen import generate_voiceover
        short_audio_path, subtitle_chunks = generate_voiceover(layman_script)
        
        if not short_audio_path or not os.path.exists(short_audio_path):
            print("❌ Shorts voiceover generation failed. Falling back to longform audio slice...")
            # Fallback to first chapter audio slice
            return False

        # Load audio clip and determine total runtime
        audio_clip = AudioFileClip(short_audio_path)
        total_duration = audio_clip.duration
        print(f"⏱️ Short Voiceover Duration: {total_duration:.2f}s (Target: 45-55s)")
        
        # ── 2. VISUAL BEATS & PACING (1.5 - 2.5s PER CUT) ────────────────────────
        visual_beats = layman_data.get("visual_beats", [])
        if not visual_beats:
            # Generate default 2.0s beats across runtime
            beat_count = max(4, int(total_duration / 2.0))
            queries = [
                "shocked person looking at smartphone", "busy highway traffic timelapse night",
                "futuristic neon server room", "hands typing fast on laptop keyboard",
                "smart city timelapse", "robot technology close up", "person watching youtube on phone"
            ]
            badges = ["WATCH THIS", "SECRET LEAK", "GAME CHANGER", "BEFORE vs AFTER", "NEW UPDATE", "ALERT"]
            visual_beats = []
            for bi in range(beat_count):
                visual_beats.append({
                    "beat_id": bi + 1,
                    "duration_seconds": total_duration / beat_count,
                    "broll_query": queries[bi % len(queries)],
                    "callout_badge": badges[bi % len(badges)] if bi in [0, 2, 4] else "",
                    "active_emoji": "⚡"
                })

        # Calculate exact duration per beat so total visual matches audio duration
        beat_dur = total_duration / max(1, len(visual_beats))
        print(f"🎬 Assembling {len(visual_beats)} rapid visual beats ({beat_dur:.2f}s per cut)...")

        rendered_cuts = []
        screenshot_path = script_json.get("screenshot_path") or script_json.get("evidence_screenshot_path")
        screenshot_used = False

        for idx, beat in enumerate(visual_beats):
            b_query = beat.get("broll_query", "modern tech artificial intelligence")
            
            # Insert evidence screenshot once at beat 2 or 3 if available
            if screenshot_path and os.path.exists(screenshot_path) and idx in [2, 3] and not screenshot_used:
                print(f"   📸 Beat {idx+1}: Inserting Evidence Screenshot: {screenshot_path}")
                cut_clip = _create_punch_in_clip_from_image(screenshot_path, beat_dur)
                screenshot_used = True
            else:
                m_type, m_path = _fetch_beat_media(b_query, beat_dur, idx, OUTPUT_DIR)
                print(f"   🎥 Beat {idx+1}/{len(visual_beats)} ({b_query[:25]}): {m_type}")
                if m_type == "video":
                    try:
                        cut_clip = _create_punch_in_clip_from_video(m_path, beat_dur)
                    except Exception as ve:
                        print(f"   ⚠️ Video processing failed, falling back to frame: {ve}")
                        cut_clip = _create_punch_in_clip_from_image(m_path, beat_dur)
                else:
                    cut_clip = _create_punch_in_clip_from_image(m_path, beat_dur)

            # Apply Callout Badge overlay if specified
            badge_text = beat.get("callout_badge", "")
            if badge_text:
                badge_img = draw_callout_badge(badge_text, beat.get("active_emoji", "🔥"))
                badge_arr = np.array(badge_img)
                badge_rgb = badge_arr[:, :, :3]
                badge_mask = VideoClip(lambda t: (badge_arr[:, :, 3] / 255.0).astype(float), is_mask=True, duration=beat_dur)
                badge_clip = ImageClip(badge_rgb, duration=beat_dur).with_mask(badge_mask).with_position(("center", 300))
                cut_clip = CompositeVideoClip([cut_clip, badge_clip], size=(1080, 1920)).with_duration(beat_dur)

            rendered_cuts.append(cut_clip)

        # Concatenate all visual cuts
        assembled_visual = concatenate_videoclips(rendered_cuts, method="compose").with_duration(total_duration)

        # ── 3. CENTER-SCREEN DYNAMIC KINETIC CAPTIONS ────────────────────────────
        print("💬 Rendering Center-Screen Kinetic Captions (1-3 words, Neon Yellow)...")
        # Build subtitle segments (group words into 1-3 word kinetic bursts)
        words_data = []
        if subtitle_chunks:
            for sc in subtitle_chunks:
                txt = sc.get("text", "").strip()
                if txt:
                    words_data.append({
                        "word": txt,
                        "start": sc.get("start", 0.0),
                        "end": sc.get("end", 0.0)
                    })
        else:
            # Fallback: distribute script words evenly across duration
            all_words = layman_script.split()
            w_step = total_duration / max(1, len(all_words))
            for wi, w in enumerate(all_words):
                words_data.append({
                    "word": w,
                    "start": wi * w_step,
                    "end": (wi + 1) * w_step
                })

        # Group words into 2-word bursts for kinetic punch
        caption_bursts = []
        chunk_size = SHORTS_MAX_WORDS_PER_CHUNK
        for i in range(0, len(words_data), chunk_size):
            slice_words = words_data[i:i + chunk_size]
            b_start = slice_words[0]["start"]
            b_end = slice_words[-1]["end"]
            words_text = [sw["word"] for sw in slice_words]
            caption_bursts.append({
                "words": words_text,
                "start": b_start,
                "end": max(b_end, b_start + 0.3),
                "word_starts": [sw["start"] for sw in slice_words]
            })

        # Create caption frame generator clip
        def make_caption_frame(t):
            active_burst = None
            active_word_idx = 0
            for b in caption_bursts:
                if b["start"] <= t < b["end"]:
                    active_burst = b
                    for w_idx, ws in enumerate(b["word_starts"]):
                        if t >= ws:
                            active_word_idx = w_idx
                    break
                    
            if active_burst:
                cap_img = draw_center_kinetic_caption(
                    active_burst["words"],
                    active_idx=active_word_idx,
                    active_color=(255, 230, 0)
                )
                return np.array(cap_img)
            else:
                return np.zeros((1920, 1080, 4), dtype=np.uint8)

        caption_video_clip = VideoClip(lambda t: make_caption_frame(t)[:, :, :3], duration=total_duration)
        caption_mask_clip = VideoClip(lambda t: (make_caption_frame(t)[:, :, 3] / 255.0).astype(float), is_mask=True, duration=total_duration)
        kinetic_caption_clip = caption_video_clip.with_mask(caption_mask_clip)

        # ── 4. FULL VIDEO CTA BANNER (FINAL 6 SECONDS) ───────────────────────────
        cta_start_time = max(0.0, total_duration - 6.0)
        cta_duration = total_duration - cta_start_time
        cta_img = draw_full_video_cta()
        cta_arr = np.array(cta_img)
        cta_rgb = cta_arr[:, :, :3]
        cta_mask = VideoClip(lambda t: (cta_arr[:, :, 3] / 255.0).astype(float), is_mask=True, duration=cta_duration)
        cta_clip = (
            ImageClip(cta_rgb, duration=cta_duration)
            .with_mask(cta_mask)
            .with_position(("center", 1680))
            .with_start(cta_start_time)
        )

        # ── 5. AUDIO MIXING (VOICEOVER + UPBEAT BGM) ─────────────────────────────
        audio_tracks = [audio_clip]
        bgm_files = glob.glob(os.path.join(MUSIC_DIR, "*.mp3"))
        if bgm_files:
            try:
                bgm_path = bgm_files[0]
                bgm_clip = AudioFileClip(bgm_path).with_effects([vfx.Loop(duration=total_duration)])
                # Duck BGM under voiceover (volume 0.10)
                bgm_clip = bgm_clip.with_volume_scaled(0.10)
                audio_tracks.append(bgm_clip)
                print(f"🎵 Layered background music: {os.path.basename(bgm_path)}")
            except Exception as bgm_err:
                print(f"⚠️ BGM loading failed (non-fatal): {bgm_err}")

        final_audio = CompositeAudioClip(audio_tracks).with_duration(total_duration)

        # ── 6. COMPOSITE & RENDER ────────────────────────────────────────────────
        final_short = CompositeVideoClip(
            [assembled_visual, kinetic_caption_clip, cta_clip],
            size=(1080, 1920)
        ).with_duration(total_duration).with_audio(final_audio)

        today_str = script_json.get("output_suffix", "")
        short_filename = f"short_nontechnical_{today_str or 'latest'}.mp4"
        short_output_path = os.path.join(OUTPUT_DIR, short_filename)

        print(f"💾 Exporting High-Energy 9:16 Short to: {short_output_path}...")
        final_short.write_videofile(
            short_output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            preset="fast",
            threads=4,
            logger=None
        )
        print("✅ Non-Technical Short rendered successfully!")

        # ── 7. UPLOAD TO YOUTUBE SHORTS ──────────────────────────────────────────
        if dry_run:
            print("🏁 [DRY RUN] Non-Technical Short generated successfully. Skipping upload.")
            return True

        short_title = layman_data.get("title") or f"{headline[:50]} 🤯 #Shorts"
        if not short_title.endswith("#Shorts"):
            short_title = f"{short_title[:50]} #Shorts"

        longform_url = f"https://youtu.be/{longform_video_id}"
        short_description = (
            f"Here is the real-world truth in 50 seconds.\n\n"
            f"👉 WATCH THE FULL IN-DEPTH STORY HERE: {longform_url}\n\n"
            f"Explained with simple analogies (ELI5). No jargon, pure substance.\n\n"
            f"Daily cutting-edge tech intelligence by VJ.\n\n"
            f"#Shorts #TechNews #AI #DidYouKnow"
        )

        optimized_meta = get_optimized_metadata(
            title=short_title,
            script=layman_script,
            sub_category=script_json.get("sub_category", "Tech News"),
            initial_keywords=script_json.get("keywords", ["AI", "Tech"]),
            initial_companies=script_json.get("companies_mentioned", []),
            initial_hashtags=["#Shorts", "#AI", "#TechNews"],
            is_shorts=True
        )

        # Generate thumbnail frame from 1/3 mark
        short_thumb_path = None
        try:
            tc = VideoFileClip(short_output_path)
            thumb_frame = tc.get_frame(min(3.0, total_duration / 3.0))
            thumb_img = Image.fromarray(thumb_frame)
            short_thumb_path = os.path.join(OUTPUT_DIR, f"thumb_short_{today_str or 'latest'}.jpg")
            thumb_img.save(short_thumb_path, "JPEG", quality=90)
            tc.close()
        except Exception as thumb_err:
            print(f"⚠️ Short thumbnail generation failed: {thumb_err}")

        print(f"🚀 Uploading Non-Technical Short to YouTube: '{short_title}'...")
        uploaded, short_video_id = upload_video(
            video_path=short_output_path,
            title=short_title,
            description=short_description,
            tags=optimized_meta.get("tags", ["Shorts", "AI", "Tech"]),
            thumbnail_path=short_thumb_path
        )

        if uploaded:
            print(f"🎉 Non-Technical Short is LIVE: https://youtu.be/{short_video_id}")
            return True
        else:
            print(f"❌ YouTube upload failed: {short_video_id}")
            return False

    except Exception as e:
        print(f"❌ Error generating Non-Technical Short: {e}")
        traceback.print_exc()
        return False