"""
main_longform.py — Daily "Did You Know" Long-Form AI Video Pipeline.

Produces a 3-minute, 16:9 landscape compilation video covering the top 5
viral AI topics of the day. Mirrors the Shorts pipeline (main.py) but adapted
for the long-form "Did You Know" format.

Usage:
  python main_longform.py --now         # Run immediately
  python main_longform.py --dry-run     # Test without upload
"""

import os
os.environ["PYTHONHASHSEED"] = "0"
import argparse
import time
import sys
import glob
import hashlib
import random
import traceback
from datetime import datetime

from config import LOGS_DIR, OUTPUT_DIR, GEMINI_API_KEY, MUSIC_DIR
from config_longform import (
    LONGFORM_TARGET_AUDIO_DURATION, LONGFORM_NUM_TOPICS,
    LONGFORM_BGM_VOLUME, LONGFORM_MAX_RETRY_ATTEMPTS,
    LONGFORM_TRACKER_FILE
)
from fetch_research_papers import fetch_tech_news, fetch_ai_tools, fetch_trending_from_newsapi
from topic_tracker import record_story, update_youtube_url
from gemini_script_longform import generate_longform_script
from audio_gen import generate_voiceover, clean_tts_text
from chunk_builder import build_chunks, redistribute_to_audio_duration
from pexels_fetcher import generate_visual_style_guide
from nano_scene_gen import generate_nano_scene_visuals
from video_gen import create_video
from screenshot_gen import capture_article_screenshot
from thumbnail_gen import generate_thumbnail
from youtube_upload import upload_video
from telegram_selector import notify_telegram
from entity_fetcher import fetch_all_entities, get_retention_layers_config


def log_message(msg):
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOGS_DIR, f"log_longform_{today}.txt")
    with open(log_path, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    print(msg)


def format_longform_description(script_data, hashtags):
    """Generate a rich YouTube description for the long-form compilation."""
    title = script_data.get("title", "5 AI Facts")
    fact_timestamps = script_data.get("fact_timestamps", [])
    topics = script_data.get("longform_topics", [])
    description_ai = script_data.get("description", "")
    
    hashtag_str = " ".join(hashtags) if hashtags else ""
    
    # Build timestamps section
    timestamps_str = "📌 TIMESTAMPS:\n"
    for ft in fact_timestamps:
        approx_s = ft.get("approx_start_seconds", 0)
        m, s = divmod(int(approx_s), 60)
        topic = ft.get("topic", f"Fact {ft.get('fact_number', '?')}")[:50]
        timestamps_str += f"{m}:{s:02d} — {topic}\n"
    
    # Build sources section
    sources_str = "📚 SOURCES:\n"
    for t in topics[:5]:
        url = t.get("source_url", "")
        name = t.get("source_name", "")
        headline = t.get("headline", "")[:60]
        if url:
            sources_str += f"🔗 {headline} ({name}): {url}\n"

    # Rotate between description templates
    desc_seed = int(hashlib.md5(title.encode()).hexdigest(), 16)
    template_idx = desc_seed % 3

    templates = [
        f"""🧠 {description_ai}

{timestamps_str}
━━━━━━━━━━━━━━━━━━━━━━
{sources_str}
━━━━━━━━━━━━━━━━━━━━━━
🚀 Get daily AI news before it trends:
📲 Telegram → https://t.me/technewsbyvj
💬 WhatsApp → https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
💼 LinkedIn → https://www.linkedin.com/in/vijayakumar-j/

⚠️ DISCLOSURE: This video uses AI-assisted production tools (TTS voiceover, AI-generated visuals). All editorial opinions, topic selection, and analysis are by VJ.

{hashtag_str}
#DidYouKnow #AIFacts #TechFacts #ArtificialIntelligence #MachineLearning""",

        f"""⚡ 5 AI facts that will blow your mind — all from the last 48 hours.

{timestamps_str}
{sources_str}
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
📬 Join the community for daily AI intelligence:
→ Telegram: https://t.me/technewsbyvj
→ WhatsApp: https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
→ LinkedIn: https://www.linkedin.com/in/vijayakumar-j/

⚠️ DISCLOSURE: AI tools are used in production (voiceover, visuals). Topic selection, research, and commentary by VJ.

{hashtag_str}
#AI #DidYouKnow #TechNews #DeepLearning #Innovation""",

        f"""👆 Which fact shocked you the most? Comment the number!

{timestamps_str}
━━━━━━━━━━━━━━━━━━━━━━
{sources_str}
🏗️ Follow for more daily AI compilations:
• 5 mind-blowing facts every day
• Curated by VJ — no fluff, no filler

📲 https://t.me/technewsbyvj
💬 https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
💼 https://www.linkedin.com/in/vijayakumar-j/

⚠️ DISCLOSURE: AI-assisted production (voiceover, visuals). Editorial direction & analysis by VJ.

{hashtag_str}
#AIDidYouKnow #TechFacts #Coding #Startup #DevTools""",
    ]

    return templates[template_idx]


def run_longform_pipeline(dry_run=False):
    """Main long-form pipeline: 5 topics → 3-min 16:9 video → YouTube."""
    log_message("=== STARTING DAILY LONG-FORM 'DID YOU KNOW' PIPELINE ===")

    # ── Clean output folder ──────────────────────────────────────────────
    if os.path.exists(OUTPUT_DIR):
        for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except Exception:
                pass
        log_message(f"Output folder cleaned: {OUTPUT_DIR}")

    # ── STEP 1: Fetch RSS + NewsAPI Articles ─────────────────────────────
    log_message("STEP 1: Fetching RSS articles (Research + Tools + Trending)...")
    rss_articles = []
    try:
        research_news = fetch_tech_news()
        ai_tool_news = fetch_ai_tools()
        trending_news = fetch_trending_from_newsapi()
        rss_articles = research_news + ai_tool_news + trending_news

        if not rss_articles:
            log_message("⚠️ All sources returned 0 articles. Pipeline will rely on Gemini Search.")
        else:
            log_message(f"✅ Fetched {len(rss_articles)} total articles "
                       f"({len(research_news)} research, {len(ai_tool_news)} tools, {len(trending_news)} trending).")
    except Exception as e:
        log_message(f"⚠️ RSS Fetch failed: {e}")

    # ── STEP 2: Generate Long-Form Script ────────────────────────────────
    attempts = 0
    script_data = None
    audio_path = None
    word_timestamps = []
    duration = 0
    min_dur, max_dur = LONGFORM_TARGET_AUDIO_DURATION
    failed_topics = []

    while attempts < LONGFORM_MAX_RETRY_ATTEMPTS:
        log_message(f"STEP 2 (Attempt {attempts + 1}/{LONGFORM_MAX_RETRY_ATTEMPTS}): "
                   f"Generating long-form 5-topic compilation script...")

        script_data = generate_longform_script(
            articles=rss_articles,
            failed_topics=failed_topics
        )

        if not script_data:
            log_message("ERROR: Long-form script generation failed.")
            attempts += 1
            if attempts % 3 == 0:
                log_message("⏳ Potential rate limit. Sleeping 60s...")
                time.sleep(60)
            continue

        # Mark as longform for video_gen
        script_data["slot"] = "Slot L (Long-form)"
        script_data["is_longform"] = True

        title = script_data.get("title", "5 AI Facts!")
        script = script_data.get("script", "")
        log_message(f"Generated Title: {title}")
        log_message(f"Script word count: {len(script.split())}")
        
        topics = script_data.get("longform_topics", [])
        log_message(f"Topics covered: {len(topics)}")
        for i, t in enumerate(topics):
            log_message(f"  Fact {i+1}: {t.get('headline', 'Unknown')}")

        # ── STEP 2b: Capture Screenshots for Each Topic ──────────────────
        log_message("STEP 2b: Capturing article screenshots for evidence...")
        screenshots_captured = 0
        for i, topic in enumerate(topics):
            url = topic.get("source_url", "")
            if url:
                ss_filename = f"screenshot_longform_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                ss_path = capture_article_screenshot(url, ss_filename)
                if ss_path:
                    topic["screenshot_path"] = ss_path
                    screenshots_captured += 1
                    if i == 0:
                        # Use first topic's screenshot as primary
                        script_data["screenshot_path"] = ss_path
        
        log_message(f"✅ Captured {screenshots_captured}/{len(topics)} article screenshots.")
        
        # For longform, screenshots are helpful but not mandatory (unlike Shorts)
        if not script_data.get("screenshot_path") and screenshots_captured == 0:
            log_message("⚠️ No screenshots captured. Proceeding anyway (long-form is less dependent).")

        # ── STEP 3: Generate Audio ───────────────────────────────────────
        log_message("STEP 3: Generating voiceover + word timestamps...")
        custom_map = script_data.get("phonetic_pronunciation_map", {})

        # Select intro video for lip-sync (rotation)
        intro_videos = glob.glob("assets/video/*.mp4")
        if not intro_videos:
            intro_videos = ["assets/video/Firefly_video_final.mp4"]
        
        headline = script_data.get("original_news_headline", "")
        video_idx = int(hashlib.md5(headline.encode()).hexdigest(), 16) % len(intro_videos)
        script_data["lipsync_face_path"] = intro_videos[video_idx]
        log_message(f"Selected Lip-Sync Template: {intro_videos[video_idx]}")

        audio_path, duration, word_timestamps = generate_voiceover(
            script, custom_phonetic_map=custom_map, api_key=GEMINI_API_KEY
        )

        if not audio_path:
            log_message("ERROR: Audio generation failed.")
            attempts += 1
            continue

        if duration < min_dur:
            log_message(f"Audio too short ({duration:.1f}s < {min_dur}s). Retrying with longer script...")
            attempts += 1
            continue

        if duration > max_dur + 30:  # Allow 30s grace for long-form
            log_message(f"Audio too long ({duration:.1f}s > {max_dur + 30}s). Retrying...")
            attempts += 1
            continue

        log_message(f"Audio OK: {duration:.1f}s | {len(word_timestamps)} word timestamps")
        break  # ← Success

    if not audio_path or not script_data:
        log_message("ERROR: Could not generate valid assets. Aborting.")
        return False

    # ── STEP 4: Reserve Topics in Tracker ────────────────────────────────
    log_message("STEP 4: Reserving topics in tracker...")
    title = script_data.get("title", "5 AI Facts!")
    keywords = script_data.get("keywords", [])
    hashtags = script_data.get("hashtags", [])

    # Record the compilation as a single entry
    companies_all = []
    for t in script_data.get("longform_topics", []):
        companies_all.append(t.get("source_name", ""))
    
    record_story(
        title, script_data.get("original_news_headline"),
        "AI Did You Know", companies_all, keywords, 7,
        "compilation", "pending_upload", script_data.get("original_news_url"),
        tracker_file=LONGFORM_TRACKER_FILE
    )

    # ── STEP 5: Build Visual Chunks ──────────────────────────────────────
    log_message("STEP 5: Grouping words into visual chunks...")
    sub_chunks = script_data.get("subtitle_chunks", [])
    for sc in sub_chunks:
        if "text" in sc:
            sc["text"] = clean_tts_text(sc["text"], phonetic=False)

    chunks = build_chunks(word_timestamps, sub_chunks)
    chunks = redistribute_to_audio_duration(chunks, duration)
    log_message(f"Built {len(chunks)} visual chunks from {len(word_timestamps)} words.")

    # ── STEP 6: Fetch Entities ───────────────────────────────────────────
    log_message("STEP 6: Fetching entity photos and company logos...")
    script_data = fetch_all_entities(script_data)

    retention_config = get_retention_layers_config()
    script_data["retention_config"] = retention_config

    # ── STEP 7: Generate Per-Sentence Visuals (16:9) ─────────────────────
    log_message("STEP 7: Generating per-sentence nano-scene backgrounds (16:9 Imagen)...")
    topic_context = f"Did You Know AI Facts Compilation: {title}"
    style_guide = generate_visual_style_guide(topic_context)

    chunks = generate_nano_scene_visuals(chunks, topic_context, style_guide=style_guide, aspect_ratio="16:9")

    nano_success = sum(1 for c in chunks if c.get("visual_path") and "Nano-Scene" in c.get("source", ""))
    if nano_success < len(chunks) * 0.5:
        log_message(f"⚠️ Nano-scene only generated {nano_success}/{len(chunks)} visuals. Using fallback...")
        from pexels_fetcher import fetch_all_chunk_visuals
        chunks = fetch_all_chunk_visuals(chunks, topic_context=topic_context, script_data=script_data, is_longform=True)
    else:
        log_message(f"✅ Nano-scene generated {nano_success}/{len(chunks)} per-sentence backgrounds.")

    # ── STEP 8: Visual Variety (Color Theme) ─────────────────────────────
    visual_palettes = [
        {"background": "#121212", "accent": "#00E5FF", "text": "#ffffff"},
        {"background": "#0D0D1A", "accent": "#FFD700", "text": "#ffffff"},
        {"background": "#1A0000", "accent": "#FF4444", "text": "#ffffff"},
        {"background": "#0F1A12", "accent": "#00FF7F", "text": "#ffffff"},
        {"background": "#10002B", "accent": "#E0AAFF", "text": "#ffffff"},
        {"background": "#1A1A1D", "accent": "#F5A623", "text": "#ffffff"},
    ]
    script_data["color_theme"] = random.choice(visual_palettes)

    # ── STEP 9: Render Video (16:9) ──────────────────────────────────────
    log_message("STEP 9: Rendering final 16:9 video with engagement layers...")
    try:
        video_path = create_video(audio_path, script_data, chunks)
        if not video_path or not os.path.exists(video_path):
            raise Exception("Video file not created.")
    except Exception as e:
        traceback.print_exc()
        log_message(f"ERROR: Video render failed: {e}")
        return False

    # ── STEP 10: Generate Thumbnail (16:9) ───────────────────────────────
    log_message("STEP 10: Generating 16:9 thumbnail...")
    thumbnail_path = generate_thumbnail(script_data)

    if dry_run:
        log_message("🏁 DRY RUN complete. Skipping upload.")
        log_message(f"   Video: {video_path}")
        log_message(f"   Thumbnail: {thumbnail_path}")
        return True

    # ── STEP 11: Upload to YouTube ───────────────────────────────────────
    log_message("STEP 11: Uploading to YouTube...")
    description = format_longform_description(script_data, hashtags)

    # Title selection
    if script_data.get("title_options"):
        title = random.choice(script_data["title_options"])

    tags = list(set(keywords + [t.replace("#", "") for t in hashtags]))[:15]

    uploaded, result = upload_video(
        video_path, title, description, tags,
        thumbnail_path=thumbnail_path,
        comment_hook=script_data.get("comment_hook")
    )
    if not uploaded:
        log_message(f"ERROR: YouTube upload failed: {result}")
        return False

    youtube_url = f"https://youtu.be/{result}"
    log_message(f"SUCCESS: {youtube_url}")

    # ── STEP 12: Update Tracker ──────────────────────────────────────────
    update_youtube_url(script_data.get("original_news_headline"), youtube_url, tracker_file=LONGFORM_TRACKER_FILE)

    # ── STEP 13: Cleanup ─────────────────────────────────────────────────
    log_message("STEP 13: Cleaning up output folder...")
    cleaned = 0
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
        try:
            if os.path.isfile(f):
                os.remove(f)
                cleaned += 1
        except Exception:
            pass

    log_message("=== LONG-FORM PIPELINE COMPLETED SUCCESSFULLY ===")

    notify_telegram(
        f"✅ Long-Form Video Live!\n\n{title}\n{youtube_url}\n\n"
        f"📊 Duration: {duration:.0f}s | Topics: {len(script_data.get('longform_topics', []))}\n"
        f"🧹 Output cleaned ({cleaned} files).",
        "🎬"
    )

    # ── STEP 14: Kill orphan processes ───────────────────────────────────
    log_message("STEP 14: Terminating leftover ffmpeg processes...")
    try:
        os.system("pkill -9 -f ffmpeg")
    except Exception:
        pass

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily 'Did You Know' Long-Form AI Video Pipeline")
    parser.add_argument("--now", action="store_true", help="Run pipeline immediately.")
    parser.add_argument("--dry-run", action="store_true", help="Run without uploading to YouTube.")
    args = parser.parse_args()

    if args.now or args.dry_run:
        success = run_longform_pipeline(dry_run=args.dry_run)
        if not success:
            print("❌ Long-form pipeline failed. Exiting with error code.")
            sys.exit(1)
    else:
        print("Usage:")
        print("  python main_longform.py --now        # Run immediately")
        print("  python main_longform.py --dry-run    # Test without upload")
