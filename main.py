import os
import argparse
import subprocess
import time
import requests
from datetime import datetime

import glob
from config import TARGET_AUDIO_DURATION, MAX_RETRY_ATTEMPTS, LOGS_DIR, OUTPUT_DIR
from fetch_research_papers import fetch_tech_news, fetch_ai_tools
from topic_tracker import record_story
from gemini_script import pick_and_generate_script
from ecosystem_logic import get_slot_info
from audio_gen import generate_voiceover
from chunk_builder import build_chunks, redistribute_to_audio_duration
from pexels_fetcher import fetch_all_chunk_visuals
from video_gen import create_video
from screenshot_gen import capture_article_screenshot
from thumbnail_gen import generate_thumbnail
from youtube_upload import upload_video
from telegram_selector import notify_telegram


def log_message(msg):
    today    = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOGS_DIR, f"log_{today}.txt")
    with open(log_path, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    print(msg)


def format_description(ai_description, script, end_question, hashtags):
    hashtag_str = " ".join(hashtags) if hashtags else ""
    
    # Build engagement question section
    question_section = ""
    if end_question:
        question_section = f"\n💬 {end_question}\n👇 CHALLENGE: 90% get this wrong. Comment your guess!\n"
    
    return f"""🚀 JOIN THE INNER CIRCLE (WhatsApp) → https://wa.me/919585793939
🔥 Get leaked AI tools & research daily before the masses.
━━━━━━━━━━━━━━━━━━━━━━
{ai_description}
{question_section}
━━━━━━━━━━━━━━━━━━━━━━
💡 In 2026, you're either the one using AI, or the one being replaced by it.

I share what top devs & AI engineers are reading right now:

🚀 Hottest AI research (before it goes viral)
💼 High-paying jobs & hiring alerts
🛠️ Dev tools & resources that save hours
📰 Deep tech research that actually matters

Don't miss out — join free today 👇

🚀 Telegram → https://t.me/technewsbyvj
💼 LinkedIn → https://www.linkedin.com/in/vijayakumar-j/
💬 WhatsApp Channel → https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z

🔗 (Links also in Channel Header/Bio!)
━━━━━━━━━━━━━━━━━━━━━━

{hashtag_str}
#airesearch #shorts #machinelearning #ai #youtubeshorts #dailyfacts"""


def run_pipeline(custom_topic=None, topic_type="research"):
    log_message(f"=== STARTING DAILY AI PIPIELINE ({topic_type.upper()}) ===")

    # ── Clean output folder before starting ───────────────────────────────────
    if os.path.exists(OUTPUT_DIR):
        for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except Exception:
                pass
        log_message(f"Output folder cleaned: {OUTPUT_DIR}")

    # ── STEP 1: Fetch News ────────────────────────────────────────────────────
    if custom_topic:
        log_message("STEP 1: Using Custom Topic...")
        news_articles = [{"title": "Custom Topic", "description": custom_topic, "url": "", "source": {"name": "User Input"}}]
    else:
        day_name, slot, category = get_slot_info()
        log_message(f"STEP 1: Content Ecosystem Check -> Day: {day_name}, Slot: {slot}, Category: {category}")
        
        log_message(f"STEP 1: Fetching Latest {topic_type.capitalize()} (Strategy: {category})...")
        # Logic: If category is Tool or Hands-on, fetch tools. Otherwise fetch research/news.
        if "Tool" in category or "Hands-on" in category:
            news_articles = fetch_ai_tools()
        else:
            news_articles = fetch_tech_news()
        
    if not news_articles:
        log_message("ERROR: No articles fetched. Aborting.")
        return False
    log_message(f"Fetched {len(news_articles)} articles.")

    # ── STEP 2: Auto-select Topic (Telegram interaction removed) ─────────────
    if custom_topic:
        log_message("STEP 2: Using Custom Topic...")
        chosen_article = news_articles[0]
    else:
        log_message("STEP 2: Telegram selection disabled — Gemini will auto-pick the best story.")
        chosen_article = None

    # ── STEP 3: Script Generation (with retry) ────────────────────────────────
    attempts = 0
    script_data = None
    audio_path  = None
    word_timestamps = []
    duration    = 0
    extra_instruction = ""
    min_dur, max_dur = TARGET_AUDIO_DURATION

    while attempts < MAX_RETRY_ATTEMPTS:
        log_message(f"STEP 3 (Attempt {attempts+1}): Generating Script...")
        script_data = pick_and_generate_script(
            news_articles, extra_instruction, forced_article=chosen_article, topic_type=topic_type
        )

        if not script_data:
            log_message("ERROR: Script generation failed.")
            attempts += 1
            continue

        title  = script_data.get("title", "Tech News!")
        script = script_data.get("script", "")
        voice  = script_data.get("edge_tts_voice", "en-US-AndrewNeural")
        emotion = script_data.get("edge_tts_emotion", "calm")
        log_message(f"Story: {script_data.get('original_news_headline')}")
        log_message(f"Breaking Level: {script_data.get('breaking_news_level')}")

        # ── STEP 4: Generate Audio + Word Timestamps ──────────────────────────
        log_message("STEP 4: Generating voiceover + word timestamps...")
        custom_map = script_data.get("phonetic_pronunciation_map", {})
        
        has_kaggle = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
        use_local_only = os.environ.get("USE_LOCAL_ONLY") == "true"
        
        if has_kaggle and not use_local_only:
            from kaggle_handover import trigger_kaggle_gpu_job
            results = trigger_kaggle_gpu_job(script_data, voice, emotion, custom_map)
            if results:
                audio_path = results.get("audio_path")
                duration = results.get("duration")
                word_timestamps = results.get("word_timestamps")
                script_data["kaggle_lipsync_path"] = results.get("lipsync_path")
                log_message("✅ Recieved Audio and Lip-Sync from Kaggle GPU!")
            else:
                log_message("⚠️ Kaggle Handover failed. Falling back to local generation.")
                audio_path, duration, word_timestamps = generate_voiceover(script, voice, emotion, custom_phonetic_map=custom_map)
        else:
            audio_path, duration, word_timestamps = generate_voiceover(script, voice, emotion, custom_phonetic_map=custom_map)
        

        if not audio_path:
            log_message("ERROR: Audio generation failed.")
            attempts += 1
            continue

        if duration < min_dur:
            log_message(f"Audio too short ({duration:.1f}s < {min_dur}s). Retrying...")
            extra_instruction = f"The previous script was too short at {duration:.0f}s. Make the script longer, aim for 65-70 seconds of speaking."
            attempts += 1
            continue

        log_message(f"Audio OK: {duration:.1f}s | {len(word_timestamps)} word timestamps")
        break   # ← success

    if not audio_path or not script_data or duration < min_dur:
        log_message("ERROR: Could not generate valid assets. Aborting.")
        return False

    # ── STEP 4b: Capture Article Screenshot ──────────────────────────────────
    log_message("STEP 4b: Capturing article screenshot...")
    news_url = script_data.get("original_news_url")
    if news_url:
        screenshot_filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        screenshot_path = capture_article_screenshot(news_url, screenshot_filename)
        if screenshot_path:
            script_data["screenshot_path"] = screenshot_path
            log_message(f"Screenshot captured: {screenshot_path}")
        else:
            log_message("Warning: Screenshot capture failed.")
    else:
        log_message("No article URL found for screenshot.")

    # ── STEP 5: Build Visual Chunks ───────────────────────────────────────────
    log_message("STEP 5: Grouping words into visual chunks...")
    from audio_gen import clean_tts_text
    sub_chunks = script_data.get("subtitle_chunks", [])
    for sc in sub_chunks:
        if "text" in sc:
            sc["text"] = clean_tts_text(sc["text"])
            
    chunks = build_chunks(word_timestamps, sub_chunks)
    chunks = redistribute_to_audio_duration(chunks, duration)
    log_message(f"Built {len(chunks)} visual chunks from {len(word_timestamps)} words.")

    # ── STEP 6: Fetch Entities (People/Companies) ─────────────────────────────
    log_message("STEP 6: Fetching entity photos and company logos...")
    from entity_fetcher import fetch_all_entities
    script_data = fetch_all_entities(script_data)

    # ── STEP 7: Fetch Per-Chunk Visuals (Decision Tree) ───────────────────────
    log_message("STEP 7: Fetching per-chunk visuals from Pexels/Imagen...")
    topic_context = script_data.get("original_news_headline", title)
    chunks = fetch_all_chunk_visuals(chunks, topic_context=topic_context, script_data=script_data)

    # ── STEP 8: Render Video ──────────────────────────────────────────────────
    log_message("STEP 8: Rendering final video with all engagement layers...")
    try:
        title  = script_data.get("title")
        script = script_data.get("script")
        subcat = script_data.get("sub_category", "")
        companies   = script_data.get("companies_mentioned", [])
        keywords    = script_data.get("keywords", [])
        hashtags    = script_data.get("hashtags", [])
        breaking_level = script_data.get("breaking_news_level", 0)
        voice_used  = script_data.get("edge_tts_voice")

        # Visual Variety Override for Anti-Bot Monetization
        import random
        visual_styles_palettes = [
            {"background": "#121212", "accent": "#00E5FF", "text": "#ffffff"}, # Cyber Cyan
            {"background": "#0D0D1A", "accent": "#FFD700", "text": "#ffffff"}, # Dark Gold
            {"background": "#1A0000", "accent": "#FF4444", "text": "#ffffff"}, # Deep Red
            {"background": "#0F1A12", "accent": "#00FF7F", "text": "#ffffff"}, # Hacker Green
            {"background": "#10002B", "accent": "#E0AAFF", "text": "#ffffff"}, # Neon Purple
            {"background": "#1A1A1D", "accent": "#F5A623", "text": "#ffffff"}  # Amber Black
        ]
        chosen_style = random.choice(visual_styles_palettes)
        # Override the AI's color theme with our explicit variety matrix
        script_data["color_theme"] = chosen_style

        video_path = create_video(audio_path, script_data, chunks)
        if not video_path or not os.path.exists(video_path):
            raise Exception("Video file not created.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log_message(f"ERROR: Video render failed: {e}")
        return False

    # ── STEP 8: Generate Thumbnail ────────────────────────────────────────────
    log_message("STEP 8: Generating thumbnail...")
    thumbnail_path = generate_thumbnail(script_data)

    # ── STEP 9: Automatic Upload (Consent removed) ──────────────────────────
    log_message("STEP 9: Telegram upload consent removed. Proceeding to upload...")
    approved = True

    # ── STEP 10: YouTube Upload ───────────────────────────────────────────────
    log_message("STEP 10: Uploading to YouTube...")
    ai_desc = script_data.get("description", "")
    description = format_description(ai_desc, script, script_data.get("end_question", ""), hashtags)
    # Ensure variety in titles using the options if generated
    if script_data.get("title_options"):
        title = random.choice(script_data["title_options"])
    
    # Algorithm Optimization: Append primary hashtag to title for Shorts feed boost
    if hashtags:
        primary_tag = hashtags[0]
        if primary_tag.lower() not in title.lower():
            # Ensure we don't exceed 100 character limit
            if len(title) + len(primary_tag) < 95:
                title = f"{title} {primary_tag}"

    tags = list(set(keywords + companies + [t.replace("#", "") for t in hashtags]))[:15]

    success, result = upload_video(video_path, title, description, tags)
    if not success:
        log_message(f"ERROR: YouTube upload failed: {result}")
        return False

    youtube_url = f"https://youtu.be/{result}"
    log_message(f"SUCCESS: {youtube_url}")

    # ── STEP 11: Update Tracker ───────────────────────────────────────────────
    log_message("STEP 11: Updating story tracker...")
    record_story(
        title, script_data.get("original_news_headline"),
        subcat, companies, keywords, breaking_level,
        voice_used, youtube_url, script_data.get("original_news_url")
    )

    # ── STEP 12: Cleanup Output Folder ────────────────────────────────────────
    log_message("STEP 12: Cleaning up output folder...")
    cleaned_count = 0
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
        try:
            if os.path.isfile(f):
                os.remove(f)
                cleaned_count += 1
        except Exception as e:
            log_message(f"Failed to delete {f}: {e}")

    log_message("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    
    notify_telegram(f"✅ Video Live & Pipeline Finished!\n\n{title}\n{youtube_url}\n\n🧹 Output folder cleaned ({cleaned_count} files removed).", "🚀")
    
    return True


def run_local(custom_topic=None, topic_type="research"):
    # XTTS server launch removed. Calling pipeline directly.
    run_pipeline(custom_topic, topic_type=topic_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Run pipeline immediately.")
    parser.add_argument("--topic", type=str, help="Run pipeline with a specific custom topic.", default=None)
    parser.add_argument("--type", type=str, choices=["research", "tools"], default="research", help="Content type mapped to the schedule")
    args = parser.parse_args()

    if args.now or args.topic:
        run_local(args.topic, topic_type=args.type)
    else:
        print("Usage: python main.py --now")
        print("For scheduled runs: python scheduler.py")
