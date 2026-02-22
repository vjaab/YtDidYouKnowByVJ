import os
import argparse
from datetime import datetime

import glob
from config import TARGET_AUDIO_DURATION, MAX_RETRY_ATTEMPTS, LOGS_DIR, OUTPUT_DIR
from fetch_news import fetch_tech_news
from topic_tracker import record_story
from gemini_script import pick_and_generate_script
from audio_gen import generate_voiceover
from chunk_builder import build_chunks, redistribute_to_audio_duration
from pexels_fetcher import fetch_all_chunk_visuals
from video_gen import create_video
from thumbnail_gen import generate_thumbnail
from youtube_upload import upload_video
from telegram_selector import send_topic_selection, send_upload_consent, notify_telegram


def log_message(msg):
    today    = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOGS_DIR, f"log_{today}.txt")
    with open(log_path, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    print(msg)


def format_description(script, end_question, hashtags):
    hashtag_str = " ".join(hashtags) if hashtags else ""
    return f"""{script}

━━━━━━━━━━━━━━━━━━━━━━
💡 Every day you're not learning, someone else is getting ahead.

I share what top devs & AI engineers are reading right now:

🚀 Hottest AI research (before it goes viral)
💼 High-paying jobs & hiring alerts
🛠️ Dev tools & resources that save hours
📰 Tech news that actually matters

Don't miss out — join free today 👇

📲 Telegram → https://t.me/technewsbyvj
━━━━━━━━━━━━━━━━━━━━━━

{hashtag_str}
#technews #shorts #tech #ai #youtubeshorts #dailyfacts"""


def run_pipeline():
    log_message("=== STARTING DAILY TECH NEWS SHORTS PIPELINE ===")

    # ── STEP 1: Fetch News ────────────────────────────────────────────────────
    log_message("STEP 1: Fetching Latest AI/Tech News (last 24h)...")
    news_articles = fetch_tech_news()
    if not news_articles:
        log_message("ERROR: No articles fetched. Aborting.")
        return False
    log_message(f"Fetched {len(news_articles)} articles.")

    # ── STEP 2: Telegram topic selection ─────────────────────────────────────
    log_message("STEP 2: Sending topic list to Telegram for your selection...")
    chosen_article = send_topic_selection(news_articles)
    if chosen_article:
        log_message(f"User selected: {chosen_article.get('title')}")
    else:
        log_message("No Telegram selection — Gemini will auto-pick best story.")

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
            news_articles, extra_instruction, forced_article=chosen_article
        )

        if not script_data:
            log_message("ERROR: Script generation failed.")
            attempts += 1
            continue

        title  = script_data.get("title", "Tech News!")
        script = script_data.get("script", "")
        voice  = script_data.get("edge_tts_voice", "en-US-GuyNeural")
        emotion = script_data.get("edge_tts_emotion", "excited")
        log_message(f"Story: {script_data.get('original_news_headline')}")
        log_message(f"Breaking Level: {script_data.get('breaking_news_level')}")

        # ── STEP 4: Generate Audio + Word Timestamps ──────────────────────────
        log_message("STEP 4: Generating voiceover + word timestamps...")
        audio_path, duration, word_timestamps = generate_voiceover(script, voice, emotion)

        if not audio_path:
            log_message("ERROR: Audio generation failed.")
            attempts += 1
            continue

        if duration < min_dur:
            log_message(f"Audio too short ({duration:.1f}s < {min_dur}s). Retrying...")
            extra_instruction = "The previous script was too short. Expand it significantly."
            attempts += 1
            continue
        elif duration > max_dur:
            log_message(f"Audio too long ({duration:.1f}s > {max_dur}s). Retrying...")
            extra_instruction = "The previous script was too long. Make it shorter."
            attempts += 1
            continue

        log_message(f"Audio OK: {duration:.1f}s | {len(word_timestamps)} word timestamps")
        break   # ← success

    if not audio_path or not script_data or not (min_dur <= duration <= max_dur):
        log_message("ERROR: Could not generate valid assets. Aborting.")
        return False

    # ── STEP 5: Build Visual Chunks ───────────────────────────────────────────
    log_message("STEP 5: Grouping words into visual chunks...")
    chunks = build_chunks(word_timestamps)
    chunks = redistribute_to_audio_duration(chunks, duration)
    log_message(f"Built {len(chunks)} visual chunks from {len(word_timestamps)} words.")

    # ── STEP 6: Fetch Per-Chunk Visuals (Pexels video → photo → Imagen) ───────
    log_message("STEP 6: Fetching per-chunk visuals from Pexels...")
    topic_context = script_data.get("original_news_headline", title)
    chunks = fetch_all_chunk_visuals(chunks, topic_context=topic_context)

    # ── STEP 7: Render Video ──────────────────────────────────────────────────
    log_message("STEP 7: Rendering final video with karaoke subtitles...")
    try:
        title  = script_data.get("title")
        script = script_data.get("script")
        subcat = script_data.get("sub_category", "")
        companies   = script_data.get("companies_mentioned", [])
        keywords    = script_data.get("keywords", [])
        hashtags    = script_data.get("hashtags", [])
        breaking_level = script_data.get("breaking_news_level", 0)
        voice_used  = script_data.get("edge_tts_voice")

        video_path = create_video(audio_path, script_data, chunks)
        if not video_path or not os.path.exists(video_path):
            raise Exception("Video file not created.")
    except Exception as e:
        log_message(f"ERROR: Video render failed: {e}")
        return False

    # ── STEP 8: Generate Thumbnail ────────────────────────────────────────────
    log_message("STEP 8: Generating thumbnail...")
    thumbnail_path = generate_thumbnail(script_data)

    # ── STEP 9: Telegram Upload Consent ──────────────────────────────────────
    log_message("STEP 9: Requesting upload consent via Telegram...")
    approved = send_upload_consent(thumbnail_path, title, duration)
    if not approved:
        log_message("Upload skipped by user or timed out.")
        notify_telegram(f"Video saved locally.\nTitle: {title}", "⚠️")
        return True

    # ── STEP 10: YouTube Upload ───────────────────────────────────────────────
    log_message("STEP 10: Uploading to YouTube...")
    description = format_description(script, script_data.get("end_question", ""), hashtags)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Run pipeline immediately.")
    args = parser.parse_args()

    if args.now:
        run_pipeline()
    else:
        print("Usage: python main.py --now")
        print("For scheduled runs: python scheduler.py")
