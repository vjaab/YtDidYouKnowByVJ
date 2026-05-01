import os
os.environ["PYTHONHASHSEED"] = "0"
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


def format_description(ai_description, script, hashtags, slot="Slot A", chunks=None):
    hashtag_str = " ".join(hashtags) if hashtags else ""
    
    # ── Action-Oriented Summary ──
    # Ensure the summary is crisp and lacks large blocks of text
    clean_summary = ai_description.split(". ")[0] + "." # Just the first hard-hitting line for the summary
    if len(clean_summary) > 150: clean_summary = clean_summary[:147] + "..."

    # ── Timestamp Logic (for Slot C / Long-form) ──
    timestamps_str = ""
    if "Slot C" in slot and chunks:
        timestamps_str = "\n📌 TECHNICAL BREAKDOWN:\n"
        # Pick 3-5 key points to cite
        step = max(1, len(chunks) // 4)
        for i in range(0, len(chunks), step):
            if i >= len(chunks): break
            chunk = chunks[i]
            t = chunk["start"]
            m, s = divmod(int(t), 60)
            h, m = divmod(m, 60)
            ts = f"[{h:02d}:{m:02d}:{s:02d}]"
            label = chunk["text"].split(".")[0][:40] + "..."
            timestamps_str += f"{ts} {label}\n"
        timestamps_str += "━━━━━━━━━━━━━━━━━━━━━━\n"

    return f"""🚀 ELITE AI ENGINEERING → https://wa.me/919585793939
🔥 Automated Agentic Systems & Cost-Optimized AI.
━━━━━━━━━━━━━━━━━━━━━━
💡 {clean_summary}
{timestamps_str}
━━━━━━━━━━━━━━━━━━━━━━
🛠️ IMPLEMENTATION BLUEPRINT:

🚀 SOTA AGENTIC LOOPS (Local & Cloud Hybrid)
💼 GPU INFRA & COST-OPTIMIZED SCALING
🛠️ OPEN SOURCE SOVEREIGNTY (Llama, Kokoro, Whisper)
📰 ARXIV-TO-PRODUCTION WORKFLOWS

Join the 1% building the future 👇

🚀 Telegram → https://t.me/technewsbyvj
💼 LinkedIn → https://www.linkedin.com/in/vijayakumar-j/
💬 WhatsApp Dev Channel → https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z

🔗 (Blueprints & SDKs in Bio!)
━━━━━━━━━━━━━━━━━━━━━━

{hashtag_str}
#agenticai #llmops #python #machinelearning #aiarchitecture #shorts"""


def generate_pinned_comment(script_data, next_series_slot):
    from ecosystem_logic import get_series_identity
    series = get_series_identity(next_series_slot)
    tease  = script_data.get("next_video_tease", "something big tomorrow")
    hook   = script_data.get("comment_hook", "What do you think?")

    return (
        f"🔔 {hook}\n\n"
        f"Tomorrow on {series['name']}: {tease}\n\n"
        f"👇 Drop your prediction below — let's see who gets it right."
    )

def run_pipeline(topic_type="research"):
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

    # ── STEP 1: Content Ecosystem Check ───────────────────────────────────────
    day_name, slot, category = get_slot_info()
    log_message(f"STEP 1: Content Ecosystem Check -> Day: {day_name}, Slot: {slot}, Category: {category}")
    
    # ── STEP 2: Selection Strategy (RSS Fetch) ────────────────────────────────
    log_message(f"STEP 2: Fetching RSS articles (Research + Tools)...")
    rss_articles = []
    try:
        # Fetch both to give Gemini more options
        research_news = fetch_tech_news()
        ai_tool_news = fetch_ai_tools()
        rss_articles = research_news + ai_tool_news
            
        if not rss_articles:
            log_message("⚠️ All RSS feeds returned 0 articles.")
        else:
            log_message(f"✅ Fetched {len(rss_articles)} total articles ({len(research_news)} research, {len(ai_tool_news)} tools).")
    except Exception as e:
        log_message(f"⚠️ RSS Fetch failed: {e}")

    # ── STEP 3: Script Generation (with retry) ────────────────────────────────
    attempts = 0
    script_data = None
    audio_path  = None
    word_timestamps = []
    duration    = 0
    extra_instruction = ""
    min_dur, max_dur = TARGET_AUDIO_DURATION

    while attempts < MAX_RETRY_ATTEMPTS:
        log_message(f"STEP 3 (Attempt {attempts+1}): Gemini Searching & Generating Script...")
        
        script_data = pick_and_generate_script(
            articles=rss_articles, extra_instruction=extra_instruction, forced_article=None, topic_type=topic_type
        )

        if not script_data:
            log_message("ERROR: Script generation failed.")
            attempts += 1
            continue
            
        # Store slot info for downstream rendering (e.g. aspect ratio)
        script_data["slot"] = slot

        title  = script_data.get("title", "Tech News!")
        script = script_data.get("script", "")
        log_message(f"Story: {script_data.get('original_news_headline')}")
        log_message(f"Breaking Level: {script_data.get('breaking_news_level')}")

        # ── STEP 4: Generate Audio + Word Timestamps ──────────────────────────
        log_message("STEP 4: Generating voiceover + word timestamps...")
        custom_map = script_data.get("phonetic_pronunciation_map", {})
        
        has_kaggle = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
        use_local_only = os.environ.get("USE_LOCAL_ONLY") == "true"
        
        if has_kaggle and not use_local_only:
            from kaggle_handover import trigger_kaggle_gpu_job
            results = trigger_kaggle_gpu_job(script_data, custom_map)
            if results:
                audio_path = results.get("audio_path")
                duration = results.get("duration")
                word_timestamps = results.get("word_timestamps")
                ls_path = results.get("lipsync_path")
                script_data["kaggle_lipsync_path"] = ls_path
                
                # Verify files exist on disk before reporting success
                ls_received = ls_path and os.path.exists(ls_path)
                audio_received = audio_path and os.path.exists(audio_path)
                
                if audio_received and ls_received:
                    log_message("✅ Received Audio and Lip-Sync from Kaggle GPU!")
                elif audio_received:
                    log_message("✅ Received Audio from Kaggle GPU! (Lip-Sync was missing/failed)")
                else:
                    # CRITICAL: Kaggle results missing audio — pipeline failure
                    log_message("❌ Kaggle job finished but critical audio output is missing.")
                    return False
            else:
                log_message("❌ Kaggle Handover failed or job reported an error.")
                return False
        else:
            audio_path, duration, word_timestamps = generate_voiceover(script, custom_phonetic_map=custom_map)
        

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

    # ── STEP 4b: Capture Article & Evidence Screenshots ──────────────────────
    log_message("STEP 4b: Capturing article and evidence screenshots...")
    news_url = script_data.get("original_news_url")
    evidence_url = script_data.get("use_case_evidence_url")
    
    if news_url:
        screenshot_filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        screenshot_path = capture_article_screenshot(news_url, screenshot_filename)
        if screenshot_path:
            script_data["screenshot_path"] = screenshot_path
            log_message(f"Main screenshot captured: {screenshot_path}")
            
    if evidence_url and "http" in evidence_url:
        evidence_filename = f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        evidence_path = capture_article_screenshot(evidence_url, evidence_filename)
        if evidence_path:
            script_data["evidence_screenshot_path"] = evidence_path
            log_message(f"Evidence screenshot captured: {evidence_path}")
    else:
        log_message("No valid evidence URL found for secondary screenshot.")

    # ── STEP 5: Build Visual Chunks ───────────────────────────────────────────
    log_message("STEP 5: Grouping words into visual chunks...")
    from audio_gen import clean_tts_text
    sub_chunks = script_data.get("subtitle_chunks", [])
    for sc in sub_chunks:
        if "text" in sc:
            sc["text"] = clean_tts_text(sc["text"], phonetic=False)
            
    chunks = build_chunks(word_timestamps, sub_chunks)
    chunks = redistribute_to_audio_duration(chunks, duration)
    log_message(f"Built {len(chunks)} visual chunks from {len(word_timestamps)} words.")

    # ── STEP 6: Fetch Entities (People/Companies) ─────────────────────────────
    log_message("STEP 6: Fetching entity photos and company logos...")
    from entity_fetcher import fetch_all_entities, get_retention_layers_config
    script_data = fetch_all_entities(script_data)
    
    # Enable Kinetic Layers (Production Spec 2026)
    retention_config = get_retention_layers_config()
    script_data["retention_config"] = retention_config
    log_message(f"Engagement Layers Active: {list(retention_config.keys())}")

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
    description = format_description(ai_desc, script, hashtags, slot=slot, chunks=chunks)
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

    uploaded, result = upload_video(
        video_path, title, description, tags, 
        thumbnail_path=thumbnail_path, comment_hook=script_data.get("comment_hook")
    )
    if not uploaded:
        log_message(f"ERROR: YouTube upload failed: {result}")
        return False

    youtube_url = f"https://youtu.be/{result}"
    log_message(f"SUCCESS: {youtube_url}")

    # ── STEP 10b: Generate Pinned Comment ───────────────────────────────────
    from ecosystem_logic import get_next_slot
    next_slot = get_next_slot(slot)
    pinned_comment = generate_pinned_comment(script_data, next_slot)
    log_message(f"📌 PINNED COMMENT TEMPLATE:\n\n{pinned_comment}\n")

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


def run_local(topic_type="research"):
    # XTTS server launch removed. Calling pipeline directly.
    success = run_pipeline(topic_type=topic_type)
    if not success:
        import sys
        print("❌ Pipeline failed. Exiting with error code.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Run pipeline immediately.")
    parser.add_argument("--type", type=str, choices=["research", "tools"], default="research", help="Content type mapped to the schedule")
    args = parser.parse_args()

    if args.now:
        run_local(topic_type=args.type)
    else:
        print("Usage: python main.py --now")
        print("For scheduled runs: python scheduler.py")
