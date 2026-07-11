"""
main_longform.py — Chaptered Deep-Dive Long-Form AI Video Pipeline.

CHAPTERED FORMAT (2026-07):
  Produces a 5-25 minute, 16:9 landscape deep-dive video. Depth is driven by
  the story, not a fixed target. Topic depth rotates weekly:
    Mon/Wed/Fri: 2-3 thematically linked stories
    Tue/Thu/Sat/Sun: 1 single deep story

  Replaces the old 8-topic "Did You Know" compilation format entirely.

Usage:
  python main_longform.py --now         # Run immediately
  python main_longform.py --dry-run     # Test without upload
"""

import os
os.environ["PYTHONHASHSEED"] = "0"
import argparse
import subprocess
import time
import sys
import socket
socket.setdefaulttimeout(20)
import glob
import hashlib
import random
import traceback
from datetime import datetime

from config import LOGS_DIR, OUTPUT_DIR, GEMINI_API_KEY, MUSIC_DIR
from config_longform import (
    LONGFORM_TARGET_AUDIO_DURATION, LONGFORM_MAX_CHAPTERS,
    LONGFORM_VISUAL_BEATS_PER_CHAPTER, LONGFORM_BGM_VOLUME,
    LONGFORM_MAX_RETRY_ATTEMPTS, LONGFORM_TRACKER_FILE,
    LONGFORM_WORD_COUNT_TARGET, get_topic_depth_mode
)
from fetch_research_papers import fetch_tech_news, fetch_ai_tools, fetch_trending_from_newsapi, fetch_reddit_news
from kaggle_handover import trigger_kaggle_gpu_job
from topic_tracker import record_story, update_youtube_url
from gemini_script_longform import generate_longform_script
from audio_gen import generate_voiceover, clean_tts_text
from chunk_builder import build_chunks, build_chapter_aware_chunks, redistribute_to_audio_duration
from pexels_fetcher import generate_visual_style_guide
from nano_scene_gen import generate_nano_scene_visuals
from video_gen import create_video
from screenshot_gen import capture_article_screenshot
from thumbnail_gen import generate_thumbnail
from youtube_upload import upload_video
from telegram_selector import notify_telegram
from entity_fetcher import fetch_all_entities, get_retention_layers_config
from tags_helper import get_optimized_metadata


def log_message(msg):
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOGS_DIR, f"log_longform_{today}.txt")
    with open(log_path, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    print(msg)


def format_longform_description(script_data, hashtags):
    """Generate a rich YouTube description for the chaptered deep-dive."""
    chapters = script_data.get("chapters", [])
    fact_timestamps = script_data.get("fact_timestamps", [])
    title = script_data.get("title", "AI Deep Dive")
    topics = script_data.get("longform_topics", [])
    description_ai = script_data.get("description", "")
    depth_mode = script_data.get("depth_mode", "single")

    hashtag_str = " ".join(hashtags) if hashtags else ""

    # Build chapter timestamps
    timestamps_str = "📌 CHAPTERS:\n"
    if chapters:
        for ch in chapters:
            start_s = ch.get("approx_start_seconds", 0)
            m, s = divmod(int(start_s), 60)
            ch_title = ch.get("chapter_title", f"Chapter {ch.get('chapter_number', '?')}")[:60]
            timestamps_str += f"{m}:{s:02d} — {ch_title}\n"
    elif fact_timestamps:
        for ft in fact_timestamps:
            start_s = ft.get("approx_start_seconds", 0)
            m, s = divmod(int(start_s), 60)
            topic = ft.get("topic", "Section")[:60]
            timestamps_str += f"{m}:{s:02d} — {topic}\n"

    # Build sources section
    sources_str = "📚 SOURCES:\n"
    for t in topics:
        url = t.get("source_url", "")
        name = t.get("source_name", "")
        headline = t.get("headline", "")[:60]
        if url:
            sources_str += f"🔗 {headline} ({name}): {url}\n"

    # Rotate between description templates
    desc_seed = int(hashlib.md5(title.encode()).hexdigest(), 16)
    template_idx = desc_seed % 3

    templates = [
        f"""📲 Get daily AI news drops before they trend:
🚀 Telegram → https://t.me/technewsbyvj
💬 WhatsApp → https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
━━━━━━━━━━━━━━━━━━━━━━
🧠 {description_ai}

{timestamps_str}
━━━━━━━━━━━━━━━━━━━━━━
{sources_str}
━━━━━━━━━━━━━━━━━━━━━━
💼 VJ LinkedIn: https://www.linkedin.com/in/vijayakumar-j/

⚠️ DISCLOSURE: This video uses AI-assisted production tools (TTS voiceover, AI-generated visuals). All editorial opinions, topic selection, and analysis are by VJ.

{hashtag_str}""",

        f"""📲 Join the community for daily AI intelligence:
🚀 Telegram: https://t.me/technewsbyvj
💬 WhatsApp: https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
━━━━━━━━━━━━━━━━━━━━━━
⚡ A deep dive into one of the biggest stories in AI this week.

{timestamps_str}
{sources_str}
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
💼 LinkedIn: https://www.linkedin.com/in/vijayakumar-j/

⚠️ DISCLOSURE: AI tools are used in production (voiceover, visuals). Topic selection, research, and commentary by VJ.

{hashtag_str}""",

        f"""📲 Direct Links to Communities:
🚀 Telegram → https://t.me/technewsbyvj
💬 WhatsApp → https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
━━━━━━━━━━━━━━━━━━━━━━
👆 What's your take? Drop your thoughts in the comments!

{timestamps_str}
━━━━━━━━━━━━━━━━━━━━━━
{sources_str}
━━━━━━━━━━━━━━━━━━━━━━
💼 LinkedIn: https://www.linkedin.com/in/vijayakumar-j/

⚠️ DISCLOSURE: AI-assisted production (voiceover, visuals). Editorial direction & analysis by VJ.

{hashtag_str}""",
    ]

    return templates[template_idx]


def run_longform_pipeline(dry_run=False):
    """Main chaptered deep-dive pipeline: research → script → render → upload."""
    depth_mode = get_topic_depth_mode()
    log_message(f"=== STARTING CHAPTERED DEEP-DIVE PIPELINE (mode={depth_mode}) ===")

    # Initialize Kaggle configuration flags
    has_kaggle = os.getenv("KAGGLE_USERNAME") is not None or os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
    use_local_only = os.environ.get("USE_LOCAL_ONLY") == "true"

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
    log_message("STEP 1: Fetching RSS articles...")
    rss_articles = []
    try:
        # Fetch vidIQ trending topics
        vidiq_news = []
        try:
            from vidiq_trending import get_pipeline_topics
            vidiq_raw = get_pipeline_topics(category="AI Did You Know")
            for item in vidiq_raw:
                vidiq_news.append({
                    "title": item["original_title"] if item.get("original_title") else item["title"],
                    "description": f"vidIQ Score: {item.get('score', 60)} (Volume: {item.get('search_volume')}, Competition: {item.get('competition')})",
                    "source": {"name": item.get("source", "vidIQ")},
                    "url": item.get("url", ""),
                    "publishedAt": datetime.now().isoformat(),
                    "type": "trending",
                    "_engagement_score": item.get("score", 60)
                })
            log_message(f"📈 vidIQ: {len(vidiq_news)} topics.")
        except Exception as ex:
            log_message(f"⚠️ vidIQ failed (non-fatal): {ex}")

        research_news = fetch_tech_news(hours=48)
        ai_tool_news = fetch_ai_tools(hours=48)
        trending_news = fetch_trending_from_newsapi()

        try:
            reddit_news = fetch_reddit_news(hours=48)
        except Exception:
            reddit_news = []

        try:
            from fetch_research_papers import fetch_x_trending_ai_topics
            x_news = fetch_x_trending_ai_topics()
        except Exception:
            x_news = []

        try:
            from trending_engine import fetch_github_trending_ai
            github_news = fetch_github_trending_ai()
        except Exception:
            github_news = []

        rss_articles = github_news + vidiq_news + research_news + ai_tool_news + trending_news + x_news + reddit_news

        if not rss_articles:
            log_message("⚠️ All sources returned 0 articles. Pipeline will rely on Gemini Search.")
        else:
            log_message(f"✅ Fetched {len(rss_articles)} total articles.")
    except Exception as e:
        log_message(f"⚠️ RSS Fetch failed: {e}")

    # ── STEP 2: Generate Chaptered Script ────────────────────────────────
    attempts = 0
    script_data = None
    best_script_data = None
    best_word_count = 0
    audio_path = None
    word_timestamps = []
    duration = 0
    min_dur, max_dur = LONGFORM_TARGET_AUDIO_DURATION
    failed_topics = []

    while attempts < LONGFORM_MAX_RETRY_ATTEMPTS:
        log_message(f"STEP 2 (Attempt {attempts + 1}/{LONGFORM_MAX_RETRY_ATTEMPTS}): "
                   f"Generating chaptered deep-dive script...")

        script_data = generate_longform_script(
            articles=rss_articles,
            failed_topics=failed_topics
        )

        if not script_data:
            log_message("ERROR: Script generation failed.")
            attempts += 1
            if attempts % 3 == 0:
                log_message("⏳ Potential rate limit. Sleeping 60s...")
                time.sleep(60)
            continue

        # Track best candidate
        script = script_data.get("script", "")
        word_count = len(script.split())
        if word_count > best_word_count:
            best_word_count = word_count
            best_script_data = dict(script_data)

        # Mark as longform
        script_data["slot"] = "Slot L (Long-form)"
        script_data["is_longform"] = True

        title = script_data.get("title", "AI Deep Dive")
        chapters = script_data.get("chapters", [])
        log_message(f"Generated Title: {title}")
        log_message(f"Script: {word_count} words, {len(chapters)} chapters")
        for i, ch in enumerate(chapters):
            log_message(f"  Ch{i+1}: {ch.get('chapter_title', 'Untitled')} "
                       f"({len(ch.get('visual_beats', []))} beats)")

        # ── STEP 3: Generate Audio ───────────────────────────────────────
        log_message("STEP 3: Generating voiceover...")
        custom_map = script_data.get("phonetic_pronunciation_map", {})

        # Select intro video for lip-sync
        intro_videos = glob.glob("assets/video/*.mp4")
        if not intro_videos:
            intro_videos = ["assets/video/Firefly_video_final.mp4"]
        from topic_tracker import get_next_avatar
        selected_avatar = get_next_avatar(intro_videos, tracker_file=LONGFORM_TRACKER_FILE)
        script_data["lipsync_face_path"] = selected_avatar

        # ── Word Count Pre-Flight ────────────────────────────────────────
        expected_dur = word_count / 2.33  # ~140 WPM

        if expected_dur < min_dur:
            log_message(f"⚠️ Script too short ({word_count} words, ~{expected_dur:.0f}s < {min_dur}s). Retrying...")
            attempts += 1
            script_data = None
            continue
        elif expected_dur > max_dur + 60:
            log_message(f"⚠️ Script too long ({word_count} words, ~{expected_dur:.0f}s > {max_dur + 60}s). Retrying...")
            attempts += 1
            script_data = None
            continue

        # ── STEP 2b: Capture Screenshots ─────────────────────────────────
        log_message("STEP 2b: Capturing article screenshots...")
        topics = script_data.get("longform_topics", [])
        screenshots_captured = 0
        for i, topic in enumerate(topics):
            url = topic.get("source_url", "")
            if url:
                ss_filename = f"screenshot_longform_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                ss_path = capture_article_screenshot(
                    url, ss_filename, desktop=True,
                    headline=topic.get("headline")
                )
                if ss_path:
                    topic["screenshot_path"] = ss_path
                    screenshots_captured += 1
                    if i == 0:
                        script_data["screenshot_path"] = ss_path
        log_message(f"✅ Captured {screenshots_captured}/{len(topics)} screenshots.")

        if has_kaggle and not use_local_only:
            log_message("Attempting Kaggle GPU handover...")
            results = trigger_kaggle_gpu_job(script_data, custom_map)

            kaggle_failed = False
            if results is None:
                kaggle_failed = True
            elif isinstance(results, dict) and "error" in results:
                kaggle_failed = True
                log_message(f"❌ Kaggle failed: [{results['error']}] {results.get('message', '')}")

            if not kaggle_failed:
                audio_path = results.get("audio_path")
                duration = results.get("duration")
                word_timestamps = results.get("word_timestamps")
                ls_path = results.get("lipsync_path")
                script_data["kaggle_lipsync_path"] = ls_path

                ls_received = ls_path and os.path.exists(ls_path)
                audio_received = audio_path and os.path.exists(audio_path)

                if audio_received and ls_received:
                    log_message("✅ Audio + Lip-Sync from Kaggle GPU!")
                elif audio_received:
                    log_message("✅ Audio from Kaggle (lip-sync missing)")
                else:
                    log_message("❌ Kaggle output missing. Retrying...")
                    attempts += 1
                    continue
            else:
                log_message("🔄 Kaggle unavailable. Using cloud TTS...")
                try:
                    notify_telegram(
                        f"🔄 Kaggle fallback (Long-form)\n"
                        f"Title: {script_data.get('title', 'Unknown')}",
                        "⚠️"
                    )
                except Exception:
                    pass
                audio_path, duration, word_timestamps = generate_voiceover(
                    script, custom_phonetic_map=custom_map, api_key=GEMINI_API_KEY
                )
                script_data["kaggle_lipsync_path"] = None
                script_data["skip_avatar"] = True
        else:
            audio_path, duration, word_timestamps = generate_voiceover(
                script, custom_phonetic_map=custom_map, api_key=GEMINI_API_KEY
            )

        if not audio_path:
            log_message("ERROR: Audio generation failed.")
            attempts += 1
            continue

        if duration < min_dur:
            log_message(f"Audio too short ({duration:.1f}s < {min_dur}s). Retrying...")
            attempts += 1
            continue

        if duration > max_dur + 60:
            log_message(f"Audio too long ({duration:.1f}s > {max_dur + 60}s). Retrying...")
            attempts += 1
            continue

        log_message(f"Audio OK: {duration:.1f}s | {len(word_timestamps)} word timestamps")
        break

    # ── FALLBACK ──────────────────────────────────────────────────────────
    if not audio_path and best_script_data:
        expected_best_dur = best_word_count / 2.33
        if expected_best_dur < 30:
            log_message(f"❌ Best candidate too short ({best_word_count} words). Aborting.")
        else:
            log_message(f"⚠️ [FALLBACK] Using best candidate ({best_word_count} words)...")
            script_data = best_script_data
            script_data["slot"] = "Slot L (Long-form)"
            script_data["is_longform"] = True

            script = script_data.get("script", "")
            custom_map = script_data.get("phonetic_pronunciation_map", {})

            intro_videos = glob.glob("assets/video/*.mp4")
            if not intro_videos:
                intro_videos = ["assets/video/Firefly_video_final.mp4"]
            from topic_tracker import get_next_avatar
            selected_avatar = get_next_avatar(intro_videos, tracker_file=LONGFORM_TRACKER_FILE)
            script_data["lipsync_face_path"] = selected_avatar

            if has_kaggle and not use_local_only:
                results = trigger_kaggle_gpu_job(script_data, custom_map)
                kaggle_failed = results is None or (isinstance(results, dict) and "error" in results)
                if not kaggle_failed:
                    audio_path = results.get("audio_path")
                    duration = results.get("duration")
                    word_timestamps = results.get("word_timestamps")
                    script_data["kaggle_lipsync_path"] = results.get("lipsync_path")
                    if not (audio_path and os.path.exists(audio_path)):
                        kaggle_failed = True
                if kaggle_failed:
                    audio_path, duration, word_timestamps = generate_voiceover(
                        script, custom_phonetic_map=custom_map, api_key=GEMINI_API_KEY
                    )
                    script_data["kaggle_lipsync_path"] = None
                    script_data["skip_avatar"] = True
            else:
                audio_path, duration, word_timestamps = generate_voiceover(
                    script, custom_phonetic_map=custom_map, api_key=GEMINI_API_KEY
                )

    if not audio_path or not script_data:
        log_message("ERROR: Could not generate valid assets. Aborting.")
        return False

    # ── STEP 4: Reserve Topics in Tracker ────────────────────────────────
    log_message("STEP 4: Reserving topics in tracker...")
    title = script_data.get("title", "AI Deep Dive")
    keywords = script_data.get("keywords", [])
    hashtags = script_data.get("hashtags", [])

    companies_all = []
    for t in script_data.get("longform_topics", []):
        companies_all.append(t.get("source_name", ""))

    record_story(
        title, script_data.get("original_news_headline"),
        "AI Deep Dive", companies_all, keywords, 7,
        "deep_dive", "pending_upload", script_data.get("original_news_url"),
        avatar_used=script_data.get("lipsync_face_path"),
        tracker_file=LONGFORM_TRACKER_FILE
    )

    # ── STEP 5: Build Visual Chunks (CHAPTER-AWARE) ──────────────────────
    log_message("STEP 5: Building visual chunks...")
    sub_chunks = script_data.get("subtitle_chunks", [])
    for sc in sub_chunks:
        if "text" in sc:
            sc["text"] = clean_tts_text(sc["text"], phonetic=False)

    chapters = script_data.get("chapters", [])
    if chapters:
        # Chapter-aware chunking: ~15-30 chunks instead of ~200+
        chunks = build_chapter_aware_chunks(
            word_timestamps, chapters,
            subtitle_chunks=sub_chunks,
            max_beats_per_chapter=LONGFORM_VISUAL_BEATS_PER_CHAPTER
        )
    else:
        # Fallback to standard chunking if no chapters
        chunks = build_chunks(word_timestamps, sub_chunks)

    chunks = redistribute_to_audio_duration(chunks, duration)
    log_message(f"Built {len(chunks)} visual chunks from {len(word_timestamps)} words.")

    # ── STEP 6: Fetch Entities ───────────────────────────────────────────
    log_message("STEP 6: Fetching entity photos and logos...")
    script_data = fetch_all_entities(script_data)

    retention_config = get_retention_layers_config()
    script_data["retention_config"] = retention_config

    # ── STEP 7: Visual Generation (16:9) ─────────────────────────────────
    log_message("STEP 7: Generating visuals (16:9)...")
    topic_context = f"Deep Dive: {title}"
    style_guide = generate_visual_style_guide(topic_context)

    from pexels_fetcher import fetch_all_chunk_visuals
    chunks = fetch_all_chunk_visuals(chunks, topic_context=topic_context, script_data=script_data, is_longform=True)

    def is_valid_engagement_source(source):
        if not source:
            return False
        valid_keywords = ["Veo", "Imagen", "Screenshot", "HuggingFace", "Cloudflare", "Pollinations", "Nano-Scene", "Pexels"]
        return any(k in source for k in valid_keywords)

    gen_success = sum(1 for c in chunks if c.get("visual_path") and is_valid_engagement_source(c.get("source")))

    if gen_success < len(chunks) * 0.5:
        log_message(f"⚠️ Primary visuals: {gen_success}/{len(chunks)}. Falling back to nano-scene...")
        chunks = generate_nano_scene_visuals(chunks, topic_context, style_guide=style_guide, aspect_ratio="16:9")

        final_success = sum(1 for c in chunks if c.get("visual_path") and os.path.exists(c["visual_path"]) and is_valid_engagement_source(c.get("source")))

        if final_success < len(chunks) * 0.5:
            log_message(f"⚠️ Both generators failed ({final_success}/{len(chunks)}). Using whiteboard fallback.")
            from whiteboard_gen import generate_whiteboard_visuals
            chunks = generate_whiteboard_visuals(chunks, topic_context, is_longform=True)
    else:
        log_message(f"✅ Visuals: {gen_success}/{len(chunks)} generated successfully.")

    # ── STEP 8: Color Theme ──────────────────────────────────────────────
    title = script_data.get("title", "")
    subcat = script_data.get("sub_category", "")
    keywords = script_data.get("keywords", [])

    def get_color_theme_for_topic(category, title, keywords):
        cat_lower = str(category).lower()
        title_lower = str(title).lower()
        kw_lower = [str(k).lower() for k in keywords]

        red_indicators = ["security", "privacy", "hack", "scam", "leak", "danger", "threat", "cyber", "exploit", "warning", "ban"]
        if any(x in cat_lower or x in title_lower for x in red_indicators) or any(x in k for k in kw_lower for x in red_indicators):
            return {"background": "#1A0000", "accent": "#FF4444", "text": "#ffffff"}

        green_indicators = ["code", "coding", "developer", "github", "repo", "automation", "workflow", "python", "javascript", "programming"]
        if any(x in cat_lower or x in title_lower for x in green_indicators) or any(x in k for k in kw_lower for x in green_indicators):
            return {"background": "#0F1A12", "accent": "#00FF7F", "text": "#ffffff"}

        gold_indicators = ["money", "earn", "business", "startup", "finance", "career", "job", "salary", "million", "billion", "market"]
        if any(x in cat_lower or x in title_lower for x in gold_indicators) or any(x in k for k in kw_lower for x in gold_indicators):
            return {"background": "#0D0D1A", "accent": "#FFD700", "text": "#ffffff"}

        purple_indicators = ["model", "llm", "gemini", "openai", "gpt", "claude", "meta", "nvidia", "deepseek", "anthropic", "apple", "microsoft"]
        if any(x in cat_lower or x in title_lower for x in purple_indicators) or any(x in k for k in kw_lower for x in purple_indicators):
            return {"background": "#10002B", "accent": "#E0AAFF", "text": "#ffffff"}

        return {"background": "#121212", "accent": "#00E5FF", "text": "#ffffff"}

    chosen_style = get_color_theme_for_topic(subcat, title, keywords)
    script_data["color_theme"] = chosen_style

    # ── STEP 9: Render Video (16:9) ──────────────────────────────────────
    log_message("STEP 9: Rendering final 16:9 video...")
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
        log_message(f"   Chunks: {len(chunks)} (chapters: {len(chapters)})")
        log_message(f"   Duration: {duration:.1f}s | Words: {len(script.split())}")
        return True

    # ── STEP 11: Upload to YouTube ───────────────────────────────────────
    log_message("STEP 11: Uploading to YouTube...")

    # Title selection
    if script_data.get("title_options"):
        title = random.choice(script_data["title_options"])

    initial_people = [p.get("name") for p in script_data.get("people", [])] if script_data.get("people") else []
    optimized_metadata = get_optimized_metadata(
        title=title,
        script=script,
        sub_category=script_data.get("sub_category", ""),
        initial_keywords=keywords,
        initial_companies=companies_all,
        initial_people=initial_people,
        initial_hashtags=hashtags,
        is_shorts=False
    )
    hashtags = optimized_metadata["hashtags"]
    tags = optimized_metadata["tags"]

    log_message(f"Optimized Tags: {tags}")
    log_message(f"Optimized Hashtags: {hashtags}")

    description = format_longform_description(script_data, hashtags)

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

    # ── STEP 12.5: Shorts Cross-Promotion Teaser ────────────────────────
    from config_longform import LONGFORM_GENERATE_SHORTS_TEASER
    if LONGFORM_GENERATE_SHORTS_TEASER:
        log_message("STEP 12.5: Generating Shorts Teaser...")
        try:
            from shorts_teaser import generate_and_upload_shorts_teaser
            generate_and_upload_shorts_teaser(script_data, result, dry_run=dry_run)
        except Exception as e:
            log_message(f"⚠️ Shorts teaser failed: {e}")
            traceback.print_exc()

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

    log_message("=== CHAPTERED DEEP-DIVE PIPELINE COMPLETED SUCCESSFULLY ===")

    notify_telegram(
        f"✅ Deep-Dive Video Live!\n\n{title}\n{youtube_url}\n\n"
        f"📊 Duration: {duration:.0f}s | Chapters: {len(chapters)} | Mode: {depth_mode}\n"
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
    parser = argparse.ArgumentParser(description="Chaptered Deep-Dive Long-Form AI Video Pipeline")
    parser.add_argument("--now", action="store_true", help="Run pipeline immediately.")
    parser.add_argument("--dry-run", action="store_true", help="Run without uploading to YouTube.")
    args = parser.parse_args()

    if args.now or args.dry_run:
        success = run_longform_pipeline(dry_run=args.dry_run)
        if not success:
            print("❌ Pipeline failed.")
            sys.exit(1)
    else:
        print("Usage:")
        print("  python main_longform.py --now        # Run immediately")
        print("  python main_longform.py --dry-run    # Test without upload")
