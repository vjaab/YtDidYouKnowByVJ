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
import subprocess
import time
import sys
import socket
socket.setdefaulttimeout(20) # Prevent network socket calls (e.g. feedparser) from hanging indefinitely
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
from kaggle_handover import trigger_kaggle_gpu_job
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
from tags_helper import get_optimized_metadata



def log_message(msg):
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOGS_DIR, f"log_longform_{today}.txt")
    with open(log_path, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    print(msg)


def format_longform_description(script_data, hashtags):
    """Generate a rich YouTube description for the long-form compilation."""
    fact_timestamps = script_data.get("fact_timestamps", [])
    num_facts = script_data.get("num_facts") or len(fact_timestamps) or 10
    title = script_data.get("title", f"{num_facts} AI Facts")
    topics = script_data.get("longform_topics", [])
    description_ai = script_data.get("description", "")
    
    # Target high-RPM regions (USA, UK, Canada, Australia, NZ, Singapore, South Korea, Japan, Europe)
    target_hashtags = ["#TechUSA", "#TechUK", "#TechCanada", "#TechAustralia", "#TechNZ", "#TechSingapore", "#TechSouthKorea", "#TechJapan", "#TechEurope", "#English"]
    all_hashtags = list(hashtags) if hashtags else []
    for tag in target_hashtags:
        if tag not in all_hashtags:
            all_hashtags.append(tag)
    hashtag_str = " ".join(all_hashtags)
    
    # Build timestamps section
    timestamps_str = "📌 TIMESTAMPS:\n"
    timestamps_str += "0:00 — Introduction / Hook\n"
    last_start = 0
    for ft in fact_timestamps:
        approx_s = ft.get("approx_start_seconds", 0)
        m, s = divmod(int(approx_s), 60)
        fact_num = ft.get("fact_number", "?")
        # Skip non-numeric/recap markers
        if not isinstance(fact_num, int):
            continue
        topic = ft.get("topic", f"Fact {fact_num}")[:50]
        timestamps_str += f"{m}:{s:02d} — Fact {fact_num}: {topic}\n"
        last_start = max(last_start, approx_s)
        
    m_out, s_out = divmod(int(last_start + 35), 60)
    timestamps_str += f"{m_out}:{s_out:02d} — Outro & Discussion\n"
    
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

{hashtag_str}
#DidYouKnow #AIFacts #TechFacts #ArtificialIntelligence #MachineLearning""",

        f"""📲 Join the community for daily AI intelligence:
🚀 Telegram: https://t.me/technewsbyvj
💬 WhatsApp: https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
━━━━━━━━━━━━━━━━━━━━━━
⚡ {num_facts} AI facts that will blow your mind — all from the last 48 hours.

{timestamps_str}
{sources_str}
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
💼 LinkedIn: https://www.linkedin.com/in/vijayakumar-j/

⚠️ DISCLOSURE: AI tools are used in production (voiceover, visuals). Topic selection, research, and commentary by VJ.

{hashtag_str}
#AI #DidYouKnow #TechNews #DeepLearning #Innovation""",

        f"""📲 Direct Links to Communities:
🚀 Telegram → https://t.me/technewsbyvj
💬 WhatsApp → https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
━━━━━━━━━━━━━━━━━━━━━━
👆 Which fact shocked you the most? Comment the number!

{timestamps_str}
━━━━━━━━━━━━━━━━━━━━━━
{sources_str}
━━━━━━━━━━━━━━━━━━━━━━
💼 LinkedIn: https://www.linkedin.com/in/vijayakumar-j/

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
        # Fetch vidIQ trending topics
        vidiq_news = []
        try:
            from vidiq_trending import get_pipeline_topics
            vidiq_raw = get_pipeline_topics(category="AI Did You Know")
            for item in vidiq_raw:
                vidiq_news.append({
                    "title": item["original_title"] if item.get("original_title") else item["title"],
                    "description": f"vidIQ Opportunity Score: {item.get('score', 60)} (Volume: {item.get('search_volume')}, Competition: {item.get('competition')})",
                    "source": {"name": item.get("source", "vidIQ")},
                    "url": item.get("url", ""),
                    "publishedAt": datetime.now().isoformat(),
                    "type": "trending",
                    "_engagement_score": item.get("score", 60)
                })
            log_message(f"📈 vidIQ: Injected {len(vidiq_news)} high-signal topics.")
        except Exception as ex:
            log_message(f"⚠️ vidIQ Fetch failed (non-fatal): {ex}")

        research_news = fetch_tech_news()
        ai_tool_news = fetch_ai_tools()
        trending_news = fetch_trending_from_newsapi()
        
        # Fetch X.com trending AI topics
        try:
            from fetch_research_papers import fetch_x_trending_ai_topics
            x_news = fetch_x_trending_ai_topics()
        except Exception as ex:
            log_message(f"⚠️ Failed to import x_trending_fetcher: {ex}")
            x_news = []
            
        rss_articles = vidiq_news + research_news + ai_tool_news + trending_news + x_news

        if not rss_articles:
            log_message("⚠️ All sources returned 0 articles. Pipeline will rely on Gemini Search.")
        else:
            log_message(f"✅ Fetched {len(rss_articles)} total articles "
                       f"({len(vidiq_news)} vidIQ, {len(research_news)} research, {len(ai_tool_news)} tools, {len(trending_news)} trending, {len(x_news)} X.com).")
    except Exception as e:
        log_message(f"⚠️ RSS/X Fetch failed: {e}")

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
                ss_path = capture_article_screenshot(
                    url, 
                    ss_filename, 
                    desktop=True, 
                    headline=topic.get("headline")
                )
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

        # ── STEP 3: Generate Audio + Lip-Sync ────────────────────────────
        log_message("STEP 3: Generating voiceover + word timestamps...")
        custom_map = script_data.get("phonetic_pronunciation_map", {})

        # Select intro video for lip-sync (rotation)
        intro_videos = glob.glob("assets/video/*.mp4")
        if not intro_videos:
            intro_videos = ["assets/video/Firefly_video_final.mp4"]
        
        from topic_tracker import get_next_avatar
        selected_avatar = get_next_avatar(intro_videos, tracker_file=LONGFORM_TRACKER_FILE)
        script_data["lipsync_face_path"] = selected_avatar
        log_message(f"Selected Lip-Sync Template: {selected_avatar}")

        # ── Kaggle GPU Handover (Audio + Lip-Sync) ───────────────────────
        has_kaggle = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
        use_local_only = os.environ.get("USE_LOCAL_ONLY") == "true"

        if has_kaggle and not use_local_only:
            log_message("Attempting Kaggle GPU handover for audio + lip-sync...")
            results = trigger_kaggle_gpu_job(script_data, custom_map)
            
            # Check if Kaggle returned a structured error (dict with "error" key)
            kaggle_failed = False
            if results is None:
                kaggle_failed = True
                log_message("❌ Kaggle Handover returned None (unexpected failure).")
            elif isinstance(results, dict) and "error" in results:
                kaggle_failed = True
                error_type = results["error"]
                error_msg = results.get("message", "Unknown")
                log_message(f"❌ Kaggle Handover failed: [{error_type}] {error_msg}")
            
            if not kaggle_failed:
                # Kaggle succeeded — use its results
                audio_path = results.get("audio_path")
                duration = results.get("duration")
                word_timestamps = results.get("word_timestamps")
                ls_path = results.get("lipsync_path")
                script_data["kaggle_lipsync_path"] = ls_path

                # Verify files exist on disk
                ls_received = ls_path and os.path.exists(ls_path)
                audio_received = audio_path and os.path.exists(audio_path)

                if audio_received and ls_received:
                    log_message("✅ Received Audio and Lip-Sync from Kaggle GPU!")
                elif audio_received:
                    log_message("✅ Received Audio from Kaggle GPU! (Lip-Sync was missing/failed)")
                else:
                    log_message("❌ Kaggle job finished but critical audio output is missing.")
                    attempts += 1
                    continue
            else:
                # ── FALLBACK: Generate audio via cloud TTS (no GPU needed) ──
                log_message("🔄 Kaggle GPU unavailable. Falling back to cloud TTS (ElevenLabs/Edge)...")
                log_message("⚠️ Lip-sync and avatar will be SKIPPED for this video (requires GPU).")
                
                try:
                    notify_telegram(
                        f"🔄 Kaggle GPU fallback activated (Long-form)\n\n"
                        f"Using cloud TTS instead. Avatar/lip-sync skipped.\n"
                        f"Title: {script_data.get('title', 'Unknown')}",
                        "⚠️"
                    )
                except Exception:
                    pass
                
                audio_path, duration, word_timestamps = generate_voiceover(
                    script, custom_phonetic_map=custom_map, api_key=GEMINI_API_KEY
                )
                script_data["kaggle_lipsync_path"] = None  # No lip-sync available
                script_data["skip_avatar"] = True           # Skip avatar PiP in video render
        else:
            # Local fallback: generate audio without Kaggle GPU
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
        avatar_used=script_data.get("lipsync_face_path"),
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

    # ── STEP 7: Google Veo / Google Image Per-Sentence Visual Generation (16:9) ──────
    log_message("STEP 7: Generating visuals using Google Veo and Google Image (16:9, primary option)...")
    topic_context = f"Did You Know AI Facts Compilation: {title}"
    style_guide = generate_visual_style_guide(topic_context)

    from pexels_fetcher import fetch_all_chunk_visuals
    chunks = fetch_all_chunk_visuals(chunks, topic_context=topic_context, script_data=script_data, is_longform=True)

    # Check success rate of the primary generator
    def is_valid_engagement_source(source):
        if not source:
            return False
        valid_keywords = ["Veo", "Imagen", "Screenshot", "HuggingFace", "Cloudflare", "Pollinations", "Nano-Scene", "Pexels"]
        return any(k in source for k in valid_keywords)

    gen_success = sum(1 for c in chunks if c.get("visual_path") and is_valid_engagement_source(c.get("source")))
    
    if gen_success < len(chunks) * 0.5:
        log_message(f"⚠️ Primary visual generator only generated {gen_success}/{len(chunks)} visuals. Falling back to nano-scene engine (16:9 Imagen)...")
        chunks = generate_nano_scene_visuals(chunks, topic_context, style_guide=style_guide, aspect_ratio="16:9")
        
        # Check if nano-scene engine ALSO failed to produce new visuals (16:9)
        final_success = sum(1 for c in chunks if c.get("visual_path") and os.path.exists(c["visual_path"]) and is_valid_engagement_source(c.get("source")))
        
        if final_success < len(chunks) * 0.5:
            log_message(f"⚠️ Both primary and nano-scene visual generation failed (only {final_success}/{len(chunks)} succeeded). Falling back to whiteboard animation videos!")
            from whiteboard_gen import generate_whiteboard_visuals
            chunks = generate_whiteboard_visuals(chunks, topic_context, is_longform=True)
    else:
        log_message(f"✅ Primary visual generator successfully created {gen_success}/{len(chunks)} clips/images.")

    # ── STEP 8: Visual Variety (Color Theme) ─────────────────────────────
    title = script_data.get("title", "")
    subcat = script_data.get("sub_category", "")
    keywords = script_data.get("keywords", [])

    def get_color_theme_for_topic(category, title, keywords):
        cat_lower = str(category).lower()
        title_lower = str(title).lower()
        kw_lower = [str(k).lower() for k in keywords]
        
        # 1. Cyber Security / Privacy / Threat / Danger / Scam -> Deep Red
        red_indicators = ["security", "privacy", "hack", "scam", "leak", "danger", "scary", "threat", "cyber", "exploit", "warning", "ban"]
        if any(x in cat_lower or x in title_lower for x in red_indicators) or any(x in k for k in kw_lower for x in red_indicators):
            return {"background": "#1A0000", "accent": "#FF4444", "text": "#ffffff"} # Deep Red
            
        # 2. Coding / Development / GitHub / Automation / Tech Tips -> Hacker Green
        green_indicators = ["code", "coding", "developer", "github", "repo", "automation", "workflow", "tip", "hack", "productivity", "python", "javascript", "programming"]
        if any(x in cat_lower or x in title_lower for x in green_indicators) or any(x in k for k in kw_lower for x in green_indicators):
            return {"background": "#0F1A12", "accent": "#00FF7F", "text": "#ffffff"} # Hacker Green
            
        # 3. Business / Finance / Startup / Career / Hustle / Money -> Dark Gold
        gold_indicators = ["money", "earn", "hustle", "business", "startup", "finance", "career", "job", "salary", "million", "billion", "market", "stock", "cost"]
        if any(x in cat_lower or x in title_lower for x in gold_indicators) or any(x in k for k in kw_lower for x in gold_indicators):
            return {"background": "#0D0D1A", "accent": "#FFD700", "text": "#ffffff"} # Dark Gold
            
        # 4. Neural Nets / Models / LLMs / Big Tech Giants (OpenAI, Gemini, Meta) -> Neon Purple
        purple_indicators = ["model", "llm", "gemini", "openai", "gpt", "claude", "meta", "nvidia", "deepseek", "anthropic", "apple", "microsoft"]
        if any(x in cat_lower or x in title_lower for x in purple_indicators) or any(x in k for k in kw_lower for x in purple_indicators):
            return {"background": "#10002B", "accent": "#E0AAFF", "text": "#ffffff"} # Neon Purple

        # 5. Default/General AI / High-tech -> Cyber Cyan
        return {"background": "#121212", "accent": "#00E5FF", "text": "#ffffff"} # Cyber Cyan

    chosen_style = get_color_theme_for_topic(subcat, title, keywords)
    script_data["color_theme"] = chosen_style

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
    
    # Title selection
    if script_data.get("title_options"):
        title = random.choice(script_data["title_options"])

    # Generate dynamic, optimized hashtags and tags
    initial_people = [p.get("name") for p in script_data.get("people", [])] if script_data.get("people") else []
    optimized_metadata = get_optimized_metadata(
        title=title,
        script=script,
        sub_category=script_data.get("sub_category", ""),
        initial_keywords=keywords,
        initial_companies=companies_all,
        initial_people=initial_people,
        initial_hashtags=hashtags
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

    # ── STEP 12.5: Shorts Cross-Promotion Teaser ──────────────────────────
    from config_longform import LONGFORM_GENERATE_SHORTS_TEASER
    if LONGFORM_GENERATE_SHORTS_TEASER:
        log_message("STEP 12.5: Generating and uploading Shorts Teaser...")
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
