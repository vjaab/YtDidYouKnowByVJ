import os
os.environ["PYTHONHASHSEED"] = "0"
import argparse
import subprocess
import time
import sys
import socket
socket.setdefaulttimeout(20) # Prevent network socket calls (e.g. feedparser) from hanging indefinitely
import requests
import glob
import hashlib
import random
import traceback
from datetime import datetime

from config import TARGET_AUDIO_DURATION, MAX_RETRY_ATTEMPTS, LOGS_DIR, OUTPUT_DIR, GEMINI_API_KEY, ENABLE_TRENDING_ENGINE
from fetch_research_papers import fetch_tech_news, fetch_ai_tools
from topic_tracker import record_story, update_youtube_url, get_next_topic_type_by_ratio, get_next_target_country, get_next_avatar
from gemini_script import pick_and_generate_script
from ecosystem_logic import get_slot_info, get_series_identity, get_next_slot
from audio_gen import generate_voiceover, clean_tts_text
from chunk_builder import build_chunks, redistribute_to_audio_duration
from pexels_fetcher import fetch_all_chunk_visuals, generate_visual_style_guide
from nano_scene_gen import generate_nano_scene_visuals
from video_gen import create_video
from screenshot_gen import capture_article_screenshot
from thumbnail_gen import generate_thumbnail
from youtube_upload import upload_video
from x_upload import upload_video_to_x
from instagram_upload import upload_reel_to_instagram
from telegram_selector import notify_telegram as real_notify_telegram
from entity_fetcher import fetch_all_entities, get_retention_layers_config
from kaggle_handover import trigger_kaggle_gpu_job
from tags_helper import get_optimized_metadata



def log_message(msg):
    today    = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOGS_DIR, f"log_{today}.txt")
    with open(log_path, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    print(msg)


def format_description(ai_description, script, hashtags, slot="Slot A", chunks=None, relevant_links=[], source_url=""):
    hashtag_str = " ".join(hashtags) if hashtags else ""
    
    # ── Action-Oriented Summary ──
    clean_summary = ai_description.split(". ")[0] + "."
    if len(clean_summary) > 150: clean_summary = clean_summary[:147] + "..."

    # ── Timestamp Logic (for Slot C / Long-form) ──
    timestamps_str = ""
    if "Slot C" in slot and chunks:
        timestamps_str = "\n📌 TECHNICAL BREAKDOWN:\n"
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

    # ── Resources & Blueprints ──
    links_str = ""
    if relevant_links:
        links_str = "\n".join([f"🔗 {link}" for link in relevant_links[:3]]) + "\n"
        
    source_str = f"📰 SOURCE ARTICLE: {source_url}\n" if source_url else ""

    # ── YPP COMPLIANCE: Rotate between 4 description templates ──
    # Deterministic selection via headline hash to avoid identical descriptions
    import hashlib
    desc_seed = int(hashlib.md5(clean_summary.encode()).hexdigest(), 16)
    template_idx = desc_seed % 4

    templates = [
        # Template 0: Original style
        f"""🎙️ Curated & analyzed by VJ — daily AI news for engineers.
🚀 JOIN FOR MORE AI NEWS → https://t.me/technewsbyvj
━━━━━━━━━━━━━━━━━━━━━━
💡 {clean_summary}
{timestamps_str}
━━━━━━━━━━━━━━━━━━━━━━
{source_str}━━━━━━━━━━━━━━━━━━━━━━
🛠️ RESOURCES & DEEP DIVES:

{links_str if links_str else "🚀 FULL AI NEWS ARCHIVE ON TELEGRAM"}
💼 Career & Industry Insights
🛠️ Open Source Tools & Reviews
📰 Daily Research Breakdowns

Join the community 👇

🚀 Telegram → https://t.me/technewsbyvj
💼 LinkedIn → https://www.linkedin.com/in/vijayakumar-j/
💬 WhatsApp Dev Channel → https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z

⚠️ DISCLOSURE: This video uses AI-assisted production tools (TTS voiceover, AI-generated visuals). All editorial opinions, topic selection, and analysis are by VJ.

{hashtag_str}""",

        # Template 1: Minimalist authority style
        f"""⚡ {clean_summary}
{timestamps_str}
{source_str}
{links_str}
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
📬 Get daily AI news before it trends:
→ Telegram: https://t.me/technewsbyvj
→ WhatsApp: https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
→ LinkedIn: https://www.linkedin.com/in/vijayakumar-j/

🔧 Full analysis & deep dives → Telegram

⚠️ DISCLOSURE: AI tools are used in production (voiceover, visuals). Topic selection, research, and commentary by VJ.

{hashtag_str}""",

        # Template 2: Newsletter/community style
        f"""What happened: {clean_summary}
{timestamps_str}
Why it matters: This changes how engineers build, deploy, and scale AI systems.

{source_str}
📚 Resources mentioned in this video:
{links_str if links_str else "→ Check the pinned comment for links"}

━━━━━━━━━━━━━━━━━━━━━━
🧠 Want the full breakdown?
Join engineers getting daily AI research drops:

📲 Telegram → https://t.me/technewsbyvj
💬 WhatsApp → https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
🔗 All links → bio

⚠️ DISCLOSURE: This video uses AI-assisted tools for production. All editorial decisions and analysis are human-driven by VJ.

{hashtag_str}""",

        # Template 3: Hook-first engagement style
        f"""👆 If this blew your mind, you need to see what's on Telegram.
━━━━━━━━━━━━━━━━━━━━━━
💡 {clean_summary}
{timestamps_str}
{source_str}
{links_str}
━━━━━━━━━━━━━━━━━━━━━━
🏗️ Follow for more AI news:
• Daily AI research & industry analysis
• Curated by VJ — no fluff, no filler
• The stories that matter for engineers

📲 https://t.me/technewsbyvj
💬 https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
💼 https://www.linkedin.com/in/vijayakumar-j/

⚠️ DISCLOSURE: AI-assisted production (voiceover, visuals). Editorial direction & analysis by VJ.

{hashtag_str}""",
    ]

    return templates[template_idx]


def generate_pinned_comment(script_data, next_series_slot):
    series = get_series_identity(next_series_slot)
    tease  = script_data.get("next_video_tease", "something big tomorrow")
    hook   = script_data.get("comment_hook", "What do you think?")
    cta    = script_data.get("identity_cta", "Join the elite builders.")

    return (
        f"🔔 {hook}\n\n"
        f"Tomorrow on {series['name']}: {tease}\n\n"
        f"👇 {cta}"
    )

def run_pipeline(topic_type="auto", dry_run=False):
    # Local mock or real Telegram notification based on dry_run
    def notify_telegram(msg, emoji=""):
        if dry_run:
            print(f"📢 [TELEGRAM MOCK] {emoji} {msg}")
        else:
            real_notify_telegram(msg, emoji)

    if topic_type == "auto":
        topic_type = get_next_topic_type_by_ratio()
    target_country = get_next_target_country()
    log_message(f"=== STARTING DAILY AI PIPELINE ({topic_type.upper()}) | COUNTRY: {target_country} ===")

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
    
    # ── STEP 2: Selection Strategy (RSS + Trending Engine) ─────────────────────
    log_message(f"STEP 2: Fetching articles (RSS + Trending Engine)...")
    rss_articles = []
    try:
        # Fetch vidIQ trending topics
        vidiq_news = []
        try:
            from vidiq_trending import get_pipeline_topics
            vidiq_raw = get_pipeline_topics(category=category)
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

        # Fetch both to give Gemini more options
        research_news = fetch_tech_news()
        ai_tool_news = fetch_ai_tools()
        
        # Fetch X.com trending AI topics
        try:
            from fetch_research_papers import fetch_x_trending_ai_topics
            x_news = fetch_x_trending_ai_topics()
        except Exception as ex:
            log_message(f"⚠️ Failed to import x_trending_fetcher: {ex}")
            x_news = []
            
        # Fetch GitHub trending AI repos (Conflict Fix: prioritize trending GitHub projects)
        try:
            from trending_engine import fetch_github_trending_ai
            github_news = fetch_github_trending_ai()
        except Exception as ex:
            log_message(f"⚠️ GitHub Fetch failed: {ex}")
            github_news = []
            
        rss_articles = github_news + vidiq_news + research_news + ai_tool_news + x_news
        
        # ── TRENDING ENGINE (Phase 1): YouTube, Reddit, GitHub signals ──
        trending_articles = []
        if ENABLE_TRENDING_ENGINE:
            try:
                from trending_engine import fetch_all_trending_signals
                trending_articles = fetch_all_trending_signals(target_country=target_country)
                rss_articles = trending_articles + rss_articles  # Trending FIRST for priority
                log_message(f"🔥 Trending Engine injected {len(trending_articles)} high-signal articles for geo={target_country}.")
            except Exception as ex:
                log_message(f"⚠️ Trending Engine failed (non-fatal): {ex}")
            
        if not rss_articles:
            log_message("⚠️ All feeds returned 0 articles.")
        else:
            log_message(f"✅ Fetched {len(rss_articles)} total articles ({len(github_news)} GitHub, {len(vidiq_news)} vidIQ, {len(research_news)} research, {len(ai_tool_news)} tools, {len(x_news)} X.com, {len(trending_articles)} trending).")
    except Exception as e:
        log_message(f"⚠️ RSS/Trending Fetch failed: {e}")

    # ── STEP 3: Script Generation (with retry) ────────────────────────────────
    # Screenshot is MANDATORY — if we can't capture it, we reject the topic and retry.
    attempts = 0
    script_data = None
    audio_path  = None
    word_timestamps = []
    duration    = 0
    extra_instruction = ""
    min_dur, max_dur = TARGET_AUDIO_DURATION
    if topic_type == "vaibhav":
        min_dur = 45
        max_dur = 65
    failed_topics = []  # Track topics whose screenshots failed so Gemini avoids them

    while attempts < MAX_RETRY_ATTEMPTS:
        log_message(f"STEP 3 (Attempt {attempts+1}/{MAX_RETRY_ATTEMPTS}): Gemini Searching & Generating Script...")
        
        # Build avoidance instruction from failed topics (duplicates or screenshot failures)
        screenshot_avoid = ""
        if failed_topics:
            avoid_lines = "\n".join([f"- {t}" for t in failed_topics])
            screenshot_avoid = (
                f"\n\nCRITICAL: The following topics/URLs were REJECTED (either because they are duplicates or their article screenshot could not be captured). "
                f"DO NOT pick these again. Choose a DIFFERENT unique story:\n{avoid_lines}\n"
            )
        
        combined_instruction = extra_instruction + screenshot_avoid
        
        script_data = pick_and_generate_script(
            articles=rss_articles, extra_instruction=combined_instruction, forced_article=None, topic_type=topic_type, failed_topics=failed_topics, target_country=target_country
        )

        if not script_data:
            log_message("ERROR: Script generation failed.")
            attempts += 1
            if attempts % 3 == 0:
                log_message("⏳ Gemini API potentially rate-limited. Sleeping 60s before next attempt...")
                time.sleep(60)
            continue
            
        # Store slot info for downstream rendering (e.g. aspect ratio)
        script_data["slot"] = slot

        title  = script_data.get("title", "Tech News!")
        script = script_data.get("script", "")
        log_message(f"Selected Headline: {script_data.get('original_news_headline')}")
        log_message(f"Selected URL: {script_data.get('original_news_url')}")
        log_message(f"Breaking Level: {script_data.get('breaking_news_level')}")

        # ── STEP 3b: Capture Article Screenshot FIRST (MANDATORY) ────────────
        # Capture screenshot BEFORE audio to fail fast and avoid wasting API costs.
        log_message("STEP 3b: Capturing article screenshot (MANDATORY — before audio)...")
        news_url = script_data.get("original_news_url")
        screenshot_captured = False
        
        if news_url:
            screenshot_filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            screenshot_path = capture_article_screenshot(
                news_url, 
                screenshot_filename, 
                headline=script_data.get("original_news_headline")
            )
            if screenshot_path:
                script_data["screenshot_path"] = screenshot_path
                log_message(f"✅ Main screenshot captured: {screenshot_path}")
                screenshot_captured = True
        
        if not screenshot_captured:
            # Screenshot is MANDATORY — reject this topic and try another
            failed_headline = script_data.get("original_news_headline", title)
            failed_url = news_url or "unknown"
            failed_topics.append(failed_headline)
            if failed_url != "unknown":
                failed_topics.append(failed_url)
            log_message(f"❌ Article screenshot FAILED for: {failed_headline}")
            log_message(f"   URL was: {failed_url}")
            log_message(f"   Rejecting this topic and picking a different one... ({len(failed_topics)} items rejected so far)")
            
            # Reset — skip audio generation entirely for this topic
            script_data = None
            attempts += 1
            continue

        # ── STEP 3c: Capture Evidence Screenshot (optional) ──────────────────
        evidence_url = script_data.get("use_case_evidence_url")
        if evidence_url and "http" in evidence_url:
            evidence_filename = f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            evidence_path = capture_article_screenshot(
                evidence_url, 
                evidence_filename, 
                headline=script_data.get("title")
            )
            if evidence_path:
                script_data["evidence_screenshot_path"] = evidence_path
                log_message(f"Evidence screenshot captured: {evidence_path}")
        else:
            log_message("No valid evidence URL found for secondary screenshot.")

        # ── STEP 3d: Fetch and Validate Entity Tags (MANDATORY for Shorts) ──
        is_longform = "Slot C" in slot
        if not is_longform:
            log_message("STEP 3d: Fetching and validating entity tags for Short...")
            script_data = fetch_all_entities(script_data)
            
            # Find entities with name, description, and successfully downloaded logo
            valid_entities = []
            for ent_list_key in ["companies", "people", "key_entities"]:
                for ent in script_data.get(ent_list_key, []):
                    name = ent.get("name")
                    desc = ent.get("description")
                    logo_path = ent.get("local_logo_path") or ent.get("local_hq_path") or ent.get("local_image_path")
                    if name and desc and logo_path and os.path.exists(logo_path):
                        if not any(e.get("name") == name for e in valid_entities):
                            valid_entities.append(ent)
            
            if not valid_entities:
                log_message("❌ Short validation FAILED: No valid entity tags (logo + name + description) found.")
                failed_headline = script_data.get("original_news_headline", title)
                failed_topics.append(failed_headline)
                script_data = None
                attempts += 1
                continue
            else:
                log_message(f"✅ Found {len(valid_entities)} valid entity tags for the Short.")

        # ── STEP 4: Generate Audio + Word Timestamps ──────────────────────────
        log_message("STEP 4: Generating voiceover + word timestamps...")
        custom_map = script_data.get("custom_map", {})
        
        # Select Intro Video for Lip-Sync (Rotation)
        intro_videos = glob.glob("assets/video/*.mp4")
        if not intro_videos:
            intro_videos = ["assets/video/Firefly_video_final.mp4"]
            
        selected_avatar = get_next_avatar(intro_videos)
        script_data["lipsync_face_path"] = selected_avatar
        log_message(f"Selected Lip-Sync Template: {selected_avatar} (from {len(intro_videos)} options)")
        
        has_kaggle = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
        use_local_only = os.environ.get("USE_LOCAL_ONLY") == "true"
        # ── Word Count Pre-Flight Gate ────────────────────────────────────────
        script = script_data.get("script", "")
        word_count = len(script.split())
        expected_dur = word_count / 2.33  # ~140 WPM (2.33 words per second)
        
        if expected_dur < min_dur:
            log_message(f"⚠️ Script word count too low ({word_count} words, expected ~{expected_dur:.1f}s < {min_dur}s). Retrying script generation early to save GPU time.")
            if (attempts % 3 == 2):
                failed_headline = script_data.get("original_news_headline", title)
                failed_topics.append(failed_headline)
                log_message(f"⚠️ Topic '{failed_headline}' repeatedly too short. Skipping.")
                extra_instruction = ""
            else:
                target_seconds = "45-60" if topic_type == "vaibhav" else "25-35"
                extra_instruction = f"The previous script was too short at {word_count} words (expected ~{expected_dur:.0f}s). Make the script longer, aim for at least {int(min_dur * 2.4)} words (approx {target_seconds} seconds)."
            attempts += 1
            script_data = None
            continue

        if has_kaggle and not use_local_only:
            results = trigger_kaggle_gpu_job(script_data, custom_map)
            
            # Check if Kaggle returned a structured error (new: dict with "error" key)
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
                # ── FALLBACK: Generate audio via cloud TTS (no GPU needed) ──
                log_message("🔄 Kaggle GPU unavailable. Falling back to cloud TTS (ElevenLabs/Edge)...")
                log_message("⚠️ Lip-sync will be SKIPPED for this video (requires GPU).")
                
                try:
                    notify_telegram(
                        f"🔄 Kaggle GPU fallback activated\n\n"
                        f"Using cloud TTS instead. Lip-sync skipped.\n"
                        f"Topic: {script_data.get('original_news_headline', 'Unknown')}",
                        "⚠️"
                    )
                except Exception:
                    pass
                
                audio_path, duration, word_timestamps = generate_voiceover(
                    script, custom_phonetic_map=custom_map, api_key=GEMINI_API_KEY
                )
                script_data["kaggle_lipsync_path"] = None  # No lip-sync available
                script_data["skip_avatar"] = True           # Skip avatar PiP in video render
                kaggle_fallback_used = True
        else:
            audio_path, duration, word_timestamps = generate_voiceover(script, custom_phonetic_map=custom_map, api_key=GEMINI_API_KEY)
        

        if not audio_path:
            log_message("ERROR: Audio generation failed.")
            failed_headline = script_data.get("original_news_headline", title)
            failed_topics.append(failed_headline)
            attempts += 1
            continue

        if duration < min_dur:
            log_message(f"Audio too short ({duration:.1f}s < {min_dur}s). Retrying...")
            # If we've tried multiple times and it's still too short, maybe just skip this topic
            if (attempts % 3 == 2):
                failed_headline = script_data.get("original_news_headline", title)
                failed_topics.append(failed_headline)
                log_message(f"⚠️ Topic '{failed_headline}' repeatedly too short. Skipping.")
                extra_instruction = ""
            else:
                target_seconds = "45-60" if topic_type == "vaibhav" else "25-35"
                extra_instruction = f"The previous script was too short at {duration:.0f}s. Make the script slightly longer, aim for {target_seconds} seconds of speaking."
            attempts += 1
            continue

        log_message(f"Audio OK: {duration:.1f}s | {len(word_timestamps)} word timestamps")
        
        break   # ← full success (script + screenshot + audio all OK)

    if not audio_path or not script_data or duration < min_dur:
        log_message("ERROR: Could not generate valid assets. Aborting.")
        return False
    
    if not script_data.get("screenshot_path"):
        log_message("ERROR: Could not capture article screenshot after all retries. Aborting.")
        return False

    # ── STEP 4.5: Reserve Topic Early ─────────────────────────────────────────
    log_message("STEP 4.5: Reserving topic in tracker to prevent reuse on failure...")
    title  = script_data.get("title", "Tech News!")
    subcat = script_data.get("sub_category", "")
    # Normalize companies_mentioned to list of strings (handle both string and dict formats)
    raw_companies = script_data.get("companies_mentioned", [])
    companies = []
    for c in raw_companies:
        if isinstance(c, dict):
            name = c.get("name")
            if name:
                companies.append(name)
        elif isinstance(c, str):
            companies.append(c)
    keywords    = script_data.get("keywords", [])
    breaking_level = script_data.get("breaking_news_level", 0)
    voice_used  = script_data.get("edge_tts_voice")
    
    record_story(
        title, script_data.get("original_news_headline"),
        subcat, companies, keywords, breaking_level,
        voice_used, "pending_upload", script_data.get("original_news_url"),
        topic_type=topic_type, target_country=target_country,
        avatar_used=script_data.get("lipsync_face_path")
    )

    # ── STEP 5: Build Visual Chunks ───────────────────────────────────────────
    log_message("STEP 5: Grouping words into visual chunks...")
    sub_chunks = script_data.get("subtitle_chunks", [])
    for sc in sub_chunks:
        if "text" in sc:
            sc["text"] = clean_tts_text(sc["text"], phonetic=False)
            
    chunks = build_chunks(word_timestamps, sub_chunks)
    chunks = redistribute_to_audio_duration(chunks, duration)
    log_message(f"Built {len(chunks)} visual chunks from {len(word_timestamps)} words.")

    # ── STEP 6: Fetch Entities (People/Companies) ─────────────────────────────
    log_message("STEP 6: Fetching entity photos and company logos...")
    script_data = fetch_all_entities(script_data)
    
    # Enable Kinetic Layers (Production Spec 2026)
    retention_config = get_retention_layers_config()
    script_data["retention_config"] = retention_config
    log_message(f"Engagement Layers Active: {list(retention_config.keys())}")

    # ── STEP 7: Google Veo / Google Image Per-Sentence Visual Generation ──────
    log_message("STEP 7: Generating visuals using Google Veo and Google Image (primary option)...")
    topic_context = script_data.get("original_news_headline", title)
    is_longform = "Slot C" in slot
    
    # Generate global visual style guide for consistency
    style_guide = generate_visual_style_guide(topic_context)
    
    # Try the primary generator first (Veo and Imagen)
    chunks = fetch_all_chunk_visuals(chunks, topic_context=topic_context, script_data=script_data, is_longform=is_longform)
    
    # Check success rate of the primary generator
    def is_valid_engagement_source(source):
        if not source:
            return False
        valid_keywords = ["Veo", "Imagen", "Screenshot", "HuggingFace", "Cloudflare", "Pollinations", "Nano-Scene", "Pexels"]
        return any(k in source for k in valid_keywords)

    gen_success = sum(1 for c in chunks if c.get("visual_path") and is_valid_engagement_source(c.get("source")))
    
    if gen_success < len(chunks) * 0.5:
        log_message(f"⚠️ Primary visual generator only generated {gen_success}/{len(chunks)} visuals. Falling back to nano-scene engine (Imagen)...")
        chunks = generate_nano_scene_visuals(chunks, topic_context, style_guide=style_guide)
        
        # Check if nano-scene engine ALSO failed to produce new visuals
        final_success = sum(1 for c in chunks if c.get("visual_path") and os.path.exists(c["visual_path"]) and is_valid_engagement_source(c.get("source")))
        
        if final_success < len(chunks) * 0.5:
            log_message(f"⚠️ Both primary and nano-scene visual generation failed (only {final_success}/{len(chunks)} succeeded). Falling back to whiteboard animation videos!")
            from whiteboard_gen import generate_whiteboard_visuals
            chunks = generate_whiteboard_visuals(chunks, topic_context, is_longform=is_longform)
    else:
        log_message(f"✅ Primary visual generator successfully created {gen_success}/{len(chunks)} clips/images.")

    # ── STEP 8: Render Video ──────────────────────────────────────────────────
    log_message("STEP 8: Rendering final video with all engagement layers...")
    try:
        title  = script_data.get("title")
        script = script_data.get("script")
        subcat = script_data.get("sub_category", "")
        # Normalize companies_mentioned to list of strings
        raw_companies = script_data.get("companies_mentioned", [])
        companies = []
        for c in raw_companies:
            if isinstance(c, dict):
                name = c.get("name")
                if name:
                    companies.append(name)
            elif isinstance(c, str):
                companies.append(c)
        keywords    = script_data.get("keywords", [])
        hashtags    = script_data.get("hashtags", [])
        breaking_level = script_data.get("breaking_news_level", 0)
        voice_used  = script_data.get("edge_tts_voice")

        # Determine visual style theme matching the tone/topic of the spoken content
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

        video_path = create_video(audio_path, script_data, chunks)
        if not video_path or not os.path.exists(video_path):
            raise Exception("Video file not created.")
    except Exception as e:
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
    # Ensure variety in titles using the options if generated
    ai_desc = script_data.get("description", "")
    if script_data.get("title_options"):
        title = random.choice(script_data["title_options"])
    
    # Generate dynamic, optimized hashtags and tags
    initial_people = [p.get("name") for p in script_data.get("people", [])] if script_data.get("people") else []
    optimized_metadata = get_optimized_metadata(
        title=title,
        script=script,
        sub_category=script_data.get("sub_category", ""),
        initial_keywords=script_data.get("keywords", []),
        initial_companies=script_data.get("companies_mentioned", []),
        initial_people=initial_people,
        initial_hashtags=script_data.get("hashtags", []),
        is_shorts=True
    )
    hashtags = optimized_metadata["hashtags"]
    tags = optimized_metadata["tags"]
    
    log_message(f"Optimized Tags: {tags}")
    log_message(f"Optimized Hashtags: {hashtags}")

    # ── Output Metadata ──
    relevant_links = script_data.get("relevant_links", [])
    source_url = script_data.get("original_news_url", "")
    description = format_description(ai_desc, script, hashtags, slot=slot, chunks=chunks, relevant_links=relevant_links, source_url=source_url)
    
    # Algorithm Optimization: Append primary hashtag to title for Shorts feed boost
    if hashtags:
        primary_tag = hashtags[0]
        if primary_tag.lower() not in title.lower():
            # Ensure we don't exceed 100 character limit
            if len(title) + len(primary_tag) < 95:
                title = f"{title} {primary_tag}"


    # YouTube Upload
    if dry_run:
        print("🧪 [DRY RUN] Simulating YouTube upload...")
        uploaded, result = True, "MOCK_VIDEO_ID"
    else:
        uploaded, result = upload_video(
            video_path=video_path,
            title=title,
            description=description,
            tags=tags,
            thumbnail_path=thumbnail_path,
            comment_hook=script_data.get("comment_hook")
        )

    if not uploaded:
        log_message(f"ERROR: YouTube upload failed: {result}")
        return False

    youtube_url = f"https://youtu.be/{result}"
    log_message(f"SUCCESS: {youtube_url}")

    # ── STEP 10c: X.com Auto-Post ─────────────────────────────────────────────
    log_message("STEP 10c: Auto-posting Short to X.com...")
    try:
        x_post_text = f"🔥 {title}\n\nFull breakdown: {youtube_url}\n\n" + " ".join(hashtags)
        if dry_run:
            print("🧪 [DRY RUN] Simulating X.com auto-post...")
            x_uploaded, x_result = True, "MOCK_TWEET_ID"
        else:
            x_uploaded, x_result = upload_video_to_x(video_path, x_post_text)

        if x_uploaded:
            log_message(f"SUCCESS: Posted video to X.com! Tweet ID: {x_result}")
        else:
            log_message(f"WARNING: X.com posting skipped/failed: {x_result}")
    except Exception as ex:
        log_message(f"WARNING: X.com auto-post failed: {ex}")

    # ── STEP 10d: Instagram Reels Auto-Post ───────────────────────────────────
    log_message("STEP 10d: Auto-posting Reel to Instagram...")
    try:
        ig_caption = f"🔥 {title}\n\n" + " ".join(hashtags[:15])
        if dry_run:
            print("🧪 [DRY RUN] Simulating Instagram Reel upload...")
            ig_uploaded, ig_result = True, "MOCK_IG_REEL_ID"
        else:
            ig_uploaded, ig_result = upload_reel_to_instagram(video_path, ig_caption)

        if ig_uploaded:
            log_message(f"SUCCESS: Posted Reel to Instagram! ID: {ig_result}")
        else:
            log_message(f"WARNING: Instagram posting skipped/failed: {ig_result}")
    except Exception as ex:
        log_message(f"WARNING: Instagram auto-post failed: {ex}")

    # ── STEP 10b: Generate Pinned Comment ───────────────────────────────────
    next_slot = get_next_slot(slot)
    pinned_comment = generate_pinned_comment(script_data, next_slot)
    log_message(f"📌 PINNED COMMENT TEMPLATE:\n\n{pinned_comment}\n")

    # ── STEP 11: Update Tracker ───────────────────────────────────────────────
    log_message("STEP 11: Updating YouTube URL in tracker...")
    update_youtube_url(script_data.get("original_news_headline"), youtube_url)

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
    
    # ── STEP 13: Terminate Orphan Processes ───────────────────────────────────
    log_message("STEP 13: Terminating leftover ffmpeg processes to prevent pipeline hang...")
    try:
        # In CI/CD, orphaned ffmpeg processes can hold stdout open and cause the job to hang
        os.system("pkill -9 -f ffmpeg")
    except Exception as e:
        log_message(f"Process cleanup failed: {e}")
    
    return True


def run_local(topic_type="auto", dry_run=False):
    # XTTS server launch removed. Calling pipeline directly.
    success = run_pipeline(topic_type=topic_type, dry_run=dry_run)
    if not success:
        print("❌ Pipeline failed. Exiting with error code.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Run pipeline immediately.")
    parser.add_argument("--type", type=str, choices=["auto", "research", "tools", "news", "tech_trends", "vaibhav"], default="auto", help="Content type mapped to the schedule")
    parser.add_argument("--dry-run", action="store_true", help="Run without uploading to YouTube/X.com/Telegram.")
    args = parser.parse_args()

    if args.now or args.dry_run:
        run_local(topic_type=args.type, dry_run=args.dry_run)
    else:
        print("Usage: python main.py --now")
        print("For dry runs: python main.py --dry-run")
        print("For scheduled runs: python scheduler.py")
