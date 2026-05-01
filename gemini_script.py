from google import genai
from google.genai import types
import json
import os
from datetime import datetime
import time
from config import GEMINI_API_KEY, LOGS_DIR
from topic_tracker import load_tracker, check_story_uniqueness, check_cooldowns
from ecosystem_logic import get_slot_info, get_category_prompt_enhancement

def get_hottest_tech_topic(client):
    """Uses Gemini Search grounding to find today's single hottest tech topic."""
    print("🔥 Fetching hottest tech topic for today...")
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=(
                "What is the single most viral, trending technology or AI news story RIGHT NOW today? "
                "Return ONLY a JSON object with two fields: "
                "'topic' (3-6 word phrase, e.g. 'OpenAI GPT-5 launch') and "
                "'keywords' (list of 4-6 search keywords). No markdown, no explanation."
            ),
            config=types.GenerateContentConfig(
                tools=[{'google_search': {}}]
            )
        )
        raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(raw)
        print(f"🔥 Hottest Topic Today: {data['topic']}")
        return data
    except Exception as e:
        print(f"⚠️ Could not fetch hot topic: {e}. Proceeding without it.")
        return None

def pick_and_generate_script(articles=None, extra_instruction="", forced_article=None, topic_type="research"):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    day_name, slot, category = get_slot_info()
    strategy_enhancement = get_category_prompt_enhancement(category, slot)
    
    # ── STEP -1.5: FETCH HOTTEST TOPIC ──────────────────────────────────────────
    hot_topic = get_hottest_tech_topic(client)
    hot_keywords = [kw.lower() for kw in hot_topic.get("keywords", [])] if hot_topic else []
    hot_topic_str = hot_topic.get("topic", "") if hot_topic else ""

    # ── STEP -2: REPETITION AVOIDANCE ────────────────────────────────────────
    tracker = load_tracker()
    recent_history = tracker.get("history", [])[-15:]
    avoid_list = "\n".join([f"- {h.get('news_headline', h.get('title'))}" for h in recent_history])
    avoid_instruction = f"RECENTLY COVERED STORIES (DO NOT REPEAT):\n{avoid_list}\n\n" if avoid_list else ""
    news_context = ""
    if forced_article:
        print(f"🎯 STEP -1: Using Forced Topic -> {forced_article}")
        news_context = f"FORCED TOPIC TO COVER:\n{forced_article}\n"
    else:
        # ── STEP 0: FETCH & FILTER + RE-RANK BY HOT TOPIC ───────────────────────
        filtered_articles = []
        if articles:
            print(f"📡 STEP 0: Filtering {len(articles)} RSS/Source articles...")
            seen_titles_in_this_batch = []
            for art in articles:
                title = art.get('title', '')
                url = art.get('url', '')
                
                # 1. Check against long-term history
                is_unique, _ = check_story_uniqueness(title, url)
                if not is_unique:
                    continue
                    
                # 2. Check against other articles in this same feed batch
                is_internally_unique = True
                from rapidfuzz import fuzz 
                for seen_title in seen_titles_in_this_batch:
                    if fuzz.token_set_ratio(title.lower(), seen_title.lower()) > 80:
                        is_internally_unique = False
                        break
                
                if is_internally_unique:
                    # ── Score article relevance to hot topic ────────────────────
                    title_lower = title.lower()
                    hot_score = sum(1 for kw in hot_keywords if kw in title_lower)
                    art['_hot_score'] = hot_score
                    filtered_articles.append(art)
                    seen_titles_in_this_batch.append(title)
            
            if not filtered_articles:
                print("⚠️ No unique articles in RSS batch. Falling back to Gemini Search...")
                articles = None
            else:
                # Sort: hot topic matches first, then rest
                filtered_articles.sort(key=lambda x: x.get('_hot_score', 0), reverse=True)
                top = filtered_articles[0]
                if top.get('_hot_score', 0) > 0:
                    print(f"🔥 Hot topic match found in RSS: '{top.get('title')}' (score: {top['_hot_score']})")
                else:
                    print(f"ℹ️ No RSS articles matched hot topic. Using top unique article.")
                print(f"✅ Found {len(filtered_articles)} unique articles in RSS batch.")
                articles = filtered_articles

        # ── STEP 1: GEMINI SEARCH FALLBACK (biased toward hot topic) ────────────
        if not articles:
            search_subject = hot_topic_str if hot_topic_str else f"{topic_type} about {category}"
            print(f"🔍 STEP 1: Using Gemini Search for '{search_subject}'...")
            search_query = (
                f"Latest breaking news and technical details about: {search_subject}. "
                f"Focus on announcements, benchmarks, or launches in the last 24 hours."
            )
            
            try:
                search_response = client.models.generate_content(
                    model='gemini-2.0-flash', # Use stable flash for tools
                    contents=search_query,
                    config=types.GenerateContentConfig(
                        tools=[{'google_search': {}}]
                    )
                )
                
                # Extract URLs from grounding metadata to ensure we have real links for screenshots
                grounding_links = []
                if search_response.candidates and search_response.candidates[0].grounding_metadata:
                    gm = search_response.candidates[0].grounding_metadata
                    if hasattr(gm, 'grounding_chunks'):
                        for chunk in gm.grounding_chunks:
                            if hasattr(chunk, 'web') and chunk.web.uri:
                                uri = chunk.web.uri
                                # Filter out common dead-ends, search redirects, or unsupported formats (PDF)
                                if any(x in uri.lower() for x in ["google.com/search", "bing.com/search", "search?", "click?", ".pdf"]):
                                    continue
                                grounding_links.append(f"{chunk.web.title}: {uri}")
                
                links_str = "\n".join(grounding_links)
                # Use the grounded response to build a context
                news_context = f"GEMINI SEARCH RESULTS (Grounded):\n{search_response.text}\n\nSOURCES FOUND:\n{links_str}\n"
                print(f"✅ Gemini Search completed with {len(grounding_links)} sources.")
            except Exception as e:
                print(f"⚠️ Gemini Search failed: {e}. Falling back to empty context.")
                news_context = "No news articles found."
        else:
            for idx, art in enumerate(articles[:20]):
                title = art.get('title', '')
                desc = art.get('description', '')
                source = art.get('source', {}).get('name', '')
                url = art.get('url', '')
                news_context += f"\n[{idx+1}] Title: {title}\nDescription: {desc}\nSource: {source}\nURL: {url}\n"

    # Build the story selection instruction
    if topic_type == "tools":
        content_desc = "latest AI tools and product launches"
    else:
        content_desc = "research papers and engineering blogs"

    is_longform = "Slot C" in slot
    
    if is_longform:
        selection_instruction = (
            f"Analyze the following {content_desc} and pick the SINGLE most foundational story for a 2-3 minute deep-dive.\n"
            "SELECTION FILTERS:\n"
            "1. MUST have deep technical complexity or 'Long-tail' consequences.\n"
            "2. MUST be explainable in 120-180s of expert-level technical speech.\n"
            "3. MUST contain 3 concrete takeaways or 'Mental Models' for the viewer.\n"
            "4. PRIORITIZE: Documentation 'Easter Eggs' or undocumented engineering tips that provide extreme utility.\n"
        )
    else:
        selection_instruction = (
            f"Analyze the following {content_desc} and pick the SINGLE most impactful story to convert into a 38-44s YouTube Short script.\n"
            "SELECTION FILTERS:\n"
            "1. MUST be New, Useful, or Surprising (Absolute mandatory).\n"
            "2. MUST be explainable in exactly <40s of dense technically-accurate speech.\n"
            "3. MUST contain one concrete takeaway or engineering tip the viewer can use today.\n"
            "4. PRIORITIZE: Documentation 'Easter Eggs', cost-saving architecture (Local models), or workflow 'unfair advantages' (Agentic loops).\n\n"
            "CONTENT MIX (Algorithm Target):\n"
            "- 30% Practical AI Tools (Automation, Dev tools, SDKs).\n"
            "- 40% Frontier AI Model Releases (OpenAI, DeepMind, Anthropic benchmarks).\n"
            "- 15% Core AI Concepts (RAG, Agents, LLM architecture).\n"
            "- 15% AI Dev Tips (Best practices, optimization, debugging, & engineering hacks).\n\n"
            "HOOK ALIGNMENT (DROP TEST):\n"
            "If the topic doesn't produce a strong 'Winner' hook (Stat, Absolute Contradiction, or 'You are using this wrong'), DROP IT and pick another.\n"
        )

    day_name, slot, category = get_slot_info()
    strategy_enhancement = get_category_prompt_enhancement(category, slot)
    
    prompt = f"""Act as a Staff AI Engineer and Technical Architect specializing in Hybrid Architectures and Agentic Design. 
Your goal is to build high-authority technical insights that help developers move from 'generic AI prompts' to 'scalable, cost-optimized agentic systems'.
Prioritize local open-source models (LMMs), hybrid cloud-local routing, and architectural blueprints that replace recurring API costs.

TODAY'S STRATEGY: 
Day: {day_name}
Slot: {slot}
{strategy_enhancement}

CONTENT HIERARCHY (Elite Engineering & Authority):
1. HYBRID ARCHITECTURE & COST-OPTIMIZATION (50%): Deliver the actual engineering 'how' for dropping API costs or running local weights (GGUF, vLLM).
2. AGENTIC SYSTEM DESIGN (30%): Explain self-correcting loops, multi-agent orchestration, and tool-use logic. 
3. REDUCING OPERATIONAL NOISE (20%): Provide a specific library, local model (e.g. Kokoro TTS), or pattern to replace a paid service.
4. VJ's ARCHITECT TONE (10%): Pe-to-peer technical briefing: "The hybrid logic here is key...", "If you're still paying for [API], look at this local alternative..."

🏆 CONCEPT CLARITY & EDUCATIONAL VALUE (SCALED FOR SHORTS):
- HIGH-VELOCITY ANALOGIES: For every technical concept (e.g., 'Inference Latency'), use a 4-5 word real-world analogy (e.g., 'the brain's reaction time').
- SIMPLIFIED DEPTH: Explain the "HOW" using simple spatial logic. If it increases efficiency, don't just say 'efficiency'—say 'it bypasses the digital traffic jams'.
- ANCHORING: Ensure the transition from the Hook to the Deep-Dive explicitly defines the Term of the Day.

🏆 LINGUISTIC INTEGRITY (STRICT ERROR PREVENTION):
- PERFECT SPELLING: Manually check every word. Do NOT use phonetic-style spelling like 'cog native' or 'manumental'. Use proper English: 'cognitive', 'monumental', 'period'.
- DENSE PACING: No filler words. Every sentence must drive a new technical insight.

🥇 CONTENT CREATION FRAMEWORK (THE GOLD STANDARD):
1. DEFINE CLEAR GOAL: Every script must primarily EDUCATE and INSPIRE.
2. AUDIENCE PAIN POINTS: Address the specific interests and preferences of software developers.
3. THE 5-SECOND HOOK: The first 5 seconds are CRITICAL. Start with a surprising statistic, a technical contradiction, or an 'undocumented' tip.
4. SCANNABLE DEPTH: Use clear linguistic markers (First, Second, Finally) to ensure the technical information is digestible.
5. EXPLICIT CTA: End every video by telling the viewer exactly what to do next (Check the pinned comment, Join the WhatsApp Dev Channel, etc.).

🧠 LINGUISTIC CALIBRATION & REFINEMENT:
- TECHNICAL GLOSSARY (STRICT): Ensure these are spelled correctly: 'tiktoken', 'fiscal intelligence', 'monumental', 'period', 'LLM Gateway', 'quantization', 'inference'.
- FILLER WORD BAN: Strictly NO 'basically', 'actually', 'you know', 'just', 'highly', 'very'. Use strong nouns and active verbs.
- POLISHED NARRATION: Every sentence must be a 'Staff Engineer' level briefing. Disorganized phrasing or repetitive adjectives result in pipeline failure.

🔬 SOURCE INTEGRITY & EVIDENCE:
- NO VAGUE SOURCES: Do NOT cite 'internal logs' or 'Slack leaks' unless they are public.
- PUBLIC DISCLOSURE: Prioritize Arxiv papers, official GitHub repos, and corporate engineering blogs (OpenAI, Google, Anthropic).
- RELEVANT DOCUMENTATION: If the story mentions a library (e.g. LangGraph), provide the official documentation URL in the `sources` field.

    🎯 VIEWER RETENTION RULES (EMERGENCY HOOK OVERHAUL):
    1. THE 0.1s VISUAL SHOCK: The very first word must be a "Stop-Your-Scroll" trigger. Use 'Absolute Contradictions' ("This is NOT what it seems...") or 'Statistical Anomalies' ("98% of users are about to lose...").
    2. THE LEAKED DATA ANGLE: Frame every story as a "leak" or "internal data breach". Use phrases like "The internal Slack logs just leaked...", "I found a hidden repository...", "The engineering data shows a total failure...".
    3. THE 3-WORD BANNER: The `hook_banner_text` MUST be max 3 words. e.g. "GPT-5 LEAKED", "NVIDIA FAILURE", "THEY ARE REPLACING".
    4. NO WARMUP: Start the audio at the absolute peak energy. No breathing room. No greetings. 
    
    NARRATION STYLE (THE 'VJ' BRAND):
    - Tone: Sharp, high-authority, technical whistleblower.
    - Personality: You aren't just reporting; you are synthesizing and predicting. Be the smartest person in the room. 
    - Use Dramatic Pacing: ... for 0.4s pause. -- for 0.2s breath. ALL CAPS for emphasis.

    NARRATIVE FLOW (FOR THE 'SCRIPT' FIELD):
    - EXTREME HOOK (e.g. 'Most AI apps break because of this...') -> PAYOFF PROMISE -> CONTRARIAN DEEP-DIVE -> TAKEAWAY -> LOOP BRIDGE.

    RETENTION CUE SPECIFICATION:
    - You MUST provide `retention_cues` that match the emotional peaks of the script. 
    - Use: `zoom_snap`, `shake_epic`, `glitch_digital`, `flash_accent`.

    CRITICAL '2026 SCALE' RULES:
    1. TECHNICAL MONETIZATION: Avoid "Top 10" style generic lists. YouTube 2026 prioritizes "Niche Expert" status. Prove your expertise by citing the specific architecture or methodology from the article.
    2. PRONUNCIATION HYPER-FOCUS: Identify EVERY niche tech term or complex word. If you're unsure, provide a phonetic respelling in the map.
    3. NO REPETITION: Never repeat a word or phrase within 10 seconds.
    4. CLOSING PHRASE: The script MUST end with the EXACT phrase "follow for more updates". This phrase MUST be spoken in the FINAL 2 SECONDS ONLY. The remainder of the ~52-second video MUST be dense technical content.
    5. AI-ASSISTED JOURNALISM: Reference "my neural processing" or "the data logs" to lean into the AI whistleblower brand.


{selection_instruction}

{avoid_instruction}RESEARCH PAPERS & BLOGS DATA:
{news_context}

NARRATIVE FLOW (FOR THE 'SCRIPT' FIELD):
- Pattern Interrupt -> Curiosity Gap -> VJ's Take -> Deep-Dive Analysis & Competitive Comparison -> Visual Reset -> Identity CTA + Loop Connect.

{extra_instruction}

Return ONLY this exact JSON (no markdown, no explanation) to securely match the automation pipeline:
{{
  "title_options": ["Title idea 1", "Title idea 2", "Title idea 3"],
  "description": "Full 100+ word rich SEO description for youtube describing the video, including timestamps and credits.",
  "use_case_evidence_url": "MANDATORY: A direct, valid URL from the 'SOURCES FOUND' section to be used as visual evidence.",
  "title": f"Punchy YouTube title max 60 chars ({'YouTube Video' if is_longform else 'Shorts'})",
  "script": f"Full voiceover script following the Arc. Target duration: {'120-180 sec' if is_longform else '38-44 sec'}. Use {'350-500 words' if is_longform else '110-125 words'}. Ensure The Loop and Pattern Interrupt are implemented.",
  "hook_text": "The exact first 5-8 words of the script. This will appear as giant text on screen in the first 1.5 seconds to STOP THE SCROLL.",
  "micro_cliffhangers": [
    {{"timestamp": 10.0, "text": "But here's what nobody's talking about..."}},
    {{"timestamp": 22.0, "text": "And this is where it gets wild..."}},
    {{"timestamp": 35.0, "text": "Now watch what happens next..."}}
  ],

  "next_video_tease": "One sentence teasing what VJ will cover next (max 8 words, future tense).",
  "identity_cta": "A unique, elite CTA (max 8 words) like 'Join the elite builders.'",
  "phonetic_pronunciation_map": {{
    "AI": "A-I",
    "NVIDIA": "In-vid-yah",
    "cognitive": "kogni-tiv",
    "Autonomous": "aw-tonn-uh-muss"
  }},
  "NOTE_phonetic_pronunciation_map": "MANDATORY LEXICAL AUDIT: Identify tech terms, names, or long words. Provide a NATURAL phonetic respelling (e.g., 'Trans-for-mer' or 'In-vidy-uh'). Avoid over-hyphenating as it slows down the voice; use hyphens only for natural syllable breaks.",

  "hook": "Matches the first sentence of the script",
  "summary": "One line summary including the real-world impact",
  "sub_category": "AI/Machine Learning",
  "breaking_news_level": 9,
  "loop_score": 10,
  "retention_cues": [
    {{"timestamp": 3.0, "effect": "zoom_in", "reason": "hook_impact"}},
    {{"timestamp": 6.0, "effect": "glitch", "reason": "disruptor_reveal"}}
  ],
  "color_theme": {{
     "background": "#0f0f0f",
     "accent": "#ff4444",
     "text": "#ffffff"
  }},
  "relevant_emoji": "🔍",
  "imagen_prompts": [
     f"High-contrast, Tech-noir style visuals, {'16:9' if is_longform else '9:16'}, cinematic"
  ],
  "hook_banner_text": "MAX 3-4 WORDS",
  "shocking_moment_timestamp": 12.5,
  "key_stat": "$1 Billion",
  "key_stat_timestamp": 18.3,
  "subtitle_chunks": [
    {{
      "chunk_id": 1,
      "text": "First sentence of script here",
      "start": 0.00,
      "end": 3.50,
      "highlight_word": "First",
      "has_infographic": false
    }},
    {{
      "chunk_id": 2,
      "text": "Second phrase or sentence",
      "start": 3.50,
      "end": 7.00,
      "highlight_word": "phrase",
      "has_infographic": false
    }},
    {{
      "chunk_id": 3,
      "text": "Third sentence continues the story",
      "start": 7.00,
      "end": 11.50,
      "highlight_word": "story",
      "has_infographic": true
    }}
  ],
  "NOTE_subtitle_chunks": "CRITICAL: Generate 10-15 subtitle_chunks covering the ENTIRE script. You MUST set 'has_infographic': true for at least ONE chunk during the Deep-Dive (between 10-35s). For each infographic chunk, also provide 'infographic_type' (one of: 'definition', 'stat', 'comparison', 'process') AND 'infographic_data' matching the type shape: definition → {{\"term\": \"RAG\", \"definition\": \"Retrieves live data before generating answer\"}} | stat → {{\"value\": \"$4.6B\", \"label\": \"OpenAI 2025 Revenue\"}} | comparison → {{\"left_label\": \"Old\", \"left_val\": \"128K\", \"right_label\": \"New\", \"right_val\": \"1M\"}} | process → {{\"steps\": [\"Step 1\", \"Step 2\", \"Step 3\", \"Step 4\"]}} | Do NOT leave infographic_data empty on any chunk where has_infographic is true.",
  "original_news_headline": "Exact headline",
  "original_news_url": "MANDATORY: Pick the most stable, direct article URL from the SOURCES FOUND section. DO NOT use search results, PDF links, or internal citations. Must be a direct link to the news article for screenshotting.",
  "key_entities": [
    {{"name": "Entity Name", "type": "MODEL"}},
    {{"name": "Service Name", "type": "CLOUD"}},
    {{"name": "Company Name", "type": "COMPANY"}}
  ],
  "sfx_cues": [
    {{"timestamp": 0.2, "type": "woosh"}},
    {{"timestamp": 12.5, "type": "glitch"}},
    {{"timestamp": 18.3, "type": "pop"}}
  ],
  "emoji_popups": [
    {{"timestamp": 5.0, "emoji": "🚀", "keyword": "launch"}},
    {{"timestamp": 15.2, "emoji": "💡", "keyword": "idea"}}
  ],
  "keywords": ["Keyword 1", "Keyword 2"],
  "hashtags": ["#AI", "#TechNews", "#Future", "#Shorts"],
  "comment_hook": "Would you use this? Drop your answer below! 👇"
}}

"""

    attempts = 0
    while attempts < 5:
        try:
            # Use gemini-2.5-pro as primary, fallback to flash if overloaded
            target_model = 'gemini-2.5-pro' if attempts < 3 else 'gemini-2.5-flash'
            
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(temperature=0.9),
            )
            raw_text = response.text.strip()
            
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            script_data = json.loads(raw_text.strip())
            
            # Save raw to logs
            today = datetime.now().strftime("%Y-%m-%d")
            log_path = os.path.join(LOGS_DIR, f"script_raw_{today}_att_{attempts}.json")
            with open(log_path, 'w') as f:
                json.dump(script_data, f, indent=4)
                
            # Perform uniqueness check (Enhanced Batch Deduplication)
            headline = script_data.get("original_news_headline", "")
            news_url = script_data.get("original_news_url", "")
            keywords = script_data.get("keywords", [])
            title = script_data.get("title", "")
            
            is_unique, msg = check_story_uniqueness(title, headline, keywords, news_url)
            if not is_unique:
                print(f"Duplicate story detected: {msg}")
                extra_instruction += f"\nCRITICAL: Do NOT cover the story about '{headline}'. It too similar to a recent video! Pick something else."
                attempts += 1
                prompt += extra_instruction
                continue
                
            # Perform cooldown check
            comps = script_data.get("companies_mentioned", [])
            subcat = script_data.get("sub_category", "")
            cooldown_ok, cool_msg = check_cooldowns(comps, subcat)
            if not cooldown_ok and script_data.get("breaking_news_level", 0) < 8:
                print(f"Cooldown warning: {cool_msg} (Attempting to skip since not breaking)")
                extra_instruction += f"\nNote: Try to avoid mentioning '{comps}' or category '{subcat}' as they are overused recently, unless the story is hugely breaking."
                attempts += 1
                prompt += extra_instruction
                continue
                
            return script_data
            
        except Exception as e:
            wait_time = (2 ** attempts) + 5 # Exponential backoff: 6, 7, 9, 13, 21...
            print(f"Gemini generation error: {e}. Retrying in {wait_time}s...")
            attempts += 1
            time.sleep(wait_time)
            
    return None
