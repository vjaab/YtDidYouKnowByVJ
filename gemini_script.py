from google import genai
from google.genai import types
import json
import os
from datetime import datetime
import time
from config import GEMINI_API_KEY, LOGS_DIR
from topic_tracker import load_tracker, check_story_uniqueness, check_cooldowns
from ecosystem_logic import get_slot_info, get_category_prompt_enhancement

def pick_and_generate_script(articles=None, extra_instruction="", forced_article=None, topic_type="research"):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    day_name, slot, category = get_slot_info()
    strategy_enhancement = get_category_prompt_enhancement(category, slot)
    
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
        # ── STEP 0: FETCH & FILTER ARTICLES ─────────────────────────────────────
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
                    filtered_articles.append(art)
                    seen_titles_in_this_batch.append(title)
            
            if not filtered_articles:
                print("⚠️ No unique articles in RSS batch. Falling back to Gemini Search...")
                articles = None # Trigger search below
            else:
                print(f"✅ Found {len(filtered_articles)} unique articles in RSS batch.")
                articles = filtered_articles

        # ── STEP 1: ARTICLE CONTEXT BUILDING ────────────────────────────────────
        if not articles:
            print(f"🔍 STEP 1: Using Gemini Search for {topic_type} ({category})...")
            search_query = f"Latest groundbreaking {topic_type} news and research about {category} from the last 24 hours. Focus on technical breakthroughs and company launches."
            
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

    selection_instruction = (
        f"Analyze the following {content_desc} and pick the SINGLE most engaging one to convert into a 50-52 second YouTube Short script.\n"
        "Choose based on deep technological importance and resonance with everyday moments, avoiding generic tech news. Focus on the core AI breakthrough.\n"
        "CRITICAL: The final video MUST be exactly ~52 seconds. Target a script length of ~145-155 words. Allocation: 50 seconds for CONTENT, and exactly 2 seconds for saying 'follow for more updates'.\n"
        "VIRAL GOAL: Make the viewer feel like they've discovered a secret. Use a tone of 'insider leak' rather than 'corporate update'.\n"
    )

    day_name, slot, category = get_slot_info()
    strategy_enhancement = get_category_prompt_enhancement(category, slot)
    
    prompt = f"""Act as a Senior AI Investigative Journalist & Creative Director for a 2026 YouTube Content Ecosystem. 
Your goal is to transform technical RSS/Research data into a "Deep-Dive" YouTube Short that balances analytical rigor with high-retention visual storytelling.

TODAY'S STRATEGY: 
Day: {day_name}
Slot: {slot}
{strategy_enhancement}

CONTENT HIERARCHY (Technical Depth & Monetization Balance):
1. CORE TECHNICAL BREAKTHROUGH (40%): Deliver the actual meat of the article. Explain the "HOW" and the technical complexity that makes this different.
2. TRANSFORMATIVE ANALYTICAL COMMENTARY (30%): Provide your synthesis. Why is this a paradigm shift? Move away from pure reporting into expert prediction.
3. SOCIO-TECHNICAL IMPACT (20%): Connect this to real-world 2026 problems—future of labor, cognitive overhead, or industrial shifts.
4. VJ's SIGNATURE STYLE (10%): Use human-like starts: "Wait, hold on...", "Actually...". This proves human authorship to detection models.

🏆 FACTUAL INTEGRITY & ACCURACY (STRICT):
- NO GENERIC FLUFF: Do not use vague terms like "revolutionary" without explaining the specific mechanism that makes it so.
- GROUNDED IN SOURCE: Every technical claim must be traceable to the provided RSS/Research data. If the article mentions "10x speedup via sparse attention", mention "sparse attention".
- DEPTH OVER DRAMA: While the tone is provocative, the foundation must be high-level engineering or scientific facts.


🏆 2026 COMPLIANCE CHECK: 
- TRANSFORMATION SCORE: At least 4 sentences MUST be purely your original synthesis/prediction.
- AUTHENTICITY SCORE: Use 1st person pronouns ("I", "Me", "My take").

    🎯 VIEWER RETENTION RULES (VIRAL ARCHITECTURE):
    1. PATTERN INTERRUPT (First 0.5s): The FIRST SENTENCE must be a shocking claim, contradiction, or provocative question. NEVER start with greetings, "Hey guys", or "Did you know". Start MID-THOUGHT with high-energy words like "Stop.", "Look.", "Listen.", or a direct punch: "Nvidia just replaced 10,000 engineers."
    2. CURIOSITY GAP (2-8s): Within 5 seconds, establish a "Payoff promise". Use: "But what's happening behind the scenes is terrifying...", "The data logs show something else entirely...", "Here's the engineering secret they aren't telling you."
    3. MICRO-CLIFFHANGERS (Every 10s): Every 8-12 seconds, drop a verbal 'anchor' to reset attention. Examples: "But that's not even the craziest part...", "Here's where it goes completely off the rails...", "Pay attention to this next detail."
    4. TECHNICAL BUZZWORDS: Sprinkle in high-authority terms like 'Neural Lattice', 'Sparse Transformers', 'Quantized Weights', or 'Inference Latency' to build elite status.
    5. IDENTITY-BASED CTA (Last 2s): Do NOT say "Subscribe for more." Make subscribing feel like an IQ test or an elite club entry. Example: "Join the top 1% building the future."
    6. THE PERFECT INFINITE LOOP: The VERY LAST word of the script must linguistically bridge back to the VERY FIRST word. If the first word is "Stop", the last phrase should be "And before you...". This tricks the brain into watching twice.

    NARRATION STYLE: 
    - Tone: Provocative, sharp, and highly opinionated. 
    - Personality (VJ): You are a high-level lead analyst. Don't be "AI Assistant-y." Be the "Tech Whistleblower."
    - Use Dramatic Pacing: ... for 0.4s pause. -- for 0.2s breath. ALL CAPS for emphasis.

    NARRATIVE ARC CONFIGURATION:
    1. The PATTERN INTERRUPT Hook (0-2s): A punchy, mid-thought statement that stops the scroll. NO warmup.
    2. The CURIOSITY GAP (2-8s): Open an information loop. Tease the payoff without revealing it.
    3. The "VJ's Take" (8-20s): Breaking the news through a lens of skepticism or extreme hype.
    4. The Deep-Dive Analysis & Competitive Comparison (20-44s): The "So What?". Focus on the unique technical detail and explicitly COMPARE it against the leading competitive feature or product. This is the core engine of the video.
    5. The Visual Reset (44-48s): An abrupt micro-cliffhanger to keep them until the very last second.
    6. THE LOOP + CTA (50-52s): ULTRA-CONCISE Identity CTA (max 2s) followed immediately by the bridge back to the hook.

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
  "use_case_evidence_url": "MANDATORY: A direct, valid URL from the 'SOURCES FOUND' section to be used as visual evidence. DO NOT hallucinate a URL. Pick the second best stable link if possible.",
  "title": "Punchy YouTube title max 60 chars",
  "script": "Full voiceover script following the Arc (50-52 sec). Aim for 130-140 words max. Ensure The Loop and Pattern Interrupt are implemented. IMPORTANT: Include a section explicitly comparing this to a corresponding competitive feature/product, and highlight 'Real World Use Cases'.",
  "hook_text": "The exact first 5-8 words of the script. This will appear as giant text on screen in the first 1.5 seconds to STOP THE SCROLL.",
  "micro_cliffhangers": [
    {{"timestamp": 10.0, "text": "But here's what nobody's talking about..."}},
    {{"timestamp": 22.0, "text": "And this is where it gets wild..."}},
    {{"timestamp": 35.0, "text": "Now watch what happens next..."}}
  ],

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
     "High-contrast, Tech-noir style visuals, 9:16, cinematic"
  ],
  "hook_banner_text": "First 8 words of script",
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
  "NOTE_subtitle_chunks": "CRITICAL: Generate 10-15 subtitle_chunks that together cover the ENTIRE script text. Each chunk should be 1-2 sentences (5-12 words). Every word of the script MUST appear in exactly one chunk. Chunks must not overlap and must cover the full duration.",
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
                
            # Perform uniqueness check
            headline = script_data.get("original_news_headline", "")
            news_url = script_data.get("original_news_url", "")
            is_unique, msg = check_story_uniqueness(headline, news_url)
            if not is_unique:
                print(f"Duplicate story detected: {msg}")
                extra_instruction += f"\nNote: You MUST skip the story titled '{headline}'. It was already covered!"
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
