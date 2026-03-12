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

    # ── STEP 0: GEMINI SEARCH (If no articles provided) ─────────────────────
    if not articles:
        print(f"🔍 STEP 0: Using Gemini Search for {topic_type} ({category})...")
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
                            grounding_links.append(f"{chunk.web.title}: {chunk.web.uri}")
            
            links_str = "\n".join(grounding_links)
            # Use the grounded response to build a context
            news_context = f"GEMINI SEARCH RESULTS (Grounded):\n{search_response.text}\n\nSOURCES FOUND:\n{links_str}\n"
            print(f"✅ Gemini Search completed with {len(grounding_links)} sources.")
        except Exception as e:
            print(f"⚠️ Gemini Search failed: {e}. Falling back to empty context.")
            news_context = "No news articles found."
    else:
        # ── Pre-filter articles (Unique against history AND against each other) ──
        filtered_articles = []
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
            print("No unique articles remaining to process.")
            return None
            
        articles = filtered_articles 
        
        news_context = ""
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
        "CRITICAL: The final video MUST be under 60 seconds. Target a script length of ~135 words.\n"
    )

    day_name, slot, category = get_slot_info()
    strategy_enhancement = get_category_prompt_enhancement(category, slot)
    
    prompt = f"""Act as a Senior AI Investigative Journalist & Creative Director for a 2026 YouTube Content Ecosystem. 
Your goal is to transform technical RSS/Research data into a "Deep-Dive" YouTube Short that balances analytical rigor with high-retention visual storytelling.

TODAY'S STRATEGY: 
Day: {day_name}
Slot: {slot}
{strategy_enhancement}

CONTENT HIERARCHY (Anti-Repetitive & Monetization Compliance):
1. TRANSFORMATIVE COMMENTARY (50%): This is the absolute law for 2026 monetization. Pure reporting is DEAD. You must spend half the script on YOUR analysis. Include "Hot Takes", "Unpopular Opinions", or "Why this looks like a corporate cover-up." 
2. UNIQUE PERSPECTIVE (20%): Connect this to a 2026 world problem. (e.g., "This isn't just about a model; it's about the future of cognitive labor...").
3. THE NARRATIVE ARC (20%): Structure it as a "Personal Leak" or an "Internal Investigation" vibe.
4. HUMAN IMPERFECTIONS (10%): Use the persona (VJ). Use stutter-starts: "Wait, hold on...", "Actually...", "Believe it or not...". This "Proves" human authorship to AI detection algorithms.

🏆 2026 COMPLIANCE CHECK: 
- TRANSFORMATION SCORE: At least 4 sentences MUST be purely your original synthesis/prediction.
- AUTHENTICITY SCORE: Use 1st person pronouns ("I", "Me", "My take").

🎯 VIEWER RETENTION RULES (CRITICAL — These decide if viewers swipe or stay):
1. PATTERN INTERRUPT (First 0.5s): The FIRST SENTENCE must be a shocking claim, contradiction, or provocative question. NEVER start with greetings, "Hey guys", or "Did you know". Start MID-THOUGHT as if the viewer walked into a conversation already happening. Examples: "Google just killed passwords. Forever.", "This AI writes code 10x faster than you.", "90% of developers don't know this exists."
2. CURIOSITY GAP (2-8s): After the hook, OPEN AN INFORMATION LOOP the brain NEEDS to close. Use phrases like: "but here's what nobody is talking about...", "and the reason will surprise you...", "but wait until you see WHY..."
3. MICRO-CLIFFHANGERS (Every 10s): Plant 3-4 teasers throughout the script to prevent mid-video swipe. Examples: "But that's not even the craziest part...", "And here's where it gets really interesting...", "Now pay attention to this next part..."
4. INTERACTIVE CHALLENGE: Include ONE moment where you challenge the viewer: "Pause and guess...", "Comment your prediction...", "90% get this wrong — are you in the 10%?"
5. IDENTITY-BASED CTA (Last 5s): Do NOT say "Subscribe for more." Instead, make subscribing feel like joining an identity: "If you're the kind of person who wants to know about AI before everyone else... you know what to do." or "Follow if you want to stay ahead of the curve."
6. REPLAY TRIGGER: Reference something from the first 2 seconds in the last 5 seconds. This creates an infinite loop and boosts replay rate (the #1 algorithm signal).

NARRATION STYLE: 
- Tone: Provocative, sharp, and highly opinionated. 
- Personality (VJ): You are a high-level lead analyst. Don't be "AI Assistant-y." Be the "Tech Whistleblower."
- Use Dramatic Pacing: ... for 0.4s pause. -- for 0.2s breath. ALL CAPS for emphasis.

NARRATIVE ARC CONFIGURATION:
1. The PATTERN INTERRUPT Hook (0-2s): A punchy, mid-thought statement that stops the scroll. NO warmup.
2. The CURIOSITY GAP (2-8s): Open an information loop. Tease the payoff without revealing it.
3. The "VJ's Take" (8-20s): Breaking the news through a lens of skepticism or extreme hype.
4. The Deep-Dive Analysis (20-38s): The "So What?" for the viewer's wallet or brain. Include the INTERACTIVE CHALLENGE here.
5. The Visual Reset (38-44s): An abrupt micro-cliffhanger to wake up the lurkers.
6. The IDENTITY CTA + Infinite Loop (44-52s): Identity-based subscribe CTA, then bridge the final sentence back to the hook for max replay rate.

CRITICAL '2026 SCALE' RULES:
1. MONETIZATION PROTECTION: If the script is >50% factual summary, it will be flagged as "Reused Content." PUSH THE OPINION.
2. AGGRESSIVE ENGAGEMENT: Use "Bait & Switch" questions. Start a sentence, pause, and say "Actually, tell me in the comments first: do you think this is ethical? Now look at this..."
3. NO REPETITION: Never repeat a word or phrase within 10 seconds.
4. WHATSAPP & TELEGRAM CTA: You MUST give a 1-second pause (...) before the end. Speak the CTA fully: "Send your suggestions and feedback on my WhatsApp and Telegram (links in bio), I read everything... See you in the next one."
5. AI DISCLOSURE: Embrace the "AI-Journalist" persona. Mention your neural nets or analytical processing to build the brand.

{selection_instruction}

RESEARCH PAPERS & BLOGS DATA:
{news_context}

NARRATIVE FLOW (FOR THE 'SCRIPT' FIELD):
- Pattern Interrupt -> Curiosity Gap -> VJ's Take -> Deep-Dive + Challenge -> Visual Reset -> Identity CTA + Loop Connect.

{extra_instruction}

Return ONLY this exact JSON (no markdown, no explanation) to securely match the automation pipeline:
{{
  "title_options": ["Title idea 1", "Title idea 2", "Title idea 3"],
  "description": "Full 100+ word rich SEO description for youtube describing the video, including timestamps and credits.",
  "quiz_tone": "Investigative",
  "title": "Punchy YouTube title max 60 chars",
  "script": "Full voiceover script following the Arc (50-52 sec). Aim for 130-140 words max. Ensure The Loop and Pattern Interrupt are implemented.",
  "hook_text": "The exact first 5-8 words of the script. This will appear as giant text on screen in the first 1.5 seconds to STOP THE SCROLL.",
  "micro_cliffhangers": [
    {{"timestamp": 10.0, "text": "But here's what nobody's talking about..."}},
    {{"timestamp": 22.0, "text": "And this is where it gets wild..."}},
    {{"timestamp": 35.0, "text": "Now watch what happens next..."}}
  ],
  "interactive_challenge": {{
    "timestamp": 25.0,
    "text": "90% get this wrong — comment your guess!",
    "type": "comment_challenge"
  }},
  "identity_cta": "If you're serious about AI, you know what to do.",
  "phonetic_pronunciation_map": {{
    "AI": "A-I",
    "NVIDIA": "En-vid-ee-uh",
    "Algorithm": "al-go-rith-um",
    "Autonomous": "aw-ton-uh-mus"
  }},
  "NOTE_phonetic_pronunciation_map": "MANDATORY LEXICAL AUDIT: Identify EVERY word that is: 1) A tech/scientific term, 2) A word with >3 syllables, 3) A homograph with ambiguous sounds (e.g., 'read' vs 'read'), or 4) Any English word that a machine might pronounce 'flat' or wrong. Provide a hyphenated phonetic respelling. This corrected text is what will be SPOKEN, but subtitles will show ORIGINAL text.",
  "hook": "Matches the first sentence of the script",
  "summary": "One line summary",
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
  "original_news_url": "MANDATORY: Pick the most relevant full URL from the SOURCES FOUND section (e.g. https://domain.com/path). DO NOT leave as placeholder.",
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
  "comment_hook": "Would you use this? Drop your answer below! 👇",
  "end_question": "A thought-provoking question for the audience to comment on"
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
