from google import genai
import json
import os
from datetime import datetime
import time
from config import GEMINI_API_KEY, LOGS_DIR
from topic_tracker import load_tracker, check_story_uniqueness, check_cooldowns
from ecosystem_logic import get_slot_info, get_category_prompt_enhancement

def pick_and_generate_script(articles, extra_instruction="", forced_article=None, topic_type="research"):
    client = genai.Client(api_key=GEMINI_API_KEY, http_options={'timeout': 60.0})
    
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
        from rapidfuzz import fuzz # In case it's not imported here
        for seen_title in seen_titles_in_this_batch:
            if fuzz.token_set_ratio(title.lower(), seen_title.lower()) > 80:
                is_internally_unique = False
                break
        
        if is_internally_unique:
            filtered_articles.append(art)
            seen_titles_in_this_batch.append(title)
    
    if not filtered_articles and not forced_article:
        print("No unique articles remaining to process.")
        return None
        
    articles = filtered_articles # Use the filtered list
    
    news_context = ""
    for idx, art in enumerate(articles[:20]):
        title = art.get('title', '')
        desc = art.get('description', '')
        source = art.get('source', {}).get('name', '')
        url = art.get('url', '')
        image_url = art.get('urlToImage', '')
        news_context += f"\n[{idx+1}] Title: {title}\nDescription: {desc}\nSource: {source}\nURL: {url}\nImage URL: {image_url}\n"

    # Build the story selection instruction
    if topic_type == "tools":
        content_desc = "latest AI tools and product launches"
    else:
        content_desc = "research papers and engineering blogs"

    if forced_article:
        forced_title = forced_article.get('title', '')
        forced_url   = forced_article.get('url', '')
        selection_instruction = (
            f"The user has MANUALLY SELECTED this specific story — you MUST write the script about it:\n"
            f"Title: {forced_title}\nURL: {forced_url}\n"
            f"Do NOT pick a different story. Skip the selection process entirely."
        )
    else:
        selection_instruction = (
            f"Analyze the following {content_desc} and pick the SINGLE most engaging one to convert into a 60-second YouTube Short script.\n"
            "Choose based on deep technological importance and resonance with everyday moments, avoiding generic tech news. Focus on the core AI breakthrough.\n"
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
1. TRANSFORMATIVE COMMENTARY (40%): This is the #1 rule for monetization. Do NOT just summarize. You must ADD VALUE. Include your personal synthesis, skepticism, or "Why I'm genuinely panicked/excited about this." 
2. UNIQUE PERSPECTIVE (20%): Relate the news to a niche human problem or a 2026 future scenario that no one else is talking about.
3. THE NARRATIVE ARC (20%): Structure it as a personal discovery or a secret you're sharing, not a PR statement.
4. HUMAN IMPERFECTIONS (10%): Use your persona (VJ). Use filler words like "Wait...", "Look...", "Honestly...", "Correct me if I'm wrong but...", and "I was just reading this and—". These "Human-Glitch" hooks prove to YouTube's manual reviewers that this isn't a low-effort automated bot.
5. DATA & VISUALS (10%): Specific numbers and pattern interrupts to stay fast-paced.

NARRATION STYLE: 
- Tone: High-energy, investigative, and slightly opinionated. 
- Personality (VJ): You are a tech-native digital creator in 2026. You don't just "report" the news; you "deconstruct" it. If a company is doing something anti-consumer, CALL THEM OUT. If a paper is revolutionary, nerd out about it.
- Use Dramatic Pacing: Use ellipses (...) for 0.5s pauses and em-dashes (—) for abrupt shifts in thought. ALL CAPS for high-emphasis words.

NARRATIVE ARC CONFIGURATION:
1. The "Human First" Hook (0-5s): Start with a personal reaction or a direct question to the viewer's life. (e.g., "I think we just lost the war on deepfakes, listen to this...")
2. The "VJ's Take" (5-20s): The news itself, but filtered through your skepticism or excitement.
3. The Deep-Dive Analysis (20-40s): Why this is a "Black Swan" event or a predictable corporate move.
4. The Visual Reset (40-45s): A sudden punchy sentence to wake up the lurkers.
5. The Infinite Loop (45-58s): Design the ending so the last 3 words logically and phonetically CRASH back into the first sentence. The loop must be "God-Tier" seamless for high retention scores.

CRITICAL '2026 SCALE' RULES:
1. MONETIZATION PROTECTION: YouTube flags "Reused Content." To avoid this, your SCRIPT must contain 3+ sentences of purely original analysis/opinion that isn't in the source text.
2. AGGRESSIVE ENGAGEMENT: Use "Bait & Switch" questions. Start a sentence, pause, and say "Actually, tell me in the comments first: do you think this is ethical? Now look at this..."
3. NO REPETITION: Never repeat a word or phrase within 10 seconds unless for rhetorical effect.
4. WHATSAPP & CTA: Always include: "Join the 1% in my WhatsApp (link in bio) for the raw data I can't post here."
5. AI DISCLOSURE: Embrace the "AI-Journalist" persona. Use phrases like "As an AI, even I'm worried about this..." or "My neural nets are screaming right now."

{selection_instruction}

RESEARCH PAPERS & BLOGS DATA:
{news_context}

NARRATIVE FLOW (FOR THE 'SCRIPT' FIELD):
- Hook -> Disruptor -> So What? -> Visual Pivot -> Infinite Loop Connect.

{extra_instruction}

Return ONLY this exact JSON (no markdown, no explanation) to securely match the automation pipeline:
{{
  "title_options": ["Title idea 1", "Title idea 2", "Title idea 3"],
  "description": "Full 100+ word rich SEO description for youtube describing the video, including timestamps and credits.",
  "quiz_tone": "Investigative",
  "title": "Punchy YouTube title max 60 chars",
  "script": "Full voiceover script following the Arc (50-58 sec). Ensure The Loop is implemented.",
  "phonetic_pronunciation_map": {{
    "AI": "A.I.",
    "NVIDIA": "en-vid-ee-uh",
    "LLM": "L L M",
    "Pexels": "Peks-uls"
  }},
  "NOTE_phonetic_pronunciation_map": "Identify EVERY word in your script that might be mispronounced (tech terms, ambiguous English words, names). Provide a phonetic respelling that a standard TTS model will pronounce perfectly. Use this to correct pronunciations for any complex English words.",
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
  "original_news_url": "Exact url",
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

IMPORTANT: voice is ALWAYS en-US-AndrewNeural, which is a warm male voice. Do not suggest any other voice.
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
