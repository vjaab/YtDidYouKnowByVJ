from google import genai
import json
import os
from datetime import datetime
import time
from config import GEMINI_API_KEY, LOGS_DIR
from topic_tracker import load_tracker, check_story_uniqueness, check_cooldowns
from ecosystem_logic import get_slot_info, get_category_prompt_enhancement

def pick_and_generate_script(articles, extra_instruction="", forced_article=None, topic_type="research"):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
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

CONTENT HIERARCHY (Monetization & Retention Strategy):
1. ORIGINAL COMMENTARY (35%): Why this news matters to a human audience.
2. COMPARATIVE ANALYSIS (25%): Competitive landscape (e.g., AWS vs. OpenAI).
3. THE DATA (20%): Extract specific numbers for the "Infographic" cards.
4. VISUAL DIRECTION (10%): Specify "Pattern Interrupts" every 3-5 seconds (glitch, zoom, or split-screen).
5. NARRATION (10%): Humanized, Conversational tone (Gen-Z/Millennial "Vlog" style). Use human imperfections like ('...uhm', 'wait...', 'listen, honestly') to break the robot feel. These act as "Human-Glitch" hooks to grab attention. Avoid textbook grammar; use sentence fragments.

NARRATIVE ARC CONFIGURATION:
1. The 2026 Identity Hook (0-5s): A shocking human-voiced statement. If generating a QUIZ, do not reveal the answer yet.
2. The Disruptor (5-15s): The news leak or tool reveal.
3. The 'So What?' (15-35s): The analytical core. Why is this a game changer?
4. The Visual Pivot (35-45s): A sudden shift in the script tone/visuals to reset the viewer's attention.
5. The Prediction & The Infinite Loop (45-55s): Since Shorts loop, the last 3 words must logically or phonetically FLOW back to the first 3 words of the script.

CRITICAL '2026 SCALE' RULES:
1. THE INFINITE LOOP: {{"end_fragment": "and that's why...", "start_fragment": "...AI news is moving fast."}}
   You MUST write the final sentence of the script so it leads perfectly back to the first sentence of the Hook. 
   Result: The viewer watches it 1.5 times before they realize it looped.
2. PATTERN INTERRUPTS: Every 3 seconds, a change in visual state is MANDATORY (Zoom-in, Glitch transition, or Reaction Split-screen).
3. COMMENT BAITING: If this is an AI QUIZ, NEVER reveal the answer until the last 5 seconds. Explicitly tell users to 'Pause and comment your guess now!'.
4. VOCAL PACING & EMPHASIS: You must know exactly where to pause for dramatic effect. Constantly use ellipses (...) and em-dashes (—) to force the AI voice to pause. For the most important words that need audio emphasis, WRITE THEM IN ALL CAPS. 
   CRITICAL: NEVER write bracketed instructions like '[pause]', '(silence)', or '[intense music]'. Do NOT write word descriptions of sounds. Only use punctuation (... or —) to create pauses.
5. THE HUMAN-GLITCH: Include subtle breaths or conversational 'actually...' to trigger the 'Authenticity' flag. Avoid robotic lists.

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
