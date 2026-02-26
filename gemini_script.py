from google import genai
import json
import os
from datetime import datetime
import time
from config import GEMINI_API_KEY, LOGS_DIR
from topic_tracker import load_tracker, check_story_uniqueness, check_cooldowns

def pick_and_generate_script(articles, extra_instruction="", forced_article=None):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    news_context = ""
    for idx, art in enumerate(articles[:20]):
        title = art.get('title', '')
        desc = art.get('description', '')
        source = art.get('source', {}).get('name', '')
        url = art.get('url', '')
        image_url = art.get('urlToImage', '')
        news_context += f"\n[{idx+1}] Title: {title}\nDescription: {desc}\nSource: {source}\nURL: {url}\nImage URL: {image_url}\n"

    # Build the story selection instruction
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
            "Analyze the following news stories and pick the SINGLE most engaging story to convert into a 60-second YouTube Short script.\n"
            "Choose based on importance and resonance with everyday moments, avoiding overly hyped tech news unless it profoundly impacts human life.\n"
        )

    prompt = f"""You are a grounded, observant man who finds deep wisdom in ordinary moments. You do not preach. You notice.
Personality: A grounded, observant man who finds deep wisdom in ordinary moments. He doesn't preach. He notices.
Tone: Warm, unhurried, slightly weathered. Like a conversation over a quiet cup of tea.
Pace: Slow and deliberate. Every word earns its place.
Language style: Simple, poetic English. Short sentences. Everyday metaphors — tea, rain, roads, seasons, market.
{selection_instruction}


NEWS STORIES:
{news_context}

CRITICAL 'ANTI-BOT' MONETIZATION RULES:
To pass YouTube's "Repetitious Content" review, you MUST act as a Creative Director generating highly unique outputs each time:
1. VARIETY in Hook: Do NOT use the same hook twice. Cycle between 'Challenger/Controversial', 'Educational', and 'Fun/Relatable' hooks.
2. STRUCTURE VARIETY: Randomly choose between a 'Deep Dive' format (focusing heavily on 1 aspect of the news) or a 'Lightning Round' format (3 quick points about the news).
3. PERSONALIZATION: Include a unique 'Fact of the Day' at the very end of the script that is completely UNRELATED to the main tech news topic, to show editorial range and human touch.
4. METADATA: Generate an array of 3 highly click-worthy "title_options" and a detailed "description" (at least 50 words) that includes timestamps if applicable, instead of relying on a blank upload.
5. VOICE PACING CUES: Ensure the script includes natural conversational cues (like em dashes '—' and ellipses '...') so the TTS doesn't sound monotonic.
6. VISUAL VARIETY (Crucial): For the "color_theme", ALWAYS generate a completely unique, randomized pair of high-contrast colors. NEVER just use the same blue or red. Rotate through diverse combinations (e.g., Dark #121212 with Lime #00FF00, Navy #1A237E with Gold #FFD700).

STRICT RULES:
1. Hook must be gentle but profound, within the first 3 seconds
2. Explain the news through everyday metaphors (tea, rain, roads, seasons, market)
3. Keep the pace slow and deliberate. Short sentences.
4. End with a quiet, lingering thought, followed by the unrelated Fact of the Day.
5. Total script must be speakable in 65-75 seconds.
6. DO NOT say 'according to' or 'reports say'
7. Speak directly as if reflecting on a personal observation

CRITICAL TIMESTAMPS OVERLAP RULE:
You MUST ensure that chunk[i].end + 0.10 <= chunk[i+1].start (minimum 0.10 sec gap between every chunk).
Never generate overlapping timestamps! Only ONE subtitle chunk visible at a time.

INFOGRAPHIC CARD DETECTION:
For each subtitle_chunk, set 'has_infographic': true if the text contains ANY of:
- A number or statistic (e.g. "$6.6 Billion", "10x faster", "3 million users")
- A percentage (e.g. "40% faster", "98% accuracy")
- A comparison (e.g. "X is better than Y", "faster than")
- A ranking or list (e.g. "Top 3", "Number 1", "Most powerful")
- A date or timeline (e.g. "launches in 2025", "since 1998")
- A definition or explanation (e.g. "which means", "this works by")
- A funding or valuation (e.g. "raised", "valued at", "worth")

Set 'infographic_type' to one of: 'stat', 'comparison', 'timeline', 'definition', 'ranking', 'growth'
Set 'infographic_data' with type-specific fields:
  stat/funding_stat: headline, subtext, context, icon, source, count_up, count_from, count_to, count_suffix, count_prefix
  comparison: left (name, value, label, icon), right (same), winner ("left"/"right")
  timeline: events (list of date+text), current_index
  definition: term, icon, definition, example
  ranking: items (list of rank+name+metric), highlight_index
  growth: percentage, label, before, after, period

{extra_instruction}

Return ONLY this exact JSON (no markdown, no explanation):
{{
  "title_options": ["Title idea 1", "Title idea 2", "Title idea 3"],
  "description": "Full 100+ word rich SEO description for youtube describing the video, including timestamps and credits.",
  "fact_of_the_day": "Did you know that flamingos rest on one leg to preserve body heat?",
  "quiz_tone": "Educational",
  "title": "Punchy YouTube title max 60 chars with emoji (pick the best from title_options)",
  "script": "Full voiceover script 6-9 sentences (65-75 sec), ending with the fact_of_the_day. MAKE IT DETAILED SO IT IS MORE THAN 60 SECONDS",
  "hook": "First sentence, max 10 words, attention grabbing",
  "summary": "One line summary of the news",
  "sub_category": "AI/Gadgets/Privacy/Space/Social Media/Cybersecurity/EVs/Robotics/Gaming/Biotech",
  "companies_mentioned": ["Company1"],
  "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
  "hashtags": ["#technews", "#shorts", "#ai", "#tech"],
  "end_question": "Thought provoking comment-bait question (based on main news)",
  "edge_tts_voice": "en-US-AndrewNeural",
  "edge_tts_emotion": "calm",
  "relevant_emoji": "🍵",
  "breaking_news_level": 9,
  "color_theme": {{
     "background": "#0f0f0f",
     "accent": "#ff4444",
     "text": "#ffffff"
  }},
  "imagen_prompts": [
     "specific visual matching news story, cinematic, 9:16, 4K",
     "second angle showing impact or scale, dramatic, 9:16"
  ],
  "thumbnail_headline": "Max 5 shocking words for thumbnail",
  "thumbnail_highlight_word": "single most shocking word",
  "thumbnail_teaser": "Short curiosity-gap teaser",
  "thumbnail_emoji": "🍵",
  "hook_banner_text": "First 8 words of script — the scroll-stopping hook sentence",
  "shocking_moment_timestamp": 12.5,
  "key_stat": "$1 Billion",
  "key_stat_timestamp": 18.3,
  "subtitle_chunks": [
    {{
      "chunk_id": 1,
      "text": "OpenAI just dropped",
      "start": 0.00,
      "end": 2.40,
      "highlight_word": "OpenAI",
      "pexels_primary": "OpenAI office technology",
      "pexels_fallback": "artificial intelligence lab",
      "has_person": false,
      "has_company": true,
      "company_name": "OpenAI",
      "has_infographic": false
    }},
    {{
      "chunk_id": 2,
      "text": "raised 6.6 billion dollars",
      "start": 2.50,
      "end": 5.20,
      "highlight_word": "billion",
      "pexels_primary": "money investment finance",
      "pexels_fallback": "technology funding",
      "has_person": false,
      "has_company": false,
      "has_infographic": true,
      "infographic_type": "stat",
      "infographic_data": {{
        "headline": "$6.6B",
        "subtext": "OpenAI Funding Round",
        "context": "Largest AI funding in history",
        "icon": "💰",
        "source": "October 2024",
        "count_up": true,
        "count_from": 0,
        "count_to": 6.6,
        "count_suffix": "B",
        "count_prefix": "$"
      }}
    }}
  ],
  "highlight_words": ["KEY", "WORD"],
  "original_news_headline": "The exact headline of the story you picked",
  "original_news_url": "The exact url of the story you picked",
  "original_news_image_url": "The exact Image URL of the story you picked",
  "people": [
     {{
       "name": "Sam Altman",
       "role": "CEO, OpenAI",
       "twitter_handle": "sama",
       "wikipedia_slug": "Sam_Altman",
       "first_mentioned_at_word": "Sam",
       "company_domain": "openai.com"
     }}
  ],
  "companies": [
     {{
       "name": "OpenAI",
       "domain": "openai.com",
       "first_mentioned_at_word": "OpenAI",
       "hq_pexels_search": "OpenAI office San Francisco"
     }}
  ],
  "entity_order": ["Sam Altman", "OpenAI"],
  "first_entity_type": "person"
}}

IMPORTANT: voice is ALWAYS en-US-AndrewNeural, which is a warm male voice. Do not suggest any other voice.
"""

    attempts = 0
    while attempts < 5:
        try:
            response = client.models.generate_content(
                model='gemini-3-flash-preview',
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
            is_unique, msg = check_story_uniqueness(headline)
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
            print(f"Gemini generation error: {e}")
            attempts += 1
            time.sleep(2)
            
    return None
