from google import genai
import json
import os
from datetime import datetime
import time
from config import GEMINI_API_KEY, LOGS_DIR
from topic_tracker import load_tracker, check_story_uniqueness, check_cooldowns

def pick_and_generate_script(articles, extra_instruction="", forced_article=None, topic_type="research"):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # ── Pre-filter articles to avoid repeats ──────────────────────────────────
    filtered_articles = []
    for art in articles:
        title = art.get('title', '')
        url = art.get('url', '')
        is_unique, _ = check_story_uniqueness(title, url)
        if is_unique:
            filtered_articles.append(art)
    
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

    prompt = f"""Act as a Senior AI Investigative Journalist & Creative Director. 
Your goal is to transform technical RSS/Research data into a "Deep-Dive" YouTube Short that balances analytical rigor with high-retention visual storytelling.

CONTENT HIERARCHY (Monetization & Retention Strategy):
1. ORIGINAL COMMENTARY (35%): Why this news matters to a human audience.
2. COMPARATIVE ANALYSIS (25%): Competitive landscape (e.g., AWS vs. OpenAI).
3. THE DATA (20%): Extract specific numbers for the "Infographic" cards.
4. VISUAL DIRECTION (10%): Specify "Pattern Interrupts" every 3-5 seconds.
5. NARRATION (10%): Conversational, punchy tone. Avoid "AI-isms" like "In the rapidly evolving landscape."

NARRATIVE ARC CONFIGURATION:
1. The Identity Hook (0-5s): A shocking statement paired with a high-contrast visual "Identity" (match the first sentence).
2. The Disruptor (5-15s): The news leak or breakthrough briefly explained.
3. The 'So What?' (15-35s): The analytical core. Why is this a game changer?
4. The Visual Pivot (35-45s): A sudden shift in the script tone/visuals to reset the viewer's attention span.
5. The Prediction & The Loop (45-55s): A bold prediction that transitions perfectly back into the Hook’s first sentence for infinite replayability.

CRITICAL 'ANTI-BOT' & RETENTION RULES:
1. PATTERN INTERRUPTS: The visual_cues/retention_cues must change every 3 seconds (Zoom, Pan, B-roll switch).
2. VOICE PACING: Use em dashes (—) and ellipses (...) for TTS humanization.
3. THE LOOP: The final 3 words must phonetically or logically lead back into the first 3 words of the Hook.
4. PROOF OF HUMANITY: End with a unique 'Fact of the Day' at the very end of the script that is completely UNRELATED to the main topic.
5. TEMPORAL ADAPTATION: 
   - If topic_type is 'research': Focus on "First Principles" and engineering breakthroughs.
   - If topic_type is 'tools': Focus on "UI/UX" and immediate productivity gains.

{selection_instruction}

RESEARCH PAPERS & BLOGS DATA:
{news_context}

NARRATIVE FLOW (FOR THE 'SCRIPT' FIELD):
- Hook -> Disruptor -> So What? -> Visual Pivot -> Prediction/Loop -> Fact of the Day.

{extra_instruction}

Return ONLY this exact JSON (no markdown, no explanation) to securely match the automation pipeline:
{{
  "title_options": ["Title idea 1", "Title idea 2", "Title idea 3"],
  "description": "Full 100+ word rich SEO description for youtube describing the video, including timestamps and credits.",
  "fact_of_the_day": "Unrelated fact here",
  "quiz_tone": "Investigative",
  "title": "Punchy YouTube title max 60 chars",
  "script": "Full voiceover script following the Arc (65-75 sec), ending with the fact_of_the_day. Ensure The Loop is implemented.",
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
      "text": "The first sentence",
      "start": 0.00,
      "end": 2.40,
      "highlight_word": "First",
      "has_infographic": false
    }}
  ],
  "original_news_headline": "Exact headline",
  "original_news_url": "Exact url"
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
            print(f"Gemini generation error: {e}")
            attempts += 1
            time.sleep(2)
            
    return None
