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
            "Choose based on:\n"
            "- Shock factor / surprise element\n"
            "- Relevance to general audience (not just developers)\n"
            "- Viral potential\n"
            "- Recency (breaking news preferred)\n"
            "- Avoid: funding rounds, stock prices, earnings reports\n"
            "- Prefer: AI breakthroughs, gadgets, privacy scandals, space tech, robot news, social media drama, scientific discoveries, product launches"
        )

    prompt = f"""You are a viral YouTube Shorts script writer specializing in tech news. 
{selection_instruction}


NEWS STORIES:
{news_context}

STRICT RULES:
1. Hook must grab attention in first 3 seconds
2. Explain WHY this matters to everyday people
3. Use simple language, no heavy jargon
4. Add surprising or shocking angle if possible
5. End with a thought-provoking question to boost comments
6. Total script must be speakable in 30-55 seconds
7. Tone: Excited, urgent, conversational
8. DO NOT mention the source website name
9. DO NOT say 'according to' or 'reports say'
10. Speak directly as if YOU discovered this news

{extra_instruction}

Return ONLY this exact JSON (no markdown, no explanation):
{{
  "title": "Punchy YouTube title max 60 chars with emoji",
  "script": "Full voiceover script 4-6 sentences, 35-55 sec",
  "hook": "First sentence, max 10 words, attention grabbing",
  "summary": "One line summary of the news",
  "sub_category": "AI/Gadgets/Privacy/Space/Social Media/Cybersecurity/EVs/Robotics/Gaming/Biotech",
  "companies_mentioned": ["Company1"],
  "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
  "hashtags": ["#technews", "#shorts", "#ai", "#tech"],
  "end_question": "Thought provoking comment-bait question",
  "edge_tts_voice": "en-US-GuyNeural",
  "edge_tts_emotion": "newscast",
  "relevant_emoji": "🤖",
  "breaking_news_level": 9,
  "color_theme": {{
     "background": "#0f0f0f",
     "accent": "#ff4444",
     "text": "#ffffff"
  }},
  "imagen_prompts": [
     "specific visual matching news story, cinematic, 9:16, 4K",
     "second angle showing impact or scale, dramatic, 9:16",
     "third visual showing technology or product, sharp, 9:16",
     "fourth atmospheric establishing shot, 9:16"
  ],
  "thumbnail_headline": "Max 5 shocking words for thumbnail (e.g. 'AI Just Killed Search')",
  "thumbnail_highlight_word": "single most shocking word from thumbnail_headline",
  "thumbnail_teaser": "Short curiosity-gap teaser (e.g. 'Nobody is talking about this')",
  "thumbnail_emoji": "🤖",
  "hook_banner_text": "First 8 words of script — the scroll-stopping hook sentence",
  "shocking_moment_timestamp": 12.5,
  "key_stat": "$1 Billion",
  "key_stat_timestamp": 18.3,
  "highlight_words": ["KEY", "STAT", "WORD"],
  "original_news_headline": "The exact headline of the story you picked",
  "original_news_url": "The exact url of the story you picked",
  "original_news_image_url": "The exact Image URL of the story you picked"
}}

IMPORTANT: voice is ALWAYS en-US-GuyNeural. Do not suggest any other voice.
"""

    attempts = 0
    while attempts < 5:
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
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
