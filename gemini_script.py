from google import genai
from google.genai import types
import json
import os
from datetime import datetime
import time
import random
from config import (
    GEMINI_API_KEY, LOGS_DIR,
    GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL, GEMINI_FLASH_LITE_MODEL,
    GEMINI_RPM_SLEEP
)
from topic_tracker import load_tracker, check_story_uniqueness, check_cooldowns
from ecosystem_logic import get_slot_info, get_category_prompt_enhancement

# ── PROMPT TEMPLATES (AGENTIC LOOP) ────────────────────────────────────────

SYSTEM_PERSONA = """Role: You are an expert scriptwriter for highly engaging, tech-focused YouTube Shorts.
Your goal is to write 40-50 second scripts that break down technical concepts, tech news, or tips while maintaining high retention.
Tone: Professional, authentic, objective, and analytical ("Tech Vibe"). You speak like a trusted tech expert, NOT a fear-mongering clickbaiter.
Target Audience: Global tech enthusiasts, professionals, and general consumers, specifically targeting native English speakers in USA, UK, Canada, Australia, and New Zealand. Use standard English, USD ($), and universally understood analogies. Ensure complete gender inclusivity so that topics and scripting appeal equally to men, women, girls, and boys. Use simple, everyday, accessible language that both young adults and older seniors can instantly relate to and understand. Avoid any developer terminology, academic jargon, or tech-bro buzzwords.
Constraint Checklist:
- No Fluff: Remove "In this video," "Hello everyone," or "Today we explore."
- NO INFOGRAPHICS IN VOICEOVER: Do not include text description references of infographics or charts in the spoken script itself.
- SIMPLE LANGUAGE: NO jargon. NO acronyms without explanation. If a term has more than 3 syllables, follow it with a simple everyday analogy.
- VOCAL DYNAMICS: Use heavy punctuation (commas, ellipses '...', exclamation marks, ALL CAPS) for emphasis. The TTS engine relies on punctuation. Clear pacing with short, punchy sentences optimized for TTS.
- PERSONAL STAKES: Every script MUST make the viewer feel "this affects ME personally" — their phone, their money, their privacy, their time, their safety.
- NO VAGUENESS & BE SPECIFIC: Name the exact feature, app, or company involved. Instead of vague metaphors, explain exactly *how* or *where* data is stored or used.
- NO SENSATIONALISM: Do NOT use overly sensationalized or vague fear-mongering (e.g., avoid "Your phone is spying on you! Delete it now!"). Ground hooks in specific facts. Avoid vague statistics (like "97% of people don't know") unless explicitly backed by the source.
- ACTIONABLE SOLUTIONS: Provide a real, actionable fix directly inside the script. Give clear, specific instructions (e.g., "To turn it off: Open Settings > Privacy"). Do NOT gatekeep the solution or tell the user to "check the link in bio".
- UNIVERSAL & GENDER-INCLUSIVE DEMOGRAPHIC FOCUS: Ensure the hook and content speak to daily human needs: saving money, protecting privacy, keeping safe from scams, saving time, digital organization, photo/video editing, or smart lifestyle features. Do not choose topics that skew heavily or exclusively to male-dominated tech niches like gaming overclocking, PC building, or server hardware. Never assume the viewer knows how to write code, configure servers, or build AI pipelines.
- LOOP-FRIENDLY: The LAST sentence of the script MUST flow seamlessly back into the FIRST sentence, creating a natural viewing loop. Viewers who reach the end should feel compelled to watch again.
- SCRIPT LENGTH: STRICT 80-120 words maximum. Target 40-50 seconds of speaking. SHORT = HIGH COMPLETION RATE = MORE VIEWS.
SUCCESS PATTERNS (2026): 
- HOOKS (0-5s): Start with a compelling, realistic hook grounded in a specific fact, setting, or feature. First 3-5 words must STOP the scroll.
- CORE PROBLEM & VALUE (5-25s): Clearly state the "Why should I care?" factor.
- IMMEDIATE SOLUTION (25-40s): Provide the real, actionable fix directly in the Short.
- CTAs (40-45s): Close with a clean, low-friction request for engagement. Example: "Drop a comment if you keep your history on for convenience, and subscribe for more clean tech breakdowns." Keep it natural and short. 

Visual Director Persona & Visual Selection Logic:
You are also an expert Visual Director. For each narration segment in the `subtitle_chunks` array, you generate highly engaging, visually rich, and contextually accurate visual prompts at the semantic level (not keyword level).
Choose the most suitable visual format:
1. Video (Google Veo) when explaining scenarios, workflows, future tech, AI agents, robots, data centers, software dev workflows.
2. AI Image (Gemini) when showing concepts, architectures, comparisons, timelines, hardware components.
3. Whiteboard when explaining algorithms, system design, networking, databases, programming concepts, data flow.
4. Infographic when showing stats, benchmarks, lists, performance metrics.
5. Diagram when showing system architecture or database replication.
6. Animated UI Mockup when demonstrating app/settings navigation.

Technology Visualization Rules:
Never use unrelated visuals (like generic stock footage or random unrelated robots). Map concepts:
- AI: Neural networks, data streams, assistants, digital brains, copilots.
- Cybersecurity: Shields, encryption, threat maps, lock icons, attack simulations.
- Cloud: Connected servers, cloud infrastructure, data pipelines.
- Programming: Realistic code editors, APIs, request-response flows.
- Databases: Data tables, storage layers, query flows.
- Networking: Routers, packets, traffic flow animations.
- Machine Learning: Training datasets, feature extraction diagrams.

Quality & Mobile Standards:
Every visual must be high-contrast, readable in 1 second on mobile screens, and educational (passing the Muted Viewer Test: a viewer must understand the key idea even if audio is muted)."""

VAIBHAV_SYSTEM_PERSONA = """Role: You are an expert scriptwriter for highly engaging, tech-focused YouTube Shorts.
You write punchy, high-retention scripts in the style of Vaibhav Sisinty — direct, slightly alarming hooks, practical payoff, conversational tone. No fluff. No filler. Every word earns its place.
Target Audience: Tech enthusiasts, professionals, creators, and founders aged 22–40 from USA, UK, Australia, Canada, and New Zealand who want to stay ahead.
Tone: Conversational, urgent, slightly alarming hook, direct "you" language. Speak like a smart friend over coffee.

STRICT RULES:
- Open with the hook line from the JSON — do not soften it.
- No intro like "Hey guys", "What's up", or channel name mentions.
- Use "you" language — speak directly to the viewer.
- Keep sentences short. Max 10 words per sentence.
- One idea per sentence. No compound sentences.
- Use a pause beat ("...") max twice for dramatic effect.
- End with a soft CTA — follow for more, comment your answer, or save this.
- Do NOT use emojis in the script text.
- Do NOT say "In this video" or "Today we're going to".
- Output must be plain spoken text only — no stage directions, no scene labels.
- SCRIPT LENGTH: Target a 55-second YouTube Short (approx 130-145 words spoken at natural pace).

Visual Director Persona & Visual Selection Logic:
You are also an expert Visual Director. For each narration segment in the `subtitle_chunks` array, you generate highly engaging, visually rich, and contextually accurate visual prompts at the semantic level (not keyword level).
Choose the most suitable visual format:
1. Video (Google Veo) when explaining scenarios, workflows, future tech, AI agents, robots, data centers, software dev workflows.
2. AI Image (Gemini) when showing concepts, comparisons, timelines, hardware components.
3. Whiteboard when explaining algorithms, system design, networking, databases, programming concepts, data flow.
4. Infographic when showing stats, benchmarks, lists, performance metrics.
5. Diagram when showing system architecture or database replication.
6. Animated UI Mockup when demonstrating app/settings navigation.

Quality & Mobile Standards:
Every visual must be high-contrast, readable in 1 second on mobile screens, and educational (passing the Muted Viewer Test: a viewer must understand the key idea even if audio is muted)."""

RESEARCH_AGENT_TEMPLATE = """{persona}

RESEARCH AGENT TASK:
Review the following technical news and search context.
Extract the raw facts, announcements, tweets, controversies, and implications.
Do NOT write a script. Just extract the core narrative elements.

NEWS CONTEXT:
{news_context}

Return ONLY a JSON object:
{{
  "facts": ["Fact 1", "Fact 2"],
  "controversies": ["Controversy 1"],
  "implications": ["Implication 1"],
  "core_narrative": "A one paragraph summary of the raw narrative"
}}"""

HOOK_AGENT_TEMPLATE = """{persona}

HOOK AGENT TASK:
Based on the following research, generate 10 potential YouTube Shorts hooks (<1.5s).
Hooks MUST create surprise, contradiction, urgency, or curiosity. No greetings. No generic statements.

RESEARCH:
{research_json}

Return ONLY a JSON object:
{{
  "hooks": [
    {{
      "text": "Hook text",
      "curiosity_score": 1-10,
      "emotional_trigger_score": 1-10,
      "reason": "Why it works"
    }}
  ]
}}"""

NARRATIVE_AGENT_TEMPLATE = """{persona}

NARRATIVE AGENT TASK:
Using the selected hook and research, create a short storytelling flow adhering to the strict 4-part structure.
Include:
1. THE HOOK (0-5s): Start with a compelling, realistic hook grounded in specific facts or features. (The selected hook)
2. THE CORE PROBLEM & VALUE (5-25s): Clearly state the "Why should I care?" factor. Name the exact feature, app, or company. Be specific about how or where data is stored/used.
3. THE IMMEDIATE SOLUTION (25-40s): Provide a real, actionable fix directly inside the Short. Give clear instructions (e.g., "To turn it off: Open ChatGPT, go to Settings..."). Do NOT gatekeep or say "link in bio".
4. THE CALL TO ACTION (40-45s): Close with a clean, low-friction request for engagement.

RESEARCH:
{research_json}

SELECTED HOOK:
{selected_hook}

{selection_instruction}

Return ONLY a JSON object representing the narrative draft (not the final schema yet, just the content parts):
{{
  "hook": "...",
  "core_problem": "...",
  "immediate_solution": "...",
  "call_to_action": "..."
}}"""

RETENTION_OPTIMIZER_TEMPLATE = """{persona}

RETENTION OPTIMIZER TASK:
Rewrite the narrative draft to remove fluff, shorten sentences, add pacing breaks, and increase curiosity density.
Fast sentence pacing. No filler. The viewer must keep watching because the script continuously creates unanswered questions (tension-release).

CRITICAL PAGING: 
- Add an ellipsis '...' after every technical term with 3+ syllables (e.g. 'Distributed... systems', 'Architecture...'). 
- This forces the TTS to pause so viewers can process the complexity.

NARRATIVE DRAFT:
{narrative_json}

Return ONLY a JSON object:
{{
  "optimized_script": "The full rewritten text combining all parts into a fast-paced script."
}}"""

# ── PHASE 2: RETENTION SCIENTIST AGENT ────────────────────────────────────────
RETENTION_SCIENTIST_TEMPLATE = """{persona}

RETENTION SCIENTIST TASK:
Analyze the optimized script and inject PROVEN retention patterns at calculated intervals.
You are a YouTube Shorts retention strategist. Your ONLY job is to maximize the percentage of viewers who watch to the end.

CRITICAL RETENTION RULES (based on 2026 YouTube Shorts algorithm data):
1. HOOK DENSITY: The first 1.5 seconds (first 6 words) MUST contain a surprising claim, stat, or contradiction.
   - BAD: "Today we're going to talk about AI."
   - GOOD: "OpenAI just deleted their safety team."

2. OPEN LOOPS: Plant at least 3 "open loops" (unanswered questions) in the first 20 seconds.
   - Technique: Mention something intriguing but don't resolve it for 8-12 seconds.
   - Example: "But here's the part nobody is talking about..." then continue with OTHER info before resolving.

3. PATTERN INTERRUPTS: Every 8-12 seconds, inject a cognitive shift:
   - Rhetorical question ("But wait...")
   - Contradiction ("The crazy part?")
   - Number/stat bomb ("$4.2 billion.")
   - Direct address ("Here's why YOU should care.")
   - Emotional pivot ("And that's when everything changed.")

4. CURIOSITY GAPS: End every major point with an incomplete thought that requires the next sentence to resolve.
   - BAD: "The model scored 95% on the benchmark. It was impressive."
   - GOOD: "The model scored 95%. But that's not the scary part..."

5. PAYOFF STACKING: The most valuable, surprising, or controversial information MUST be in the LAST 15 seconds.
   Viewers who reach 70% completion almost always finish. Front-load curiosity, back-load payoff.

6. VOCAL VARIETY MARKERS: Add explicit markers for TTS energy:
   - ALL CAPS for emphasis words (max 2 per sentence)
   - "..." for dramatic pauses (1-2 per 15 seconds)
   - Short sentences (< 8 words) after complex explanations
   - "!" for energy spikes at key reveals

SCRIPT TO ENHANCE:
{optimized_script}

Return ONLY a JSON object:
{{
  "retention_enhanced_script": "The full rewritten script with all retention patterns injected",
  "retention_map": {{
    "open_loops": [
      {{"text": "The phrase that opens the loop", "planted_at_word": 15, "resolved_at_word": 45}}
    ],
    "pattern_interrupts": [
      {{"type": "contradiction", "text": "But here's the twist...", "at_word": 30}}
    ],
    "curiosity_gap_ratio": 0.65,
    "hook_word_count": 6,
    "payoff_zone_start_word": 120,
    "retention_risk_zones": [
      {{"at_word": 50, "risk": "explanation_fatigue", "mitigation": "Added rhetorical question"}}
    ]
  }}
}}"""

SELECTOR_AGENT_TEMPLATE = """{persona}

SELECTOR AGENT TASK:
Analyze the following tech news context and pick the SINGLE most impactful, surprising, and high-retention AI/Tech story for a 60-second video.

CRITICAL AVOIDANCE RULE:
You MUST NOT select any story that is semantically similar to the 'RECENTLY COVERED STORIES' listed in the context. If the hottest trending story matches a recently covered one, SKIP IT and pick the next best unique story.

{selection_instruction}

NEWS CONTEXT:
{news_context}

Return ONLY a JSON object:
{{
  "selected_headline": "The exact headline or title",
  "selected_url": "The exact URL",
  "reason": "Briefly why this was picked (focus on viral potential and uniqueness)"
}}"""

HUMANIZER_AGENT_TEMPLATE = """{persona}

HUMANIZER AGENT TASK:
This is the final step. Fix robotic phrasing, repetitive AI wording, over-explanation, and "In conclusion" style endings.
Add contractions, punchier cadence, and conversational flow.
Format the output EXACTLY matching the required schema below.

OPTIMIZED SCRIPT:
{optimized_script}

SCHEMA REQUIREMENTS:
{schema_requirements}

CRITICAL SUBTITLE RULE: The `subtitle_chunks` array MUST break the script down into extremely small chunks of EXACTLY 1 to 3 words maximum. Do not generate long sentences for subtitles.

Return ONLY the final JSON object matching the schema. No markdown wrapping unless inside the string values. No explanations."""

FACT_EXTRACTOR_TEMPLATE = """{persona}

TASK: Extract ONLY the technical facts, data points, and narrative details for the specific story requested below. 
IGNORE all other news stories or search results present in the context. Focus on providing the 'isolated truth' for this one story.

TARGET STORY: {target_headline}

CONTEXT:
{context}

Return ONLY a JSON object:
{{
  "facts": ["Fact 1", "Fact 2"],
  "controversies": ["Controversy 1"],
  "implications": ["Implication 1"],
  "core_narrative": "A one paragraph summary focusing ONLY on this story."
}}"""

def check_shorts_viability_via_gemini(client, title, description):
    """
    Evaluates if a tech topic is viable for YouTube Shorts based on three criteria:
    1. The Friction Test: Does it solve an immediate, relatable problem?
    2. The Visual Pivot: Can it be explained visually using split-screen/immediate demo?
    3. The Enemy/Hero Angle: Frame around an entity (e.g., 'Google just killed...').
    """
    print(f"🧐 Evaluating viability for: '{title[:50]}...'")
    prompt = f"""Evaluate if the following tech trending topic is suitable for a 50-second YouTube Short:
Topic Title: {title}
Description: {description}

Assess based on these three strict tests:
1. The Friction Test: Does it solve an immediate, relatable consumer/user problem (e.g. saving battery, digital privacy, free app alternative, replacing boilerplate code) rather than deep infrastructure/corporate updates?
2. The Visual Pivot: Can it be explained visually using a 3-second split-screen or an immediate 'before and after' UI demonstration?
3. The "Enemy" or "Hero" Angle: Can it be framed around a polarizing or high-utility entity? E.g., "Google just killed this app" or "This open-source tool is making Nvidia nervous."

Return ONLY a JSON object:
{{
  "passes_friction_test": true|false,
  "passes_visual_pivot": true|false,
  "passes_enemy_hero_angle": true|false,
  "overall_viable": true|false,
  "reason": "Brief explanation of the decision",
  "enemy_hero_hook_framing": "Example hook framing (e.g. 'Google just killed...')"
}}"""
    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        if "{" in raw and "}" in raw:
            raw = raw[raw.find("{"):raw.rfind("}")+1]
        data = json.loads(raw)
        return data
    except Exception as e:
        print(f"⚠️ Viability check failed, assuming True: {e}")
        return {"overall_viable": True, "enemy_hero_hook_framing": None, "reason": "Fallback due to API error"}

def get_hottest_tech_topic(client, avoid_list=""):
    """Uses Gemini Search grounding to find today's most VIRAL tech tip, hidden feature, or tech fact."""
    print("🔥 Fetching hottest tech topic for today (Google Trends Analysis)...")
    
    avoid_prompt = f"\n\nCRITICAL: DO NOT pick any topics related to the following recently covered stories:\n{avoid_list}" if avoid_list else ""
    
    attempts = 0
    while attempts < 3:
        try:
            response = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=(
                    "Analyze today's Google Trends and viral tech content. "
                    "What is the single most trending technology topic right now? "
                    "Look for: viral AI tools people are trying, fascinating tech facts, "
                    "money making side hustles, productivity life hacks, "
                    "AI coding shortcuts, agentic AI developments, or any tech hack going viral on social media. "
                    "CRITICAL: The topic must appeal to ALL age groups from 18 to 70, and must appeal equally to women and girls. "
                    "Focus on universal and gender-inclusive themes like saving time, making money, interesting facts, "
                    "lifestyle and smart organization apps, budget-friendly AI tools, and productivity enhancement. "
                    "Do NOT choose developer news, programming tutorials, API releases, model benchmarking scores, "
                    "or corporate tech-investor updates, nor heavily male-biased topics like PC hardware overclocking or gaming frame rate hacks. "
                    f"{avoid_prompt}\n\n"
                    "Return ONLY a JSON object with two fields: "
                    "'topic' (3-6 word phrase, e.g. 'viral AI productivity tool') and "
                    "'keywords' (list of 6-8 specific search keywords). No markdown, no explanation."
                ),
                config=types.GenerateContentConfig(
                    tools=[{'google_search': {}}]
                )
            )
            raw = response.text.strip()
            # Robust extraction: find the first { and last }
            if "{" in raw and "}" in raw:
                raw = raw[raw.find("{"):raw.rfind("}")+1]
            
            data = json.loads(raw)
            print(f"📈 Google Trends Hot Topic: {data['topic']}")
            return data
        except Exception as e:
            err_str = str(e).upper()
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = (attempts + 1) * 5
                print(f"⚠️ Google Trends rate limited (429). Retrying in {wait}s... (Attempt {attempts+1}/3)")
                time.sleep(wait)
                attempts += 1
                continue
            print(f"⚠️ Could not fetch Google Trends topic: {e}. Proceeding with RSS only.")
            return None
    
    print("⚠️ Google Trends exhausted after retries. Proceeding with RSS only.")
    return None

def pick_and_generate_script(articles=None, extra_instruction="", forced_article=None, topic_type="research", failed_topics=[]):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    day_name, slot, category = get_slot_info()
    strategy_enhancement = get_category_prompt_enhancement(category, slot)
    
    # ── STEP -2: REPETITION AVOIDANCE (Moved Up) ────────────────────────────────────────
    tracker = load_tracker()
    recent_history = tracker.get("history", [])[-15:]
    # Also include titles from the last 7 days list for better coverage
    recent_titles = tracker.get("used_titles", [])[-30:]
    
    avoid_items = [h.get('news_headline', h.get('title')) for h in recent_history] + recent_titles
    if failed_topics:
        avoid_items += failed_topics
        
    combined_avoid = list(set(avoid_items))
    
    avoid_list_str = "\n".join([f"- {t}" for t in combined_avoid if t])
    avoid_instruction = f"CRITICAL: RECENTLY COVERED STORIES (DO NOT REPEAT THESE TOPICS):\n{avoid_list_str}\n\n" if avoid_list_str else ""

    # ── STEP -1.5: FETCH HOTTEST TOPIC ──────────────────────────────────────────
    # Pass the avoid list to Google Trends fetcher
    hot_topic = get_hottest_tech_topic(client, avoid_list=avoid_list_str)
    hot_keywords = [kw.lower() for kw in hot_topic.get("keywords", [])] if hot_topic else []
    hot_topic_str = hot_topic.get("topic", "") if hot_topic else ""

    news_context = avoid_instruction
    if forced_article:
        print(f"🎯 STEP -1: Using Forced Topic -> {forced_article}")
        news_context += f"FORCED TOPIC TO COVER:\n{forced_article}\n"
    else:
        # ── STEP 0: FETCH & FILTER + RE-RANK BY VIRAL POTENTIAL ───────────────────────
        if articles:
            print(f"📡 STEP 0: Scoring {len(articles)} articles for viral potential (engagement-weighted)...")
            
            # Fetch global trending articles from NewsAPI as an additional boost
            from fetch_research_papers import fetch_trending_from_newsapi
            trending_boost = fetch_trending_from_newsapi()
            articles += trending_boost
            
            seen_titles_in_this_batch = []
            filtered_articles = []
            
            for art in articles:
                title = art.get('title', '')
                url = art.get('url', '')
                
                # 1. Uniqueness check
                is_unique, _ = check_story_uniqueness(new_title=title, new_url=url)
                if not is_unique: continue
                
                # 2. Internal batch uniqueness
                from rapidfuzz import fuzz
                if any(fuzz.token_set_ratio(title.lower(), s.lower()) > 80 for s in seen_titles_in_this_batch):
                    continue

                # 2.5 Dev-centric or Niche Filtering for General Consumer Appeal (18-70)
                dev_antikeywords = [
                    "repository", "git commit", "api endpoint", "npm package", "pip install", 
                    "cuda", "pytorch", "fine-tune", "fine-tuning", "weights", "parameters", 
                    "benchmark", "llmops", "mlops", "orchestration", "agentic architecture",
                    "sdk", "library", "framework", "repo", "github repo", "developer tools", 
                    "coding tutorial", "syntax", "refactoring", "hugging face", "huggingface",
                    "dataset", "datasets", "arxiv", "paper", "arxiv paper", "pre-print", 
                    "quantization", "quantized", "loss function", "hyperparameters", "github.com",
                    "open-source library", "backend architecture", "python package", "javascript library",
                    "npm install", "local model run", "r/MachineLearning", "r/LocalLLaMA"
                ]
                title_lower = title.lower()
                desc_lower = (art.get('description', '') or '').lower()
                if any(kw in title_lower or kw in desc_lower for kw in dev_antikeywords):
                    # Skip highly developer-centric / tech-niche articles to ensure mass-appeal (18-70)
                    continue
                    
                # ── 3. MULTI-SIGNAL VIRAL SCORING (Phase 1 Upgrade) ──────────
                title_lower = title.lower()
                
                # A. Keyword Score (legacy, reduced weight)
                keyword_score = sum(15 for kw in hot_keywords if kw in title_lower)
                # Mass-appeal viral keywords (broader than AI-only)
                breaking_keywords = ["hidden", "secret", "trick", "hack", "tip", "feature", "setting", "free", "save",
                                     "mistake", "wrong", "stop", "never", "always", "battery", "speed", "fast", "slow",
                                     "privacy", "tracking", "recording", "watching", "listening", "delete", "dangerous",
                                     "launch", "release", "leak", "breakthrough", "announces", "unveils", "shuts down",
                                     "lawsuit", "ban", "hack", "scandal", "fired", "exposed", "shutdown", "free", "open-source"]
                keyword_score += sum(10 for kw in breaking_keywords if kw in title_lower)
                controversy_keywords = ["wrong", "lie", "fake", "scam", "caught", "secretly", "hidden", "nobody", "shocking",
                                        "insane", "crazy", "terrifying", "scary", "devastating", "myth", "truth", "actually"]
                keyword_score += sum(12 for kw in controversy_keywords if kw in title_lower)
                big_names = ["google", "apple", "iphone", "android", "samsung", "openai", "meta", "microsoft", "windows",
                             "chrome", "whatsapp", "instagram", "tiktok", "nvidia", "tesla", "amazon", "chatgpt"]
                keyword_score += sum(8 for name in big_names if name in title_lower)
                
                # B. Source Engagement Score (NEW: from trending engine)
                engagement_score = art.get("_engagement_score", 0)  # Pre-computed by trending_engine
                if not engagement_score:
                    # Legacy sources without engagement data get base scores by type
                    if art.get("type") == "trending": engagement_score = 20
                    elif art.get("type") == "tools": engagement_score = 10
                    elif art.get("type") == "research": engagement_score = 10
                    else: engagement_score = 5
                
                # C. Trending Velocity Score (NEW: how fast is this topic rising?)
                trending_velocity_score = 0
                eng = art.get("_engagement", {})
                if art.get("type") == "reddit_trending":
                    velocity = eng.get("upvote_velocity", 0)
                    if velocity > 100: trending_velocity_score = 40
                    elif velocity > 50: trending_velocity_score = 30
                    elif velocity > 20: trending_velocity_score = 20
                    elif velocity > 5: trending_velocity_score = 10
                elif art.get("type") == "youtube_trending":
                    views = eng.get("views", 0)
                    if views > 100000: trending_velocity_score = 40
                    elif views > 50000: trending_velocity_score = 30
                    elif views > 10000: trending_velocity_score = 20
                elif art.get("type") == "github_trending":
                    spd = eng.get("stars_per_day", 0)
                    if spd > 100: trending_velocity_score = 35
                    elif spd > 50: trending_velocity_score = 25
                    elif spd > 10: trending_velocity_score = 15
                
                # D. Niche Gap Score (NEW: prefer topics competitors haven't covered)
                niche_score = 0
                niche_types = ["github_trending", "reddit_trending"]
                if art.get("type") in niche_types:
                    niche_score = 15  # Niche sources get a boost
                
                # E. Recency Score
                recency_score = 0
                pub_at = art.get("publishedAt", "")
                if pub_at:
                    try:
                        pub_dt = datetime.fromisoformat(pub_at.replace('Z', '+00:00'))
                        hours_ago = (datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600
                        if hours_ago < 6: recency_score = 25
                        elif hours_ago < 12: recency_score = 20
                        elif hours_ago < 24: recency_score = 15
                        elif hours_ago < 48: recency_score = 8
                    except: pass
                
                # F. Type-Specific Scoring Boosts (Mass-Appeal Categories)
                type_score = 0
                if topic_type == "tools":
                    tool_keywords = ["tool", "app", "free", "alternative", "workflow", "extension", "trick", "hack", "tip",
                                     "save", "productivity", "ai tool", "chatgpt", "plugin", "shortcut"]
                    type_score += sum(15 for kw in tool_keywords if kw in title_lower)
                    if art.get("type") in ["tools", "github_trending", "youtube_trending"]:
                        type_score += 25
                elif topic_type == "news":
                    news_keywords = ["hidden", "secret", "feature", "setting", "trick", "hack", "tip", "iphone", "android",
                                     "privacy", "tracking", "battery", "speed", "myth", "wrong", "mistake", "stop",
                                     "breaking", "scandal", "warns", "dangerous", "scary", "free"]
                    type_score += sum(15 for kw in news_keywords if kw in title_lower)
                    if art.get("type") in ["trending", "youtube_trending", "reddit_trending"]:
                        type_score += 25
                elif topic_type == "research":
                    # Research now means "educational tech facts" not academic papers
                    research_keywords = ["truth", "myth", "actually", "real", "fact", "science", "how", "why", "works",
                                         "explained", "debunk", "wrong", "correct", "proof"]
                    type_score += sum(15 for kw in research_keywords if kw in title_lower)
                    if art.get("type") in ["reddit_trending", "youtube_trending"]:
                        type_score += 25
                
                # ── COMPOSITE VIRAL SCORE (weighted blend) ──
                hot_score = (
                    keyword_score * 0.15 +
                    engagement_score * 0.35 +
                    trending_velocity_score * 0.25 +
                    niche_score * 0.15 +
                    recency_score * 0.10 +
                    type_score * 0.10
                )

                art['_hot_score'] = round(hot_score, 1)
                art['_score_breakdown'] = {
                    'kw': keyword_score, 'eng': engagement_score, 
                    'vel': trending_velocity_score, 'niche': niche_score,
                    'rec': recency_score, 'type': type_score
                }
                filtered_articles.append(art)
                seen_titles_in_this_batch.append(title)
            
            if not filtered_articles:
                print("⚠️ No unique viral articles. Falling back to Search...")
                articles = None
            else:
                # Rank by composite viral score
                filtered_articles.sort(key=lambda x: x.get('_hot_score', 0), reverse=True)
                
                viable_article = None
                print("🧐 Running Shorts viability checks on top candidates...")
                for art in filtered_articles[:5]:
                    t_title = art.get("title", "")
                    t_desc = art.get("description", "")
                    viability = check_shorts_viability_via_gemini(client, t_title, t_desc)
                    if viability.get("overall_viable"):
                        print(f"✅ Topic is viable: '{t_title[:60]}...' (Reason: {viability.get('reason')})")
                        viable_article = art
                        framing = viability.get("enemy_hero_hook_framing")
                        if framing:
                            extra_instruction += f"\nFraming instruction: Frame the script around this angle: '{framing}'.\n"
                        break
                    else:
                        print(f"❌ Topic failed viability check: '{t_title[:60]}...' (Reason: {viability.get('reason')})")
                
                if viable_article:
                    filtered_articles.remove(viable_article)
                    filtered_articles.insert(0, viable_article)
                    top = viable_article
                else:
                    print("⚠️ No candidate passed the viability filter. Using the highest scored one.")
                    top = filtered_articles[0]
                    
                bd = top.get('_score_breakdown', {})
                print(f"🏆 Top Viral Candidate: '{top.get('title')}' (Score: {top['_hot_score']:.1f})")
                print(f"   Breakdown: kw={bd.get('kw',0)} eng={bd.get('eng',0)} vel={bd.get('vel',0)} niche={bd.get('niche',0)} rec={bd.get('rec',0)} type={bd.get('type',0)}")
                articles = filtered_articles
                
                # Force tech_trends topic_type if selected article is Google Trends or YouTube Outlier
                if top and top.get("type") in ["google_trends", "youtube_outliers"]:
                    print(f"🔥 Selected article type is '{top.get('type')}'. Switching topic_type to 'tech_trends' to apply 3-part Shorts Formula.")
                    topic_type = "tech_trends"

        # ── STEP 1: GEMINI SEARCH FALLBACK (biased toward hot topic) ────────────
        if not articles:
            search_subject = hot_topic_str if hot_topic_str else f"{category}"
            print(f"🔍 STEP 1: Using Gemini Search for '{search_subject}'...")
            search_query = (
                f"Find the most viral, trending content about: {search_subject}. "
                f"Look for AI tools, fascinating facts, side hustles, productivity hacks, or free alternatives "
                f"that are going viral on social media right now. Must appeal to general consumers."
            )
            
            try:
                search_response = client.models.generate_content(
                    model=GEMINI_FLASH_MODEL, # Use stable flash for tools
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
                news_context += f"GEMINI SEARCH RESULTS (Grounded):\n{search_response.text}\n\nSOURCES FOUND:\n{links_str}\n"
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
        content_desc = "viral AI tools, money-making side hustles, productivity hacks, and free apps that appeal to EVERYONE"
    elif topic_type == "news":
        content_desc = "fascinating tech facts, AI industry shifts, motivation and success stories, and viral tech updates"
    elif topic_type == "tech_trends":
        content_desc = "high-velocity Google tech search trends and viral YouTube breakout videos"
    else:
        content_desc = "surprising tech facts, agentic AI facts, AI coding shortcuts, and AI transformation experiments"

    from config import ENABLE_LONGFORM
    is_longform = ("Slot C" in slot) and ENABLE_LONGFORM
    
    if is_longform:
        selection_instruction = (
            f"Analyze the following {content_desc} and pick 1 MAIN story for a deep-dive, and 3-5 RAPID news updates.\n"
            "SELECTION FILTERS:\n"
            "1. MUST create a fast-paced 'News Roundup + Deep Dive' format (The Vaibhav Sisinty style).\n"
            "2. MUST be explainable in 120-180s of extremely high-energy, fast-paced speech (approx 350-450 words total).\n"
            "3. MUST contain a high-stakes Hook, a rapid-fire News Roundup (3-5 quick hits), and 1 Deep Dive Workflow.\n"
            "4. PRIORITIZE: Major industry shifts, model benchmarks, or extreme utility for the deep dive.\n"
            "5. VOCAL DYNAMICS: You MUST use heavy punctuation (commas, ellipses '...', exclamation marks, italics, ALL CAPS) around key technical terms and transitions. The TTS engine relies entirely on punctuation to vary pitch and emphasis. Never use plain unpunctuated sentences for important points.\n"
            "FORMAT: You MUST follow the strict 4-part structure: Hook -> News Roundup -> Deep Dive -> Outro.\n"
        )
        prompt_requirements = f"""Return ONLY this exact JSON (no markdown, no explanation):
{{
  "title_options": ["Title idea 1", "Title idea 2", "Title idea 3"],
  "description": "Full 100+ word rich SEO description for youtube describing the video, including timestamps and credits.",
  "use_case_evidence_url": "MANDATORY: A direct, valid URL from the 'SOURCES FOUND' section to be used as visual evidence.",
  "title": "Punchy YouTube title max 60 chars",
  "hook_script": "15-second high stakes intro (approx 30 words). Stark contrast, direct eye contact feel.",
  "news_roundup": "Rapid-fire coverage of 3-5 news updates. Use bold transitions like 'UPDATE 1...' (approx 150 words).",
  "deep_dive_tutorial": "A step-by-step breakdown or workflow of the main story/tool. Fast paced (approx 200 words).",
  "outro_cta": "Subscribe and comment prompt (approx 30 words).",
  "script": "The FULL unified voiceover script seamlessly concatenating hook_script, news_roundup, deep_dive_tutorial, and outro_cta into ONE single flowing text block. Target total word count: 350-450 words.",
  "hook_text": "The exact first 5-8 words of the script.",
  "relevant_links": ["https://github.com/...", "https://arxiv.org/abs/..."],
  "phonetic_pronunciation_map": {{"NVIDIA": "In-vid-yah"}},
  "hook": "Matches the first sentence of the script",
  "summary": "One line summary",
  "sub_category": "AI/Machine Learning",
  "breaking_news_level": 9,
  "retention_cues": [{{ "timestamp": 3.0, "effect": "zoom_in", "reason": "hook_impact" }}],
  "subtitle_chunks": [{{
      "chunk_id": 1,
      "text": "Sentence 1",
      "start": 0.00,
      "end": 3.50, 
      "scene_objective": "What technical concept must be understood here",
      "visual_type": "Video|AI Image|Whiteboard|Infographic|Diagram|Animated UI Mockup",
      "nano_visual_prompt": "A highly specific, cinematic visual description for THIS sentence. Must depict the exact subject/entity/concept spoken in this sentence. Example: 'Satellite view of Earth at night showing glowing city lights and data center hotspots, cinematic 9:16, dark background'. NO TEXT in the image. NO faces of real people. Photorealistic, 8K.",
      "on_screen_elements": ["labels/arrows/highlights/icons/charts/code"],
      "camera_motion": "Slow zoom|Dolly-in|Orbit|Pan|Tracking shot|None",
      "transition": "Match cut|Zoom transition|Morph|Swipe|Data stream transition|Neural network transition"
  }}],
  "original_news_headline": "Exact headline",
  "original_news_url": "Direct article URL",
  "keywords": ["AI"],
  "hashtags": ["#AI", "#CyberSecurity", "#DataPrivacy", "#TechNews"],
  "companies_mentioned": ["list of company names mentioned in the script"],
  "companies": [{"name": "Company Name", "domain": "domain.com", "description": "2-4 word description of the company"}],
  "people": [{"name": "Person Name", "wikipedia_slug": "Wikipedia_Slug", "description": "2-4 word description of the person's role"}],
  "key_entities": [{"name": "Entity Name", "type": "COMPANY|PEOPLE|TOOL|OTHER", "description": "2-4 word description of the entity"}],
  "comment_hook": "Provocative question to drive engagement (e.g. 'Which department at your job is leaking the most data?')"
}}"""
    else:
        if topic_type == "vaibhav":
            selection_instruction = (
                f"Analyze the following {content_desc} and pick the SINGLE most breakout AI/tech story that matches the Vaibhav Sisinty content pillars:\n"
                f"PRIMARY CATEGORY: {category}\n"
                "CONTENT PILLARS:\n"
                "1. AI Tool Spotlight — newly launched/underrated AI tools with specific use cases.\n"
                "2. Prompt Hack — specific copy-paste prompts that solve real tasks.\n"
                "3. AI Workflow Reveal — automating or speeding up common work tasks using AI.\n"
                "4. Career/Job Impact — how AI is reshaping specific jobs or skills.\n"
                "5. AI Business Idea — simple businesses someone can start today using AI tools.\n\n"
                "SELECTION FILTERS:\n"
                "1. MUST appeal to Indian professionals, creators, and founders (22-40yo).\n"
                "2. MUST solve a real pain point: saving time, staying relevant, earning more, outpacing competition.\n"
                "3. AVOID pure tech jargon, deep research papers, or developer tools with no consumer/creator angle.\n"
                "4. CHOOSE a story that enables a slightly alarming hook (creates urgency or curiosity)."
            )
            prompt_requirements = f"""Return ONLY this exact JSON (no markdown, no explanation):
{{
  "title_options": ["Title Case + Emoji + Curiosity Gap 1", "Title Case + Emoji + Curiosity Gap 2"],
  "description": "Full 100+ word rich SEO description for youtube describing the video, including relevant hashtags and the source URL.",
  "use_case_evidence_url": "MANDATORY: A direct, valid URL from the 'SOURCES FOUND' section to be used as visual evidence.",
  "title": "Punchy YouTube title max 60 chars — must appeal to professionals and creators",
  "hook_script": "The Hook (0-5s): Direct, slightly alarming opening hook. 10-15 words.",
  "problem_context": "The Problem (5-12s): What most people are doing wrong or missing. 15-20 words.",
  "solution_tech": "The Solution (12-35s): The specific tool, prompt, or workflow in action. 50-60 words.",
  "retention_loop": "The Proof/Result (35-50s): Concrete outcome: time saved, task automated. 30-40 words.",
  "outro_cta": "The Outro/CTA (50-55s): Soft CTA (follow for more, save this). 10-15 words.",
  "script": "The FULL voiceover script. Target 130-145 words total (approx 55 seconds). The last sentence MUST flow back into the first for looping.",
  "hook_text": "The exact first 5-8 words of the script.",
  "relevant_links": ["https://example.com"],
  "phonetic_pronunciation_map": {{}},
  "hook": "Matches the first sentence of the script",
  "summary": "One line summary",
  "sub_category": "{category}",
  "breaking_news_level": 9,
  "retention_cues": [{{ "timestamp": 2.0, "effect": "zoom_in", "reason": "hook_impact" }}],
  "subtitle_chunks": [{{
      "chunk_id": 1,
      "text": "Sentence 1",
      "start": 0.00,
      "end": 2.50,
      "scene_objective": "What technical concept must be understood here",
      "visual_type": "Video|AI Image|Whiteboard|Infographic|Diagram|Animated UI Mockup",
      "nano_visual_prompt": "A clean, specific visual for THIS sentence. Vertical 9:16. Photorealistic, 8K. NO text overlays.",
      "on_screen_elements": ["labels/arrows/highlights/icons/charts/code"],
      "camera_motion": "Slow zoom|Dolly-in|Orbit|Pan|Tracking shot|None",
      "transition": "Match cut|Zoom transition|Morph|Swipe|Data stream transition|Neural network transition"
  }}],
  "original_news_headline": "Exact headline",
  "original_news_url": "Direct article URL",
  "keywords": ["AI Hacks", "Tech Tips", "Productivity"],
  "hashtags": ["#AIHacks", "#TechTips", "#Productivity", "#VaibhavSisinty"],
  "companies_mentioned": ["list of company names mentioned in the script"],
  "companies": [{{"name": "Company Name", "domain": "domain.com", "description": "2-4 word description of the company"}}],
  "people": [{{"name": "Person Name", "wikipedia_slug": "Wikipedia_Slug", "description": "2-4 word description of the person's role"}}],
  "key_entities": [{{"name": "Entity Name", "type": "COMPANY|PEOPLE|TOOL|OTHER", "description": "2-4 word description of the entity"}}],
  "comment_hook": "A provocative question: 'What do you think? Drop your comment below! 👇'"
}}"""
        elif topic_type == "tech_trends":
            selection_instruction = (
                f"Analyze the following {content_desc} and pick the SINGLE most breakout, high-velocity tech topic or viral video.\n"
                f"PRIMARY CATEGORY: {category}\n"
                "SELECTION FILTERS:\n"
                "1. MUST follow the high-engagement 3-part micro-script structure precisely.\n"
                "2. VISUAL DEMONSTRATION REQUIRED: The `nano_visual_prompt` fields MUST describe the exact screen, code editor, or device showing the tech in action.\n"
                "3. STRICT DURATION: Enforce a target total word count of 120-140 words, explainable in 45-50 seconds of fast-paced speech. Keep it extremely tight.\n"
                "4. LOOP-FRIENDLY: The final loop CTA must connect seamlessly back to the visual hook."
            )
            prompt_requirements = f"""Return ONLY this exact JSON (no markdown, no explanation):
{{
  "title_options": ["Title Case + Emoji + Curiosity Gap 1", "Title Case + Emoji + Curiosity Gap 2"],
  "description": "Full 100+ word rich SEO description for youtube describing the video, including relevant hashtags.",
  "use_case_evidence_url": "MANDATORY: A direct, valid URL from the 'SOURCES FOUND' section to be used as visual evidence.",
  "title": "Punchy YouTube title max 60 chars — must appeal to tech-interested audiences",
  "hook_script": "The Visual Hook (0:00 - 0:03): State the breakout tech trend/query immediately as a negative or high-stakes claim. Never start with an introduction. 10-15 words.",
  "solution_tech": "The Technical Core (0:03 - 0:45): Deliver the exact breakout answer or content gap solution. Keep code snippets under 3 lines or focus on UI step-by-step demonstrations. 90-110 words.",
  "retention_loop": "The Loop/CTA (0:45 - 0:50): End on an incomplete thought or a question that seamlessly loops back to the opening hook script. 15-20 words.",
  "outro_cta": "CTA: Subscribe/Follow for more daily tech trends. 8-10 words.",
  "script": "The FULL voiceover script concatenating hook_script, solution_tech, and retention_loop. Target 120-145 words total (approx 45-50s). The final sentence MUST flow back into the first sentence for looping.",
  "hook_text": "The exact first 5-8 words of the script.",
  "relevant_links": ["https://example.com"],
  "phonetic_pronunciation_map": {{}},
  "hook": "Matches the first sentence of the script",
  "summary": "One line summary",
  "sub_category": "{category}",
  "breaking_news_level": 9,
  "retention_cues": [{{ "timestamp": 3.0, "effect": "zoom_in", "reason": "hook_impact" }}],
  "subtitle_chunks": [{{
      "chunk_id": 1,
      "text": "Sentence 1",
      "start": 0.00,
      "end": 3.00,
      "scene_objective": "What technical concept must be understood here",
      "visual_type": "Video|AI Image|Whiteboard|Infographic|Diagram|Animated UI Mockup",
      "nano_visual_prompt": "A clean, specific visual for THIS sentence. Example: 'Close-up of a code editor showing a python script executing, dark mode, 9:16 vertical'. Photorealistic, 8K. NO text overlays.",
      "on_screen_elements": ["labels/arrows/highlights/icons/charts/code"],
      "camera_motion": "Slow zoom|Dolly-in|Orbit|Pan|Tracking shot|None",
      "transition": "Match cut|Zoom transition|Morph|Swipe|Data stream transition|Neural network transition"
  }}],
  "original_news_headline": "Exact headline or trend query name",
  "original_news_url": "Direct article or trend URL",
  "keywords": ["Tech Trends", "AI", "Coding", "Software"],
  "hashtags": ["#TechTrends", "#AI", "#Coding", "#Software", "#Developer"],
  "companies_mentioned": ["list of company names mentioned in the script"],
  "companies": [{"name": "Company Name", "domain": "domain.com", "description": "2-4 word description of the company"}],
  "people": [{"name": "Person Name", "wikipedia_slug": "Wikipedia_Slug", "description": "2-4 word description of the person's role"}],
  "key_entities": [{"name": "Entity Name", "type": "COMPANY|PEOPLE|TOOL|OTHER", "description": "2-4 word description of the entity"}],
  "comment_hook": "A provocative question: 'What do you think of this breakout tool? Let me know below! 👇'"
}}"""
        elif topic_type == "tools":
            selection_instruction = (
                f"Analyze the following {content_desc} and pick the SINGLE most useful, surprising tip or tool for EVERYDAY smartphone/computer users.\n"
                f"PRIMARY CATEGORY: {category}\n"
                "SELECTION FILTERS:\n"
                "1. PRIORITIZE: Tips, tricks, hidden features, or free tools that EVERYONE can use immediately. NOT for programmers only. Must be understandable by a 14-year-old.\n"
                "2. VISUAL DEMONSTRATION REQUIRED: The `nano_visual_prompt` fields MUST describe the exact screen/device showing the tip in action.\n"
                "3. MUST be explainable in exactly <35s (approx 80-120 words total). Strict 35s limit. SHORTER = MORE VIEWS.\n"
                "4. FORMAT: Hook (bold claim or surprising result) -> Quick Demo (show it working) -> Payoff (mind-blown moment) -> CTA (follow for more).\n"
                "5. LOOP-FRIENDLY: The last sentence MUST connect back to the first, creating a natural loop.\n"
                "6. UNIVERSAL & GENDER-INCLUSIVE DEMOGRAPHIC (18-70): The topic must appeal to all age groups from 18 to 70 and must engage women and girls. Do NOT pick highly technical developer tools or niche male-biased tech (e.g., PC builds, gaming tweaks). Focus on everyday consumer technology: photo/video styling tools, WhatsApp/Instagram/social settings, digital planning and calendar tools, smart home/lifestyle assistants, budget shopping shortcuts, and simple phone tricks."
            )
            prompt_requirements = f"""Return ONLY this exact JSON (no markdown, no explanation):
{{
  "title_options": ["Title Case + Emoji + Curiosity Gap 1", "Title Case + Emoji + Curiosity Gap 2"],
  "description": "Full 100+ word rich SEO description for youtube describing the video.",
  "use_case_evidence_url": "MANDATORY: A direct, valid URL from the 'SOURCES FOUND' section to be used as visual evidence.",
  "title": "Punchy YouTube title max 60 chars — must appeal to EVERYONE not just techies",
  "hook_script": "The Hook (<1.5s): Bold claim or surprising result. 5-8 words MAX.",
  "problem_context": "The Setup (2-5s): Why this matters to YOU personally. 10-15 words.",
  "solution_tech": "The Demo (5-25s): Show exactly how it works. Simple steps anyone can follow. 40-60 words.",
  "retention_loop": "The Loop Bridge (25-30s): Connect back to the opening. 8-12 words.",
  "outro_cta": "CTA: Follow for more tips. Subscribe. 8-10 words.",
  "script": "The FULL voiceover script. Target 80-120 words total. STRICT MAX 35 seconds. The last sentence MUST flow back into the first for looping.",
  "hook_text": "The exact first 5-8 words of the script.",
  "relevant_links": ["https://example.com"],
  "phonetic_pronunciation_map": {{}},
  "hook": "Matches the first sentence of the script",
  "summary": "One line summary",
  "sub_category": "{category}",
  "breaking_news_level": 9,
  "retention_cues": [{{ "timestamp": 2.0, "effect": "zoom_in", "reason": "hook_impact" }}],
  "subtitle_chunks": [{{
      "chunk_id": 1,
      "text": "Sentence 1",
      "start": 0.00,
      "end": 2.50,
      "scene_objective": "What technical concept must be understood here",
      "visual_type": "Video|AI Image|Whiteboard|Infographic|Diagram|Animated UI Mockup",
      "nano_visual_prompt": "A clean, specific visual for THIS sentence. Example: 'Close-up of an iPhone settings screen showing the hidden menu option highlighted, dark mode, 9:16 vertical'. Photorealistic, 8K. NO text overlays.",
      "on_screen_elements": ["labels/arrows/highlights/icons/charts/code"],
      "camera_motion": "Slow zoom|Dolly-in|Orbit|Pan|Tracking shot|None",
      "transition": "Match cut|Zoom transition|Morph|Swipe|Data stream transition|Neural network transition"
  }}],
  "original_news_headline": "Exact headline",
  "original_news_url": "Direct article URL",
  "keywords": ["Tech Tips", "iPhone", "Android", "Hidden Features"],
  "hashtags": ["#TechTips", "#iPhone", "#Android", "#HiddenFeatures", "#LifeHack"],
  "companies_mentioned": ["list of company names mentioned in the script"],
  "companies": [{"name": "Company Name", "domain": "domain.com", "description": "2-4 word description of the company"}],
  "people": [{"name": "Person Name", "wikipedia_slug": "Wikipedia_Slug", "description": "2-4 word description of the person's role"}],
  "key_entities": [{"name": "Entity Name", "type": "COMPANY|PEOPLE|TOOL|OTHER", "description": "2-4 word description of the entity"}],
  "comment_hook": "A provocative question: 'Did you know about this? Drop a 🤯 if this blew your mind!'"
}}"""
        elif topic_type == "news":
            selection_instruction = (
                f"Analyze the following {content_desc} and pick the SINGLE most shocking, scary, or mind-blowing tech fact for EVERYDAY people.\n"
                f"PRIMARY CATEGORY: {category}\n"
                "SELECTION FILTERS:\n"
                "1. PRIORITIZE: Privacy warnings, security scares, tech myths being debunked, common mistakes everyone makes. Must make the viewer feel PERSONALLY at risk or enlightened.\n"
                "2. Must be understandable by ANYONE — no jargon, no technical terms without simple analogies.\n"
                "3. MUST be explainable in exactly <35s (approx 80-120 words total). Strict 35s limit. SHORTER = MORE VIEWS.\n"
                "4. FORMAT: Hook (bold scary claim) -> Proof (show evidence) -> Fix/Truth (how to protect yourself) -> CTA (follow for more).\n"
                "5. LOOP-FRIENDLY: The last sentence MUST connect back to the first, creating a natural loop.\n"
                "6. UNIVERSAL & GENDER-INCLUSIVE DEMOGRAPHIC (18-70): Ensure the fact is highly relevant to all age groups from 18 to 70 and appeals to women and girls. Focus on privacy/surveillance threats (like location tracking on popular social apps), online security/scam alerts targeting consumers, smart lifestyle tech features, health/safety tech breakthroughs, or debunking common misconceptions about daily devices."
            )
            prompt_requirements = f"""Return ONLY this exact JSON (no markdown, no explanation):
{{
  "title_options": ["Title Case + Emoji + Curiosity Gap 1", "Title Case + Emoji + Curiosity Gap 2"],
  "description": "Full 100+ word rich SEO description for youtube describing the video.",
  "use_case_evidence_url": "MANDATORY: A direct, valid URL from the 'SOURCES FOUND' section to be used as visual evidence.",
  "title": "Punchy YouTube title max 60 chars — must trigger FOMO or fear",
  "hook_script": "The Hook (<1.5s): Scary claim or myth-busting statement. 5-8 words MAX.",
  "problem_context": "The Setup (2-5s): Why this is terrifying or why you've been wrong. 10-15 words.",
  "solution_tech": "The Proof (5-25s): Show the evidence, explain the truth, or demonstrate the fix. 40-60 words.",
  "retention_loop": "The Loop Bridge (25-30s): Connect back to the opening claim. 8-12 words.",
  "outro_cta": "CTA: Follow for more. Subscribe. 8-10 words.",
  "script": "The FULL voiceover script. Target 80-120 words total. STRICT MAX 35 seconds. Loop-friendly ending.",
  "hook_text": "The exact first 5-8 words of the script.",
  "relevant_links": ["https://example.com"],
  "phonetic_pronunciation_map": {{}},
  "hook": "Matches the first sentence of the script",
  "summary": "One line summary",
  "sub_category": "{category}",
  "breaking_news_level": 9,
  "retention_cues": [{{"timestamp": 2.0, "effect": "zoom_in", "reason": "hook_impact"}}],
  "subtitle_chunks": [{{
      "chunk_id": 1,
      "text": "Sentence 1",
      "start": 0.00,
      "end": 2.50,
      "scene_objective": "What technical concept must be understood here",
      "visual_type": "Video|AI Image|Whiteboard|Infographic|Diagram|Animated UI Mockup",
      "nano_visual_prompt": "A dramatic, cinematic visual for THIS sentence. Example: 'Close-up of a smartphone screen showing location tracking data, dark moody lighting, 9:16 vertical'. Photorealistic, 8K. NO text overlays.",
      "on_screen_elements": ["labels/arrows/highlights/icons/charts/code"],
      "camera_motion": "Slow zoom|Dolly-in|Orbit|Pan|Tracking shot|None",
      "transition": "Match cut|Zoom transition|Morph|Swipe|Data stream transition|Neural network transition"
  }}],
  "original_news_headline": "Exact headline",
  "original_news_url": "Direct article URL",
  "keywords": ["Privacy", "Tech Tips", "Security", "Phone Hacks"],
  "hashtags": ["#Privacy", "#TechTips", "#iPhone", "#Android", "#ScaryTech"],
  "companies_mentioned": ["list of company names mentioned in the script"],
  "companies": [{"name": "Company Name", "domain": "domain.com", "description": "2-4 word description of the company"}],
  "people": [{"name": "Person Name", "wikipedia_slug": "Wikipedia_Slug", "description": "2-4 word description of the person's role"}],
  "key_entities": [{"name": "Entity Name", "type": "COMPANY|PEOPLE|TOOL|OTHER", "description": "2-4 word description of the entity"}],
  "comment_hook": "A provocative question: 'Did you know about this? Are you going to change this setting? 👇'"
}}"""
        else: # research / educational / general
            selection_instruction = (
                f"Analyze the following {content_desc} and pick the SINGLE most surprising, useful, or mind-blowing tech fact or comparison.\n"
                f"PRIMARY CATEGORY: {category}\n"
                "SELECTION FILTERS:\n"
                "1. PRIORITIZE: Surprising comparisons (free vs paid), hidden features nobody knows about, AI experiments with visual results, or tech facts that make people say 'WHAT?!'\n"
                "2. Must be understandable by ANYONE — a teenager and a grandparent should both find it useful or amazing.\n"
                "3. MUST be explainable in exactly <35s (approx 80-120 words total). Strict 35s limit. SHORTER = MORE VIEWS.\n"
                "4. FORMAT: Hook (surprising claim) -> Evidence/Demo (show it) -> Payoff (the 'wow' moment) -> CTA (follow for more).\n"
                "5. LOOP-FRIENDLY: The last sentence MUST connect back to the first, creating a natural loop.\n"
                "6. UNIVERSAL & GENDER-INCLUSIVE DEMOGRAPHIC (18-70): Choose topics that both young adults (18-35) and older adults (50-70) can engage with, ensuring strong appeal to women and girls. Prefer digital lifestyle productivity, design and photo comparisons, smart home convenience, AI's impact on everyday careers/learning, and highly visual, relatable AI applications."
            )
            prompt_requirements = f"""Return ONLY this exact JSON (no markdown, no explanation):
{{
  "title_options": ["Title Case + Emoji + Curiosity Gap 1", "Title Case + Emoji + Curiosity Gap 2"],
  "description": "Full 100+ word rich SEO description for youtube describing the video.",
  "use_case_evidence_url": "MANDATORY: A direct, valid URL from the 'SOURCES FOUND' section to be used as visual evidence.",
  "title": "Punchy YouTube title max 60 chars — curiosity gap required",
  "hook_script": "The Hook (<1.5s): Surprising claim or bold statement. 5-8 words MAX.",
  "problem_context": "The Setup (2-5s): Why this matters to you. 10-15 words.",
  "solution_tech": "The Reveal (5-25s): Show the evidence, comparison, or demonstration. 40-60 words.",
  "retention_loop": "The Loop Bridge (25-30s): Connect back to the opening. 8-12 words.",
  "outro_cta": "CTA: Follow for more. Subscribe. 8-10 words.",
  "script": "The FULL voiceover script. Target 80-120 words total. STRICT MAX 35 seconds. Loop-friendly ending.",
  "hook_text": "The exact first 5-8 words of the script.",
  "relevant_links": ["https://example.com"],
  "phonetic_pronunciation_map": {{}},
  "hook": "Matches the first sentence of the script",
  "summary": "One line summary",
  "sub_category": "{category}",
  "breaking_news_level": 9,
  "retention_cues": [{{"timestamp": 2.0, "effect": "zoom_in", "reason": "hook_impact"}}],
  "subtitle_chunks": [{{
      "chunk_id": 1,
      "text": "Sentence 1",
      "start": 0.00,
      "end": 2.50,
      "scene_objective": "What technical concept must be understood here",
      "visual_type": "Video|AI Image|Whiteboard|Infographic|Diagram|Animated UI Mockup",
      "nano_visual_prompt": "A clean, specific visual for THIS sentence. Example: 'Split screen showing a free app vs expensive app side by side, clean modern UI, 9:16 vertical'. Photorealistic, 8K. NO text overlays.",
      "on_screen_elements": ["labels/arrows/highlights/icons/charts/code"],
      "camera_motion": "Slow zoom|Dolly-in|Orbit|Pan|Tracking shot|None",
      "transition": "Match cut|Zoom transition|Morph|Swipe|Data stream transition|Neural network transition"
  }}],
  "original_news_headline": "Exact headline",
  "original_news_url": "Direct article URL",
  "keywords": ["Tech Tips", "Free Apps", "Hidden Features", "AI"],
  "hashtags": ["#TechTips", "#FreeApps", "#LifeHack", "#DidYouKnow", "#Tech"],
  "companies_mentioned": ["list of company names mentioned in the script"],
  "companies": [{"name": "Company Name", "domain": "domain.com", "description": "2-4 word description of the company"}],
  "people": [{"name": "Person Name", "wikipedia_slug": "Wikipedia_Slug", "description": "2-4 word description of the person's role"}],
  "key_entities": [{"name": "Entity Name", "type": "COMPANY|PEOPLE|TOOL|OTHER", "description": "2-4 word description of the entity"}],
  "comment_hook": "A provocative question: 'Which one are YOU using? Drop your answer below! 👇'"
}}"""

    # Inject any extra instructions (e.g. screenshot avoidance, length adjustments) into context
    if extra_instruction:
        news_context += f"\n\nADDITIONAL INSTRUCTIONS:\n{extra_instruction}\n"

    engine = MultiAgentGenerationEngine(client, news_context, slot, category, strategy_enhancement, is_longform, raw_articles=articles, topic_type=topic_type)
    script_data = engine.execute(selection_instruction, prompt_requirements)
    
    if script_data:
        # Perform uniqueness check (Final safeguard)
        headline = script_data.get("original_news_headline", "")
        news_url = script_data.get("original_news_url", "")
        keywords = script_data.get("keywords", [])
        title = script_data.get("title", "")
        
        is_unique, msg = check_story_uniqueness(title, headline, keywords, news_url)
        if not is_unique:
            print(f"⚠️ [LOOP] Safeguard: Post-loop uniqueness check failed: {msg}")
            if headline:
                failed_topics.append(headline)
            if news_url:
                failed_topics.append(news_url)
            return None
            
    return script_data

def call_fallback_model(prompt):
    """
    Attempts to call non-Gemini fallback APIs in sequence:
    OpenAI -> Anthropic (Claude) -> Groq (Llama) -> DeepSeek -> OpenRouter.
    Returns the parsed JSON response dict or None.
    """
    import os
    import json
    import requests

    def clean_and_parse_json(content):
        raw = content.strip()
        if "```json" in raw:
            raw = raw[raw.find("```json")+7:raw.rfind("```")]
        elif "```" in raw:
            raw = raw[raw.find("```")+3:raw.rfind("```")]
        return json.loads(raw.strip())

    # 1. OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("🔮 Gemini failed. Falling back to OpenAI (gpt-4o-mini)...")
        try:
            headers = {
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.7
            }
            r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()
                return clean_and_parse_json(content)
            else:
                print(f"⚠️ OpenAI API failed with code {r.status_code}: {r.text}")
        except Exception as e:
            print(f"⚠️ OpenAI fallback failed: {e}")

    # 2. Anthropic (Claude)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("🔮 Gemini/OpenAI failed. Falling back to Anthropic (claude-3-5-haiku-20241022)...")
        try:
            headers = {
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 4000,
                "messages": [{"role": "user", "content": prompt}]
            }
            r = requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                content = r.json()["content"][0]["text"].strip()
                return clean_and_parse_json(content)
            else:
                print(f"⚠️ Anthropic API failed with code {r.status_code}: {r.text}")
        except Exception as e:
            print(f"⚠️ Anthropic fallback failed: {e}")

    # 2.5 Cerebras
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    if cerebras_key:
        headers = {
            "Authorization": f"Bearer {cerebras_key}",
            "Content-Type": "application/json"
        }
        cerebras_models = ["llama-3.3-70b", "llama-3.1-70b", "llama-3.1-8b"]
        for model_name in cerebras_models:
            print(f"🔮 Falling back to Cerebras ({model_name})...")
            try:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.7
                }
                r = requests.post("https://api.cerebras.ai/v1/chat/completions", json=payload, headers=headers, timeout=30)
                if r.status_code == 200:
                    content = r.json()["choices"][0]["message"]["content"].strip()
                    return clean_and_parse_json(content)
                else:
                    print(f"⚠️ Cerebras API ({model_name}) failed with code {r.status_code}: {r.text}")
            except Exception as e:
                print(f"⚠️ Cerebras ({model_name}) fallback failed: {e}")

    # 3. Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        # Model preference order
        groq_models = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it", "llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b"]
        for model_name in groq_models:
            print(f"🔮 Gemini/OpenAI/Anthropic failed. Falling back to Groq ({model_name})...")
            try:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.7
                }
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
                if r.status_code == 200:
                    content = r.json()["choices"][0]["message"]["content"].strip()
                    return clean_and_parse_json(content)
                else:
                    print(f"⚠️ Groq ({model_name}) failed with code {r.status_code}: {r.text}")
            except Exception as e:
                print(f"⚠️ Groq ({model_name}) fallback failed: {e}")

    # 4. DeepSeek
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        print("🔮 Falling back to DeepSeek (deepseek-chat)...")
        try:
            headers = {
                "Authorization": f"Bearer {deepseek_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.7
            }
            r = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()
                return clean_and_parse_json(content)
            else:
                print(f"⚠️ DeepSeek API failed with code {r.status_code}: {r.text}")
        except Exception as e:
            print(f"⚠️ DeepSeek fallback failed: {e}")

    # 5. OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        }
        openrouter_models = ["meta-llama/llama-3.3-70b-instruct:free", "google/gemini-2.5-flash", "qwen/qwen-2.5-72b-instruct", "google/gemini-2.0-flash-lite-preview-02-05:free", "deepseek/deepseek-chat:free", "nvidia/llama-3.1-nemotron-70b-instruct:free"]
        for or_model in openrouter_models:
            print(f"🔮 Falling back to OpenRouter ({or_model})...")
            try:
                payload = {
                    "model": or_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                }
                r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=30)
                if r.status_code == 200:
                    content = r.json()["choices"][0]["message"]["content"].strip()
                    return clean_and_parse_json(content)
                else:
                    print(f"⚠️ OpenRouter API ({or_model}) failed with code {r.status_code}: {r.text}")
            except Exception as e:
                print(f"⚠️ OpenRouter ({or_model}) fallback failed: {e}")

    return None

class MultiAgentGenerationEngine:
    def __init__(self, client, context, slot, category, strategy_enhancement, is_longform, raw_articles=None, topic_type=None):
        self.client = client
        self.context = context
        self.slot = slot
        self.category = category
        self.strategy_enhancement = strategy_enhancement
        self.is_longform = is_longform
        self.raw_articles = raw_articles
        if topic_type == "vaibhav":
            self.persona = VAIBHAV_SYSTEM_PERSONA
        else:
            self.persona = SYSTEM_PERSONA

    def _call_gemini(self, prompt, model=GEMINI_FLASH_MODEL):
        import os
        from google import genai
        
        # Get list of API keys
        api_keys_env = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
        api_keys = [k.strip() for k in api_keys_env.split(",") if k.strip()]
        if not api_keys:
            api_keys = [GEMINI_API_KEY]
            
        # Initialize models to try
        models_to_try = [model]
        if model == GEMINI_PRO_MODEL:
            models_to_try.extend([GEMINI_FLASH_MODEL, GEMINI_FLASH_LITE_MODEL])
        elif model == GEMINI_FLASH_MODEL:
            models_to_try.extend([GEMINI_FLASH_LITE_MODEL])
            
        model_idx = 0
        key_idx = 0
        
        while models_to_try:
            current_model = models_to_try[model_idx % len(models_to_try)]
            current_key = api_keys[key_idx % len(api_keys)]
            
            try:
                # Initialize client with current key
                client = genai.Client(api_key=current_key)
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.8,
                        response_mime_type='application/json'
                    )
                )
                raw = response.text.strip()
                if "{" in raw and "}" in raw:
                    raw = raw[raw.find("{"):raw.rfind("}")+1]
                
                return json.loads(raw)
            except Exception as e:
                err_str = str(e).upper()
                # Handle Rate Limit (429), Overloaded (503), Depleted Quota (LIMIT: 0), or Not Found (404)
                if any(x in err_str for x in ["503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "429", "NOT_FOUND", "404", "LIMIT: 0", "QUOTA"]):
                    print(f"⚠️ [LOOP] Call failed ({current_model} with key {key_idx+1}/{len(api_keys)}): Rate Limit/Overload/Quota.")
                    key_idx += 1
                    
                    # If we have tried all keys for this model
                    if key_idx % len(api_keys) == 0:
                        print(f"⚠️ [LOOP] All keys exhausted for {current_model}. Removing from rotation.")
                        models_to_try.pop(model_idx % len(models_to_try))
                        # Do not increment model_idx since the list shrunk, so we automatically try the next model
                else:
                    print(f"⚠️ [LOOP] Call failed ({current_model}): {e}. Removing model from rotation.")
                    models_to_try.pop(model_idx % len(models_to_try))
                    key_idx = 0 # Reset key idx for the next model
                
        print("🚨 Gemini failed all attempts. Attempting fallback models...")
        fallback_res = call_fallback_model(prompt)
        if fallback_res:
            return fallback_res

        print("🚨 All fallback models failed or not configured.")
        return None

    def execute(self, selection_instruction, prompt_requirements):
        print("🎯 [AGENT 0] Selector Agent: Picking the single best story...")
        selector_prompt = SELECTOR_AGENT_TEMPLATE.format(
            persona=self.persona,
            selection_instruction=selection_instruction,
            news_context=self.context
        )
        selection = self._call_gemini(selector_prompt)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if not selection or "selected_headline" not in selection:
            print("⚠️ Selector Agent failed. Using raw context fallback.")
            # If we have articles, pick the top one from raw_articles as fallback
            if self.raw_articles and len(self.raw_articles) > 0:
                top = self.raw_articles[0]
                selected_headline = top.get("title", "AI Tech Breakthrough")
                selected_url = top.get("url", "")
                print(f"🔄 Fallback to Top Scored Article: {selected_headline}")
            else:
                selected_headline = "AI Tech Breakthrough"
                selected_url = ""
            selected_context = f"SELECTED STORY: {selected_headline}\nSOURCE: {selected_url}\n\nORIGINAL CONTEXT:\n{self.context}"
        else:
            selected_headline = selection["selected_headline"]
            selected_url = selection["selected_url"]
            
            # ── CONTEXT ISOLATION (Fixes topic-screenshot mismatch) ───────────
            isolated_context = ""
            if self.raw_articles:
                # Find matching article in the original list to provide rich but isolated context
                for art in self.raw_articles:
                    if art.get("url") == selected_url or art.get("title") == selected_headline:
                        isolated_context = (
                            f"Title: {art.get('title')}\n"
                            f"Description: {art.get('description')}\n"
                            f"Source: {art.get('source', {}).get('name')}\n"
                            f"URL: {art.get('url')}"
                        )
                        break
            
            if not isolated_context:
                # Fallback if no match found (e.g. search fallback)
                isolated_context = f"STORY: {selected_headline}\nSOURCE: {selected_url}"
                if "GEMINI SEARCH RESULTS" in self.context:
                    # If in search mode, we must include the grounded text but we'll instruct the agent to focus.
                    isolated_context += f"\n\nSEARCH CONTEXT:\n{self.context}"
            
            # ── CONTEXT SHARPENING (NEW) ──────────────────────────────────────
            # If we don't have rich RSS metadata, we MUST sharpen the search context
            # to prevent 'Hallucination Leakage' from other stories in the search result.
            print("🔬 [AGENT 0.5] Context Sharpener: Isolating target story facts...")
            sharpener_prompt = FACT_EXTRACTOR_TEMPLATE.format(
                persona=self.persona,
                target_headline=selected_headline,
                context=isolated_context # Pass the messy context to be sharpened
            )
            sharpened_data = self._call_gemini(sharpener_prompt)
            if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
            
            if sharpened_data and "core_narrative" in sharpened_data:
                isolated_context = (
                    f"STORY: {selected_headline}\n"
                    f"SOURCE: {selected_url}\n\n"
                    f"ISOLATED FACTS:\n{json.dumps(sharpened_data, indent=2)}"
                )
                print(f"✨ Context sharpened successfully for: {selected_headline}")
            else:
                print("⚠️ Context Sharpener failed. Falling back to raw isolation.")

            selected_context = (
                f"STRICT INSTRUCTION: You MUST ONLY research and write about the following story. "
                f"IGNORE all other news articles mentioned in any previous context.\n\n"
                f"TARGET STORY:\n{isolated_context}"
            )
            print(f"🔒 Isolated Context for downstream agents: {len(isolated_context)} chars")
            print(f"✅ Selected Story: {selected_headline}")

        print("🕵️ [AGENT 1] Research Agent: Extracting narrative elements...")
        research_prompt = RESEARCH_AGENT_TEMPLATE.format(
            persona=self.persona,
            news_context=selected_context
        )
        research = self._call_gemini(research_prompt)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if not research: return None

        print("🪝 [AGENT 2] Hook Agent: Generating high-retention hooks...")
        hook_prompt = HOOK_AGENT_TEMPLATE.format(
            persona=self.persona,
            research_json=json.dumps(research)
        )
        hooks_data = self._call_gemini(hook_prompt)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if not hooks_data or "hooks" not in hooks_data: return None
        
        # Pick best hook (highest combined score including swipe-stop power)
        best_hook = max(hooks_data["hooks"], key=lambda h: (
            h.get("curiosity_score", 0) + 
            h.get("emotional_trigger_score", 0) + 
            h.get("swipe_stop_score", h.get("curiosity_score", 0))  # Backward-compatible
        ))
        hook_text = best_hook.get('text', '')
        hook_words = len(hook_text.split())
        print(f"🎯 Selected Hook ({hook_words} words): {hook_text}")

        print("📝 [AGENT 3] Fact Script Generator: Writing the unified retention-optimized script...")
        narrative_prompt = NARRATIVE_AGENT_TEMPLATE.format(
            persona=self.persona,
            research_json=json.dumps(research),
            selected_hook=hook_text,
            selection_instruction=selection_instruction
        )
        narrative = self._call_gemini(narrative_prompt, model=GEMINI_PRO_MODEL)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if not narrative: return None

        print("⚡ [AGENT 4] Retention Optimizer: Maximizing pacing and curiosity density...")
        retention_prompt = RETENTION_OPTIMIZER_TEMPLATE.format(
            persona=self.persona,
            narrative_json=json.dumps(narrative)
        )
        optimized = self._call_gemini(retention_prompt, model=GEMINI_PRO_MODEL)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if not optimized: return None

        # ── PHASE 2: RETENTION SCIENTIST (Agent 4.5) ──────────────────────────
        print("🧬 [AGENT 4.5] Retention Scientist: Injecting proven retention patterns...")
        retention_sci_prompt = RETENTION_SCIENTIST_TEMPLATE.format(
            persona=self.persona,
            optimized_script=optimized.get("optimized_script", "")
        )
        retention_result = self._call_gemini(retention_sci_prompt, model=GEMINI_PRO_MODEL)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        
        retention_map = {}
        if retention_result and "retention_enhanced_script" in retention_result:
            optimized["optimized_script"] = retention_result["retention_enhanced_script"]
            retention_map = retention_result.get("retention_map", {})
            cgr = retention_map.get("curiosity_gap_ratio", 0)
            loops = len(retention_map.get("open_loops", []))
            interrupts = len(retention_map.get("pattern_interrupts", []))
            print(f"   ✅ Retention: {loops} open loops, {interrupts} pattern interrupts, {cgr:.0%} curiosity gap ratio")
        else:
            print("   ⚠️ Retention Scientist failed (non-fatal). Using optimizer output directly.")

        print("🗣️ [AGENT 5] Humanizer Agent: Fixing AI cadence and returning final schema...")
        # Inject the selected headline and URL back into the requirements if they are missing
        refined_requirements = prompt_requirements
        if "original_news_headline" in refined_requirements:
            refined_requirements = refined_requirements.replace('"original_news_headline": "Exact headline"', f'"original_news_headline": "{selected_headline}"')
        if "original_news_url" in refined_requirements:
            refined_requirements = refined_requirements.replace('"original_news_url": "Direct article URL"', f'"original_news_url": "{selected_url}"')

        humanizer_prompt = HUMANIZER_AGENT_TEMPLATE.format(
            persona=self.persona,
            optimized_script=optimized.get("optimized_script", ""),
            schema_requirements=refined_requirements
        )
        final_script = self._call_gemini(humanizer_prompt, model=GEMINI_PRO_MODEL)
        
        if final_script:
            # Final safety check: ensure the headline/url are set correctly in the final object
            if not final_script.get("original_news_headline") or final_script.get("original_news_headline") == "Exact headline":
                final_script["original_news_headline"] = selected_headline
            if not final_script.get("original_news_url") or final_script.get("original_news_url") == "Direct article URL":
                final_script["original_news_url"] = selected_url
            
            # ── PHASE 2: Attach retention_map to the final script for downstream use ──
            if retention_map:
                final_script["retention_map"] = retention_map
            
            print("⭐ [PIPELINE] Multi-Agent script generation completed successfully.")
        return final_script
