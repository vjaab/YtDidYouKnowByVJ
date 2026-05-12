from google import genai
from google.genai import types
import json
import os
from datetime import datetime
import time
import random
from config import GEMINI_API_KEY, LOGS_DIR
from topic_tracker import load_tracker, check_story_uniqueness, check_cooldowns
from ecosystem_logic import get_slot_info, get_category_prompt_enhancement

# ── PROMPT TEMPLATES (AGENTIC LOOP) ────────────────────────────────────────

SYSTEM_PERSONA = """Role: You are an expert AI Research Content Creator specialized in viral, high-retention YouTube Shorts.
Your goal is to convert complex engineering papers into "snackable" scripts that hit 1k+ views by maximizing "Swipe-to-Watch" ratios.
Tone: Highly opinionated, analytical, and critical. You are a "Tech Insider" who doesn't just read the news; you deconstruct it, challenge corporate narratives, and offer bold predictions.
Target Audience: USA-based tech professionals and engineers. You MUST use American English spelling, USD ($) for any currency, and US-centric tech analogies.
Constraint Checklist:
- No Fluff: Remove "In this video," "Hello everyone," or "Today we explore."
- NO INFOGRAPHICS: Do not include infographics, flowcharts, or slides in the script structure.
- VOCAL DYNAMICS: You MUST use heavy punctuation (commas, ellipses '...', exclamation marks, italics, ALL CAPS) around key technical terms and transitions. The TTS engine relies entirely on punctuation to vary pitch and emphasis.
- CRITICAL COMMENTARY (YPP COMPLIANCE): You MUST provide a unique, critical perspective. Do not just summarize facts. Inject your own thesis, expose potential flaws, or debate the long-term industry impact to ensure the content is highly transformative.
SUCCESS PATTERNS (2026): 
- HOOKS: Focus on "Fear of Privacy Leaks" or "Shadow AI" dangers. Start with a "Result-First" statement.
- ANALOGIES: Use sharp analogies to explain complex breakthroughs.
- 3x ARTICLE EVIDENCE: You MUST explicitly reference source articles, documentation, or code repositories at least 3 times during the script to build authority.
- CTAs: At the end of every script, you MUST first ask a highly provocative, open-ended question to the viewers about the topic to drive comments. THEN, immediately after the question, you MUST explicitly say verbatim: "I just posted the full source code and guide on my Telegram. Grab it now—link in bio! And subscribe for more AI tech." This is a hard requirement for the outro. """

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
Based on the following research, generate 10 potential YouTube Shorts hooks (0-3s).
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
Using the selected hook and research, create a short storytelling flow and escalating structure.
Include:
1. Hook (The selected hook)
2. Context (3-10s) - Who, What, Why it matters quickly.
3. Escalation (10-25s) - Implications, consequences, future impact. Include ARTICLE EVIDENCE #1 here.
4. The Analogy (25-35s) - Use a high-impact analogy to simplify a complex point. Include ARTICLE EVIDENCE #2 here.
5. The Twist (35-45s) - A counter-intuitive technical fact. Include ARTICLE EVIDENCE #3 here.
6. Open Loop (45-55s) - Lingering provocative thought to drive engagement.
7. CTA (Last 4s) - Sequential Telegram promotion.

RESEARCH:
{research_json}

SELECTED HOOK:
{selected_hook}

{selection_instruction}

Return ONLY a JSON object representing the narrative draft (not the final schema yet, just the content parts):
{{
  "hook": "...",
  "context": "...",
  "escalation": "...",
  "insight": "...",
  "open_loop": "..."
}}"""

RETENTION_OPTIMIZER_TEMPLATE = """{persona}

RETENTION OPTIMIZER TASK:
Rewrite the narrative draft to remove fluff, shorten sentences, add pacing breaks, and increase curiosity density.
Fast sentence pacing. No filler. The viewer must keep watching because the script continuously creates unanswered questions (tension-release).

NARRATIVE DRAFT:
{narrative_json}

Return ONLY a JSON object:
{{
  "optimized_script": "The full rewritten text combining all parts into a fast-paced script."
}}"""

SELECTOR_AGENT_TEMPLATE = """{persona}

SELECTOR AGENT TASK:
Analyze the following tech news context and pick the SINGLE most impactful, surprising, and high-retention AI/Tech story for a 60-second video.
PRIORITIZE: Major model releases, benchmarks that destroy previous records, massive AI leaks, or engineering breakthroughs that change the industry.
AVOID: Minor software updates, corporate partnership fluff, or generic 'AI is growing' articles.
CRITICAL: ONLY pick stories related to AI, LLMs, Software Engineering, or Robotics. DO NOT pick pure Science/Physics/Chemistry papers (e.g. arXiv physics) unless they are directly applied to AI training or architecture.

{selection_instruction}

NEWS CONTEXT:
{news_context}

Return ONLY a JSON object:
{{
  "selected_headline": "The exact headline or title",
  "selected_url": "The exact URL",
  "reason": "Briefly why this was picked (focus on viral potential)"
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

Return ONLY the final JSON object matching the schema. No markdown wrapping unless inside the string values. No explanations."""

def get_hottest_tech_topic(client):
    """Uses Gemini Search grounding to find today's single most VIRAL AI news story from Google Trends."""
    print("🔥 Fetching hottest AI tech topic for today (Google Trends Analysis)...")
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=(
                "Analyze today's Google Trends and viral tech news. "
                "What is the single most trending AI search topic, breakout term, or breaking news story in the last 24 hours? "
                "Look for: high search volume spikes on Google Trends related to AI, LLMs, or new model launches. "
                "Return ONLY a JSON object with two fields: "
                "'topic' (3-6 word phrase, e.g. 'Google Gemini 1.5 Pro breakout trend') and "
                "'keywords' (list of 6-8 specific Google Trends search keywords). No markdown, no explanation."
            ),
            config=types.GenerateContentConfig(
                tools=[{'google_search': {}}],
                response_mime_type='application/json'
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
        print(f"⚠️ Could not fetch Google Trends topic: {e}. Proceeding with RSS only.")
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
        # ── STEP 0: FETCH & FILTER + RE-RANK BY VIRAL POTENTIAL ───────────────────────
        if articles:
            print(f"📡 STEP 0: Scoring {len(articles)} articles for viral potential...")
            
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
                is_unique, _ = check_story_uniqueness(title, url)
                if not is_unique: continue
                
                # 2. Internal batch uniqueness
                from rapidfuzz import fuzz
                if any(fuzz.token_set_ratio(title.lower(), s.lower()) > 80 for s in seen_titles_in_this_batch):
                    continue
                    
                # 3. Viral Potential Scoring
                title_lower = title.lower()
                hot_score = sum(15 for kw in hot_keywords if kw in title_lower)
                
                # Additional weight for 'Trending' type from NewsAPI
                if art.get("type") == "trending":
                    hot_score += 20
                
                # Keyword density for "Breaking" signals
                breaking_keywords = ["launch", "release", "leak", "breakthrough", "benchmark", "announces", "unveils", "shuts down"]
                hot_score += sum(10 for kw in breaking_keywords if kw in title_lower)
                
                # Recency Boost (Last 24h gets +15)
                pub_at = art.get("publishedAt", "")
                if pub_at:
                    try:
                        pub_dt = datetime.fromisoformat(pub_at.replace('Z', '+00:00'))
                        if datetime.now(timezone.utc) - pub_dt < timedelta(hours=24):
                            hot_score += 15
                    except: pass

                art['_hot_score'] = hot_score
                filtered_articles.append(art)
                seen_titles_in_this_batch.append(title)
            
            if not filtered_articles:
                print("⚠️ No unique viral articles. Falling back to Search...")
                articles = None
            else:
                # Rank by viral score
                filtered_articles.sort(key=lambda x: x.get('_hot_score', 0), reverse=True)
                top = filtered_articles[0]
                print(f"🏆 Top Viral Candidate: '{top.get('title')}' (Score: {top['_hot_score']})")
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
  "retention_cues": [{{"timestamp": 3.0, "effect": "zoom_in", "reason": "hook_impact"}}],
  "subtitle_chunks": [{{
      "chunk_id": 1, "text": "Sentence 1", "start": 0.00, "end": 3.50, 
      "has_infographic": true, "infographic_type": "process|slide", 
      "infographic_data": {{"steps": ["Step 1", "Step 2"]}}
  }}],
  "original_news_headline": "Exact headline",
  "original_news_url": "Direct article URL",
  "keywords": ["AI"],
  "hashtags": ["#AI", "#CyberSecurity", "#DataPrivacy", "#TechNews"],
  "comment_hook": "Provocative question to drive engagement (e.g. 'Which department at your job is leaking the most data?')"
}}"""
    else:
        selection_instruction = (
            f"Analyze the following {content_desc} and pick the SINGLE most impactful story to convert into a 38-44s YouTube Short script.\n"
            f"PRIMARY CATEGORY: {category}\n"
            "SELECTION FILTERS:\n"
            f"1. PRIORITIZE: Stories related to '{category}'. However, if no high-impact story exists for this category today, you ARE AUTHORIZED to pick the single most viral/surprising AI story from any other category instead.\n"
            "2. MUST be New, Useful, or Surprising (Absolute mandatory).\n"
            "3. MUST be explainable in exactly <40s of dense technically-accurate speech (approx 120-140 words total).\n"
            "4. MUST contain one concrete takeaway or engineering tip the viewer can use today.\n"
            "5. PACING: Keep sentences short (under 12 words) for better TTS pacing and Micro-Cut boundaries.\n\n"
            "FORMAT: You MUST follow the strict 5-part structure: Hook -> Problem -> Solution -> Retention Loop -> Call to Action.\n\n"
            "HOOK ALIGNMENT (DROP TEST):\n"
            "If the topic doesn't produce a strong 'Winner' hook (Stat, Absolute Contradiction, or 'You are using this wrong'), DROP IT and pick another.\n"
        )
        prompt_requirements = f"""Return ONLY this exact JSON (no markdown, no explanation):
{{
  "title_options": ["Title Case + Emoji + Curiosity Gap 1", "Title Case + Emoji + Curiosity Gap 2"],
  "description": "Full 100+ word rich SEO description for youtube describing the video, including timestamps and credits.",
  "use_case_evidence_url": "MANDATORY: A direct, valid URL from the 'SOURCES FOUND' section to be used as visual evidence.",
  "title": "Punchy YouTube title max 60 chars",
  "hook_script": "The Hook (0-3s): Start with a Result-First statement. Do not introduce the paper title immediately. Focus on impact (e.g., '90% cheaper', '10x faster'). Approx 10 words.",
  "problem_context": "The Problem (3-10s): Briefly state the bottleneck this research solves. Approx 20 words.",
  "solution_tech": "The Solution (10-45s): Explain the core technical breakthrough using analogies. Keep sentences UNDER 12 words. Approx 80 words.",
  "retention_loop": "The Retention Loop (45-55s): End with a cliffhanger or a seamless bridge that leads back to the start of the video. Approx 15 words.",
  "outro_cta": "Call to Action: Include one engagement trigger (e.g., 'Check the pinned repo' or 'Drop a comment if you'd use this'). Approx 10 words.",
  "script": "The FULL unified voiceover script seamlessly concatenating hook_script, problem_context, solution_tech, retention_loop, and outro_cta into ONE single flowing text block. Target total duration: 38-44 sec (approx 120-140 words).",
  "hook_text": "The exact first 5-8 words of the script.",
  "relevant_links": ["https://github.com/...", "https://arxiv.org/abs/..."],
  "phonetic_pronunciation_map": {{"NVIDIA": "In-vid-yah"}},
  "hook": "Matches the first sentence of the script",
  "summary": "One line summary",
  "sub_category": "AI/Machine Learning",
  "breaking_news_level": 9,
  "retention_cues": [{{"timestamp": 3.0, "effect": "zoom_in", "reason": "hook_impact"}}],
  "subtitle_chunks": [{{
      "chunk_id": 1, "text": "Sentence 1", "start": 0.00, "end": 2.50
  }}],
  "original_news_headline": "Exact headline",
  "original_news_url": "Direct article URL",
  "keywords": ["AI"],
  "hashtags": ["#AI", "#Tech", "#MachineLearning", "#Python", "#OpenAI"],
  "comment_hook": "A custom question targeting the seed audience: 'Which part of this architecture surprised you most? Let's discuss! 👇'"
}}"""

    # Inject any extra instructions (e.g. screenshot avoidance, length adjustments) into context
    if extra_instruction:
        news_context += f"\n\nADDITIONAL INSTRUCTIONS:\n{extra_instruction}\n"

    engine = MultiAgentGenerationEngine(client, news_context, slot, category, strategy_enhancement, is_longform, raw_articles=articles)
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
            return None
            
    return script_data

class MultiAgentGenerationEngine:
    def __init__(self, client, context, slot, category, strategy_enhancement, is_longform, raw_articles=None):
        self.client = client
        self.context = context
        self.slot = slot
        self.category = category
        self.strategy_enhancement = strategy_enhancement
        self.is_longform = is_longform
        self.raw_articles = raw_articles

    def _call_gemini(self, prompt, model='gemini-2.0-flash'):
        attempts = 0
        current_model = model
        while attempts < 5:
            try:
                response = self.client.models.generate_content(
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
                # Handle Overloaded (503) or Resource Exhausted (429)
                if any(x in err_str for x in ["503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "429"]):
                    wait_time = (5 ** (attempts + 1)) + random.uniform(2, 5) # Progressive wait: 7s, 27s, 127s...
                    
                    # Model Fallback Logic
                    if current_model == 'gemini-2.5-pro':
                        print(f"⚠️ [LOOP] Model {current_model} is UNAVAILABLE. Falling back to gemini-2.0-flash...")
                        current_model = 'gemini-2.0-flash'
                        continue
                    elif current_model == 'gemini-2.0-flash' and attempts >= 1:
                        print(f"⚠️ [LOOP] Model {current_model} is OVERLOADED. Falling back to gemini-1.5-flash...")
                        current_model = 'gemini-1.5-flash'
                        continue
                        
                    print(f"⚠️ [LOOP] Call failed ({current_model}): Rate Limit/Overload. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️ [LOOP] Call failed ({current_model}): {e}. Retrying...")
                    time.sleep(3)
                attempts += 1
        return None

    def execute(self, selection_instruction, prompt_requirements):
        print("🎯 [AGENT 0] Selector Agent: Picking the single best story...")
        selector_prompt = SELECTOR_AGENT_TEMPLATE.format(
            persona=SYSTEM_PERSONA,
            selection_instruction=selection_instruction,
            news_context=self.context
        )
        selection = self._call_gemini(selector_prompt)
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

            selected_context = (
                f"STRICT INSTRUCTION: You MUST ONLY research and write about the following story. "
                f"IGNORE all other news articles mentioned in any previous context.\n\n"
                f"TARGET STORY:\n{isolated_context}"
            )
            print(f"✅ Selected Story: {selected_headline}")
            print(f"🔒 Isolated Context for downstream agents: {len(isolated_context)} chars")

        print("🕵️ [AGENT 1] Research Agent: Extracting narrative elements...")
        research_prompt = RESEARCH_AGENT_TEMPLATE.format(
            persona=SYSTEM_PERSONA,
            news_context=selected_context
        )
        research = self._call_gemini(research_prompt)
        if not research: return None

        print("🪝 [AGENT 2] Hook Agent: Generating high-retention hooks...")
        hook_prompt = HOOK_AGENT_TEMPLATE.format(
            persona=SYSTEM_PERSONA,
            research_json=json.dumps(research)
        )
        hooks_data = self._call_gemini(hook_prompt)
        if not hooks_data or "hooks" not in hooks_data: return None
        
        # Pick best hook (highest combined score)
        best_hook = max(hooks_data["hooks"], key=lambda h: h.get("curiosity_score", 0) + h.get("emotional_trigger_score", 0))
        print(f"🎯 Selected Hook: {best_hook.get('text')}")

        print("📖 [AGENT 3] Narrative Agent: Building escalating structure...")
        narrative_prompt = NARRATIVE_AGENT_TEMPLATE.format(
            persona=SYSTEM_PERSONA,
            research_json=json.dumps(research),
            selected_hook=best_hook.get("text"),
            selection_instruction=selection_instruction
        )
        narrative = self._call_gemini(narrative_prompt, model='gemini-2.5-pro')
        if not narrative: return None

        print("⚡ [AGENT 4] Retention Optimizer: Maximizing pacing and curiosity density...")
        retention_prompt = RETENTION_OPTIMIZER_TEMPLATE.format(
            persona=SYSTEM_PERSONA,
            narrative_json=json.dumps(narrative)
        )
        optimized = self._call_gemini(retention_prompt, model='gemini-2.5-pro')
        if not optimized: return None

        print("🗣️ [AGENT 5] Humanizer Agent: Fixing AI cadence and returning final schema...")
        # Inject the selected headline and URL back into the requirements if they are missing
        refined_requirements = prompt_requirements
        if "original_news_headline" in refined_requirements:
            refined_requirements = refined_requirements.replace('"original_news_headline": "Exact headline"', f'"original_news_headline": "{selected_headline}"')
        if "original_news_url" in refined_requirements:
            refined_requirements = refined_requirements.replace('"original_news_url": "Direct article URL"', f'"original_news_url": "{selected_url}"')

        humanizer_prompt = HUMANIZER_AGENT_TEMPLATE.format(
            persona=SYSTEM_PERSONA,
            optimized_script=optimized.get("optimized_script", ""),
            schema_requirements=refined_requirements
        )
        final_script = self._call_gemini(humanizer_prompt, model='gemini-2.5-pro')
        
        if final_script:
            # Final safety check: ensure the headline/url are set correctly in the final object
            if not final_script.get("original_news_headline") or final_script.get("original_news_headline") == "Exact headline":
                final_script["original_news_headline"] = selected_headline
            if not final_script.get("original_news_url") or final_script.get("original_news_url") == "Direct article URL":
                final_script["original_news_url"] = selected_url
            
            print("⭐ [PIPELINE] Multi-Agent script generation completed successfully.")
        return final_script
