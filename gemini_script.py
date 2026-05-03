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

SYSTEM_PERSONA = """Act as a Staff AI Engineer and Technical Architect specializing in Hybrid Architectures and Agentic Design. 
Your goal is to build high-authority technical insights that help developers move from 'generic AI prompts' to 'scalable, cost-optimized agentic systems'.
Prioritize local open-source models (LMMs), hybrid cloud-local routing, and architectural blueprints that replace recurring API costs."""

PLANNING_TEMPLATE = """{persona}

ANALYZER TASK:
Review the following technical news and search context. 
Identify 3-4 distinct 'Technical Angles' or 'Architectural Blueprints' that would be highly valuable for a Staff Engineer.
Consider: Cost-optimization, Local model replacement of APIs, or Agentic Loop efficiency.

NEWS CONTEXT:
{news_context}

Return ONLY a JSON list of objects:
[{{ "angle": "Short title", "insight": "One sentence technical insight", "reason": "Why this matters for devs" }}]"""

CRITIQUE_TEMPLATE = """{persona}

CRITIQUE TASK:
Evaluate the following AI Video Script draft for a professional developer audience.
Identify:
1. TECHNICAL SHALLOW SPOTS: Where did the script remain too generic?
2. FILLER WORDS: Did it use forbidden words (basically, actually, just, etc.)?
3. HOOK VELOCITY: Is the 5-second hook a 'Stop-Your-Scroll' trigger?
4. PERSONA ALIGNMENT: Does it sound like a technical peer briefing or a news reporter?

SCRIPT DRAFT:
{script_json}

Return ONLY a JSON object:
{{
  "score": 0.0-10.0,
  "technical_critique": "Draft improvements here",
  "persona_critique": "Draft improvements here",
  "specific_fixes": ["List of exact changes to make"]
}}"""

REFINEMENT_TEMPLATE = """{persona}

REFINEMENT TASK:
Rewrite the script draft based on the following critique. 
Ensure ALL technical glossary terms are correct (tiktoken, etc.) and ALL filler words are removed.
Increase the 'Staff Engineer' authority.

CRITIQUE:
{critique_json}

ORIGINAL DRAFT:
{original_draft}

Return the FINAL corrected JSON matching the original schema. No explanation."""

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
            "1. MUST be a topic that can be broken down into 3 to 5 actionable tools, techniques, or architectural principles.\n"
            "2. MUST be explainable in 120-180s of expert-level technical speech (approx 350-450 words total).\n"
            "3. MUST contain a highly relatable Hook, a Workflow demonstration, and a strong Conceptual Warning (The Caveat).\n"
            "4. PRIORITIZE: Documentation 'Easter Eggs', optimization hacks, or workflow structures that provide extreme utility.\n"
            "5. VOCAL DYNAMICS: You MUST use heavy punctuation (commas, ellipses '...', exclamation marks, italics, ALL CAPS) around key technical terms and transitions. The TTS engine relies entirely on punctuation to vary pitch and emphasis. Never use plain unpunctuated sentences for important points.\n"
            "FORMAT: You MUST follow the strict 6-part structure: Hook -> Why It Matters -> Core Content (3-5 items) -> Workflow -> Caveat -> Starting Point -> Outro.\n"
        )
        prompt_requirements = f"""Return ONLY this exact JSON (no markdown, no explanation):
{{
  "title_options": ["Title idea 1", "Title idea 2", "Title idea 3"],
  "description": "Full 100+ word rich SEO description for youtube describing the video, including timestamps and credits.",
  "use_case_evidence_url": "MANDATORY: A direct, valid URL from the 'SOURCES FOUND' section to be used as visual evidence.",
  "title": "Punchy YouTube title max 60 chars",
  "hook_script": "30-second personal story / problem setup (approx 60 words). Stark contrast, direct eye contact feel.",
  "section_1_why_it_matters": "Context, data point, and common mistake (approx 60 words).",
  "section_2_core_content": "Break down 3-5 specific tools, architectural principles, or tips based on the topic. Give specific prompts/formulas for each (approx 150 words).",
  "section_3_workflow": "A step-by-step chaining of the core content on a real problem. Fast paced (approx 70 words).",
  "section_4_caveat": "The critical mistake to avoid. Serious tone (approx 40 words).",
  "section_5_starting_point": "One immediate action to take today (approx 30 words).",
  "outro_cta": "Subscribe and comment prompt.",
  "script": "The FULL unified voiceover script seamlessly concatenating hook_script, section_1, section_2, section_3, section_4, section_5, and outro_cta into ONE single flowing text block. Target total word count: 350-450 words.",
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
  "hashtags": ["#AI"],
  "comment_hook": "Would you use this?"
}}"""
    else:
        selection_instruction = (
            f"Analyze the following {content_desc} and pick the SINGLE most impactful story to convert into a 38-44s YouTube Short script.\n"
            "SELECTION FILTERS:\n"
            "1. MUST be New, Useful, or Surprising (Absolute mandatory).\n"
            "2. MUST be explainable in exactly <40s of dense technically-accurate speech (approx 120-140 words total).\n"
            "3. MUST contain one concrete takeaway or engineering tip the viewer can use today.\n"
            "4. PRIORITIZE: Documentation 'Easter Eggs', cost-saving architecture (Local models), or workflow 'unfair advantages' (Agentic loops).\n"
            "5. VOCAL DYNAMICS: You MUST use heavy punctuation (commas, ellipses '...', exclamation marks, italics, ALL CAPS) around key technical terms and transitions. The TTS engine relies entirely on punctuation to vary pitch and emphasis. Never use plain unpunctuated sentences for important points.\n\n"
            "FORMAT: You MUST follow the strict 6-part structure: Hook -> Why It Matters -> Core Content (1-2 items) -> Workflow -> Caveat -> Starting Point -> Outro.\n\n"
            "CONTENT MIX (Algorithm Target):\n"
            "- 30% Practical AI Tools (Automation, Dev tools, SDKs).\n"
            "- 40% Frontier AI Model Releases (OpenAI, DeepMind, Anthropic benchmarks).\n"
            "- 15% Core AI Concepts (RAG, Agents, LLM architecture).\n"
            "- 15% AI Dev Tips (Best practices, optimization, debugging, & engineering hacks).\n\n"
            "HOOK ALIGNMENT (DROP TEST):\n"
            "If the topic doesn't produce a strong 'Winner' hook (Stat, Absolute Contradiction, or 'You are using this wrong'), DROP IT and pick another.\n"
        )
        prompt_requirements = f"""Return ONLY this exact JSON (no markdown, no explanation):
{{
  "title_options": ["Title idea 1", "Title idea 2", "Title idea 3"],
  "description": "Full 100+ word rich SEO description for youtube describing the video, including timestamps and credits.",
  "use_case_evidence_url": "MANDATORY: A direct, valid URL from the 'SOURCES FOUND' section to be used as visual evidence.",
  "title": "Punchy YouTube title max 60 chars",
  "hook_script": "5-second personal story / problem setup (approx 15 words). Stark contrast, direct eye contact feel.",
  "section_1_why_it_matters": "Context, data point, and common mistake (approx 15 words).",
  "section_2_core_content": "Break down 1-2 specific tools, architectural principles, or tips. Give specific prompts/formulas for each (approx 40 words).",
  "section_3_workflow": "A step-by-step chaining of the core content on a real problem. Fast paced (approx 20 words).",
  "section_4_caveat": "The critical mistake to avoid. Serious tone (approx 10 words).",
  "section_5_starting_point": "One immediate action to take today (approx 10 words).",
  "outro_cta": "Subscribe and comment prompt.",
  "script": "The FULL unified voiceover script seamlessly concatenating hook_script, section_1, section_2, section_3, section_4, section_5, and outro_cta into ONE single flowing text block. Target total duration: 38-44 sec (approx 120-140 words).",
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
      "infographic_data": {{"steps": ["Step 1", "Step 2"]}} // If slide: {{"title": "Architecture", "bullet_points": ["Point 1", "Point 2"]}}
  }}],
  "original_news_headline": "Exact headline",
  "original_news_url": "Direct article URL",
  "keywords": ["AI"],
  "hashtags": ["#AI"],
  "comment_hook": "Would you use this?"
}}"""

    engine = AgenticGenerationEngine(client, news_context, slot, category, strategy_enhancement, is_longform)
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

class AgenticGenerationEngine:
    def __init__(self, client, context, slot, category, strategy_enhancement, is_longform):
        self.client = client
        self.context = context
        self.slot = slot
        self.category = category
        self.strategy_enhancement = strategy_enhancement
        self.is_longform = is_longform

    def _call_gemini(self, prompt, model='gemini-2.0-flash'):
        attempts = 0
        while attempts < 3:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.8)
                )
                raw = response.text.strip()
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0].strip()
                
                return json.loads(raw)
            except Exception as e:
                print(f"⚠️ [LOOP] Call failed ({model}): {e}. Retrying...")
                attempts += 1
                time.sleep(2)
        return None

    def plan(self):
        print("🔍 [LOOP] Phase 1: Planning Technical Angles...")
        prompt = PLANNING_TEMPLATE.format(
            persona=SYSTEM_PERSONA,
            news_context=self.context
        )
        return self._call_gemini(prompt)

    def generate_draft(self, plan, selection_instruction, prompt_requirements):
        print("✍️ [LOOP] Phase 2: Generating Initial Draft...")
        full_prompt = f"{SYSTEM_PERSONA}\n\nPLAN SELECTED: {json.dumps(plan)}\n\n{selection_instruction}\n\nDATA:\n{self.context}\n\n{prompt_requirements}"
        return self._call_gemini(full_prompt, model='gemini-2.5-pro')

    def critique(self, draft):
        print("🧐 [LOOP] Phase 3: Critiquing Technical Depth...")
        prompt = CRITIQUE_TEMPLATE.format(
            persona=SYSTEM_PERSONA,
            script_json=json.dumps(draft)
        )
        return self._call_gemini(prompt)

    def refine(self, draft, critique, requirements):
        print("🛠️ [LOOP] Phase 4: Refining & Polishing...")
        prompt = REFINEMENT_TEMPLATE.format(
            persona=SYSTEM_PERSONA,
            critique_json=json.dumps(critique),
            original_draft=json.dumps(draft),
            schema_requirements=requirements
        )
        return self._call_gemini(prompt, model='gemini-2.5-pro')

    def execute(self, selection_instruction, prompt_requirements):
        plan_angles = self.plan()
        selected_angle = plan_angles[0] if (plan_angles and isinstance(plan_angles, list)) else {}

        draft = self.generate_draft(selected_angle, selection_instruction, prompt_requirements)
        if not draft: return None

        iterations = 0
        max_iters = 1 
        while iterations < max_iters:
            feedback = self.critique(draft)
            if not feedback: break
            
            score = feedback.get("score", 0)
            if score >= 9.2:
                print(f"⭐ [LOOP] Quality Score: {score}/10. Ready.")
                break
            
            print(f"🔄 [LOOP] Quality Score: {score}/10. Iterating ({iterations+1}/{max_iters})...")
            refined = self.refine(draft, feedback, prompt_requirements)
            if refined:
                draft = refined
            iterations += 1
            
        return draft
