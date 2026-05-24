"""
gemini_script_longform.py — Multi-Agent Script Generation for "Did You Know" Compilation Videos.

Generates a 3-minute, 5-topic, 16:9 landscape "Did You Know" script using a
multi-agent pipeline mirroring the Shorts gemini_script.py architecture.

Agents:
  0. Topic Discovery Agent — Finds top 5 viral AI topics
  1. Research Agent (×5) — Extracts facts per topic
  2. Fact Script Generator (×5) — Writes 25-35s per-fact scripts
  3. Compilation Assembler — Stitches into one flowing 3-min script
  4. Retention Optimizer — Maximizes pacing at drop-off points
  5. Humanizer — Fixes AI cadence, returns final JSON schema
"""

from google import genai
from google.genai import types
import json
import os
import time
import random
from datetime import datetime
from config import GEMINI_API_KEY, LOGS_DIR
from topic_tracker import load_tracker, check_story_uniqueness
from config_longform import (
    LONGFORM_NUM_TOPICS, LONGFORM_PER_TOPIC_DURATION,
    LONGFORM_WORD_COUNT_TARGET, LONGFORM_TARGET_AUDIO_DURATION,
    LONGFORM_TRACKER_FILE
)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PERSONA_LONGFORM = """Role: You are an elite AI Research Content Architect who creates "Did You Know" style compilation videos that go VIRAL on YouTube.

Format: 16:9 Landscape, 3-minute "Rapid-Fire AI Facts" video.
Structure: 5 mind-blowing AI facts, each lasting 25-35 seconds, stitched with smooth narrative bridges and escalating intensity.

Tone: Authoritative yet conversational. Like a tech-savvy friend who just discovered something INSANE and can't wait to tell you. Think Mark Rober meets Fireship — clear, punchy, and impossible to look away from.

Target Audience: USA-based tech enthusiasts, AI curious professionals, and engineers. Use American English, USD ($), and US-centric analogies.

GOLDEN RULES:
1. EACH FACT starts with "Did you know..." or a variant ("Here's something insane...", "Most people have no idea that...", "This one blew my mind...", "Nobody is talking about this but...").
2. ESCALATING INTENSITY: Fact 1 = interesting, Fact 3 = surprising, Fact 5 = mind-blowing. The viewer must feel the video gets BETTER.
3. BRIDGES: Each fact MUST end with a 1-sentence bridge that teases the next fact, creating an irresistible curiosity loop. Example: "But that's nothing compared to what Google just did..."
4. NO FLUFF: Zero filler words. No "In this video", "Hello everyone", "Today we explore". Start with the first fact IMMEDIATELY after a 5-second meta-hook intro.
5. VOCAL DYNAMICS: Heavy punctuation (commas, ellipses '...', ALL CAPS) for TTS emphasis. Pause after every technical term with 3+ syllables.
6. EVIDENCE: Each fact MUST reference a specific source (company, paper, benchmark, or study) for authority.
7. PERSONAL STAKES: At least 2 of the 5 facts MUST explain how this affects the viewer personally (job, privacy, daily life).
8. CTA: The final 10 seconds = provocative question + "Follow me on Telegram for daily AI facts. Link is on my channel home page. And subscribe for more mind-blowing AI content!"
9. NO INFOGRAPHICS: Do not include infographics, flowcharts, or slides in the script structure.
10. CLARITY OVER JARGON: If you use a complex term like 'Quantization', follow it immediately with a simple 3-word analogy (e.g., '...essentially data compression').
11. SUBJECT CLARITY: Always clearly state the primary subject name (e.g., "Ferrari", "IBM") in the first 2 seconds of each fact's hook. Never start with a dangling verb or pronoun without naming the subject aloud first.
12. COMPLETE SENTENCES: Every single sentence MUST be grammatically complete and fully resolved. Never truncate, cut short, or leave a thought unfinished.
13. SMOOTH PHRASING: Avoid awkward phrasing or word salads. Read the script internally to ensure extremely smooth, professional tech-journalist transitions (e.g., write "The next voice-phishing attack..." instead of "Next fishing attack...")."""


TOPIC_DISCOVERY_TEMPLATE = """{persona}

TOPIC DISCOVERY AGENT TASK:
Search today's AI landscape and find the TOP 5 most viral, surprising, and mind-blowing AI facts, discoveries, or announcements from the last 72 hours.

SOURCES TO ANALYZE:
{news_context}

SELECTION CRITERIA (in order of priority):
1. SHOCK VALUE: "Wait, WHAT?!" reaction. Facts that make people stop scrolling.
2. RECENCY: Happened in the last 24-72 hours. Fresh news > old knowledge.
3. PERSONAL IMPACT: Affects the viewer's job, privacy, money, or daily routine.
4. BIG NAMES: Google, OpenAI, Meta, NVIDIA, Apple, Anthropic = more clicks.
5. CONTROVERSY: Lawsuits, leaks, ethical scandals, unexpected failures.
6. VARIETY: All 5 topics MUST be from DIFFERENT areas of AI (no two about the same company or subfield).

ESCALATION ORDER:
- Topic 1: Interesting/Cool (Warm-up hook)
- Topic 2: Useful/Practical (Value delivery)
- Topic 3: Surprising/Counterintuitive (Pattern interrupt)
- Topic 4: Scary/Concerning (Emotional peak)
- Topic 5: Mind-blowing/World-changing (Climax)

AVOIDANCE LIST (DO NOT select topics similar to these):
{avoid_list}

Return ONLY a JSON object:
{{
  "topics": [
    {{
      "rank": 1,
      "headline": "Exact headline or discovery",
      "source_url": "Direct URL to the source article",
      "source_name": "Publication or Company name",
      "shock_level": 7,
      "category": "Model Launch",
      "one_liner": "One-sentence summary of why this is mind-blowing",
      "search_keywords": ["keyword1", "keyword2", "keyword3"]
    }}
  ]
}}"""


FACT_RESEARCH_TEMPLATE = """{persona}

RESEARCH AGENT TASK:
Extract ONLY the technical facts, data points, and narrative details for the specific story below.
IGNORE all other news stories. Focus on providing the 'isolated truth' for this one story.

TARGET STORY: {target_headline}
SOURCE URL: {source_url}

ADDITIONAL CONTEXT:
{context}

Return ONLY a JSON object:
{{
  "facts": ["Fact 1", "Fact 2", "Fact 3"],
  "controversies": ["Controversy 1"],
  "implications": ["Implication 1", "Implication 2"],
  "key_numbers": ["$2.5 billion", "10x faster", "95% accuracy"],
  "core_narrative": "A one paragraph summary focusing ONLY on this story."
}}"""


FACT_SCRIPT_TEMPLATE = """{persona}

FACT SCRIPT GENERATOR TASK:
Write a 25-35 second script segment for ONE "Did You Know" fact in a compilation video.

FACT #{fact_number} of 5 (Intensity Level: {intensity})
TOPIC: {topic_headline}
SOURCE: {source_url}
RESEARCH CONTEXT:
{research_context}

STRUCTURE (MANDATORY):
1. HOOK (2-3s): Start with "Did you know..." or a variant. Pattern interrupt. Use ALL CAPS on the key reveal word.
2. CORE FACT (8-12s): Explain the discovery or news with a sharp analogy. Keep sentences UNDER 12 words.
3. EVIDENCE (5-8s): Reference the specific source, paper, or benchmark. Mention the company/lab by name.
4. PERSONAL STAKES (5-8s): Why should the viewer care RIGHT NOW? Use "YOU" and "YOUR".
5. BRIDGE (2-3s): Tease the next fact. Create an irresistible open loop.
   {bridge_instruction}

WORD COUNT: 70-95 words (for 25-35 seconds of speech at ~2.7 words/second).

VOCAL DYNAMICS:
- Use '...' after technical terms with 3+ syllables (e.g., "Quantization... is essentially...")
- Use ALL CAPS for emphasis on 2-3 key words per fact
- Use commas for natural pauses
- Use exclamation marks for energy spikes
- Keep sentences SHORT. Under 12 words. Punchy.

Return ONLY a JSON object:
{{
  "fact_number": {fact_number},
  "script": "The full voiceover text for this 25-35 second fact segment. DO NOT include timestamps or speaker labels.",
  "hook_text": "The first 5-8 words of the script",
  "key_stat": "One memorable number or data point (e.g., '10x faster')",
  "nano_visual_prompts": [
    {{
      "sentence": "The exact sentence from the script this visual accompanies",
      "visual_prompt": "Highly specific, cinematic visual description. 16:9 LANDSCAPE format. Dark tech aesthetic, dramatic lighting. NO text in the image. NO faces of real people. Photorealistic, 8K. Example: 'Wide-angle aerial view of a massive data center complex at night, glowing blue server racks visible through glass walls, cinematic fog, dramatic volumetric lighting, 16:9 landscape'",
      "duration_estimate": 4.5
    }}
  ],
  "transition_style": "glitch",
  "source_reference": "Exact source attribution (e.g., 'According to OpenAI's blog post...')"
}}"""


COMPILATION_ASSEMBLER_TEMPLATE = """{persona}

COMPILATION ASSEMBLER TASK:
You have 5 individual fact scripts. Assemble them into ONE seamless, flowing 3-minute voiceover script with smooth transitions.

INDIVIDUAL FACT SCRIPTS:
{fact_scripts_json}

ASSEMBLY RULES:
1. INTRO (5s): Start with a punchy meta-hook. Examples:
   - "Five AI facts that will change how you see the world... starting NOW."
   - "These five AI discoveries... are things NOBODY is talking about."
   - "You won't believe what AI did THIS week. Here are five facts that prove it."
2. Ensure bridges between facts feel NATURAL, not forced. Remove redundant transitions.
3. MAINTAIN escalating intensity (Fact 1 = warm, Fact 5 = mind-blown).
4. At approximately the halfway point (after Fact 3), insert a meta-comment like: "And this next one... this is the one that kept ME up last night."
5. FACT SIGNPOSTS: Each fact segment in the script MUST explicitly start with the spoken signpost 'Fact number [one/two/three/four/five].' followed by a comma or ellipsis for a natural pause (e.g., "Fact number one. Did you know..." or "Fact number two... Ferrari is using...").
6. OUTRO (10s): After Fact 5, add the CTA:
   "Which one shocked you the most? Drop it in the comments! Follow me on Telegram for daily AI facts just like these... link is on my channel home page. And subscribe for more mind-blowing AI content!"
7. TOTAL WORD COUNT: {min_words}-{max_words} words (for ~3 minutes at ~2.7 words/second).
8. CRITICAL SUBTITLE RULE: The `subtitle_chunks` array MUST break the script down into extremely small chunks of EXACTLY 1 to 3 words maximum. Do not generate long sentences for subtitles.
9. DYNAMIC ATTRACTIVE TITLES: The generated "title" and "title_options" MUST be extremely attractive, high-CTR, click-enticing titles (max 65 chars, with 1 emoji) designed dynamically around the most shocking or mind-blowing topics covered in this specific script's 5 facts (e.g. referencing a specific company, fear factor, or insane capability, rather than generic placeholders). Make them custom and highly relevant to your actual news content. Use high-performing dynamic formats such as:
   - "5 AI Facts That [Shocking/Fear Action] (e.g., '5 AI Facts That Keep Engineers Awake At Night 💀')"
   - "5 [Intensity] AI Discoveries That [Benefit/Shock] (e.g., '5 INSANE AI Facts Nobody Is Telling You 🤫')"
   - "This AI Fact [Curiosity Action] (e.g., '5 AI Facts That Prove The Future Is Already Here 🤖')"

Return ONLY this exact JSON:
{{
  "title_options": ["Title 1 (max 70 chars, curiosity gap + emoji)", "Title 2", "Title 3"],
  "title": "Best title for YouTube (max 70 chars, curiosity gap + emoji)",
  "description": "Full 150+ word SEO description for YouTube. Include fact timestamps (e.g., 0:00 Fact 1, 0:30 Fact 2, etc.), credits, and SEO keywords. Include AI disclosure.",
  "script": "The FULL unified voiceover script. Intro + all 5 facts with bridges + outro. One continuous flowing text block.",
  "fact_timestamps": [
    {{"fact_number": 1, "approx_start_seconds": 5, "topic": "Topic headline"}},
    {{"fact_number": 2, "approx_start_seconds": 35, "topic": "Topic headline"}},
    {{"fact_number": 3, "approx_start_seconds": 65, "topic": "Topic headline"}},
    {{"fact_number": 4, "approx_start_seconds": 100, "topic": "Topic headline"}},
    {{"fact_number": 5, "approx_start_seconds": 135, "topic": "Topic headline"}}
  ],
  "subtitle_chunks": [
    {{
      "chunk_id": 1,
      "text": "Exactly 1-3 words for subtitle display",
      "start": 0.00,
      "end": 1.50,
      "nano_visual_prompt": "16:9 landscape cinematic visual for this moment. Dark tech aesthetic, no text, no faces. 8K photorealistic. Must depict the exact subject/entity/concept spoken. Example: 'Satellite view of Earth at night showing glowing city lights and data center hotspots, cinematic 16:9, dark background'",
      "fact_number": 0
    }}
  ],
  "keywords": ["AI", "Did You Know", "Tech Facts", "Machine Learning", "Artificial Intelligence"],
  "hashtags": ["#AI", "#DidYouKnow", "#TechFacts", "#MachineLearning", "#ArtificialIntelligence"],
  "comment_hook": "Provocative question for comments (e.g., 'Which fact shocked you the most? Comment the number!')",
  "phonetic_pronunciation_map": {{"NVIDIA": "In-vid-yah", "LLaMA": "Lah-mah"}},
  "retention_cues": [
    {{"timestamp": 3.0, "effect": "zoom_in", "reason": "intro_hook"}},
    {{"timestamp": 30.0, "effect": "transition_glitch", "reason": "fact_1_to_2_bridge"}},
    {{"timestamp": 60.0, "effect": "transition_glitch", "reason": "fact_2_to_3_bridge"}},
    {{"timestamp": 90.0, "effect": "flash_accent", "reason": "halfway_pattern_interrupt"}},
    {{"timestamp": 100.0, "effect": "transition_glitch", "reason": "fact_3_to_4_bridge"}},
    {{"timestamp": 135.0, "effect": "transition_glitch", "reason": "fact_4_to_5_bridge"}}
  ],
  "original_news_headline": "Compilation: 5 AI Facts - [Today's Date]",
  "original_news_url": "Primary source URL from Fact 1",
  "use_case_evidence_url": "Primary source URL from the most impactful fact",
  "metric_popups": [
    {{"text": "1,000 tok/sec", "timestamp": 8.5, "fact_number": 1}},
    {{"text": "93% Zero-Click", "timestamp": 42.0, "fact_number": 3}}
  ]
}}"""


RETENTION_OPTIMIZER_LONGFORM_TEMPLATE = """{persona}

RETENTION OPTIMIZER TASK (LONG-FORM):
This is a 3-minute compilation video. Viewer drop-off is the #1 enemy.
Rewrite the assembled script to MAXIMIZE retention at these critical points:

CRITICAL DROP-OFF POINTS:
- 0:30 mark (after first fact — viewer decides to stay or leave)
- 1:00 mark (the "minute barrier" — must feel like progress)
- 1:30 mark (halfway — needs a STRONG pattern interrupt)
- 2:30 mark (final stretch — must feel the climax building)

OPTIMIZATIONS:
1. Shorten any sentence over 15 words.
2. Add '...' pauses after every 3+ syllable technical term.
3. Ensure each fact ENDS with an open-loop bridge to the next.
4. At approximately the halfway mark, ensure there is a meta-comment like:
   "And this next one... this is the one that kept ME up last night."
5. Remove ALL filler: "basically", "essentially", "actually", "literally", "so".
6. Add vocal dynamics: commas, ellipses, exclamation marks, ALL CAPS on key words.
7. Verify TOTAL word count is between {min_words} and {max_words}.
8. DO NOT remove the spoken "Fact number [one/two/three/four/five]" signposts at the start of each fact segment.
9. COMPLETE SENTENCES: Ensure every single sentence remains grammatically complete and fully resolved. Never truncate, cut off, or leave a phrase half-finished.

ASSEMBLED SCRIPT:
{assembled_script}

Return ONLY a JSON object:
{{
  "optimized_script": "The full rewritten script with all optimizations applied.",
  "word_count": 450,
  "estimated_duration_seconds": 175,
  "retention_hooks_added": ["0:30 bridge strengthened", "1:00 pattern interrupt added"]
}}"""


HUMANIZER_TEMPLATE = """{persona}

HUMANIZER AGENT TASK:
This is the final step. Fix robotic phrasing, repetitive AI wording, over-explanation, and "In conclusion" style endings.
Add contractions, punchier cadence, and conversational flow.
Format the output EXACTLY matching the required schema below.

OPTIMIZED SCRIPT:
{optimized_script}

FULL COMPILATION DATA:
{compilation_data}

SCHEMA REQUIREMENTS:
{schema_requirements}

CRITICAL SUBTITLE RULE: The `subtitle_chunks` array MUST break the script down into extremely small chunks of EXACTLY 1 to 3 words maximum. Do not generate long sentences for subtitles.

Return ONLY the final JSON object matching the schema. No markdown wrapping. No explanations."""


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class LongformGenerationEngine:
    """Multi-agent script generation engine for 5-topic "Did You Know" videos."""

    def __init__(self, client, news_context, avoid_list_str):
        self.client = client
        self.news_context = news_context
        self.avoid_list_str = avoid_list_str

    def _call_gemini(self, prompt, model='gemini-2.0-flash', use_search=False):
        """Call Gemini with retry logic and model fallback."""
        attempts = 0
        current_model = model
        while attempts < 5:
            try:
                config_kwargs = {
                    'temperature': 0.8,
                    'response_mime_type': 'application/json'
                }
                if use_search:
                    config_kwargs['tools'] = [{'google_search': {}}]
                    del config_kwargs['response_mime_type']  # Search doesn't support JSON mode

                response = self.client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs)
                )
                raw = response.text.strip()
                if "{" in raw and "}" in raw:
                    raw = raw[raw.find("{"):raw.rfind("}") + 1]
                return json.loads(raw)
            except Exception as e:
                err_str = str(e).upper()
                if any(x in err_str for x in ["503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "429"]):
                    wait_time = (5 ** (attempts + 1)) + random.uniform(2, 5)
                    if current_model == 'gemini-2.5-pro':
                        print(f"⚠️ [LONGFORM] {current_model} unavailable. Falling back to gemini-2.0-flash...")
                        current_model = 'gemini-2.0-flash'
                        continue
                    elif current_model == 'gemini-2.0-flash' and attempts >= 1:
                        print(f"⚠️ [LONGFORM] {current_model} overloaded. Falling back to gemini-2.5-flash...")
                        current_model = 'gemini-2.5-flash'
                        continue
                    print(f"⚠️ [LONGFORM] Rate limit ({current_model}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️ [LONGFORM] Call failed ({current_model}): {e}. Retrying...")
                    time.sleep(3)
                attempts += 1
        return None

    def _call_gemini_search(self, query):
        """Call Gemini with Google Search grounding (no JSON mode)."""
        attempts = 0
        while attempts < 3:
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=query,
                    config=types.GenerateContentConfig(
                        tools=[{'google_search': {}}]
                    )
                )
                # Extract grounding links
                grounding_links = []
                if response.candidates and response.candidates[0].grounding_metadata:
                    gm = response.candidates[0].grounding_metadata
                    if hasattr(gm, 'grounding_chunks'):
                        for chunk in gm.grounding_chunks:
                            if hasattr(chunk, 'web') and chunk.web.uri:
                                uri = chunk.web.uri
                                if any(x in uri.lower() for x in ["google.com/search", "bing.com/search", ".pdf"]):
                                    continue
                                grounding_links.append(f"{chunk.web.title}: {uri}")
                return response.text, grounding_links
            except Exception as e:
                err_str = str(e).upper()
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = (attempts + 1) * 5
                    print(f"⚠️ [LONGFORM SEARCH] Rate limited. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"⚠️ [LONGFORM SEARCH] Failed: {e}")
                    return "", []
                attempts += 1
        return "", []

    # ── STEP 0: Topic Discovery ──────────────────────────────────────────────
    def discover_topics(self):
        """Find the top 5 viral AI topics for today's compilation."""
        print("🔥 [AGENT 0] Topic Discovery: Finding top 5 viral AI facts...")

        # Enrich context with live search
        search_text, search_links = self._call_gemini_search(
            "What are the top 5 most surprising, viral, or breaking AI news stories "
            "in the last 48 hours? Include model launches, benchmarks, controversies, "
            "privacy scandals, and open-source breakthroughs."
        )
        enriched_context = self.news_context
        if search_text:
            links_str = "\n".join(search_links[:15])
            enriched_context += (
                f"\n\nGEMINI SEARCH RESULTS (Live, Grounded):\n{search_text}\n\n"
                f"SOURCES FOUND:\n{links_str}\n"
            )

        prompt = TOPIC_DISCOVERY_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            news_context=enriched_context,
            avoid_list=self.avoid_list_str
        )
        result = self._call_gemini(prompt)
        if not result or "topics" not in result:
            print("⚠️ Topic Discovery failed. Falling back to search-only topics...")
            return None
        
        topics = result["topics"][:LONGFORM_NUM_TOPICS]
        for i, t in enumerate(topics):
            print(f"   📌 Fact {i+1}: {t.get('headline', 'Unknown')} (Shock: {t.get('shock_level', '?')}/10)")
        return topics

    # ── STEP 1: Research per topic ───────────────────────────────────────────
    def research_topic(self, topic):
        """Deep-research a single topic for fact extraction."""
        headline = topic.get("headline", "")
        source_url = topic.get("source_url", "")
        keywords = topic.get("search_keywords", [])
        
        print(f"   🔬 Researching: {headline}")

        # Enrich with targeted search
        search_query = f"Latest technical details, benchmarks, and implications about: {headline}. " \
                       f"Keywords: {', '.join(keywords)}. Focus on specific data points and numbers."
        search_text, search_links = self._call_gemini_search(search_query)
        
        context = f"HEADLINE: {headline}\nSOURCE: {source_url}\n"
        if search_text:
            context += f"\nSEARCH RESULTS:\n{search_text}\n"
            context += f"\nSOURCES:\n" + "\n".join(search_links[:5])

        prompt = FACT_RESEARCH_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            target_headline=headline,
            source_url=source_url,
            context=context
        )
        return self._call_gemini(prompt)

    # ── STEP 2: Generate script for one fact ─────────────────────────────────
    def generate_fact_script(self, topic, research_data, fact_number):
        """Generate a 25-35s script for a single fact."""
        headline = topic.get("headline", "")
        source_url = topic.get("source_url", "")
        
        intensity_map = {1: "warm/interesting", 2: "useful/practical", 
                         3: "surprising/counterintuitive", 4: "scary/concerning", 
                         5: "mind-blowing/climax"}
        intensity = intensity_map.get(fact_number, "interesting")

        # Bridge instruction varies per fact
        if fact_number < LONGFORM_NUM_TOPICS:
            bridge_instruction = (
                "BRIDGE: End with a teaser sentence for the NEXT fact. "
                "Example: 'But that's nothing compared to what happens next...' "
                "or 'And it gets even crazier from here...'"
            )
        else:
            bridge_instruction = (
                "BRIDGE: This is the LAST fact. End with a powerful closing statement, "
                "then transition to the CTA. Example: 'And THAT... is what nobody saw coming.'"
            )

        research_context = json.dumps(research_data, indent=2) if research_data else "No detailed research available."

        prompt = FACT_SCRIPT_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            fact_number=fact_number,
            intensity=intensity,
            topic_headline=headline,
            source_url=source_url,
            research_context=research_context,
            bridge_instruction=bridge_instruction
        )
        
        print(f"   📝 [AGENT 2.{fact_number}] Generating Fact #{fact_number} script...")
        return self._call_gemini(prompt, model='gemini-2.5-pro')

    # ── STEP 3: Assemble compilation ─────────────────────────────────────────
    def assemble_compilation(self, fact_scripts):
        """Stitch 5 fact scripts into one seamless 3-minute compilation."""
        print("🎬 [AGENT 3] Compilation Assembler: Stitching 5 facts into one video...")
        
        min_words, max_words = LONGFORM_WORD_COUNT_TARGET

        prompt = COMPILATION_ASSEMBLER_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            fact_scripts_json=json.dumps(fact_scripts, indent=2),
            min_words=min_words,
            max_words=max_words
        )
        return self._call_gemini(prompt, model='gemini-2.5-pro')

    # ── STEP 4: Optimize retention ───────────────────────────────────────────
    def optimize_retention(self, assembled_script):
        """Maximize retention at critical drop-off points."""
        print("⚡ [AGENT 4] Retention Optimizer: Maximizing pacing...")
        
        min_words, max_words = LONGFORM_WORD_COUNT_TARGET

        prompt = RETENTION_OPTIMIZER_LONGFORM_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            assembled_script=assembled_script,
            min_words=min_words,
            max_words=max_words
        )
        return self._call_gemini(prompt, model='gemini-2.5-pro')

    # ── STEP 5: Humanize and finalize ────────────────────────────────────────
    def humanize_and_finalize(self, optimized_script, compilation_data, schema_requirements):
        """Fix AI cadence and return final JSON schema."""
        print("🗣️ [AGENT 5] Humanizer: Fixing AI cadence...")

        prompt = HUMANIZER_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            optimized_script=optimized_script,
            compilation_data=json.dumps(compilation_data, indent=2),
            schema_requirements=schema_requirements
        )
        return self._call_gemini(prompt, model='gemini-2.5-pro')

    # ── FULL PIPELINE ────────────────────────────────────────────────────────
    def execute(self):
        """Run the full multi-agent pipeline end-to-end."""
        # 0. Discover topics
        topics = self.discover_topics()
        if not topics or len(topics) < 3:
            print("❌ [LONGFORM] Could not discover enough topics. Aborting.")
            return None

        # 1-2. Research + Generate script per topic
        fact_scripts = []
        successful_topics = []
        for i, topic in enumerate(topics):
            fact_num = i + 1
            
            # Research
            research = self.research_topic(topic)
            if not research:
                print(f"   ⚠️ Research failed for Fact #{fact_num}. Using headline only.")
                research = {"core_narrative": topic.get("one_liner", ""), "facts": [], "implications": []}
            
            # Generate script
            fact_script = self.generate_fact_script(topic, research, fact_num)
            if fact_script:
                fact_script["topic"] = topic  # Attach topic metadata
                fact_scripts.append(fact_script)
                successful_topics.append(topic)
                print(f"   ✅ Fact #{fact_num} script generated ({len(fact_script.get('script', '').split())} words)")
            else:
                print(f"   ❌ Fact #{fact_num} script generation failed. Skipping.")

        if len(fact_scripts) < 3:
            print(f"❌ [LONGFORM] Only {len(fact_scripts)}/5 facts generated. Need at least 3. Aborting.")
            return None

        # 3. Assemble compilation
        compilation = self.assemble_compilation(fact_scripts)
        if not compilation or "script" not in compilation:
            print("❌ [LONGFORM] Compilation assembly failed. Aborting.")
            return None

        assembled_script = compilation.get("script", "")
        word_count = len(assembled_script.split())
        print(f"   📊 Assembled script: {word_count} words")

        # 4. Optimize retention
        optimized = self.optimize_retention(assembled_script)
        if optimized and "optimized_script" in optimized:
            final_script = optimized["optimized_script"]
            print(f"   📊 Optimized script: {len(final_script.split())} words")
        else:
            print("   ⚠️ Retention optimization failed. Using assembled script as-is.")
            final_script = assembled_script

        # 5. Humanize and get final JSON
        min_words, max_words = LONGFORM_WORD_COUNT_TARGET
        schema_requirements = COMPILATION_ASSEMBLER_TEMPLATE.split("Return ONLY this exact JSON:")[1] if "Return ONLY this exact JSON:" in COMPILATION_ASSEMBLER_TEMPLATE else ""

        final_data = self.humanize_and_finalize(final_script, compilation, schema_requirements)
        
        if not final_data:
            print("   ⚠️ Humanizer failed. Using compilation data directly.")
            final_data = compilation
            final_data["script"] = final_script

        # Ensure critical fields are populated
        today_str = datetime.now().strftime("%Y-%m-%d")
        if not final_data.get("original_news_headline") or "Exact" in final_data.get("original_news_headline", ""):
            final_data["original_news_headline"] = f"5 AI Facts That Blew My Mind - {today_str}"
        if not final_data.get("original_news_url") or "Primary" in final_data.get("original_news_url", ""):
            final_data["original_news_url"] = successful_topics[0].get("source_url", "") if successful_topics else ""
        
        # Attach successful topics metadata for downstream use
        final_data["longform_topics"] = successful_topics
        final_data["fact_scripts"] = fact_scripts
        final_data["is_longform"] = True
        final_data["longform_format"] = "did_you_know"

        print(f"⭐ [LONGFORM PIPELINE] Multi-agent generation completed: {len(final_data.get('script', '').split())} words, {len(successful_topics)} facts.")
        return final_data


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_longform_script(articles=None, failed_topics=None):
    """Main entry point for long-form "Did You Know" script generation."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    if failed_topics is None:
        failed_topics = []

    # ── Build news context from RSS articles ──────────────────────────────
    news_context = ""
    if articles:
        for idx, art in enumerate(articles[:25]):
            title = art.get('title', '')
            desc = art.get('description', '')
            source = art.get('source', {}).get('name', '')
            url = art.get('url', '')
            news_context += f"\n[{idx+1}] Title: {title}\nDescription: {desc}\nSource: {source}\nURL: {url}\n"
    
    if not news_context:
        news_context = "No RSS articles available. Use Gemini Search to find today's top AI stories."

    # ── Build avoidance list from tracker ─────────────────────────────────
    tracker = load_tracker(tracker_file=LONGFORM_TRACKER_FILE)
    recent_history = tracker.get("history", [])[-20:]
    recent_titles = tracker.get("used_titles", [])[-40:]
    
    avoid_items = [h.get('news_headline', h.get('title')) for h in recent_history] + recent_titles
    if failed_topics:
        avoid_items += failed_topics
    combined_avoid = list(set([a for a in avoid_items if a]))
    avoid_list_str = "\n".join([f"- {t}" for t in combined_avoid]) if combined_avoid else "None"

    # Add avoidance instruction to context
    if combined_avoid:
        news_context = (
            f"CRITICAL: RECENTLY COVERED STORIES (DO NOT REPEAT THESE TOPICS):\n{avoid_list_str}\n\n"
            + news_context
        )

    # ── Run multi-agent pipeline ──────────────────────────────────────────
    engine = LongformGenerationEngine(client, news_context, avoid_list_str)
    script_data = engine.execute()

    if script_data:
        # Final uniqueness safeguard
        headline = script_data.get("original_news_headline", "")
        news_url = script_data.get("original_news_url", "")
        is_unique, msg = check_story_uniqueness(headline, new_url=news_url, tracker_file=LONGFORM_TRACKER_FILE)
        if not is_unique:
            print(f"⚠️ [LONGFORM] Post-generation uniqueness check failed: {msg}")
            # For compilations this is less strict — we allow it since the headline is synthetic

    return script_data
