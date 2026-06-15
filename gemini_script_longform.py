"""
gemini_script_longform.py — Multi-Agent Script Generation for "Did You Know" Compilation Videos.

VIRAL RETENTION OVERHAUL (2026-06-01):
  - Scaled from 5-fact/3-min → 10-fact/8-min format
  - Added: Cold Open, Midpoint Twist, Recap Bumpers, Escalation Labels
  - Enhanced: Retention psychology, pattern interrupts, curiosity cliffhangers
  - Improved: Title CTR formulas, description SEO, chapter markers

Generates an 8-minute, 10-topic, 16:9 landscape "Did You Know" script using a
multi-agent pipeline mirroring the Shorts gemini_script.py architecture.

Agents:
  0. Topic Discovery Agent — Finds top 10 viral AI topics
  1. Research Agent (×10) — Extracts facts per topic
  2. Fact Script Generator (×10) — Writes 35-50s per-fact scripts
  3. Compilation Assembler — Stitches into one flowing 8-min script
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
from config import (
    GEMINI_API_KEY, LOGS_DIR,
    GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL, GEMINI_FLASH_LITE_MODEL,
    GEMINI_RPM_SLEEP
)
from topic_tracker import load_tracker, check_story_uniqueness
from config_longform import (
    LONGFORM_NUM_TOPICS, LONGFORM_PER_TOPIC_DURATION,
    LONGFORM_WORD_COUNT_TARGET, LONGFORM_TARGET_AUDIO_DURATION,
    LONGFORM_TRACKER_FILE, LONGFORM_RECAP_EVERY_N_FACTS,
    LONGFORM_COLD_OPEN_DURATION, LONGFORM_MIDPOINT_TWIST_FACT
)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PERSONA_LONGFORM = """Role: You are an elite AI Research Content Architect who creates "Did You Know" style compilation videos that go VIRAL on YouTube — targeting millions of views and 50%+ average retention.

Format: 16:9 Landscape, 8-minute "Rapid-Fire AI Facts" video.
Structure: 10 mind-blowing AI facts, each lasting 35-50 seconds, stitched with smooth narrative bridges and escalating intensity.

Tone: Authoritative yet conversational. Like a tech-savvy friend who just discovered something INSANE and can't wait to tell you. Think Mark Rober meets Fireship — clear, punchy, and impossible to look away from.

Target Audience: USA-based tech enthusiasts, AI curious professionals, and engineers. Use American English, USD ($), and US-centric analogies.

═══════════════════════════════════════════════════
GOLDEN RULES (NON-NEGOTIABLE):
═══════════════════════════════════════════════════

1. EACH FACT starts with "Did you know..." or a variant ("Here's something insane...", "Most people have no idea that...", "This one blew my mind...", "Nobody is talking about this but...").
2. ESCALATING INTENSITY: Fact 1 = interesting, Fact 5 = surprising, Fact 10 = mind-blowing. The viewer must feel the video gets BETTER with every fact.
3. BRIDGES: Each fact MUST end with a curiosity cliffhanger that FORCES the viewer to keep watching. Never let the viewer feel like a natural stopping point. Examples:
   - "But that's nothing compared to what Google just did..."
   - "And the NEXT fact? It makes this look like child's play."
   - "Wait until you hear what happened three days AFTER this announcement..."
4. NO FLUFF: Zero filler words. No "In this video", "Hello everyone", "Today we explore".
5. COLD OPEN: Start with a 15-second teaser of the MOST SHOCKING fact (Fact 8, 9, or 10) before the intro. Example: "In just a moment, I'll show you something that made the CEO of Google lose sleep. But first..." — then begin Fact 1.
6. VOCAL DYNAMICS: Heavy punctuation (commas, ellipses '...', ALL CAPS) for TTS emphasis. Pause after every technical term with 3+ syllables.
7. EVIDENCE: Each fact MUST reference a specific source (company, paper, benchmark, or study) for authority.
8. PERSONAL STAKES: At least 4 of the 10 facts MUST explain how this affects the viewer personally (job, privacy, daily life, money).
9. PATTERN INTERRUPT EVERY 30 SECONDS: Insert a micro-hook, rhetorical question, or pattern interrupt at least once every 30 seconds. Examples:
   - "Wait... did I hear that right?"
   - "Now pause and think about that for a second."
   - "And here's where it gets really interesting..."
10. POWER OF 3: Each fact should have 3 components — (1) the hook, (2) the evidence/data, (3) the personal impact.
11. RECAP BUMPERS: After every 3 facts, insert a 5-second recap: "So far we've covered [X], [Y], and [Z] — but this NEXT fact completely changes everything."
12. MIDPOINT TWIST: At the halfway mark (after Fact 5), insert a dramatic escalation: "Okay, those first five facts were just the warm-up. Everything from here on out... is on a completely different level."
13. CTA: The final 15 seconds = provocative question + "Drop a comment with which fact shocked you most! Follow me on Telegram for daily AI facts. Link is on my channel home page. And hit subscribe — because tomorrow's video? It's even crazier."
14. NO INFOGRAPHICS: Do not include infographics, flowcharts, or slides in the script structure.
15. CLARITY OVER JARGON: If you use a complex term like 'Quantization', follow it immediately with a simple 3-word analogy (e.g., '...essentially data compression').
16. SUBJECT CLARITY: Always clearly state the primary subject name (e.g., "Ferrari", "IBM") in the first 2 seconds of each fact's hook. Never start with a dangling verb or pronoun without naming the subject aloud first.
17. COMPLETE SENTENCES: Every single sentence MUST be grammatically complete and fully resolved. Never truncate, cut short, or leave a thought unfinished.
18. SMOOTH PHRASING: Avoid awkward phrasing or word salads. Read the script internally to ensure extremely smooth, professional tech-journalist transitions.
19. ESCALATION LABELS: At facts 7, 8, 9, 10 — verbally label the intensity: "This NEXT fact ranks number three on our shock meter..." to gamify the viewing experience.
20. END SCREEN SAFE ZONE: The last 20 seconds of the script must be structured so the video can show YouTube end screen cards without covering important visuals."""


TOPIC_DISCOVERY_TEMPLATE = """{persona}

TOPIC DISCOVERY AGENT TASK:
Search today's AI landscape and find the TOP 10 most viral, surprising, and mind-blowing AI facts, discoveries, or announcements from the last 72 hours.

SOURCES TO ANALYZE:
{news_context}

SELECTION CRITERIA (in order of priority):
1. SHOCK VALUE: "Wait, WHAT?!" reaction. Facts that make people stop scrolling.
2. RECENCY: Happened in the last 24-72 hours. Fresh news > old knowledge.
3. PERSONAL IMPACT: Affects the viewer's job, privacy, money, or daily routine.
4. BIG NAMES: Google, OpenAI, Meta, NVIDIA, Apple, Anthropic, Microsoft, xAI, Amazon = more clicks.
5. CONTROVERSY: Lawsuits, leaks, ethical scandals, unexpected failures.
6. VARIETY: All 10 topics MUST be from DIFFERENT areas of AI (no two about the same company or subfield).

ESCALATION ORDER (10-fact format):
- Topic 1: Interesting/Cool (Warm-up hook)
- Topic 2: Useful/Practical (Value delivery)
- Topic 3: Surprising/Counterintuitive (First pattern interrupt)
- Topic 4: Industry-shaking (Power move)
- Topic 5: Scary/Concerning (First emotional peak)
- Topic 6: Useful/Life-changing tool (Recovery — value delivery)
- Topic 7: Counterintuitive/Underdog (Surprise factor)
- Topic 8: Controversial/Divisive (Debate fuel — drives comments)
- Topic 9: Alarming/Privacy-related (Emotional peak #2)
- Topic 10: Mind-blowing/World-changing (Ultimate climax)

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
      "search_keywords": ["keyword1", "keyword2", "keyword3"],
      "personal_impact": "How this affects the average viewer"
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

Go DEEP. Find:
1. Specific numbers, benchmarks, and data points (e.g., "93% accuracy", "$2.5 billion", "10x faster")
2. Who said what — direct quotes from executives or researchers
3. Timeline — when did this happen, and what's next?
4. Competition context — how does this compare to rivals?
5. Personal impact — how does this affect a software engineer, a student, a business owner?

Return ONLY a JSON object:
{{
  "facts": ["Fact 1", "Fact 2", "Fact 3", "Fact 4", "Fact 5"],
  "controversies": ["Controversy 1"],
  "implications": ["Implication 1", "Implication 2"],
  "key_numbers": ["$2.5 billion", "10x faster", "95% accuracy"],
  "quotes": ["Direct quote from a key figure"],
  "timeline": "When this happened and what's expected next",
  "competitive_context": "How this compares to competitors",
  "core_narrative": "A detailed paragraph summary focusing ONLY on this story."
}}"""


FACT_SCRIPT_TEMPLATE = """{persona}

FACT SCRIPT GENERATOR TASK:
Write a 35-50 second script segment for ONE "Did You Know" fact in an 8-minute, 10-fact compilation video.

FACT #{fact_number} of 10 (Intensity Level: {intensity})
TOPIC: {topic_headline}
SOURCE: {source_url}
RESEARCH CONTEXT:
{research_context}

STRUCTURE (MANDATORY — "Power of 3"):
1. HOOK (3-5s): Start with "Did you know..." or a variant. Pattern interrupt. Use ALL CAPS on the key reveal word. Name the subject (company/person) in the FIRST sentence.
2. CORE FACT + EVIDENCE (12-18s): Explain the discovery or news with a sharp analogy. Reference the specific source, paper, or benchmark. Use concrete numbers. Keep sentences UNDER 15 words.
3. PERSONAL STAKES (8-12s): Why should the viewer care RIGHT NOW? Use "YOU" and "YOUR". Be specific about how this affects their job, privacy, finances, or daily life.
4. BRIDGE (3-5s): Tease the next fact. Create an irresistible open loop that FORCES the viewer to keep watching.
   {bridge_instruction}

{escalation_label_instruction}

WORD COUNT: 100-140 words (for 35-50 seconds of speech at ~2.8 words/second).

VOCAL DYNAMICS:
- Use '...' after technical terms with 3+ syllables (e.g., "Quantization... is essentially...")
- Use ALL CAPS for emphasis on 2-3 key words per fact
- Use commas for natural pauses
- Use exclamation marks for energy spikes
- Keep sentences SHORT. Under 15 words. Punchy.
- Include at least ONE pattern interrupt (rhetorical question, "wait...", "think about that")

Return ONLY a JSON object:
{{
  "fact_number": {fact_number},
  "script": "The full voiceover text for this 35-50 second fact segment. DO NOT include timestamps or speaker labels.",
  "hook_text": "The first 5-8 words of the script",
  "key_stat": "One memorable number or data point (e.g., '10x faster')",
  "personal_impact_sentence": "The single most impactful sentence about how this affects the viewer",
  "shock_level": 8,
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
You have {num_facts} individual fact scripts. Assemble them into ONE seamless, flowing 8-minute voiceover script with smooth transitions and maximum retention architecture.

INDIVIDUAL FACT SCRIPTS:
{fact_scripts_json}

ASSEMBLY RULES (CRITICAL FOR VIRAL PERFORMANCE):

1. COLD OPEN (15s): Start with a teaser of the MOST SHOCKING fact (Fact 8, 9, or 10). Examples:
   - "In just a moment, I'll reveal something that made the CEO of Google lose sleep. But FIRST..."
   - "One of these ten facts is so disturbing, it was ALMOST censored. Stick around to find out which one."
   - "By the end of this video, you'll see AI completely differently. And fact number nine? That's the one that changed everything for ME."
   Then transition: "Let's start with fact number one."

2. FACT SIGNPOSTS: Each fact segment MUST start with the spoken signpost "Fact number [one/two/.../ten]." followed by a comma or ellipsis.

3. BRIDGES: Ensure bridges between facts feel NATURAL, not forced. Remove redundant transitions. Each bridge must create an OPEN LOOP — an unanswered question that can only be resolved by watching the next fact.

4. ESCALATING INTENSITY: Facts 1-3 = warm/interesting, Facts 4-6 = surprising/useful, Facts 7-10 = mind-blowing/alarming.

5. RECAP BUMPERS: After Fact 3 and Fact 6, insert a quick recap:
   - After Fact 3: "So far we've covered [Fact 1 topic], [Fact 2 topic], and [Fact 3 topic]. But this NEXT fact? It completely changes the game."
   - After Fact 6: "We're past the halfway mark, and we've already seen [brief summary]. But honestly... the craziest stuff is still ahead."

6. MIDPOINT TWIST (after Fact 5): Insert a dramatic escalation: "Okay, those first five facts were just the warm-up. Everything from here on out... is on a completely different level."

7. ESCALATION LABELS (Facts 7-10): Verbally label the intensity to gamify the experience:
   - Fact 7: "This next fact ranks number FOUR on our shock meter..."
   - Fact 8: "Fact eight... and this one ranks number THREE..."
   - Fact 9: "This is number TWO on our shock list... and for good reason."
   - Fact 10: "And the NUMBER ONE most mind-blowing AI fact of today..."

8. PATTERN INTERRUPTS: Insert a micro-hook every ~30 seconds. Examples:
   - "Now pause and think about that for a second."
   - "And here's where it gets really interesting..."
   - "Wait... did they really just say that?"

9. OUTRO (20s): After Fact 10, add the CTA:
   "Which fact shocked you the MOST? Drop the number in the comments — I read every single one! If you learned something new today, smash that subscribe button... because tomorrow's video? It's even crazier. And for daily AI facts before they trend, follow me on Telegram... link is on my channel home page. I'll see you in the next one."

10. TOTAL WORD COUNT: {min_words}-{max_words} words (for ~8 minutes at ~2.8 words/second).

11. CRITICAL SUBTITLE RULE: The `subtitle_chunks` array MUST break the script down into extremely small chunks of EXACTLY 1 to 3 words maximum. Do not generate long sentences for subtitles.

12. DYNAMIC ATTRACTIVE TITLES: Generate titles that are EXTREMELY specific to the actual content of this video — not generic. Use the most shocking/recognizable company name or capability from the 10 facts. Max 60 chars + 1 emoji. High-CTR formulas:
    - "10 AI Facts That [Shocking Action] 🤯" (e.g., "10 AI Facts That Keep Engineers Awake 💀")
    - "Did You Know These 10 [Adjective] AI Facts? 😱"
    - "[Company Name] Just Did Something INSANE + 9 More AI Facts 🔥"
    - "10 AI Facts Nobody Is Telling You [Year] 🤫"

Return ONLY this exact JSON:
{{
  "title_options": ["Title 1 (max 60 chars + emoji)", "Title 2", "Title 3", "Title 4", "Title 5"],
  "title": "Best title for YouTube (max 60 chars + emoji, curiosity gap)",
  "description": "Full 300+ word SEO description for YouTube. Must include: fact-by-fact timestamps (e.g., 0:00 Cold Open, 0:15 Fact 1, etc.), credits, SEO keywords, AI disclosure, links to Telegram and WhatsApp.",
  "script": "The FULL unified voiceover script. Cold open + intro + all 10 facts with bridges + recap bumpers + midpoint twist + outro. One continuous flowing text block.",
  "cold_open_fact_number": 9,
  "fact_timestamps": [
    {{"fact_number": 0, "approx_start_seconds": 0, "topic": "Cold Open Teaser"}},
    {{"fact_number": 1, "approx_start_seconds": 15, "topic": "Topic headline"}},
    {{"fact_number": 2, "approx_start_seconds": 55, "topic": "Topic headline"}},
    {{"fact_number": 3, "approx_start_seconds": 95, "topic": "Topic headline"}},
    {{"fact_number": "recap_1", "approx_start_seconds": 130, "topic": "Recap 1-3"}},
    {{"fact_number": 4, "approx_start_seconds": 140, "topic": "Topic headline"}},
    {{"fact_number": 5, "approx_start_seconds": 180, "topic": "Topic headline + Midpoint Twist"}},
    {{"fact_number": 6, "approx_start_seconds": 225, "topic": "Topic headline"}},
    {{"fact_number": "recap_2", "approx_start_seconds": 260, "topic": "Recap 4-6"}},
    {{"fact_number": 7, "approx_start_seconds": 270, "topic": "Topic headline"}},
    {{"fact_number": 8, "approx_start_seconds": 315, "topic": "Topic headline"}},
    {{"fact_number": 9, "approx_start_seconds": 360, "topic": "Topic headline"}},
    {{"fact_number": 10, "approx_start_seconds": 405, "topic": "Topic headline"}},
    {{"fact_number": "outro", "approx_start_seconds": 445, "topic": "CTA + Outro"}}
  ],
  "subtitle_chunks": [
    {{
      "chunk_id": 1,
      "text": "Exactly 1-3 words for subtitle display",
      "start": 0.00,
      "end": 1.50,
      "nano_visual_prompt": "16:9 landscape cinematic visual. Dark tech aesthetic, no text, no faces. 8K photorealistic.",
      "fact_number": 0
    }}
  ],
  "keywords": ["AI", "Did You Know", "Tech Facts", "Machine Learning", "Artificial Intelligence", "AI News 2026", "10 AI Facts"],
  "hashtags": ["#AI", "#DidYouKnow", "#TechFacts", "#MachineLearning", "#ArtificialIntelligence", "#AINews"],
  "comment_hook": "Provocative question for comments (e.g., 'Which fact shocked you the most? Drop the number!')",
  "phonetic_pronunciation_map": {{"NVIDIA": "In-vid-yah", "LLaMA": "Lah-mah"}},
  "best_fact_for_shorts": {{
    "fact_number": 9,
    "reason": "Highest shock value and standalone watchability",
    "hook_for_shorts": "This is just 1 of 10 insane AI facts. Full video linked above!"
  }},
  "retention_cues": [
    {{"timestamp": 3.0, "effect": "zoom_in", "reason": "cold_open_hook"}},
    {{"timestamp": 15.0, "effect": "transition_glitch", "reason": "cold_open_to_fact_1"}},
    {{"timestamp": 55.0, "effect": "transition_glitch", "reason": "fact_1_to_2_bridge"}},
    {{"timestamp": 130.0, "effect": "flash_accent", "reason": "recap_bumper_1"}},
    {{"timestamp": 225.0, "effect": "flash_accent", "reason": "midpoint_twist"}},
    {{"timestamp": 260.0, "effect": "flash_accent", "reason": "recap_bumper_2"}},
    {{"timestamp": 405.0, "effect": "zoom_snap", "reason": "fact_10_climax"}}
  ],
  "original_news_headline": "Compilation: 10 AI Facts - [Today's Date]",
  "original_news_url": "Primary source URL from the most impactful fact",
  "use_case_evidence_url": "Primary source URL from the most impactful fact",
  "metric_popups": [
    {{"text": "1,000 tok/sec", "timestamp": 25.0, "fact_number": 1}},
    {{"text": "93% Zero-Click", "timestamp": 70.0, "fact_number": 2}}
  ]
}}"""


RETENTION_OPTIMIZER_LONGFORM_TEMPLATE = """{persona}

RETENTION OPTIMIZER TASK (LONG-FORM — 8 MINUTES):
This is an 8-minute, 10-fact compilation video. Viewer drop-off is the #1 enemy.
Rewrite the assembled script to MAXIMIZE retention at these critical points:

CRITICAL DROP-OFF POINTS (YouTube analytics data):
- 0:30 mark (after cold open — viewer decides to stay or leave) — MOST IMPORTANT
- 1:00 mark (after Fact 1 — "is this worth my time?")
- 2:00 mark (the "two-minute cliff" — biggest drop-off point for longform)
- 3:00 mark (attention fatigue — needs a STRONG recap bumper)
- 4:00 mark (halfway — needs the midpoint twist to re-engage)
- 5:00 mark (post-midpoint — verify escalation is working)
- 6:00 mark (final third — must feel the climax building)
- 7:00 mark (approaching outro — maintain urgency)

OPTIMIZATIONS:
1. Shorten any sentence over 18 words.
2. Add '...' pauses after every 3+ syllable technical term.
3. Ensure each fact ENDS with an open-loop bridge to the next.
4. Verify the cold open teaser is maximally intriguing (name a specific company or number).
5. Verify recap bumpers are present after Facts 3 and 6.
6. Verify midpoint twist is present after Fact 5.
7. Verify escalation labels are present for Facts 7-10.
8. Remove ALL filler: "basically", "essentially", "actually", "literally", "so", "in fact", "as a matter of fact".
9. Add vocal dynamics: commas, ellipses, exclamation marks, ALL CAPS on key words.
10. Verify TOTAL word count is between {min_words} and {max_words}.
11. DO NOT remove the spoken "Fact number [one/two/.../ten]" signposts at the start of each fact segment.
12. COMPLETE SENTENCES: Ensure every single sentence remains grammatically complete and fully resolved.
13. Pattern interrupts: Ensure at least one micro-hook every 30 seconds (rhetorical question, "wait...", "think about that").

ASSEMBLED SCRIPT:
{assembled_script}

Return ONLY a JSON object:
{{
  "optimized_script": "The full rewritten script with all optimizations applied.",
  "word_count": 1200,
  "estimated_duration_seconds": 430,
  "retention_hooks_added": ["0:30 cold open strengthened", "2:00 pattern interrupt added", "4:00 midpoint twist verified"],
  "sentences_shortened": 12,
  "filler_words_removed": 8
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

CRITICAL RULES:
1. The `subtitle_chunks` array MUST break the script down into extremely small chunks of EXACTLY 1 to 3 words maximum.
2. Every sentence must sound like it's spoken by a real person, not read from a teleprompter.
3. Replace "Furthermore" with "And", "However" with "But", "Additionally" with "Plus".
4. Add contractions: "it is" → "it's", "they are" → "they're", "do not" → "don't".
5. The `best_fact_for_shorts` field MUST identify the single most viral fact for Shorts cross-promotion.
6. YouTube chapter timestamps in `fact_timestamps` MUST be accurate based on word count estimates.

Return ONLY the final JSON object matching the schema. No markdown wrapping. No explanations."""


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

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

    # 3. Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        # Model preference order: gpt-oss-120b -> qwen3-32b -> llama-3.3-70b-versatile
        groq_models = ["gpt-oss-120b", "qwen3-32b", "llama-3.3-70b-versatile"]
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
        print("🔮 Falling back to OpenRouter (meta-llama/llama-3.3-70b-instruct)...")
        try:
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "meta-llama/llama-3.3-70b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()
                return clean_and_parse_json(content)
            else:
                print(f"⚠️ OpenRouter API failed with code {r.status_code}: {r.text}")
        except Exception as e:
            print(f"⚠️ OpenRouter fallback failed: {e}")

    return None

class LongformGenerationEngine:
    """Multi-agent script generation engine for 10-topic "Did You Know" videos."""

    def __init__(self, client, news_context, avoid_list_str):
        self.client = client
        self.news_context = news_context
        self.avoid_list_str = avoid_list_str

    def _call_gemini(self, prompt, model=GEMINI_FLASH_MODEL, use_search=False):
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
                if any(x in err_str for x in ["503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "429", "NOT_FOUND", "404"]):
                    # Gentler exponential backoff capped at 60s
                    wait_time = min((2 ** (attempts + 1)) + random.uniform(2, 5), 60.0)
                    if current_model == GEMINI_PRO_MODEL:
                        print(f"⚠️ [LONGFORM] {current_model} unavailable. Falling back to {GEMINI_FLASH_MODEL}...")
                        current_model = GEMINI_FLASH_MODEL
                        continue
                    elif current_model == GEMINI_FLASH_MODEL and attempts >= 1:
                        print(f"⚠️ [LONGFORM] {current_model} overloaded. Falling back to {GEMINI_FLASH_LITE_MODEL}...")
                        current_model = GEMINI_FLASH_LITE_MODEL
                        continue
                    print(f"⚠️ [LONGFORM] Rate limit ({current_model}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️ [LONGFORM] Call failed ({current_model}): {e}. Retrying...")
                    time.sleep(3)
                attempts += 1
        
        print("🚨 Gemini failed all attempts. Attempting fallback models...")
        fallback_res = call_fallback_model(prompt)
        if fallback_res:
            return fallback_res

        print("🚨 All fallback models failed or not configured.")
        return None

    def _call_gemini_search(self, query):
        """Call Gemini with Google Search grounding (no JSON mode)."""
        attempts = 0
        while attempts < 3:
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_FLASH_MODEL,
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
        """Find the top 10 viral AI topics for today's compilation."""
        print("🔥 [AGENT 0] Topic Discovery: Finding top 10 viral AI facts...")

        # Enrich context with live search
        search_text, search_links = self._call_gemini_search(
            "What are the top 10 most surprising, viral, or breaking AI news stories "
            "in the last 48 hours? Include model launches, benchmarks, controversies, "
            "privacy scandals, open-source breakthroughs, AI regulation, startup funding, "
            "and unexpected AI applications. Be specific with company names and numbers."
        )
        enriched_context = self.news_context
        if search_text:
            links_str = "\n".join(search_links[:20])
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
                       f"Keywords: {', '.join(keywords)}. Focus on specific data points, numbers, " \
                       f"executive quotes, and competitive comparisons."
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
        """Generate a 35-50s script for a single fact."""
        headline = topic.get("headline", "")
        source_url = topic.get("source_url", "")
        
        intensity_map = {
            1: "warm/interesting", 2: "useful/practical", 
            3: "surprising/counterintuitive", 4: "industry-shaking/power move",
            5: "scary/concerning", 6: "useful/life-changing",
            7: "counterintuitive/underdog", 8: "controversial/divisive",
            9: "alarming/privacy-related", 10: "mind-blowing/climax"
        }
        intensity = intensity_map.get(fact_number, "interesting")

        # Bridge instruction varies per fact
        if fact_number < LONGFORM_NUM_TOPICS:
            bridge_instruction = (
                "BRIDGE: End with a curiosity cliffhanger for the NEXT fact. "
                "Example: 'But that's nothing compared to what happens next...' "
                "or 'And it gets even crazier from here...'"
            )
        else:
            bridge_instruction = (
                "BRIDGE: This is the LAST fact (#10). End with a powerful closing statement, "
                "then transition to the CTA. Example: 'And THAT... is what nobody saw coming.'"
            )

        # Escalation labels for facts 7-10
        escalation_label_instruction = ""
        if fact_number >= 7:
            rank = 10 - fact_number + 1  # Fact 7 = rank 4, Fact 10 = rank 1
            escalation_label_instruction = (
                f"ESCALATION LABEL: This fact ranks #{rank} on the shock meter. "
                f"Start with a verbal label like: 'This NEXT fact ranks number {rank} on our shock meter...' "
                f"BEFORE the 'Did you know' hook."
            )

        research_context = json.dumps(research_data, indent=2) if research_data else "No detailed research available."

        prompt = FACT_SCRIPT_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            fact_number=fact_number,
            intensity=intensity,
            topic_headline=headline,
            source_url=source_url,
            research_context=research_context,
            bridge_instruction=bridge_instruction,
            escalation_label_instruction=escalation_label_instruction
        )
        
        print(f"   📝 [AGENT 2.{fact_number}] Generating Fact #{fact_number} script ({intensity})...")
        return self._call_gemini(prompt, model=GEMINI_PRO_MODEL)

    # ── STEP 3: Assemble compilation ─────────────────────────────────────────
    def assemble_compilation(self, fact_scripts):
        """Stitch fact scripts into one seamless 8-minute compilation."""
        num_facts = len(fact_scripts)
        print(f"🎬 [AGENT 3] Compilation Assembler: Stitching {num_facts} facts into one video...")
        
        min_words, max_words = LONGFORM_WORD_COUNT_TARGET

        prompt = COMPILATION_ASSEMBLER_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            fact_scripts_json=json.dumps(fact_scripts, indent=2),
            num_facts=num_facts,
            min_words=min_words,
            max_words=max_words
        )
        return self._call_gemini(prompt, model=GEMINI_PRO_MODEL)

    # ── STEP 4: Optimize retention ───────────────────────────────────────────
    def optimize_retention(self, assembled_script):
        """Maximize retention at critical drop-off points."""
        print("⚡ [AGENT 4] Retention Optimizer: Maximizing pacing for 8-minute video...")
        
        min_words, max_words = LONGFORM_WORD_COUNT_TARGET

        prompt = RETENTION_OPTIMIZER_LONGFORM_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            assembled_script=assembled_script,
            min_words=min_words,
            max_words=max_words
        )
        return self._call_gemini(prompt, model=GEMINI_PRO_MODEL)

    # ── STEP 5: Humanize and finalize ────────────────────────────────────────
    def humanize_and_finalize(self, optimized_script, compilation_data, schema_requirements):
        """Fix AI cadence and return final JSON schema."""
        print("🗣️ [AGENT 5] Humanizer: Fixing AI cadence for 10-fact format...")

        prompt = HUMANIZER_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            optimized_script=optimized_script,
            compilation_data=json.dumps(compilation_data, indent=2),
            schema_requirements=schema_requirements
        )
        return self._call_gemini(prompt, model=GEMINI_PRO_MODEL)

    # ── FULL PIPELINE ────────────────────────────────────────────────────────
    def execute(self):
        """Run the full multi-agent pipeline end-to-end."""
        # 0. Discover topics
        topics = self.discover_topics()
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if not topics or len(topics) < 5:
            print("❌ [LONGFORM] Could not discover enough topics. Aborting.")
            return None

        # 1-2. Research + Generate script per topic
        fact_scripts = []
        successful_topics = []
        for i, topic in enumerate(topics):
            fact_num = i + 1
            
            # Research
            research = self.research_topic(topic)
            if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
            if not research:
                print(f"   ⚠️ Research failed for Fact #{fact_num}. Using headline only.")
                research = {"core_narrative": topic.get("one_liner", ""), "facts": [], "implications": []}
            
            # Generate script
            fact_script = self.generate_fact_script(topic, research, fact_num)
            if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
            if fact_script:
                fact_script["topic"] = topic  # Attach topic metadata
                fact_scripts.append(fact_script)
                successful_topics.append(topic)
                print(f"   ✅ Fact #{fact_num} script generated ({len(fact_script.get('script', '').split())} words)")
            else:
                print(f"   ❌ Fact #{fact_num} script generation failed. Skipping.")

        if len(fact_scripts) < 5:
            print(f"❌ [LONGFORM] Only {len(fact_scripts)}/10 facts generated. Need at least 5. Aborting.")
            return None

        # 3. Assemble compilation
        compilation = self.assemble_compilation(fact_scripts)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if not compilation or "script" not in compilation:
            print("❌ [LONGFORM] Compilation assembly failed. Aborting.")
            return None

        assembled_script = compilation.get("script", "")
        word_count = len(assembled_script.split())
        print(f"   📊 Assembled script: {word_count} words")

        # 4. Optimize retention
        optimized = self.optimize_retention(assembled_script)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if optimized and "optimized_script" in optimized:
            final_script = optimized["optimized_script"]
            opt_word_count = len(final_script.split())
            print(f"   📊 Optimized script: {opt_word_count} words")
            print(f"   📊 Retention hooks added: {optimized.get('retention_hooks_added', [])}")
            print(f"   📊 Filler words removed: {optimized.get('filler_words_removed', 0)}")
        else:
            print("   ⚠️ Retention optimization failed. Using assembled script as-is.")
            final_script = assembled_script

        # 5. Humanize and get final JSON
        min_words, max_words = LONGFORM_WORD_COUNT_TARGET
        schema_requirements = COMPILATION_ASSEMBLER_TEMPLATE.split("Return ONLY this exact JSON:")[1] if "Return ONLY this exact JSON:" in COMPILATION_ASSEMBLER_TEMPLATE else ""

        final_data = self.humanize_and_finalize(final_script, compilation, schema_requirements)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        
        if not final_data:
            print("   ⚠️ Humanizer failed. Using compilation data directly.")
            final_data = compilation
            final_data["script"] = final_script

        # Ensure critical fields are populated
        today_str = datetime.now().strftime("%Y-%m-%d")
        if not final_data.get("original_news_headline") or "Exact" in final_data.get("original_news_headline", ""):
            final_data["original_news_headline"] = f"10 AI Facts That Blew My Mind - {today_str}"
        if not final_data.get("original_news_url") or "Primary" in final_data.get("original_news_url", ""):
            final_data["original_news_url"] = successful_topics[0].get("source_url", "") if successful_topics else ""
        
        # Attach successful topics metadata for downstream use
        final_data["longform_topics"] = successful_topics
        final_data["fact_scripts"] = fact_scripts
        final_data["is_longform"] = True
        final_data["longform_format"] = "did_you_know"
        final_data["num_facts"] = len(successful_topics)

        # Ensure best_fact_for_shorts is populated
        if not final_data.get("best_fact_for_shorts"):
            # Pick the fact with the highest shock level
            best_shock = 0
            best_fact_num = len(successful_topics)
            for fs in fact_scripts:
                sl = fs.get("shock_level", 0)
                if sl > best_shock:
                    best_shock = sl
                    best_fact_num = fs.get("fact_number", best_fact_num)
            final_data["best_fact_for_shorts"] = {
                "fact_number": best_fact_num,
                "reason": "Highest shock level",
                "hook_for_shorts": "This is just 1 of 10 insane AI facts. Full video linked above!"
            }

        print(f"⭐ [LONGFORM PIPELINE] Multi-agent generation completed: "
              f"{len(final_data.get('script', '').split())} words, {len(successful_topics)} facts.")
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
        for idx, art in enumerate(articles[:30]):  # Increased from 25 to 30 for more coverage
            title = art.get('title', '')
            desc = art.get('description', '')
            source = art.get('source', {}).get('name', '')
            url = art.get('url', '')
            news_context += f"\n[{idx+1}] Title: {title}\nDescription: {desc}\nSource: {source}\nURL: {url}\n"
    
    if not news_context:
        news_context = "No RSS articles available. Use Gemini Search to find today's top AI stories."

    # ── Build avoidance list from tracker ─────────────────────────────────
    tracker = load_tracker(tracker_file=LONGFORM_TRACKER_FILE)
    recent_history = tracker.get("history", [])[-30:]  # Increased lookback for 10-fact format
    recent_titles = tracker.get("used_titles", [])[-60:]
    
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
