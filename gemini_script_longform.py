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
# PROMPT TEMPLATES (VAIBHAV SISINTY FORMAT)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PERSONA_LONGFORM = """Role: You are an elite AI Content Architect and growth hacker who creates high-utility, fast-paced, "no-fluff" YouTube long-form videos in the style of Vaibhav Sisinty — targeting millions of views, extremely active comment sections, and 50%+ average retention from professionals, creators, and founders.

Format: 16:9 Landscape, 7-9 minutes, fast-paced, high-utility explainer video.
Structure:
- A main topic deep-dive (approx 4-5 minutes) covering: Hook -> The Problem -> The Framework -> The Live Demo (Prompts & workflow) -> The Payoff/Result.
- A news roundup (approx 3-4 minutes) covering 5-8 hot AI updates or tool releases from the last 72 hours, adding massive value and scope to the video.

Tone: High-energy, tactical, authoritative yet conversational. Like a growth-focused founder or tech product manager who has just optimized a workflow and is showing you exactly how to do it. Bias toward action: "Build, don't talk" and "Taste over tech".

Target Audience: Founders, creators, knowledge workers, and professionals from high-RPM countries (USA, UK, Canada, Australia, Singapore, India). Use standard English, clean numbers, and clear analogies. Use commas and ellipses '...' for natural speech pacing in TTS.

HIGH-RETENTION VIDEO CONCEPTS (MANDATORY FRAMING):
The main topic deep-dive must always fit into one of these three series format structures:
1. The "Visual Code Archaeology" Series:
   - Hook archetype: "How [Famous App/Game/System] handles [Massive Problem] in just 50 lines of code."
   - Focus: Simplify production engineering. Visualize the DAG of Git, Spotify's music recommendations, or Doom (1993) rendering 3D graphics on 90s hardware.
2. "AI Battles: Model vs. Model" Series:
   - Hook archetype: "We forced 5 AI models to build the same application. The results were terrifying."
   - Focus: Side-by-side programmatic LLM execution, comparing GPT-4o, Gemini, and DeepSeek on code quality, errors, and MLOps benchmarks.
3. "System Design Chronicles" Series:
   - Hook archetype: "What happens inside Netflix's servers when millions of people hit 'Play' at the exact same millisecond?"
   - Focus: Infrastructure, vector DBs vs SQL, scaling secrets (Netflix, WhatsApp servers handling billions of messages without crashes).

CRUCIAL RETENTION RULES:
- The 3-Second Rule: First sentence must immediately state high stakes (e.g. "This line of code caused a $500 million dollar blackout..."). NO welcome messages, intros, or channel name references.
- Visual Reset Rule: Specify visual cues, camera zoom, text pops, background color shift, or code highlighting at least every 4 seconds in the visual segments.
- Cut the Fluff: Programmatically strip out any phrase or sentence that doesn't actively advance the technical value or story."""


TOPIC_DISCOVERY_TEMPLATE = """{persona}

TOPIC DISCOVERY AGENT TASK:
Search today's AI landscape and discover:
1. One MAIN BREAKOUT topic (the main_topic): This must be a major AI release, a revolutionary tool, an insane prompt hack, or a life-changing automated workflow with massive utility for professionals or creators.
2. 5 to 7 SECONDARY topics (news_updates): These must be hot AI updates, tool launches, or interesting tech facts from the last 72 hours.

SOURCES TO ANALYZE:
{news_context}

SELECTION CRITERIA (in order of priority):
- High utility: solves a real pain point (saving time, making money, future-proofing jobs).
- Urgent/Breaking: happened in the last 24-72 hours.
- Mass appeal: avoid dry developer-only tutorials, PC overclocking, or academic benchmarks without consumer application.

AVOIDANCE LIST (DO NOT select topics similar to these):
{avoid_list}

Return ONLY a JSON object:
{{
  "main_topic": {{
    "headline": "Headline or breakout tool name",
    "source_url": "Direct source article URL",
    "source_name": "Company/source name",
    "one_liner": "Why this tool/hack is a game-changer for professionals",
    "search_keywords": ["keyword1", "keyword2", "keyword3"],
    "content_pillar": "AI Tool Spotlight"
  }},
  "news_updates": [
    {{
      "rank": 2,
      "headline": "Secondary news update or tool headline",
      "source_url": "Direct source URL",
      "source_name": "Company/source name",
      "one_liner": "Quick update summary",
      "search_keywords": ["keyword1", "keyword2"]
    }}
  ]
}}"""


FACT_RESEARCH_TEMPLATE = """{persona}

RESEARCH AGENT TASK:
Extract ONLY the technical facts, data points, and narrative details for the specific story below.

TARGET STORY: {target_headline}
SOURCE URL: {source_url}

ADDITIONAL CONTEXT:
{context}

Go DEEP. Find:
1. Specific steps, inputs, or workflow description.
2. Prompts used or how a user can replicate it.
3. Competitor comparison or background context.
4. Key numbers (e.g. 10x faster, $10/month, free).

Return ONLY a JSON object:
{{
  "details": ["Detail 1", "Detail 2"],
  "prompts_or_steps": ["Step 1", "Step 2"],
  "competitor_context": "How it compares to competitors",
  "implications": ["Implication 1", "Implication 2"],
  "facts": ["Fact 1", "Fact 2"],
  "core_narrative": "A detailed paragraph summary focusing ONLY on this story."
}}"""


DEEP_DIVE_SCRIPT_TEMPLATE = """{persona}

DEEP DIVE SCRIPT ARCHITECT TASK:
Write the main deep-dive segment (4-5 minutes) for a Vaibhav Sisinty-style YouTube video.

MAIN TOPIC: {headline}
RESEARCH CONTEXT:
{research_context}

STRUCTURE (MANDATORY):
1. THE COLD OPEN & HOOK (0:00 - 0:45): Challenge a common belief or state a bold, slightly alarming claim about the tool/breakthrough. Make it high stakes. Name the tool immediately. (approx 100-120 words)
2. THE PROBLEM/GAP (0:45 - 1:30): Explain why the traditional way of doing this task (or using basic AI prompts) is slow, expensive, or outdated. (approx 100-120 words)
3. THE FRAMEWORK/SECRET SAUCE (1:30 - 2:30): Introduce a specific framework, rule, or concept (e.g. GPS Framework, Loop Engineering, Magic Prompt formula) to solve the problem. Explain the theory simply. (approx 130-150 words)
4. THE LIVE DEMO/EXECUTION (2:30 - 4:15): Walk through the exact step-by-step prompts, tools, or inputs. Describe the screen action clearly so the video generator can place visuals/mockups. (approx 200-240 words)
5. THE PAYOFF/RESULT (4:15 - 5:00): Show the concrete result — time saved, money earned, or business automated. (approx 100-120 words)

VOCAL DYNAMICS:
- Use heavy punctuation (..., ALL CAPS, exclamation points, commas) to vary TTS pitch/cadence.
- Short, punchy sentences. Under 15 words.
- No filler words. Bias toward action.

Return ONLY a JSON object:
{{
  "headline": "{headline}",
  "hook_script": "The Hook section text.",
  "problem_script": "The Problem section text.",
  "framework_script": "The Framework section text.",
  "demo_script": "The Demo section text.",
  "payoff_script": "The Payoff section text.",
  "script": "The full concatenated deep-dive script.",
  "key_prompt_used": "The primary prompt or workflow highlight",
  "nano_visual_prompts": [
    {{
      "sentence": "Exact sentence from the script",
      "visual_prompt": "16:9 cinematic landscape visual prompt (no text, no real faces, dark tech aesthetic).",
      "duration_estimate": 4.5
    }}
  ]
}}"""


UPDATE_SCRIPT_TEMPLATE = """{persona}

NEWS UPDATE SCRIPT GENERATOR TASK:
Write a rapid-fire 15-20 second news update segment.

UPDATE TOPIC: {headline}
RESEARCH CONTEXT:
{research_context}

RULES:
1. Start with a direct signpost like "Next up..." or "In other news...".
2. Explain the update simply with 1 key metric or detail.
3. Focus on utility/impact for professionals.
4. Word count: 40-50 words.

Return ONLY a JSON object:
{{
  "headline": "{headline}",
  "script": "The full voiceover script for this update.",
  "nano_visual_prompts": [
    {{
      "sentence": "Exact sentence from the script",
      "visual_prompt": "16:9 cinematic landscape visual prompt.",
      "duration_estimate": 5.0
    }}
  ]
}}"""


COMPILATION_ASSEMBLER_TEMPLATE = """{persona}

COMPILATION ASSEMBLER TASK:
Stitch the main deep-dive script and the news updates into a single cohesive, high-retention 7-9 minute voiceover script.

DEEP DIVE SEGMENT:
{deep_dive_json}

NEWS UPDATES SEGMENTS:
{updates_json}

ASSEMBLY RULES:
1. Transition smoothly from the deep-dive payoff to the news updates using a bridge like: "But [Main Tool] isn't the only thing changing the game this week. Here are [X] more massive AI updates you need to know to stay ahead."
2. Label each update clearly (e.g. "Update number one: ...", "Update number two: ...") to maintain a structured pacing.
3. After the last update, transition to the Outro/CTA: "Which of these updates or tools are you going to use first? Let me know in the comments below — I read every single one! If you want my exact prompts, playbooks, and tool lists, join my Telegram community. The link is on my channel homepage. Make sure to subscribe to stay ahead, and I'll see you in the next one."
4. Total word count target: {min_words}-{max_words} words (~7-9 minutes of speech).
5. SUBTITLES: The `subtitle_chunks` array must break the script down into small chunks of exactly 1 to 3 words maximum.
6. Generate dynamic and attractive titles cases for professionals and creators.

Return ONLY this exact JSON:
{{
  "title_options": ["Curiosity-gap title 1", "Curiosity-gap title 2", "Curiosity-gap title 3", "Curiosity-gap title 4"],
  "title": "Main YouTube Title (max 60 chars + emoji)",
  "description": "Full 300+ word rich SEO description including stamps, credits, Telegram/WhatsApp links, and resources.",
  "script": "The full concatenated continuous script.",
  "fact_timestamps": [
    {{"section": "Intro", "approx_start_seconds": 0, "topic": "Introduction & The Paradox"}},
    {{"section": "Problem", "approx_start_seconds": 45, "topic": "Why basic prompts fail"}},
    {{"section": "Framework", "approx_start_seconds": 90, "topic": "The framework/solution"}},
    {{"section": "Demo", "approx_start_seconds": 150, "topic": "Live walkthrough & prompts"}},
    {{"section": "Updates Intro", "approx_start_seconds": 300, "topic": "Rapid AI News Roundup"}},
    {{"section": "Update 1", "approx_start_seconds": 315, "topic": "Update 1 topic"}},
    {{"section": "Outro", "approx_start_seconds": 440, "topic": "Outro & Resources"}}
  ],
  "subtitle_chunks": [
    {{
      "chunk_id": 1,
      "text": "1-3 words subtitle",
      "start": 0.00,
      "end": 1.50,
      "nano_visual_prompt": "16:9 landscape visual prompt."
    }}
  ],
  "keywords": ["AI Hacks", "Tech Tips", "Productivity"],
  "hashtags": ["#AIHacks", "#TechTips", "#Productivity", "#VaibhavSisinty"],
  "comment_hook": "Provocative question for comments",
  "phonetic_pronunciation_map": {{}}
}}"""


RETENTION_OPTIMIZER_LONGFORM_TEMPLATE = """{persona}

RETENTION OPTIMIZER TASK (LONG-FORM — 8 MINUTES):
Optimize this Vaibhav Sisinty explainer style script to maximize pacing and viewer retention.
Ensure there are no filler words and add vocal dynamics (commas, ellipses, exclamation marks, ALL CAPS on key words).
Shorten any sentence over 18 words.

STRICT RETENTION ENFORCEMENTS:
1. 3-Second Rule: Check that the very first sentence immediately states the stakes (e.g. "This line of code caused a $500 million dollar blackout..."). Strip out any intro greeting.
2. 4-Second Visual Reset: Make sure visual cues in the visual layout details shift at least every 4 seconds.
3. Cut the Fluff: Remove filler words and redundant phrases that don't add direct value, but do NOT delete engineering details or facts.
4. Length Preservation: The script MUST maintain its target length of 1100–1350 words (approx 8 minutes of speech) to ensure mid-roll ad eligibility. Do NOT summarize, compress, or delete any of the 8 topic segments or facts. Retain all details and explain them in a fast-paced, highly detailed manner.

ASSEMBLED SCRIPT:
{assembled_script}

Return ONLY a JSON object:
{{
  "optimized_script": "The full rewritten script with all optimizations applied.",
  "word_count": 1200,
  "estimated_duration_seconds": 450,
  "retention_hooks_added": ["cold open hook strengthened", "bridge transition optimized"],
  "sentences_shortened": 5,
  "filler_words_removed": 12
}}"""


HUMANIZER_TEMPLATE = """{persona}

HUMANIZER AGENT TASK:
Polishes the wording, adds contractions, fixes robotic/repetitive phrasing, ensures extremely smooth flow.
Formats the final output EXACTLY matching the required schema below.

OPTIMIZED SCRIPT:
{optimized_script}

FULL COMPILATION DATA:
{compilation_data}

SCHEMA REQUIREMENTS:
{schema_requirements}

Return ONLY the final JSON object matching the schema. No markdown wrapping. No explanations."""


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level cache for failed Cloudflare models (permanent errors like 400/403/404)
_FAILED_CLOUDFLARE_MODELS = set()

def call_fallback_model(prompt):
    """
    Attempts to call non-Gemini fallback APIs in sequence:
    Groq (Llama) -> Cloudflare Workers AI -> OpenAI -> Anthropic (Claude) -> DeepSeek -> OpenRouter.
    Returns the parsed JSON response dict or None.
    """
    import os
    import json
    import requests
    from gemini_script import is_model_exhausted, mark_model_exhausted

    def clean_and_parse_json(content):
        raw = content.strip()
        if "```json" in raw:
            raw = raw[raw.find("```json")+7:raw.rfind("```")]
        elif "```" in raw:
            raw = raw[raw.find("```")+3:raw.rfind("```")]
        return json.loads(raw.strip())

    # 0. Groq (fast, reliable, high quota - prioritized first)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        # Model preference order: only verified working models
        groq_models = [
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "openai/gpt-oss-20b",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b"
        ]
        for model_name in groq_models:
            if is_model_exhausted("groq_models", model_name):
                print(f"⏭️ Skipping known-bad Groq model: {model_name}")
                continue
                
            # Determine max_tokens and TPM limit dynamically (Conflict Fix: limit max_tokens on small TPM models)
            if model_name in ("llama-3.1-8b-instant", "qwen/qwen3-32b"):
                max_tokens = 1536
                tpm_limit = 6000
            else:
                max_tokens = 4096
                tpm_limit = None
                
            # TPM budget check: skip if combined requested tokens (prompt + max completion tokens) exceeds the limit
            if tpm_limit:
                approx_prompt_tokens = len(prompt) // 4
                if approx_prompt_tokens + max_tokens > tpm_limit:
                    print(f"⏭️ Skipping Groq model {model_name} because requested budget ({approx_prompt_tokens} prompt + {max_tokens} completion) exceeds TPM limit ({tpm_limit})")
                    continue
                
            print(f"🔮 Falling back to Groq ({model_name})...")
            try:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.7,
                    "max_tokens": max_tokens
                }
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
                if r.status_code == 200:
                    content = r.json()["choices"][0]["message"]["content"].strip()
                    return clean_and_parse_json(content)
                else:
                    print(f"⚠️ Groq ({model_name}) failed with code {r.status_code}: {r.text}")
                    if r.status_code == 429:
                        mark_model_exhausted("groq_models", model_name, r.text)
            except Exception as e:
                print(f"⚠️ Groq ({model_name}) fallback failed: {e}")

    # 1. Cloudflare Workers AI (fast, free tier available)
    cloudflare_token = os.getenv("CF_API_TOKEN") or os.getenv("CLOUDFLARE_API_TOKEN")
    cloudflare_account_id = os.getenv("CF_ACCOUNT_ID") or os.getenv("CLOUDFLARE_ACCOUNT_ID")
    if cloudflare_token and cloudflare_account_id:
        try:
            from config import CLOUDFLARE_MODELS
        except ImportError:
            CLOUDFLARE_MODELS = [
                "@cf/meta/llama-3.3-70b-instruct",
                "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                "@cf/zai-org/glm-4.7-flash",
                "@cf/openai/gpt-oss-120b",
                "@cf/nvidia/nemotron-3-120b-a12b",
                "@cf/meta/llama-4-scout-17b-16e-instruct",
                "@cf/qwen/qwq-32b",
                "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
                "@cf/meta/llama-3.1-70b-instruct",
                "@cf/qwen/qwen2.5-72b-instruct",
            ]
        headers = {
            "Authorization": f"Bearer {cloudflare_token}",
            "Content-Type": "application/json"
        }
        # gpt-oss models use Chat Completions format
        gpt_oss_models = {"@cf/openai/gpt-oss-120b", "@cf/openai/gpt-oss-20b"}
        for model_name in CLOUDFLARE_MODELS:
            # Skip models that already failed
            if is_model_exhausted("cloudflare_models", model_name):
                print(f"⏭️ Skipping known-bad Cloudflare model: {model_name}")
                continue
            print(f"🔮 Falling back to Cloudflare ({model_name})...")
            try:
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.7,
                    "max_tokens": 4096
                }
                r = requests.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{cloudflare_account_id}/ai/run/{model_name}",
                    json=payload,
                    headers=headers,
                    timeout=60
                )
                if r.status_code == 200:
                    result = r.json()["result"]
                    # Handle different response formats
                    if model_name in gpt_oss_models:
                        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    else:
                        content = result.get("response", "").strip()
                    if content:
                        return clean_and_parse_json(content)
                    else:
                        print(f"⚠️ Cloudflare ({model_name}) returned empty content")
                else:
                    print(f"⚠️ Cloudflare ({model_name}) failed with code {r.status_code}: {r.text}")
                    # Cache permanent failures: 400 (bad model), 403 (no access), 404 (not found)
                    # Also check response body for Cloudflare error code 4006 (quota exhausted)
                    error_code = None
                    try:
                        error_data = r.json()
                        if isinstance(error_data, dict):
                            errors = error_data.get("errors", [])
                            if errors and isinstance(errors, list):
                                error_code = errors[0].get("code")
                    except:
                        pass
                    
                    if r.status_code in (400, 403, 404) or error_code == 4006:
                        mark_model_exhausted("cloudflare_models", model_name, r.text)
                        if error_code == 4006:
                            print(f"🚫 Cloudflare quota exhausted (code 4006) - marking entire provider as dead")
                            try:
                                from config import CLOUDFLARE_MODELS
                                for m in CLOUDFLARE_MODELS:
                                    mark_model_exhausted("cloudflare_models", m, "Cloudflare quota exhausted (4006)")
                            except ImportError:
                                pass
            except Exception as e:
                print(f"⚠️ Cloudflare ({model_name}) fallback failed: {e}")

    # 1.5. OpenCode Zen (Nemotron 3 Ultra Free - free tier, reasoning support)
    opencode_key = os.getenv("OPENCODE_API_KEY")
    if opencode_key:
        model_name = "nemotron-3-ultra-free"
        if not is_model_exhausted("opencode_models", model_name):
            print(f"🔮 Falling back to OpenCode Zen ({model_name})...")
            headers = {
                "Authorization": f"Bearer {opencode_key}",
                "Content-Type": "application/json"
            }
            try:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                }
                r = requests.post("https://opencode.ai/zen/v1/chat/completions", json=payload, headers=headers, timeout=60)
                if r.status_code == 200:
                    content = r.json()["choices"][0]["message"]["content"].strip()
                    return clean_and_parse_json(content)
                else:
                    print(f"⚠️ OpenCode Zen ({model_name}) failed with code {r.status_code}: {r.text}")
                    if r.status_code == 429:
                        mark_model_exhausted("opencode_models", model_name, r.text)
            except Exception as e:
                print(f"⚠️ OpenCode Zen ({model_name}) fallback failed: {e}")

    # 2. OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("🔮 Falling back to OpenAI (gpt-4o-mini)...")
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

    # 3. Anthropic (Claude)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("🔮 Falling back to Anthropic (claude-3-5-haiku-20241022)...")
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
    """Multi-agent script generation engine for Vaibhav Sisinty format videos."""

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
        """Find the main topic and news updates for today's video."""
        print("🔥 [AGENT 0] Topic Discovery: Finding main breakout topic + news updates...")

        # Enrich context with live search
        search_text, search_links = self._call_gemini_search(
            "What are the top 10 most trending, viral, or breakout AI news stories, engineering breakdowns, "
            "or system design chronicles in the last 48 hours? Include model launches, code archaeology "
            "deep dives, model vs model comparisons, or massive server scaling secrets (e.g. Netflix, WhatsApp, Git, Spotify, Doom)."
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
        if not result or "main_topic" not in result:
            print("⚠️ Topic Discovery failed.")
            return None
        
        main_t = result.get("main_topic", {})
        news_u = result.get("news_updates", [])
        print(f"   📌 Main topic discovered: {main_t.get('headline', 'Unknown')}")
        print(f"   📌 Discovered {len(news_u)} secondary news updates.")
        return result

    # ── STEP 1: Research per topic ───────────────────────────────────────────
    def research_topic(self, topic):
        """Deep-research a single topic for fact extraction."""
        headline = topic.get("headline") or ""
        source_url = topic.get("source_url") or ""
        keywords = topic.get("search_keywords") or []
        
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

    # ── STEP 2a: Generate script for main deep-dive topic ────────────────────
    def generate_deep_dive_script(self, topic, research_data):
        headline = topic.get("headline", "")
        research_context = json.dumps(research_data, indent=2) if research_data else "No detailed research available."
        prompt = DEEP_DIVE_SCRIPT_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            headline=headline,
            research_context=research_context
        )
        print(f"   📝 [AGENT 2.DEEP_DIVE] Generating main deep dive script for {headline}...")
        return self._call_gemini(prompt, model=GEMINI_PRO_MODEL)

    # ── STEP 2b: Generate script for news update ─────────────────────────────
    def generate_update_script(self, topic, research_data):
        headline = topic.get("headline", "")
        research_context = json.dumps(research_data, indent=2) if research_data else "No detailed research available."
        prompt = UPDATE_SCRIPT_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            headline=headline,
            research_context=research_context
        )
        print(f"   📝 [AGENT 2.UPDATE] Generating news update script for {headline}...")
        return self._call_gemini(prompt, model=GEMINI_PRO_MODEL)

    # ── STEP 3: Assemble compilation ─────────────────────────────────────────
    def assemble_compilation(self, deep_dive_script, updates_scripts):
        """Stitch deep-dive and news updates into one seamless video script."""
        print("🎬 [AGENT 3] Compilation Assembler: Stitching deep-dive and news updates...")
        min_words, max_words = LONGFORM_WORD_COUNT_TARGET
        prompt = COMPILATION_ASSEMBLER_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            deep_dive_json=json.dumps(deep_dive_script, indent=2),
            updates_json=json.dumps(updates_scripts, indent=2),
            min_words=min_words,
            max_words=max_words
        )
        return self._call_gemini(prompt, model=GEMINI_PRO_MODEL)

    # ── STEP 4: Optimize retention ───────────────────────────────────────────
    def optimize_retention(self, assembled_script):
        """Maximize retention at critical drop-off points."""
        print("⚡ [AGENT 4] Retention Optimizer: Maximizing pacing for video...")
        
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
        print("🗣️ [AGENT 5] Humanizer: Fixing AI cadence for final schema...")

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
        topics_data = self.discover_topics()
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if not topics_data or "main_topic" not in topics_data:
            print("❌ [LONGFORM] Could not discover main topic. Aborting.")
            return None

        main_topic = topics_data["main_topic"]
        news_updates = topics_data.get("news_updates", [])

        # 1. Research main topic
        main_research = self.research_topic(main_topic)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if not main_research:
            main_research = {"core_narrative": main_topic.get("one_liner", ""), "details": [], "prompts_or_steps": []}

        # 2. Generate deep-dive script
        deep_dive_script = self.generate_deep_dive_script(main_topic, main_research)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if not deep_dive_script:
            print("❌ [LONGFORM] Deep-dive script generation failed. Aborting.")
            return None
        deep_dive_script["topic"] = main_topic

        # 3. Research + generate scripts for news updates
        updates_scripts = []
        successful_topics = [main_topic]
        
        # Max news updates count: total topics (LONGFORM_NUM_TOPICS = 8) - 1 main topic = 7 updates
        target_updates_count = LONGFORM_NUM_TOPICS - 1
        news_updates = news_updates[:target_updates_count]

        for i, topic in enumerate(news_updates):
            update_num = i + 1
            # Research
            research = self.research_topic(topic)
            if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
            if not research:
                research = {"core_narrative": topic.get("one_liner", ""), "details": [], "facts": []}
            
            # Generate update script
            update_script = self.generate_update_script(topic, research)
            if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
            if update_script:
                update_script["topic"] = topic
                updates_scripts.append(update_script)
                successful_topics.append(topic)
                print(f"   ✅ News Update #{update_num} script generated ({len(update_script.get('script', '').split())} words)")
            else:
                print(f"   ❌ News Update #{update_num} script generation failed. Skipping.")

        # 4. Assemble compilation
        compilation = self.assemble_compilation(deep_dive_script, updates_scripts)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if not compilation or "script" not in compilation:
            print("❌ [LONGFORM] Compilation assembly failed. Aborting.")
            return None

        assembled_script = compilation.get("script", "")
        word_count = len(assembled_script.split())
        print(f"   📊 Assembled script: {word_count} words")

        # 5. Optimize retention
        optimized = self.optimize_retention(assembled_script)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if optimized and "optimized_script" in optimized:
            final_script = optimized["optimized_script"]
            opt_word_count = len(final_script.split())
            print(f"   📊 Optimized script: {opt_word_count} words")
        else:
            print("   ⚠️ Retention optimization failed. Using assembled script as-is.")
            final_script = assembled_script

        # 6. Humanize and get final JSON
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
            final_data["original_news_headline"] = f"Top AI Updates That Feel Illegal - {today_str}"
        if not final_data.get("original_news_url") or "Primary" in final_data.get("original_news_url", ""):
            final_data["original_news_url"] = successful_topics[0].get("source_url", "") if successful_topics else ""
        
        # Attach successful topics metadata for downstream use
        final_data["longform_topics"] = successful_topics
        
        # For compatibility with main_longform.py which uses fact_scripts to check shock levels:
        combined_fact_scripts = []
        dd_fs = {
            "fact_number": 1,
            "script": deep_dive_script.get("script", ""),
            "shock_level": 8,
            "key_stat": deep_dive_script.get("key_prompt_used", "Deep Dive"),
            "transition_style": "glitch",
            "source_reference": main_topic.get("source_name", "")
        }
        combined_fact_scripts.append(dd_fs)
        
        for idx, us in enumerate(updates_scripts):
            us_fs = {
                "fact_number": idx + 2,
                "script": us.get("script", ""),
                "shock_level": 7,
                "key_stat": "News Update",
                "transition_style": "zoom",
                "source_reference": us.get("topic", {}).get("source_name", "")
            }
            combined_fact_scripts.append(us_fs)
            
        final_data["fact_scripts"] = combined_fact_scripts
        final_data["is_longform"] = True
        final_data["longform_format"] = "vaibhav"
        final_data["num_facts"] = len(successful_topics)

        # Ensure best_fact_for_shorts is populated
        if not final_data.get("best_fact_for_shorts"):
            final_data["best_fact_for_shorts"] = {
                "fact_number": 1,
                "reason": "Main deep dive topic hook",
                "hook_for_shorts": "This is how to 10x your productivity. Full guide in the long video!"
            }

        print(f"⭐ [LONGFORM PIPELINE] Multi-agent generation completed: "
              f"{len(final_data.get('script', '').split())} words, {len(successful_topics)} facts.")
        return final_data


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_longform_script(articles=None, failed_topics=None):
    """Main entry point for long-form "Did You Know" script generation."""
    from gemini_script import _get_active_gemini_key
    active_key = _get_active_gemini_key()
    client = genai.Client(api_key=active_key)
    
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
