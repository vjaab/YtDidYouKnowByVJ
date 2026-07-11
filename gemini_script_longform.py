"""
gemini_script_longform.py — Chaptered Deep-Dive Script Generation Pipeline.

CHAPTERED FORMAT OVERHAUL (2026-07):
  - Replaced 8-topic compilation with chaptered deep-dive (Fireship/MKBHD/Johnny Harris)
  - 4-agent pipeline: Topic Discovery → Research → Chaptered Script → Retention+Humanizer
  - Supports two modes: "single" (1 deep story) and "multi" (2-3 linked stories)
  - Output includes chapters[] with visual_beats[] for memory-efficient chunk rendering

Agents:
  0. Topic Discovery Agent — Finds the richest story (or 2-3 linked stories)
  1. Research Agent — Deep-researches selected story/stories
  2. Chaptered Script Architect — Writes 3-5 chapter cold-open-to-payoff script
  3. Retention Optimizer + Humanizer — Combined pass with chapter-preservation guard
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
    LONGFORM_MAX_CHAPTERS, LONGFORM_VISUAL_BEATS_PER_CHAPTER,
    LONGFORM_WORD_COUNT_TARGET, LONGFORM_TARGET_AUDIO_DURATION,
    LONGFORM_TRACKER_FILE, LONGFORM_COLD_OPEN_DURATION,
    get_topic_depth_mode
)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES (CHAPTERED DEEP-DIVE FORMAT)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PERSONA_LONGFORM = """Role: You are an elite tech video scriptwriter who creates chaptered deep-dive videos combining Fireship's density, MKBHD's clarity, and Johnny Harris's narrative structure. You target engineers, founders, and tech professionals who want substance, not surface-level coverage.

Format: 16:9 Landscape, 5-25 minutes (depth-driven, not target-driven), chaptered deep-dive.

Tone: Dense, fast-paced, technically precise. Short sentences. Active voice. No throat-clearing, no filler. React to facts with genuine surprise or opinion. Use contractions naturally (it's, you're, don't, can't). Mix punchy 5-word sentences with longer 15-word flowing ones.

Target Audience: Engineers, founders, creators from high-RPM countries (USA, UK, Canada, Australia, Singapore, India). Standard English, clean numbers, clear analogies.

MANDATORY WRITING CONSTRAINTS:
- First sentence must immediately state high stakes. NO "hey guys", NO channel intros, NO welcomes.
- Every sentence must earn its place. If it doesn't add a new fact, stake, or turn, cut it.
- Hard constraint: ZERO em dashes, en dashes, spaced em dashes, or double hyphens. Use commas, colons, periods, or parentheses.
- Do NOT use: "delve", "fostering", "tapestry", "intricate", "pivotal", "vibrant", "enduring", "enhance", "additionally", "landscape", "testament", "underscore", "showcase", "groundbreaking", "breathtaking".
- Do NOT use: "serves as", "stands as", "is a testament", "evolving landscape", "Not only... but...", "boasts a".
- Do NOT use passive voice or subjectless fragments ("No configuration needed" -> "You don't need a configuration").
- Avoid vague attributions: "Experts say", "Industry reports", "Observers note". Name specific sources or cut the claim.
- Use simple copulas: "is", "are", "has". Not "serves as", "features", "stands as".
- Do NOT force descriptions into groups of three (rule of three).
- Use heavy punctuation (commas, ellipses, ALL CAPS on key words, exclamation points) for TTS vocal dynamics."""


TOPIC_DISCOVERY_SINGLE_TEMPLATE = """{persona}

TOPIC DISCOVERY AGENT TASK (SINGLE DEEP STORY MODE):
Find the ONE story from the last 72 hours with the most depth, narrative potential, and utility for a tech audience. This story must be rich enough to sustain a 10-25 minute chaptered deep-dive.

Prioritize stories that have:
1. A specific, surprising claim or data point (not just "company X released Y")
2. A clear "why this matters" angle with real consequences
3. Enough technical depth for 3-5 chapters of analysis
4. A strong hook that works as a cold open

SOURCES TO ANALYZE:
{news_context}

AVOIDANCE LIST (DO NOT select topics similar to these):
{avoid_list}

Return ONLY a JSON object:
{{
  "selected_stories": [
    {{
      "headline": "The specific story headline",
      "source_url": "Direct source URL",
      "source_name": "Company or publication name",
      "why_this_story": "2-3 sentences on why this has the most depth and narrative potential",
      "search_keywords": ["keyword1", "keyword2", "keyword3"],
      "suggested_chapters": ["Chapter angle 1", "Chapter angle 2", "Chapter angle 3"]
    }}
  ],
  "depth_mode": "single"
}}"""


TOPIC_DISCOVERY_MULTI_TEMPLATE = """{persona}

TOPIC DISCOVERY AGENT TASK (MULTI-STORY MODE):
Find 2-3 stories from the last 72 hours that are THEMATICALLY CONNECTED by a single throughline. Do NOT combine unrelated stories. The throughline must be a genuine shared trend, cause, or consequence.

Examples of valid throughlines:
- "Three companies all racing to solve the same problem this week"
- "A single technical breakthrough that triggered reactions across the industry"
- "Multiple signals pointing to the same fundamental shift in AI"

Examples of INVALID combinations (do not do this):
- Random unrelated AI news crammed together
- Stories linked only by "they're all about AI"

SOURCES TO ANALYZE:
{news_context}

AVOIDANCE LIST (DO NOT select topics similar to these):
{avoid_list}

Return ONLY a JSON object:
{{
  "selected_stories": [
    {{
      "headline": "Story headline",
      "source_url": "Direct source URL",
      "source_name": "Company or publication name",
      "why_this_story": "How this connects to the throughline",
      "search_keywords": ["keyword1", "keyword2"]
    }}
  ],
  "throughline": "The single sentence that connects all selected stories",
  "depth_mode": "multi"
}}"""


RESEARCH_TEMPLATE = """{persona}

RESEARCH AGENT TASK:
Deep-research the following story. Go beyond the headline. Find the technical details, data points, competitive context, and real implications that make this story worth 10+ minutes of someone's time.

TARGET STORY: {headline}
SOURCE URL: {source_url}
ADDITIONAL CONTEXT:
{context}

Find:
1. Specific technical details, architecture decisions, or implementation specifics
2. Concrete numbers: benchmarks, costs, speeds, user counts, funding amounts
3. Competitive context: who else is doing this, how does it compare
4. The "so what": real consequences for engineers, founders, or the industry
5. Any contrarian angles or criticisms worth addressing
6. Historical context: what came before, why now

Return ONLY a JSON object:
{{
  "core_narrative": "A detailed 3-4 paragraph summary with specific facts and numbers",
  "technical_details": ["Detail 1 with specific numbers", "Detail 2"],
  "competitive_context": "How this compares to alternatives",
  "implications": ["Real consequence 1", "Real consequence 2"],
  "contrarian_angles": ["Criticism or counterpoint worth addressing"],
  "key_numbers": ["$X funding", "Y% improvement", "Z users"],
  "chapter_angles": ["Suggested angle for chapter 1", "Angle 2", "Angle 3"]
}}"""


CHAPTERED_SCRIPT_TEMPLATE = """{persona}

CHAPTERED SCRIPT ARCHITECT TASK:
Write the full narration script for a chaptered deep-dive YouTube video.

{story_context}

STRUCTURAL RULES (non-negotiable):

1. COLD OPEN HOOK (0-15s): Start with the single sharpest, most specific claim or question from the story. No channel intro, no "hey guys welcome back." The hook must be answerable only by watching further.

2. 3 TO 5 CHAPTERS: Each chapter is a self-contained beat that:
   - Answers ONE sub-question completely (good abandonment: a viewer who only wanted that answer can leave satisfied, and that's rewarded in 2026 retention scoring)
   - Has its own mini-payoff or "aha" moment
   - Naturally escalates in stakes from Chapter 1 to the final chapter

3. RUNTIME IS DEPTH-DRIVEN: Do not pad to hit a duration. Write until the story is fully told, then stop. Target range: {min_words}-{max_words} words (~{min_min}-{max_min} minutes). Never stretch content to fill time.

4. VISUAL ECONOMY: Write in beats that can each hold ONE strong supporting visual for 8-15 seconds of narration. Do NOT fragment into one visual per sentence. Group related claims into a single narrated beat. Max {max_beats} visual beats per chapter.

5. PACING: Short sentences, active voice, no throat-clearing. If a sentence doesn't add a new fact, stake, or turn, remove it.

6. ENDING: Close the final chapter with a direct payoff, then a single-sentence takeaway, then a natural CTA: "If you found this useful, subscribe for more deep-dives like this. I'll see you in the next one."

Return ONLY valid JSON matching this schema exactly:
{{
  "title": "YouTube title (max 80 chars, curiosity-driven)",
  "title_options": ["Title option 1", "Title option 2", "Title option 3"],
  "script": "The FULL narration script, all chapters concatenated. No headers or labels inside the script text.",
  "description": "2-3 sentence video description for YouTube",
  "sub_category": "e.g. AI Models, Security, Open Source, Developer Tools",
  "chapters": [
    {{
      "chapter_number": 1,
      "chapter_title": "Short label e.g. 'The Claim That Started It All'",
      "chapter_text": "The narration text for just this chapter",
      "approx_start_seconds": 0,
      "visual_beats": [
        {{
          "beat_text": "The sub-portion of narration this visual covers (8-15s worth)",
          "visual_direction": "What should be shown: screenshot of X, diagram showing Y, B-roll of Z. Be specific."
        }}
      ]
    }}
  ],
  "fact_timestamps": [
    {{"topic": "Chapter title or section label", "approx_start_seconds": 0}}
  ],
  "subtitle_chunks": [
    {{"text": "1-3 word subtitle chunk", "start": 0.0, "end": 1.5}}
  ],
  "longform_topics": [
    {{"headline": "Source story headline", "source_name": "Source name", "source_url": "URL"}}
  ],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "hashtags": ["#hashtag1", "#hashtag2"],
  "people": [{{"name": "Person mentioned in script"}}],
  "companies_mentioned": [{{"name": "Company name"}}],
  "phonetic_pronunciation_map": {{"unusual_word": "phonetic spelling"}},
  "comment_hook": "A provocative question to pin as first comment",
  "original_news_headline": "The primary source headline",
  "original_news_url": "The primary source URL"
}}

QUALITY CHECK before finalizing:
- Does the hook work without any prior context?
- Could a viewer leave after any chapter and feel satisfied?
- Is every sentence earning its place?
- Are visual_beats grouped (not one per sentence)?
- Is the word count between {min_words}-{max_words}?"""


RETENTION_HUMANIZER_TEMPLATE = """{persona}

RETENTION OPTIMIZER + HUMANIZER TASK (COMBINED PASS):
Optimize this chaptered script for maximum retention and remove all AI-generated writing patterns.

CRITICAL PRESERVATION RULES:
1. Do NOT delete, merge, or reorder chapters. The chapter structure is sacred.
2. Do NOT reduce total word count by more than 10%. The depth is intentional.
3. Do NOT remove technical details, numbers, or competitive comparisons.
4. Do NOT add generic filler to pad word count.

RETENTION OPTIMIZATIONS:
- First sentence must immediately state stakes. Strip any remaining intro greeting.
- Shorten sentences over 18 words. Split compound sentences.
- Add vocal dynamics: commas for pauses, ellipses for suspense, ALL CAPS on 1-2 key words per chapter, exclamation for genuine surprise.
- Ensure each chapter ends with a micro-cliffhanger or payoff that pulls into the next.

HUMANIZER AUDIT (fix all of these):
1. Remove grandiose statements ("stands as", "is a testament", "pivotal moment", "evolving landscape")
2. Replace AI vocabulary ("delve", "fostering", "tapestry", "intricate", "vibrant", "enhance", "additionally", "showcase", "underscore")
3. Convert passive voice to active voice
4. Remove vague attributions ("Experts say", "Industry reports")
5. Replace elaborate verbs with simple copulas ("is", "are", "has")
6. Remove "Not only... but..." parallelisms
7. Eliminate ALL em dashes, en dashes, spaced em dashes, double hyphens. Use commas, colons, or periods.
8. Use contractions naturally (it's, you're, don't, can't)
9. Vary sentence rhythm: mix 5-word punches with 15-word flows

SCRIPT TO OPTIMIZE:
{script}

COMPILATION DATA:
{compilation_data}

Return ONLY valid JSON:
{{
  "title": "Optimized YouTube title",
  "title_options": ["Title 1", "Title 2", "Title 3"],
  "script": "The full optimized and humanized script",
  "description": "2-3 sentence description",
  "sub_category": "category",
  "chapters": [same structure as input, with optimized chapter_text],
  "fact_timestamps": [same structure as input],
  "subtitle_chunks": [{{"text": "1-3 word chunk"}}],
  "longform_topics": [same as input],
  "keywords": ["keyword1", "keyword2"],
  "hashtags": ["#hashtag1", "#hashtag2"],
  "people": [{{"name": "Person"}}],
  "companies_mentioned": [{{"name": "Company"}}],
  "phonetic_pronunciation_map": {{}},
  "comment_hook": "Provocative question",
  "original_news_headline": "Primary headline",
  "original_news_url": "Primary URL",
  "optimization_stats": {{
    "original_word_count": 0,
    "optimized_word_count": 0,
    "sentences_shortened": 0,
    "ai_patterns_fixed": 0,
    "em_dashes_removed": 0
  }}
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE (reuse the existing fallback infrastructure from the old module)
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level cache for failed Cloudflare models
_FAILED_CLOUDFLARE_MODELS = set()

# Groq daily token limits
GROQ_TPD_LIMITS = {
    "llama-3.3-70b-versatile": 1_000_000,
    "qwen/qwen3-32b": 500_000,
    "openai/gpt-oss-20b": 500_000,
    "llama-3.1-8b-instant": 1_000_000,
    "openai/gpt-oss-120b": 500_000
}

def is_groq_model_near_limit(model_name):
    """Returns True if the Groq model has consumed 90%+ of its daily token limit."""
    try:
        from gemini_script import _load_cache
        cache = _load_cache()
        usage = cache.get("groq_token_usage", {}).get(model_name, {})
        today_str = datetime.now().strftime("%Y-%m-%d")
        if usage.get("date") == today_str:
            tokens_used = usage.get("tokens", 0)
            limit = GROQ_TPD_LIMITS.get(model_name, 500_000)
            if tokens_used >= limit * 0.9:
                print(f"⚠️ [CACHE] {model_name} at {tokens_used}/{limit} TPD ({tokens_used/limit:.1%}) — skipping")
                return True
    except Exception as e:
        print(f"⚠️ Error checking Groq token limit: {e}")
    return False

def _update_groq_token_usage(model_name, tokens_increment):
    """Updates the token usage counters in the shared api_exhaustion_cache.json."""
    try:
        from gemini_script import _load_cache, CACHE_FILE
        data = _load_cache()
        if "groq_token_usage" not in data:
            data["groq_token_usage"] = {}
        today_str = datetime.now().strftime("%Y-%m-%d")
        current = data["groq_token_usage"].get(model_name, {})
        if current.get("date") == today_str:
            new_tokens = current.get("tokens", 0) + tokens_increment
        else:
            new_tokens = tokens_increment
        data["groq_token_usage"][model_name] = {"tokens": new_tokens, "date": today_str}
        tmp_file = CACHE_FILE + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_file, CACHE_FILE)
    except Exception as e:
        print(f"⚠️ Warning: failed to save Groq token usage ({e})")


def call_fallback_model(prompt):
    """
    Attempts non-Gemini fallback APIs in sequence:
    Groq -> Cloudflare Workers AI -> OpenCode Zen -> Cerebras -> NVIDIA NIM ->
    Mistral -> GitHub Models -> OpenAI -> Anthropic -> DeepSeek -> OpenRouter.
    Returns parsed JSON dict or None.
    """
    import requests
    from gemini_script import is_model_exhausted, mark_model_exhausted

    def clean_and_parse_json(content):
        raw = content.strip()
        if "```json" in raw:
            raw = raw[raw.find("```json")+7:raw.rfind("```")]
        elif "```" in raw:
            raw = raw[raw.find("```")+3:raw.rfind("```")]
        return json.loads(raw.strip())

    def safe_extract_choices(response_or_json, provider_name):
        try:
            data = response_or_json if isinstance(response_or_json, dict) else response_or_json.json()
            choices = data.get("choices")
            if choices and len(choices) > 0:
                msg = choices[0].get("message")
                if msg and "content" in msg and msg["content"]:
                    return msg["content"].strip()
        except Exception as e:
            print(f"⚠️ [{provider_name}] Parse failed: {e}")
        return None

    def safe_extract_anthropic(response):
        try:
            data = response.json()
            content_list = data.get("content")
            if content_list and len(content_list) > 0:
                text = content_list[0].get("text")
                if text:
                    return text.strip()
        except Exception as e:
            print(f"⚠️ [Anthropic] Parse failed: {e}")
        return None

    # 0. Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        groq_models = ["llama-3.3-70b-versatile", "qwen/qwen3-32b", "openai/gpt-oss-20b", "llama-3.1-8b-instant", "openai/gpt-oss-120b"]
        for model_name in groq_models:
            if is_model_exhausted("groq_models", model_name) or is_groq_model_near_limit(model_name):
                continue
            max_tokens = 1536 if model_name in ("llama-3.1-8b-instant", "qwen/qwen3-32b") else 4096
            print(f"🔮 Falling back to Groq ({model_name})...")
            try:
                payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "temperature": 0.7, "max_tokens": max_tokens}
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
                if r.status_code == 200:
                    res_json = r.json()
                    total_tokens = res_json.get("usage", {}).get("total_tokens", 0)
                    if total_tokens > 0:
                        _update_groq_token_usage(model_name, total_tokens)
                    content = safe_extract_choices(res_json, "Groq")
                    if content:
                        return clean_and_parse_json(content)
                elif r.status_code == 429:
                    mark_model_exhausted("groq_models", model_name, r.text)
            except Exception as e:
                print(f"⚠️ Groq ({model_name}) failed: {e}")

    # 1. Cloudflare Workers AI
    cloudflare_token = os.getenv("CF_API_TOKEN") or os.getenv("CLOUDFLARE_API_TOKEN")
    cloudflare_account_id = os.getenv("CF_ACCOUNT_ID") or os.getenv("CLOUDFLARE_ACCOUNT_ID")
    if cloudflare_token and cloudflare_account_id:
        try:
            from config import CLOUDFLARE_MODELS
        except ImportError:
            CLOUDFLARE_MODELS = ["@cf/meta/llama-3.3-70b-instruct", "@cf/qwen/qwen2.5-72b-instruct"]
        gpt_oss_models = {"@cf/openai/gpt-oss-120b", "@cf/openai/gpt-oss-20b"}
        headers = {"Authorization": f"Bearer {cloudflare_token}", "Content-Type": "application/json"}
        for model_name in CLOUDFLARE_MODELS:
            if is_model_exhausted("cloudflare_models", model_name):
                continue
            print(f"🔮 Falling back to Cloudflare ({model_name})...")
            try:
                payload = {"messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "temperature": 0.7, "max_tokens": 4096}
                r = requests.post(f"https://api.cloudflare.com/client/v4/accounts/{cloudflare_account_id}/ai/run/{model_name}", json=payload, headers=headers, timeout=60)
                if r.status_code == 200:
                    result = r.json()["result"]
                    content = safe_extract_choices(result, "Cloudflare") if model_name in gpt_oss_models else result.get("response", "").strip()
                    if content:
                        return clean_and_parse_json(content)
                else:
                    if r.status_code in (400, 403, 404):
                        mark_model_exhausted("cloudflare_models", model_name, r.text)
            except Exception as e:
                print(f"⚠️ Cloudflare ({model_name}) failed: {e}")

    # 2. Cerebras
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    if cerebras_key:
        headers = {"Authorization": f"Bearer {cerebras_key}", "Content-Type": "application/json"}
        for model_name in ["gpt-oss-120b", "zai-glm-4.7", "gemma-4-31b"]:
            if is_model_exhausted("cerebras_models", model_name):
                continue
            print(f"🔮 Falling back to Cerebras ({model_name})...")
            try:
                payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "temperature": 0.7}
                r = requests.post("https://api.cerebras.ai/v1/chat/completions", json=payload, headers=headers, timeout=30)
                if r.status_code == 200:
                    content = safe_extract_choices(r, "Cerebras")
                    if content:
                        return clean_and_parse_json(content)
                elif r.status_code == 429:
                    mark_model_exhausted("cerebras_models", model_name, r.text)
            except Exception as e:
                print(f"⚠️ Cerebras ({model_name}) failed: {e}")

    # 3. DeepSeek
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        print("🔮 Falling back to DeepSeek...")
        try:
            headers = {"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"}
            payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "temperature": 0.7}
            r = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                content = safe_extract_choices(r, "DeepSeek")
                if content:
                    return clean_and_parse_json(content)
        except Exception as e:
            print(f"⚠️ DeepSeek failed: {e}")

    # 4. OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        for model_name in ["openrouter/free", "meta-llama/llama-3.3-70b-instruct"]:
            if is_model_exhausted("openrouter_models", model_name):
                continue
            print(f"🔮 Falling back to OpenRouter ({model_name})...")
            try:
                headers = {"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"}
                payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
                r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=30)
                if r.status_code == 200:
                    content = safe_extract_choices(r, "OpenRouter")
                    if content:
                        return clean_and_parse_json(content)
                elif r.status_code in (429, 403):
                    mark_model_exhausted("openrouter_models", model_name, r.text)
            except Exception as e:
                print(f"⚠️ OpenRouter ({model_name}) failed: {e}")

    return None


import concurrent.futures
import re

def get_retry_after(e):
    """Parses Retry-After header or delay hints from exception string."""
    err_str = str(e)
    match = re.search(r'(?:retry|wait|backoff|delay)\s*(?:after|of)?\s*(\d+(?:\.\d+)?)\s*(s|sec|second|ms|millisecond|minute|min)?', err_str, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if unit in ('ms', 'millisecond'):
            val = val / 1000.0
        elif unit in ('minute', 'min'):
            val = val * 60.0
        return max(val, 3.0)
    for attr in ('response', 'http_response', 'http_err'):
        if hasattr(e, attr):
            resp = getattr(e, attr)
            if resp and hasattr(resp, 'headers'):
                retry_after = resp.headers.get('Retry-After') or resp.headers.get('retry-after')
                if retry_after:
                    try:
                        return max(float(retry_after), 3.0)
                    except:
                        pass
    return 3.0

_TIMEOUT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=10)

def execute_with_timeout(func, timeout, *args, **kwargs):
    """Executes a function with a hard wall-clock timeout."""
    future = _TIMEOUT_EXECUTOR.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        print(f"🚨 [TIMEOUT] {func.__name__} timed out after {timeout}s.")
        return None
    except Exception as ex:
        print(f"🚨 [ERROR] Exception in {func.__name__}: {ex}")
        return None


class ChapteredScriptEngine:
    """4-agent chaptered deep-dive script generation engine."""

    def __init__(self, client, news_context, avoid_list_str, depth_mode="single"):
        self.client = client
        self.news_context = news_context
        self.avoid_list_str = avoid_list_str
        self.depth_mode = depth_mode  # "single" or "multi"

    def _call_gemini(self, prompt, model=GEMINI_FLASH_MODEL, use_search=False):
        """Call Gemini with strict retry logic and fast model fallback."""
        from gemini_script import is_model_exhausted, mark_model_exhausted

        model_sequence = []
        if model == GEMINI_PRO_MODEL:
            model_sequence = [GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL, GEMINI_FLASH_LITE_MODEL]
        elif model == GEMINI_FLASH_MODEL:
            model_sequence = [GEMINI_FLASH_MODEL, GEMINI_FLASH_LITE_MODEL]
        else:
            model_sequence = [GEMINI_FLASH_LITE_MODEL]

        active_sequence = [m for m in model_sequence if not is_model_exhausted("gemini_models", m)]

        for m in active_sequence:
            for attempt in range(2):
                try:
                    config_kwargs = {'temperature': 0.8, 'response_mime_type': 'application/json'}
                    if use_search:
                        config_kwargs['tools'] = [{'google_search': {}}]
                        del config_kwargs['response_mime_type']

                    response = self.client.models.generate_content(
                        model=m, contents=prompt,
                        config=types.GenerateContentConfig(**config_kwargs)
                    )
                    raw = response.text.strip()
                    if "{" in raw and "}" in raw:
                        raw = raw[raw.find("{"):raw.rfind("}") + 1]
                    return json.loads(raw)
                except Exception as e:
                    err_str = str(e).upper()
                    is_rate = any(x in err_str for x in ["503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "429", "OVERLOADED"])
                    if is_rate:
                        wait = get_retry_after(e)
                        print(f"⚠️ [LONGFORM] {m} rate-limited. Attempt {attempt+1}/2. Waiting {wait}s...")
                        mark_model_exhausted("gemini_models", m, err_str)
                        if attempt == 0:
                            time.sleep(wait)
                            continue
                        else:
                            break
                    else:
                        print(f"⚠️ [LONGFORM] Call failed ({m}): {e}. Retrying in 2s...")
                        time.sleep(2)

        print("🚨 Gemini failed all models. Attempting fallback models...")
        fallback_res = call_fallback_model(prompt)
        if fallback_res:
            return fallback_res
        return None

    def _call_gemini_search(self, query):
        """Call Gemini with Google Search grounding."""
        from gemini_script import is_model_exhausted
        models = [GEMINI_FLASH_MODEL, GEMINI_FLASH_LITE_MODEL]
        for m in models:
            if is_model_exhausted("gemini_models", m):
                continue
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=m, contents=query,
                        config=types.GenerateContentConfig(tools=[{'google_search': {}}])
                    )
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
                    if any(x in err_str for x in ["429", "RESOURCE_EXHAUSTED", "OVERLOADED"]):
                        if attempt == 0:
                            time.sleep(get_retry_after(e))
                            continue
                        break
                    else:
                        print(f"⚠️ [LONGFORM SEARCH] Failed: {e}")
                        break
        return "", []

    # ── AGENT 0: Topic Discovery ─────────────────────────────────────────────
    def discover_topics(self):
        """Find the story/stories for today's video based on depth mode."""
        print(f"🔥 [AGENT 0] Topic Discovery (mode={self.depth_mode})...")

        # Enrich with live search
        search_text, search_links = self._call_gemini_search(
            "What are the top trending AI breakthroughs, open-source releases, major tech announcements, "
            "or developer tool launches in the last 48 hours? Include GitHub trending repos and technical details."
        )
        enriched_context = self.news_context
        if search_text:
            links_str = "\n".join(search_links[:20])
            enriched_context += f"\n\nGEMINI SEARCH RESULTS:\n{search_text}\n\nSOURCES:\n{links_str}\n"

        if self.depth_mode == "multi":
            template = TOPIC_DISCOVERY_MULTI_TEMPLATE
        else:
            template = TOPIC_DISCOVERY_SINGLE_TEMPLATE

        prompt = template.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            news_context=enriched_context,
            avoid_list=self.avoid_list_str
        )
        result = self._call_gemini(prompt)
        if not result or "selected_stories" not in result:
            print("⚠️ Topic Discovery failed.")
            return None

        stories = result.get("selected_stories", [])
        print(f"   📌 Selected {len(stories)} story/stories:")
        for s in stories:
            print(f"      - {s.get('headline', 'Unknown')}")
        if self.depth_mode == "multi":
            print(f"   📌 Throughline: {result.get('throughline', 'N/A')}")
        return result

    # ── AGENT 1: Research ────────────────────────────────────────────────────
    def research_story(self, story):
        """Deep-research a single story."""
        headline = story.get("headline", "")
        source_url = story.get("source_url", "")
        keywords = story.get("search_keywords", [])

        print(f"   🔬 Researching: {headline}")

        search_query = (
            f"Latest technical details, benchmarks, competitive analysis about: {headline}. "
            f"Keywords: {', '.join(keywords)}. Focus on specific numbers, architecture, and implications."
        )
        search_text, search_links = self._call_gemini_search(search_query)

        context = f"HEADLINE: {headline}\nSOURCE: {source_url}\n"
        if search_text:
            context += f"\nSEARCH RESULTS:\n{search_text}\n"
            context += f"\nSOURCES:\n" + "\n".join(search_links[:5])

        prompt = RESEARCH_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            headline=headline,
            source_url=source_url,
            context=context
        )
        return self._call_gemini(prompt)

    # ── AGENT 2: Chaptered Script Architect ───────────────────────────────────
    def generate_chaptered_script(self, stories, research_results):
        """Generate the full chaptered script from researched stories."""
        print("📝 [AGENT 2] Chaptered Script Architect...")

        # Build story context
        story_parts = []
        for i, (story, research) in enumerate(zip(stories, research_results)):
            story_parts.append(f"""
STORY {i+1}: {story.get('headline', 'Unknown')}
SOURCE: {story.get('source_url', '')}
WHY THIS STORY: {story.get('why_this_story', '')}
RESEARCH:
{json.dumps(research, indent=2) if research else 'No detailed research available.'}
""")

        if self.depth_mode == "multi" and len(stories) > 1:
            story_context = f"""MODE: MULTI-STORY (2-3 thematically linked stories)
THROUGHLINE: Connect these stories through their shared theme. Each story gets its own chapter(s).

{"".join(story_parts)}"""
        else:
            story_context = f"""MODE: SINGLE DEEP STORY
Write the deepest possible analysis of this one story across 3-5 chapters.

{story_parts[0] if story_parts else 'No story context available.'}"""

        min_words, max_words = LONGFORM_WORD_COUNT_TARGET
        min_min = min_words // 140  # minutes at 140 WPM
        max_min = max_words // 140

        prompt = CHAPTERED_SCRIPT_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            story_context=story_context,
            min_words=min_words,
            max_words=max_words,
            min_min=min_min,
            max_min=max_min,
            max_beats=LONGFORM_VISUAL_BEATS_PER_CHAPTER
        )
        return self._call_gemini(prompt, model=GEMINI_PRO_MODEL)

    # ── AGENT 3: Retention + Humanizer ───────────────────────────────────────
    def optimize_and_humanize(self, script_data):
        """Combined retention optimization and humanization pass."""
        print("⚡ [AGENT 3] Retention Optimizer + Humanizer...")

        script = script_data.get("script", "")
        prompt = RETENTION_HUMANIZER_TEMPLATE.format(
            persona=SYSTEM_PERSONA_LONGFORM,
            script=script,
            compilation_data=json.dumps(script_data, indent=2)
        )
        return self._call_gemini(prompt, model=GEMINI_PRO_MODEL)

    # ── FULL PIPELINE ────────────────────────────────────────────────────────
    def execute(self):
        """Run the full 4-agent pipeline end-to-end."""

        # 0. Discover topics (90s timeout)
        topics_data = execute_with_timeout(self.discover_topics, 90)
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if not topics_data or "selected_stories" not in topics_data:
            print("❌ [LONGFORM] Could not discover topics. Aborting.")
            return None

        stories = topics_data["selected_stories"]

        # 1. Research each story (120s timeout per story)
        research_results = []
        for i, story in enumerate(stories):
            print(f"🚀 Researching story {i+1}/{len(stories)} with 120s timeout...")
            result = execute_with_timeout(self.research_story, 120, story)
            if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
            if result:
                research_results.append(result)
            else:
                research_results.append({
                    "core_narrative": story.get("why_this_story", ""),
                    "technical_details": [], "implications": []
                })

        # 2. Generate chaptered script (180s timeout)
        script_data = execute_with_timeout(
            lambda: self.generate_chaptered_script(stories, research_results), 180
        )
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)
        if not script_data or "script" not in script_data:
            print("❌ [LONGFORM] Chaptered script generation failed. Aborting.")
            return None

        word_count = len(script_data.get("script", "").split())
        print(f"   📊 Chaptered script: {word_count} words, {len(script_data.get('chapters', []))} chapters")

        # 3. Optimize + Humanize (120s timeout)
        final_data = execute_with_timeout(
            lambda: self.optimize_and_humanize(script_data), 120
        )
        if GEMINI_RPM_SLEEP > 0: time.sleep(GEMINI_RPM_SLEEP)

        if not final_data or "script" not in final_data:
            print("   ⚠️ Optimization/humanization failed. Using raw script.")
            final_data = script_data

        # Verify the optimizer didn't cut more than 10%
        original_wc = word_count
        final_wc = len(final_data.get("script", "").split())
        if final_wc < original_wc * 0.85:
            print(f"   ⚠️ Optimizer cut too aggressively ({original_wc} -> {final_wc} words). Reverting to pre-optimization script.")
            final_data["script"] = script_data["script"]
            final_data["chapters"] = script_data.get("chapters", [])

        # Ensure critical fields
        today_str = datetime.now().strftime("%Y-%m-%d")
        if not final_data.get("original_news_headline"):
            final_data["original_news_headline"] = stories[0].get("headline", f"AI Deep Dive - {today_str}")
        if not final_data.get("original_news_url"):
            final_data["original_news_url"] = stories[0].get("source_url", "")

        # Attach metadata
        final_data["longform_topics"] = stories
        final_data["is_longform"] = True
        final_data["longform_format"] = "chaptered"
        final_data["depth_mode"] = self.depth_mode
        final_data["num_facts"] = len(stories)

        # Build fact_scripts for backward compatibility with shorts_teaser.py
        combined_fact_scripts = []
        for i, story in enumerate(stories):
            combined_fact_scripts.append({
                "fact_number": i + 1,
                "script": story.get("headline", ""),
                "shock_level": 8,
                "key_stat": "Deep Dive" if i == 0 else "Linked Story",
                "transition_style": "glitch" if i == 0 else "zoom",
                "source_reference": story.get("source_name", "")
            })
        final_data["fact_scripts"] = combined_fact_scripts

        if not final_data.get("best_fact_for_shorts"):
            final_data["best_fact_for_shorts"] = {
                "fact_number": 1,
                "reason": "Hook from the main deep-dive chapter",
                "hook_for_shorts": "You need to see this. Full deep-dive in the long video!"
            }

        final_wc = len(final_data.get("script", "").split())
        print(f"⭐ [LONGFORM] Pipeline complete: {final_wc} words, "
              f"{len(final_data.get('chapters', []))} chapters, "
              f"{len(stories)} story/stories (mode={self.depth_mode})")
        return final_data


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_longform_script(articles=None, failed_topics=None):
    """Main entry point for longform chaptered deep-dive script generation."""
    from gemini_script import _get_active_gemini_key
    active_key = _get_active_gemini_key()
    client = genai.Client(api_key=active_key)

    if failed_topics is None:
        failed_topics = []

    # Determine today's depth mode
    depth_mode = get_topic_depth_mode()
    print(f"📋 Longform depth mode for today: {depth_mode}")

    # Build news context from RSS articles
    news_context = ""
    if articles:
        display_articles = articles[:30]
        for idx, art in enumerate(display_articles):
            title = art.get('title', '')
            desc = art.get('description', '')
            source = art.get('source', {}).get('name', '')
            url = art.get('url', '')
            news_context += f"\n[{idx+1}] Title: {title}\nDescription: {desc}\nSource: {source}\nURL: {url}\n"

    if not news_context:
        news_context = "No RSS articles available. Use Gemini Search to find today's top AI stories."

    # Build avoidance list from tracker
    tracker = load_tracker(tracker_file=LONGFORM_TRACKER_FILE)
    recent_history = tracker.get("history", [])[-30:]
    recent_titles = tracker.get("used_titles", [])[-60:]

    avoid_items = [h.get('news_headline', h.get('title')) for h in recent_history] + recent_titles
    if failed_topics:
        avoid_items += failed_topics
    combined_avoid = list(set([a for a in avoid_items if a]))
    avoid_list_str = "\n".join([f"- {t}" for t in combined_avoid]) if combined_avoid else "None"

    if combined_avoid:
        news_context = (
            f"CRITICAL: RECENTLY COVERED STORIES (DO NOT REPEAT):\n{avoid_list_str}\n\n"
            + news_context
        )

    # Run the chaptered pipeline
    engine = ChapteredScriptEngine(client, news_context, avoid_list_str, depth_mode)
    script_data = engine.execute()

    if script_data:
        headline = script_data.get("original_news_headline", "")
        news_url = script_data.get("original_news_url", "")
        is_unique, msg = check_story_uniqueness(headline, new_url=news_url, tracker_file=LONGFORM_TRACKER_FILE)
        if not is_unique:
            print(f"⚠️ [LONGFORM] Post-generation uniqueness check: {msg}")

    return script_data
