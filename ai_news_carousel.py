#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_news_carousel.py — AI News Carousel Content Generator
Generates structured JSON for 6-slide AI news carousels using LLM.
"""

import os
import json
import sys
import argparse
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

try:
    from trending_engine import fetch_all_trending_signals
    TRENDING_ENGINE_AVAILABLE = True
except ImportError:
    TRENDING_ENGINE_AVAILABLE = False

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

TRACKER_FILE = os.path.join(str(Path(__file__).parent), "news_log.json")
CAROUSEL_TRACKER_FILE = os.path.join(str(Path(__file__).parent), "instagram_carousel_log.json")
SIMILARITY_THRESHOLD = 75

AI_CATEGORIES = [
    "OpenAI",
    "Google / Gemini",
    "Anthropic / Claude",
    "Meta AI",
    "Microsoft",
    "NVIDIA",
    "AWS",
    "Open-source AI",
    "AI agents",
    "AI developer tools",
]

def load_topic_tracker() -> Dict:
    """Load the topic tracker from news_log.json."""
    if not os.path.exists(TRACKER_FILE):
        return {"used_titles": [], "used_keywords": [], "history": []}
    try:
        with open(TRACKER_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"used_titles": [], "used_keywords": [], "history": []}


def load_carousel_tracker() -> Dict:
    """Load the Instagram carousel topic tracker."""
    if not os.path.exists(CAROUSEL_TRACKER_FILE):
        return {"used_titles": [], "used_keywords": [], "history": []}
    try:
        with open(CAROUSEL_TRACKER_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"used_titles": [], "used_keywords": [], "history": []}


def save_carousel_tracker(tracker: Dict) -> None:
    """Save the Instagram carousel topic tracker."""
    try:
        with open(CAROUSEL_TRACKER_FILE, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save carousel tracker: {e}")


def record_carousel_topic(title: str, url: str, keywords: List[str], source: str) -> None:
    """Record a topic as used in the carousel tracker."""
    tracker = load_carousel_tracker()
    
    entry = {
        "title": title,
        "news_source_url": url,
        "keywords": keywords,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    tracker.setdefault("history", []).append(entry)
    tracker.setdefault("used_titles", []).append(title)
    tracker.setdefault("used_keywords", []).extend(keywords)
    
    # Keep only last 100 entries to prevent unbounded growth
    if len(tracker["history"]) > 100:
        tracker["history"] = tracker["history"][-100:]
    if len(tracker["used_titles"]) > 100:
        tracker["used_titles"] = tracker["used_titles"][-100:]
    if len(tracker["used_keywords"]) > 200:
        tracker["used_keywords"] = tracker["used_keywords"][-200:]
    
    save_carousel_tracker(tracker)
    print(f"📝 Recorded carousel topic: {title[:60]}...")

def is_topic_unique(title: str, url: str = "", keywords: List[str] = None, check_youtube: bool = True, check_carousel: bool = True) -> tuple:
    """Check if a topic is unique compared to previously used topics (YouTube + Carousel)."""
    if check_youtube:
        tracker = load_topic_tracker()
        
        if not tracker.get("used_titles") and not tracker.get("history"):
            pass
        else:
            if url:
                for entry in tracker.get("history", []):
                    if isinstance(entry, dict) and entry.get("news_source_url") == url:
                        return False, f"Exact URL already covered (YouTube): {url}"
            
            if RAPIDFUZZ_AVAILABLE:
                headlines_to_check = set(tracker.get("used_titles", []))
                for existing_title in headlines_to_check:
                    score = fuzz.token_set_ratio(title.lower(), existing_title.lower())
                    if score > SIMILARITY_THRESHOLD:
                        return False, f"Semantic match found (YouTube, score {score}): '{existing_title}'"
            
            if keywords:
                recent_keywords = set()
                for entry in tracker.get("history", [])[-20:]:
                    if url and entry.get("news_source_url") == url:
                        continue
                    recent_keywords.update([k.lower() for k in entry.get("keywords", [])])
                
                new_k_set = set([k.lower() for k in keywords])
                if new_k_set:
                    intersection = new_k_set.intersection(recent_keywords)
                    overlap_pct = (len(intersection) / len(new_k_set)) * 100
                    if overlap_pct > 70:
                        return False, f"High keyword overlap ({overlap_pct:.0f}%) with recent YouTube stories"
    
    if check_carousel:
        carousel_tracker = load_carousel_tracker()
        
        if not carousel_tracker.get("used_titles") and not carousel_tracker.get("history"):
            pass
        else:
            if url:
                for entry in carousel_tracker.get("history", []):
                    if isinstance(entry, dict) and entry.get("news_source_url") == url:
                        return False, f"Exact URL already covered (Carousel): {url}"
            
            if RAPIDFUZZ_AVAILABLE:
                headlines_to_check = set(carousel_tracker.get("used_titles", []))
                for existing_title in headlines_to_check:
                    score = fuzz.token_set_ratio(title.lower(), existing_title.lower())
                    if score > SIMILARITY_THRESHOLD:
                        return False, f"Semantic match found (Carousel, score {score}): '{existing_title}'"
            
            if keywords:
                recent_keywords = set()
                for entry in carousel_tracker.get("history", [])[-20:]:
                    if url and entry.get("news_source_url") == url:
                        continue
                    recent_keywords.update([k.lower() for k in entry.get("keywords", [])])
                
                new_k_set = set([k.lower() for k in keywords])
                if new_k_set:
                    intersection = new_k_set.intersection(recent_keywords)
                    overlap_pct = (len(intersection) / len(new_k_set)) * 100
                    if overlap_pct > 70:
                        return False, f"High keyword overlap ({overlap_pct:.0f}%) with recent Carousel stories"
    
    return True, "Unique"

def filter_unique_stories(stories: List[Dict]) -> List[Dict]:
    """Filter stories to only include unique topics not previously covered (YouTube + Carousel)."""
    unique_stories = []
    for story in stories:
        title = story.get("title", "")
        url = story.get("url", "")
        description = story.get("description", "")
        
        keywords = []
        if description:
            keywords = re.findall(r'\b[A-Za-z]{4,}\b', description.lower())
        
        is_unique, reason = is_topic_unique(title, url, keywords, check_youtube=True, check_carousel=True)
        if is_unique:
            unique_stories.append(story)
        else:
            print(f"  🔄 Skipping duplicate topic: {title[:60]}... ({reason})")
    
    print(f"✅ Filtered {len(stories)} stories -> {len(unique_stories)} unique stories")
    return unique_stories

SCORING_WEIGHTS = {
    "freshness": 0.25,
    "developer_value": 0.25,
    "impact": 0.20,
    "novelty": 0.15,
    "visual_potential": 0.10,
    "audience_fit": 0.05,
}

MIN_SCORE_THRESHOLD = 4.5


def score_story(story: Dict) -> float:
    """Score a story based on weighted criteria."""
    scores = {
        "freshness": story.get("_freshness_score", 5),
        "developer_value": story.get("_developer_value_score", 5),
        "impact": story.get("_impact_score", 5),
        "novelty": story.get("_novelty_score", 5),
        "visual_potential": story.get("_visual_potential_score", 5),
        "audience_fit": story.get("_audience_fit_score", 5),
    }
    return sum(scores[k] * SCORING_WEIGHTS[k] for k in scores)


def select_best_story(stories: List[Dict], run_context: str = "") -> Optional[Dict]:
    """Select the best story based on scoring and record it in carousel tracker.
    
    Args:
        stories: List of candidate stories
        run_context: Optional context string (e.g., date + run number) for deterministic tiebreaking
    """
    if not stories:
        return None
    
    # Pre-filter: Check against carousel tracker BEFORE scoring to avoid duplicates
    carousel_tracker = load_carousel_tracker()
    recent_carousel_urls = set()
    recent_carousel_titles = set()
    for entry in carousel_tracker.get("history", [])[-20:]:  # Check last 20 carousel topics
        if isinstance(entry, dict):
            if entry.get("news_source_url"):
                recent_carousel_urls.add(entry["news_source_url"])
            if entry.get("title"):
                recent_carousel_titles.add(entry["title"].lower())
    
    filtered_stories = []
    for story in stories:
        url = story.get("url", "")
        title = story.get("title", "").lower()
        
        # Skip if exact URL already used in carousel
        if url and url in recent_carousel_urls:
            print(f"  🔄 Skipping (URL already in carousel history): {story.get('title')[:60]}...")
            continue
        
        # Skip if title semantically matches recent carousel titles
        if RAPIDFUZZ_AVAILABLE and title:
            is_dup = False
            for recent_title in recent_carousel_titles:
                score = fuzz.token_set_ratio(title, recent_title)
                if score > SIMILARITY_THRESHOLD:
                    print(f"  🔄 Skipping (semantic match in carousel history, score {score}): {story.get('title')[:60]}...")
                    is_dup = True
                    break
            if is_dup:
                continue
        
        filtered_stories.append(story)
    
    if not filtered_stories:
        print("⚠️ All stories filtered out as duplicates, falling back to original list")
        filtered_stories = stories
    
    scored = [(score_story(s), s) for s in filtered_stories]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Deterministic tiebreaking using run_context
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 0.1:
        # Scores are very close, use run_context hash for consistent selection
        import hashlib
        ctx_hash = int(hashlib.md5(run_context.encode()).hexdigest()[:8], 16)
        idx = ctx_hash % len(scored)
        best_score, best_story = scored[idx]
        print(f"🎲 Tiebreaker: Selected index {idx} from top {len(scored)} (run_context hash)")
    else:
        best_score, best_story = scored[0]
    
    if best_score >= MIN_SCORE_THRESHOLD:
        best_story["_score"] = best_score
        
        # Record the selected topic in carousel tracker to prevent reuse
        title = best_story.get("title", "")
        url = best_story.get("url", "")
        description = best_story.get("description", "")
        source = best_story.get("source", {}).get("name", "Unknown")
        
        keywords = []
        if description:
            keywords = re.findall(r'\b[A-Za-z]{4,}\b', description.lower())
        
        record_carousel_topic(title, url, keywords, source)
        
        return best_story
    return None


try:
    from content_validator import validate_carousel_content, repair_carousel
    CONTENT_VALIDATOR_AVAILABLE = True
except ImportError:
    CONTENT_VALIDATOR_AVAILABLE = False

AVAILABLE_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
]

def build_carousel_prompt(story: Dict, min_slides: int = 5, max_slides: int = 8) -> str:
    """Build a comprehensive prompt for LLM to generate rich, dynamic educational carousel content."""
    title = story.get("title", "")
    description = story.get("description", "")
    url = story.get("url", "")
    source = story.get("source", {}).get("name", "Unknown")
    category = story.get("_predicted_category", "AI Engineering")
    today_str = datetime.now().strftime('%d %b %Y')
    
    # Extract key technical entities from description for LLM to use
    import re
    version_matches = re.findall(r'\b(v?\d+\.\d+(\.\d+)?(-[a-zA-Z0-9]+)?)\b', description)
    metrics = re.findall(r'(\d+(?:\.\d+)?%|\d+(?:,\d{3})*(?:\.\d+)?[km]?|\$\d+(?:\.\d+)?[km]?|\d+(?:\.\d+)?x|\d+(?:\.\d+)?ms|\d+(?:\.\d+)?s|\d+(?:\.\d+)?GB|\d+(?:\.\d+)?TB)', description, re.IGNORECASE)
    code_keywords = re.findall(r'\b(API|SDK|CLI|GPU|CPU|RAM|VRAM|LLM|RAG|MCP|JSON|YAML|SQL|NoSQL|REST|GraphQL|gRPC|WebSocket|Docker|Kubernetes|Terraform|Ansible|CI/CD|GitHub|GitLab|VS Code|Cursor|Copilot|Python|TypeScript|JavaScript|Rust|Go|Java|C\+\+)\b', description, re.IGNORECASE)
    company_names = re.findall(r'\b(OpenAI|Anthropic|Google|Meta|Microsoft|Amazon|AWS|NVIDIA|Vercel|Netflix|Uber|Stripe|Airbnb|Shopify|Databricks|Snowflake|MongoDB|Redis|PostgreSQL|Elasticsearch|Kafka|RabbitMQ|Prometheus|Grafana|Datadog|LangChain|LlamaIndex|AutoGen|CrewAI|LangGraph|Ollama|LM Studio|vLLM|TGIF|Hugging Face)\b', description, re.IGNORECASE)
    
    extracted_info = []
    if version_matches:
        extracted_info.append(f"Versions mentioned: {', '.join(set(v[0] for v in version_matches))}")
    if metrics:
        extracted_info.append(f"Metrics mentioned: {', '.join(set(metrics))}")
    if code_keywords:
        extracted_info.append(f"Technologies mentioned: {', '.join(set(code_keywords))}")
    if company_names:
        extracted_info.append(f"Companies mentioned: {', '.join(set(company_names))}")
    
    extracted_context = "\n".join(extracted_info) if extracted_info else "No specific technical details extracted from description."
    
    return f"""You are an elite developer educator and tech visual designer creating high-engagement Instagram educational carousels for @vijayakumarj_ai (daily AI & engineering updates for software engineers, tech leads, and AI practitioners).

STORY CONTEXT:
Title: {title}
Description: {description}
Source: {source}
URL: {url}
Category: {category}
Date: {today_str}

EXTRACTED TECHNICAL DETAILS (USE THESE SPECIFICS):
{extracted_context}

OBJECTIVE:
Generate a visually varied, professional {min_slides} to {max_slides} slide carousel JSON.
Do NOT make every slide a boring bullet list.
Use dynamic storytelling layouts so each slide has a distinct, purposeful visual composition.

AVAILABLE LAYOUT TYPES:
1. "hero_hook" (Slide 1): High-impact bold hook headline, subtitle, and badge.
2. "architecture_diagram": System diagram with structured nodes & connections (e.g. Client -> Agent -> LLM -> VectorDB).
3. "process_flow": 3-4 numbered execution steps with titles & explanations.
4. "before_after": Clear before vs after comparison (e.g., Old way vs New paradigm).
5. "common_mistake": Wrong way (❌) vs Right way (✅) side-by-side or stacked.
6. "code_block": Real, copy-pasteable syntax-valid code snippet (Python, JS, or Bash) with explanation.
7. "real_world_scenario": Production use case from a named company (e.g. Netflix, Uber, Stripe) with Problem, Solution, Result.
8. "metrics_cards": 2-3 key metrics/benchmarks with labels, values, and deltas.
9. "side_by_side": Two-column comparison of 2 approaches/tools.
10. "quiz_choice" or "quiz_predict_output": Engaging quick question for engineers (with options, answer, explanation).
11. "checklist": 3-5 practical checklist items for implementing this tech.
12. "takeaway": Final slide with key lessons and CTA to follow @vijayakumarj_ai.

DETAILED CONTENT REQUIREMENTS PER SLIDE TYPE:

**hero_hook / big_number / scenario_question**: 
- Include SPECIFIC version numbers, release names, or metric headlines from the story
- Eyebrow: Category badge (e.g., "GEMINI 2.5 RELEASE", "RUST 1.80", "AWS LAMBDA UPDATE")

**architecture_diagram / process_flow / layered_stack / input_output / request_response / timeline**:
- Use ACTUAL component names, service names, API endpoints from the story
- Include specific protocols (gRPC, REST, WebSocket), data formats (Protobuf, JSON, Avro)
- Show real latency numbers, throughput if mentioned

**code_block / code_output**:
- Write REAL, RUNNABLE code using the ACTUAL library/SDK/API from the story
- Include specific method names, parameter values, config options from the release
- Show imports, initialization, and a concrete use case
- Language: match the story's ecosystem (Python for AI/ML, TypeScript for Vercel/Next.js, Rust for systems, Go for cloud)

**before_after / common_mistake / side_by_side / myth_vs_fact**:
- Compare SPECIFIC old vs new APIs, config flags, CLI commands
- Reference actual breaking changes, deprecated methods, new parameters
- Include version-specific migration details

**real_world_scenario / analogy**:
- Name ACTUAL companies/products from the story or well-known adopters
- Include SPECIFIC problem metrics (e.g., "500k req/s", "2TB/day", "99.99% SLA")
- Quote real results if available in source

**metrics_cards / three_column / checklist**:
- Use EXACT numbers from the story (latency, cost, throughput, context window, model size)
- If no numbers in story, infer realistic benchmarks for the technology class
- Include units: ms, %, $/1M tokens, GB, tokens/sec, requests/sec

**quiz_choice / quiz_predict_output**:
- Test KNOWLEDGE of the specific feature/API from the story
- Options should include plausible but incorrect alternatives
- Explanation must reference the story's technical details

**takeaway**:
- Action items SPECIFIC to this technology/release
- Include migration commands, config flags, documentation URLs
- Reference the actual version/release name

JSON OUTPUT SPECIFICATION:
Output strictly valid JSON with this structure (no markdown wrapping outside json):
{{
  "headline": "Short punchy headline under 60 chars - INCLUDE VERSION/RELEASE NAME",
  "summary": "2-3 sentence executive summary with SPECIFIC metrics/names from story",
  "source": "{source}",
  "source_url": "{url}",
  "date": "{today_str}",
  "category": "{category}",
  "slides": [
    {{
      "slide_number": 1,
      "layout_type": "hero_hook",
      "eyebrow": "SPECIFIC CATEGORY BADGE (e.g., GEMINI 2.5, RUST 1.80, AWS RE:INVENT)",
      "title": "Hook with SPECIFIC version/feature name from story",
      "subtitle": "One-sentence impact statement with concrete metric or capability",
      "body": "2-3 sentences: what changed, who it affects, why it matters NOW"
    }},
    {{ ... remaining slides with REAL data from story ... }}
  ]
}}

CRITICAL RULES:
1. Return strictly {min_slides} to {max_slides} slides tailored to THIS SPECIFIC STORY.
2. The first slide MUST be 'hero_hook' or 'big_number'.
3. The last slide MUST be 'takeaway'.
4. Include AT LEAST ONE code snippet or architecture/process diagram.
5. Include AT LEAST ONE comparison ('before_after', 'common_mistake', 'side_by_side') or real-world scenario.
6. Mobile readable: Keep bullet texts punchy (under 15 words each).
7. ABSOLUTELY NO generic placeholder text. Every field must reference the actual story.
8. Brand handle is @vijayakumarj_ai.
9. If story lacks specifics, infer realistic technical details for that technology class.
10. Code snippets MUST be syntactically correct and use real APIs from the story's ecosystem."""


def _get_active_gemini_model(client) -> str:
    """Find the best available Gemini model from candidate list."""
    for model_name in AVAILABLE_GEMINI_MODELS:
        try:
            # Quick check or return candidate
            return model_name
        except Exception:
            continue
    return "gemini-2.0-flash"


def generate_carousel_json(story: Dict, min_slides: int = 5, max_slides: int = 8) -> Dict:
    """Generate carousel content using LLM with available model resolution and content validation."""
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        print("ℹ️ Gemini not available or API key missing, generating dynamic fallback carousel")
        return generate_fallback_carousel(story)
    
    prompt = build_carousel_prompt(story, min_slides=min_slides, max_slides=max_slides)
    client = genai.Client(api_key=GEMINI_API_KEY)

    carousel = None
    last_error = None

    # Try preferred models in order
    for model_name in AVAILABLE_GEMINI_MODELS:
        try:
            print(f"🤖 Attempting generation with model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict) and "slides" in parsed and len(parsed["slides"]) >= 4:
                carousel = parsed
                carousel["_model_used"] = model_name
                print(f"✅ LLM generation successful with {model_name} ({len(carousel['slides'])} slides)")
                break
        except Exception as e:
            last_error = e
            print(f"⚠️ Model {model_name} failed: {e}")
            continue

    if not carousel:
        print(f"⚠️ All LLM models failed ({last_error}), using dynamic fallback")
        return generate_fallback_carousel(story)

    # Validate and repair content
    carousel["_story"] = story
    if CONTENT_VALIDATOR_AVAILABLE:
        is_valid, issues = validate_carousel_content(carousel)
        if not is_valid:
            print(f"⚠️ Content validator reported {len(issues)} issues, running auto-repair...")
            carousel = repair_carousel(carousel)
    
    return carousel


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate text at word boundary, adding ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.7:
        return truncated[:last_space] + '…'
    return truncated + '…'


def _extract_story_details(story: Dict) -> Dict[str, Any]:
    """Extract fine-grained technical entities, metrics, versions, and category from a news story."""
    title = story.get("title", "")
    description = story.get("description", "")
    source_name = story.get("source", {}).get("name", "") if isinstance(story.get("source"), dict) else str(story.get("source", ""))
    
    # 1. Company / Primary Ecosystem (Prioritize title and source over description comparisons)
    entity = "AI Engineering"
    ecosystem = "general"
    
    ecosystem_patterns = [
        (r'\b(Cursor|Anysphere)\b', "Cursor AI", "cursor"),
        (r'\b(LangGraph|LangChain)\b', "LangChain", "langgraph"),
        (r'\b(Vercel|AI SDK)\b', "Vercel", "vercel"),
        (r'\b(NVIDIA|Nemotron)\b', "NVIDIA", "nvidia"),
        (r'\b(Anthropic|Claude|Sonnet|Opus|Artifacts)\b', "Anthropic", "anthropic"),
        (r'\b(Meta|Llama|FAIR|PyTorch)\b', "Meta AI", "meta"),
        (r'\b(Google|Gemini|DeepMind|Gemma)\b', "Google", "google"),
        (r'\b(OpenAI|ChatGPT|GPT-4o|GPT-4|o1|o3)\b', "OpenAI", "openai"),
        (r'\b(Mistral|Mixtral|Codestral)\b', "Mistral AI", "mistral"),
        (r'\b(DeepSeek)\b', "DeepSeek", "deepseek"),
        (r'\b(Hugging\s*Face|Transformers)\b', "Hugging Face", "huggingface"),
        (r'\b(vLLM)\b', "vLLM", "vllm"),
        (r'\b(Ollama)\b', "Ollama", "ollama"),
        (r'\b(Microsoft|Copilot|Phi-3|Phi-4)\b', "Microsoft", "microsoft"),
    ]
    
    # First evaluate primary text (title + source)
    primary_text = f"{title} {source_name}"
    for pattern, ent_name, eco_key in ecosystem_patterns:
        if re.search(pattern, primary_text, re.IGNORECASE):
            entity = ent_name
            ecosystem = eco_key
            break
            
    # If not found in title/source, fallback to description
    if ecosystem == "general":
        for pattern, ent_name, eco_key in ecosystem_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                entity = ent_name
                ecosystem = eco_key
                break
            
    if entity == "AI Engineering" and source_name and source_name.lower() not in ["unknown", "ai insights", "newsletter_ai"]:
        entity = source_name

    # 2. Version / Parameter size
    full_text = f"{title} {description} {source_name}"
    version_match = re.search(r'\b(v?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?)\b', full_text)
    version = version_match.group(1) if version_match else ""
    
    param_match = re.search(r'\b(\d+(?:\.\d+)?[Bb])\b', full_text)
    param_size = param_match.group(1).upper() if param_match else ""
    
    # 3. Context window
    context_match = re.search(r'\b(\d+(?:\.\d+)?\s*(?:[KkMm]|million)?\s*(?:tokens?|context))\b', full_text, re.IGNORECASE)
    context_window = context_match.group(1).strip() if context_match else ""
    
    # 4. Metrics: Funding, Percentages, Latency, Dev count
    funding_match = re.search(r'(\$\d+(?:\.\d+)?\s*(?:[MBKmbk]|million|billion)?(?:\s*valuation)?)', full_text, re.IGNORECASE)
    funding_str = funding_match.group(1).strip() if funding_match else ""
    
    percent_matches = re.findall(r'(\d+(?:\.\d+)?%)', full_text)
    percentage = percent_matches[0] if percent_matches else ""
    
    devs_match = re.search(r'(\d+[KkMm]\+?\s*(?:developers|devs|users|engineers))', full_text, re.IGNORECASE)
    devs_count = devs_match.group(1).strip() if devs_match else ""
    
    benchmark_match = re.search(r'\b(HumanEval|MBPP|MMLU|GSM8K|MATH|SWE-bench|Chatbot Arena)\b', full_text, re.IGNORECASE)
    benchmark_name = benchmark_match.group(1) if benchmark_match else ""

    # 5. Technical Category Classification with word boundaries
    title_lower = title.lower()
    desc_lower = description.lower()
    full_lower = f"{title_lower} {desc_lower}"
    
    if re.search(r'\b(raises|raised|funding|valuation|series [abc]|seed round|acquired|acquisition)\b', full_lower):
        category = "funding_business"
    elif re.search(r'\b(beats|outperforms|benchmark|benchmarks|humaneval|mbpp|mmlu|sota|arxiv|paper)\b', title_lower):
        category = "research_benchmark"
    elif re.search(r'\b(editor|ide|cursor|copilot|autocomplete|code editing|multi-file)\b', title_lower):
        category = "developer_tool"
    elif re.search(r'\b(framework|sdk|library|langgraph|langchain|vercel|vllm|ollama)\b', title_lower) or re.search(r'\b(framework|sdk|library)\b', desc_lower):
        category = "framework_release"
    elif re.search(r'\b(gpt|claude|gemini|llama|nemotron|mistral|deepseek|model|weights|frontier model|context window)\b', full_lower):
        category = "model_release"
    elif re.search(r'\b(beats|outperforms|benchmark|benchmarks|humaneval|mbpp|mmlu|sota)\b', desc_lower):
        category = "research_benchmark"
    else:
        category = "general_ai"

    return {
        "entity": entity,
        "ecosystem": ecosystem,
        "version": version,
        "param_size": param_size,
        "context_window": context_window,
        "funding": funding_str,
        "percentage": percentage,
        "devs_count": devs_count,
        "benchmark": benchmark_name,
        "category": category,
        "title": title,
        "description": description,
        "source": source_name,
        "url": story.get("url", "")
    }


def _get_ecosystem_code_snippet(info: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return tailored (code, language, explanation) based on ecosystem & story details."""
    eco = info["ecosystem"]
    entity = info["entity"]
    
    if eco == "openai":
        code = (
            "from openai import OpenAI\n\n"
            "client = OpenAI()\n"
            "response = client.chat.completions.create(\n"
            "    model='gpt-4o',\n"
            "    messages=[\n"
            "        {'role': 'system', 'content': 'You are a high-throughput reasoning agent'},\n"
            "        {'role': 'user', 'content': 'Process real-time multimodal inputs'}\n"
            "    ],\n"
            "    response_format={'type': 'json_object'}\n"
            ")\n"
            "print(response.choices[0].message.content)"
        )
        return code, "python", "Native multimodal model interface with 50% reduced inference latency."

    elif eco == "anthropic":
        code = (
            "import anthropic\n\n"
            "client = anthropic.Anthropic()\n"
            "message = client.messages.create(\n"
            "    model='claude-3-5-sonnet-20241022',\n"
            "    max_tokens=2048,\n"
            "    messages=[\n"
            "        {'role': 'user', 'content': 'Architect production distributed cache with TTL'}\n"
            "    ]\n"
            ")\n"
            "print(message.content[0].text)"
        )
        return code, "python", "Artifacts-capable client with state-of-the-art coding benchmark performance."

    elif eco == "google":
        code = (
            "from google import genai\n\n"
            "client = genai.Client()\n"
            "response = client.models.generate_content(\n"
            "    model='gemini-2.0-flash',\n"
            "    contents='Analyze 2M token context repository architecture'\n"
            ")\n"
            "print(response.text)"
        )
        return code, "python", "Unified GenAI client handling massive context windows with sub-second time-to-first-token."

    elif eco == "meta":
        model_name = f"meta-llama/Meta-Llama-3.1-{info['param_size'] or '8B'}-Instruct"
        code = (
            "from transformers import pipeline\n"
            "import torch\n\n"
            f"pipe = pipeline(\n"
            f"    'text-generation',\n"
            f"    model='{model_name}',\n"
            "    torch_dtype=torch.bfloat16,\n"
            "    device_map='auto'\n"
            ")\n"
            "output = pipe('Benchmark memory throughput', max_new_tokens=256)\n"
            "print(output[0]['generated_text'])"
        )
        return code, "python", "Open weights pipeline deployment supporting commercial use and local inference."

    elif eco == "langgraph":
        code = (
            "from langgraph.graph import StateGraph, END\n"
            "from typing import TypedDict\n\n"
            "class AgentState(TypedDict):\n"
            "    messages: list\n"
            "    next_step: str\n\n"
            "builder = StateGraph(AgentState)\n"
            "builder.add_node('reason', run_reasoning_node)\n"
            "builder.add_edge('reason', END)\n"
            "app = builder.compile()"
        )
        return code, "python", "Cyclic multi-agent state graph with native checkpointing and human-in-the-loop support."

    elif eco == "vercel":
        code = (
            "import { streamText } from 'ai';\n"
            "import { openai } from '@ai-sdk/openai';\n\n"
            "export async function POST(req: Request) {\n"
            "  const { prompt } = await req.json();\n"
            "  const result = streamText({\n"
            "    model: openai('gpt-4o'),\n"
            "    prompt,\n"
            "  });\n"
            "  return result.toDataStreamResponse();\n"
            "}"
        )
        return code, "typescript", "Full React Server Components & streaming Generative UI pipeline in under 15 lines."

    elif eco == "cursor":
        code = (
            "# .cursorrules - Project Engineering Configuration\n"
            "version: 2.0\n"
            "rules:\n"
            "  - 'Always enforce TypeScript strict null checks'\n"
            "  - 'Use parameterized queries for all database mutations'\n"
            "  - 'Include automated unit tests for any new service method'\n"
            "  - 'Prefer streaming endpoints over buffered batch responses'"
        )
        return code, "yaml", "Deep codebase index instructions driving automated multi-file agentic refactoring."

    elif eco == "nvidia":
        code = (
            "from vllm import LLM, SamplingParams\n\n"
            "llm = LLM(\n"
            "    model='nvidia/Nemotron-3-Ultra',\n"
            "    tensor_parallel_size=2,\n"
            "    gpu_memory_utilization=0.90\n"
            ")\n"
            "prompts = ['Optimize low-level CUDA kernel execution']\n"
            "outputs = llm.generate(prompts, SamplingParams(temperature=0.2))\n"
            "print(outputs[0].outputs[0].text)"
        )
        return code, "python", "High-throughput vLLM engine inference achieving benchmark-topping coding scores."

    else:
        code = (
            "import httpx\n"
            "import asyncio\n\n"
            "# Production resilient AI gateway client\n"
            "async def call_ai_service(payload: dict) -> str:\n"
            "    async with httpx.AsyncClient(timeout=30.0) as client:\n"
            "        resp = await client.post(\n"
            "            'https://api.internal-ai.dev/v1/inference',\n"
            "            json=payload,\n"
            "            headers={'Authorization': 'Bearer $KEY'}\n"
            "        )\n"
            "        return resp.json()['result']"
        )
        return code, "python", "Production microservice integration pattern with structured validation and retries."


def _get_metrics_for_story(info: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build 3 precise, story-relevant metric cards."""
    cat = info["category"]
    
    if cat == "funding_business":
        return [
            {"label": "Funding Secured", "value": info["funding"] or "$60M+", "delta": "Series Round"},
            {"label": "Valuation", "value": "$400M+", "delta": "Market Cap"},
            {"label": "Active Builders", "value": info["devs_count"] or "40k+ Devs", "delta": "Rapid Adoption"}
        ]
    elif cat == "research_benchmark":
        bench = info["benchmark"] or "HumanEval"
        return [
            {"label": f"{bench} Score", "value": "88.4%", "delta": "+14% vs Baseline"},
            {"label": "Coding Accuracy", "value": "Top 1%", "delta": "SOTA Frontier"},
            {"label": "Model Weights", "value": "Open Source", "delta": "Commercial License"}
        ]
    elif cat == "developer_tool":
        return [
            {"label": "Dev Productivity", "value": "+40%", "delta": "Faster Delivery"},
            {"label": "Multi-File Edits", "value": "Autonomous", "delta": "Full Codebase"},
            {"label": "Daily Users", "value": info["devs_count"] or "40k+", "delta": "Growing Fast"}
        ]
    elif cat == "framework_release":
        ver_text = info["version"] or "Latest"
        return [
            {"label": "Release Version", "value": ver_text, "delta": "Production Stable"},
            {"label": "Execution Engine", "value": "Cyclic / Async", "delta": "Stateful Graph"},
            {"label": "Streaming Latency", "value": "< 20ms", "delta": "Zero Buffering"}
        ]
    else:  # model_release or general_ai
        ctx = info["context_window"] or ("2M Tokens" if "2m" in info["title"].lower() else "128k")
        cost_delta = f"-{info['percentage']}" if info["percentage"] else "-50% Drop"
        return [
            {"label": "Context Length", "value": ctx, "delta": "Massive Capacity"},
            {"label": "Inference Latency", "value": "180ms", "delta": "Sub-Second"},
            {"label": "API Cost", "value": "$0.075", "delta": cost_delta}
        ]


def generate_fallback_carousel(story: Dict) -> Dict:
    """Generate dynamic, high-quality, story-specific fallback carousel when LLM unavailable."""
    info = _extract_story_details(story)
    today_str = datetime.now().strftime('%d %b %Y')
    
    entity = info["entity"]
    category_raw = info["category"]
    cat_display_map = {
        "model_release": "FOUNDATION MODELS",
        "framework_release": "AGENT FRAMEWORKS",
        "developer_tool": "DEVELOPER TOOLING",
        "funding_business": "AI VENTURE & ECOSYSTEM",
        "research_benchmark": "BENCHMARKS & RESEARCH",
        "general_ai": "AI ENGINEERING"
    }
    category_badge = cat_display_map.get(category_raw, "AI ENGINEERING")
    
    headline = _truncate_at_word_boundary(info["title"], 55)
    summary = info["description"][:250] if info["description"] else f"Key architectural updates and production insights from {info['source']}."
    
    # Slide 1: Hook details
    hook_eyebrow = f"{entity.upper()} UPDATE"
    if info["version"]:
        hook_eyebrow = f"{entity.upper()} {info['version'].upper()}"
    elif info["param_size"]:
        hook_eyebrow = f"{entity.upper()} {info['param_size']}"
    hook_eyebrow = hook_eyebrow[:24]

    hook_title = _truncate_at_word_boundary(info["title"], 50)
    hook_subtitle = f"What engineers must know about {entity}'s latest production release."
    hook_body = (
        f"{info['description'][:140]} " if len(info["description"]) > 20
        else f"Major architectural upgrade announced for {entity} with real-world developer impact. "
    )
    if not any(char in (hook_title + hook_body) for char in ["?", "!", "what", "why", "how"]):
        hook_body += "What does this mean for your production architecture?"

    # Slide 2: Pipeline Steps tailored to category
    if category_raw == "framework_release":
        steps_data = [
            {"step": "1", "title": "Define State Schema", "desc": "Declare shared context types & memory stores."},
            {"step": "2", "title": "Compile Cyclic Graph", "desc": "Route conditional branches with fallback logic."},
            {"step": "3", "title": "Stream Execution", "desc": "Yield structured tokens with persistent checkpoints."}
        ]
    elif category_raw == "developer_tool":
        steps_data = [
            {"step": "1", "title": "Index Repository AST", "desc": "Map symbols, dependencies, and call trees."},
            {"step": "2", "title": "Semantic Retrieval", "desc": "Ground reasoning with relevant workspace files."},
            {"step": "3", "title": "Multi-File Patching", "desc": "Execute cohesive diffs across the entire project."}
        ]
    elif category_raw == "funding_business":
        steps_data = [
            {"step": "1", "title": "Prove Developer Traction", "desc": "Drive bottom-up adoption across engineering teams."},
            {"step": "2", "title": "Scale Compute Infrastructure", "desc": "Secure GPU clusters & low-latency inference nodes."},
            {"step": "3", "title": "Expand Enterprise Tier", "desc": "Roll out SOC2 compliance, SSO, and audit telemetry."}
        ]
    elif category_raw == "research_benchmark":
        steps_data = [
            {"step": "1", "title": "Standardized Evaluation", "desc": "Run rigorous tests across coding and logic suites."},
            {"step": "2", "title": "Chain-of-Thought Audit", "desc": "Verify reasoning accuracy with zero data leakage."},
            {"step": "3", "title": "Open Weight Verification", "desc": "Validate reproducible weights across community GPUs."}
        ]
    else:  # model_release / general_ai
        steps_data = [
            {"step": "1", "title": "Multimodal Ingestion", "desc": "Parse text, vision, and audio in a single pass."},
            {"step": "2", "title": "Attention Routing", "desc": "Process context through high-throughput KV cache."},
            {"step": "3", "title": "Strict Tool Dispatch", "desc": "Generate typed arguments matching JSON schemas."}
        ]

    # Slide 3: Before vs After tailored to category
    if category_raw == "framework_release":
        before_items = [
            "Brittle linear chains with hard-coded logic",
            "Stateless execution without time-travel debugging",
            "Complex custom glue code for agent handoffs"
        ]
        after_items = [
            "Cyclic state graphs with native looping support",
            "Built-in state persistence and checkpointing",
            "Unified streaming interface with observability"
        ]
    elif category_raw == "developer_tool":
        before_items = [
            "Context-blind single-file autocompletions",
            "Manual copy-pasting of terminal errors",
            "High cognitive load when refactoring large repos"
        ]
        after_items = [
            "Global codebase awareness and deep semantic search",
            "Autonomous terminal command execution and debugging",
            "One-click multi-file edits with instant rollback"
        ]
    elif category_raw == "funding_business":
        before_items = [
            "Resource constraints limiting compute capacity",
            "Slow release cycles due to constrained GPU access",
            "Single-tenant infrastructure bottlenecks"
        ]
        after_items = [
            "Massive capital expansion into frontier model training",
            "Enterprise-grade SLA guarantees and zero-latency clusters",
            "Accelerated product velocity and talent acquisition"
        ]
    elif category_raw == "research_benchmark":
        before_items = [
            "High hallucination rates on complex edge cases",
            "Heavy latency overhead during deep reasoning steps",
            "Vendor lock-in with closed proprietary APIs"
        ]
        after_items = [
            "SOTA scores on HumanEval and real-world coding",
            "Sub-second response times with optimized kernels",
            "Open weights available for private on-prem deployment"
        ]
    else:  # model_release / general_ai
        before_items = [
            "Rigid 8k-32k token limits restricting input scope",
            "High per-token API costs hindering production scale",
            "Separate models needed for voice, vision, and text"
        ]
        after_items = [
            "Massive context capacity handling full codebases",
            "Up to 50% lower inference costs for all workloads",
            "Unified native multimodal reasoning in real time"
        ]

    # Slide 4: Code block & explanation
    code_text, code_lang, code_expl = _get_ecosystem_code_snippet(info)

    # Slide 5: Metrics cards
    metrics_data = _get_metrics_for_story(info)

    # Slide 6: Takeaways
    if category_raw == "framework_release":
        takeaways_data = [
            f"Upgrade {entity} dependencies to unlock stateful cyclic graphs.",
            "Implement typed state schemas for deterministic agent execution.",
            "Enable persistent checkpointing before deploying to production."
        ]
    elif category_raw == "developer_tool":
        takeaways_data = [
            f"Configure project rules for {entity} to maintain coding standards.",
            "Leverage multi-file refactoring to clear legacy technical debt.",
            "Integrate automated linting and test execution in workflows."
        ]
    elif category_raw == "funding_business":
        takeaways_data = [
            f"Track {entity}'s roadmap for upcoming enterprise features.",
            "Benchmark API pricing as competitive rounds reduce token costs.",
            "Evaluate self-hosted vs managed tiers for your architecture."
        ]
    elif category_raw == "research_benchmark":
        takeaways_data = [
            f"Run independent benchmarks of {entity} on your internal data.",
            "Audit licensing terms before shipping open weights to production.",
            "Explore quantizations (AWQ/GGUF) for cost-effective local serving."
        ]
    else:
        takeaways_data = [
            f"Evaluate {entity}'s latest capabilities against existing models.",
            "Take advantage of larger context windows to simplify RAG pipelines.",
            "Track production latency and token spend to capture cost savings."
        ]

    slides = [
        {
            "slide_number": 1,
            "layout_type": "hero_hook",
            "type": "hook",
            "eyebrow": hook_eyebrow,
            "title": hook_title,
            "subtitle": hook_subtitle,
            "body": hook_body
        },
        {
            "slide_number": 2,
            "layout_type": "process_flow",
            "type": "what_happened",
            "eyebrow": "UNDER THE HOOD",
            "title": "Core Execution Pipeline",
            "body": f"How {entity} executes this workflow in modern production systems:",
            "steps": steps_data
        },
        {
            "slide_number": 3,
            "layout_type": "before_after",
            "type": "whats_new",
            "eyebrow": "PARADIGM SHIFT",
            "title": "Before vs After This Update",
            "body": "Comparing previous engineering constraints with the new architecture.",
            "before_title": "Legacy Approach",
            "before_items": before_items,
            "after_title": "Modern Architecture",
            "after_items": after_items
        },
        {
            "slide_number": 4,
            "layout_type": "code_block",
            "type": "real_world_example",
            "eyebrow": "PRODUCTION CODE",
            "title": "Clean Minimal Snippet",
            "body": f"Real-world implementation example using {entity}'s official API:",
            "code": code_text,
            "language": code_lang,
            "code_explanation": code_expl
        },
        {
            "slide_number": 5,
            "layout_type": "metrics_cards",
            "type": "why_matters",
            "eyebrow": "PERFORMANCE IMPACT",
            "title": "Measurable Benchmarks",
            "body": f"Production efficiency and developer metrics for {entity}:",
            "metrics": metrics_data
        },
        {
            "slide_number": 6,
            "layout_type": "takeaway",
            "type": "takeaway_cta",
            "eyebrow": "KEY TAKEAWAYS",
            "title": "Action Items For Engineers",
            "body": "Immediate steps software engineers and architects should take:",
            "takeaways": takeaways_data,
            "cta": "Save this guide • Follow @vijayakumarj_ai for daily AI engineering"
        }
    ]

    return {
        "headline": headline,
        "summary": summary,
        "source": info["source"] or "AI Engineering",
        "source_url": info["url"],
        "date": today_str,
        "category": category_badge,
        "slides": slides,
        "_story": story
    }


def fetch_ai_news_stories(run_context: str = "") -> List[Dict]:
    """Fetch AI news stories from trending engine with fallback to curated samples.
    
    Args:
        run_context: Optional context for source rotation awareness
    """
    if not TRENDING_ENGINE_AVAILABLE:
        return filter_unique_stories(get_fallback_stories())
    
    try:
        print("🔍 Fetching AI trending signals...")
        # Use default sources from config (includes source rotation)
        from config import TRENDING_SOURCES as DEFAULT_SOURCES
        signals = fetch_all_trending_signals(
            target_country="US",
            category="AI & Tech Tools",
            sources=DEFAULT_SOURCES
        )
        
        stories = []
        for signal in signals:
            story = {
                "title": signal.get("title", "Untitled"),
                "description": signal.get("description", ""),
                "url": signal.get("url", ""),
                "source": signal.get("source", {"name": "Unknown"}),
                "type": signal.get("type", ""),
                "_engagement": signal.get("_engagement", {}),
                "_freshness_score": min(10, signal.get("_engagement", {}).get("views", 0) / 10000 + 3),
                "_developer_value_score": 7 if "developer" in signal.get("description", "").lower() or "api" in signal.get("description", "").lower() else 5,
                "_impact_score": 6,
                "_novelty_score": 6,
                "_visual_potential_score": 5,
                "_audience_fit_score": 7,
            }
            stories.append(story)
        
        if stories:
            print(f"✅ Fetched {len(stories)} AI news stories from trending engine")
            return filter_unique_stories(stories)
        
        print("⚠️ Trending engine returned no signals, using fallback stories")
        return filter_unique_stories(get_fallback_stories(run_context=run_context))
        
    except Exception as e:
        print(f"⚠️ Failed to fetch trending signals: {e}, using fallback")
        return filter_unique_stories(get_fallback_stories(run_context=run_context))


def get_fallback_stories(run_context: str = "") -> List[Dict]:
    """Provide curated fallback AI news stories when trending engine unavailable.
    
    Args:
        run_context: Optional context for deterministic but varied selection
    """
    import random
    import hashlib
    
    # Use run_context for seed if available, otherwise date
    if run_context:
        seed = int(hashlib.md5(run_context.encode()).hexdigest()[:8], 16)
    else:
        today = datetime.now()
        seed = int(today.strftime("%Y%m%d"))
    random.seed(seed)
    
    fallback_stories = [
        {
            "title": "OpenAI Releases GPT-4o with Real-time Voice and Vision",
            "description": "OpenAI launched GPT-4o, a multimodal model with real-time voice conversation, vision understanding, and improved coding capabilities. Available via API with 50% lower cost than GPT-4 Turbo.",
            "url": "https://openai.com/index/gpt-4o/",
            "source": {"name": "OpenAI"},
            "type": "newsletter_ai",
            "_engagement": {"views": 500000},
            "_freshness_score": 9,
            "_developer_value_score": 9,
            "_impact_score": 10,
            "_novelty_score": 9,
            "_visual_potential_score": 8,
            "_audience_fit_score": 10,
        },
        {
            "title": "Anthropic Launches Claude 3.5 Sonnet with Artifacts Feature",
            "description": "Claude 3.5 Sonnet outperforms GPT-4o on coding benchmarks. New Artifacts feature lets developers see, edit, and iterate on code, documents, and designs in a dedicated window alongside the chat.",
            "url": "https://www.anthropic.com/news/claude-3-5-sonnet",
            "source": {"name": "Anthropic"},
            "type": "newsletter_ai",
            "_engagement": {"views": 300000},
            "_freshness_score": 9,
            "_developer_value_score": 10,
            "_impact_score": 9,
            "_novelty_score": 8,
            "_visual_potential_score": 9,
            "_audience_fit_score": 10,
        },
        {
            "title": "Google Gemini 1.5 Pro Gets 2M Context Window",
            "description": "Google expanded Gemini 1.5 Pro's context window to 2 million tokens, enabling analysis of entire codebases, hours of video, or hundreds of documents in a single prompt. Available in AI Studio and Vertex AI.",
            "url": "https://blog.google/technology/ai/google-gemini-update-2m-context/",
            "source": {"name": "Google"},
            "type": "newsletter_ai",
            "_engagement": {"views": 200000},
            "_freshness_score": 8,
            "_developer_value_score": 9,
            "_impact_score": 9,
            "_novelty_score": 8,
            "_visual_potential_score": 7,
            "_audience_fit_score": 9,
        },
        {
            "title": "Meta Releases Llama 3.1 405B - Open Source Frontier Model",
            "description": "Meta open-sourced Llama 3.1 405B, the largest openly available foundation model. Matches GPT-4o on benchmarks. Includes 8B and 70B variants. Available for commercial use with permissive license.",
            "url": "https://ai.meta.com/blog/meta-llama-3-1/",
            "source": {"name": "Meta AI"},
            "type": "huggingface_trending",
            "_engagement": {"views": 400000},
            "_freshness_score": 9,
            "_developer_value_score": 10,
            "_impact_score": 10,
            "_novelty_score": 9,
            "_visual_potential_score": 8,
            "_audience_fit_score": 10,
        },
        {
            "title": "Cursor AI Editor Raises $60M at $400M Valuation",
            "description": "AI-powered code editor Cursor secured Series A funding. Features include natural language editing, multi-file changes, and deep codebase understanding. 40k+ developers using it daily.",
            "url": "https://cursor.sh/blog/series-a",
            "source": {"name": "Cursor"},
            "type": "newsletter_ai",
            "_engagement": {"views": 150000},
            "_freshness_score": 8,
            "_developer_value_score": 10,
            "_impact_score": 8,
            "_novelty_score": 7,
            "_visual_potential_score": 8,
            "_audience_fit_score": 10,
        },
        {
            "title": "LangGraph v0.1 Released - Production-Ready Agent Framework",
            "description": "LangChain released LangGraph v0.1 with stateful agents, human-in-the-loop, and streaming support. Enables building complex multi-agent workflows with cyclic graphs and persistence.",
            "url": "https://blog.langchain.dev/langgraph-v0-1/",
            "source": {"name": "LangChain"},
            "type": "github_trending",
            "_engagement": {"views": 100000},
            "_freshness_score": 8,
            "_developer_value_score": 9,
            "_impact_score": 8,
            "_novelty_score": 7,
            "_visual_potential_score": 7,
            "_audience_fit_score": 9,
        },
        {
            "title": "NVIDIA Nemotron 3 Ultra Beats GPT-4 on Coding Benchmarks",
            "description": "NVIDIA's new Nemotron 3 Ultra model achieves state-of-the-art results on HumanEval and MBPP coding benchmarks. Released with open weights for research and commercial use.",
            "url": "https://research.nvidia.com/labs/nemotron/",
            "source": {"name": "NVIDIA"},
            "type": "arxiv_papers",
            "_engagement": {"views": 80000},
            "_freshness_score": 7,
            "_developer_value_score": 9,
            "_impact_score": 8,
            "_novelty_score": 8,
            "_visual_potential_score": 6,
            "_audience_fit_score": 8,
        },
        {
            "title": "Vercel AI SDK 3.0 - Streaming, Tools, and Generative UI",
            "description": "Vercel released AI SDK 3.0 with React Server Components support, generative UI components, and improved streaming. Makes it trivial to build AI-native applications with Next.js.",
            "url": "https://vercel.com/blog/ai-sdk-3",
            "source": {"name": "Vercel"},
            "type": "github_trending",
            "_engagement": {"views": 120000},
            "_freshness_score": 8,
            "_developer_value_score": 9,
            "_impact_score": 8,
            "_novelty_score": 7,
            "_visual_potential_score": 8,
            "_audience_fit_score": 9,
        },
    ]
    
    # Score all fallback stories
    for story in fallback_stories:
        story["_score"] = (
            story["_freshness_score"] * 0.25 +
            story["_developer_value_score"] * 0.25 +
            story["_impact_score"] * 0.20 +
            story["_novelty_score"] * 0.15 +
            story["_visual_potential_score"] * 0.10 +
            story["_audience_fit_score"] * 0.05
        )
    
    # Sort by score and return top stories
    fallback_stories.sort(key=lambda x: x["_score"], reverse=True)
    selected = fallback_stories[:5]
    
    print(f"✅ Using {len(selected)} curated fallback AI news stories")
    return selected


def main():
    parser = argparse.ArgumentParser(description="Generate AI News Carousel")
    parser.add_argument("--topic", type=str, help="Specific topic/title to generate")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--output-dir", type=str, default="output/social_images", help="Output directory")
    parser.add_argument("--run-context", type=str, help="Run context for deterministic selection (e.g., date-runNumber)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    story = None
    
    if args.topic:
        stories = fetch_ai_news_stories(run_context=args.run_context or "")
        story = next((s for s in stories if args.topic.lower() in s.get("title", "").lower()), None)
        if not story:
            print(f"❌ Topic not found: {args.topic}")
            sys.exit(1)
    else:
        stories = fetch_ai_news_stories(run_context=args.run_context or "")
        story = select_best_story(stories, run_context=args.run_context or "")
        if not story:
            print("❌ No stories meet quality threshold")
            sys.exit(1)
    
    print(f"📰 Selected story: {story.get('title')}")
    print(f"   Score: {story.get('_score', 'N/A')}")
    print(f"   Source: {story.get('source', {}).get('name')}")
    
    carousel = generate_carousel_json(story)
    
    safe_title = "".join(c for c in story.get("title", "ai_news") if c.isalnum() or c in " -_").strip()[:50]
    safe_title = safe_title.replace(" ", "_")
    
    carousel_path = output_dir / f"carousel_{safe_title}.json"
    with open(carousel_path, "w") as f:
        json.dump(carousel, f, indent=2)
    
    print(f"✅ Carousel JSON saved: {carousel_path}")
    
    if args.dry_run:
        print(json.dumps(carousel, indent=2))
    
    return carousel_path


if __name__ == "__main__":
    main()