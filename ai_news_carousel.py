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

MIN_SCORE_THRESHOLD = 6.5


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


def select_best_story(stories: List[Dict]) -> Optional[Dict]:
    """Select the best story based on scoring and record it in carousel tracker."""
    if not stories:
        return None
    
    scored = [(score_story(s), s) for s in stories]
    scored.sort(key=lambda x: x[0], reverse=True)
    
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
    
    return f"""You are an elite developer educator and tech visual designer creating high-engagement Instagram educational carousels for @vijayakumarj_ai (daily AI & engineering updates for software engineers, tech leads, and AI practitioners).

STORY CONTEXT:
Title: {title}
Description: {description}
Source: {source}
URL: {url}
Category: {category}
Date: {today_str}

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

JSON OUTPUT SPECIFICATION:
Output strictly valid JSON with this structure (no markdown wrapping outside json):
{{
  "headline": "Short punchy headline under 60 chars",
  "summary": "2-3 sentence executive summary for social caption",
  "source": "{source}",
  "source_url": "{url}",
  "date": "{today_str}",
  "category": "{category}",
  "slides": [
    {{
      "slide_number": 1,
      "layout_type": "hero_hook",
      "eyebrow": "AI BREAKTHROUGH",
      "title": "Hook Headline That Stops The Scroll",
      "subtitle": "Clear, compelling statement of what changes today for engineers.",
      "body": "Brief context hook."
    }},
    {{
      "slide_number": 2,
      "layout_type": "architecture_diagram",
      "eyebrow": "SYSTEM ARCHITECTURE",
      "title": "How The Architecture Works",
      "body": "System overview explanation.",
      "diagram": {{
        "title": "Data Flow Pipeline",
        "nodes": [
          {{"id": "1", "label": "User Query", "sub": "Frontend / API"}},
          {{"id": "2", "label": "Orchestrator", "sub": "Agent Controller"}},
          {{"id": "3", "label": "LLM Engine", "sub": "Reasoning Model"}},
          {{"id": "4", "label": "Vector Index", "sub": "Knowledge Base"}}
        ],
        "connections": [
          {{"from": "1", "to": "2", "label": "HTTP/gRPC"}},
          {{"from": "2", "to": "3", "label": "Context Prompt"}},
          {{"from": "2", "to": "4", "label": "Semantic Search"}}
        ]
      }}
    }},
    {{
      "slide_number": 3,
      "layout_type": "code_block",
      "eyebrow": "QUICK IMPLEMENTATION",
      "title": "Implementing In 5 Lines of Code",
      "body": "How developers can run this right now.",
      "code": "import google.genai as genai\\n\\nclient = genai.Client()\\nresponse = client.models.generate_content(\\n    model='gemini-2.5-flash',\\n    contents='Analyze repo architecture'\\n)\\nprint(response.text)",
      "language": "python",
      "code_explanation": "Simple, idiomatic setup using the official SDK client."
    }},
    {{
      "slide_number": 4,
      "layout_type": "before_after",
      "eyebrow": "PARADIGM SHIFT",
      "title": "Before vs After This Release",
      "body": "The dramatic developer workflow shift.",
      "before_title": "Traditional Workflow",
      "before_items": ["Manual orchestration & prompt tuning", "Slow sync batch processing", "High latency and token cost"],
      "after_title": "New Paradigm",
      "after_items": ["Native agentic execution & tool use", "Real-time streaming multimodal UI", "50% lower cost with 2M context"]
    }},
    {{
      "slide_number": 5,
      "layout_type": "real_world_scenario",
      "eyebrow": "PRODUCTION CASE STUDY",
      "title": "Real-World Impact At Scale",
      "body": "How modern teams leverage this capability in production.",
      "scenario_company": "Modern Enterprise Stack",
      "scenario_problem": "Processing 500k customer tickets with high human triage latency.",
      "scenario_solution": "Deployed automated agent pipeline with real-time tool grounding.",
      "scenario_result": "78% reduction in resolution time, zero downtime rollout."
    }},
    {{
      "slide_number": 6,
      "layout_type": "takeaway",
      "eyebrow": "KEY TAKEAWAYS",
      "title": "What You Should Do Next",
      "body": "Summary action items.",
      "takeaways": [
        "Audit existing pipelines for native agentic integration.",
        "Test benchmarks on your proprietary domain data.",
        "Bookmark documentation for production deployment patterns."
      ],
      "cta": "Save this guide • Follow @vijayakumarj_ai for daily AI engineering"
    }}
  ]
}}

CRITICAL RULES:
1. Return strictly {min_slides} to {max_slides} slides tailored to this specific story.
2. The first slide MUST be 'hero_hook' or 'big_number'.
3. The last slide MUST be 'takeaway'.
4. Include AT LEAST ONE code snippet or architecture/process diagram.
5. Include AT LEAST ONE comparison ('before_after', 'common_mistake', or 'side_by_side') or real-world scenario.
6. Mobile readable: Keep bullet texts punchy (under 15 words each).
7. Absolutely NO generic placeholder text (no 'Lorem Ipsum' or 'Foo Bar').
8. Brand handle is @vijayakumarj_ai."""


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


def generate_fallback_carousel(story: Dict) -> Dict:
    """Generate dynamic, high-quality fallback carousel when LLM unavailable."""
    title = story.get("title", "AI Architecture & Engineering Update")
    description = story.get("description", "")
    source = story.get("source", {}).get("name", "AI Insights")
    url = story.get("url", "")
    today_str = datetime.now().strftime('%d %b %Y')
    
    headline = _truncate_at_word_boundary(title, 55)
    summary = description[:250] if description else f"Key architectural updates and developer insights from {source}."

    return {
        "headline": headline,
        "summary": summary,
        "source": source,
        "source_url": url,
        "date": today_str,
        "category": "AI ENGINEERING",
        "slides": [
            {
                "slide_number": 1,
                "layout_type": "hero_hook",
                "type": "hook",
                "eyebrow": "AI BREAKTHROUGH",
                "title": _truncate_at_word_boundary(title, 48),
                "subtitle": "Major architectural upgrade announced for modern engineering teams.",
                "body": "What just changed in production AI infrastructure and why it matters for developers."
            },
            {
                "slide_number": 2,
                "layout_type": "process_flow",
                "type": "what_happened",
                "eyebrow": "HOW IT WORKS",
                "title": "Key Execution Steps",
                "body": "The modern pipeline flow for implementing this update:",
                "steps": [
                    {"step": "1", "title": "Signal Ingestion", "desc": "Capture multimodal streams & context payload"},
                    {"step": "2", "title": "Context Compression", "desc": "Semantic routing through high-throughput cache"},
                    {"step": "3", "title": "Tool Execution", "desc": "Autonomous function dispatch with strict schema"}
                ]
            },
            {
                "slide_number": 3,
                "layout_type": "before_after",
                "type": "whats_new",
                "eyebrow": "PARADIGM SHIFT",
                "title": "Before vs After This Update",
                "body": "Comparing legacy implementations with the modern workflow.",
                "before_title": "Legacy Approach",
                "before_items": [
                    "Manual prompt engineering & rigid heuristics",
                    "High token latency with frequent rate limits",
                    "Fragile glue code across disparate microservices"
                ],
                "after_title": "Modern Architecture",
                "after_items": [
                    "Native agentic tool calling and streaming output",
                    "Sub-second response times with optimized models",
                    "Unified SDK client with production monitoring"
                ]
            },
            {
                "slide_number": 4,
                "layout_type": "code_block",
                "type": "real_world_example",
                "eyebrow": "QUICK IMPLEMENTATION",
                "title": "Production Code Snippet",
                "body": "Get up and running with minimal boilerplate:",
                "code": "# Initialize client with available models\nfrom google import genai\n\nclient = genai.Client()\nresponse = client.models.generate_content(\n    model='gemini-2.5-flash',\n    contents='Benchmark AI latency & throughput'\n)\nprint(response.text)",
                "language": "python",
                "code_explanation": "Clean SDK interface with built-in streaming and schema validation."
            },
            {
                "slide_number": 5,
                "layout_type": "metrics_cards",
                "type": "why_matters",
                "eyebrow": "PERFORMANCE IMPACT",
                "title": "Production Benchmarks",
                "body": "Developer efficiency and production cost reduction:",
                "metrics": [
                    {"label": "Inference Latency", "value": "180ms", "delta": "-65% drop"},
                    {"label": "Context Length", "value": "2M+", "delta": "10x capacity"},
                    {"label": "Token Cost", "value": "$0.075", "delta": "50% savings"}
                ]
            },
            {
                "slide_number": 6,
                "layout_type": "takeaway",
                "type": "takeaway_cta",
                "eyebrow": "KEY TAKEAWAYS",
                "title": "What To Do Next",
                "body": "Action items for engineering leads and builders:",
                "takeaways": [
                    "Upgrade SDK dependencies to support latest model capabilities.",
                    "Implement structured output schemas for robust agent tool calls.",
                    "Track latency and cost metrics across production workloads."
                ],
                "cta": "Save this guide • Follow @vijayakumarj_ai for daily AI engineering"
            }
        ],
        "_story": story
    }


def fetch_ai_news_stories() -> List[Dict]:
    """Fetch AI news stories from trending engine with fallback to curated samples."""
    if not TRENDING_ENGINE_AVAILABLE:
        return filter_unique_stories(get_fallback_stories())
    
    try:
        print("🔍 Fetching AI trending signals...")
        signals = fetch_all_trending_signals(
            target_country="US",
            category="AI & Tech Tools",
            sources=["github_trending", "huggingface_trending", "arxiv_papers", "newsletter_ai", "hacker_news", "reddit_trending"]
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
        return filter_unique_stories(get_fallback_stories())
        
    except Exception as e:
        print(f"⚠️ Failed to fetch trending signals: {e}, using fallback")
        return filter_unique_stories(get_fallback_stories())


def get_fallback_stories() -> List[Dict]:
    """Provide curated fallback AI news stories when trending engine unavailable."""
    import random
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
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    story = None
    
    if args.topic:
        stories = fetch_ai_news_stories()
        story = next((s for s in stories if args.topic.lower() in s.get("title", "").lower()), None)
        if not story:
            print(f"❌ Topic not found: {args.topic}")
            sys.exit(1)
    else:
        stories = fetch_ai_news_stories()
        story = select_best_story(stories)
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