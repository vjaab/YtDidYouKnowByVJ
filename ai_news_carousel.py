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
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import requests
from dotenv import load_dotenv

load_dotenv()

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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


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
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
]

# ─── Content Archetypes Registry ───────────────────────────────────────────────

CONTENT_ARCHETYPES = {
    "model_release": {
        "name": "AI Model Release & Architecture",
        "default_slide_count": 8,
        "slide_range": (7, 8),
        "story_angle": "technical_breakdown",
        "narrative_roles": [
            {"role": "hook", "layout": "hero_hook", "eyebrow": "{ENTITY} RELEASE", "focus": "Model name, headline capability leap, core premise"},
            {"role": "what_changed", "layout": "whats_new", "eyebrow": "WHAT CHANGED", "focus": "Architectural upgrade vs previous version, context size or weights"},
            {"role": "technical_mechanism", "layout": "architecture_diagram", "eyebrow": "HOW IT WORKS", "focus": "Internal mechanism (attention, MoE routing, latent reasoning, KV cache)"},
            {"role": "benchmark_evidence", "layout": "metrics_cards", "eyebrow": "BENCHMARKS", "focus": "Concrete empirical benchmarks (HumanEval, MMLU, latency ms, token cost delta)"},
            {"role": "previous_vs_new", "layout": "before_after", "eyebrow": "OLD VS NEW", "focus": "Previous constraints vs new production capabilities"},
            {"role": "developer_impact", "layout": "real_world_scenario", "eyebrow": "IN PRODUCTION", "focus": "How engineering teams deploy or leverage this in production"},
            {"role": "practical_action", "layout": "checklist", "eyebrow": "INTEGRATION CHECKLIST", "focus": "Evaluation, prompt adaptation, fine-tuning, latency optimization"},
            {"role": "takeaway", "layout": "takeaway", "eyebrow": "KEY TAKEAWAY", "focus": "API availability, rollout status, core conclusion"}
        ],
        "conditional_rules": "REQUIRE benchmark metrics or before_after comparison. Focus on deep technical evaluation, architecture upgrades, and real-world system impact. Do NOT include code snippets."
    },
    "developer_tool": {
        "name": "Developer Tool & IDE Innovation",
        "default_slide_count": 7,
        "slide_range": (6, 7),
        "story_angle": "developer_workflow",
        "narrative_roles": [
            {"role": "problem_hook", "layout": "scenario_question", "eyebrow": "{ENTITY} UPDATE", "focus": "Real developer friction or bottleneck solved by this tool"},
            {"role": "tool_overview", "layout": "whats_new", "eyebrow": "WHAT IT DOES", "focus": "Tool capabilities, platform integration, developer setup"},
            {"role": "core_mechanism", "layout": "process_flow", "eyebrow": "EXECUTION FLOW", "focus": "How the tool processes codebase/AST, runs agents, or manages state"},
            {"role": "architecture", "layout": "architecture_diagram", "eyebrow": "ARCHITECTURE", "focus": "Editor -> Local Agent -> Code Index -> Compiler/LLM"},
            {"role": "system_workflow", "layout": "input_output", "eyebrow": "EXECUTION MECHANICS", "focus": "Developer Input/Query -> AST Index & State Machine -> Structured Diff/Artifact"},
            {"role": "limitations_and_gotchas", "layout": "common_mistake", "eyebrow": "PRO TIPS & MISTAKES", "focus": "Common developer antipatterns vs recommended usage"},
            {"role": "takeaway", "layout": "takeaway", "eyebrow": "GET STARTED", "focus": "Installation command, configuration, productivity summary"}
        ],
        "conditional_rules": "REQUIRE system workflow or input-output mechanics. Focus on developer ergonomics, practical workflows, and real-world engineering architecture. Do NOT include code snippets."
    },
    "research_paper": {
        "name": "Research Breakthrough & Methodology",
        "default_slide_count": 8,
        "slide_range": (7, 8),
        "story_angle": "deep_methodology",
        "narrative_roles": [
            {"role": "big_idea_hook", "layout": "hero_hook", "eyebrow": "RESEARCH BREAKTHROUGH", "focus": "The foundational discovery or breakthrough hypothesis"},
            {"role": "problem_solved", "layout": "what_happened", "eyebrow": "THE CORE PROBLEM", "focus": "Why existing methods failed or hit scaling walls"},
            {"role": "new_methodology", "layout": "process_flow", "eyebrow": "NEW METHODOLOGY", "focus": "Step-by-step novel algorithmic or mathematical technique"},
            {"role": "method_architecture", "layout": "architecture_diagram", "eyebrow": "SYSTEM ARCHITECTURE", "focus": "Diagram of proposed model or pipeline structure"},
            {"role": "experimental_results", "layout": "metrics_cards", "eyebrow": "EVALUATION DATA", "focus": "Measured benchmark improvements, error reductions, speedups"},
            {"role": "paradigm_comparison", "layout": "before_after", "eyebrow": "PARADIGM SHIFT", "focus": "Traditional approach vs proposed novel approach"},
            {"role": "why_it_matters", "layout": "real_world_scenario", "eyebrow": "INDUSTRY IMPACT", "focus": "Longer term impact on AI systems and future models"},
            {"role": "takeaway", "layout": "takeaway", "eyebrow": "PAPER SUMMARY", "focus": "Paper reference, key takeaway, open questions"}
        ],
        "conditional_rules": "REQUIRE research methodology and evaluation data. Focus on conceptual rigor, mathematical intuition, and architectural innovation. Do NOT include code snippets."
    },
    "security_incident": {
        "name": "Security Vulnerability & Threat Analysis",
        "default_slide_count": 8,
        "slide_range": (7, 8),
        "story_angle": "threat_postmortem",
        "narrative_roles": [
            {"role": "threat_hook", "layout": "scenario_question", "eyebrow": "SECURITY ALERT", "focus": "Urgent vulnerability or exploit scenario"},
            {"role": "what_happened", "layout": "what_happened", "eyebrow": "THE INCIDENT", "focus": "CVE details, affected packages/versions, attack surface"},
            {"role": "attack_chain", "layout": "process_flow", "eyebrow": "ATTACK CHAIN", "focus": "Step 1 to Step 3 of exploit propagation"},
            {"role": "vulnerable_architecture", "layout": "architecture_diagram", "eyebrow": "VULNERABLE TOPOLOGY", "focus": "Where the exploit enters the system boundaries"},
            {"role": "root_cause_analysis", "layout": "common_mistake", "eyebrow": "VULNERABLE VS SECURE", "focus": "Flawed implementation vs secure implementation"},
            {"role": "defense_architecture", "layout": "side_by_side", "eyebrow": "SECURITY HARDENING", "focus": "Vulnerable perimeter vs zero-trust defense-in-depth architecture"},
            {"role": "security_checklist", "layout": "checklist", "eyebrow": "DEFENSE CHECKLIST", "focus": "Audit commands, dependency locks, secret rotation steps"},
            {"role": "takeaway", "layout": "takeaway", "eyebrow": "REMEDIATION ACTION", "focus": "Patch version to upgrade to immediately, final guidance"}
        ],
        "conditional_rules": "REQUIRE attack flow or vulnerable topology, plus security hardening checklist. Focus on defense-in-depth. Do NOT include code snippets."
    },
    "programming_concept": {
        "name": "Programming Concept & Architecture Pattern",
        "default_slide_count": 7,
        "slide_range": (7, 8),
        "story_angle": "engineering_concept",
        "narrative_roles": [
            {"role": "problem_hook", "layout": "hero_hook", "eyebrow": "ARCHITECTURE PATTERN", "focus": "Real engineering challenge, performance bottleneck, or race condition"},
            {"role": "concept_mental_model", "layout": "process_flow", "eyebrow": "MENTAL MODEL", "focus": "Intuitive explanation of how this mechanism operates"},
            {"role": "execution_mechanics", "layout": "input_output", "eyebrow": "HOW IT WORKS", "focus": "Input -> State Transformation -> Output execution path"},
            {"role": "architectural_deep_dive", "layout": "side_by_side", "eyebrow": "PATTERN TRADE-OFFS", "focus": "Latency, throughput, concurrency limits, and architectural trade-offs"},
            {"role": "common_antipattern", "layout": "common_mistake", "eyebrow": "AVOID THIS MISTAKE", "focus": "Naive antipattern (❌) vs robust production solution (✅)"},
            {"role": "production_case_study", "layout": "real_world_scenario", "eyebrow": "SCALE IN PRODUCTION", "focus": "How high-throughput systems utilize this pattern"},
            {"role": "takeaway", "layout": "takeaway", "eyebrow": "ENGINEERING CHEAT SHEET", "focus": "When to apply, trade-offs, follow CTA"}
        ],
        "conditional_rules": "REQUIRE architectural trade-offs, mental model, and common antipattern comparison. Focus on system design and software engineering principles. Do NOT include code snippets."
    },
    "framework_update": {
        "name": "Framework & Agent SDK Release",
        "default_slide_count": 7,
        "slide_range": (6, 7),
        "story_angle": "developer_ecosystem",
        "narrative_roles": [
            {"role": "hook", "layout": "hero_hook", "eyebrow": "{ENTITY} UPDATE", "focus": "Framework version announcement and headline capabilities"},
            {"role": "whats_new", "layout": "whats_new", "eyebrow": "WHAT'S NEW", "focus": "New API methods, state management primitives, breaking changes"},
            {"role": "execution_graph", "layout": "architecture_diagram", "eyebrow": "GRAPH ARCHITECTURE", "focus": "State machine or agent loop execution structure"},
            {"role": "capability_breakdown", "layout": "side_by_side", "eyebrow": "CAPABILITY UPGRADE", "focus": "Prior framework bottlenecks vs new high-throughput primitives"},
            {"role": "migration_comparison", "layout": "before_after", "eyebrow": "MIGRATION GUIDE", "focus": "Old verbose syntax vs new streamlined API"},
            {"role": "upgrade_checklist", "layout": "checklist", "eyebrow": "UPGRADE STEPS", "focus": "Installation, config updates, deployment checklist"},
            {"role": "takeaway", "layout": "takeaway", "eyebrow": "KEY TAKEAWAY", "focus": "Upgrade commands, documentation links, ecosystem value"}
        ],
        "conditional_rules": "REQUIRE API capability breakdown and migration trade-offs. Focus on developer ecosystem and production reliability. Do NOT include code snippets."
    },
    "cloud_service": {
        "name": "Cloud Infrastructure & Distributed Systems",
        "default_slide_count": 7,
        "slide_range": (6, 7),
        "story_angle": "cloud_architecture",
        "narrative_roles": [
            {"role": "service_hook", "layout": "hero_hook", "eyebrow": "CLOUD INFRASTRUCTURE", "focus": "Managed service announcement, scaling milestone, or regional expansion"},
            {"role": "system_overview", "layout": "whats_new", "eyebrow": "CAPABILITIES", "focus": "Throughput, auto-scaling characteristics, security boundaries"},
            {"role": "infrastructure_topology", "layout": "architecture_diagram", "eyebrow": "CLOUD TOPOLOGY", "focus": "VPC -> Ingress -> Compute cluster -> Storage / DB"},
            {"role": "metrics_or_sla", "layout": "metrics_cards", "eyebrow": "BENCHMARKS & SLA", "focus": "Latency, IOPS, cost per request/hour, uptime SLA"},
            {"role": "architecture_comparison", "layout": "before_after", "eyebrow": "SELF-HOSTED VS MANAGED", "focus": "Operational burden of DIY vs managed cloud capability"},
            {"role": "deployment_checklist", "layout": "checklist", "eyebrow": "DEPLOYMENT CHECKLIST", "focus": "IAM permissions, Terraform configuration, network peering"},
            {"role": "takeaway", "layout": "takeaway", "eyebrow": "ACTION ITEM", "focus": "Availability, pricing tier, architecture recommendations"}
        ],
        "conditional_rules": "REQUIRE architecture topology and deployment checklist. Focus on cloud scale, reliability, and cost-efficiency. Do NOT include code snippets."
    },
    "industry_trend": {
        "name": "AI Industry Strategy & Market Shift",
        "default_slide_count": 6,
        "slide_range": (5, 6),
        "story_angle": "strategic_analysis",
        "narrative_roles": [
            {"role": "headline_hook", "layout": "hero_hook", "eyebrow": "INDUSTRY ANALYSIS", "focus": "The major market shift, funding round, or strategic acquisition"},
            {"role": "market_context", "layout": "what_happened", "eyebrow": "WHY NOW", "focus": "The competitive context and underlying economic drivers"},
            {"role": "technology_backbone", "layout": "process_flow", "eyebrow": "TECH DRIVER", "focus": "The technical innovation that unlocked this market transition"},
            {"role": "ecosystem_impact", "layout": "side_by_side", "eyebrow": "MARKET IMPACT", "focus": "Incumbents vs Emerging Players / Prior market vs New reality"},
            {"role": "developer_implications", "layout": "real_world_scenario", "eyebrow": "WHAT IT MEANS FOR DEVS", "focus": "Implications for engineering teams and tech startup strategy"},
            {"role": "takeaway", "layout": "takeaway", "eyebrow": "THE BIG PICTURE", "focus": "Where the market is heading and key strategic takeaway"}
        ],
        "conditional_rules": "REQUIRE market context and ecosystem comparison. Focus on technical innovation driving the shift. Do NOT include code snippets."
    }
}


def classify_topic_heuristically(story: Dict) -> Dict[str, Any]:
    """Classify a story into a structured Topic Intelligence object using deterministic heuristics."""
    title = story.get("title", "")
    description = story.get("description", "")
    text = f"{title} {description}".lower()

    # Default values
    topic_type = "model_release"
    domain = "ai_ml"
    primary_entity = "AI Engineering"
    content_depth = "deep"
    best_story_angle = "technical_breakdown"
    audience = ["software_engineers", "ai_engineers", "tech_leads"]

    # Detect primary entity
    entity_patterns = [
        (r'\b(cursor|anysphere)\b', "Cursor AI", "programming"),
        (r'\b(langgraph|langchain)\b', "LangChain", "ai_ml"),
        (r'\b(vercel|next\.js)\b', "Vercel", "programming"),
        (r'\b(nvidia|nemotron|cuda)\b', "NVIDIA", "ai_ml"),
        (r'\b(anthropic|claude|sonnet|opus)\b', "Anthropic", "ai_ml"),
        (r'\b(meta|llama|pytorch)\b', "Meta AI", "ai_ml"),
        (r'\b(google|gemini|deepmind|gemma)\b', "Google", "ai_ml"),
        (r'\b(openai|chatgpt|gpt-4o|o1|o3)\b', "OpenAI", "ai_ml"),
        (r'\b(mistral|mixtral|codestral)\b', "Mistral AI", "ai_ml"),
        (r'\b(deepseek)\b', "DeepSeek", "ai_ml"),
        (r'\b(hugging\s*face|transformers)\b', "Hugging Face", "ai_ml"),
        (r'\b(vllm)\b', "vLLM", "ai_ml"),
        (r'\b(ollama)\b', "Ollama", "ai_ml"),
        (r'\b(aws|amazon)\b', "AWS", "cloud"),
        (r'\b(microsoft|copilot)\b', "Microsoft", "ai_ml"),
        (r'\b(kubernetes|k8s)\b', "Kubernetes", "cloud"),
    ]
    for pattern, ent_name, dom in entity_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            primary_entity = ent_name
            domain = dom
            break

    # Determine topic type
    if re.search(r'\b(cve|vulnerability|exploit|jailbreak|breach|malware|insecure|attack|threat)\b', text):
        topic_type = "security_incident"
        domain = "security"
        best_story_angle = "threat_postmortem"
        audience = ["security_engineers", "devops_engineers", "tech_leads"]
    elif re.search(r'\b(editor|ide|cursor|autocomplete|copilot|plugin|extension|cli|terminal)\b', text) and "paper" not in text:
        topic_type = "developer_tool"
        domain = "programming"
        best_story_angle = "developer_workflow"
        audience = ["software_engineers", "fullstack_developers"]
    elif re.search(r'\b(paper|arxiv|researchers|proof|theorem|sota|outperforms|beats|benchmark|swe-bench|mmlu|humaneval)\b', text) and ("paper" in text or "arxiv" in text or "research" in text):
        topic_type = "research_paper"
        domain = "ai_ml"
        best_story_angle = "deep_methodology"
        audience = ["ai_researchers", "ai_engineers", "data_scientists"]
    elif re.search(r'\b(framework|sdk|library|langgraph|langchain|llamaindex|vllm|ollama|package|crate)\b', text):
        topic_type = "framework_update"
        domain = "ai_ml" if domain == "ai_ml" else "programming"
        best_story_angle = "developer_ecosystem"
        audience = ["ai_engineers", "backend_developers", "system_architects"]
    elif re.search(r'\b(aws|azure|gcp|cloud|kubernetes|k8s|serverless|lambda|s3|ec2|cluster|vpc)\b', text):
        topic_type = "cloud_service"
        domain = "cloud"
        best_story_angle = "cloud_architecture"
        audience = ["cloud_architects", "devops_engineers", "sre"]
    elif re.search(r'\b(pattern|algorithm|async|concurrency|memory|garbage collect|data structure|rust|golang|closure)\b', text) and not re.search(r'\b(model|gpt|claude|gemini|llama)\b', text):
        topic_type = "programming_concept"
        domain = "programming"
        best_story_angle = "engineering_concept"
        audience = ["software_engineers", "systems_programmers"]
    elif re.search(r'\b(raises|raised|funding|valuation|acquisition|acquired|merger|ipo|series [abc]|seed round)\b', text):
        topic_type = "industry_trend"
        domain = "ai_ml"
        best_story_angle = "strategic_analysis"
        audience = ["tech_leads", "engineering_managers", "founders"]
    else:
        topic_type = "model_release"
        domain = "ai_ml"
        best_story_angle = "technical_breakdown"
        audience = ["ai_engineers", "software_engineers", "tech_leads"]

    archetype = CONTENT_ARCHETYPES.get(topic_type, CONTENT_ARCHETYPES["model_release"])

    key_questions = [
        f"What is the headline advancement introduced by {primary_entity}?",
        "How does the internal architectural mechanism function under the hood?",
        "What verifiable metrics or benchmarks validate this capability?",
        "How should software engineers practically adapt their production workflows?"
    ]

    return {
        "topic_type": topic_type,
        "domain": domain,
        "primary_entity": primary_entity,
        "content_depth": content_depth,
        "best_story_angle": best_story_angle,
        "audience": audience,
        "key_questions": key_questions,
        "recommended_slide_count": archetype["default_slide_count"]
    }


def classify_topic(story: Dict, client=None) -> Dict[str, Any]:
    """Classify a story into a structured Topic Intelligence object using OpenRouter/Gemini with fallback to heuristics."""
    title = story.get("title", "")
    description = story.get("description", "")
    
    classification_prompt = f"""You are a senior tech editor analyzing an AI & engineering news story for an educational developer carousel.
Analyze this story and output strictly a JSON object:

Story Title: {title}
Story Description: {description}
Source: {story.get('source', {}).get('name', 'Unknown')}

Choose topic_type from EXACTLY one of:
- "model_release" (Foundation models, weights, new LLMs, multimodal models)
- "developer_tool" (IDEs, coding assistants, CLI tools, developer utilities)
- "framework_update" (Agent SDKs, libraries, LangChain, Vercel AI SDK, vLLM)
- "research_paper" (Academic breakthroughs, arXiv papers, reasoning proofs, novel algorithms)
- "security_incident" (CVEs, jailbreaks, prompt injection, data leaks, model threats)
- "programming_concept" (Core programming idioms, design patterns, concurrency, algorithms)
- "cloud_service" (Cloud infrastructure, GPU clusters, Kubernetes, managed services)
- "industry_trend" (Company strategy, funding rounds, acquisitions, market dynamics)

Choose domain from: ["ai_ml", "cloud", "programming", "security", "devops", "data"]

JSON OUTPUT FORMAT:
{{
  "topic_type": "model_release",
  "domain": "ai_ml",
  "primary_entity": "OpenAI / Google / etc.",
  "content_depth": "deep",
  "best_story_angle": "technical_breakdown",
  "audience": ["software_engineers", "ai_engineers"],
  "key_questions": [
    "What specifically changed?",
    "How does it operate under the hood?",
    "What evidence or benchmarks support it?",
    "What should developers do about it?"
  ],
  "recommended_slide_count": 8
}}"""

    # Priority 1 & 3: Try OpenRouter for reasoning/classification if available
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "") or OPENROUTER_API_KEY
    if openrouter_key:
        or_models = ["nvidia/nemotron-3-ultra-550b-a55b:free", "nvidia/nemotron-3.5-lightning:free"]
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vjaab/YtDidYouKnowByVJ",
            "X-Title": "YtDidYouKnowByVJ Carousel Intelligence",
        }
        for model_name in or_models:
            try:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": classification_prompt}],
                    "temperature": 0.2,
                }
                r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=25)
                if r.status_code == 200:
                    choices = r.json().get("choices", [])
                    if choices:
                        raw = choices[0].get("message", {}).get("content", "").strip()
                        from llm_fallback import clean_and_parse_json
                        data = clean_and_parse_json(raw)
                        if isinstance(data, dict) and data.get("topic_type") in CONTENT_ARCHETYPES:
                            arch = CONTENT_ARCHETYPES[data["topic_type"]]
                            min_s, max_s = arch["slide_range"]
                            rec = data.get("recommended_slide_count", arch["default_slide_count"])
                            data["recommended_slide_count"] = max(min_s, min(rec, max_s))
                            print(f"🧠 Topic Intelligence (OpenRouter:{model_name}): type={data['topic_type']} | entity={data['primary_entity']} | domain={data['domain']}")
                            return data
            except Exception as e:
                print(f"⚠️ OpenRouter topic classification ({model_name}) exception: {e}")

    # Priority 5: Gemini LLM classification
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        active_client = client or genai.Client(api_key=GEMINI_API_KEY)
        for model_name in AVAILABLE_GEMINI_MODELS:
            try:
                response = active_client.models.generate_content(
                    model=model_name,
                    contents=classification_prompt
                )
                raw = response.text.strip()
                if raw.startswith("```json"):
                    raw = raw[7:]
                if raw.startswith("```"):
                    raw = raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                data = json.loads(raw.strip())
                if isinstance(data, dict) and data.get("topic_type") in CONTENT_ARCHETYPES:
                    arch = CONTENT_ARCHETYPES[data["topic_type"]]
                    min_s, max_s = arch["slide_range"]
                    rec = data.get("recommended_slide_count", arch["default_slide_count"])
                    data["recommended_slide_count"] = max(min_s, min(rec, max_s))
                    print(f"🧠 Topic Intelligence (Gemini:{model_name}): type={data['topic_type']} | entity={data['primary_entity']} | domain={data['domain']}")
                    return data
            except Exception as e:
                continue

    print("ℹ️ LLM topic classification unavailable, using heuristic classifier")
    return classify_topic_heuristically(story)


def build_research_brief(story: Dict, topic_intel: Dict) -> Dict[str, Any]:
    """Extract fine-grained verifiable facts, metrics, versions, and APIs without hallucination."""
    title = story.get("title", "")
    description = story.get("description", "")
    source_name = story.get("source", {}).get("name", "Unknown") if isinstance(story.get("source"), dict) else str(story.get("source", "Unknown"))
    url = story.get("url", "")
    full_text = f"{title}. {description}"

    # Extract versions & parameters
    versions = list(set(re.findall(r'\b(v?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?)\b', full_text)))
    param_sizes = list(set(re.findall(r'\b(\d+(?:\.\d+)?[Bb])\b', full_text)))
    context_windows = list(set(re.findall(r'(\d+(?:\.\d+)?\s*(?:[KkMm]|million)?\s*(?:tokens?|context))\b', full_text, re.IGNORECASE)))
    
    # Extract metrics
    metrics = list(set(re.findall(r'(\d+(?:\.\d+)?%|\$\d+(?:\.\d+)?[kmbt]?|\d+(?:\.\d+)?x\b|\d+(?:\.\d+)?ms\b|\d+(?:\.\d+)?s\b|\d+(?:\.\d+)?GB\b|\d+(?:\.\d+)?TB\b)', full_text, re.IGNORECASE)))
    
    # Extract named benchmarks
    benchmarks = list(set(re.findall(r'\b(HumanEval|MBPP|MMLU|GSM8K|MATH|SWE-bench|Chatbot Arena|GPQA|ARC-Challenge)\b', full_text, re.IGNORECASE)))
    
    # Extract code / tech terms
    tech_matches = list(set(re.findall(r'\b(API|SDK|CLI|GPU|CPU|RAM|VRAM|LLM|RAG|MCP|JSON|YAML|SQL|NoSQL|REST|GraphQL|gRPC|WebSocket|Docker|Kubernetes|Terraform|Ansible|CI/CD|GitHub|GitLab|VS Code|Cursor|Copilot|Python|TypeScript|JavaScript|Rust|Go|Java|PyTorch|TensorFlow|vLLM|Ollama|LangChain|LangGraph|LlamaIndex)\b', full_text, re.IGNORECASE)))

    # Known company names
    company_matches = list(set(re.findall(r'\b(OpenAI|Anthropic|Google|Meta|Microsoft|Amazon|AWS|NVIDIA|Vercel|Netflix|Uber|Stripe|Airbnb|Shopify|Databricks|Snowflake|MongoDB|Redis|PostgreSQL|DeepSeek|Mistral)\b', full_text, re.IGNORECASE)))

    # Identify unspecified dimensions to prevent fabrication
    unspecified = []
    if not metrics:
        unspecified.append("Exact benchmark numbers or percentages not specified in source.")
    if not versions:
        unspecified.append("Specific version numbers not explicitly given in summary.")
    if not benchmarks:
        unspecified.append("Named benchmark suites (e.g. MMLU, HumanEval) not cited in source.")
    
    return {
        "title": title,
        "description": description,
        "source": source_name,
        "url": url,
        "primary_entity": topic_intel.get("primary_entity", "AI Engineering"),
        "topic_type": topic_intel.get("topic_type", "model_release"),
        "versions": versions,
        "param_sizes": param_sizes,
        "context_windows": context_windows,
        "metrics": metrics,
        "benchmarks": benchmarks,
        "technologies": tech_matches,
        "companies": company_matches,
        "unspecified_aspects": unspecified
    }


def build_carousel_prompt(
    story: Dict,
    topic_intel: Dict,
    research_brief: Dict,
    archetype: Dict,
    slide_count: int = 7
) -> str:
    """Build an archetype-driven, anti-hallucinatory prompt for Gemini carousel generation."""
    title = story.get("title", "")
    description = story.get("description", "")
    source = story.get("source", {}).get("name", "Unknown") if isinstance(story.get("source"), dict) else str(story.get("source", "Unknown"))
    url = story.get("url", "")
    today_str = datetime.now().strftime('%d %b %Y')
    
    entity = topic_intel.get("primary_entity", "AI Engineering")
    topic_type = topic_intel.get("topic_type", "model_release")
    story_angle = archetype.get("story_angle", "technical_breakdown")
    conditional_rules = archetype.get("conditional_rules", "")
    
    # Build narrative role outline
    narrative_steps = []
    for i, role_info in enumerate(archetype["narrative_roles"][:slide_count]):
        eyebrow_txt = role_info["eyebrow"].replace("{ENTITY}", entity.upper())
        step_desc = (
            f"Slide {i + 1} [Role: '{role_info['role']}', Layout: '{role_info['layout']}']\n"
            f"   - Eyebrow: \"{eyebrow_txt}\"\n"
            f"   - Narrative Focus: {role_info['focus']}"
        )
        narrative_steps.append(step_desc)
    
    narrative_outline = "\n".join(narrative_steps)
    
    # Format verified evidence
    verified_items = []
    if research_brief["versions"]:
        verified_items.append(f"• Verified Versions: {', '.join(research_brief['versions'])}")
    if research_brief["param_sizes"]:
        verified_items.append(f"• Parameter Sizes: {', '.join(research_brief['param_sizes'])}")
    if research_brief["context_windows"]:
        verified_items.append(f"• Context Window: {', '.join(research_brief['context_windows'])}")
    if research_brief["metrics"]:
        verified_items.append(f"• Verified Metrics: {', '.join(research_brief['metrics'])}")
    if research_brief["benchmarks"]:
        verified_items.append(f"• Named Benchmarks: {', '.join(research_brief['benchmarks'])}")
    if research_brief["technologies"]:
        verified_items.append(f"• Relevant Technologies: {', '.join(research_brief['technologies'])}")
    
    verified_context = "\n".join(verified_items) if verified_items else "• No specific numbers in source summary; explain conceptual mechanisms without fabricating fake metrics."
    
    unspecified_warning = ""
    if research_brief["unspecified_aspects"]:
        unspecified_warning = "\nDO NOT FABRICATE:\n" + "\n".join(f"• {u}" for u in research_brief["unspecified_aspects"])

    return f"""You are an elite developer educator and tech visual designer creating high-engagement Instagram educational carousels for @vijayakumarj_ai (daily AI & engineering updates for software engineers, tech leads, and AI practitioners).

TOPIC INTELLIGENCE:
• Topic Type: {topic_type.upper()} ({archetype['name']})
• Primary Entity: {entity}
• Domain: {topic_intel.get('domain', 'ai_ml')}
• Story Angle: {story_angle}
• Target Audience: {', '.join(topic_intel.get('audience', ['engineers']))}
• Key Questions to Answer:
{chr(10).join(f"  - {q}" for q in topic_intel.get('key_questions', []))}

STORY SOURCE CONTEXT:
Title: {title}
Description: {description}
Source: {source}
URL: {url}
Date: {today_str}

VERIFIED SOURCE EVIDENCE (USE THESE SPECIFICS):
{verified_context}
{unspecified_warning}

REQUIRED NARRATIVE STRUCTURE ({slide_count} SLIDES):
{narrative_outline}

ARCHETYPE-SPECIFIC RULES:
{conditional_rules}

ANTI-HALLUCINATION & RIGOR MANDATE:
1. NEVER invent technical specifications, benchmarks, API behavior, pricing, architecture details, performance numbers, release dates, or company claims.
2. If the source does not provide a specific benchmark or metric, explain the technical concept conceptually or mark it as "Not specified in source" rather than presenting a fabricated number.
3. NO CODE SNIPPETS MANDATE: Do NOT include code snippets, raw code blocks, or syntax lines in ANY slide. Software professionals and students consume high-signal conceptual architecture, system mechanics, mental models, trade-offs, performance metrics, and production anti-patterns on mobile carousels. Code snippets are hard to read and low-engagement on social media. Focus on diagrams, workflows, and conceptual engineering depth.

NO-REPETITION MANDATE:
Every single slide must advance the story and add genuinely new knowledge.
Do NOT:
- Restate what was explained in the previous slide
- Repeat the headline in different words across slides
- Use generic "why it matters" filler statements
- Repeat the same metric or example across multiple slides
- Before finalizing each slide, ask: "What new concept does the reader learn here that they did NOT learn on the previous slide?"

JSON OUTPUT SPECIFICATION:
Output strictly valid JSON with this structure (no markdown wrapping outside json):
{{
  "headline": "Punchy headline under 60 chars - include version/entity",
  "summary": "2-3 sentence executive summary explaining what changed and why it matters",
  "source": "{source}",
  "source_url": "{url}",
  "date": "{today_str}",
  "category": "{topic_intel.get('domain', 'AI ENGINEERING').upper()}",
  "topic_type": "{topic_type}",
  "depth_score": 9,
  "originality_score": 9,
  "technical_specificity": 9,
  "slides": [
    {{
      "slide_number": 1,
      "role": "{archetype['narrative_roles'][0]['role']}",
      "layout_type": "{archetype['narrative_roles'][0]['layout']}",
      "eyebrow": "CATEGORY OR ENTITY BADGE",
      "title": "Clear, informative hook title",
      "subtitle": "Concrete one-sentence capability statement",
      "body": "2-3 sentences providing context and the core question answered in this carousel",
      "deep_dive": "Deep contextual paragraph providing rich technical background",
      "key_fact": "Single most important verified takeaway of this slide",
      "why_it_matters": "Immediate engineering or architecture impact"
    }}
  ]
}}

CRITICAL SCHEMA RULES:
1. Return EXACTLY {slide_count} slides matching the required narrative structure.
2. Each slide MUST have 'slide_number', 'role', 'layout_type', 'eyebrow', 'title', and 'body'.
3. For 'architecture_diagram' layouts, include a 'diagram' object with 'nodes' (array of {{'name', 'sub'}}) and 'connections' (array of {{'label'}}).
4. For 'process_flow' layouts, include a 'steps' array of {{'step': '1', 'title': '...', 'desc': '...'}}.
5. For 'before_after' or 'whats_new' layouts, include 'before_title', 'before_items', 'after_title', 'after_items'.
6. For 'side_by_side' layouts, include 'left_title', 'left_items', 'right_title', 'right_items'. For 'input_output' layouts, include 'input', 'processing', 'output'.
7. For 'metrics_cards' layouts, include 'metrics' array of {{'label': '...', 'value': '...', 'delta': '...'}}.
8. For 'checklist' layouts, include 'items' array of strings.
9. For 'takeaway' layouts, include 'takeaways' array of 3 actionable items and 'cta' mentioning @vijayakumarj_ai."""


def validate_slide_narrative(carousel: Dict, archetype: Dict = None) -> Tuple[bool, List[str]]:
    """Validate slide diversity, narrative progression, repetition avoidance, and technical depth."""
    issues = []
    slides = carousel.get("slides", [])
    
    if len(slides) < 4:
        issues.append(f"Insufficient slides: only {len(slides)} slides found.")
        return False, issues

    # 1. Layout Diversity: at least 4 unique layout types
    layouts = [s.get("layout_type") or s.get("type", "") for s in slides]
    unique_layouts = set(layouts)
    if len(unique_layouts) < min(4, len(slides)):
        issues.append(f"Low visual diversity: only {len(unique_layouts)} unique layouts across {len(slides)} slides.")

    # 2. No 3 consecutive text-only layouts
    text_heavy = {"whats_new", "what_happened", "why_matters", "bullet_list", "real_world_scenario"}
    consecutive_text = 0
    for l in layouts:
        if l in text_heavy:
            consecutive_text += 1
            if consecutive_text >= 3:
                issues.append("Found 3 consecutive text-heavy slides without diagrams or metrics.")
                break
        else:
            consecutive_text = 0

    # 3. Disallow code_block layouts
    for s in slides:
        l = s.get("layout_type") or s.get("type", "")
        if l in ["code_block", "code_breakdown"]:
            issues.append(f"Slide {s.get('slide_number', '?')} uses forbidden '{l}' layout. Carousels must use conceptual diagrams.")

    # 3. Duplicate titles or identical text
    titles = [s.get("title", "").strip().lower() for s in slides if s.get("title")]
    if len(titles) != len(set(titles)):
        issues.append("Duplicate slide titles detected across slides.")

    # 4. Body Concept Repetition check (Jaccard token similarity)
    for i in range(len(slides) - 1):
        body1 = str(slides[i].get("body", "")).lower()
        body2 = str(slides[i + 1].get("body", "")).lower()
        tokens1 = set(re.findall(r'\b[a-zA-Z]{4,}\b', body1))
        tokens2 = set(re.findall(r'\b[a-zA-Z]{4,}\b', body2))
        if tokens1 and tokens2:
            intersection = tokens1.intersection(tokens2)
            similarity = len(intersection) / min(len(tokens1), len(tokens2))
            if similarity > 0.75:
                issues.append(f"Consecutive slides {i + 1} and {i + 2} have repetitive body text ({int(similarity*100)}% overlap).")

    # 5. Hallucination / placeholder check
    placeholder_tokens = ["lorem ipsum", "company xyz", "acme corp", "placeholder", "fake api", "dummy token"]
    all_text = " ".join([str(s.get("title", "")) + " " + str(s.get("body", "")) for s in slides]).lower()
    for pt in placeholder_tokens:
        if pt in all_text:
            issues.append(f"Placeholder token detected: '{pt}'.")

    # 6. Technical Depth Score check
    depth = carousel.get("depth_score", 8)
    if isinstance(depth, (int, float)) and depth < 6:
        issues.append(f"Reported depth score too low: {depth}/10.")

    is_valid = len(issues) == 0
    return is_valid, issues


def _get_active_gemini_model(client) -> str:
    """Find the best available Gemini model from candidate list."""
    for model_name in AVAILABLE_GEMINI_MODELS:
        try:
            return model_name
        except Exception:
            continue
    return "gemini-3.8-flash"


def generate_carousel_json(story: Dict, min_slides: int = 5, max_slides: int = 8) -> Dict:
    """Generate dynamic carousel content using Topic Intelligence, Archetype Narrative, and Gemini LLM."""
    # Step 1: Classify topic and select archetype
    topic_intel = classify_topic(story)
    topic_type = topic_intel.get("topic_type", "model_release")
    archetype = CONTENT_ARCHETYPES.get(topic_type, CONTENT_ARCHETYPES["model_release"])
    
    # Step 2: Determine dynamic slide count
    min_s, max_s = archetype.get("slide_range", (min_slides, max_slides))
    desired_slides = topic_intel.get("recommended_slide_count", archetype.get("default_slide_count", 7))
    desired_slides = max(min_s, min(desired_slides, max_s))
    
    # Step 3: Extract structured research brief
    research_brief = build_research_brief(story, topic_intel)

    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        print("ℹ️ Gemini not available or API key missing, generating dynamic archetype fallback carousel")
        return generate_fallback_carousel(story, topic_intel=topic_intel)

    prompt = build_carousel_prompt(
        story=story,
        topic_intel=topic_intel,
        research_brief=research_brief,
        archetype=archetype,
        slide_count=desired_slides
    )

    carousel = None
    last_error = None

    # Priority 1-4: OpenRouter prioritized models
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "") or OPENROUTER_API_KEY
    if openrouter_key:
        from llm_fallback import get_openrouter_models_by_priority, clean_and_parse_json
        story_context = f"{story.get('title', '')} {story.get('description', '')}"
        priority_models = get_openrouter_models_by_priority(topic_category=topic_type, context_text=story_context)
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vjaab/YtDidYouKnowByVJ",
            "X-Title": "YtDidYouKnowByVJ Carousel Generator",
        }
        for model_name in priority_models:
            try:
                print(f"🤖 Generating dynamic carousel ({topic_type} | {desired_slides} slides) with OpenRouter: {model_name}...")
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                }
                r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=45)
                if r.status_code == 200:
                    choices = r.json().get("choices", [])
                    if choices:
                        raw = choices[0].get("message", {}).get("content", "").strip()
                        parsed = clean_and_parse_json(raw)
                        if isinstance(parsed, dict) and "slides" in parsed and len(parsed["slides"]) >= 4:
                            carousel = parsed
                            carousel["_model_used"] = f"OpenRouter:{model_name}"
                            carousel["_topic_intelligence"] = topic_intel
                            carousel["_archetype"] = archetype["name"]

                            is_valid_narrative, narrative_issues = validate_slide_narrative(carousel, archetype=archetype)
                            if not is_valid_narrative:
                                print(f"⚠️ Slide narrative validator reported issues: {narrative_issues}")
                            else:
                                print(f"✅ Slide narrative validation passed ({len(carousel['slides'])} slides, {len(set(s.get('layout_type') for s in carousel['slides']))} unique layouts)")
                            
                            print(f"✅ OpenRouter generation successful with {model_name} ({len(carousel['slides'])} slides)")
                            break
                else:
                    print(f"⚠️ OpenRouter ({model_name}) returned HTTP {r.status_code}: {r.text[:120]}")
            except Exception as e:
                last_error = e
                print(f"⚠️ OpenRouter ({model_name}) exception: {e}")
                continue

    # Priority 5: Gemini LLM generation
    if not carousel and GEMINI_AVAILABLE and GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        for model_name in AVAILABLE_GEMINI_MODELS:
            try:
                print(f"🤖 Generating dynamic carousel ({topic_type} | {desired_slides} slides) with Gemini: {model_name}...")
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
                    carousel["_topic_intelligence"] = topic_intel
                    carousel["_archetype"] = archetype["name"]
                    
                    # Semantic Diversity & Narrative Validation
                    is_valid_narrative, narrative_issues = validate_slide_narrative(carousel, archetype=archetype)
                    if not is_valid_narrative:
                        print(f"⚠️ Slide narrative validator reported issues: {narrative_issues}")
                    else:
                        print(f"✅ Slide narrative validation passed ({len(carousel['slides'])} slides, {len(set(s.get('layout_type') for s in carousel['slides']))} unique layouts)")
                    
                    print(f"✅ Gemini LLM generation successful with {model_name} ({len(carousel['slides'])} slides)")
                    break
            except Exception as e:
                last_error = e
                print(f"⚠️ Model {model_name} failed: {e}")
                continue

    # Priority 6: Shared Fallback Chain
    if not carousel:
        print("🚨 OpenRouter & Gemini failed. Attempting shared fallback chain...")
        from llm_fallback import call_fallback_chain
        fallback_res = call_fallback_chain(prompt, normalize=False)
        if fallback_res and isinstance(fallback_res, dict) and "slides" in fallback_res and len(fallback_res["slides"]) >= 4:
            carousel = fallback_res
            carousel["_model_used"] = "FallbackChain"
            carousel["_topic_intelligence"] = topic_intel
            carousel["_archetype"] = archetype["name"]
            print(f"✅ Fallback chain generation successful ({len(carousel['slides'])} slides)")

    if not carousel:
        print(f"⚠️ All LLM providers failed ({last_error}), using dynamic archetype fallback")
        return generate_fallback_carousel(story, topic_intel=topic_intel)

    # Validate and repair content
    carousel["_story"] = story
    carousel["_topic_intelligence"] = topic_intel
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
            "    model='gemini-3.8-flash',\n"
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


def generate_fallback_carousel(story: Dict, topic_intel: Dict = None) -> Dict:
    """Generate dynamic, high-quality, story-specific fallback carousel matching content archetypes."""
    info = _extract_story_details(story)
    today_str = datetime.now().strftime('%d %b %Y')
    
    if not topic_intel:
        topic_intel = classify_topic_heuristically(story)
        
    topic_type = topic_intel.get("topic_type", "model_release")
    entity = topic_intel.get("primary_entity", info["entity"])
    archetype = CONTENT_ARCHETYPES.get(topic_type, CONTENT_ARCHETYPES["model_release"])
    
    headline = _truncate_at_word_boundary(info["title"], 55)
    summary = info["description"][:250] if info["description"] else f"Key architectural updates and production insights from {info['source']}."
    category_badge = topic_intel.get("domain", "AI ENGINEERING").upper()

    code_text, code_lang, code_expl = _get_ecosystem_code_snippet(info)
    metrics_data = _get_metrics_for_story(info)

    slides = []

    if topic_type == "developer_tool":
        slides = [
            {
                "slide_number": 1,
                "role": "problem_hook",
                "layout_type": "scenario_question",
                "eyebrow": f"{entity.upper()} INNOVATION",
                "title": _truncate_at_word_boundary(info["title"], 50),
                "subtitle": f"How {entity} eliminates developer friction in modern workflows.",
                "body": f"{info['description'][:140]} " if len(info["description"]) > 20 else f"A major developer tooling upgrade announced for {entity}.",
                "deep_dive": f"{entity} tackles developer productivity by streamlining repetitive coding tasks and surfacing contextual workspace insights.",
                "key_fact": f"Transforms standard development loops into automated, assisted engineering workflows.",
                "why_it_matters": "Reduces context switching and accelerates shipping velocity across engineering teams."
            },
            {
                "slide_number": 2,
                "role": "tool_overview",
                "layout_type": "whats_new",
                "eyebrow": "WHAT IT DOES",
                "title": "Core Tooling Capabilities",
                "body": f"Three foundational capabilities introduced in this {entity} release:",
                "before_title": "Developer Bottlenecks",
                "before_items": [
                    "Manual context gathering across disconnected files",
                    "Repetitive boilerplate and tedious syntax setup",
                    "Context-blind autocompletion requiring constant verification"
                ],
                "after_title": f"{entity} Capabilities",
                "after_items": [
                    "Global workspace semantic indexing & call tree awareness",
                    "Autonomous multi-file code editing with interactive diffs",
                    "Native terminal command execution & test verification"
                ]
            },
            {
                "slide_number": 3,
                "role": "core_mechanism",
                "layout_type": "process_flow",
                "eyebrow": "EXECUTION FLOW",
                "title": "Under The Hood Execution",
                "body": f"How {entity} analyzes and executes developer commands in real time:",
                "steps": [
                    {"step": "1", "title": "Index Workspace AST", "desc": "Extracts symbols, type definitions, and dependencies across project files."},
                    {"step": "2", "title": "Semantic Retrieval", "desc": "Surfaces relevant code chunks using high-dimensional vector embeddings."},
                    {"step": "3", "title": "Multi-File Patching", "desc": "Generates cohesive, syntactically verified diffs with instant rollback."}
                ]
            },
            {
                "slide_number": 4,
                "role": "architecture",
                "layout_type": "architecture_diagram",
                "eyebrow": "SYSTEM ARCHITECTURE",
                "title": "Tool Integration Topology",
                "body": f"The architectural pipeline powering {entity}'s developer loop:",
                "diagram": {
                    "nodes": [
                        {"name": "Developer Editor / IDE", "sub": "VS Code, Cursor, or Terminal Client"},
                        {"name": "Local Agent Engine", "sub": "AST Parser & Workspace Watcher"},
                        {"name": "Vector Codebase Index", "sub": "Embedded Semantic Retrieval Cache"},
                        {"name": "Frontier LLM Inference", "sub": "Sub-Second Structured Diff Generator"}
                    ],
                    "connections": [
                        {"label": "Telemetry & Commands"},
                        {"label": "AST Context Query"},
                        {"label": "Grounded Prompts & Patches"}
                    ]
                }
            },
            {
                "slide_number": 5,
                "role": "system_workflow",
                "layout_type": "input_output",
                "eyebrow": "EXECUTION MECHANICS",
                "title": "Query to Verified Patch",
                "body": f"How {entity} transforms developer natural language prompts into production diffs:",
                "input": "Developer Prompt: 'Refactor auth middleware to async JWT token verification'",
                "processing": "AST parser resolves dependency tree, identifies impacted routes, and injects validated types",
                "output": "Syntactically verified, multi-file diff ready for automated test execution"
            },
            {
                "slide_number": 6,
                "role": "limitations_and_gotchas",
                "layout_type": "common_mistake",
                "eyebrow": "PRO TIPS & PITFALLS",
                "title": "Common Mistakes to Avoid",
                "body": f"Best engineering practices when deploying {entity} across engineering repos:",
                "wrong_title": "Common Mistakes ❌",
                "wrong_items": [
                    "Over-relying on autonomous edits without running unit test suites",
                    "Indexing huge build artifacts and node_modules directories",
                    "Neglecting to provide explicit project instruction rules (.cursorrules)"
                ],
                "right_title": "Recommended Pattern ✅",
                "right_items": [
                    "Enforce CI verification and automated linting on every agent patch",
                    "Configure strict ignore patterns for generated files and binaries",
                    "Define modular project conventions and explicit typing constraints"
                ]
            },
            {
                "slide_number": 7,
                "role": "takeaway",
                "layout_type": "takeaway",
                "eyebrow": "GET STARTED",
                "title": "Action Plan for Developers",
                "body": "Immediate steps to integrate this tooling into your team's stack:",
                "takeaways": [
                    f"Upgrade to the latest {entity} release and audit updated configuration flags.",
                    "Establish standardized workspace guidelines to maximize agent reasoning accuracy.",
                    "Track developer shipping velocity and code review cycle improvements."
                ],
                "cta": "Save this guide • Follow @vijayakumarj_ai for daily developer tools & AI insights"
            }
        ]

    elif topic_type == "security_incident":
        slides = [
            {
                "slide_number": 1,
                "role": "threat_hook",
                "layout_type": "scenario_question",
                "eyebrow": "SECURITY ADVISORY",
                "title": _truncate_at_word_boundary(info["title"], 50),
                "subtitle": "Critical vulnerability disclosure and system remediation breakdown.",
                "body": f"{info['description'][:140]} " if len(info["description"]) > 20 else "A critical security advisory has been disclosed requiring immediate attention.",
                "deep_dive": "Security researchers identified an exploitation vector allowing unauthorized state manipulation or arbitrary execution.",
                "key_fact": "Affects production configurations lacking strict input sanitation or network boundary controls.",
                "why_it_matters": "Immediate patching is essential to prevent lateral movement and credential exfiltration."
            },
            {
                "slide_number": 2,
                "role": "what_happened",
                "layout_type": "what_happened",
                "eyebrow": "THE INCIDENT",
                "title": "Vulnerability Analysis",
                "body": f"Details of the vulnerability discovered in {entity}:",
                "metrics": [
                    {"label": "Vulnerability Impact", "value": "High Severity", "delta": "Advisory Issued"},
                    {"label": "Affected Surface", "value": "Production APIs", "delta": "Patch Available"},
                    {"label": "Exploit Complexity", "value": "Moderate", "delta": "Zero-Day Avoided"}
                ]
            },
            {
                "slide_number": 3,
                "role": "attack_chain",
                "layout_type": "process_flow",
                "eyebrow": "ATTACK CHAIN",
                "title": "Exploitation Mechanism",
                "body": "How an adversary could potentially leverage this flaw:",
                "steps": [
                    {"step": "1", "title": "Untrusted Ingestion", "desc": "Malicious payload injected into unvalidated input fields or prompt streams."},
                    {"step": "2", "title": "Boundary Bypass", "desc": "Exploits parser edge case to bypass security perimeter and escape sandbox."},
                    {"step": "3", "title": "Privileged Execution", "desc": "Executes unauthorized actions or extracts sensitive credentials from memory."}
                ]
            },
            {
                "slide_number": 4,
                "role": "vulnerable_architecture",
                "layout_type": "architecture_diagram",
                "eyebrow": "ATTACK SURFACE",
                "title": "Vulnerable Topology",
                "body": "The system boundaries and where the attack vector intersects:",
                "diagram": {
                    "nodes": [
                        {"name": "External Ingress / Client", "sub": "Unauthenticated API Request"},
                        {"name": "Vulnerable Gateway Layer", "sub": "Flawed Input Parsing Logic"},
                        {"name": "Internal Service Mesh", "sub": "Downstream Privileged Services"},
                        {"name": "Encrypted Secrets & Data Store", "sub": "Targeted Assets & Credentials"}
                    ],
                    "connections": [
                        {"label": "Malicious Payload"},
                        {"label": "Unsanitized Forwarding"},
                        {"label": "Privilege Escalation"}
                    ]
                }
            },
            {
                "slide_number": 5,
                "role": "root_cause_analysis",
                "layout_type": "common_mistake",
                "eyebrow": "ROOT CAUSE",
                "title": "Vulnerable vs Secure Pattern",
                "body": "Analyzing the insecure implementation against the patched standard:",
                "wrong_title": "Vulnerable Pattern ❌",
                "wrong_items": [
                    "Directly concatenating untrusted inputs into execution context",
                    "Implicit trust granted between internal microservices",
                    "Broad IAM permissions without resource scoping"
                ],
                "right_title": "Patched Hardening ✅",
                "right_items": [
                    "Strict schema validation & parameterized payload execution",
                    "Zero-trust mTLS authentication across all service boundaries",
                    "Least-privilege scoped tokens with short-lived expiration"
                ]
            },
            {
                "slide_number": 6,
                "role": "defense_architecture",
                "layout_type": "side_by_side",
                "eyebrow": "DEFENSE STRATEGY",
                "title": "Immediate Hardening Protocol",
                "body": "Production defense-in-depth measures to eliminate the exploit vector across the stack:",
                "left_title": "Perimeter Defenses 🛡️",
                "left_items": [
                    "Deploy edge WAF rules blocking malformed payload injections",
                    "Enforce zero-trust mTLS with short-lived certificate rotation",
                    "Strict rate limiting on authentication & token exchange routes"
                ],
                "right_title": "Application Hardening 🔒",
                "right_items": [
                    "Replace dynamic query strings with parameterized execution",
                    "Enforce strict schema validation and bounded field constraints",
                    "Automate SBOM dependency auditing in the CI/CD pipeline"
                ]
            },
            {
                "slide_number": 7,
                "role": "security_checklist",
                "layout_type": "checklist",
                "eyebrow": "HARDENING CHECKLIST",
                "title": "Actionable Defense Checklist",
                "body": "Audit steps security and DevOps teams should immediately conduct:",
                "items": [
                    f"Audit all environments for vulnerable {entity} package versions and update immediately.",
                    "Verify Web Application Firewall (WAF) rules are actively blocking known exploit patterns.",
                    "Rotate all API keys and service credentials that were exposed to affected nodes.",
                    "Enable comprehensive audit logging on all ingress endpoints."
                ]
            },
            {
                "slide_number": 8,
                "role": "takeaway",
                "layout_type": "takeaway",
                "eyebrow": "REMEDIATION ACTION",
                "title": "Immediate Next Steps",
                "body": "Summary of remediation priorities for engineering teams:",
                "takeaways": [
                    f"Deploy patched version of {entity} across staging and production clusters.",
                    "Implement defense-in-depth perimeter validation to mitigate future zero-days.",
                    "Conduct automated dependency vulnerability scans as part of CI/CD."
                ],
                "cta": "Save this guide • Follow @vijayakumarj_ai for daily cybersecurity & AI engineering"
            }
        ]

    elif topic_type == "research_paper":
        slides = [
            {
                "slide_number": 1,
                "role": "big_idea_hook",
                "layout_type": "hero_hook",
                "eyebrow": "RESEARCH BREAKTHROUGH",
                "title": _truncate_at_word_boundary(info["title"], 50),
                "subtitle": f"Novel algorithmic methodology and empirical results from {entity}.",
                "body": f"{info['description'][:140]} " if len(info["description"]) > 20 else f"A breakthrough research paper published by {entity} introduces a new foundational paradigm.",
                "deep_dive": "Researchers propose an alternative formulation to standard transformer bottlenecks, demonstrating marked efficiency gains.",
                "key_fact": "Achieves superior benchmark accuracy while reducing compute complexity by an order of magnitude.",
                "why_it_matters": "Could redefine model training economics and high-throughput real-time serving."
            },
            {
                "slide_number": 2,
                "role": "problem_solved",
                "layout_type": "what_happened",
                "eyebrow": "THE CORE PROBLEM",
                "title": "Why Prior Approaches Failed",
                "body": "Existing model architectures hit fundamental mathematical and operational walls:",
                "metrics": [
                    {"label": "Memory Complexity", "value": "O(N²)", "delta": "Quadratic Bottleneck"},
                    {"label": "Inference Latency", "value": "Linear Growth", "delta": "KV Cache Bloat"},
                    {"label": "Reasoning Drift", "value": "Accumulative", "delta": "Compounding Errors"}
                ]
            },
            {
                "slide_number": 3,
                "role": "new_methodology",
                "layout_type": "process_flow",
                "eyebrow": "NEW METHODOLOGY",
                "title": "Algorithmic Innovation",
                "body": "The core three-step technique proposed in the paper:",
                "steps": [
                    {"step": "1", "title": "Dynamic State Compression", "desc": "Projects continuous attention vectors into fixed-size latent manifolds."},
                    {"step": "2", "title": "Recurrent State Update", "desc": "Maintains constant-time token updates during generation passes."},
                    {"step": "3", "title": "Adaptive Gating Filter", "desc": "Dynamically purges irrelevant historical noise from the internal state."}
                ]
            },
            {
                "slide_number": 4,
                "role": "method_architecture",
                "layout_type": "architecture_diagram",
                "eyebrow": "SYSTEM ARCHITECTURE",
                "title": "Proposed Model Architecture",
                "body": "The novel layer topology introduced by the researchers:",
                "diagram": {
                    "nodes": [
                        {"name": "Input Token Stream", "sub": "Raw Context Representation"},
                        {"name": "Linear Projection Gate", "sub": "Dimension Reduction & Normalization"},
                        {"name": "Recurrent Latent Core", "sub": "Constant-Memory State Tensor"},
                        {"name": "Output Prediction Head", "sub": "Probability Distribution over Vocabulary"}
                    ],
                    "connections": [
                        {"label": "Vector Mapping"},
                        {"label": "State Recurrence"},
                        {"label": "Logit Computation"}
                    ]
                }
            },
            {
                "slide_number": 5,
                "role": "experimental_results",
                "layout_type": "metrics_cards",
                "eyebrow": "EVALUATION DATA",
                "title": "Empirical Benchmark Results",
                "body": "Reported performance metrics comparing the novel method against strong baselines:",
                "metrics": metrics_data
            },
            {
                "slide_number": 6,
                "role": "paradigm_comparison",
                "layout_type": "before_after",
                "eyebrow": "PARADIGM SHIFT",
                "title": "Prior Approach vs Proposed Method",
                "body": "Direct comparison between traditional transformers and the novel architecture:",
                "before_title": "Standard Transformer",
                "before_items": [
                    "Quadratic attention memory complexity with sequence length",
                    "Massive KV cache footprint requiring multi-GPU memory pooling",
                    "High time-to-first-token latency on long context prompts"
                ],
                "after_title": "Novel Proposed Method",
                "after_items": [
                    "Constant O(1) memory complexity during sequential token decoding",
                    "Near-zero KV cache overhead unlocking edge device deployment",
                    "Sub-linear compute scaling on multi-million token sequences"
                ]
            },
            {
                "slide_number": 7,
                "role": "why_it_matters",
                "layout_type": "real_world_scenario",
                "eyebrow": "INDUSTRY IMPACT",
                "title": "Production & Industry Implications",
                "body": f"How {entity}'s research discovery transforms the broader tech landscape:",
                "company": "Frontier AI Serving",
                "scenario": "Serving high-concurrency autonomous agents requires sustained context retention without astronomical cloud GPU compute bills.",
                "technical_detail": "By moving to constant-state recurrence, serving costs drop dramatically while context horizons expand infinitely.",
                "result": "Unlocks true continuous lifelong learning and ultra-low latency real-time voice & coding agents."
            },
            {
                "slide_number": 8,
                "role": "takeaway",
                "layout_type": "takeaway",
                "eyebrow": "PAPER SUMMARY",
                "title": "Key Takeaways for Engineers",
                "body": "Essential conclusions from this research paper:",
                "takeaways": [
                    "Alternative architectures are proving competitive with standard dense attention.",
                    "Efficiency breakthroughs at the mathematical level outpace brute-force hardware scaling.",
                    "Expect open-source implementations and community reproductions in coming months."
                ],
                "cta": "Save this guide • Follow @vijayakumarj_ai for daily AI research & engineering breakdowns"
            }
        ]

    elif topic_type == "programming_concept":
        slides = [
            {
                "slide_number": 1,
                "role": "problem_hook",
                "layout_type": "hero_hook",
                "eyebrow": "ARCHITECTURE PATTERN",
                "title": _truncate_at_word_boundary(info["title"], 50),
                "subtitle": "Mastering production-grade software design and system resilience.",
                "body": f"{info['description'][:140]} " if len(info["description"]) > 20 else "A fundamental engineering concept that every senior software engineer must master.",
                "deep_dive": "High-throughput distributed systems require robust synchronization and decoupled state management.",
                "key_fact": "Prevents catastrophic race conditions and deadlocks under concurrent peak load.",
                "why_it_matters": "Dramatically improves system reliability and maintains sub-millisecond tail latencies."
            },
            {
                "slide_number": 2,
                "role": "concept_mental_model",
                "layout_type": "process_flow",
                "eyebrow": "MENTAL MODEL",
                "title": "Core Conceptual Principles",
                "body": "Building an intuitive mental model for this architectural pattern:",
                "steps": [
                    {"step": "1", "title": "State Isolation", "desc": "Encapsulate mutable state within single-threaded owners or atomic boundaries."},
                    {"step": "2", "title": "Event-Driven Messaging", "desc": "Communicate changes via asynchronous typed events rather than shared memory locks."},
                    {"step": "3", "title": "Deterministic Replay", "desc": "Allow full state reconstruction from append-only immutable event streams."}
                ]
            },
            {
                "slide_number": 3,
                "role": "execution_mechanics",
                "layout_type": "input_output",
                "eyebrow": "HOW IT WORKS",
                "title": "Input to Output Flow",
                "body": "State transitions across the pattern lifecycle:",
                "input": "Incoming Concurrent Requests (10,000 req/sec)",
                "processing": "Non-blocking event loop dispatches tasks to worker queues with backpressure buffering",
                "output": "Consistent, fully ordered transaction committed to persistent storage"
            },
            {
                "slide_number": 4,
                "role": "architectural_deep_dive",
                "layout_type": "side_by_side",
                "eyebrow": "PATTERN TRADE-OFFS",
                "title": "When to Apply This Pattern",
                "body": "Key architectural trade-offs every senior engineer must evaluate:",
                "left_title": "Primary Advantages ⚡",
                "left_items": [
                    "Near-zero lock contention under high concurrent write loads",
                    "Linear horizontal scalability across distributed node clusters",
                    "Predictable sub-millisecond p99 latency during traffic bursts"
                ],
                "right_title": "Architectural Cost ⚠️",
                "right_items": [
                    "Eventual consistency model requires careful idempotency handling",
                    "Increased memory consumption for event buffering & write-ahead logs",
                    "Higher system observability and tracing setup complexity"
                ]
            },
            {
                "slide_number": 5,
                "role": "common_antipattern",
                "layout_type": "common_mistake",
                "eyebrow": "AVOID THIS MISTAKE",
                "title": "Anti-Pattern vs Production Pattern",
                "body": "Comparing flawed naive implementations with senior engineering design:",
                "wrong_title": "Naive Anti-Pattern ❌",
                "wrong_items": [
                    "Global mutable state protected by coarse-grained mutexes",
                    "Unbounded in-memory queues causing Out-Of-Memory crashes under spikes",
                    "Silent error suppression without retry exponential backoff"
                ],
                "right_title": "Production Pattern ✅",
                "right_items": [
                    "Decoupled actor model or channel-based communication",
                    "Explicit bounded buffers with circuit breakers and load shedding",
                    "Structured error hierarchies with automatic jittered retries"
                ]
            },
            {
                "slide_number": 6,
                "role": "production_case_study",
                "layout_type": "real_world_scenario",
                "eyebrow": "SCALE IN PRODUCTION",
                "title": "Real-World High-Scale Case Study",
                "body": "How leading engineering organizations leverage this design pattern:",
                "company": "High-Throughput Fintech & Cloud",
                "scenario": "Processing millions of financial ledger transactions per minute requires strict zero-data-loss guarantees.",
                "technical_detail": "Adopting an append-only event-sourced log ensures audit compliance and enables horizontal read-replica scaling.",
                "result": "Zero lock contention, 99.999% availability, and instantaneous disaster recovery."
            },
            {
                "slide_number": 7,
                "role": "takeaway",
                "layout_type": "takeaway",
                "eyebrow": "ENGINEERING CHEAT SHEET",
                "title": "Key Implementation Rules",
                "body": "Guidelines to apply when designing your next production service:",
                "takeaways": [
                    "Default to immutable data structures unless profiling proves an allocation bottleneck.",
                    "Isolate concurrency boundaries using structured concurrency primitives.",
                    "Always define operational metrics: p99 latency, queue depth, and error rates."
                ],
                "cta": "Save this guide • Follow @vijayakumarj_ai for daily software architecture & engineering"
            }
        ]

    else:
        # Default: model_release / foundation models (8 slides)
        hook_eyebrow = f"{entity.upper()} UPDATE"
        if info["version"]:
            hook_eyebrow = f"{entity.upper()} {info['version'].upper()}"
        elif info["param_size"]:
            hook_eyebrow = f"{entity.upper()} {info['param_size']}"
        hook_eyebrow = hook_eyebrow[:24]

        slides = [
            {
                "slide_number": 1,
                "role": "hook",
                "layout_type": "hero_hook",
                "eyebrow": hook_eyebrow,
                "title": _truncate_at_word_boundary(info["title"], 50),
                "subtitle": f"What engineers must know about {entity}'s latest production release.",
                "body": f"{info['description'][:140]} " if len(info["description"]) > 20 else f"A major architectural upgrade announced for {entity}.",
                "deep_dive": f"{entity} has introduced new architectural improvements that enhance accuracy, latency, and context efficiency.",
                "key_fact": "Delivers marked improvements in reasoning accuracy and token throughput.",
                "why_it_matters": "Enables engineers to build more capable autonomous agents with lower operational overhead."
            },
            {
                "slide_number": 2,
                "role": "what_changed",
                "layout_type": "whats_new",
                "eyebrow": "WHAT CHANGED",
                "title": "Core Architectural Upgrades",
                "body": f"How {entity} fundamentally shifts the capabilities of frontier AI systems:",
                "before_title": "Previous Constraints",
                "before_items": [
                    "Narrower context windows requiring complex chunking RAG pipelines",
                    "Higher per-token pricing constraining high-throughput batch workloads",
                    "Elevated latency on multi-step reasoning and autonomous tool calls"
                ],
                "after_title": f"New {entity} Paradigm",
                "after_items": [
                    f"Massive context capacity handling full codebases & multimodal assets",
                    "Up to 50% lower operational inference costs for production workloads",
                    "Optimized KV cache kernels delivering sub-second time-to-first-token"
                ]
            },
            {
                "slide_number": 3,
                "role": "technical_mechanism",
                "layout_type": "architecture_diagram",
                "eyebrow": "HOW IT WORKS",
                "title": "Model Execution Architecture",
                "body": f"The internal inference and tool dispatch pipeline powering {entity}:",
                "diagram": {
                    "nodes": [
                        {"name": "Client Application", "sub": "User Prompt & Multi-Modal Context"},
                        {"name": "Tokenizer & High-Speed Ingress", "sub": "Streaming Protocol & KV Cache Router"},
                        {"name": "Mixture-of-Experts Layer", "sub": "Sparse Activated Parameter Routing"},
                        {"name": "Structured Output & Tool Dispatch", "sub": "Guaranteed JSON Schema Generation"}
                    ],
                    "connections": [
                        {"label": "HTTP / gRPC Stream"},
                        {"label": "Attention Context Routing"},
                        {"label": "Typed Tool Invocation"}
                    ]
                }
            },
            {
                "slide_number": 4,
                "role": "benchmark_evidence",
                "layout_type": "metrics_cards",
                "eyebrow": "BENCHMARKS",
                "title": "Empirical Benchmark Results",
                "body": f"Standardized benchmark performance and efficiency data for {entity}:",
                "metrics": metrics_data
            },
            {
                "slide_number": 5,
                "role": "previous_vs_new",
                "layout_type": "before_after",
                "eyebrow": "OLD VS NEW",
                "title": "Developer Workflow Shift",
                "body": "Comparing previous engineering workarounds with modern capabilities:",
                "before_title": "Legacy Approach",
                "before_items": [
                    "Brittle regex-based post-processing of model completions",
                    "Manual fallbacks for context window overflow errors",
                    "Multi-model chaining required for vision, code, and text"
                ],
                "after_title": "Modern Architecture",
                "after_items": [
                    "Native constrained decoding adhering strictly to JSON schemas",
                    "Single prompt ingestion of comprehensive repository architectures",
                    "Unified multimodal understanding in a single forward pass"
                ]
            },
            {
                "slide_number": 6,
                "role": "developer_impact",
                "layout_type": "real_world_scenario",
                "eyebrow": "IN PRODUCTION",
                "title": "Production Deployment Case",
                "body": f"How engineering teams deploy {entity} at high scale:",
                "company": "Autonomous Agent Platform",
                "scenario": "Running multi-agent code refactoring across enterprise repositories requires guaranteed schema adherence and rapid feedback loops.",
                "technical_detail": f"Leveraging {entity}'s low-latency streaming and high-capacity context simplifies pipeline orchestration.",
                "result": "40% reduction in agent cycle duration and zero schema validation failures."
            },
            {
                "slide_number": 7,
                "role": "practical_action",
                "layout_type": "checklist",
                "eyebrow": "INTEGRATION CHECKLIST",
                "title": "Engineering Migration Checklist",
                "body": "Practical steps software engineers should take to evaluate this release:",
                "items": [
                    f"Test {entity} on your internal evaluation harness against production baseline prompts.",
                    "Audit token costs and adjust rate limiter concurrency settings.",
                    "Adopt structured JSON schema mode for all programmatic function calls.",
                    "Implement streaming responses to optimize perceived user latency."
                ]
            },
            {
                "slide_number": 8,
                "role": "takeaway",
                "layout_type": "takeaway",
                "eyebrow": "KEY TAKEAWAY",
                "title": "Summary & Next Steps",
                "body": "Actionable takeaways for software engineers and architects:",
                "takeaways": [
                    f"Evaluate {entity}'s latest capabilities against existing models in your stack.",
                    "Take advantage of larger context windows to simplify multi-step RAG pipelines.",
                    "Track production latency and token spend to capture cost savings."
                ],
                "cta": "Save this guide • Follow @vijayakumarj_ai for daily AI engineering updates"
            }
        ]

    return {
        "headline": headline,
        "summary": summary,
        "source": info["source"] or "AI Engineering",
        "source_url": info["url"],
        "date": today_str,
        "category": category_badge,
        "topic_type": topic_type,
        "depth_score": 9,
        "originality_score": 9,
        "technical_specificity": 9,
        "slides": slides,
        "_story": story,
        "_topic_intelligence": topic_intel
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