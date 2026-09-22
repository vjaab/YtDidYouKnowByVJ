#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visual_strategy.py — Visual Strategy Engine for carousel generation.
Classifies topics, selects themes, chooses layouts, and produces
a structured visual strategy JSON that drives the renderer.
"""

import random
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

from carousel_history import (
    suggest_avoided_themes,
    suggest_avoided_layouts,
    is_layout_pattern_too_similar,
    get_recent_layout_patterns,
)

# Load carousel topic tracker for topic-level avoidance
def load_carousel_topic_tracker() -> Dict:
    """Load the Instagram carousel topic tracker."""
    import os
    import json
    from pathlib import Path
    
    TRACKER_FILE = Path(__file__).parent / "instagram_carousel_log.json"
    if not os.path.exists(TRACKER_FILE):
        return {"used_titles": [], "used_keywords": [], "history": []}
    try:
        with open(TRACKER_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"used_titles": [], "used_keywords": [], "history": []}


def get_recent_carousel_topics(n: int = 10) -> List[str]:
    """Get the last N carousel topic titles."""
    tracker = load_carousel_topic_tracker()
    return [entry.get("title", "") for entry in tracker.get("history", [])[-n:]]


def get_recent_carousel_keywords(n: int = 10) -> Set[str]:
    """Get keywords from recent carousel topics."""
    tracker = load_carousel_topic_tracker()
    keywords = set()
    for entry in tracker.get("history", [])[-n:]:
        keywords.update([k.lower() for k in entry.get("keywords", [])])
    return keywords

# ─── Layout Type Registry ──────────────────────────────────────────────────────

LAYOUT_TYPES = {
    # Hook / Opening layouts
    "hero_hook": {
        "purpose": "hook",
        "description": "Full-bleed gradient with large bold typography",
        "supports": ["title", "subtitle", "emoji_icon"],
    },
    "big_number": {
        "purpose": "hook",
        "description": "Giant statistic or number with context text",
        "supports": ["number", "label", "context"],
    },
    "scenario_question": {
        "purpose": "hook",
        "description": "Scenario-based hook: Imagine you're building X...",
        "supports": ["scenario", "question", "context_icon"],
    },
    # Explanation layouts
    "bullet_list": {
        "purpose": "explanation",
        "description": "Clean bullet list with accent dots",
        "supports": ["title", "bullets"],
    },
    "side_by_side": {
        "purpose": "comparison",
        "description": "Two-column comparison (A vs B)",
        "supports": ["left_title", "left_items", "right_title", "right_items"],
    },
    "before_after": {
        "purpose": "comparison",
        "description": "Before/After split with red/green coding",
        "supports": ["before_title", "before_items", "after_title", "after_items"],
    },
    "common_mistake": {
        "purpose": "comparison",
        "description": "❌ Wrong vs ✅ Right approach",
        "supports": ["wrong_title", "wrong_items", "right_title", "right_items"],
    },
    "myth_vs_fact": {
        "purpose": "comparison",
        "description": "Myth-busting format",
        "supports": ["myth", "fact", "explanation"],
    },
    # Architecture / Flow layouts
    "architecture_diagram": {
        "purpose": "technical",
        "description": "Box-and-arrow architecture diagram",
        "supports": ["nodes", "connections", "labels"],
    },
    "process_flow": {
        "purpose": "technical",
        "description": "Numbered step-by-step vertical flow",
        "supports": ["steps"],
    },
    "input_output": {
        "purpose": "technical",
        "description": "Input → Processing → Output flow",
        "supports": ["input", "processing", "output"],
    },
    "layered_stack": {
        "purpose": "technical",
        "description": "Layered architecture stack (top to bottom)",
        "supports": ["layers"],
    },
    "request_response": {
        "purpose": "technical",
        "description": "Request/Response flow between components",
        "supports": ["requester", "steps", "responder"],
    },
    "timeline": {
        "purpose": "technical",
        "description": "Horizontal or vertical timeline",
        "supports": ["events"],
    },
    # Code layouts
    "code_block": {
        "purpose": "code",
        "description": "Syntax-highlighted code with explanation",
        "supports": ["language", "code", "explanation", "highlight_lines"],
    },
    "code_output": {
        "purpose": "code",
        "description": "Code snippet + its output side by side",
        "supports": ["language", "code", "output", "explanation"],
    },
    # Real-world layouts
    "real_world_scenario": {
        "purpose": "example",
        "description": "Named company/product scenario with technical context",
        "supports": ["company", "scenario", "technical_detail", "diagram"],
    },
    "analogy": {
        "purpose": "example",
        "description": "Real-world analogy: X is like Y",
        "supports": ["real_world", "technical", "mapping"],
    },
    # Data / Metrics layouts
    "metrics_cards": {
        "purpose": "data",
        "description": "2-4 metric cards with numbers and deltas",
        "supports": ["metrics"],
    },
    "three_column": {
        "purpose": "data",
        "description": "Three-column comparison cards",
        "supports": ["columns"],
    },
    "checklist": {
        "purpose": "data",
        "description": "Checkmark list of key points",
        "supports": ["items", "title"],
    },
    # Quiz layouts
    "quiz_predict_output": {
        "purpose": "quiz",
        "description": "Code + What does this print?",
        "supports": ["code", "options", "answer_index"],
    },
    "quiz_choice": {
        "purpose": "quiz",
        "description": "Multiple choice question",
        "supports": ["question", "options", "answer_index", "explanation"],
    },
    # CTA / Closing layouts
    "takeaway": {
        "purpose": "cta",
        "description": "Key takeaways + save/follow CTA",
        "supports": ["takeaways", "cta_text"],
    },
    "decision_tree": {
        "purpose": "technical",
        "description": "Simple decision tree / flowchart",
        "supports": ["question", "yes_path", "no_path"],
    },
}

# ─── Visual Themes ──────────────────────────────────────────────────────────────

VISUAL_THEMES = {
    "TECH_DARK": {
        "name": "Tech Dark",
        "css_class": "theme--tech-dark",
        "bg": "#0B0E14",
        "card_bg": "#151921",
        "card_border": "#1E2633",
        "text_primary": "#F1F3F8",
        "text_secondary": "#8A93A8",
        "text_muted": "#5A6478",
        "accent": "#60A5FA",
        "accent_secondary": "#A78BFA",
        "accent_success": "#34D399",
        "accent_danger": "#FB7185",
        "accent_warning": "#FBBF24",
        "code_bg": "#0D1117",
        "gradient_start": "#60A5FA",
        "gradient_end": "#A78BFA",
        "domains": ["ai_ml", "programming", "devops", "general"],
    },
    "CLEAN_LIGHT": {
        "name": "Clean Light",
        "css_class": "theme--clean-light",
        "bg": "#FFFFFF",
        "card_bg": "#F8FAFC",
        "card_border": "#E2E8F0",
        "text_primary": "#0F172A",
        "text_secondary": "#475569",
        "text_muted": "#94A3B8",
        "accent": "#3B82F6",
        "accent_secondary": "#8B5CF6",
        "accent_success": "#10B981",
        "accent_danger": "#EF4444",
        "accent_warning": "#F59E0B",
        "code_bg": "#1E293B",
        "gradient_start": "#3B82F6",
        "gradient_end": "#8B5CF6",
        "domains": ["career", "general", "data"],
    },
    "TERMINAL": {
        "name": "Terminal",
        "css_class": "theme--terminal",
        "bg": "#0C0C0C",
        "card_bg": "#1A1A1A",
        "card_border": "#2D2D2D",
        "text_primary": "#33FF33",
        "text_secondary": "#00CC00",
        "text_muted": "#006600",
        "accent": "#33FF33",
        "accent_secondary": "#00FFFF",
        "accent_success": "#33FF33",
        "accent_danger": "#FF3333",
        "accent_warning": "#FFFF33",
        "code_bg": "#0C0C0C",
        "gradient_start": "#33FF33",
        "gradient_end": "#00FFFF",
        "domains": ["security", "programming", "devops"],
    },
    "BLUEPRINT": {
        "name": "Blueprint",
        "css_class": "theme--blueprint",
        "bg": "#0A1628",
        "card_bg": "#0F2140",
        "card_border": "#1A3A6B",
        "text_primary": "#E0F0FF",
        "text_secondary": "#7EB8E0",
        "text_muted": "#4A7BA8",
        "accent": "#00B4D8",
        "accent_secondary": "#0096C7",
        "accent_success": "#48CAE4",
        "accent_danger": "#FF6B6B",
        "accent_warning": "#FFD166",
        "code_bg": "#071120",
        "gradient_start": "#00B4D8",
        "gradient_end": "#0077B6",
        "domains": ["cloud", "architecture", "devops"],
    },
    "FUTURISTIC": {
        "name": "Futuristic",
        "css_class": "theme--futuristic",
        "bg": "#0D0221",
        "card_bg": "#1A0A3E",
        "card_border": "#2D1B69",
        "text_primary": "#F0E6FF",
        "text_secondary": "#B39DDB",
        "text_muted": "#7C6BA0",
        "accent": "#E040FB",
        "accent_secondary": "#7C4DFF",
        "accent_success": "#69F0AE",
        "accent_danger": "#FF5252",
        "accent_warning": "#FFD740",
        "code_bg": "#0A0118",
        "gradient_start": "#E040FB",
        "gradient_end": "#7C4DFF",
        "domains": ["ai_ml", "general"],
    },
    "MINIMAL": {
        "name": "Minimal",
        "css_class": "theme--minimal",
        "bg": "#FAFAFA",
        "card_bg": "#FFFFFF",
        "card_border": "#EEEEEE",
        "text_primary": "#212121",
        "text_secondary": "#616161",
        "text_muted": "#9E9E9E",
        "accent": "#1A1A1A",
        "accent_secondary": "#424242",
        "accent_success": "#2E7D32",
        "accent_danger": "#C62828",
        "accent_warning": "#EF6C00",
        "code_bg": "#263238",
        "gradient_start": "#424242",
        "gradient_end": "#212121",
        "domains": ["career", "general", "programming"],
    },
    "DATA_DASHBOARD": {
        "name": "Data Dashboard",
        "css_class": "theme--dashboard",
        "bg": "#111827",
        "card_bg": "#1F2937",
        "card_border": "#374151",
        "text_primary": "#F9FAFB",
        "text_secondary": "#D1D5DB",
        "text_muted": "#6B7280",
        "accent": "#06B6D4",
        "accent_secondary": "#8B5CF6",
        "accent_success": "#10B981",
        "accent_danger": "#EF4444",
        "accent_warning": "#F59E0B",
        "code_bg": "#0F172A",
        "gradient_start": "#06B6D4",
        "gradient_end": "#8B5CF6",
        "domains": ["data", "ai_ml", "cloud"],
    },
}

# ─── Topic Domain Classification ────────────────────────────────────────────────

DOMAIN_KEYWORDS = {
    "ai_ml": [
        "gpt", "llm", "language model", "transformer", "neural", "deep learning",
        "machine learning", "rag", "embedding", "vector", "diffusion", "generative",
        "openai", "anthropic", "claude", "gemini", "llama", "mistral", "huggingface",
        "fine-tune", "fine tune", "inference", "token", "prompt", "agent", "chatbot",
        "copilot", "ai", "artificial intelligence", "nlp", "computer vision",
        "reinforcement learning", "multimodal",
    ],
    "cloud": [
        "aws", "azure", "gcp", "s3", "lambda", "ec2", "ecs", "eks", "fargate",
        "cloudfront", "route53", "dynamodb", "rds", "sqs", "sns", "kinesis",
        "terraform", "pulumi", "cloudformation", "serverless", "kubernetes", "k8s",
        "docker", "container", "microservices", "load balancer", "cdn", "api gateway",
        "cloud", "iaas", "paas", "saas",
    ],
    "programming": [
        "python", "javascript", "typescript", "java", "rust", "go", "golang",
        "react", "next.js", "vue", "angular", "node.js", "django", "flask",
        "fastapi", "spring", "list", "tuple", "dictionary", "array", "hash",
        "algorithm", "data structure", "decorator", "closure", "async", "await",
        "generator", "iterator", "class", "inheritance", "polymorphism",
        "design pattern", "solid", "clean code", "refactor", "compilation",
    ],
    "security": [
        "security", "vulnerability", "exploit", "hack", "breach", "encryption",
        "authentication", "authorization", "oauth", "jwt", "ssl", "tls", "firewall",
        "iam", "rbac", "zero trust", "penetration", "malware", "ransomware",
        "credential", "secret", "api key", "xss", "sql injection", "csrf",
        "cybersecurity", "soc", "siem",
    ],
    "devops": [
        "ci/cd", "pipeline", "jenkins", "github actions", "gitlab", "deployment",
        "monitoring", "observability", "prometheus", "grafana", "datadog",
        "logging", "elk", "splunk", "scaling", "autoscaling", "infrastructure",
        "devops", "sre", "reliability", "incident", "on-call", "runbook",
        "ansible", "chef", "puppet", "helm",
    ],
    "data": [
        "database", "sql", "nosql", "postgresql", "mysql", "mongodb", "redis",
        "elasticsearch", "kafka", "rabbitmq", "streaming", "etl", "data pipeline",
        "data warehouse", "bigquery", "snowflake", "spark", "hadoop", "analytics",
        "dashboard", "visualization", "pandas", "numpy", "data engineering",
    ],
    "career": [
        "interview", "resume", "career", "salary", "job", "hiring", "roadmap",
        "skill", "learning", "bootcamp", "certification", "freelance", "remote",
        "mentor", "portfolio", "linkedin", "networking", "promotion", "senior",
        "staff", "principal", "manager", "lead",
    ],
}


def classify_domain(topic: str, description: str = "") -> str:
    """Classify a topic into a domain based on keyword matching."""
    text = f"{topic} {description}".lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[domain] = score

    if not scores:
        return "general"
    return max(scores, key=scores.get)


def classify_difficulty(description: str = "") -> str:
    """Estimate difficulty level from description."""
    text = description.lower()
    advanced_terms = [
        "architecture", "distributed", "consensus", "sharding", "replication",
        "eventually consistent", "cap theorem", "raft", "paxos", "vector clock",
        "transformer", "attention mechanism", "fine-tune", "quantization",
    ]
    beginner_terms = [
        "basics", "introduction", "what is", "beginner", "getting started",
        "first", "simple", "easy", "tutorial", "101",
    ]
    adv = sum(1 for t in advanced_terms if t in text)
    beg = sum(1 for t in beginner_terms if t in text)
    if adv >= 2:
        return "advanced"
    if beg >= 2:
        return "beginner"
    return "intermediate"


# ─── Storytelling Patterns ───────────────────────────────────────────────────────

# Each pattern is a list of (purpose, candidate_layout_types) tuples.
# The engine picks one layout from each candidate list.

STORYTELLING_PATTERNS = {
    "problem_solution": {
        "description": "Problem → Why → Solution → Example → Takeaway",
        "suitable_for": ["programming", "devops", "security", "cloud"],
        "slides": [
            ("hook", ["hero_hook", "scenario_question", "big_number"]),
            ("explanation", ["bullet_list", "process_flow"]),
            ("technical", ["architecture_diagram", "process_flow", "layered_stack", "code_block"]),
            ("example", ["real_world_scenario", "code_output", "analogy"]),
            ("comparison", ["common_mistake", "before_after"]),
            ("cta", ["takeaway"]),
        ],
    },
    "comparison_deep_dive": {
        "description": "Hook → Compare A vs B → Use cases → Code → Quiz",
        "suitable_for": ["programming", "cloud", "data"],
        "slides": [
            ("hook", ["hero_hook", "scenario_question"]),
            ("comparison", ["side_by_side", "three_column"]),
            ("example", ["real_world_scenario", "analogy"]),
            ("code", ["code_block", "code_output"]),
            ("quiz", ["quiz_predict_output", "quiz_choice"]),
            ("cta", ["takeaway"]),
        ],
    },
    "architecture_walkthrough": {
        "description": "Hook → Architecture → Component deep-dive → Flow → Real example → Takeaway",
        "suitable_for": ["cloud", "devops", "ai_ml", "data"],
        "slides": [
            ("hook", ["hero_hook", "big_number", "scenario_question"]),
            ("technical", ["architecture_diagram", "layered_stack"]),
            ("explanation", ["bullet_list", "process_flow"]),
            ("technical", ["request_response", "input_output"]),
            ("example", ["real_world_scenario"]),
            ("cta", ["takeaway"]),
        ],
    },
    "concept_explained": {
        "description": "Hook → What → How → Why → Mistake → Takeaway",
        "suitable_for": ["programming", "ai_ml", "general"],
        "slides": [
            ("hook", ["hero_hook", "scenario_question"]),
            ("explanation", ["bullet_list", "analogy"]),
            ("technical", ["process_flow", "architecture_diagram", "code_block"]),
            ("data", ["metrics_cards", "checklist"]),
            ("comparison", ["common_mistake", "myth_vs_fact"]),
            ("cta", ["takeaway"]),
        ],
    },
    "news_breakdown": {
        "description": "Breaking news → What → Why it matters → Impact → Try it → Takeaway",
        "suitable_for": ["ai_ml", "general", "cloud", "programming"],
        "slides": [
            ("hook", ["hero_hook", "big_number"]),
            ("explanation", ["bullet_list"]),
            ("technical", ["architecture_diagram", "process_flow", "input_output"]),
            ("comparison", ["before_after", "side_by_side"]),
            ("example", ["real_world_scenario", "code_block"]),
            ("cta", ["takeaway"]),
        ],
    },
    "tutorial_flow": {
        "description": "Hook → Prerequisites → Steps → Code → Output → Takeaway",
        "suitable_for": ["programming", "devops", "cloud"],
        "slides": [
            ("hook", ["hero_hook", "scenario_question"]),
            ("explanation", ["checklist", "bullet_list"]),
            ("code", ["code_block"]),
            ("code", ["code_output"]),
            ("comparison", ["common_mistake", "before_after"]),
            ("cta", ["takeaway"]),
        ],
    },
    "career_roadmap": {
        "description": "Hook → Current state → Path → Skills → Resources → Action",
        "suitable_for": ["career", "general"],
        "slides": [
            ("hook", ["hero_hook", "big_number"]),
            ("explanation", ["bullet_list", "timeline"]),
            ("data", ["checklist", "three_column"]),
            ("example", ["real_world_scenario"]),
            ("quiz", ["quiz_choice"]),
            ("cta", ["takeaway"]),
        ],
    },
    "security_incident": {
        "description": "Scenario → What went wrong → How it works → Protection → Checklist → Takeaway",
        "suitable_for": ["security"],
        "slides": [
            ("hook", ["scenario_question", "hero_hook"]),
            ("technical", ["process_flow", "architecture_diagram"]),
            ("comparison", ["common_mistake", "before_after"]),
            ("explanation", ["bullet_list", "checklist"]),
            ("example", ["real_world_scenario"]),
            ("cta", ["takeaway"]),
        ],
    },
}


def select_theme(
    domain: str,
    avoided_themes: Set[str] = None,
    run_context: str = "",
) -> Tuple[str, Dict]:
    """Select a visual theme based on domain, avoiding recently used themes.
    
    Args:
        domain: The content domain
        avoided_themes: Set of theme IDs to avoid
        run_context: Optional context for deterministic selection
    """
    if avoided_themes is None:
        avoided_themes = set()

    # Find themes that match this domain
    candidates = []
    for theme_id, theme in VISUAL_THEMES.items():
        if domain in theme["domains"] or "general" in theme["domains"]:
            if theme_id not in avoided_themes:
                candidates.append((theme_id, theme))

    # Fallback: if all matching themes are avoided, use any theme
    if not candidates:
        candidates = [
            (tid, t)
            for tid, t in VISUAL_THEMES.items()
            if tid not in avoided_themes
        ]

    # Final fallback: just pick any theme
    if not candidates:
        candidates = list(VISUAL_THEMES.items())

    # Deterministic selection using run_context hash
    if run_context:
        import hashlib
        ctx_hash = int(hashlib.md5(run_context.encode()).hexdigest()[:8], 16)
        # Prefer domain-matching themes first
        domain_matches = [(tid, t) for tid, t in candidates if domain in t["domains"]]
        if domain_matches:
            idx = ctx_hash % len(domain_matches)
            theme_id, theme = domain_matches[idx]
        else:
            idx = ctx_hash % len(candidates)
            theme_id, theme = candidates[idx]
    else:
        # Weighted random: prefer domain-matching themes
        domain_matches = [
            (tid, t) for tid, t in candidates if domain in t["domains"]
        ]
        if domain_matches and random.random() < 0.8:
            theme_id, theme = random.choice(domain_matches)
        else:
            theme_id, theme = random.choice(candidates)

    return theme_id, theme


def select_storytelling_pattern(
    domain: str,
    avoided_patterns: List[List[str]] = None,
    run_context: str = "",
    topic: str = "",
) -> Tuple[str, Dict]:
    """Select a storytelling pattern based on domain and topic.
    
    Args:
        domain: The content domain
        avoided_patterns: List of recent layout patterns to avoid
        run_context: Optional context for deterministic selection
        topic: The carousel topic for topic-level avoidance
    """
    if avoided_patterns is None:
        avoided_patterns = []

    # Find patterns suitable for this domain
    candidates = []
    for pattern_id, pattern in STORYTELLING_PATTERNS.items():
        if domain in pattern["suitable_for"] or "general" in pattern["suitable_for"]:
            candidates.append((pattern_id, pattern))

    if not candidates:
        candidates = list(STORYTELLING_PATTERNS.items())

    # Topic-level avoidance: check if pattern would repeat recent carousel topics
    recent_topics = get_recent_carousel_topics(10)
    recent_keywords = get_recent_carousel_keywords(10)
    
    # Score each pattern based on topic overlap
    scored_candidates = []
    for pattern_id, pattern in candidates:
        # Generate proposed layouts for this pattern
        proposed_layouts = [
            random.choice(slot[1]) for slot in pattern["slides"]
        ]
        
        # Check layout similarity
        layout_penalty = 0
        if is_layout_pattern_too_similar(proposed_layouts, avoided_patterns):
            layout_penalty = 100  # Strong penalty for similar layout
        
        # Topic overlap check (simplified - would need topic to fully check)
        topic_penalty = 0
        
        score = layout_penalty + topic_penalty
        scored_candidates.append((score, pattern_id, pattern))
    
    # Sort by penalty (lower is better)
    scored_candidates.sort(key=lambda x: x[0])
    
    # Deterministic selection from top candidates with low penalty
    if run_context:
        import hashlib
        ctx_hash = int(hashlib.md5(run_context.encode()).hexdigest()[:8], 16)
        # Pick from top 3 candidates with lowest penalty
        top_candidates = [c for c in scored_candidates if c[0] == scored_candidates[0][0]]
        if len(top_candidates) > 1:
            top_candidates = top_candidates[:3]
        idx = ctx_hash % len(top_candidates)
        _, pattern_id, pattern = top_candidates[idx]
    else:
        # Pick from top candidates with lowest penalty
        min_penalty = scored_candidates[0][0]
        top_candidates = [c for c in scored_candidates if c[0] == min_penalty]
        if len(top_candidates) > 1:
            top_candidates = top_candidates[:3]
        _, pattern_id, pattern = random.choice(top_candidates)
    
    return pattern_id, pattern


def _determine_slide_count(pattern: Dict, domain: str) -> int:
    """Determine optimal slide count (5-8) for the carousel."""
    base = len(pattern["slides"])  # Typically 6
    # Some domains benefit from extra slides
    if domain in ("cloud", "ai_ml", "security"):
        # 50% chance of adding an extra slide
        if random.random() < 0.5:
            return min(base + 1, 8)
    if domain == "career":
        # Career topics can be shorter
        if random.random() < 0.3:
            return max(base - 1, 5)
    return min(max(base, 5), 8)


# ─── Semantic Visual Motifs by Topic Type ─────────────────────────────────────

TOPIC_VISUAL_MOTIFS = {
    "model_release": {
        "visual_concept": "neural_intelligence",
        "visual_motifs": ["attention_heads", "activation_nodes", "latent_vectors", "context_streams"],
        "hero_visual": "model_architecture",
        "accent_style": "cyber_matrix",
    },
    "developer_tool": {
        "visual_concept": "developer_workspace",
        "visual_motifs": ["ast_nodes", "syntax_tokens", "file_tree", "terminal_prompt"],
        "hero_visual": "ide_window",
        "accent_style": "modern_ide",
    },
    "security_incident": {
        "visual_concept": "threat_vector",
        "visual_motifs": ["attack_path", "lock_shield", "warning_badge", "perimeter_mesh"],
        "hero_visual": "security_perimeter",
        "accent_style": "threat_alert",
    },
    "research_paper": {
        "visual_concept": "algorithmic_rigor",
        "visual_motifs": ["benchmark_radar", "matrix_transform", "coordinate_grid", "proof_graph"],
        "hero_visual": "methodology_flow",
        "accent_style": "academic_precision",
    },
    "programming_concept": {
        "visual_concept": "system_architecture",
        "visual_motifs": ["call_stack", "state_machine", "memory_registers", "concurrency_lanes"],
        "hero_visual": "execution_flow",
        "accent_style": "clean_systems",
    },
    "framework_update": {
        "visual_concept": "agent_orchestration",
        "visual_motifs": ["cyclic_graph", "state_node", "tool_dispatch", "checkpoint_queue"],
        "hero_visual": "state_graph",
        "accent_style": "agentic_high_tech",
    },
    "cloud_service": {
        "visual_concept": "distributed_cloud",
        "visual_motifs": ["vpc_edges", "service_clusters", "ingress_gateway", "data_mesh"],
        "hero_visual": "cloud_topology",
        "accent_style": "cloud_blueprint",
    },
    "industry_trend": {
        "visual_concept": "market_dynamics",
        "visual_motifs": ["growth_vectors", "delta_cards", "market_radar", "capital_flow"],
        "hero_visual": "market_shift",
        "accent_style": "executive_pulse",
    },
}

TOPIC_PREFERRED_PATTERNS = {
    "model_release": "news_breakdown",
    "developer_tool": "problem_solution",
    "security_incident": "security_incident",
    "programming_concept": "concept_explained",
    "research_paper": "architecture_walkthrough",
    "framework_update": "tutorial_flow",
    "cloud_service": "architecture_walkthrough",
    "industry_trend": "news_breakdown",
}


def create_visual_strategy(
    carousel: Dict,
    story: Optional[Dict] = None,
    run_context: str = "",
) -> Dict[str, Any]:
    """
    Create a complete, topic-aware visual strategy for a carousel.

    Args:
        carousel: The carousel JSON from LLM (with headline, slides, etc.)
        story: Optional original story metadata
        run_context: Optional context for deterministic selection (date + run number)

    Returns:
        Visual strategy dict with theme, layout, visual motifs, and per-slide instructions.
    """
    headline = carousel.get("headline", "")
    description = ""
    if story:
        description = story.get("description", "")
    elif carousel.get("summary"):
        description = carousel["summary"]

    # Extract topic intelligence if available
    topic_intel = carousel.get("_topic_intelligence") or {}
    topic_type = topic_intel.get("topic_type") or carousel.get("topic_type", "model_release")
    
    # Classify domain and difficulty
    domain = topic_intel.get("domain") or classify_domain(headline, description)
    difficulty = classify_difficulty(description)

    # Get avoidance lists
    avoided_themes = suggest_avoided_themes(7)
    avoided_patterns = suggest_avoided_layouts(7)

    # Select theme with run_context for deterministic selection
    theme_id, theme = select_theme(domain, avoided_themes, run_context=run_context)

    # Preferred pattern from topic intelligence
    preferred_pattern = TOPIC_PREFERRED_PATTERNS.get(topic_type)
    if preferred_pattern and preferred_pattern in STORYTELLING_PATTERNS:
        pattern_id = preferred_pattern
        pattern = STORYTELLING_PATTERNS[pattern_id]
    else:
        pattern_id, pattern = select_storytelling_pattern(
            domain, avoided_patterns, run_context=run_context, topic=headline
        )

    # Align with actual generated slides if present
    actual_slides = carousel.get("slides", [])
    if actual_slides:
        slide_count = len(actual_slides)
        slides_strategy = []
        layout_pattern = []
        for i, s in enumerate(actual_slides):
            chosen_layout = s.get("layout_type") or s.get("type", "hero_hook")
            layout_pattern.append(chosen_layout)
            title_pos = "center" if chosen_layout in ("hero_hook", "big_number", "scenario_question") else ("top-left" if i % 2 == 0 else "top-center")
            slides_strategy.append({
                "slide_number": i + 1,
                "role": s.get("role", "slide"),
                "purpose": s.get("role") or LAYOUT_TYPES.get(chosen_layout, {}).get("purpose", "explanation"),
                "layout_type": chosen_layout,
                "title_position": title_pos,
                "supports": LAYOUT_TYPES.get(chosen_layout, {}).get("supports", []),
            })
    else:
        # Determine slide count dynamically
        slide_count = _determine_slide_count(pattern, domain)
        slides_strategy = []
        pattern_slides = pattern["slides"]

        if slide_count > len(pattern_slides):
            extra_options = [
                ("data", ["metrics_cards", "checklist"]),
                ("comparison", ["common_mistake", "myth_vs_fact"]),
                ("quiz", ["quiz_choice", "quiz_predict_output"]),
                ("example", ["real_world_scenario", "analogy"]),
            ]
            for i in range(slide_count - len(pattern_slides)):
                pattern_slides = list(pattern_slides)
                insert_idx = len(pattern_slides) - 1
                extra = extra_options[i % len(extra_options)]
                pattern_slides.insert(insert_idx, extra)
        elif slide_count < len(pattern_slides):
            pattern_slides = list(pattern_slides)
            while len(pattern_slides) > slide_count:
                mid = len(pattern_slides) // 2
                pattern_slides.pop(mid)

        layout_pattern = []
        for i, (purpose, layout_candidates) in enumerate(pattern_slides):
            chosen_layout = random.choice(layout_candidates)
            layout_pattern.append(chosen_layout)
            title_pos = "center" if chosen_layout in ("hero_hook", "big_number", "scenario_question") else "top-left"
            slides_strategy.append({
                "slide_number": i + 1,
                "purpose": purpose,
                "layout_type": chosen_layout,
                "title_position": title_pos,
                "supports": LAYOUT_TYPES.get(chosen_layout, {}).get("supports", []),
            })

    # Retrieve semantic visual motifs
    motifs_config = TOPIC_VISUAL_MOTIFS.get(topic_type, TOPIC_VISUAL_MOTIFS["model_release"])

    strategy = {
        "topic": headline,
        "topic_type": topic_type,
        "domain": domain,
        "difficulty": difficulty,
        "audience": "students_and_professionals",
        "visual_theme": theme_id,
        "theme_css_class": theme["css_class"],
        "theme_colors": {
            "bg": theme["bg"],
            "card_bg": theme["card_bg"],
            "card_border": theme["card_border"],
            "text_primary": theme["text_primary"],
            "text_secondary": theme["text_secondary"],
            "text_muted": theme["text_muted"],
            "accent": theme["accent"],
            "accent_secondary": theme["accent_secondary"],
            "accent_success": theme["accent_success"],
            "accent_danger": theme["accent_danger"],
            "accent_warning": theme["accent_warning"],
            "code_bg": theme["code_bg"],
            "gradient_start": theme["gradient_start"],
            "gradient_end": theme["gradient_end"],
        },
        "visual_concept": motifs_config["visual_concept"],
        "visual_motifs": motifs_config["visual_motifs"],
        "hero_visual": motifs_config["hero_visual"],
        "accent_style": motifs_config["accent_style"],
        "storytelling_pattern": pattern_id,
        "slide_count": slide_count,
        "layout_pattern": layout_pattern,
        "slides": slides_strategy,
        "brand": {
            "handle": "@vijayakumarj_ai",
            "name": "Vijayakumar J",
            "short_name": "VJ",
        },
    }

    print(f"🎨 Visual strategy: topic_type={topic_type}, theme={theme_id}, pattern={pattern_id}, slides={slide_count}")
    print(f"   Motifs: {', '.join(motifs_config['visual_motifs'])} | Concept: {motifs_config['visual_concept']}")
    print(f"   Layouts: {' → '.join(layout_pattern)}")

    return strategy
