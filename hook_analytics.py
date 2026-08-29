"""
hook_analytics.py — Hook Pattern Performance Analytics
Tracks A/B test results for hook patterns per category to optimize selection.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import LOGS_DIR

HOOK_ANALYTICS_FILE = Path(LOGS_DIR) / "hook_analytics.json"

# 5 Proven Viral Hook Patterns (from HOOK_AGENT_TEMPLATE)
HOOK_PATTERNS = [
    {
        "id": "negative_warning",
        "name": "Negative/Warning",
        "description": "You're probably using [Tool] wrong / Stop paying for [Tool] / Don't make this mistake",
        "examples": [
            "You're probably using {tool} wrong",
            "Stop paying for {tool}",
            "Don't make this {tool} mistake",
            "Your {workflow} is wasting hours"
        ]
    },
    {
        "id": "result_first",
        "name": "Result-First Reveal",
        "description": "Show the payoff immediately / This tool saves hours / Free alternative to paid tool",
        "examples": [
            "This AI tool saves developers hours",
            "I found a free alternative to {paid_tool}",
            "This GitHub repo replaces {popular_tool}",
            "Built a {result} in {time}"
        ]
    },
    {
        "id": "stat_contradiction",
        "name": "Stat/Contradiction",
        "description": "Google just killed X / $4.2B wasted / 90% don't know this",
        "examples": [
            "{company} just killed {feature}",
            "${amount} wasted on {thing}",
            "{percent}% of developers don't know this",
            "Everyone thinks X, but actually Y"
        ]
    },
    {
        "id": "curiosity_gap",
        "name": "Curiosity Gap",
        "description": "Most don't know this feature / Your app is secretly doing X / What company hides",
        "examples": [
            "Most developers don't know this {feature}",
            "Your {app} is secretly doing {thing}",
            "What {company} doesn't want you to know",
            "The {setting} hiding in plain sight"
        ]
    },
    {
        "id": "personal_stake",
        "name": "Personal Stake",
        "description": "Your code has flaw / AI reads your data / You're overpaying",
        "examples": [
            "Your code has this security flaw",
            "This AI reads your private data",
            "You're overpaying for {service}",
            "Your {data} is exposed right now"
        ]
    }
]

# Default performance scores (will be updated from analytics)
DEFAULT_PATTERN_SCORES = {
    "negative_warning": 5.0,
    "result_first": 5.0,
    "stat_contradiction": 5.0,
    "curiosity_gap": 5.0,
    "personal_stake": 5.0,
}


def _load_analytics() -> Dict:
    """Load hook analytics from JSON file."""
    if HOOK_ANALYTICS_FILE.exists():
        try:
            with open(HOOK_ANALYTICS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "categories": {},
        "global": {pattern["id"]: {"views": 0, "retention": 0.0, "engagement": 0.0, "count": 0} 
                  for pattern in HOOK_PATTERNS},
        "last_updated": None
    }


def _save_analytics(data: Dict):
    """Save hook analytics to JSON file."""
    HOOK_ANALYTICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = datetime.now().isoformat()
    with open(HOOK_ANALYTICS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_hook_analytics(category: Optional[str] = None) -> Dict:
    """Get current hook analytics, optionally filtered by category."""
    data = _load_analytics()
    if category:
        return data.get("categories", {}).get(category, {})
    return data


def record_hook_performance(
    category: str,
    pattern_id: str,
    variant_id: str,
    views: int,
    retention_rate: float,
    engagement_rate: float,
    watch_time_seconds: float = 0
):
    """
    Record performance metrics for a hook pattern/variant.
    Called after video publishes with analytics data.
    """
    data = _load_analytics()
    
    # Update category-level analytics
    if category not in data["categories"]:
        data["categories"][category] = {}
    
    cat = data["categories"][category]
    if pattern_id not in cat:
        cat[pattern_id] = {"variants": {}, "total_views": 0, "total_videos": 0, "avg_retention": 0.0, "avg_engagement": 0.0}
    
    pattern = cat[pattern_id]
    
    # Update variant
    if variant_id not in pattern["variants"]:
        pattern["variants"][variant_id] = {"views": 0, "retention": 0.0, "engagement": 0.0, "count": 0}
    
    variant = pattern["variants"][variant_id]
    variant["views"] = variant.get("views", 0) + views
    variant["count"] = variant.get("count", 0) + 1
    
    # Running average for retention
    old_ret = variant.get("retention", 0.0)
    variant["retention"] = (old_ret * (variant["count"] - 1) + retention_rate) / variant["count"]
    
    old_eng = variant.get("engagement", 0.0)
    variant["engagement"] = (old_eng * (variant["count"] - 1) + engagement_rate) / variant["count"]
    
    # Update pattern totals
    pattern["total_views"] = pattern.get("total_views", 0) + views
    pattern["total_videos"] = pattern.get("total_videos", 0) + 1
    
    # Recalculate pattern averages
    total_ret = sum(v.get("retention", 0.0) * v.get("count", 1) for v in pattern["variants"].values())
    total_eng = sum(v.get("engagement", 0.0) * v.get("count", 1) for v in pattern["variants"].values())
    total_cnt = sum(v.get("count", 1) for v in pattern["variants"].values())
    
    pattern["avg_retention"] = total_ret / max(total_cnt, 1)
    pattern["avg_engagement"] = total_eng / max(total_cnt, 1)
    
    # Update global
    glob = data["global"].setdefault(pattern_id, {"views": 0, "retention": 0.0, "engagement": 0.0, "count": 0})
    glob["views"] = glob.get("views", 0) + views
    glob["count"] = glob.get("count", 0) + 1
    glob["retention"] = (glob.get("retention", 0.0) * (glob["count"] - 1) + retention_rate) / glob["count"]
    glob["engagement"] = (glob.get("engagement", 0.0) * (glob["count"] - 1) + engagement_rate) / glob["count"]
    
    _save_analytics(data)
    print(f"📊 Hook analytics updated: {category}/{pattern_id}/{variant_id} - {views} views, {retention_rate:.1%} retention")


def select_hook_patterns_for_category(category: str, num_patterns: int = 3) -> List[str]:
    """
    Select the best hook patterns for a category based on historical performance.
    Uses Thompson Sampling for exploration/exploitation balance.
    """
    import random
    import math
    
    data = _load_analytics()
    cat_data = data.get("categories", {}).get(category, {})
    global_data = data.get("global", {})
    
    pattern_scores = {}
    
    for pattern in HOOK_PATTERNS:
        pid = pattern["id"]
        
        # Get category-specific stats
        cat_stats = cat_data.get(pid, {})
        cat_ret = cat_stats.get("avg_retention", 0.0)
        cat_eng = cat_stats.get("avg_engagement", 0.0)
        cat_cnt = cat_stats.get("total_videos", 0)
        
        # Get global stats as prior
        glob_stats = global_data.get(pid, {})
        glob_ret = glob_stats.get("retention", 0.0)
        glob_eng = glob_stats.get("engagement", 0.0)
        glob_cnt = glob_stats.get("count", 0)
        
        # Weighted score: 70% category, 30% global (with smoothing)
        if cat_cnt > 0:
            weight = min(cat_cnt / 10.0, 0.7)  # Max 70% weight to category after 10 videos
        else:
            weight = 0.0
        
        # Combined score (retention + engagement)
        cat_score = (cat_ret + cat_eng) / 2 if cat_cnt > 0 else 0.5
        glob_score = (glob_ret + glob_eng) / 2 if glob_cnt > 0 else 0.5
        
        score = weight * cat_score + (1 - weight) * glob_score
        
        # Thompson Sampling: add exploration bonus
        # Less tested patterns get a bonus
        exploration = math.sqrt(2 * math.log(max(cat_cnt + glob_cnt + 1, 2)) / max(cat_cnt + glob_cnt, 1))
        
        pattern_scores[pid] = score + 0.1 * exploration
    
    # Sort by score descending, pick top N
    sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [pid for pid, _ in sorted_patterns[:num_patterns]]
    
    # Always ensure diversity - if top patterns are too similar, add variety
    if len(selected) >= 2:
        # Check if we have both negative and positive patterns
        has_negative = any(p in selected for p in ["negative_warning", "stat_contradiction"])
        has_positive = any(p in selected for p in ["result_first", "curiosity_gap"])
        
        if not has_negative and "negative_warning" not in selected:
            # Replace lowest scored with negative_warning for variety
            selected[-1] = "negative_warning"
        elif not has_positive and "result_first" not in selected:
            selected[-1] = "result_first"
    
    print(f"🎯 Selected hook patterns for {category}: {selected}")
    print(f"   Scores: {pattern_scores}")
    
    return selected


def get_hook_pattern_examples(pattern_id: str, topic: str = "") -> List[str]:
    """Get example hooks for a pattern, optionally customized for topic."""
    for pattern in HOOK_PATTERNS:
        if pattern["id"] == pattern_id:
            examples = pattern["examples"][:]
            # Simple topic substitution
            if topic:
                examples = [ex.replace("{tool}", topic).replace("{feature}", topic).replace("{app}", topic) 
                          for ex in examples]
            return examples
    return []


def get_selected_hook_info(pattern_id: str) -> Dict:
    """Get full pattern info for a pattern ID."""
    for pattern in HOOK_PATTERNS:
        if pattern["id"] == pattern_id:
            return pattern
    return {}


def update_hook_variant_views(
    video_id: str,
    category: str,
    pattern_id: str,
    variant_id: str,
    views: int,
    retention_rate: float,
    engagement_rate: float
):
    """
    Update analytics when YouTube Analytics data becomes available.
    Call this from a scheduled job that pulls YouTube Analytics.
    """
    record_hook_performance(category, pattern_id, variant_id, views, retention_rate, engagement_rate)


def get_category_leaderboard(category: str, top_n: int = 5) -> List[Dict]:
    """Get top performing hook variants for a category."""
    data = _load_analytics()
    cat_data = data.get("categories", {}).get(category, {})
    
    results = []
    for pattern_id, pattern_data in cat_data.items():
        for variant_id, variant_data in pattern_data.get("variants", {}).items():
            results.append({
                "pattern": pattern_id,
                "variant": variant_id,
                "views": variant_data.get("views", 0),
                "retention": variant_data.get("retention", 0.0),
                "engagement": variant_data.get("engagement", 0.0),
                "count": variant_data.get("count", 0),
                "score": (variant_data.get("retention", 0.0) + variant_data.get("engagement", 0.0)) / 2
            })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def print_analytics_summary():
    """Print a summary of hook analytics for debugging."""
    data = _load_analytics()
    print("\n📊 HOOK ANALYTICS SUMMARY")
    print("=" * 60)
    
    for category, cat_data in data.get("categories", {}).items():
        print(f"\n📁 Category: {category}")
        for pattern_id, pattern_data in cat_data.items():
            total_v = pattern_data.get("total_videos", 0)
            avg_ret = pattern_data.get("avg_retention", 0.0)
            avg_eng = pattern_data.get("avg_engagement", 0.0)
            print(f"  {pattern_id}: {total_v} videos, {avg_ret:.1%} retention, {avg_eng:.1%} engagement")
            for var_id, var_data in pattern_data.get("variants", {}).items():
                print(f"    {var_id}: {var_data.get('views', 0)} views, {var_data.get('retention', 0.0):.1%} ret")
    
    print("\n🌍 Global:")
    for pid, glob in data.get("global", {}).items():
        if glob.get("count", 0) > 0:
            print(f"  {pid}: {glob['count']} videos, {glob['retention']:.1%} ret, {glob['engagement']:.1%} eng")


# Convenience function for pipeline integration
def get_optimized_hook_prompt(category: str, research: Dict) -> str:
    """
    Generate an optimized hook prompt that includes the best patterns for the category.
    """
    selected_patterns = select_hook_patterns_for_category(category, num_patterns=3)
    
    pattern_descriptions = []
    for pid in selected_patterns:
        info = get_hook_pattern_info(pid)
        examples = get_hook_pattern_examples(pid)
        pattern_descriptions.append(f"""
  {info['name']} ({pid}):
    Description: {info['description']}
    Examples: {examples[:2]}
""")
    
    return f"""HOOK AGENT TASK:
Generate 3 distinct A/B TEST HOOK VARIANTS for YouTube Shorts A/B testing.
Based on analytics for category '{category}', use these OPTIMIZED patterns:

{''.join(pattern_descriptions)}

RULES:
- First 3 words MUST stop the scroll
- Max 8 words. One clause only.
- Each variant = same pattern, different angles on the topic
- Return ONLY JSON with ab_test_variants array

RESEARCH:
{json.dumps(research, indent=2)}
"""


import json  # for get_optimized_hook_prompt