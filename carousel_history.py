#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
carousel_history.py — Anti-repetition tracker for carousel generation.
Tracks recent themes, layout patterns, and topics to ensure visual variety.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

HISTORY_FILE = Path(__file__).parent / ".carousel_history.json"
MAX_HISTORY = 20  # Keep last 20 carousels


def load_history() -> Dict:
    """Load carousel history from disk."""
    if not HISTORY_FILE.exists():
        return {
            "carousels": [],
            "last_updated": None,
        }
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"carousels": [], "last_updated": None}


def save_history(history: Dict):
    """Save carousel history to disk."""
    history["last_updated"] = datetime.now(timezone.utc).isoformat()
    # Trim to MAX_HISTORY
    history["carousels"] = history["carousels"][-MAX_HISTORY:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def record_carousel(
    topic: str,
    theme: str,
    layout_pattern: List[str],
    domain: str = "",
    slide_count: int = 6,
):
    """Record a generated carousel for anti-repetition tracking."""
    history = load_history()
    entry = {
        "topic": topic,
        "theme": theme,
        "layout_pattern": layout_pattern,
        "domain": domain,
        "slide_count": slide_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    history["carousels"].append(entry)
    save_history(history)
    print(f"📊 Carousel history updated: {len(history['carousels'])} entries")


def get_recent_themes(n: int = 5) -> List[str]:
    """Get the last N themes used."""
    history = load_history()
    return [c["theme"] for c in history["carousels"][-n:]]


def get_recent_layout_patterns(n: int = 5) -> List[List[str]]:
    """Get the last N layout patterns used."""
    history = load_history()
    return [c["layout_pattern"] for c in history["carousels"][-n:]]


def get_recent_domains(n: int = 5) -> List[str]:
    """Get the last N domains used."""
    history = load_history()
    return [c.get("domain", "") for c in history["carousels"][-n:]]


def suggest_avoided_themes(n: int = 3) -> Set[str]:
    """Return themes that should be avoided (used in last N carousels)."""
    return set(get_recent_themes(n))


def suggest_avoided_layouts(n: int = 3) -> List[List[str]]:
    """Return layout patterns to avoid (used in last N carousels)."""
    return get_recent_layout_patterns(n)


def is_layout_pattern_too_similar(
    proposed: List[str], recent: List[List[str]], threshold: float = 0.7
) -> bool:
    """Check if a proposed layout pattern is too similar to recent ones."""
    if not recent:
        return False
    for past in recent:
        if not past:
            continue
        # Calculate overlap ratio
        min_len = min(len(proposed), len(past))
        if min_len == 0:
            continue
        matches = sum(1 for a, b in zip(proposed, past) if a == b)
        overlap = matches / min_len
        if overlap >= threshold:
            return True
    return False
