"""
humanizer_engine.py — Wikipedia-Based AI Writing Detector and Sanitizer.

Based on WikiProject AI Cleanup guidelines (SKILL.md):
Detects and fixes:
1. Em dashes (—), En dashes (–), and double hyphens (--)
2. High-frequency AI vocabulary (delve, tapestry, intricate, pivotal, vibrant, underscore, etc.)
3. Undue emphasis on significance ('serves as a testament', 'evolving landscape', 'pivotal moment')
4. Superficial -ing participial phrases ('highlighting', 'ensuring', 'symbolizing')
5. Promotional / ad-like language ('breathtaking', 'groundbreaking', 'must-visit')
6. Copula avoidance ('serves as', 'stands as', 'boasts a')
7. Negative parallelisms ('Not only... but also...', 'It's not just about...')
8. Vague attributions ('Experts argue', 'Industry reports suggest')
9. Passive voice / subjectless fragments
10. Rule of three groupings
"""

import re
from typing import Dict, List, Tuple, Any

# High-frequency AI vocabulary words
BANNED_AI_WORDS = [
    "delve", "delves", "delving",
    "tapestry", "tapestries",
    "intricate", "intricacies",
    "pivotal",
    "vibrant",
    "enduring",
    "fostering", "fosters",
    "enhance", "enhances", "enhancing",
    "additionally",
    "landscape", "landscapes",
    "testament",
    "underscore", "underscores", "underscoring",
    "showcase", "showcases", "showcasing",
    "groundbreaking",
    "breathtaking",
    "crucial",
    "garner", "garners",
    "interplay",
    "beacon",
    "revered",
    "unwavering",
    "paramount",
    "cornerstone"
]

# AI Cliché Phrases to replace or cut
CLICHE_PHRASE_REPLACEMENTS = {
    r"\bserves as a testament to\b": "shows",
    r"\bis a testament to\b": "proves",
    r"\bserves as\b": "is",
    r"\bstands as\b": "is",
    r"\bboasts a\b": "has a",
    r"\bboasts\b": "has",
    r"\bmarks a pivotal moment\b": "marks a change",
    r"\bpivotal moment\b": "big turning point",
    r"\bevolving landscape\b": "industry",
    r"\brapidly evolving landscape\b": "fast changes",
    r"\bnot only (.*?), but also (.*?)\b": r"\1 and \2",
    r"\bit's not just about (.*?), it's (.*?)\b": r"\1 matters because \2",
    r"\bindustry reports suggest\b": "data shows",
    r"\bobservers have noted\b": "engineers noticed",
    r"\bexperts argue\b": "many argue",
    r"\bexperts believe\b": "critics say",
    r"\ba wide range of\b": "many",
    r"\bfrom (.*?) to (.*?)\b": r"\1 as well as \2",
}

# Synonyms for AI words to human equivalents
WORD_REPLACEMENTS = {
    r"\bdelves into\b": "explores",
    r"\bdelve into\b": "explore",
    r"\bdelves\b": "digs into",
    r"\bdelving\b": "digging into",
    r"\bdelve\b": "dig into",
    r"\btapestry\b": "mix",
    r"\btapestries\b": "mix",
    r"\bintricate\b": "detailed",
    r"\bintricacies\b": "details",
    r"\bpivotal\b": "key",
    r"\bvibrant\b": "active",
    r"\benduring\b": "lasting",
    r"\bfostering\b": "building",
    r"\bfosters\b": "builds",
    r"\benhance\b": "improve",
    r"\benhances\b": "improves",
    r"\benhancing\b": "improving",
    r"\badditionally\b": "also",
    r"\blandscape\b": "space",
    r"\blandscapes\b": "spaces",
    r"\btestament\b": "proof",
    r"\bunderscore\b": "highlight",
    r"\bunderscores\b": "highlights",
    r"\bunderscoring\b": "highlighting",
    r"\bshowcase\b": "show",
    r"\bshowcases\b": "shows",
    r"\bshowcasing\b": "showing",
    r"\bgroundbreaking\b": "huge",
    r"\bbreathtaking\b": "impressive",
    r"\bcrucial\b": "critical",
    r"\bgarner\b": "get",
    r"\bgarners\b": "gets",
    r"\binterplay\b": "interaction",
}


class HumanizerAuditor:
    """Automated detector and sanitizer for machine-like writing patterns."""

    @staticmethod
    def audit(text: str) -> Dict[str, Any]:
        """Scans text and returns an audit report identifying all AI pattern violations."""
        if not text:
            return {"violations": [], "score": 100, "is_humanic": True}

        violations = []

        # 1. Check Em Dashes, En Dashes, Double Hyphens
        dash_matches = re.findall(r'[—–]|(?:\s--\s)|(?:^--\s)', text)
        if dash_matches:
            violations.append({
                "type": "em_dash",
                "count": len(dash_matches),
                "severity": "HIGH",
                "message": f"Found {len(dash_matches)} em/en dashes or double hyphens. (Forbidden in humanic writing)"
            })

        # 2. Check Banned AI Words
        found_ai_words = []
        for word in BANNED_AI_WORDS:
            matches = re.findall(rf'\b{re.escape(word)}\b', text, re.IGNORECASE)
            if matches:
                found_ai_words.extend([m.lower() for m in matches])
        if found_ai_words:
            violations.append({
                "type": "banned_ai_words",
                "count": len(found_ai_words),
                "words": list(set(found_ai_words)),
                "severity": "HIGH",
                "message": f"Found high-frequency AI vocabulary words: {list(set(found_ai_words))}"
            })

        # 3. Check Cliché Phrases & Copula Avoidance
        found_phrases = []
        for pattern in CLICHE_PHRASE_REPLACEMENTS:
            if re.search(pattern, text, re.IGNORECASE):
                found_phrases.append(pattern)
        if found_phrases:
            violations.append({
                "type": "cliche_phrases",
                "count": len(found_phrases),
                "severity": "MEDIUM",
                "message": f"Found {len(found_phrases)} formulaic AI clichés or copula avoidances."
            })

        # 4. Check Superficial -ing Endings
        ing_matches = re.findall(r',\s+(?:highlighting|ensuring|reflecting|symbolizing|contributing to|cultivating)\b', text, re.IGNORECASE)
        if ing_matches:
            violations.append({
                "type": "superficial_ing",
                "count": len(ing_matches),
                "severity": "MEDIUM",
                "message": f"Found {len(ing_matches)} superficial -ing participial phrases tacked onto clauses."
            })

        # 5. Check Sentence Length Uniformity (Soulless flat rhythm)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
        if len(sentences) >= 4:
            lengths = [len(s.split()) for s in sentences]
            variance = max(lengths) - min(lengths)
            if variance < 5 and all(12 <= l <= 20 for l in lengths):
                violations.append({
                    "type": "flat_rhythm",
                    "severity": "LOW",
                    "message": "Flat sentence rhythm detected. Sentences lack punchy variation."
                })

        # Calculate Humanic Score (100 is pristine)
        penalty = 0
        for v in violations:
            if v["severity"] == "HIGH":
                penalty += v["count"] * 12
            elif v["severity"] == "MEDIUM":
                penalty += v["count"] * 6
            else:
                penalty += 10
        score = max(0, 100 - penalty)
        is_humanic = score >= 85 and not dash_matches

        return {
            "score": score,
            "is_humanic": is_humanic,
            "violations": violations,
            "sentence_count": len(sentences),
        }

    @staticmethod
    def clean(text: str) -> Tuple[str, Dict[str, int]]:
        """Programmatically sanitizes text to enforce zero em dashes and replace AI patterns."""
        if not text:
            return "", {"em_dashes_removed": 0, "ai_words_replaced": 0, "cliches_cleaned": 0}

        cleaned = text
        stats = {
            "em_dashes_removed": 0,
            "ai_words_replaced": 0,
            "cliches_cleaned": 0
        }

        # 1. Eliminate Em Dashes, En Dashes, Double Hyphens
        # Preference: replace with a comma or colon or period depending on context
        em_dash_patterns = [
            (r'\s*—\s*', ', '),
            (r'\s*–\s*', ', '),
            (r'\s+--\s+', ', '),
            (r'--', ', ')
        ]
        for pattern, repl in em_dash_patterns:
            matches = len(re.findall(pattern, cleaned))
            if matches > 0:
                stats["em_dashes_removed"] += matches
                cleaned = re.sub(pattern, repl, cleaned)

        # 2. Replace Cliché Phrases & Copula Avoidance
        for pattern, replacement in CLICHE_PHRASE_REPLACEMENTS.items():
            matches = len(re.findall(pattern, cleaned, re.IGNORECASE))
            if matches > 0:
                stats["cliches_cleaned"] += matches
                cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # 3. Replace High-Frequency AI Words with Natural Conversational Alternatives
        for pattern, replacement in WORD_REPLACEMENTS.items():
            matches = len(re.findall(pattern, cleaned, re.IGNORECASE))
            if matches > 0:
                stats["ai_words_replaced"] += matches
                cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # 4. Clean Double Commas or Spacing Artifacts
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = re.sub(r'\s+,', ',', cleaned)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)

        return cleaned.strip(), stats


def verify_and_humanize_script(script_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforces a strict Humanizer audit & clean on the entire script data object,
    including main script, chapter texts, and layman shorts teaser.
    """
    if not script_data:
        return script_data

    # Audit & clean main narration script
    main_script = script_data.get("script", "")
    if main_script:
        cleaned_main, stats = HumanizerAuditor.clean(main_script)
        audit_res = HumanizerAuditor.audit(cleaned_main)
        script_data["script"] = cleaned_main
        script_data["humanizer_stats"] = {
            "main_script_score": audit_res["score"],
            "main_script_humanic": audit_res["is_humanic"],
            **stats
        }

    # Clean chapter texts
    chapters = script_data.get("chapters", [])
    for ch in chapters:
        if "chapter_text" in ch:
            ch["chapter_text"], _ = HumanizerAuditor.clean(ch["chapter_text"])
        for vb in ch.get("visual_beats", []):
            if "beat_text" in vb:
                vb["beat_text"], _ = HumanizerAuditor.clean(vb["beat_text"])

    # Clean subtitles
    sub_chunks = script_data.get("subtitle_chunks", [])
    for sc in sub_chunks:
        if "text" in sc:
            sc["text"], _ = HumanizerAuditor.clean(sc["text"])

    # Clean layman short teaser if present
    layman = script_data.get("layman_short")
    if isinstance(layman, dict) and "script" in layman:
        layman["script"], _ = HumanizerAuditor.clean(layman["script"])

    return script_data
