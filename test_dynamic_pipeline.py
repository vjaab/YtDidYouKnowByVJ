#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_dynamic_pipeline.py — Unit and integration tests for the dynamic carousel pipeline.
Tests topic intelligence, archetype selection, research brief, prompt building,
slide narrative validation, and visual strategy.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from ai_news_carousel import (
    CONTENT_ARCHETYPES,
    classify_topic_heuristically,
    build_research_brief,
    build_carousel_prompt,
    validate_slide_narrative,
    generate_fallback_carousel,
)
from visual_strategy import create_visual_strategy

def test_topic_classification():
    print("\n--- Test 1: Heuristic Topic Classification ---")
    stories = [
        {
            "title": "OpenAI Unveils GPT-4.5 with Advanced Multimodal Reasoning and 1M Context",
            "description": "OpenAI announced GPT-4.5 featuring unprecedented reasoning capabilities and 50% reduced inference costs across API calls.",
            "expected_type": "model_release"
        },
        {
            "title": "Cursor 2.0 Released: Autonomous Full-Codebase Multi-File Editing in VS Code",
            "description": "The Cursor AI team shipped an editor update with deep AST indexing, allowing autonomous refactoring across multiple files.",
            "expected_type": "developer_tool"
        },
        {
            "title": "Critical CVE-2024-9981 Exploit Found in LangChain Agent Sandboxing Logic",
            "description": "Security researchers discovered a prompt injection vulnerability in agent tool execution allowing arbitrary host commands.",
            "expected_type": "security_incident"
        },
        {
            "title": "DeepSeek Researchers Publish Mathematical Proof of Linear Attention Scaling on ArXiv",
            "description": "A new research paper proves constant memory complexity for reasoning models, beating standard quadratic transformers on HumanEval.",
            "expected_type": "research_paper"
        },
        {
            "title": "Understanding Event-Driven Concurrency Patterns in Modern High-Throughput Systems",
            "description": "A comprehensive deep dive into lock-free concurrency, state isolation, and append-only event streams.",
            "expected_type": "programming_concept"
        },
        {
            "title": "Anthropic Raises $4B Series D Valuation to Expand Compute Infrastructure",
            "description": "Anthropic secured major funding rounds to scale out frontier model training clusters.",
            "expected_type": "industry_trend"
        }
    ]

    for s in stories:
        intel = classify_topic_heuristically(s)
        print(f"Title: {s['title'][:50]}...")
        print(f"  -> Predicted: {intel['topic_type']} (domain: {intel['domain']}, entity: {intel['primary_entity']})")
        assert intel["topic_type"] == s["expected_type"], f"Expected {s['expected_type']}, got {intel['topic_type']}"
    print("✅ All topic classifications matched expected archetypes!")


def test_research_brief_and_prompts():
    print("\n--- Test 2: Research Brief & Prompt Architecture ---")
    story = {
        "title": "Google DeepMind Launches Gemini 2.5 Flash with 2M Token Context",
        "description": "Google released Gemini 2.5 Flash with 2M tokens context window and 40% latency reduction. Achieving 92.1% on MMLU benchmark.",
        "source": {"name": "Google DeepMind"},
        "url": "https://deepmind.google/gemini"
    }
    intel = classify_topic_heuristically(story)
    brief = build_research_brief(story, intel)
    
    assert "2.5" in str(brief["versions"]) or "2M" in str(brief["context_windows"])
    assert any("40%" in m or "92.1%" in m for m in brief["metrics"])
    print(f"  Verified Metrics: {brief['metrics']}")
    print(f"  Verified Versions: {brief['versions']}")
    print(f"  Verified Context: {brief['context_windows']}")

    archetype = CONTENT_ARCHETYPES[intel["topic_type"]]
    prompt = build_carousel_prompt(story, intel, brief, archetype, slide_count=8)
    
    assert "ANTI-HALLUCINATION & RIGOR MANDATE" in prompt
    assert "NO-REPETITION MANDATE" in prompt
    assert "Slide 1 [Role: 'hook'" in prompt
    assert "Slide 4 [Role: 'benchmark_evidence'" in prompt
    print("✅ Prompt contains strict anti-hallucination and archetype narrative instructions!")


def test_slide_narrative_validator():
    print("\n--- Test 3: Slide Narrative Diversity Validator ---")
    
    # Repetitive / bad carousel
    bad_carousel = {
        "depth_score": 5,
        "slides": [
            {"slide_number": 1, "layout_type": "hero_hook", "title": "Gemini 2.5 is here", "body": "Gemini 2.5 has been released by Google today."},
            {"slide_number": 2, "layout_type": "whats_new", "title": "Gemini 2.5 is here", "body": "Gemini 2.5 has been released by Google today with new features."},
            {"slide_number": 3, "layout_type": "what_happened", "title": "What happened", "body": "Google released Gemini 2.5 today for all developers."},
            {"slide_number": 4, "layout_type": "why_matters", "title": "Why it matters", "body": "It matters because Gemini 2.5 is very good."},
        ]
    }
    is_valid, issues = validate_slide_narrative(bad_carousel)
    assert not is_valid, "Validator should have failed bad carousel"
    print(f"  Correctly rejected repetitive carousel with issues: {issues}")

    # Good dynamic fallback carousel
    story = {
        "title": "Cursor AI 2.0 Revolutionizes Multi-File Coding",
        "description": "Cursor released 2.0 with AST codebase indexing and full multi-file autonomous diffs.",
        "source": {"name": "Cursor AI"},
        "url": "https://cursor.com"
    }
    good_carousel = generate_fallback_carousel(story)
    is_valid_good, good_issues = validate_slide_narrative(good_carousel)
    assert is_valid_good, f"Good fallback should pass validation, got issues: {good_issues}"
    print(f"✅ Dynamic fallback carousel passed narrative diversity validator ({len(good_carousel['slides'])} slides)!")


def test_visual_strategy_integration():
    print("\n--- Test 4: Visual Strategy Semantic Motifs ---")
    story = {
        "title": "Critical Zero-Day Exploit in AWS Lambda Ingress",
        "description": "A CVE exploit allowing sandbox escapes in AWS Lambda serverless execution.",
        "source": {"name": "Cloud Security Alert"},
        "url": "https://cve.mitre.org"
    }
    carousel = generate_fallback_carousel(story)
    strat = create_visual_strategy(carousel, story=story)
    
    assert strat["topic_type"] == "security_incident"
    assert strat["visual_concept"] == "threat_vector"
    assert "attack_path" in strat["visual_motifs"]
    assert strat["storytelling_pattern"] == "security_incident"
    assert strat["slide_count"] == len(carousel["slides"])
    print(f"  Visual Concept: {strat['visual_concept']}")
    print(f"  Motifs: {strat['visual_motifs']}")
    print(f"  Pattern: {strat['storytelling_pattern']} (Slides: {strat['slide_count']})")
    print("✅ Visual strategy correctly adopts semantic motifs from topic intelligence!")


if __name__ == "__main__":
    test_topic_classification()
    test_research_brief_and_prompts()
    test_slide_narrative_validator()
    test_visual_strategy_integration()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
