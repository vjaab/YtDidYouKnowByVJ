import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from templates.config import TemplateConfig, FACEBOOK_CONFIG, INSTAGRAM_CONFIG
from cartoon_dialogue_engine import clean_bubble_text, build_dialogue_prompt
from ai_news_carousel import build_carousel_prompt


def test_clean_bubble_text_length_and_artifacts():
    # Long text with more than 12 words should be trimmed to 12 words
    long_text = "Wait, did you know that 99% of the internet is actually underwater in ocean cables and deep trenches?"
    cleaned = clean_bubble_text(long_text, max_words=12)
    words = cleaned.split()
    assert len(words) <= 12
    assert cleaned.endswith("...")

    # Artifacts like [pause] or (continue) should be stripped cleanly
    artifact_text = "[pause] Did you know that? (continue) It's true!"
    cleaned_artifact = clean_bubble_text(artifact_text, max_words=12)
    assert "[pause]" not in cleaned_artifact
    assert "(continue)" not in cleaned_artifact
    assert "Did you know that?" in cleaned_artifact


def test_template_config_high_readability_tokens():
    cfg = TemplateConfig()
    # High font size requirements for 1080x1350 canvas mobile viewports
    assert cfg.font_size_base >= 28
    assert cfg.font_size_sm >= 20
    assert cfg.font_size_lg >= 36
    assert cfg.font_size_xl >= 48
    assert cfg.font_size_2xl >= 64

    dict_repr = cfg.to_dict()
    assert dict_repr["font_size_base"] >= 28
    assert dict_repr["font_size_2xl"] >= 64


def test_cartoon_dialogue_prompt_mandates_crisp_concise_rules():
    prompt = build_dialogue_prompt("did_you_know", "Why GPS Needs Einstein")
    assert "STRICTLY 6 TO 12 WORDS PER BUBBLE" in prompt
    assert "ZERO text overload" in prompt


def test_ai_news_carousel_prompt_mandates_crisp_copy():
    dummy_story = {"title": "GPT-5 Released", "description": "New model launched."}
    dummy_intel = {"primary_entity": "OpenAI", "topic_type": "model_release"}
    dummy_brief = {
        "versions": [],
        "param_sizes": [],
        "context_windows": [],
        "metrics": [],
        "benchmarks": [],
        "technologies": [],
        "unspecified_aspects": [],
    }
    dummy_archetype = {
        "name": "Model Release",
        "story_angle": "technical_breakdown",
        "conditional_rules": "",
        "narrative_roles": [
            {"role": "hook", "layout": "whats_new", "eyebrow": "RELEASE", "focus": "Overview"}
        ],
    }
    prompt = build_carousel_prompt(dummy_story, dummy_intel, dummy_brief, dummy_archetype, slide_count=1)
    assert "CRISP & CONCISE COPY (MOBILE READABILITY)" in prompt
    assert "strictly under 25 words" in prompt
