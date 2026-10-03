"""Tests for Tier-2 Shorts growth optimizations:
- Audience reach multiplier (consumer tech boosted, niche ML penalized)
- Weekly schedule consolidated around mass-appeal pillars
- Topic type allocation balanced for high-retention discovery (did_you_know, tools, news)
- Target country sequence includes high-velocity regions (IN, US, etc.)
- YouTube Outlier hunter specifically targets Shorts (videoDuration='short')
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trending_engine import compute_audience_reach_multiplier
from ecosystem_logic import WEEKLY_SCHEDULE
from topic_tracker import get_next_topic_type_by_ratio, get_next_target_country


def test_audience_reach_multiplier_boosts_consumer_tech():
    consumer_article = {
        "title": "Secret iPhone battery hack you never knew existed",
        "description": "Stop doing this with your charger. Apple hidden setting saves your battery life.",
    }
    mult = compute_audience_reach_multiplier(consumer_article)
    assert mult >= 1.5, f"Expected consumer tech boost >= 1.5, got {mult}"


def test_audience_reach_multiplier_penalizes_niche_ml():
    niche_article = {
        "title": "OrcaSAQ-2-Cyber-27B-Uncensored-GGUF Model Release",
        "description": "Q4_K_M quantized weights released with LoRA checkpoint on HuggingFace Hub.",
    }
    mult = compute_audience_reach_multiplier(niche_article)
    assert mult <= 0.4, f"Expected niche ML penalty <= 0.4, got {mult}"


def test_weekly_schedule_mass_appeal_pillars():
    allowed_pillars = {
        "Facts & Trivia",
        "AI & Tech Tools",
        "Tech Gadgets & Inventions",
        "Coding & Development Hacks",
    }
    for day, categories in WEEKLY_SCHEDULE.items():
        for cat in categories:
            assert cat in allowed_pillars, f"Day {day} contains non-mass-appeal category '{cat}'"


def test_topic_allocation_returns_mass_appeal_type():
    allocated = get_next_topic_type_by_ratio()
    assert allocated in ["did_you_know", "tools", "news"], f"Unexpected topic type: {allocated}"


def test_target_country_sequence_includes_india():
    # Cold start or traversal should include India (IN)
    with patch("topic_tracker.load_tracker", return_value={"history": []}):
        country = get_next_target_country()
        assert country == "IN"


def test_youtube_outlier_targets_shorts():
    import trending_engine
    with patch("trending_engine.YOUTUBE_DATA_API_KEY", "mock_key"), \
         patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": []}
        mock_get.return_value = mock_resp

        trending_engine.fetch_youtube_outlier_trends(target_country="IN", category="AI & Tech Tools")

        # Verify search query called with videoDuration='short'
        called = False
        for call_args in mock_get.call_args_list:
            params = call_args[1].get("params", {})
            if params.get("type") == "video":
                assert params.get("videoDuration") == "short"
                called = True
                break
        assert called, "Search was not called with videoDuration='short'"
