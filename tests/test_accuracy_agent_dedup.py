import pytest
import os
import sys
import json
from unittest.mock import patch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cartoon_dialogue_engine import fetch_or_select_did_you_know_fact
from content_accuracy_agent import verify_content_accuracy, scan_for_script_artifacts
from ai_news_carousel import is_topic_unique


def test_fetch_or_select_does_not_pollute_trackers_when_record_false():
    """Verify that fetch_or_select_did_you_know_fact with record=False does not write to tracker."""
    with patch("telegram_approval_handler.record_topic_in_tracker") as mock_record_yt, \
         patch("ai_news_carousel.record_carousel_topic") as mock_record_c:
        fact = fetch_or_select_did_you_know_fact(platform="facebook", record=False)
        assert fact is not None
        assert "title" in fact
        mock_record_yt.assert_not_called()
        mock_record_c.assert_not_called()


def test_uniqueness_check_passes_on_selected_fact():
    """Verify that selected fact is not considered a duplicate of itself."""
    fact = fetch_or_select_did_you_know_fact(platform="facebook", record=False)
    title = fact["title"]
    # Check uniqueness
    unique, reason = is_topic_unique(title)
    # The selected unseen seed fact must be unique against current tracker
    assert unique is True, f"Expected unique=True for freshly selected fact, but got {reason}"


def test_verify_content_accuracy_reads_bubble_text():
    """Verify that verify_content_accuracy correctly parses bubble dialogue in slides."""
    carousel_data = {
        "headline": "A Single GPU Chip Contains More Transistors Than Stars in the Milky Way",
        "hook": "Did You Know GPUs Have More Transistors Than Stars? ⭐",
        "takeaway": "Modern semiconductor lithography packs billions of transistors onto tiny silicon wafers.",
        "slides": [
            {"speaker": "byte", "emotion": "curious", "title": "Stargazing Chips ⭐", "bubble": "Did you know a single GPU has more transistors than stars in our galaxy?"},
            {"speaker": "phone", "emotion": "shocked", "title": "Milky Way Math 🌌", "bubble": "The Milky Way has around 100 to 400 billion stars."},
            {"speaker": "byte", "emotion": "thinking", "title": "Silicon Megastructure 🔬", "bubble": "Nvidia's Blackwell GPU packs an unbelievable 208 billion transistors on one chip!"},
            {"speaker": "phone", "emotion": "excited", "title": "Atomic Engineering ⚡", "bubble": "Each transistor is just nanometers across—smaller than a strand of human DNA."},
            {"speaker": "byte", "emotion": "smug", "title": "Everyday Magic 💡", "bubble": "Follow @vijayakumarj_ai for daily mind-blowing technology facts!", "is_takeaway": True}
        ]
    }
    # Heuristic / deterministic scan should pass
    audit = verify_content_accuracy(carousel_data["headline"], carousel_data, caption="Fascinating GPU facts!")
    assert audit["artifact_free"] is True
    assert "reason" in audit


def test_scan_for_script_artifacts():
    """Verify that forbidden artifacts like [pause], [break], (continue) are caught."""
    clean_texts = ["Hello world", "Did you know that GPUs are fast?"]
    clean, msg = scan_for_script_artifacts(clean_texts)
    assert clean is True

    dirty_texts = ["Hello world [pause] wait for it"]
    clean, msg = scan_for_script_artifacts(dirty_texts)
    assert clean is False
    assert "pause" in msg
