import unittest
import os
import json
import tempfile
import sys

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SOURCE_SEQUENCE, TRENDING_SOURCES
from topic_tracker import (
    get_next_topic_source,
    record_topic_source,
    detect_source_from_url,
    record_story,
    load_tracker
)
from trending_engine import compute_engagement_score


class TestSequentialSourceRotation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.tracker_file = os.path.join(self.test_dir.name, "test_news_log.json")
        self.state_file = os.path.join(self.test_dir.name, "test_source_tracker.json")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_source_sequence_configuration(self):
        self.assertEqual(SOURCE_SEQUENCE, ["github", "medium", "huggingface_hub"])
        self.assertIn("github", TRENDING_SOURCES)
        self.assertIn("medium", TRENDING_SOURCES)
        self.assertIn("huggingface_hub", TRENDING_SOURCES)

    def test_detect_source_from_url(self):
        self.assertEqual(detect_source_from_url("https://github.com/torvalds/linux"), "github")
        self.assertEqual(detect_source_from_url("https://medium.com/@author/great-ai-article"), "medium")
        self.assertEqual(detect_source_from_url("https://towardsdatascience.com/fast-rag"), "medium")
        self.assertEqual(detect_source_from_url("https://towardsdev.com/tools-for-devs"), "medium")
        self.assertEqual(detect_source_from_url("https://huggingface.co/models/deepseek-ai"), "huggingface_hub")
        self.assertEqual(detect_source_from_url("https://huggingface.co/datasets/wikipedia"), "huggingface_hub")
        self.assertIsNone(detect_source_from_url("https://example.com/other-news"))

    def test_cold_start_defaults_to_github(self):
        next_source = get_next_topic_source(tracker_file=self.tracker_file, state_file=self.state_file)
        self.assertEqual(next_source, "github")

    def test_sequential_rotation_cycle(self):
        # 1. Start from empty -> next is github
        src1 = get_next_topic_source(tracker_file=self.tracker_file, state_file=self.state_file)
        self.assertEqual(src1, "github")

        # Record github in state file
        record_topic_source("github", state_file=self.state_file)

        # 2. After github -> next should be medium
        src2 = get_next_topic_source(tracker_file=self.tracker_file, state_file=self.state_file)
        self.assertEqual(src2, "medium")

        # Record medium in state file
        record_topic_source("medium", state_file=self.state_file)

        # 3. After medium -> next should be huggingface_hub
        src3 = get_next_topic_source(tracker_file=self.tracker_file, state_file=self.state_file)
        self.assertEqual(src3, "huggingface_hub")

        # Record huggingface_hub in state file
        record_topic_source("huggingface_hub", state_file=self.state_file)

        # 4. After huggingface_hub -> wraps around to github
        src4 = get_next_topic_source(tracker_file=self.tracker_file, state_file=self.state_file)
        self.assertEqual(src4, "github")

    def test_rotation_from_news_log_history(self):
        # Simulate an existing news_log.json with huggingface_hub as last entry
        initial_tracker = {
            "used_titles": [],
            "history": [
                {
                    "date": "2026-09-20",
                    "title": "A Cool Tool",
                    "news_source_url": "https://github.com/foo/bar",
                    "topic_source": "github"
                },
                {
                    "date": "2026-09-21",
                    "title": "Medium Tech Article",
                    "news_source_url": "https://medium.com/@dev/post",
                    "topic_source": "medium"
                },
                {
                    "date": "2026-09-22",
                    "title": "HF Model Launch",
                    "news_source_url": "https://huggingface.co/org/model",
                    "topic_source": "huggingface_hub"
                }
            ]
        }
        with open(self.tracker_file, "w") as f:
            json.dump(initial_tracker, f)

        # State file doesn't exist yet, should infer from history that last was huggingface_hub -> next is github
        next_source = get_next_topic_source(tracker_file=self.tracker_file, state_file=self.state_file)
        self.assertEqual(next_source, "github")

    def test_record_story_stores_topic_source(self):
        record_story(
            title="Medium AI Story",
            news_headline="Big AI Update",
            subcategory="AI Tools",
            companies=["OpenAI"],
            keywords=["AI"],
            breaking_news_level=8,
            voice_used="en-US-GuyNeural",
            youtube_url="https://youtube.com/shorts/123",
            news_source_url="https://medium.com/@dev/big-ai-update",
            topic_type="tools",
            topic_source="medium",
            tracker_file=self.tracker_file
        )

        tracker = load_tracker(self.tracker_file)
        history = tracker.get("history", [])
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].get("topic_source"), "medium")

    def test_medium_engagement_score(self):
        med_art = {
            "title": "Medium: Future of AI",
            "type": "medium_rss",
            "_engagement": {"curated_score": 35, "tag": "artificial-intelligence"}
        }
        score = compute_engagement_score(med_art)
        self.assertGreaterEqual(score, 60)


if __name__ == "__main__":
    unittest.main()
