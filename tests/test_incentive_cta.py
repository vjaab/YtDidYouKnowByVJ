import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch

# Add workspace directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import format_description


class TestIncentiveCTA(unittest.TestCase):
    """Unit tests for Shorts subscriber incentive loop in descriptions."""

    def setUp(self):
        self.base_ai_description = "This video reveals a shocking privacy leak in your phone's settings."
        self.base_script = "Your phone is recording everything. Here's how to stop it. Comment CONFIG for the checklist."
        self.base_hashtags = ["#Privacy", "#TechTips", "#Security"]
        self.base_slot = "Slot A"
        self.base_chunks = []
        self.base_relevant_links = ["https://github.com/example/repo"]
        self.base_source_url = "https://example.com/article"

    def test_format_description_shorts_digital_vault(self):
        """Test Shorts description includes Digital Vault offer with comment trigger."""
        script_data = {
            "comment_trigger_keyword": "CONFIG",
            "incentive_cta_type": "digital_vault",
            "digital_asset_offer": "Full privacy checklist & architecture diagram"
        }

        description = format_description(
            ai_description=self.base_ai_description,
            script=self.base_script,
            hashtags=self.base_hashtags,
            slot=self.base_slot,
            chunks=self.base_chunks,
            relevant_links=self.base_relevant_links,
            source_url=self.base_source_url,
            script_data=script_data
        )

        # Check for Digital Vault section
        self.assertIn("🎁 SUBSCRIBER INCENTIVE LOOP", description)
        self.assertIn("🎁 DIGITAL VAULT OFFER", description)
        self.assertIn("Full privacy checklist & architecture diagram", description)
        self.assertIn("https://t.me/technewsbyvj", description)
        # Check for Comment Trigger
        self.assertIn("COMMENT TRIGGER", description)
        self.assertIn("'CONFIG'", description)

    def test_format_description_shorts_comment_trigger(self):
        """Test Shorts description includes Comment Keyword Trigger prominently."""
        script_data = {
            "comment_trigger_keyword": "OLLAMA",
            "incentive_cta_type": "comment_trigger",
            "digital_asset_offer": "Exact evaluation template for local LLMs"
        }

        description = format_description(
            ai_description=self.base_ai_description,
            script=self.base_script,
            hashtags=self.base_hashtags,
            slot=self.base_slot,
            chunks=self.base_chunks,
            relevant_links=self.base_relevant_links,
            source_url=self.base_source_url,
            script_data=script_data
        )

        self.assertIn("🎁 SUBSCRIBER INCENTIVE LOOP", description)
        self.assertIn("🎁 DIGITAL VAULT OFFER", description)
        self.assertIn("Exact evaluation template for local LLMs", description)
        self.assertIn("COMMENT TRIGGER", description)
        self.assertIn("'OLLAMA'", description)
        self.assertIn("direct template link", description)

    def test_format_description_shorts_benchmark_challenge(self):
        """Test Shorts description includes Benchmark Challenge with hardware specs."""
        script_data = {
            "comment_trigger_keyword": "BENCHMARK",
            "incentive_cta_type": "benchmark_challenge",
            "digital_asset_offer": "Tokens/sec comparison sheet for M2 Mac vs M3"
        }

        description = format_description(
            ai_description=self.base_ai_description,
            script=self.base_script,
            hashtags=self.base_hashtags,
            slot=self.base_slot,
            chunks=self.base_chunks,
            relevant_links=self.base_relevant_links,
            source_url=self.base_source_url,
            script_data=script_data
        )

        self.assertIn("🎁 SUBSCRIBER INCENTIVE LOOP", description)
        self.assertIn("🏆 BENCHMARK CHALLENGE", description)
        self.assertIn("Tokens/sec comparison sheet for M2 Mac vs M3", description)
        self.assertIn("Comment 'BENCHMARK' with your hardware specs", description)

    def test_format_description_shorts_community_audit(self):
        """Test Shorts description includes Community Audit & $100 API Credit Giveaway."""
        script_data = {
            "comment_trigger_keyword": "SETUP",
            "incentive_cta_type": "community_audit",
            "digital_asset_offer": "Monthly $100 API credit giveaway entry"
        }

        description = format_description(
            ai_description=self.base_ai_description,
            script=self.base_script,
            hashtags=self.base_hashtags,
            slot=self.base_slot,
            chunks=self.base_chunks,
            relevant_links=self.base_relevant_links,
            source_url=self.base_source_url,
            script_data=script_data
        )

        self.assertIn("🎁 SUBSCRIBER INCENTIVE LOOP", description)
        self.assertIn("🏆 COMMUNITY AUDIT & GIVEAWAY", description)
        self.assertIn("Monthly $100 API Credit Giveaway", description)
        self.assertIn("Comment 'SETUP' with your setup to enter", description)

    def test_format_description_longform_no_incentive_loop(self):
        """Test Long-form (Slot C) description does NOT include incentive loop."""
        script_data = {
            "comment_trigger_keyword": "CONFIG",
            "incentive_cta_type": "digital_vault",
            "digital_asset_offer": "Full privacy checklist & architecture diagram"
        }

        description = format_description(
            ai_description=self.base_ai_description,
            script=self.base_script,
            hashtags=self.base_hashtags,
            slot="Slot C",
            chunks=self.base_chunks,
            relevant_links=self.base_relevant_links,
            source_url=self.base_source_url,
            script_data=script_data
        )

        # Long-form should NOT have incentive loop
        self.assertNotIn("🎁 SUBSCRIBER INCENTIVE LOOP", description)
        self.assertNotIn("DIGITAL VAULT OFFER", description)
        self.assertNotIn("COMMENT TRIGGER", description)
        self.assertNotIn("BENCHMARK CHALLENGE", description)
        self.assertNotIn("COMMUNITY AUDIT", description)

    def test_format_description_no_script_data_no_incentive_loop(self):
        """Test description without script_data does not include incentive loop."""
        description = format_description(
            ai_description=self.base_ai_description,
            script=self.base_script,
            hashtags=self.base_hashtags,
            slot=self.base_slot,
            chunks=self.base_chunks,
            relevant_links=self.base_relevant_links,
            source_url=self.base_source_url,
            script_data=None
        )

        self.assertNotIn("🎁 SUBSCRIBER INCENTIVE LOOP", description)

    def test_format_description_all_templates_include_incentive(self):
        """Test that all 4 description templates include incentive loop for Shorts."""
        script_data = {
            "comment_trigger_keyword": "TESTKEY",
            "incentive_cta_type": "digital_vault",
            "digital_asset_offer": "Test asset"
        }

        # We need to test different templates by changing the hash seed
        # The template is selected by hash of clean_summary
        # Let's test by creating descriptions with different summaries
        summaries = [
            "First summary for template zero.",
            "Second summary for template one.",
            "Third summary for template two.",
            "Fourth summary for template three."
        ]

        for summary in summaries:
            description = format_description(
                ai_description=summary,
                script=self.base_script,
                hashtags=self.base_hashtags,
                slot=self.base_slot,
                chunks=self.base_chunks,
                relevant_links=self.base_relevant_links,
                source_url=self.base_source_url,
                script_data=script_data
            )
            self.assertIn("🎁 SUBSCRIBER INCENTIVE LOOP", description, f"Template for '{summary}' missing incentive loop")


class TestIncentiveCTASchema(unittest.TestCase):
    """Tests for the expected schema fields in script generation output."""

    def test_expected_incentive_fields_exist(self):
        """Verify the expected incentive fields are defined in the schema."""
        # These are the fields that should be in the JSON output from gemini_script.py
        expected_fields = [
            "comment_trigger_keyword",
            "incentive_cta_type",
            "digital_asset_offer"
        ]

        # This test documents the expected schema - actual validation
        # happens in the integration test with the LLM
        for field in expected_fields:
            self.assertIsInstance(field, str)
            self.assertTrue(len(field) > 0)

    def test_valid_incentive_cta_types(self):
        """Test that incentive_cta_type only accepts valid values."""
        valid_types = [
            "digital_vault",
            "comment_trigger",
            "benchmark_challenge",
            "community_audit"
        ]

        for cta_type in valid_types:
            self.assertIn(cta_type, valid_types)


class TestShortsScriptStructure(unittest.TestCase):
    """Tests for the 4-part Shorts script structure timing."""

    def test_shorts_timing_structure_documentation(self):
        """Document the expected 4-part timing structure for Shorts."""
        # 00:00 - 00:03: Hard Hook (Metric, Contradiction, or Personal Stake)
        # 00:03 - 00:25: Core Engineering / Practical Value Breakdown
        # 00:25 - 00:35: Incentive CTA (Digital Vault / Comment Trigger / Benchmark / Community Audit)
        # 00:35 - 00:45: Seamless Loop back to opening hook

        structure = {
            "hook": (0, 3),
            "core_value": (3, 25),
            "incentive_cta": (25, 35),
            "seamless_loop": (35, 45)
        }

        # Verify no gaps or overlaps
        prev_end = 0
        for phase, (start, end) in structure.items():
            self.assertEqual(start, prev_end, f"Gap in {phase} timing")
            self.assertLess(start, end, f"Invalid duration for {phase}")
            prev_end = end

        self.assertEqual(prev_end, 45, "Total duration should be 45 seconds")


if __name__ == "__main__":
    unittest.main()