import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemini_script_longform import (
    TOPIC_DISCOVERY_SINGLE_TEMPLATE,
    TOPIC_DISCOVERY_MULTI_TEMPLATE,
    CHAPTERED_SCRIPT_TEMPLATE,
)
from main_longform import format_longform_description


class TestLongformGrowthPipeline(unittest.TestCase):
    def test_topic_discovery_templates_include_mass_appeal(self):
        self.assertIn("MASS APPEAL & HIGH-REACH PRIORITIZATION", TOPIC_DISCOVERY_SINGLE_TEMPLATE)
        self.assertIn("Personal Impact Factor", TOPIC_DISCOVERY_SINGLE_TEMPLATE)
        self.assertIn("MASS APPEAL & HIGH-REACH PRIORITIZATION", TOPIC_DISCOVERY_MULTI_TEMPLATE)
        self.assertIn("personal stakes", TOPIC_DISCOVERY_SINGLE_TEMPLATE.lower())

    def test_chaptered_script_template_rules(self):
        self.assertIn("NAME THE SPECIFIC TOOL/MODEL EARLY", CHAPTERED_SCRIPT_TEMPLATE)
        self.assertIn("OPEN-LOOP BRIDGES", CHAPTERED_SCRIPT_TEMPLATE)
        self.assertIn("Micro-Cliffhanger Bridges", CHAPTERED_SCRIPT_TEMPLATE)
        self.assertIn("ELI5", CHAPTERED_SCRIPT_TEMPLATE)

    def test_longform_description_formats_pinned_discussion(self):
        dummy_script = {
            "title": "Why Senior Engineers Are Quitting Old Tools",
            "description": "A deep dive into next-gen developer tooling.",
            "comment_hook": "What's the one developer tool you refuse to give up? Drop it below!",
            "chapters": [
                {"chapter_number": 1, "chapter_title": "The Shift", "approx_start_seconds": 0},
                {"chapter_number": 2, "chapter_title": "The Real Problem", "approx_start_seconds": 60},
            ]
        }
        desc = format_longform_description(dummy_script, ["#AI", "#Coding"])
        self.assertIn("PINNED DISCUSSION", desc)
        self.assertIn("What's the one developer tool you refuse to give up?", desc)
        self.assertIn("CHAPTERS:", desc)
