import unittest
from config_longform import (
    LONGFORM_TARGET_AUDIO_DURATION,
    LONGFORM_WORD_COUNT_TARGET,
    LONGFORM_MAX_CHAPTERS,
    LONGFORM_VISUAL_BEATS_PER_CHAPTER,
)
from gemini_script_longform import (
    SYSTEM_PERSONA_LONGFORM,
    TOPIC_DISCOVERY_SINGLE_TEMPLATE,
    RESEARCH_TEMPLATE,
    CHAPTERED_SCRIPT_TEMPLATE,
)

class TestLongformDuration(unittest.TestCase):
    def test_audio_duration_range(self):
        """Longform audio duration must strictly be 2 to 3 minutes (120 to 180 seconds)."""
        min_dur, max_dur = LONGFORM_TARGET_AUDIO_DURATION
        self.assertEqual(min_dur, 120, "Minimum audio duration should be 120 seconds (2 mins)")
        self.assertEqual(max_dur, 180, "Maximum audio duration should be 180 seconds (3 mins)")

    def test_word_count_target(self):
        """Longform word count target must map to 2-3 minutes at 140 WPM."""
        min_words, max_words = LONGFORM_WORD_COUNT_TARGET
        self.assertGreaterEqual(min_words, 260)
        self.assertLessEqual(min_words, 300)
        self.assertGreaterEqual(max_words, 400)
        self.assertLessEqual(max_words, 450)
        # Expected duration at 140 WPM (2.33 words/sec)
        min_expected = min_words / 2.33
        max_expected = max_words / 2.33
        self.assertAlmostEqual(min_expected, 120, delta=10)
        self.assertAlmostEqual(max_expected, 180, delta=10)

    def test_max_chapters_and_visual_beats(self):
        """Chapters and beats must be scaled for a 2-3 minute video."""
        self.assertLessEqual(LONGFORM_MAX_CHAPTERS, 3)
        self.assertLessEqual(LONGFORM_VISUAL_BEATS_PER_CHAPTER, 5)

    def test_prompt_constraints_mention_2_to_3_mins(self):
        """Prompt templates must explicitly enforce 2 to 3 minutes."""
        self.assertIn("2 to 3 minutes", SYSTEM_PERSONA_LONGFORM)
        self.assertIn("2 to 3 minute", TOPIC_DISCOVERY_SINGLE_TEMPLATE)
        self.assertIn("2 to 3 minutes", RESEARCH_TEMPLATE)
        self.assertIn("2 TO 3 MINUTES ONLY", CHAPTERED_SCRIPT_TEMPLATE)

if __name__ == "__main__":
    unittest.main()
