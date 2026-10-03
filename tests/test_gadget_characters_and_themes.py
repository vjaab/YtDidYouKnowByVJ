import unittest
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cartoon_dialogue_engine import (
    VALID_SPEAKERS,
    CHARACTER_METADATA,
    CAROUSEL_THEMES,
    get_character_image_path,
    resolve_dialogue_characters,
    resolve_carousel_theme,
)


class TestGadgetCharactersAndThemes(unittest.TestCase):
    def test_all_gadgets_registered(self):
        expected_gadgets = [
            "phone", "computer", "watch", "chip", "camera",
            "earbuds", "battery", "vr", "server", "controller", "drone"
        ]
        for gadget in expected_gadgets:
            self.assertIn(gadget, VALID_SPEAKERS, f"Gadget {gadget} should be in VALID_SPEAKERS")
            self.assertIn(gadget, CHARACTER_METADATA, f"Gadget {gadget} should be in CHARACTER_METADATA")
            meta = CHARACTER_METADATA[gadget]
            self.assertTrue(len(meta["tag"]) > 0)
            self.assertTrue(len(meta["emoji"]) > 0)

    def test_asha_removed(self):
        self.assertNotIn("asha", VALID_SPEAKERS, "Asha should not be in VALID_SPEAKERS")
        self.assertNotIn("asha", CHARACTER_METADATA, "Asha should not be in CHARACTER_METADATA")

    def test_character_sprite_existence(self):
        for speaker in ["phone", "computer", "watch", "chip", "camera", "earbuds", "battery", "vr", "server", "controller", "drone"]:
            path = get_character_image_path(speaker, "neutral")
            self.assertIsNotNone(path, f"Sprite for {speaker} neutral should exist")
            self.assertTrue(path.exists(), f"Sprite file {path} should exist on disk")

    def test_themes_exist(self):
        expected_themes = [
            "monochrome_noir", "paper_editorial", "crimson_ember",
            "emerald_terminal", "luxury_gold", "arctic_frost", "tokyo_midnight",
            "neon_cyber", "cyber_matrix", "tech_blueprint", "amber_solaris",
            "synthwave_plum", "swiss_minimal"
        ]
        for theme_id in expected_themes:
            self.assertIn(theme_id, CAROUSEL_THEMES, f"Theme {theme_id} must be defined in CAROUSEL_THEMES")
            theme = resolve_carousel_theme(theme_id)
            self.assertIn("bg_gradient", theme)
            self.assertIn("card_bg", theme)
            self.assertIn("accent_primary", theme)
            self.assertIn("cta_bg", theme)

    def test_resolve_dialogue_characters_explicit(self):
        c1, c2 = resolve_dialogue_characters("vj_phone")
        self.assertEqual(c1, "phone")
        self.assertEqual(c2, "vj")

        c1, c2 = resolve_dialogue_characters("phone_vj")
        self.assertEqual(c1, "phone")
        self.assertEqual(c2, "vj")

    def test_resolve_dialogue_characters_topic_keyword(self):
        c1, c2 = resolve_dialogue_characters("auto", topic="Smartphone lithium battery optimization")
        self.assertIn(c1, ["battery", "phone"])
        self.assertEqual(c2, "vj")


if __name__ == "__main__":
    unittest.main()
