"""
layout_engine.py — Layout Engine for deterministic visual variety.
Extracts layout/profile generation logic from video_gen.py for modularity and testing.
"""

import os
import random
import hashlib as _hashlib
from typing import Dict, Any, Optional, List, Tuple

# Import configuration
from config import ASSETS_DIR, ENABLE_LAYOUT_VARIATION, FORCE_LAYOUT_TYPE

# Layout type constants
VALID_LAYOUT_TYPES = ["split_screen", "hero_center", "asymmetric", "side_strip", "top_center", "corner_cycling"]

# Palette definitions (matching original video_gen.py)
PALETTES = [
    {
        "name": "Cyberpunk Neon",
        "highlight": (255, 0, 128, 255),
        "normal": (255, 255, 255, 255),
        "box_fill": (15, 5, 25, 220),
        "accent": (255, 0, 128),
        "font_file": "Montserrat-ExtraBold.ttf"
    },
    {
        "name": "Electric Cyan",
        "highlight": (0, 230, 255, 255),
        "normal": (240, 240, 240, 255),
        "box_fill": (10, 15, 25, 220),
        "accent": (0, 230, 255),
        "font_file": "Montserrat-Bold.ttf"
    },
    {
        "name": "Dark Mode Terminal",
        "highlight": (50, 255, 50, 255),
        "normal": (200, 240, 200, 255),
        "box_fill": (5, 12, 5, 230),
        "accent": (50, 255, 50),
        "font_file": "Roboto-Bold.ttf"
    },
    {
        "name": "Clean Minimalist Gold",
        "highlight": (255, 215, 0, 255),
        "normal": (255, 255, 255, 255),
        "box_fill": (20, 20, 20, 215),
        "accent": (255, 215, 0),
        "font_file": "Montserrat-Bold.ttf"
    },
    {
        "name": "Retro Synthwave",
        "highlight": (255, 110, 0, 255),
        "normal": (255, 250, 220, 255),
        "box_fill": (20, 10, 30, 220),
        "accent": (255, 110, 0),
        "font_file": "Montserrat-ExtraBold.ttf"
    }
]

# Legacy default theme
DEFAULT_THEME = {
    "name": "Legacy Cyan/Yellow",
    "highlight": (204, 255, 0, 255),
    "normal": (255, 255, 255, 255),
    "box_fill": (0, 0, 0, 215),
    "accent": (255, 214, 0),
    "font_file": "Montserrat-Bold.ttf"
}

# Layout bounds per type
LAYOUT_BOUNDS = {
    "split_screen": {"image_y_start_pct": 0.42, "image_height_pct": 0.58},
    "hero_center": {"image_y_start_pct": 0.20, "image_height_pct": 0.60},
    "asymmetric": {"image_y_start_pct": 0.0, "image_height_pct": 1.0},
    "side_strip": {"image_y_start_pct": 0.0, "image_height_pct": 1.0},
    "top_center": {"image_y_start_pct": 0.15, "image_height_pct": 0.70},
    "corner_cycling": {"image_y_start_pct": 0.0, "image_height_pct": 1.0},
}

# CTA configuration
CTA_PILL_COLORS = [
    (255, 214, 0),      # Gold
    (0, 200, 255),      # Cyan
    (255, 100, 100),    # Coral
    (180, 130, 255),    # Purple
]

CTA_HEADLINES = [
    "Full {topic} guide + source code",
    "Get the complete {topic} breakdown",
    "{topic} implementation playbook",
    "Deep dive: {topic} explained",
]

CTA_DESCRIPTIONS = [
    "Join the community 🚀",
    "Free access — link in bio 📥",
    "Grab it before it's gone ⚡",
    "Level up your stack 🔧",
]

PARTICLE_STYLES = ["bokeh", "digital", "stars", "digital_rain", "lens_dust"]
GRADIENT_POSITIONS = ["bottom", "top"]
PROGRESS_BAR_POSITIONS = ["bottom", "top"]


class LayoutEngine:
    """
    Deterministic layout/profile generator for video visual variety.
    
    Generates unique but reproducible layout profiles based on content hash,
    breaking 'template fingerprint' that triggers YouTube's Inauthentic Content policy.
    """
    
    def __init__(
        self,
        enable_variation: bool = None,
        force_layout: str = None,
        seed_salt: str = ""
    ):
        self.enable_variation = enable_variation or (os.environ.get("ENABLE_LAYOUT_VARIATION", "0") == "1")
        self.force_layout = force_layout or os.environ.get("FORCE_LAYOUT_TYPE")
        self.seed_salt = seed_salt
        
        # Validate force_layout
        if self.force_layout and self.force_layout not in VALID_LAYOUT_TYPES:
            print(f"⚠️ Invalid FORCE_LAYOUT_TYPE: {self.force_layout}. Ignoring.")
            self.force_layout = None
    
    def generate_profile(
        self,
        seed_string: str,
        dominant_color: Optional[Tuple[int, int, int]] = None,
        layout_type: Optional[str] = None,
        daily_layout: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete layout profile for a video.
        
        Args:
            seed_string: Unique string to seed the RNG (e.g., headline + date)
            dominant_color: Optional RGB tuple from screenshot to tint theme
            layout_type: Override layout type (overrides daily/random)
            daily_layout: Layout from daily schedule (ecosystem_logic)
            
        Returns:
            Complete profile dict with layout, theme, and visual parameters
        """
        # Deterministic seed from content
        full_seed = f"{seed_string}{self.seed_salt}"
        seed = int(_hashlib.md5(full_seed.encode('utf-8')).hexdigest(), 16)
        rng = random.Random(seed)
        
        # Resolve layout type priority: explicit > force > daily > random
        if layout_type and layout_type in VALID_LAYOUT_TYPES:
            resolved_layout = layout_type
        elif self.force_layout and self.force_layout in VALID_LAYOUT_TYPES:
            resolved_layout = self.force_layout
        elif daily_layout and daily_layout in VALID_LAYOUT_TYPES:
            resolved_layout = daily_layout
        elif self.enable_variation:
            resolved_layout = rng.choice(VALID_LAYOUT_TYPES)
        else:
            resolved_layout = "asymmetric"  # Legacy default
        
        # Get layout bounds
        bounds = LAYOUT_BOUNDS.get(resolved_layout, LAYOUT_BOUNDS["asymmetric"])
        
        # Select theme
        theme = self._select_theme(rng, dominant_color)
        
        # Select CTA variant
        cta_variant = rng.randint(0, 3)
        cta_pill_color = CTA_PILL_COLORS[cta_variant]
        cta_headline_template = CTA_HEADLINES[cta_variant]
        cta_description = CTA_DESCRIPTIONS[cta_variant]
        
        # Generate visual parameters
        gradient_height_pct = rng.uniform(0.40, 0.50)
        gradient_position = rng.choice(GRADIENT_POSITIONS)
        title_bottom_gap = rng.randint(165, 220)
        particle_style = rng.choice(PARTICLE_STYLES)
        progress_bar_height = rng.randint(4, 8)
        progress_bar_position = rng.choice(PROGRESS_BAR_POSITIONS)
        hook_transition_time = rng.uniform(3.5, 5.0)
        avatar_x_offset = rng.randint(-60, 60)
        subtitle_y_jitter = rng.randint(-30, 30)
        
        profile = {
            "layout_type": resolved_layout,
            "theme": theme,
            "image_y_start_pct": bounds["image_y_start_pct"],
            "image_height_pct": bounds["image_height_pct"],
            "gradient_height_pct": gradient_height_pct,
            "gradient_position": gradient_position,
            "title_bottom_gap": title_bottom_gap,
            "particle_style": particle_style,
            "progress_bar_height": progress_bar_height,
            "progress_bar_position": progress_bar_position,
            "hook_transition_time": hook_transition_time,
            "avatar_x_offset": avatar_x_offset,
            "subtitle_y_jitter": subtitle_y_jitter,
            "cta_pill_color": cta_pill_color,
            "cta_headline_template": cta_headline_template,
            "cta_description": cta_description,
        }
        
        print(f"🎲 Layout Profile: type={resolved_layout}, theme={theme['name']}, "
              f"gradient={gradient_position}@{gradient_height_pct:.0%}, "
              f"particles={particle_style}, progress={progress_bar_position}@{progress_bar_height}px, "
              f"cta_variant={cta_variant}")
        return profile
    
    def _select_theme(self, rng: random.Random, dominant_color: Optional[Tuple[int, int, int]]) -> Dict:
        """Select and optionally customize a color theme."""
        if not self.enable_variation:
            theme = DEFAULT_THEME.copy()
        else:
            palette_idx = rng.randint(0, len(PALETTES) - 1)
            theme = PALETTES[palette_idx].copy()
        
        # Apply dominant color tinting
        if dominant_color:
            theme["accent"] = dominant_color
            r, g, b = dominant_color
            brightness = (0.299 * r + 0.587 * g + 0.114 * b)
            if brightness < 200:
                vibrant_color = tuple(min(255, int(c * 1.5)) for c in dominant_color)
                theme["highlight"] = (*vibrant_color, 255)
            else:
                theme["highlight"] = (*dominant_color, 255)
        
        return theme
    
    @staticmethod
    def get_valid_layout_types() -> List[str]:
        """Return list of valid layout types."""
        return VALID_LAYOUT_TYPES.copy()
    
    @staticmethod
    def validate_profile(profile: Dict) -> bool:
        """Validate a layout profile has all required fields."""
        required = [
            "layout_type", "theme", "image_y_start_pct", "image_height_pct",
            "gradient_height_pct", "gradient_position", "title_bottom_gap",
            "particle_style", "progress_bar_height", "progress_bar_position",
            "hook_transition_time", "avatar_x_offset", "subtitle_y_jitter",
            "cta_pill_color", "cta_headline_template", "cta_description"
        ]
        return all(key in profile for key in required)


# Convenience function for backward compatibility
def generate_layout_profile(
    seed_string: str,
    dominant_color: Optional[Tuple[int, int, int]] = None,
    enable_variation: bool = None,
    force_layout: str = None,
    daily_layout: Optional[str] = None,
    seed_salt: str = ""
) -> Dict[str, Any]:
    """Backward-compatible function matching original video_gen.py signature."""
    engine = LayoutEngine(
        enable_variation=enable_variation,
        force_layout=force_layout,
        seed_salt=seed_salt
    )
    return engine.generate_profile(
        seed_string=seed_string,
        dominant_color=dominant_color,
        daily_layout=daily_layout
    )


# Standalone test
if __name__ == "__main__":
    # Test deterministic generation
    engine = LayoutEngine(enable_variation=True, seed_salt="test_v1")
    
    test_topics = [
        "AI Tool Spotlight: Local LLMs",
        "GitHub Repo You Should Know - Day 1",
        "Free Alternative to Claude Code",
        "Stop Paying for This AI Tool",
        "This AI Saves Developers Hours"
    ]
    
    print("=" * 60)
    print("LAYOUT ENGINE TEST")
    print("=" * 60)
    
    for topic in test_topics:
        profile = engine.generate_profile(
            seed_string=topic,
            dominant_color=(0, 200, 255)
        )
        print(f"\nTopic: {topic}")
        print(f"  Layout: {profile['layout_type']}")
        print(f"  Theme: {profile['theme']['name']}")
        print(f"  Particles: {profile['particle_style']}")
        print(f"  CTA: {profile['cta_headline_template']}")
    
    print("\n" + "=" * 60)
    print("Determinism test (same seed should produce same result):")
    p1 = engine.generate_profile("Test Topic")
    p2 = engine.generate_profile("Test Topic")
    print(f"  Profile 1 layout: {p1['layout_type']}, theme: {p1['theme']['name']}")
    print(f"  Profile 2 layout: {p2['layout_type']}, theme: {p2['theme']['name']}")
    print(f"  Match: {p1['layout_type'] == p2['layout_type'] and p1['theme']['name'] == p2['theme']['name']}")