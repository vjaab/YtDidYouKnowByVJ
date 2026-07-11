"""
config_longform.py — Configuration for the 16:9 long-form pipeline.

CHAPTERED DEEP-DIVE FORMAT (2026-07):
  - Replaced 8-topic compilation with chaptered deep-dive (Fireship/MKBHD/Johnny Harris hybrid)
  - Duration is depth-driven, not target-driven (5-25 min flexible)
  - Topic depth rotates weekly: 3 days multi-story, 4 days single deep story
  - Reduces visual clip count from ~200 to ~25 (fixes memory/SIGTERM)
"""
import os
from datetime import datetime

# ── Directory Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LONGFORM_TRACKER_FILE = os.path.join(BASE_DIR, "longform_news_log.json")

# ── Duration & Resolution ─────────────────────────────────────────────────────
# FLEXIBLE: Let the story dictate runtime.
# A single rich story earns 15-25 min. A thinner but solid story gets 5-8 min.
# Mid-roll ads kick in at 8+ min, so aim there when depth supports it.
LONGFORM_TARGET_AUDIO_DURATION = (300, 1500)  # 5-25 min flexible range
LONGFORM_RESOLUTION = (1920, 1080)            # 16:9 Landscape
LONGFORM_FPS = 30

# ── Content Structure ─────────────────────────────────────────────────────────
# Topic depth rotation: determines whether today's video is a single deep-dive
# or 2-3 thematically linked stories. Based on day of week.
#   Mon/Wed/Fri (3 days): multi-story (2-3 thematically linked)
#   Tue/Thu/Sat/Sun (4 days): single deep story
LONGFORM_DEPTH_SCHEDULE = {
    0: "multi",   # Monday
    1: "single",  # Tuesday
    2: "multi",   # Wednesday
    3: "single",  # Thursday
    4: "multi",   # Friday
    5: "single",  # Saturday
    6: "single",  # Sunday
}

def get_topic_depth_mode():
    """Returns 'single' or 'multi' based on today's day of the week."""
    day = datetime.now().weekday()
    return LONGFORM_DEPTH_SCHEDULE.get(day, "single")

LONGFORM_MAX_CHAPTERS = 5                     # Max chapters per deep-dive
LONGFORM_VISUAL_BEATS_PER_CHAPTER = 6         # Max visual beats per chapter (caps clip count)
LONGFORM_WORD_COUNT_TARGET = (700, 3500)      # ~5-25 min at 140 WPM
LONGFORM_FORMAT = "chaptered"                 # The format flag for downstream branching

# Legacy aliases kept for backward compat in video_gen.py branch checks
LONGFORM_NUM_TOPICS = 1                       # Default (overridden by depth mode)
LONGFORM_PER_TOPIC_DURATION = (10, 15)        # Unused but kept for imports

# ── Upload Schedule ───────────────────────────────────────────────────────────
LONGFORM_UPLOAD_TIME = "04:30"                # 10:00 AM IST = 04:30 UTC

# ── Audio ─────────────────────────────────────────────────────────────────────
LONGFORM_BGM_VOLUME = 0.09                    # Atmospheric BGM
LONGFORM_BGM_INTENSITY_RAMP = True            # BGM volume ramps in final chapter

# ── Retry Logic ───────────────────────────────────────────────────────────────
LONGFORM_MAX_RETRY_ATTEMPTS = 8               # Fewer retries needed with 1-3 topics
LONGFORM_PER_TOPIC_RETRIES = 3                # Retries per individual topic

# ── Transition Effects ────────────────────────────────────────────────────────
LONGFORM_TRANSITION_DURATION = 0.8            # Seconds between chapters (cinematic)
LONGFORM_TRANSITION_STYLES = ["glitch", "zoom", "slide", "fade", "wipe", "shatter"]

# ── Retention Engineering ─────────────────────────────────────────────────────
LONGFORM_PATTERN_INTERRUPT_INTERVAL = 30      # Seconds between pattern interrupts
LONGFORM_VISUAL_HOLD_MAX = 2.5                # Max seconds before forced visual change
LONGFORM_COLD_OPEN_DURATION = 15              # Cold open teaser before first chapter (seconds)

# Legacy aliases for any remaining references
LONGFORM_RECAP_EVERY_N_FACTS = 3
LONGFORM_MIDPOINT_TWIST_FACT = 3
LONGFORM_INTRO_DURATION = 5
LONGFORM_OUTRO_DURATION = 5

# ── Shorts Cross-Promotion ───────────────────────────────────────────────────
LONGFORM_GENERATE_SHORTS_TEASER = True        # Auto-generate 60s Shorts teaser
LONGFORM_SHORTS_TEASER_DURATION = (50, 58)    # Shorts teaser duration range
