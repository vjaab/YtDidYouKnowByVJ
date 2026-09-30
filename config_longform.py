"""
config_longform.py — Configuration for the 16:9 long-form pipeline.

CHAPTERED DEEP-DIVE FORMAT (2026-07):
  - Replaced 8-topic compilation with chaptered deep-dive (Fireship/MKBHD/Johnny Harris hybrid)
  - Duration is strictly 2 to 3 mins (120-180 seconds, 280-420 words)
  - Topic depth rotates weekly: 3 days multi-story, 4 days single deep story
  - Compact visual clip count (fixes memory/SIGTERM)
"""
import os
from datetime import datetime

# ── Directory Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LONGFORM_TRACKER_FILE = os.path.join(BASE_DIR, "longform_news_log.json")

# ── Duration & Resolution ─────────────────────────────────────────────────────
# STRICT: 2 to 3 mins duration only (120 to 180 seconds)
LONGFORM_TARGET_AUDIO_DURATION = (120, 180)  # 2-3 min (120-180 seconds)
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
    """Returns 'multi' to always pick 2-3 thematically linked topics."""
    return "multi"

LONGFORM_MAX_CHAPTERS = 3                     # Max chapters per deep-dive (fits 2-3 min runtime)
LONGFORM_VISUAL_BEATS_PER_CHAPTER = 4         # Max visual beats per chapter (caps clip count)
LONGFORM_WORD_COUNT_TARGET = (280, 420)       # Strictly 2-3 min at 140 WPM (120-180 seconds)
LONGFORM_FORMAT = "chaptered"                 # The format flag for downstream branching

# Legacy aliases kept for backward compat in video_gen.py branch checks
LONGFORM_NUM_TOPICS = 1                       # Default (overridden by depth mode)
LONGFORM_PER_TOPIC_DURATION = (10, 15)        # Unused but kept for imports

# ── Upload Schedule ───────────────────────────────────────────────────────────
LONGFORM_UPLOAD_TIME = "13:30"                # 07:00 PM IST = 13:30 UTC (Global peak tech window)

# ── Audio ─────────────────────────────────────────────────────────────────────
LONGFORM_BGM_VOLUME = 0.09                    # Atmospheric BGM
LONGFORM_BGM_INTENSITY_RAMP = True            # BGM volume ramps in final chapter

# ── Retry Logic ───────────────────────────────────────────────────────────────
LONGFORM_MAX_RETRY_ATTEMPTS = 5               # Reduced from 8 — timeout fixes make fewer retries sufficient
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
LONGFORM_GENERATE_SHORTS_TEASER = False       # Disabled — do not upload Shorts of longform videos
LONGFORM_SHORTS_TEASER_DURATION = (50, 58)    # Shorts teaser duration range
