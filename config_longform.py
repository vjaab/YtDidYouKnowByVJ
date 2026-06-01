"""
config_longform.py — Configuration for the daily "Did You Know" 16:9 long-form pipeline.

Separate from config.py so that the Shorts pipeline is never affected.

VIRAL RETENTION OVERHAUL (2026-06-01):
  - Scaled from 3-min / 5-fact → 8-min / 10-fact format
  - Enables mid-roll ads (8+ min threshold)
  - Matches top-performing "Did You Know" channels (Bright Side, BE AMAZED)
"""
import os

# ── Directory Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LONGFORM_TRACKER_FILE = os.path.join(BASE_DIR, "longform_news_log.json")

# ── Duration & Resolution ─────────────────────────────────────────────────────
LONGFORM_TARGET_AUDIO_DURATION = (420, 500)   # 7–8.5 min of speech (mid-roll eligible)
LONGFORM_RESOLUTION = (1920, 1080)            # 16:9 Landscape
LONGFORM_FPS = 30

# ── Content Structure ─────────────────────────────────────────────────────────
LONGFORM_NUM_TOPICS = 10                      # 10 facts per compilation (doubled)
LONGFORM_PER_TOPIC_DURATION = (35, 50)        # Seconds per individual fact (deeper)
LONGFORM_INTRO_DURATION = 15                  # 15s intro hook (cold open + intro)
LONGFORM_OUTRO_DURATION = 20                  # 20s CTA outro (end screen safe zone)

# ── Upload Schedule ───────────────────────────────────────────────────────────
LONGFORM_UPLOAD_TIME = "04:30"                # 10:00 AM IST = 04:30 UTC

# ── Audio ─────────────────────────────────────────────────────────────────────
LONGFORM_BGM_VOLUME = 0.09                    # Atmospheric BGM for 8-min content
LONGFORM_WORD_COUNT_TARGET = (1100, 1350)     # ~150 WPM for 8 min
LONGFORM_BGM_INTENSITY_RAMP = True            # BGM volume ramps up for last 3 facts

# ── Retry Logic ───────────────────────────────────────────────────────────────
LONGFORM_MAX_RETRY_ATTEMPTS = 12              # More retries since 10 topics
LONGFORM_PER_TOPIC_RETRIES = 3                # Retries per individual topic

# ── Transition Effects ────────────────────────────────────────────────────────
LONGFORM_TRANSITION_DURATION = 0.8            # Seconds between facts (cinematic)
LONGFORM_TRANSITION_STYLES = ["glitch", "zoom", "slide", "fade", "wipe", "shatter"]

# ── Retention Engineering ─────────────────────────────────────────────────────
LONGFORM_PATTERN_INTERRUPT_INTERVAL = 30      # Seconds between pattern interrupts
LONGFORM_VISUAL_HOLD_MAX = 2.5                # Max seconds before forced visual change
LONGFORM_RECAP_EVERY_N_FACTS = 3              # Recap bumper every N facts
LONGFORM_COLD_OPEN_DURATION = 15              # Cold open teaser before intro (seconds)
LONGFORM_MIDPOINT_TWIST_FACT = 5              # Fact number where midpoint twist occurs

# ── Shorts Cross-Promotion ───────────────────────────────────────────────────
LONGFORM_GENERATE_SHORTS_TEASER = True        # Auto-generate 60s Shorts teaser
LONGFORM_SHORTS_TEASER_DURATION = (50, 58)    # Shorts teaser duration range
