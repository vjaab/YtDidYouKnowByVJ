"""
config_longform.py — Configuration for the 16:9 long-form pipeline in Vaibhav Sisinty format.

Separate from config.py so that the Shorts pipeline is never affected.

VAIBHAV SISINTY FORMAT OVERHAUL (2026-06-28):
  - Transitioned from 10-fact compilation to: 1 Main Topic Deep Dive + News Roundup updates
  - Enables mid-roll ads (8+ min threshold)
  - Follows high-utility, tactical explainer style
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
LONGFORM_NUM_TOPICS = 8                       # 1 Main Topic + 7 news updates
LONGFORM_PER_TOPIC_DURATION = (10, 15)        # Seconds per individual fact/segment (deeper)
LONGFORM_INTRO_DURATION = 5                   # 5s intro hook (cold open + intro)
LONGFORM_OUTRO_DURATION = 5                   # 5s CTA outro (end screen safe zone)
LONGFORM_WORD_COUNT_TARGET = (1100, 1350)     # ~150 WPM for 8 min
LONGFORM_TARGET_AUDIO_DURATION = (480, 540)   # 8-9 minutes target duration (mid-roll eligible)

# ── Upload Schedule ───────────────────────────────────────────────────────────
LONGFORM_UPLOAD_TIME = "04:30"                # 10:00 AM IST = 04:30 UTC

# ── Audio ─────────────────────────────────────────────────────────────────────
LONGFORM_BGM_VOLUME = 0.09                    # Atmospheric BGM for 8-min content
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
