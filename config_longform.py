"""
config_longform.py — Configuration for the daily "Did You Know" 16:9 long-form pipeline.

Separate from config.py so that the Shorts pipeline is never affected.
"""

# ── Duration & Resolution ─────────────────────────────────────────────────────
LONGFORM_TARGET_AUDIO_DURATION = (150, 180)   # 2.5–3 min of speech
LONGFORM_RESOLUTION = (1920, 1080)            # 16:9 Landscape
LONGFORM_FPS = 30

# ── Content Structure ─────────────────────────────────────────────────────────
LONGFORM_NUM_TOPICS = 5                       # 5 facts per compilation
LONGFORM_PER_TOPIC_DURATION = (25, 35)        # Seconds per individual fact
LONGFORM_INTRO_DURATION = 5                   # 5s intro hook
LONGFORM_OUTRO_DURATION = 10                  # 10s CTA outro

# ── Upload Schedule ───────────────────────────────────────────────────────────
LONGFORM_UPLOAD_TIME = "04:30"                # 10:00 AM IST = 04:30 UTC

# ── Audio ─────────────────────────────────────────────────────────────────────
LONGFORM_BGM_VOLUME = 0.05                    # Lower BGM for 3-min content
LONGFORM_WORD_COUNT_TARGET = (420, 500)       # ~150 WPM for 3 min

# ── Retry Logic ───────────────────────────────────────────────────────────────
LONGFORM_MAX_RETRY_ATTEMPTS = 8               # More retries since 5 topics
LONGFORM_PER_TOPIC_RETRIES = 3                # Retries per individual topic

# ── Transition Effects ────────────────────────────────────────────────────────
LONGFORM_TRANSITION_DURATION = 0.5            # Seconds between facts
LONGFORM_TRANSITION_STYLES = ["glitch", "zoom", "slide", "fade"]
