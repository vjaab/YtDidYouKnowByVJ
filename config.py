import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
YOUTUBE_CLIENT_SECRET_FILE = os.getenv("YOUTUBE_CLIENT_SECRET_FILE", "client_secret.json")
VEO_MODEL_ID = "veo-3.1-generate-preview"

# X.com (Twitter) API Credentials
X_API_KEY = os.getenv("X_API_KEY", "")
X_API_SECRET = os.getenv("X_API_SECRET", "")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET", "")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "")

# Trending Engine API Keys (Phase 1)
YOUTUBE_DATA_API_KEY = os.getenv("YOUTUBE_DATA_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")


# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")
MUSIC_DIR = os.path.join(ASSETS_DIR, "music")
SFX_DIR = os.path.join(ASSETS_DIR, "sfx")
TRACKER_FILE = os.path.join(BASE_DIR, "news_log.json")

# Create required directories
for d in [OUTPUT_DIR, LOGS_DIR, FONTS_DIR, MUSIC_DIR, SFX_DIR]:
    os.makedirs(d, exist_ok=True)

# Application Settings
TIMEZONE = "Asia/Kolkata"
UPLOAD_TIMES = ["08:30", "20:00"]  # 2/day schedule (Morning and Evening IST)
MAX_RETRY_ATTEMPTS = 10
SIMILARITY_THRESHOLD = 75
CATEGORY_COOLDOWN_DAYS = 3
BGM_VOLUME = 0.07
TARGET_AUDIO_DURATION = (15, 35) # Shorter Shorts = higher completion rates = more algorithmic push

# Global Feature Flags
ENABLE_LONGFORM = False
ENABLE_TRENDING_ENGINE = True    # Phase 1: YouTube/Reddit/GitHub trending aggregation

# Engagement & Retention Pillars (Production Spec 2026)
ENABLE_KINETIC_CAPTIONS = True
ENABLE_AUDIO_DUCKING = True
ENABLE_PERIODIC_CUTS = True
ENABLE_EVIDENCE_SCREENSHOTS = True
ENABLE_HORMOZI_STYLING = True

# Retention Engine Settings (Phase 3 & 4)
VISUAL_CUT_TARGET_SECONDS = 2.0   # Target visual change frequency (was ~4s, now 2s)
ENABLE_CINEMATIC_TRANSITIONS = True  # Whip pan, zoom punch, flash cut, glitch
ENABLE_STRATEGIC_SFX = True       # Whoosh/bass at pattern interrupts
ENABLE_DYNAMIC_BGM_CURVE = True   # BGM energy follows Hook→Body→Payoff→CTA
TRENDING_NICHE_BIAS = 0.15         # 0=prefer broad topics, 1=prefer niche topics

# Model Configurations (to easily switch/override via env variables or custom values)
GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
GEMINI_FLASH_LITE_MODEL = os.getenv("GEMINI_FLASH_LITE_MODEL", "gemini-2.5-flash-lite")

# API Call Spacing Delay to prevent rate-limiting on Free Tier keys
GEMINI_RPM_SLEEP = float(os.getenv("GEMINI_RPM_SLEEP", "2.0"))
