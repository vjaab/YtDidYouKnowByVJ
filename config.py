import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
YOUTUBE_CLIENT_SECRET_FILE = os.getenv("YOUTUBE_CLIENT_SECRET_FILE", "client_secret.json")

# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")
MUSIC_DIR = os.path.join(ASSETS_DIR, "music")
TRACKER_FILE = os.path.join(BASE_DIR, "news_log.json")

# Create required directories
for d in [OUTPUT_DIR, LOGS_DIR, FONTS_DIR, MUSIC_DIR]:
    os.makedirs(d, exist_ok=True)

# Application Settings
TIMEZONE = "Asia/Kolkata"
UPLOAD_TIMES = ["04:00", "13:30"] # Slot A and B (Shorts only)
MAX_RETRY_ATTEMPTS = 5
SIMILARITY_THRESHOLD = 75
CATEGORY_COOLDOWN_DAYS = 3
BGM_VOLUME = 0.07
TARGET_AUDIO_DURATION = (38, 44) # Optimized for replay-ability (Algorithmic Spec 2026)

# Engagement & Retention Pillars (Production Spec 2026)
ENABLE_KINETIC_CAPTIONS = True
ENABLE_AUDIO_DUCKING = True
ENABLE_PERIODIC_CUTS = True
ENABLE_EVIDENCE_SCREENSHOTS = True
ENABLE_HORMOZI_STYLING = True
