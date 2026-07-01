import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
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

# Cloudflare Workers AI
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")

# Text Generation Models (LLMs)
CLOUDFLARE_TEXT_MODELS = [
    # Latest flagship models
    "@cf/meta/llama-3.3-70b-instruct",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/meta/llama-3.1-70b-instruct",
    "@cf/meta/llama-3.1-8b-instruct",
    "@cf/meta/llama-3.1-8b-instruct-fp8",
    "@cf/meta/llama-3.1-8b-instruct-fast",
    "@cf/meta/llama-3.2-1b-instruct",
    "@cf/meta/llama-3.2-3b-instruct",
    "@cf/meta/llama-3.2-11b-vision-instruct",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/meta/llama-guard-3-8b",
    
    # Qwen models
    "@cf/qwen/qwen2.5-72b-instruct",
    "@cf/qwen/qwen2.5-coder-32b-instruct",
    "@cf/qwen/qwq-32b",
    "@cf/qwen/qwen3-30b-a3b-fp8",
    
    # Mistral models
    "@cf/mistral/mistral-small-3.1-24b-instruct",
    
    # Google Gemma
    "@cf/google/gemma-4-26b-a4b-it",
    
    # Z.ai GLM
    "@cf/zai-org/glm-4.7-flash",
    "@cf/zai-org/glm-5.2",
    
    # Moonshot Kimi
    "@cf/moonshotai/kimi-k2.5",
    "@cf/moonshotai/kimi-k2.6",
    "@cf/moonshotai/kimi-k2.7-code",
    
    # NVIDIA Nemotron
    "@cf/nvidia/nemotron-3-120b-a12b",
    
    # DeepSeek
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    
    # OpenAI
    "@cf/openai/gpt-oss-120b",
    "@cf/openai/gpt-oss-20b",
    
    # Others
    "@cf/baai/bge-reranker-base",
    "@cf/ibm-granite/granite-4.0-h-micro",
    "@cf/aisingapore/gemma-sea-lion-v4-27b-it",
    "@cf/nousresearch/hermes-2-pro-mistral-7b",
    "@cf/defog/sqlcoder-7b-2",
    "@cf/microsoft/phi-2",
    "@cf/meta/bart-large-cnn",
    "@cf/meta/m2m100-1.2b",
]

# Text-to-Image Models
CLOUDFLARE_TEXT_TO_IMAGE_MODELS = [
    "@cf/blackforestlabs/flux-2-dev",
    "@cf/blackforestlabs/flux-2-klein-4b",
    "@cf/blackforestlabs/flux-2-klein-9b",
    "@cf/blackforestlabs/flux-1-schnell",
    "@cf/runwayml/stable-diffusion-v1-5-img2img",
    "@cf/runwayml/stable-diffusion-v1-5-inpainting",
    "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "@cf/bytedance/stable-diffusion-xl-lightning",
    "@cf/lykon/dreamshaper-8-lcm",
    "@cf/leonardo/lucid-origin",
    "@cf/leonardo/phoenix-1.0",
]

# Text-to-Video Models (if available)
CLOUDFLARE_TEXT_TO_VIDEO_MODELS = [
    # Note: Cloudflare Workers AI currently doesn't have native text-to-video models
]

# Image-to-Text / Vision Models
CLOUDFLARE_IMAGE_TO_TEXT_MODELS = [
    "@cf/meta/llama-3.2-11b-vision-instruct",
    "@cf/llava-hf/llava-1.5-7b-hf",
    "@cf/unum/uform-gen2-qwen-500m",
]

# Text-to-Speech Models
CLOUDFLARE_TEXT_TO_SPEECH_MODELS = [
    "@cf/deepgram/aura-1",
    "@cf/deepgram/aura-2-en",
    "@cf/deepgram/aura-2-es",
    "@cf/myshell/melotts",
]

# Automatic Speech Recognition (Speech-to-Text)
CLOUDFLARE_SPEECH_TO_TEXT_MODELS = [
    "@cf/openai/whisper",
    "@cf/openai/whisper-large-v3-turbo",
    "@cf/openai/whisper-tiny-en",
    "@cf/deepgram/nova-3",
    "@cf/deepgram/flux",
]

# Embedding Models
CLOUDFLARE_EMBEDDING_MODELS = [
    "@cf/baai/bge-large-en-v1.5",
    "@cf/baai/bge-base-en-v1.5",
    "@cf/baai/bge-small-en-v1.5",
    "@cf/baai/bge-m3",
    "@cf/google/embeddinggemma-300m",
    "@cf/qwen/qwen3-embedding-0.6b",
    "@cf/pfnet/plamo-embedding-1b",
]

# Translation Models
CLOUDFLARE_TRANSLATION_MODELS = [
    "@cf/ai4bharat/indictrans2-en-indic-1b",
    "@cf/meta/m2m100-1.2b",
]

# Classification / Other Models
CLOUDFLARE_CLASSIFICATION_MODELS = [
    "@cf/huggingface/distilbert-sst-2-int8",
    "@cf/meta/detr-resnet-50",
    "@cf/microsoft/resnet-50",
]

# Voice Activity Detection
CLOUDFLARE_VAD_MODELS = [
    "@cf/pipecat/smart-turn-v2",
]

# Legacy/Deprecated models (kept for compatibility)
CLOUDFLARE_LEGACY_MODELS = [
    "@cf/meta/llama-2-7b-chat-fp16",
    "@cf/meta/llama-2-7b-chat-int8",
    "@cf/meta/llama-2-7b-chat-hf-lora",
    "@cf/meta/llama-3-8b-instruct",
    "@cf/meta/llama-3-8b-instruct-awq",
    "@cf/meta/meta-llama-3-8b-instruct",
    "@cf/meta/llama-3.1-8b-instruct-awq",
    "@cf/mistral/mistral-7b-instruct-v0.2-lora",
]

# Default model list for fallback (prioritized by capability)
CLOUDFLARE_MODELS = [
    # Top tier - best for reasoning/complex tasks
    "@cf/meta/llama-3.3-70b-instruct",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/zai-org/glm-4.7-flash",
    "@cf/openai/gpt-oss-120b",
    "@cf/nvidia/nemotron-3-120b-a12b",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/qwen/qwq-32b",
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    
    # High quality - good balance
    "@cf/meta/llama-3.1-70b-instruct",
    "@cf/qwen/qwen2.5-72b-instruct",
    "@cf/qwen/qwen3-30b-a3b-fp8",
    "@cf/qwen/qwen2.5-coder-32b-instruct",
    "@cf/mistral/mistral-small-3.1-24b-instruct",
    "@cf/google/gemma-4-26b-a4b-it",
    "@cf/moonshotai/kimi-k2.7-code",
    "@cf/moonshotai/kimi-k2.6",
    
    # Fast/Efficient
    "@cf/meta/llama-3.1-8b-instruct",
    "@cf/meta/llama-3.1-8b-instruct-fp8",
    "@cf/meta/llama-3.1-8b-instruct-fast",
    "@cf/meta/llama-3.2-3b-instruct",
    "@cf/zai-org/glm-5.2",
    
    # Lightweight
    "@cf/meta/llama-3.2-1b-instruct",
    "@cf/openai/gpt-oss-20b",
    "@cf/ibm-granite/granite-4.0-h-micro",
]

# All models combined for reference
CLOUDFLARE_ALL_MODELS = (
    CLOUDFLARE_TEXT_MODELS + 
    CLOUDFLARE_TEXT_TO_IMAGE_MODELS + 
    CLOUDFLARE_TEXT_TO_VIDEO_MODELS + 
    CLOUDFLARE_IMAGE_TO_TEXT_MODELS + 
    CLOUDFLARE_TEXT_TO_SPEECH_MODELS + 
    CLOUDFLARE_SPEECH_TO_TEXT_MODELS + 
    CLOUDFLARE_EMBEDDING_MODELS + 
    CLOUDFLARE_TRANSLATION_MODELS + 
    CLOUDFLARE_CLASSIFICATION_MODELS + 
    CLOUDFLARE_VAD_MODELS
)

# Model Configurations (to easily switch/override via env variables or custom values)
GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
GEMINI_FLASH_LITE_MODEL = os.getenv("GEMINI_FLASH_LITE_MODEL", "gemini-2.5-flash-lite")

# API Call Spacing Delay to prevent rate-limiting on Free Tier keys
GEMINI_RPM_SLEEP = float(os.getenv("GEMINI_RPM_SLEEP", "2.0"))
