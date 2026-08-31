"""
schemas.py — Pydantic models for type-safe data structures across the pipeline.
Replaces ad-hoc dicts with validated, documented schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class LayoutType(str, Enum):
    SPLIT_SCREEN = "split_screen"
    HERO_CENTER = "hero_center"
    ASYMMETRIC = "asymmetric"
    SIDE_STRIP = "side_strip"
    TOP_CENTER = "top_center"
    CORNER_CYCLING = "corner_cycling"


class VisualType(str, Enum):
    VIDEO = "Video"
    AI_IMAGE = "AI Image"
    WHITEBOARD = "Whiteboard"
    INFOGRAPHIC = "Infographic"
    DIAGRAM = "Diagram"
    ANIMATED_UI_MOCKUP = "Animated UI Mockup"
    CODE_SNIPPET = "Code Snippet"
    SCREEN_RECORDING = "Screen Recording"
    FLOWCHART = "Flowchart"
    TERMINAL_OUTPUT = "Terminal Output"
    GITHUB_UI = "GitHub UI"
    SIDE_BY_SIDE_COMPARISON = "Side-by-side Comparison"
    ARCHITECTURE_DIAGRAM = "Architecture Diagram"


class IncentiveCTAType(str, Enum):
    DIGITAL_VAULT = "digital_vault"
    COMMENT_TRIGGER = "comment_trigger"
    BENCHMARK_CHALLENGE = "benchmark_challenge"
    COMMUNITY_AUDIT = "community_audit"
    SAVE_TRIGGER = "save_trigger"
    SHARE_TRIGGER = "share_trigger"


class TopicType(str, Enum):
    RESEARCH = "research"
    TOOLS = "tools"
    NEWS = "news"
    TECH_TRENDS = "tech_trends"
    VAIBHAV = "vaibhav"
    INTERVIEW_QUESTIONS = "interview_questions"
    QUIZ = "quiz"


class SlotType(str, Enum):
    SLOT_A = "Slot A (Discovery)"
    SLOT_B = "Slot B (Deep Dive)"
    SLOT_C = "Slot C (Longform)"
    SLOT_L = "Slot L (Technical/Deep Dive)"


class Platform(str, Enum):
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    X = "x"
    TIKTOK = "tiktok"
    LINKEDIN = "linkedin"


# ═══════════════════════════════════════════════════════════════════════
# CORE MODELS
# ═══════════════════════════════════════════════════════════════════════

class Theme(BaseModel):
    """Color theme configuration."""
    model_config = ConfigDict(extra="allow")
    
    name: str
    highlight: tuple[int, int, int, int]  # RGBA
    normal: tuple[int, int, int, int]
    box_fill: tuple[int, int, int, int]
    accent: tuple[int, int, int]
    font_file: str


class LayoutProfile(BaseModel):
    """Complete layout profile for a video."""
    model_config = ConfigDict(extra="allow")
    
    layout_type: LayoutType
    theme: Theme
    image_y_start_pct: float
    image_height_pct: float
    gradient_height_pct: float
    gradient_position: Literal["bottom", "top"]
    title_bottom_gap: int
    particle_style: str
    progress_bar_height: int
    progress_bar_position: Literal["bottom", "top"]
    hook_transition_time: float
    avatar_x_offset: int
    subtitle_y_jitter: int
    cta_pill_color: tuple[int, int, int]
    cta_headline_template: str
    cta_description: str


class WordTimestamp(BaseModel):
    """Individual word with timing from stable-ts/Whisper."""
    word: str
    start: float
    end: float


class Chunk(BaseModel):
    """Visual chunk aligned to audio timestamps."""
    model_config = ConfigDict(extra="allow")
    
    chunk_id: int
    text: str
    start: float
    end: float
    duration: float
    words: List[WordTimestamp] = []
    
    # Visual metadata from nano_scene_gen
    visual_type: Optional[VisualType] = None
    nano_visual_prompt: Optional[str] = None
    visual_path: Optional[str] = None
    scene_objective: Optional[str] = None
    on_screen_elements: List[str] = []
    camera_motion: Optional[str] = None
    transition: Optional[str] = None
    
    # Infographic data
    has_infographic: bool = False
    infographic_type: Optional[str] = None
    infographic_data: Optional[Dict] = None
    
    # Settings mockup
    is_setting_chunk: bool = False
    
    # Render style metadata
    render_style: Optional[str] = None
    overlay_elements: List[str] = []
    camera_motion: Optional[str] = None
    relevance_score: Optional[int] = None


class ScriptData(BaseModel):
    """Complete script data from gemini_script pipeline."""
    model_config = ConfigDict(extra="allow")
    
    # Core identification
    title: str
    hook: str
    hook_text: str
    script: str
    summary: str
    sub_category: str
    
    # Visual structure
    chunks: List[Chunk] = []
    subtitle_chunks: List[Dict] = []
    
    # Retention engineering
    retention_map: Optional[Dict] = None
    retention_cues: List[Dict] = []
    
    # Incentive/CTA
    incentive_cta_type: IncentiveCTAType
    comment_trigger_keyword: str
    digital_asset_offer: str
    comment_hook: str
    
    # Series/episodic
    series_name: Optional[str] = None
    episode_number: Optional[int] = None
    
    # Metadata
    keywords: List[str] = []
    hashtags: List[str] = []
    companies_mentioned: List[str] = []
    people: List[str] = []
    relevant_links: List[str] = []
    
    # Source tracking
    original_news_headline: str
    original_news_url: str
    use_case_evidence_url: Optional[str] = None
    
    # Editorial compliance
    editorial_perspective: Optional[str] = None
    editorial_angle: Optional[str] = None
    content_fingerprint: Optional[str] = None
    
    # Timestamps
    fact_timestamps: List[Dict] = []
    word_timestamps: List[WordTimestamp] = []
    
    # Production flags
    is_longform: bool = False
    breaking_news_level: int = 5
    
    # Screenshot paths
    screenshot_path: Optional[str] = None
    evidence_screenshot_path: Optional[str] = None
    is_github_readme: Optional[bool] = None
    
    # Output
    output_suffix: Optional[str] = None
    color_theme: Optional[Dict] = None


class VideoMetadata(BaseModel):
    """Metadata for video upload."""
    model_config = ConfigDict(extra="allow")
    
    title: str
    description: str
    tags: List[str]
    hashtags: List[str]
    thumbnail_path: Optional[str] = None
    category_id: str = "28"  # Science & Technology
    privacy_status: Literal["public", "private", "unlisted"] = "public"
    made_for_kids: bool = False
    publish_at: Optional[datetime] = None


class PlatformCaption(BaseModel):
    """Platform-specific caption data."""
    platform: Platform
    caption: str
    hashtags: List[str]
    first_comment: Optional[str] = None  # For Facebook/LinkedIn
    link_in_bio: bool = False


class PipelineRun(BaseModel):
    """Single pipeline execution record."""
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    topic_type: TopicType
    slot: SlotType
    category: str
    run_index: int
    status: Literal["running", "completed", "failed"] = "running"
    
    # Outputs
    script_data: Optional[ScriptData] = None
    video_path: Optional[str] = None
    youtube_url: Optional[str] = None
    youtube_video_id: Optional[str] = None
    
    # Cross-platform
    instagram_post_id: Optional[str] = None
    facebook_post_id: Optional[str] = None
    x_post_id: Optional[str] = None
    
    # Metrics (populated later)
    views_24h: Optional[int] = None
    retention_rate: Optional[float] = None
    ctr: Optional[float] = None
    
    # Errors
    error_message: Optional[str] = None
    error_stage: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def validate_script_data(data: Dict) -> ScriptData:
    """Validate and parse raw script data from Gemini."""
    return ScriptData.model_validate(data)


def validate_layout_profile(data: Dict) -> LayoutProfile:
    """Validate layout profile."""
    return LayoutProfile.model_validate(data)


def validate_chunk(data: Dict) -> Chunk:
    """Validate single chunk."""
    return Chunk.model_validate(data)


def validate_chunks(data: List[Dict]) -> List[Chunk]:
    """Validate list of chunks."""
    return [Chunk.model_validate(c) for c in data]


# ═══════════════════════════════════════════════════════════════════════
# SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════

def script_data_to_json(script: ScriptData, indent: int = 2) -> str:
    """Serialize ScriptData to JSON string."""
    return script.model_dump_json(indent=indent)


def script_data_from_json(json_str: str) -> ScriptData:
    """Deserialize ScriptData from JSON string."""
    return ScriptData.model_validate_json(json_str)


def chunks_to_json(chunks: List[Chunk], indent: int = 2) -> str:
    """Serialize chunks to JSON."""
    return json.dumps([c.model_dump() for c in chunks], indent=indent)


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test ScriptData validation
    test_script = {
        "title": "Test Video",
        "hook": "You're using AI wrong",
        "hook_text": "You're using AI wrong",
        "script": "Test script content",
        "summary": "Test summary",
        "sub_category": "AI & Tech Tools",
        "chunks": [
            {
                "chunk_id": 1,
                "text": "First chunk",
                "start": 0.0,
                "end": 3.0,
                "duration": 3.0,
                "visual_type": "AI Image",
                "nano_visual_prompt": "AI robot coding"
            }
        ],
        "incentive_cta_type": "comment_trigger",
        "comment_trigger_keyword": "AI",
        "digital_asset_offer": "AI prompt pack",
        "comment_hook": "What do you think?",
        "original_news_headline": "Test headline",
        "original_news_url": "https://example.com",
        "keywords": ["AI", "coding"],
        "hashtags": ["#AI", "#Coding"],
        "companies_mentioned": ["OpenAI"],
        "people": ["Sam Altman"],
        "relevant_links": ["https://github.com"],
        "fact_timestamps": [],
    }
    
    print("Testing ScriptData validation...")
    script = validate_script_data(test_script)
    print(f"✅ Validated: {script.title}")
    print(f"   Chunks: {len(script.chunks)}")
    print(f"   CTA Type: {script.incentive_cta_type}")
    print(f"   Hook: {script.hook}")
    
    # Test LayoutProfile
    test_layout = {
        "layout_type": "split_screen",
        "theme": {
            "name": "Test Theme",
            "highlight": (255, 0, 128, 255),
            "normal": (255, 255, 255, 255),
            "box_fill": (15, 5, 25, 220),
            "accent": (255, 0, 128),
            "font_file": "Montserrat-ExtraBold.ttf"
        },
        "image_y_start_pct": 0.42,
        "image_height_pct": 0.58,
        "gradient_height_pct": 0.45,
        "gradient_position": "bottom",
        "title_bottom_gap": 180,
        "particle_style": "bokeh",
        "progress_bar_height": 6,
        "progress_bar_position": "bottom",
        "hook_transition_time": 4.0,
        "avatar_x_offset": 10,
        "subtitle_y_jitter": 5,
        "cta_pill_color": (255, 214, 0),
        "cta_headline_template": "Full {topic} guide + source code",
        "cta_description": "Join the community 🚀"
    }
    
    print("\nTesting LayoutProfile validation...")
    layout = validate_layout_profile(test_layout)
    print(f"✅ Validated: {layout.layout_type}")
    print(f"   Theme: {layout.theme.name}")
    
    # Test serialization
    print("\nTesting JSON serialization...")
    json_str = script_data_to_json(script)
    print(f"✅ Serialized to JSON ({len(json_str)} chars)")
    
    parsed = script_data_from_json(json_str)
    print(f"✅ Deserialized: {parsed.title}")
    
    print("\n✅ All schema tests passed!")