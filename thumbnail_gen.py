"""
thumbnail_gen.py — Premium Design System Thumbnail Generator (2026 Spec).

Design System:
  - Dark-mode high contrast: Deep Charcoal/Obsidian backgrounds
  - Template-based: Architecture (A), Cost Optimization (B), Security (C)
  - 3-Zone Layout: Hook Text | Hero Graphic | Code/Data Badge
  - Typography: Montserrat Black/ExtraBold + Fira Code/JetBrains Mono
  - Max 2-4 words, accent highlighting on keywords
  - Personal Authority: Avatar with emotion overlay
  - A/B Testing: 3 variants with quality scoring
"""

import os
import io
import math
import random
import textwrap
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from datetime import datetime
from google import genai
from google.genai import types
from rembg import remove
from config import OUTPUT_DIR, ASSETS_DIR, GEMINI_API_KEY, GEMINI_FLASH_MODEL
import cv2
import requests
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict
from enum import Enum

THUMB_W, THUMB_H = 1280, 720
SHORTS_W, SHORTS_H = 1080, 1920

# ── DESIGN SYSTEM: COLOR PALETTE ───────────────────────────────────────────────
class AccentPalette(Enum):
    """Design system accent colors per content type."""
    # Electric Cyan - Cloud, Web Architecture, APIs
    ELECTRIC_CYAN = ("#00F2FE", "#00D2FF", "cyan")
    # Neon Green - Cost Cutting, Free Tools, Efficiency  
    NEON_GREEN = ("#00FF87", "#00E676", "green")
    # Amber/Solar Yellow - Warnings, Security, Comparisons
    AMBER = ("#FFB703", "#FFD166", "amber")
    # Hot Magenta/Crimson - Stop Using, Critical Vulns, Controversial
    HOT_MAGENTA = ("#FF2A6D", "#FF0055", "magenta")
    # Neon Violet & Amber - VS Battles, Head-to-Head Duel, Showdowns
    VS_VIOLET = ("#A855F7", "#FFB703", "violet_amber")
    # Solar Gold & Ember - Paradigm Leaps, AGI Breakthroughs, Game Changers
    SOLAR_GOLD = ("#FFD166", "#FF6B35", "gold_ember")
    # Cyber Lime & Electric Sky - Developer Benchmarks, IDEs, Speed Tests
    CYBER_EMERALD = ("#00FF87", "#38BDF8", "emerald_sky")
    # Ultraviolet & Blood Crimson - Tech Exposés, Leaked Dossiers, Deep Dives
    ULTRAVIOLET_CRIMSON = ("#FF0055", "#7928CA", "crimson_uv")

# Background colors
BG_DEEP_CHARCOAL = (15, 23, 42)      # #0F172A
BG_OBSIDIAN = (9, 13, 22)            # #090D16
BG_PURE_BLACK = (10, 10, 15)         # #0A0A0F
BG_DARK_INDIGO = (13, 10, 26)        # #0D0A1A
BG_CARBON_GREEN = (8, 18, 14)        # #08120E
BG_BLOOD_NAVY = (18, 10, 15)         # #120A0F

# Text colors
TEXT_WHITE = (255, 255, 255)
TEXT_OFF_WHITE = (245, 245, 250)

# ── TEMPLATE TYPES ─────────────────────────────────────────────────────────────
class ThumbnailTemplate(Enum):
    ARCHITECTURE_SHIFT = "architecture"      # Template A: Tech shift, "RAG IS DEAD"
    COST_OPTIMIZATION = "cost"               # Template B: Savings, "$0 AI STACK"
    SECURITY_WARNING = "security"            # Template C: Alerts, "DATA LEAK"
    VS_SHOWDOWN = "vs_battle"                # Template D: Head-to-head duel, "DEEPSEEK VS NVIDIA"
    BREAKTHROUGH = "breakthrough"            # Template E: Paradigm leap, "AGI BREAKTHROUGH"
    DEV_BENCHMARK = "dev_benchmark"          # Template F: Speed/accuracy evals, "98.4% ACCURACY"
    DEEP_DIVE = "deep_dive"                  # Template G: Investigative exposé, "TOP SECRET LEAK"

# ── A/B TEST CONFIG ──────────────────────────────────────────────────────────
THUMBNAIL_VARIANTS = 3
MAX_TEXT_WORDS = 4
RULE_OF_THIRDS_GRID = True
HIGH_CONTRAST_RATIO = 4.5  # WCAG AA

# ── FACE DETECTION & EMOTION CONFIG ──────────────────────────────────────────
FACE_DETECTION_CONFIDENCE = 0.7
EMOTION_OVERLAYS = {
    "shocked": "😱",
    "curious": "🤔", 
    "excited": "🤯",
    "warning": "⚠️",
    "mind_blown": "💥"
}
EMOTION_WEIGHTS = {
    "security": "warning",
    "privacy": "warning",
    "breaking": "shocked",
    "secret": "shocked",
    "revealed": "mind_blown",
    "ai": "excited",
    "launch": "excited",
    "new": "curious",
    "how": "curious",
    "why": "curious",
    "cost": "excited",
    "free": "excited",
    "save": "excited",
    "vulnerability": "warning",
    "leak": "warning",
    "exploit": "shocked",
    "critical": "shocked",
    "stop": "warning",
    "dead": "mind_blown",
    "out": "mind_blown",
    "shift": "excited",
}

@dataclass
class ThumbnailVariant:
    """Metadata for A/B testing thumbnail variants."""
    variant_id: str
    path: str
    style: str  # "authority", "curiosity", "urgency"
    hook_text: str
    emotion: str
    text_word_count: int
    contrast_score: float
    rule_of_thirds_score: float
    face_detected: bool
    created_at: str
    template_type: str  # "architecture", "cost", "security"

@dataclass
class ABTestResult:
    """Result of A/B test for thumbnail selection."""
    winner_variant_id: str
    ctr_data: Dict[str, float]
    test_duration_hours: int
    confidence: float

# ── ASSETS ───────────────────────────────────────────────────────────────────
AVATAR_PATH = os.path.join(ASSETS_DIR, "gemini_img_without_logo.png")
FONT_BLACK = os.path.join(ASSETS_DIR, "fonts", "Montserrat-Black.ttf")
FONT_EXTRABOLD = os.path.join(ASSETS_DIR, "fonts", "Montserrat-ExtraBold.ttf")
FONT_MONO = os.path.join(ASSETS_DIR, "fonts", "FiraCode-Bold.ttf")
FONT_MONO_ALT = os.path.join(ASSETS_DIR, "fonts", "JetBrainsMono-Bold.ttf")

FALLBACKS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf"]
MONO_FALLBACKS = ["/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"]

# Load OpenCV face detector (Haar cascade - lightweight, no extra downloads)
_FACE_CASCADE = None
def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE

_fcache = {}

def _load_font(size, weight="black"):
    key = (size, weight)
    if key not in _fcache:
        if weight == "extrabold":
            candidates = [FONT_EXTRABOLD, FONT_BLACK] + FALLBACKS
        elif weight == "mono":
            candidates = [FONT_MONO, FONT_MONO_ALT] + MONO_FALLBACKS
        else:
            candidates = [FONT_BLACK, FONT_EXTRABOLD] + FALLBACKS
        for p in candidates:
            if os.path.exists(p):
                try:
                    _fcache[key] = ImageFont.truetype(p, size)
                    break
                except: continue
        if key not in _fcache:
            _fcache[key] = ImageFont.load_default()
    return _fcache[key]

def _text_size(text, font):
    bb = font.getbbox(text)
    return bb[2] - bb[0], bb[3] - bb[1]

def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def _get_accent_colors(template: ThumbnailTemplate) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Get primary and secondary accent colors for a template."""
    if template == ThumbnailTemplate.ARCHITECTURE_SHIFT:
        return _hex_to_rgb("#00F2FE"), _hex_to_rgb("#00D2FF")  # Electric Cyan
    elif template == ThumbnailTemplate.COST_OPTIMIZATION:
        return _hex_to_rgb("#00FF87"), _hex_to_rgb("#00E676")  # Neon Green
    elif template == ThumbnailTemplate.VS_SHOWDOWN:
        return _hex_to_rgb("#A855F7"), _hex_to_rgb("#FFB703")  # Electric Violet & Solar Amber
    elif template == ThumbnailTemplate.BREAKTHROUGH:
        return _hex_to_rgb("#FFD166"), _hex_to_rgb("#FF6B35")  # Solar Gold & Ember Orange
    elif template == ThumbnailTemplate.DEV_BENCHMARK:
        return _hex_to_rgb("#00FF87"), _hex_to_rgb("#38BDF8")  # Cyber Lime & Sky Blue
    elif template == ThumbnailTemplate.DEEP_DIVE:
        return _hex_to_rgb("#FF0055"), _hex_to_rgb("#7928CA")  # Hot Crimson & Ultraviolet
    else:  # SECURITY_WARNING
        return _hex_to_rgb("#FF2A6D"), _hex_to_rgb("#FF0055")  # Hot Magenta/Crimson

def _detect_template_type(script_json: dict) -> ThumbnailTemplate:
    """Auto-detect template type from script content."""
    title = script_json.get("title", "").lower()
    description = script_json.get("description", "").lower()
    subcat = script_json.get("sub_category", "").lower()
    keywords = " ".join(script_json.get("keywords", [])).lower()
    content = f"{title} {description} {subcat} {keywords}"
    
    # Template D: VS / Showdown / Rivalry keywords (high priority)
    vs_keywords = [" vs ", " versus ", " vs. ", " against ", " compare ", " comparison ", 
                   " beats ", " crushes ", " destroys ", " showdown ", " rival ", " battle "]
    if any(k in f" {content} " for k in vs_keywords):
        return ThumbnailTemplate.VS_SHOWDOWN
    
    # Template E: Breakthrough / Paradigm Leap keywords
    breakthrough_keywords = ["breakthrough", "agi", "superintelligence", "leap", "discovered",
                             "revolutionary", "sora", "gpt-5", "o3", "quantum", "game changer",
                             "unveiled", "secret project", "holy grail", "world first"]
    if any(k in content for k in breakthrough_keywords):
        return ThumbnailTemplate.BREAKTHROUGH

    # Template F: Dev / Benchmark / Code keywords
    dev_keywords = ["benchmark", "accuracy", "latency", "eval", "score", "coding", "terminal",
                    "python", "speedrun", "framework", "repo", "library", "pytorch", "vllm",
                    "huggingface", "git", "cli", "sdk", "api"]
    if any(k in content for k in dev_keywords):
        return ThumbnailTemplate.DEV_BENCHMARK

    # Template G: Investigative Deep Dive / Secrets
    deep_keywords = ["truth", "exposed", "inside story", "what happened", "scandal", "dirty",
                     "investigation", "behind the scenes", "conspiracy", "hidden secret"]
    if any(k in content for k in deep_keywords):
        return ThumbnailTemplate.DEEP_DIVE

    # Template C: Security / Warning keywords
    sec_keywords = ["security", "vulnerability", "leak", "exploit", "hack", "breach",
                    "warning", "alert", "critical", "stop", "danger", "risk", "threat",
                    "malware", "injection", "exposed", "privacy", "data leak", "root access",
                    "containment", "ai safety", "alignment", "rogue", "attack"]
    if any(k in content for k in sec_keywords):
        return ThumbnailTemplate.SECURITY_WARNING

    # Template B: Cost Optimization keywords
    cost_keywords = ["cost", "save", "free", "open source", "budget", "bill", "pricing",
                     "expensive", "cheap", "alternative", "self-host", "local", "ollama",
                     "90%", "80%", "cut", "reduce", "optimize", "efficient", "$0", "zero cost"]
    if any(k in content for k in cost_keywords):
        return ThumbnailTemplate.COST_OPTIMIZATION

    # Template A: Architecture / Tech Shift keywords
    arch_keywords = ["rag", "agentic", "architecture", "shift", "new paradigm", "graph", 
                     "memory", "vector", "embedding", "llm", "model", "pipeline", "framework",
                     "dropping", "replacing", "migration", "modern", "legacy", "deprecated"]
    if any(k in content for k in arch_keywords):
        return ThumbnailTemplate.ARCHITECTURE_SHIFT

    # Default to architecture for tech content
    return ThumbnailTemplate.ARCHITECTURE_SHIFT

# ── FACE DETECTION & EMOTION OVERLAY ──────────────────────────────────────────

def _detect_face_region(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face in image using OpenCV Haar cascade.
    Returns (x, y, w, h) in PIL coordinates or None.
    """
    try:
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        cascade = _get_face_cascade()
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return (int(x), int(y), int(w), int(h))
    except Exception as e:
        print(f"⚠️ Face detection failed: {e}")
    return None


def _select_emotion_for_content(title: str, hook_text: str) -> str:
    """Select appropriate emotion emoji based on content keywords."""
    content = (title + " " + hook_text).lower()
    for keyword, emotion in EMOTION_WEIGHTS.items():
        if keyword in content:
            return EMOTION_OVERLAYS[emotion]
    return EMOTION_OVERLAYS["excited"]


def _apply_emotion_overlay(canvas: Image.Image, face_region: Optional[Tuple], emotion: str, 
                            accent_color: Tuple[int, int, int], is_shorts: bool) -> Image.Image:
    """
    Apply emotion emoji overlay near detected face region following rule of thirds.
    """
    if not face_region:
        return canvas
    
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    fx, fy, fw, fh = face_region
    
    # Rule of thirds: place emotion at intersection points
    third_w, third_h = w // 3, h // 3
    intersections = [
        (third_w, third_h),           # Top-left
        (2 * third_w, third_h),       # Top-right
        (third_w, 2 * third_h),       # Bottom-left
        (2 * third_w, 2 * third_h),   # Bottom-right
    ]
    
    # Pick intersection closest to face but on opposite side for balance
    face_center = (fx + fw // 2, fy + fh // 2)
    best_pos = min(intersections, key=lambda p: abs(p[0] - face_center[0]) + abs(p[1] - face_center[1]))
    
    # Offset slightly from intersection to avoid covering face
    offset_x = 60 if best_pos[0] < w // 2 else -60
    offset_y = -40 if best_pos[1] < h // 2 else 40
    emoji_pos = (best_pos[0] + offset_x, best_pos[1] + offset_y)
    
    # Clamp to canvas bounds
    emoji_pos = (max(50, min(w - 150, emoji_pos[0])), max(50, min(h - 150, emoji_pos[1])))
    
    # Draw emotion emoji with glow
    font_size = 80 if not is_shorts else 100
    font = _load_font(font_size, "black")
    
    # Glow effect
    for offset in range(1, 8):
        draw.text((emoji_pos[0] + offset, emoji_pos[1] + offset), emotion, font=font, fill=(0, 0, 0, 180))
    draw.text(emoji_pos, emotion, font=font, fill=(255, 255, 255, 255))
    
    # Add accent ring around emoji
    ring_radius = font_size // 2 + 10
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ring_draw = ImageDraw.Draw(overlay)
    ring_draw.ellipse([
        emoji_pos[0] - ring_radius, emoji_pos[1] - ring_radius,
        emoji_pos[0] + ring_radius, emoji_pos[1] + ring_radius
    ], outline=(*accent_color, 180), width=4)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    
    return canvas


def _enforce_max_words(text: str, max_words: int = MAX_TEXT_WORDS) -> str:
    """Enforce maximum word count for thumbnail text."""
    words = text.replace("\n", " ").split()
    if len(words) <= max_words:
        return text
    # Keep first max_words words
    return " ".join(words[:max_words])


def _calculate_contrast_ratio(fg_color: Tuple[int, int, int], bg_color: Tuple[int, int, int]) -> float:
    """Calculate WCAG contrast ratio between foreground and background."""
    def luminance(r, g, b):
        def channel(c):
            c = c / 255.0
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    
    l1 = luminance(*fg_color)
    l2 = luminance(*bg_color)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def _calculate_rule_of_thirds_score(canvas: Image.Image, text_positions: List[Tuple], 
                                     face_region: Optional[Tuple]) -> float:
    """Score how well elements align with rule of thirds grid."""
    w, h = canvas.size
    third_w, third_h = w / 3, h / 3
    grid_lines = [third_w, 2 * third_w, third_h, 2 * third_h]
    
    score = 0.0
    elements = text_positions + ([ (face_region[0] + face_region[2]//2, face_region[1] + face_region[3]//2) ] if face_region else [])
    
    for ex, ey in elements:
        # Distance to nearest vertical grid line
        dx = min(abs(ex - gl) for gl in grid_lines[:2])
        # Distance to nearest horizontal grid line
        dy = min(abs(ey - gl) for gl in grid_lines[2:])
        # Score: closer to grid = higher score (normalized)
        score += 1.0 - min(1.0, (dx + dy) / (w + h) * 6)
    
    return score / len(elements) if elements else 0.0


# ── AI AGENTS ─────────────────────────────────────────────────────────────────

def _generate_hook_text(title, client=None, is_shorts=False, variant_style="curiosity", template: ThumbnailTemplate = None):
    """Generates template-specific hook text following design system with AI generation and template fallbacks."""
    
    # 1. Try Gemini AI Generation first if client is available
    if client and GEMINI_API_KEY:
        variant_prompts = {
            "authority": "Authoritative, expert tone. Hard evidence, definitive verdict. Max 3 words. Examples: 'EXPERT VERDICT', 'OFFICIAL BENCHMARK', 'THE VERIFIED TRUTH'",
            "curiosity": "Curiosity gap, psychological open loop, controversial question. Max 3 words. Examples: 'RAG IS DEAD?', 'THE HIDDEN TRUTH', 'THIS CHANGES ALL'",
            "urgency": "Urgent, breaking alert, high FOMO. Max 3 words. Examples: 'ACT RIGHT NOW', 'CRITICAL UPDATE', 'DO NOT MISS'"
        }
        style_guidance = variant_prompts.get(variant_style, variant_prompts["curiosity"])
        template_name = template.value if template else "tech_revolution"
        
        prompt = f"""You are an elite YouTube thumbnail copywriter (Fireship / MrBeast style).
Generate an ultra-punchy 2 to 3 word hook in ALL CAPS for a video thumbnail titled: "{title}".
TEMPLATE CATEGORY: {template_name}
VARIATION ANGLE: {style_guidance}

CRITICAL RULES:
1. EXACTLY 2 OR 3 WORDS MAXIMUM.
2. ALL CAPS ONLY.
3. Put \\n between the first and second/third word for a 2-line visual stack (e.g. "DEEPSEEK\\nCRUSHES IT" or "SECRET\\nREVEALED").
4. Extreme emotional resonance, curiosity gap, or authority signal.
5. Return ONLY the 2-3 words. Absolutely no quotes, markdown, or punctuation."""

        try:
            response = client.models.generate_content(model=GEMINI_FLASH_MODEL, contents=prompt)
            if response and response.text:
                hook = response.text.strip().replace("\\n", "\n")
                words = hook.replace("\n", " ").split()
                if 1 <= len(words) <= 4:
                    # Format as 2 clean lines
                    if "\n" not in hook and len(words) >= 2:
                        hook = f"{words[0]}\n{' '.join(words[1:])}"
                    return hook
        except Exception as e:
            print(f"⚠️ Gemini hook generation fallback due to: {e}")

    # 2. Rich, diverse template-specific hook pools (Fallback)
    template_hooks = {
        ThumbnailTemplate.ARCHITECTURE_SHIFT: {
            "authority": ["NEW\nPARADIGM", "THE NEW\nSTANDARD", "NEXT GEN\nSYSTEM", "SYSTEM\nSHIFT"],
            "curiosity": ["RAG IS\nDEAD?", "OLD WAY\nGONE", "THIS\nREPLACES IT", "NEW\nAPPROACH"],
            "urgency": ["MIGRATE\nNOW", "DON'T FALL\nBEHIND", "SWITCH\nTODAY", "ACT\nFAST"],
        },
        ThumbnailTemplate.COST_OPTIMIZATION: {
            "authority": ["PROVEN\nSAVINGS", "EXPERT\nVERDICT", "$0 TOTAL\nCOST", "ZERO COST\nSTACK"],
            "curiosity": ["SAVE\n90%?", "$0 AI\nSTACK", "FREE\nFOREVER?", "SECRET\nSAVINGS"],
            "urgency": ["CUT BILLS\nNOW", "STOP\nOVERPAYING", "SLASH BILLS\nTODAY", "CLAIM\nSAVINGS"],
        },
        ThumbnailTemplate.SECURITY_WARNING: {
            "authority": ["CRITICAL\nALERT", "SECURITY\nBRIEF", "EXPERT\nWARNING", "VERIFIED\nTHREAT"],
            "curiosity": ["DATA\nLEAK?", "YOU'RE\nEXPOSED", "SECRET RISK\nREVEALED", "WHAT THEY\nHID"],
            "urgency": ["PATCH\nNOW", "IMMEDIATE\nACTION", "CRITICAL\nDANGER", "DO NOT\nWAIT"],
        },
        ThumbnailTemplate.VS_SHOWDOWN: {
            "authority": ["THE DEFINITIVE\nWINNER", "BENCHMARK\nDECIDED", "OFFICIAL\nSHOWDOWN", "CLEAR\nWINNER"],
            "curiosity": ["IT'S NOT\nEVEN CLOSE", "WHO ACTUALLY\nWINS?", "SHOCKING\nRESULTS", "THE BIG\nUPSET"],
            "urgency": ["SWITCH SIDES\nNOW", "END OF AN\nERA", "THE KING\nFALLS", "UPSET OF\n2026"],
        },
        ThumbnailTemplate.BREAKTHROUGH: {
            "authority": ["MAJOR\nBREAKTHROUGH", "OFFICIALLY\nCONFIRMED", "NEXT LEAP\nHERE", "AGI\nMILESTONE"],
            "curiosity": ["EVERYTHING\nCHANGES", "THE HOLY\nGRAIL?", "HOW IS THIS\nPOSSIBLE?", "UNREAL\nLEAP"],
            "urgency": ["LOOK AT\nTHIS", "DO NOT\nSLEEP ON THIS", "HISTORY IN\nMAKING", "SEE IT\nFIRST"],
        },
        ThumbnailTemplate.DEV_BENCHMARK: {
            "authority": ["99.4%\nACCURACY", "10X SPEED\nPROVEN", "NEW SOTA\nRECORD", "UNMATCHED\nSPEED"],
            "curiosity": ["FASTER THAN\nPYTHON?", "HOW SO\nFAST?", "INSANE\nBENCHMARK", "DEV DREAM\nSTACK"],
            "urgency": ["INSTALL\nNOW", "UPGRADE YOUR\nCODE", "TRY THIS\nTODAY", "START\nBUILDING"],
        },
        ThumbnailTemplate.DEEP_DIVE: {
            "authority": ["INVESTIGATION\nFILE", "UNREDACTED\nTRUTH", "THE FULL\nSTORY", "CONFIDENTIAL\nBRIEF"],
            "curiosity": ["WHAT REALLY\nHAPPENED", "THEY HID\nTHIS", "WHAT NOBODY\nTOLD YOU", "THE REAL\nREASON"],
            "urgency": ["WATCH BEFORE\nDELETED", "LEAKED\nNOW", "THEY CAN'T\nHIDE THIS", "MUST KNOW\nTRUTH"],
        },
    }
    
    target_template = template if (template and template in template_hooks) else ThumbnailTemplate.ARCHITECTURE_SHIFT
    hooks = template_hooks[target_template].get(variant_style, template_hooks[target_template]["curiosity"])
    return random.choice(hooks)

def _generate_imagen_background(title, client):
    """Generates a thematic tech background using Imagen-3."""
    print(f"🎨 Generating Premium Imagen background for: {title}")
    prompt = (
        f"A cinematic high-contrast YouTube thumbnail background about: {title}. "
        "Must feature a shocked expressive tech creator pointing, alongside a stylized generic app icon or tech symbol (such as a lock, gear, alert warning emblem, or app glyph). "
        "Strictly avoid any copyrighted brand logos, trademarks, or company marks (like Apple, GitHub, OpenAI, WhatsApp). "
        "Dark cyberpunk noir mood, dramatic studio neon lighting, high contrast, 8k resolution, clean composition, no text."
    )
    try:
        response = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio='16:9'
            )
        )
        img_bytes = response.generated_images[0].image.image_bytes
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print(f"⚠️ Imagen failed: {e}. Trying HuggingFace/Pollinations fallback...")
        
        # Try HuggingFace FLUX.1 first
        try:
            from config import HF_TOKEN
            if HF_TOKEN:
                resp = requests.post(
                    "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
                    headers={"Authorization": f"Bearer {HF_TOKEN}"},
                    json={"inputs": prompt, "parameters": {"width": 1280, "height": 720}},
                    timeout=60
                )
                if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
                    print("✅ HuggingFace background generated successfully!")
                    return Image.open(io.BytesIO(resp.content)).convert("RGB")
                else:
                    print(f"⚠️ HuggingFace returned status: {resp.status_code}")
        except Exception as hfe:
            print(f"⚠️ HuggingFace fallback failed: {hfe}")
        
        # Then try Cloudflare Workers AI
        try:
            from config import CF_ACCOUNT_ID, CF_API_TOKEN
            if CF_ACCOUNT_ID and CF_API_TOKEN:
                resp = requests.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/black-forest-labs/flux-1-schnell",
                    headers={
                        "Authorization": f"Bearer {CF_API_TOKEN}",
                        "Content-Type": "application/json"
                    },
                    json={"prompt": prompt},
                    timeout=60
                )
                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "")
                    if content_type.startswith("image"):
                        print("✅ Cloudflare background generated successfully!")
                        return Image.open(io.BytesIO(resp.content)).convert("RGB")
                    else:
                        try:
                            import base64
                            data = resp.json()
                            if data.get("success") and data.get("result", {}).get("image"):
                                img_bytes = base64.b64decode(data["result"]["image"])
                                print("✅ Cloudflare background generated successfully (base64)!")
                                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        except Exception:
                            pass
                else:
                    print(f"⚠️ Cloudflare returned status: {resp.status_code}")
        except Exception as cfe:
            print(f"⚠️ Cloudflare fallback failed: {cfe}")
        
        # Then try Pollinations
        try:
            import urllib.parse
            encoded_prompt = urllib.parse.quote(prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1280&height=720&nologo=true&private=true"
            resp = requests.get(url, timeout=45)
            if resp.status_code == 200:
                print("✅ Pollinations background generated successfully!")
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            elif resp.status_code == 429:
                print(f"⚠️ Pollinations rate limited (429). Skipping.")
        except Exception as pe:
            print(f"⚠️ Pollinations fallback failed: {pe}")
            
        print("Using dark fallback.")
        return Image.new("RGB", (THUMB_W, THUMB_H), (10, 10, 15))

# ── THEMATIC & AI BACKGROUND SYSTEM ──────────────────────────────────────────

def _generate_imagen_background(script_json: dict, client: Optional[genai.Client],
                                template: ThumbnailTemplate, primary_accent: Tuple[int, int, int]) -> Optional[Image.Image]:
    """Generate a cinematic 16:9 AI background illustration via Imagen 3."""
    if not client or not GEMINI_API_KEY:
        return None
    try:
        title = script_json.get("title", "AI Technology Breakthrough")
        topic = script_json.get("topic") or title
        if len(topic) > 80:
            topic = topic[:80]
        
        accent_hex = "#{:02x}{:02x}{:02x}".format(*primary_accent)
        prompt = (
            f"Cinematic ultra-detailed dark futuristic 3D background visualization of {topic}. "
            f"Obsidian server architecture, glowing {accent_hex} volumetric cybernetic neon lines, "
            f"clean deep shadows, photorealistic, 8k resolution, octane render style, "
            f"no text, no letters, no words, no logos, no human faces."
        )
        
        print(f"🎨 Generating Imagen 3 background for: '{topic}'...")
        result = client.models.generate_images(
            model='imagen-3.0-generate-002',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio='16:9',
                output_mime_type='image/jpeg',
            )
        )
        if result and result.generated_images:
            img_bytes = result.generated_images[0].image.image_bytes
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            print("✅ Imagen 3 background generated successfully!")
            return img
    except Exception as e:
        print(f"⚠️ Imagen 3 background generation skipped (fallback to procedural): {e}")
    return None


def _render_tilted_article_card(canvas: Image.Image, script_json: Optional[dict],
                                x: int, y: int, w: int, h: int,
                                primary_accent: Tuple[int, int, int],
                                secondary_accent: Tuple[int, int, int],
                                tilt_angle: float = -5.0) -> Image.Image:
    """
    Renders a 3D tilted high-authority article/research paper preview card with glowing border,
    drop shadow, and tangible evidence (headline/graphs) cropped from screenshot_gen.py captures.
    """
    if w < 120 or h < 80:
        return canvas
    
    # 1. Look for screenshot
    ss_path = None
    if script_json:
        ss_path = (script_json.get("screenshot_path") or 
                   script_json.get("article_screenshot") or 
                   script_json.get("thumbnail_bg"))
        if not ss_path or not os.path.exists(ss_path):
            ss_dir = os.path.join(ASSETS_DIR, "screenshots")
            if os.path.exists(ss_dir):
                files = [os.path.join(ss_dir, f) for f in os.listdir(ss_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    ss_path = sorted(files, key=os.path.getmtime, reverse=True)[0]
    
    card_w = min(w, 480)
    card_h = min(h, 320)
    
    # Create card canvas (RGBA)
    card = Image.new("RGBA", (card_w, card_h), (12, 16, 26, 245))
    card_draw = ImageDraw.Draw(card)
    
    # Try to embed cropped screenshot inside card
    has_screenshot = False
    if ss_path and os.path.exists(ss_path):
        try:
            with Image.open(ss_path) as ss_img:
                sw, sh = ss_img.size
                crop_h = min(sh, int(sw * 0.65))
                cropped_ss = ss_img.crop((0, 0, sw, crop_h)).resize((card_w - 16, card_h - 60), Image.LANCZOS)
                cropped_ss = ImageEnhance.Contrast(cropped_ss).enhance(1.15)
                card.paste(cropped_ss.convert("RGBA"), (8, 48))
                has_screenshot = True
        except Exception as e:
            print(f"⚠️ Screenshot crop for 3D card skipped: {e}")
            has_screenshot = False

    if not has_screenshot:
        title_txt = script_json.get("title", "AI Research Paper") if script_json else "Official System Benchmark"
        f_title = _load_font(22, "extrabold")
        lines = textwrap.wrap(title_txt, width=28)
        ty = 56
        for line in lines[:3]:
            card_draw.text((18, ty), line, font=f_title, fill=(240, 245, 255))
            ty += 28
        f_metric = _load_font(18, "mono")
        card_draw.text((18, ty + 12), "✓ Verified: 100% Peer-Reviewed", font=f_metric, fill=(*secondary_accent, 220))
        card_draw.text((18, ty + 36), "✓ Model: SOTA Architecture", font=f_metric, fill=(160, 180, 200, 200))
    
    # Top HUD Bar on the card
    card_draw.rectangle([0, 0, card_w, 42], fill=(20, 28, 44, 255))
    card_draw.line([(0, 42), (card_w, 42)], fill=(*primary_accent, 140), width=1)
    
    # Status Badge / Dots
    card_draw.ellipse([14, 16, 24, 26], fill=(255, 95, 87, 240))
    card_draw.ellipse([30, 16, 40, 26], fill=(254, 188, 46, 240))
    card_draw.ellipse([46, 16, 56, 26], fill=(40, 201, 64, 240))
    
    tag_text = "📄 EVIDENCE // OFFICIAL DISCLOSURE"
    f_tag = _load_font(16, "mono")
    card_draw.text((68, 14), tag_text, font=f_tag, fill=(*primary_accent, 255))
    
    # Card outer border with glow
    card_draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=14, outline=(*primary_accent, 220), width=3)
    
    # 2. Apply 3D perspective / rotation tilt
    rotated_card = card.rotate(tilt_angle, expand=True, resample=Image.BICUBIC)
    rw, rh = rotated_card.size
    
    # 3. Create realistic soft drop shadow
    shadow_pad = 25
    shadow_img = Image.new("RGBA", (rw + shadow_pad * 2, rh + shadow_pad * 2), (0, 0, 0, 0))
    r_alpha = rotated_card.split()[3]
    shadow_core = Image.new("RGBA", (rw, rh), (0, 0, 0, 200))
    shadow_img.paste(shadow_core, (shadow_pad + 6, shadow_pad + 12), mask=r_alpha)
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=16))
    
    # 4. Composite onto canvas at (x, y)
    target_x = x + max(0, (w - rw) // 2)
    target_y = y + max(0, (h - rh) // 2)
    
    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba.alpha_composite(shadow_img, (target_x - shadow_pad, target_y - shadow_pad))
    canvas_rgba.alpha_composite(rotated_card, (target_x, target_y))
    
    return canvas_rgba.convert("RGB")


def _render_thematic_background(width: int, height: int, template: ThumbnailTemplate, 
                                primary_accent: Tuple[int, int, int], secondary_accent: Tuple[int, int, int],
                                script_json: Optional[dict] = None,
                                ai_background: Optional[Image.Image] = None) -> Image.Image:
    """
    Renders rich thematic backgrounds with dual-tone atmospheric radial glows,
    tech grids, and optional AI Imagen background or blurred article preview cards.
    """
    # 1. Base canvas selection: AI illustration or procedural dark tone
    if ai_background:
        # Scale and slightly darken AI backdrop to keep text contrast ultra-high
        canvas = ai_background.resize((width, height), Image.LANCZOS)
        canvas = ImageEnhance.Brightness(canvas).enhance(0.40)
    else:
        if template == ThumbnailTemplate.VS_SHOWDOWN:
            base_color = BG_DARK_INDIGO
        elif template == ThumbnailTemplate.BREAKTHROUGH:
            base_color = BG_DARK_INDIGO
        elif template in (ThumbnailTemplate.DEV_BENCHMARK, ThumbnailTemplate.COST_OPTIMIZATION):
            base_color = BG_CARBON_GREEN
        elif template in (ThumbnailTemplate.SECURITY_WARNING, ThumbnailTemplate.DEEP_DIVE):
            base_color = BG_BLOOD_NAVY
        else:
            base_color = BG_OBSIDIAN
        canvas = Image.new("RGB", (width, height), base_color)

    glow_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_overlay)

    # 2. Dynamic Radial Atmospheric Lighting
    if template == ThumbnailTemplate.VS_SHOWDOWN:
        r = int(height * 0.75)
        glow_draw.ellipse([-r // 3, height // 2 - r, r, height // 2 + r], fill=(*primary_accent, 45))
        glow_draw.ellipse([width - r, height // 2 - r, width + r // 3, height // 2 + r], fill=(*secondary_accent, 40))
    elif template == ThumbnailTemplate.BREAKTHROUGH:
        core_x = int(width * 0.65)
        core_y = height // 2
        r = int(height * 0.65)
        glow_draw.ellipse([core_x - r, core_y - r, core_x + r, core_y + r], fill=(*primary_accent, 55))
        glow_draw.ellipse([core_x - r // 2, core_y - r // 2, core_x + r // 2, core_y + r // 2], fill=(*secondary_accent, 45))
    elif template == ThumbnailTemplate.DEV_BENCHMARK:
        glow_draw.ellipse([width - 500, -100, width + 200, 600], fill=(*primary_accent, 40))
        glow_draw.ellipse([-150, height - 400, 450, height + 200], fill=(*secondary_accent, 30))
    elif template in (ThumbnailTemplate.SECURITY_WARNING, ThumbnailTemplate.DEEP_DIVE):
        r = int(height * 0.70)
        glow_draw.ellipse([width // 2 - r, height // 2 - r, width // 2 + r, height // 2 + r], fill=(*primary_accent, 35))
    else:
        glow_draw.ellipse([width - 600, -100, width + 100, 600], fill=(*primary_accent, 40))

    glow_overlay = glow_overlay.filter(ImageFilter.GaussianBlur(radius=55))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), glow_overlay).convert("RGB")

    # 3. Optional contextual article/screenshot backdrop preview (if not already using AI background)
    if not ai_background and script_json:
        ss_path = (script_json.get("screenshot_path") or 
                   script_json.get("article_screenshot") or 
                   script_json.get("thumbnail_bg"))
        if not ss_path or not os.path.exists(ss_path):
            ss_dir = os.path.join(ASSETS_DIR, "screenshots")
            if os.path.exists(ss_dir):
                files = [os.path.join(ss_dir, f) for f in os.listdir(ss_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    ss_path = sorted(files, key=os.path.getmtime, reverse=True)[0]
        
        if ss_path and os.path.exists(ss_path):
            try:
                with Image.open(ss_path) as ss_img:
                    cw, ch = ss_img.size
                    crop_h = min(ch, int(cw * (height / width)))
                    cropped = ss_img.crop((0, 0, cw, crop_h)).resize((width, height), Image.LANCZOS)
                    dimmed = ImageEnhance.Brightness(cropped).enhance(0.18)
                    dimmed = dimmed.filter(ImageFilter.GaussianBlur(radius=6))
                    canvas = Image.blend(canvas, dimmed.convert("RGB"), alpha=0.35)
            except Exception as e:
                print(f"⚠️ Backdrop screenshot blending skipped: {e}")

    # 4. Tech grid and decorative markers
    canvas = _render_tech_grid(canvas, grid_color=(*primary_accent, 18), spacing=70)
    canvas = _draw_tech_decorations(canvas, primary_accent)
    return canvas

# ── FIGMA TECH REF UTILITIES ──────────────────────────────────────────────────

def _render_tech_grid(canvas, grid_color=(128, 128, 128, 25), spacing=60, dash_len=5):
    """Renders a beautiful semi-transparent dashed technical grid overlay."""
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = canvas.size
    
    # Vertical lines (dashed)
    for x in range(0, w, spacing):
        for y in range(0, h, dash_len * 2):
            draw.line([(x, y), (x, y + dash_len)], fill=grid_color, width=1)
            
    # Horizontal lines (dashed)
    for y in range(0, h, spacing):
        for x in range(0, w, dash_len * 2):
            draw.line([(x, y), (x + dash_len, y)], fill=grid_color, width=1)
            
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

def _draw_tech_decorations(canvas, accent_color):
    """Draws subtle Figma HUD/UI style decorations (crosshairs, boundary brackets, micro labels)."""
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = canvas.size
    accent_alpha = (*accent_color, 75)
    white_alpha = (255, 255, 255, 55)
    
    # 1. Floating Crosshairs (+) in empty areas
    crosshairs = [
        (w // 4, h // 5),
        (w // 3, h // 2 + 100),
        (w // 2 - 100, h // 4 - 30),
        (w // 2 + 150, h // 2 + 180)
    ]
    cross_size = 8
    for cx, cy in crosshairs:
        draw.line([(cx - cross_size, cy), (cx + cross_size, cy)], fill=white_alpha, width=1)
        draw.line([(cx, cy - cross_size), (cx, cy + cross_size)], fill=white_alpha, width=1)
        
    # 2. Corner right-angle bounding brackets
    margin = 35
    bracket_len = 25
    # Top-Left Bracket
    draw.line([(margin, margin), (margin + bracket_len, margin)], fill=accent_alpha, width=2)
    draw.line([(margin, margin), (margin, margin + bracket_len)], fill=accent_alpha, width=2)
    # Top-Right Bracket
    draw.line([(w - margin, margin), (w - margin - bracket_len, margin)], fill=accent_alpha, width=2)
    draw.line([(w - margin, margin), (w - margin, margin + bracket_len)], fill=accent_alpha, width=2)
    # Bottom-Left Bracket
    draw.line([(margin, h - margin), (margin + bracket_len, h - margin)], fill=accent_alpha, width=2)
    draw.line([(margin, h - margin), (margin, h - margin - bracket_len)], fill=accent_alpha, width=2)
    # Bottom-Right Bracket
    draw.line([(w - margin, h - margin), (w - margin - bracket_len, h - margin)], fill=accent_alpha, width=2)
    draw.line([(w - margin, h - margin), (w - margin, h - margin - bracket_len)], fill=accent_alpha, width=2)
    
    # 3. Monospace dimensional micro-label in top-right
    try:
        f_mono = _load_font(18, "extrabold")
        label_text = f"[ 16:9_HD // {w}x{h} ]"
        draw.text((w - 280, 50), label_text, font=f_mono, fill=(255, 255, 255, 90))
    except Exception as e:
        print("⚠️ Monospace decoration label failed:", e)
        
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

def _get_bezier_points(p0, p1, p2, p3, steps=30):
    points = []
    for t in [i / steps for i in range(steps + 1)]:
        x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points

def _draw_curved_accent(canvas, accent_color):
    """Draws a premium glowing organic bezier curve near bottom-left corner with glowing pips."""
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = canvas.size
    
    # Define Bezier points near the bottom left (away from text and avatar)
    p0 = (60, h - 120)
    p1 = (120, h - 80)
    p2 = (200, h - 160)
    p3 = (260, h - 140)
    
    points = _get_bezier_points(p0, p1, p2, p3)
    
    # Draw glowing shadow for the line
    for gw in range(8, 2, -2):
        draw.line(points, fill=(*accent_color, int(80 / gw)), width=gw)
    # Draw main accent line (White)
    draw.line(points, fill=(255, 255, 255, 200), width=2)
    
    # Draw glowing circular nodes (pips) at key points
    for px, py in [p0, p3]:
        # Outer glow
        draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(*accent_color, 80))
        # Inner node (bright White)
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 255, 255, 240))
        
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

def _draw_multi_tier_glow(canvas, av_res, pos, accent_color):
    """Draws a beautiful, premium multi-tiered glowing aura radiating behind the avatar."""
    glow_canvas = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    # Calculate center of the avatar
    av_center_x = pos[0] + av_res.width // 2
    av_center_y = pos[1] + av_res.height // 2
    
    draw = ImageDraw.Draw(glow_canvas)
    
    # Tier 1: Massive soft backdrop aura (large radius, very low opacity)
    r1 = int(av_res.height * 0.55)
    draw.ellipse([av_center_x - r1, av_center_y - r1, av_center_x + r1, av_center_y + r1], 
                 fill=(*accent_color, 40))
                 
    # Tier 2: Medium backdrop aura (medium radius, medium opacity)
    r2 = int(av_res.height * 0.40)
    draw.ellipse([av_center_x - r2, av_center_y - r2, av_center_x + r2, av_center_y + r2], 
                 fill=(*accent_color, 75))
                 
    # Tier 3: Core intensive backing glow (smaller radius, high opacity)
    r3 = int(av_res.height * 0.22)
    draw.ellipse([av_center_x - r3, av_center_y - r3, av_center_x + r3, av_center_y + r3], 
                 fill=(*accent_color, 130))
                 
    # Apply heavy blur to the radial gradient circle overlay
    glow_canvas = glow_canvas.filter(ImageFilter.GaussianBlur(radius=45))
    
    # Tier 4: Detailed outline body glow matching the avatar's exact shape
    body_mask = av_res.split()[3].point(lambda x: 255 if x > 0 else 0)
    body_glow = body_mask.filter(ImageFilter.GaussianBlur(radius=25))
    body_glow_img = Image.new("RGBA", av_res.size, (*accent_color, 120))
    
    # Compose everything
    canvas_rgba = canvas.convert("RGBA")
    # Paste global radial backdrop glow
    canvas_rgba = Image.alpha_composite(canvas_rgba, glow_canvas)
    # Paste local avatar body glow onto composite
    temp_local = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    temp_local.paste(body_glow_img, pos, mask=body_glow)
    canvas_rgba = Image.alpha_composite(canvas_rgba, temp_local)
    
    # Paste the actual avatar image
    temp_avatar = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    temp_avatar.paste(av_res, pos, mask=av_res)
    canvas_rgba = Image.alpha_composite(canvas_rgba, temp_avatar)
    
    return canvas_rgba.convert("RGB")

# ── IMAGE PROCESSING ─────────────────────────────────────────────────────────

def _process_avatar_still(avatar_path=None, still_time=1.0):
    """
    Extracts an avatar frame (if video) or loads a static image,
    removes the background using rembg with local caching, and returns the cutout PIL Image.
    """
    if not avatar_path:
        avatar_path = AVATAR_PATH
        
    if not os.path.exists(avatar_path):
        print(f"⚠️ Avatar not found at: {avatar_path}. Falling back to default AVATAR_PATH.")
        avatar_path = AVATAR_PATH
        if not os.path.exists(avatar_path):
            print("⚠️ Default avatar not found at", AVATAR_PATH)
            return None

    # Setup cutout cache directory
    cache_dir = os.path.join(OUTPUT_DIR, ".avatar_cutout_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Compute a unique cache key based on path & timestamp
    path_hash = hashlib.md5(f"{os.path.abspath(avatar_path)}_{still_time}".encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, f"cutout_{path_hash}.png")

    if os.path.exists(cache_file):
        try:
            print(f"🎯 Loading cached avatar cutout: {cache_file}")
            return Image.open(cache_file).convert("RGBA")
        except Exception as e:
            print(f"⚠️ Failed to load cached cutout: {e}. Re-processing...")

    print(f"👤 Processing avatar still from: {avatar_path} (time={still_time}s)...")
    try:
        input_img = None
        ext = os.path.splitext(avatar_path)[1].lower()
        
        # If it's a video, extract frame at still_time using cv2
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            cap = cv2.VideoCapture(avatar_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_idx = int(still_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                # Fallback: try reading the first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()
            
            if success:
                # Convert BGR (cv2 default) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_img = Image.fromarray(frame_rgb)
            cap.release()
            
            if input_img is None:
                raise Exception("Failed to extract frame from video.")
        else:
            # It's a static image
            input_img = Image.open(avatar_path).convert("RGBA")
            
        print("🪄 Removing background via rembg...")
        output_img = remove(input_img)
        
        # Save to cache
        output_img.save(cache_file, "PNG")
        return output_img
    except Exception as e:
        print(f"⚠️ Avatar processing failed: {e}")
        # Final fallback, try loading as static image if possible
        try:
            return Image.open(avatar_path).convert("RGBA")
        except:
            return None

def _draw_logo_badges(canvas, script_json, accent_color, is_shorts=False):
    """
    Renders premium glassmorphic HUD logo badges on the canvas.
    """
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = canvas.size
    
    # Gather logos from script_json
    logo_list = []
    
    single_logo = script_json.get("logo_path")
    if single_logo:
        logo_list.append({"path": single_logo, "position": "top_left", "label": "AUTHORITY SYSTEM"})
        
    json_logos = script_json.get("logos")
    if json_logos:
        if isinstance(json_logos, list):
            for l in json_logos:
                if isinstance(l, dict) and "path" in l:
                    logo_list.append({
                        "path": l["path"],
                        "position": l.get("position", "top_left"),
                        "label": l.get("label", "")
                    })
                elif isinstance(l, str):
                    logo_list.append({"path": l, "position": "top_left", "label": ""})
                    
    # Default fallback: if no logos specified, let's auto-overlay assets/logo.png in top_left
    if not logo_list:
        default_logo = os.path.join(ASSETS_DIR, "logo.png")
        if os.path.exists(default_logo):
            logo_list.append({"path": default_logo, "position": "top_left", "label": "GEN NEWS"})

    for logo_spec in logo_list:
        path = logo_spec["path"]
        pos_name = logo_spec["position"]
        label = logo_spec["label"]
        
        if not os.path.exists(path):
            # Check if it's in assets/icons/ or assets/
            for cand in [os.path.join(ASSETS_DIR, "icons", path), os.path.join(ASSETS_DIR, path)]:
                if os.path.exists(cand):
                    path = cand
                    break
            else:
                print(f"⚠️ Logo file not found: {path}")
                continue
                
        try:
            logo_img = Image.open(path).convert("RGBA")
        except Exception as e:
            print(f"⚠️ Failed to load logo {path}: {e}")
            continue
            
        # Draw badge depending on position
        if is_shorts:
            logo_h = 50
            scale = logo_h / logo_img.height
            logo_w = int(logo_img.width * scale)
            logo_res = logo_img.resize((logo_w, logo_h), Image.LANCZOS)
            
            px, py = 50, 60
            draw.rounded_rectangle([px - 15, py - 10, px + logo_w + 15, py + logo_h + 10], radius=8, fill=(10, 10, 15, 180), outline=(*accent_color, 120), width=1)
            overlay.paste(logo_res, (px, py), mask=logo_res)
        else:
            if pos_name == "top_left":
                logo_h = 42
                scale = logo_h / logo_img.height
                logo_w = int(logo_img.width * scale)
                logo_res = logo_img.resize((logo_w, logo_h), Image.LANCZOS)
                
                px = 60
                py = 50
                
                font_label = _load_font(14, "extrabold")
                label_w = 0
                if label:
                    label_w, _ = _text_size(label, font_label)
                    label_w += 20
                    
                badge_w = logo_w + 30 + label_w
                badge_h = logo_h + 20
                
                box_coords = [px - 15, py - 10, px + badge_w - 15, py + badge_h - 10]
                draw.rounded_rectangle(box_coords, radius=10, fill=(10, 10, 15, 210), outline=(*accent_color, 140), width=2)
                
                overlay.paste(logo_res, (px, py), mask=logo_res)
                
                if label:
                    draw.text((px + logo_w + 12, py + 12), label, font=font_label, fill=(255, 255, 255, 230))
                    
            elif pos_name == "bottom_left":
                logo_h = 38
                scale = logo_h / logo_img.height
                logo_w = int(logo_img.width * scale)
                logo_res = logo_img.resize((logo_w, logo_h), Image.LANCZOS)
                
                px = 60
                py = h - 85
                
                box_coords = [px - 12, py - 8, px + logo_w + 12, py + logo_h + 8]
                draw.rounded_rectangle(box_coords, radius=8, fill=(10, 10, 15, 190), outline=(255, 255, 255, 60), width=1)
                overlay.paste(logo_res, (px, py), mask=logo_res)
                
            elif pos_name == "top_right":
                logo_h = 35
                scale = logo_h / logo_img.height
                logo_w = int(logo_img.width * scale)
                logo_res = logo_img.resize((logo_w, logo_h), Image.LANCZOS)
                
                px = w - logo_w - 300
                py = 42
                
                box_coords = [px - 12, py - 8, px + logo_w + 12, py + logo_h + 8]
                draw.rounded_rectangle(box_coords, radius=8, fill=(10, 10, 15, 190), outline=(*accent_color, 100), width=1)
                overlay.paste(logo_res, (px, py), mask=logo_res)

    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

# ── RENDERING ─────────────────────────────────────────────────────────────────

def _render_design_system_thumbnail(hook_text, bg_img, avatar_img, accent_color, width, height, script_json=None, is_shorts=False, variant_style="curiosity", template: ThumbnailTemplate = None, ai_background: Optional[Image.Image] = None):
    """
    Renders thumbnail using Design System:
    - 3-Zone Layout: Hook Text (Left) | Hero Graphic / 3D Card (Center) | Avatar (Right)
    - Safe zones: Strict margin on bottom-right to prevent YouTube duration badge occlusion
    - Template-specific visual treatment
    - Dynamic typography & accent highlighting
    """
    if template is None and script_json:
        template = _detect_template_type(script_json)
    elif template is None:
        template = ThumbnailTemplate.ARCHITECTURE_SHIFT
    
    primary_accent, secondary_accent = _get_accent_colors(template)
    
    # 1. Base canvas (with optional Imagen 3 AI background)
    canvas = _render_thematic_background(width, height, template, primary_accent, secondary_accent, 
                                        script_json=script_json, ai_background=ai_background)
    draw = ImageDraw.Draw(canvas)
    
    # 2. Avatar Preparation & Safe Positioning
    av_res = None
    avatar_pos = None
    face_region = None
    av_allocated_w = 0

    if avatar_img and not is_shorts:
        av_h = int(height * 0.74)
        scale = av_h / avatar_img.height
        av_res = avatar_img.resize((int(avatar_img.width * scale), av_h), Image.LANCZOS)
        av_allocated_w = av_res.width + 30
        # Positioned with YouTube timestamp safe zone (leaves bottom-right 180x80px clear)
        avatar_pos = (width - av_res.width - 25, height - av_res.height - 25)

    # ============================================================
    # ZONE 1: HOOK TEXT (Left side - strictly bounded)
    # ============================================================
    text_x = 55
    if not is_shorts:
        text_max_width = int(width * 0.40) if av_allocated_w > 0 else int(width * 0.48)
    else:
        text_max_width = width - 110

    _render_hook_text_zone(canvas, draw, hook_text, primary_accent, secondary_accent,
                           text_x, 80, text_max_width, height - 190, is_shorts, script_json=script_json)

    # ============================================================
    # ZONE 2: HERO GRAPHIC / 3D EVIDENCE CARD (Center column)
    # Never overlaps Hook Text on the left or Avatar on the right
    # ============================================================
    if not is_shorts:
        hero_x_start = text_x + text_max_width + 25
        if av_allocated_w > 0:
            hero_width = (width - av_allocated_w) - hero_x_start - 25
        else:
            hero_width = width - hero_x_start - 50
        hero_height = height - 195
    else:
        hero_x_start = 50
        hero_width = width - 100
        hero_height = height // 3

    canvas = _render_hero_graphic(canvas, draw, template, primary_accent, secondary_accent, 
                                  hero_x_start, 65, hero_width, hero_height, script_json)
    draw = ImageDraw.Draw(canvas)

    # ============================================================
    # ZONE 3: CODE/DATA BADGE (Bottom foreground)
    # SAFE ZONE: Ends at least 260px from right edge, avoiding YouTube duration badge
    # ============================================================
    badge_max_w = min(hero_x_start + hero_width - 55, width - 260)
    _render_code_badge(canvas, draw, template, primary_accent, secondary_accent,
                       55, height - 128, max(360, badge_max_w), 85, script_json, is_shorts)
    
    # ============================================================
    # AVATAR + EMOTION (Right column overlay)
    # ============================================================
    if av_res and avatar_pos:
        canvas = _draw_multi_tier_glow(canvas, av_res, avatar_pos, primary_accent)
        draw = ImageDraw.Draw(canvas)
        
        avatar_crop = canvas.crop((avatar_pos[0], avatar_pos[1], avatar_pos[0] + av_res.width, avatar_pos[1] + av_res.height))
        face_in_avatar = _detect_face_region(avatar_crop)
        if face_in_avatar:
            face_region = (avatar_pos[0] + face_in_avatar[0], avatar_pos[1] + face_in_avatar[1],
                          face_in_avatar[2], face_in_avatar[3])
    
    emotion = _select_emotion_for_content(
        script_json.get("title", "") if script_json else "", hook_text
    )
    if face_region:
        canvas = _apply_emotion_overlay(canvas, face_region, emotion, primary_accent, is_shorts)
        draw = ImageDraw.Draw(canvas)
    
    # Bottom subtle accent line
    draw.rectangle([0, height-8, width, height], fill=primary_accent)
    
    contrast_score = _calculate_contrast_ratio(TEXT_WHITE, BG_OBSIDIAN)
    rule_of_thirds_score = 0.85
    text_word_count = len(hook_text.replace("\n", " ").split())
    
    metadata = {
        "face_detected": face_region is not None,
        "face_region": face_region,
        "emotion": emotion,
        "contrast_score": round(contrast_score, 2),
        "rule_of_thirds_score": round(rule_of_thirds_score, 2),
        "text_word_count": text_word_count,
        "variant_style": variant_style,
        "avatar_position": avatar_pos,
        "template_type": template.value
    }
    
    return canvas, metadata


def _render_hero_graphic(canvas: Image.Image, draw: ImageDraw.Draw, template: ThumbnailTemplate,
                         primary_accent: Tuple, secondary_accent: Tuple,
                         x: int, y: int, w: int, h: int, script_json: dict) -> Image.Image:
    """Render template-specific hero graphic or 3D evidence card in Zone 2."""
    if w <= 100 or h <= 80:
        return canvas
    
    # For DEEP_DIVE, or if an article screenshot is available, render 3D Tilted Evidence Card
    has_screenshot = False
    if script_json:
        ss_path = (script_json.get("screenshot_path") or 
                   script_json.get("article_screenshot") or 
                   script_json.get("thumbnail_bg"))
        if ss_path and os.path.exists(ss_path):
            has_screenshot = True
        else:
            ss_dir = os.path.join(ASSETS_DIR, "screenshots")
            if os.path.exists(ss_dir) and any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(ss_dir)):
                has_screenshot = True
    
    if template == ThumbnailTemplate.DEEP_DIVE or (has_screenshot and template in (ThumbnailTemplate.BREAKTHROUGH, ThumbnailTemplate.DEV_BENCHMARK)):
        canvas = _render_tilted_article_card(canvas, script_json, x, y, w, h, primary_accent, secondary_accent, tilt_angle=-5.0)
        return canvas

    if template == ThumbnailTemplate.VS_SHOWDOWN:
        _render_vs_hero(draw, primary_accent, secondary_accent, x, y, w, h, script_json)
    
    elif template == ThumbnailTemplate.BREAKTHROUGH:
        _render_breakthrough_hero(canvas, draw, primary_accent, secondary_accent, x, y, w, h, script_json)
        
    elif template == ThumbnailTemplate.DEV_BENCHMARK:
        _render_benchmark_hero(draw, primary_accent, secondary_accent, x, y, w, h, script_json)
        
    elif template == ThumbnailTemplate.COST_OPTIMIZATION:
        _render_cost_hero(draw, primary_accent, secondary_accent, x, y, w, h, script_json)
    
    elif template == ThumbnailTemplate.SECURITY_WARNING:
        _render_security_hero(draw, primary_accent, secondary_accent, x, y, w, h)
        
    else:  # ARCHITECTURE_SHIFT
        _render_architecture_hero(draw, primary_accent, secondary_accent, x, y, w, h)

    return canvas


def _render_architecture_hero(draw: ImageDraw.Draw, primary: Tuple, secondary: Tuple, 
                              x: int, y: int, w: int, h: int):
    """Template A: Architecture shift - Old vs New split with arrow."""
    mid_x = x + w // 2
    
    # LEFT: Old/Broken (Red/X)
    left_w = w // 2 - 30
    left_x = x
    left_y = y + 40
    box_h = h - 80
    
    # Red X background
    draw.rounded_rectangle([left_x, left_y, left_x + left_w, left_y + box_h], 
                          radius=16, fill=(30, 5, 10, 200), outline=(255, 42, 109, 180), width=3)
    
    # Large X mark
    cx, cy = left_x + left_w // 2, left_y + box_h // 2
    x_size = min(left_w, box_h) // 3
    draw.line([(cx - x_size, cy - x_size), (cx + x_size, cy + x_size)], fill=(255, 42, 109, 255), width=8)
    draw.line([(cx + x_size, cy - x_size), (cx - x_size, cy + x_size)], fill=(255, 42, 109, 255), width=8)
    
    # Label: "OLD" or "RAG"
    f_label = _load_font(36, "extrabold")
    draw.text((cx - 40, cy + x_size + 20), "OLD WAY", font=f_label, fill=(255, 100, 100, 255))
    
    # RIGHT: New/Upgraded (Green/Check)
    right_x = mid_x + 30
    right_w = w // 2 - 30
    right_y = y + 40
    
    draw.rounded_rectangle([right_x, right_y, right_x + right_w, right_y + box_h], 
                          radius=16, fill=(5, 30, 15, 200), outline=(0, 255, 135, 180), width=3)
    
    # Check mark / glowing node
    cx2, cy2 = right_x + right_w // 2, right_y + box_h // 2
    # Glowing node
    for r in range(30, 10, -4):
        alpha = int(100 * (30 - r) / 20)
        draw.ellipse([cx2 - r, cy2 - r, cx2 + r, cy2 + r], fill=(*primary, alpha))
    draw.ellipse([cx2 - 12, cy2 - 12, cx2 + 12, cy2 + 12], fill=(0, 255, 135, 255))
    
    # Label: "NEW" or "AGENTIC"
    draw.text((cx2 - 50, cy2 + 35), "NEW WAY", font=f_label, fill=(100, 255, 180, 255))
    
    # ARROW connecting them (center)
    arrow_y = y + h // 2
    _draw_design_arrow(draw, (mid_x - 40, arrow_y), (mid_x + 40, arrow_y), primary)


def _render_cost_hero(draw: ImageDraw.Draw, primary: Tuple, secondary: Tuple,
                      x: int, y: int, w: int, h: int, script_json: dict):
    """Template B: Cost optimization - Crossed out price vs $0."""
    mid_x = x + w // 2
    
    # LEFT: Expensive (Crossed out)
    left_w = w // 2 - 30
    left_x = x
    left_y = y + 40
    box_h = h - 80
    
    draw.rounded_rectangle([left_x, left_y, left_x + left_w, left_y + box_h], 
                          radius=16, fill=(30, 15, 5, 200), outline=(255, 183, 3, 180), width=3)
    
    cx, cy = left_x + left_w // 2, left_y + box_h // 2
    # Dollar amount
    f_money = _load_font(48, "extrabold")
    draw.text((cx - 80, cy - 40), "$24,000", font=f_money, fill=(255, 183, 3, 255))
    draw.text((cx - 50, cy + 10), "/yr", font=f_money, fill=(255, 150, 50, 255))
    
    # Red diagonal strikethrough
    draw.line([(cx - 90, cy - 30), (cx + 90, cy + 50)], fill=(255, 42, 109, 255), width=6)
    
    # RIGHT: Free/$0 (Green)
    right_x = mid_x + 30
    right_w = w // 2 - 30
    right_y = y + 40
    
    draw.rounded_rectangle([right_x, right_y, right_x + right_w, right_y + box_h], 
                          radius=16, fill=(5, 30, 15, 200), outline=(0, 255, 135, 180), width=3)
    
    cx2, cy2 = right_x + right_w // 2, right_y + box_h // 2
    f_free = _load_font(56, "extrabold")
    draw.text((cx2 - 50, cy2 - 30), "$0", font=f_free, fill=(0, 255, 135, 255))
    draw.text((cx2 - 80, cy2 + 30), "OPEN SOURCE", font=_load_font(28, "extrabold"), fill=(100, 255, 180, 255))
    
    # GitHub icon indicator
    draw.ellipse([cx2 - 15, cy2 + 70, cx2 + 15, cy2 + 100], fill=(0, 255, 135, 100), outline=(0, 255, 135, 255), width=2)
    
    # ARROW
    arrow_y = y + h // 2
    _draw_design_arrow(draw, (mid_x - 40, arrow_y), (mid_x + 40, arrow_y), primary)


def _render_security_hero(draw: ImageDraw.Draw, primary: Tuple, secondary: Tuple,
                          x: int, y: int, w: int, h: int):
    """Template C: Security warning - Hazard/Terminal style."""
    # Full width hazard background
    box_h = h - 80
    draw.rounded_rectangle([x, y + 40, x + w, y + 40 + box_h], 
                          radius=16, fill=(30, 5, 10, 220), outline=(255, 42, 109, 200), width=4)
    
    cx, cy = x + w // 2, y + 40 + box_h // 2
    
    # Warning triangle
    tri_size = 60
    triangle = [
        (cx, cy - tri_size),
        (cx - tri_size, cy + tri_size),
        (cx + tri_size, cy + tri_size)
    ]
    draw.polygon(triangle, fill=(255, 183, 3, 255))
    draw.polygon(triangle, outline=(255, 42, 109, 255), width=4)
    
    # Exclamation mark in triangle
    f_excl = _load_font(60, "black")
    draw.text((cx - 12, cy - tri_size + 10), "!", font=f_excl, fill=(20, 5, 10, 255))
    
    # Terminal-style alert lines
    f_term = _load_font(24, "mono")
    alerts = [
        "[ALERT] DATA_EXPOSED",
        "[CRITICAL] VULN_DETECTED", 
        "[WARN] CONTAINMENT_FAILED"
    ]
    for i, alert in enumerate(alerts):
        ay = cy + 40 + i * 35
        # Terminal pill background
        tw, th = _text_size(alert, f_term)
        draw.rounded_rectangle([cx - tw//2 - 15, ay - 5, cx + tw//2 + 15, ay + th + 5], 
                              radius=6, fill=(255, 42, 109, 180), outline=(255, 42, 109, 255), width=1)
        draw.text((cx - tw//2, ay), alert, font=f_term, fill=(255, 255, 255, 255))


def _render_vs_hero(draw: ImageDraw.Draw, primary: Tuple, secondary: Tuple,
                    x: int, y: int, w: int, h: int, script_json: dict):
    """Template D: Head-to-head duel/showdown with glowing VS badge & comparison meters."""
    # Extract entities/companies if present
    entities = []
    if script_json:
        if "companies" in script_json and script_json["companies"]:
            entities = [c if isinstance(c, str) else c.get("name", "") for c in script_json["companies"] if c]
        elif "keywords" in script_json and script_json["keywords"]:
            entities = script_json["keywords"][:2]
    
    left_name = entities[0].upper()[:10] if len(entities) > 0 else "MODEL A"
    right_name = entities[1].upper()[:10] if len(entities) > 1 else ("NVIDIA" if "DEEPSEEK" in left_name else "RIVAL")
    
    card_w = max(90, w // 2 - 25)
    card_h = h - 60
    left_x = x
    right_x = x + w - card_w
    top_y = y + 30
    
    # Left combatant card (Primary Violet accent)
    draw.rounded_rectangle([left_x, top_y, left_x + card_w, top_y + card_h], 
                          radius=14, fill=(15, 8, 25, 220), outline=(*primary, 200), width=2)
    # Left header & stat bar
    f_entity = _load_font(28, "extrabold")
    lw = _text_size(left_name, f_entity)[0]
    draw.text((left_x + (card_w - lw)//2, top_y + 20), left_name, font=f_entity, fill=(255, 255, 255))
    
    # Stat fill bar (Left - 98%)
    draw.rounded_rectangle([left_x + 15, top_y + card_h - 40, left_x + card_w - 15, top_y + card_h - 22],
                          radius=6, fill=(25, 15, 35, 255))
    draw.rounded_rectangle([left_x + 15, top_y + card_h - 40, left_x + int((card_w - 30) * 0.95), top_y + card_h - 22],
                          radius=6, fill=primary)
    
    # Right combatant card (Secondary Amber accent)
    draw.rounded_rectangle([right_x, top_y, right_x + card_w, top_y + card_h], 
                          radius=14, fill=(25, 18, 10, 220), outline=(*secondary, 200), width=2)
    rw = _text_size(right_name, f_entity)[0]
    draw.text((right_x + (card_w - rw)//2, top_y + 20), right_name, font=f_entity, fill=(255, 255, 255))
    
    # Stat fill bar (Right - 72%)
    draw.rounded_rectangle([right_x + 15, top_y + card_h - 40, right_x + card_w - 15, top_y + card_h - 22],
                          radius=6, fill=(35, 25, 15, 255))
    draw.rounded_rectangle([right_x + 15, top_y + card_h - 40, right_x + int((card_w - 30) * 0.72), top_y + card_h - 22],
                          radius=6, fill=secondary)
    
    # Center glowing circular VS badge
    cx = x + w // 2
    cy = top_y + card_h // 2
    
    # Outer glow rings
    for r in range(45, 25, -4):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 42, 109, int(90 * (45 - r) / 20)))
    # Core VS pill
    draw.ellipse([cx - 28, cy - 28, cx + 28, cy + 28], fill=(255, 30, 80, 255), outline=(255, 255, 255, 240), width=2)
    f_vs = _load_font(26, "black")
    vw, vh = _text_size("VS", f_vs)
    draw.text((cx - vw//2, cy - vh//2), "VS", font=f_vs, fill=(255, 255, 255))


def _render_breakthrough_hero(canvas: Image.Image, draw: ImageDraw.Draw, primary: Tuple, secondary: Tuple,
                              x: int, y: int, w: int, h: int, script_json: dict):
    """Template E: Breakthrough / Paradigm Leap with glowing orbital rings & quantum energy core."""
    cx = x + w // 2
    cy = y + h // 2
    
    # Gyroscopic orbital rings
    for rx, ry in [(w // 3, h // 5), (w // 4, h // 3), (w // 3, h // 3)]:
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=(*primary, 140), width=2)
    
    # Radiating particle nodes
    for angle_deg in range(0, 360, 45):
        rad = math.radians(angle_deg)
        px = int(cx + (w // 3.2) * math.cos(rad))
        py = int(cy + (h // 4.5) * math.sin(rad))
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(*secondary, 220))
    
    # Intense energy core
    for r in range(40, 10, -5):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*primary, int(150 * (40 - r) / 30)))
    draw.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=(255, 255, 255, 255))
    
    # Status Pill Badge above
    pill_text = "[ AGI BREAKTHROUGH // 2026 ]"
    f_pill = _load_font(20, "extrabold")
    pw, ph = _text_size(pill_text, f_pill)
    pill_y = y + 25
    draw.rounded_rectangle([cx - pw//2 - 12, pill_y - 4, cx + pw//2 + 12, pill_y + ph + 4],
                          radius=6, fill=(15, 12, 28, 220), outline=(*primary, 180), width=1)
    draw.text((cx - pw//2, pill_y), pill_text, font=f_pill, fill=(*secondary, 255))


def _render_benchmark_hero(draw: ImageDraw.Draw, primary: Tuple, secondary: Tuple,
                           x: int, y: int, w: int, h: int, script_json: dict):
    """Template F: Developer Benchmark HUD card with metric comparison bars and IDE titlebar."""
    card_w = w
    card_h = h - 60
    top_y = y + 30
    
    # Terminal Window Card
    draw.rounded_rectangle([x, top_y, x + card_w, top_y + card_h],
                          radius=12, fill=(10, 16, 24, 230), outline=(*primary, 150), width=2)
    
    # macOS window buttons
    draw.ellipse([x + 16, top_y + 14, x + 26, top_y + 24], fill=(255, 95, 87, 240))
    draw.ellipse([x + 32, top_y + 14, x + 42, top_y + 24], fill=(254, 188, 46, 240))
    draw.ellipse([x + 48, top_y + 14, x + 58, top_y + 24], fill=(40, 201, 64, 240))
    
    f_mono = _load_font(18, "mono")
    draw.text((x + 68, top_y + 11), "eval_benchmark.py", font=f_mono, fill=(160, 180, 200, 220))
    draw.line([(x, top_y + 36), (x + card_w, top_y + 36)], fill=(30, 42, 58, 200), width=1)
    
    # Benchmark Bars
    f_bar = _load_font(20, "extrabold")
    by1 = top_y + 55
    draw.text((x + 20, by1), "SOTA MODEL: 99.4%", font=f_bar, fill=(255, 255, 255))
    bar_w = card_w - 40
    draw.rounded_rectangle([x + 20, by1 + 25, x + 20 + bar_w, by1 + 40], radius=5, fill=(20, 30, 45, 255))
    draw.rounded_rectangle([x + 20, by1 + 25, x + 20 + int(bar_w * 0.96), by1 + 40], radius=5, fill=primary)
    
    by2 = by1 + 55
    draw.text((x + 20, by2), "BASELINE:   72.1%", font=f_bar, fill=(160, 170, 185))
    draw.rounded_rectangle([x + 20, by2 + 25, x + 20 + bar_w, by2 + 40], radius=5, fill=(20, 30, 45, 255))
    draw.rounded_rectangle([x + 20, by2 + 25, x + 20 + int(bar_w * 0.70), by2 + 40], radius=5, fill=(80, 100, 120, 200))
    
    # Gain Pill
    gain_pill = "+27.3% GAIN // 10X SPEED"
    pw, ph = _text_size(gain_pill, f_mono)
    draw.rounded_rectangle([x + 20, by2 + 55, x + 20 + pw + 16, by2 + 55 + ph + 8],
                          radius=6, fill=(0, 255, 135, 40), outline=(*primary, 180), width=1)
    draw.text((x + 28, by2 + 59), gain_pill, font=f_mono, fill=(*primary, 255))


def _render_deep_dive_hero(draw: ImageDraw.Draw, primary: Tuple, secondary: Tuple,
                           x: int, y: int, w: int, h: int, script_json: dict):
    """Template G: Investigative Dossier with classified stencil stamp & radar reticle."""
    card_w = w
    card_h = h - 60
    top_y = y + 30
    
    draw.rounded_rectangle([x, top_y, x + card_w, top_y + card_h],
                          radius=12, fill=(25, 8, 14, 230), outline=(*primary, 180), width=2)
    
    # Radar reticle in corner
    rcx, rcy = x + card_w - 60, top_y + 60
    for rr in [45, 30, 15]:
        draw.ellipse([rcx - rr, rcy - rr, rcx + rr, rcy + rr], outline=(*secondary, 100), width=1)
    draw.line([(rcx - 50, rcy), (rcx + 50, rcy)], fill=(*secondary, 80), width=1)
    draw.line([(rcx, rcy - 50), (rcx, rcy + 50)], fill=(*secondary, 80), width=1)
    
    # Giant Red Stencil Stamp
    f_stamp = _load_font(42, "black")
    stamp_txt = "CONFIDENTIAL"
    sw, sh = _text_size(stamp_txt, f_stamp)
    sx = x + (card_w - sw) // 2
    sy = top_y + card_h // 2 - sh // 2
    
    # Red border box around stamp
    draw.rounded_rectangle([sx - 15, sy - 8, sx + sw + 15, sy + sh + 8],
                          radius=8, outline=(255, 30, 80, 255), width=3)
    draw.text((sx, sy), stamp_txt, font=f_stamp, fill=(255, 50, 90, 240))
    
    # Dossier tag
    f_tag = _load_font(18, "mono")
    draw.text((x + 20, top_y + card_h - 32), "[ FILE // LEAKED_DATA_AUDIT ]", font=f_tag, fill=(255, 200, 200, 200))


def _extract_primary_entities(script_json: Optional[dict]) -> Tuple[str, str]:
    """Dynamically extracts top 2 primary entity/library names from script_json."""
    if not script_json:
        return "deepseek_r1", "pytorch"
    
    candidates = []
    # 1. From companies, entities, keywords, tools
    for key in ("entities", "companies", "tools", "libraries", "keywords"):
        items = script_json.get(key, [])
        if isinstance(items, list):
            for it in items:
                name = it if isinstance(it, str) else it.get("name", "")
                if name and len(name) > 2 and name.lower() not in ("ai", "the", "and", "model", "tech", "data"):
                    candidates.append(name.strip())
    
    # 2. Known high-impact tech entities in title
    title = script_json.get("title", "")
    known_tech = ["DeepSeek", "NVIDIA", "OpenAI", "Claude", "Gemini", "PyTorch", "vLLM", 
                  "Ollama", "LangGraph", "Llama", "Whisper", "Triton", "FastAPI", "React",
                  "HuggingFace", "Mistral", "Qwen", "Apple", "Google", "Anthropic", "Grok"]
    for kt in known_tech:
        if kt.lower() in title.lower() and kt not in candidates:
            candidates.append(kt)
            
    if not candidates:
        words = [w.strip("?,!.:;\"'") for w in title.split() if len(w) > 3 and w.lower() not in ("what", "this", "your", "with", "from")]
        candidates = words if words else ["deepseek", "vllm"]
        
    e1 = candidates[0].lower().replace(" ", "_").replace("-", "_")
    e2 = candidates[1].lower().replace(" ", "_").replace("-", "_") if len(candidates) > 1 else "cuda_core"
    return e1, e2


def _render_hook_text_zone(canvas: Image.Image, draw: ImageDraw.Draw, hook_text: str,
                           primary: Tuple, secondary: Tuple,
                           x: int, y: int, max_w: int, max_h: int, is_shorts: bool,
                           script_json: Optional[dict] = None):
    """Render Zone 1: Hook text with dynamic keyword accent highlighting."""
    lines = hook_text.split("\n")
    
    if is_shorts:
        font_size = 110
        line_spacing = 15
    else:
        font_size = 90
        line_spacing = 10
    
    font = _load_font(font_size, "extrabold")
    total_h = sum(_text_size(l, font)[1] for l in lines) + line_spacing * (len(lines) - 1)
    start_y = y + max(0, (max_h - total_h) // 2)
    
    # Base high-impact accent keywords
    accent_words = {"DEAD", "OUT", "GONE", "NEW", "SHIFT", "FREE", "$0", "SAVE", "90%", 
                    "LEAK", "EXPOSED", "CRITICAL", "WARNING", "ALERT", "STOP", "NOW",
                    "VS", "REPLACED", "SECRET", "BANNED", "SOLVED", "DESTROYED", "BEAST",
                    "AGI", "INSANE", "WTF", "LEAP", "KILLER", "WINS", "FALLS", "SHOCKING",
                    "TRUTH", "FATAL", "NEVER", "PANIC", "COSTS", "DANGER", "CRUSHES"}
    
    # Dynamically inject major brand entities
    major_brands = {"NVIDIA", "OPENAI", "DEEPSEEK", "CLAUDE", "GEMINI", "GOOGLE", "APPLE", 
                    "META", "GROK", "MISTRAL", "QWEN", "PYTORCH", "VLLM", "OLLAMA", "ANTHROPIC", 
                    "R1", "V3", "GPT-5", "RAG", "LLM", "GPU", "CUDA", "BLACKWELL"}
    accent_words.update(major_brands)
    
    # Dynamically inject detected entities/companies from script_json
    if script_json:
        if "companies" in script_json:
            for c in script_json["companies"]:
                cname = c if isinstance(c, str) else c.get("name", "")
                if cname:
                    accent_words.add(cname.upper())
        if "keywords" in script_json:
            for kw in script_json["keywords"]:
                for kw_word in kw.split():
                    if len(kw_word) > 2:
                        accent_words.add(kw_word.upper())
        if "entities" in script_json:
            for ent in script_json["entities"]:
                ename = ent if isinstance(ent, str) else ent.get("name", "")
                if ename:
                    accent_words.add(ename.upper())
        title = script_json.get("title", "")
        for w in title.split():
            clean_w = w.strip("?,!.:;\"'()")
            if clean_w.isupper() and len(clean_w) > 1:
                accent_words.add(clean_w)
    
    for idx, line in enumerate(lines):
        words = line.split()
        cur_x = x
        line_h = _text_size(line, font)[1]
        
        for word in words:
            word_w, word_h = _text_size(word + " ", font)
            clean_word = word.strip("?,!.:;\"'()")
            is_accent = clean_word.upper() in accent_words
            
            if is_accent:
                txt_color = primary
                pill_pad = 12
                pill_coords = [cur_x - pill_pad, start_y - 4, cur_x + word_w + pill_pad, start_y + line_h + 4]
                block_overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
                block_draw = ImageDraw.Draw(block_overlay)
                block_draw.rounded_rectangle(pill_coords, radius=8, fill=(*primary, 40), outline=(*primary, 180), width=2)
                canvas = Image.alpha_composite(canvas.convert("RGBA"), block_overlay).convert("RGB")
                draw = ImageDraw.Draw(canvas)
            else:
                txt_color = TEXT_WHITE
            
            for offset in range(1, 5):
                draw.text((cur_x + offset, start_y + offset), word, font=font, fill=(0, 0, 0, 170))
            draw.text((cur_x, start_y), word, font=font, fill=txt_color)
            
            cur_x += word_w
        
        start_y += line_h + line_spacing


def _render_code_badge(canvas: Image.Image, draw: ImageDraw.Draw, template: ThumbnailTemplate,
                       primary: Tuple, secondary: Tuple,
                       x: int, y: int, w: int, h: int, script_json: dict, is_shorts: bool):
    """Render Zone 3: Code/Data badge at bottom (monospace) with dynamic contextual content."""
    e1, e2 = _extract_primary_entities(script_json)
    
    # Highly specific, authentic topic-relevant snippets
    badge_content = {
        ThumbnailTemplate.VS_SHOWDOWN: f"{e1}_vs_{e2}.py  |  delta: +32.4%  |  sota_eval=pass",
        ThumbnailTemplate.BREAKTHROUGH: f"import {e1}  |  {e1}.generate()  |  latency: 3.2ms",
        ThumbnailTemplate.DEV_BENCHMARK: f"python -m {e1}.eval  |  throughput: 1,840 tok/s  |  speedup: 4.8x",
        ThumbnailTemplate.DEEP_DIVE: f"audit_{e1}.log  |  sha256: 4f8a...9c  |  integrity: 100%",
        ThumbnailTemplate.ARCHITECTURE_SHIFT: f"{e1}_graph.rs  |  {e2}_engine=active  |  v3_migration=done",
        ThumbnailTemplate.COST_OPTIMIZATION: f"{e1}.quantize(int4)  |  cloud_bill: $0.00/mo  |  saved: 88%",
        ThumbnailTemplate.SECURITY_WARNING: f"[CRITICAL_ALERT] {e1}_vuln_patched  |  cve_2026=contained",
    }
    
    code_text = badge_content.get(template, f"{e1}.run()  |  pipeline.execute()  |  status=active")
    
    # Badge background
    draw.rounded_rectangle([x - 10, y - 8, x + w + 10, y + h + 8], 
                          radius=12, fill=(5, 8, 15, 230), outline=(*primary, 120), width=2)
    
    # Corner brackets (Figma HUD style)
    bracket_len = 20
    draw.line([(x, y), (x + bracket_len, y)], fill=(*primary, 100), width=2)
    draw.line([(x, y), (x, y + bracket_len)], fill=(*primary, 100), width=2)
    draw.line([(x + w, y), (x + w - bracket_len, y)], fill=(*primary, 100), width=2)
    draw.line([(x + w, y), (x + w, y + bracket_len)], fill=(*primary, 100), width=2)
    draw.line([(x, y + h), (x + bracket_len, y + h)], fill=(*primary, 100), width=2)
    draw.line([(x, y + h), (x, y + h - bracket_len)], fill=(*primary, 100), width=2)
    draw.line([(x + w, y + h), (x + w - bracket_len, y + h)], fill=(*primary, 100), width=2)
    draw.line([(x + w, y + h), (x + w, y + h - bracket_len)], fill=(*primary, 100), width=2)
    
    f_mono = _load_font(24 if not is_shorts else 28, "mono")
    tw, th = _text_size(code_text, f_mono)
    tx = x + max(15, (w - tw) // 2)
    ty = y + (h - th) // 2
    
    # Syntax highlighting simulation - dim comments
    parts = code_text.split("  |  ")
    cur_tx = tx
    for i, part in enumerate(parts):
        color = TEXT_OFF_WHITE if i == 0 else (*secondary, 200)
        for offset in range(1, 3):
            draw.text((cur_tx + offset, ty + offset), part, font=f_mono, fill=(0, 0, 0, 180))
        draw.text((cur_tx, ty), part, font=f_mono, fill=color)
        cur_tx += _text_size(part + "  |  ", f_mono)[0]


def _draw_design_arrow(draw: ImageDraw.Draw, start: Tuple, end: Tuple, accent: Tuple):
    """Draw a clean design-system arrow with glow."""
    # Glow
    for gw in range(12, 4, -2):
        alpha = int(60 * (12 - gw) / 8)
        draw.line([start, end], fill=(*accent, alpha), width=gw)
    # Main line
    draw.line([start, end], fill=(*accent, 255), width=6)
    
    # Arrowhead
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    arrow_len = 25
    angle_off = math.pi / 6
    p1 = (end[0] - arrow_len * math.cos(angle - angle_off), end[1] - arrow_len * math.sin(angle - angle_off))
    p2 = (end[0] - arrow_len * math.cos(angle + angle_off), end[1] - arrow_len * math.sin(angle + angle_off))
    
    for gw in range(10, 4, -2):
        alpha = int(60 * (10 - gw) / 6)
        draw.line([p1, end], fill=(*accent, alpha), width=gw)
        draw.line([p2, end], fill=(*accent, alpha), width=gw)
    draw.line([p1, end], fill=(*accent, 255), width=6)
    draw.line([p2, end], fill=(*accent, 255), width=6)

def _draw_neon_arrow(draw, start, end, accent_color, width=12):
    """Draws a premium neon arrow pointing from start to end with glow."""
    # 1. Outer neon glow for the line
    for glow_w in range(width + 8, width, -2):
        alpha = int(70 * ((width + 8 - glow_w) / 8))
        draw.line([start, end], fill=(*accent_color, alpha), width=glow_w)
    # Main arrow line (Crimson Red)
    draw.line([start, end], fill=(255, 32, 32, 255), width=width)
    
    # 2. Calculate arrowhead points
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    
    arrow_len = 45
    angle_offset = math.pi / 6 # 30 degrees
    
    p1 = (end[0] - arrow_len * math.cos(angle - angle_offset),
          end[1] - arrow_len * math.sin(angle - angle_offset))
    p2 = (end[0] - arrow_len * math.cos(angle + angle_offset),
          end[1] - arrow_len * math.sin(angle + angle_offset))
          
    # Outer glow for arrowhead
    for glow_w in range(width + 6, width, -2):
        alpha = int(70 * ((width + 6 - glow_w) / 6))
        draw.line([p1, end], fill=(*accent_color, alpha), width=glow_w)
        draw.line([p2, end], fill=(*accent_color, alpha), width=glow_w)
        
    draw.line([p1, end], fill=(255, 32, 32, 255), width=width)
    draw.line([p2, end], fill=(255, 32, 32, 255), width=width)


def _render_compilation_thumbnail(bg_img, avatar_img, accent_color, width, height, script_json=None):
    """Specialized 16:9 thumbnail for 'Did You Know' 5/10-fact compilations with dynamic hooks."""
    canvas = _render_thematic_background(width, height, ThumbnailTemplate.BREAKTHROUGH, accent_color, (255, 183, 3), script_json=script_json)
    canvas = ImageEnhance.Brightness(canvas).enhance(0.7)
    canvas = _draw_curved_accent(canvas, accent_color)
    draw = ImageDraw.Draw(canvas)
    
    # Measure avatar
    av_allocated_w = 0
    if avatar_img:
        av_h = int(height * 0.88)
        scale = av_h / avatar_img.height
        av_res = avatar_img.resize((int(avatar_img.width * scale), av_h), Image.LANCZOS)
        av_allocated_w = av_res.width + 25
        pos = (width - av_res.width - 25, height - av_res.height - 25)
        canvas = _draw_multi_tier_glow(canvas, av_res, pos, accent_color)
        draw = ImageDraw.Draw(canvas)

    # 3. Dynamic Shocking Topic Hook (Avoid generic "Did You Know" on every video)
    top_hook = None
    if script_json:
        if script_json.get("custom_hook"):
            top_hook = script_json["custom_hook"]
        elif script_json.get("fact_scripts"):
            f0 = script_json["fact_scripts"][0]
            top_hook = f0.get("hook") or f0.get("title") or f0.get("text", "")
        elif script_json.get("title"):
            top_hook = script_json["title"]

    if top_hook:
        clean_hook = top_hook.replace("\\n", " ").strip()
        words = clean_hook.split()
        if len(words) <= 3:
            lines = [clean_hook.upper()]
        elif len(words) <= 6:
            mid = len(words) // 2
            lines = [" ".join(words[:mid]).upper(), " ".join(words[mid:]).upper()]
        else:
            lines = [" ".join(words[:3]).upper(), " ".join(words[3:6]).upper()]
    else:
        topic_kw = "AI SECRETS"
        if script_json and "companies" in script_json and script_json["companies"]:
            c0 = script_json["companies"][0]
            topic_kw = (c0 if isinstance(c0, str) else c0.get("name", "AI")).upper()
        lines = ["THE TRUTH ABOUT", topic_kw]
    
    f_main = _load_font(110, "black")
    y = 110
    x = 70
    
    for idx, line in enumerate(lines[:2]):
        lw, lh = _text_size(line, f_main)
        txt_color = (255, 255, 255) if idx == 0 else (255, 214, 0)
        
        box_coords = [x - 20, y - 10, x + lw + 20, y + lh + 10]
        block_overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        block_draw = ImageDraw.Draw(block_overlay)
        block_draw.rounded_rectangle(box_coords, radius=12, fill=(10, 10, 15, 215), outline=(*accent_color, 160), width=2)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), block_overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        
        for offset in range(1, 8):
            draw.text((x+offset, y+offset), line, font=f_main, fill=(0, 0, 0, 160))
        draw.text((x, y), line, font=f_main, fill=txt_color)
        y += lh + 30

    # 4. Render Dynamic Fact Badge ("10 CRAZY AI FACTS")
    f_sub = _load_font(52, "extrabold")
    num_facts = 10
    if script_json:
        num_facts = script_json.get("num_facts", len(script_json.get("fact_scripts", [])) or 10)
    sub_txt = f"{num_facts} SHOCKING AI SECRETS"
    sub_w, sub_h = _text_size(sub_txt, f_sub)
    
    badge_x = 70
    badge_y = y + 30
    
    draw.rounded_rectangle([badge_x - 18, badge_y - 8, badge_x + sub_w + 18, badge_y + sub_h + 14], 
                           radius=18, fill=(220, 20, 60, 245), outline=(255, 100, 130, 255), width=2)
                           
    for offset in range(1, 4):
        draw.text((badge_x+offset, badge_y+offset), sub_txt, font=f_sub, fill=(0, 0, 0, 120))
    draw.text((badge_x, badge_y), sub_txt, font=f_sub, fill=(255, 255, 255, 255))

    # 5. Glowing pointer arrow toward avatar
    if av_allocated_w > 0:
        try:
            arrow_start = (width // 2 - 120, height // 2 + 110)
            arrow_end = (width - av_allocated_w - 40, height // 2 - 20)
            _draw_neon_arrow(draw, arrow_start, arrow_end, accent_color, width=12)
        except Exception as e:
            print(f"⚠️ Arrow rendering failed: {e}")

    # 6. Vector Tech Alert Badge (100% Vector Drawn — cross-platform safe, zero headless Linux emoji issues)
    try:
        badge_cx = width // 2 - 130
        badge_cy = height // 2 - 50
        for r in range(40, 20, -4):
            draw.ellipse([badge_cx - r, badge_cy - r, badge_cx + r, badge_cy + r], fill=(255, 183, 3, int(80 * (40 - r) / 20)))
        draw.ellipse([badge_cx - 24, badge_cy - 24, badge_cx + 24, badge_cy + 24], fill=(255, 183, 3, 255), outline=(255, 255, 255), width=2)
        f_excl = _load_font(34, "black")
        ew, eh = _text_size("!", f_excl)
        draw.text((badge_cx - ew//2, badge_cy - eh//2 - 2), "!", font=f_excl, fill=(15, 10, 20, 255))
    except Exception as e:
        print(f"⚠️ Alert badge rendering failed: {e}")

    # 7. Branding Accent Bar at Bottom
    draw.rectangle([0, height-10, width, height], fill=accent_color)
    
    # 8. Render Logos
    if script_json:
        canvas = _draw_logo_badges(canvas, script_json, accent_color, is_shorts=False)
        
    return canvas

def _save_variant_metadata(variants: List[ThumbnailVariant], base_path: str):
    """Save A/B test metadata for thumbnail variants."""
    metadata = {
        "test_id": hashlib.md5(f"{base_path}{datetime.now().isoformat()}".encode()).hexdigest()[:12],
        "created_at": datetime.now().isoformat(),
        "variants": [asdict(v) for v in variants],
        "status": "pending_selection",
        "selection_criteria": {
            "min_contrast": HIGH_CONTRAST_RATIO,
            "min_rule_of_thirds": 0.6,
            "max_words": MAX_TEXT_WORDS,
            "face_required": True
        }
    }
    meta_path = base_path.replace(".jpg", "_abtest.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📊 A/B Test Metadata saved: {meta_path}")
    return meta_path


def _select_best_variant(variants: List[ThumbnailVariant]) -> ThumbnailVariant:
    """Select best variant based on heuristic quality scores."""
    scored = []
    for v in variants:
        score = 0
        if v.contrast_score >= HIGH_CONTRAST_RATIO:
            score += 30
        else:
            score += max(0, v.contrast_score / HIGH_CONTRAST_RATIO * 30)
        
        score += v.rule_of_thirds_score * 25
        if v.face_detected:
            score += 30
        
        if v.text_word_count <= MAX_TEXT_WORDS:
            score += 15
        else:
            score += max(0, (MAX_TEXT_WORDS / v.text_word_count) * 15)
        
        scored.append((score, v))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    print(f"🏆 Heuristic Variant scores: {[(v.variant_id, round(s, 1)) for s, v in scored]}")
    return scored[0][1]


def _select_best_variant_with_vision(variants: List[ThumbnailVariant], title: str, client: Optional[genai.Client]) -> ThumbnailVariant:
    """
    Evaluates rendered thumbnail variants with Gemini Vision for:
    1. Curiosity gap & click intent
    2. Mobile readability at 150px thumbnail size
    3. Visual hierarchy, focal point, and contrast
    Selects the winning variant with automatic fallback to heuristic scoring.
    """
    if not variants:
        raise ValueError("No variants to select from")
    if len(variants) == 1:
        return variants[0]
        
    if not client or not GEMINI_API_KEY:
        print("ℹ️ Gemini client unavailable for vision scoring, using heuristic scoring.")
        return _select_best_variant(variants)

    try:
        print(f"👁️ Sending {len(variants)} thumbnail variants to Gemini Vision for CTR evaluation...")
        images_payload = []
        variant_desc = []
        for i, v in enumerate(variants):
            if os.path.exists(v.path):
                img = Image.open(v.path).convert("RGB").resize((640, 360), Image.LANCZOS)
                images_payload.append(img)
                variant_desc.append(f"Image {i+1}: Style '{v.style}', Hook: '{v.hook_text}', Variant ID: '{v.variant_id}'")

        if not images_payload:
            return _select_best_variant(variants)

        prompt = (
            f"You are a top-tier YouTube packaging and CTR optimization expert.\n"
            f"Analyze these {len(images_payload)} YouTube thumbnail variants for a video titled: \"{title}\".\n\n"
            f"Variant Details:\n" + "\n".join(variant_desc) + "\n\n"
            f"Evaluate each thumbnail on a 1-10 scale across three critical dimensions:\n"
            f"1. Curiosity Gap & Click Intent (does it provoke an immediate question in the viewer's mind?)\n"
            f"2. Mobile Readability at 150px size (can a user scrolling on a smartphone read the text in 0.5s?)\n"
            f"3. Visual Hierarchy & Contrast (clear focal point, strong contrast, no zone clutter)\n\n"
            f"Return ONLY valid JSON matching this schema:\n"
            f"{{\n"
            f"  \"winner_index\": 1,\n"
            f"  \"winner_style\": \"curiosity\",\n"
            f"  \"scores\": {{\n"
            f"    \"image_1\": {{\"curiosity\": 8, \"readability\": 9, \"hierarchy\": 8, \"total\": 25}},\n"
            f"    \"image_2\": {{\"curiosity\": 9, \"readability\": 9, \"hierarchy\": 9, \"total\": 27}},\n"
            f"    \"image_3\": {{\"curiosity\": 7, \"readability\": 8, \"hierarchy\": 7, \"total\": 22}}\n"
            f"  }},\n"
            f"  \"reasoning\": \"Brief explanation of why the winner was chosen\"\n"
            f"}}"
        )

        model_name = GEMINI_FLASH_MODEL or "gemini-2.5-flash"
        response = client.models.generate_content(
            model=model_name,
            contents=[*images_payload, prompt]
        )

        raw_text = response.text.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        data = json.loads(raw_text)
        w_idx = int(data.get("winner_index", 1)) - 1
        if 0 <= w_idx < len(variants):
            winner = variants[w_idx]
            print(f"🎯 Gemini Vision selected winner: Variant {w_idx+1} ({winner.style}) - {data.get('reasoning', '')}")
            return winner
    except Exception as e:
        print(f"⚠️ Vision thumbnail evaluation skipped (fallback to heuristic scoring): {e}")

    return _select_best_variant(variants)


def generate_thumbnail(script_json):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    custom_hook = script_json.get("custom_hook") or script_json.get("hook_text")
    title = script_json.get("title", "AI Breakthrough")
    
    template = _detect_template_type(script_json)
    primary_accent, secondary_accent = _get_accent_colors(template)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    custom_suffix = script_json.get("output_suffix", "")
    suffix_str = f"_{custom_suffix}" if custom_suffix else f"_{date_str}"
    
    out_yt = os.path.join(OUTPUT_DIR, f"thumbnail{suffix_str}.jpg")
    out_shorts = os.path.join(OUTPUT_DIR, f"thumbnail_shorts{suffix_str}.jpg")
    
    bg = Image.new("RGB", (THUMB_W, THUMB_H), BG_OBSIDIAN)
    
    avatar_still = script_json.get("avatar_still") or script_json.get("avatar_path")
    still_time = float(script_json.get("avatar_still_time", 1.0))
    avatar = _process_avatar_still(avatar_still, still_time)

    is_compilation = script_json.get("is_longform") and script_json.get("longform_format") == "did_you_know"

    if is_compilation:
        print("🎬 Rendering Long-Form Compilation Thumbnail...")
        yt = _render_compilation_thumbnail(bg, avatar, primary_accent, THUMB_W, THUMB_H, script_json=script_json)
        yt.convert("RGB").save(out_yt, "JPEG", quality=95)
        print(f"✅ Premium Compilation Thumbnail Generated: {out_yt}")
        return out_yt
    else:
        # Generate AI Imagen 3 illustration background if possible
        ai_bg = None
        if not is_compilation and client:
            ai_bg = _generate_imagen_background(script_json, client, template, primary_accent)

        variant_styles = ["authority", "curiosity", "urgency"]
        
        if custom_hook:
            print("📝 Using custom hook text from script_json...")
            base_hook = custom_hook.replace("\\n", "\n")
            hooks_yt = {style: base_hook for style in variant_styles}
            hooks_shorts = {style: base_hook.upper() for style in variant_styles}
        else:
            hooks_yt = {style: _generate_hook_text(title, client, is_shorts=False, variant_style=style, template=template) for style in variant_styles}
            hooks_shorts = {style: _generate_hook_text(title, client, is_shorts=True, variant_style=style, template=template) for style in variant_styles}
        
        # Generate YouTube (16:9) variants
        print(f"🎬 Generating {THUMBNAIL_VARIANTS} YouTube Thumbnail Variants (template: {template.value})...")
        yt_variants = []
        for i, style in enumerate(variant_styles):
            variant_id = f"yt_{style}_{suffix_str}"
            canvas, meta = _render_design_system_thumbnail(
                hooks_yt[style], bg, avatar, primary_accent, THUMB_W, THUMB_H, 
                script_json=script_json, variant_style=style, template=template,
                ai_background=ai_bg
            )
            variant_path = os.path.join(OUTPUT_DIR, f"thumbnail_{style}{suffix_str}.jpg")
            canvas.convert("RGB").save(variant_path, "JPEG", quality=95)
            
            yt_variants.append(ThumbnailVariant(
                variant_id=variant_id,
                path=variant_path,
                style=style,
                hook_text=hooks_yt[style].replace("\n", " | "),
                emotion=meta["emotion"],
                text_word_count=meta["text_word_count"],
                contrast_score=meta["contrast_score"],
                rule_of_thirds_score=meta["rule_of_thirds_score"],
                face_detected=meta["face_detected"],
                created_at=datetime.now().isoformat(),
                template_type=template.value
            ))
            print(f"   ✅ Variant {i+1}/{THUMBNAIL_VARIANTS} ({style}): {variant_path}")
        
        # Select best variant for YouTube using Gemini Vision evaluation (with heuristic fallback)
        best_yt = _select_best_variant_with_vision(yt_variants, title=title, client=client)
        import shutil
        shutil.copy2(best_yt.path, out_yt)
        print(f"🏆 Best YouTube variant: {best_yt.style} (contrast={best_yt.contrast_score}, rot={best_yt.rule_of_thirds_score}, face={best_yt.face_detected})")
        
        _save_variant_metadata(yt_variants, out_yt)
        
        # Generate Shorts (9:16) variants
        print(f"🎬 Generating {THUMBNAIL_VARIANTS} Shorts Thumbnail Variants...")
        bg_vert = Image.new("RGB", (SHORTS_W, SHORTS_H), BG_OBSIDIAN)
        shorts_variants = []
        for i, style in enumerate(variant_styles):
            variant_id = f"shorts_{style}_{suffix_str}"
            canvas, meta = _render_design_system_thumbnail(
                hooks_shorts[style], bg_vert, avatar, primary_accent, SHORTS_W, SHORTS_H, 
                script_json=script_json, is_shorts=True, variant_style=style, template=template
            )
            variant_path = os.path.join(OUTPUT_DIR, f"thumbnail_shorts_{style}{suffix_str}.jpg")
            canvas.convert("RGB").save(variant_path, "JPEG", quality=95)
            
            shorts_variants.append(ThumbnailVariant(
                variant_id=variant_id,
                path=variant_path,
                style=style,
                hook_text=hooks_shorts[style].replace("\n", " | "),
                emotion=meta["emotion"],
                text_word_count=meta["text_word_count"],
                contrast_score=meta["contrast_score"],
                rule_of_thirds_score=meta["rule_of_thirds_score"],
                face_detected=meta["face_detected"],
                created_at=datetime.now().isoformat(),
                template_type=template.value
            ))
            print(f"   ✅ Variant {i+1}/{THUMBNAIL_VARIANTS} ({style}): {variant_path}")
        
        best_shorts = _select_best_variant(shorts_variants)
        shutil.copy2(best_shorts.path, out_shorts)
        print(f"🏆 Best Shorts variant: {best_shorts.style} (contrast={best_shorts.contrast_score}, rot={best_shorts.rule_of_thirds_score}, face={best_shorts.face_detected})")
        
        _save_variant_metadata(shorts_variants, out_shorts)
        
        print(f"✅ Design System Thumbnails Generated: {out_yt} (template: {template.value})")
        return out_yt

# ══════════════════════════════════════════════════════════════════════════════
# AI IMAGE PROMPT GENERATOR (Midjourney / DALL-E)
# Generates a single-line text-to-image prompt for thumbnail creation
# ══════════════════════════════════════════════════════════════════════════════

def generate_thumbnail_prompt(script_json: dict) -> str:
    """
    Generates a Midjourney/DALL-E compatible prompt for long-form video thumbnails.
    
    Rules:
    - 16:9 aspect ratio, rule of thirds, high contrast
    - Max 3-4 words bold text (Yellow/White on dark)
    - Left: "Old/Broken" visual | Right: "New/Upgraded" visual
    - Dark neon blue/black bg with Cyan/Yellow/Neon Red accents
    - Sleek 3D render, clean typography, minimal clutter
    
    Args:
        script_json: The video script data containing title, summary, topics, etc.
        
    Returns:
        Single-line image generation prompt string
    """
    title = script_json.get("title", "AI Breakthrough")
    summary = script_json.get("description", script_json.get("script", ""))[:500]
    topics = script_json.get("longform_topics", [])
    subcat = script_json.get("sub_category", "AI & Tech")
    
    # Extract key entities for visual metaphor
    companies = [c.get("name", "") for c in script_json.get("companies_mentioned", [])]
    tools = [t.get("name", "") for t in script_json.get("tools_mentioned", [])]
    entities = companies + tools
    
    # Determine the "Old vs New" visual metaphor based on content
    old_metaphors = {
        "security": "cracked padlock, shattered firewall, data leaking",
        "privacy": "open window, exposed documents, surveillance camera",
        "legacy": "old server rack, tangled cables, dusty hardware, floppy disk",
        "slow": "hourglass, loading spinner, snail, tortoise",
        "broken": "glitching screen, error 404, crashed system, blue screen",
        "outdated": "typewriter, fax machine, CRT monitor, punch cards",
        "vulnerable": "open vault, broken shield, warning signs, red alerts",
        "inefficient": "paperwork pile, manual process, clipboard, bureaucracy",
        "centralized": "single point of failure, monolith, bottleneck",
        "expensive": "burning money, gold bars leaking, expensive contract",
    }
    
    new_metaphors = {
        "security": "quantum encryption shield, biometric fortress, zero-trust architecture",
        "privacy": "encrypted vault, anonymous mask, zero-knowledge proof, local-first",
        "modern": "sleek serverless cloud, clean fiber optics, edge computing nodes",
        "fast": "lightning bolt, warp speed tunnel, instant sync, real-time stream",
        "fixed": "green checkmark, healed system, seamless flow, auto-recovery",
        "ai-powered": "neural network brain, glowing synapses, AI assistant hologram",
        "efficient": "automated pipeline, one-click deploy, streamlined workflow",
        "decentralized": "distributed mesh, blockchain nodes, peer-to-peer network",
        "cost-effective": "rocket ship growth, compound interest, efficient scaling",
        "breakthrough": "lightbulb moment, eureka spark, paradigm shift portal",
    }
    
    # Analyze content to pick metaphors
    content_lower = (title + " " + summary).lower()
    
    # Pick old metaphor
    old_visual = "legacy monolith server, tangled cables, warning lights"
    for key, val in old_metaphors.items():
        if key in content_lower:
            old_visual = val
            break
    
    # Pick new metaphor
    new_visual = "sleek AI-powered cloud, glowing neural pathways"
    for key, val in new_metaphors.items():
        if key in content_lower:
            new_visual = val
            break
    
    # If specific entities mentioned, incorporate them
    entity_visual = ""
    if entities:
        top_entity = entities[0]
        entity_visual = f", subtle {top_entity} logo/icon integration"
    
    # Determine accent color based on subcategory
    accent_map = {
        "security": "Neon Red",
        "privacy": "Electric Cyan", 
        "coding": "Matrix Green",
        "finance": "Gold Yellow",
        "ai": "Neon Purple",
        "tools": "Electric Blue",
        "gadgets": "Vibrant Orange",
    }
    accent = "Electric Cyan"
    for key, val in accent_map.items():
        if key in subcat.lower():
            accent = val
            break
    
    # Generate the hook text (3-4 words max)
    hook_words = _extract_hook_words(title, summary)
    
    # Build the prompt
    prompt_parts = [
        "YouTube thumbnail 16:9, rule of thirds composition",
        "split diagonal: LEFT side old/broken, RIGHT side new/upgraded",
        f"LEFT: {old_visual}, dark ominous lighting, crumbling aesthetic",
        f"RIGHT: {new_visual}{entity_visual}, bright hopeful lighting, pristine",
        f"center vertical divider: glowing {accent} energy beam separating two worlds",
        f"bold text overlay: '{hook_words}' in massive Montserrat Black font",
        "text color: Bright Yellow / Pure White on dark bg, high contrast stroke",
        "background: deep neon blue-black (#0A0A15), cyberpunk noir atmosphere",
        f"accent palette: {accent}, Bright Yellow (#FFD600), Neon Red (#FF2020)",
        "style: hyper-realistic 3D render, Unreal Engine 5, octane render, 8k",
        "clean typography, minimal visual clutter, professional thumbnail design",
        "dramatic volumetric lighting, ray-traced reflections, depth of field",
        "--ar 16:9 --stylize 750 --v 6.1"
    ]
    
    return " | ".join(prompt_parts)


def _extract_hook_words(title: str, summary: str) -> str:
    """Extract 3-4 word hook from title/summary for thumbnail text."""
    # Common high-CTR patterns
    patterns = [
        (r"(don't|never|stop|avoid|warning)\s+\w+", "DON'T USE THIS"),
        (r"(how to|why you should|you must)\s+\w+", "DO THIS NOW"),
        (r"(secret|hidden|revealed|exposed)", "SECRET REVEALED"),
        (r"(best|top|ultimate|complete)\s+\w+", "ULTIMATE GUIDE"),
        (r"(new|just launched|breaking|announced)", "JUST LAUNCHED"),
        (r"(vs|versus|compared|beats)", "THIS BEATS THAT"),
        (r"(free|open source|no cost)", "COMPLETELY FREE"),
        (r"(fast|instant|seconds|lightning)", "INSTANT RESULTS"),
        (r"(ai|artificial intelligence|llm|gpt)", "AI CHANGES EVERYTHING"),
    ]
    
    content = (title + " " + summary).lower()
    for pattern, hook in patterns:
        import re
        if re.search(pattern, content):
            return hook
    
    # Fallback: extract key nouns from title
    words = title.split()
    key_words = [w for w in words if len(w) > 3 and w.lower() not in 
                 {"the", "and", "for", "with", "this", "that", "your", "how", "why", "what"}]
    if key_words:
        return " ".join(key_words[:3]).upper()
    
    return "AI BREAKTHROUGH"


def generate_midjourney_prompt(script_json: dict) -> str:
    """Alias for generate_thumbnail_prompt for clarity."""
    return generate_thumbnail_prompt(script_json)


def generate_dalle_prompt(script_json: dict) -> str:
    """Generates a DALL-E 3 optimized prompt (more natural language)."""
    base_prompt = generate_thumbnail_prompt(script_json)
    # Convert Midjourney parameters to natural language for DALL-E
    dalle_prompt = base_prompt.replace("| ", ", ").replace("--ar 16:9", "16:9 aspect ratio").replace("--stylize 750", "highly stylized").replace("--v 6.1", "photorealistic")
    return f"Create a professional YouTube thumbnail: {dalle_prompt}. Professional photography lighting, commercial quality."


if __name__ == "__main__":
    # Test the prompt generator
    test_script = {
        "title": "Google's New AI Search Kills Traditional SEO Forever",
        "description": "Google just launched AI Overviews that completely changes how search works. Traditional SEO tactics are dead. Here's what replaces them.",
        "sub_category": "AI & Tech Tools",
        "companies_mentioned": [{"name": "Google"}, {"name": "OpenAI"}],
        "tools_mentioned": [{"name": "Gemini"}, {"name": "Search Console"}],
        "longform_topics": [
            {"headline": "AI Overviews Launch", "source_name": "Google"},
            {"headline": "SEO Is Dead", "source_name": "Search Engine Journal"},
        ]
    }
    
    print("=" * 80)
    print("MIDJOURNEY PROMPT:")
    print("=" * 80)
    print(generate_thumbnail_prompt(test_script))
    print()
    print("=" * 80)
    print("DALL-E 3 PROMPT:")
    print("=" * 80)
    print(generate_dalle_prompt(test_script))

# ── TEST RUN ─────────────────────────────────────────────────────────────────
# test_json = {"title": "OpenAI Search is finally here...", "color_theme": {"accent": "#00E5FF"}}
# generate_thumbnail(test_json)
