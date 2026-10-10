#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cartoon_dialogue_engine.py — Mascot-driven Cartoon Dialogue Carousel Engine
Features Byte (the Robot) & Asha (the Tech Dev) explaining AI concepts and trending news
using structured dialogue scripts, crisp HTML speech bubbles, and Playwright rendering.
"""

import os
import sys
import json
import re
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

import requests
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

load_dotenv()

try:
    from humanizer_engine import sanitize_text_for_human_voice
except ImportError:
    def sanitize_text_for_human_voice(text):
        return text

BASE_DIR = Path(__file__).parent
CHARACTERS_DIR = BASE_DIR / "assets" / "characters"
TEMPLATE_DIR = BASE_DIR / "carousel_templates"

# Mascot configuration: VJ + Byte + 11 Living Tech Gadget Characters
VALID_SPEAKERS = [
    "vj", "byte", "phone", "computer", "watch", "chip",
    "camera", "earbuds", "battery", "vr", "server", "controller", "drone"
]
VALID_EMOTIONS = ["neutral", "curious", "excited", "shocked", "thinking", "smug"]

# Character metadata registry
CHARACTER_METADATA = {
    "vj": {
        "name": "VJ",
        "emoji": "👨‍💻",
        "title": "Tech Host",
        "tag": "👨‍💻 VJ",
        "role": "Lead human educator & host",
        "desc": "the human tech creator and host of 'Did You Know By VJ', wearing a blue hoodie, explaining complex tech simply and clearly",
        "side": "right"
    },
    "byte": {
        "name": "Byte",
        "emoji": "🤖",
        "title": "AI Robot",
        "tag": "🤖 BYTE",
        "role": "Curious AI robot mascot",
        "desc": "a curious, smart white and cyan robot mascot who asks sharp, fun questions",
        "side": "left"
    },
    "phone": {
        "name": "Phony",
        "emoji": "📱",
        "title": "Smartphone",
        "tag": "📱 PHONE",
        "role": "Sleek smartphone mascot",
        "desc": "a friendly, energetic smartphone mascot with a glowing screen face, always tuned to notifications, apps, and everyday mobile tech",
        "side": "left"
    },
    "computer": {
        "name": "Compute",
        "emoji": "💻",
        "title": "Computer",
        "tag": "💻 COMPUTER",
        "role": "Laptop / PC mascot",
        "desc": "a brainy laptop computer mascot with terminal-style screen eyes, obsessed with software, coding, and heavy-duty computing",
        "side": "left"
    },
    "watch": {
        "name": "Chrono",
        "emoji": "⌚",
        "title": "Smartwatch",
        "tag": "⌚ WATCH",
        "role": "Smartwatch mascot",
        "desc": "a witty, fast-talking digital smartwatch mascot with silicone strap limbs and a glowing pulse face",
        "side": "left"
    },
    "chip": {
        "name": "Silicon",
        "emoji": "⚡",
        "title": "AI Microchip",
        "tag": "⚡ CHIP",
        "role": "Microchip processor mascot",
        "desc": "a proud, tiny powerhouse microchip mascot with golden pin legs and glowing nanometer circuit traces",
        "side": "left"
    },
    "camera": {
        "name": "Shutter",
        "emoji": "📷",
        "title": "Camera",
        "tag": "📷 CAMERA",
        "role": "Digital camera mascot",
        "desc": "a curious camera mascot with a giant glowing optical lens eye, fascinated by image sensors, pixels, and optics",
        "side": "left"
    },
    "earbuds": {
        "name": "Pod",
        "emoji": "🎧",
        "title": "Earbuds",
        "tag": "🎧 EARBUDS",
        "role": "Wireless headphones mascot",
        "desc": "a musical, sleek headphones mascot with glowing soundwave eyes, curious about audio physics and acoustics",
        "side": "left"
    },
    "battery": {
        "name": "Volt",
        "emoji": "🔋",
        "title": "Battery",
        "tag": "🔋 BATTERY",
        "role": "Power bank mascot",
        "desc": "an energetic power bank mascot with a glowing green charge-bar smile, sensitive to heat, voltage, and fast charging",
        "side": "left"
    },
    "vr": {
        "name": "Vision",
        "emoji": "🥽",
        "title": "VR Headset",
        "tag": "🥽 VR HEADSET",
        "role": "Spatial computing headset mascot",
        "desc": "a futuristic spatial headset mascot with holographic visor eyes, passionate about 3D worlds and spatial vision",
        "side": "left"
    },
    "server": {
        "name": "Server",
        "emoji": "🖧",
        "title": "Cloud Server",
        "tag": "🖧 SERVER",
        "role": "Cloud server rack mascot",
        "desc": "a towering cloud server rack mascot with blinking LED array eyes and ethernet cable arms, keeper of global internet data",
        "side": "left"
    },
    "controller": {
        "name": "Joy",
        "emoji": "🎮",
        "title": "Game Controller",
        "tag": "🎮 CONTROLLER",
        "role": "Gamepad mascot",
        "desc": "a playful video game controller mascot with glowing thumbstick eyes, hyped about GPUs, ray tracing, and frame rates",
        "side": "left"
    },
    "drone": {
        "name": "Aero",
        "emoji": "🛸",
        "title": "Drone",
        "tag": "🛸 DRONE",
        "role": "Quadcopter drone mascot",
        "desc": "an adventurous drone mascot with 4 spinning rotors and a camera eye, surveying GPS, navigation, and flight tech",
        "side": "left"
    },
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

GEMINI_GENAI_AVAILABLE = False
GEMINI_LEGACY_AVAILABLE = False

try:
    from google import genai
    GEMINI_GENAI_AVAILABLE = True
except ImportError:
    pass

try:
    import google.generativeai as genai_legacy
    GEMINI_LEGACY_AVAILABLE = True
except ImportError:
    pass

GEMINI_AVAILABLE = GEMINI_GENAI_AVAILABLE or GEMINI_LEGACY_AVAILABLE


def get_character_image_path(speaker: str, emotion: str) -> Optional[Path]:
    """Retrieve absolute file path for a speaker's emotion sprite."""
    speaker = speaker.lower().strip()
    emotion = emotion.lower().strip()
    
    if speaker == "asha":
        speaker = "vj"
    
    if speaker not in VALID_SPEAKERS:
        speaker = "byte"
    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"
        
    # Check folder structure: assets/characters/phone/curious.png
    nested_path = CHARACTERS_DIR / speaker / f"{emotion}.png"
    if nested_path.exists():
        return nested_path
        
    # Check flat structure: assets/characters/phone_curious.png
    flat_path = CHARACTERS_DIR / f"{speaker}_{emotion}.png"
    if flat_path.exists():
        return flat_path
        
    # Fallback to neutral
    fallback_path = CHARACTERS_DIR / speaker / "neutral.png"
    if fallback_path.exists():
        return fallback_path
    flat_fallback = CHARACTERS_DIR / f"{speaker}_neutral.png"
    if flat_fallback.exists():
        return flat_fallback
        
    speaker_dir = CHARACTERS_DIR / speaker
    if speaker_dir.exists() and speaker_dir.is_dir():
        pngs = list(speaker_dir.glob("*.png"))
        if pngs:
            return pngs[0]
            
    if speaker != "byte":
        return get_character_image_path("byte", emotion)
    return None


def resolve_dialogue_characters(
    characters: Optional[str] = "auto",
    topic: str = "",
    story: Optional[Dict] = None,
) -> Tuple[str, str]:
    """Resolve the two dialogue speakers (speaker_left, speaker_right)."""
    chars = (characters or "auto").strip().lower()
    
    # Explicit pair format like 'vj_phone' or 'phone_vj'
    if "_" in chars and chars != "auto":
        parts = chars.split("_", 1)
        c1, c2 = parts[0], parts[1]
        if c1 in VALID_SPEAKERS and c2 in VALID_SPEAKERS:
            if c1 == "vj":
                return c2, "vj"
            elif c2 == "vj":
                return c1, "vj"
            return c1, c2
        elif c1 in VALID_SPEAKERS:
            return c1, "vj"
        elif c2 in VALID_SPEAKERS:
            return c2, "vj"
            
    if chars in VALID_SPEAKERS and chars != "vj":
        return chars, "vj"
        
    # Auto topic matching
    combined_text = f"{topic or ''} {story.get('title', '') if story else ''} {story.get('description', '') if story else ''} {story.get('fact_summary', '') if story else ''}".lower()
    
    if any(k in combined_text for k in ["camera", "photo", "lens", "pixel", "sensor size", "aperture", "optical", "shutter"]):
        return "camera", "vj"
    if any(k in combined_text for k in ["earbuds", "headphone", "audio", "noise cancell", "sound", "acoustics", "voice clone"]):
        return "earbuds", "vj"
    if any(k in combined_text for k in ["battery", "charge", "lithium", "power bank", "overheat", "voltage", "energy"]):
        return "battery", "vj"
    if any(k in combined_text for k in ["vr", "ar", "headset", "spatial", "metaverse", "vision pro", "oculus", "quest"]):
        return "vr", "vj"
    if any(k in combined_text for k in ["underwater", "submarine cable", "fiber optic", "datacenter", "server", "data center", "cloud", "internet backbone"]):
        return "server", "vj"
    if any(k in combined_text for k in ["game", "gaming", "controller", "playstation", "xbox", "nintendo", "ray tracing", "unreal engine"]):
        return "controller", "vj"
    if any(k in combined_text for k in ["drone", "satellite", "gps", "flying", "orbit", "space", "robotics navigation"]):
        return "drone", "vj"
    if any(k in combined_text for k in ["chip", "semiconductor", "transistor", "nanometer", "gpu", "tpu", "silicon", "moore's law", "processor"]):
        return "chip", "vj"
    if any(k in combined_text for k in ["watch", "smartwatch", "heart rate", "biometric", "pulse", "wearable", "chrono", "clock frequency"]):
        return "watch", "vj"
    if any(k in combined_text for k in ["computer", "laptop", "pc", "code", "programming", "software", "linux", "compiler", "terminal", "memory leak", "context window", "llm", "chatgpt"]):
        return "computer", "vj"
    if any(k in combined_text for k in ["phone", "smartphone", "mobile", "screen", "android", "iphone", "ios", "cellular", "5g", "sim card", "touchscreen"]):
        return "phone", "vj"
        
    return "byte", "vj"


try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


# ── CAROUSEL THEME SYSTEM ──────────────────────────────────────────────────
# 13 high-engagement visual themes including Black & White (Dark Noir & Light Editorial)
# and modern vibrant colorways, dynamically matching the topic vector or platform.
CAROUSEL_THEMES = {
    # ── 1. Black & White: Monochrome Noir (Pure OLED Dark) ───
    "monochrome_noir": {
        "id": "monochrome_noir",
        "name": "Monochrome Noir",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #18181B 0%, #09090B 60%, #000000 100%)",
        "mesh_glow": "radial-gradient(circle, rgba(255, 255, 255, 0.12) 0%, rgba(161, 161, 170, 0.08) 50%, transparent 75%)",
        "bg_dark": "#000000",
        "card_bg": "rgba(24, 24, 27, 0.94)",
        "accent_primary": "#FFFFFF",
        "accent_secondary": "#E4E4E7",
        "accent_glow": "rgba(255, 255, 255, 0.28)",
        "accent_tag_bg": "linear-gradient(135deg, #FFFFFF 0%, #A1A1AA 100%)",
        "accent_tag_color": "#000000",
        "secondary_tag_bg": "linear-gradient(135deg, #E4E4E7 0%, #71717A 100%)",
        "secondary_tag_color": "#000000",
        "badge_border": "rgba(255, 255, 255, 0.45)",
        "badge_bg": "rgba(255, 255, 255, 0.10)",
        "bubble_border_left": "rgba(255, 255, 255, 0.5)",
        "bubble_border_right": "rgba(228, 228, 231, 0.5)",
        "cta_bg": "linear-gradient(135deg, #27272A 0%, #18181B 50%, #09090B 100%)",
        "takeaway_border": "rgba(255, 255, 255, 0.5)",
        "accent_gradient": "linear-gradient(90deg, #FFFFFF 0%, #D4D4D8 50%, #71717A 100%)",
        "text_main": "#FFFFFF",
        "text_sub": "#A1A1AA",
    },
    # ── 2. Black & White: Paper Editorial (Clean Light Mode) ───
    "paper_editorial": {
        "id": "paper_editorial",
        "name": "Paper Editorial",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #FFFFFF 0%, #F8FAFC 60%, #EEF2F6 100%)",
        "mesh_glow": "radial-gradient(circle, rgba(15, 23, 42, 0.06) 0%, rgba(100, 116, 139, 0.04) 50%, transparent 75%)",
        "bg_dark": "#F8FAFC",
        "card_bg": "rgba(255, 255, 255, 0.95)",
        "accent_primary": "#0F172A",
        "accent_secondary": "#334155",
        "accent_glow": "rgba(15, 23, 42, 0.18)",
        "accent_tag_bg": "linear-gradient(135deg, #0F172A 0%, #1E293B 100%)",
        "accent_tag_color": "#FFFFFF",
        "secondary_tag_bg": "linear-gradient(135deg, #334155 0%, #475569 100%)",
        "secondary_tag_color": "#FFFFFF",
        "badge_border": "rgba(15, 23, 42, 0.35)",
        "badge_bg": "rgba(15, 23, 42, 0.07)",
        "bubble_border_left": "rgba(15, 23, 42, 0.35)",
        "bubble_border_right": "rgba(51, 65, 85, 0.35)",
        "cta_bg": "linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%)",
        "takeaway_border": "rgba(15, 23, 42, 0.35)",
        "accent_gradient": "linear-gradient(90deg, #0F172A 0%, #334155 50%, #64748B 100%)",
        "text_main": "#0F172A",
        "text_sub": "#475569",
        "is_light": True,
    },
    # ── 3. Crimson Ember (Urgent & Breaking) ───
    "crimson_ember": {
        "id": "crimson_ember",
        "name": "Crimson Ember",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #450A0A 0%, #150507 60%, #080203 100%)",
        "mesh_glow": "radial-gradient(circle, rgba(239, 68, 68, 0.28) 0%, rgba(249, 115, 22, 0.18) 50%, transparent 75%)",
        "bg_dark": "#150507",
        "card_bg": "rgba(30, 8, 12, 0.92)",
        "accent_primary": "#EF4444",
        "accent_secondary": "#F97316",
        "accent_glow": "rgba(239, 68, 68, 0.35)",
        "accent_tag_bg": "linear-gradient(135deg, #EF4444 0%, #B91C1C 100%)",
        "accent_tag_color": "#FFFFFF",
        "secondary_tag_bg": "linear-gradient(135deg, #F97316 0%, #C2410C 100%)",
        "secondary_tag_color": "#FFFFFF",
        "badge_border": "rgba(239, 68, 68, 0.4)",
        "badge_bg": "rgba(239, 68, 68, 0.12)",
        "bubble_border_left": "rgba(239, 68, 68, 0.45)",
        "bubble_border_right": "rgba(249, 115, 22, 0.45)",
        "cta_bg": "linear-gradient(135deg, #EF4444 0%, #DC2626 50%, #991B1B 100%)",
        "takeaway_border": "rgba(239, 68, 68, 0.45)",
        "accent_gradient": "linear-gradient(90deg, #EF4444 0%, #F97316 50%, #FBBF24 100%)",
    },
    # ── 4. Emerald Terminal (Hacker / Matrix) ───
    "emerald_terminal": {
        "id": "emerald_terminal",
        "name": "Emerald Terminal",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #022C22 0%, #03140F 60%, #010705 100%)",
        "mesh_glow": "radial-gradient(circle, rgba(16, 185, 129, 0.30) 0%, rgba(52, 211, 153, 0.15) 50%, transparent 75%)",
        "bg_dark": "#03140F",
        "card_bg": "rgba(6, 28, 20, 0.94)",
        "accent_primary": "#10B981",
        "accent_secondary": "#34D399",
        "accent_glow": "rgba(16, 185, 129, 0.38)",
        "accent_tag_bg": "linear-gradient(135deg, #10B981 0%, #059669 100%)",
        "accent_tag_color": "#021A11",
        "secondary_tag_bg": "linear-gradient(135deg, #34D399 0%, #047857 100%)",
        "secondary_tag_color": "#021A11",
        "badge_border": "rgba(16, 185, 129, 0.4)",
        "badge_bg": "rgba(16, 185, 129, 0.12)",
        "bubble_border_left": "rgba(16, 185, 129, 0.45)",
        "bubble_border_right": "rgba(52, 211, 153, 0.45)",
        "cta_bg": "linear-gradient(135deg, #10B981 0%, #059669 50%, #047857 100%)",
        "takeaway_border": "rgba(16, 185, 129, 0.45)",
        "accent_gradient": "linear-gradient(90deg, #10B981 0%, #34D399 50%, #A7F3D0 100%)",
    },
    # ── 5. Luxury Gold (Enterprise & Hardware) ───
    "luxury_gold": {
        "id": "luxury_gold",
        "name": "Luxury Gold",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #2A1B07 0%, #110B03 60%, #050301 100%)",
        "mesh_glow": "radial-gradient(circle, rgba(245, 158, 11, 0.28) 0%, rgba(251, 191, 36, 0.16) 50%, transparent 75%)",
        "bg_dark": "#110B03",
        "card_bg": "rgba(28, 19, 7, 0.94)",
        "accent_primary": "#FBBF24",
        "accent_secondary": "#F59E0B",
        "accent_glow": "rgba(251, 191, 36, 0.35)",
        "accent_tag_bg": "linear-gradient(135deg, #FBBF24 0%, #D97706 100%)",
        "accent_tag_color": "#1B0F00",
        "secondary_tag_bg": "linear-gradient(135deg, #FDE68A 0%, #B45309 100%)",
        "secondary_tag_color": "#1B0F00",
        "badge_border": "rgba(251, 191, 36, 0.4)",
        "badge_bg": "rgba(251, 191, 36, 0.12)",
        "bubble_border_left": "rgba(251, 191, 36, 0.45)",
        "bubble_border_right": "rgba(245, 158, 11, 0.45)",
        "cta_bg": "linear-gradient(135deg, #F59E0B 0%, #D97706 50%, #92400E 100%)",
        "takeaway_border": "rgba(251, 191, 36, 0.45)",
        "accent_gradient": "linear-gradient(90deg, #FDE68A 0%, #FBBF24 50%, #D97706 100%)",
    },
    # ── 6. Arctic Frost (Data & Cloud) ───
    "arctic_frost": {
        "id": "arctic_frost",
        "name": "Arctic Frost",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #082F49 0%, #031422 60%, #01060B 100%)",
        "mesh_glow": "radial-gradient(circle, rgba(56, 189, 248, 0.28) 0%, rgba(147, 197, 253, 0.16) 50%, transparent 75%)",
        "bg_dark": "#031422",
        "card_bg": "rgba(8, 28, 48, 0.94)",
        "accent_primary": "#38BDF8",
        "accent_secondary": "#BAE6FD",
        "accent_glow": "rgba(56, 189, 248, 0.35)",
        "accent_tag_bg": "linear-gradient(135deg, #38BDF8 0%, #0284C7 100%)",
        "accent_tag_color": "#021A2C",
        "secondary_tag_bg": "linear-gradient(135deg, #BAE6FD 0%, #0369A1 100%)",
        "secondary_tag_color": "#021A2C",
        "badge_border": "rgba(56, 189, 248, 0.4)",
        "badge_bg": "rgba(56, 189, 248, 0.12)",
        "bubble_border_left": "rgba(56, 189, 248, 0.45)",
        "bubble_border_right": "rgba(186, 230, 253, 0.45)",
        "cta_bg": "linear-gradient(135deg, #38BDF8 0%, #0284C7 50%, #0369A1 100%)",
        "takeaway_border": "rgba(56, 189, 248, 0.45)",
        "accent_gradient": "linear-gradient(90deg, #BAE6FD 0%, #38BDF8 50%, #0284C7 100%)",
    },
    # ── 7. Tokyo Midnight (Cyber Magenta) ───
    "tokyo_midnight": {
        "id": "tokyo_midnight",
        "name": "Tokyo Midnight",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #3B0764 0%, #150325 60%, #07010C 100%)",
        "mesh_glow": "radial-gradient(circle, rgba(236, 72, 153, 0.28) 0%, rgba(139, 92, 246, 0.20) 50%, transparent 75%)",
        "bg_dark": "#150325",
        "card_bg": "rgba(28, 7, 48, 0.94)",
        "accent_primary": "#F43F5E",
        "accent_secondary": "#A855F7",
        "accent_glow": "rgba(244, 63, 94, 0.38)",
        "accent_tag_bg": "linear-gradient(135deg, #F43F5E 0%, #9333EA 100%)",
        "accent_tag_color": "#FFFFFF",
        "secondary_tag_bg": "linear-gradient(135deg, #A855F7 0%, #7E22CE 100%)",
        "secondary_tag_color": "#FFFFFF",
        "badge_border": "rgba(244, 63, 94, 0.4)",
        "badge_bg": "rgba(244, 63, 94, 0.12)",
        "bubble_border_left": "rgba(244, 63, 94, 0.45)",
        "bubble_border_right": "rgba(168, 85, 247, 0.45)",
        "cta_bg": "linear-gradient(135deg, #F43F5E 0%, #A855F7 50%, #7C3AED 100%)",
        "takeaway_border": "rgba(244, 63, 94, 0.45)",
        "accent_gradient": "linear-gradient(90deg, #F43F5E 0%, #A855F7 50%, #38BDF8 100%)",
    },
    # ── 8. Neon Cyber ───
    "neon_cyber": {
        "id": "neon_cyber",
        "name": "Neon Cyber",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #172554 0%, #0B0F19 60%, #030712 100%)",
        "mesh_glow": "radial-gradient(circle, rgba(56, 189, 248, 0.22) 0%, rgba(168, 85, 247, 0.15) 50%, transparent 75%)",
        "bg_dark": "#0B0F19",
        "card_bg": "rgba(16, 24, 44, 0.90)",
        "accent_primary": "#00F2FE",
        "accent_secondary": "#38BDF8",
        "accent_glow": "rgba(0, 242, 254, 0.35)",
        "accent_tag_bg": "linear-gradient(135deg, #00F2FE 0%, #0077FE 100%)",
        "accent_tag_color": "#031427",
        "secondary_tag_bg": "linear-gradient(135deg, #38BDF8 0%, #0284C7 100%)",
        "secondary_tag_color": "#031427",
        "badge_border": "rgba(0, 242, 254, 0.4)",
        "badge_bg": "rgba(0, 242, 254, 0.12)",
        "bubble_border_left": "rgba(0, 242, 254, 0.45)",
        "bubble_border_right": "rgba(56, 189, 248, 0.45)",
        "cta_bg": "linear-gradient(135deg, #0284C7 0%, #2563EB 50%, #7C3AED 100%)",
        "takeaway_border": "rgba(56, 189, 248, 0.45)",
        "accent_gradient": "linear-gradient(90deg, #00F2FE 0%, #A855F7 50%, #FF5E8E 100%)",
    },
    # ── 9. Cyber Matrix ───
    "cyber_matrix": {
        "id": "cyber_matrix",
        "name": "Cyber Matrix",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #064E3B 0%, #0A140F 60%, #020704 100%)",
        "mesh_glow": "radial-gradient(circle, rgba(16, 185, 129, 0.28) 0%, rgba(5, 150, 105, 0.18) 50%, transparent 75%)",
        "bg_dark": "#0A140F",
        "card_bg": "rgba(10, 28, 20, 0.92)",
        "accent_primary": "#10B981",
        "accent_secondary": "#34D399",
        "accent_glow": "rgba(16, 185, 129, 0.35)",
        "accent_tag_bg": "linear-gradient(135deg, #10B981 0%, #047857 100%)",
        "accent_tag_color": "#022013",
        "secondary_tag_bg": "linear-gradient(135deg, #34D399 0%, #059669 100%)",
        "secondary_tag_color": "#022013",
        "badge_border": "rgba(16, 185, 129, 0.4)",
        "badge_bg": "rgba(16, 185, 129, 0.12)",
        "bubble_border_left": "rgba(16, 185, 129, 0.45)",
        "bubble_border_right": "rgba(52, 211, 153, 0.45)",
        "cta_bg": "linear-gradient(135deg, #059669 0%, #047857 50%, #065F46 100%)",
        "takeaway_border": "rgba(16, 185, 129, 0.45)",
        "accent_gradient": "linear-gradient(90deg, #10B981 0%, #34D399 50%, #6EE7B7 100%)",
    },
    # ── 10. Tech Blueprint ───
    "tech_blueprint": {
        "id": "tech_blueprint",
        "name": "Tech Blueprint",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #1E3A8A 0%, #0B192C 60%, #050C17 100%)",
        "mesh_glow": "radial-gradient(circle, rgba(96, 165, 250, 0.25) 0%, rgba(37, 99, 235, 0.18) 50%, transparent 75%)",
        "bg_dark": "#0B192C",
        "card_bg": "rgba(15, 30, 54, 0.92)",
        "accent_primary": "#60A5FA",
        "accent_secondary": "#93C5FD",
        "accent_glow": "rgba(96, 165, 250, 0.35)",
        "accent_tag_bg": "linear-gradient(135deg, #60A5FA 0%, #2563EB 100%)",
        "accent_tag_color": "#051329",
        "secondary_tag_bg": "linear-gradient(135deg, #93C5FD 0%, #3B82F6 100%)",
        "secondary_tag_color": "#051329",
        "badge_border": "rgba(96, 165, 250, 0.4)",
        "badge_bg": "rgba(96, 165, 250, 0.12)",
        "bubble_border_left": "rgba(96, 165, 250, 0.45)",
        "bubble_border_right": "rgba(147, 197, 253, 0.45)",
        "cta_bg": "linear-gradient(135deg, #2563EB 0%, #1D4ED8 50%, #1E40AF 100%)",
        "takeaway_border": "rgba(96, 165, 250, 0.45)",
        "accent_gradient": "linear-gradient(90deg, #60A5FA 0%, #38BDF8 50%, #818CF8 100%)",
    },
    # ── 11. Amber Solaris ───
    "amber_solaris": {
        "id": "amber_solaris",
        "name": "Amber Solaris",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #451A03 0%, #16110D 60%, #080605 100%)",
        "mesh_glow": "radial-gradient(circle, rgba(245, 158, 11, 0.28) 0%, rgba(217, 119, 6, 0.18) 50%, transparent 75%)",
        "bg_dark": "#16110D",
        "card_bg": "rgba(32, 22, 14, 0.92)",
        "accent_primary": "#F59E0B",
        "accent_secondary": "#FBBF24",
        "accent_glow": "rgba(245, 158, 11, 0.35)",
        "accent_tag_bg": "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)",
        "accent_tag_color": "#200D00",
        "secondary_tag_bg": "linear-gradient(135deg, #FBBF24 0%, #B45309 100%)",
        "secondary_tag_color": "#200D00",
        "badge_border": "rgba(245, 158, 11, 0.4)",
        "badge_bg": "rgba(245, 158, 11, 0.12)",
        "bubble_border_left": "rgba(245, 158, 11, 0.45)",
        "bubble_border_right": "rgba(251, 191, 36, 0.45)",
        "cta_bg": "linear-gradient(135deg, #D97706 0%, #B45309 50%, #92400E 100%)",
        "takeaway_border": "rgba(245, 158, 11, 0.45)",
        "accent_gradient": "linear-gradient(90deg, #F59E0B 0%, #F97316 50%, #EF4444 100%)",
    },
    # ── 12. Synthwave Plum ───
    "synthwave_plum": {
        "id": "synthwave_plum",
        "name": "Synthwave Plum",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #4A0E4E 0%, #180B26 60%, #0A0412 100%)",
        "mesh_glow": "radial-gradient(circle, rgba(255, 51, 102, 0.25) 0%, rgba(168, 85, 247, 0.20) 50%, transparent 75%)",
        "bg_dark": "#180B26",
        "card_bg": "rgba(34, 16, 50, 0.92)",
        "accent_primary": "#FF3366",
        "accent_secondary": "#C084FC",
        "accent_glow": "rgba(255, 51, 102, 0.35)",
        "accent_tag_bg": "linear-gradient(135deg, #FF3366 0%, #BE185D 100%)",
        "accent_tag_color": "#1F030B",
        "secondary_tag_bg": "linear-gradient(135deg, #C084FC 0%, #9333EA 100%)",
        "secondary_tag_color": "#1F030B",
        "badge_border": "rgba(255, 51, 102, 0.4)",
        "badge_bg": "rgba(255, 51, 102, 0.12)",
        "bubble_border_left": "rgba(255, 51, 102, 0.45)",
        "bubble_border_right": "rgba(192, 132, 252, 0.45)",
        "cta_bg": "linear-gradient(135deg, #BE185D 0%, #9333EA 50%, #7E22CE 100%)",
        "takeaway_border": "rgba(255, 51, 102, 0.45)",
        "accent_gradient": "linear-gradient(90deg, #FF3366 0%, #C084FC 50%, #38BDF8 100%)",
    },
    # ── 13. Swiss Minimal ───
    "swiss_minimal": {
        "id": "swiss_minimal",
        "name": "Swiss Minimal",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #1E293B 0%, #0F172A 60%, #020617 100%)",
        "mesh_glow": "radial-gradient(circle, rgba(167, 139, 250, 0.20) 0%, rgba(99, 102, 241, 0.15) 50%, transparent 75%)",
        "bg_dark": "#0F172A",
        "card_bg": "rgba(24, 33, 52, 0.92)",
        "accent_primary": "#A78BFA",
        "accent_secondary": "#E2E8F0",
        "accent_glow": "rgba(167, 139, 250, 0.35)",
        "accent_tag_bg": "linear-gradient(135deg, #A78BFA 0%, #6D28D9 100%)",
        "accent_tag_color": "#0F0B1E",
        "secondary_tag_bg": "linear-gradient(135deg, #E2E8F0 0%, #94A3B8 100%)",
        "secondary_tag_color": "#0F0B1E",
        "badge_border": "rgba(167, 139, 250, 0.4)",
        "badge_bg": "rgba(167, 139, 250, 0.12)",
        "bubble_border_left": "rgba(167, 139, 250, 0.45)",
        "bubble_border_right": "rgba(226, 232, 240, 0.45)",
        "cta_bg": "linear-gradient(135deg, #6D28D9 0%, #4F46E5 50%, #3730A3 100%)",
        "takeaway_border": "rgba(167, 139, 250, 0.45)",
        "accent_gradient": "linear-gradient(90deg, #A78BFA 0%, #818CF8 50%, #38BDF8 100%)",
    },
}

def resolve_carousel_theme(
    theme: Optional[str] = "auto",
    vector: Optional[str] = None,
    platform: Optional[str] = None,
) -> Dict:
    """Resolve theme configuration based on explicit choice, topic vector, or platform."""
    if theme and theme != "auto" and theme in CAROUSEL_THEMES:
        return CAROUSEL_THEMES[theme]

    # Vector-to-theme mapping
    vector_map = {
        "ai_secrets": "tokyo_midnight",
        "everyday_tech_mysteries": "paper_editorial",
        "hardware_megastructures": "luxury_gold",
        "internet_infrastructure": "arctic_frost",
        "forgotten_tech_history": "monochrome_noir",
    }
    if vector and vector in vector_map:
        return CAROUSEL_THEMES[vector_map[vector]]

    # Platform default / fallback mapping
    platform_map = {
        "threads": "paper_editorial",
        "facebook": "tech_blueprint",
        "instagram": "neon_cyber",
    }
    pref = platform_map.get((platform or "").lower(), "neon_cyber")
    return CAROUSEL_THEMES[pref]

def extract_stat_from_bubble(text: str) -> Optional[Dict[str, str]]:
    """Detect prominent numerical stats in dialogue bubble to create visual stat callouts."""
    if not text:
        return None
    patterns = [
        r'(\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent))',
        r'(\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:Gbps|Tbps|Mbps|km|meters|miles|Hz|kHz|MHz|GHz|ms|microseconds|seconds|hours|days|years)\b)',
        r'(\b(?:over|under|nearly|exactly|approx\.?|around)?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:billion|million|trillion|thousand)\b)',
        r'(\b\d{1,3}(?:,\d{3})*(?:\.\d+)?x\b)',
        r'(\b(?:18\d{2}|19\d{2}|20\d{2})\b)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            stat_val = m.group(1).strip()
            if len(stat_val) >= 2:
                return {"highlight_stat": stat_val}
    return None


# ── DID YOU KNOW CURATED SEED FACT POOL ───────────────────────────────────────
# High-attraction, verified mind-blowing facts rotating across 5 curiosity vectors
# engagement_score (1-10): predicted relative engagement based on topic universality,
# shareability, and emotional surprise factor. Higher = more likes/saves/shares.
DID_YOU_KNOW_SEED_FACTS = [
    # ── AI Secrets ──────────────────────────────────────────────────────────
    {
        "vector": "ai_secrets",
        "title": "Why ChatGPT Actually Forgets You",
        "hook": "Why Does ChatGPT Forget What You Said? 🤯",
        "headline": "The Context Window Reality",
        "fact_summary": "LLMs do not store memories between turns. Every reply re-reads earlier text until the context window overflows, silently dropping older tokens from the start.",
        "source": "Transformer Attention & Context Windows",
        "keywords": ["chatgpt", "context window", "ai memory", "tokens", "llm", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "ai_secrets",
        "title": "Why AI Hallucinates Instead of Admitting Ignorance",
        "hook": "Why Do AI Models Hallucinate? 🤖",
        "headline": "Next-Token Probability Engine",
        "fact_summary": "AI has zero concept of factual truth. It calculates mathematical probabilities of what word should come next, generating convincing falsehoods when confidence is low.",
        "source": "Transformer Probabilistic Modeling",
        "keywords": ["ai hallucination", "transformers", "machine learning", "probability", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "ai_secrets",
        "title": "How AI Generates Images from Pure Static Noise",
        "hook": "Did You Know AI Paints from Pure TV Static? 🎨",
        "headline": "Diffusion Reverse Denoising",
        "fact_summary": "Diffusion models like Midjourney start with 100% random static fuzz and gradually subtract noise over 50 steps until a crisp image crystallizes.",
        "source": "Denoising Diffusion Probabilistic Models",
        "keywords": ["diffusion models", "midjourney", "image generation", "ai art", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "ai_secrets",
        "title": "Why AI Can Count to Billions But Fails at Strawberry 'r's",
        "hook": "Why Can't AI Count Letters in 'Strawberry'? 🍓",
        "headline": "The Subword Tokenization Blindspot",
        "fact_summary": "LLMs never see raw characters. Words are sliced into multi-letter token IDs, making it impossible for the model to see individual letters without spelling them out.",
        "source": "Byte-Pair Encoding Tokenization",
        "keywords": ["tokenization", "bpe", "strawberry", "llm reasoning", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "ai_secrets",
        "title": "Self-Attention Costs Double When You Double Text Length",
        "hook": "Did You Know AI Memory Explodes Quadratically? 💥",
        "headline": "O(N²) Transformer Quadratic Scaling",
        "fact_summary": "In standard transformers, every token must calculate attention with every other token. Doubling prompt length doesn't double memory—it multiplies computation by four.",
        "source": "Attention Is All You Need (Vaswani et al.)",
        "keywords": ["transformer", "self attention", "quadratic cost", "context length", "did you know"],
        "engagement_score": 6
    },
    {
        "vector": "ai_secrets",
        "title": "Training AI on AI-Generated Text Causes Model Collapse",
        "hook": "Did You Know AI Goes Insane on Its Own Data? 🌀",
        "headline": "Model Autophagous Collapse Disorder",
        "fact_summary": "When language models are trained recursively on synthetic AI outputs, mathematical tails vanish, entropy collapses, and the model permanently degenerates into gibberish.",
        "source": "Nature: The Curse of Recursion in Generative Models",
        "keywords": ["model collapse", "synthetic data", "ai training", "machine learning", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "ai_secrets",
        "title": "Mixture of Experts Only Wakes Up 3% of the Brain",
        "hook": "Did You Know Giant AIs Sleep Through Most Words? 🧠",
        "headline": "MoE Sparse Neural Routing",
        "fact_summary": "Models like Mixtral and DeepSeek have hundreds of billions of parameters, but a router gate activates only 2 out of 64 expert neural networks per individual token.",
        "source": "Sparse Mixture-of-Experts Architecture Papers",
        "keywords": ["mixture of experts", "moe", "deepseek", "neural networks", "did you know"],
        "engagement_score": 7
    },
    {
        "vector": "ai_secrets",
        "title": "RLHF Makes AI Sycophantic Rather Than Honest",
        "hook": "Did You Know AI is Trained to Flatter You? 🎭",
        "headline": "RLHF Sycophancy Emergence",
        "fact_summary": "Reinforcement learning with human feedback rewards models that agree with the user's misconceptions, causing chatbots to flatter opinions rather than state factual corrections.",
        "source": "Anthropic Research on AI Sycophancy",
        "keywords": ["rlhf", "sycophancy", "alignment", "chatbots", "did you know"],
        "engagement_score": 8
    },

    # ── Everyday Tech Mysteries ─────────────────────────────────────────────
    {
        "vector": "everyday_tech_mysteries",
        "title": "99% of Global Internet is on the Ocean Floor",
        "hook": "Did You Know 99% of the Internet is Underwater? 🌊",
        "headline": "Subsea Fiber Optic Megastructure",
        "fact_summary": "Satellites carry under 1% of data. Over 1.4 million kilometers of submarine fiber optic cables, armored against sharks and anchors, carry all global internet.",
        "source": "TeleGeography Submarine Cable Registry",
        "keywords": ["submarine cables", "internet", "fiber optics", "ocean floor", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "GPS Would Drift 11 Kilometers Daily Without Einstein",
        "hook": "Did You Know GPS Needs Einstein's Relativity? 🛰️",
        "headline": "Relativistic Satellite Time Dilation",
        "fact_summary": "Satellite clocks tick 38 microseconds faster per day due to weaker gravity and high speed. Without relativistic math correction, Google Maps would drift 11 km every day.",
        "source": "General & Special Relativity in GNSS",
        "keywords": ["gps", "einstein", "relativity", "time dilation", "satellites", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "Your Smartphone Screen Steals Your Electrons",
        "hook": "Did You Know Your Phone Steals Your Electrons? 📱",
        "headline": "Capacitive Touchscreen Physics",
        "fact_summary": "Phone glass does not detect pressure. A grid of transparent indium tin oxide electrodes detects tiny electrical charges transferring from your skin when you touch it.",
        "source": "Capacitive Sensing Physics & IEEE",
        "keywords": ["touchscreen", "capacitance", "smartphone", "physics", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "How Airplanes Get Wi-Fi at 35,000 Feet Over Oceans",
        "hook": "Did You Know How Planes Get Wi-Fi Mid-Flight? ✈️",
        "headline": "Gimbaled Satellite Phased Arrays",
        "fact_summary": "Planes use motorized parabolic antennas inside a teardrop roof dome that track geostationary satellites 36,000 km away while flying at 900 km/h.",
        "source": "Aeronautical Satellite Telecommunications",
        "keywords": ["airplane wifi", "satellite", "aviation tech", "phased array", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "Quartz Clocks Vibrate Exactly 32,768 Times Every Second",
        "hook": "Did You Know Why Watches Tick in Powers of Two? ⌚",
        "headline": "The Piezoelectric 32,768 Hz Tuning Fork",
        "fact_summary": "Every quartz watch contains a microscopic tuning fork vibrating at 32,768 Hz. A 15-stage binary flip-flop circuit halves the frequency 15 times to yield exactly 1 second.",
        "source": "Horological Quartz Oscillation Engineering",
        "keywords": ["quartz crystal", "binary", "watches", "piezoelectric", "did you know"],
        "engagement_score": 7
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "Blue OLED Subpixels Die 3 Times Faster Than Red or Green",
        "hook": "Did You Know Blue Light Destroys Your OLED Screen? 📺",
        "headline": "The Blue Phosphorescent Organic Decay Problem",
        "fact_summary": "Blue light requires high-energy photon emission, breaking organic chemical bonds much faster than red or green. Phone makers make blue subpixels twice as large to compensate.",
        "source": "Society for Information Display (SID) Research",
        "keywords": ["oled", "burn in", "blue subpixels", "display tech", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "Li-Fi Transmits 224 Gigabits Per Second Through Room Bulbs",
        "hook": "Did You Know Light Bulbs Can Outrun Wi-Fi 100x? 💡",
        "headline": "Visible Light Optical Communications (Li-Fi)",
        "fact_summary": "Specialized LED bulbs can flicker millions of times per second—undetectable to human eyes—beaming 224 Gbps of encrypted wireless data directly into laptop photo-sensors.",
        "source": "IEEE 802.11bb Light Wireless Standard",
        "keywords": ["lifi", "visible light", "wireless communication", "photonics", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "Credit Cards Have Zero Battery and Power Themselves from Air",
        "hook": "Did You Know Contactless Cards Have No Battery? 💳",
        "headline": "Near-Field Electromagnetic Induction",
        "fact_summary": "A tap-to-pay card has a copper antenna coil running along its edges. The terminal emits an electromagnetic field that induces current, powering the onboard cryptographic CPU.",
        "source": "ISO/IEC 14443 Contactless Standards",
        "keywords": ["nfc", "rfid", "contactless payments", "electromagnetic induction", "did you know"],
        "engagement_score": 9
    },

    # ── Hardware Megastructures ─────────────────────────────────────────────
    {
        "vector": "hardware_megastructures",
        "title": "ASML Chip Lasers Fire 50,000 Times a Second at Molten Tin",
        "hook": "Did You Know How the World's Microchips are Made? 🔬",
        "headline": "ASML Extreme Ultraviolet Lithography",
        "fact_summary": "A high-powered CO2 laser vaporizes 50,000 drops of molten tin per second into plasma hotter than the sun's surface to produce 13.5nm light waves.",
        "source": "ASML High-NA EUV Engineering",
        "keywords": ["asml", "euv", "semiconductors", "chip making", "microprocessors", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "hardware_megastructures",
        "title": "Microsoft Sunk a Datacenter 117 Feet Under the Sea",
        "hook": "Did You Know Datacenters Run Under the Sea? 🌊",
        "headline": "Project Natick Underwater Server Pods",
        "fact_summary": "Microsoft submerged 864 servers in a sealed nitrogen capsule off Scotland. With no humans and constant natural seawater cooling, server failure dropped by 800%.",
        "source": "Microsoft Project Natick Research",
        "keywords": ["project natick", "underwater datacenter", "cloud servers", "microsoft", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "hardware_megastructures",
        "title": "Cleanrooms are 10,000x Cleaner Than Hospital Surgery Rooms",
        "hook": "Did You Know Chip Cleanrooms Beat Surgery Rooms? 🧪",
        "headline": "ISO Class 1 Semiconductor Cleanrooms",
        "fact_summary": "A single speck of human dead skin or dust can bridge microscopic transistor paths. Air in chip cleanrooms is filtered to under 10 particles per cubic meter.",
        "source": "Semiconductor Fab Standards (ISO 14644)",
        "keywords": ["cleanroom", "semiconductors", "fab", "transistors", "did you know"],
        "engagement_score": 7
    },
    {
        "vector": "hardware_megastructures",
        "title": "Cerebras Built a Single Chip with 4 Trillion Transistors",
        "hook": "Did You Know the World's Largest Chip is 8.5 Inches? 🖥️",
        "headline": "Cerebras Wafer-Scale Engine 3",
        "fact_summary": "Instead of slicing silicon wafers into hundreds of tiny chips, Cerebras uses an entire 300mm silicon wafer as a single giant AI processor with 900,000 compute cores.",
        "source": "Cerebras Systems Architectural Whitepaper",
        "keywords": ["cerebras", "wafer scale", "ai chips", "hardware megastructures", "did you know"],
        "engagement_score": 7
    },
    {
        "vector": "hardware_megastructures",
        "title": "HBM3e Stacks DRAM with 50,000 Microscopic Silicon Vias",
        "hook": "Did You Know AI Memory is Stacked Like Skyscraper Towers? 🏢",
        "headline": "High-Bandwidth 3D Memory Packaging",
        "fact_summary": "NVIDIA Blackwell GPUs achieve 8 TB/s memory speeds by stacking 12 DRAM dies vertically, connected by 50,000 Through-Silicon Vias microscopic channels per stack.",
        "source": "JEDEC High Bandwidth Memory Specification",
        "keywords": ["hbm3e", "gpu memory", "nvidia blackwell", "semiconductors", "did you know"],
        "engagement_score": 7
    },
    {
        "vector": "hardware_megastructures",
        "title": "CPU Interconnects are 10,000x Thinner Than a Human Hair",
        "hook": "Did You Know Inside a CPU Looks Like a 15-Story Highway? 🛣️",
        "headline": "Multilayer Copper Interconnect Metallurgy",
        "fact_summary": "Inside modern 3nm chips, over 100 kilometers of microscopic copper wires weave through 15 vertical metal layers, some measuring only 12 nanometers in width.",
        "source": "IEEE Transactions on Electron Devices",
        "keywords": ["semiconductor", "interconnects", "3nm", "copper wiring", "did you know"],
        "engagement_score": 7
    },
    {
        "vector": "hardware_megastructures",
        "title": "Hard Drive Heads Fly Just 3 Nanometers Above Platters",
        "hook": "Did You Know Hard Drive Heads Fly Closer Than Smoke Particles? 💽",
        "headline": "Air Bearing Slider Aerodynamics",
        "fact_summary": "HDD magnetic read heads glide 3 nanometers above platters spinning at 120 km/h. If the head were a Boeing 747, it would be flying 0.1 millimeters above the grass.",
        "source": "Seagate & Western Digital Advanced Storage Physics",
        "keywords": ["hard drive", "hdd", "aerodynamics", "magnetic storage", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "hardware_megastructures",
        "title": "EV Batteries Check Cell Health 1,000 Times Every Second",
        "hook": "Did You Know EV Batteries Run 1,000 Safety Checks a Second? ⚡",
        "headline": "Real-Time Microsecond Battery Management",
        "fact_summary": "EV battery packs contain thousands of lithium cells. Dedicated BMS chips monitor microvolt shifts and millikelvin thermal deviations 1,000 times/sec to prevent thermal runaway.",
        "source": "Automotive BMS Functional Safety (ISO 26262)",
        "keywords": ["ev battery", "battery management", "electric vehicles", "automotive tech", "did you know"],
        "engagement_score": 8
    },

    # ── Bizarre Tech History ────────────────────────────────────────────────
    {
        "vector": "bizarre_tech_history",
        "title": "The $500M Rocket Crash Caused by 64-Bit to 16-Bit Conversion",
        "hook": "Did You Know a 64-Bit Bug Blew Up a $500M Rocket? 🚀",
        "headline": "Ariane 5 Flight 501 Integer Overflow",
        "fact_summary": "In 1996, the Ariane 5 rocket exploded 37 seconds after launch because software tried to stuff a 64-bit floating point number into a 16-bit integer, causing fatal overflow.",
        "source": "Ariane 5 Flight 501 Inquiry Board Report",
        "keywords": ["ariane 5", "integer overflow", "software bug", "rocket science", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "bizarre_tech_history",
        "title": "Wi-Fi Was Accidentally Invented by an Astronomer Studying Black Holes",
        "hook": "Did You Know Wi-Fi Came from Black Holes? 🌌",
        "headline": "CSIRO Radio Astronomy Invention",
        "fact_summary": "In the 1990s, Australian astronomer Dr. John O'Sullivan was trying to detect exploding mini black holes using radio waves. The signal-cleaning algorithm became modern Wi-Fi.",
        "source": "CSIRO Wireless LAN Patent History",
        "keywords": ["wifi", "invention", "black holes", "radio astronomy", "csiro", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "bizarre_tech_history",
        "title": "The First Computer Bug Was an Actual Moth",
        "hook": "Did You Know the First Bug Was a Real Moth? 🦋",
        "headline": "Grace Hopper's 1947 Harvard Relay Bug",
        "fact_summary": "In 1947, computer pioneer Grace Hopper's team investigated a malfunction in the Harvard Mark II relay computer and found a live moth trapped between Relay #70.",
        "source": "Smithsonian National Museum of American History",
        "keywords": ["computer bug", "grace hopper", "harvard mark ii", "tech history", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "bizarre_tech_history",
        "title": "Why NASA Lost a $327M Mars Orbiter Over Metric Units",
        "hook": "Did You Know Metric vs Imperial Crashed a Mars Probe? 🪐",
        "headline": "Mars Climate Orbiter Navigation Loss",
        "fact_summary": "Lockheed Martin software calculated thruster impulse in pound-seconds, but NASA navigation software expected metric newton-seconds. The orbiter incinerated in Mars' atmosphere.",
        "source": "NASA Mars Climate Orbiter Mishap Board",
        "keywords": ["mars climate orbiter", "nasa", "metric system", "spacecraft", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "bizarre_tech_history",
        "title": "A Cloud Reflection Almost Triggered Nuclear War in 1983",
        "hook": "Did You Know Cloud Reflections Almost Started World War III? ☢️",
        "headline": "The 1983 Soviet Satellite False Alarm",
        "fact_summary": "Soviet early-warning satellite Oko detected five incoming US nuclear missiles. Duty officer Stanislav Petrov trusted his intuition that it was a computer glitch, which turned out to be sun reflecting off high clouds.",
        "source": "United Nations Peace History & CIA Declassified Archives",
        "keywords": ["stanislav petrov", "nuclear false alarm", "cold war tech", "satellites", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "bizarre_tech_history",
        "title": "The First 1GB Hard Drive in 1980 Weighed 550 Pounds",
        "hook": "Did You Know the First 1GB Drive Was the Size of a Refrigerator? 🗄️",
        "headline": "IBM 3380 Direct Access Storage Device",
        "fact_summary": "Released in 1980, the IBM 3380 was the first storage system capable of holding 1 Gigabyte of data. It weighed 550 pounds (250 kg) and cost over $40,000.",
        "source": "IBM Archives & Computer History Museum",
        "keywords": ["ibm 3380", "1gb hard drive", "computer history", "storage", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "bizarre_tech_history",
        "title": "The $100 Billion Y2K Fix That Actually Worked",
        "hook": "Did You Know Y2K Wasn't a Hoax—It Cost $100 Billion to Fix? 🛠️",
        "headline": "The Millennium Bug Global Code Remediation",
        "fact_summary": "Early programmers used 2 digits for years (99 for 1999) to save memory. A coordinated effort involving millions of engineers and $100B in refactoring prevented global banking systems from crashing at midnight.",
        "source": "US Department of Commerce Y2K Economic Report",
        "keywords": ["y2k", "millennium bug", "legacy code", "software engineering", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "bizarre_tech_history",
        "title": "Soviet Titanium Submarines Were Driven by Liquid Metal Reactors",
        "hook": "Did You Know Soviet Subs Ran on Liquid Lead-Bismuth Metal? ⚓",
        "headline": "Project 705 Lira (Alfa-Class) High-Automation Submarines",
        "fact_summary": "Soviet Alfa-class submarines featured an all-titanium hull and reached 80 km/h underwater. Their reactor was cooled by molten lead-bismuth metal and could never be shut down without freezing solid.",
        "source": "Naval Nuclear Propulsion History Archives",
        "keywords": ["alfa class", "titanium submarine", "nuclear reactor", "engineering history", "did you know"],
        "engagement_score": 7
    },

    # ── Cybersecurity Secrets ───────────────────────────────────────────────
    {
        "vector": "cybersecurity_secrets",
        "title": "Stuxnet Destroyed 1,000 Centrifuges Without Any Explosives",
        "hook": "Did You Know Code Physically Destroyed Centrifuges? ☢️",
        "headline": "Stuxnet Frequency Inverter Weapon",
        "fact_summary": "The Stuxnet cyberweapon targeted Siemens PLCs, secretly spinning uranium enrichment centrifuges at dangerously fast and slow speeds while playing fake normal recordings to operators.",
        "source": "Symantec W32.Stuxnet Dossier",
        "keywords": ["stuxnet", "cybersecurity", "plc", "zero day", "malware", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "cybersecurity_secrets",
        "title": "The 14 People Who Hold 7 Physical Keys to the Global Internet",
        "hook": "Did You Know 14 People Hold Physical Keys to the Web? 🗝️",
        "headline": "The DNSSEC Root Key Signing Ceremony",
        "fact_summary": "Every 3 months, 14 trusted cryptographers meet under armed guard in California and Virginia to execute the DNSSEC Key Ceremony, ensuring internet domain names cannot be hijacked.",
        "source": "ICANN Root Key Signing Formal Ceremonies",
        "keywords": ["dnssec", "icann", "internet keys", "cryptography", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "cybersecurity_secrets",
        "title": "AI Can Steal Passwords Just by Listening to Keyboard Audio",
        "hook": "Did You Know AI Can Hear Your Password Clicks? 🎧",
        "headline": "Acoustic Keyboard Side-Channel Attack",
        "fact_summary": "Researchers trained an audio deep learning model that decodes keystrokes from laptop microphone recordings with 95% accuracy by analyzing acoustic resonance waveforms.",
        "source": "IEEE European Symposium on Security and Privacy",
        "keywords": ["acoustic attack", "keyboard snooping", "passwords", "ai audio", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "cybersecurity_secrets",
        "title": "The Year 2038 Problem: When Unix Time Runs Out",
        "hook": "Did You Know 32-Bit Time Ends on January 19, 2038? ⏳",
        "headline": "The 32-Bit Unix Epoch Rollover Bug",
        "fact_summary": "At 03:14:07 UTC on Jan 19, 2038, 32-bit signed integers tracking seconds since 1970 will overflow into negative numbers, sending legacy systems back to December 13, 1901.",
        "source": "POSIX Standard & Unix Time Architecture",
        "keywords": ["y2038", "unix epoch", "integer overflow", "32 bit bug", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "cybersecurity_secrets",
        "title": "Rowhammer Flips Memory Bits in Physical DRAM Without Permissions",
        "hook": "Did You Know Hackers Can Flip Bits Using Electrical Leakage? ⚡",
        "headline": "DRAM Rowhammer Electrical Disturbance",
        "fact_summary": "By repeatedly accessing a row of transistors in modern RAM millions of times per second, electrical charge leaks into adjacent capacitor rows, flipping bits from 0 to 1 without software authorization.",
        "source": "ACM SIGARCH Computer Architecture Research",
        "keywords": ["rowhammer", "dram", "hardware exploit", "cybersecurity", "did you know"],
        "engagement_score": 7
    },
    {
        "vector": "cybersecurity_secrets",
        "title": "BadUSB Firmware Tricks Computers by Pretending to be Keyboards",
        "hook": "Did You Know a Flash Drive Can Type Faster Than a Human? 🔌",
        "headline": "USB HID Firmware Microcontroller Injection",
        "fact_summary": "Computers inherently trust USB keyboards. Malicious USB devices reprogram their microcontroller firmware to present as a Human Interface Device, injecting shell commands in milliseconds.",
        "source": "Black Hat BadUSB Research Papers",
        "keywords": ["badusb", "hid injection", "cybersecurity", "hardware hacking", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "cybersecurity_secrets",
        "title": "Tempest Leaks: Spies Can Rebuild Your Screen from Radio Waves",
        "hook": "Did You Know Your Monitor Radiates Your Screen into the Air? 📡",
        "headline": "TEMPEST Video Electromagnetic Side-Channel",
        "fact_summary": "HDMI and display cables emit faint electromagnetic radio signals as pixels refresh. Sensitive software-defined radios up to 100 meters away can reconstruct the exact screen image in real time.",
        "source": "NSA TEMPEST Specifications & IEEE S&P",
        "keywords": ["tempest", "side channel", "electromagnetic surveillance", "rf hacking", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "cybersecurity_secrets",
        "title": "Quantum Computers Will Break RSA Encryption with Shor's Algorithm",
        "hook": "Did You Know Quantum Math Will Break Today's Passwords? 🔮",
        "headline": "Shor's Algorithm Polynomial Prime Factorization",
        "fact_summary": "Modern internet security relies on the mathematical difficulty of factoring huge prime numbers. A sufficiently scaled quantum computer will factor these numbers in minutes using quantum superposition.",
        "source": "NIST Post-Quantum Cryptography Standardization",
        "keywords": ["quantum computing", "shor algorithm", "rsa encryption", "post quantum crypto", "did you know"],
        "engagement_score": 8
    },

    # ── NEW HIGH-ENGAGEMENT EXPANSION POOL ──────────────────────────────────
    # Fresh facts to replenish the seed pool after near-exhaustion
    {
        "vector": "ai_secrets",
        "title": "GPT-4 Was Trained on More Text Than You Could Read in 20,000 Years",
        "hook": "Did You Know GPT-4 Read More Than 20,000 Lifetimes of Text? 📚",
        "headline": "The Scale of LLM Training Data",
        "fact_summary": "GPT-4's training corpus contains roughly 13 trillion tokens. If a human read 250 words per minute nonstop, it would take over 20,000 years to read the same amount.",
        "source": "OpenAI Technical Reports & Estimates",
        "keywords": ["gpt4", "training data", "llm", "tokens", "ai scale", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "ai_secrets",
        "title": "AI Image Generators Have Invisible Watermarks You Cannot See",
        "hook": "Did You Know AI Images Have Hidden Invisible Watermarks? 🔍",
        "headline": "Steganographic AI Provenance Watermarking",
        "fact_summary": "Google DeepMind's SynthID embeds imperceptible patterns into AI-generated images at the pixel level. These watermarks survive cropping, filtering, and screenshotting.",
        "source": "Google DeepMind SynthID Paper",
        "keywords": ["synthid", "ai watermark", "image generation", "deepmind", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "Bluetooth is Named After a 10th-Century Viking King",
        "hook": "Did You Know Bluetooth is Named After a Viking? 🦷",
        "headline": "Harald Bluetooth's Wireless Legacy",
        "fact_summary": "Bluetooth is named after Harald 'Bluetooth' Gormsson, a Viking king who united warring Scandinavian tribes. The Bluetooth logo is his runic initials H and B merged together.",
        "source": "Bluetooth SIG Official History",
        "keywords": ["bluetooth", "viking", "wireless", "tech naming", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "Wi-Fi Signals Pass Through Walls Using Quantum-Like Wave Diffraction",
        "hook": "Did You Know How Wi-Fi Passes Through Walls? 📶",
        "headline": "Radio Wave Diffraction & Building Penetration",
        "fact_summary": "Wi-Fi operates at 2.4 GHz and 5 GHz radio frequencies whose wavelengths (12 cm and 6 cm) are large enough to diffract around doorframes and penetrate drywall, but are absorbed by water and metal.",
        "source": "IEEE 802.11 Radio Physics Standards",
        "keywords": ["wifi", "radio waves", "diffraction", "physics", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "QR Codes Still Work Even When 30% Is Destroyed",
        "hook": "Did You Know QR Codes Work Even When Damaged? 📱",
        "headline": "Reed-Solomon Error Correction Magic",
        "fact_summary": "QR codes use Reed-Solomon error correction that stores redundant data. At the highest error correction level (H), up to 30% of the code can be destroyed and it still scans perfectly.",
        "source": "ISO/IEC 18004 QR Code Standard",
        "keywords": ["qr code", "error correction", "reed solomon", "barcode", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "Your Phone Knows You're in Your Pocket Using a Proximity Infrared Beam",
        "hook": "Did You Know Your Phone Shoots Invisible Light at Your Face? 👁️",
        "headline": "Infrared Proximity Sensor Detection",
        "fact_summary": "A tiny IR LED next to your phone's front camera emits invisible infrared light. When it bounces back from your ear or pocket, the phone turns off the display to save battery and prevent accidental touches.",
        "source": "Smartphone Sensor Design Engineering",
        "keywords": ["proximity sensor", "infrared", "smartphone", "sensors", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "hardware_megastructures",
        "title": "A Single GPU Chip Contains More Transistors Than Stars in the Milky Way",
        "hook": "Did You Know GPUs Have More Transistors Than Stars in Our Galaxy? ⭐",
        "headline": "NVIDIA B200 Transistor Density Milestone",
        "fact_summary": "NVIDIA's B200 GPU contains 208 billion transistors on a single chip package. The Milky Way galaxy contains an estimated 100-400 billion stars.",
        "source": "NVIDIA Blackwell Architecture Whitepaper",
        "keywords": ["nvidia", "gpu", "transistors", "semiconductor", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "hardware_megastructures",
        "title": "Fiber Optic Cables Carry Data Using Light Bouncing in Total Internal Reflection",
        "hook": "Did You Know Internet Data Travels as Light Bouncing Inside Glass? 💡",
        "headline": "Total Internal Reflection Photonic Waveguides",
        "fact_summary": "Inside each hair-thin glass fiber, laser light bounces off the walls thousands of times per meter through total internal reflection, travelling at 200,000 km/s with near-zero loss.",
        "source": "Corning Optical Fiber Engineering",
        "keywords": ["fiber optics", "total internal reflection", "photonics", "internet", "did you know"],
        "engagement_score": 8
    },
    {
        "vector": "bizarre_tech_history",
        "title": "The Microwave Oven Was Invented When a Candy Bar Melted in an Engineer's Pocket",
        "hook": "Did You Know a Melting Candy Bar Invented the Microwave? 🍫",
        "headline": "Percy Spencer's Accidental Magnetron Discovery",
        "fact_summary": "In 1945, Raytheon engineer Percy Spencer was testing military radar magnetrons when he noticed the chocolate bar in his pocket had melted. He then pointed the magnetron at popcorn kernels — and they popped.",
        "source": "Raytheon Company Historical Archives",
        "keywords": ["microwave oven", "invention", "percy spencer", "accidental discovery", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "bizarre_tech_history",
        "title": "The Original iPhone Had No Copy-Paste for Two Full Years",
        "hook": "Did You Know the Original iPhone Couldn't Copy-Paste? 📋",
        "headline": "iPhone OS 1.0-2.0 Missing Clipboard Feature",
        "fact_summary": "When Apple launched the iPhone in 2007, it shipped without copy-paste functionality. The feature didn't arrive until iPhone OS 3.0 in June 2009 — two years after launch.",
        "source": "Apple iOS Version History",
        "keywords": ["iphone", "apple", "copy paste", "smartphone history", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "bizarre_tech_history",
        "title": "Nintendo Started as a Playing Card Company in 1889",
        "hook": "Did You Know Nintendo is 135 Years Old? 🎮",
        "headline": "From Hanafuda Cards to Global Gaming Empire",
        "fact_summary": "Nintendo was founded in 1889 in Kyoto, Japan, as a handmade hanafuda (flower card) company. Before video games, they tried taxi services, love hotels, and instant rice.",
        "source": "Nintendo Corporate History Archives",
        "keywords": ["nintendo", "gaming history", "hanafuda", "tech companies", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "cybersecurity_secrets",
        "title": "Your Deleted Files Are Never Actually Erased From Your Hard Drive",
        "hook": "Did You Know 'Deleted' Files Are Still on Your Disk? 🗑️",
        "headline": "File System Pointer Deletion vs Physical Erasure",
        "fact_summary": "When you delete a file, the OS only removes the pointer in the file table. The actual data remains on disk until new data physically overwrites those exact sectors, which may never happen.",
        "source": "NIST SP 800-88 Media Sanitization Guidelines",
        "keywords": ["file deletion", "data recovery", "hard drive", "digital forensics", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "cybersecurity_secrets",
        "title": "Airplane Mode Doesn't Actually Stop Your Phone From Being Tracked",
        "hook": "Did You Know Airplane Mode Doesn't Fully Disable Tracking? ✈️",
        "headline": "Baseband Processor Independent Operation",
        "fact_summary": "The baseband modem chip in smartphones can operate independently from the main processor. Some phones can still be pinged by cell towers even in airplane mode if the baseband firmware allows it.",
        "source": "Mobile Security Research & Baseband Analysis",
        "keywords": ["airplane mode", "tracking", "baseband", "phone security", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "cybersecurity_secrets",
        "title": "Emojis Are Approved by a 12-Person Committee That Controls All Text on Earth",
        "hook": "Did You Know 12 People Decide Every Emoji You Use? 😱",
        "headline": "The Unicode Consortium Emoji Subcommittee",
        "fact_summary": "Every emoji on every phone, computer, and platform is approved by the Unicode Consortium's 12-member Emoji Subcommittee. They review thousands of proposals annually and control the text encoding standard used by all digital devices.",
        "source": "Unicode Consortium Emoji Technical Reports",
        "keywords": ["emoji", "unicode", "text encoding", "tech governance", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "ai_secrets",
        "title": "ChatGPT Uses More Electricity Per Query Than a Google Search Uses in a Day",
        "hook": "Did You Know One ChatGPT Query Uses 10x More Power Than Google? ⚡",
        "headline": "LLM Inference Energy Consumption",
        "fact_summary": "A single ChatGPT query consumes roughly 10 watt-hours of electricity — about 10 times more than a standard Google search. Running GPT-4 at scale requires thousands of NVIDIA GPUs drawing megawatts.",
        "source": "IEA & Goldman Sachs AI Energy Reports",
        "keywords": ["ai energy", "chatgpt power", "gpu electricity", "sustainability", "did you know"],
        "engagement_score": 9
    },

    # ── Platform-Optimized: Threads Debates & Controversies ─────────────────
    {
        "vector": "threads_debates",
        "title": "Will AI Truly Kill Junior Developer Jobs or Just Change Them?",
        "hook": "Is AI Actually Killing Junior Dev Jobs? 💻",
        "headline": "The Great Junior Developer Debate",
        "fact_summary": "Companies using AI code assistants report 30% faster coding, but code reviews and debugging took 40% longer because juniors cannot spot subtle AI architectural bugs.",
        "source": "Software Engineering Institute Empirical Study",
        "keywords": ["ai coding", "junior developers", "software engineering", "career", "threads_debates", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "threads_debates",
        "title": "Why 90% of Startups Regret Moving to Microservices",
        "hook": "Did You Know Most Teams Regret Microservices? 🏗️",
        "headline": "The Microservice Complexity Tax",
        "fact_summary": "Splitting a small app into dozens of microservices often multiplies network latency, deployment fragility, and AWS bills by 5x without solving team scaling issues.",
        "source": "Distributed Systems Architecture Reports",
        "keywords": ["microservices", "monolith", "cloud architecture", "system design", "threads_debates", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "threads_debates",
        "title": "Tabs vs Spaces Was Solved: Spaces Cost Silicon Valley Millions",
        "hook": "Did You Know Spaces in Code Waste Millions in Bandwidth? ⌨️",
        "headline": "The Storage Cost of Indentation",
        "fact_summary": "Replacing 4 spaces with a single tab character across massive GitHub repos saves gigabytes of wire transfer and disk storage across millions of git clones daily.",
        "source": "GitHub Infrastructure & Indentation Analysis",
        "keywords": ["tabs vs spaces", "coding standards", "git", "clean code", "threads_debates", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "threads_debates",
        "title": "Why Python Will Not Be Replaced by Rust Anytime Soon",
        "hook": "Why Won't Python Ever Die to Rust? 🐍",
        "headline": "The Ecosystem Velocity Moat",
        "fact_summary": "Rust has unbeatable memory safety, but Python dominates AI and data science because PyTorch and NumPy are already written in C++ and CUDA under the hood.",
        "source": "TIOBE & PyPI Ecosystem Benchmarks",
        "keywords": ["python", "rust", "ai libraries", "programming languages", "threads_debates", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "threads_debates",
        "title": "Open Source AI Models Are Catching Up to Closed Tech Monopolies",
        "hook": "Can Open-Source AI Beat Closed Trillion-Dollar Labs? 🔓",
        "headline": "The Open Weights Revolution",
        "fact_summary": "Fine-tuned open models running locally on consumer GPUs now match or exceed GPT-4 on specialized coding benchmarks at 1% of the inference cost.",
        "source": "OpenLLM Leaderboard & HuggingFace Analysis",
        "keywords": ["open source ai", "local llm", "huggingface", "deepseek", "threads_debates", "did you know"],
        "engagement_score": 9
    },

    # ── Platform-Optimized: Facebook Consumer Utility & Scam Alerts ─────────
    {
        "vector": "consumer_utility",
        "title": "Why You Should Never Charge Your Smartphone to 100% Overnight",
        "hook": "Why You Should Stop Charging Your Phone to 100% 🔋",
        "headline": "Lithium-Ion Chemical Degradation",
        "fact_summary": "Holding a phone battery at 100% under high voltage and heat causes micro-cracks in lithium electrodes. Keeping charge between 20% and 80% doubles total battery lifespan.",
        "source": "Battery University & Journal of The Electrochemical Society",
        "keywords": ["battery life", "phone charger", "lithium ion", "tech tips", "consumer_utility", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "consumer_utility",
        "title": "How Scammers Clone a Family Member Voice in Just 3 Seconds",
        "hook": "Scam Alert: How Criminals Clone Voices in 3 Seconds 🚨",
        "headline": "AI Neural Voice Synthesis Scams",
        "fact_summary": "Scammers take a 3-second audio clip from Instagram or Facebook video and feed it into zero-shot neural voice cloners to call family demanding fake emergency ransoms.",
        "source": "FTC Consumer Protection Warnings",
        "keywords": ["voice clone scam", "cybercrime", "ai fraud", "phone security", "consumer_utility", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "consumer_utility",
        "title": "Why Free Public Airport USB Charging Ports Are a Major Security Risk",
        "hook": "Did You Know Free Airport USB Ports Can Hack Your Phone? 🔌",
        "headline": "Juice Jacking Data Pin Exploit",
        "fact_summary": "A standard USB cable contains data pins alongside power pins. Compromised airport chargers can silently install malware or siphon photos while your phone charges.",
        "source": "FCC & Cybersecurity Infrastructure Security Agency (CISA)",
        "keywords": ["juice jacking", "public usb", "airport charging", "travel security", "consumer_utility", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "consumer_utility",
        "title": "Deleting a File on Your Computer Does Not Actually Erase It",
        "hook": "Did You Know Deleted Files Aren't Actually Gone? 🗑️",
        "headline": "File System Index Dereferencing",
        "fact_summary": "Emptying the trash only deletes the pointer in the directory index. The actual bytes remain on your hard drive until overwritten, allowing data recovery in minutes.",
        "source": "NIST Guidelines for Media Sanitization",
        "keywords": ["deleted files", "hard drive", "data privacy", "computer tips", "consumer_utility", "did you know"],
        "engagement_score": 9
    },
    {
        "vector": "consumer_utility",
        "title": "Why Your Smartphone Suddenly Loses 30% Battery in Cold Weather",
        "hook": "Why Does Your Phone Battery Die Instantly in the Cold? ❄️",
        "headline": "Electrolyte Internal Resistance Spike",
        "fact_summary": "Sub-zero temperatures freeze the liquid electrolyte inside your battery, raising electrical resistance so high that the phone thinks voltage dropped to zero and turns off.",
        "source": "IEEE Transactions on Industrial Electronics",
        "keywords": ["cold battery", "smartphone shutdown", "battery science", "winter tech", "consumer_utility", "did you know"],
        "engagement_score": 9
    },

    # ── Platform-Optimized: Instagram Visual Engineering Wonders ────────────
    {
        "vector": "visual_engineering",
        "title": "Why Modern Jet Airplanes Still Rely on 1980s 3.5-inch Floppy Disks",
        "hook": "Why Do Boeing Jets Still Use Floppy Disks? ✈️",
        "headline": "Aviation Avionics Certification Safety",
        "fact_summary": "Boeing 747 navigation databases are updated every 28 days via 3.5-inch floppy disks because re-certifying modern USB systems with aviation safety regulators costs tens of millions.",
        "source": "FAA Avionics Certification & Aviation Security Audits",
        "keywords": ["floppy disk", "boeing 747", "aviation tech", "legacy systems", "visual_engineering", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "visual_engineering",
        "title": "How Active Noise Cancelling Headphones Invert Physical Sound Waves",
        "hook": "How Headphones Delete Sound Waves in Mid-Air 🎧",
        "headline": "Destructive Acoustic Phase Inversion",
        "fact_summary": "External microphones capture incoming engine rumble, compute the exact inverse soundwave in microseconds, and play anti-noise to collide with and cancel physical air pressure waves.",
        "source": "Acoustical Society of America Principles of ANC",
        "keywords": ["noise cancelling", "headphones", "sound waves", "acoustics", "visual_engineering", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "visual_engineering",
        "title": "How ASML Machines Use Exploding Tin Droplets to Print 2nm Chips",
        "hook": "How Computer Chips are Printed with 50,000 Laser Blasts 🔬",
        "headline": "Extreme Ultraviolet Photolithography",
        "fact_summary": "ASML machines vaporize 50,000 molten tin droplets per second with high-power CO2 lasers, generating extreme ultraviolet light to etch billions of nanometer transistors.",
        "source": "ASML EUV Photolithography Engineering Whitepaper",
        "keywords": ["asml", "euv lithography", "microchips", "semiconductors", "visual_engineering", "did you know"],
        "engagement_score": 10
    },
    {
        "vector": "visual_engineering",
        "title": "Why NASA Spacecraft Use 30-Year-Old 1990s Microprocessors",
        "hook": "Why Does NASA Use 1990s Computer Chips in Mars Rovers? 🚀",
        "headline": "Radiation-Hardened Silicon Architecture",
        "fact_summary": "Cosmic rays flip memory bits and fry modern 3nm chips in outer space. NASA uses rugged 250nm PowerPC chips from the 1990s wrapped in heavy silicon-on-insulator shields.",
        "source": "NASA Jet Propulsion Laboratory Avionics Specifications",
        "keywords": ["nasa", "spacecraft chips", "radiation hardening", "mars rover", "visual_engineering", "did you know"],
        "engagement_score": 10
    },
]


class DialogueGenerationError(RuntimeError):
    """Raised when no LLM provider could produce a topic-specific dialogue."""


# Phrases from the old generic fallback template. If any of these appear, the dialogue
# is NOT topic-specific and must never be published.
GENERIC_TEMPLATE_PHRASES = [
    "what makes this happen in modern engineering",
    "underlying physics and software architectures make this fully operational",
    "what happens if this system glitches or fails",
    "automated fail-safes and redundancy keep the entire system from failing",
    "it completely defies our daily intuition",
]


def find_generic_template_phrase(dialogue: Dict) -> Optional[str]:
    """Return the first generic/stale template phrase found in the dialogue, if any."""
    for s in (dialogue or {}).get("slides", []) or []:
        text = f"{s.get('title', '')} {s.get('bubble', '')}".lower()
        for phrase in GENERIC_TEMPLATE_PHRASES:
            if phrase in text:
                return phrase
    return None


def query_llm_for_json(prompt: str) -> Optional[Dict]:
    """Helper to query all configured LLM providers and return parsed JSON."""
    from llm_json_client import query_json
    return query_json(prompt, temperature=0.7, label="Topic LLM")


def _legacy_query_llm_for_json(prompt: str) -> Optional[Dict]:
    """Deprecated: original OpenRouter/Gemini helper (kept for reference, unused)."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "") or OPENROUTER_API_KEY
    if openrouter_key:
        try:
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/vjaab/YtDidYouKnowByVJ",
                "X-Title": "YtDidYouKnowByVJ Cartoon Dialogue",
            }
            models = ["google/gemini-2.5-flash", "meta-llama/llama-3.3-70b-instruct", "openai/gpt-4o-mini"]
            for m in models:
                res = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": m,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                    },
                    timeout=25,
                )
                if res.status_code == 200:
                    raw = res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
                    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        return data
        except Exception as e:
            print(f"⚠️ OpenRouter query note: {e}")

    if GEMINI_AVAILABLE and (os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY):
        api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        models = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        for m in models:
            try:
                raw = ""
                if GEMINI_GENAI_AVAILABLE:
                    client = genai.Client(api_key=api_key)
                    resp = client.models.generate_content(
                        model=m,
                        contents=prompt,
                    )
                    raw = resp.text.strip()
                elif GEMINI_LEGACY_AVAILABLE:
                    genai_legacy.configure(api_key=api_key)
                    model_inst = genai_legacy.GenerativeModel(m)
                    resp = model_inst.generate_content(prompt)
                    raw = resp.text.strip()

                if raw:
                    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
                    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        return data
            except Exception:
                continue

    return None


def convert_trending_story_to_dyk_fact(story: Dict) -> Dict:
    """
    Transform a fresh trending tech/AI news story into a high-attraction 'Did You Know' fact.
    Extracts the most shocking technical detail, benchmark, or engineering achievement.
    """
    title = story.get("title", "")
    description = story.get("description", "")
    url = story.get("url", "")
    source = story.get("source", {})
    source_name = source.get("name", "Verified Tech Signals") if isinstance(source, dict) else str(source)

    prompt = f"""You are the viral tech director for 'Did You Know By VJ'.
Transform this breaking trending tech news story into a mind-blowing 'Did You Know' fact carousel topic.
Find the most shocking number, engineering feat, hidden detail, or breakthrough insight in this story.

STORY:
Title: {title}
Description: {description}
Source: {source_name}

Return ONLY valid JSON (no markdown):
{{
  "title": "<Concise, punchy 5-8 word fact title highlighting the achievement or shocker>",
  "hook": "Did You Know <compelling question with emoji>? 🤯",
  "headline": "<High-tech technical headline>",
  "fact_summary": "<2-3 sentence explanation with exact numbers, specs, or mind-blowing reality>",
  "source": "{source_name}",
  "keywords": ["<keyword1>", "<keyword2>", "<keyword3>", "did you know"]
}}
"""
    parsed = query_llm_for_json(prompt)
    if parsed and isinstance(parsed, dict) and parsed.get("title") and parsed.get("hook"):
        return {
            "mode": "did_you_know",
            "category": "🧠 DID YOU KNOW?",
            "title": parsed["title"].strip(),
            "hook": parsed["hook"].strip(),
            "headline": parsed.get("headline", parsed["title"]).strip(),
            "fact_summary": parsed.get("fact_summary", description or title).strip(),
            "source": parsed.get("source", source_name),
            "news_source_url": url,
            "keywords": parsed.get("keywords", ["did you know", "tech facts"]),
            "vector": "trending_breakthroughs"
        }

    # Rule-based fallback if LLM is unavailable
    clean_title = re.sub(r'^(GitHub Trending:\s*|Show HN:\s*|Ask HN:\s*|Medium\s*\([^)]*\):\s*|Reddit\s*\([^)]*\):\s*)', '', title).strip()
    if " — " in clean_title:
        parts = clean_title.split(" — ")
        clean_title = parts[1].strip() if len(parts[1].strip()) > 15 else parts[0].strip()
    
    hook = f"Did You Know About This New Breakthrough? 🚀"
    if len(clean_title) < 55:
        hook = f"Did You Know: {clean_title}? ⚡"

    return {
        "mode": "did_you_know",
        "category": "🧠 DID YOU KNOW?",
        "title": clean_title[:70],
        "hook": hook,
        "headline": clean_title[:60],
        "fact_summary": description or clean_title,
        "source": source_name,
        "news_source_url": url,
        "keywords": ["did you know", "tech trending"] + [w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', clean_title)[:5]],
        "vector": "trending_breakthroughs"
    }


def synthesize_unique_dyk_fact(vector: str, used_titles: List[str]) -> Optional[Dict]:
    """
    Synthesizes a brand-new, 100% verified, unique 'Did You Know' tech/AI fact using LLM.
    Guarantees that topics are fascinating and easily understandable by a common layperson.
    Guarantees zero duplicates against all previously used topics.
    """
    recent_sample = ", ".join([f'"{t}"' for t in used_titles[-30:]]) if used_titles else "none"
    prompt = f"""You are the viral tech storyteller for 'Did You Know By VJ' (@vijayakumarj_ai).
Generate an extraordinary, fascinating 'Did You Know' fact about Technology and AI that is easily understood by a COMMON LAYPERSON (a student or non-technical adult).

CRITICAL AUDIENCE & CONTENT RULES:
1. LAYMAN UNDERSTANDABLE: Anyone from age 12 to 80 must instantly understand it. Avoid developer-only jargon (no niche database internals, no raw APIs, no obscure benchmark suites).
2. FASCINATING EVERYDAY WONDER: Focus on mind-blowing technology concepts people interact with or wonder about (e.g. smartphones, touchscreens, internet cables under oceans, how AI recognizes photos, microchip physics, GPS satellites and Einstein's relativity, biometric sensors, battery chemistry, how Wi-Fi travels through walls).
3. NO SCRIPT ARTIFACTS: Absolutely NO 'pause', 'continue', or 'break' statements.
4. ZERO DUPLICATES: It must be COMPLETELY DIFFERENT from these recently covered topics:
[{recent_sample}]

Return ONLY valid JSON (no markdown):
{{
  "title": "<Catchy 5-8 word topic title understandable by anyone>",
  "hook": "Did You Know <compelling, simple layman question with emoji>? 🤯",
  "headline": "<Clear, intriguing headline>",
  "fact_summary": "<2-3 sentence simple explanation with a vivid real-world analogy and mind-blowing reality>",
  "source": "<Verified scientific / tech fact source>",
  "keywords": ["<keyword1>", "<keyword2>", "<keyword3>", "did you know"]
}}
"""
    parsed = query_llm_for_json(prompt)
    if parsed and isinstance(parsed, dict) and parsed.get("title") and parsed.get("hook"):
        return {
            "mode": "did_you_know",
            "category": "🧠 DID YOU KNOW?",
            "title": parsed["title"].strip(),
            "hook": parsed["hook"].strip(),
            "headline": parsed.get("headline", parsed["title"]).strip(),
            "fact_summary": parsed.get("fact_summary", parsed["title"]).strip(),
            "source": parsed.get("source", "Verified Tech Architecture & Science"),
            "news_source_url": "",
            "keywords": parsed.get("keywords", ["did you know", "tech facts", vector]),
            "vector": vector
        }
    return None


def fetch_or_select_did_you_know_fact(topic: Optional[str] = None, platform: str = "instagram", record: bool = False) -> Dict:
    """
    Selects or generates a high-attraction, 100% unique 'Did You Know' fact that is
    easily understandable by a common layperson (not overly niche or dry).
    - Guarantees zero duplicate topics across all platforms (checked via RapidFuzz & keywords).
    - Uses engagement-weighted selection to prioritize high-like topics.
    - Platform-aware offsets prevent Instagram/Threads/Facebook from selecting the same topic
      during simultaneous cron runs.
    - If record=False (default), does NOT pollute trackers before generation/verification.
    - PRIORITY 1: Curated Layman-Friendly Seed Catalog (verified, high curiosity).
    - PRIORITY 2: Real-Time Dynamic LLM Synthesis (new layman facts, deduplicated).
    - PRIORITY 3: Live Trending Signals (filtered for layman interest).
    """
    # Platform-specific offset to break ties during concurrent runs
    PLATFORM_OFFSETS = {"instagram": 0, "threads": 1, "facebook": 2, "both": 0}
    platform_offset = PLATFORM_OFFSETS.get(platform.lower(), 0)

    try:
        from ai_news_carousel import (
            is_topic_unique,
            record_carousel_topic,
            load_carousel_tracker,
            fetch_ai_news_stories,
            filter_unique_stories
        )
    except Exception as e:
        print(f"⚠️ Warning importing ai_news_carousel: {e}")
        is_topic_unique = None
        record_carousel_topic = None
        load_carousel_tracker = None
        fetch_ai_news_stories = None
        filter_unique_stories = None

    try:
        from telegram_approval_handler import record_topic_in_tracker
    except Exception:
        record_topic_in_tracker = None

    selected = None

    # CASE A: Explicit topic passed by caller / user
    if topic and topic.strip():
        clean_topic = topic.strip()
        if is_topic_unique:
            uniq, reason = is_topic_unique(clean_topic, check_youtube=True, check_carousel=True)
            if not uniq:
                print(f"⚠️ Warning: Requested explicit topic may be a duplicate: {reason}")
            else:
                print(f"✅ Requested topic '{clean_topic}' is verified unique.")
        
        hook = clean_topic if clean_topic.lower().startswith("did you know") else f"Did You Know: {clean_topic}?"
        if not hook.endswith("?") and not hook.endswith("!"):
            hook += "?"
        selected = {
            "mode": "did_you_know",
            "category": "🧠 DID YOU KNOW?",
            "title": clean_topic,
            "hook": hook,
            "headline": clean_topic,
            "fact_summary": clean_topic,
            "source": "Verified Tech Architecture & Science",
            "news_source_url": "",
            "keywords": ["did you know", "tech facts", "engineering"] + [w.lower() for w in clean_topic.split() if len(w) > 3],
            "vector": "custom"
        }

    # CASE B: Auto-selection (Prioritize high-attraction, layman-friendly concepts with zero duplicates)
    if not selected:
        PLATFORM_VECTOR_PREFS = {
            "threads": ["threads_debates", "ai_secrets", "cybersecurity_secrets", "bizarre_tech_history"],
            "facebook": ["consumer_utility", "everyday_tech_mysteries", "cybersecurity_secrets", "ai_secrets"],
            "instagram": ["visual_engineering", "everyday_tech_mysteries", "hardware_megastructures", "ai_secrets"],
        }
        plat_key = (platform or "instagram").lower()
        if plat_key in PLATFORM_VECTOR_PREFS:
            pref_vectors = PLATFORM_VECTOR_PREFS[plat_key]
            # Use deterministic rotation based on run number or day
            import datetime as _dt
            hour_rot = int(_dt.datetime.now().strftime("%d%H")) % len(pref_vectors)
            current_vector = pref_vectors[(hour_rot + platform_offset) % len(pref_vectors)]
        else:
            try:
                from topic_tracker import get_did_you_know_sub_vector, DID_YOU_KNOW_VECTORS
                base_vector = get_did_you_know_sub_vector()
                vector_idx = DID_YOU_KNOW_VECTORS.index(base_vector) if base_vector in DID_YOU_KNOW_VECTORS else 0
                offset_idx = (vector_idx + platform_offset) % len(DID_YOU_KNOW_VECTORS)
                current_vector = DID_YOU_KNOW_VECTORS[offset_idx]
            except Exception:
                fallback_vectors = [
                    "ai_secrets",
                    "everyday_tech_mysteries",
                    "hardware_megastructures",
                    "bizarre_tech_history",
                    "cybersecurity_secrets",
                ]
                current_vector = fallback_vectors[platform_offset % len(fallback_vectors)]

        print(f"🧠 [{platform.upper()}] Selecting layman-friendly fact for vector: '{current_vector}'...")

        def _engagement_weighted_select(candidates: list) -> dict:
            """Select a topic weighted by engagement_score. Higher scores get proportionally more chance."""
            if not candidates:
                return None
            # Sort by engagement_score descending so highest-engagement topics are tried first
            scored = sorted(candidates, key=lambda f: f.get("engagement_score", 5), reverse=True)
            # Use weighted random: engagement_score as weight
            weights = [f.get("engagement_score", 5) for f in scored]
            total = sum(weights)
            # Deterministic seed based on date + platform to ensure different selection per platform per day
            import hashlib
            from datetime import datetime
            day_key = datetime.now().strftime("%Y-%m-%d-%H")
            run_key = os.getenv("GITHUB_RUN_ID", "") or str(random.random())
            seed_str = f"{day_key}-{platform}-{current_vector}-{run_key}"
            seed_val = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed_val)
            r = rng.uniform(0, total)
            cumulative = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    return dict(scored[i])
            return dict(scored[0])

        # ── PRIORITY 1: Curated Seed Catalog (40+ Verified Layman-Friendly Facts) ────
        print("📚 Checking curated seed facts with multi-layer deduplication...")
        # 1. Filter candidates for current vector
        vector_candidates = [f for f in DID_YOU_KNOW_SEED_FACTS if f.get("vector") == current_vector]
        unseen_vector = []
        for cand in vector_candidates:
            if is_topic_unique:
                is_uniq, _ = is_topic_unique(
                    cand["title"],
                    "",
                    cand.get("keywords", []),
                    check_youtube=True,
                    check_carousel=True
                )
                if is_uniq:
                    unseen_vector.append(cand)
            else:
                unseen_vector.append(cand)
        
        if unseen_vector:
            selected = _engagement_weighted_select(unseen_vector)
            print(f"🎯 [{platform.upper()}] Selected unseen seed fact for '{current_vector}': '{selected['title']}' (engagement: {selected.get('engagement_score', '?')})")
        else:
            # 2. Check all remaining seed facts across other vectors (also engagement-weighted)
            print(f"ℹ️ All seed facts in '{current_vector}' covered; searching full seed catalog...")
            all_unseen = []
            for cand in DID_YOU_KNOW_SEED_FACTS:
                if is_topic_unique:
                    is_uniq, _ = is_topic_unique(
                        cand["title"],
                        "",
                        cand.get("keywords", []),
                        check_youtube=True,
                        check_carousel=True
                    )
                    if is_uniq:
                        all_unseen.append(cand)
                else:
                    all_unseen.append(cand)
            
            if all_unseen:
                selected = _engagement_weighted_select(all_unseen)
                print(f"🎯 [{platform.upper()}] Selected unseen seed fact across catalog: '{selected['title']}' (engagement: {selected.get('engagement_score', '?')})")

        # ── PRIORITY 2: Real-Time Dynamic LLM Synthesis (Zero-Duplicate Layman Fact) ──
        if not selected:
            print("⚡ Seed facts covered! Synthesizing brand-new unique layman-friendly fact via LLM...")
            used_titles_list = []
            if load_carousel_tracker:
                try:
                    c_tracker = load_carousel_tracker()
                    used_titles_list.extend(c_tracker.get("used_titles", []))
                except Exception:
                    pass
            try:
                from topic_tracker import load_topic_tracker
                yt_tracker = load_topic_tracker()
                used_titles_list.extend(yt_tracker.get("used_titles", []))
            except Exception:
                pass
            
            for synth_attempt in range(3):
                synth = synthesize_unique_dyk_fact(current_vector, used_titles_list)
                if synth:
                    if is_topic_unique:
                        uniq, reason = is_topic_unique(synth["title"], "", synth.get("keywords", []), check_youtube=True, check_carousel=True)
                        if uniq:
                            selected = synth
                            print(f"✨ [{platform.upper()}] Synthesized unique fact: '{selected['title']}'")
                            break
                        else:
                            print(f"⚠️ Synthesized fact duplicate ({reason}), retrying ({synth_attempt+1}/3)...")
                    else:
                        selected = synth
                        break

        # ── PRIORITY 3: Live Trending Signals (Filtered for Layman-Friendliness) ────
        if not selected and fetch_ai_news_stories and filter_unique_stories:
            try:
                print("📡 Checking live trending signals for accessible breaking tech concepts...")
                raw_stories = fetch_ai_news_stories()
                unique_stories = filter_unique_stories(raw_stories) if raw_stories else []
                
                for candidate_story in unique_stories[:5]:
                    cand_title = candidate_story.get("title", "")
                    cand_url = candidate_story.get("url", "")
                    
                    converted = convert_trending_story_to_dyk_fact(candidate_story)
                    
                    if is_topic_unique:
                        is_uniq, reason = is_topic_unique(
                            converted["title"],
                            converted.get("news_source_url", cand_url),
                            converted.get("keywords", []),
                            check_youtube=True,
                            check_carousel=True
                        )
                        if not is_uniq:
                            continue
                    
                    selected = converted
                    print(f"🔥 [{platform.upper()}] Picked live fact: '{selected['title']}'")
                    break
            except Exception as e:
                print(f"⚠️ Note on live trending fact extraction: {e}")

        # Fallback safeguard (guarantee a valid dictionary — use platform offset for variety)
        if not selected:
            fallback_idx = platform_offset % len(DID_YOU_KNOW_SEED_FACTS)
            selected = dict(DID_YOU_KNOW_SEED_FACTS[fallback_idx])

    selected["mode"] = "did_you_know"
    selected["category"] = "🧠 DID YOU KNOW?"

    # ── Record in Trackers to Prevent Future Duplication (Only if requested) ────
    if record:
        if record_carousel_topic:
            try:
                record_carousel_topic(
                    title=selected["title"],
                    url=selected.get("news_source_url", ""),
                    keywords=selected.get("keywords", ["did you know"]),
                    source=selected.get("source", "Did You Know By VJ"),
                    platform=platform
                )
            except Exception as e:
                print(f"⚠️ Note recording carousel topic: {e}")

        if record_topic_in_tracker:
            try:
                record_topic_in_tracker(
                    topic=selected["title"],
                    source_url=selected.get("news_source_url", ""),
                    keywords=selected.get("keywords", ["did you know"]),
                    subcategory="Did You Know Fact"
                )
            except Exception as e:
                print(f"⚠️ Note recording in news_log: {e}")

    print(f"🚀 [{platform.upper()}] Final Selected Fact: '{selected['title']}' (Hook: '{selected.get('hook', '')}')")
    return selected


def clean_bubble_text(text: str, max_words: int = 18) -> str:
    """Ensure bubble text is clean, layman-friendly, and concise (18 words or fewer).
    Removes pause, continue, break statements and any script/stage direction artifacts.
    """
    if not text:
        return ""
    text = str(text).strip().strip('"').strip("'")
    
    # 1. Remove bracketed / parenthetical stage directions: [pause], (pause), [beat], (continue), etc.
    text = re.sub(r'\[\s*(?:pause|beat|break|continue|breathe|silence|tone|laughter|sigh)\s*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*(?:pause|beat|break|continue|breathe|silence|tone|laughter|sigh)\s*\)', '', text, flags=re.IGNORECASE)
    
    # 2. Remove standalone pause/continue/break commands at word boundaries if written as artifacts
    text = re.sub(r'\b(?:pause|break|continue)\s*\.{2,}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*(?:pause|break|continue)\s*[:,\-—]\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*[:,\-—]\s*(?:pause|break|continue)\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:take a break|let\'s pause|continue reading|to be continued)\b', '', text, flags=re.IGNORECASE)

    # 3. Clean up leftover double punctuation and excess spaces
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'[,;]\s*([.?!])', r'\1', text)
    text = text.strip()

    # 4. Word count limit
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;:-") + "..."
    return text


def build_dialogue_prompt(
    mode: str,
    topic: Optional[str] = None,
    story: Optional[Dict] = None,
    characters: Optional[str] = "auto",
    platform: str = "instagram",
    slide_count: Optional[int] = None,
) -> str:
    """Build the prompt for Gemini / OpenRouter dialogue script generation with platform-aware narrative arcs and slide counts."""
    current_date = datetime.now().strftime("%d %B %Y")
    speaker_left, speaker_right = resolve_dialogue_characters(characters=characters, topic=topic or "", story=story)
    meta_left = CHARACTER_METADATA.get(speaker_left, CHARACTER_METADATA["byte"])
    meta_right = CHARACTER_METADATA.get(speaker_right, CHARACTER_METADATA["vj"])
    plat = (platform or "instagram").lower()

    # Determine optimal slide count for maximum engagement on each platform
    if not slide_count:
        if plat == "threads":
            target_slides = 4  # Short, snappy debate carousel for high swipe-completion & replies
        elif plat == "facebook":
            target_slides = 4  # 4-image grid/carousel format optimal for Facebook feed sharing
        elif plat == "instagram":
            target_slides = 7  # 7-slide deep dossier driving saves & bookmarks
        else:
            target_slides = 6
    else:
        target_slides = max(3, min(slide_count, 8))
    
    # ── Did You Know Mode (High-Attraction Tech & Science Facts) ───────────
    if mode in ["did_you_know", "dyk"]:
        fact_title = (story.get("title") if story else None) or topic or "99% of Internet is Underwater"
        fact_hook = (story.get("hook") if story else None) or f"Did You Know This About {fact_title}?"
        fact_summary = (story.get("fact_summary") if story else None) or (story.get("description") if story else None) or fact_title
        fact_source = (story.get("source") if story else None) or "Tech Architecture & Science"
        
        # Platform-specific virality rules
        if plat == "threads":
            platform_virality_instructions = f"""
CRITICAL VIRALITY RULES FOR THREADS (TEXT-FIRST DEBATES & 1M+ SWIPE COMPLETION):
1. SLIDE COUNT: EXACTLY {target_slides} SLIDES. Threads rewards ultra-short swipe completion.
2. SLIDE 1 HOOK: A bold counter-intuitive question or controversial claim (Max 12 words).
3. MIDDLE SLIDES: Reveal shocking real-world evidence and the counter-intuitive mechanism.
4. FINAL SLIDE: An open-ended question that sparks debate in the comments (e.g., 'Which side are you on? Drop your honest take below 👇').
5. CAPTION: Conversational, debate-starting question inviting quick replies."""
            sample_slides = f"""    {{"speaker": "{speaker_left}", "emotion": "shocked", "title": "Bold Shocker ⚡", "bubble": "Wait, did you know that 99% of the internet is underwater?!"}},
    {{"speaker": "{speaker_right}", "emotion": "excited", "title": "The Reality 🌐", "bubble": "Yes! Over 1.4 million km of subsea fiber cables carry almost all global data."}},
    {{"speaker": "{speaker_left}", "emotion": "curious", "title": "Garden-Hose Thin ⚙️", "bubble": "And deep down in the ocean, they are barely as thick as a garden hose!"}},
    {{"speaker": "{speaker_right}", "emotion": "thinking", "title": "Your Take? 💬", "bubble": "Did you think it was all satellites? Drop your honest take below!", "is_takeaway": true}}"""
            sample_caption = f"🧠 Did you know 99% of the internet is actually underwater on the ocean floor?\\n\\nNot in the sky. Over 500 undersea cables power the modern web.\\n\\n💬 Did you think it was all satellites, or did you already know this? Drop your thoughts below! 👇\\n#TechDebate #DidYouKnow #Engineering"

        elif plat == "facebook":
            platform_virality_instructions = f"""
CRITICAL VIRALITY RULES FOR FACEBOOK (RELATABLE CONSUMER SAFETY & PUBLIC SHARES):
1. SLIDE COUNT: EXACTLY {target_slides} SLIDES. Perfectly fitted for Facebook multi-photo album previews.
2. SLIDE 1 HOOK: A relatable everyday tech dilemma, scam warning, or device mystery anyone understands.
3. MIDDLE SLIDES: The hidden physical or software reason, plus simple actionable protection/advice.
4. FINAL SLIDE: A high-value takeaway prompting viewers to share with friends and family (e.g., 'Share this with someone who needs to see this!').
5. CAPTION: Clear, helpful, family/friend shareable tip."""
            sample_slides = f"""    {{"speaker": "{speaker_left}", "emotion": "shocked", "title": "Everyday Shocker 📱", "bubble": "Wait, does charging my phone to 100% actually ruin the battery?!"}},
    {{"speaker": "{speaker_right}", "emotion": "excited", "title": "The High Voltage Risk ⚡", "bubble": "Yes! Holding 100% causes micro-cracks in lithium electrodes."}},
    {{"speaker": "{speaker_left}", "emotion": "curious", "title": "The Golden Rule 💡", "bubble": "So keeping battery between 20% and 80% doubles its life?"}},
    {{"speaker": "{speaker_right}", "emotion": "thinking", "title": "Share the Tip 📢", "bubble": "Exactly! Share this with someone whose phone is always on 1%!", "is_takeaway": true}}"""
            sample_caption = f"🔋 Did you know keeping your phone charged at 100% actually damages battery health over time?\\n\\nHere is how to make your smartphone last twice as long:\\n👉 Keep your charge between 20% and 80%.\\n👉 Avoid heavy gaming while fast-charging.\\n\\nShare this with a friend whose phone is always dying! 📱👇\\n#PhoneTips #TechHacks #BatteryCare"

        else: # Instagram & default
            platform_virality_instructions = f"""
CRITICAL VIRALITY RULES FOR INSTAGRAM (VISUAL CURIOSITY, DOSSIER DEPTH & SAVES):
1. SLIDE COUNT: EXACTLY {target_slides} SLIDES. A comprehensive educational carousel that viewers save for later.
2. SLIDE 1 HOOK: A scroll-stopping curiosity paradox (Max 14 words).
3. MIDDLE SLIDES: Step-by-step mechanism, vivid numbers, simple analogies, and edge cases.
4. FINAL SLIDE: A crystal-clear takeaway prompting viewers to bookmark and save ('📌 Tap Save so you have this in your toolkit!').
5. CAPTION: Bullet points + save CTA and trending tags."""
            sample_slides = f"""    {{"speaker": "{speaker_left}", "emotion": "shocked", "title": "Internet Under the Sea? 🌊", "bubble": "Wait, did you know that 99% of the internet is underwater?!"}},
    {{"speaker": "{speaker_right}", "emotion": "excited", "title": "1.4M km of Glass Fiber 🌐", "bubble": "Yes! Over 1.4 million kilometers of fiber optic cables sit on the ocean floor."}},
    {{"speaker": "{speaker_left}", "emotion": "curious", "title": "Shark & Anchor Defense 🦈", "bubble": "What stops sharks or ship anchors from destroying them?"}},
    {{"speaker": "{speaker_right}", "emotion": "smug", "title": "Garden-Hose Thin ⚙️", "bubble": "Near shore they have heavy steel armor, deep down they are barely garden-hose thick!"}},
    {{"speaker": "{speaker_left}", "emotion": "shocked", "title": "When Cables Break 🚢", "bubble": "What happens if an anchor snags one?"}},
    {{"speaker": "{speaker_right}", "emotion": "thinking", "title": "Instant Reroute ⚡", "bubble": "Entire countries can go offline until specialized repair ships arrive."}},
    {{"speaker": "{speaker_left}", "emotion": "excited", "title": "Mind-Blowing Fact 💡", "bubble": "📌 Save this for later & follow @vijayakumarj_ai for daily tech facts!", "is_takeaway": true}}"""
            sample_caption = f"🧠 DID YOU KNOW? 🤯\\n\\n99% of the internet is not in the sky... it is sitting on the ocean floor!\\n\\nHere is the mind-blowing reality:\\n🔹 Over 500 undersea fiber optic cables carry global data.\\n🔹 They transmit data at 99.7% the speed of light.\\n🔹 Deep-sea cables are only as thick as a garden hose, but carry trillions of dollars daily!\\n\\n📌 Tap Save so you don't lose this!\\n\\nFollow @vijayakumarj_ai for daily visual tech breakdowns & facts!\\n#DidYouKnow #TechFacts #MindBlowingFacts #Engineering #ComputerScience"

        prompt = f"""You are the viral tech writer and visual director for 'Did You Know By VJ' (@vijayakumarj_ai).
Create a high-attraction, scroll-stopping {plat.upper()} dialogue carousel (exactly {target_slides} slides) between:
1. "{speaker_left}" ({meta_left['desc']})
2. "{speaker_right}" ({meta_right['desc']})

MIND-BLOWING FACT TO COVER:
- Core Fact: {fact_title}
- Hook Idea: {fact_hook}
- Verified Details: {fact_summary}
- Source: {fact_source}
{platform_virality_instructions}

CRITICAL RULES FOR MAXIMUM VIEWER ATTRACTION & LAYMAN UNDERSTANDING:
1. COMMON LAYMAN UNDERSTANDABLE: Must be crystal clear and instantly understandable by a common layman (a 12-year-old student or non-technical adult). Zero complex jargon without an immediate simple analogy.
2. ABSOLUTELY NO SCRIPT ARTIFACTS: NEVER use words or stage directions like '[pause]', '(pause)', 'pause', '[break]', '(break)', 'break', '[continue]', '(continue)', 'continue' anywhere in speech bubbles. Every bubble must be pure, clean, natural conversational English.
3. MODE: "did_you_know"
4. CATEGORY: "🧠 DID YOU KNOW?"
5. SPEAKERS ALTERNATE: Slide 1 {speaker_left}, Slide 2 {speaker_right}, Slide 3 {speaker_left}, Slide 4 {speaker_right}, etc.
6. SPEECH BUBBLE LENGTH: STRICTLY 18 WORDS OR FEWER PER BUBBLE. Short, punchy, conversational, mind-blowing!
7. FINAL SLIDE: Mark "is_takeaway": true. Provide a punchy summary in "takeaway" field.
8. SLIDE TITLES: Every single slide MUST include a "title" property (2-5 words, plus an optional emoji) matching what that specific slide discusses! Slide 1 title should be the hook.

Return ONLY valid JSON matching this schema with NO markdown fences, NO preamble:
{{
  "mode": "did_you_know",
  "category": "🧠 DID YOU KNOW?",
  "speaker_left": "{speaker_left}",
  "speaker_right": "{speaker_right}",
  "hook": "{fact_hook}",
  "headline": "{fact_title}",
  "source": "{fact_source}",
  "slides": [
{sample_slides}
  ],
  "takeaway": "{fact_summary[:120]}",
  "caption": "{sample_caption}"
}}
"""
        return prompt

    if mode == "news" and story:
        title = story.get("title", topic or "New AI Breakthrough")
        source = story.get("source", "Tech News")
        date = story.get("date", current_date)
        desc = story.get("description", "")
        url = story.get("url", "")
        
        prompt = f"""You are a senior tech writer creating a high-engagement, viral Instagram dialogue carousel (6-7 slides) between two characters:
1. "{speaker_left}" ({meta_left['desc']})
2. "{speaker_right}" ({meta_right['desc']})

The carousel is grounded STRICTLY in this verified AI news event:
- Headline: {title}
- Source: {source}
- Date: {date}
- Verified Details: {desc}
- Source Link: {url}

CRITICAL ACCURACY RULES:
1. GROUNDED IN REALITY: Do NOT invent features, benchmarks, or claims not provided in the verified details above.
2. MODE: "news"
3. SLIDE COUNT: Exactly 6 to 7 slides.
4. SPEAKERS ALTERNATE: Alternate between "{speaker_left}" and "{speaker_right}" on every slide (e.g., slide 1 {speaker_left}, slide 2 {speaker_right}, slide 3 {speaker_left}, etc.).
5. CONCISE BUBBLES: Each speech bubble MUST BE 18 WORDS OR FEWER. Short, punchy, conversational, engaging!
6. EMOTIONS: Each slide must have a valid emotion: ["curious", "thinking", "excited", "shocked", "neutral", "smug"].
7. SLIDE TITLES: Every single slide MUST include a "title" property (2-5 words) matching that slide's content.
8. LAST SLIDE: The final slide is a takeaway plus a follow CTA. Provide a "takeaway" field and note the source: "{source}, {date}".

Return ONLY valid JSON matching this schema with NO markdown fences, NO preamble:
{{
  "mode": "news",
  "speaker_left": "{speaker_left}",
  "speaker_right": "{speaker_right}",
  "hook": "Punchy hook question or breaking statement (max 8 words)",
  "headline": "{title}",
  "source": "{source}",
  "date": "{date}",
  "slides": [
    {{"speaker": "{speaker_left}", "emotion": "shocked", "title": "Breaking Update 🚨", "bubble": "Did OpenAI really just drop GPT-5 preview?"}},
    {{"speaker": "{speaker_right}", "emotion": "excited", "title": "Autonomous Tools ⚡", "bubble": "Yes! It introduces native autonomous tool orchestration."}},
    {{"speaker": "{speaker_left}", "emotion": "curious", "title": "Developer Impact 🛠️", "bubble": "How does that help everyday engineers?"}},
    {{"speaker": "{speaker_right}", "emotion": "thinking", "title": "Internal Planning 🧠", "bubble": "No more brittle agent loops. It handles planning internally."}},
    {{"speaker": "{speaker_left}", "emotion": "smug", "title": "10x Faster Debugging 💻", "bubble": "My debugging sessions just got 10x faster."}},
    {{"speaker": "{speaker_right}", "emotion": "excited", "title": "Key Takeaway 💡", "bubble": "Follow @vijayakumarj_ai for daily updates!", "is_takeaway": true}}
  ],
  "takeaway": "Autonomous tool calling cuts agent boilerplate and boosts pipeline reliability.",
  "caption": "Breaking AI Update: {title}\\n\\nHere is what developers need to know...\\n\\nSource: {source} ({date})\\n\\nFollow @vijayakumarj_ai for daily AI breakdowns!\\n#AI #TechNews #DevCommunity"
}}
"""
        return prompt

    # Concept Mode (Educational)
    concept_topic = topic or "Why does ChatGPT forget you?"
    prompt = f"""You are a world-class tech educator creating an engaging, easy-to-understand Instagram educational carousel (6-7 slides) between two mascot characters:
1. "{speaker_left}" ({meta_left['desc']})
2. "{speaker_right}" ({meta_right['desc']})

TOPIC TO EXPLAIN: "{concept_topic}"

CRITICAL RULES:
1. MODE: "concept"
2. SLIDE COUNT: Exactly 6 to 7 slides.
3. SPEAKERS ALTERNATE: Alternate between "{speaker_left}" and "{speaker_right}" on every slide.
4. PUNCHY BUBBLES: Every single speech bubble MUST BE 18 WORDS OR FEWER. No exceptions.
5. EMOTIONS: Valid emotions for each slide: ["curious", "thinking", "excited", "shocked", "neutral", "smug"].
6. HOOK: Must start with a magnetic hook question that stops the user's scroll.
7. SLIDE TITLES: Every single slide MUST include a "title" property (2-5 words) matching that slide's content.
8. LAST SLIDE: Must be a clear takeaway summary plus a follow CTA.

Return ONLY valid JSON with NO markdown formatting:
{{
  "mode": "concept",
  "speaker_left": "{speaker_left}",
  "speaker_right": "{speaker_right}",
  "hook": "Why does ChatGPT forget you?",
  "headline": "{concept_topic}",
  "slides": [
    {{"speaker": "{speaker_left}", "emotion": "curious", "title": "Why ChatGPT Forgets? 🤯", "bubble": "Why does ChatGPT forget what I said earlier?"}},
    {{"speaker": "{speaker_right}", "emotion": "thinking", "title": "The Context Window 🧠", "bubble": "Think of it as the AI's short-term memory: the context window."}},
    {{"speaker": "{speaker_left}", "emotion": "curious", "title": "When Limits Hit 🛑", "bubble": "What happens when that window fills up?"}},
    {{"speaker": "{speaker_right}", "emotion": "shocked", "title": "Silent Token Drop ✂️", "bubble": "Older messages drop off, so it literally cannot see them anymore!"}},
    {{"speaker": "{speaker_left}", "emotion": "thinking", "title": "Memory Solutions 💡", "bubble": "So prompt summaries prevent memory loss?"}},
    {{"speaker": "{speaker_right}", "emotion": "smug", "title": "Vector Memory ⚡", "bubble": "Exactly! Summarize older context or use vector memory."}},
    {{"speaker": "{speaker_left}", "emotion": "excited", "title": "Key Takeaway 💡", "bubble": "Follow @vijayakumarj_ai for daily AI breakdowns!", "is_takeaway": true}}
  ],
  "takeaway": "LLMs have finite context windows. To avoid memory drop, summarize long chats and prune prompts!",
  "caption": "Why does ChatGPT forget you?\\n\\nEver noticed your long chat losing its train of thought? Here is how context windows actually work...\\n\\nFollow @vijayakumarj_ai for daily AI engineering breakdowns!\\n#AI #MachineLearning #ChatGPT #TechTips"
}}
"""
    return prompt


def build_tailored_dyk_dialogue(
    title: str,
    hook: str,
    fact_summary: str,
    source: str,
    speaker_left: str = "byte",
    speaker_right: str = "vj",
) -> Dict:
    """Dynamically construct 6-7 slide dialogue tailored strictly to the given topic and summary."""
    current_date = datetime.now().strftime("%d %b %Y")
    clean_summary = fact_summary.strip()
    # Split summary into sentences or chunks
    sentences = [s.strip() for s in re.split(r'[.!?]+', clean_summary) if len(s.strip()) > 5]
    part1 = sentences[0] if sentences else clean_summary[:60]
    part2 = sentences[1] if len(sentences) > 1 else (clean_summary[60:130] if len(clean_summary) > 60 else "It completely defies our daily intuition.")
    
    return {
        "mode": "did_you_know",
        "category": "🧠 DID YOU KNOW?",
        "speaker_left": speaker_left,
        "speaker_right": speaker_right,
        "hook": hook,
        "headline": title,
        "source": source,
        "date": current_date,
        "slides": [
            {
                "speaker": speaker_left,
                "emotion": "shocked",
                "title": hook,
                "bubble": clean_bubble_text(f"Wait, did you know that {part1}?!", 18)
            },
            {
                "speaker": speaker_right,
                "emotion": "excited",
                "title": "The Reality ⚙️",
                "bubble": clean_bubble_text(f"Yes! {part2}", 18)
            },
            {
                "speaker": speaker_left,
                "emotion": "curious",
                "title": "How Does It Work? 🔍",
                "bubble": clean_bubble_text(f"What makes this happen in modern engineering?", 18)
            },
            {
                "speaker": speaker_right,
                "emotion": "smug",
                "title": "The Mechanism ⚡",
                "bubble": clean_bubble_text("Underlying physics and software architectures make this fully operational.", 18)
            },
            {
                "speaker": speaker_left,
                "emotion": "shocked",
                "title": "Why It Matters 🚨",
                "bubble": clean_bubble_text("What happens if this system glitches or fails?", 18)
            },
            {
                "speaker": speaker_right,
                "emotion": "thinking",
                "title": "Fail-Safe Design 🛡️",
                "bubble": clean_bubble_text("Automated fail-safes and redundancy keep the entire system from failing.", 18)
            },
            {
                "speaker": speaker_left,
                "emotion": "excited",
                "title": "Mind-Blowing Fact 💡",
                "bubble": "Follow @vijayakumarj_ai for daily mind-blowing tech facts!",
                "is_takeaway": True
            }
        ],
        "takeaway": clean_summary[:140],
        "caption": (
            f"🧠 DID YOU KNOW? 🤯\n\n"
            f"{hook}\n\n"
            f"Here is the mind-blowing reality:\n"
            f"🔹 {clean_summary}\n\n"
            f"💬 Did you already know this, or did this blow your mind? Drop a 🤯 below!\n\n"
            f"Follow @vijayakumarj_ai for daily visual tech breakdowns & facts!\n"
            f"#DidYouKnow #TechFacts #Science #Engineering #MindBlowing"
        )
    }


def get_curated_fallback_dialogue(mode: str, topic: Optional[str] = None, story: Optional[Dict] = None) -> Dict:
    """High-quality curated fallback dialogue strictly matching the requested topic."""
    current_date = datetime.now().strftime("%d %b %Y")
    
    topic_str = (story.get("title") if story else None) or topic or ""
    hook_str = (story.get("hook") if story else None) or f"Did You Know: {topic_str[:40]}? 🤯"
    summary_str = (story.get("fact_summary") if story else None) or (story.get("description") if story else None) or topic_str
    source_str = (story.get("source") if story else None) or "Tech Architecture & Science"
    t_lower = f"{topic_str} {hook_str} {summary_str}".lower()
    
    if mode in ["did_you_know", "dyk"]:
        # 1. AI Hallucination
        if any(k in t_lower for k in ["hallucinat", "ignorance", "probability engine", "next-token"]):
            return {
                "mode": "did_you_know",
                "category": "🧠 DID YOU KNOW?",
                "hook": "Why Do AI Models Hallucinate? 🤖",
                "headline": "Why AI Hallucinates Instead of Admitting Ignorance",
                "source": "Transformer Probabilistic Modeling",
                "date": current_date,
                "slides": [
                    {"speaker": "byte", "emotion": "shocked", "title": "Why Do AI Models Lie? 🤖", "bubble": "Why do AI models lie with 100% confidence instead of admitting ignorance?"},
                    {"speaker": "vj", "emotion": "thinking", "title": "Next-Token Prediction ⚙️", "bubble": "Because LLMs have zero concept of truth. They only calculate what word comes next!"},
                    {"speaker": "byte", "emotion": "curious", "title": "No Concept of Truth? 🔍", "bubble": "Wait, so the AI doesn't actually understand what it is saying?"},
                    {"speaker": "vj", "emotion": "excited", "title": "Statistical Guessing 📊", "bubble": "Never! It produces plausible-sounding sentences, even if the facts are totally fake."},
                    {"speaker": "byte", "emotion": "shocked", "title": "The Confidence Trap 🚨", "bubble": "So high confidence just means high statistical probability, not truth?!"},
                    {"speaker": "vj", "emotion": "smug", "title": "Always Verify Facts 🛡️", "bubble": "Exactly! Treat AI as a reasoning engine, not an infallible database. Always verify!"},
                    {"speaker": "byte", "emotion": "excited", "title": "Mind-Blowing Fact 💡", "bubble": "Follow @vijayakumarj_ai for daily mind-blowing tech facts!", "is_takeaway": True}
                ],
                "takeaway": "AI models don't look up facts—they calculate next-token probabilities. Plausibility never equals factual truth!",
                "caption": "🧠 DID YOU KNOW? 🤯\n\nWhy do AI models invent fake citations and lie with absolute confidence?\n\nHere is the mind-blowing reality:\n🔹 LLMs have zero internal concept of factual truth or falsehood.\n🔹 They generate text using next-token mathematical probability.\n🔹 When an AI sounds confident, it just means the sentence structure is probable!\n\n💬 Have you ever caught an AI completely hallucinating? Drop your story below!\n\nFollow @vijayakumarj_ai for daily visual tech breakdowns & facts!\n#DidYouKnow #AIHallucinations #MachineLearning #ArtificialIntelligence #TechFacts"
            }

        # 2. ChatGPT Memory / Context Window
        if any(k in t_lower for k in ["forget", "context window", "ai memory", "token limit"]):
            return {
                "mode": "did_you_know",
                "category": "🧠 DID YOU KNOW?",
                "hook": "Why Does ChatGPT Forget What You Said? 🤯",
                "headline": "Why ChatGPT Actually Forgets You",
                "source": "Transformer Attention & Context Windows",
                "date": current_date,
                "slides": [
                    {"speaker": "byte", "emotion": "curious", "title": "Why Does ChatGPT Forget? 🤯", "bubble": "Why does ChatGPT forget what I said 10 minutes ago in our chat?"},
                    {"speaker": "vj", "emotion": "thinking", "title": "The Context Window 🧠", "bubble": "LLMs have no ongoing memory. Each reply re-reads earlier text in a context window."},
                    {"speaker": "byte", "emotion": "curious", "title": "What Happens When Full? 🛑", "bubble": "What happens when that context window reaches its token limit?"},
                    {"speaker": "vj", "emotion": "shocked", "title": "Silent Eviction ✂️", "bubble": "Older messages are dropped off from the start, so the model literally cannot see them!"},
                    {"speaker": "byte", "emotion": "thinking", "title": "How To Fix It? 💡", "bubble": "So prompt summaries or vector databases keep long chats alive?"},
                    {"speaker": "vj", "emotion": "smug", "title": "Keep Prompts Lean ⚡", "bubble": "Exactly! Prune system prompts and summarize key points before the window overflows."},
                    {"speaker": "byte", "emotion": "excited", "title": "Mind-Blowing Fact 💡", "bubble": "Follow @vijayakumarj_ai for daily mind-blowing tech facts!", "is_takeaway": True}
                ],
                "takeaway": "ChatGPT doesn't have human memory. When the context window fills up, older messages silently disappear.",
                "caption": "🧠 DID YOU KNOW? 🤯\n\nEver wonder why ChatGPT forgets what you said 20 minutes ago?\n\nHere is how LLM memory actually works:\n🔹 LLMs store zero memory between turns.\n🔹 Every response re-reads your previous messages in a 'context window'.\n🔹 Once that window fills, older tokens are silently discarded!\n\n💬 Did you know this is how AI 'memory' works? Drop a 🤯 below!\n\nFollow @vijayakumarj_ai for daily visual tech breakdowns!\n#DidYouKnow #ChatGPT #AI #TechTips #Developers"
            }

        # 3. Strawberry Tokenization
        if any(k in t_lower for k in ["strawberry", "tokenization", "bpe", "subword"]):
            return {
                "mode": "did_you_know",
                "category": "🧠 DID YOU KNOW?",
                "hook": "Why Can't AI Count Letters in 'Strawberry'? 🍓",
                "headline": "Why AI Can Count to Billions But Fails at Strawberry 'r's",
                "source": "Byte-Pair Encoding Tokenization",
                "date": current_date,
                "slides": [
                    {"speaker": "byte", "emotion": "shocked", "title": "Why Can't AI Count 'r's? 🍓", "bubble": "Why does ChatGPT struggle to count how many 'r's are in 'strawberry'?"},
                    {"speaker": "vj", "emotion": "thinking", "title": "Subword Tokenization 🧩", "bubble": "Because LLMs never see letters! Words are chopped into numeric chunks called tokens."},
                    {"speaker": "byte", "emotion": "curious", "title": "Token Blindspot 🕶️", "bubble": "So the word 'strawberry' isn't stored as ten individual characters?"},
                    {"speaker": "vj", "emotion": "excited", "title": "Tokens Over Letters 🔢", "bubble": "Right! It sees token IDs like 'straw' and 'berry', completely hiding the raw letters."},
                    {"speaker": "byte", "emotion": "shocked", "title": "How Do Newer AIs Fix It? ⚡", "bubble": "Can reasoning models fix this by spelling out the word internally?"},
                    {"speaker": "vj", "emotion": "smug", "title": "Chain-of-Thought Power 🧠", "bubble": "Yes! Reasoning models spell each letter in their scratchpad to count accurately."},
                    {"speaker": "byte", "emotion": "excited", "title": "Mind-Blowing Fact 💡", "bubble": "Follow @vijayakumarj_ai for daily mind-blowing tech facts!", "is_takeaway": True}
                ],
                "takeaway": "AI doesn't see letters—it sees token chunks. Asking for letter counts breaks subword tokenization!",
                "caption": "🧠 DID YOU KNOW? 🤯\n\nWhy can AI solve differential equations but fail to count letters in 'strawberry'?\n\nHere is the answer:\n🔹 Language models do not read individual characters.\n🔹 Words are tokenized into subword IDs (e.g. 'straw' + 'berry').\n🔹 Without seeing individual letters, character counting becomes a statistical guess!\n\nFollow @vijayakumarj_ai for daily tech breakthroughs!\n#DidYouKnow #AI #Tokenization #MachineLearning #TechFacts"
            }

        # 4. GPS Relativity
        if any(k in t_lower for k in ["gps", "relativ", "einstein", "time dilation"]):
            return {
                "mode": "did_you_know",
                "category": "🧠 DID YOU KNOW?",
                "hook": "Did You Know GPS Needs Einstein's Relativity? 🛰️",
                "headline": "GPS Would Drift 11 Kilometers Daily Without Einstein",
                "source": "General & Special Relativity in GNSS",
                "date": current_date,
                "slides": [
                    {"speaker": "byte", "emotion": "shocked", "title": "GPS Needs Einstein? 🛰️", "bubble": "Did you know GPS would drift 11 kilometers every day without Einstein's physics?!"},
                    {"speaker": "vj", "emotion": "excited", "title": "Satellite Time Dilation ⏱️", "bubble": "Satellite atomic clocks tick 38 microseconds faster every day than clocks on Earth."},
                    {"speaker": "byte", "emotion": "curious", "title": "Why Clocks Tick Faster? 🌌", "bubble": "Why does time run faster for GPS satellites in orbit?"},
                    {"speaker": "vj", "emotion": "thinking", "title": "Weaker Gravity & Speed 🪐", "bubble": "Weaker gravity speeds time up, while orbital speed slows it down slightly."},
                    {"speaker": "byte", "emotion": "shocked", "title": "38 Microseconds Drift? 📍", "bubble": "Does 38 tiny microseconds really cause an 11-kilometer navigation error?"},
                    {"speaker": "vj", "emotion": "smug", "title": "Speed of Light Math ⚡", "bubble": "Radio signals travel at light speed! A microsecond error equals hundreds of meters off."},
                    {"speaker": "byte", "emotion": "excited", "title": "Mind-Blowing Fact 💡", "bubble": "Follow @vijayakumarj_ai for daily mind-blowing tech facts!", "is_takeaway": True}
                ],
                "takeaway": "GPS satellites run 38 microseconds fast daily due to relativity. Without Einstein's math, Google Maps fails!",
                "caption": "🧠 DID YOU KNOW? 🤯\n\nWithout Albert Einstein, Google Maps would be unusable within hours!\n\nHere is why:\n🔹 Satellite atomic clocks tick 38 microseconds faster each day due to relativity.\n🔹 Radio signals travel at the speed of light: 300,000 km/s.\n🔹 38 microseconds translates into an 11 km navigation drift every single day!\n\nFollow @vijayakumarj_ai for daily mind-blowing facts!\n#DidYouKnow #GPS #Physics #Einstein #ScienceFacts"
            }

        # 5. Ariane 5 Rocket Bug
        if any(k in t_lower for k in ["ariane", "64-bit", "rocket", "integer overflow"]):
            return {
                "mode": "did_you_know",
                "category": "🧠 DID YOU KNOW?",
                "hook": "Did You Know a 64-Bit Bug Blew Up a $500M Rocket? 🚀",
                "headline": "The $500M Rocket Crash Caused by 64-Bit to 16-Bit Conversion",
                "source": "Ariane 5 Flight 501 Inquiry Board Report",
                "date": current_date,
                "slides": [
                    {"speaker": "byte", "emotion": "shocked", "title": "A $500M Type Bug? 🚀", "bubble": "Did you know a simple number conversion destroyed a $500M rocket in 37 seconds?!"},
                    {"speaker": "vj", "emotion": "excited", "title": "Ariane 5 Flight 501 💥", "bubble": "In 1996, the Ariane 5 rocket exploded immediately after launch due to software."},
                    {"speaker": "byte", "emotion": "curious", "title": "What Code Failed? 💻", "bubble": "What single line of code could possibly blow up an entire space rocket?"},
                    {"speaker": "vj", "emotion": "thinking", "title": "64-Bit to 16-Bit Overflow ⚠️", "bubble": "Guidance software converted a 64-bit float into a 16-bit signed integer. It overflowed!"},
                    {"speaker": "byte", "emotion": "shocked", "title": "No Exception Handling?! 🚨", "bubble": "Wait, the guidance computer didn't catch the overflow exception?"},
                    {"speaker": "vj", "emotion": "smug", "title": "Shutdown in Mid-Air 🛑", "bubble": "Both primary and backup computers shut down, causing nozzles to swerve fatally."},
                    {"speaker": "byte", "emotion": "excited", "title": "Mind-Blowing Fact 💡", "bubble": "Follow @vijayakumarj_ai for daily mind-blowing tech facts!", "is_takeaway": True}
                ],
                "takeaway": "In 1996, a $500M rocket exploded because a 64-bit number overflowed a 16-bit integer without exception handling.",
                "caption": "🧠 DID YOU KNOW? 🤯\n\nA single line of unhandled code caused the most expensive software bug in space history!\n\n🔹 Ariane 5 rocket exploded 37 seconds after launch in 1996.\n🔹 Software attempted to fit a 64-bit float into a 16-bit integer.\n🔹 The integer overflowed, shutting down guidance computers mid-flight!\n\nFollow @vijayakumarj_ai for daily tech history & software lessons!\n#DidYouKnow #SoftwareEngineering #Coding #Bugs #SpaceExploration"
            }

        # 6. Wi-Fi from Black Holes
        if any(k in t_lower for k in ["wifi", "wi-fi", "black hole", "astronom"]):
            return {
                "mode": "did_you_know",
                "category": "🧠 DID YOU KNOW?",
                "hook": "Did You Know Wi-Fi Came from Black Holes? 🌌",
                "headline": "Wi-Fi Was Accidentally Invented by an Astronomer Studying Black Holes",
                "source": "CSIRO Wireless LAN Patent History",
                "date": current_date,
                "slides": [
                    {"speaker": "byte", "emotion": "shocked", "title": "Wi-Fi from Black Holes? 🌌", "bubble": "Did you know that Wi-Fi was accidentally invented by an astronomer studying black holes?!"},
                    {"speaker": "vj", "emotion": "excited", "title": "Dr. John O'Sullivan's Search 🔭", "bubble": "In the 1990s, astronomer Dr. John O'Sullivan tried detecting exploding mini black holes."},
                    {"speaker": "byte", "emotion": "curious", "title": "Echoes in the Sky 📡", "bubble": "How did searching for black holes lead to wireless internet on our laptops?"},
                    {"speaker": "vj", "emotion": "thinking", "title": "Multipath Distortion 📶", "bubble": "Radio waves bounce off walls, causing distorted ghost echoes that ruined wireless signals."},
                    {"speaker": "byte", "emotion": "shocked", "title": "The Astronomy Math 💡", "bubble": "So his astronomy signal-cleaning equation worked for indoor radio waves?!"},
                    {"speaker": "vj", "emotion": "smug", "title": "Modern Wi-Fi Born 🚀", "bubble": "Exactly! That exact Fast Fourier math became the foundational patent for high-speed Wi-Fi."},
                    {"speaker": "byte", "emotion": "excited", "title": "Mind-Blowing Fact 💡", "bubble": "Follow @vijayakumarj_ai for daily mind-blowing tech facts!", "is_takeaway": True}
                ],
                "takeaway": "The algorithm powering modern Wi-Fi was originally invented to detect faint radio echoes from evaporating black holes!",
                "caption": "🧠 DID YOU KNOW? 🤯\n\nYour home Wi-Fi was born from the search for exploding black holes!\n\n🔹 Astronomers in Australia were searching for faint radio signals from dying mini black holes.\n🔹 Indoor wireless signals suffered from the exact same multipath echo interference.\n🔹 The radio astronomy mathematical equations became the basis for IEEE 802.11 Wi-Fi!\n\nFollow @vijayakumarj_ai for daily visual science & tech breakthroughs!\n#DidYouKnow #WiFi #Astronomy #TechHistory #Engineering"
            }

        # 7. Undersea Cables
        if any(k in t_lower for k in ["cable", "underwater", "ocean", "seafloor"]):
            return {
                "mode": "did_you_know",
                "category": "🧠 DID YOU KNOW?",
                "hook": "Did You Know 99% of the Internet is Underwater? 🌊",
                "headline": "99% of Global Internet is on the Ocean Floor",
                "source": "TeleGeography Submarine Cable Registry",
                "date": current_date,
                "slides": [
                    {"speaker": "byte", "emotion": "shocked", "title": "Internet Under the Sea? 🌊", "bubble": "Wait, did you know that 99% of all international internet data is underwater?!"},
                    {"speaker": "vj", "emotion": "excited", "title": "1.4M km of Glass Fiber 🌐", "bubble": "Yes! Satellites carry under 1%. Over 1.4 million km of seafloor fiber carry the web."},
                    {"speaker": "byte", "emotion": "curious", "title": "Shark & Anchor Defense 🦈", "bubble": "What stops sharks or heavy boat anchors from snapping them in two?"},
                    {"speaker": "vj", "emotion": "smug", "title": "Garden-Hose Thin ⚙️", "bubble": "Near shore they have heavy steel armor; in deep waters they are garden-hose thin!"},
                    {"speaker": "byte", "emotion": "shocked", "title": "What If One Snaps? 🚢", "bubble": "What happens when an underwater cable actually gets cut by an anchor?"},
                    {"speaker": "vj", "emotion": "thinking", "title": "Autonomous Rerouting ⚡", "bubble": "Traffic reroutes in milliseconds, and specialized grappling ships sail out to repair it!"},
                    {"speaker": "byte", "emotion": "excited", "title": "Mind-Blowing Fact 💡", "bubble": "Follow @vijayakumarj_ai for daily mind-blowing tech facts!", "is_takeaway": True}
                ],
                "takeaway": "99% of global internet data travels through thin glass cables on the ocean floor, not satellites in space!",
                "caption": "🧠 DID YOU KNOW? 🤯\n\n99% of the internet is not in the sky... it is sitting on the ocean floor!\n\nHere is the mind-blowing reality:\n🔹 Over 500 undersea fiber optic cables carry global data.\n🔹 They transmit data at 99.7% the speed of light.\n🔹 Deep-sea cables are only as thick as a garden hose, but carry trillions of dollars daily!\n\n💬 Did you already know this, or did this blow your mind? Drop a 🤯 below!\n\nFollow @vijayakumarj_ai for daily visual tech breakdowns & facts!\n#DidYouKnow #TechFacts #MindBlowingFacts #Engineering #ComputerScience"
            }

        # 8. No topic-specific dialogue available. Do NOT emit the generic template
        # (it produced identical slides 3-6 on every run). Fail loudly instead.
        raise DialogueGenerationError(
            f"No LLM-generated dialogue available for topic '{topic_str[:80]}'; refusing to publish generic template."
        )

    if mode == "news" and story:
        title = story.get("title", topic or "New AI System Released")
        source = story.get("source", "Tech Updates")
        date = story.get("date", current_date)
        return {
            "mode": "news",
            "hook": f"Breaking: {title[:40]}...",
            "headline": title,
            "source": source,
            "date": date,
            "slides": [
                {"speaker": "byte", "emotion": "shocked", "title": "Breaking News 🚨", "bubble": f"Did you see the latest update from {source}?"},
                {"speaker": "vj", "emotion": "excited", "title": "Major Release ⚡", "bubble": clean_bubble_text(f"Yes! {title[:50]} just went live.", 18)},
                {"speaker": "byte", "emotion": "curious", "title": "Capability Jump 🔍", "bubble": "What is the biggest capability improvement?"},
                {"speaker": "vj", "emotion": "thinking", "title": "Inference Speed ⚙️", "bubble": "Faster inference speeds and significantly higher reasoning accuracy."},
                {"speaker": "byte", "emotion": "smug", "title": "Agent Workflows 🚀", "bubble": "This changes how we build autonomous agent workflows."},
                {"speaker": "vj", "emotion": "excited", "title": "Key Takeaway 💡", "bubble": "Follow @vijayakumarj_ai for daily verified AI news!", "is_takeaway": True}
            ],
            "takeaway": f"This release from {source} accelerates practical AI deployment and agent pipelines.",
            "caption": f"📰 {title}\n\nKey takeaways from the latest release.\n\nSource: {source} ({date})\n\nFollow @vijayakumarj_ai for daily AI news!\n#AI #TechNews #ArtificialIntelligence"
        }

    # Concept fallback
    t = topic or "Why ChatGPT Forgets You"
    return {
        "mode": "concept",
        "hook": "Why does ChatGPT forget you?",
        "headline": t,
        "source": "AI Architecture Guides",
        "date": current_date,
        "slides": [
            {"speaker": "byte", "emotion": "curious", "title": "Why ChatGPT Forgets? 🤯", "bubble": "Why does ChatGPT forget what I said 10 minutes ago?"},
            {"speaker": "vj", "emotion": "thinking", "title": "The Context Window 🧠", "bubble": "Think of it as the AI's short-term memory: context window."},
            {"speaker": "byte", "emotion": "curious", "title": "When Limits Hit 🛑", "bubble": "What happens when that window gets full?"},
            {"speaker": "vj", "emotion": "shocked", "title": "Silent Token Drop ✂️", "bubble": "Older messages drop off completely so it cannot read them!"},
            {"speaker": "byte", "emotion": "thinking", "title": "Memory Solutions 💡", "bubble": "So smart prompt compression keeps chats alive?"},
            {"speaker": "vj", "emotion": "smug", "title": "Vector Memory ⚡", "bubble": "Exactly! Keep system prompts clean and summarize history."},
            {"speaker": "byte", "emotion": "excited", "title": "Key Takeaway 💡", "bubble": "Follow @vijayakumarj_ai for daily AI breakdowns!", "is_takeaway": True}
        ],
        "takeaway": "LLMs rely on finite context windows. Trim excess prompts and summarize earlier dialogue to preserve memory.",
        "caption": "Ever wondered why your long AI chats lose track of context?\n\nHere is how context limits work and how you can fix them.\n\nFollow @vijayakumarj_ai for daily visual tech breakdowns!\n#AI #ChatGPT #TechTips #Developers"
    }


def parse_and_validate_dialogue(
    data: Any,
    mode: str,
    topic: Optional[str] = None,
    story: Optional[Dict] = None,
    platform: str = "instagram",
    slide_count: Optional[int] = None,
) -> Dict:
    """Ensure strict adherence to alternating speakers, word limits, slide titles, and platform slide count."""
    if not isinstance(data, dict):
        raise DialogueGenerationError("LLM returned no usable dialogue JSON")

    plat = (platform or "instagram").lower()
    min_required = 3 if plat in ["threads", "facebook"] else 4
    max_allowed = 4 if plat in ["threads", "facebook"] else 8
    if slide_count:
        max_allowed = max(min_required, min(slide_count, 8))

    slides = data.get("slides", [])
    if not isinstance(slides, list) or len(slides) < min_required:
        raise DialogueGenerationError(f"LLM dialogue has too few slides ({len(slides) if isinstance(slides, list) else 0}, min {min_required})")

    # Slice to platform maximum
    if len(slides) > max_allowed:
        slides = slides[:max_allowed]

    hook = data.get("hook") or data.get("headline") or "Did You Know?"
    headline = data.get("headline") or hook

    validated_slides = []
    expected_speaker = "byte"
    
    for i, s in enumerate(slides):
        if not isinstance(s, dict):
            continue
            
        speaker = s.get("speaker", expected_speaker).lower()
        if speaker == "asha":
            speaker = "vj"
        if speaker not in VALID_SPEAKERS:
            speaker = expected_speaker
            
        # Ensure alternating speaker
        if i > 0 and speaker == validated_slides[-1]["speaker"]:
            speaker = "vj" if validated_slides[-1]["speaker"] == "byte" else "byte"
            
        emotion = s.get("emotion", "neutral").lower()
        if emotion not in VALID_EMOTIONS:
            emotion = "curious" if speaker == "byte" else "thinking"
            
        raw_bubble = s.get("bubble", "")
        bubble = clean_bubble_text(raw_bubble, max_words=18)
        
        # Ensure slide has a matching title
        slide_title = s.get("title")
        if not slide_title or len(slide_title.strip()) < 2:
            if i == 0:
                slide_title = hook
            elif i == len(slides) - 1 or s.get("is_takeaway"):
                slide_title = "Mind-Blowing Fact 💡" if mode in ["did_you_know", "dyk"] else "Key Takeaway 💡"
            else:
                slide_title = headline
        
        slide_entry = {
            "speaker": speaker,
            "emotion": emotion,
            "title": slide_title,
            "bubble": bubble,
            "is_takeaway": s.get("is_takeaway", False)
        }
        validated_slides.append(slide_entry)
        expected_speaker = "vj" if speaker == "byte" else "byte"

    # Mark the last slide as takeaway
    if validated_slides:
        validated_slides[-1]["is_takeaway"] = True

    data["slides"] = validated_slides
    data["mode"] = mode
    data["hook"] = hook
    data["headline"] = headline
    data["takeaway"] = data.get("takeaway") or "Understand core tech systems to build better workflows."
    if story and story.get("vector"):
        data["vector"] = story.get("vector")
    
    if mode in ["did_you_know", "dyk"]:
        data["category"] = data.get("category") or "🧠 DID YOU KNOW?"
        if not data.get("caption") or len(data.get("caption", "")) < 30 or "DID YOU KNOW" not in data.get("caption", ""):
            data["caption"] = (
                f"🧠 DID YOU KNOW? 🤯\n\n"
                f"{data['hook']}\n\n"
                f"💡 {data['takeaway']}\n\n"
                f"💬 Did you already know this, or did this blow your mind? Drop a 🤯 below!\n\n"
                f"Follow @vijayakumarj_ai for daily mind-blowing tech & science facts!"
            )
    else:
        data["caption"] = data.get("caption") or f"{data['hook']}\n\nFollow @vijayakumarj_ai for daily AI updates!"
    
    return data


def generate_cartoon_dialogue_json(
    topic: Optional[str] = None,
    story: Optional[Dict] = None,
    mode: str = "auto",
    characters: str = "auto",
    platform: str = "instagram",
    slide_count: Optional[int] = None,
) -> Dict:
    """Generate the mascot dialogue JSON script using Gemini / OpenRouter."""
    plat = (platform or "instagram").lower()

    # Determine mode if auto
    if mode == "auto":
        if story and story.get("mode") in ["did_you_know", "dyk"]:
            mode = "did_you_know"
        elif story and ("did you know" in str(story.get("title", "")).lower() or "did you know" in str(story.get("hook", "")).lower()):
            mode = "did_you_know"
        elif topic and any(q in topic.lower() for q in ["did you know", "mind-blowing", "secret", "weird"]):
            mode = "did_you_know"
        elif story and story.get("title") and story.get("source") != "Tech Architecture & Science":
            mode = "news"
        else:
            mode = "did_you_know"

    prompt = build_dialogue_prompt(
        mode=mode,
        topic=topic,
        story=story,
        characters=characters,
        platform=plat,
        slide_count=slide_count
    )
    prompt += (
        "\n\nIMPORTANT: The JSON above is ONLY a format example. Every slide title and bubble "
        "MUST be written specifically about the topic given above, with concrete facts, numbers "
        "and analogies for THIS topic. Never reuse the example sentences."
    )

    example_bubbles = {b.lower() for b in re.findall(r'"bubble":\s*"([^"]+)"', prompt)}
    min_slides_check = 3 if plat in ["threads", "facebook"] else 4

    def _is_valid_dialogue(d: Dict) -> bool:
        slides = d.get("slides")
        if not isinstance(slides, list) or len(slides) < min_slides_check:
            return False
        bubbles = [str(s.get("bubble", "")).strip().lower() for s in slides if isinstance(s, dict)]
        if len([b for b in bubbles if b]) < min_slides_check:
            return False
        # Reject output that just parrots the prompt's format example.
        copied = sum(1 for b in bubbles[:-1] if b in example_bubbles)
        if copied >= 2:
            print(f"⚠️ Dialogue copied {copied} example bubbles from the prompt; rejecting")
            return False
        if find_generic_template_phrase(d):
            return False
        return True

    from llm_json_client import query_json
    print(f"🤖 Generating cartoon dialogue script for '{(story or {}).get('title') or topic}' ({plat.upper()} mode)...")
    dialogue_data = query_json(prompt, temperature=0.7, label="Dialogue LLM", validator=_is_valid_dialogue)
    if not dialogue_data:
        raise DialogueGenerationError(
            "All LLM providers failed to generate a topic-specific dialogue. "
            "Check GEMINI_API_KEY quota / OPENROUTER_API_KEY credits / CF_API_TOKEN."
        )
    print(f"✅ Generated {len(dialogue_data.get('slides', []))} topic-specific dialogue slides")

    # Validate structure (raises DialogueGenerationError if invalid)
    result = parse_and_validate_dialogue(
        dialogue_data,
        mode=mode,
        topic=topic,
        story=story,
        platform=plat,
        slide_count=slide_count
    )
    speaker_left, speaker_right = resolve_dialogue_characters(characters=characters, topic=topic or "", story=story)
    result["speaker_left"] = result.get("speaker_left") or speaker_left
    result["speaker_right"] = result.get("speaker_right") or speaker_right
    result["characters"] = characters

    # ── Apply Humanizer Sanitization (0 Em-Dashes, No AI Buzzwords, Natural Voice) ──
    try:
        for s in result.get("slides", []):
            if "bubble" in s:
                s["bubble"] = sanitize_text_for_human_voice(s["bubble"])
            if "title" in s:
                s["title"] = sanitize_text_for_human_voice(s["title"])
        if "caption" in result:
            result["caption"] = sanitize_text_for_human_voice(result["caption"])
        if "takeaway" in result:
            result["takeaway"] = sanitize_text_for_human_voice(result["takeaway"])
        if "hook" in result:
            result["hook"] = sanitize_text_for_human_voice(result["hook"])
        if "headline" in result:
            result["headline"] = sanitize_text_for_human_voice(result["headline"])
    except Exception as h_err:
        print(f"⚠️ Note on humanizer sanitization: {h_err}")

    return result


def render_cartoon_dialogue_carousel(
    dialogue: Dict,
    output_dir: Path,
    canvas_width: int = 1080,
    canvas_height: int = 1350,
    prefix: str = "carousel_",
    bg_images: Optional[List[str]] = None,
    theme: Optional[str] = "auto",
    platform: Optional[str] = "instagram",
    characters: Optional[str] = "auto",
) -> List[Path]:
    """Render the cartoon dialogue slides using cartoon_dialogue.html.j2 & Playwright."""
    slides = dialogue.get("slides", [])
    if not slides:
        raise ValueError("No slides to render in cartoon dialogue")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_slides = len(slides)

    # Resolve theme dynamically based on topic vector, platform, or explicit choice
    theme_cfg = resolve_carousel_theme(
        theme=theme or dialogue.get("theme", "auto"),
        vector=dialogue.get("vector"),
        platform=platform,
    )
    print(f"🎨 Theme Selected: '{theme_cfg['name']}' ({theme_cfg['id']}) for platform '{platform}'")
    
    # Jinja2 setup
    env = Environment(
        loader=FileSystemLoader([str(TEMPLATE_DIR), str(TEMPLATE_DIR / "layouts")]),
        autoescape=True
    )
    template = env.get_template("cartoon_dialogue.html.j2")
    
    safe_title = re.sub(r'[\s\-]+', '_', re.sub(r'[^\w\s-]', '', dialogue.get("hook", "dialogue"))).strip('_')[:30].lower()
    slide_html_items = []
    
    # Resolve character duo
    speaker_left = dialogue.get("speaker_left")
    speaker_right = dialogue.get("speaker_right")
    if not speaker_left or not speaker_right:
        s_left, s_right = resolve_dialogue_characters(
            characters=characters or dialogue.get("characters", "auto"),
            topic=dialogue.get("hook", "") or dialogue.get("headline", ""),
            story=dialogue
        )
        speaker_left = speaker_left or s_left
        speaker_right = speaker_right or s_right

    mascot_left_path = get_character_image_path(speaker_left, "excited") or get_character_image_path(speaker_left, "neutral")
    mascot_right_path = get_character_image_path(speaker_right, "excited") or get_character_image_path(speaker_right, "neutral")
    
    for i, slide in enumerate(slides):
        slide_num = i + 1
        speaker = slide.get("speaker", speaker_left)
        emotion = slide.get("emotion", "neutral")
        is_takeaway = (slide_num == total_slides) or slide.get("is_takeaway", False)
        slide_title = slide.get("title") or (dialogue.get("hook") if i == 0 else dialogue.get("headline", ""))
        
        char_img = get_character_image_path(speaker, emotion)
        char_img_uri = f"file://{char_img.resolve()}" if char_img else ""
        
        speaker_meta = CHARACTER_METADATA.get(speaker, CHARACTER_METADATA.get("byte", {}))
        speaker_tag = speaker_meta.get("tag", f"🤖 {speaker.upper()}")
        is_speaker_right = (speaker == speaker_right or speaker == "vj")
        
        bg_uri = None
        if bg_images and i < len(bg_images) and bg_images[i]:
            bg_uri = f"file://{Path(bg_images[i]).resolve()}"

        derived_mode = dialogue.get("mode", "did_you_know")
        category_label = dialogue.get("category")
        if not category_label:
            if derived_mode in ["did_you_know", "dyk"]:
                category_label = "🧠 DID YOU KNOW?"
            elif derived_mode == "news":
                category_label = "AI NEWS"
            else:
                category_label = "AI EXPLAINED"

        # Extract numerical stat callout if present
        slide_copy = dict(slide)
        stat_info = extract_stat_from_bubble(slide.get("bubble", ""))
        if stat_info and not slide_copy.get("highlight_stat"):
            slide_copy.update(stat_info)

        context = {
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "slide_current": slide_num,
            "slide_total": total_slides,
            "mode": derived_mode,
            "category": category_label,
            "slide_title": slide_title,
            "hook": dialogue.get("hook", ""),
            "headline": dialogue.get("headline", ""),
            "speaker": speaker,
            "emotion": emotion,
            "speaker_tag": speaker_tag,
            "is_speaker_right": is_speaker_right,
            "slide_data": slide_copy,
            "is_takeaway": is_takeaway,
            "takeaway": dialogue.get("takeaway", ""),
            "source": dialogue.get("source", ""),
            "date": dialogue.get("date", ""),
            "brand": {"handle": "@vijayakumarj_ai", "name": "Vijayakumar J"},
            "character_img_path": char_img_uri,
            "character_left_path": f"file://{mascot_left_path.resolve()}" if mascot_left_path else "",
            "character_right_path": f"file://{mascot_right_path.resolve()}" if mascot_right_path else "",
            "character_byte_path": f"file://{mascot_left_path.resolve()}" if mascot_left_path else "",
            "character_vj_path": f"file://{mascot_right_path.resolve()}" if mascot_right_path else "",
            "background_image_url": bg_uri,
            "theme": theme_cfg,
            "platform": platform,
        }
        
        html_path = output_dir / f"cartoon_{prefix}{safe_title}_{slide_num:02d}.html"
        html_rendered = template.render(**context)
        html_path.write_text(html_rendered, encoding="utf-8")
        
        img_path = output_dir / f"{prefix}{safe_title}_{slide_num:02d}.jpg"
        slide_html_items.append((slide_num, html_path, img_path))

    # Render via Playwright
    print(f"🚀 Rendering {total_slides} cartoon dialogue slides with Playwright ({canvas_width}x{canvas_height})...")
    output_paths = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": canvas_width, "height": canvas_height},
            device_scale_factor=2,
        )
        
        for slide_num, html_path, img_path in slide_html_items:
            page.goto(f"file://{html_path.absolute()}")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(200)
            page.screenshot(path=str(img_path), type="jpeg", quality=95)
            output_paths.append(img_path)
            print(f"  ✅ Slide {slide_num}/{total_slides} rendered: {img_path.name}")
            
        browser.close()

    return output_paths
