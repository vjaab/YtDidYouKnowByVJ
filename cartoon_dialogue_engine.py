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

BASE_DIR = Path(__file__).parent
CHARACTERS_DIR = BASE_DIR / "assets" / "characters"
TEMPLATE_DIR = BASE_DIR / "carousel_templates"

# Mascot configuration
VALID_SPEAKERS = ["byte", "vj"]
VALID_EMOTIONS = ["neutral", "curious", "excited", "shocked", "thinking", "smug"]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def get_character_image_path(speaker: str, emotion: str) -> Optional[Path]:
    """Retrieve absolute file path for a speaker's emotion sprite."""
    speaker = speaker.lower().strip()
    emotion = emotion.lower().strip()
    
    # Backward compatibility alias
    if speaker == "asha":
        speaker = "vj"
    
    if speaker not in VALID_SPEAKERS:
        speaker = "byte"
    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"
        
    # Check folder structure: assets/characters/vj/curious.png
    nested_path = CHARACTERS_DIR / speaker / f"{emotion}.png"
    if nested_path.exists():
        return nested_path
        
    # Check flat structure: assets/characters/vj_curious.png
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
        
    # Fallback to neutral
    fallback_path = CHARACTERS_DIR / speaker / "neutral.png"
    if fallback_path.exists():
        return fallback_path
        
    return None


try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


# ── DID YOU KNOW CURATED SEED FACT POOL ───────────────────────────────────────
# High-attraction, verified mind-blowing facts rotating across 5 curiosity vectors
DID_YOU_KNOW_SEED_FACTS = [
    # ── AI Secrets ──────────────────────────────────────────────────────────
    {
        "vector": "ai_secrets",
        "title": "Why ChatGPT Actually Forgets You",
        "hook": "Why Does ChatGPT Forget What You Said? 🤯",
        "headline": "The Context Window Reality",
        "fact_summary": "LLMs do not store memories between turns. Every reply re-reads earlier text until the context window overflows, silently dropping older tokens from the start.",
        "source": "Transformer Attention & Context Windows",
        "keywords": ["chatgpt", "context window", "ai memory", "tokens", "llm", "did you know"]
    },
    {
        "vector": "ai_secrets",
        "title": "Why AI Hallucinates Instead of Admitting Ignorance",
        "hook": "Why Do AI Models Hallucinate? 🤖",
        "headline": "Next-Token Probability Engine",
        "fact_summary": "AI has zero concept of factual truth. It calculates mathematical probabilities of what word should come next, generating convincing falsehoods when confidence is low.",
        "source": "Transformer Probabilistic Modeling",
        "keywords": ["ai hallucination", "transformers", "machine learning", "probability", "did you know"]
    },
    {
        "vector": "ai_secrets",
        "title": "How AI Generates Images from Pure Static Noise",
        "hook": "Did You Know AI Paints from Pure TV Static? 🎨",
        "headline": "Diffusion Reverse Denoising",
        "fact_summary": "Diffusion models like Midjourney start with 100% random static fuzz and gradually subtract noise over 50 steps until a crisp image crystallizes.",
        "source": "Denoising Diffusion Probabilistic Models",
        "keywords": ["diffusion models", "midjourney", "image generation", "ai art", "did you know"]
    },
    {
        "vector": "ai_secrets",
        "title": "Why AI Can Count to Billions But Fails at Strawberry 'r's",
        "hook": "Why Can't AI Count Letters in 'Strawberry'? 🍓",
        "headline": "The Subword Tokenization Blindspot",
        "fact_summary": "LLMs never see raw characters. Words are sliced into multi-letter token IDs, making it impossible for the model to see individual letters without spelling them out.",
        "source": "Byte-Pair Encoding Tokenization",
        "keywords": ["tokenization", "bpe", "strawberry", "llm reasoning", "did you know"]
    },

    # ── Everyday Tech Mysteries ─────────────────────────────────────────────
    {
        "vector": "everyday_tech_mysteries",
        "title": "99% of Global Internet is on the Ocean Floor",
        "hook": "Did You Know 99% of the Internet is Underwater? 🌊",
        "headline": "Subsea Fiber Optic Megastructure",
        "fact_summary": "Satellites carry under 1% of data. Over 1.4 million kilometers of submarine fiber optic cables, armored against sharks and anchors, carry all global internet.",
        "source": "TeleGeography Submarine Cable Registry",
        "keywords": ["submarine cables", "internet", "fiber optics", "ocean floor", "did you know"]
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "GPS Would Drift 11 Kilometers Daily Without Einstein",
        "hook": "Did You Know GPS Needs Einstein's Relativity? 🛰️",
        "headline": "Relativistic Satellite Time Dilation",
        "fact_summary": "Satellite clocks tick 38 microseconds faster per day due to weaker gravity and high speed. Without relativistic math correction, Google Maps would drift 11 km every day.",
        "source": "General & Special Relativity in GNSS",
        "keywords": ["gps", "einstein", "relativity", "time dilation", "satellites", "did you know"]
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "Your Smartphone Screen Steals Your Electrons",
        "hook": "Did You Know Your Phone Steals Your Electrons? 📱",
        "headline": "Capacitive Touchscreen Physics",
        "fact_summary": "Phone glass does not detect pressure. A grid of transparent indium tin oxide electrodes detects tiny electrical charges transferring from your skin when you touch it.",
        "source": "Capacitive Sensing Physics & IEEE",
        "keywords": ["touchscreen", "capacitance", "smartphone", "physics", "did you know"]
    },
    {
        "vector": "everyday_tech_mysteries",
        "title": "How Airplanes Get Wi-Fi at 35,000 Feet Over Oceans",
        "hook": "Did You Know How Planes Get Wi-Fi Mid-Flight? ✈️",
        "headline": "Gimbaled Satellite Phased Arrays",
        "fact_summary": "Planes use motorized parabolic antennas inside a teardrop roof dome that track geostationary satellites 36,000 km away while flying at 900 km/h.",
        "source": "Aeronautical Satellite Telecommunications",
        "keywords": ["airplane wifi", "satellite", "aviation tech", "phased array", "did you know"]
    },

    # ── Hardware Megastructures ─────────────────────────────────────────────
    {
        "vector": "hardware_megastructures",
        "title": "ASML Chip Lasers Fire 50,000 Times a Second at Molten Tin",
        "hook": "Did You Know How the World's Microchips are Made? 🔬",
        "headline": "ASML Extreme Ultraviolet Lithography",
        "fact_summary": "A high-powered CO2 laser vaporizes 50,000 drops of molten tin per second into plasma hotter than the sun's surface to produce 13.5nm light waves.",
        "source": "ASML High-NA EUV Engineering",
        "keywords": ["asml", "euv", "semiconductors", "chip making", "microprocessors", "did you know"]
    },
    {
        "vector": "hardware_megastructures",
        "title": "Microsoft Sunk a Datacenter 117 Feet Under the Sea",
        "hook": "Did You Know Datacenters Run Under the Sea? 🌊",
        "headline": "Project Natick Underwater Server Pods",
        "fact_summary": "Microsoft submerged 864 servers in a sealed nitrogen capsule off Scotland. With no humans and constant natural seawater cooling, server failure dropped by 800%.",
        "source": "Microsoft Project Natick Research",
        "keywords": ["project natick", "underwater datacenter", "cloud servers", "microsoft", "did you know"]
    },
    {
        "vector": "hardware_megastructures",
        "title": "Cleanrooms are 10,000x Cleaner Than Hospital Surgery Rooms",
        "hook": "Did You Know Chip Cleanrooms Beat Surgery Rooms? 🧪",
        "headline": "ISO Class 1 Semiconductor Cleanrooms",
        "fact_summary": "A single speck of human dead skin or dust can bridge microscopic transistor paths. Air in chip cleanrooms is filtered to under 10 particles per cubic meter.",
        "source": "Semiconductor Fab Standards (ISO 14644)",
        "keywords": ["cleanroom", "semiconductors", "fab", "transistors", "did you know"]
    },

    # ── Bizarre Tech History ────────────────────────────────────────────────
    {
        "vector": "bizarre_tech_history",
        "title": "The $500M Rocket Crash Caused by 64-Bit to 16-Bit Conversion",
        "hook": "Did You Know a 64-Bit Bug Blew Up a $500M Rocket? 🚀",
        "headline": "Ariane 5 Flight 501 Integer Overflow",
        "fact_summary": "In 1996, the Ariane 5 rocket exploded 37 seconds after launch because software tried to stuff a 64-bit floating point number into a 16-bit integer, causing fatal overflow.",
        "source": "Ariane 5 Flight 501 Inquiry Board Report",
        "keywords": ["ariane 5", "integer overflow", "software bug", "rocket science", "did you know"]
    },
    {
        "vector": "bizarre_tech_history",
        "title": "Wi-Fi Was Accidentally Invented by an Astronomer Studying Black Holes",
        "hook": "Did You Know Wi-Fi Came from Black Holes? 🌌",
        "headline": "CSIRO Radio Astronomy Invention",
        "fact_summary": "In the 1990s, Australian astronomer Dr. John O'Sullivan was trying to detect exploding mini black holes using radio waves. The signal-cleaning algorithm became modern Wi-Fi.",
        "source": "CSIRO Wireless LAN Patent History",
        "keywords": ["wifi", "invention", "black holes", "radio astronomy", "csiro", "did you know"]
    },
    {
        "vector": "bizarre_tech_history",
        "title": "The First Computer Bug Was an Actual Moth",
        "hook": "Did You Know the First Bug Was a Real Moth? 🦋",
        "headline": "Grace Hopper's 1947 Harvard Relay Bug",
        "fact_summary": "In 1947, computer pioneer Grace Hopper's team investigated a malfunction in the Harvard Mark II relay computer and found a live moth trapped between Relay #70.",
        "source": "Smithsonian National Museum of American History",
        "keywords": ["computer bug", "grace hopper", "harvard mark ii", "tech history", "did you know"]
    },
    {
        "vector": "bizarre_tech_history",
        "title": "Why NASA Lost a $327M Mars Orbiter Over Metric Units",
        "hook": "Did You Know Metric vs Imperial Crashed a Mars Probe? 🪐",
        "headline": "Mars Climate Orbiter Navigation Loss",
        "fact_summary": "Lockheed Martin software calculated thruster impulse in pound-seconds, but NASA navigation software expected metric newton-seconds. The orbiter incinerated in Mars' atmosphere.",
        "source": "NASA Mars Climate Orbiter Mishap Board",
        "keywords": ["mars climate orbiter", "nasa", "metric system", "spacecraft", "did you know"]
    },

    # ── Cybersecurity Secrets ───────────────────────────────────────────────
    {
        "vector": "cybersecurity_secrets",
        "title": "Stuxnet Destroyed 1,000 Centrifuges Without Any Explosives",
        "hook": "Did You Know Code Physically Destroyed Centrifuges? ☢️",
        "headline": "Stuxnet Frequency Inverter Weapon",
        "fact_summary": "The Stuxnet cyberweapon targeted Siemens PLCs, secretly spinning uranium enrichment centrifuges at dangerously fast and slow speeds while playing fake normal recordings to operators.",
        "source": "Symantec W32.Stuxnet Dossier",
        "keywords": ["stuxnet", "cybersecurity", "plc", "zero day", "malware", "did you know"]
    },
    {
        "vector": "cybersecurity_secrets",
        "title": "The 14 People Who Hold 7 Physical Keys to the Global Internet",
        "hook": "Did You Know 14 People Hold Physical Keys to the Web? 🗝️",
        "headline": "The DNSSEC Root Key Signing Ceremony",
        "fact_summary": "Every 3 months, 14 trusted cryptographers meet under armed guard in California and Virginia to execute the DNSSEC Key Ceremony, ensuring internet domain names cannot be hijacked.",
        "source": "ICANN Root Key Signing Formal Ceremonies",
        "keywords": ["dnssec", "icann", "internet keys", "cryptography", "did you know"]
    },
    {
        "vector": "cybersecurity_secrets",
        "title": "AI Can Steal Passwords Just by Listening to Keyboard Audio",
        "hook": "Did You Know AI Can Hear Your Password Clicks? 🎧",
        "headline": "Acoustic Keyboard Side-Channel Attack",
        "fact_summary": "Researchers trained an audio deep learning model that decodes keystrokes from laptop microphone recordings with 95% accuracy by analyzing acoustic resonance waveforms.",
        "source": "IEEE European Symposium on Security and Privacy",
        "keywords": ["acoustic attack", "keyboard snooping", "passwords", "ai audio", "did you know"]
    },
    {
        "vector": "cybersecurity_secrets",
        "title": "The Year 2038 Problem: When Unix Time Runs Out",
        "hook": "Did You Know 32-Bit Time Ends on January 19, 2038? ⏳",
        "headline": "The 32-Bit Unix Epoch Rollover Bug",
        "fact_summary": "At 03:14:07 UTC on Jan 19, 2038, 32-bit signed integers tracking seconds since 1970 will overflow into negative numbers, sending legacy systems back to December 13, 1901.",
        "source": "POSIX Standard & Unix Time Architecture",
        "keywords": ["y2038", "unix epoch", "integer overflow", "32 bit bug", "did you know"]
    }
]


def fetch_or_select_did_you_know_fact(topic: Optional[str] = None) -> Dict:
    """
    Selects or generates a high-attraction 'Did You Know' fact.
    - If topic is provided, formats it into a DYK fact structure.
    - If no topic, cycles through DID_YOU_KNOW_VECTORS from topic_tracker,
      filters out recently used topics from instagram_carousel_log.json,
      and records the selection.
    """
    if topic:
        clean_topic = topic.strip()
        hook = clean_topic if clean_topic.lower().startswith("did you know") else f"Did You Know: {clean_topic}?"
        if not hook.endswith("?") and not hook.endswith("!"):
            hook += "?"
        return {
            "mode": "did_you_know",
            "category": "🧠 DID YOU KNOW?",
            "title": clean_topic,
            "hook": hook,
            "headline": clean_topic,
            "fact_summary": clean_topic,
            "source": "Verified Tech Architecture & Science",
            "keywords": ["did you know", "tech facts", "engineering"] + [w.lower() for w in clean_topic.split() if len(w) > 3]
        }

    # Auto-selection from topic tracker vectors
    try:
        from topic_tracker import get_did_you_know_sub_vector
        current_vector = get_did_you_know_sub_vector()
    except Exception:
        current_vector = "ai_secrets"

    print(f"🧠 Selecting 'Did You Know' fact for active vector: '{current_vector}'...")

    # Load tracker history to prevent duplicates
    used_titles = set()
    try:
        from ai_news_carousel import load_carousel_tracker, record_carousel_topic
        tracker = load_carousel_tracker()
        for t in tracker.get("used_titles", []):
            used_titles.add(str(t).lower())
        for h in tracker.get("history", []):
            if isinstance(h, dict) and h.get("title"):
                used_titles.add(h["title"].lower())
    except Exception as e:
        print(f"⚠️ Carousel tracker note: {e}")
        record_carousel_topic = None

    # Filter seed facts
    vector_candidates = [f for f in DID_YOU_KNOW_SEED_FACTS if f.get("vector") == current_vector]
    if not vector_candidates:
        vector_candidates = DID_YOU_KNOW_SEED_FACTS

    unseen_candidates = []
    for cand in vector_candidates:
        cand_title = cand["title"].lower()
        is_dup = False
        if cand_title in used_titles:
            is_dup = True
        elif RAPIDFUZZ_AVAILABLE:
            for used in used_titles:
                if fuzz.token_set_ratio(cand_title, used) > 70:
                    is_dup = True
                    break
        if not is_dup:
            unseen_candidates.append(cand)

    # Pick candidate
    selected = None
    if unseen_candidates:
        selected = random.choice(unseen_candidates)
    else:
        print("⚠️ All seed facts for current vector have been used recently; picking from full seed catalog...")
        all_unseen = [f for f in DID_YOU_KNOW_SEED_FACTS if f["title"].lower() not in used_titles]
        if all_unseen:
            selected = random.choice(all_unseen)
        else:
            selected = random.choice(vector_candidates)

    selected = dict(selected)
    selected["mode"] = "did_you_know"
    selected["category"] = "🧠 DID YOU KNOW?"

    # Record selection in carousel tracker
    if record_carousel_topic:
        try:
            record_carousel_topic(
                title=selected["title"],
                url="",
                keywords=selected.get("keywords", ["did you know"]),
                source=selected.get("source", "Did You Know By VJ")
            )
        except Exception as e:
            print(f"⚠️ Note recording carousel topic: {e}")

    print(f"🎯 Selected 'Did You Know' fact: '{selected['title']}' ({selected.get('vector', 'general')})")
    return selected


def clean_bubble_text(text: str, max_words: int = 18) -> str:
    """Ensure bubble text is concise (18 words or fewer) without overflow."""
    text = text.strip().strip('"').strip("'")
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;:-") + "..."
    return text


def build_dialogue_prompt(mode: str, topic: Optional[str] = None, story: Optional[Dict] = None) -> str:
    """Build the prompt for Gemini / OpenRouter dialogue script generation."""
    current_date = datetime.now().strftime("%d %B %Y")
    
    # ── Did You Know Mode (High-Attraction Tech & Science Facts) ───────────
    if mode in ["did_you_know", "dyk"]:
        fact_title = (story.get("title") if story else None) or topic or "99% of Internet is Underwater"
        fact_hook = (story.get("hook") if story else None) or f"Did You Know This About {fact_title}?"
        fact_summary = (story.get("fact_summary") if story else None) or (story.get("description") if story else None) or fact_title
        fact_source = (story.get("source") if story else None) or "Tech Architecture & Science"
        
        prompt = f"""You are the viral tech writer and visual director for 'Did You Know By VJ' (@vijayakumarj_ai).
Create a high-attraction, scroll-stopping Instagram dialogue carousel (exactly 6 to 7 slides) between:
1. "byte" (a curious, smart robot mascot who represents the fascinated audience. Expresses shock, disbelief, and asks the burning questions)
2. "vj" (the human tech creator and host of 'Did You Know By VJ', wearing a blue hoodie with 'TECH' logo. Explains the mind-blowing reality with calm expertise, exact numbers, and vivid analogies)

MIND-BLOWING FACT TO COVER:
- Core Fact: {fact_title}
- Hook Idea: {fact_hook}
- Verified Details: {fact_summary}
- Source: {fact_source}

CRITICAL RULES FOR MAXIMUM VIEWER ATTRACTION:
1. MODE: "did_you_know"
2. CATEGORY: "🧠 DID YOU KNOW?"
3. SLIDE 1 HOOK: Must start with a scroll-stopping question: "Did you know that...?" or a counter-intuitive paradox. Max 14 words.
4. SPEAKERS ALTERNATE: Slide 1 byte, Slide 2 vj, Slide 3 byte, Slide 4 vj, Slide 5 byte, Slide 6 vj (Slide 7 takeaway).
5. SPEECH BUBBLE LENGTH: STRICTLY 18 WORDS OR FEWER PER BUBBLE. Short, punchy, conversational, mind-blowing!
6. EMOTIONAL ARC:
   - Byte: "shocked" or "curious" on slide 1 ("Wait, did you know that...?").
   - VJ: "excited" or "thinking" on slide 2 revealing the scale & numbers.
   - Byte: "curious" or "thinking" on slide 3 asking the technical question.
   - VJ: "smug" or "excited" on slide 4 explaining the engineering mechanism.
   - Byte: "shocked" on slide 5 asking the crazy consequence or edge case.
   - VJ: "thinking" or "smug" on slide 6 delivering the punchline.
7. FINAL SLIDE: Mark "is_takeaway": true. Provide a punchy summary in "takeaway" field.
8. CAPTION: Engaging Instagram caption with Did You Know format, 3 bullet points, an engagement question ("Did you already know this? Drop a 🤯 below!"), and viral hashtags.

Return ONLY valid JSON matching this schema with NO markdown fences, NO preamble:
{{
  "mode": "did_you_know",
  "category": "🧠 DID YOU KNOW?",
  "hook": "Did You Know 99% of the Internet is Underwater? 🌊",
  "headline": "{fact_title}",
  "source": "{fact_source}",
  "slides": [
    {{"speaker": "byte", "emotion": "shocked", "bubble": "Wait, did you know that 99% of the internet is underwater?!"}},
    {{"speaker": "vj", "emotion": "excited", "bubble": "Yes! Over 1.4 million kilometers of fiber optic cables sit on the ocean floor."}},
    {{"speaker": "byte", "emotion": "curious", "bubble": "What stops sharks or anchors from destroying them?"}},
    {{"speaker": "vj", "emotion": "smug", "bubble": "Near shore they have heavy steel armor, deep down they are barely garden-hose thick!"}},
    {{"speaker": "byte", "emotion": "shocked", "bubble": "What happens if an anchor snags one?"}},
    {{"speaker": "vj", "emotion": "thinking", "bubble": "Entire countries can go offline until specialized repair ships arrive."}},
    {{"speaker": "byte", "emotion": "excited", "bubble": "Follow @vijayakumarj_ai for daily mind-blowing tech facts!", "is_takeaway": true}}
  ],
  "takeaway": "99% of international data relies on physical seafloor cables, not satellites. The cloud is literally on the ocean floor!",
  "caption": "🧠 DID YOU KNOW? 🤯\\n\\n99% of the internet is not in the sky... it is sitting on the ocean floor!\\n\\nHere is the mind-blowing reality:\\n🔹 Over 500 undersea fiber optic cables carry global data.\\n🔹 They transmit data at 99.7% the speed of light.\\n🔹 Deep-sea cables are only as thick as a garden hose, but carry trillions of dollars daily!\\n\\n💬 Did you already know this, or did this blow your mind? Drop a 🤯 in the comments!\\n\\nFollow @vijayakumarj_ai for daily visual tech breakdowns & facts!\\n#DidYouKnow #TechFacts #MindBlowingFacts #Engineering #ComputerScience"
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
1. "byte" (a curious, smart robot mascot)
2. "vj" (the human tech creator and host of 'Did You Know By VJ', wearing a blue hoodie, explaining complex tech simply and clearly)

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
4. SPEAKERS ALTERNATE: Alternate between "byte" and "vj" on every slide (e.g., slide 1 byte, slide 2 vj, slide 3 byte, etc.).
5. CONCISE BUBBLES: Each speech bubble MUST BE 18 WORDS OR FEWER. Short, punchy, conversational, engaging!
6. EMOTIONS: Each slide must have a valid emotion: ["curious", "thinking", "excited", "shocked", "neutral", "smug"].
7. LAST SLIDE: The final slide is a takeaway plus a follow CTA. Provide a "takeaway" field and note the source: "{source}, {date}".

Return ONLY valid JSON matching this schema with NO markdown fences, NO preamble:
{{
  "mode": "news",
  "hook": "Punchy hook question or breaking statement (max 8 words)",
  "headline": "{title}",
  "source": "{source}",
  "date": "{date}",
  "slides": [
    {{"speaker": "byte", "emotion": "shocked", "bubble": "Did OpenAI really just drop GPT-5 preview?"}},
    {{"speaker": "vj", "emotion": "excited", "bubble": "Yes! It introduces native autonomous tool orchestration."}},
    {{"speaker": "byte", "emotion": "curious", "bubble": "How does that help everyday engineers?"}},
    {{"speaker": "vj", "emotion": "thinking", "bubble": "No more brittle agent loops. It handles planning internally."}},
    {{"speaker": "byte", "emotion": "smug", "bubble": "My debugging sessions just got 10x faster."}},
    {{"speaker": "vj", "emotion": "excited", "bubble": "Follow @vijayakumarj_ai for daily updates!", "is_takeaway": true}}
  ],
  "takeaway": "Autonomous tool calling cuts agent boilerplate and boosts pipeline reliability.",
  "caption": "Breaking AI Update: {title}\\n\\nHere is what developers need to know...\\n\\nSource: {source} ({date})\\n\\nFollow @vijayakumarj_ai for daily AI breakdowns!\\n#AI #TechNews #DevCommunity"
}}
"""
        return prompt

    # Concept Mode (Educational)
    concept_topic = topic or "Why does ChatGPT forget you?"
    prompt = f"""You are a world-class tech educator creating an engaging, easy-to-understand Instagram educational carousel (6-7 slides) between two mascot characters:
1. "byte" (a curious, smart robot mascot who asks great questions)
2. "vj" (the human tech creator and host of 'Did You Know By VJ', wearing a blue hoodie, who explains complex tech simply)

TOPIC TO EXPLAIN: "{concept_topic}"

CRITICAL RULES:
1. MODE: "concept"
2. SLIDE COUNT: Exactly 6 to 7 slides.
3. SPEAKERS ALTERNATE: Alternate between "byte" and "vj" on every slide.
4. PUNCHY BUBBLES: Every single speech bubble MUST BE 18 WORDS OR FEWER. No exceptions.
5. EMOTIONS: Valid emotions for each slide: ["curious", "thinking", "excited", "shocked", "neutral", "smug"].
6. HOOK: Must start with a magnetic hook question that stops the user's scroll.
7. LAST SLIDE: Must be a clear takeaway summary plus a follow CTA.

Return ONLY valid JSON with NO markdown formatting:
{{
  "mode": "concept",
  "hook": "Why does ChatGPT forget you?",
  "headline": "{concept_topic}",
  "slides": [
    {{"speaker": "byte", "emotion": "curious", "bubble": "Why does ChatGPT forget what I said earlier?"}},
    {{"speaker": "vj", "emotion": "thinking", "bubble": "Think of it as the AI's short-term memory: the context window."}},
    {{"speaker": "byte", "emotion": "curious", "bubble": "What happens when that window fills up?"}},
    {{"speaker": "vj", "emotion": "shocked", "bubble": "Older messages drop off, so it literally cannot see them anymore!"}},
    {{"speaker": "byte", "emotion": "thinking", "bubble": "So prompt summaries prevent memory loss?"}},
    {{"speaker": "vj", "emotion": "smug", "bubble": "Exactly! Summarize older context or use vector memory."}},
    {{"speaker": "byte", "emotion": "excited", "bubble": "Follow @vijayakumarj_ai for daily AI breakdowns!", "is_takeaway": true}}
  ],
  "takeaway": "LLMs have finite context windows. To avoid memory drop, summarize long chats and prune prompts!",
  "caption": "Why does ChatGPT forget you?\\n\\nEver noticed your long chat losing its train of thought? Here is how context windows actually work...\\n\\nFollow @vijayakumarj_ai for daily AI engineering breakdowns!\\n#AI #MachineLearning #ChatGPT #TechTips"
}}
"""
    return prompt


def get_curated_fallback_dialogue(mode: str, topic: Optional[str] = None, story: Optional[Dict] = None) -> Dict:
    """High-quality curated fallback dialogue if all LLM endpoints fail."""
    current_date = datetime.now().strftime("%d %b %Y")
    
    if mode in ["did_you_know", "dyk"]:
        t = (story.get("title") if story else None) or topic or "99% of the Internet is Underwater"
        hook = (story.get("hook") if story else None) or f"Did You Know: {t[:40]}? 🌊"
        source = (story.get("source") if story else None) or "Subsea Cable Registry"
        return {
            "mode": "did_you_know",
            "category": "🧠 DID YOU KNOW?",
            "hook": hook,
            "headline": t,
            "source": source,
            "date": current_date,
            "slides": [
                {"speaker": "byte", "emotion": "shocked", "bubble": "Wait, did you know that 99% of the internet is underwater?!"},
                {"speaker": "vj", "emotion": "excited", "bubble": "Yes! Satellites carry almost nothing. 1.4M km of glass fiber sit on the seafloor."},
                {"speaker": "byte", "emotion": "curious", "bubble": "What stops sharks or heavy boat anchors from snapping them?"},
                {"speaker": "vj", "emotion": "smug", "bubble": "Near shore they have thick steel armor wire. Deep down they are garden-hose thin!"},
                {"speaker": "byte", "emotion": "shocked", "bubble": "What happens when an underwater cable actually gets cut?"},
                {"speaker": "vj", "emotion": "thinking", "bubble": "Traffic reroutes in milliseconds, and specialized grappling ships sail out to fish it up!"},
                {"speaker": "byte", "emotion": "excited", "bubble": "Follow @vijayakumarj_ai for daily mind-blowing tech facts!", "is_takeaway": True}
            ],
            "takeaway": "99% of global internet data travels through thin glass cables on the ocean floor, not satellites in space!",
            "caption": "🧠 DID YOU KNOW? 🤯\n\n99% of the internet is not in the sky... it is sitting on the ocean floor!\n\nHere is the mind-blowing reality:\n🔹 Over 500 undersea fiber optic cables carry global data.\n🔹 They transmit data at 99.7% the speed of light.\n🔹 Deep-sea cables are only as thick as a garden hose, but carry trillions of dollars daily!\n\n💬 Did you already know this, or did this blow your mind? Drop a 🤯 below!\n\nFollow @vijayakumarj_ai for daily visual tech breakdowns & facts!\n#DidYouKnow #TechFacts #MindBlowingFacts #Engineering #ComputerScience"
        }

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
                {"speaker": "byte", "emotion": "shocked", "bubble": f"Did you see the latest update from {source}?"},
                {"speaker": "vj", "emotion": "excited", "bubble": f"Yes! {title[:60]} just went live."},
                {"speaker": "byte", "emotion": "curious", "bubble": "What is the biggest capability improvement?"},
                {"speaker": "vj", "emotion": "thinking", "bubble": "Faster inference speeds and significantly higher reasoning accuracy."},
                {"speaker": "byte", "emotion": "smug", "bubble": "This changes how we build agent workflows."},
                {"speaker": "vj", "emotion": "excited", "bubble": "Follow @vijayakumarj_ai for daily verified AI news!", "is_takeaway": True}
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
            {"speaker": "byte", "emotion": "curious", "bubble": "Why does ChatGPT forget what I said 10 minutes ago?"},
            {"speaker": "vj", "emotion": "thinking", "bubble": "Think of it as the AI's short-term memory: context window."},
            {"speaker": "byte", "emotion": "curious", "bubble": "What happens when that window gets full?"},
            {"speaker": "vj", "emotion": "shocked", "bubble": "Older messages drop off completely so it cannot read them!"},
            {"speaker": "byte", "emotion": "thinking", "bubble": "So smart prompt compression keeps chats alive?"},
            {"speaker": "vj", "emotion": "smug", "bubble": "Exactly! Keep system prompts clean and summarize history."},
            {"speaker": "byte", "emotion": "excited", "bubble": "Follow @vijayakumarj_ai for daily AI breakdowns!", "is_takeaway": True}
        ],
        "takeaway": "LLMs rely on finite context windows. Trim excess prompts and summarize earlier dialogue to preserve memory.",
        "caption": "Ever wondered why your long AI chats lose track of context?\n\nHere is how context limits work and how you can fix them.\n\nFollow @vijayakumarj_ai for daily visual tech breakdowns!\n#AI #ChatGPT #TechTips #Developers"
    }


def parse_and_validate_dialogue(data: Any, mode: str, topic: Optional[str] = None, story: Optional[Dict] = None) -> Dict:
    """Ensure strict adherence to alternating speakers, word limits, and slide count."""
    if not isinstance(data, dict):
        return get_curated_fallback_dialogue(mode, topic, story)

    slides = data.get("slides", [])
    if not isinstance(slides, list) or len(slides) < 4:
        return get_curated_fallback_dialogue(mode, topic, story)

    # Ensure 6–7 slides
    if len(slides) > 7:
        slides = slides[:7]
    elif len(slides) < 6:
        # If 4 or 5 slides, ensure we add conclusion slide
        pass

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
        
        slide_entry = {
            "speaker": speaker,
            "emotion": emotion,
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
    data["hook"] = data.get("hook") or data.get("headline") or "AI Insights"
    data["headline"] = data.get("headline") or data["hook"]
    data["takeaway"] = data.get("takeaway") or "Understand core tech systems to build better workflows."
    
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
    mode: str = "auto"
) -> Dict:
    """Generate the mascot dialogue JSON script using Gemini / OpenRouter."""
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

    prompt = build_dialogue_prompt(mode=mode, topic=topic, story=story)
    dialogue_data = None

    # Priority 1: OpenRouter models if key is present
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "") or OPENROUTER_API_KEY
    if openrouter_key:
        try:
            print("🤖 Generating cartoon dialogue script with OpenRouter...")
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
                        "temperature": 0.4,
                    },
                    timeout=30,
                )
                if res.status_code == 200:
                    raw = res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
                    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
                    dialogue_data = json.loads(raw)
                    print(f"✅ OpenRouter ({m}) generated {len(dialogue_data.get('slides', []))} dialogue slides")
                    break
        except Exception as e:
            print(f"⚠️ OpenRouter generation note: {e}")

    # Priority 2: Google GenAI (Gemini)
    if not dialogue_data and GEMINI_AVAILABLE and GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            models = ["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.5-pro"]
            for m in models:
                try:
                    print(f"🤖 Generating cartoon dialogue with Gemini ({m})...")
                    resp = client.models.generate_content(
                        model=m,
                        contents=prompt,
                    )
                    raw = resp.text.strip()
                    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
                    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict) and "slides" in parsed and len(parsed["slides"]) >= 4:
                        dialogue_data = parsed
                        print(f"✅ Gemini ({m}) generated {len(dialogue_data.get('slides', []))} dialogue slides")
                        break
                except Exception as model_err:
                    print(f"⚠️ Gemini model {m} note: {model_err}")
                    continue
        except Exception as e:
            print(f"⚠️ Gemini client setup note: {e}")

    # Validate and fallback if needed
    result = parse_and_validate_dialogue(dialogue_data, mode=mode, topic=topic, story=story)
    return result


def render_cartoon_dialogue_carousel(
    dialogue: Dict,
    output_dir: Path,
    canvas_width: int = 1080,
    canvas_height: int = 1350,
    prefix: str = "carousel_",
    bg_images: Optional[List[str]] = None,
) -> List[Path]:
    """Render the cartoon dialogue slides using cartoon_dialogue.html.j2 & Playwright."""
    slides = dialogue.get("slides", [])
    if not slides:
        raise ValueError("No slides to render in cartoon dialogue")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_slides = len(slides)
    
    # Jinja2 setup
    env = Environment(
        loader=FileSystemLoader([str(TEMPLATE_DIR), str(TEMPLATE_DIR / "layouts")]),
        autoescape=True
    )
    template = env.get_template("cartoon_dialogue.html.j2")
    
    safe_title = re.sub(r'[\s\-]+', '_', re.sub(r'[^\w\s-]', '', dialogue.get("hook", "dialogue"))).strip('_')[:30].lower()
    slide_html_items = []
    
    # Mascot image paths for final celebratory slide
    byte_excited_path = get_character_image_path("byte", "excited") or get_character_image_path("byte", "neutral")
    vj_excited_path = get_character_image_path("vj", "excited") or get_character_image_path("vj", "neutral")
    asha_excited_path = vj_excited_path or get_character_image_path("asha", "excited")
    
    for i, slide in enumerate(slides):
        slide_num = i + 1
        speaker = slide.get("speaker", "byte")
        emotion = slide.get("emotion", "neutral")
        is_takeaway = (slide_num == total_slides) or slide.get("is_takeaway", False)
        
        char_img = get_character_image_path(speaker, emotion)
        char_img_uri = f"file://{char_img.resolve()}" if char_img else ""
        
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

        context = {
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "slide_current": slide_num,
            "slide_total": total_slides,
            "mode": derived_mode,
            "category": category_label,
            "hook": dialogue.get("hook", ""),
            "headline": dialogue.get("headline", ""),
            "speaker": speaker,
            "emotion": emotion,
            "slide_data": slide,
            "is_takeaway": is_takeaway,
            "takeaway": dialogue.get("takeaway", ""),
            "source": dialogue.get("source", ""),
            "date": dialogue.get("date", ""),
            "brand": {"handle": "@vijayakumarj_ai", "name": "Vijayakumar J"},
            "character_img_path": char_img_uri,
            "character_byte_path": f"file://{byte_excited_path.resolve()}" if byte_excited_path else "",
            "character_vj_path": f"file://{vj_excited_path.resolve()}" if vj_excited_path else "",
            "character_asha_path": f"file://{asha_excited_path.resolve()}" if asha_excited_path else "",
            "background_image_url": bg_uri,
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
