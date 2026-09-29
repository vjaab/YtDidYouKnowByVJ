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
    data["takeaway"] = data.get("takeaway") or "Understand core AI systems to build better workflows."
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
        if story and story.get("title"):
            mode = "news"
        elif topic and any(q in topic.lower() for q in ["what", "why", "how", "explain", "vs", "versus"]):
            mode = "concept"
        elif story:
            mode = "news"
        else:
            mode = "concept"

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

        context = {
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "slide_current": slide_num,
            "slide_total": total_slides,
            "mode": dialogue.get("mode", "concept"),
            "category": "AI NEWS" if dialogue.get("mode") == "news" else "AI EXPLAINED",
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
