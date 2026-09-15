#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_news_carousel.py — AI News Carousel Content Generator
Generates structured JSON for 6-slide AI news carousels using LLM.
"""

import os
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

try:
    from trending_engine import fetch_all_trending_signals
    TRENDING_ENGINE_AVAILABLE = True
except ImportError:
    TRENDING_ENGINE_AVAILABLE = False

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

AI_CATEGORIES = [
    "OpenAI",
    "Google / Gemini",
    "Anthropic / Claude",
    "Meta AI",
    "Microsoft",
    "NVIDIA",
    "AWS",
    "Open-source AI",
    "AI agents",
    "AI developer tools",
]

SCORING_WEIGHTS = {
    "freshness": 0.25,
    "developer_value": 0.25,
    "impact": 0.20,
    "novelty": 0.15,
    "visual_potential": 0.10,
    "audience_fit": 0.05,
}

MIN_SCORE_THRESHOLD = 6.5


def score_story(story: Dict) -> float:
    """Score a story based on weighted criteria."""
    scores = {
        "freshness": story.get("_freshness_score", 5),
        "developer_value": story.get("_developer_value_score", 5),
        "impact": story.get("_impact_score", 5),
        "novelty": story.get("_novelty_score", 5),
        "visual_potential": story.get("_visual_potential_score", 5),
        "audience_fit": story.get("_audience_fit_score", 5),
    }
    return sum(scores[k] * SCORING_WEIGHTS[k] for k in scores)


def select_best_story(stories: List[Dict]) -> Optional[Dict]:
    """Select the best story based on scoring."""
    if not stories:
        return None
    
    scored = [(score_story(s), s) for s in stories]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    best_score, best_story = scored[0]
    if best_score >= MIN_SCORE_THRESHOLD:
        best_story["_score"] = best_score
        return best_story
    return None


def build_carousel_prompt(story: Dict) -> str:
    """Build the prompt for LLM to generate carousel content."""
    title = story.get("title", "")
    description = story.get("description", "")
    url = story.get("url", "")
    source = story.get("source", {}).get("name", "Unknown")
    category = story.get("_predicted_category", "AI News")
    
    return f"""You are an expert AI news curator creating Instagram carousels for @vijayakumarj_ai (daily AI updates for engineers/developers).

STORY:
Title: {title}
Description: {description}
Source: {source}
URL: {url}
Category: {category}

Generate a 6-slide carousel JSON with this EXACT structure:
{{
  "headline": "Short punchy headline (max 60 chars)",
  "summary": "2-3 sentence summary for caption",
  "source": "{source}",
  "source_url": "{url}",
  "date": "{datetime.now().strftime('%d %b %y')}",
  "slides": [
    {{
      "type": "hook",
      "title": "HOOK HEADLINE",
      "body": "Big statement: What just changed?",
      "visual_hint": "Description of visual element for this slide"
    }},
    {{
      "type": "what_happened",
      "title": "WHAT HAPPENED?",
      "body": ["Bullet 1", "Bullet 2", "Bullet 3"],
      "visual_hint": "Simple diagram or icon suggestion"
    }},
    {{
      "type": "whats_new",
      "title": "WHAT'S NEW?",
      "body": "Key technical change in plain English",
      "visual_hint": "Before/After comparison or architecture diagram"
    }},
    {{
      "type": "why_matters",
      "title": "WHY DOES IT MATTER?",
      "body": "Developer/business/user impact\nBefore: X → After: Y",
      "visual_hint": "Impact visualization or metric comparison"
    }},
    {{
      "type": "real_world_example",
      "title": "REAL-WORLD EXAMPLE",
      "body": "Code snippet / workflow / use case\n\"Here's how you can use this\"",
      "visual_hint": "Code block or workflow screenshot"
    }},
    {{
      "type": "takeaway_cta",
      "title": "KEY TAKEAWAYS",
      "body": ["Takeaway 1", "Takeaway 2", "Takeaway 3"],
      "visual_hint": "Save this post / Follow @vijayakumarj_ai"
    }}
  ]
}}

RULES:
- Keep text concise, scannable, engineer-friendly
- Slide 1: Hook with BIG headline, strong visual
- Slide 2: 2-3 short bullets, date/source
- Slide 3: Key technical change, simple diagram
- Slide 4: Impact with Before→After format
- Slide 5: Practical example (code/workflow)
- Slide 6: 3 key points + "Save this post" + "Follow @vijayakumarj_ai"
- NO fluff, NO marketing speak, NO emojis in body text
- Visual hints are for the renderer, not shown to user"""


def generate_carousel_json(story: Dict) -> Dict:
    """Generate carousel content using LLM."""
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return generate_fallback_carousel(story)
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = build_carousel_prompt(story)
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
        )
        
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        carousel = json.loads(text.strip())
        carousel["_story"] = story
        return carousel
        
    except Exception as e:
        print(f"⚠️ LLM generation failed: {e}, using fallback")
        return generate_fallback_carousel(story)


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate text at word boundary, adding ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.7:
        return truncated[:last_space] + '…'
    return truncated + '…'


def generate_fallback_carousel(story: Dict) -> Dict:
    """Generate fallback carousel when LLM unavailable."""
    title = story.get("title", "AI Update")
    description = story.get("description", "")
    source = story.get("source", {}).get("name", "Unknown")
    url = story.get("url", "")
    
    return {
        "headline": _truncate_at_word_boundary(title, 60),
        "summary": description[:300] if description else f"Latest update from {source}",
        "source": source,
        "source_url": url,
        "date": datetime.now().strftime('%d %b %y'),
        "slides": [
            {
                "type": "hook",
                "title": _truncate_at_word_boundary(title, 50),
                "body": "Major AI development just announced",
                "visual_hint": "Company logo + breaking news style"
            },
            {
                "type": "what_happened",
                "title": "WHAT HAPPENED?",
                "body": [
                    f"{source} announced a significant update",
                    "Key capabilities expanded for developers",
                    "Available now for testing"
                ],
                "visual_hint": "Bullet list with checkmarks"
            },
            {
                "type": "whats_new",
                "title": "WHAT'S NEW?",
                "body": description[:200] if description else "Technical improvements and new features",
                "visual_hint": "Before/After feature comparison"
            },
            {
                "type": "why_matters",
                "title": "WHY DOES IT MATTER?",
                "body": "Developer impact: Faster iteration, lower costs\nBefore: Manual setup → After: One-click deploy",
                "visual_hint": "Metric comparison chart"
            },
            {
                "type": "real_world_example",
                "title": "REAL-WORLD EXAMPLE",
                "body": "Try it now:\n1. Visit the platform\n2. Enable the new feature\n3. Test with your use case",
                "visual_hint": "Code snippet or workflow steps"
            },
            {
                "type": "takeaway_cta",
                "title": "KEY TAKEAWAYS",
                "body": [
                    "Significant capability improvement",
                    "Reduces development time",
                    "Available for immediate testing"
                ],
                "visual_hint": "Save icon + Follow @vijayakumarj_ai"
            }
        ],
        "_story": story
    }


def fetch_ai_news_stories() -> List[Dict]:
    """Fetch AI news stories from trending engine with fallback to curated samples."""
    if not TRENDING_ENGINE_AVAILABLE:
        return get_fallback_stories()
    
    try:
        print("🔍 Fetching AI trending signals...")
        signals = fetch_all_trending_signals(
            target_country="US",
            category="AI & Tech Tools",
            sources=["github_trending", "huggingface_trending", "arxiv_papers", "newsletter_ai", "hacker_news", "reddit_trending"]
        )
        
        stories = []
        for signal in signals:
            story = {
                "title": signal.get("title", "Untitled"),
                "description": signal.get("description", ""),
                "url": signal.get("url", ""),
                "source": signal.get("source", {"name": "Unknown"}),
                "type": signal.get("type", ""),
                "_engagement": signal.get("_engagement", {}),
                "_freshness_score": min(10, signal.get("_engagement", {}).get("views", 0) / 10000 + 3),
                "_developer_value_score": 7 if "developer" in signal.get("description", "").lower() or "api" in signal.get("description", "").lower() else 5,
                "_impact_score": 6,
                "_novelty_score": 6,
                "_visual_potential_score": 5,
                "_audience_fit_score": 7,
            }
            stories.append(story)
        
        if stories:
            print(f"✅ Fetched {len(stories)} AI news stories from trending engine")
            return stories
        
        print("⚠️ Trending engine returned no signals, using fallback stories")
        return get_fallback_stories()
        
    except Exception as e:
        print(f"⚠️ Failed to fetch trending signals: {e}, using fallback")
        return get_fallback_stories()


def get_fallback_stories() -> List[Dict]:
    """Provide curated fallback AI news stories when trending engine unavailable."""
    import random
    today = datetime.now()
    seed = int(today.strftime("%Y%m%d"))
    random.seed(seed)
    
    fallback_stories = [
        {
            "title": "OpenAI Releases GPT-4o with Real-time Voice and Vision",
            "description": "OpenAI launched GPT-4o, a multimodal model with real-time voice conversation, vision understanding, and improved coding capabilities. Available via API with 50% lower cost than GPT-4 Turbo.",
            "url": "https://openai.com/index/gpt-4o/",
            "source": {"name": "OpenAI"},
            "type": "newsletter_ai",
            "_engagement": {"views": 500000},
            "_freshness_score": 9,
            "_developer_value_score": 9,
            "_impact_score": 10,
            "_novelty_score": 9,
            "_visual_potential_score": 8,
            "_audience_fit_score": 10,
        },
        {
            "title": "Anthropic Launches Claude 3.5 Sonnet with Artifacts Feature",
            "description": "Claude 3.5 Sonnet outperforms GPT-4o on coding benchmarks. New Artifacts feature lets developers see, edit, and iterate on code, documents, and designs in a dedicated window alongside the chat.",
            "url": "https://www.anthropic.com/news/claude-3-5-sonnet",
            "source": {"name": "Anthropic"},
            "type": "newsletter_ai",
            "_engagement": {"views": 300000},
            "_freshness_score": 9,
            "_developer_value_score": 10,
            "_impact_score": 9,
            "_novelty_score": 8,
            "_visual_potential_score": 9,
            "_audience_fit_score": 10,
        },
        {
            "title": "Google Gemini 1.5 Pro Gets 2M Context Window",
            "description": "Google expanded Gemini 1.5 Pro's context window to 2 million tokens, enabling analysis of entire codebases, hours of video, or hundreds of documents in a single prompt. Available in AI Studio and Vertex AI.",
            "url": "https://blog.google/technology/ai/google-gemini-update-2m-context/",
            "source": {"name": "Google"},
            "type": "newsletter_ai",
            "_engagement": {"views": 200000},
            "_freshness_score": 8,
            "_developer_value_score": 9,
            "_impact_score": 9,
            "_novelty_score": 8,
            "_visual_potential_score": 7,
            "_audience_fit_score": 9,
        },
        {
            "title": "Meta Releases Llama 3.1 405B - Open Source Frontier Model",
            "description": "Meta open-sourced Llama 3.1 405B, the largest openly available foundation model. Matches GPT-4o on benchmarks. Includes 8B and 70B variants. Available for commercial use with permissive license.",
            "url": "https://ai.meta.com/blog/meta-llama-3-1/",
            "source": {"name": "Meta AI"},
            "type": "huggingface_trending",
            "_engagement": {"views": 400000},
            "_freshness_score": 9,
            "_developer_value_score": 10,
            "_impact_score": 10,
            "_novelty_score": 9,
            "_visual_potential_score": 8,
            "_audience_fit_score": 10,
        },
        {
            "title": "Cursor AI Editor Raises $60M at $400M Valuation",
            "description": "AI-powered code editor Cursor secured Series A funding. Features include natural language editing, multi-file changes, and deep codebase understanding. 40k+ developers using it daily.",
            "url": "https://cursor.sh/blog/series-a",
            "source": {"name": "Cursor"},
            "type": "newsletter_ai",
            "_engagement": {"views": 150000},
            "_freshness_score": 8,
            "_developer_value_score": 10,
            "_impact_score": 8,
            "_novelty_score": 7,
            "_visual_potential_score": 8,
            "_audience_fit_score": 10,
        },
        {
            "title": "LangGraph v0.1 Released - Production-Ready Agent Framework",
            "description": "LangChain released LangGraph v0.1 with stateful agents, human-in-the-loop, and streaming support. Enables building complex multi-agent workflows with cyclic graphs and persistence.",
            "url": "https://blog.langchain.dev/langgraph-v0-1/",
            "source": {"name": "LangChain"},
            "type": "github_trending",
            "_engagement": {"views": 100000},
            "_freshness_score": 8,
            "_developer_value_score": 9,
            "_impact_score": 8,
            "_novelty_score": 7,
            "_visual_potential_score": 7,
            "_audience_fit_score": 9,
        },
        {
            "title": "NVIDIA Nemotron 3 Ultra Beats GPT-4 on Coding Benchmarks",
            "description": "NVIDIA's new Nemotron 3 Ultra model achieves state-of-the-art results on HumanEval and MBPP coding benchmarks. Released with open weights for research and commercial use.",
            "url": "https://research.nvidia.com/labs/nemotron/",
            "source": {"name": "NVIDIA"},
            "type": "arxiv_papers",
            "_engagement": {"views": 80000},
            "_freshness_score": 7,
            "_developer_value_score": 9,
            "_impact_score": 8,
            "_novelty_score": 8,
            "_visual_potential_score": 6,
            "_audience_fit_score": 8,
        },
        {
            "title": "Vercel AI SDK 3.0 - Streaming, Tools, and Generative UI",
            "description": "Vercel released AI SDK 3.0 with React Server Components support, generative UI components, and improved streaming. Makes it trivial to build AI-native applications with Next.js.",
            "url": "https://vercel.com/blog/ai-sdk-3",
            "source": {"name": "Vercel"},
            "type": "github_trending",
            "_engagement": {"views": 120000},
            "_freshness_score": 8,
            "_developer_value_score": 9,
            "_impact_score": 8,
            "_novelty_score": 7,
            "_visual_potential_score": 8,
            "_audience_fit_score": 9,
        },
    ]
    
    # Score all fallback stories
    for story in fallback_stories:
        story["_score"] = (
            story["_freshness_score"] * 0.25 +
            story["_developer_value_score"] * 0.25 +
            story["_impact_score"] * 0.20 +
            story["_novelty_score"] * 0.15 +
            story["_visual_potential_score"] * 0.10 +
            story["_audience_fit_score"] * 0.05
        )
    
    # Sort by score and return top stories
    fallback_stories.sort(key=lambda x: x["_score"], reverse=True)
    selected = fallback_stories[:5]
    
    print(f"✅ Using {len(selected)} curated fallback AI news stories")
    return selected


def main():
    parser = argparse.ArgumentParser(description="Generate AI News Carousel")
    parser.add_argument("--topic", type=str, help="Specific topic/title to generate")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--output-dir", type=str, default="output/social_images", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    story = None
    
    if args.topic:
        stories = fetch_ai_news_stories()
        story = next((s for s in stories if args.topic.lower() in s.get("title", "").lower()), None)
        if not story:
            print(f"❌ Topic not found: {args.topic}")
            sys.exit(1)
    else:
        stories = fetch_ai_news_stories()
        story = select_best_story(stories)
        if not story:
            print("❌ No stories meet quality threshold")
            sys.exit(1)
    
    print(f"📰 Selected story: {story.get('title')}")
    print(f"   Score: {story.get('_score', 'N/A')}")
    print(f"   Source: {story.get('source', {}).get('name')}")
    
    carousel = generate_carousel_json(story)
    
    safe_title = "".join(c for c in story.get("title", "ai_news") if c.isalnum() or c in " -_").strip()[:50]
    safe_title = safe_title.replace(" ", "_")
    
    carousel_path = output_dir / f"carousel_{safe_title}.json"
    with open(carousel_path, "w") as f:
        json.dump(carousel, f, indent=2)
    
    print(f"✅ Carousel JSON saved: {carousel_path}")
    
    if args.dry_run:
        print(json.dumps(carousel, indent=2))
    
    return carousel_path


if __name__ == "__main__":
    main()