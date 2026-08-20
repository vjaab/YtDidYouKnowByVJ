#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_image.py — Generate and post educational infographics to Instagram, Facebook, and Telegram.

Uses structured educational content pipeline with LLM-generated content contracts,
HTML/CSS templates, and Playwright rendering.
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

from config import BASE_DIR, OUTPUT_DIR, GEMINI_API_KEY

# Import new education pipeline
sys.path.insert(0, str(BASE_DIR))
from education_pipeline import EducationPipeline
from content.schemas import DifficultyLevel


# ── Telegram ────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else ""


# ── Default Educational Topics ─────────────────────────────────────────────────
DEFAULT_TOPICS = [
    "Python Decorators",
    "AWS Lambda",
    "Java Streams API",
    "RAG Pipeline Architecture",
    "Kubernetes Pods and Services",
    "Docker Multi-stage Builds",
    "Git Branching Strategies",
    "System Design: Load Balancers",
    "SQL Joins Explained",
    "CI/CD Pipeline Stages",
    "Python Async/Await",
    "AWS DynamoDB vs RDS",
    "Java CompletableFuture",
    "Vector Databases",
    "Kubernetes Ingress",
    "Docker Compose for Microservices",
    "Git Rebase vs Merge",
    "System Design: Caching Strategies",
    "SQL Indexing",
    "GitHub Actions Workflows",
]


def send_image_to_telegram(image_path: str, caption: str = "") -> bool:
    """Send an image to Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured — skipping")
        return False

    try:
        import requests
        with open(image_path, "rb") as f:
            files = {"photo": f}
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption[:1024],
                "parse_mode": "HTML",
            }
            resp = requests.post(f"{TELEGRAM_BASE_URL}/sendPhoto", data=data, files=files, timeout=30)
            resp.raise_for_status()
            print(f"📱 Telegram image sent successfully")
            return True
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")
        return False


def send_telegram_message(message: str, emoji: str = "ℹ️"):
    """Send a plain notification to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import requests
        requests.post(
            f"{TELEGRAM_BASE_URL}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": f"{emoji} {message}", "parse_mode": "HTML"},
            timeout=15
        )
    except Exception as e:
        print(f"Telegram notify failed: {e}")


def load_topics_from_file() -> list:
    """Load topics from topics.json if exists."""
    topics_file = Path(BASE_DIR) / "topics.json"
    if topics_file.exists():
        try:
            with open(topics_file) as f:
                data = json.load(f)
                return data.get("topics", []) if isinstance(data, dict) else data
        except Exception as e:
            print(f"⚠️ Failed to load topics.json: {e}")
    return []


def get_next_topic_round_robin() -> str:
    """Get the next topic in round-robin fashion."""
    all_topics = load_topics_from_file() or DEFAULT_TOPICS
    if not all_topics:
        return DEFAULT_TOPICS[0]
    
    # Track file for round-robin state
    state_file = Path(BASE_DIR) / ".topic_index.json"
    
    current_index = 0
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
                current_index = state.get("index", 0)
        except Exception:
            current_index = 0
    
    # Get current topic
    topic = all_topics[current_index % len(all_topics)]
    
    # Update index for next run
    next_index = (current_index + 1) % len(all_topics)
    try:
        with open(state_file, "w") as f:
            json.dump({"index": next_index}, f)
    except Exception as e:
        print(f"⚠️ Failed to save topic index: {e}")
    
    return topic


async def run_education_pipeline(dry_run=False, topic=None, topics=None, audience=None, difficulty=None):
    """Main pipeline using the new education content system."""
    print("=" * 60)
    print("📚 EDUCATIONAL CONTENT PIPELINE STARTED")
    print("=" * 60)

    if dry_run:
        print("🧪 DRY RUN MODE - No actual posting")

    # Get topics
    if topic:
        topic_list = [topic]
    elif topics:
        topic_list = topics
    else:
        # Round-robin: pick one topic per run
        topic_list = [get_next_topic_round_robin()]
    
    if audience is None:
        audience = ["students", "developers"]
    if difficulty is None:
        difficulty = DifficultyLevel.INTERMEDIATE

    print(f"\n📋 Topic to process: {topic_list[0]}")
    print(f"👥 Audience: {', '.join(audience)}")
    print(f"📊 Difficulty: {difficulty.value}")

    # Initialize pipeline
    pipeline = EducationPipeline(
        gemini_api_key=GEMINI_API_KEY,
        output_dir=str(OUTPUT_DIR),
    )

    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not set! Please add it to GitHub secrets.")
        return False, "GEMINI_API_KEY not configured"

    if dry_run:
        print("\n🤖 Generating content (dry run)...")
        contents = pipeline.generate_batch(topic_list, audience, difficulty)
        for content in contents:
            print(f"\n📝 {content.topic} ({content.category.value})")
            print(f"   Difficulty: {content.difficulty.value}")
            print(f"   Hook: {content.hook}")
            print(f"   Strategy: {[v.value for v in content.visual_strategy]}")
            if content.quiz:
                print(f"   Quiz: {content.quiz.question[:60]}...")
        return True, "DRY_RUN_SUCCESS"

    # Process all topics
    print("\n🚀 Processing topics...")
    results = await pipeline.process_batch(topic_list, audience, difficulty)

    # Send to Telegram
    print("\n📱 Sending to Telegram...")
    total_slides_sent = 0
    for result in results:
        if "error" in result:
            continue
        
        outputs = result.get("outputs", {})
        
        # Send Facebook poster
        fb_path = outputs.get("facebook")
        if fb_path and os.path.exists(fb_path):
            caption = f"📚 {result['topic']}\n\n{result['content'].get('takeaway', '')}\n\nFollow @Vijayakumarj_ai for more!"
            send_image_to_telegram(fb_path, caption)
            total_slides_sent += 1
        
        # Send Instagram carousel slides (first 3)
        ig_paths = outputs.get("instagram", [])
        for i, slide_path in enumerate(ig_paths[:3]):
            if os.path.exists(slide_path):
                slide_num = i + 1
                caption = f"📚 {result['topic']} - Slide {slide_num}/5\n\nSwipe for more!\n\nFollow @Vijayakumarj_ai"
                send_image_to_telegram(slide_path, caption)
                total_slides_sent += 1

    # Notify Telegram
    success_count = sum(1 for r in results if "error" not in r)
    send_telegram_message(
        f"✅ Education Pipeline Complete\n"
        f"Topics: {success_count}/{len(results)}\n"
        f"Slides sent to Telegram: {total_slides_sent}",
        emoji="🤖"
    )

    # Save results
    pipeline.save_results(results)

    print("\n" + "=" * 60)
    print("✅ EDUCATIONAL CONTENT PIPELINE COMPLETED")
    print("=" * 60)
    
    return True, "SUCCESS"


def main():
    parser = argparse.ArgumentParser(description="Generate and post educational infographics")
    parser.add_argument("--now", action="store_true", help="Run immediately")
    parser.add_argument("--dry-run", action="store_true", help="Preview without posting")
    parser.add_argument("--topic", type=str, help="Single topic to process")
    parser.add_argument("--topics", nargs="+", help="Multiple topics to process")
    parser.add_argument("--audience", nargs="+", default=["students", "developers"])
    parser.add_argument("--difficulty", choices=["beginner", "intermediate", "advanced"], default="intermediate")
    args = parser.parse_args()

    if not args.now and not args.dry_run:
        print("Usage: python post_image.py --now       # Run and post")
        print("       python post_image.py --dry-run   # Preview only")
        print("       python post_image.py --now --topic 'Python Decorators'")
        sys.exit(1)

    success, result = asyncio.run(run_education_pipeline(
        dry_run=args.dry_run,
        topic=args.topic,
        topics=args.topics,
        audience=args.audience,
        difficulty=DifficultyLevel(args.difficulty)
    ))
    
    if not success:
        print(f"Pipeline failed: {result}")
        sys.exit(1)
    else:
        print(f"Pipeline succeeded: {result}")


if __name__ == "__main__":
    main()