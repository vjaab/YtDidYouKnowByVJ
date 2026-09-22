#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
telegram_approval_handler.py — Handle Telegram approval workflow for auto-posting.
Supports both single images and carousels.
"""

import os
import sys
import json
import time
import re
import argparse
import requests
from pathlib import Path
from datetime import datetime

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else ""

# State file for tracking approval
STATE_FILE = Path(__file__).parent / ".telegram_approval_state.json"
TRACKER_FILE = Path(__file__).parent / "news_log.json"

def record_topic_in_tracker(topic: str, source_url: str = "", keywords: list = None, subcategory: str = "AI News", companies: list = None):
    """Record a successfully posted topic in the news_log.json tracker."""
    try:
        if TRACKER_FILE.exists():
            with open(TRACKER_FILE, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
        else:
            tracker = {
                "used_titles": [],
                "used_keywords": [],
                "used_companies": {},
                "used_subcategories": {},
                "last_7_days_stories": [],
                "last_3_days_subcategories": [],
                "last_3_days_companies": [],
                "total_uploaded": 0,
                "last_upload": None,
                "history": []
            }
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Add to used_titles
        tracker.setdefault("used_titles", []).append(topic)
        
        # Add keywords
        if keywords:
            tracker.setdefault("used_keywords", []).extend(keywords)
            tracker["used_keywords"] = list(set(tracker["used_keywords"]))
        
        # Add companies
        if companies:
            tracker.setdefault("used_companies", {})
            for comp in companies:
                if isinstance(comp, dict):
                    comp_name = comp.get("name")
                else:
                    comp_name = comp
                if comp_name:
                    tracker["used_companies"][comp_name] = tracker["used_companies"].get(comp_name, 0) + 1
        
        # Add subcategory
        tracker.setdefault("used_subcategories", {})
        tracker["used_subcategories"][subcategory] = tracker["used_subcategories"].get(subcategory, 0) + 1
        
        # Update rolling windows
        tracker.setdefault("last_7_days_stories", []).append(topic)
        if len(tracker["last_7_days_stories"]) > 7:
            tracker["last_7_days_stories"].pop(0)
        
        tracker.setdefault("last_3_days_subcategories", []).append(subcategory)
        if len(tracker["last_3_days_subcategories"]) > 3:
            tracker["last_3_days_subcategories"].pop(0)
        
        if companies:
            for comp in companies:
                comp_name = comp.get("name") if isinstance(comp, dict) else comp
                if comp_name:
                    tracker.setdefault("last_3_days_companies", []).append(comp_name)
            if len(tracker["last_3_days_companies"]) > 5:
                tracker["last_3_days_companies"] = tracker["last_3_days_companies"][-5:]
        
        tracker["total_uploaded"] = tracker.get("total_uploaded", 0) + 1
        tracker["last_upload"] = today
        
        # Add to history
        history_entry = {
            "date": today,
            "title": topic,
            "news_headline": topic,
            "sub_category": subcategory,
            "companies": companies or [],
            "keywords": keywords or [],
            "breaking_news_level": "normal",
            "voice_used": "system",
            "youtube_url": "",
            "facebook_post_id": None,
            "news_source_url": source_url,
            "target_country": "US",
            "avatar_used": None,
            "topic_type": "instagram_carousel"
        }
        tracker.setdefault("history", []).append(history_entry)
        
        # Save
        with open(TRACKER_FILE, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=4)
        
        print(f"✅ Recorded topic in tracker: {topic}")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to record topic in tracker: {e}")
        return False

def send_approval_request(topic: str, ig_images: list, fb_images: list, caption_file: str, poll_file: str = "", is_carousel: bool = False) -> int:
    """Send images to Telegram with approval buttons."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Telegram not configured")
    
    # Read caption
    caption = ""
    if Path(caption_file).exists():
        with open(caption_file) as f:
            caption = f.read()
    
    # Send Instagram images
    ig_msg_ids = []
    if is_carousel and len(ig_images) > 1:
        # Send as media group (album) for carousel
        media = []
        files = {}
        for i, ig_img in enumerate(ig_images):
            files[f"photo{i}"] = open(ig_img, "rb")
            media.append({
                "type": "photo",
                "media": f"attach://photo{i}",
                "caption": f"📸 <b>Instagram Carousel (4:5)</b> — Slide {i+1}/{len(ig_images)}\n\n{caption}" if i == 0 else "",
                "parse_mode": "HTML",
            })
        
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "media": json.dumps(media),
        }
        
        resp = requests.post(f"{TELEGRAM_BASE_URL}/sendMediaGroup", data=data, files=files, timeout=60)
        for f in files.values():
            f.close()
        resp.raise_for_status()
        ig_msg_ids = [msg["message_id"] for msg in resp.json()["result"]]
    else:
        # Send individual images
        for ig_img in ig_images:
            with open(ig_img, "rb") as f:
                files = {"photo": f}
                data = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": f"📸 <b>Instagram (4:5)</b>\n\n{caption}",
                    "parse_mode": "HTML",
                }
                resp = requests.post(f"{TELEGRAM_BASE_URL}/sendPhoto", data=data, files=files, timeout=30)
                resp.raise_for_status()
                ig_msg_ids.append(resp.json()["result"]["message_id"])
    
    # Send Facebook image
    fb_msg_ids = []
    for fb_img in fb_images:
        with open(fb_img, "rb") as f:
            files = {"photo": f}
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": f"📘 <b>Facebook (9:16)</b>\n\n{caption}",
                "parse_mode": "HTML",
            }
            resp = requests.post(f"{TELEGRAM_BASE_URL}/sendPhoto", data=data, files=files, timeout=30)
            resp.raise_for_status()
            fb_msg_ids.append(resp.json()["result"]["message_id"])
    
    # Send approval message with inline keyboard
    carousel_text = "Carousel" if is_carousel else "Image"
    keyboard = {
        "inline_keyboard": [
            [
                {"text": f"✅ Approve & Post {carousel_text} (IG+FB)", "callback_data": f"approve_all:{topic}"},
                {"text": f"✅ Approve Instagram Only", "callback_data": f"approve_ig:{topic}"},
            ],
            [
                {"text": f"✅ Approve Facebook Only", "callback_data": f"approve_fb:{topic}"},
            ],
            [
                {"text": "❌ Reject", "callback_data": f"reject:{topic}"},
                {"text": "🔄 Regenerate", "callback_data": f"regenerate:{topic}"},
            ],
        ]
    }
    
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"🤖 <b>Approval Required</b>\n\nTopic: <b>{topic}</b>\nType: {carousel_text} ({len(ig_images)} IG slides)\n\nReview the images above and choose an action:",
        "parse_mode": "HTML",
        "reply_markup": json.dumps(keyboard),
    }
    
    resp = requests.post(f"{TELEGRAM_BASE_URL}/sendMessage", json=data, timeout=30)
    resp.raise_for_status()
    approval_msg_id = resp.json()["result"]["message_id"]
    
    # Save state
    state = {
        "topic": topic,
        "ig_images": ig_images,
        "fb_images": fb_images,
        "caption_file": caption_file,
        "poll_file": poll_file,
        "approval_msg_id": approval_msg_id,
        "ig_msg_ids": ig_msg_ids,
        "fb_msg_ids": fb_msg_ids,
        "is_carousel": is_carousel,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
    }
    
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
    
    return approval_msg_id

def wait_for_approval(timeout: int = 3600) -> dict:
    """Wait for Telegram callback (polling for approval)."""
    print(f"⏳ Waiting for Telegram approval (timeout: {timeout}s)...")
    
    start_time = time.time()
    last_update_id = 0
    
    while time.time() - start_time < timeout:
        try:
            # Poll for updates
            params = {"offset": last_update_id + 1, "timeout": 30}
            resp = requests.get(f"{TELEGRAM_BASE_URL}/getUpdates", params=params, timeout=35)
            resp.raise_for_status()
            updates = resp.json().get("result", [])
            
            for update in updates:
                last_update_id = update["update_id"]
                
                if "callback_query" in update:
                    callback = update["callback_query"]
                    data = callback["data"]
                    user_id = callback["from"]["id"]
                    
                    # Verify it's from the right chat
                    if str(callback["message"]["chat"]["id"]) != TELEGRAM_CHAT_ID:
                        continue
                    
                    # Parse action
                    action, topic = data.split(":", 1)
                    
                    # Answer callback query
                    requests.post(
                        f"{TELEGRAM_BASE_URL}/answerCallbackQuery",
                        json={"callback_query_id": callback["id"], "text": f"Action: {action}"},
                        timeout=10,
                    )
                    
                    # Load state
                    if STATE_FILE.exists():
                        with open(STATE_FILE) as f:
                            state = json.load(f)
                    else:
                        state = {}
                    
                    state["action"] = action
                    state["approved_by"] = user_id
                    state["decided_at"] = datetime.utcnow().isoformat()
                    
                    with open(STATE_FILE, "w") as f:
                        json.dump(state, f)
                    
                    # Edit approval message
                    action_text = {
                        "approve_all": "✅ Approved - Posting to Instagram & Facebook",
                        "approve_ig": "✅ Approved - Posting to Instagram only",
                        "approve_fb": "✅ Approved - Posting to Facebook only",
                        "reject": "❌ Rejected - Not posting",
                        "regenerate": "🔄 Regenerating...",
                    }.get(action, f"Action: {action}")
                    
                    requests.post(
                        f"{TELEGRAM_BASE_URL}/editMessageText",
                        json={
                            "chat_id": TELEGRAM_CHAT_ID,
                            "message_id": callback["message"]["message_id"],
                            "text": action_text,
                            "parse_mode": "HTML",
                        },
                        timeout=10,
                    )
                    
                    return state
            
            time.sleep(5)
            
        except Exception as e:
            print(f"Polling error: {e}")
            time.sleep(10)
    
    print("⏰ Approval timeout")
    return {"action": "timeout"}

def post_to_platforms(state: dict, platform: str = "both"):
    """Post to approved platforms based on state action."""
    action = state.get("action", "")
    topic = state.get("topic", "")
    ig_images = state.get("ig_images", [])
    fb_images = state.get("fb_images", [])
    caption_file = state.get("caption_file", "")
    is_carousel = state.get("is_carousel", False)
    
    # Read caption
    caption = ""
    if Path(caption_file).exists():
        with open(caption_file) as f:
            caption = f.read()
    
    results = {}
    
    # Determine which platforms to post to based on action and platform arg
    post_ig = (action in ["approve_all", "approve_ig"]) and platform in ["both", "instagram"]
    post_fb = (action in ["approve_all", "approve_fb"]) and platform in ["both", "facebook"]
    
    if post_ig and ig_images:
        print(f"📸 Posting to Instagram ({'Carousel' if is_carousel else 'Single Image'})...")
        sys.path.insert(0, str(Path(__file__).parent))
        from post_to_instagram import post_to_instagram
        
        try:
            # For carousel, join paths with comma
            ig_image_arg = ",".join(ig_images) if is_carousel else ig_images[0]
            post_id = post_to_instagram(ig_image_arg, caption)
            results["instagram"] = post_id
        except Exception as e:
            results["instagram"] = f"ERROR: {e}"
    
    if post_fb and fb_images:
        print("📘 Posting to Facebook...")
        from post_to_facebook import post_to_facebook
        try:
            post_id = post_to_facebook(fb_images[0], caption)
            results["facebook"] = post_id
        except Exception as e:
            results["facebook"] = f"ERROR: {e}"
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Telegram approval handler")
    parser.add_argument("--send-for-approval", action="store_true", help="Send images for approval")
    parser.add_argument("--wait-and-post", action="store_true", help="Wait for approval and post")
    parser.add_argument("--topic", required=True, help="Topic name")
    parser.add_argument("--ig-images", nargs="+", help="Instagram image paths")
    parser.add_argument("--fb-images", nargs="+", help="Facebook image paths")
    parser.add_argument("--caption-file", help="Caption file path")
    parser.add_argument("--poll-file", help="Poll file path")
    parser.add_argument("--hashtags", help="Hashtags string")
    parser.add_argument("--timeout", type=int, default=3600, help="Approval timeout (seconds)")
    parser.add_argument("--carousel", action="store_true", help="Images form a carousel")
    parser.add_argument("--platform", choices=["both", "instagram", "facebook"], default="both", help="Platform to post to")
    args = parser.parse_args()
    
    def set_gha_output(key: str, value: str):
        github_output = os.getenv("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"{key}={value}\n")
    
    if args.send_for_approval:
        # Parse image lists
        ig_images = args.ig_images or []
        fb_images = args.fb_images or []
        
        # If comma-separated strings
        if len(ig_images) == 1 and "," in ig_images[0]:
            ig_images = ig_images[0].split(",")
        if len(fb_images) == 1 and "," in fb_images[0]:
            fb_images = fb_images[0].split(",")
        
        send_approval_request(
            args.topic, ig_images, fb_images, 
            args.caption_file, args.poll_file or "",
            is_carousel=args.carousel
        )
        print("✅ Approval request sent to Telegram")
        
    elif args.wait_and_post:
        state = wait_for_approval(args.timeout)
        
        action = state.get("action", "timeout")
        set_gha_output("approval_action", action)
        
        if action == "timeout":
            print("⏰ Timeout - no approval received")
            requests.post(
                f"{TELEGRAM_BASE_URL}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": "⏰ Approval timeout - no action taken",
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            sys.exit(1)
        elif action == "reject":
            print("❌ Rejected by user")
            sys.exit(0)
        elif action == "regenerate":
            print("🔄 Regeneration requested")
            sys.exit(2)  # Special exit code for regeneration
        
        print(f"✅ Approved with action: {action}")
        results = post_to_platforms(state, args.platform)
        
        # Report results
        for platform, result in results.items():
            if "ERROR" in str(result):
                print(f"❌ {platform}: {result}")
            else:
                print(f"✅ {platform}: {result}")
        
        # Record topic in tracker if successfully posted to at least one platform
        success = any("ERROR" not in str(v) for v in results.values())
        if success:
            # Extract keywords from caption file
            keywords = []
            if Path(args.caption_file).exists():
                with open(args.caption_file, 'r') as f:
                    caption_text = f.read()
                    keywords = re.findall(r'#(\w+)', caption_text)
            
            # Extract source URL from state if available
            source_url = ""
            if "carousel_json" in state:
                try:
                    with open(state["carousel_json"], 'r') as f:
                        carousel = json.load(f)
                        source_url = carousel.get("source_url", "")
                except:
                    pass
            
            record_topic_in_tracker(
                topic=args.topic,
                source_url=source_url,
                keywords=keywords,
                subcategory="AI News",
                companies=[]
            )
        
        # Send completion message
        result_text = "\n".join([f"{'✅' if 'ERROR' not in str(v) else '❌'} {k}: {v}" for k, v in results.items()])
        requests.post(
            f"{TELEGRAM_BASE_URL}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": f"📊 <b>Posting Complete</b>\n\n{result_text}",
                "parse_mode": "HTML",
            },
            timeout=10,
        )
        
    else:
        print("Use --send-for-approval or --wait-and-post")
        sys.exit(1)

if __name__ == "__main__":
    main()