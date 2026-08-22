#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
telegram_approval_handler.py — Handle Telegram approval workflow for auto-posting.
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime, timedelta

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else ""

# State file for tracking approval
STATE_FILE = Path(__file__).parent / ".telegram_approval_state.json"

def send_approval_request(topic: str, ig_images: list, fb_images: list, caption_file: str, poll_file: str = "") -> int:
    """Send images to Telegram with approval buttons."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Telegram not configured")
    
    # Read caption
    caption = ""
    if Path(caption_file).exists():
        with open(caption_file) as f:
            caption = f.read()
    
    # Send Instagram image
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
            ig_msg_id = resp.json()["result"]["message_id"]
    
    # Send Facebook image
    for fb_img in fb_images:
        with open(fb_img, "rb") as f:
            files = {"photo": f}
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": f"📘 <b>Facebook (1.91:1)</b>\n\n{caption}",
                "parse_mode": "HTML",
            }
            resp = requests.post(f"{TELEGRAM_BASE_URL}/sendPhoto", data=data, files=files, timeout=30)
            resp.raise_for_status()
            fb_msg_id = resp.json()["result"]["message_id"]
    
    # Send approval message with inline keyboard
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "✅ Approve & Post All", "callback_data": f"approve_all:{topic}"},
                {"text": "✅ Approve Instagram Only", "callback_data": f"approve_ig:{topic}"},
            ],
            [
                {"text": "✅ Approve Facebook Only", "callback_data": f"approve_fb:{topic}"},
                {"text": "✅ Approve + X/LinkedIn", "callback_data": f"approve_all_cross:{topic}"},
            ],
            [
                {"text": "❌ Reject", "callback_data": f"reject:{topic}"},
                {"text": "🔄 Regenerate", "callback_data": f"regenerate:{topic}"},
            ],
        ]
    }
    
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"🤖 <b>Approval Required</b>\n\nTopic: <b>{topic}</b>\n\nReview the images above and choose an action:",
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
        "ig_msg_id": ig_msg_id,
        "fb_msg_id": fb_msg_id,
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
                        "approve_all": "✅ Approved - Posting to Instagram, Facebook, X, LinkedIn",
                        "approve_ig": "✅ Approved - Posting to Instagram only",
                        "approve_fb": "✅ Approved - Posting to Facebook only",
                        "approve_all_cross": "✅ Approved - Posting to all platforms",
                        "reject": "❌ Rejected - Not posting",
                        "regenerate": "🔄 Regenerating images...",
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

def post_to_platforms(state: dict):
    """Post to approved platforms based on state action."""
    action = state.get("action", "")
    topic = state.get("topic", "")
    ig_images = state.get("ig_images", [])
    fb_images = state.get("fb_images", [])
    caption_file = state.get("caption_file", "")
    
    # Read caption
    caption = ""
    if Path(caption_file).exists():
        with open(caption_file) as f:
            caption = f.read()
    
    results = {}
    
    # Determine which platforms to post to
    post_ig = action in ["approve_all", "approve_ig", "approve_all_cross"]
    post_fb = action in ["approve_all", "approve_fb", "approve_all_cross"]
    post_x = action in ["approve_all_cross"]
    post_linkedin = action in ["approve_all_cross"]
    
    if post_ig and ig_images:
        print("📸 Posting to Instagram...")
        # Import and call post_to_instagram
        sys.path.insert(0, str(Path(__file__).parent))
        from post_to_instagram import post_to_instagram
        for img in ig_images:
            try:
                post_id = post_to_instagram(img, caption)
                results["instagram"] = post_id
            except Exception as e:
                results["instagram"] = f"ERROR: {e}"
    
    if post_fb and fb_images:
        print("📘 Posting to Facebook...")
        from post_to_facebook import post_to_facebook
        for img in fb_images:
            try:
                post_id = post_to_facebook(img, caption)
                results["facebook"] = post_id
            except Exception as e:
                results["facebook"] = f"ERROR: {e}"
    
    if post_x and ig_images:
        print("🐦 Posting to X...")
        from post_to_x import post_to_x
        for img in ig_images:
            try:
                post_id = post_to_x(img, caption)
                results["x"] = post_id
            except Exception as e:
                results["x"] = f"ERROR: {e}"
    
    if post_linkedin and fb_images:
        print("💼 Posting to LinkedIn...")
        from post_to_linkedin import post_to_linkedin
        for img in fb_images:
            try:
                post_id = post_to_linkedin(img, caption)
                results["linkedin"] = post_id
            except Exception as e:
                results["linkedin"] = f"ERROR: {e}"
    
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
            args.caption_file, args.poll_file or ""
        )
        print("✅ Approval request sent to Telegram")
        
    elif args.wait_and_post:
        state = wait_for_approval(args.timeout)
        
        action = state.get("action", "timeout")
        set_gha_output("approval_action", action)
        
        if action == "timeout":
            print("⏰ Timeout - no approval received")
            # Send timeout notification
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
        results = post_to_platforms(state)
        
        # Report results
        for platform, result in results.items():
            if "ERROR" in str(result):
                print(f"❌ {platform}: {result}")
            else:
                print(f"✅ {platform}: {result}")
        
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