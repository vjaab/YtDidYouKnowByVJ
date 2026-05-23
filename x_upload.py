"""
x_upload.py — Robust Auto-Posting Module for X.com (Twitter).

Implements Twitter Media Upload v1.1 API (chunked video upload) + API v2 for Tweeting.
Supports credentials loading, async transcode status polling, and error boundaries.
"""

import os
import time
import requests
from requests_oauthlib import OAuth1
from config import X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET

def _check_credentials():
    """Verifies that all required X API credentials are configured."""
    keys = [X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET]
    return all(k and k.strip() for k in keys)

def upload_video_to_x(video_path, post_text):
    """
    Uploads a video to X.com using OAuth 1.0a User Context and posts it.
    
    Args:
        video_path (str): Absolute path to the .mp4 video file.
        post_text (str): Body text of the tweet.
        
    Returns:
        tuple: (bool success, str result_message_or_id)
    """
    if not _check_credentials():
        return False, "Skipped: X.com credentials not configured in .env"
        
    if not os.path.exists(video_path):
        return False, f"Error: Video file not found at {video_path}"

    print(f"📡 Initializing chunked video upload to X.com for: {video_path}")
    auth = OAuth1(X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET)
    upload_url = "https://upload.twitter.com/1.1/media/upload.json"
    
    # ── STEP 1: INIT UPLOAD ──
    file_size = os.path.getsize(video_path)
    init_data = {
        "command": "INIT",
        "media_type": "video/mp4",
        "total_bytes": file_size,
        "media_category": "tweet_video"
    }
    
    try:
        req_init = requests.post(upload_url, data=init_data, auth=auth, timeout=30)
        if req_init.status_code not in (200, 201, 202):
            return False, f"INIT failed (HTTP {req_init.status_code}): {req_init.text}"
            
        media_id = req_init.json().get("media_id_string")
        if not media_id:
            return False, f"INIT failed: No media_id returned. Response: {req_init.text}"
    except Exception as e:
        return False, f"INIT exception: {e}"

    print(f"✔ INIT complete. Media ID: {media_id}. Starting chunked APPEND...")

    # ── STEP 2: APPEND CHUNKS ──
    chunk_size = 4 * 1024 * 1024  # 4MB chunks
    segment_index = 0
    
    try:
        with open(video_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                print(f"   Uploading chunk #{segment_index} ({len(chunk)} bytes)...")
                append_data = {
                    "command": "APPEND",
                    "media_id": media_id,
                    "segment_index": segment_index
                }
                files = {"media": chunk}
                
                req_append = requests.post(upload_url, data=append_data, files=files, auth=auth, timeout=60)
                if req_append.status_code not in (200, 201, 202):
                    return False, f"APPEND chunk #{segment_index} failed: {req_append.text}"
                
                segment_index += 1
    except Exception as e:
        return False, f"APPEND exception: {e}"

    print("✔ APPEND complete. Finalizing media upload...")

    # ── STEP 3: FINALIZE ──
    finalize_data = {
        "command": "FINALIZE",
        "media_id": media_id
    }
    
    try:
        req_finalize = requests.post(upload_url, data=finalize_data, auth=auth, timeout=30)
        if req_finalize.status_code not in (200, 201, 202):
            return False, f"FINALIZE failed: {req_finalize.text}"
            
        finalize_json = req_finalize.json()
    except Exception as e:
        return False, f"FINALIZE exception: {e}"

    # ── STEP 4: STATUS POLLING (Asynchronous Transcode Check) ──
    print("⏳ Waiting for X.com backend to transcode and verify video processing...")
    processing_info = finalize_json.get("processing_info")
    
    if processing_info:
        state = processing_info.get("state")
        while state in ("pending", "in_progress"):
            check_after = processing_info.get("check_after_secs", 5)
            print(f"   Processing state: {state}. Sleeping for {check_after}s...")
            time.sleep(check_after)
            
            try:
                status_params = {
                    "command": "STATUS",
                    "media_id": media_id
                }
                req_status = requests.get(upload_url, params=status_params, auth=auth, timeout=20)
                if req_status.status_code == 200:
                    status_json = req_status.json()
                    processing_info = status_json.get("processing_info", {})
                    state = processing_info.get("state")
                else:
                    print(f"⚠️ STATUS check failed (HTTP {req_status.status_code}): {req_status.text}")
            except Exception as e:
                print(f"⚠️ STATUS check exception: {e}")
                time.sleep(5)
                
        if state == "failed":
            error_msg = processing_info.get("error", {}).get("message", "Unknown transcoding error")
            return False, f"Transcoding failed on X.com: {error_msg}"
            
    print("✔ Video transcode succeeded! Creating the Tweet...")

    # ── STEP 5: POST TWEET (API v2) ──
    tweet_url = "https://api.twitter.com/2/tweets"
    tweet_payload = {
        "text": post_text[:280],  # Standard X character limit safety bounds
        "media": {"media_ids": [media_id]}
    }
    
    try:
        response = requests.post(tweet_url, json=tweet_payload, auth=auth, timeout=30)
        if response.status_code == 201:
            tweet_id = response.json().get("data", {}).get("id")
            print(f"🎉 Success! Tweet posted to X.com. ID: {tweet_id}")
            return True, tweet_id
        else:
            return False, f"TWEET post failed (HTTP {response.status_code}): {response.text}"
    except Exception as e:
        return False, f"TWEET exception: {e}"

if __name__ == "__main__":
    # Test script run
    print("Testing credentials check...")
    if _check_credentials():
        print("✔ Credentials configured correctly!")
    else:
        print("⚠️ Credentials missing or empty in .env. Auto-posting will be skipped gracefully.")
