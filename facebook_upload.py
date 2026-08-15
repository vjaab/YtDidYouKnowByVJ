# -*- coding: utf-8 -*-
"""
facebook_upload.py — Facebook Page Reels/Video Upload Module via Meta Graph API.

Uploads videos to Facebook Page as Reels using the video_reels API with upload_phase workflow.
Also provides fallback to regular /videos endpoint.

Flow:
  1. Start upload session (POST /page_id/video_reels with upload_phase=start)
  2. Download video from public URL
  3. Upload binary to rupload.facebook.com with Offset/Content-Length/X-Entity-Length
  4. Finish upload with video_state=PUBLISHED
  5. Poll for processing status

Env vars expected:
    FB_PAGE_ID            - Facebook Page ID
    FB_PAGE_ACCESS_TOKEN  - Page access token with pages_manage_posts, pages_show_list, pages_read_engagement

Video hosting:
    Expects a public video URL (from GitHub Releases, or direct public URL)
"""

import os
import time
import json
import requests
from datetime import datetime, timedelta

from config import (
    BASE_DIR,
)

# ── Constants ────────────────────────────────────────────────────────────────
GRAPH_API_VERSION = "v25.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


def _check_facebook_credentials():
    """Verifies that Facebook Page credentials are configured."""
    fb_page_id = os.getenv("FB_PAGE_ID")
    fb_page_access_token = os.getenv("FB_PAGE_ACCESS_TOKEN")
    return all(k and k.strip() for k in [fb_page_id, fb_page_access_token])


# ── Facebook API Helpers ──────────────────────────────────────────────────────

def _parse_meta_error(response_text: str) -> dict:
    """Parse Meta Graph API error response into structured dict.
    Handles malformed JSON with unescaped quotes in error messages.
    """
    # First try normal parsing
    try:
        data = json.loads(response_text)
        error = data.get("error", {})
        return {
            "message": error.get("message", "Unknown error"),
            "type": error.get("type", "Unknown"),
            "code": error.get("code", 0),
            "error_subcode": error.get("error_subcode"),
            "fbtrace_id": error.get("fbtrace_id"),
            "raw": data
        }
    except json.JSONDecodeError:
        pass
    
    # Fallback: extract fields using regex (handles malformed JSON from Meta)
    try:
        import re
        
        # Extract error object content
        error_match = re.search(r'"error"\s*:\s*(\{.*?\})\s*[},\]]', response_text, re.DOTALL)
        if error_match:
            error_content = error_match.group(1)
        else:
            error_content = response_text
        
        # Extract individual fields
        code_match = re.search(r'"code"\s*:\s*(\d+)', error_content)
        message_match = re.search(r'"message"\s*:\s*"(.+?)(?:\"\s*,\s*"|\"\s*})', error_content)
        type_match = re.search(r'"type"\s*:\s*"([^"]*)"', error_content)
        fbtrace_match = re.search(r'"fbtrace_id"\s*:\s*"([^"]*)"', error_content)
        subcode_match = re.search(r'"error_subcode"\s*:\s*(\d+)', error_content)
        
        return {
            "message": message_match.group(1) if message_match else "Unknown error",
            "type": type_match.group(1) if type_match else "Unknown",
            "code": int(code_match.group(1)) if code_match else 0,
            "error_subcode": int(subcode_match.group(1)) if subcode_match else None,
            "fbtrace_id": fbtrace_match.group(1) if fbtrace_match else None,
            "raw": {"parse_method": "regex_fallback"}
        }
    except Exception as e:
        return {"message": response_text[:200], "type": "ParseError", "code": 0, "error_subcode": None, "fbtrace_id": None, "raw": {"parse_error": str(e)}}


def _is_retryable_error(error_code: int, error_subcode: int = None) -> bool:
    """Check if error is transient and worth retrying."""
    retryable_codes = {4, 17, 32, 80000, 80001, 80004, 80006}
    retryable_subcodes = {2446079, 2446071}
    return error_code in retryable_codes or error_subcode in retryable_subcodes


def _retry_with_backoff(func, max_attempts=3, base_delay=2, *args, **kwargs):
    """Execute function with exponential backoff retry for transient errors."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except requests.HTTPError as e:
            last_error = e
            # Use captured response_text if available, fallback to e.response.text
            response_text = getattr(e, 'response_text', None) or (e.response.text if e.response else "")
            error_data = _parse_meta_error(response_text)
            if not _is_retryable_error(error_data["code"], error_data["error_subcode"]):
                raise
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                print(f"⚠️ Retryable error (attempt {attempt}/{max_attempts}): {error_data['message']} (code {error_data['code']}). Waiting {delay}s...")
                time.sleep(delay)
            else:
                raise
    raise last_error


def _validate_fb_token(fb_page_id: str, fb_page_access_token: str) -> dict:
    """Validate Facebook Page token has required permissions."""
    try:
        # Try with permissions field first
        resp = requests.get(
            f"{GRAPH_API_BASE}/me",
            params={
                "fields": "id,name,permissions",
                "access_token": fb_page_access_token
            },
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            perms = set()
            permissions_data = data.get("permissions", {}).get("data", [])
            if permissions_data:
                perms = {p["permission"] for p in permissions_data}
            required = {"pages_show_list", "pages_read_engagement", "pages_manage_posts", "pages_manage_metadata"}
            missing = required - perms
            return {
                "valid": True,
                "page_id": data.get("id"),
                "page_name": data.get("name"),
                "missing_permissions": list(missing),
                "all_permissions": list(perms)
            }
        # If permissions field fails, try without it
        if "Tried accessing nonexisting field" in resp.text:
            resp = requests.get(
                f"{GRAPH_API_BASE}/me",
                params={
                    "fields": "id,name",
                    "access_token": fb_page_access_token
                },
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "valid": True,
                    "page_id": data.get("id"),
                    "page_name": data.get("name"),
                    "missing_permissions": ["permissions field not accessible"],
                    "all_permissions": []
                }
        return {"valid": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ── Main Upload Function ──────────────────────────────────────────────────────

def post_video_to_facebook_page(video_url: str, caption: str) -> tuple:
    """
    Posts a video to Facebook Page as a Reel using the video_reels API.
    
    Args:
        video_url: Public URL to the video file
        caption: Caption/description for the video
        
    Returns:
        tuple: (video_id, error_message) - video_id is None on failure
    """
    fb_page_id = os.getenv("FB_PAGE_ID")
    fb_page_access_token = os.getenv("FB_PAGE_ACCESS_TOKEN")
    
    if not fb_page_id or not fb_page_access_token:
        return None, "Facebook Page credentials not configured (FB_PAGE_ID, FB_PAGE_ACCESS_TOKEN)"
    
    # Validate token permissions
    print("📡 [Facebook] Validating Page token permissions...")
    token_info = _validate_fb_token(fb_page_id, fb_page_access_token)
    if not token_info["valid"]:
        print(f"⚠️ [Facebook] Token validation failed: {token_info['error']}")
        return None, f"Token validation failed: {token_info['error']}"
    
    print(f"✔ [Facebook] Token valid for Page: {token_info['page_name']} ({token_info['page_id']})")
    if token_info["missing_permissions"]:
        print(f"⚠️ [Facebook] Missing recommended permissions: {token_info['missing_permissions']}")
    
    try:
        # Step 1: Start upload session - get video_id and upload_url
        print("📡 [Facebook] Starting video upload session...")
        start_payload = {
            "upload_phase": "start",
            "video_state": "DRAFT",
            "description": caption[:2200],
            "access_token": fb_page_access_token,
        }
        print(f"📡 [Facebook] Start phase payload: {start_payload}")
        
        def _start_phase():
            resp = requests.post(
                f"{GRAPH_API_BASE}/{fb_page_id}/video_reels",
                data=start_payload,
                timeout=30,
            )
            response_text = resp.text
            print(f"📡 [Facebook] Start phase response: {resp.status_code} - {response_text}")
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                e.response_text = response_text
                raise
            return resp.json()
        
        start_data = _retry_with_backoff(_start_phase)
        video_id = start_data["video_id"]
        upload_url = start_data["upload_url"]
        print(f"✔ Upload session started. Video ID: {video_id}")
        print(f"📤 Upload URL: {upload_url}")
        
        # Step 2: Download video for binary upload
        # The rupload endpoint requires binary upload with Offset/Content-Length headers
        print("📤 [Facebook] Downloading video for binary upload...")
        try:
            video_resp = requests.get(video_url, timeout=120, stream=True)
            video_resp.raise_for_status()
            video_data = video_resp.content
            video_size = len(video_data)
            print(f"✔ Downloaded video: {video_size:,} bytes ({video_size/1024/1024:.1f} MB)")
        except Exception as e:
            print(f"⚠️ [Facebook] Download failed: {e}")
            raise
        
        # Step 3: Upload video binary to rupload.facebook.com
        print("📤 [Facebook] Uploading video binary to rupload.facebook.com...")
        upload_headers = {
            "Authorization": f"OAuth {fb_page_access_token}",
            "Offset": "0",
            "Content-Length": str(video_size),
            "X-Entity-Length": str(video_size),
            "Content-Type": "video/mp4",
        }
        
        def _upload_phase():
            upload_resp = requests.post(
                upload_url,
                headers=upload_headers,
                data=video_data,
                timeout=180,
            )
            response_text = upload_resp.text
            print(f"📤 [Facebook] Upload phase response: {upload_resp.status_code} - {response_text}")
            try:
                upload_resp.raise_for_status()
            except requests.HTTPError as e:
                e.response_text = response_text
                raise
            return upload_resp.json()
        
        upload_result = _retry_with_backoff(_upload_phase)
        print(f"✔ Video upload initiated: {upload_result}")
        
        # Step 4: Finish upload with description and publish
        print("📡 [Facebook] Finishing upload and publishing...")
        finish_payload = {
            "video_id": video_id,
            "upload_phase": "finish",
            "video_state": "PUBLISHED",
            "description": caption[:2200],
            "access_token": fb_page_access_token,
        }
        print(f"📡 [Facebook] Finish phase payload: {finish_payload}")
        
        def _finish_phase():
            resp = requests.post(
                f"{GRAPH_API_BASE}/{fb_page_id}/video_reels",
                data=finish_payload,
                timeout=30,
            )
            response_text = resp.text
            print(f"📡 [Facebook] Finish phase response: {resp.status_code} - {response_text}")
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                e.response_text = response_text
                raise
            return resp.json()
        
        finish_result = _retry_with_backoff(_finish_phase)
        print(f"✔ Upload finished: {finish_result}")
        
        # Step 5: Poll for processing status
        deadline = time.time() + 300  # 5 min timeout
        while time.time() < deadline:
            resp = requests.get(
                f"{GRAPH_API_BASE}/{video_id}",
                params={"fields": "status", "access_token": fb_page_access_token},
                timeout=30,
            )
            resp.raise_for_status()
            status = resp.json().get("status", {})
            processing_phase = status.get("processing_phase", {}).get("status")
            video_status = status.get("video_status")
            print(f"📊 [Facebook] Polling status: processing_phase={processing_phase}, video_status={video_status}")
            if processing_phase == "complete" or video_status == "published":
                break
            if processing_phase == "error" or video_status == "error":
                error_msg = status.get("processing_phase", {}).get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"Facebook Reel {video_id} failed processing: {error_msg}")
            time.sleep(5)
        else:
            # Don't fail on timeout - video might still process in background
            print(f"⚠️ Facebook Reel {video_id} still processing after 300s, but publish was initiated")
        
        print(f"🎉 Facebook Reel published! ID: {video_id}")
        return video_id, None
        
    except requests.HTTPError as e:
        response_text = getattr(e, 'response_text', None) or (e.response.text if e.response else "")
        error_data = _parse_meta_error(response_text)
        print(f"⚠️ video_reels endpoint failed: code={error_data['code']}, message={error_data['message']}, subcode={error_data['error_subcode']}, fbtrace_id={error_data['fbtrace_id']}")
        return None, f"video_reels API failed: {error_data['message']} (code {error_data['code']}, subcode {error_data['error_subcode']})"
    except Exception as e:
        print(f"⚠️ Facebook upload exception: {e}")
        return None, f"Facebook upload exception: {e}"


# ── Fallback: Regular Video Post ──────────────────────────────────────────────

def post_video_to_facebook_page_fallback(video_url: str, caption: str) -> tuple:
    """Fallback: Post as regular video via /videos endpoint (not Reel)."""
    fb_page_id = os.getenv("FB_PAGE_ID")
    fb_page_access_token = os.getenv("FB_PAGE_ACCESS_TOKEN")
    
    if not fb_page_id or not fb_page_access_token:
        return None, "Facebook Page credentials not configured"
    
    try:
        print("📡 [Facebook Fallback] Posting via /videos endpoint...")
        fallback_payload = {
            "file_url": video_url,
            "description": caption[:2200],
            "published": "true",
            "access_token": fb_page_access_token,
        }
        print(f"📡 [Facebook Fallback] Payload: {fallback_payload}")
        
        def _fallback_post():
            resp = requests.post(
                f"{GRAPH_API_BASE}/{fb_page_id}/videos",
                data=fallback_payload,
                timeout=60
            )
            response_text = resp.text
            print(f"📡 [Facebook Fallback] Response: {resp.status_code} - {response_text}")
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                e.response_text = response_text
                raise
            return resp.json()
        
        result = _retry_with_backoff(_fallback_post)
        video_id = result.get("id")
        print(f"🎉 Facebook Video published (fallback)! ID: {video_id}")
        return video_id, None
    except requests.HTTPError as e:
        response_text = getattr(e, 'response_text', None) or (e.response.text if e.response else "")
        error_data = _parse_meta_error(response_text)
        print(f"⚠️ Fallback /videos failed: code={error_data['code']}, message={error_data['message']}")
        return None, f"Fallback /videos failed: {error_data['message']} (code {error_data['code']})"
    except Exception as e:
        print(f"⚠️ Fallback exception: {e}")
        return None, f"Fallback exception: {e}"


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Facebook Reels Upload Module — Diagnostic Check")
    print("=" * 50)
    
    if _check_facebook_credentials():
        print("✔ Facebook Page credentials: Configured")
    else:
        print("⚠️ Facebook Page credentials: MISSING — set FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN in .env")
        missing = []
        if not os.getenv("FB_PAGE_ID"): missing.append("FB_PAGE_ID")
        if not os.getenv("FB_PAGE_ACCESS_TOKEN"): missing.append("FB_PAGE_ACCESS_TOKEN")
        print(f"   Missing keys: {', '.join(missing)}")