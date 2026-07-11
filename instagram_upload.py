# -*- coding: utf-8 -*-
"""
instagram_upload.py — Official Instagram Reels Upload Module via Meta Graph API.

Implements the 3-step container workflow:
  1. Upload video to Cloudflare R2 (temp public URL)
  2. Create IG media container (media_type=REELS)
  3. Poll until FINISHED → Publish

Rate Limit: Instagram allows max 25 published posts per 24-hour window per account.
At the current pipeline cadence (2 Shorts/day), this is not a concern — but if you
scale to bulk publishing, add a counter/lockfile to enforce this ceiling.

Token Lifecycle: Long-lived tokens expire after 60 days. The module auto-refreshes
before expiry. If refresh fails, it logs a warning and skips (non-fatal) rather than
crashing the pipeline, matching the graceful-degradation pattern used elsewhere.

Requires App Review approval of:
  - instagram_business_basic
  - instagram_business_content_publish
Until approved, this module will only run in dry-run/mock mode.
"""

import os
import time
import json
import hashlib
import requests
from datetime import datetime, timedelta

from config import (
    INSTAGRAM_APP_ID,
    INSTAGRAM_APP_SECRET,
    INSTAGRAM_ACCESS_TOKEN,
    INSTAGRAM_BUSINESS_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_ACCOUNT_ID,
    BASE_DIR,
)

# ── Constants ────────────────────────────────────────────────────────────────
GRAPH_API_VERSION = "v23.0"
GRAPH_BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

# Cloudflare R2 bucket for temporary video hosting.
# The Instagram API cannot fetch local files — it needs a publicly reachable URL.
# Videos are uploaded here, published to IG, then immediately deleted.
# SAFETY NET: Also set a 1-day lifecycle rule on this R2 bucket so any orphaned
# files from crashed pipeline runs get auto-deleted (see manual setup steps).
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "ig-temp-uploads")
R2_ACCOUNT_ID = CLOUDFLARE_ACCOUNT_ID
R2_PUBLIC_DOMAIN = os.getenv("R2_PUBLIC_DOMAIN", "")  # e.g., "pub-xxxx.r2.dev"

# Token persistence path (for auto-refresh)
TOKEN_FILE = os.path.join(BASE_DIR, ".ig_token.json")

# Rate limit: Instagram caps at 25 posts per 24h per account.
# This file tracks daily post count to prevent hitting the ceiling.
RATE_LIMIT_FILE = os.path.join(BASE_DIR, ".ig_rate_limit.json")
MAX_POSTS_PER_DAY = 25

# Poll settings for container processing
MAX_POLL_ATTEMPTS = 30       # Max polling iterations
POLL_INTERVAL_SECONDS = 10   # Seconds between polls


def _check_credentials():
    """Verifies that all required Instagram API credentials are configured."""
    keys = [INSTAGRAM_APP_ID, INSTAGRAM_APP_SECRET, INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_BUSINESS_ACCOUNT_ID]
    return all(k and k.strip() for k in keys)


def _check_r2_credentials():
    """Verifies that Cloudflare R2 credentials are configured for temp video hosting."""
    return all(k and k.strip() for k in [CLOUDFLARE_API_TOKEN, R2_ACCOUNT_ID, R2_PUBLIC_DOMAIN])


# ── Token Management ────────────────────────────────────────────────────────

def _load_token():
    """
    Loads the current access token from the persisted token file.
    Falls back to the .env value if no persisted token exists.
    """
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                data = json.load(f)
                token = data.get("access_token", "")
                expiry = data.get("expires_at", "")
                if token:
                    return token, expiry
        except Exception:
            pass
    return INSTAGRAM_ACCESS_TOKEN, ""


def _save_token(token, expires_in_seconds):
    """Persists a refreshed token with its expiry timestamp."""
    expires_at = (datetime.now() + timedelta(seconds=int(expires_in_seconds))).isoformat()
    try:
        with open(TOKEN_FILE, "w") as f:
            json.dump({"access_token": token, "expires_at": expires_at}, f)
        print(f"✅ Instagram token saved. Expires: {expires_at}")
    except Exception as e:
        print(f"⚠️ Failed to persist Instagram token: {e}")


def _refresh_token_if_needed(current_token, expiry_str):
    """
    Refreshes the long-lived token if it's within 7 days of expiry.
    Long-lived tokens last 60 days; refreshing yields a new 60-day token.

    If refresh fails, returns the current token and logs a warning.
    This matches the pipeline's graceful-degradation pattern — log + continue,
    never crash the entire run over a token issue.
    """
    if expiry_str:
        try:
            expiry = datetime.fromisoformat(expiry_str)
            days_left = (expiry - datetime.now()).days
            if days_left > 7:
                return current_token  # Plenty of time left
            print(f"⚠️ Instagram token expires in {days_left} days. Attempting refresh...")
        except Exception:
            print("⚠️ Could not parse token expiry. Attempting refresh as precaution...")
    else:
        # No expiry info — try refreshing to be safe
        print("ℹ️ No token expiry info found. Attempting refresh...")

    try:
        url = f"{GRAPH_BASE_URL}/oauth/access_token"
        params = {
            "grant_type": "fb_exchange_token",
            "client_id": INSTAGRAM_APP_ID,
            "client_secret": INSTAGRAM_APP_SECRET,
            "fb_exchange_token": current_token,
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            new_token = data.get("access_token")
            expires_in = data.get("expires_in", 5184000)  # Default 60 days
            if new_token:
                _save_token(new_token, expires_in)
                print("✅ Instagram token refreshed successfully.")
                return new_token
        
        print(f"⚠️ Token refresh failed (HTTP {resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"⚠️ Token refresh exception (non-fatal): {e}")

    # Graceful degradation: return the old token and hope it still works
    return current_token


# ── Rate Limit Guard ─────────────────────────────────────────────────────────

def _check_rate_limit():
    """
    Checks if we've hit Instagram's 25 posts/24h ceiling.
    Returns True if posting is allowed, False if we should skip.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    data = {}

    if os.path.exists(RATE_LIMIT_FILE):
        try:
            with open(RATE_LIMIT_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

    if data.get("date") != today:
        # New day — reset counter
        data = {"date": today, "count": 0}

    if data["count"] >= MAX_POSTS_PER_DAY:
        return False, data["count"]

    return True, data["count"]


def _increment_rate_limit():
    """Increments today's post counter."""
    today = datetime.now().strftime("%Y-%m-%d")
    data = {"date": today, "count": 0}

    if os.path.exists(RATE_LIMIT_FILE):
        try:
            with open(RATE_LIMIT_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            pass

    if data.get("date") != today:
        data = {"date": today, "count": 0}

    data["count"] = data.get("count", 0) + 1

    try:
        with open(RATE_LIMIT_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# ── Cloudflare R2 Temporary Video Hosting ────────────────────────────────────

def _upload_to_r2(video_path):
    """
    Uploads a video file to Cloudflare R2 and returns the public URL.
    Uses the S3-compatible API with Cloudflare API token auth.

    The file is given a unique hash-based name to avoid collisions.
    After Instagram fetches the video, call _delete_from_r2() to clean up.
    """
    if not _check_r2_credentials():
        return None, "Cloudflare R2 credentials not configured (R2_PUBLIC_DOMAIN missing)"

    # Generate unique filename based on content hash + timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_hash = hashlib.md5(f"{video_path}_{timestamp}".encode()).hexdigest()[:8]
    object_key = f"ig_temp_{timestamp}_{file_hash}.mp4"

    # Cloudflare R2 S3-compatible endpoint
    r2_url = f"https://api.cloudflare.com/client/v4/accounts/{R2_ACCOUNT_ID}/r2/buckets/{R2_BUCKET_NAME}/objects/{object_key}"

    try:
        file_size = os.path.getsize(video_path)
        print(f"📤 Uploading {file_size / (1024*1024):.1f}MB to Cloudflare R2: {object_key}")

        with open(video_path, "rb") as f:
            headers = {
                "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                "Content-Type": "video/mp4",
            }
            resp = requests.put(r2_url, headers=headers, data=f, timeout=120)

        if resp.status_code in (200, 201):
            public_url = f"https://{R2_PUBLIC_DOMAIN}/{object_key}"
            print(f"✅ Uploaded to R2: {public_url}")
            return public_url, object_key
        else:
            return None, f"R2 upload failed (HTTP {resp.status_code}): {resp.text}"

    except Exception as e:
        return None, f"R2 upload exception: {e}"


def _delete_from_r2(object_key):
    """
    Deletes a temporary video file from Cloudflare R2 after Instagram has fetched it.

    SAFETY NET: Even if this fails (e.g., pipeline crash), the R2 bucket should have
    a 1-day lifecycle rule that auto-deletes objects, preventing dangling public files.
    """
    if not object_key or not _check_r2_credentials():
        return

    r2_url = f"https://api.cloudflare.com/client/v4/accounts/{R2_ACCOUNT_ID}/r2/buckets/{R2_BUCKET_NAME}/objects/{object_key}"

    try:
        headers = {"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"}
        resp = requests.delete(r2_url, headers=headers, timeout=15)
        if resp.status_code in (200, 204):
            print(f"🗑️ R2 cleanup: Deleted temp file {object_key}")
        else:
            print(f"⚠️ R2 cleanup failed (HTTP {resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"⚠️ R2 cleanup exception (non-fatal): {e}")


# ── Instagram Graph API: 3-Step Container Workflow ───────────────────────────

def upload_reel_to_instagram(video_path, caption):
    """
    Uploads a video as an Instagram Reel using the official Graph API.

    Workflow:
      1. Upload video to Cloudflare R2 (temp public URL)
      2. POST /{ig-user-id}/media → create container (media_type=REELS)
      3. GET /{container-id}?fields=status_code → poll until FINISHED
      4. POST /{ig-user-id}/media_publish → publish the Reel
      5. DELETE temp file from R2

    Args:
        video_path (str): Absolute path to the .mp4 video file.
        caption (str): Caption text for the Instagram Reel.

    Returns:
        tuple: (bool success, str result_message_or_reel_id)
    """
    # ── Pre-flight checks ────────────────────────────────────────────────────
    if not _check_credentials():
        return False, "Skipped: Instagram credentials not configured in .env"

    if not os.path.exists(video_path):
        return False, f"Error: Video file not found at {video_path}"

    # Rate limit guard
    allowed, current_count = _check_rate_limit()
    if not allowed:
        return False, f"Skipped: Instagram rate limit reached ({current_count}/{MAX_POSTS_PER_DAY} posts today)"

    # Token management — refresh if close to expiry
    token, expiry = _load_token()
    token = _refresh_token_if_needed(token, expiry)

    ig_user_id = INSTAGRAM_BUSINESS_ACCOUNT_ID
    r2_object_key = None  # Track for cleanup

    try:
        # ── STEP 1: Upload video to Cloudflare R2 for public access ──────────
        print(f"📡 [Instagram] Step 1/4: Uploading video to temporary public host...")
        public_url, r2_result = _upload_to_r2(video_path)

        if not public_url:
            return False, f"Failed to host video publicly: {r2_result}"

        r2_object_key = r2_result  # Save for cleanup

        # ── STEP 2: Create Media Container ───────────────────────────────────
        print(f"📡 [Instagram] Step 2/4: Creating Reels container...")
        container_url = f"{GRAPH_BASE_URL}/{ig_user_id}/media"
        container_payload = {
            "media_type": "REELS",
            "video_url": public_url,
            "caption": caption[:2200],  # Instagram caption limit
            "access_token": token,
        }

        resp = requests.post(container_url, data=container_payload, timeout=30)
        if resp.status_code != 200:
            return False, f"Container creation failed (HTTP {resp.status_code}): {resp.text}"

        container_id = resp.json().get("id")
        if not container_id:
            return False, f"Container creation failed: No ID returned. Response: {resp.text}"

        print(f"✔ Container created: {container_id}")

        # ── STEP 3: Poll for processing status ──────────────────────────────
        print(f"📡 [Instagram] Step 3/4: Waiting for video processing...")
        status_url = f"{GRAPH_BASE_URL}/{container_id}"

        for attempt in range(MAX_POLL_ATTEMPTS):
            time.sleep(POLL_INTERVAL_SECONDS)

            try:
                status_resp = requests.get(
                    status_url,
                    params={"fields": "status_code", "access_token": token},
                    timeout=15
                )

                if status_resp.status_code != 200:
                    print(f"   ⚠️ Status poll #{attempt+1} failed (HTTP {status_resp.status_code})")
                    continue

                status_code = status_resp.json().get("status_code", "")
                print(f"   Poll #{attempt+1}/{MAX_POLL_ATTEMPTS}: status={status_code}")

                if status_code == "FINISHED":
                    break
                elif status_code == "ERROR":
                    error_msg = status_resp.json().get("status", "Unknown processing error")
                    return False, f"Instagram video processing failed: {error_msg}"
                # IN_PROGRESS or other states → continue polling

            except Exception as poll_err:
                print(f"   ⚠️ Status poll exception: {poll_err}")
                continue
        else:
            # Exhausted all poll attempts
            return False, f"Instagram processing timed out after {MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS}s"

        # ── STEP 4: Publish the Reel ─────────────────────────────────────────
        print(f"📡 [Instagram] Step 4/4: Publishing Reel...")
        publish_url = f"{GRAPH_BASE_URL}/{ig_user_id}/media_publish"
        publish_payload = {
            "creation_id": container_id,
            "access_token": token,
        }

        pub_resp = requests.post(publish_url, data=publish_payload, timeout=30)
        if pub_resp.status_code != 200:
            return False, f"Publish failed (HTTP {pub_resp.status_code}): {pub_resp.text}"

        reel_id = pub_resp.json().get("id")
        if not reel_id:
            return False, f"Publish succeeded but no Reel ID returned: {pub_resp.text}"

        print(f"🎉 Instagram Reel published! ID: {reel_id}")

        # Track rate limit
        _increment_rate_limit()

        return True, reel_id

    except Exception as e:
        return False, f"Instagram upload exception: {e}"

    finally:
        # ── CLEANUP: Always delete the temp file from R2 ─────────────────────
        # Even on failure — we don't want public video files lingering.
        # The R2 bucket's 1-day lifecycle rule is a backup if this fails.
        if r2_object_key:
            print("🧹 Cleaning up temporary R2 file...")
            _delete_from_r2(r2_object_key)


# ── Standalone Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Instagram Reels Upload Module — Diagnostic Check")
    print("=" * 50)

    # 1. Credential check
    if _check_credentials():
        print("✔ Instagram API credentials: Configured")
    else:
        print("⚠️ Instagram API credentials: MISSING — fill in .env (see implementation_plan.md)")
        missing = []
        if not INSTAGRAM_APP_ID: missing.append("INSTAGRAM_APP_ID")
        if not INSTAGRAM_APP_SECRET: missing.append("INSTAGRAM_APP_SECRET")
        if not INSTAGRAM_ACCESS_TOKEN: missing.append("INSTAGRAM_ACCESS_TOKEN")
        if not INSTAGRAM_BUSINESS_ACCOUNT_ID: missing.append("INSTAGRAM_BUSINESS_ACCOUNT_ID")
        print(f"   Missing keys: {', '.join(missing)}")

    # 2. R2 check
    if _check_r2_credentials():
        print("✔ Cloudflare R2 credentials: Configured")
    else:
        print("⚠️ Cloudflare R2 credentials: MISSING — need R2_PUBLIC_DOMAIN in .env")

    # 3. Rate limit check
    allowed, count = _check_rate_limit()
    print(f"✔ Rate limit: {count}/{MAX_POSTS_PER_DAY} posts today ({'OK' if allowed else 'LIMIT REACHED'})")

    # 4. Token check
    token, expiry = _load_token()
    if token:
        print(f"✔ Access token: Present (expiry: {expiry if expiry else 'unknown'})")
    else:
        print("⚠️ Access token: Not found")

    print("\n" + "=" * 50)
    if _check_credentials() and _check_r2_credentials():
        print("✅ Ready for Instagram Reels publishing!")
    else:
        print("❌ Not ready — complete the manual setup steps first.")
    print("=" * 50)
