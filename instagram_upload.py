# -*- coding: utf-8 -*-
"""
instagram_upload.py — Official Instagram Reels Upload Module via Meta Graph API.

Implements the 3-step container workflow:
  1. Create IG media container (media_type=REELS) with a public video URL
  2. Poll until FINISHED -> Publish
  3. Return published media ID

Rate Limit: Instagram allows max 25 published posts per 24-hour window per account.
At the current pipeline cadence (2 Shorts/day), this is not a concern -- but if you
scale to bulk publishing, add a counter/lockfile to enforce this ceiling.

Token Lifecycle: Long-lived tokens expire after 60 days. The module auto-refreshes
before expiry. If refresh fails, it logs a warning and skips (non-fatal) rather than
crashing the pipeline, matching the graceful-degradation pattern used elsewhere.

Requires App Review approval of:
  - instagram_business_basic
  - instagram_business_content_publish
Until approved, this module will only run in dry-run/mock mode.

Env vars expected:
    IG_USER_ID       - Instagram Business Account ID
    IG_ACCESS_TOKEN  - long-lived access token with publish permission
    
    # Video hosting (choose ONE):
    # Option A: S3-compatible (R2, S3, MinIO, DO Spaces, etc.)
    # Supports both S3_* and R2_* (Cloudflare R2) naming conventions:
    S3_ENDPOINT_URL / R2_ENDPOINT_URL   - e.g., https://<account-id>.r2.cloudflarestorage.com
    S3_ACCESS_KEY_ID / R2_ACCESS_KEY_ID
    S3_SECRET_ACCESS_KEY / R2_SECRET_ACCESS_KEY
    S3_BUCKET_NAME / R2_BUCKET_NAME     - e.g., ig-temp-uploads
    S3_PUBLIC_DOMAIN / R2_PUBLIC_DOMAIN - e.g., pub-xxx.r2.dev (for public URL construction)
    # Option B: file.io (simple temporary file hosting)
    FILE_IO_API_KEY       - Optional API key for file.io (higher limits, longer expiry)
    # Option C: Pre-signed/public URL provided externally
    IG_VIDEO_PUBLIC_URL   - If set, skip upload and use this URL directly
"""

import os
import time
import json
import hashlib
import requests
from datetime import datetime, timedelta

from config import (
    BASE_DIR,
)

# ── Constants ────────────────────────────────────────────────────────────────
GRAPH_API_VERSION = "v21.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

# Token persistence path (for auto-refresh)
TOKEN_FILE = os.path.join(BASE_DIR, ".ig_token.json")

# Rate limit: Instagram caps at 25 posts per 24h per account.
# This file tracks daily post count to prevent hitting the ceiling.
RATE_LIMIT_FILE = os.path.join(BASE_DIR, ".ig_rate_limit.json")
MAX_POSTS_PER_DAY = 25

# Poll settings for container processing
MAX_POLL_ATTEMPTS = 60       # Max polling iterations (5 min at 5s interval)
POLL_INTERVAL_SECONDS = 5    # Seconds between polls


def _check_credentials():
    """Verifies that all required Instagram API credentials are configured."""
    ig_user_id = os.getenv("IG_USER_ID")
    ig_access_token = os.getenv("IG_ACCESS_TOKEN")
    return all(k and k.strip() for k in [ig_user_id, ig_access_token])


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
    return os.getenv("IG_ACCESS_TOKEN", ""), ""


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
        # Note: This requires app_id and app_secret in config/env
        # If not available, we'll skip refresh and hope the token works
        from config import INSTAGRAM_APP_ID, INSTAGRAM_APP_SECRET
        if not INSTAGRAM_APP_ID or not INSTAGRAM_APP_SECRET:
            print("⚠️ App ID/Secret not configured. Skipping token refresh.")
            return current_token

        url = f"{GRAPH_API_BASE}/oauth/access_token"
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


# ── Rate Limit Guard ────────────────────────────────────────────────────────

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

# ── S3-Compatible Temporary Video Hosting ────────────────────────────────────

def _check_s3_credentials():
    """Verifies that S3-compatible credentials are configured for temp video hosting.
    
    Supports both S3_* and R2_* (Cloudflare R2) environment variable naming conventions.
    """
    # Try S3_* first, then fall back to R2_*
    endpoint = os.getenv("S3_ENDPOINT_URL") or os.getenv("R2_ENDPOINT_URL")
    access_key = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("R2_SECRET_ACCESS_KEY")
    bucket = os.getenv("S3_BUCKET_NAME") or os.getenv("R2_BUCKET_NAME")
    public_domain = os.getenv("S3_PUBLIC_DOMAIN") or os.getenv("R2_PUBLIC_DOMAIN")
    
    return all(k and k.strip() for k in [endpoint, access_key, secret_key, bucket, public_domain])


def upload_video_to_s3(video_path):
    """
    Uploads a video file to an S3-compatible storage and returns the public URL.
    Works with Cloudflare R2, AWS S3, MinIO, DigitalOcean Spaces, etc.
    
    The file is given a unique hash-based name to avoid collisions.
    After Instagram fetches the video, call delete_from_s3() to clean up.
    """
    # Support both S3_* and R2_* env var naming conventions
    S3_ENDPOINT_URL = (os.getenv("S3_ENDPOINT_URL") or os.getenv("R2_ENDPOINT_URL", "")).rstrip('/')
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME") or os.getenv("R2_BUCKET_NAME", "ig-temp-uploads")
    S3_PUBLIC_DOMAIN = os.getenv("S3_PUBLIC_DOMAIN") or os.getenv("R2_PUBLIC_DOMAIN")
    S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("R2_ACCESS_KEY_ID")
    S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("R2_SECRET_ACCESS_KEY")

    if not all(k and k.strip() for k in [S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET_NAME, S3_PUBLIC_DOMAIN]):
        return None, "S3-compatible storage credentials not configured (need S3_ENDPOINT_URL/R2_ENDPOINT_URL, S3_ACCESS_KEY_ID/R2_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY/R2_SECRET_ACCESS_KEY, S3_BUCKET_NAME/R2_BUCKET_NAME, S3_PUBLIC_DOMAIN/R2_PUBLIC_DOMAIN)"

    # Generate unique filename based on content hash + timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_hash = hashlib.md5(f"{video_path}_{timestamp}".encode()).hexdigest()[:8]
    object_key = f"ig_temp_{timestamp}_{file_hash}.mp4"

    # S3-compatible PUT URL
    s3_url = f"{S3_ENDPOINT_URL}/{S3_BUCKET_NAME}/{object_key}"

    try:
        file_size = os.path.getsize(video_path)
        print(f"📤 Uploading {file_size / (1024*1024):.1f}MB to S3-compatible storage: {object_key}")

        with open(video_path, "rb") as f:
            # AWS Signature V4 would be needed for real AWS S3
            # For R2 and compatible, we can use simpler auth or pre-signed URLs
            # Here we use the simplest approach: for R2, use Bearer token
            # For generic S3, you'd need boto3 or requests-aws4auth
            import hmac
            import hashlib as hl
            
            # Simple approach: if endpoint contains cloudflare/r2, use Bearer
            # Otherwise, we'd need proper AWS SigV4 (use boto3 for production)
            if "cloudflare" in S3_ENDPOINT_URL.lower() or "r2" in S3_ENDPOINT_URL.lower():
                headers = {
                    "Authorization": f"Bearer {S3_SECRET_ACCESS_KEY}",  # R2 uses API token as secret
                    "Content-Type": "video/mp4",
                }
            else:
                # For other S3-compatible, fall back to basic approach
                # In production, use boto3 with proper credentials
                headers = {
                    "Content-Type": "video/mp4",
                }
            
            resp = requests.put(s3_url, headers=headers, data=f, timeout=120)

        if resp.status_code in (200, 201):
            public_url = f"https://{S3_PUBLIC_DOMAIN}/{object_key}"
            print(f"✅ Uploaded to S3: {public_url}")
            return public_url, object_key
        else:
            return None, f"S3 upload failed (HTTP {resp.status_code}): {resp.text}"

    except Exception as e:
        return None, f"S3 upload exception: {e}"


def delete_from_s3(object_key):
    """
    Deletes a temporary video file from S3-compatible storage after Instagram has fetched it.
    """
    # Support both S3_* and R2_* env var naming conventions
    S3_ENDPOINT_URL = (os.getenv("S3_ENDPOINT_URL") or os.getenv("R2_ENDPOINT_URL", "")).rstrip('/')
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME") or os.getenv("R2_BUCKET_NAME", "ig-temp-uploads")
    S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("R2_SECRET_ACCESS_KEY")

    if not object_key or not all(k and k.strip() for k in [S3_ENDPOINT_URL, S3_BUCKET_NAME, S3_SECRET_ACCESS_KEY]):
        return

    s3_url = f"{S3_ENDPOINT_URL}/{S3_BUCKET_NAME}/{object_key}"

    try:
        headers = {}
        if "cloudflare" in S3_ENDPOINT_URL.lower() or "r2" in S3_ENDPOINT_URL.lower():
            headers["Authorization"] = f"Bearer {S3_SECRET_ACCESS_KEY}"
        
        resp = requests.delete(s3_url, headers=headers, timeout=15)
        if resp.status_code in (200, 204):
            print(f"🗑️ S3 cleanup: Deleted temp file {object_key}")
        else:
            print(f"⚠️ S3 cleanup failed (HTTP {resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"⚠️ S3 cleanup exception (non-fatal): {e}")


# ── file.io Temporary Video Hosting ───────────────────────────────────────────

def _check_fileio_credentials():
    """Verifies file.io is available (works without API key, but key gives better limits)."""
    return True  # file.io works without credentials


def upload_video_to_fileio(video_path):
    """
    Uploads a video file to file.io and returns the public download URL.
    
    file.io provides temporary file hosting with automatic expiry.
    Free tier: 100MB max, 1 download or 14 days expiry (whichever comes first).
    With API key: Higher limits, customizable expiry.
    
    Note: Instagram needs to fetch the video during container processing.
    The link must remain valid long enough for Instagram to download it.
    """
    api_key = os.getenv("FILE_IO_API_KEY")
    
    file_size = os.path.getsize(video_path)
    size_mb = file_size / (1024 * 1024)
    
    if size_mb > 100 and not api_key:
        return None, f"File too large ({size_mb:.1f}MB) for free file.io tier (100MB limit). Use FILE_IO_API_KEY for larger files or use S3-compatible storage."

    print(f"📤 Uploading {size_mb:.1f}MB to file.io...")

    try:
        url = "https://file.io"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Options: set expiry to maximum (14 days = 1209600 seconds)
        # Note: free tier limits to 1 download OR 14 days
        data = {
            "expires": "14d",  # Maximum expiry
            "maxDownloads": "1",  # Instagram only needs to download once
        }
        
        with open(video_path, "rb") as f:
            files = {"file": (os.path.basename(video_path), f, "video/mp4")}
            # allow_redirects=True is needed because file.io returns 301 to Cloudflare
            resp = requests.post(url, headers=headers, data=data, files=files, timeout=180, allow_redirects=True)

        # Handle case where response might not be JSON (e.g., HTML error page)
        content_type = resp.headers.get('Content-Type', '')
        if 'application/json' not in content_type:
            return None, f"file.io returned non-JSON response (HTTP {resp.status_code}): {resp.text[:200]}"

        if resp.status_code in (200, 201):
            try:
                result = resp.json()
            except json.JSONDecodeError as e:
                return None, f"file.io upload failed: Invalid JSON response: {e}"
            if result.get("success"):
                public_url = result.get("link")
                expires = result.get("expires", "unknown")
                print(f"✅ Uploaded to file.io: {public_url} (expires: {expires})")
                return public_url, public_url  # Return URL as both public_url and "object_key" for tracking
            else:
                return None, f"file.io upload failed: {result}"
        else:
            return None, f"file.io upload failed (HTTP {resp.status_code}): {resp.text}"

    except Exception as e:
        return None, f"file.io upload exception: {e}"


def delete_from_fileio(object_key):
    """
    file.io files auto-delete after expiry or download. No manual cleanup needed.
    This is a no-op for compatibility with the cleanup pattern.
    """
    if object_key:
        print(f"🗑️ file.io cleanup: Files auto-expire after download/expiry (no manual delete needed)")


# ── Instagram Graph API: 3-Step Container Workflow ────────────────────────

def create_reels_container(video_url: str, caption: str, share_to_feed: bool = True) -> str:
    """Step 1: create a media container. Returns the container/creation ID."""
    ig_user_id = os.getenv("IG_USER_ID")
    access_token = os.getenv("IG_ACCESS_TOKEN")
    
    resp = requests.post(
        f"{GRAPH_API_BASE}/{ig_user_id}/media",
        data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption[:2200],  # Instagram caption limit
            "share_to_feed": str(share_to_feed).lower(),
            "access_token": access_token,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def wait_for_container(container_id: str, timeout_s: int = 300, poll_every_s: int = 5) -> None:
    """Step 2: poll until Instagram finishes processing the uploaded video."""
    access_token = os.getenv("IG_ACCESS_TOKEN")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.get(
            f"{GRAPH_API_BASE}/{container_id}",
            params={"fields": "status_code", "access_token": access_token},
            timeout=30,
        )
        resp.raise_for_status()
        status = resp.json()["status_code"]
        if status == "FINISHED":
            return
        if status == "ERROR":
            raise RuntimeError(f"Container {container_id} failed processing")
        time.sleep(poll_every_s)
    raise TimeoutError(f"Container {container_id} still processing after {timeout_s}s")


def publish_container(container_id: str) -> str:
    """Step 3: publish the finished container. Returns the published media ID."""
    ig_user_id = os.getenv("IG_USER_ID")
    access_token = os.getenv("IG_ACCESS_TOKEN")
    
    resp = requests.post(
        f"{GRAPH_API_BASE}/{ig_user_id}/media_publish",
        data={"creation_id": container_id, "access_token": access_token},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def upload_reel_to_instagram(video_path: str, caption: str):
    """
    Uploads a video as an Instagram Reel using the official Graph API.

    Workflow:
      1. Get public video URL (from file.io, S3-compatible upload, or IG_VIDEO_PUBLIC_URL env var)
      2. POST /{ig-user-id}/media → create container (media_type=REELS)
      3. GET /{container-id}?fields=status_code → poll until FINISHED
      4. POST /{ig-user-id}/media_publish → publish the Reel
      5. Cleanup (auto for file.io, manual for S3)

    Args:
        video_path (str): Absolute path to the .mp4 video file.
        caption (str): Caption text for the Instagram Reel.

    Returns:
        tuple: (bool success, str result_message_or_reel_id)
    """
    # ── Pre-flight checks ────────────────────────────────────────────────────
    if not _check_credentials():
        return False, "Skipped: Instagram credentials not configured (IG_USER_ID, IG_ACCESS_TOKEN)"

    if not os.path.exists(video_path):
        return False, f"Error: Video file not found at {video_path}"

    # Rate limit guard
    allowed, current_count = _check_rate_limit()
    if not allowed:
        return False, f"Skipped: Instagram rate limit reached ({current_count}/{MAX_POSTS_PER_DAY} posts today)"

    # Token management — refresh if close to expiry
    token, expiry = _load_token()
    token = _refresh_token_if_needed(token, expiry)
    
    # Update env var for the helper functions to use the refreshed token
    if token != os.getenv("IG_ACCESS_TOKEN"):
        os.environ["IG_ACCESS_TOKEN"] = token

    upload_method = None  # Track which method was used: "fileio", "s3", or "direct"
    upload_key = None     # Track for cleanup
    public_url = None

    try:
        # ── STEP 1: Get public video URL ───────────────────────────────────────
        print(f"📡 [Instagram] Step 1/4: Getting public video URL...")
        
        # Option A: Direct public URL from env (for pre-hosted videos)
        direct_url = os.getenv("IG_VIDEO_PUBLIC_URL")
        if direct_url and direct_url.strip():
            public_url = direct_url.strip()
            upload_method = "direct"
            print(f"✅ Using provided public URL: {public_url}")
        
        # Option B: Upload to file.io (simple, no infrastructure needed)
        if not public_url:
            print(f"📤 Trying file.io for temporary hosting...")
            public_url, fileio_result = upload_video_to_fileio(video_path)
            if public_url:
                upload_method = "fileio"
                upload_key = fileio_result
                print(f"✅ Uploaded to file.io")
            else:
                print(f"⚠️ file.io upload failed: {fileio_result}")
        
        # Option C: Upload to S3-compatible storage
        if not public_url:
            public_url, s3_result = upload_video_to_s3(video_path)
            if not public_url:
                return False, f"Failed to host video publicly: {s3_result}"
            upload_method = "s3"
            upload_key = s3_result  # Save for cleanup

        # ── STEP 2: Create Media Container ───────────────────────────────────
        print(f"📡 [Instagram] Step 2/4: Creating Reels container...")
        container_id = create_reels_container(public_url, caption)
        print(f"✔ Container created: {container_id}")

        # ── STEP 3: Poll for processing status ──────────────────────────────
        print(f"📡 [Instagram] Step 3/4: Waiting for video processing...")
        wait_for_container(container_id)
        print(f"✔ Video processing finished")

        # ── STEP 4: Publish the Reel ─────────────────────────────────────────
        print(f"📡 [Instagram] Step 4/4: Publishing Reel...")
        reel_id = publish_container(container_id)
        print(f"🎉 Instagram Reel published! ID: {reel_id}")

        # Track rate limit
        _increment_rate_limit()

        return True, reel_id

    except requests.HTTPError as e:
        return False, f"Instagram API error (HTTP {e.response.status_code}): {e.response.text}"
    except Exception as e:
        return False, f"Instagram upload exception: {e}"

    finally:
        # ── CLEANUP: Delete temp file based on upload method ──────────────────
        if upload_method == "s3" and upload_key:
            print("🧹 Cleaning up temporary S3 file...")
            delete_from_s3(upload_key)
        elif upload_method == "fileio" and upload_key:
            print("🧹 Cleaning up file.io reference...")
            delete_from_fileio(upload_key)
        # For "direct" method, no cleanup needed


# ── Standalone Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Instagram Reels Upload Module — Diagnostic Check")
    print("=" * 50)

    # 1. Credential check
    if _check_credentials():
        print("✔ Instagram API credentials: Configured")
    else:
        print("⚠️ Instagram API credentials: MISSING — set IG_USER_ID and IG_ACCESS_TOKEN in .env")
        missing = []
        if not os.getenv("IG_USER_ID"): missing.append("IG_USER_ID")
        if not os.getenv("IG_ACCESS_TOKEN"): missing.append("IG_ACCESS_TOKEN")
        print(f"   Missing keys: {', '.join(missing)}")

    # 2. Video hosting options
    print("\n📦 Video hosting options:")
    if _check_s3_credentials():
        print("  ✔ S3-compatible storage (S3_* or R2_*): Configured")
    else:
        print("  ⚠️ S3-compatible storage: Not configured (optional)")
        print("     Set: S3_ENDPOINT_URL/R2_ENDPOINT_URL, S3_ACCESS_KEY_ID/R2_ACCESS_KEY_ID,")
        print("          S3_SECRET_ACCESS_KEY/R2_SECRET_ACCESS_KEY, S3_BUCKET_NAME/R2_BUCKET_NAME,")
        print("          S3_PUBLIC_DOMAIN/R2_PUBLIC_DOMAIN")
    
    if _check_fileio_credentials():
        print("  ✔ file.io: Available (no config needed, optional FILE_IO_API_KEY for larger files)")
    else:
        print("  ⚠️ file.io: Not available")
    
    if os.getenv("IG_VIDEO_PUBLIC_URL"):
        print("  ✔ Direct public URL (IG_VIDEO_PUBLIC_URL): Provided")

    # 3. Rate limit check
    allowed, count = _check_rate_limit()
    print(f"\n✔ Rate limit: {count}/{MAX_POSTS_PER_DAY} posts today ({'OK' if allowed else 'LIMIT REACHED'})")

    # 4. Token check
    token, expiry = _load_token()
    if token:
        print(f"✔ Access token: Present (expiry: {expiry if expiry else 'unknown'})")
    else:
        print("⚠️ Access token: Not found")

    print("\n" + "=" * 50)
    if _check_credentials():
        print("✅ Ready for Instagram Reels publishing!")
    else:
        print("❌ Not ready — complete the manual setup steps first.")
    print("=" * 50)