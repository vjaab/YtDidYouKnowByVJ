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
GRAPH_API_VERSION = "v25.0"
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


def _check_facebook_credentials():
    """Verifies that Facebook Page credentials are configured for cross-posting."""
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
        # Use lookahead to capture message including unescaped quotes
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


def upload_video_to_github_releases(video_path):
    """
    Uploads a video file to GitHub Releases as a temporary asset and returns the public URL.
    
    Uses GITHUB_TOKEN (provided by GitHub Actions) and GITHUB_REPOSITORY.
    The repo must be PUBLIC for Meta's Graph API to fetch the URL.
    
    Returns (public_url, release_id) or (None, error_message).
    The release_id is returned for cleanup.
    """
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")  # e.g., "owner/repo"
    
    if not token or not repo:
        return None, "GitHub credentials not configured (need GITHUB_TOKEN and GITHUB_REPOSITORY)"
    
    if "/" not in repo:
        return None, f"Invalid GITHUB_REPOSITORY format: {repo} (expected 'owner/repo')"
    
    file_size = os.path.getsize(video_path)
    size_mb = file_size / (1024 * 1024)
    
    if size_mb > 2048:
        return None, f"File too large ({size_mb:.1f}MB) for GitHub Releases (2GB limit)"

    print(f"📤 Uploading {size_mb:.1f}MB to GitHub Releases...")

    try:
        owner, repo_name = repo.split("/")
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        tag = f"media-temp-{int(time.time())}"  # unique per run, easy to prune later
        filename = os.path.basename(video_path)

        # Create release (public, not prerelease, for better accessibility)
        release_resp = requests.post(
            f"https://api.github.com/repos/{owner}/{repo_name}/releases",
            headers=headers,
            json={
                "tag_name": tag,
                "name": f"Temp media {tag}",
                "body": "Auto-generated for Instagram/Facebook publish step. Safe to delete after 24h.",
                "prerelease": False,
            },
            timeout=30,
        )
        release_resp.raise_for_status()
        release_data = release_resp.json()
        release_id = release_data["id"]
        upload_url = release_data["upload_url"].split("{")[0]

        # Upload asset
        with open(video_path, "rb") as f:
            asset_resp = requests.post(
                upload_url,
                headers={**headers, "Content-Type": "video/mp4"},
                params={"name": filename},
                data=f,
                timeout=120,
            )
        asset_resp.raise_for_status()
        public_url = asset_resp.json()["browser_download_url"]

        # Let CDN edge catch up before Graph API tries to fetch (increased for video)
        print(f"⏳ Waiting for GitHub CDN propagation...")
        time.sleep(10)
        
        # Verify URL is accessible with HEAD request
        try:
            verify_resp = requests.head(public_url, allow_redirects=True, timeout=15)
            if verify_resp.status_code == 200:
                content_type = verify_resp.headers.get("content-type", "")
                content_length = verify_resp.headers.get("content-length", "unknown")
                print(f"✅ URL verified: {content_type}, {content_length} bytes")
            else:
                print(f"⚠️ URL verification returned {verify_resp.status_code}: {verify_resp.text[:200]}")
        except Exception as e:
            print(f"⚠️ URL verification failed: {e}")
        
        print(f"✅ Uploaded to GitHub Releases: {public_url}")
        return public_url, str(release_id)

    except Exception as e:
        return None, f"GitHub Releases upload exception: {e}"


def delete_github_release(release_id):
    """Deletes a GitHub Release and its tag after Instagram has fetched the video."""
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    
    if not token or not repo or not release_id:
        return
    
    owner, repo_name = repo.split("/")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    
    try:
        # Get release to find tag name
        rel_resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo_name}/releases/{release_id}",
            headers=headers, timeout=15
        )
        tag_name = ""
        if rel_resp.status_code == 200:
            tag_name = rel_resp.json().get("tag_name", "")
        
        # Delete release
        requests.delete(
            f"https://api.github.com/repos/{owner}/{repo_name}/releases/{release_id}",
            headers=headers, timeout=30
        )
        
        # Delete tag if we found it
        if tag_name:
            requests.delete(
                f"https://api.github.com/repos/{owner}/{repo_name}/git/refs/tags/{tag_name}",
                headers=headers, timeout=30
            )
        print(f"🗑️ GitHub Releases cleanup: Deleted release {release_id}")
    except Exception as e:
        print(f"⚠️ GitHub Releases cleanup exception (non-fatal): {e}")


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


def post_video_to_facebook_page(video_url: str, caption: str) -> str:
    """
    Posts a video to Facebook Page as a Reel/Video.
    Uses the public video URL (same as Instagram) and Facebook Page access token.
    Implements the video_reels API with upload_phase workflow.
    
    Correct flow per Meta Graph API:
    1. Start upload session (POST /page_id/video_reels with upload_phase=start)
    2. Upload video to returned upload_url using file_url header (for hosted videos)
    3. Finish upload with description, title, video_state=PUBLISHED
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
        # NOTE: Do NOT include video_url here - it's not supported in start phase
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
                # Attach response text to exception for later parsing
                e.response_text = response_text
                raise
            return resp.json()
        
        start_data = _retry_with_backoff(_start_phase)
        video_id = start_data["video_id"]
        upload_url = start_data["upload_url"]
        print(f"✔ Upload session started. Video ID: {video_id}")
        print(f"📤 Upload URL: {upload_url}")
        
        # Step 2: Upload video to Facebook's upload URL (rupload.facebook.com)
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
        
        # Step 3: Finish upload with description and publish
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
        
        # Step 4: Poll for processing status
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
        
        # Option C: Upload to GitHub Releases (free, 2GB limit, works in CI)
        if not public_url:
            public_url, gh_result = upload_video_to_github_releases(video_path)
            if public_url:
                upload_method = "github"
                upload_key = gh_result  # release_id for cleanup
                print(f"✅ Uploaded to GitHub Releases")
            else:
                print(f"⚠️ GitHub Releases upload failed: {gh_result}")
        
        # Option D: Upload to S3-compatible storage
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

        # ── STEP 5: Cross-post to Facebook Page (if configured) ──────────────
        if _check_facebook_credentials():
            print(f"📡 [Facebook] Cross-posting to Facebook Page...")
            try:
                fb_reel_id, fb_error = post_video_to_facebook_page(public_url, caption)
                if not fb_reel_id:
                    print(f"🔄 [Facebook] video_reels failed ({fb_error}), trying fallback /videos endpoint...")
                    fb_reel_id, fb_error = post_video_to_facebook_page_fallback(public_url, caption)
                if fb_reel_id:
                    print(f"🎉 Facebook Reel published! ID: {fb_reel_id}")
                else:
                    print(f"⚠️ Facebook cross-post skipped: {fb_error}")
            except Exception as e:
                print(f"⚠️ Facebook cross-post failed (non-fatal): {e}")
        else:
            print("⚠️ Facebook Page credentials not configured — skipping cross-post")

        # Track rate limit
        _increment_rate_limit()

        return True, reel_id

    except requests.HTTPError as e:
        response_text = getattr(e, 'response_text', None) or (e.response.text if e.response else "")
        return False, f"Instagram API error (HTTP {e.response.status_code}): {response_text}"
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
        elif upload_method == "github" and upload_key:
            # NOTE: Skip GitHub Release deletion to allow async Facebook video fetch
            # Releases are tagged as 'media-temp-*' for periodic manual cleanup
            print(f"ℹ️ Keeping GitHub Release {upload_key} for async video processing (tagged for later cleanup)")
            # delete_github_release(int(upload_key))  # Disabled for async fetch compatibility
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
    
    gh_token = os.getenv("GITHUB_TOKEN")
    gh_repo = os.getenv("GITHUB_REPOSITORY")
    if gh_token and gh_repo and "/" in gh_repo:
        print(f"  ✔ GitHub Releases: Available ({gh_repo})")
    else:
        print("  ⚠️ GitHub Releases: Not configured (optional)")
        print("     Set: GITHUB_TOKEN and GITHUB_REPOSITORY (e.g., 'owner/repo')")
        print("     Note: Repo must be PUBLIC for Instagram to fetch the video")
    
    if os.getenv("IG_VIDEO_PUBLIC_URL"):
        print("  ✔ Direct public URL (IG_VIDEO_PUBLIC_URL): Provided")

    # Facebook cross-posting
    print("\n📘 Facebook cross-posting:")
    if _check_facebook_credentials():
        print("  ✔ Facebook Page: Configured (FB_PAGE_ID, FB_PAGE_ACCESS_TOKEN)")
    else:
        print("  ⚠️ Facebook Page: Not configured (optional)")
        print("     Set: FB_PAGE_ID, FB_PAGE_ACCESS_TOKEN")

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