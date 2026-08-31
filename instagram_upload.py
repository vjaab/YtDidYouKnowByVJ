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
    # Option A: GitHub Releases (free, 2GB limit, works in CI)
    GITHUB_TOKEN          - Provided by GitHub Actions
    GITHUB_REPOSITORY     - e.g., "owner/repo" (repo must be PUBLIC)
    # Option B: Pre-signed/public URL provided externally
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

from facebook_upload import (
    post_video_to_facebook_page,
    post_video_to_facebook_page_fallback,
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


# ── GitHub Releases Temporary Video Hosting ──────────────────────────────────

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
        asset_data = asset_resp.json()
        if not isinstance(asset_data, dict) or "browser_download_url" not in asset_data:
            raise RuntimeError(f"Unexpected GitHub asset response: {asset_data}")
        public_url = asset_data["browser_download_url"]

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
    data = resp.json()
    if not isinstance(data, dict) or "id" not in data:
        raise RuntimeError(f"Unexpected response from create_reels_container: {data}")
    return data["id"]


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
        data = resp.json()
        if not isinstance(data, dict) or "status_code" not in data:
            raise RuntimeError(f"Unexpected response from wait_for_container: {data}")
        status = data["status_code"]
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
    data = resp.json()
    if not isinstance(data, dict) or "id" not in data:
        raise RuntimeError(f"Unexpected response from publish_container: {data}")
    return data["id"]


def upload_reel_to_instagram(video_path: str, caption: str):
    """
    Uploads a video as an Instagram Reel using the official Graph API.

    Workflow:
      1. Get public video URL (from GitHub Releases, or IG_VIDEO_PUBLIC_URL env var)
      2. POST /{ig-user-id}/media → create container (media_type=REELS)
      3. GET /{container-id}?fields=status_code → poll until FINISHED
      4. POST /{ig-user-id}/media_publish → publish the Reel
      5. Cleanup (GitHub Release kept for async Facebook video fetch)

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

    upload_method = None  # Track which method was used: "github", or "direct"
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
        
        # Option B: Upload to GitHub Releases (free, 2GB limit, works in CI)
        if not public_url:
            public_url, gh_result = upload_video_to_github_releases(video_path)
            if public_url:
                upload_method = "github"
                upload_key = gh_result  # release_id for cleanup
                print(f"✅ Uploaded to GitHub Releases")
            else:
                return False, f"Failed to host video publicly: {gh_result}"

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
        if upload_method == "github" and upload_key:
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