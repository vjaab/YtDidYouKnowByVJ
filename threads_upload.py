# -*- coding: utf-8 -*-
"""
threads_upload.py — Threads Publishing Module via Meta Graph API.

Implements the 2-step container workflow (same as Instagram Reels):
  1. Create Threads media container (media_type=VIDEO) with a public video URL
  2. Publish the container

Rate Limit: 250 publishing actions per user per 24-hour window (as of Sep 2025).
No linked Instagram account required — needs Tech Provider Verification on Meta app
and threads_content_publish scope.

Env vars expected:
    THREADS_USER_ID       - Threads User ID (from /me endpoint)
    THREADS_ACCESS_TOKEN  - Long-lived access token with threads_content_publish scope
    
    # Video hosting (choose ONE):
    # Option A: GitHub Releases (free, 2GB limit, works in CI)
    GITHUB_TOKEN          - Provided by GitHub Actions
    GITHUB_REPOSITORY     - e.g., "owner/repo" (repo must be PUBLIC)
    # Option B: Pre-signed/public URL provided externally
    THREADS_VIDEO_PUBLIC_URL   - If set, skip upload and use this URL directly
"""

import os
import time
import json
import requests
from datetime import datetime, timedelta

from config import BASE_DIR

# ── Constants ────────────────────────────────────────────────────────────────
GRAPH_API_VERSION = "v1.0"
GRAPH_API_BASE = f"https://graph.threads.net/{GRAPH_API_VERSION}"

# Token persistence path (for auto-refresh)
TOKEN_FILE = os.path.join(BASE_DIR, ".threads_token.json")

# Rate limit: Threads caps at 250 posts per 24h per user.
RATE_LIMIT_FILE = os.path.join(BASE_DIR, ".threads_rate_limit.json")
MAX_POSTS_PER_DAY = 250

# Poll settings for container processing
MAX_POLL_ATTEMPTS = 60
POLL_INTERVAL_SECONDS = 5


def _check_credentials():
    """Verifies that all required Threads API credentials are configured."""
    threads_user_id = os.getenv("THREADS_USER_ID")
    threads_access_token = os.getenv("THREADS_ACCESS_TOKEN")
    return all(k and k.strip() for k in [threads_user_id, threads_access_token])


# ── Token Management ────────────────────────────────────────────────────────

def _load_token():
    """Loads the current access token from the persisted token file."""
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
    return os.getenv("THREADS_ACCESS_TOKEN", ""), ""


def _save_token(token, expires_in_seconds):
    """Persists a refreshed token with its expiry timestamp."""
    expires_at = (datetime.now() + timedelta(seconds=int(expires_in_seconds))).isoformat()
    try:
        with open(TOKEN_FILE, "w") as f:
            json.dump({"access_token": token, "expires_at": expires_at}, f)
        print(f"✅ Threads token saved. Expires: {expires_at}")
    except Exception as e:
        print(f"⚠️ Failed to persist Threads token: {e}")


def _refresh_token_if_needed(current_token, expiry_str):
    """Thread tokens are already long-lived (60 days) and don't use fb_exchange_token.
    They need to be manually regenerated via the app dashboard when expired.
    This function just returns the current token."""
    if expiry_str:
        try:
            expiry = datetime.fromisoformat(expiry_str)
            days_left = (expiry - datetime.now()).days
            if days_left <= 7:
                print(f"⚠️ Threads token expires in {days_left} days. Please regenerate via Meta App Dashboard.")
        except Exception:
            pass
    return current_token


# ── Rate Limit Guard ────────────────────────────────────────────────────────

def _check_rate_limit():
    """Checks if we've hit Threads' 250 posts/24h ceiling."""
    today = datetime.now().strftime("%Y-%m-%d")
    data = {}

    if os.path.exists(RATE_LIMIT_FILE):
        try:
            with open(RATE_LIMIT_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

    if data.get("date") != today:
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
    """Uploads a video file to GitHub Releases and returns the public URL."""
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")

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

        # Check if repo is public (Threads needs public access to fetch video)
        repo_resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo_name}",
            headers=headers, timeout=15
        )
        if repo_resp.status_code == 200:
            repo_data = repo_resp.json()
            if repo_data.get("private", True):
                print(f"⚠️ WARNING: Repository {owner}/{repo_name} is PRIVATE. Threads cannot fetch video from private repos!")
                print(f"   Make the repo public or use THREADS_VIDEO_PUBLIC_URL with a public CDN URL.")
        else:
            print(f"⚠️ Could not verify repo visibility: {repo_resp.status_code}")

        tag = f"media-threads-{int(time.time())}"
        filename = os.path.basename(video_path)

        release_resp = requests.post(
            f"https://api.github.com/repos/{owner}/{repo_name}/releases",
            headers=headers,
            json={
                "tag_name": tag,
                "name": f"Temp media {tag}",
                "body": "Auto-generated for Threads publish step. Safe to delete after 24h.",
                "prerelease": False,
            },
            timeout=30,
        )
        release_resp.raise_for_status()
        release_data = release_resp.json()
        release_id = release_data["id"]
        upload_url = release_data["upload_url"].split("{")[0]

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

        print(f"⏳ Waiting for GitHub CDN propagation...")
        time.sleep(10)

        try:
            verify_resp = requests.head(public_url, allow_redirects=True, timeout=15)
            if verify_resp.status_code == 200:
                content_type = verify_resp.headers.get("content-type", "")
                content_length = verify_resp.headers.get("content-length", "unknown")
                print(f"✅ URL verified: {content_type}, {content_length} bytes")
        except Exception as e:
            print(f"⚠️ URL verification failed: {e}")

        print(f"✅ Uploaded to GitHub Releases: {public_url}")
        return public_url, str(release_id)

    except Exception as e:
        return None, f"GitHub Releases upload exception: {e}"


def delete_github_release(release_id):
    """Deletes a GitHub Release after Threads has fetched the video."""
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")

    if not token or not repo or not release_id:
        return

    owner, repo_name = repo.split("/")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    try:
        rel_resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo_name}/releases/{release_id}",
            headers=headers, timeout=15
        )
        tag_name = ""
        if rel_resp.status_code == 200:
            tag_name = rel_resp.json().get("tag_name", "")

        requests.delete(
            f"https://api.github.com/repos/{owner}/{repo_name}/releases/{release_id}",
            headers=headers, timeout=30
        )

        if tag_name:
            requests.delete(
                f"https://api.github.com/repos/{owner}/{repo_name}/git/refs/tags/{tag_name}",
                headers=headers, timeout=30
            )
        print(f"🗑️ GitHub Releases cleanup: Deleted release {release_id}")
    except Exception as e:
        print(f"⚠️ GitHub Releases cleanup exception (non-fatal): {e}")


# ── Threads Graph API: 2-Step Container Workflow ────────────────────────────

def create_threads_container(video_url: str, caption: str, reply_to_id: str = None) -> str:
    """Step 1: create a media container. Returns the container/creation ID.
    
    Args:
        video_url: Public URL of the video
        caption: Text caption for the post
        reply_to_id: Optional ID of the post to reply to (for threading)
    """
    threads_user_id = os.getenv("THREADS_USER_ID")
    access_token = os.getenv("THREADS_ACCESS_TOKEN")

    data = {
        "media_type": "VIDEO",
        "video_url": video_url,
        "text": caption[:2200],  # Threads text limit
        "access_token": access_token,
    }
    if reply_to_id:
        data["reply_to_id"] = reply_to_id

    resp = requests.post(
        f"{GRAPH_API_BASE}/{threads_user_id}/threads",
        data=data,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def create_threads_reply(post_id: str, reply_text: str) -> str:
    """Create a text-only reply to an existing Threads post.
    
    Args:
        post_id: The ID of the post to reply to
        reply_text: Text content of the reply
        
    Returns:
        The reply post ID
    """
    threads_user_id = os.getenv("THREADS_USER_ID")
    access_token = os.getenv("THREADS_ACCESS_TOKEN")

    resp = requests.post(
        f"{GRAPH_API_BASE}/{threads_user_id}/threads",
        data={
            "text": reply_text[:2200],
            "reply_to_id": post_id,
            "access_token": access_token,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def wait_for_container(container_id: str, timeout_s: int = 300, poll_every_s: int = 5) -> None:
    """Step 2: poll until Threads finishes processing the uploaded video."""
    access_token = os.getenv("THREADS_ACCESS_TOKEN")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.get(
            f"{GRAPH_API_BASE}/{container_id}",
            params={"fields": "status,error_message", "access_token": access_token},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "UNKNOWN")
        if status == "FINISHED":
            return
        if status == "ERROR":
            error_msg = data.get("error_message", "No error message provided")
            raise RuntimeError(f"Container {container_id} failed processing: {error_msg}")
        time.sleep(poll_every_s)
    raise TimeoutError(f"Container {container_id} still processing after {timeout_s}s")


def publish_container(container_id: str) -> str:
    """Step 3: publish the finished container. Returns the published media ID."""
    threads_user_id = os.getenv("THREADS_USER_ID")
    access_token = os.getenv("THREADS_ACCESS_TOKEN")

    resp = requests.post(
        f"{GRAPH_API_BASE}/{threads_user_id}/threads_publish",
        data={"creation_id": container_id, "access_token": access_token},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def upload_video_to_threads(video_path: str, caption: str, source_url: str = None):
    """
    Uploads a video as a Threads post using the official Graph API.

    Workflow:
      1. Get public video URL (from GitHub Releases, or THREADS_VIDEO_PUBLIC_URL env var)
      2. POST /{threads-user-id}/threads → create container (media_type=VIDEO)
      3. GET /{container-id}?fields=status,error_message → poll until FINISHED
      4. POST /{threads-user-id}/threads_publish → publish the post
      5. (Optional) Post a reply with source link

    Args:
        video_path (str): Absolute path to the .mp4 video file.
        caption (str): Caption text for the Threads post.
        source_url (str): Optional source article URL to reply with.

    Returns:
        tuple: (bool success, str result_message_or_post_id)
    """
    if not _check_credentials():
        return False, "Skipped: Threads credentials not configured (THREADS_USER_ID, THREADS_ACCESS_TOKEN)"

    if not os.path.exists(video_path):
        return False, f"Error: Video file not found at {video_path}"

    # Log video specs for debugging
    try:
        import subprocess
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,duration,pix_fmt",
            "-of", "csv=p=0", video_path
        ], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            v_info = result.stdout.strip().split(",")
            print(f"📹 Video specs: codec={v_info[0]}, {v_info[1]}x{v_info[2]}, duration={v_info[3]}s, pix_fmt={v_info[4]}")
        
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate",
            "-of", "csv=p=0", video_path
        ], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            a_info = result.stdout.strip().split(",")
            print(f"🔊 Audio specs: codec={a_info[0]}, sample_rate={a_info[1]}")
        
        file_size = os.path.getsize(video_path)
        print(f"📦 File size: {file_size / (1024*1024):.1f} MB")
    except Exception as e:
        print(f"⚠️ Could not probe video: {e}")

    allowed, current_count = _check_rate_limit()
    if not allowed:
        return False, f"Skipped: Threads rate limit reached ({current_count}/{MAX_POSTS_PER_DAY} posts today)"

    token, expiry = _load_token()
    token = _refresh_token_if_needed(token, expiry)

    if token != os.getenv("THREADS_ACCESS_TOKEN"):
        os.environ["THREADS_ACCESS_TOKEN"] = token

    upload_method = None
    upload_key = None
    public_url = None

    try:
        print(f"📡 [Threads] Step 1/4: Getting public video URL...")

        direct_url = os.getenv("THREADS_VIDEO_PUBLIC_URL")
        if direct_url and direct_url.strip():
            public_url = direct_url.strip()
            upload_method = "direct"
            print(f"✅ Using provided public URL: {public_url}")

        if not public_url:
            public_url, gh_result = upload_video_to_github_releases(video_path)
            if public_url:
                upload_method = "github"
                upload_key = gh_result
                print(f"✅ Uploaded to GitHub Releases")
            else:
                return False, f"Failed to host video publicly: {gh_result}"

        print(f"📡 [Threads] Step 2/4: Creating Threads container...")
        container_id = create_threads_container(public_url, caption)
        print(f"✔ Container created: {container_id}")

        print(f"📡 [Threads] Step 3/4: Waiting for video processing...")
        wait_for_container(container_id)
        print(f"✔ Video processing finished")

        print(f"📡 [Threads] Step 4/4: Publishing Thread...")
        post_id = publish_container(container_id)
        print(f"🎉 Threads post published! ID: {post_id}")

        # Post reply with source link if provided
        if source_url:
            print(f"📡 [Threads] Posting source link reply...")
            reply_text = f"📰 Source: {source_url}"
            try:
                reply_id = create_threads_reply(post_id, reply_text)
                print(f"✅ Source reply posted! ID: {reply_id}")
            except Exception as e:
                print(f"⚠️ Failed to post source reply: {e}")

        _increment_rate_limit()

        return True, post_id

    except requests.HTTPError as e:
        response_text = getattr(e, 'response_text', None) or (e.response.text if e.response else "")
        return False, f"Threads API error (HTTP {e.response.status_code}): {response_text}"
    except Exception as e:
        return False, f"Threads upload exception: {e}"

    finally:
        if upload_method == "github" and upload_key:
            print(f"ℹ️ Keeping GitHub Release {upload_key} for async video processing (tagged for later cleanup)")
        # For "direct" method, no cleanup needed


# ── Standalone Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Threads Upload Module — Diagnostic Check")
    print("=" * 50)

    if _check_credentials():
        print("✔ Threads API credentials: Configured")
    else:
        print("⚠️ Threads API credentials: MISSING — set THREADS_USER_ID and THREADS_ACCESS_TOKEN in .env")
        missing = []
        if not os.getenv("THREADS_USER_ID"): missing.append("THREADS_USER_ID")
        if not os.getenv("THREADS_ACCESS_TOKEN"): missing.append("THREADS_ACCESS_TOKEN")
        print(f"   Missing keys: {', '.join(missing)}")

    print("\n📦 Video hosting options:")
    gh_token = os.getenv("GITHUB_TOKEN")
    gh_repo = os.getenv("GITHUB_REPOSITORY")
    if gh_token and gh_repo and "/" in gh_repo:
        print(f"  ✔ GitHub Releases: Available ({gh_repo})")
    else:
        print("  ⚠️ GitHub Releases: Not configured (optional)")
        print("     Set: GITHUB_TOKEN and GITHUB_REPOSITORY (e.g., 'owner/repo')")
        print("     Note: Repo must be PUBLIC for Threads to fetch the video")

    if os.getenv("THREADS_VIDEO_PUBLIC_URL"):
        print("  ✔ Direct public URL (THREADS_VIDEO_PUBLIC_URL): Provided")

    allowed, count = _check_rate_limit()
    print(f"\n✔ Rate limit: {count}/{MAX_POSTS_PER_DAY} posts today ({'OK' if allowed else 'LIMIT REACHED'})")

    token, expiry = _load_token()
    if token:
        print(f"✔ Access token: Present (expiry: {expiry if expiry else 'unknown'})")
    else:
        print("⚠️ Access token: Not found")

    print("\n" + "=" * 50)
    if _check_credentials():
        print("✅ Ready for Threads publishing!")
    else:
        print("❌ Not ready — complete the manual setup steps first.")
    print("=" * 50)