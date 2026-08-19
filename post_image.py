#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_image.py — Generate and post images to Instagram (carousel/single), Facebook, and Telegram.

Uses Meta Graph API for Instagram/Facebook image posting and Telegram Bot API for photo sending.
"""

import os
import sys
import json
import time
import random
import argparse
import hashlib
import requests
from datetime import datetime, timedelta
from pathlib import Path

from config import BASE_DIR, OUTPUT_DIR, GEMINI_API_KEY

# Import image generation
from image_gen import generate_images

# ── Constants ────────────────────────────────────────────────────────────────
GRAPH_API_VERSION = "v25.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

# Token persistence
IG_TOKEN_FILE = os.path.join(BASE_DIR, ".ig_token.json")

# Rate limit tracking
IG_RATE_LIMIT_FILE = os.path.join(BASE_DIR, ".ig_rate_limit.json")
MAX_POSTS_PER_DAY = 25

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else ""


# ── Helper Functions ──────────────────────────────────────────────────────────

def _check_ig_credentials():
    ig_user_id = os.getenv("IG_USER_ID")
    ig_access_token = os.getenv("IG_ACCESS_TOKEN")
    return all(k and k.strip() for k in [ig_user_id, ig_access_token])


def _check_fb_credentials():
    fb_page_id = os.getenv("FB_PAGE_ID")
    fb_page_access_token = os.getenv("FB_PAGE_ACCESS_TOKEN")
    return all(k and k.strip() for k in [fb_page_id, fb_page_access_token])


def _load_ig_token():
    if os.path.exists(IG_TOKEN_FILE):
        try:
            with open(IG_TOKEN_FILE, "r") as f:
                data = json.load(f)
                token = data.get("access_token", "")
                expiry = data.get("expires_at", "")
                if token:
                    return token, expiry
        except Exception:
            pass
    return os.getenv("IG_ACCESS_TOKEN", ""), ""


def _save_ig_token(token, expires_in_seconds):
    expires_at = (datetime.now() + timedelta(seconds=int(expires_in_seconds))).isoformat()
    try:
        with open(IG_TOKEN_FILE, "w") as f:
            json.dump({"access_token": token, "expires_at": expires_at}, f)
        print(f"✅ Instagram token saved. Expires: {expires_at}")
    except Exception as e:
        print(f"⚠️ Failed to persist Instagram token: {e}")


def _refresh_ig_token_if_needed(current_token, expiry_str):
    from datetime import timedelta
    if expiry_str:
        try:
            expiry = datetime.fromisoformat(expiry_str)
            days_left = (expiry - datetime.now()).days
            if days_left > 7:
                return current_token
            print(f"⚠️ Instagram token expires in {days_left} days. Attempting refresh...")
        except Exception:
            print("⚠️ Could not parse token expiry. Attempting refresh as precaution...")
    else:
        print("ℹ️ No token expiry info found. Attempting refresh...")

    try:
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
            expires_in = data.get("expires_in", 5184000)
            if new_token:
                _save_ig_token(new_token, expires_in)
                print("✅ Instagram token refreshed successfully.")
                return new_token

        print(f"⚠️ Token refresh failed (HTTP {resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"⚠️ Token refresh exception (non-fatal): {e}")

    return current_token


def _check_ig_rate_limit():
    today = datetime.now().strftime("%Y-%m-%d")
    data = {}

    if os.path.exists(IG_RATE_LIMIT_FILE):
        try:
            with open(IG_RATE_LIMIT_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

    if data.get("date") != today:
        data = {"date": today, "count": 0}

    if data["count"] >= MAX_POSTS_PER_DAY:
        return False, data["count"]

    return True, data["count"]


def _increment_ig_rate_limit():
    today = datetime.now().strftime("%Y-%m-%d")
    data = {"date": today, "count": 0}

    if os.path.exists(IG_RATE_LIMIT_FILE):
        try:
            with open(IG_RATE_LIMIT_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            pass

    if data.get("date") != today:
        data = {"date": today, "count": 0}

    data["count"] = data.get("count", 0) + 1

    try:
        with open(IG_RATE_LIMIT_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def upload_image_to_github_releases(image_path):
    """Uploads an image file to GitHub Releases and returns the public URL."""
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")

    if not token or not repo:
        return None, "GitHub credentials not configured"

    if "/" not in repo:
        return None, f"Invalid GITHUB_REPOSITORY format: {repo}"

    file_size = os.path.getsize(image_path)
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

        tag = f"image-temp-{int(time.time())}"
        filename = os.path.basename(image_path)

        release_resp = requests.post(
            f"https://api.github.com/repos/{owner}/{repo_name}/releases",
            headers=headers,
            json={
                "tag_name": tag,
                "name": f"Temp image {tag}",
                "body": "Auto-generated for Instagram/Facebook image publish. Safe to delete after 24h.",
                "prerelease": False,
            },
            timeout=30,
        )
        release_resp.raise_for_status()
        release_data = release_resp.json()
        release_id = release_data["id"]
        upload_url = release_data["upload_url"].split("{")[0]

        with open(image_path, "rb") as f:
            asset_resp = requests.post(
                upload_url,
                headers={**headers, "Content-Type": "image/jpeg"},
                params={"name": filename},
                data=f,
                timeout=60,
            )
        asset_resp.raise_for_status()
        public_url = asset_resp.json()["browser_download_url"]

        print(f"⏳ Waiting for GitHub CDN propagation...")
        time.sleep(5)

        try:
            verify_resp = requests.head(public_url, allow_redirects=True, timeout=15)
            if verify_resp.status_code == 200:
                print(f"✅ URL verified: {verify_resp.headers.get('content-type')}, {verify_resp.headers.get('content-length')} bytes")
            else:
                print(f"⚠️ URL verification returned {verify_resp.status_code}")
        except Exception as e:
            print(f"⚠️ URL verification failed: {e}")

        print(f"✅ Uploaded to GitHub Releases: {public_url}")
        return public_url, str(release_id)

    except Exception as e:
        return None, f"GitHub Releases upload exception: {e}"


def delete_github_release(release_id):
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


# ── Instagram Image/Carousel Posting ──────────────────────────────────────────

def create_ig_image_container(image_url: str, caption: str, is_carousel_item: bool = False) -> str:
    """Create an Instagram image container (or carousel item)."""
    ig_user_id = os.getenv("IG_USER_ID")
    access_token = os.getenv("IG_ACCESS_TOKEN")

    media_type = "IMAGE"
    if is_carousel_item:
        media_type = "IMAGE"  # Carousel items are IMAGE type

    resp = requests.post(
        f"{GRAPH_API_BASE}/{ig_user_id}/media",
        data={
            "media_type": media_type,
            "image_url": image_url,
            "caption": caption[:2200],
            "access_token": access_token,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def create_ig_carousel_container(caption: str, children_ids: list) -> str:
    """Create a carousel container from child media IDs."""
    ig_user_id = os.getenv("IG_USER_ID")
    access_token = os.getenv("IG_ACCESS_TOKEN")

    resp = requests.post(
        f"{GRAPH_API_BASE}/{ig_user_id}/media",
        data={
            "media_type": "CAROUSEL",
            "caption": caption[:2200],
            "children": ",".join(children_ids),
            "access_token": access_token,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def wait_for_ig_container(container_id: str, timeout_s: int = 120, poll_every_s: int = 3) -> None:
    """Poll until Instagram finishes processing the container."""
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


def publish_ig_container(container_id: str) -> str:
    """Publish the finished Instagram container."""
    ig_user_id = os.getenv("IG_USER_ID")
    access_token = os.getenv("IG_ACCESS_TOKEN")

    resp = requests.post(
        f"{GRAPH_API_BASE}/{ig_user_id}/media_publish",
        data={"creation_id": container_id, "access_token": access_token},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def post_image_to_instagram(image_paths: list, caption: str, as_carousel: bool = True):
    """
    Posts image(s) to Instagram as carousel (multiple) or single image.
    
    Args:
        image_paths: List of local image file paths
        caption: Caption text
        as_carousel: If True and multiple images, post as carousel
    """
    if not _check_ig_credentials():
        return False, "Skipped: Instagram credentials not configured"

    if not image_paths:
        return False, "Error: No images provided"

    allowed, current_count = _check_ig_rate_limit()
    if not allowed:
        return False, f"Skipped: Instagram rate limit reached ({current_count}/{MAX_POSTS_PER_DAY} posts today)"

    token, expiry = _load_ig_token()
    token = _refresh_ig_token_if_needed(token, expiry)
    if token != os.getenv("IG_ACCESS_TOKEN"):
        os.environ["IG_ACCESS_TOKEN"] = token

    upload_keys = []
    public_urls = []

    try:
        print(f"📡 [Instagram] Uploading {len(image_paths)} image(s)...")
        
        for i, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                return False, f"Error: Image file not found at {img_path}"
            
            direct_url = os.getenv("IG_VIDEO_PUBLIC_URL")
            if direct_url and direct_url.strip() and i == 0:
                public_url = direct_url.strip()
                print(f"✅ Using provided public URL for image {i+1}")
            else:
                public_url, gh_result = upload_image_to_github_releases(img_path)
                if public_url:
                    upload_keys.append(gh_result)
                    print(f"✅ Uploaded image {i+1} to GitHub Releases")
                else:
                    return False, f"Failed to host image {i+1}: {gh_result}"
            public_urls.append(public_url)

        if as_carousel and len(public_urls) > 1:
            print(f"📡 [Instagram] Creating carousel with {len(public_urls)} images...")
            child_ids = []
            for j, url in enumerate(public_urls):
                print(f"  Creating carousel item {j+1}/{len(public_urls)}...")
                child_id = create_ig_image_container(url, caption, is_carousel_item=True)
                wait_for_ig_container(child_id, timeout_s=60)
                child_ids.append(child_id)
                print(f"  ✔ Item {j+1} ready: {child_id}")

            container_id = create_ig_carousel_container(caption, child_ids)
            print(f"✔ Carousel container created: {container_id}")

            wait_for_ig_container(container_id)
            print(f"✔ Carousel processing finished")

            post_id = publish_ig_container(container_id)
            print(f"🎉 Instagram Carousel published! ID: {post_id}")
        else:
            # Single image
            print(f"📡 [Instagram] Creating single image post...")
            container_id = create_ig_image_container(public_urls[0], caption)
            print(f"✔ Container created: {container_id}")

            wait_for_ig_container(container_id)
            print(f"✔ Image processing finished")

            post_id = publish_ig_container(container_id)
            print(f"🎉 Instagram Image published! ID: {post_id}")

        if _check_fb_credentials():
            print(f"📡 [Facebook] Cross-posting image to Facebook Page...")
            fb_post_id, fb_error = post_image_to_facebook_page(public_urls[0], caption)
            if fb_post_id:
                print(f"🎉 Facebook Image published! ID: {fb_post_id}")
            else:
                print(f"⚠️ Facebook cross-post skipped: {fb_error}")
        else:
            print("⚠️ Facebook Page credentials not configured — skipping cross-post")

        _increment_ig_rate_limit()
        return True, post_id

    except requests.HTTPError as e:
        response_text = getattr(e, 'response_text', None) or (e.response.text if e.response else "")
        return False, f"Instagram API error (HTTP {e.response.status_code}): {response_text}"
    except Exception as e:
        return False, f"Instagram upload exception: {e}"
    finally:
        for key in upload_keys:
            print(f"ℹ️ Keeping GitHub Release {key} for async processing (tagged for later cleanup)")


# ── Facebook Image Posting ────────────────────────────────────────────────────

def post_image_to_facebook_page(image_url: str, caption: str) -> tuple:
    """Post a single image to Facebook Page via Graph API."""
    fb_page_id = os.getenv("FB_PAGE_ID")
    fb_page_access_token = os.getenv("FB_PAGE_ACCESS_TOKEN")

    if not fb_page_id or not fb_page_access_token:
        return None, "Facebook Page credentials not configured"

    try:
        print("📡 [Facebook] Posting image to Page...")
        payload = {
            "url": image_url,
            "caption": caption[:2200],
            "published": "true",
            "access_token": fb_page_access_token,
        }

        def _post():
            resp = requests.post(
                f"{GRAPH_API_BASE}/{fb_page_id}/photos",
                data=payload,
                timeout=60,
            )
            response_text = resp.text
            print(f"📡 [Facebook] Response: {resp.status_code} - {response_text}")
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                e.response_text = response_text
                raise
            return resp.json()

        result = _post()
        post_id = result.get("id") or result.get("post_id")
        print(f"🎉 Facebook Image published! ID: {post_id}")
        return post_id, None

    except requests.HTTPError as e:
        response_text = getattr(e, 'response_text', None) or (e.response.text if e.response else "")
        return None, f"Facebook API error (HTTP {e.response.status_code}): {response_text}"
    except Exception as e:
        return None, f"Facebook upload exception: {e}"


# ── Telegram Image Sending ────────────────────────────────────────────────────

def send_image_to_telegram(image_path: str, caption: str = "") -> bool:
    """Send an image to Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured — skipping")
        return False

    try:
        with open(image_path, "rb") as f:
            files = {"photo": f}
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption[:1024],  # Telegram caption limit
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
        requests.post(
            f"{TELEGRAM_BASE_URL}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": f"{emoji} {message}", "parse_mode": "HTML"},
            timeout=15
        )
    except Exception as e:
        print(f"Telegram notify failed: {e}")


# ── Topic/Content Selection ──────────────────────────────────────────────────

def fetch_latest_news_for_image_post():
    """Fetch latest tech news from news_log.json for image post content."""
    news_log_path = os.path.join(BASE_DIR, "news_log.json")
    if not os.path.exists(news_log_path):
        return []

    try:
        with open(news_log_path, "r") as f:
            data = json.load(f)
        articles = data.get("articles", [])
        return articles[:5]  # Top 5
    except Exception as e:
        print(f"Failed to load news log: {e}")
        return []


def generate_image_prompts_from_news(articles, count=4):
    """Generate image generation prompts from news articles."""
    if not articles:
        return [
            "Futuristic AI technology concept, neural networks visualization, digital art",
            "Cutting-edge machine learning research, data visualization, sci-fi aesthetic",
            "Advanced robotics and automation, modern tech laboratory, cinematic lighting",
            "Quantum computing breakthrough, abstract technology art, photorealistic"
        ]

    prompts = []
    for art in articles[:count]:
        title = art.get("title", "Tech News")
        desc = art.get("description", "")[:200]
        prompt = f"Tech news illustration: {title}. {desc}. Futuristic digital art, cinematic lighting, photorealistic, no text"
        prompts.append(prompt)

    while len(prompts) < count:
        prompts.append("Technology innovation concept, AI and future tech, digital art")

    return prompts[:count]


def build_image_caption(articles):
    """Build caption for the image post from news articles."""
    if not articles:
        return "🤖 Daily Tech Insights\n\n#AI #TechNews #Innovation #MachineLearning #FutureTech"

    main_article = articles[0]
    title = main_article.get("title", "Tech Update")[:80]
    source = main_article.get("source", {}).get("name", "Tech News")

    hashtags = ["#AI", "#TechNews", "#Innovation", "#MachineLearning", "#FutureTech", "#Technology"]
    
    caption = f"📰 {title}\n\n📍 Source: {source}\n\n"
    caption += " ".join(hashtags)
    
    return caption


# ── Main Pipeline ────────────────────────────────────────────────────────────

def run_image_post_pipeline(dry_run=False):
    """Main pipeline for generating and posting images."""
    print("=" * 60)
    print("🖼️  IMAGE POST PIPELINE STARTED")
    print("=" * 60)

    if dry_run:
        print("🧪 DRY RUN MODE - No actual posting")

    # 1. Fetch news/topics
    print("\n📰 Fetching latest tech news...")
    articles = fetch_latest_news_for_image_post()
    if articles:
        print(f"  Found {len(articles)} articles")
        for i, art in enumerate(articles[:3], 1):
            print(f"  {i}. {art.get('title', 'Untitled')[:60]}")
    else:
        print("  No articles found, using fallback prompts")

    # 2. Generate image prompts
    print("\n🎨 Generating image prompts...")
    prompts = generate_image_prompts_from_news(articles, count=4)
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. {p[:80]}...")

    # 3. Generate images
    print("\n🖼️  Generating images...")
    keywords = []
    if articles:
        for art in articles[:3]:
            title = art.get("title", "")
            keywords.extend(title.split()[:3])

    image_paths = generate_images(
        prompts=prompts,
        image_url=None,
        keywords=keywords[:6],
        aspect_ratio="9:16"  # Instagram portrait format
    )

    if not image_paths:
        print("❌ Image generation failed!")
        return False, "Image generation failed"

    print(f"✅ Generated {len(image_paths)} images:")
    for p in image_paths:
        print(f"  - {p}")

    if dry_run:
        print("\n🧪 DRY RUN: Would post the following:")
        print(f"  Images: {len(image_paths)}")
        caption = build_image_caption(articles)
        print(f"  Caption: {caption[:100]}...")
        return True, "DRY_RUN_SUCCESS"

    # 4. Build caption
    caption = build_image_caption(articles)

    # 5. Post to Instagram (DISABLED - only Telegram)
    # print("\n📸 Posting to Instagram...")
    # as_carousel = len(image_paths) > 1
    # ig_success, ig_result = post_image_to_instagram(image_paths, caption, as_carousel=as_carousel)
    # if ig_success:
    #     print(f"✅ Instagram: {ig_result}")
    # else:
    #     print(f"❌ Instagram failed: {ig_result}")

    # 6. Send to Telegram (send first image or all)
    print("\n📱 Sending to Telegram...")
    if image_paths:
        tg_caption = f"🖼️ Daily Tech Visual\n\n{caption[:800]}"
        send_image_to_telegram(image_paths[0], tg_caption)
        # Send additional images as separate messages
        for img_path in image_paths[1:3]:  # Send up to 3 total
            send_image_to_telegram(img_path, "📸 More from today's tech visual")

    # 7. Notify Telegram
    send_telegram_message(
        f"✅ Image Post Pipeline Complete\n"
        f"Images: {len(image_paths)}\n"
        f"Telegram: Sent",
        emoji="🤖"
    )

    print("\n" + "=" * 60)
    print("✅ IMAGE POST PIPELINE COMPLETED")
    print("=" * 60)
    
    return ig_success, ig_result if not ig_success else "SUCCESS"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and post images to Instagram, Facebook, Telegram")
    parser.add_argument("--now", action="store_true", help="Run immediately")
    parser.add_argument("--dry-run", action="store_true", help="Preview without posting")
    args = parser.parse_args()

    if not args.now and not args.dry_run:
        print("Usage: python post_image.py --now       # Run and post")
        print("       python post_image.py --dry-run   # Preview only")
        sys.exit(1)

    success, result = run_image_post_pipeline(dry_run=args.dry_run)
    if not success:
        print(f"Pipeline failed: {result}")
        sys.exit(1)
    else:
        print(f"Pipeline succeeded: {result}")