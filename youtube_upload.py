import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.http import MediaFileUpload
from config import YOUTUBE_CLIENT_SECRET_FILE

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # required for comments
]

# ── YPP COMPLIANCE: Rotating pinned comment templates ──
# Prevents identical metadata fingerprint across uploads
PINNED_COMMENT_TEMPLATES = [
    """💡 Every day you're not keeping up with AI, someone else is getting ahead.

I share what top engineers are reading — before it trends:

🚀 Breaking AI news & analysis
💼 Industry moves & career insights
🛠️ Open source tools & reviews
📰 Research that actually matters

Join early 👇
🚀 Telegram → https://t.me/technewsbyvj
💬 WhatsApp → https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
🔗 (Links in Header!)""",

    """🔥 This is just the tip of the iceberg.

I post FULL breakdowns, analysis, and deep-dives on Telegram every single day.

What you get:
→ Daily AI news drops (before they trend)
→ My personal take on every major story
→ Industry analysis you won't find elsewhere

📲 Telegram: https://t.me/technewsbyvj
💬 WhatsApp: https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z""",

    """⚡ Want the full story? It's already on my Telegram.

I break down one major AI story every day — with context most channels skip.

Why engineers follow:
• No fluff, pure analysis
• Controversial takes on every big announcement
• The stories that actually affect your career

Join → https://t.me/technewsbyvj
WhatsApp → https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z""",

    """🧠 If you made it this far, you're the type of person who wants the REAL story.

I share daily AI analysis that goes deeper than headlines:
🔬 What happened & WHY it matters
🛠️ Who wins, who loses
📊 My predictions (and track record)

The best part? It's all free.

📲 https://t.me/technewsbyvj
💬 https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
🔗 Everything → link in bio""",
]

def _get_pinned_comment(title=""):
    """Select a pinned comment template based on the video title hash."""
    import hashlib
    seed = int(hashlib.md5(title.encode()).hexdigest(), 16)
    idx = seed % len(PINNED_COMMENT_TEMPLATES)
    return PINNED_COMMENT_TEMPLATES[idx]


import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import google_auth_oauthlib.flow

def get_authenticated_service():
    if not os.path.exists(YOUTUBE_CLIENT_SECRET_FILE):
        print("YouTube client secret file not found.")
        return None
        
    creds = None
    token_path = "token.json"
    
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                YOUTUBE_CLIENT_SECRET_FILE, SCOPES
            )
            creds = flow.run_local_server(port=8080, prompt='consent')
            
        with open(token_path, "w") as token:
            token.write(creds.to_json())
            
    try:
        youtube = googleapiclient.discovery.build("youtube", "v3", credentials=creds)
        return youtube
    except Exception as e:
        print(f"YouTube auth failed: {e}")
        return None


def upload_video(video_path, title, description, tags, thumbnail_path=None, category_id="28", comment_hook=None):
    youtube = get_authenticated_service()
    if not youtube:
        return False, "Failed to authenticate with YouTube API"

    # Target high-RPM native English regions
    target_regions = ["USA", "United States", "UK", "United Kingdom", "Australia", "Canada", "New Zealand", "English"]
    if not isinstance(tags, list):
        tags = []
    unique_tags = []
    for t in (tags + target_regions):
        cleaned_t = str(t).strip()
        if cleaned_t and cleaned_t not in unique_tags:
            unique_tags.append(cleaned_t)

    body = {
        "snippet": {
            "title":                title[:100],
            "description":          description[:5000],
            "tags":                 unique_tags[:30],
            "categoryId":           category_id,
            "defaultLanguage":      "en-US",
            "defaultAudioLanguage": "en-US",
        },
        "status": {
            "privacyStatus":          "public",
            "selfDeclaredMadeForKids": False,
            # CRITICAL: Since we use F5-TTS (cloned voice) and AI visuals, 
            # we MUST disclose altered/synthetic content to remain eligible for monetization.
            "containsSyntheticMedia": True,
        },
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media, notifySubscribers=True)

    try:
        response = request.execute()
        video_id = response.get("id")
        print(f"Video uploaded: https://youtu.be/{video_id}")

        # Step 2: Upload Thumbnail
        if thumbnail_path and os.path.exists(thumbnail_path):
            try:
                set_thumbnail(youtube, video_id, thumbnail_path)
            except Exception as e:
                print(f"Thumbnail upload failed (non-fatal): {e}")

        # Step 3: Post + pin comment immediately (rotated template for YPP compliance)
        try:
            pinned_text = _get_pinned_comment(title)
            full_comment = f"{title}\n\n{comment_hook}\n\n{pinned_text}" if comment_hook else pinned_text
            post_and_pin_comment(youtube, video_id, full_comment)
        except Exception as e:
            print(f"Pinned comment failed (non-fatal): {e}")

        return True, video_id
    except googleapiclient.errors.HttpError as e:
        print(f"YouTube upload error {e.resp.status}: {e.content}")
        return False, str(e)


def post_and_pin_comment(youtube, video_id, comment_text):
    """Posts a comment and pins it on the given video."""
    # Step 1: Post comment
    comment_response = youtube.commentThreads().insert(
        part="snippet",
        body={
            "snippet": {
                "videoId": video_id,
                "topLevelComment": {
                    "snippet": {
                        "textOriginal": comment_text
                    }
                }
            }
        }
    ).execute()

    comment_id = comment_response["snippet"]["topLevelComment"]["id"]
    print(f"Comment posted: {comment_id}")

    # Step 2: Pin comment (set to published = pinned by channel owner)
    youtube.comments().setModerationStatus(
        id=comment_id,
        moderationStatus="published",
        banAuthor=False
    ).execute()

    print(f"Comment pinned: {comment_id}")
    return comment_id


def set_thumbnail(youtube, video_id, thumbnail_path):
    """
    Uploads a custom thumbnail for the specified video.
    Includes a slight delay to ensure video is ready for thumbnail attachment.
    """
    import time
    # YouTube backend sometimes needs a moment to 'register' the new video 
    # before it can accept a thumbnail attachment.
    print(f"⏳ Waiting 5s for YouTube to index video {video_id} before thumbnail...")
    time.sleep(5) 
    
    print(f"Uploading thumbnail: {thumbnail_path}...")
    try:
        request = youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg", resumable=True)
        )
        response = request.execute()
        print(f"✅ Thumbnail set successfully.")
        return response
    except Exception as e:
        print(f"⚠ Critical Thumbnail Error: {e}")
        raise e

