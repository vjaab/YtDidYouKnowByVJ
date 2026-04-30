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

PINNED_COMMENT_TEXT = """💡 Every day you're not learning, someone else is getting ahead.

I share what top devs & AI engineers are reading right now:

🚀 Hottest AI research (before it goes viral)
💼 High-paying jobs & hiring alerts
🛠️ Dev tools & resources that save hours
📰 Tech news that actually matters

Join early 👇
🚀 Telegram → https://t.me/technewsbyvj
💬 WhatsApp → https://whatsapp.com/channel/0029Vb75sw08vd1GsBm3RD1Z
🔗 (Links in Header!)"""


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

    body = {
        "snippet": {
            "title":                title[:100],
            "description":          description[:5000],
            "tags":                 tags[:15],
            "categoryId":           category_id,
            "defaultLanguage":      "en",
            "defaultAudioLanguage": "en",
        },
        "status": {
            "privacyStatus":          "public",
            "selfDeclaredMadeForKids": False,
            # CRITICAL: Since we use F5-TTS (cloned voice) and AI visuals, 
            # we MUST disclose altered/synthetic content to remain eligible for monetization.
            "selfDeclaredAlteredContent": True,
        },
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

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

        # Step 3: Post + pin comment immediately
        try:
            full_comment = f"{title}\n\n{comment_hook}\n\n{PINNED_COMMENT_TEXT}" if comment_hook else PINNED_COMMENT_TEXT
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

