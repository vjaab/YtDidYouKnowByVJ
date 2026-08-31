import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.http import MediaFileUpload
from config import YOUTUBE_CLIENT_SECRET_FILE

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # required for comments, playlists, end screens
]

# ── PLAYLIST MAPPING: Category/Series → Playlist ID ──
# Create these playlists in YouTube Studio first, then add their IDs here
# Format: "playlist_name": "PLAYLIST_ID"
PLAYLIST_MAP = {
    "AI & Tech Tools": os.getenv("PLAYLIST_AI_TOOLS", ""),
    "Tech Gadgets & Inventions": os.getenv("PLAYLIST_GADGETS", ""),
    "Finance & Tech Economy": os.getenv("PLAYLIST_FINANCE", ""),
    "Facts & Trivia": os.getenv("PLAYLIST_FACTS", ""),
    "Coding & Development Hacks": os.getenv("PLAYLIST_CODING", ""),
    "Quiz & Trivia": os.getenv("PLAYLIST_QUIZ", ""),
    "Interview Questions": os.getenv("PLAYLIST_INTERVIEW", ""),
    "Programming Language Origins": os.getenv("PLAYLIST_LANG_ORIGINS", ""),
    "Tech Company Founding Stories": os.getenv("PLAYLIST_FOUNDING", ""),
    "Famous Bugs & Glitches": os.getenv("PLAYLIST_BUGS", ""),
    "Agentic AI Facts": os.getenv("PLAYLIST_AGENTIC", ""),
    # Series-specific playlists
    "GitHub Gems": os.getenv("PLAYLIST_GITHUB_GEMS", ""),
    "Free AI Alternatives": os.getenv("PLAYLIST_AI_ALTS", ""),
    "Dev Productivity Hacks": os.getenv("PLAYLIST_DEV_HACKS", ""),
    "GitHub Repo You Should Know": os.getenv("PLAYLIST_GITHUB_REPO", ""),
    "AI Fact of the Day": os.getenv("PLAYLIST_AI_FACTS", ""),
    "Interview Question of the Day": os.getenv("PLAYLIST_INTERVIEW_Q", ""),
    "Famous Bugs & Glitches Series": os.getenv("PLAYLIST_BUG_HUNTER", ""),
    "Tech Founding Stories": os.getenv("PLAYLIST_FOUNDING_STORIES", ""),
}

def get_playlist_id_for_content(category, series_name=""):
    """Returns the appropriate playlist ID for the given category/series."""
    # Priority: series-specific playlist > category playlist
    if series_name and series_name in PLAYLIST_MAP and PLAYLIST_MAP[series_name]:
        return PLAYLIST_MAP[series_name]
    if category in PLAYLIST_MAP and PLAYLIST_MAP[category]:
        return PLAYLIST_MAP[category]
    return None

def add_video_to_playlist(youtube, video_id, playlist_id):
    """Adds a video to a playlist. Creates the playlist item."""
    if not playlist_id:
        return None
    try:
        request = youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video_id
                    }
                }
            }
        )
        response = request.execute()
        print(f"✅ Added video to playlist: {playlist_id}")
        return response
    except googleapiclient.errors.HttpError as e:
        if e.resp.status == 404:
            print(f"⚠️ Playlist not found: {playlist_id} (create it in YouTube Studio)")
        else:
            print(f"⚠️ Failed to add to playlist: {e}")
        return None

def add_end_screen(youtube, video_id, next_video_id=None, playlist_id=None, subscribe=True):
    """Adds end screen elements to the video."""
    if not next_video_id and not playlist_id and not subscribe:
        return None
    
    end_screen_elements = []
    
    # Subscribe button (always recommended)
    if subscribe:
        end_screen_elements.append({
            "type": "subscribe",
            "videoId": video_id,  # Not used for subscribe, but required
        })
    
    # Next video in series
    if next_video_id:
        end_screen_elements.append({
            "type": "video",
            "videoId": next_video_id,
            "recentUpload": False,
        })
    
    # Playlist link
    if playlist_id:
        end_screen_elements.append({
            "type": "playlist",
            "playlistId": playlist_id,
        })
    
    if not end_screen_elements:
        return None
    
    try:
        request = youtube.videos().update(
            part="endScreen",
            body={
                "id": video_id,
                "endScreen": {
                    "elements": end_screen_elements
                }
            }
        )
        response = request.execute()
        print(f"✅ End screen added to video: {video_id}")
        return response
    except googleapiclient.errors.HttpError as e:
        print(f"⚠️ End screen failed: {e}")
        return None

def get_latest_video_in_series(youtube, channel_id, series_keywords):
    """Finds the most recent video in the same series for end screen linking."""
    try:
        request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            maxResults=5,
            order="date",
            type="video",
            q=" ".join(series_keywords)
        )
        response = request.execute()
        for item in response.get("items", []):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            if any(kw.lower() in title.lower() for kw in series_keywords):
                return video_id
    except Exception as e:
        print(f"⚠️ Could not find previous video in series: {e}")
    return None

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


def upload_video(video_path, title, description, tags, thumbnail_path=None, category_id="22", comment_hook=None, is_longform=False, script_data=None):
    youtube = get_authenticated_service()
    if not youtube:
        return False, "Failed to authenticate with YouTube API"

    if not isinstance(tags, list):
        tags = []
    unique_tags = []
    for t in tags:
        cleaned_t = str(t).strip()
        if cleaned_t and cleaned_t not in unique_tags:
            unique_tags.append(cleaned_t)

    # ── YPP COMPLIANCE: Determine if mandatory AI disclosure is needed ──
    # YouTube requires disclosure for "realistic synthetic content" that could mislead viewers:
    # - Deepfakes of real people (face/body alteration)
    # - Synthetic voices of real people (celebrities, politicians, public figures)
    # - Fabricated events depicted as real
    # AI-assisted production (TTS narration, AI-generated visuals for concepts) does NOT require disclosure
    # unless it depicts a real person doing/saying something they didn't actually do/say.
    #
    # However, per YouTube's updated policy, we declare ALL videos from this pipeline as containing
    # synthetic media since they use AI-generated voice (ElevenLabs cloned voice), AI-generated visuals
    # (Pexels/whiteboard/infographics), and AI-written scripts.

    requires_ai_disclosure = True  # All videos from this pipeline use AI-generated content

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
            # AI Disclosure: Set to True only for realistic synthetic media of real people/events
            # Our pipeline uses AI narration + conceptual visuals = production assistance (no disclosure required)
            "containsSyntheticMedia": requires_ai_disclosure,
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

        # Step 3: Add to Playlist
        playlist_id = None
        series_name = ""
        if script_data:
            category = script_data.get("sub_category", "")
            series_name = script_data.get("series_name", "")
            playlist_id = get_playlist_id_for_content(category, series_name)
            if playlist_id:
                try:
                    add_video_to_playlist(youtube, video_id, playlist_id)
                except Exception as e:
                    print(f"Playlist add failed (non-fatal): {e}")

        # Step 4: Add End Screen (link to next video in series + playlist)
        try:
            channel_id = get_channel_id(youtube)
            next_video_id = None
            if series_name:
                series_keywords = [series_name.lower().replace(" ", ""), "vj", "tech"]
                next_video_id = get_latest_video_in_series(youtube, channel_id, series_keywords)
            if next_video_id or playlist_id:
                add_end_screen(youtube, video_id, next_video_id=next_video_id, playlist_id=playlist_id)
        except Exception as e:
            print(f"End screen failed (non-fatal): {e}")

        # Step 5: Post + pin comment with playlist link (rotated template for YPP compliance)
        try:
            pinned_text = _get_pinned_comment(title)
            playlist_link = f"\n\n📺 Full playlist: https://youtube.com/playlist?list={playlist_id}" if playlist_id else ""
            full_comment = f"{title}\n\n{comment_hook}\n\n{pinned_text}{playlist_link}" if comment_hook else f"{pinned_text}{playlist_link}"
            post_and_pin_comment(youtube, video_id, full_comment)
        except Exception as e:
            print(f"Pinned comment failed (non-fatal): {e}")

        return True, video_id
    except googleapiclient.errors.HttpError as e:
        print(f"YouTube upload error {e.resp.status}: {e.content}")
        return False, str(e)


def get_channel_id(youtube):
    """Gets the authenticated user's channel ID."""
    try:
        request = youtube.channels().list(part="id", mine=True)
        response = request.execute()
        if response.get("items"):
            return response["items"][0]["id"]
    except Exception as e:
        print(f"⚠️ Could not get channel ID: {e}")
    return None


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

