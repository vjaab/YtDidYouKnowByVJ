import sys
import json
from youtube_transcript_api import YouTubeTranscriptApi
import urllib.request
import re

video_id = "-PHCrjCPte8"
try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([t['text'] for t in transcript])
    print("--- TRANSCRIPT ---")
    print(text[:2000] + "..." if len(text) > 2000 else text)
except Exception as e:
    print(f"Error fetching transcript: {e}")

try:
    url = f"https://www.youtube.com/watch?v={video_id}"
    html = urllib.request.urlopen(url).read().decode('utf-8')
    title_match = re.search(r'<title>(.*?)</title>', html)
    if title_match:
        print("\n--- TITLE ---")
        print(title_match.group(1).replace(" - YouTube", ""))
except Exception as e:
    print(f"Error fetching title: {e}")
