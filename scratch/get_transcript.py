from youtube_transcript_api import YouTubeTranscriptApi

video_id = '8_vlWx1vUVk'
try:
    print(f"Fetching transcript for video: {video_id}...")
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    with open('scratch/transcript.txt', 'w', encoding='utf-8') as f:
        for entry in transcript:
            time_str = f"[{int(entry.start // 60)}:{int(entry.start % 60):02d}]"
            f.write(f"{time_str} {entry.text}\n")
    print("Successfully saved transcript to scratch/transcript.txt")
except Exception as e:
    print(f"Failed to fetch transcript: {e}")
