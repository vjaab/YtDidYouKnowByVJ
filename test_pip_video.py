import os
import sys

# Ensure yt_did_you_know_by_vj is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yt_did_you_know_by_vj'))

from video_gen import create_video

audio_path = "vj.wav"

if not os.path.exists(audio_path):
    print(f"Error: {audio_path} not found")
    sys.exit(1)

# Using a standard 10-second duration for test if actual duration isn't fetched
# You can check actual audio_duration using librosa or just make the test chunks sum up to 10.
import wave
import contextlib
with contextlib.closing(wave.open(audio_path,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    audio_duration = frames / float(rate)
    
print(f"Audio duration: {audio_duration}")

script_json = {
    "title": "Test Video for PiP Layout",
    "sub_category": "AI Magic",
    "color_theme": {"accent": "#00FF7F"},
    "original_news_headline": "Testing the new layout!",
    "breaking_news_level": 0,
}

chunk_1_dur = min(audio_duration / 2, 5.0)
chunk_2_dur = audio_duration - chunk_1_dur

chunks = [
    {
        "chunk_id": 1,
        "text": "This is the first test chunk that shows gemini image.",
        "start": 0.0,
        "end": chunk_1_dur,
        "duration": chunk_1_dur,
        "visual_path": "assets/gemini_img_without_logo.png",
        "visual_type": "photo",
        "words": [
            {"word": "This", "start": 0.0, "end": 0.5},
            {"word": "is", "start": 0.5, "end": 1.0},
            {"word": "first", "start": 1.0, "end": 2.5},
        ]
    },
    {
        "chunk_id": 2,
        "text": "This is the second chunk and it shows the youtube pic.",
        "start": chunk_1_dur,
        "end": audio_duration,
        "duration": chunk_2_dur,
        "visual_path": "assets/youtube_pic.png",
        "visual_type": "photo",
        "words": [
            {"word": "second", "start": chunk_1_dur, "end": chunk_1_dur+1.0},
            {"word": "chunk", "start": chunk_1_dur+1.0, "end": audio_duration},
        ]
    }
]

output_path = "output/test_pip_video.mp4"
print(f"Generating video to {output_path}...")

final_path = create_video(audio_path, script_json, chunks, output_path=output_path)
print(f"Result: {final_path}")
