import os
import json
from video_gen import create_video

def test_minimalist():
    # Mock script data
    script_json = {
        "title": "Minimalist Style Test",
        "sub_category": "AI Technology",
        "color_theme": {"accent": "#00E5FF"},
        "hook_banner_text": "THIS IS A MINIMALIST TEST",
        "key_stat": "0 Rs COST",
        "key_stat_timestamp": 2.0,
        "shocking_moment_timestamp": 4.0,
        "subtitle_chunks": [
            {"text": "Welcome to the new video format.", "start": 0.0, "end": 2.0},
            {"text": "This is 100 percent code driven.", "start": 2.1, "end": 4.0},
            {"text": "And optimized for YouTube monetization.", "start": 4.1, "end": 6.0}
        ]
    }
    
    # Mock chunks with word-level detail
    chunks = [
        {
            "start": 0.0, "end": 2.0, "duration": 2.0,
            "words": [
                {"word": "Welcome", "start": 0.0, "end": 0.5},
                {"word": "to", "start": 0.6, "end": 0.8},
                {"word": "the", "start": 0.9, "end": 1.1},
                {"word": "new", "start": 1.2, "end": 1.4},
                {"word": "video", "start": 1.5, "end": 1.7},
                {"word": "format.", "start": 1.8, "end": 2.0}
            ]
        },
        {
            "start": 2.1, "end": 4.0, "duration": 1.9,
            "words": [
                {"word": "This", "start": 2.1, "end": 2.3},
                {"word": "is", "start": 2.4, "end": 2.6},
                {"word": "100", "start": 2.7, "end": 3.0},
                {"word": "percent", "start": 3.1, "end": 3.4},
                {"word": "code", "start": 3.5, "end": 3.7},
                {"word": "driven.", "start": 3.8, "end": 4.0}
            ]
        },
        {
            "start": 4.1, "end": 6.0, "duration": 1.9,
            "words": [
                {"word": "And", "start": 4.1, "end": 4.3},
                {"word": "optimized", "start": 4.4, "end": 4.8},
                {"word": "for", "start": 4.9, "end": 5.1},
                {"word": "YouTube", "start": 5.2, "end": 5.5},
                {"word": "monetization.", "start": 5.6, "end": 6.0}
            ]
        }
    ]
    
    # Needs a real audio file for MoviePy, I'll use a dummy segment or an existing one if possible.
    # Actually, I'll just use a silent 6s audio clip.
    from pydub import AudioSegment
    silent = AudioSegment.silent(duration=6000)
    silent.export("output/test_audio.mp3", format="mp3")
    
    os.makedirs("output", exist_ok=True)
    video_path = create_video("output/test_audio.mp3", script_json, chunks, "output/minimalist_test.mp4")
    print(f"Test video created at: {video_path}")

if __name__ == "__main__":
    test_minimalist()
