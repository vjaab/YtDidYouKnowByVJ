import json
import os
from video_gen import create_video

SCRIPT = {"title": "Test Title", "body_text": "First chunk words sentence. Then something else passes time", "companies": ["Fake"]}
CHUNKS = [
    {
        "start": 0.0,
        "end": 2.0,
        "duration": 2.0,
        "words": [
            {"word": "Hello", "start": 0.0, "end": 1.0},
            {"word": "World", "start": 1.0, "end": 2.0}
        ]
    }
]

import sys
print("starting render test")
output=create_video("vj.wav", SCRIPT, CHUNKS, "output/minimal_test.mp4")
print("Render done:", output)
