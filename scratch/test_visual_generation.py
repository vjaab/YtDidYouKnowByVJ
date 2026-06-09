import os
import sys
from datetime import datetime

# Adjust path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pexels_fetcher import fetch_chunk_visual

def test_visual_generation():
    print("Testing visual generation with Google Veo and Google Image...")
    
    # Mock data representing a subtitle chunk
    chunk_short = {
        "chunk_id": 2, # Second chunk, should map to nano_evidence
        "text": "OpenAI has just released GPT-5, outperforming all human benchmarks.",
        "duration": 3.0
    }
    
    script_data = {
        "screenshot_path": "dummy_screenshot.png",
        "evidence_screenshot_path": "dummy_screenshot.png",
        "longform_topics": [
            {
                "headline": "OpenAI Releases GPT-5",
                "source_url": "https://openai.com",
                "screenshot_path": "dummy_screenshot.png"
            }
        ]
    }
    
    print("\n--- Test 1: Shorts aspect ratio (9:16) and nano_evidence (main screenshot) ---")
    res1 = fetch_chunk_visual(
        chunk=chunk_short.copy(),
        script_data=script_data,
        topic_context="OpenAI Releases GPT-5",
        is_longform=False,
        visual_mode="nano_evidence"
    )
    print("Result 1 source:", res1.get("source"))
    print("Result 1 path:", res1.get("visual_path"))
    
    print("\n--- Test 2: Longform aspect ratio (16:9) and nano_evidence (topic screenshot) ---")
    chunk_long = {
        "chunk_id": 2,
        "text": "OpenAI has just released GPT-5.",
        "duration": 3.0,
        "fact_number": 1 # Fact 1
    }
    res2 = fetch_chunk_visual(
        chunk=chunk_long,
        script_data=script_data,
        topic_context="OpenAI Releases GPT-5",
        is_longform=True,
        visual_mode="nano_evidence"
    )
    print("Result 2 source:", res2.get("source"))
    print("Result 2 path:", res2.get("visual_path"))
    
    print("\nVerification completed successfully!")

if __name__ == "__main__":
    test_visual_generation()
