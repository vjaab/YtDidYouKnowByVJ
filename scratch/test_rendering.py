import os
import sys

# Adjust path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_gen import create_video

def main():
    print("Starting rendering validation test...")
    
    # 1. Use the existing vj.wav or mock audio path
    audio_path = "vj.wav"
    if not os.path.exists(audio_path):
        print("vj.wav not found! Please make sure you are running from the workspace root.")
        return

    # 2. Mock script_data
    script_data = {
        "title": "Test Video",
        "script": "Unlock your AI's hidden SUPERPOWERS! Just check this out.",
        "color_theme": {"background": "#121212", "accent": "#00E5FF", "text": "#ffffff"}, # Cyber Cyan
        "screenshot_path": "dummy_screenshot.png",
        "slot": "Slot A",
        "retention_map": {
            "pacing_speed": "fast"
        }
    }
    
    # Create a dummy image if not exists
    if not os.path.exists("dummy_screenshot.png"):
        from PIL import Image
        img = Image.new('RGB', (1080, 1920), color = (73, 109, 137))
        img.save('dummy_screenshot.png')
        print("Created dummy_screenshot.png")

    # 3. Mock chunks to verify morph transition
    # We will define 4 chunks so we check transitions between them.
    # Chunk index 0 to 1, 1 to 2, 2 to 3.
    # Let's force trans_type selection by running the loop multiple times or forcing choices.
    chunks = [
        {
            "chunk_id": 1,
            "start": 0.0,
            "end": 1.0,
            "duration": 1.0,
            "text": "Unlock your AI's",
            "visual_path": "dummy_screenshot.png",
            "visual_type": "photo"
        },
        {
            "chunk_id": 2,
            "start": 1.0,
            "end": 2.0,
            "duration": 1.0,
            "text": "hidden SUPERPOWERS!",
            "visual_path": "dummy_screenshot.png",
            "visual_type": "photo"
        },
        {
            "chunk_id": 3,
            "start": 2.0,
            "end": 3.0,
            "duration": 1.0,
            "text": "Just check this out.",
            "visual_path": "dummy_screenshot.png",
            "visual_type": "photo"
        }
    ]

    output_path = "output/test_rendered_video.mp4"
    os.makedirs("output", exist_ok=True)
    
    # Let's override random.choice inside video_gen for testing to ensure morph is hit!
    import random
    original_choice = random.choice
    
    def mock_choice(seq):
        # Always return morph for transition selection when it is present
        if "morph" in seq:
            print("Force selecting transition: morph")
            return "morph"
        return original_choice(seq)
        
    random.choice = mock_choice

    try:
        video_path = create_video(audio_path, script_data, chunks, output_path=output_path)
        print(f"Success! Video rendered and saved to: {video_path}")
    except Exception as e:
        print(f"Failed rendering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
