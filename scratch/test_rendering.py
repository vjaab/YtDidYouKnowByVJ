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
        "companies": [
            {"name": "WhatsApp", "local_logo_path": "assets/icons/whatsapp_logo.png", "description": "Messaging App"},
            {"name": "Telegram", "local_logo_path": "assets/icons/telegram_logo.png", "description": "Privacy Chat"}
        ],
        "people": [
            {"name": "VJ Profile", "local_image_path": "assets/vj_profile.jpg", "description": "Tech Creator"}
        ],
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

    chunks = [
        {
            "chunk_id": 1,
            "start": 0.0,
            "end": 2.9,
            "duration": 2.9,
            "text": "Unlock WhatsApp hidden tricks.",
            "visual_path": "dummy_screenshot.png",
            "visual_type": "photo",
            "words": [
                {"word": "Unlock", "start": 0.0, "end": 0.6},
                {"word": "WhatsApp", "start": 0.6, "end": 1.2},
                {"word": "hidden", "start": 1.2, "end": 1.8},
                {"word": "tricks.", "start": 1.8, "end": 2.9}
            ]
        },
        {
            "chunk_id": 2,
            "start": 2.9,
            "end": 5.0,
            "duration": 2.1,
            "text": "Telegram is also loaded with powers.",
            "visual_path": "dummy_screenshot.png",
            "visual_type": "photo",
            "is_setting_chunk": True,
            "words": [
                {"word": "Telegram", "start": 2.9, "end": 3.4},
                {"word": "is", "start": 3.4, "end": 3.6},
                {"word": "also", "start": 3.6, "end": 4.0},
                {"word": "loaded", "start": 4.0, "end": 4.3},
                {"word": "with", "start": 4.3, "end": 4.5},
                {"word": "powers.", "start": 4.5, "end": 5.0}
            ]
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
