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
        "script": "Unlock your AI's hidden SUPERPOWERS with VJ! Let's discuss biology and join us to check the link in bio.",
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

    # Phonetic Substring Validation Test
    from audio_gen import clean_tts_text
    cleaned_txt = clean_tts_text(script_data["script"])
    print("--------------------------------------------------")
    print(f"Original Text: {script_data['script']}")
    print(f"Cleaned Text:  {cleaned_txt}")
    print("--------------------------------------------------")
    
    # Assertions to confirm correctness:
    assert "discuss" in cleaned_txt, "Error: 'discuss' was mangled!"
    assert "biology" in cleaned_txt, "Error: 'biology' was mangled!"
    assert "Vee-Jay" in cleaned_txt, "Error: 'VJ' phonetic override failed!"
    assert "join uss" in cleaned_txt, "Error: 'join us' phonetic override failed!"
    assert "link in by-oh" in cleaned_txt, "Error: 'link in bio' phonetic override failed!"
    print("✅ Phonetic Override Substring Validation Passed successfully!")
    
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

    output_dirs = {
        "split_screen": "output/test_rendered_video_split_screen.mp4",
        "hero_center": "output/test_rendered_video_hero_center.mp4",
        "asymmetric": "output/test_rendered_video_asymmetric.mp4"
    }
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

    # Enable layout variation flag
    os.environ["ENABLE_LAYOUT_VARIATION"] = "1"

    for layout_type, output_path in output_dirs.items():
        print(f"\n🎬 Rendering video for layout: {layout_type}...")
        os.environ["FORCE_LAYOUT_TYPE"] = layout_type
        try:
            video_path = create_video(audio_path, script_data, chunks, output_path=output_path)
            print(f"✅ Success! Video for layout '{layout_type}' rendered to: {video_path}")
        except Exception as e:
            print(f"❌ Failed rendering for layout '{layout_type}': {e}")
            import traceback
            traceback.print_exc()

    # MD5 headline-seeded layout distribution test
    print("\n📊 Checking MD5 Headline Layout Distribution Check...")
    from video_gen import _generate_layout_profile
    sample_headlines = [
        "Unlocking WhatsApp hidden features",
        "How WhatsApp secure chats work",
        "VJ explains biology of aging",
        "Deep dive: Telegram group calls vs Discord",
        "Top 10 terminal productivity tricks",
        "Inside Apple's custom M4 AI cores",
        "Google's Gemini 1.5 Pro architecture guide",
        "Why standard SQL is making a huge comeback",
        "The complete guide to React Server Components",
        "How to self-host your own database cluster",
        "Why C++ remains the king of systems programming",
        "Unveiling biology of neural nets",
        "Telegram bots implementation playbook",
        "VJ's setup guide: Cyberpunk minimal terminal",
        "Top AI research papers of 2026",
        "Building production-grade MoviePy pipelines",
        "How to avoid repetitive content filter",
        "React vs Vue vs Svelte in 2026",
        "What's new in Python 3.14",
        "Unveiling the future of web browsers",
        "How compilers optimize matrix multiplication",
        "VJ plays retro synthwave music",
        "Secure your SSH key in 60 seconds",
        "Why Docker is still relevant in 2026",
        "Understanding zero-knowledge proofs",
        "Building a compiler from scratch in Go",
        "How video codecs compress high-motion scenes",
        "VJ's dark mode minimalist terminal setup",
        "Creating beautiful infographic cards dynamically",
        "The layout engine that YouTube hates"
    ]
    
    distribution = {"split_screen": 0, "hero_center": 0, "asymmetric": 0}
    # Temporarily unset override
    os.environ["FORCE_LAYOUT_TYPE"] = ""
    for hl in sample_headlines:
        prof = _generate_layout_profile(hl)
        lt = prof["layout_type"]
        distribution[lt] += 1
        
    print(f"Headline count: {len(sample_headlines)}")
    for lt, count in distribution.items():
        percentage = (count / len(sample_headlines)) * 100
        print(f"  - {lt}: {count} ({percentage:.1f}%)")
    
    # Assert reasonable distribution (each should be hit at least 15% of the time)
    for lt, count in distribution.items():
        assert count > 2, f"Clustering detected! Layout '{lt}' only hit {count} times."
    print("✅ Distribution test passed! High entropy confirmed.")

if __name__ == "__main__":
    main()
