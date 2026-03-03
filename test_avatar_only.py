import os
import sys
from video_gen import _dynamic_avatar_clip
from config import OUTPUT_DIR, BASE_DIR

def test_avatar():
    audio_path = os.path.join(BASE_DIR, "vj.wav")
    accent_color = (139, 92, 246) # Purple-ish
    duration = 10 # 10 second test
    
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found at {audio_path}")
        # Create a dummy silent wav if needed? No, Wav2Lip needs sound to animate.
        return

    print(f"🚀 Starting Wav2Lip Avatar Test...")
    print(f"Audio: {audio_path}")
    
    # We call the internal function directly
    # This will trigger Wav2Lip if the folder exists
    avatar_clip = _dynamic_avatar_clip(duration, audio_path, accent_color)
    
    if avatar_clip:
        output_file = os.path.join(OUTPUT_DIR, "avatar_test_output.mp4")
        print(f"✅ Avatar clip created. Rendering test video to {output_file}...")
        avatar_clip.write_videofile(output_file, fps=24, codec="libx264")
        print(f"🏁 Test Complete! File saved to {output_file}")
    else:
        print("❌ Avatar clip generation returned None. Check logs for errors.")

if __name__ == "__main__":
    test_avatar()
