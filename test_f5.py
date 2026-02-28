import os
import torch
import soundfile as sf
from audio_gen import _generate_f5_clone, VJ_REF_WAV, VJ_REF_TEXT
import imageio_ffmpeg
from pydub import AudioSegment
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

def test_f5():
    print(f"Testing F5-TTS with reference: {VJ_REF_WAV}")
    test_text = "This is a test of the zero cost cloned voice system. It should sound precisely like VJ because we are using a 2026 local transformer model."
    output_path = "output/vj_test_clone.mp3"
    
    # Create output dir if needed
    os.makedirs("output", exist_ok=True)
    
    try:
        path, dur, word_timestamps = _generate_f5_clone(test_text, output_path)
        print(f"SUCCESS: Generated {path} ({dur:.2f}s) with {len(word_timestamps)} timestamps.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    test_f5()
