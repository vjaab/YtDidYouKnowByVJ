"""
Quick test: Run GFPGAN face enhancement on a short clip of Firefly_video_final.mp4.
This tests the _enhance_with_gfpgan() function without needing Wav2Lip.
"""
import os
import subprocess
import imageio_ffmpeg

# Use project config
from config import ASSETS_DIR, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Extract a short 3-second clip from Firefly video as our test input
src_video = os.path.join(ASSETS_DIR, "Firefly_video_final.mp4")
test_clip = os.path.join(OUTPUT_DIR, "test_gfpgan_input.mp4")

if not os.path.exists(src_video):
    print(f"ERROR: {src_video} not found!")
    exit(1)

print(f"Extracting 3-second test clip from {src_video}...")
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([
    ffmpeg_exe, "-y", "-i", src_video,
    "-t", "3", "-c", "copy", test_clip
], capture_output=True, text=True, check=True)
print(f"Test clip created: {test_clip}")

# Step 2: Run GFPGAN enhancement
from video_gen import _enhance_with_gfpgan

print("\n=== Running GFPGAN face enhancement ===")
enhanced = _enhance_with_gfpgan(test_clip)

print(f"\nInput:    {test_clip}")
print(f"Enhanced: {enhanced}")

if enhanced != test_clip:
    # Check file sizes
    orig_size = os.path.getsize(test_clip) / (1024*1024)
    enh_size = os.path.getsize(enhanced) / (1024*1024)
    print(f"\nOriginal size: {orig_size:.1f} MB")
    print(f"Enhanced size: {enh_size:.1f} MB")
    print("\n✅ GFPGAN enhancement successful! Check the output file.")
else:
    print("\n⚠️  Enhancement did not produce a new file (may have fallen back).")
