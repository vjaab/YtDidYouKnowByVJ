# -*- coding: utf-8 -*-
"""
test_wav2lip.py - Standalone Wav2Lip avatar generation test.
Tests ONLY the Wav2Lip subprocess call (no Gemini, no pipeline).
"""
import os
import sys
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR = os.path.join(BASE_DIR, "Wav2Lip")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FACE_IMG = os.path.join(ASSETS_DIR, "Firefly_video.mp4")
if not os.path.exists(FACE_IMG):
    FACE_IMG = os.path.join(ASSETS_DIR, "youtube_pic.png")
if not os.path.exists(FACE_IMG):
    FACE_IMG = os.path.join(ASSETS_DIR, "vj_profile.jpg")
AUDIO_FILE = os.path.join(BASE_DIR, "vj.wav")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "test_avatar.mp4")

def test_wav2lip():
    print("=" * 60)
    print("WAV2LIP STANDALONE TEST")
    print("=" * 60)

    # 1. Verify all prerequisites
    print("\n[1/4] Checking prerequisites...")
    checks = {
        "Wav2Lip directory": WAV2LIP_DIR,
        "inference.py": os.path.join(WAV2LIP_DIR, "inference.py"),
        "Model weights": os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth"),
        "Face image": FACE_IMG,
        "Audio file": AUDIO_FILE,
    }
    all_ok = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print("  [%s] %s: %s" % (status, name, path))
        if not exists:
            all_ok = False

    if not all_ok:
        print("\nPrerequisites check failed. Cannot proceed.")
        return False

    # 2. Check Python & environment
    print("\n[2/4] Environment info...")
    print("  Python: %s (%s)" % (sys.executable, sys.version))
    print("  PYTHONHASHSEED in env: '%s'" % os.environ.get('PYTHONHASHSEED', '(not set)'))

    # 3. Build and run Wav2Lip command
    print("\n[3/4] Running Wav2Lip inference...")

    # Build CLEAN environment - force PYTHONHASHSEED to prevent crash
    w2l_env = os.environ.copy()
    w2l_env["PYTHONHASHSEED"] = "0"
    print("  Subprocess PYTHONHASHSEED forced to: '%s'" % w2l_env['PYTHONHASHSEED'])

    cmd = [
        sys.executable, "inference.py",
        "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
        "--face", FACE_IMG,
        "--audio", AUDIO_FILE,
        "--outfile", OUTPUT_VIDEO,
        "--pads", "0", "20", "0", "0",
        "--face_det_batch_size", "2",
        "--wav2lip_batch_size", "16",
    ]
    print("  CMD: %s" % " ".join(cmd))
    print("  CWD: %s" % WAV2LIP_DIR)
    print()

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=WAV2LIP_DIR,
            capture_output=True,
            text=True,
            env=w2l_env,
            timeout=1800,
        )
        elapsed = time.time() - start

        print("  Exit code: %d" % result.returncode)
        print("  Duration: %.1fs" % elapsed)

        if result.stdout:
            print("\n  --- STDOUT (last 1000 chars) ---")
            print(result.stdout[-1000:])

        if result.stderr:
            print("\n  --- STDERR (last 1000 chars) ---")
            print(result.stderr[-1000:])

    except subprocess.TimeoutExpired:
        print("  Wav2Lip timed out after 300 seconds.")
        return False
    except Exception as e:
        print("  Exception running Wav2Lip: %s" % e)
        return False

    # 4. Check output
    print("\n[4/4] Checking output...")
    if os.path.exists(OUTPUT_VIDEO):
        size_mb = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
        print("  OK - Avatar video created: %s" % OUTPUT_VIDEO)
        print("  OK - File size: %.1f MB" % size_mb)
        return True
    else:
        print("  FAIL - Output file NOT created: %s" % OUTPUT_VIDEO)
        return False


if __name__ == "__main__":
    success = test_wav2lip()
    print("\n" + "=" * 60)
    if success:
        print("WAV2LIP TEST PASSED - Avatar generated successfully!")
    else:
        print("WAV2LIP TEST FAILED - See errors above.")
    print("=" * 60)
