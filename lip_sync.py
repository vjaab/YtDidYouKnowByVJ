"""
lip_sync.py — Unified lip-sync engine abstraction.

Supports:
  1. MuseTalk local  (Best quality, requires GPU locally)

Usage:
    from lip_sync import generate_lip_sync

    output_path = generate_lip_sync(
        face_path="assets/Firefly_video_final.mp4",
        audio_path="output/voiceover.wav",
        output_path="output/temp_lipsync.mp4",
    )
    # Returns output_path on success, None on failure.
"""

import os
import sys
import subprocess
import shutil
import time

# ── Engine directory detection ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUSETALK_DIR = os.path.join(BASE_DIR, "MuseTalk")


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _is_musetalk_ready():
    """Check if MuseTalk is cloned AND has model weights downloaded."""
    if not os.path.isdir(MUSETALK_DIR):
        return False
    v15_weights = os.path.join(MUSETALK_DIR, "models", "musetalkV15", "unet.pth")
    v10_weights = os.path.join(MUSETALK_DIR, "models", "musetalk", "pytorch_model.bin")
    vae_weights = os.path.join(MUSETALK_DIR, "models", "sd-vae", "diffusion_pytorch_model.bin")

    has_model = os.path.exists(v15_weights) or os.path.exists(v10_weights)
    has_vae = os.path.exists(vae_weights)

    if has_model and has_vae:
        return True

    print(f"MuseTalk dir exists but weights incomplete: "
          f"model={'✓' if has_model else '✗'}, vae={'✓' if has_vae else '✗'}")
    return False


def _get_musetalk_version():
    """Returns 'v15' if v1.5 weights exist, else 'v1'."""
    v15 = os.path.join(MUSETALK_DIR, "models", "musetalkV15", "unet.pth")
    return "v15" if os.path.exists(v15) else "v1"


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 1: MuseTalk (Local GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_musetalk_config(face_path, audio_path):
    """Create a temporary YAML config for MuseTalk inference."""
    config_content = f"""video_path: "{os.path.abspath(face_path)}"
audio_path: "{os.path.abspath(audio_path)}"
bbox_shift: 0
preparation: True
"""
    config_dir = os.path.join(MUSETALK_DIR, "configs", "inference")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "pipeline_auto.yaml")

    with open(config_path, "w") as f:
        f.write(config_content)

    return config_path


def _run_musetalk(face_path, audio_path, output_path, timeout=1800):
    """Run MuseTalk inference as subprocess. Returns True on success."""
    version = _get_musetalk_version()
    print(f"🎭 MuseTalk {version}: Starting lip-sync generation...")
    print(f"   Face: {face_path}")
    print(f"   Audio: {audio_path}")

    config_path = _build_musetalk_config(face_path, audio_path)
    result_dir = os.path.join(MUSETALK_DIR, "results", "pipeline_output")
    os.makedirs(result_dir, exist_ok=True)

    if version == "v15":
        unet_path = os.path.join("models", "musetalkV15", "unet.pth")
        unet_config = os.path.join("models", "musetalkV15", "musetalk.json")
    else:
        unet_path = os.path.join("models", "musetalk", "pytorch_model.bin")
        unet_config = os.path.join("models", "musetalk", "musetalk.json")

    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--inference_config", config_path,
        "--result_dir", result_dir,
        "--unet_model_path", unet_path,
        "--unet_config", unet_config,
        "--version", version,
    ]

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        env["FFMPEG_PATH"] = os.path.dirname(ffmpeg_bin)

    print(f"   CMD: {' '.join(str(c) for c in cmd)}")

    try:
        result = subprocess.run(
            cmd, cwd=MUSETALK_DIR, capture_output=True,
            text=True, env=env, timeout=timeout,
        )
        print(f"   STDOUT: {(result.stdout or '')[-800:]}")
        print(f"   STDERR: {(result.stderr or '')[-800:]}")

        if result.returncode != 0:
            print(f"   ✗ MuseTalk exited with code {result.returncode}")
            return False

        generated = _find_output_video(result_dir)
        if generated:
            shutil.copy2(generated, output_path)
            print(f"   ✓ Output: {output_path}")
            return True
        else:
            print(f"   ✗ No output video in {result_dir}")
            return False

    except subprocess.TimeoutExpired:
        print(f"   ✗ Timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 2: SadTalker (GitHub Actions CPU)
# ═══════════════════════════════════════════════════════════════════════════════

SADTALKER_DIR = os.path.join(BASE_DIR, "SadTalker")

def _is_sadtalker_ready():
    """Check if SadTalker is cloned."""
    return os.path.isdir(SADTALKER_DIR)

def _run_sadtalker(face_path, audio_path, output_path, timeout=10800):
    """Run SadTalker inference on CPU."""
    print(f"🎭 SadTalker: Starting lip-sync generation (CPU mode, might take a while)...")
    print(f"   Face: {face_path}")
    print(f"   Audio: {audio_path}")
    
    face_path = os.path.abspath(face_path)
    audio_path = os.path.abspath(audio_path)
    result_dir = os.path.abspath(os.path.join(BASE_DIR, "output", "sadtalker_output"))
    
    os.makedirs(result_dir, exist_ok=True)
    
    cmd = [
        sys.executable, "inference.py",
        "--driven_audio", audio_path,
        "--source_image", face_path,
        "--result_dir", result_dir,
        "--still",
        "--preprocess", "full",
        "--enhancer", "gfpgan",
        "--device", "cpu"
    ]
    
    print(f"   CMD: {' '.join(cmd)}")
    
    env = os.environ.copy()
    
    try:
        result = subprocess.run(
            cmd, cwd=SADTALKER_DIR, capture_output=True,
            text=True, env=env, timeout=timeout
        )
        print(f"   STDOUT: {(result.stdout or '')[-1000:]}")
        print(f"   STDERR: {(result.stderr or '')[-1000:]}")

        if result.returncode != 0:
            print(f"   ✗ SadTalker exited with code {result.returncode}")
            return False

        generated = _find_output_video(result_dir)
        if generated:
            shutil.copy2(generated, output_path)
            print(f"   ✓ Output: {output_path}")
            return True
        else:
            print(f"   ✗ No output video in {result_dir}")
            return False

    except subprocess.TimeoutExpired:
        print(f"   ✗ Timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def _find_output_video(result_dir):
    """Scan MuseTalk/SadTalker result directory for the generated .mp4."""
    for root, dirs, files in os.walk(result_dir):
        for f in sorted(files, reverse=True):
            if f.endswith(".mp4"):
                return os.path.join(root, f)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_lip_sync(face_path, audio_path, output_path, timeout=10800):
    """
    Generate a lip-synced video.

    Engine priority:
      1. MuseTalk local  (if installed + weights present)
      2. SadTalker local (if installed, runs on CPU for GitHub actions)

    Args:
        face_path:   Path to face video (.mp4) or image (.png/.jpg)
        audio_path:  Path to audio file (.wav/.mp3)
        output_path: Where the lip-synced video will be saved
        timeout:     Max seconds to wait (default 10800)
    """
    if not os.path.exists(face_path):
        print(f"🎭 Lip-sync: Face file not found: {face_path}")
        return None

    if not os.path.exists(audio_path):
        print(f"🎭 Lip-sync: Audio file not found: {audio_path}")
        return None

    # Clean up previous output
    if os.path.exists(output_path):
        os.remove(output_path)

    # ── Engine 1: MuseTalk Local ──────────────────────────────────────────────
    if _is_musetalk_ready():
        success = _run_musetalk(face_path, audio_path, output_path, timeout)
        if success and os.path.exists(output_path):
            return output_path
        print("   ⚠ MuseTalk local failed.")

    # ── Engine 2: SadTalker Local (CPU) ───────────────────────────────────────
    elif _is_sadtalker_ready():
        success = _run_sadtalker(face_path, audio_path, output_path, timeout)
        if success and os.path.exists(output_path):
            return output_path
        print("   ⚠ SadTalker failed.")

    # ── No engine succeeded ───────────────────────────────────────────────────
    print("🎭 All lip-sync engines unavailable or failed. Lip sync will be skipped.")
    return None


def get_available_engine():
    """Report which lip-sync engine will be used."""
    if _is_musetalk_ready():
        return f"MuseTalk {_get_musetalk_version()}"
    elif _is_sadtalker_ready():
        return "SadTalker (CPU)"
    else:
        return None
