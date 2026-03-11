"""
lip_sync.py — Unified lip-sync engine abstraction.

Supports:
  1. SadTalker local (Runs on CPU for GitHub actions)

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

# ── Engine directory detection ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_python_exe():
    """Find the best python executable (prefers project venv)."""
    # Check for venv in the project root
    venv_py = os.path.join(BASE_DIR, "venv", "bin", "python3")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable

# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE: SadTalker (GitHub Actions CPU)
# ═══════════════════════════════════════════════════════════════════════════════

SADTALKER_DIR = os.path.join(BASE_DIR, "SadTalker")

def _is_sadtalker_ready():
    """Check if SadTalker is cloned."""
    return os.path.isdir(SADTALKER_DIR)

def _run_sadtalker(face_path, audio_path, output_path, enhancer=None, timeout=10800):
    """Run SadTalker inference on CPU."""
    mode_str = "LITE" if not enhancer else "FULL"
    print(f"🎭 SadTalker [{mode_str}]: Starting lip-sync generation...")
    print(f"   Face: {face_path}")
    print(f"   Audio: {audio_path}")
    
    face_path = os.path.abspath(face_path)
    audio_path = os.path.abspath(audio_path)
    result_dir = os.path.abspath(os.path.join(BASE_DIR, "output", "sadtalker_output"))
    
    os.makedirs(result_dir, exist_ok=True)
    
    is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    
    preprocess = "crop"  # 'crop' gives tighter face isolation = more accurate lip-sync
    expression_scale = 1.2  # Slightly amplify facial expressions for realism
    
    cmd = [
        _get_python_exe(), "inference.py",
        "--driven_audio", audio_path,
        "--source_image", face_path,
        "--result_dir", result_dir,
        "--still",
        "--preprocess", preprocess,
        "--expression_scale", str(expression_scale),
    ]
    
    # 🏎️ Device & Quality Logic
    import torch
    has_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    
    if is_ci or not has_gpu:
        cmd.extend(["--cpu", "--size", "512", "--batch_size", "2"])
        if not enhancer:
            enhancer = "gfpgan"  # Always enhance for better face quality
        print(f"   Mode: CPU (CI/No-GPU) → Using settings (512px + GFPGAN)")
    else:
        # High-End Settings for Kaggle GPU / Local Mac GPU
        cmd.extend(["--size", "512", "--batch_size", "4"])
        if not enhancer:
            enhancer = "gfpgan"  # Enable high-end face enhancement by default
        print(f"   Mode: GPU/MPS → Using HIGH-END settings (512px + GFPGAN)")
    
    if enhancer:
        cmd.extend(["--enhancer", enhancer])
    
    print(f"   CMD: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0" # Fix for SadTalker fatal hardware/hash seed error
    # Prevents fragmentation in limited memory environments
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
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
    """Scan SadTalker result directory for the generated .mp4."""
    for root, dirs, files in os.walk(result_dir):
        for f in sorted(files, reverse=True):
            if f.endswith(".mp4"):
                return os.path.join(root, f)
    return None


def _postprocess_sadtalker(input_path, output_path):
    """Post-process SadTalker output for consistency with MuseTalk quality."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", ",".join([
                "unsharp=3:3:0.4:3:3:0.0",  # Gentle sharpen
                "eq=contrast=1.02:brightness=0.01:saturation=1.04",  # Subtle color correction
            ]),
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-c:a", "copy",
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return output_path
    except Exception as e:
        print(f"   ⚠ SadTalker post-processing skipped: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_lip_sync(face_path, audio_path, output_path, enhancer=None, timeout=10800):
    """
    Generate a lip-synced video using SadTalker.

    Args:
        face_path:   Path to face video (.mp4) or image (.png/.jpg)
        audio_path:  Path to audio file (.wav/.mp3)
        output_path: Where the lip-synced video will be saved
        enhancer:    None, 'gfpgan', or 'RestoreFormer'
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

    # ── Engine: SadTalker Local ──────────────────────────────────────────
    if _is_sadtalker_ready():
        raw_output = output_path + ".raw.mp4"
        success = _run_sadtalker(face_path, audio_path, raw_output, enhancer, timeout)
        if success and os.path.exists(raw_output):
            # Post-process for quality parity with MuseTalk
            enhanced = _postprocess_sadtalker(raw_output, output_path)
            if enhanced and os.path.exists(enhanced):
                try: os.remove(raw_output)
                except: pass
                return output_path
            # Fallback to raw
            shutil.move(raw_output, output_path)
            return output_path
        print("   ⚠ SadTalker failed.")

    # ── No engine succeeded ───────────────────────────────────────────────────
    print("🎭 Lip-sync engine unavailable or failed. Lip sync will be skipped.")
    return None


def get_available_engine():
    """Report which lip-sync engine will be used."""
    if _is_sadtalker_ready():
        return "SadTalker (CPU)"
    else:
        return None
