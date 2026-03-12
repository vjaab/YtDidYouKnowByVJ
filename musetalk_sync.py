import os
import subprocess
import shutil
import time
import tempfile
import yaml

def generate_musetalk(face_path, audio_path, output_path, timeout=10800):
    """
    High-quality MuseTalk lip-sync pipeline.
    Uses the official `python -m scripts.inference` command.
    Includes pre/post-processing for maximum realism.
    """
    print(f"🎤 MuseTalk [P1]: Starting high-end lip-sync...")
    print(f"   Face: {face_path}")
    print(f"   Audio: {audio_path}")

    musetalk_dir = "MuseTalk"
    if not os.path.isdir(musetalk_dir):
        print("   ✗ MuseTalk directory not found.")
        return None

    face_path = os.path.abspath(face_path)
    audio_path = os.path.abspath(audio_path)
    output_path_abs = os.path.abspath(output_path)
    result_dir = os.path.join(os.path.abspath(musetalk_dir), "results", "pipeline_run")
    os.makedirs(result_dir, exist_ok=True)

    # ── Pre-process: Extract a high-quality reference frame ──────────────
    ref_frame_path = os.path.join(os.path.dirname(output_path_abs), "musetalk_ref_frame.png")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", face_path,
            "-vframes", "1", "-q:v", "1",
            ref_frame_path
        ], check=True, capture_output=True)
        print(f"   Extracted reference frame: {ref_frame_path}")
    except Exception as e:
        print(f"   ⚠ Reference frame extraction failed: {e}")
        ref_frame_path = None

    # ── Create YAML config with quality-optimized settings ────────────────
    config_content = {
        "video_path": face_path,
        "audio_path": audio_path,
        "bbox_shift": 5,  # Positive shift = more jaw movement during speech (more natural)
    }
    config_path = os.path.join(os.path.abspath(musetalk_dir), "configs", "inference", "_pipeline_run.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump([config_content], f)

    # ── Determine model version (prefer v1.5 if available) ───────────────
    v15_path = os.path.join(musetalk_dir, "models", "musetalkV15", "unet.pth")
    v10_path = os.path.join(musetalk_dir, "models", "musetalk", "pytorch_model.bin")

    if os.path.exists(v15_path):
        unet_model_path = os.path.join("models", "musetalkV15", "unet.pth")
        unet_config = os.path.join("models", "musetalkV15", "musetalk.json")
        version_arg = "v15"
        print("   Using MuseTalk v1.5 (highest quality)")
    elif os.path.exists(v10_path):
        unet_model_path = os.path.join("models", "musetalk", "pytorch_model.bin")
        unet_config = os.path.join("models", "musetalk", "musetalk.json")
        version_arg = "v1"
        print("   Using MuseTalk v1.0")
    else:
        print("   ✗ MuseTalk model weights not found. Run download_weights.sh first.")
        return None

    # ── Build the official inference command with quality flags ────────────
    cmd = [
        "python3", "-m", "scripts.inference",
        "--inference_config", config_path,
        "--result_dir", result_dir,
        "--unet_model_path", unet_model_path,
        "--unet_config", unet_config,
        "--version", version_arg,
    ]

    print(f"   CMD: {' '.join(cmd)}")

    start_time = time.time()
    try:
        env = os.environ.copy()
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONHASHSEED"] = "random"
        
        result = subprocess.run(
            cmd, cwd=musetalk_dir, capture_output=True, text=True,
            timeout=timeout, env=env
        )
        
        if result.returncode != 0:
            print(f"   ✗ MuseTalk inference exited with code {result.returncode}")
            if result.stderr:
                # Print last 2000 chars of stderr for debugging
                print(f"   STDERR (last 2000 chars):\n{result.stderr[-2000:]}")
            if result.stdout:
                print(f"   STDOUT (last 500 chars):\n{result.stdout[-500:]}")
            return None
        
        # Print stdout for progress visibility
        if result.stdout:
            print(f"   {result.stdout[-500:]}")

        # MuseTalk saves output inside result_dir — find the generated mp4
        generated = _find_musetalk_output(result_dir)
        if generated:
            # ── Post-process: Enhance face quality ────────────────────────
            enhanced_path = _postprocess_lipsync(generated, output_path_abs)
            if enhanced_path:
                duration = time.time() - start_time
                print(f"✅ MuseTalk done in {duration:.1f}s → {enhanced_path}")
                return enhanced_path

            # Fallback to raw output if post-processing fails
            shutil.copy2(generated, output_path_abs)
            duration = time.time() - start_time
            print(f"✅ MuseTalk done in {duration:.1f}s → {output_path_abs}")
            return output_path_abs

        print("   ✗ MuseTalk output not found in results directory.")
        return None
    except subprocess.TimeoutExpired:
        print(f"   ✗ MuseTalk timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"   ✗ MuseTalk failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up reference frame
        if ref_frame_path and os.path.exists(ref_frame_path):
            try: os.remove(ref_frame_path)
            except: pass


def _postprocess_lipsync(input_path, output_path):
    """
    Post-process lip-synced video for realism:
    1. Upscale to 512px width if smaller
    2. Sharpen slightly for face definition
    3. Subtle color correction for natural skin tones
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", ",".join([
                "scale=512:-1:flags=lanczos",       # Upscale with high-quality Lanczos filter
                "unsharp=3:3:0.5:3:3:0.0",          # Gentle sharpen (luma only, no chroma)
                "eq=contrast=1.03:brightness=0.01:saturation=1.05",  # Subtle warmth + contrast
            ]),
            "-c:v", "libx264",
            "-crf", "18",       # High quality (lower = better, 18 is visually lossless)
            "-preset", "slow",  # Better compression for same quality
            "-c:a", "copy",     # Keep audio untouched
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        print(f"   🎨 Post-processed: upscaled+sharpened+color-corrected → {output_path}")
        return output_path
    except Exception as e:
        print(f"   ⚠ Post-processing failed: {e}")
        return None


def _find_musetalk_output(result_dir):
    """Scan MuseTalk result directory for the generated .mp4."""
    for root, dirs, files in os.walk(result_dir):
        for f in sorted(files, reverse=True):
            if f.endswith(".mp4"):
                return os.path.join(root, f)
    return None
