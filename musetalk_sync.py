import os
import subprocess
import shutil
import time
import tempfile
import yaml

def generate_musetalk(face_path, audio_path, output_path, timeout=10800):
    """
    Standard interface for MuseTalk lip-sync.
    Uses the official `python -m scripts.inference` command.
    Requires MuseTalk to be cloned and weights downloaded.
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

    # ── Create a temporary YAML config for this run ──────────────────────
    config_content = {
        "video_path": face_path,
        "audio_path": audio_path,
        "bbox_shift": 0,
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
        print("   Using MuseTalk v1.5 (recommended)")
    elif os.path.exists(v10_path):
        unet_model_path = os.path.join("models", "musetalk", "pytorch_model.bin")
        unet_config = os.path.join("models", "musetalk", "musetalk.json")
        version_arg = "v1"
        print("   Using MuseTalk v1.0")
    else:
        print("   ✗ MuseTalk model weights not found. Run download_weights.sh first.")
        return None

    # ── Build the official inference command ──────────────────────────────
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
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        subprocess.run(cmd, cwd=musetalk_dir, check=True, timeout=timeout, env=env)

        # MuseTalk saves output inside result_dir — find the generated mp4
        generated = _find_musetalk_output(result_dir)
        if generated:
            shutil.copy2(generated, output_path_abs)
            duration = time.time() - start_time
            print(f"✅ MuseTalk done in {duration:.1f}s -> {output_path_abs}")
            return output_path_abs

        print("   ✗ MuseTalk output not found in results directory.")
        return None
    except subprocess.TimeoutExpired:
        print(f"   ✗ MuseTalk timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"   ✗ MuseTalk failed: {e}")
        return None


def _find_musetalk_output(result_dir):
    """Scan MuseTalk result directory for the generated .mp4."""
    for root, dirs, files in os.walk(result_dir):
        for f in sorted(files, reverse=True):
            if f.endswith(".mp4"):
                return os.path.join(root, f)
    return None
