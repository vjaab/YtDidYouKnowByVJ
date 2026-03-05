"""
lip_sync.py — Unified lip-sync engine abstraction.

Supports (in order of preference):
  1. MuseTalk local  (Best quality, requires GPU locally)
  2. fal.ai   cloud  (Works in GitHub Actions — no GPU needed!)

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
import requests

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


def _is_fal_ready():
    """Check if fal.ai cloud API is configured."""
    return bool(os.getenv("FAL_KEY", ""))


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


def _find_output_video(result_dir):
    """Scan MuseTalk result directory for the generated .mp4."""
    for root, dirs, files in os.walk(result_dir):
        for f in sorted(files, reverse=True):
            if f.endswith(".mp4"):
                return os.path.join(root, f)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 2: fal.ai Cloud API (Works in GitHub Actions — no GPU required!)
# ═══════════════════════════════════════════════════════════════════════════════

def _upload_to_fal(file_path):
    """Upload a local file to fal.ai CDN and return the public URL."""
    fal_key = os.getenv("FAL_KEY", "")
    if not fal_key:
        return None

    file_name = os.path.basename(file_path)
    mime_map = {
        ".mp4": "video/mp4",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    ext = os.path.splitext(file_name)[1].lower()
    content_type = mime_map.get(ext, "application/octet-stream")

    print(f"   ↑ Uploading {file_name} to fal.ai CDN...")
    try:
        init_resp = requests.post(
            "https://rest.alpha.fal.ai/storage/upload/initiate",
            headers={
                "Authorization": f"Key {fal_key}",
                "Content-Type": "application/json",
            },
            json={
                "file_name": file_name,
                "content_type": content_type,
            },
            timeout=30,
        )
        init_resp.raise_for_status()
        upload_data = init_resp.json()
        upload_url = upload_data.get("upload_url")
        file_url = upload_data.get("file_url")

        if not upload_url or not file_url:
            print(f"   ✗ Upload init failed: {upload_data}")
            return None

        with open(file_path, "rb") as f:
            put_resp = requests.put(
                upload_url,
                data=f,
                headers={"Content-Type": content_type},
                timeout=300,
            )
            put_resp.raise_for_status()

        print(f"   ✓ Uploaded: {file_url}")
        return file_url

    except Exception as e:
        print(f"   ✗ Upload failed: {e}")
        return None


def _run_fal_cloud(face_path, audio_path, output_path, timeout=600):
    """
    Run MuseTalk lip-sync via fal.ai cloud API.
    Works without local GPU — perfect for GitHub Actions.
    """
    fal_key = os.getenv("FAL_KEY", "")
    if not fal_key:
        print("   ✗ FAL_KEY not set.")
        return False

    print(f"🎭 fal.ai Cloud (MuseTalk): Starting lip-sync generation...")
    print(f"   Face: {face_path}")
    print(f"   Audio: {audio_path}")

    # Upload local files to fal CDN
    face_url = _upload_to_fal(face_path)
    audio_url = _upload_to_fal(audio_path)

    if not face_url or not audio_url:
        print("   ✗ Failed to upload files to fal.ai CDN")
        return False

    headers = {
        "Authorization": f"Key {fal_key}",
        "Content-Type": "application/json",
    }

    # Submit to queue
    print("   → Submitting to fal.ai queue...")
    try:
        submit_resp = requests.post(
            "https://queue.fal.run/fal-ai/musetalk",
            headers=headers,
            json={
                "source_video_url": face_url,
                "audio_url": audio_url,
            },
            timeout=60,
        )
        submit_resp.raise_for_status()
        queue_data = submit_resp.json()
        request_id = queue_data.get("request_id")

        if not request_id:
            print(f"   ✗ No request_id in response: {queue_data}")
            return False

        print(f"   → Queued: request_id={request_id}")

    except Exception as e:
        print(f"   ✗ Queue submit failed: {e}")
        return False

    # Poll for completion
    status_url = f"https://queue.fal.run/fal-ai/musetalk/requests/{request_id}/status"
    result_url = f"https://queue.fal.run/fal-ai/musetalk/requests/{request_id}"

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            status_resp = requests.get(status_url, headers=headers, timeout=30)
            status_resp.raise_for_status()
            status_data = status_resp.json()
            status = status_data.get("status", "")

            if status == "COMPLETED":
                print("   ✓ Generation completed!")
                break
            elif status in ("FAILED", "CANCELLED"):
                print(f"   ✗ Generation {status}: {status_data}")
                return False
            else:
                elapsed = int(time.time() - start_time)
                print(f"   ⏳ Status: {status} ({elapsed}s elapsed)")
                time.sleep(5)

        except Exception as e:
            print(f"   ⚠ Status check error: {e}")
            time.sleep(5)
    else:
        print(f"   ✗ Timed out after {timeout}s")
        return False

    # Fetch result
    try:
        result_resp = requests.get(result_url, headers=headers, timeout=60)
        result_resp.raise_for_status()
        result_data = result_resp.json()

        video_info = result_data.get("video", {})
        video_url = video_info.get("url", "")

        if not video_url:
            print(f"   ✗ No video URL in result: {result_data}")
            return False

        print(f"   ↓ Downloading generated video...")
        dl_resp = requests.get(video_url, timeout=300)
        dl_resp.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(dl_resp.content)

        file_size = os.path.getsize(output_path)
        print(f"   ✓ Downloaded: {output_path} ({file_size / 1024:.0f} KB)")
        return True

    except Exception as e:
        print(f"   ✗ Result fetch failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_lip_sync(face_path, audio_path, output_path, timeout=1800):
    """
    Generate a lip-synced video.

    Engine priority:
      1. MuseTalk local  (if installed + weights present)
      2. fal.ai   cloud  (if FAL_KEY is set — works in CI!)

    Args:
        face_path:   Path to face video (.mp4) or image (.png/.jpg)
        audio_path:  Path to audio file (.wav/.mp3)
        output_path: Where the lip-synced video will be saved
        timeout:     Max seconds to wait (default 1800)

    Returns:
        str: output_path on success
        None: on failure
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
        print("   ⚠ MuseTalk local failed, trying cloud engine...")

    # ── Engine 2: fal.ai Cloud ────────────────────────────────────────────────
    if _is_fal_ready():
        success = _run_fal_cloud(face_path, audio_path, output_path, timeout=600)
        if success and os.path.exists(output_path):
            return output_path
        print("   ⚠ fal.ai cloud failed.")

    # ── No engine succeeded ───────────────────────────────────────────────────
    print("🎭 All lip-sync engines unavailable or failed. Lip sync will be skipped.")
    return None


def get_available_engine():
    """Report which lip-sync engine will be used."""
    if _is_musetalk_ready():
        return f"MuseTalk {_get_musetalk_version()}"
    elif _is_fal_ready():
        return "fal.ai Cloud (MuseTalk)"
    else:
        return None
