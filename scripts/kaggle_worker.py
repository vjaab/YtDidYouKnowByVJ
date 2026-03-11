import os
os.environ["PYTHONHASHSEED"] = "0"
import subprocess
import shutil
import sys
import time

# 🚀 KAGGLE GPU WORKER FOR YtDidYouKnowByVJ
# Designed to run on Kaggle T4 x2 or P100

def run_cmd(cmd, cwd=None, quiet=False):
    if quiet:
        print(f"Executing (Quietly): {' '.join(cmd)}")
        subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=cwd, check=True)

def setup_musetalk():
    if not os.path.isdir("MuseTalk"):
        print("📥 Cloning MuseTalk...")
        run_cmd(["git", "clone", "-q", "https://github.com/TMElyralab/MuseTalk.git"])
        
        # ═══════════════════════════════════════════════════════════════════
        # BYPASS MuseTalk's broken requirements.txt entirely.
        # It pins ancient versions (tensorflow==2.12.0, numpy==1.23.5, etc.)
        # that don't exist on Kaggle's Python 3.12.
        #
        # Instead, we install EXACTLY what MuseTalk inference needs at runtime:
        #   scripts/inference.py → imports from musetalk.utils.*
        #   musetalk/utils/* → imports diffusers, transformers, mmpose, etc.
        # ═══════════════════════════════════════════════════════════════════
        print("📦 Installing MuseTalk runtime dependencies (curated list)...")
        
        # Core ML deps that MuseTalk actually imports (Kaggle already has torch/numpy/opencv)
        musetalk_deps = [
            "diffusers",           # UNet, VAE decoder
            "accelerate",          # HuggingFace model loading
            "transformers",        # Whisper audio features
            "huggingface_hub",     # Model downloads
            "einops",              # Tensor reshaping in UNet
            "omegaconf",           # Config parsing
            "soundfile",           # Audio I/O
            "librosa",             # Audio feature extraction
            "gradio",              # (may be imported but not used for inference)
            "gdown",               # Weight downloads
            "ffmpeg-python",       # Video processing
            "moviepy",             # Video composition
            "imageio[ffmpeg]",     # Frame I/O
        ]
        
        # Install all deps in a single pip call for efficiency
        try:
            run_cmd(["pip", "install", "-q"] + musetalk_deps)
            print("   ✅ MuseTalk core deps installed")
        except Exception as e:
            print(f"   ⚠ Some MuseTalk deps failed: {e}")
            # Fallback: install one-by-one so partial failures don't block
            for dep in musetalk_deps:
                try:
                    run_cmd(["pip", "install", "-q", dep])
                except Exception:
                    print(f"   ⚠ Skipping {dep}")
        
        # ── MMLab Stack (mmcv + mmpose + mmdet) ──────────────────────────
        print("📦 Installing MMLab stack...")
        # Direct URLs for Python 3.12/CUDA 12.1 (typical for Kaggle)
        # We try mim first, then fallback to direct pip URL
        mm_index = "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html"
        
        steps = [
            (["pip", "install", "-q", "-U", "openmim", "setuptools<70", "wheel", "packaging"], "mim-core"),
            (["mim", "install", "mmengine"], "mmengine"),
            (["pip", "install", "-q", "mmcv>=2.1.0", "-f", mm_index], "mmcv"),
            (["mim", "install", "mmdet>=3.1.0"], "mmdet"),
            (["mim", "install", "mmpose>=1.1.0"], "mmpose"),
        ]
        
        mmlab_ok = True
        for cmd, name in steps:
            try:
                run_cmd(cmd)
                print(f"   ✅ {name}")
            except Exception as e:
                print(f"   ⚠ {name} failed via primary: {e}")
                # Secondary fallback for mmdet/mmpose if mim failed
                if name in ["mmdet", "mmpose"]:
                    try:
                        print(f"   Trying direct pip for {name}...")
                        run_cmd(["pip", "install", "-q", name])
                        print(f"   ✅ {name} (via direct pip)")
                    except:
                        print(f"   ❌ {name} total failure.")
                        mmlab_ok = False
                else:
                    mmlab_ok = False
        
        if not mmlab_ok:
            print("   ⚠ MMLab stack incomplete — MuseTalk may fall back to SadTalker")
        
        # ── Download MuseTalk model weights ──────────────────────────────
        print("📥 Downloading MuseTalk model weights...")
        # Skip the shell script (which lacks huggingface-cli in path) and use Python directly
        _download_musetalk_weights_manual()
    else:
        print("✓ MuseTalk already set up.")


def _download_musetalk_weights_manual():
    """Download weights via huggingface_hub API."""
    import subprocess
    models_dir = os.path.join("MuseTalk", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("   Using huggingface_hub snapshot_download...")
    try:
        # We use a python -c call to ensure it runs in a clean environment if needed
        # but calling it directly inside the script is fine too.
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id='TMElyralab/MuseTalk', 
            local_dir=models_dir, 
            allow_patterns=['musetalk/*', 'musetalkV15/*', 'dwpose/*', 'sd-vae-ft-mse/*', 'whisper/*']
        )
        print("   ✅ Weights downloaded successfully.")
    except Exception as e:
        print(f"   ❌ Weight download failed: {e}")
        # Last resort: try the shell script if it exists
        print("   Trying bash download_weights.sh as last resort...")
        try:
            run_cmd(["bash", "download_weights.sh"], cwd="MuseTalk")
        except:
            print("   ❌ Bash download also failed.")

        
def _patch_basicsr():
    """Basicsr patch for Python 3.12 (LooseVersion is gone) and modern torchvision."""
    print("🛠️ Patching basicsr for modern environments...")
    try:
        import basicsr
        path = os.path.dirname(basicsr.__file__)
        # Fix 1: functional_tensor -> functional
        deg_file = os.path.join(path, "data", "degradations.py")
        if os.path.exists(deg_file):
            with open(deg_file, 'r') as f: content = f.read()
            content = content.replace("from torchvision.transforms.functional_tensor import rgb_to_grayscale", 
                                      "from torchvision.transforms.functional import rgb_to_grayscale")
            with open(deg_file, 'w') as f: f.write(content)
        
        # Fix 2: distutils.version -> packaging.version
        arch_util = os.path.join(path, "archs", "arch_util.py")
        if os.path.exists(arch_util):
            with open(arch_util, 'r') as f: content = f.read()
            content = content.replace("from distutils.version import LooseVersion", "from packaging.version import parse as LooseVersion")
            with open(arch_util, 'w') as f: f.write(content)
        print("   ✅ Basicsr patched.")
    except Exception as e:
        print(f"   ⚠ Basicsr patch failed: {e}")


def setup_sadtalker():
    if not os.path.isdir("SadTalker"):
        print("📥 Cloning SadTalker...")
        run_cmd(["git", "clone", "-q", "https://github.com/OpenTalker/SadTalker.git"])
        
        # Apply patches for SadTalker itself
        print("🛠️ Patching SadTalker for modern environments...")
        prep_file = "SadTalker/src/face3d/util/preprocess.py"
        if os.path.exists(prep_file):
            with open(prep_file, 'r') as f: content = f.read()
            import re
            content = content.replace('np.VisibleDeprecationWarning', 'Warning')
            with open(prep_file, 'w') as f: f.write(content)

        print("📥 Downloading SadTalker Weights...")
        os.makedirs("SadTalker/checkpoints", exist_ok=True)
        run_cmd(["bash", "scripts/download_models.sh"], cwd="SadTalker", quiet=True)

def setup_project():
    # ── SYSTEM DEPENDENCIES (Kaggle Linux) ──────────────────────────────────
    print("🖥️ Installing System Dependencies (espeak-ng, ffmpeg)...")
    try:
        # In Kaggle, apt-get usually works without sudo if run as a subprocess
        subprocess.run(["apt-get", "update"], check=False)
        subprocess.run(["apt-get", "install", "-y", "espeak-ng", "ffmpeg"], check=False)
    except:
        print("⚠️ System dependency installation skipped (non-critical).")

    if os.path.isdir("YtDidYouKnowByVJ"):
        print("🧹 Removing stale repository for fresh clone...")
        shutil.rmtree("YtDidYouKnowByVJ", ignore_errors=True)
        
    print("📥 Cloning Project Repository...")
    run_cmd(["git", "clone", "-q", "https://github.com/vjaab/YtDidYouKnowByVJ.git"])
    
    # ── PYTHON DEPENDENCIES ────────────────────────────────────────────────
    print("📦 Installing Python Dependencies...")
    run_cmd(["pip", "install", "-q", "-U", "pip", "setuptools", "wheel"])
    run_cmd(["pip", "install", "-q", "-r", "requirements.txt"], cwd="YtDidYouKnowByVJ")
    
    # Force GPU specific backends for Kokoro and Audio
    run_cmd(["pip", "install", "-q", 
        "onnxruntime-gpu", "espeakng-loader",
        "f5-tts", "stable-ts", "torch", "torchvision", "torchaudio", 
        "facexlib", "gfpgan", "basicsr", "av", "yacs", "kornia", 
        "librosa", "resampy", "imageio-ffmpeg", "pyyaml", "joblib", 
        "scikit-image", "safetensors", "trimesh", "face-alignment",
        "diffusers", "transformers", "accelerate", "g2p_en",
        "--extra-index-url", "https://download.pytorch.org/whl/cu118"])

    print("�️ Applying environment patches...")
    _patch_basicsr()

def process_job():
    print("🎬 Starting GPU Job...")
    
    # Kaggle-specific execution:
    # `JOB_PAYLOAD` is injected at the top of the file by `kaggle_handover.py` before push
    import json
    
    if "JOB_PAYLOAD" in globals():
        job_data = globals()["JOB_PAYLOAD"]
    elif os.path.exists("job_data.json"):
        with open("job_data.json", 'r') as f:
            job_data = json.load(f)
    else:
        print(f"⚠ Job Data not found. Running default demo mode.")
        # Default fallback/demo logic
        return

    try:
        sys.path.append(os.getcwd())
        from audio_gen import generate_voiceover, unload_f5_model
        from lip_sync import generate_lip_sync
        from musetalk_sync import generate_musetalk
        
        script = job_data.get("script")
        voice = job_data.get("voice")
        emotion = job_data.get("emotion")
        custom_map = job_data.get("custom_map")
        
        # 🟢 STEP 1: GPU Audio (F5-TTS)
        audio_path, duration, word_timestamps = generate_voiceover(
            script, voice, emotion, custom_phonetic_map=custom_map
        )
        
        # 🔓 Unload F5-TTS
        unload_f5_model()
        
        # 🟢 STEP 2: Prep Assets & Optimize
        face_path = "assets/Firefly_video_final.mp4"
        optimized_face = "assets/Firefly_video_optimized.mp4"
        lipsync_out = "kaggle_lipsync.mp4"
        lipsync_path = None
        
        print("🏎️ Optimizing template resolution (512px) for RAM safety...")
        run_cmd([
            "ffmpeg", "-y", "-i", face_path, 
            "-vf", "scale=512:-1", 
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            optimized_face
        ])
        
        # 🏅 TIER 1: MuseTalk (Best Quality + Gestures)
        import gc
        import torch
        
        try:
            gc.collect()
            torch.cuda.empty_cache()
            lipsync_path = generate_musetalk(
                face_path=optimized_face,
                audio_path=audio_path,
                output_path=lipsync_out
            )
        except Exception as e:
            print(f"   ⚠ MuseTalk failed: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        # 🥈 TIER 2: SadTalker Fallback
        if not lipsync_path:
            print("   ↳ Falling back to SadTalker...")
            try:
                gc.collect()
                torch.cuda.empty_cache()
                lipsync_path = generate_lip_sync(
                    face_path=optimized_face,
                    audio_path=audio_path,
                    output_path=lipsync_out
                )
            except Exception as e:
                print(f"   ⚠ SadTalker failed: {e}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        # 🥉 TIER 3: Raw Fallback (Audio + Original Video)
        if not lipsync_path:
            print("   ↳ ⚠ ALL AI Engines Failed. Falling back to RAW video...")
            shutil.copy(face_path, lipsync_out)
            lipsync_path = lipsync_out
        
        # 🟢 STEP 3: Save Results
        # IMPORTANT: Copy files FIRST, before any cleanup can happen
        # Kaggle outputs from /kaggle/working/ — copy to that dir
        output_root = os.path.join(os.getcwd(), "..")
        
        results = {
            "audio_path": os.path.basename(audio_path),
            "duration": duration,
            "word_timestamps": word_timestamps,
            "lipsync_path": os.path.basename(lipsync_path) if lipsync_path else None
        }
        
        # Final Output Transfer — copy to Kaggle's /kaggle/working/ for download
        try:
            rel_audio_path = audio_path.split("YtDidYouKnowByVJ/")[-1] 
            audio_src = os.path.join(os.getcwd(), rel_audio_path)
            if os.path.exists(audio_src):
                shutil.copy(audio_src, output_root)
                print(f"   Copied audio: {audio_src} → {output_root}")
            else:
                print(f"   ⚠ Audio file not found at: {audio_src}")
            
            lipsync_src = os.path.join(os.getcwd(), lipsync_out)
            if os.path.exists(lipsync_src):
                shutil.copy(lipsync_src, output_root)
                print(f"   Copied lipsync: {lipsync_src} → {output_root}")
            else:
                print(f"   ⚠ Lipsync file not found at: {lipsync_src}")
        except Exception as copy_err:
            print(f"   ⚠ File copy failed: {copy_err}")
            
        with open(os.path.join(output_root, "results.json"), "w") as f:
            json.dump(results, f)
            
        print("✅ GPU Processing Complete.")

    finally:
        # HUGE OPTIMIZATION: Ensure cleanup happens even on failure
        print("🧹 Cleaning up repositories and models to speed up download...")
        os.chdir("..")
        if os.path.isdir("YtDidYouKnowByVJ"):
            shutil.rmtree("YtDidYouKnowByVJ", ignore_errors=True)
        if os.path.isdir("SadTalker"):
            shutil.rmtree("SadTalker", ignore_errors=True)
        if os.path.isdir("MuseTalk"):
            shutil.rmtree("MuseTalk", ignore_errors=True)

if __name__ == "__main__":
    print("--- Kaggle Worker Initiated ---")
    setup_project()
    os.chdir("YtDidYouKnowByVJ") 
    setup_musetalk()
    setup_sadtalker()
    process_job()
    print("--- Job Finished ---")
