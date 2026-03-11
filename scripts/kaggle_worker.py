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
        
        # MuseTalk's own Python dependencies — but we MUST filter out packages
        # that conflict with Kaggle's pre-installed versions (numpy, torch, opencv, g2p_en)
        print("📦 Installing MuseTalk requirements (filtered)...")
        skip_packages = {"numpy", "opencv-python", "opencv-contrib-python", 
                         "torch", "torchvision", "torchaudio", "g2p-en", "g2p_en"}
        musetalk_req = os.path.join("MuseTalk", "requirements.txt")
        if os.path.exists(musetalk_req):
            with open(musetalk_req, "r") as f:
                lines = f.readlines()
            filtered = []
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith("#"):
                    continue
                # Extract package name (before any version specifier)
                pkg_name = line_stripped.split(">=")[0].split("<=")[0].split("==")[0].split("<")[0].split(">")[0].split("[")[0].strip()
                if pkg_name.lower() in skip_packages:
                    print(f"   ⏭️  Skipping '{line_stripped}' (already installed on Kaggle)")
                    continue
                filtered.append(line_stripped)
            
            # Write filtered requirements to a temp file
            filtered_req = os.path.join("MuseTalk", "requirements_filtered.txt")
            with open(filtered_req, "w") as f:
                f.write("\n".join(filtered))
            
            run_cmd(["pip", "install", "-q", "-r", filtered_req], cwd=".")
        
        # MMLab Dependencies (same --no-build-isolation fix as GHA)
        print("📦 Installing MMLab stack (mmcv, mmpose)...")
        run_cmd(["pip", "install", "-q", "-U", "openmim", "setuptools", "wheel"])
        run_cmd(["pip", "install", "-q", "chumpy", "--no-build-isolation"])
        run_cmd(["mim", "install", "mmengine"])
        run_cmd(["pip", "install", "-q", "mmcv>=2.0.1", "--no-build-isolation"])
        run_cmd(["mim", "install", "mmdet>=3.1.0"])
        run_cmd(["mim", "install", "mmpose>=1.1.0"])
        
        # Download model weights
        print("📥 Downloading MuseTalk model weights...")
        weight_script = os.path.join("MuseTalk", "download_weights.sh")
        if os.path.exists(weight_script):
            run_cmd(["bash", "download_weights.sh"], cwd="MuseTalk")
        else:
            print("   ⚠ download_weights.sh not found, weights must be manually placed.")
    else:
        print("✓ MuseTalk already set up.")

        
def setup_sadtalker():
    if not os.path.isdir("SadTalker"):
        print("📥 Cloning SadTalker...")
        run_cmd(["git", "clone", "-q", "https://github.com/OpenTalker/SadTalker.git"])
        
        # Apply patches
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

    print("🛠️ Patching basicsr for modern torchvision compatibility...")
    import site
    packages = site.getsitepackages()
    # Try all possible package locations
    for pkg_dir in packages:
        degradations_file = os.path.join(pkg_dir, "basicsr", "data", "degradations.py")
        if os.path.exists(degradations_file):
            print(f"Found and patching: {degradations_file}")
            with open(degradations_file, "r") as f:
                content = f.read()
            content = content.replace("functional_tensor", "functional")
            with open(degradations_file, "w") as f:
                f.write(content)
            break
    else:
        print("⚠ Could not find basicsr/data/degradations.py to patch.")

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
