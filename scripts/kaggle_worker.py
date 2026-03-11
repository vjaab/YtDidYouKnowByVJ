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

# ═══════════════════════════════════════════════════════════════════════════════
# BASICSR PATCH — Must run BEFORE any import of basicsr/gfpgan/facexlib
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_basicsr():
    """
    Patch basicsr for Python 3.12 compatibility.
    MUST use file-path scanning (not `import basicsr`) because importing
    basicsr triggers the very crash we're trying to fix.
    
    Fixes:
    1. torchvision.transforms.functional_tensor → functional (removed in torchvision 0.18+)
    2. distutils.version.LooseVersion → packaging.version.parse (removed in Python 3.12)
    """
    print("🛠️ Patching basicsr for Python 3.12 compatibility...")
    import site
    try:
        # Find basicsr WITHOUT importing it
        basicsr_dir = None
        for d in site.getsitepackages():
            candidate = os.path.join(d, "basicsr")
            if os.path.isdir(candidate):
                basicsr_dir = candidate
                break
        
        if not basicsr_dir:
            # Try common Kaggle path directly
            fallback = "/usr/local/lib/python3.12/dist-packages/basicsr"
            if os.path.isdir(fallback):
                basicsr_dir = fallback
        
        if not basicsr_dir:
            print("   ⚠ basicsr not found in site-packages.")
            return

        patched = False

        # Patch 1: functional_tensor → functional
        deg_file = os.path.join(basicsr_dir, "data", "degradations.py")
        if os.path.exists(deg_file):
            with open(deg_file, 'r') as f:
                content = f.read()
            if "functional_tensor" in content:
                content = content.replace(
                    "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                    "from torchvision.transforms.functional import rgb_to_grayscale"
                )
                with open(deg_file, 'w') as f:
                    f.write(content)
                patched = True
                print("   ✅ Patched degradations.py (functional_tensor → functional)")

        # Patch 2: distutils.version → packaging.version  
        arch_util = os.path.join(basicsr_dir, "archs", "arch_util.py")
        if os.path.exists(arch_util):
            with open(arch_util, 'r') as f:
                content = f.read()
            if "from distutils.version import LooseVersion" in content:
                content = content.replace(
                    "from distutils.version import LooseVersion",
                    "from packaging.version import parse as LooseVersion"
                )
                with open(arch_util, 'w') as f:
                    f.write(content)
                patched = True
                print("   ✅ Patched arch_util.py (distutils → packaging)")
        
        # Patch 3: Also check __init__.py for any distutils imports
        init_file = os.path.join(basicsr_dir, "__init__.py")
        if os.path.exists(init_file):
            with open(init_file, 'r') as f:
                content = f.read()
            if "distutils" in content:
                content = content.replace(
                    "from distutils.version import LooseVersion",
                    "from packaging.version import parse as LooseVersion"
                )
                with open(init_file, 'w') as f:
                    f.write(content)
                patched = True

        if not patched:
            print("   ✓ basicsr already patched or no patches needed.")
        
    except Exception as e:
        print(f"   ⚠ Basicsr patch error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MUSETALK SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_musetalk():
    if not os.path.isdir("MuseTalk"):
        print("📥 Cloning MuseTalk...")
        run_cmd(["git", "clone", "-q", "https://github.com/TMElyralab/MuseTalk.git"])
        
        # ═══════════════════════════════════════════════════════════════════
        # BYPASS MuseTalk's broken requirements.txt entirely.
        # It pins ancient versions (tensorflow==2.12.0, numpy==1.23.5, etc.)
        # that don't exist on Kaggle's Python 3.12.
        # ═══════════════════════════════════════════════════════════════════
        print("📦 Installing MuseTalk runtime dependencies (curated list)...")
        
        # Core ML deps that MuseTalk actually imports
        musetalk_deps = [
            "diffusers", "accelerate", "transformers", "huggingface_hub",
            "einops", "omegaconf", "soundfile", "librosa",
            "gradio", "gdown", "ffmpeg-python", "moviepy", "imageio[ffmpeg]",
        ]
        
        try:
            run_cmd(["pip", "install", "-q"] + musetalk_deps)
            print("   ✅ MuseTalk core deps installed")
        except Exception as e:
            print(f"   ⚠ Batch install failed: {e}")
            for dep in musetalk_deps:
                try:
                    run_cmd(["pip", "install", "-q", dep])
                except Exception:
                    print(f"   ⚠ Skipping {dep}")
        
        # ── MMLab Stack ──────────────────────────────────────────────────
        _install_mmlab()
        
        # ── Download MuseTalk model weights ──────────────────────────────
        _download_musetalk_weights()
    else:
        print("✓ MuseTalk already set up.")


def _install_mmlab():
    """
    Install MMLab packages (mmengine, mmcv, mmdet, mmpose) on Kaggle Python 3.12.
    
    Strategy:
    1. `mim` tool is BROKEN on Python 3.12 (pkgutil.ImpImporter removed)
    2. mmcv needs pre-built CUDA wheels — we use MiroPsota's third-party index
    3. mmengine/mmdet/mmpose can be installed via plain pip
    4. mmpose depends on chumpy which may fail — install without deps if needed
    """
    print("📦 Installing MMLab stack (bypassing broken mim)...")
    
    # Step 0: Ensure packaging tools are compatible
    try:
        run_cmd(["pip", "install", "-q", "setuptools<70", "packaging", "wheel"])
        print("   ✅ setuptools/packaging ready")
    except:
        print("   ⚠ setuptools fix failed")
    
    # Step 1: mmengine (pure Python, no CUDA ops)
    try:
        run_cmd(["pip", "install", "-q", "mmengine"])
        print("   ✅ mmengine")
    except Exception as e:
        print(f"   ❌ mmengine failed: {e}")
    
    # Step 2: mmcv (needs CUDA ops — use pre-built wheels)
    # Detect torch version for correct wheel index
    torch_ver = "2.5"
    try:
        import torch as _t
        torch_ver = '.'.join(_t.__version__.split('+')[0].split('.')[:2])
    except:
        pass
    
    mmcv_installed = False
    
    # Try 1: MiroPsota pre-built wheels (most reliable for Py3.12)
    print(f"   Installing mmcv for torch {torch_ver}...")
    try:
        run_cmd([
            "pip", "install", "-q",
            "--extra-index-url", "https://miropsota.github.io/torch_packages_builder",
            "mmcv"
        ])
        mmcv_installed = True
        print("   ✅ mmcv (via MiroPsota wheels)")
    except:
        pass
    
    # Try 2: OpenMMLab official index
    if not mmcv_installed:
        for cuda_ver in ["cu121", "cu124", "cu118"]:
            try:
                mm_index = f"https://download.openmmlab.com/mmcv/dist/{cuda_ver}/torch{torch_ver}/index.html"
                run_cmd(["pip", "install", "-q", "mmcv>=2.0.1", "-f", mm_index])
                mmcv_installed = True
                print(f"   ✅ mmcv (via OpenMMLab {cuda_ver}/torch{torch_ver})")
                break
            except:
                continue
    
    # Try 3: mmcv-lite (no CUDA ops, but still functional for inference)
    if not mmcv_installed:
        try:
            run_cmd(["pip", "install", "-q", "mmcv-lite"])
            mmcv_installed = True
            print("   ✅ mmcv-lite (fallback, no CUDA ops)")
        except:
            print("   ❌ mmcv completely failed — MuseTalk will not work")
    
    # Step 3: mmdet
    try:
        run_cmd(["pip", "install", "-q", "mmdet"])
        print("   ✅ mmdet")
    except Exception as e:
        print(f"   ⚠ mmdet failed: {e}")
    
    # Step 4: mmpose (depends on chumpy which is broken on Py3.12)
    try:
        run_cmd(["pip", "install", "-q", "mmpose"])
        print("   ✅ mmpose")
    except:
        # chumpy build fails on Py3.12 — try without deps
        try:
            run_cmd(["pip", "install", "-q", "--no-deps", "mmpose"])
            print("   ✅ mmpose (--no-deps)")
        except Exception as e:
            print(f"   ❌ mmpose failed: {e}")


def _download_musetalk_weights():
    """
    Download MuseTalk model weights via huggingface_hub.
    
    IMPORTANT: The HF repo has files like:
      models/musetalkV15/unet.pth
      models/sd-vae-ft-mse/diffusion_pytorch_model.bin  
      models/whisper/tiny.pt
      models/dwpose/dw-ll_ucoco_384.pth
    
    Our code checks: MuseTalk/models/musetalkV15/unet.pth
    So we download into MuseTalk/ (NOT MuseTalk/models/) because
    the repo already has the models/ prefix in its file paths.
    """
    print("📥 Downloading MuseTalk model weights...")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Download into MuseTalk/ root — repo files already have models/ prefix
        snapshot_download(
            repo_id='TMElyralab/MuseTalk',
            local_dir='MuseTalk',
            allow_patterns=[
                'models/musetalk/*',
                'models/musetalkV15/*', 
                'models/dwpose/*',
                'models/sd-vae-ft-mse/*',
                'models/whisper/*',
                'models/face-parse-bisent/*',
            ]
        )
        
        # Verify critical files exist
        critical_files = [
            "MuseTalk/models/musetalkV15/unet.pth",
            "MuseTalk/models/sd-vae-ft-mse/diffusion_pytorch_model.bin",
        ]
        for f in critical_files:
            if os.path.exists(f):
                size_mb = os.path.getsize(f) / (1024*1024)
                print(f"   ✅ {f} ({size_mb:.1f}MB)")
            else:
                print(f"   ⚠ MISSING: {f}")
        
        print("   ✅ Weight download complete.")
    except Exception as e:
        print(f"   ❌ HF download failed: {e}")
        # Last resort: bash script
        try:
            run_cmd(["bash", "download_weights.sh"], cwd="MuseTalk")
        except:
            print("   ❌ Weight download completely failed.")


# ═══════════════════════════════════════════════════════════════════════════════
# SADTALKER SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_sadtalker():
    if not os.path.isdir("SadTalker"):
        print("📥 Cloning SadTalker...")
        run_cmd(["git", "clone", "-q", "https://github.com/OpenTalker/SadTalker.git"])
        
        # SadTalker-specific patches
        print("🛠️ Patching SadTalker for modern environments...")
        prep_file = "SadTalker/src/face3d/util/preprocess.py"
        if os.path.exists(prep_file):
            with open(prep_file, 'r') as f:
                content = f.read()
            content = content.replace('np.VisibleDeprecationWarning', 'Warning')
            with open(prep_file, 'w') as f:
                f.write(content)

        print("📥 Downloading SadTalker Weights...")
        os.makedirs("SadTalker/checkpoints", exist_ok=True)
        run_cmd(["bash", "scripts/download_models.sh"], cwd="SadTalker", quiet=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_project():
    # ── SYSTEM DEPENDENCIES ──────────────────────────────────────────────────
    print("🖥️ Installing System Dependencies (espeak-ng, ffmpeg)...")
    try:
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
    run_cmd(["pip", "install", "-q", "-U", "pip", "setuptools<70", "wheel", "packaging"])
    run_cmd(["pip", "install", "-q", "-r", "requirements.txt"], cwd="YtDidYouKnowByVJ")
    
    # Force GPU-specific backends
    run_cmd(["pip", "install", "-q", 
        "onnxruntime-gpu", "espeakng-loader",
        "f5-tts", "stable-ts", "torch", "torchvision", "torchaudio", 
        "facexlib", "gfpgan", "basicsr", "av", "yacs", "kornia", 
        "librosa", "resampy", "imageio-ffmpeg", "pyyaml", "joblib", 
        "scikit-image", "safetensors", "trimesh", "face-alignment",
        "diffusers", "transformers", "accelerate", "g2p_en",
        "--extra-index-url", "https://download.pytorch.org/whl/cu121"])

    # ── CRITICAL: Patch basicsr BEFORE any engine imports ──────────────────
    _patch_basicsr()


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS JOB
# ═══════════════════════════════════════════════════════════════════════════════

def process_job():
    print("🎬 Starting GPU Job...")
    
    import json
    
    if "JOB_PAYLOAD" in globals():
        job_data = globals()["JOB_PAYLOAD"]
    elif os.path.exists("job_data.json"):
        with open("job_data.json", 'r') as f:
            job_data = json.load(f)
    else:
        print(f"⚠ Job Data not found. Running default demo mode.")
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
        output_root = os.path.join(os.getcwd(), "..")
        
        results = {
            "audio_path": os.path.basename(audio_path),
            "duration": duration,
            "word_timestamps": word_timestamps,
            "lipsync_path": os.path.basename(lipsync_path) if lipsync_path else None
        }
        
        # Final Output Transfer — copy to Kaggle's /kaggle/working/
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
