import os
os.environ["PYTHONHASHSEED"] = "random"
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


def _patch_mmengine():
    """
    Monkey-patch mmengine to prevent KeyError: 'Adafactor is already registered'
    This happens when mmengine and modern torch (2.2+) both try to register Adafactor.
    """
    print("🛠️ Patching mmengine registry for Adafactor compatibility...")
    try:
        import importlib
        import mmengine
        importlib.reload(mmengine)
        
        import mmengine.optim.optimizer.builder as builder
        if hasattr(builder, 'register_transformers_optimizers'):
            orig_reg = builder.register_transformers_optimizers
            def safe_register():
                try:
                    orig_reg()
                except KeyError:
                    pass # Already registered
            builder.register_transformers_optimizers = safe_register
            # Force run it now to claim the territory
            safe_register()
            print("   ✅ mmengine Adafactor patch applied & reloaded")
    except Exception as e:
        print(f"   ⚠ Could not patch mmengine: {e}")


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
    
    # Step 0: Ensure packaging tools and global stubs are ready
    try:
        run_cmd(["pip", "install", "-q", "numpy<2.0", "setuptools<70", "packaging", "wheel", "yapf==0.40.1"])
        
        # 🛡️ Global distutils stub for Python 3.12
        import site
        sp = site.getsitepackages()[0]
        dist_dir = os.path.join(sp, "distutils")
        os.makedirs(dist_dir, exist_ok=True)
        with open(os.path.join(dist_dir, "__init__.py"), "w") as f:
            f.write("# Global distutils stub\n")
        with open(os.path.join(dist_dir, "version.py"), "w") as f:
            f.write("from packaging.version import parse as LooseVersion\n")
        print("   ✅ Global distutils/version.py stub created")
    except:
        print("   ⚠ Global stubs setup failed")
    
    # Step 1: Clean start (except mmcv which we might want to keep if wheels are rare)
    try:
        run_cmd(["pip", "uninstall", "-y", "mmdet", "mmpose", "mmengine"])
    except:
        pass

    # Step 2: mmengine (Forced 0.10.4 for Py3.12 compatibility) 
    try:
        run_cmd(["pip", "install", "-q", "mmengine==0.10.4", "--force-reinstall"])
        print("   ✅ mmengine==0.10.4 (forced)")
        
        # ── EXPERT PATCH: Physical Disk Patch for Adafactor KeyError ──────
        import site
        sp = site.getsitepackages()[0]
        builder_path = os.path.join(sp, "mmengine", "optim", "optimizer", "builder.py")
        if os.path.exists(builder_path):
            with open(builder_path, "r") as f:
                content = f.read()
            
            old_pattern = "    OPTIMIZERS.register_module(name='Adafactor', module=Adafactor)"
            new_pattern = """    try:
        OPTIMIZERS.register_module(name='Adafactor', module=Adafactor)
    except KeyError:
        pass  # Already registered by PyTorch 2.2+, skip"""
            
            if old_pattern in content and "except KeyError" not in content:
                content = content.replace(old_pattern, new_pattern)
                with open(builder_path, "w") as f:
                    f.write(content)
                print("   ✅ Physical Adafactor patch applied to builder.py")
                
                # Clear bytecode cache
                pyc_path = os.path.join(sp, "mmengine", "optim", "optimizer", "__pycache__", "builder.cpython-312.pyc")
                if os.path.exists(pyc_path):
                    os.remove(pyc_path)
                    print("   ✅ Cleared mmengine pyc cache")
            else:
                print("   ⚠ builder.py already patched or pattern not found")
    except Exception as e:
        print(f"   ❌ mmengine setup/patch failed: {e}")
    
    # Step 2: mmcv (needs CUDA ops — use pre-built wheels)
    import torch
    torch_v_full = torch.__version__
    torch_v_simple = torch_v_full.split('+')[0]
    cuda_v = torch.version.cuda
    
    print(f"🔍 Environment Check:")
    print(f"   PyTorch: {torch_v_full}")
    print(f"   CUDA: {cuda_v}")
    
    if cuda_v:
        cuda_tag = "cu" + cuda_v.replace(".", "")
    else:
        cuda_tag = "cpu"
    
    mmcv_installed = False
    
    # Check if mmcv is already installed (Kaggle often pre-installs compatible versions)
    try:
        import mmcv
        print(f"   ✓ Existing mmcv {mmcv.__version__} detected")
        mmcv_installed = True
    except:
        pass

    if not mmcv_installed or int(torch_v_simple.split('.')[0]) < 2 or int(torch_v_simple.split('.')[1]) < 9:
        # Only try re-installing mmcv if missing or on older Torch versions where wheels exist
        if int(torch_v_simple.split('.')[0]) >= 2 and int(torch_v_simple.split('.')[1]) >= 9:
            print("   ⚠️ Torch 2.9+ detected. Skipping mmcv reinstall (no prebuilt wheels yet).")
        else:
            # Strategy: Try the OpenMMLab index first with various torch version tags.
            # Kaggle might have Torch 2.5, but indexes usually stop at 2.1/2.2/2.3/2.4.
            # We try them in descending order of recency.
            for trial_v in [torch_v_simple, "2.4.0", "2.3.0", "2.2.0", "2.1.0"]:
                # Also try common CUDA tags if exact tag fails (e.g. cu121 often works on cu124)
                for trial_cuda in [cuda_tag, "cu121", "cu118"]:
                    try:
                        mm_index = f"https://download.openmmlab.com/mmcv/dist/{trial_cuda}/torch{trial_v}/index.html"
                        print(f"   Checking: {mm_index}")
                        run_cmd(["pip", "install", "-q", "mmcv==2.1.0", "-f", mm_index])
                        mmcv_installed = True
                        print(f"   ✅ mmcv==2.1.0 (via {trial_cuda}/torch{trial_v})")
                        break
                    except:
                        continue
                if mmcv_installed: break

    # Try MiroPsota pre-built wheels if OpenMMLab fails
    if not mmcv_installed:
        try:
            run_cmd([
                "pip", "install", "-q",
                "--extra-index-url", "https://miropsota.github.io/torch_packages_builder",
                "mmcv>=2.1.0,<2.2.0"
            ])
            mmcv_installed = True
            print("   ✅ mmcv (via MiroPsota wheels)")
        except:
            pass
    
    if not mmcv_installed:
        try:
            run_cmd(["pip", "install", "-q", "mmcv-lite>=2.1.0,<2.2.0"])
            mmcv_installed = True
            print("   ✅ mmcv-lite (fallback)")
        except:
            print("   ❌ mmcv completely failed")
            
    # CRITICAL: If we have mmcv (lite or full) but _ext is missing, stub it.
    try:
        import site
        sp = site.getsitepackages()[0]
        mmcv_path = os.path.join(sp, "mmcv")
        if os.path.exists(mmcv_path):
            ext_file = os.path.join(mmcv_path, "_ext.py")
            # Create a more robust stub that allows importing submodules
            print("   🛠️ Hardening mmcv._ext to prevent import crash...")
            with open(ext_file, "w") as f:
                f.write("# Robust physical stub for mmcv._ext\n")
                f.write("import sys\n")
                f.write("from types import ModuleType\n\n")
                f.write("class StubModule(ModuleType):\n")
                f.write("    def __getattr__(self, name):\n")
                f.write("        if name == '__path__': return []\n")
                f.write("        if name == '__all__': return []\n")
                f.write("        if name in ('__file__', '__name__', '__package__'): return ''\n")
                f.write("        return lambda *args, **kwargs: None\n\n")
                f.write("sys.modules['mmcv._ext'] = StubModule('mmcv._ext')\n")
                # Also stub the ops submodule which is often checked
                ops_dir = os.path.join(mmcv_path, "ops")
                os.makedirs(ops_dir, exist_ok=True)
                with open(os.path.join(ops_dir, "__init__.py"), "a") as f_ops:
                    f_ops.write("\n# Subprocess Import Protection\n")
                    f_ops.write("try:\n    from . import _ext\nexcept:\n    pass\n")
            print("   ✅ mmcv._ext hardened")
    except Exception as e:
        print(f"   ⚠ Failed to stub mmcv._ext: {e}")
    
    # Step 3: mmdet (Pinned to 3.3.0)
    try:
        run_cmd(["pip", "install", "-q", "mmdet==3.3.0"])
        print("   ✅ mmdet==3.3.0")
    except Exception as e:
        print(f"   ⚠ mmdet failed: {e}")
    
    # Step 4: mmpose (Pinned to 1.3.2)
    try:
        run_cmd(["pip", "install", "-q", "--no-deps", "mmpose==1.3.2"])
        print("   ✅ mmpose==1.3.2")
    except Exception as e:
        print(f"   ⚠ mmpose failed: {e}")
        # Install secondary deps individually (skipping broken xtcocotools build)
        for dep in ["munkres", "json_tricks", "scipy", "shapely", "face-alignment", "pycocotools"]:
            run_cmd(["pip", "install", "-q", dep])
        
        # Install chumpy with relaxed build isolation (needed by mmpose in subprocesses)
        try:
            run_cmd(["pip", "install", "-q", "--no-build-isolation", "chumpy"])
            print("   ✅ chumpy installed")
        except:
            # Create a more generous stub for chumpy to avoid AttributeErrors
            import site
            sp = site.getsitepackages()[0]
            chumpy_dir = os.path.join(sp, "chumpy")
            os.makedirs(chumpy_dir, exist_ok=True)
            with open(os.path.join(chumpy_dir, "__init__.py"), "w") as f:
                f.write("# Robust stub for chumpy\n")
                f.write("__version__ = '0.70'\n")
                f.write("import sys\n")
                f.write("from types import ModuleType\n\n")
                f.write("class Stub(ModuleType):\n")
                f.write("    def __getattr__(self, name):\n")
                f.write("        if name in ('__file__', '__name__', '__package__'): return ''\n")
                f.write("        if name == '__path__': return []\n")
                f.write("        return lambda *a, **k: Stub(name)\n")
                f.write("    def __call__(self, *a, **k): return Stub('call')\n")
                f.write("sys.modules['chumpy'] = Stub('chumpy')\n")
            print("   ✅ chumpy robust stub created")
        
        # Install xtcocotools
        try:
            run_cmd(["pip", "install", "-q", "Cython"])
            run_cmd(["pip", "install", "-q", "--no-build-isolation", "xtcocotools"])
            print("   ✅ xtcocotools")
        except:
            print("   ⚠ xtcocotools failed, creating robust stub package...")
            import site
            sp = site.getsitepackages()[0]
            xtcoco_dir = os.path.join(sp, "xtcocotools")
            os.makedirs(xtcoco_dir, exist_ok=True)
            with open(os.path.join(xtcoco_dir, "__init__.py"), "w") as f:
                f.write("# Robust stub for xtcocotools\n")
            with open(os.path.join(xtcoco_dir, "coco.py"), "w") as f:
                f.write("class COCO:\n")
                f.write("    def __init__(self, *a, **k): pass\n")
                f.write("    def __getattr__(self, name): return lambda *a, **k: []\n")
            print("   ✅ xtcocotools robust stub created")
        
        # Install mmpose core
        run_cmd(["pip", "install", "-q", "--no-deps", "mmpose"])
        print("   ✅ mmpose core installed")
    except Exception as e:
        print(f"   ⚠ mmpose setup had issues: {e}")




def _download_musetalk_weights():
    """
    Download MuseTalk model weights from multiple HuggingFace repos.
    
    The TMElyralab/MuseTalk HF repo has files at ROOT (no models/ prefix):
      musetalk/musetalk.json, musetalk/pytorch_model.bin
      musetalkV15/musetalk.json, musetalkV15/unet.pth
    
    Other weights come from separate repos:
      stabilityai/sd-vae-ft-mse  → models/sd-vae/
      openai/whisper-tiny        → models/whisper/
      yzd-v/DWPose               → models/dwpose/
    """
    print("📥 Downloading MuseTalk model weights...")
    
    models_dir = os.path.join("MuseTalk", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        
        # ── 1. MuseTalk core model (v1.0 + v1.5) ────────────────────────
        # Repo has files like: musetalkV15/unet.pth (NO models/ prefix)
        # Download into MuseTalk/models/ so path becomes models/musetalkV15/unet.pth
        print("   [1/5] Downloading MuseTalk model weights...")
        snapshot_download(
            repo_id='TMElyralab/MuseTalk',
            local_dir=models_dir,
            allow_patterns=['musetalk/*', 'musetalkV15/*']
        )

        # ── 2. SD-VAE (Stable Diffusion VAE) ─────────────────────────────
        # MuseTalk's load_all_model() uses vae_type="sd-vae" → models/sd-vae/
        print("   [2/5] Downloading SD-VAE weights...")
        sd_vae_dir = os.path.join(models_dir, "sd-vae")
        os.makedirs(sd_vae_dir, exist_ok=True)
        snapshot_download(
            repo_id='stabilityai/sd-vae-ft-mse',
            local_dir=sd_vae_dir,
            allow_patterns=['config.json', 'diffusion_pytorch_model.bin', 'diffusion_pytorch_model.safetensors']
        )
        
        # ── 3. Whisper (audio encoder) ───────────────────────────────────
        print("   [3/5] Downloading Whisper weights...")
        whisper_dir = os.path.join(models_dir, "whisper")
        os.makedirs(whisper_dir, exist_ok=True)
        snapshot_download(
            repo_id='openai/whisper-tiny',
            local_dir=whisper_dir,
            allow_patterns=['config.json', 'pytorch_model.bin', 'preprocessor_config.json',
                          'model.safetensors', 'tokenizer.json', 'vocab.json', 'merges.txt',
                          'normalizer.json', 'special_tokens_map.json', 'added_tokens.json']
        )
        
        # ── 4. DWPose (body pose estimation) ─────────────────────────────
        print("   [4/5] Downloading DWPose weights...")
        dwpose_dir = os.path.join(models_dir, "dwpose")
        os.makedirs(dwpose_dir, exist_ok=True)
        try:
            hf_hub_download(
                repo_id='yzd-v/DWPose',
                filename='dw-ll_ucoco_384.pth',
                local_dir=dwpose_dir
            )
        except Exception as e:
            print(f"   ⚠ DWPose download failed: {e}")
        
        # ── 5. Face Parse BiSeNet ────────────────────────────────────────
        print("   [5/5] Downloading Face Parse weights...")
        face_parse_dir = os.path.join(models_dir, "face-parse-bisent")
        os.makedirs(face_parse_dir, exist_ok=True)
        try:
            # gdown for Google Drive file
            run_cmd([
                "gdown", "--id", "154JgKpzCPW82qINcVieuPH3fZ2e0P812",
                "-O", os.path.join(face_parse_dir, "79999_iter.pth")
            ])
        except:
            # Fallback for newer gdown versions
            try:
                run_cmd([
                    "gdown", "154JgKpzCPW82qINcVieuPH3fZ2e0P812",
                    "-O", os.path.join(face_parse_dir, "79999_iter.pth")
                ])
            except:
                print("   ⚠ Face parse gdown failed")
        try:
            run_cmd([
                "curl", "-sL",
                "https://download.pytorch.org/models/resnet18-5c106cde.pth",
                "-o", os.path.join(face_parse_dir, "resnet18-5c106cde.pth")
            ])
        except:
            print("   ⚠ ResNet18 download failed")
        
        # ── Verify critical files ────────────────────────────────────────
        critical = {
            "MuseTalk UNet v1.5": os.path.join(models_dir, "musetalkV15", "unet.pth"),
            "SD-VAE": os.path.join(models_dir, "sd-vae", "diffusion_pytorch_model.bin"),
            "Whisper": os.path.join(models_dir, "whisper", "pytorch_model.bin"),
            "DWPose": os.path.join(models_dir, "dwpose", "dw-ll_ucoco_384.pth"),
        }
        all_ok = True
        for name, path in critical.items():
            # Also check for safetensors variant
            alt_path = path.replace('.bin', '.safetensors')
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024*1024)
                print(f"   ✅ {name}: {size_mb:.1f}MB")
            elif os.path.exists(alt_path):
                size_mb = os.path.getsize(alt_path) / (1024*1024)
                print(f"   ✅ {name}: {size_mb:.1f}MB (safetensors)")
            else:
                print(f"   ❌ MISSING: {name} ({path})")
                all_ok = False
        
        if all_ok:
            print("   ✅ All weights verified.")
        else:
            print("   ⚠ Some weights missing — MuseTalk may fall back to SadTalker.")
            
    except Exception as e:
        print(f"   ❌ Weight download failed: {e}")
        import traceback
        traceback.print_exc()


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
        
        # FIX: np.float attribute error in modern numpy
        awing_file = "SadTalker/src/face3d/util/my_awing_arch.py"
        if os.path.exists(awing_file):
            with open(awing_file, 'r') as f:
                content = f.read()
            if "np.float" in content:
                content = content.replace("np.float", "float")
                with open(awing_file, 'w') as f:
                    f.write(content)
                print("   ✅ Patched SadTalker my_awing_arch.py (np.float → float)")

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
    if not os.path.exists("YtDidYouKnowByVJ"):
        run_cmd(["git", "clone", "-q", "https://github.com/vjaab/YtDidYouKnowByVJ.git"])
    
    if os.path.isdir("YtDidYouKnowByVJ"):
        os.chdir("YtDidYouKnowByVJ")
        print(f"🏠 Switched to project directory: {os.getcwd()}")
    else:
        print("⚠️ Warning: Could not find project directory 'YtDidYouKnowByVJ' after clone.")

    
    # ── PYTHON DEPENDENCIES ────────────────────────────────────────────────
    print("📦 Installing Python Dependencies...")
    run_cmd(["pip", "install", "-q", "-U", "pip", "numpy<2.0", "setuptools<70", "wheel", "packaging"])
    run_cmd(["pip", "install", "-q", "grpcio==1.62.2", "grpcio-status==1.62.2"]) # Fix for yanked versions
    run_cmd(["pip", "install", "-q", "-r", "requirements.txt"])
    
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
        
        # 🛡️ Apply Expert Runtime Patches
        _patch_mmengine()
        os.environ["DWPOSE_DEVICE"] = "cuda" # Force DWPose to GPU
        
        # 🔍 Print Stack Versions for Debugging
        import torch
        print(f"🔍 System Check: PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
        try:
            import mmengine; print(f"🔍 mmengine: {mmengine.__version__}")
            import mmpose;   print(f"🔍 mmpose:   {mmpose.__version__}")
        except:
            print("🔍 MMLab imports failed — checking stubs...")

        from audio_gen import generate_voiceover, unload_f5_model
        from lip_sync import generate_lip_sync
        from musetalk_sync import generate_musetalk
        
        script = job_data.get("script")
        voice = job_data.get("voice")
        emotion = job_data.get("emotion")
        custom_map = job_data.get("custom_map")
        
        # 🖼️ Reference Frame Sanity Check
        face_path = "assets/Firefly_video_final.mp4"
        if not os.path.exists(face_path):
            print(f"❌ Face template missing: {face_path}")
            raise RuntimeError("Face template missing.")
        
        import cv2
        cap = cv2.VideoCapture(face_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError("Could not read reference frame from template video.")
        h, w = frame.shape[:2]
        print(f"✅ Template Video: {w}x{h} (OpenCV verified)")
        if h < 100 or w < 100:
            raise RuntimeError(f"Template video resolution too low ({w}x{h}). Face detection will fail.")
        
        # 🟢 STEP 1: GPU Audio (F5-TTS) — HARD REQUIREMENT
        audio_path, duration, word_timestamps = None, 0, []
        try:
            audio_path, duration, word_timestamps = generate_voiceover(
                script, voice, emotion, custom_phonetic_map=custom_map
            )
        except Exception as e:
            print(f"❌ F5-TTS Voice Cloning FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Pipeline aborted: F5-TTS voice cloning failed — {e}")
        
        if not audio_path or not os.path.exists(audio_path):
            raise RuntimeError("Pipeline aborted: F5-TTS produced no audio output.")
        
        # Verify the audio is actually from F5-TTS (WAV) not an Edge TTS fallback (MP3)
        if audio_path.endswith(".mp3"):
            raise RuntimeError("Pipeline aborted: F5-TTS failed silently — got MP3 fallback instead of WAV.")
        
        print(f"✅ F5-TTS succeeded: {audio_path} ({duration:.1f}s, {len(word_timestamps)} words)")
        
        # 🔓 Unload F5-TTS
        unload_f5_model()
        
        # 🟢 STEP 2: Prep Assets & Optimize
        face_path = "assets/Firefly_video_final.mp4"
        optimized_face = "assets/Firefly_video_optimized.mp4"
        lipsync_out = "kaggle_lipsync.mp4"
        
        print("🏎️ Optimizing template resolution (512px) for RAM safety...")
        run_cmd([
            "ffmpeg", "-y", "-i", face_path, 
            "-vf", "scale=512:-1", 
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            optimized_face
        ])
        
        # 🏅 MuseTalk Lip-Sync — HARD REQUIREMENT
        import gc
        import torch
        
        lipsync_path = None
        try:
            gc.collect()
            torch.cuda.empty_cache()
            lipsync_path = generate_musetalk(
                face_path=optimized_face,
                audio_path=audio_path,
                output_path=lipsync_out
            )
        except Exception as e:
            print(f"❌ MuseTalk Lip-Sync FAILED: {e}")
            raise RuntimeError(f"Pipeline aborted: MuseTalk lip-sync failed — {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        if not lipsync_path or not os.path.exists(lipsync_path):
            raise RuntimeError("Pipeline aborted: MuseTalk produced no lip-sync output.")
        
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
    setup_musetalk()
    # setup_sadtalker()
    process_job()
    print("--- Job Finished ---")
