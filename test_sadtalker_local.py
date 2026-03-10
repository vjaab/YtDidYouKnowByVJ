
import os
import sys
import subprocess

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

face_path = "assets/Firefly_video_final.mp4"
audio_path = "assets/vj.wav"
output_path = "output/local_test_sadtalker.mp4"

print("🚀 Starting Local SadTalker Lip-Sync Test...")

# 1. Check if SadTalker is present
if not os.path.isdir("SadTalker"):
    print("📦 SadTalker not found. Cloning repository...")
    subprocess.run(["git", "clone", "https://github.com/OpenTalker/SadTalker.git"], check=True)
    
# ── APPLY TECHNICAL PATCHES ───────────────────────────────────────────────
print("🛠️ Applying technical patches to SadTalker...")

# Patch 1: np.float fix
arch_file = "SadTalker/src/face3d/util/my_awing_arch.py"
if os.path.exists(arch_file):
    with open(arch_file, 'r') as f:
        content = f.read()
    with open(arch_file, 'w') as f:
        f.write(content.replace("np.float", "float"))
    print(f"   ✅ Patched {arch_file}")

# Patch 2: align_img inhomogeneous shape fix and VisibleDeprecationWarning fix
import re
prep_file = "SadTalker/src/face3d/util/preprocess.py"
if os.path.exists(prep_file):
    with open(prep_file, 'r') as f:
        content = f.read()
    
    # fix 2.1: inhomogeneous shape
    content = re.sub(r'trans_params = np\.array\(\[w0, h0, s, t\[0\], t\[1\]\]\)', 
                     'trans_params = np.array([w0, h0, s, t[0].item(), t[1].item()])', content)
    # Also handle the already patched version if it exists
    content = content.replace('np.array([w0, h0, s, t[0], t[1]], dtype=object).astype(float)',
                              'np.array([w0, h0, s, t[0].item(), t[1].item()])')
    
    # fix 2.2: VisibleDeprecationWarning
    content = re.sub(r'warnings.filterwarnings\("ignore", category=np.VisibleDeprecationWarning\)', 
                     '# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)', content)

    # fix 2.3: np.int32 often deprecated/moved
    content = content.replace(".astype(np.int32)", ".astype(int)")

    # fix 2.4: float() or round() on 0-d array fails in Numpy 2.x
    # Handle both original and partially patched states
    content = re.sub(r'float\(\(t\[0\] - w0/2\)\*s\)', '((t[0] - w0/2)*s).item()', content)
    content = re.sub(r'float\(\(h0/2 - t\[1\]\)\*s\)', '((h0/2 - t[1])*s).item()', content)
    # Fallback for if float() was already stripped but .item() wasn't added
    content = content.replace("((t[0] - w0/2)*s)).astype(int)", "((t[0] - w0/2)*s).item())")
    content = content.replace("((h0/2 - t[1])*s)).astype(int)", "((h0/2 - t[1])*s).item())")

    with open(prep_file, 'w') as f:
        f.write(content)
    print(f"   ✅ Patched {prep_file}")

# Patch 3: SadTalker's own preprocess.py fixes
main_prep_file = "SadTalker/src/utils/preprocess.py"
if os.path.exists(main_prep_file):
    with open(main_prep_file, 'r') as f:
        content = f.read()
    
    # fix 3.1: float() on 0-d array during trans_params reconstruct
    content = content.replace("float(item)", "float(item.item())")

    with open(main_prep_file, 'w') as f:
        f.write(content)
    print(f"   ✅ Patched {main_prep_file}")

# 3. Install Dependencies
print("📦 Installing SadTalker dependencies...")
python_exe = "venv/bin/python3" if os.path.exists("venv/bin/python3") else sys.executable
subprocess.run([python_exe, "-m", "pip", "install", "-r", "SadTalker/requirements.txt", "--break-system-packages"], check=False)
subprocess.run([python_exe, "-m", "pip", "install", "gdown", "kornia", "face_alignment", "yacs", "ninja", "resampy", "facexlib", "basicsr", "--break-system-packages"], check=False)

# FIX basicsr for torchvision compatibility
print("🛠️ Patching basicsr for torchvision compatibility...")
site_pkg = subprocess.check_output([python_exe, "-c", "import site; print(site.getsitepackages()[0])"]).decode().strip()
basicsr_file = os.path.join(site_pkg, "basicsr", "data", "degradations.py")
if os.path.exists(basicsr_file):
    with open(basicsr_file, 'r') as f:
        content = f.read()
    with open(basicsr_file, 'w') as f:
        f.write(content.replace("functional_tensor", "functional"))
    print(f"   ✅ Patched {basicsr_file}")

# 4. Download Models
checkpoint_file = "SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors"
if not os.path.exists(checkpoint_file):
    print("📥 Downloading SadTalker models (~1GB)...")
    subprocess.run(["bash", "scripts/download_models.sh"], cwd="SadTalker", check=True)
else:
    print("✅ SadTalker checkpoints already present. Skipping download.")

# 5. Run Inference using the project's lip_sync wrapper
print("🎬 Running Lip-Sync Inference...")
# We use the venv's python to ensure all dependencies are available
env = os.environ.copy()
env["PATH"] = os.path.abspath("venv/bin") + os.pathsep + env["PATH"]

from lip_sync import generate_lip_sync

# We'll use a 30-minute timeout for local test to avoid long hangs
result = generate_lip_sync(
    face_path=face_path,
    audio_path=audio_path,
    output_path=output_path,
    timeout=1800 # 30 mins
)

if result and os.path.exists(result):
    print(f"✅ SUCCESS! Test video saved to: {result}")
else:
    print("❌ FAILED: Lip-sync generation did not produce an output.")
