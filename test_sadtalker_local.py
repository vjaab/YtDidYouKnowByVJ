
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
    
    # 2. Apply Technical Patches (Fixes ImportError and ValueError seen in CI)
    print("🛠️ Applying technical patches to SadTalker...")
    
    # Patch 1: np.float fix
    arch_file = "SadTalker/src/face3d/util/my_awing_arch.py"
    if os.path.exists(arch_file):
        with open(arch_file, 'r') as f:
            content = f.read()
        with open(arch_file, 'w') as f:
            f.write(content.replace("np.float", "float"))
        print(f"   ✅ Patched {arch_file}")

    # Patch 2: align_img inhomogeneous shape fix
    prep_file = "SadTalker/src/face3d/util/preprocess.py"
    if os.path.exists(prep_file):
        with open(prep_file, 'r') as f:
            content = f.read()
        target = "trans_params = np.array([w0, h0, s, t[0], t[1]])"
        replacement = "trans_params = np.array([w0, h0, s, t[0], t[1]], dtype=object).astype(float)"
        if target in content:
            with open(prep_file, 'w') as f:
                f.write(content.replace(target, replacement))
            print(f"   ✅ Patched {prep_file}")
        else:
            print(f"   ⚠️ Could not find target line in {prep_file}")

# 3. Install Dependencies
print("📦 Installing SadTalker dependencies...")
python_exe = "venv/bin/python3" if os.path.exists("venv/bin/python3") else sys.executable
subprocess.run([python_exe, "-m", "pip", "install", "-r", "SadTalker/requirements.txt", "--break-system-packages"], check=False)
subprocess.run([python_exe, "-m", "pip", "install", "gdown", "--break-system-packages"], check=False)

# 4. Download Models
print("📥 Downloading SadTalker models (~1GB)...")
subprocess.run(["bash", "scripts/download_models.sh"], cwd="SadTalker", check=True)

# 5. Run Inference using the project's lip_sync wrapper
print("🎬 Running Lip-Sync Inference...")
# We use the venv's python to ensure all dependencies are available
env = os.environ.copy()
env["PATH"] = os.path.abspath("venv/bin") + os.pathsep + env["PATH"]

from lip_sync import generate_lip_sync

# We'll use a longer timeout for local CPU/GPU rendering
result = generate_lip_sync(
    face_path=face_path,
    audio_path=audio_path,
    output_path=output_path,
    timeout=10800
)

if result and os.path.exists(result):
    print(f"✅ SUCCESS! Test video saved to: {result}")
else:
    print("❌ FAILED: Lip-sync generation did not produce an output.")
