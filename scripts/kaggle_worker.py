import os
import subprocess
import shutil
import sys
import time

# 🚀 KAGGLE GPU WORKER FOR YtDidYouKnowByVJ
# Designed to run on Kaggle T4 x2 or P100

def run_cmd(cmd, cwd=None):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)

def setup_sadtalker():
    if not os.path.isdir("SadTalker"):
        print("📥 Cloning SadTalker...")
        run_cmd(["git", "clone", "https://github.com/OpenTalker/SadTalker.git"])
        
        # Apply patches
        print("🛠️ Patching SadTalker for modern environments...")
        # (Same patches as used in GitHub Actions)
        arch_file = "SadTalker/src/face3d/util/my_awing_arch.py"
        if os.path.exists(arch_file):
            with open(arch_file, 'r') as f: content = f.read()
            with open(arch_file, 'w') as f: f.write(content.replace("np.float", "float"))
            
        prep_file = "SadTalker/src/face3d/util/preprocess.py"
        if os.path.exists(prep_file):
            with open(prep_file, 'r') as f: content = f.read()
            import re
            content = re.sub(r'trans_params = np\.array\(\[w0, h0, s, t\[0\], t\[1\]\]\)', 
                            'trans_params = np.array([w0, h0, s, t[0].item(), t[1].item()])', content)
            content = content.replace('.astype(np.int32)', '.astype(int)')
            with open(prep_file, 'w') as f: f.write(content)

        # Download weight (Efficiently)
        print("📥 Downloading SadTalker Weights...")
        os.makedirs("SadTalker/checkpoints", exist_ok=True)
        # Using a faster mirror or direct download if possible
        run_cmd(["bash", "scripts/download_models.sh"], cwd="SadTalker")

def setup_project():
    if not os.path.isdir("YtDidYouKnowByVJ"):
        print("📥 Cloning Project Repository...")
        run_cmd(["git", "clone", "https://github.com/vjaab/YtDidYouKnowByVJ.git"])
    
    run_cmd(["pip", "install", "-r", "requirements.txt"], cwd="YtDidYouKnowByVJ")
    run_cmd(["pip", "install", "f5-tts", "stable-ts", "torch", "torchvision<0.17.0", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])

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

    # Reconstruct the script/audio state
    # On Kaggle, we want to run the heavy steps:
    # 1. generate_voiceover (via F5-TTS with GPU)
    # 2. generate_lip_sync (via SadTalker with GPU)
    
    sys.path.append(os.getcwd())
    from audio_gen import generate_voiceover
    from lip_sync import generate_lip_sync
    
    script = job_data.get("script")
    voice = job_data.get("voice")
    emotion = job_data.get("emotion")
    custom_map = job_data.get("custom_map")
    
    # 1. GPU Audio
    audio_path, duration, word_timestamps = generate_voiceover(
        script, voice, emotion, custom_phonetic_map=custom_map
    )
    
    # 2. GPU Lip-Sync
    face_path = "assets/Firefly_video_final.mp4"
    lipsync_out = "kaggle_lipsync.mp4"
    
    lipsync_path = generate_lip_sync(
        face_path=face_path,
        audio_path=audio_path,
        output_path=lipsync_out
    )
    
    # 3. Save Results for Pipeline Retrieval (Placing results.json in the kaggle root)
    results = {
        "audio_path": os.path.basename(audio_path),
        "duration": duration,
        "word_timestamps": word_timestamps,
        "lipsync_path": os.path.basename(lipsync_path) if lipsync_path else None
    }
    
    # Move outputs to /kaggle/working/ so the CLI correctly downloads them raw
    os.chdir("..")
    
    # Copy assets before nuking the folders
    # Note: audio_path comes as absolute from output/, so we extract the relative path from YtDidYouKnowByVJ 
    rel_audio_path = audio_path.split("YtDidYouKnowByVJ/")[-1] 
    
    shutil.copy(os.path.join("YtDidYouKnowByVJ", rel_audio_path), ".")
    if lipsync_path and os.path.exists(os.path.join("YtDidYouKnowByVJ", lipsync_path)):
        shutil.copy(os.path.join("YtDidYouKnowByVJ", lipsync_path), ".")
        
    # HUGE OPTIMIZATION: Kaggle downloads EVERYTHING in /kaggle/working/
    # We must wipe the repo and heavy model weights to prevent downloading ~3GB of files locally
    print("🧹 Cleaning up repositories and models to speed up download...")
    if os.path.isdir("YtDidYouKnowByVJ"):
        shutil.rmtree("YtDidYouKnowByVJ", ignore_errors=True)
    if os.path.isdir("SadTalker"):
        shutil.rmtree("SadTalker", ignore_errors=True)
        
    with open("results.json", "w") as f:
        json.dump(results, f)
    
    print("✅ GPU Processing Complete.")

if __name__ == "__main__":
    print("--- Kaggle Worker Initiated ---")
    setup_project()
    # Change into the project directory so all relative paths align perfectly for lip_sync.py and Kaggle CLI
    os.chdir("YtDidYouKnowByVJ") 
    setup_sadtalker()
    process_job()
    print("--- Job Finished ---")
