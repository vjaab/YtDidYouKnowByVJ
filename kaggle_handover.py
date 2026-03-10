import os
import json
import time
import subprocess
import shutil

def trigger_kaggle_gpu_job(script_data, voice, emotion, custom_map):
    """
    Saves job data, pushes to Kaggle, waits for completion, and downloads results.
    """
    print("🚀 Initiating Kaggle GPU Handover...")
    
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    job_file = os.path.join(scripts_dir, "job_data.json")
    
    # 1. Prepare Job Data
    job_payload = {
        "script": script_data.get("script"),
        "voice": voice,
        "emotion": emotion,
        "custom_map": custom_map
    }
    
    with open(job_file, "w") as f:
        json.dump(job_payload, f)
    
    # 2. Push Kernel
    print("📤 Pushing kernel to Kaggle...")
    try:
        # Use venv's kaggle if available, else system kaggle
        kaggle_cmd = "kaggle"
        if os.path.exists("venv/bin/kaggle"):
            kaggle_cmd = "venv/bin/kaggle"
            
        subprocess.run([kaggle_cmd, "kernels", "push", "-p", scripts_dir], check=True)
    except Exception as e:
        print(f"❌ Failed to push Kaggle kernel: {e}")
        return None

    # 3. Poll for Completion
    kernel_id = "vijayakumarj/ytdidyouknowbyvj-gpu-worker"
    print(f"⌛ Waiting for Kaggle job ({kernel_id}) to finish...")
    
    max_wait_mins = 60
    start_time = time.time()
    
    while (time.time() - start_time) < (max_wait_mins * 60):
        try:
            status_output = subprocess.check_output([kaggle_cmd, "kernels", "status", kernel_id], text=True)
            print(f"   Status: {status_output.strip()}")
            
            if "complete" in status_output.lower():
                 print("✅ Kaggle job finished!")
                 break
            if "error" in status_output.lower():
                 print("❌ Kaggle job failed!")
                 return None
        except Exception as e:
            print(f"⚠️ Error checking status: {e}")
            
        time.sleep(30) # Poll every 30s
    else:
        print("❌ Kaggle job timed out.")
        return None

    # 4. Download Results
    print("📥 Downloading results from Kaggle...")
    output_dir = "output"
    try:
        subprocess.run([kaggle_cmd, "kernels", "output", kernel_id, "-p", output_dir], check=True)
        
        results_file = os.path.join(output_dir, "results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results = json.load(f)
            
            # Map paths back to local project structure
            # Kaggle might save as './audio_...wav', we need to ensure they match expectations
            return results
    except Exception as e:
        print(f"❌ Failed to download Kaggle results: {e}")
        
    return None
