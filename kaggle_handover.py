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
    
    # 1. Inject Job Data directly into the script
    job_payload = {
        "script": script_data.get("script"),
        "voice": voice,
        "emotion": emotion,
        "custom_map": custom_map
    }
    
    worker_script_path = os.path.join(scripts_dir, "kaggle_worker.py")
    with open(worker_script_path, "r") as f:
        worker_code = f.read()
    
    # We replace a specific string or inject at the top
    injection = f"\nJOB_PAYLOAD = {json.dumps(job_payload)}\n"
    
    # Let's write a temporary execution file that Kaggle will upload
    temp_script_path = os.path.join(scripts_dir, "ytdidyouknowbyvj_gpu_worker.py")
    with open(temp_script_path, "w") as f:
        f.write(injection + worker_code)
        
    # Update Metadata to point to the temp script
    meta_path = os.path.join(scripts_dir, "kernel-metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    meta["code_file"] = "ytdidyouknowbyvj_gpu_worker.py"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
        
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
            
            # Map downloaded basenames back to local 'output/' path as absolute paths
            if results.get("audio_path"):
                results["audio_path"] = os.path.abspath(os.path.join(output_dir, results["audio_path"]))
            if results.get("lipsync_path"):
                results["lipsync_path"] = os.path.abspath(os.path.join(output_dir, results["lipsync_path"]))
                
            return results
    except Exception as e:
        print(f"❌ Failed to download/process Kaggle results: {e}")
        
    return None
