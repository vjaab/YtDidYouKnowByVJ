import os
import json
import time
import subprocess
import shutil


def _notify_kaggle_failure(message):
    """Send a Telegram alert when Kaggle GPU encounters issues."""
    try:
        from telegram_selector import notify_telegram
        notify_telegram(message, "🔴")
    except Exception as e:
        print(f"⚠️ Telegram notification failed: {e}")


def trigger_kaggle_gpu_job(script_data, custom_map):
    """
    Saves job data, pushes to Kaggle, waits for completion, and downloads results.
    
    Returns:
        dict with results on success, or dict with "error" key on failure:
            {"error": "queued_timeout"}  — job never left QUEUED (GPU quota/availability)
            {"error": "run_timeout"}     — job started but didn't finish in time
            {"error": "job_error"}       — Kaggle reported an execution error
            {"error": "push_failed"}     — couldn't push kernel to Kaggle
            {"error": "download_failed"} — job finished but results download failed
        Returns None only for unexpected/unknown failures.
    """
    print("🚀 Initiating Kaggle GPU Handover...")
    
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    job_file = os.path.join(scripts_dir, "job_data.json")
    
    # 1. Inject Job Data directly into the script
    from config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID
    job_payload = {
        "script": script_data.get("script"),
        "custom_map": custom_map,
        "elevenlabs_api_key": ELEVENLABS_API_KEY,
        "elevenlabs_voice_id": ELEVENLABS_VOICE_ID,
        "face_path": script_data.get("lipsync_face_path", "assets/video/Firefly_video_final.mp4")
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
        msg = f"Failed to push Kaggle kernel: {e}"
        print(f"❌ {msg}")
        _notify_kaggle_failure(f"🚨 Kaggle Push Failed\n\n{msg}\n\nPipeline will attempt cloud TTS fallback.")
        return {"error": "push_failed", "message": msg}

    # 3. Poll for Completion (with separate QUEUED vs RUNNING timeouts)
    kernel_id = "vijayakumarj/ytdidyouknowbyvj-gpu-worker"
    print(f"⌛ Waiting for Kaggle job ({kernel_id}) to finish...")
    
    max_queued_mins = 15   # Give up if stuck in QUEUED for 15 min (GPU unavailable)
    max_running_mins = 45  # Allow up to 45 min once the job actually starts running
    poll_interval_s = 30   # Poll every 30 seconds
    
    start_time = time.time()
    job_started_running = False
    running_start_time = None
    
    while True:
        elapsed_s = time.time() - start_time
        elapsed_mins = elapsed_s / 60
        
        try:
            status_output = subprocess.check_output(
                [kaggle_cmd, "kernels", "status", kernel_id], text=True
            )
            status_lower = status_output.strip().lower()
            print(f"   Status: {status_output.strip()}")
            
            # ── Job completed successfully ──
            if "complete" in status_lower:
                print("✅ Kaggle job finished!")
                break
            
            # ── Job failed with an error ──
            if "error" in status_lower:
                msg = f"Kaggle job '{kernel_id}' reported an error after {elapsed_mins:.0f} min."
                print(f"❌ {msg}")
                _notify_kaggle_failure(
                    f"🚨 Kaggle GPU Job Error\n\n{msg}\n\n"
                    f"Pipeline will attempt cloud TTS fallback."
                )
                return {"error": "job_error", "message": msg}
            
            # ── Track state transition: QUEUED → RUNNING ──
            is_queued = "queued" in status_lower
            is_running = "running" in status_lower
            
            if is_running and not job_started_running:
                job_started_running = True
                running_start_time = time.time()
                print(f"   🟢 Job started running after {elapsed_mins:.1f} min in queue.")
            
            # ── QUEUED timeout: job never started ──
            if is_queued and not job_started_running and elapsed_s > (max_queued_mins * 60):
                msg = (
                    f"Kaggle job stuck in QUEUED for {elapsed_mins:.0f} min "
                    f"(limit: {max_queued_mins} min). GPU likely unavailable or quota exhausted."
                )
                print(f"❌ {msg}")
                _notify_kaggle_failure(
                    f"⏰ Kaggle GPU Queue Timeout\n\n{msg}\n\n"
                    f"💡 Check your Kaggle GPU quota at kaggle.com/settings\n"
                    f"Pipeline will attempt cloud TTS fallback (no lip-sync)."
                )
                return {"error": "queued_timeout", "message": msg}
            
            # ── RUNNING timeout: job is running but taking too long ──
            if job_started_running and running_start_time:
                running_elapsed_s = time.time() - running_start_time
                if running_elapsed_s > (max_running_mins * 60):
                    msg = (
                        f"Kaggle job running for {running_elapsed_s/60:.0f} min "
                        f"(limit: {max_running_mins} min). Possibly stuck or processing error."
                    )
                    print(f"❌ {msg}")
                    _notify_kaggle_failure(
                        f"⏰ Kaggle GPU Run Timeout\n\n{msg}\n\n"
                        f"Pipeline will attempt cloud TTS fallback (no lip-sync)."
                    )
                    return {"error": "run_timeout", "message": msg}
                    
        except Exception as e:
            print(f"⚠️ Error checking status: {e}")
            
        time.sleep(poll_interval_s)

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
        else:
            msg = "Kaggle job completed but results.json not found in output."
            print(f"❌ {msg}")
            _notify_kaggle_failure(f"🚨 Kaggle Results Missing\n\n{msg}")
            return {"error": "download_failed", "message": msg}
    except Exception as e:
        msg = f"Failed to download/process Kaggle results: {e}"
        print(f"❌ {msg}")
        _notify_kaggle_failure(f"🚨 Kaggle Download Failed\n\n{msg}")
        return {"error": "download_failed", "message": msg}
