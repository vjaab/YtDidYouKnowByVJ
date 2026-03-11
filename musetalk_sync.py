import os
import subprocess
import time

def generate_musetalk(face_path, audio_path, output_path, timeout=10800):
    """
    Standard interface for MuseTalk lip-sync.
    Requires MuseTalk to be installed in the current environment's path.
    """
    print(f"🎤 MuseTalk [P1]: Starting high-end lip-sync...")
    print(f"   Face: {face_path}")
    print(f"   Audio: {audio_path}")

    # Check if MuseTalk exists
    if not os.path.isdir("MuseTalk"):
        print("   ✗ MuseTalk directory not found.")
        return None

    # MuseTalk typically requires a specific inference script
    # We'll use a standard shell command to run it
    # We'll assume the setup is handled in kaggle_worker.py
    
    # Standard Kaggle Python path setup
    python_exe = "python3"
    
    # Prepare the MuseTalk command (assuming standard inference.py structure)
    cmd = [
        python_exe, "inference.py",
        "--video_path", face_path,
        "--audio_path", audio_path,
        "--output_path", output_path,
        "--fps", "24",
        "--batch_size", "8" # MuseTalk is quite memory efficient
    ]

    start_time = time.time()
    try:
        # Run in MuseTalk directory
        subprocess.run(cmd, cwd="MuseTalk", check=True, timeout=timeout)
        
        # MuseTalk output might be in a subdir depending on its inference.py
        # We'll check if output_path was created
        if os.path.exists(output_path):
            duration = time.time() - start_time
            print(f"✅ MuseTalk done in {duration:.1f}s")
            return output_path
        else:
            # Check for common MuseTalk output patterns (results/output.mp4)
            fallback_out = "MuseTalk/results/output.mp4"
            if os.path.exists(fallback_out):
                os.rename(fallback_out, output_path)
                return output_path
                
        print("   ✗ MuseTalk output not found.")
        return None
    except Exception as e:
        print(f"   ✗ MuseTalk failed: {e}")
        return None
