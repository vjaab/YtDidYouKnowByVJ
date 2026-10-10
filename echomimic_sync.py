#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
echomimic_sync.py — SOTA EchoMimic Landmark-Guided Audio-Driven Portrait Animation Engine
Generates ultra-expressive, natural portraits with editable head poses and realistic lip sync.
Based on BadToBest/EchoMimic.
"""

import os
import sys
import shutil
import time
import subprocess
import tempfile
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent

def is_echomimic_available():
    """Check if EchoMimic directory and model weights exist."""
    em_dir = BASE_DIR / "EchoMimic"
    if not em_dir.is_dir():
        return False
    
    weights_dir = em_dir / "pretrained_weights"
    unet_path = weights_dir / "denoising_unet.pth"
    ref_unet = weights_dir / "reference_unet.pth"
    motion_path = weights_dir / "motion_module.pth"
    
    return unet_path.exists() or ref_unet.exists() or motion_path.exists()


def generate_echomimic_sync(face_path, audio_path, output_path, timeout=10800):
    """
    Generate realistic audio-driven portrait animation with landmark guidance,
    head poses, and natural lip synchronisation using EchoMimic.
    """
    print(f"🎭 EchoMimic [SOTA]: Generating landmark-guided portrait animation...")
    print(f"   Face: {face_path}")
    print(f"   Audio: {audio_path}")

    em_dir = BASE_DIR / "EchoMimic"
    if not em_dir.is_dir():
        print("   ✗ EchoMimic directory not found.")
        return None

    face_path_abs = os.path.abspath(face_path)
    audio_path_abs = os.path.abspath(audio_path)
    output_path_abs = os.path.abspath(output_path)
    result_dir = os.path.join(str(em_dir), "results", "pipeline_run")
    os.makedirs(result_dir, exist_ok=True)

    # ── 1. Pre-process: Normalize audio to 16kHz Mono PCM ─────────────
    norm_audio = os.path.join(tempfile.gettempdir(), f"em_16k_mono_{os.path.basename(audio_path)}.wav")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path_abs,
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
            norm_audio
        ], check=True, capture_output=True)
        audio_path_abs = norm_audio
        print(f"   🎙️ Normalized audio to 16kHz mono: {norm_audio}")
    except Exception as e:
        print(f"   ⚠ Audio normalization notice: {e}")

    # ── 2. Pre-process: Extract source reference frame if video ───────
    conformed_face = os.path.join(tempfile.gettempdir(), f"em_face_{os.path.basename(face_path)}.png")
    try:
        if face_path_abs.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            subprocess.run([
                "ffmpeg", "-y", "-i", face_path_abs,
                "-vframes", "1", "-q:v", "1",
                conformed_face
            ], check=True, capture_output=True)
            source_image = conformed_face
        else:
            source_image = face_path_abs
        print(f"   🖼️ EchoMimic Reference Face: {source_image}")
    except Exception as e:
        print(f"   ⚠ Reference extraction fallback: {e}")
        source_image = face_path_abs

    # ── 3. Build EchoMimic Config ─────────────────────────────────────
    config_data = {
        "ref_image_path": source_image,
        "audio_path": audio_path_abs,
        "result_dir": result_dir,
        "W": 512,
        "H": 512,
        "cfg": 2.5,
        "steps": 30,
        "fps": 25,
    }
    config_file = os.path.join(tempfile.gettempdir(), "echomimic_run_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    cmd = [
        sys.executable, "-m", "infer_audio2vid",
        "--config", config_file
    ]

    print(f"   CMD: {' '.join(cmd)}")
    start_time = time.time()
    try:
        env = os.environ.copy()
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        result = subprocess.run(
            cmd, cwd=str(em_dir), capture_output=True, text=True,
            timeout=timeout, env=env
        )

        if result.returncode != 0:
            print(f"   ✗ EchoMimic inference failed (Code {result.returncode})")
            if result.stderr:
                print(f"   STDERR:\n{result.stderr[-800:]}")
            return None

        # Scan for output video
        generated = None
        for root, _, files in os.walk(result_dir):
            for f in sorted(files, reverse=True):
                if f.endswith(".mp4"):
                    generated = os.path.join(root, f)
                    break
            if generated:
                break

        if generated and os.path.exists(generated):
            shutil.copy2(generated, output_path_abs)
            dur = time.time() - start_time
            print(f"✅ EchoMimic animation succeeded in {dur:.1f}s → {output_path_abs}")
            return output_path_abs

        print("   ✗ EchoMimic output video not found.")
        return None

    except subprocess.TimeoutExpired:
        print(f"   ✗ EchoMimic timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"   ✗ EchoMimic execution error: {e}")
        return None
