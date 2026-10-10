"""
lip_sync.py — Unified lip-sync engine abstraction.

Now strictly focused on MuseTalk (High-quality GPU pipeline).
"""

import os
import sys

# ── Engine directory detection ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_lip_sync(face_path, audio_path, output_path, enhancer=None, timeout=10800):
    """
    Generate realistic audio-driven portrait animation / lip-sync.
    Priority order:
    1. LivePortrait (SOTA full-head pose, eye blinks, expression dynamics)
    2. EchoMimic (SOTA landmark-guided portrait diffusion)
    3. MuseTalk (High-performance latent UNet pipeline)
    """
    if not os.path.exists(face_path):
        print(f"🎭 Lip-sync: Face file not found: {face_path}")
        return None

    if not os.path.exists(audio_path):
        print(f"🎭 Lip-sync: Audio file not found: {audio_path}")
        return None

    # Priority 1: LivePortrait (Full head dynamics + lip-sync)
    try:
        from liveportrait_sync import generate_liveportrait_sync, is_liveportrait_available
        if is_liveportrait_available():
            print("🎭 Engine Selected: LivePortrait (SOTA Head Motion + Lips)")
            lp_out = generate_liveportrait_sync(face_path, audio_path, output_path, timeout=timeout)
            if lp_out and os.path.exists(lp_out):
                return lp_out
    except Exception as e:
        print(f"   ⚠ LivePortrait error: {e}")

    # Priority 2: EchoMimic (Landmark diffusion portrait animation)
    try:
        from echomimic_sync import generate_echomimic_sync, is_echomimic_available
        if is_echomimic_available():
            print("🎭 Engine Selected: EchoMimic (Landmark-Guided Diffusion)")
            em_out = generate_echomimic_sync(face_path, audio_path, output_path, timeout=timeout)
            if em_out and os.path.exists(em_out):
                return em_out
    except Exception as e:
        print(f"   ⚠ EchoMimic error: {e}")

    # Priority 3: MuseTalk (Optimized GPU latent lip-sync)
    try:
        from musetalk_sync import generate_musetalk_sync, is_musetalk_available
        if is_musetalk_available():
            print("🎭 Engine Selected: MuseTalk (Optimized GPU)")
            musetalk_out = generate_musetalk_sync(face_path, audio_path, output_path, timeout=timeout)
            if musetalk_out and os.path.exists(musetalk_out):
                return musetalk_out
        else:
            print("🎭 Lip-sync: MuseTalk is not available (Check GPU/Installation).")
    except ImportError as e:
        print(f"   ⚠ MuseTalk import failed: {e}")
    except Exception as e:
        print(f"   ⚠ MuseTalk error: {e}")

    # ── No engine succeeded ───────────────────────────────────────────────────
    print("🎭 Lip-sync engine failed or unavailable. Aborting.")
    return None


def get_available_engine():
    """Report which lip-sync engine will be used."""
    try:
        from liveportrait_sync import is_liveportrait_available
        if is_liveportrait_available():
            return "LivePortrait (SOTA Full Head Motion)"
    except:
        pass

    try:
        from echomimic_sync import is_echomimic_available
        if is_echomimic_available():
            return "EchoMimic (Landmark-Guided Diffusion)"
    except:
        pass

    try:
        from musetalk_sync import is_musetalk_available
        if is_musetalk_available():
            return "MuseTalk (Optimized GPU)"
    except:
        pass

    return None
