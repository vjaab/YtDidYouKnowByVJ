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
    Generate a lip-synced video using MuseTalk.
    """
    if not os.path.exists(face_path):
        print(f"🎭 Lip-sync: Face file not found: {face_path}")
        return None

    if not os.path.exists(audio_path):
        print(f"🎭 Lip-sync: Audio file not found: {audio_path}")
        return None

    # Try MuseTalk (Primary & Only Engine)
    try:
        from musetalk_sync import generate_musetalk_sync, is_musetalk_available
        if is_musetalk_available():
            print("🎭 Engine: MuseTalk (Strict)")
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
        from musetalk_sync import is_musetalk_available
        if is_musetalk_available():
            return "MuseTalk (GPU)"
    except:
        pass
    return None
