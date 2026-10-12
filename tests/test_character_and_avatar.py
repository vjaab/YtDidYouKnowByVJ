"""Tests for VJ Alter-Ego Themed Skins (Option 2) and Co-Host Sidekick Mascot Duo (Option 1).

Validates:
- Preserving VJ identity anchors while adapting contextual skins (Hacker, Detective, Hologram).
- Co-host mascot selection and reactive animation clip generation.
- Cartoon dialogue engine emotion registry.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_gen import (
    get_vj_alter_ego_skin,
    get_cohost_sidekick_for_topic,
    _build_cohost_sidekick_clip,
    VJ_ALTER_EGO_REGISTRY,
    COHOST_MASCOT_REGISTRY,
)
from cartoon_dialogue_engine import VALID_EMOTIONS


def test_vj_alter_ego_skin_selection():
    # 1. Hacker / Exploit
    hacker = get_vj_alter_ego_skin(topic_text="Critical Zero-Day Exploit Found in iOS", category="Cybersecurity")
    assert hacker["skin"] == "hacker"
    assert "CYBER" in hacker["tag"]
    assert os.path.exists(hacker["path"])

    # 2. Detective / Secret
    detective = get_vj_alter_ego_skin(topic_text="Secret Hidden Setting in Android Revealed", category="Smartphones")
    assert detective["skin"] == "detective"
    assert "DETECTIVE" in detective["tag"]
    assert os.path.exists(detective["path"])

    # 3. Hologram / Quantum Singularity
    holo = get_vj_alter_ego_skin(topic_text="Quantum Singularity and 2050 AGI", category="Artificial Intelligence")
    assert holo["skin"] == "hologram"
    assert "HOLO" in holo["tag"]
    assert os.path.exists(holo["path"])

    # 4. Standard fallback
    standard = get_vj_alter_ego_skin(topic_text="Top 5 Productivity Tips", category="Productivity")
    assert standard["skin"] == "standard"


def test_cohost_mascot_selection_and_assets():
    # 1. Phone mascot
    phone = get_cohost_sidekick_for_topic(topic_text="iPhone 17 Pro Camera Features", category="Mobile")
    assert phone["mascot"] == "phone"
    assert "PHONE" in phone["tag"]
    assert len(phone["emotions"]) > 0

    # 2. Server mascot
    server = get_cohost_sidekick_for_topic(topic_text="Underwater Submarine Fiber Cables", category="Cloud")
    assert server["mascot"] == "server"
    assert "SERVER" in server["tag"]

    # 3. Chip mascot
    chip = get_cohost_sidekick_for_topic(topic_text="NVIDIA New Blackwell GPU Chip Architecture", category="Hardware")
    assert chip["mascot"] == "chip"
    assert "CHIP" in chip["tag"]

    # 4. Gaming controller mascot
    controller = get_cohost_sidekick_for_topic(topic_text="GTA 6 Ray Tracing Physics", category="Gaming")
    assert controller["mascot"] == "controller"
    assert "CONTROLLER" in controller["tag"]

    # 5. Default Byte robot
    byte_m = get_cohost_sidekick_for_topic(topic_text="Understanding Large Language Models", category="AI")
    assert byte_m["mascot"] == "byte"
    assert "BYTE" in byte_m["tag"]


def test_build_cohost_clip_and_shadow():
    mascot_info = get_cohost_sidekick_for_topic(topic_text="AI and Future Tech", category="AI")
    clip, shadow = _build_cohost_sidekick_clip(
        cohost_info=mascot_info,
        audio_duration=8.0,
        accent_color=(0, 240, 255),
        is_longform=False
    )
    assert clip is not None
    assert shadow is not None
    assert clip.mask is not None
    assert shadow.mask is not None

    # Verify frame dimensions match between visual clip and shadow
    frame_clip = clip.get_frame(0.5)
    frame_shadow = shadow.get_frame(0.5)
    assert frame_clip.shape == frame_shadow.shape


def test_cartoon_dialogue_engine_emotions():
    assert "hacker" in VALID_EMOTIONS
    assert "detective" in VALID_EMOTIONS
    assert "hologram" in VALID_EMOTIONS
