"""Tests for Tier-3 and Tier-4 Shorts growth optimizations:
- Algorithmic hook candidate ranking (brevity, filler detection, tension words)
- Frame 0 visual preservation (avatar delayed until 2.2s for Shorts)
- Pipeline schedule aligned to peak viewing hours (08:00 UTC and 14:30 UTC)
"""
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_hook_scoring_prefers_concise_high_tension_hook():
    from gemini_script import score_hook_candidate

    # Candidate hooks
    long_filler_hook = {
        "text": "In this video today we are going to look at this brand new tool that everyone is using",
        "curiosity_score": 9,
        "swipe_stop_score": 9
    }
    short_tension_hook = {
        "text": "Stop using your phone like this",
        "curiosity_score": 8,
        "swipe_stop_score": 8
    }

    score_long = score_hook_candidate(long_filler_hook)
    score_short = score_hook_candidate(short_tension_hook)

    assert score_short > score_long, f"Short hook ({score_short}) should outrank long filler hook ({score_long})"
    assert score_short >= 80, f"Expected strong score for punchy hook, got {score_short}"


def test_workflow_cron_schedules_peak_hours():
    wf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".github", "workflows", "yt-shorts-pipeline.yml")
    with open(wf_path, "r") as f:
        content = f.read()

    # Verify peak UTC cron times (08:00 UTC = 1:30 PM IST, 14:30 UTC = 8:00 PM IST)
    assert "0 8 * * *" in content, "Missing Slot 1 peak cron '0 8 * * *'"
    assert "30 14 * * *" in content, "Missing Slot 2 peak cron '30 14 * * *'"
    # Verify dead-zone 00:00 UTC (5:30 AM IST) was removed
    assert "0 0 * * 2-6" not in content, "Dead-zone cron '0 0 * * 2-6' should be removed"


def test_frame_0_avatar_delayed_for_shorts():
    vg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video_gen.py")
    with open(vg_path, "r") as f:
        content = f.read()

    assert "avatar_start_time = 2.2" in content, "Shorts avatar must be delayed to 2.2s to preserve Frame 0"
