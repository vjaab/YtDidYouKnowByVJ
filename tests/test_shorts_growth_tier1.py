"""Tests for the Tier-1 Shorts growth fixes:
- analytics rows parsed by column header (video ID is the first column)
- long visual chunks split for 2-3s pacing
- internal IDs no longer leak into public tags
- pinned comment never shows placeholder teaser
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunk_builder import split_long_chunks, MAX_VISUAL_CHUNK_SEC


# ── Analytics parsing ─────────────────────────────────────────────────────────
def test_fetch_video_analytics_parses_by_header():
    import hook_analytics_sync as has

    response = {
        "columnHeaders": [{"name": n} for n in [
            "video", "views", "estimatedMinutesWatched", "averageViewDuration",
            "averageViewPercentage", "subscribersGained", "likes", "comments", "shares"]],
        "rows": [["abc123", 1500, 20, 25, 83.3, 4, 60, 5, 7]],
    }
    service = MagicMock()
    service.reports.return_value.query.return_value.execute.return_value = response

    out = has.fetch_video_analytics(service, "abc123", "2026-10-01", "2026-10-03")
    assert out["views"] == 1500
    assert out["avg_view_duration_sec"] == 25
    assert abs(out["avg_view_percentage"] - 83.3) < 1e-6
    assert out["likes"] == 60 and out["shares"] == 7


def test_fetch_video_analytics_fallback_without_headers():
    import hook_analytics_sync as has

    response = {"rows": [["abc123", 900, 10, 20, 70.0, 1, 30, 2, 3]]}
    service = MagicMock()
    service.reports.return_value.query.return_value.execute.return_value = response
    out = has.fetch_video_analytics(service, "abc123", "2026-10-01", "2026-10-03")
    assert out["views"] == 900
    assert out["avg_view_percentage"] == 70.0


# ── Visual pacing ─────────────────────────────────────────────────────────────
def _words(n, start=0.0, step=0.45):
    return [{"word": f"w{i}", "start": start + i * step, "end": start + i * step + 0.4} for i in range(n)]


def test_long_chunk_is_split_to_short_beats():
    words = _words(28)  # ~12.6s chunk
    chunk = {"chunk_id": 1, "text": "x", "words": words, "start": words[0]["start"],
             "end": words[-1]["end"], "nano_visual_prompt": "a robot"}
    out = split_long_chunks([chunk])
    assert len(out) >= 4
    for c in out:
        assert c["end"] - c["start"] <= MAX_VISUAL_CHUNK_SEC + 0.5
    # All words preserved in order
    assert [w["word"] for c in out for w in c["words"]] == [w["word"] for w in words]
    # Continuations get a different framing prompt
    assert out[0]["nano_visual_prompt"] == "a robot"
    assert out[1]["nano_visual_prompt"] != "a robot"
    assert [c["chunk_id"] for c in out] == list(range(1, len(out) + 1))


def test_short_chunk_untouched():
    words = _words(4)
    chunk = {"chunk_id": 1, "text": "x", "words": words, "start": 0.0, "end": words[-1]["end"]}
    assert split_long_chunks([chunk]) == [chunk]


# ── Public metadata hygiene ───────────────────────────────────────────────────
def test_internal_ids_not_in_public_tags():
    import tags_helper
    with patch.object(tags_helper, "fetch_trending_hashtags_from_google", return_value=[]):
        res = tags_helper.get_optimized_metadata(
            "Hidden iPhone feature", "Your iPhone can do this hidden trick",
            editorial_perspective="Builders Lens", content_fingerprint="ca66d099ffff",
        )
    blob = str(res).lower()
    assert "fp_ca66d099" not in blob and "fpca66d099" not in blob
    assert "builders_lens" not in blob and "builderslens" not in blob


# ── Pinned comment ────────────────────────────────────────────────────────────
def test_pinned_comment_skips_placeholder_tease():
    import main
    with patch.object(main, "get_series_identity", return_value={"name": "HYBRID AI"}):
        txt = main.generate_pinned_comment({"comment_hook": "Guess A, B or C?"}, 0)
        assert "something big tomorrow" not in txt
        assert "Tomorrow on" not in txt
        txt2 = main.generate_pinned_comment({"next_video_tease": "the iPhone setting Apple hides"}, 0)
        assert "Tomorrow on HYBRID AI: the iPhone setting Apple hides" in txt2
