"""
Pipeline Checkpoint Manager — Enables resume from any stage.
Saves/loads intermediate state to JSON files for GitHub Actions artifacts.
"""
import os
import json
import glob
from datetime import datetime
from config import LOGS_DIR, BASE_DIR

CHECKPOINT_DIR = os.path.join(BASE_DIR, ".pipeline_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Long-form stages
LONGFORM_STAGES = [
    "fetch_articles",
    "generate_script",
    "generate_audio",
    "capture_screenshots",
    "build_chunks",
    "fetch_entities",
    "generate_visuals",
    "render_video",
    "generate_thumbnail",
    "upload_youtube",
    "generate_shorts_teaser",
]

# Shorts stages
SHORTS_STAGES = [
    "fetch_articles",
    "generate_script",
    "capture_screenshots",
    "fetch_entities",
    "generate_audio",
    "build_chunks",
    "generate_visuals",
    "validate_visuals",
    "render_video",
    "generate_thumbnail",
    "upload_youtube",
    "upload_x",
    "upload_instagram",
    "upload_facebook",
    "upload_telegram",
    "upload_threads",
]

def _checkpoint_path(stage: str, pipeline_type: str = "longform") -> str:
    return os.path.join(CHECKPOINT_DIR, f"{pipeline_type}_{stage}.json")

def _latest_checkpoint(pipeline_type: str = "longform") -> str | None:
    """Return the latest completed stage, or None."""
    stages = LONGFORM_STAGES if pipeline_type == "longform" else SHORTS_STAGES
    for stage in reversed(stages):
        if os.path.exists(_checkpoint_path(stage, pipeline_type)):
            return stage
    return None

def save_checkpoint(stage: str, data: dict, pipeline_type: str = "longform") -> None:
    """Save checkpoint for a stage."""
    path = _checkpoint_path(stage, pipeline_type)
    data["_checkpoint_meta"] = {
        "stage": stage,
        "pipeline_type": pipeline_type,
        "timestamp": datetime.now().isoformat(),
        "version": 1,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"💾 Checkpoint saved: {pipeline_type}:{stage}")

def load_checkpoint(stage: str, pipeline_type: str = "longform") -> dict | None:
    """Load checkpoint for a stage."""
    path = _checkpoint_path(stage, pipeline_type)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def clear_checkpoints(from_stage: str | None = None, pipeline_type: str = "longform") -> None:
    """Clear all checkpoints, or from a specific stage onwards."""
    stages = LONGFORM_STAGES if pipeline_type == "longform" else SHORTS_STAGES
    if from_stage is None:
        stages_to_clear = stages
    else:
        idx = stages.index(from_stage)
        stages_to_clear = stages[idx:]
    for stage in stages_to_clear:
        path = _checkpoint_path(stage, pipeline_type)
        if os.path.exists(path):
            os.remove(path)
    print(f"🗑️ Checkpoints cleared from: {from_stage or 'start'} ({pipeline_type})")

def get_resume_stage(pipeline_type: str = "longform") -> str | None:
    """Determine which stage to resume from."""
    latest = _latest_checkpoint(pipeline_type)
    if latest is None:
        return None
    stages = LONGFORM_STAGES if pipeline_type == "longform" else SHORTS_STAGES
    idx = stages.index(latest)
    if idx + 1 < len(stages):
        return stages[idx + 1]
    return None

def list_checkpoints(pipeline_type: str = "longform") -> dict:
    """List all existing checkpoints with metadata."""
    stages = LONGFORM_STAGES if pipeline_type == "longform" else SHORTS_STAGES
    result = {}
    for stage in stages:
        path = _checkpoint_path(stage, pipeline_type)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            result[stage] = data.get("_checkpoint_meta", {})
    return result