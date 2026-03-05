#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_musetalk.sh — One-command MuseTalk v1.5 setup
#
# This script:
#   1. Clones MuseTalk into the project directory
#   2. Installs Python dependencies (MMLab stack)
#   3. Downloads all required model weights (~3GB total)
#
# Prerequisites:
#   - Python 3.10+ with pip
#   - PyTorch (already in your requirements.txt)
#   - ffmpeg installed (brew install ffmpeg / apt install ffmpeg)
#   - ~5GB disk space for models
#
# Usage:
#   cd yt_did_you_know_by_vj
#   chmod +x setup_musetalk.sh
#   ./setup_musetalk.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MUSETALK_DIR="$SCRIPT_DIR/MuseTalk"
MODELS_DIR="$MUSETALK_DIR/models"

echo "═══════════════════════════════════════════════════════════"
echo "  🎭 MuseTalk v1.5 Setup — Better Lip Sync for YT Shorts"
echo "═══════════════════════════════════════════════════════════"

# ── Step 1: Clone MuseTalk ────────────────────────────────────────────────────
if [ ! -d "$MUSETALK_DIR" ]; then
    echo ""
    echo "📦 Step 1/4: Cloning MuseTalk repository..."
    git clone https://github.com/TMElyralab/MuseTalk.git "$MUSETALK_DIR"
else
    echo ""
    echo "📦 Step 1/4: MuseTalk directory already exists. Pulling latest..."
    cd "$MUSETALK_DIR" && git pull origin main || true
fi

cd "$MUSETALK_DIR"

# ── Step 2: Install Python dependencies ───────────────────────────────────────
echo ""
echo "📦 Step 2/4: Installing Python dependencies..."

# Install MuseTalk's own requirements
if [ -f requirements.txt ]; then
    pip install -r requirements.txt 2>&1 | tail -5
fi

# Install MMLab ecosystem (required for face detection / pose estimation)
echo "   Installing MMLab packages..."
pip install --no-cache-dir -U openmim 2>&1 | tail -2
mim install mmengine 2>&1 | tail -2
mim install "mmcv>=2.0.1" 2>&1 | tail -2
mim install "mmdet>=3.1.0" 2>&1 | tail -2
mim install "mmpose>=1.1.0" 2>&1 | tail -2

# Install whisper (audio feature extraction)
if [ -d "musetalk/whisper" ]; then
    echo "   Installing Whisper for audio features..."
    pip install --editable ./musetalk/whisper 2>&1 | tail -2
fi

# ── Step 3: Download Model Weights ────────────────────────────────────────────
echo ""
echo "📦 Step 3/4: Downloading model weights (~3GB)..."

mkdir -p "$MODELS_DIR/musetalkV15"
mkdir -p "$MODELS_DIR/musetalk"
mkdir -p "$MODELS_DIR/dwpose"
mkdir -p "$MODELS_DIR/face-parse-bisent"
mkdir -p "$MODELS_DIR/sd-vae"
mkdir -p "$MODELS_DIR/whisper"
mkdir -p "$MODELS_DIR/syncnet"

# Check if huggingface-cli is available, otherwise use wget
HF_DOWNLOAD=""
if command -v huggingface-cli &> /dev/null; then
    HF_DOWNLOAD="hf"
    echo "   Using huggingface-cli for downloads..."
fi

download_hf_file() {
    local repo="$1"
    local filename="$2"
    local dest="$3"

    if [ -f "$dest" ]; then
        echo "   ✓ Already exists: $(basename $dest)"
        return 0
    fi

    echo "   ↓ Downloading: $repo/$filename → $(basename $dest)"
    
    if [ "$HF_DOWNLOAD" = "hf" ]; then
        huggingface-cli download "$repo" "$filename" --local-dir "$(dirname $dest)" --local-dir-use-symlinks False 2>/dev/null || \
        wget -q --show-progress -O "$dest" "https://huggingface.co/$repo/resolve/main/$filename"
    else
        wget -q --show-progress -O "$dest" "https://huggingface.co/$repo/resolve/main/$filename"
    fi
}

# MuseTalk v1.5 weights
download_hf_file "TMElyralab/MuseTalk" "models/musetalkV15/unet.pth" "$MODELS_DIR/musetalkV15/unet.pth"
download_hf_file "TMElyralab/MuseTalk" "models/musetalkV15/musetalk.json" "$MODELS_DIR/musetalkV15/musetalk.json"

# MuseTalk v1.0 weights (fallback)
download_hf_file "TMElyralab/MuseTalk" "models/musetalk/pytorch_model.bin" "$MODELS_DIR/musetalk/pytorch_model.bin"
download_hf_file "TMElyralab/MuseTalk" "models/musetalk/musetalk.json" "$MODELS_DIR/musetalk/musetalk.json"

# SD-VAE (Stable Diffusion VAE for latent space)
download_hf_file "stabilityai/sd-vae-ft-mse" "diffusion_pytorch_model.bin" "$MODELS_DIR/sd-vae/diffusion_pytorch_model.bin"
download_hf_file "stabilityai/sd-vae-ft-mse" "config.json" "$MODELS_DIR/sd-vae/config.json"

# Whisper (audio feature extraction)
download_hf_file "openai/whisper-tiny" "pytorch_model.bin" "$MODELS_DIR/whisper/pytorch_model.bin"
download_hf_file "openai/whisper-tiny" "config.json" "$MODELS_DIR/whisper/config.json"
download_hf_file "openai/whisper-tiny" "preprocessor_config.json" "$MODELS_DIR/whisper/preprocessor_config.json"

# DWPose (face/body pose estimation)
download_hf_file "yzd-v/DWPose" "dw-ll_ucoco_384.pth" "$MODELS_DIR/dwpose/dw-ll_ucoco_384.pth"

# SyncNet (lip-sync quality scoring)
download_hf_file "ByteDance/LatentSync" "latentsync_syncnet.pt" "$MODELS_DIR/syncnet/latentsync_syncnet.pt"

# ResNet18 (face parsing backbone)
if [ ! -f "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth" ]; then
    echo "   ↓ Downloading: ResNet18 backbone..."
    wget -q --show-progress -O "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth" \
        "https://download.pytorch.org/models/resnet18-5c106cde.pth"
else
    echo "   ✓ Already exists: resnet18-5c106cde.pth"
fi

# Face-Parse-BiSeNet (face segmentation - hosted on Google Drive)
if [ ! -f "$MODELS_DIR/face-parse-bisent/79999_iter.pth" ]; then
    echo "   ↓ Downloading: Face-Parse-BiSeNet weights..."
    echo "   ⚠ This file is on Google Drive. If auto-download fails, manually download from:"
    echo "     https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view"
    echo "     Place it at: $MODELS_DIR/face-parse-bisent/79999_iter.pth"
    
    # Try gdown if available
    if command -v gdown &> /dev/null; then
        gdown "154JgKpzCPW82qINcVieuPH3fZ2e0P812" -O "$MODELS_DIR/face-parse-bisent/79999_iter.pth"
    else
        pip install gdown 2>/dev/null
        gdown "154JgKpzCPW82qINcVieuPH3fZ2e0P812" -O "$MODELS_DIR/face-parse-bisent/79999_iter.pth" || \
        echo "   ⚠ Auto-download failed. Please download manually (see URL above)."
    fi
else
    echo "   ✓ Already exists: 79999_iter.pth"
fi

# ── Step 4: Verify Installation ───────────────────────────────────────────────
echo ""
echo "📦 Step 4/4: Verifying installation..."

MISSING=0
check_file() {
    if [ -f "$1" ]; then
        echo "   ✓ $2"
    else
        echo "   ✗ MISSING: $2 ($1)"
        MISSING=$((MISSING + 1))
    fi
}

check_file "$MODELS_DIR/musetalkV15/unet.pth" "MuseTalk v1.5 model"
check_file "$MODELS_DIR/sd-vae/diffusion_pytorch_model.bin" "SD-VAE"
check_file "$MODELS_DIR/whisper/pytorch_model.bin" "Whisper"
check_file "$MODELS_DIR/dwpose/dw-ll_ucoco_384.pth" "DWPose"
check_file "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth" "ResNet18"
check_file "$MODELS_DIR/face-parse-bisent/79999_iter.pth" "Face-Parse-BiSeNet"

echo ""
if [ $MISSING -eq 0 ]; then
    echo "═══════════════════════════════════════════════════════════"
    echo "  ✅ MuseTalk setup complete! All weights downloaded."
    echo "  "
    echo "  Your pipeline will now automatically use MuseTalk v1.5"
    echo "  for lip-sync generation."
    echo "═══════════════════════════════════════════════════════════"
else
    echo "═══════════════════════════════════════════════════════════"
    echo "  ⚠ Setup mostly complete, but $MISSING weight(s) missing."
    echo "  Please download them manually (see messages above)."
    echo "═══════════════════════════════════════════════════════════"
fi
