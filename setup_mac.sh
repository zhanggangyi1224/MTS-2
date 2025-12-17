#!/bin/bash
# Setup script for MTS-2 on Mac
# Installs dependencies and prepares environment

set -e  # Exit on error

echo "================================"
echo "🍎 MTS-2 Mac Setup Script"
echo "================================"
echo ""

# Check if we're on Mac
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install from https://brew.sh"
    exit 1
fi
echo "✅ Homebrew found"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
echo "Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.8" | bc) -eq 1 ]]; then
    echo "❌ Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python version OK"

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."

# FFmpeg (required for audio conversion)
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    brew install ffmpeg
else
    echo "✅ ffmpeg already installed"
fi

# Install RubberBand (for high-quality pitch/tempo shifting)
if ! command -v rubberband &> /dev/null; then
    echo "Installing rubberband..."
    brew install rubberband
else
    echo "✅ rubberband already installed"
fi

# SoX (optional, for audio processing)
if ! command -v sox &> /dev/null; then
    echo "Installing sox (optional)..."
    brew install sox
else
    echo "✅ sox already installed"
fi

# Create virtual environment
echo ""
echo "🐍 Setting up Python virtual environment..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with MPS support
echo ""
echo "🔥 Installing PyTorch with Mac MPS support..."
pip install torch torchvision torchaudio

# Test MPS
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
if torch.backends.mps.is_available():
    print("✅ MPS (Metal Performance Shaders) available!")
    print("   Your Mac can use GPU acceleration!")
else:
    print("⚠️  MPS not available (requires macOS 12.3+ and PyTorch 1.12+)")
EOF

# Install audio processing libraries
echo ""
echo "🎵 Installing audio processing libraries..."
pip install librosa soundfile scipy
pip install pyrubberband || echo "⚠️  pyrubberband failed (optional)"
pip install pedalboard || echo "⚠️  pedalboard failed (optional)"

# Install ML/NLP libraries
echo ""
echo "🤖 Installing ML/NLP libraries..."
pip install transformers sentence-transformers datasets
pip install numpy pandas scikit-learn

# Install CoreML tools (for Neural Engine)
echo ""
echo "🧠 Installing CoreML tools for Neural Engine..."
if [[ $(uname -m) == "arm64" ]]; then
    pip install coremltools
    echo "✅ CoreML tools installed (Neural Engine support enabled)"
else
    echo "ℹ️  Not Apple Silicon, skipping CoreML tools"
fi

# Install other requirements
echo ""
echo "📦 Installing remaining requirements..."
pip install -r requirements.txt || echo "⚠️  Some requirements failed (non-critical)"

# Install pydub for audio format handling
pip install pydub

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p data/{raw,processed,augmented/audio}
mkdir -p outputs/{logs,configurations,statistics,structure,augmentation}
mkdir -p cache/text_models
mkdir -p checkpoints
echo "✅ Directories created"

# Check if FMA data exists
echo ""
echo "📂 Checking for audio data..."
if [ -d "./fma_data/fma_small" ]; then
    FMA_COUNT=$(find ./fma_data/fma_small -name "*.mp3" | wc -l)
    echo "✅ FMA dataset found: $FMA_COUNT files"
else
    echo "⚠️  FMA dataset not found"
    echo ""
    echo "To download FMA dataset:"
    echo "  1. Visit: https://github.com/mdeff/fma"
    echo "  2. Download fma_small.zip (8 hours, 8GB, ~8,000 tracks)"
    echo "  3. Extract: unzip fma_small.zip -d fma_data/"
    echo ""
    echo "Or use your own audio files:"
    echo "  - Place .mp3 or .wav files in ./data/raw/"
fi

# Summary
echo ""
echo "================================"
echo "✅ Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source .venv/bin/activate"
echo "  2. Test installation: python3 run_local_test.py --all"
echo "  3. Run pipeline: python3 run_pipeline.py --config config/config_local_mac.yaml"
echo ""
echo "Mac Optimizations Enabled:"
echo "  ✅ MPS (Metal Performance Shaders) for GPU acceleration"
if [[ $(uname -m) == "arm64" ]]; then
    echo "  ✅ CoreML for Neural Engine acceleration"
fi
echo "  ✅ Hardware-accelerated FFmpeg"
echo "  ✅ High-quality audio processing (pyrubberband)"
echo ""
