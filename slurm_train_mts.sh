#!/bin/bash
#SBATCH --job-name=mts_training
#SBATCH --output=slurm_logs/mts_train_%j.out
#SBATCH --error=slurm_logs/mts_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================================================
# MTS Model Training on HPC
# ============================================================================
# This script:
# 1. Sets up the environment
# 2. Checks/prepares data
# 3. Trains the MTS model on GPU
# 4. Generates test samples
# ============================================================================

echo "======================================================================"
echo "MTS Model Training - HPC"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "======================================================================"

# ============================================================================
# Environment Setup
# ============================================================================
echo ""
echo "📦 Setting up environment..."

# Load required modules (adjust for your HPC system)
module purge
module load python/3.9
module load cuda/11.8
module load cudnn/8.6

# Activate virtual environment (create if doesn't exist)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.deps_installed" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    touch venv/.deps_installed
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# ============================================================================
# Check GPU
# ============================================================================
echo ""
echo "🔧 Checking GPU..."
nvidia-smi
echo ""

python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# ============================================================================
# Data Preparation
# ============================================================================
echo ""
echo "======================================================================"
echo "📊 Data Preparation"
echo "======================================================================"

# Check if data already exists
if [ -f "outputs/mts_final_dataset.csv" ]; then
    echo "✅ Found existing processed data"
    echo "   Skipping data preparation"
else
    echo "⚠️  No processed data found"
    echo "   Running data preparation pipeline..."

    # Check if FMA data exists
    if [ ! -d "fma_data/fma_small" ]; then
        echo "❌ FMA dataset not found!"
        echo "   Please download FMA dataset first:"
        echo "   python3 fetch_fma_dataset.py --size small"
        exit 1
    fi

    # Run data preparation pipeline
    echo "🚀 Running data preparation..."
    python3 run_pipeline.py --config config/config_hpc.yaml

    if [ $? -ne 0 ]; then
        echo "❌ Data preparation failed!"
        exit 1
    fi

    echo "✅ Data preparation complete"
fi

# ============================================================================
# Training Configuration
# ============================================================================
echo ""
echo "======================================================================"
echo "🎯 Training Configuration"
echo "======================================================================"

# Training parameters
BATCH_SIZE=8  # Larger batch size on GPU
EPOCHS=200    # More epochs for better quality
LR=1e-4
SAVE_EVERY=20

# Create directories
mkdir -p checkpoints
mkdir -p slurm_logs
mkdir -p generated_samples

echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Save every: $SAVE_EVERY epochs"

# ============================================================================
# Main Training
# ============================================================================
echo ""
echo "======================================================================"
echo "🚀 Starting Training"
echo "======================================================================"
echo ""

python3 -u train_mts_hpc.py \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --save-every $SAVE_EVERY \
    --device cuda \
    2>&1 | tee slurm_logs/training_${SLURM_JOB_ID}.log

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "✅ Training complete!"

# ============================================================================
# Test Generation
# ============================================================================
echo ""
echo "======================================================================"
echo "🎵 Generating Test Samples"
echo "======================================================================"

python3 test_generation.py 2>&1 | tee slurm_logs/generation_${SLURM_JOB_ID}.log

if [ $? -eq 0 ]; then
    echo "✅ Test samples generated successfully"
else
    echo "⚠️  Test generation had errors (check log)"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "📊 Training Summary"
echo "======================================================================"

# Find best checkpoint
BEST_CHECKPOINT=$(ls -t checkpoints/mts_*.pt 2>/dev/null | head -1)

if [ -n "$BEST_CHECKPOINT" ]; then
    echo "Best checkpoint: $BEST_CHECKPOINT"
    CHECKPOINT_SIZE=$(du -h "$BEST_CHECKPOINT" | cut -f1)
    echo "Checkpoint size: $CHECKPOINT_SIZE"
fi

# Count generated samples
SAMPLE_COUNT=$(ls generated_samples/*.wav 2>/dev/null | wc -l)
echo "Generated samples: $SAMPLE_COUNT"

# Show training time
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "======================================================================"

echo ""
echo "✅ Job complete!"
echo ""
echo "📁 Output files:"
echo "   Checkpoints: checkpoints/"
echo "   Samples: generated_samples/"
echo "   Logs: slurm_logs/"
echo ""
echo "🎵 Next steps:"
echo "   1. Check training log: slurm_logs/mts_train_${SLURM_JOB_ID}.out"
echo "   2. Listen to samples: ls generated_samples/"
echo "   3. Run long-form generation: python3 generate_long_form.py"
echo "======================================================================"
