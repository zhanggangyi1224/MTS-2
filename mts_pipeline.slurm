#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 05:00:00
#SBATCH -J mts-data-pipeline
#SBATCH -o /data/gpfs/projects/punim2072/MTS/out/%x-%j.out
#SBATCH -e /data/gpfs/projects/punim2072/MTS/err/%x-%j.err
#SBATCH --mail-user=GangyiZ@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A punim2072

echo "================================"
echo "MTS Data Pipeline - Job Started"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "================================"

# Load HPC-available modules (based on softwarelist.txt)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loading HPC modules..."
module purge
module load GCCcore/11.3.0
module load CUDA/11.7.0
module load Anaconda3/2024.02-1
# Load dependencies first
module load GCC/11.3.0
module load OpenMPI/4.1.4
# Now load dependent modules
module load SciPy-bundle/2022.05
module load matplotlib/3.5.2
module load scikit-learn/1.1.2
module load h5py/3.7.0
module load FFmpeg/5.0.1
module load PyYAML/6.0
module load tqdm/4.64.0

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loaded modules:"
module list 2>&1

# Set up environment variables
export PROJECT_DIR=/data/gpfs/projects/punim2072/MTS/MTS/MTS
export DATA_DIR=$PROJECT_DIR/data
export OUTPUT_DIR=$PROJECT_DIR/outputs
export PYTHONPATH=$PROJECT_DIR:$PROJECT_DIR/src:$PYTHONPATH
export PYTHONUNBUFFERED=1

# CUDA settings
export CUDA_VISIBLE_DEVICES=0

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Environment setup complete"
echo "  PROJECT_DIR: $PROJECT_DIR"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Create directories
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating directories..."
mkdir -p /data/gpfs/projects/punim2072/MTS/{out,err}
mkdir -p $PROJECT_DIR/{data,outputs,logs,src,config}
mkdir -p $DATA_DIR/{raw,processed,augmented}
mkdir -p $OUTPUT_DIR/{configurations,statistics,structure,augmentation}

# Navigate to project directory
cd $PROJECT_DIR

# Initialize conda
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Initializing conda..."
eval "$(conda shell.bash hook)"

# Activate conda environment
CONDA_ENV_NAME="mts-data"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to activate environment"
    conda env list
    exit 1
fi

# Install psutil for batch processing memory management
pip install --user psutil

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python: $(which python)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Version: $(python --version)"

# Install packages
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing packages..."
pip install --upgrade pip setuptools wheel

# Install PyTorch 2.1.2 with CUDA 11.8 for H100 support
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing PyTorch for H100..."
pip install --user torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install main requirements (skip torch/torchaudio since already installed)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing requirements..."
pip install -r $PROJECT_DIR/requirements.txt --exists-action i

# Install optional packages separately (continue on failure)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing optional packages..."
pip install madmom>=0.16.1 --no-build-isolation 2>/dev/null || echo "[WARNING] madmom failed"
pip install openai-whisper>=20230314 2>/dev/null || echo "[WARNING] whisper failed"
pip install pyrubberband>=0.3.0 2>/dev/null || echo "[WARNING] pyrubberband failed"
pip install pedalboard>=0.7.0 2>/dev/null || echo "[WARNING] pedalboard failed"

# Fix run_pipeline.py if needed
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking run_pipeline.py..."
if [ -f "$PROJECT_DIR/run_pipeline.py" ]; then
    # Fix the syntax error in the file
    sed -i 's/if \*\*name\*\* == "__main__":/if __name__ == "__main__":/' $PROJECT_DIR/run_pipeline.py
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fixed run_pipeline.py syntax"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating run_pipeline.py..."
    cat > $PROJECT_DIR/run_pipeline.py << 'EOF'
"""
Main runner script for MTS Data Pipeline
"""
import os
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run pipeline
from pipeline import main

if __name__ == "__main__":
    main()
EOF
fi

echo "================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Verifying installations..."
python -c "
import sys
print('Python:', sys.version)
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except: pass

required = ['numpy', 'pandas', 'librosa', 'transformers', 'datasets', 'scipy', 'torchaudio']
for pkg in required:
    try:
        __import__(pkg)
        print(f'{pkg}: OK')
    except:
        print(f'{pkg}: MISSING')
"
echo "================================"

# Check config file
if [ ! -f "$PROJECT_DIR/config/config.yaml" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating config.yaml..."
    cat > $PROJECT_DIR/config/config.yaml << EOF
data:
  dataset_name: "ccmusic-database/music_genre"
  cache_dir: "$CACHE_DIR"
  data_dir: "$DATA_DIR"
  output_dir: "$OUTPUT_DIR"
  target_sample_rate: 22050
  use_simulated_data: true

augmentation:
  enabled: true
  augmentation_factor: 3

text_generation:
  enabled: true
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  prompt_pool_size: 1000

data_organization:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

structure_processing:
  enabled: true
  structure_subset_size: 300

pipeline:
  save_intermediate_results: true
  continue_on_error: true

logging:
  level: "INFO"
EOF
fi

# Run the pipeline using run_pipeline.py
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting MTS Pipeline..."
echo "================================"

cd $PROJECT_DIR

# Check if batch processor exists, use it for memory efficiency
if [ -f "$PROJECT_DIR/run_batch_pipeline.py" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using batch processing for memory efficiency..."
    python -u run_batch_pipeline.py --batch-size 100 --max-memory 75
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using standard pipeline..."
    python -u run_pipeline.py
fi

EXIT_CODE=$?

# Report results
echo "================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline completed with exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
    # List output files
    echo "Output files:"
    find $OUTPUT_DIR -type f -name "*.json" -o -name "*.csv" -o -name "*.md" | head -20
else
    echo "Status: FAILED"
fi

# Cleanup
conda deactivate

echo "================================"
echo "Job Summary"
echo "Job ID: $SLURM_JOB_ID"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "================================"

exit $EXIT_CODE