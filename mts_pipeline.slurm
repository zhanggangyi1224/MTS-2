#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 7-00:00:00
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

# Load HPC modules
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loading HPC modules..."
module purge
module load GCCcore/11.3.0
module load CUDA/11.7.0
module load Anaconda3/2024.02-1
module load GCC/11.3.0
module load OpenMPI/4.1.4
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
export PROJECT_DIR=${SLURM_SUBMIT_DIR:-/data/gpfs/projects/punim2072/MTS/MTS/MTS-2}
export CACHE_DIR=$PROJECT_DIR/cache
export DATA_DIR=$PROJECT_DIR/data
export OUTPUT_DIR=$PROJECT_DIR/outputs
export PYTHONPATH=$PROJECT_DIR:$PROJECT_DIR/src:$PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Environment setup complete"
echo "  PROJECT_DIR: $PROJECT_DIR"
echo "  PYTHONPATH: $PYTHONPATH"

# Create directories
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating directories..."
mkdir -p /data/gpfs/projects/punim2072/MTS/{out,err}
mkdir -p $PROJECT_DIR/{data,outputs,logs,checkpoints}
mkdir -p $DATA_DIR/{raw,processed,augmented}
mkdir -p $OUTPUT_DIR/{configurations,statistics,structure,augmentation}

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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python: $(which python)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Version: $(python --version)"

# Install packages
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing packages..."
pip install --upgrade pip setuptools wheel

# Install PyTorch 2.1.2 with CUDA 11.8 for H100 support
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing PyTorch for H100..."
pip install --user torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install base requirements
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing base requirements..."
pip install -r $PROJECT_DIR/requirements.txt --exists-action i

# Install complete requirements (model dependencies)
if [ -f "$PROJECT_DIR/requirements_complete.txt" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing complete requirements..."
    pip install -r $PROJECT_DIR/requirements_complete.txt --exists-action i || true
fi

# CRITICAL: Install einops explicitly (needed by diffusion_unet.py)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing model dependencies..."
pip install --user einops>=0.6.0
pip install --user encodec>=0.1.1 || echo "[WARNING] encodec install failed - will use mock tokenization"
pip install --user pydub>=0.25.1

# Install psutil for memory management
pip install --user psutil

# Enforce compatible transformers/sentence-transformers versions
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reinstalling transformers stack..."
pip uninstall -y transformers sentence-transformers >/dev/null 2>&1 || true
pip install "transformers==4.36.0" "sentence-transformers==2.2.2"

# Install optional packages (continue on failure)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing optional packages..."
pip install Cython>=0.29.0 --user 2>/dev/null || true
pip install madmom>=0.16.1 --no-build-isolation --user 2>/dev/null || echo "[WARNING] madmom failed - will use librosa fallback"
pip install pyrubberband>=0.3.0 --user 2>/dev/null || echo "[WARNING] pyrubberband failed - will use librosa fallback"
pip install pedalboard>=0.7.0 --user 2>/dev/null || echo "[WARNING] pedalboard failed - will use basic effects"

echo "================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Verifying installations..."
python -c "
import sys
print('Python:', sys.version)

# Core packages
print('\\n--- Core Packages ---')
required = ['numpy', 'pandas', 'librosa', 'transformers', 'datasets', 'scipy', 'torchaudio', 'soundfile']
for pkg in required:
    try:
        __import__(pkg)
        print(f'{pkg}: OK')
    except ImportError:
        print(f'{pkg}: MISSING')

# PyTorch and CUDA
print('\\n--- PyTorch ---')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'PyTorch: ERROR - {e}')

# New model dependencies
print('\\n--- Model Dependencies (Phase 2) ---')
model_deps = ['einops', 'encodec', 'pydub']
for pkg in model_deps:
    try:
        __import__(pkg)
        print(f'{pkg}: OK')
    except ImportError:
        print(f'{pkg}: MISSING (optional for data pipeline)')

# Check ffmpeg
import shutil
ffmpeg = shutil.which('ffmpeg')
print(f'\\nffmpeg: {\"OK\" if ffmpeg else \"MISSING\"} ({ffmpeg})')
"
echo "================================"

# Verify project structure including new files
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Verifying project structure..."
python -c "
from pathlib import Path

project_dir = Path('$PROJECT_DIR')

# Data pipeline files (Phase 1 - REQUIRED)
phase1_files = [
    'src/__init__.py',
    'src/pipeline.py',
    'src/batch_processor.py',
    'src/data_loader.py',
    'src/augmentation.py',
    'src/text_generation.py',
    'src/data_organization.py',
    'src/structure_processing.py',
    'src/data/__init__.py',
]

# Model files (Phase 2 - scaffolds, not required for data pipeline)
phase2_files = [
    'src/data/structure_detection.py',
    'src/models/__init__.py',
    'src/models/encodec_wrapper.py',
    'src/models/diffusion_unet.py',
    'src/models/mts_model.py',
    'src/training/__init__.py',
    'src/training/trainer.py',
    'src/training/losses.py',
    'src/inference/__init__.py',
    'src/inference/generator.py',
]

print('=== Phase 1: Data Pipeline Files (REQUIRED) ===')
missing_phase1 = []
for f in phase1_files:
    path = project_dir / f
    if path.exists():
        print(f'  ✓ {f}')
    else:
        print(f'  ✗ {f} MISSING')
        missing_phase1.append(f)

print('\\n=== Phase 2: Model Files (scaffolds for future use) ===')
missing_phase2 = []
for f in phase2_files:
    path = project_dir / f
    if path.exists():
        print(f'  ✓ {f}')
    else:
        print(f'  ✗ {f} MISSING')
        missing_phase2.append(f)

if missing_phase1:
    print(f'\\n❌ ERROR: {len(missing_phase1)} required Phase 1 files missing!')
    exit(1)
else:
    print('\\n✓ All Phase 1 files present - data pipeline ready')

if missing_phase2:
    print(f'⚠️  {len(missing_phase2)} Phase 2 files missing (not needed for data pipeline)')
else:
    print('✓ All Phase 2 files present - model training ready')
"

VERIFY_EXIT=$?
if [ $VERIFY_EXIT -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Required files missing!"
    exit 1
fi

# Fix config if needed (ensure MP3 format)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking config..."
python -c "
import yaml
from pathlib import Path

config_path = Path('$PROJECT_DIR/config/batch_config.yaml')
if config_path.exists():
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    changed = False
    
    # Fix audio format
    if config.get('augmentation', {}).get('audio_format') != 'mp3':
        print('Fixing audio_format: mp4 -> mp3')
        config.setdefault('augmentation', {})['audio_format'] = 'mp3'
        changed = True
    
    # Fix sample rate for EnCodec compatibility
    if config.get('data', {}).get('target_sample_rate') != 24000:
        print('Fixing target_sample_rate: -> 24000')
        config.setdefault('data', {})['target_sample_rate'] = 24000
        changed = True
    
    if changed:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print('Config updated!')
    else:
        print('Config already correct')
"

# Select config file
if [ -f "$PROJECT_DIR/config/batch_config.yaml" ]; then
    PIPELINE_CONFIG="config/batch_config.yaml"
else
    PIPELINE_CONFIG="config/config.yaml"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using config: $PIPELINE_CONFIG"

# Run the pipeline
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting MTS Pipeline..."
echo "================================"

cd $PROJECT_DIR

if [ -f "$PROJECT_DIR/run_batch_pipeline.py" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using batch processing for memory efficiency..."
    python -u run_batch_pipeline.py --config $PIPELINE_CONFIG --batch-size 100 --max-memory 75
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using standard pipeline..."
    python -u run_pipeline.py --config config/config.yaml
fi

EXIT_CODE=$?

# Report results
echo "================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline completed with exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
    echo ""
    echo "Output files:"
    find $OUTPUT_DIR -type f \( -name "*.json" -o -name "*.csv" -o -name "*.md" \) 2>/dev/null | head -20
    
    echo ""
    echo "Summary:"
    echo "  JSON files: $(find $OUTPUT_DIR -name '*.json' 2>/dev/null | wc -l)"
    echo "  CSV files: $(find $OUTPUT_DIR -name '*.csv' 2>/dev/null | wc -l)"
    echo "  MP3 files: $(find $DATA_DIR -name '*.mp3' 2>/dev/null | wc -l)"
else
    echo "Status: FAILED"
    echo "Check error logs for details"
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
