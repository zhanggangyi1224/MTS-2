#!/bin/bash
#
# MTS-2 Data Preparation Only (SLURM)
# Downloads and prepares dataset, creates augmented audio, then stops
# Use this to get real audio data that you can test locally on Mac
#
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH -J mts-data-prep-only
#SBATCH -o /data/gpfs/projects/punim2072/MTS/out/%x-%j.out
#SBATCH -e /data/gpfs/projects/punim2072/MTS/err/%x-%j.err
#SBATCH --mail-user=GangyiZ@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A punim2072

echo "================================"
echo "MTS-2 Data Preparation Only"
echo "Purpose: Download dataset, create augmented audio"
echo "Output: Real audio files for local Mac testing"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
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

# Create directories
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating directories..."
mkdir -p /data/gpfs/projects/punim2072/MTS/{out,err}
mkdir -p $PROJECT_DIR/{data,outputs,logs,checkpoints}
mkdir -p $DATA_DIR/{raw,processed,augmented,augmented/audio}
mkdir -p $OUTPUT_DIR/{configurations,statistics,augmentation}

# Set permissions
chmod 755 $DATA_DIR/augmented/audio

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
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python: $(which python)"

# Install packages
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing packages..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA
pip install --user torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r $PROJECT_DIR/requirements.txt --exists-action i

# Install audio libraries
pip install --user librosa soundfile scipy
pip install --user pyrubberband || echo "[WARNING] pyrubberband failed"
pip install --user pydub

echo "================================"

# Create data-only configuration
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating data preparation config..."
cat > $PROJECT_DIR/config/data_prep_only.yaml << 'EOF'
# Data Preparation Only Configuration
# For SLURM: Download dataset and create augmented audio

data:
  dataset_name: "ccmusic-database/music_genre"
  cache_dir: "./cache"
  data_dir: "./data"
  output_dir: "./outputs"
  target_sample_rate: 24000  # EnCodec-compatible
  use_simulated_data: false  # Use REAL data
  use_fma_dataset: false     # Use CCMusic
  huggingface_token: null    # Add if needed

augmentation:
  enabled: true
  augmentation_factor: 3     # Create 3 augmented versions
  preserve_original: true
  save_audio: true          # MUST save audio files
  audio_format: "mp3"       # MP3 for space efficiency
  suppress_warnings: true

  techniques:
    pitch_shift:
      enabled: true
      probability: 0.8
      range_semitones: [-2, 2]
      preserve_tempo: true

    tempo_scale:
      enabled: true
      probability: 0.8
      range_factor: [0.9, 1.1]
      preserve_pitch: true

    noise_addition:
      enabled: true
      probability: 0.4
      snr_db_range: [25, 40]
      noise_types: ["gaussian", "pink"]

    eq_filter:
      enabled: true
      probability: 0.3
      gain_range_db: [-2, 2]

    reverb:
      enabled: true
      probability: 0.25
      room_sizes: ["small", "medium", "large"]
      wet_mix_range: [0.1, 0.3]

text_generation:
  enabled: false  # Skip for data prep only

data_organization:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_state: 42
  stratify_by_genre: true

structure_processing:
  enabled: false  # Skip for data prep only

pipeline:
  save_intermediate_results: true
  continue_on_error: true
  parallel_processing: false

logging:
  level: "INFO"
  save_logs: true
EOF

echo "✅ Config created: config/data_prep_only.yaml"

# Run data preparation only
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting data preparation..."
echo "================================"

python3 << 'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

import yaml
from data_loader import MTSDataLoader
from augmentation import MTSAudioAugmentation
import json
from datetime import datetime

print("="*60)
print("MTS-2 Data Preparation Only")
print("="*60)

# Load config
with open("config/data_prep_only.yaml") as f:
    config = yaml.safe_load(f)

# Step 1: Load dataset
print("\n[Step 1] Loading dataset...")
loader = MTSDataLoader(
    cache_dir=config["data"]["cache_dir"],
    data_dir=config["data"]["data_dir"],
    target_sr=config["data"]["target_sample_rate"]
)

# Try CCMusic first, fallback to simulated
try:
    if config["data"].get("use_simulated_data", True):
        dataset_info = loader._create_fallback_dataset()
    else:
        dataset_info = loader.load_ccmusic_dataset(
            config["data"]["dataset_name"],
            use_auth_token=config["data"].get("huggingface_token")
        )
except Exception as e:
    print(f"⚠️  Dataset load failed: {e}")
    print("Using simulated dataset...")
    dataset_info = loader._create_fallback_dataset()

# Process features
print("\n[Step 2] Processing features...")
songs = loader.extract_features_from_spectrograms(dataset_info)

# Save processed songs
processed_file = loader.save_processed_dataset(songs, "processed_songs.json")
print(f"✅ Saved processed songs: {processed_file}")

# Step 2: Create augmented audio
print("\n[Step 3] Creating augmented audio...")
augmenter = MTSAudioAugmentation(
    sample_rate=config["data"]["target_sample_rate"],
    augmentation_factor=config["augmentation"]["augmentation_factor"],
    preserve_original=config["augmentation"]["preserve_original"],
    suppress_warnings=config["augmentation"]["suppress_warnings"]
)

# Augment dataset
augmented_songs = augmenter.augment_dataset(
    songs,
    output_dir=f"{config['data']['data_dir']}/augmented",
    save_audio=config["augmentation"]["save_audio"],
    audio_format=config["augmentation"]["audio_format"]
)

print(f"\n✅ Created {len(augmented_songs)} augmented songs")

# Save summary
summary = {
    "preparation_date": datetime.now().isoformat(),
    "total_original_songs": len(songs),
    "total_augmented_songs": len(augmented_songs),
    "augmentation_factor": config["augmentation"]["augmentation_factor"],
    "sample_rate": config["data"]["target_sample_rate"],
    "audio_format": config["augmentation"]["audio_format"],
    "audio_directory": f"{config['data']['data_dir']}/augmented/audio",
    "dataset_source": "simulated" if dataset_info.get("is_simulated") else "ccmusic"
}

summary_file = Path(config["data"]["output_dir"]) / "data_prep_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Summary saved: {summary_file}")
print("\n" + "="*60)
print("Data Preparation Complete!")
print("="*60)
print(f"Audio files location: {config['data']['data_dir']}/augmented/audio/")
print(f"Total files: {len(augmented_songs)}")
print("\nYou can now:")
print("  1. Copy one audio file to your Mac for testing")
print("  2. Use run_local_with_slurm_data.py to test locally")
print("="*60)

PYEOF

EXIT_CODE=$?

# Report results
echo "================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Data preparation completed with exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: ✅ SUCCESS"
    echo ""
    echo "📊 Output Summary:"

    AUDIO_DIR="$DATA_DIR/augmented/audio"
    if [ -d "$AUDIO_DIR" ]; then
        AUDIO_COUNT=$(find $AUDIO_DIR -type f \( -name '*.mp3' -o -name '*.wav' \) 2>/dev/null | wc -l)
        AUDIO_SIZE=$(du -sh $AUDIO_DIR 2>/dev/null | awk '{print $1}')

        echo "  📂 Audio directory: $AUDIO_DIR"
        echo "  🎵 Total audio files: $AUDIO_COUNT"
        echo "  💾 Total size: $AUDIO_SIZE"

        if [ $AUDIO_COUNT -gt 0 ]; then
            echo ""
            echo "  🎼 Sample audio files (first 10):"
            find $AUDIO_DIR -type f \( -name '*.mp3' -o -name '*.wav' \) 2>/dev/null | head -10 | while read file; do
                SIZE=$(ls -lh "$file" | awk '{print $5}')
                echo "    - $(basename "$file") ($SIZE)"
            done

            echo ""
            echo "  📋 To copy one file to your Mac for testing:"
            FIRST_FILE=$(find $AUDIO_DIR -type f \( -name '*.mp3' -o -name '*.wav' \) 2>/dev/null | head -1)
            if [ ! -z "$FIRST_FILE" ]; then
                echo "    scp $USER@spartan.hpc.unimelb.edu.au:$FIRST_FILE ~/Downloads/"
            fi
        fi
    fi

    echo ""
    echo "📁 Key Output Files:"
    find $OUTPUT_DIR -type f \( -name "*.json" -o -name "*.csv" \) 2>/dev/null | head -10
else
    echo "Status: ❌ FAILED"
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
