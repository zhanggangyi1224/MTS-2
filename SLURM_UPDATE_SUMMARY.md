# SLURM Pipeline Update Summary

## What Was Fixed

### 1. **Audio Path Bug** ✅
**Problem**: Training failed with "Error opening ''" for all samples

**Solution**:
- Added `audio_path` column to CSV generation in `src/batch_processor.py`
- Created `regenerate_dataset_csv.py` to fix existing datasets
- Your local dataset has been regenerated with audio paths

### 2. **FMA Data Loading** ✅
**Problem**: No automatic checking/downloading of FMA dataset

**Solution**:
- Added Phase 0 to SLURM script: FMA Data Verification
- Automatically checks if data exists
- Downloads and extracts if missing
- Resumes interrupted downloads

### 3. **Pre-Training Validation** ✅
**Problem**: Pipeline didn't verify dataset integrity before training

**Solution**:
- Added comprehensive validation before Phase 2
- Checks for audio_path column
- Verifies audio files exist
- Auto-repairs CSV if needed

### 4. **Dataset Path Verification** ✅
**Problem**: Training would silently fail on missing files without clear errors

**Solution**:
- Created `verify_dataset.py` - standalone verification script
- Enhanced `train_mts_hpc.py` dataset loader with path checking
- Checks sample of files before training starts
- Provides detailed error messages with file paths
- Shows which files are missing and why

## Files Modified

1. ✅ **src/batch_processor.py** - Added audio_path to CSV generation
2. ✅ **regenerate_dataset_csv.py** - New script for CSV regeneration
3. ✅ **verify_dataset.py** - New script for dataset integrity verification
4. ✅ **train_mts_hpc.py** - Enhanced dataset loader with path verification
5. ✅ **mts_pipeline.slurm** - Enhanced with Phase 0 and validation checks
6. ✅ **outputs/mts_final_dataset.csv** - Regenerated with audio paths

## New SLURM Pipeline Flow

```
Phase 0: FMA Data Verification (NEW!)
├── Check if fma_data/fma_metadata exists
├── Check if fma_data/fma_small exists (>7000 MP3s)
├── Download missing data with wget -c (resumable)
├── Extract zip files
└── Verify completion

Phase 1: Data Preparation
├── Skip if outputs/mts_final_dataset.csv exists
└── Run batch pipeline (now includes audio_path in CSV)

Phase 2: Model Training
├── NEW: Verify CSV exists
├── NEW: Check for audio_path column
├── NEW: Auto-regenerate CSV if needed
├── NEW: Count and verify audio files
├── NEW: Run detailed dataset verification (verify_dataset.py)
│   ├── Check 50 sample files exist
│   ├── Report missing/empty paths
│   └── Calculate success rate
├── Load dataset (with built-in path verification)
├── Train model
└── Save checkpoints

Phase 3: Sample Generation
├── Load best checkpoint
└── Generate test samples
```

## What to Transfer to HPC

Transfer these updated files to your HPC environment:

```bash
# Required (contains the fix)
src/batch_processor.py
mts_pipeline.slurm
train_mts_hpc.py

# Recommended (for verification and auto-repair)
regenerate_dataset_csv.py
verify_dataset.py

# Optional (if you want to use the fixed CSV)
outputs/mts_final_dataset.csv
```

## Transfer Commands

```bash
# From your local machine:
cd /Users/zhanggangyi/Desktop/MTS-2

# Transfer updated files
scp src/batch_processor.py username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/src/
scp mts_pipeline.slurm username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/
scp train_mts_hpc.py username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/
scp regenerate_dataset_csv.py username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/
scp verify_dataset.py username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/

# Optional: Transfer fixed CSV
scp outputs/mts_final_dataset.csv username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/outputs/
```

Replace `username` with your actual HPC username.

## How to Run on HPC

After transferring the files:

```bash
# SSH to HPC
ssh username@spartan.hpc.unimelb.edu.au

# Navigate to project directory
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2

# Submit the job
sbatch mts_pipeline.slurm
```

## What Will Happen

### Scenario 1: FMA Data Doesn't Exist on HPC
```
Phase 0: Downloads FMA data (7.5 GB, ~20-30 min)
Phase 1: Skipped (CSV exists) OR runs data prep
Phase 2: Validates CSV → Starts training
Phase 3: Generates samples
```

### Scenario 2: FMA Data Already Exists
```
Phase 0: Verifies data → Skip download (fast!)
Phase 1: Skipped (CSV exists) OR runs data prep
Phase 2: Validates CSV → Starts training
Phase 3: Generates samples
```

### Scenario 3: CSV Missing audio_path Column
```
Phase 0: Verifies FMA data
Phase 1: May skip or run
Phase 2: Detects missing audio_path → Auto-regenerates CSV → Starts training
Phase 3: Generates samples
```

## New Verification System

### How It Works

The pipeline now has **3 layers of verification**:

**Layer 1: SLURM Pre-Checks** (Lines 546-674 in mts_pipeline.slurm)
- Verifies CSV exists
- Checks audio_path column present
- Counts audio files in directories
- Compares file counts to dataset rows

**Layer 2: Detailed Verification Script** (`verify_dataset.py`)
- Samples 50 files from CSV
- Checks each file path exists
- Reports empty paths, missing files, path errors
- Calculates success rate (must be >80% to pass)

**Layer 3: Dataset Loader Verification** (`train_mts_hpc.py`)
- Verifies paths when loading dataset
- Checks first 10 samples on initialization
- Logs detailed errors for each missing file
- Uses fallback audio if file not found

### Manual Verification

You can manually verify the dataset anytime:

```bash
# Check dataset integrity
python3 verify_dataset.py

# Check first 100 samples
python3 verify_dataset.py --max-check 100

# Check all samples (slower)
python3 verify_dataset.py --max-check 0
```

Output example:
```
======================================================================
Verification Results
======================================================================
Total checked:  50
Valid files:    47 ✅
Missing files:  3 ⚠️
Empty paths:    0 ✅
Path errors:    0 ✅

Success Rate: 94.0%
⚠️  WARNING: Some files missing but >80% valid
   Training may proceed with fallback audio for missing files
```

## Expected Output

When you check the output log, you'll see:

```
======================================================================
        MTS-2 Complete Pipeline - Single Job Execution
======================================================================
Phase 0: FMA Data Verification (check/download dataset)
Phase 1: Data Preparation (augmentation + text generation)
Phase 2: Model Training (GPU-accelerated)
Phase 3: Sample Generation (quality validation)
======================================================================

...

======================================================================
PHASE 0: FMA DATA VERIFICATION
======================================================================
  ✅ FMA metadata found: fma_data/fma_metadata
  ✅ FMA audio found: 8000 files in fma_data/fma_small

✅ FMA dataset is ready!

======================================================================
PHASE 1: DATA PREPARATION
======================================================================
✅ Found existing processed data: outputs/mts_final_dataset.csv
   Skipping data preparation phase

======================================================================
PHASE 2: MODEL TRAINING
======================================================================
🔍 Verifying dataset before training...
  📊 Dataset contains: 150 samples
  ✅ Audio paths present in CSV
  ✅ Augmented audio files: 100
  ✅ FMA audio files: 8000
  ✅ Sufficient audio files available for training

✅ Dataset verification complete - ready for training!

🚀 Starting MTS model training...
```

## Troubleshooting

### If Phase 0 fails to download:
The script will continue with a warning. You can manually download:
```bash
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2/fma_data
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_metadata.zip
unzip fma_small.zip
```

### If CSV regeneration fails:
Check the log for errors, then manually run:
```bash
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2
conda activate mts-data
python regenerate_dataset_csv.py
```

### If training still shows empty path errors:
```bash
# Verify CSV has audio_path column:
head -1 outputs/mts_final_dataset.csv

# Should include: ...,audio_path

# Check a few sample rows:
head -5 outputs/mts_final_dataset.csv | cut -d',' -f1,18
```

## Summary

✅ **Bug Fixed**: CSV now includes audio_path column
✅ **Auto-Download**: FMA data downloads automatically if missing
✅ **Validation**: Comprehensive checks before training
✅ **Auto-Repair**: CSV regenerates automatically if needed
✅ **Ready to Deploy**: Transfer files and submit job

The pipeline is now much more robust and will handle edge cases automatically!

---

**Next Step**: Transfer the updated files to HPC and submit the job with `sbatch mts_pipeline.slurm`
