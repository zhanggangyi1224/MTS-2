# Urgent Fix Steps - Current HPC Job Will Fail

## Current Situation

Your HPC job (ID: 19878746) is running but will fail at Phase 2 because:
1. ❌ Dataset CSV still has CCMusic data with empty audio_paths
2. ❌ SLURM script on HPC is OLD (packages installing to home directory)

## Immediate Actions Required

### Step 1: Transfer Updated Files (Do This NOW)

```bash
# From your local machine
cd /Users/zhanggangyi/Desktop/MTS-2

# Transfer ALL updated files at once
scp mts_pipeline.slurm \
    train_mts_hpc.py \
    verify_dataset.py \
    src/batch_processor.py \
    regenerate_dataset_csv.py \
    gangyiz@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/
```

Replace `gangyiz` with your actual username if different.

### Step 2: Cancel Current Job and Clean Up

```bash
# SSH to HPC
ssh gangyiz@spartan.hpc.unimelb.edu.au

# Cancel the running job
scancel 19878746

# Navigate to project
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2

# Delete the broken CSV with CCMusic data
rm outputs/mts_final_dataset.csv

# Optional: Also delete intermediate files to force fresh start
rm -f outputs/step3_labeled_dataset.csv
rm -rf outputs/augmentation/step2_augmentation_results.json
```

### Step 3: Verify Configuration

```bash
# Check that config uses FMA dataset
grep "use_fma_dataset" config/config_hpc.yaml

# Should show:
#   use_fma_dataset: true

# If it shows false, edit it:
nano config/config_hpc.yaml
# Change line 11 to: use_fma_dataset: true
# Save: Ctrl+O, Enter, Ctrl+X
```

### Step 4: Verify FMA Data Exists

```bash
# Check if FMA data is downloaded
ls -lh fma_data/fma_small/ | head -5

# Count MP3 files
find fma_data/fma_small -name "*.mp3" | wc -l
# Should show ~8000

# If no files found, Phase 0 will download them automatically
```

### Step 5: Resubmit Job

```bash
# Make sure you're in the project directory
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2

# Submit the updated job
sbatch mts_pipeline.slurm

# Note the new job ID, then monitor
tail -f /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-*.out
```

## What Changed in Updated Files

### 1. mts_pipeline.slurm
- ✅ Packages install to `$PROJECT_DIR/packages` (not `~/.local`)
- ✅ Added `PIP_PREFIX` environment variable
- ✅ All `pip install` commands use `--prefix $PIP_PREFIX`

### 2. train_mts_hpc.py
- ✅ Handles NaN audio_path values without crashing
- ✅ Better error messages for debugging

### 3. verify_dataset.py
- ✅ Properly detects NaN values in audio_path column
- ✅ Won't crash when encountering non-string values

### 4. src/batch_processor.py
- ✅ Already fixed - generates CSV with audio_path column
- ✅ Works with FMA data when `use_fma_dataset: true`

## Expected Output After Resubmitting

### Phase 0 (if FMA data exists):
```
✅ FMA metadata found: fma_data/fma_metadata
✅ FMA audio found: 8000 files in fma_data/fma_small
✅ FMA dataset is ready!
```

### Phase 1 (will run because CSV deleted):
```
⚠️  No processed data found, running data preparation...
🎵 Loading FMA audio from fma_data/fma_small (50 files)...
[Creates NEW CSV with FMA data and proper audio paths]
✅ Phase 1 complete
```

### Phase 2 (should succeed):
```
🔍 Verifying dataset before training...
✅ audio_path column found
✅ 50/50 sample files verified successfully
Success Rate: 100.0%
🚀 Starting MTS model training...
```

## Why Current Job Will Fail

Looking at your job output, packages are installing to:
```
/home/gangyiz/.local/lib/python3.10/site-packages
```

This means the HPC has the OLD SLURM script. Also, when Phase 2 starts, it will find the CCMusic CSV with NaN audio paths and fail with:
```
TypeError: expected str, bytes or os.PathLike object, not float
```

## Quick Verification After Transfer

```bash
# SSH to HPC
ssh gangyiz@spartan.hpc.unimelb.edu.au
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2

# Check SLURM script has PIP_PREFIX
grep "PIP_PREFIX" mts_pipeline.slurm
# Should show: export PIP_PREFIX=$PROJECT_DIR/packages

# Check pip install commands use --prefix
grep "pip install --prefix" mts_pipeline.slurm | head -3
# Should show several commands with --prefix $PIP_PREFIX
```

## Summary

**Critical Issues:**
1. ❌ HPC has old SLURM script (packages go to home directory)
2. ❌ Dataset CSV has CCMusic data with empty audio_paths

**Solution:**
1. ✅ Transfer updated files (mts_pipeline.slurm, train_mts_hpc.py, verify_dataset.py, src/batch_processor.py)
2. ✅ Cancel current job (scancel 19878746)
3. ✅ Delete broken CSV (rm outputs/mts_final_dataset.csv)
4. ✅ Resubmit job (sbatch mts_pipeline.slurm)

**Result:**
- Packages install to project directory
- Phase 1 creates FMA dataset with valid audio paths
- Training succeeds

---

**Time Estimate:**
- File transfer: 1 minute
- Cleanup: 1 minute
- Phase 0: 1 minute (data exists)
- Phase 1: 10-20 minutes (process 50 FMA songs)
- Phase 2: 4-8 hours (training)

**Next job will work!** 🚀
