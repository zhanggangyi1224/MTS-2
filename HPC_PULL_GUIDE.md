# HPC Pull Guide - Update Code via Git

## ✅ Local Changes Committed and Pushed

All fixes have been committed to git and pushed to the repository.

**Commit:** `0412ee6` - Fix audio path bug and enhance SLURM pipeline

## Steps to Update HPC

### Step 1: SSH to HPC and Navigate to Project

```bash
ssh gangyiz@spartan.hpc.unimelb.edu.au
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2
```

### Step 2: Cancel Running Job (if still running)

```bash
# Check if job is still running
squeue -u gangyiz

# If job 19878746 is still running, cancel it
scancel 19878746
```

### Step 3: Stash or Backup Any Local Changes on HPC

```bash
# Check what files are modified on HPC
git status

# If there are local changes you want to keep, stash them
git stash

# Or if you want to discard HPC changes and use the new code
git reset --hard HEAD
```

### Step 4: Pull Latest Changes

```bash
# Pull the latest changes from main branch
git pull origin main

# Verify the pull was successful
git log --oneline -1
# Should show: 0412ee6 Fix audio path bug and enhance SLURM pipeline
```

### Step 5: Verify Updated Files

```bash
# Check that key files were updated
ls -lh mts_pipeline.slurm train_mts_hpc.py verify_dataset.py
ls -lh src/batch_processor.py regenerate_dataset_csv.py

# Verify SLURM script has PIP_PREFIX
grep "PIP_PREFIX" mts_pipeline.slurm
# Should show: export PIP_PREFIX=$PROJECT_DIR/packages
```

### Step 6: Delete Old Broken CSV

```bash
# Delete the CCMusic CSV with empty audio paths
rm outputs/mts_final_dataset.csv

# Optional: Also delete intermediate files to force fresh start
rm -f outputs/step3_labeled_dataset.csv
rm -rf outputs/augmentation/step2_augmentation_results.json
```

### Step 7: Verify Configuration

```bash
# Check that config uses FMA dataset
grep "use_fma_dataset" config/config_hpc.yaml
# Should show: use_fma_dataset: true

# If it shows false, edit it
nano config/config_hpc.yaml
# Change line 11 to: use_fma_dataset: true
# Save: Ctrl+O, Enter, Ctrl+X
```

### Step 8: Resubmit Job

```bash
# Submit the updated job
sbatch mts_pipeline.slurm

# Note the new job ID
# Example output: Submitted batch job 19878800

# Monitor the output
tail -f /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-*.out
```

## What to Expect

### Phase 0: FMA Data Verification (~1 min)
```
======================================================================
PHASE 0: FMA DATA VERIFICATION
======================================================================
  ✅ FMA metadata found: fma_data/fma_metadata
  ✅ FMA audio found: 8000 files in fma_data/fma_small

✅ FMA dataset is ready!
```

### Phase 1: Data Preparation (~10-20 min)
```
======================================================================
PHASE 1: DATA PREPARATION
======================================================================
⚠️  No processed data found, running data preparation...
🎵 Loading FMA audio from fma_data/fma_small (50 files)...
Loading FMA audio: 100%|████████████| 50/50 [00:30<00:00]
✅ Loaded 50 FMA tracks with audio.

[Processing pipeline...]
✅ Phase 1 complete
```

### Phase 2: Model Training (~4-8 hours)
```
======================================================================
PHASE 2: MODEL TRAINING
======================================================================
🔍 Verifying dataset before training...
  ✅ audio_path column found
  📊 Dataset contains: 150 samples
  ✅ Audio paths present in CSV

🔍 Running detailed dataset verification...
======================================================================
Verification Results
======================================================================
Total checked:  50
Valid files:    50 ✅
Missing files:  0 ✅
Empty paths:    0 ✅
Path errors:    0 ✅

Success Rate: 100.0%
✅ PASSED: All checked files exist

🚀 Starting MTS model training...
📖 Loading dataset from: outputs/mts_final_dataset.csv
✅ Loaded train set: 120 samples
✅ 10/10 sample files verified successfully

Epoch 1/200
  Train loss: X.XXXX
  Val loss: X.XXXX
```

## Troubleshooting

### If git pull shows merge conflicts:

```bash
# Reset to remote state
git fetch origin
git reset --hard origin/main
```

### If you get "changes would be overwritten" error:

```bash
# Stash local changes
git stash

# Or discard local changes
git reset --hard HEAD

# Then pull again
git pull origin main
```

### If FMA data doesn't exist:

Phase 0 will automatically download it (~7.5 GB, 20-30 minutes).

### If you still see CCMusic data in logs:

Make sure you deleted the old CSV:
```bash
rm outputs/mts_final_dataset.csv
ls -lh outputs/mts_final_dataset.csv  # Should show "No such file"
```

## Verification Checklist

After pulling and before submitting job:

- [ ] Git pull successful (commit 0412ee6)
- [ ] Old job canceled (if running)
- [ ] Broken CSV deleted
- [ ] Config uses FMA dataset (use_fma_dataset: true)
- [ ] New job submitted (sbatch mts_pipeline.slurm)

## Key Changes in This Update

### Code Fixes
1. ✅ `src/batch_processor.py` - Adds audio_path to CSV
2. ✅ `train_mts_hpc.py` - Handles NaN values, prevents TypeError
3. ✅ `verify_dataset.py` - Validates dataset with NaN handling
4. ✅ `mts_pipeline.slurm` - Phase 0, validation, project packages
5. ✅ `regenerate_dataset_csv.py` - Rebuilds CSV from intermediates

### Documentation Added
1. 📄 `README_FIXES.md` - Master summary
2. 📄 `URGENT_FIX_STEPS.md` - Quick reference
3. 📄 `FIX_CCMUSIC_TO_FMA.md` - Dataset switching guide
4. 📄 `PACKAGE_INSTALLATION_CHANGES.md` - Package paths
5. 📄 `HPC_PULL_GUIDE.md` - This file

## Timeline

- **Git pull**: < 1 minute
- **Cancel job + delete CSV**: < 1 minute
- **Resubmit**: < 1 minute
- **Phase 0**: ~1 minute (data exists)
- **Phase 1**: ~10-20 minutes
- **Phase 2**: ~4-8 hours
- **Phase 3**: ~5-10 minutes

**Total:** ~5-8 hours after you run `git pull`

## Success Indicators

Everything is working when you see:

✅ **Git pull successful** - Shows commit 0412ee6
✅ **Phase 0 complete** - FMA dataset verified
✅ **Phase 1 complete** - CSV created with FMA data
✅ **Verification 100%** - All audio files found
✅ **Training starts** - No TypeError, loads samples successfully

---

## Quick Commands Summary

```bash
# Complete update workflow
ssh gangyiz@spartan.hpc.unimelb.edu.au
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2
scancel 19878746  # Cancel old job
git pull origin main  # Pull updates
rm outputs/mts_final_dataset.csv  # Delete broken CSV
sbatch mts_pipeline.slurm  # Submit new job
tail -f /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-*.out  # Monitor
```

**That's it!** 🚀 Much easier than using `scp`!
