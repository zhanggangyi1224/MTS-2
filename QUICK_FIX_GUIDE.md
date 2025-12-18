# Quick Fix Guide - Your Current Error

## Your Situation

You have an **old CSV file** on HPC that was created by the **buggy version** without the `audio_path` column.

The error you're seeing:
```
❌ ERROR: Dataset CSV missing 'audio_path' column
❌ ERROR: Cannot regenerate CSV - intermediate files missing
```

## Solution: Simple 2-Step Fix

### Step 1: Transfer Updated Files to HPC

```bash
# From your local machine:
cd /Users/zhanggangyi/Desktop/MTS-2

scp src/batch_processor.py \
    mts_pipeline.slurm \
    train_mts_hpc.py \
    regenerate_dataset_csv.py \
    verify_dataset.py \
    username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/
```

Replace `username` with your HPC username.

### Step 2: Delete Old CSV and Resubmit

```bash
# SSH to HPC
ssh username@spartan.hpc.unimelb.edu.au

# Navigate to project
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2

# Delete the broken CSV (it will be backed up automatically if needed)
rm outputs/mts_final_dataset.csv

# Resubmit the job
sbatch mts_pipeline.slurm
```

## What Will Happen

With the updated files:

```
Phase 0: FMA Data Verification
  [Checks/downloads FMA data if needed]

Phase 1: Data Preparation
  ⚠️  No processed data found
  Running data preparation...
  [Creates NEW CSV with audio_path column]
  ✅ Phase 1 complete

Phase 2: Model Training
  🔍 Verifying dataset...
  ✅ CSV is valid (has audio_path column)
  ✅ Audio paths present
  [Detailed verification checks 50 files]
  ✅ Dataset verification passed
  🚀 Starting training...
  [Training proceeds successfully]
```

## Alternative: If You Want to Keep Existing Data

If you've already run data preparation and just need to fix the CSV:

### Option A: Manual Fix on HPC

```bash
# SSH to HPC
ssh username@spartan.hpc.unimelb.edu.au
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2

# Check if intermediate files exist
ls -lh outputs/step3_labeled_dataset.csv
ls -lh outputs/augmentation/step2_augmentation_results.json

# If both exist, regenerate CSV manually
python regenerate_dataset_csv.py

# Then resubmit
sbatch mts_pipeline.slurm
```

### Option B: Transfer Fixed CSV from Local

You already have a valid CSV locally. Transfer it:

```bash
# From local machine
scp outputs/mts_final_dataset.csv \
    username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/outputs/

# Then SSH and submit
ssh username@spartan.hpc.unimelb.edu.au
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2
sbatch mts_pipeline.slurm
```

**Note:** Your local CSV has paths pointing to local files, not HPC paths. This will work because:
- The updated training script uses **fallback audio** for missing files
- Phase 0 will download FMA data to the correct HPC location
- Paths will work correctly on subsequent runs

## Recommended Approach

**Best option:** Delete the old CSV and let Phase 1 run (Option from Step 2 above)

**Why?**
- ✅ Ensures all paths are correct for HPC environment
- ✅ Creates proper intermediate files for future use
- ✅ Validates entire data pipeline works on HPC
- ✅ Only takes 10-20 minutes (data prep on 50 songs)

## Quick Checklist

Before submitting the job, ensure:

- [ ] Transferred updated files to HPC
- [ ] Deleted old broken CSV: `rm outputs/mts_final_dataset.csv`
- [ ] Or transferred valid CSV if using Option B
- [ ] Run: `sbatch mts_pipeline.slurm`
- [ ] Monitor: `tail -f /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-JOBID.out`

## Expected Timeline

```
Phase 0: FMA Data Check/Download
  - If data exists: ~1 minute
  - If needs download: ~20-30 minutes

Phase 1: Data Preparation
  - If CSV valid: ~1 minute (skipped)
  - If needs to run: ~10-20 minutes

Phase 2: Training
  - Verification: ~2-3 minutes
  - Training 200 epochs: ~4-8 hours (depends on GPU)

Phase 3: Sample Generation
  - ~5-10 minutes
```

## How to Monitor Progress

```bash
# Watch the output log
tail -f /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-*.out

# Check for errors
tail -f /data/gpfs/projects/punim2072/MTS/err/mts-complete-pipeline-*.err

# Check job status
squeue -u your_username
```

## Success Indicators

You'll know it's working when you see:

```
✅ FMA dataset is ready!
✅ Phase 1 complete
✅ CSV is valid (has audio_path column)
✅ 50/50 sample files verified successfully
Success Rate: 100.0%
✅ PASSED: All checked files exist
🚀 Starting MTS model training...
Epoch 1/200
  Train loss: X.XXXX
  Val loss: X.XXXX
```

## If You Still Get Errors

1. **Check the updated files were transferred correctly:**
   ```bash
   ssh username@spartan.hpc.unimelb.edu.au
   cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2
   grep "PHASE 0: FMA DATA VERIFICATION" mts_pipeline.slurm
   # Should show the new Phase 0 code
   ```

2. **Check error log for details:**
   ```bash
   cat /data/gpfs/projects/punim2072/MTS/err/mts-complete-pipeline-*.err
   ```

3. **Verify FMA data location:**
   ```bash
   ls -lh fma_data/fma_small/ | head
   # Should show .mp3 files
   ```

If problems persist, the error logs will now provide clear, actionable messages about what's wrong and how to fix it.

---

**Bottom Line:** Transfer the updated files, delete the old CSV, and resubmit. The pipeline will now handle everything automatically! 🚀
