# MTS-2 Bug Fixes & Enhancements - Master Summary

## Current Status: ⚠️ **ACTION REQUIRED**

Your HPC job (19878746) is running but **will fail** because it's using old files. You need to transfer updated files and restart.

## Quick Start - Do This Now

See **[URGENT_FIX_STEPS.md](URGENT_FIX_STEPS.md)** for immediate actions.

**Summary:**
```bash
# 1. Transfer files (from local machine)
scp mts_pipeline.slurm train_mts_hpc.py verify_dataset.py \
    src/batch_processor.py regenerate_dataset_csv.py \
    gangyiz@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/

# 2. On HPC
scancel 19878746  # Cancel current job
rm outputs/mts_final_dataset.csv  # Delete broken CSV
sbatch mts_pipeline.slurm  # Resubmit with fixed files
```

## What Was Fixed

### Problem 1: Empty Audio Paths (CRITICAL)
- **Issue**: Training failed with `Error opening '': System error` for all samples
- **Cause**: CSV missing `audio_path` column
- **Fix**: Updated `src/batch_processor.py` to include audio_path in CSV generation
- **Status**: ✅ Fixed

### Problem 2: CCMusic Data Instead of FMA (CRITICAL)
- **Issue**: Current HPC dataset has CCMusic data with NaN audio paths
- **Cause**: Old data preparation run used CCMusic dataset
- **Fix**: Delete old CSV, let Phase 1 regenerate with FMA data (config already set to `use_fma_dataset: true`)
- **Status**: ⚠️ Requires action (delete CSV on HPC)

### Problem 3: No FMA Data Verification
- **Issue**: Pipeline didn't check if FMA data exists before running
- **Fix**: Added Phase 0 to SLURM script - auto-downloads FMA data if missing
- **Status**: ✅ Fixed

### Problem 4: TypeError on NaN Values
- **Issue**: Training crashes with `TypeError: expected str, bytes or os.PathLike object, not float`
- **Cause**: Code didn't handle NaN audio_path values properly
- **Fix**: Updated `train_mts_hpc.py` and `verify_dataset.py` to handle NaN values
- **Status**: ✅ Fixed

### Problem 5: Packages Installing to Home Directory
- **Issue**: Packages install to `~/.local` (home quota limits)
- **Fix**: Updated SLURM script to install packages to `$PROJECT_DIR/packages`
- **Status**: ✅ Fixed (but HPC has old SLURM script)

## Documentation Files

### Primary Guides
1. **[URGENT_FIX_STEPS.md](URGENT_FIX_STEPS.md)** ⚠️ **START HERE**
   - Immediate actions for current HPC job
   - Step-by-step commands to transfer files and restart

2. **[FIX_CCMUSIC_TO_FMA.md](FIX_CCMUSIC_TO_FMA.md)**
   - Detailed guide for switching from CCMusic to FMA dataset
   - Explains why CCMusic data has empty audio paths
   - Shows what FMA dataset will look like

3. **[PACKAGE_INSTALLATION_CHANGES.md](PACKAGE_INSTALLATION_CHANGES.md)**
   - Explains package installation path changes
   - Shows how packages now install to project directory
   - Troubleshooting for package-related issues

### Reference Guides
4. **[COMPLETE_FIX_SUMMARY.md](COMPLETE_FIX_SUMMARY.md)**
   - Comprehensive overview of all fixes
   - Three-layer verification system details
   - Troubleshooting common errors

5. **[SLURM_UPDATE_SUMMARY.md](SLURM_UPDATE_SUMMARY.md)**
   - SLURM pipeline enhancements
   - Transfer commands and deployment instructions
   - Expected output for each phase

6. **[FIX_AUDIO_PATH_BUG.md](FIX_AUDIO_PATH_BUG.md)**
   - Technical details of the audio_path bug
   - Code changes in batch_processor.py
   - Status checklist

7. **[QUICK_FIX_GUIDE.md](QUICK_FIX_GUIDE.md)**
   - Quick reference for specific error scenarios
   - Alternative solutions if automatic fixes fail

## Files That Need Transfer

### Critical (Must Transfer)
1. `mts_pipeline.slurm` - Updated with Phase 0, validation, and package paths
2. `train_mts_hpc.py` - NaN handling prevents crashes
3. `verify_dataset.py` - NaN handling for verification
4. `src/batch_processor.py` - Generates CSV with audio_path column
5. `regenerate_dataset_csv.py` - Can regenerate CSV from intermediate files

### Transfer Command
```bash
cd /Users/zhanggangyi/Desktop/MTS-2

scp mts_pipeline.slurm \
    train_mts_hpc.py \
    verify_dataset.py \
    regenerate_dataset_csv.py \
    gangyiz@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/

scp src/batch_processor.py \
    gangyiz@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/src/
```

## What Will Happen After Fix

### Phase 0: FMA Data Verification (~1 min if data exists)
```
✅ FMA metadata found: fma_data/fma_metadata
✅ FMA audio found: 8000 files in fma_data/fma_small
✅ FMA dataset is ready!
```

### Phase 1: Data Preparation (~10-20 min)
```
⚠️  No processed data found, running data preparation...
🎵 Loading FMA audio from fma_data/fma_small (50 files)...
[Creates augmented audio]
[Generates text prompts]
[Analyzes structure]
📊 Final dataset created: outputs/mts_final_dataset.csv
   Total samples: 150 (50 original + 100 augmented)
   ✅ All samples have valid audio_path values
```

### Phase 2: Model Training (~4-8 hours)
```
🔍 Verifying dataset before training...
✅ audio_path column found
✅ 50/50 sample files verified successfully
Success Rate: 100.0%
✅ PASSED: All checked files exist

🚀 Starting MTS model training...
Epoch 1/200
  Train loss: X.XXXX
  Val loss: X.XXXX
[Training proceeds successfully]
```

## Key Changes Summary

### Before
❌ CSV missing audio_path column
❌ Training crashes with TypeError
❌ No FMA data verification
❌ Packages install to home directory
❌ CCMusic data with empty paths
❌ No pre-training validation

### After
✅ CSV includes audio_path column
✅ NaN values handled gracefully
✅ Phase 0 auto-downloads FMA data
✅ Packages install to project directory
✅ FMA data with valid audio paths
✅ Three-layer verification system

## Verification Checklist

After transferring files and resubmitting job:

- [ ] Files transferred to HPC
- [ ] Old job canceled (scancel 19878746)
- [ ] Broken CSV deleted (rm outputs/mts_final_dataset.csv)
- [ ] Config uses FMA dataset (use_fma_dataset: true)
- [ ] New job submitted (sbatch mts_pipeline.slurm)
- [ ] Phase 0 completes successfully (FMA data verified)
- [ ] Phase 1 creates new CSV with FMA data
- [ ] Phase 2 verification shows 100% success rate
- [ ] Training starts without errors

## Support

If you encounter issues:

1. Check the specific error in the SLURM output log
2. Find the error in the troubleshooting section of COMPLETE_FIX_SUMMARY.md
3. Follow the recommended solution

**Common issues:**
- "Cannot regenerate CSV - intermediate files missing" → Delete CSV, rerun Phase 1
- "No valid files found" → Check FMA data downloaded, verify paths
- "TypeError: expected str" → Transfer updated train_mts_hpc.py

## Timeline

**Immediate (< 5 minutes):**
- Transfer files
- Cancel current job
- Delete broken CSV
- Resubmit job

**Phase 0 (~1 minute):**
- Verify FMA data exists

**Phase 1 (~10-20 minutes):**
- Process 50 FMA songs
- Generate 100 augmented versions
- Create CSV with valid audio paths

**Phase 2 (~4-8 hours):**
- Train model for 200 epochs

**Phase 3 (~5-10 minutes):**
- Generate test samples

**Total time to completion:** ~5-8 hours (after you transfer files)

## Success Indicators

You'll know everything is working when you see:

```
[Phase 1 Log]
✅ Loaded 50 FMA tracks with audio.

[Phase 2 Log]
Success Rate: 100.0%
✅ PASSED: All checked files exist

[Training Log]
✅ Loaded train set: 120 samples
✅ 10/10 sample files verified successfully
🚀 Starting MTS model training...
Epoch 1/200
```

---

**Bottom Line:** Transfer the 5 updated files to HPC, delete the broken CSV, and resubmit the job. Everything else is automatic! 🚀

**Start with:** [URGENT_FIX_STEPS.md](URGENT_FIX_STEPS.md)
