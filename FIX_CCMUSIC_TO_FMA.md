# Fixing CCMusic Dataset Issue - Switch to FMA

## Problem

The current dataset CSV contains CCMusic data without audio paths:
```
ID: ccmusic_0000, ccmusic_0001, etc.
audio_path: NaN (empty/missing)
```

This causes training to fail with:
```
TypeError: expected str, bytes or os.PathLike object, not float
```

## Root Cause

The data preparation phase ran with CCMusic dataset configuration, but CCMusic data doesn't include actual audio files, only metadata. The configuration should use FMA dataset instead.

## Solution

You need to **delete the old CSV** and **regenerate with FMA data**.

### Step 1: Transfer Updated Files to HPC

The following files have been updated to handle NaN values properly:

```bash
cd /Users/zhanggangyi/Desktop/MTS-2

# Transfer updated files
scp train_mts_hpc.py \
    verify_dataset.py \
    mts_pipeline.slurm \
    username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/
```

Replace `username` with your HPC username.

### Step 2: Delete Old CSV on HPC

```bash
# SSH to HPC
ssh username@spartan.hpc.unimelb.edu.au

# Navigate to project
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2

# Delete the old CCMusic CSV
rm outputs/mts_final_dataset.csv

# Optional: Also delete intermediate files if they exist
rm -f outputs/step3_labeled_dataset.csv
rm -f outputs/augmentation/step2_augmentation_results.json
```

### Step 3: Verify Configuration

Check that the configuration file uses FMA dataset:

```bash
# Check the config
grep "use_fma_dataset" config/config_hpc.yaml

# Should show:
# use_fma_dataset: true
```

If it shows `false`, edit it:
```bash
# Edit config
nano config/config_hpc.yaml

# Change line 11 to:
# use_fma_dataset: true

# Save and exit (Ctrl+O, Enter, Ctrl+X)
```

### Step 4: Verify FMA Data Exists

```bash
# Check FMA data
ls -lh fma_data/fma_small/ | head
ls -lh fma_data/fma_metadata/tracks.csv

# Count MP3 files
find fma_data/fma_small -name "*.mp3" | wc -l
# Should show ~8000 files
```

If FMA data doesn't exist, the SLURM script Phase 0 will download it automatically.

### Step 5: Resubmit the Job

```bash
# Submit the job
sbatch mts_pipeline.slurm

# Monitor output
tail -f /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-*.out
```

## What Will Happen

### Phase 0: FMA Data Verification
```
======================================================================
PHASE 0: FMA DATA VERIFICATION
======================================================================
[TIMESTAMP] Checking FMA dataset availability...
  ✅ FMA metadata found: fma_data/fma_metadata
  ✅ FMA audio found: 8000 files in fma_data/fma_small

✅ FMA dataset is ready!
```

### Phase 1: Data Preparation
Since you deleted the old CSV, Phase 1 will run and create a NEW CSV with FMA data:

```
======================================================================
PHASE 1: DATA PREPARATION
======================================================================
⚠️  No processed data found, running data preparation...
[Using batch processing for memory efficiency...]

🎵 Loading FMA audio from fma_data/fma_small (50 files)...
Loading FMA audio: 100%|████████████| 50/50 [00:30<00:00]
✅ Loaded 50 FMA tracks with audio.

[Data pipeline processing...]
- Step 1: Load data ✅
- Step 2: Augmentation ✅ (creates augmented audio files)
- Step 3: Text generation ✅
- Step 4: Structure analysis ✅
- Step 5: Final CSV ✅

📊 Final dataset created: outputs/mts_final_dataset.csv
   Total samples: 150 (50 original + 100 augmented)
   All samples have audio_path populated
```

The new CSV will have IDs like:
```
fma_000002, fma_000003, etc.
audio_path: fma_data/fma_small/000/000002.mp3
```

### Phase 2: Model Training
```
======================================================================
PHASE 2: MODEL TRAINING
======================================================================
🔍 Verifying dataset before training...
  ✅ audio_path column found
  📊 Dataset contains: 150 samples
  ✅ Audio paths present in CSV
  ✅ Augmented audio files: 100
  ✅ FMA audio files: 8000

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

Training will now proceed successfully!
```

## Changes Made to Fix NaN Handling

### 1. train_mts_hpc.py
**Fixed line 105-106** and **line 146** to handle NaN values:
```python
# Before (would crash on NaN):
if not audio_path:
    empty_paths.append(...)

# After (handles NaN properly):
if pd.isna(audio_path) or not isinstance(audio_path, str) or not audio_path.strip():
    empty_paths.append(...)
```

### 2. verify_dataset.py
**Fixed line 73** to handle NaN values:
```python
# Before:
if not audio_path or pd.isna(audio_path):

# After (more robust):
if pd.isna(audio_path) or not isinstance(audio_path, str) or not audio_path.strip():
```

These changes ensure that:
- NaN values (from pandas) are detected before Path() is called
- Non-string values are handled gracefully
- Empty strings are detected properly

## Verification

After the job completes, verify the new dataset:

```bash
# Check CSV header
head -1 outputs/mts_final_dataset.csv | tr ',' '\n' | grep -n "audio_path"

# Should show: 18:audio_path (or similar line number)

# Check sample IDs
head -5 outputs/mts_final_dataset.csv | cut -d',' -f1

# Should show:
# id
# fma_000002
# fma_000005
# etc.

# Check audio paths (column 18 in the CSV)
tail -n +2 outputs/mts_final_dataset.csv | cut -d',' -f18 | head -5

# Should show actual file paths:
# fma_data/fma_small/000/000002.mp3
# fma_data/fma_small/000/000005.mp3
# etc.
```

## Configuration Details

The key configuration in `config/config_hpc.yaml`:

```yaml
data:
  dataset_name: ccmusic-database/music_genre  # Not used when use_fma_dataset is true
  use_simulated_data: false  # Don't use simulated data
  use_fma_dataset: true      # ✅ Use real FMA audio
  fma_audio_dir: ./fma_data/fma_small
  fma_metadata_path: ./fma_data/fma_metadata/tracks.csv
  fma_max_files: 1000        # Limit to 1000 files (you can adjust)
  fma_clip_duration: 30.0    # Load 30 seconds per track
```

## Timeline

**First run after fixes:**
- Phase 0: ~1 minute (data already exists)
- Phase 1: ~10-20 minutes (processing 50 songs)
- Phase 2: ~4-8 hours (training 200 epochs)
- Phase 3: ~5-10 minutes (sample generation)

**Subsequent runs:**
- Phase 0: ~1 minute (skip)
- Phase 1: ~1 minute (skip - CSV exists)
- Phase 2: ~4-8 hours
- Phase 3: ~5-10 minutes

## Troubleshooting

### If Phase 0 downloads take too long:
The download is ~7.5 GB and may take 20-30 minutes. It's resumable with `wget -c`, so if it fails, just resubmit the job.

### If you still see "No valid files found":
Check that FMA data downloaded correctly:
```bash
find fma_data/fma_small -name "*.mp3" | wc -l
# Should be ~8000
```

### If you want to use more FMA files:
Edit `config/config_hpc.yaml`:
```yaml
fma_max_files: 2000  # Use 2000 instead of 1000
```
This will create a larger dataset (e.g., 2000 original + 6000 augmented = 8000 total samples).

## Summary

✅ **Root cause**: Old CSV has CCMusic data without audio paths
✅ **Solution**: Delete old CSV, regenerate with FMA data
✅ **Files updated**: train_mts_hpc.py, verify_dataset.py to handle NaN
✅ **Configuration**: use_fma_dataset: true (already set)
✅ **Next step**: Transfer files, delete CSV, resubmit job

---

**Bottom line**: Delete `outputs/mts_final_dataset.csv` on HPC and resubmit the job. Phase 1 will create a proper CSV with FMA audio paths, and training will succeed! 🚀
