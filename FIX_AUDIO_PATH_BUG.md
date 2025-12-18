# Audio Path Bug Fix & SLURM Data Loading Enhancement

## Problem
The training script was failing with the error:
```
Error loading sample 1972: Error opening '': System error.
```

**Root cause**: The `mts_final_dataset.csv` file was missing the `audio_path` column, causing the training script to try loading files with empty paths.

**Additional issue**: The SLURM script was not checking if FMA data exists before running the pipeline.

## Files Changed

### 1. `src/batch_processor.py` (Lines 724-746)
**Fixed**: Added `audio_path` field to the CSV generation in `_create_final_csv()` method.

The fix handles both:
- **Original songs**: Use `audio_path` field (from data loader)
- **Augmented songs**: Use `audio_file_path` field (from augmentation)

```python
# Get audio path - check both fields (augmented uses audio_file_path, original uses audio_path)
audio_path = song.get("audio_file_path", "") or song.get("audio_path", "")

row = {
    # ... other fields ...
    "audio_path": audio_path
}
```

### 2. `regenerate_dataset_csv.py` (NEW)
**Added**: Script to regenerate the CSV from existing intermediate files without rerunning the entire pipeline.

This script:
- Reads `outputs/step3_labeled_dataset.csv` for metadata
- Reads `outputs/augmentation/step2_augmentation_results.json` for audio paths
- Reads `outputs/structure/structure_annotations.json` for structure info
- Combines all data and generates a new CSV with audio paths

### 3. `mts_pipeline.slurm` (ENHANCED)
**Added**: Phase 0 - FMA Data Verification and Download

New features:
- **Automatic data checking**: Verifies FMA metadata and audio files exist
- **Smart downloading**: Downloads and extracts FMA data if missing
- **Resume capability**: Detects partially downloaded files and resumes
- **Pre-training validation**: Verifies CSV has audio_path column before training
- **Audio file counting**: Ensures sufficient audio files exist for training
- **Auto-repair**: Automatically regenerates CSV if audio_path column is missing

The SLURM script now has 4 phases:
- **Phase 0**: FMA Data Verification (check/download dataset)
- **Phase 1**: Data Preparation (augmentation + text generation)
- **Phase 2**: Model Training (GPU-accelerated)
- **Phase 3**: Sample Generation (quality validation)

## How to Apply the Fix

### Option 1: Regenerate CSV from existing data (Quick)
If you already have processed data and just need to add audio paths:

```bash
python3 regenerate_dataset_csv.py
```

This will create a new `outputs/mts_final_dataset.csv` with the `audio_path` column.

### Option 2: Rerun the pipeline (Complete)
If you're running the full pipeline from scratch:

The fix is already in `src/batch_processor.py`, so just run:

```bash
# For HPC
sbatch mts_pipeline.slurm

# For local
python3 run_batch_pipeline.py --config config/batch_config.yaml
```

## Verification

After applying the fix, verify the CSV has audio paths:

```bash
# Check CSV header includes audio_path
head -1 outputs/mts_final_dataset.csv

# Should show: id,original_id,title,...,audio_path

# Check sample rows
head -5 outputs/mts_final_dataset.csv | cut -d, -f1,18
```

Expected output:
```
id,audio_path
fma_000002,fma_data/fma_small/002/000002.mp3
fma_000002_aug_01,data/augmented/audio/fma_000002_aug_01.wav
```

## Testing on HPC

Before running the full training, test with a small number of samples:

```bash
# Test training with 1 epoch
python3 train_mts_hpc.py \
    --batch-size 4 \
    --epochs 1 \
    --dataset outputs/mts_final_dataset.csv
```

The training should now load samples successfully instead of showing "Error opening ''" errors.

## Path Structure

The CSV now contains two types of audio paths:

1. **Original FMA songs**:
   - Format: `fma_data/fma_small/XXX/00XXXX.mp3`
   - Example: `fma_data/fma_small/002/000002.mp3`

2. **Augmented songs**:
   - Format: `data/augmented/audio/{id}.wav`
   - Example: `data/augmented/audio/fma_000002_aug_01.wav`

## Future Prevention

The fix is permanent in the codebase:
- ✅ `src/batch_processor.py` now includes audio paths in CSV generation
- ✅ Future pipeline runs will automatically include audio paths
- ✅ The regeneration script is available for manual fixes if needed

## Status
- [x] Bug identified
- [x] Fix implemented in batch_processor.py
- [x] Regeneration script created
- [x] CSV regenerated with audio paths (150 rows, 0 missing)
- [x] Verified audio files exist
- [x] SLURM script enhanced with Phase 0 (FMA data verification)
- [x] Added pre-training validation checks
- [x] Added auto-repair for missing audio_path column
- [ ] Tested on HPC (pending next SLURM job)

## SLURM Script Enhancements Detail

### Phase 0: FMA Data Verification

The script now checks for FMA data before starting the pipeline:

```bash
# Checks performed:
1. Verify FMA metadata directory exists with tracks.csv
2. Verify FMA audio directory has >7000 MP3 files
3. Download missing data automatically using wget
4. Extract zip files if needed
5. Re-verify after download
```

**Benefits**:
- No more manual data download steps
- Handles incomplete downloads (resume with -c flag)
- Clear error messages if download fails
- Pipeline continues even if FMA data unavailable (uses fallback)

### Phase 2 Pre-Training Validation

Before starting training, the script now:

```bash
# Validation checks:
1. Verify mts_final_dataset.csv exists
2. Check if audio_path column is present in CSV
3. Auto-regenerate CSV if audio_path missing
4. Count total samples in dataset
5. Sample and verify audio paths are not empty
6. Count augmented audio files
7. Count FMA audio files
8. Verify total audio files >= dataset rows
```

**Benefits**:
- Catches the audio_path bug automatically
- Provides detailed pre-flight checks
- Auto-repairs if possible
- Clear error messages with actionable steps
- Prevents wasted GPU time on broken datasets

## What Happens on Next HPC Run

When you submit the SLURM job, it will:

1. **Phase 0**: Check if FMA data exists at `$PROJECT_DIR/fma_data/`
   - If exists and complete → Skip download
   - If missing → Download automatically (7.2 GB + 342 MB)
   - If partial → Resume download

2. **Phase 1**: Either skip (if CSV exists) or run data preparation
   - Now generates CSV with audio_path column (fix applied)

3. **Phase 2**: Pre-training validation
   - Check CSV has audio_path column
   - If missing → Auto-regenerate using regenerate_dataset_csv.py
   - Verify audio files exist
   - Then start training

4. **Phase 3**: Sample generation (if training succeeds)
