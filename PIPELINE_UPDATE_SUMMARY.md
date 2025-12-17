# MTS-2 Pipeline Update Summary

**Date**: 2025-12-17
**Update**: Unified SLURM script for single-job submission

## What Changed

### Before: Three Separate Jobs
You previously needed to submit **three different SLURM scripts**:
1. `slurm_data_prep_only.sh` - Data preparation
2. `slurm_train_mts.sh` - Model training
3. Manual generation step

This meant:
- Submitting multiple jobs sequentially
- Waiting between submissions
- Managing dependencies manually

### After: One Complete Pipeline Job
Now you submit **one script** that does everything:

```bash
sbatch mts_pipeline.slurm
```

## Key Improvements

### 1. Single Submission
- **Old**: 3 separate `sbatch` commands
- **New**: 1 `sbatch` command
- **Benefit**: Less manual work, no need to monitor and resubmit

### 2. Automatic Phase Progression
The script automatically moves through three phases:

```
Phase 1: Data Preparation
   ↓ (if successful)
Phase 2: Model Training
   ↓ (if successful)
Phase 3: Sample Generation
```

### 3. Smart Skipping
- If `outputs/mts_final_dataset.csv` exists → **skips Phase 1**
- If `checkpoints/mts_best.pt` exists → **resumes training**
- Saves time on re-runs!

### 4. Better Resource Allocation
Updated SLURM parameters for the complete workflow:

| Parameter | Old (data only) | New (complete) |
|-----------|----------------|----------------|
| CPUs      | 4              | 8              |
| Memory    | 24GB           | 64GB           |
| Time      | 12 hours       | 7 days         |
| Job Name  | mts-data-pipeline | mts-complete-pipeline |

### 5. Comprehensive Exit Handling
Each phase has its own exit code:
- Phase 1 fails → Stop (can't train without data)
- Phase 2 fails → Stop (can't generate without model)
- Phase 3 fails → Continue (non-critical, job still successful)

### 6. Detailed Summary Report
At the end, you get a complete summary:

```
====================================================================
                    COMPLETE PIPELINE SUMMARY
====================================================================
Phase Results:
  1. Data Preparation:  ✅ SUCCESS
  2. Model Training:    ✅ SUCCESS
  3. Sample Generation: ✅ SUCCESS

Overall Status: ✅ SUCCESS

📁 Output Locations:
  Data:        ./data/augmented/audio/
  Checkpoints: ./checkpoints/
  Samples:     ./generated_samples/
  Outputs:     ./outputs/

📊 Statistics:
  Dataset rows: 1000
  Audio files:  3000 (2.5G)
  Checkpoints:  10
  Samples:      5

🎉 All phases completed successfully!
```

## File Changes

### Modified Files
- **[mts_pipeline.slurm](mts_pipeline.slurm)** (593 lines)
  - Added Phase 2: Model Training (lines 393-467)
  - Added Phase 3: Sample Generation (lines 469-509)
  - Added comprehensive summary (lines 511-593)
  - Updated SLURM headers for complete workflow

### New Files
- **[SLURM_SUBMISSION_GUIDE.md](SLURM_SUBMISSION_GUIDE.md)** - Complete usage guide
- **[PIPELINE_UPDATE_SUMMARY.md](PIPELINE_UPDATE_SUMMARY.md)** - This file

### Preserved Files
- `slurm_data_prep_only.sh` - Still available for data-only runs
- `slurm_train_mts.sh` - Still available for training-only runs
- All Python scripts unchanged

## How to Use

### Quick Start
```bash
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2
sbatch mts_pipeline.slurm
```

### Monitor Progress
```bash
# Watch output
tail -f /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-<JOB_ID>.out

# Check job status
squeue -u $USER
```

### Check Results
```bash
# View final summary
cat /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-<JOB_ID>.out | tail -50

# Check outputs
ls -lh outputs/mts_final_dataset.csv
ls -lh checkpoints/
ls -lh generated_samples/
```

## Workflow Comparison

### Old Workflow
```mermaid
graph LR
    A[Submit Data Prep] --> B[Wait]
    B --> C[Check Completion]
    C --> D[Submit Training]
    D --> E[Wait]
    E --> F[Check Completion]
    F --> G[Manual Generation]
```

### New Workflow
```mermaid
graph LR
    A[Submit Pipeline] --> B[Automatic Execution]
    B --> C[Get Results]
```

## Training Configuration

Default settings in the unified script:

```bash
BATCH_SIZE=8      # Batch size for training
EPOCHS=200        # Total training epochs
LR=1e-4          # Learning rate
SAVE_EVERY=20     # Save checkpoint every N epochs
```

Easily customizable by editing lines 403-406 in [mts_pipeline.slurm](mts_pipeline.slurm:403-406).

## Benefits

1. **Time Savings**: No manual intervention between phases
2. **Error Handling**: Automatic failure detection and reporting
3. **Resource Efficiency**: Single job allocation for entire workflow
4. **Reproducibility**: One command runs everything consistently
5. **Resume Support**: Can resume interrupted training automatically
6. **Smart Caching**: Skips completed phases on re-run

## Backward Compatibility

Your old scripts still work if you need them:
- `slurm_data_prep_only.sh` - For data preparation only
- `slurm_train_mts.sh` - For training only

But the new unified script is **recommended** for complete runs.

## Technical Details

### Phase 1: Data Preparation
- Uses existing `run_batch_pipeline.py` or `run_pipeline.py`
- Creates `outputs/mts_final_dataset.csv`
- Generates audio files in `data/augmented/audio/`
- Exit code stored in `$DATA_PREP_EXIT_CODE`

### Phase 2: Model Training
- Runs `train_mts_hpc.py` with optimized parameters
- Saves checkpoints to `checkpoints/`
- Best model: `checkpoints/mts_best.pt`
- Exit code stored in `$TRAIN_EXIT_CODE`

### Phase 3: Sample Generation
- Runs `test_generation.py` if available
- Creates samples in `generated_samples/`
- Non-critical (job succeeds even if this fails)
- Exit code stored in `$GEN_EXIT_CODE`

### Final Exit Code Logic
```bash
if Phase1 SUCCESS and Phase2 SUCCESS:
    exit 0  # Success
elif Phase1 FAILED:
    exit $DATA_PREP_EXIT_CODE
else:
    exit $TRAIN_EXIT_CODE
```

## Testing Recommendations

Before running the full 200-epoch training:

1. **Test run** with reduced epochs (edit line 404):
   ```bash
   EPOCHS=5  # Quick test
   ```

2. **Submit** and monitor:
   ```bash
   sbatch mts_pipeline.slurm
   tail -f /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-*.out
   ```

3. **Verify** all phases complete successfully

4. **Full run** with 200 epochs (revert line 404):
   ```bash
   EPOCHS=200  # Production
   ```

## Questions?

- **How do I only run data prep?** Use `slurm_data_prep_only.sh`
- **How do I resume training?** Just resubmit, it auto-resumes
- **How do I skip data prep?** It auto-skips if CSV exists
- **Can I change epochs?** Yes, edit line 404 in the script
- **What if Phase 3 fails?** Job still succeeds if training worked

## Summary

✅ Single job submission replaces multiple submissions
✅ Automatic phase progression with smart skipping
✅ Better resource allocation (8 CPUs, 64GB, 7 days)
✅ Comprehensive error handling and reporting
✅ Resume support for interrupted training
✅ Detailed final summary with statistics

**You're ready to go! Just run:**
```bash
sbatch mts_pipeline.slurm
```

---

**For detailed usage instructions**, see [SLURM_SUBMISSION_GUIDE.md](SLURM_SUBMISSION_GUIDE.md)
