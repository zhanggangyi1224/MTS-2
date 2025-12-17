# MTS-2 Complete Pipeline - Single SLURM Job Submission Guide

## Overview

The `mts_pipeline.slurm` script has been updated to run **all three phases in a single job**:

1. **Phase 1: Data Preparation** - Downloads dataset and creates augmented audio files
2. **Phase 2: Model Training** - Trains the MTS model on GPU (200 epochs)
3. **Phase 3: Sample Generation** - Generates test audio samples to validate quality

## Quick Start

### Submit the Complete Pipeline

```bash
# Navigate to project directory
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2

# Submit single job (runs all phases)
sbatch mts_pipeline.slurm
```

That's it! One command runs everything.

## Job Configuration

The unified job uses these resources:

- **Partition**: `gpu-h100` (H100 GPU)
- **GPU**: 1x H100
- **CPUs**: 8 cores
- **Memory**: 64GB
- **Time limit**: 7 days
- **Job name**: `mts-complete-pipeline`

## What Happens During Execution

### Phase 1: Data Preparation (30-60 min)
- Downloads CCMusic dataset or creates simulated data
- Processes audio features
- Creates augmented versions (3x augmentation)
- Saves to `data/augmented/audio/`
- **Smart skip**: If `outputs/mts_final_dataset.csv` exists, skips this phase

### Phase 2: Model Training (12-48 hours)
- Loads prepared dataset
- Trains MTS model for 200 epochs
- Saves checkpoints every 20 epochs
- Saves best model to `checkpoints/mts_best.pt`
- **Smart resume**: If checkpoint exists, resumes training

### Phase 3: Sample Generation (5-10 min)
- Loads best checkpoint
- Generates test audio samples
- Saves to `generated_samples/`
- Non-critical (job succeeds even if this fails)

## Monitoring Progress

### Check job status
```bash
squeue -u $USER
```

### View live output (data preparation + training progress)
```bash
tail -f /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-<JOB_ID>.out
```

### View errors (if any)
```bash
tail -f /data/gpfs/projects/punim2072/MTS/err/mts-complete-pipeline-<JOB_ID>.err
```

## Output Locations

After successful completion:

```
MTS-2/
├── data/augmented/audio/          # Augmented audio files (MP3/WAV)
├── outputs/
│   ├── mts_final_dataset.csv      # Complete dataset
│   └── ...                         # Other output files
├── checkpoints/
│   ├── mts_best.pt                # Best model checkpoint
│   ├── mts_epoch_20.pt            # Checkpoint at epoch 20
│   ├── mts_epoch_40.pt            # Checkpoint at epoch 40
│   └── ...                         # More checkpoints
└── generated_samples/
    ├── sample_0.wav               # Generated test sample
    └── ...                         # More samples
```

## Smart Features

### 1. Resume Training
If training is interrupted, just resubmit:
```bash
sbatch mts_pipeline.slurm
```
It will automatically resume from the last checkpoint.

### 2. Skip Data Preparation
If you already have `outputs/mts_final_dataset.csv`, Phase 1 is automatically skipped.

### 3. Graceful Failure Handling
- Phase 1 fails → Job stops, no training
- Phase 2 fails → Job stops, no generation
- Phase 3 fails → Job continues (non-critical)

## Customizing Training Parameters

Edit the script at line 402-406 to adjust training:

```bash
BATCH_SIZE=8      # Increase for faster training (needs more memory)
EPOCHS=200        # Reduce for faster experimentation
LR=1e-4          # Learning rate
SAVE_EVERY=20     # Save checkpoint frequency
```

## Checking Results

### Final summary in job output
```bash
cat /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-<JOB_ID>.out | tail -50
```

Look for:
```
====================================================================
                    COMPLETE PIPELINE SUMMARY
====================================================================
Phase Results:
  1. Data Preparation:  ✅ SUCCESS
  2. Model Training:    ✅ SUCCESS
  3. Sample Generation: ✅ SUCCESS

Overall Status: ✅ SUCCESS
```

### Download generated samples
```bash
# From your local machine
scp username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/generated_samples/*.wav ~/Downloads/
```

## Troubleshooting

### Job not starting
```bash
# Check queue position
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>
```

### Out of memory during training
Reduce batch size in the script:
```bash
BATCH_SIZE=4  # Line 403
```

### Training too slow
If you need results faster, reduce epochs:
```bash
EPOCHS=50  # Line 404
```

### Data preparation keeps running
If dataset already exists but it's re-running Phase 1, check:
```bash
ls -lh outputs/mts_final_dataset.csv
```
If missing, data prep will run again.

## Comparison with Old Workflow

### Old Way (Multiple Jobs)
```bash
# Job 1: Data preparation
sbatch slurm_data_prep_only.sh
# Wait for completion...

# Job 2: Training
sbatch slurm_train_mts.sh
# Wait for completion...

# Job 3: Generation (manual)
```

### New Way (Single Job)
```bash
sbatch mts_pipeline.slurm
# Everything runs automatically!
```

## Advanced: Running Only Specific Phases

If you want to run just one phase, use the old scripts:

- Data only: `sbatch slurm_data_prep_only.sh`
- Training only: `sbatch slurm_train_mts.sh`

But the unified script is recommended for full pipeline runs.

## Email Notifications

You'll receive emails at `GangyiZ@student.unimelb.edu.au` for:
- Job start
- Job completion
- Job failure

## Next Steps After Successful Run

1. **Check training log** to see loss curves
2. **Download and listen** to generated samples
3. **Run long-form generation** for full songs:
   ```bash
   python3 generate_long_form.py
   ```
4. **Experiment** with different prompts and durations

## Support

If you encounter issues:
1. Check the error log file
2. Review the job output file
3. Verify GPU availability: `nvidia-smi`
4. Check disk space: `df -h`

---

**Last Updated**: 2025-12-17
**Script Version**: Unified Complete Pipeline v2
