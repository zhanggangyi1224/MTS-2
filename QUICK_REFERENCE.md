# MTS-2 Quick Reference Card

## Single Command to Run Everything

```bash
sbatch mts_pipeline.slurm
```

That's it! This runs:
1. Data preparation (download + augmentation)
2. Model training (200 epochs on GPU)
3. Sample generation (test outputs)

---

## Essential Commands

### Submit Job
```bash
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2
sbatch mts_pipeline.slurm
```

### Check Status
```bash
squeue -u $USER
```

### Watch Progress (live)
```bash
tail -f /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-*.out
```

### View Final Results
```bash
# Last 50 lines show summary
cat /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-<JOB_ID>.out | tail -50
```

### Cancel Job
```bash
scancel <JOB_ID>
```

---

## Output Locations

```
data/augmented/audio/     → Augmented audio files
outputs/                  → Dataset CSV and statistics
checkpoints/              → Trained model files
generated_samples/        → Generated test audio
```

---

## Key Files

- **mts_pipeline.slurm** - Main unified script (submit this)
- **SLURM_SUBMISSION_GUIDE.md** - Detailed documentation
- **PIPELINE_UPDATE_SUMMARY.md** - What changed

---

## Customization

Edit `mts_pipeline.slurm` lines 403-406:

```bash
BATCH_SIZE=8      # Training batch size
EPOCHS=200        # Number of epochs (reduce for testing)
LR=1e-4          # Learning rate
SAVE_EVERY=20     # Checkpoint frequency
```

---

## Smart Features

✅ **Auto-skip data prep** if CSV exists
✅ **Auto-resume training** from checkpoint
✅ **Graceful failure** handling per phase
✅ **Email notifications** on start/end/fail

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Job pending too long | Check queue: `squeue -p gpu-h100` |
| Out of memory | Reduce `BATCH_SIZE=4` (line 403) |
| Need faster test | Set `EPOCHS=5` (line 404) |
| Data prep re-running | Check `outputs/mts_final_dataset.csv` exists |

---

## Download Results

```bash
# From your local machine
scp username@spartan.hpc.unimelb.edu.au:/path/to/generated_samples/*.wav ~/Downloads/
```

---

## Expected Timeline

- **Data Prep**: 30-60 minutes
- **Training**: 12-48 hours (depends on data size)
- **Generation**: 5-10 minutes
- **Total**: ~1-2 days for full run

---

## Status Indicators

```
✅ SUCCESS  - Phase completed
❌ FAILED   - Phase failed, job stopped
⚠️  SKIPPED - Phase skipped (non-critical)
```

---

## Need Help?

1. Check error log: `/data/gpfs/projects/punim2072/MTS/err/mts-complete-pipeline-<JOB_ID>.err`
2. Review output log for details
3. See [SLURM_SUBMISSION_GUIDE.md](SLURM_SUBMISSION_GUIDE.md) for full documentation

---

**Quick Start**: `sbatch mts_pipeline.slurm` 🚀
