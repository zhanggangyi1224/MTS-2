# Quick Reference Card

## ⚡ Fastest Way to Update HPC

```bash
ssh gangyiz@spartan.hpc.unimelb.edu.au
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2
scancel $(squeue -u gangyiz -h -o "%i")  # Cancel all your jobs
git pull origin main
rm outputs/mts_final_dataset.csv
sbatch mts_pipeline.slurm
```

## 📋 What Was Fixed

| Issue | Fix | File |
|-------|-----|------|
| Empty audio paths | Added audio_path column | `src/batch_processor.py` |
| TypeError on NaN | Handle NaN values | `train_mts_hpc.py`, `verify_dataset.py` |
| CCMusic → FMA | Delete old CSV, regenerate with FMA | Config already set |
| Packages in ~home | Install to project directory | `mts_pipeline.slurm` |
| No FMA verification | Auto-download in Phase 0 | `mts_pipeline.slurm` |

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| **[HPC_PULL_GUIDE.md](HPC_PULL_GUIDE.md)** | **START HERE** - How to pull updates on HPC |
| [README_FIXES.md](README_FIXES.md) | Master summary of all fixes |
| [URGENT_FIX_STEPS.md](URGENT_FIX_STEPS.md) | Alternative: Using scp instead of git |
| [FIX_CCMUSIC_TO_FMA.md](FIX_CCMUSIC_TO_FMA.md) | Why CCMusic → FMA switch |
| [PACKAGE_INSTALLATION_CHANGES.md](PACKAGE_INSTALLATION_CHANGES.md) | Package path details |

## ✅ Verification Commands

```bash
# Check git updated
git log --oneline -1
# Should show: c05c61e Add HPC pull guide

# Check SLURM has updates
grep "PIP_PREFIX" mts_pipeline.slurm
# Should show: export PIP_PREFIX=$PROJECT_DIR/packages

# Check CSV deleted
ls outputs/mts_final_dataset.csv
# Should show: No such file or directory

# Check config
grep "use_fma_dataset" config/config_hpc.yaml
# Should show: use_fma_dataset: true
```

## 🎯 Expected Results

**Phase 0 (1 min):** ✅ FMA data verified  
**Phase 1 (10-20 min):** ✅ 50 FMA songs → 150 samples with valid audio paths  
**Phase 2 (4-8 hrs):** ✅ 100% verification rate → Training succeeds  
**Phase 3 (5-10 min):** ✅ Generates test samples

## 🔍 Monitor Job

```bash
# Watch output
tail -f /data/gpfs/projects/punim2072/MTS/out/mts-complete-pipeline-*.out

# Check errors
tail -f /data/gpfs/projects/punim2072/MTS/err/mts-complete-pipeline-*.err

# Check job status
squeue -u gangyiz
```

## 🚨 If Something Goes Wrong

| Problem | Solution |
|---------|----------|
| Git merge conflict | `git reset --hard origin/main` |
| Still sees CCMusic | Delete CSV: `rm outputs/mts_final_dataset.csv` |
| TypeError still occurs | Check git updated: `git log --oneline -1` |
| Packages still in ~home | Re-pull: `git pull origin main` |
| FMA data not found | Phase 0 auto-downloads (wait 20-30 min) |

---

**Need more details?** → [HPC_PULL_GUIDE.md](HPC_PULL_GUIDE.md)
