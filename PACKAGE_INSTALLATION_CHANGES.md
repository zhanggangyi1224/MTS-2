# Package Installation Path Changes

## Summary

Modified the SLURM pipeline to install all Python packages to the **project directory** instead of the home directory (`~/.local`).

## Changes Made

### 1. Environment Variables (Lines 57-65)

**Added:**
- `PIP_PREFIX=$PROJECT_DIR/packages` - Directory for pip packages
- Updated `PYTHONPATH` to include `$PIP_PREFIX/lib/python3.11/site-packages`

```bash
export PROJECT_DIR=${SLURM_SUBMIT_DIR:-/data/gpfs/projects/punim2072/MTS/MTS/MTS-2}
export CACHE_DIR=$PROJECT_DIR/cache
export DATA_DIR=$PROJECT_DIR/data
export OUTPUT_DIR=$PROJECT_DIR/outputs
export PIP_PREFIX=$PROJECT_DIR/packages  # NEW
export PYTHONPATH=$PIP_PREFIX/lib/python3.11/site-packages:$PROJECT_DIR:$PROJECT_DIR/src:$PYTHONPATH  # UPDATED
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
```

### 2. Directory Creation (Lines 71-77)

**Added:**
- Create `$PROJECT_DIR/packages` directory
- Create `$PIP_PREFIX/lib/python3.11/site-packages` directory

```bash
mkdir -p $PROJECT_DIR/{data,outputs,logs,checkpoints,packages}  # Added 'packages'
mkdir -p $PIP_PREFIX/lib/python3.11/site-packages  # NEW
```

### 3. Pip Installation Commands

**Changed ALL `pip install` commands from:**
```bash
pip install --user <package>
```

**To:**
```bash
pip install --prefix $PIP_PREFIX <package>
```

**Modified commands:**
- Line 112: `pip install --upgrade pip setuptools wheel` (unchanged - upgrades pip itself)
- Line 116: PyTorch installation - changed `--user` to `--prefix $PIP_PREFIX`
- Line 120: Base requirements - added `--prefix $PIP_PREFIX`
- Line 125: Complete requirements - added `--prefix $PIP_PREFIX`
- Line 130: einops - changed `--user` to `--prefix $PIP_PREFIX`
- Line 131: encodec - changed `--user` to `--prefix $PIP_PREFIX`
- Line 132: pydub - changed `--user` to `--prefix $PIP_PREFIX`
- Line 135: psutil - changed `--user` to `--prefix $PIP_PREFIX`
- Line 140: transformers/sentence-transformers - added `--prefix $PIP_PREFIX`
- Lines 144-147: Optional packages (Cython, madmom, pyrubberband, pedalboard) - changed `--user` to `--prefix $PIP_PREFIX`

## Benefits

### Before (Home Directory Installation)
```
❌ Packages installed to: ~/.local/lib/python3.11/site-packages
❌ May conflict with other user projects
❌ Home directory quota limits
❌ Not portable - tied to specific user account
```

### After (Project Directory Installation)
```
✅ Packages installed to: /data/gpfs/projects/punim2072/MTS/MTS/MTS-2/packages
✅ Isolated per-project environment
✅ Uses project storage quota (not home quota)
✅ Portable - can be shared/transferred with project
✅ Easier to clean up when project is complete
```

## Installation Paths

All packages will now be installed to:
```
$PROJECT_DIR/packages/
├── bin/                          # Executable scripts
├── lib/
│   └── python3.11/
│       └── site-packages/        # Python packages
│           ├── torch/
│           ├── transformers/
│           ├── einops/
│           ├── encodec/
│           └── ... (all other packages)
└── ...
```

## PYTHONPATH Priority

The new `PYTHONPATH` ensures packages are found in this order:
1. **Project packages**: `$PIP_PREFIX/lib/python3.11/site-packages` (HIGHEST PRIORITY)
2. **Project root**: `$PROJECT_DIR`
3. **Project source**: `$PROJECT_DIR/src`
4. **System packages**: Original `$PYTHONPATH`

## Important Notes

### Python Version
The configuration uses **Python 3.11** in the path:
```bash
$PIP_PREFIX/lib/python3.11/site-packages
```

If your HPC uses a different Python version (e.g., 3.9, 3.10, 3.12), you may need to:
1. Check Python version on HPC: `python --version`
2. Update the path accordingly in the SLURM script

### Verification
When the job runs, you'll see:
```
[TIMESTAMP] Environment setup complete
  PROJECT_DIR: /data/gpfs/projects/punim2072/MTS/MTS/MTS-2
  PIP_PREFIX: /data/gpfs/projects/punim2072/MTS/MTS/MTS-2/packages
  PYTHONPATH: /data/gpfs/projects/punim2072/MTS/MTS/MTS-2/packages/lib/python3.11/site-packages:...
```

### Storage Requirements
Packages will require approximately **5-8 GB** of storage in the project directory:
- PyTorch + torchaudio: ~3-4 GB
- Transformers + sentence-transformers: ~1-2 GB
- Other dependencies: ~1-2 GB

Ensure your project directory has sufficient quota.

### Cleanup
To remove all installed packages:
```bash
rm -rf $PROJECT_DIR/packages
```

This is safe - you can always reinstall packages by running the SLURM job again.

## Testing

After transferring the updated SLURM script to HPC, the first job submission will:
1. Create the `packages/` directory
2. Install all packages to `$PROJECT_DIR/packages/`
3. Verify installations work correctly
4. Proceed with the pipeline

## What to Transfer to HPC

Updated file:
```bash
mts_pipeline.slurm
```

Transfer command:
```bash
cd /Users/zhanggangyi/Desktop/MTS-2
scp mts_pipeline.slurm username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/
```

## Troubleshooting

### If packages aren't found after installation:

1. **Check Python version**:
   ```bash
   python --version
   # If it shows Python 3.X where X != 11, update the path in SLURM script
   ```

2. **Verify PYTHONPATH**:
   ```bash
   echo $PYTHONPATH
   # Should include: .../packages/lib/python3.11/site-packages
   ```

3. **Check package directory**:
   ```bash
   ls -lh $PROJECT_DIR/packages/lib/python3.*/site-packages/
   # Should show installed packages
   ```

4. **Test import**:
   ```bash
   python -c "import torch; print(torch.__file__)"
   # Should show path in project directory
   ```

## Summary

✅ **All pip installations now use project directory**
✅ **No more ~/.local installations**
✅ **Isolated project environment**
✅ **Ready for HPC deployment**

---

**Next Step**: Transfer the updated `mts_pipeline.slurm` to HPC and submit the job with `sbatch mts_pipeline.slurm`
