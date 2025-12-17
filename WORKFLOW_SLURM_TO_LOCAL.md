# Workflow: SLURM Dataset → Local Mac Testing

**Best Practice**: Prepare full dataset on SLURM, then test quality locally on Mac

---

## 📋 Complete Workflow

### Step 1: Prepare Dataset on SLURM

Run the data preparation script on SLURM to download and augment the full dataset:

```bash
# On your local Mac, submit the job
ssh your-username@spartan.hpc.unimelb.edu.au
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2

# Submit data preparation job
sbatch slurm_data_prep_only.sh
```

**What it does**:
- Downloads CCMusic dataset (or uses simulated data)
- Processes all songs at 24000 Hz (EnCodec-compatible)
- Creates 3x augmented versions
- Saves all audio as MP3 files
- Takes ~2-6 hours depending on dataset size

**Output location on SLURM**:
```
/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/data/augmented/audio/
```

---

### Step 2: Check Job Status

```bash
# Check if job is running
squeue -u your-username

# View output log
tail -f /data/gpfs/projects/punim2072/MTS/out/mts-data-prep-only-JOBID.out

# Check for completion
ls -lh /data/gpfs/projects/punim2072/MTS/MTS/MTS-2/data/augmented/audio/ | head
```

When complete, you'll see output like:
```
✅ SUCCESS
🎵 Total audio files: 5100
💾 Total size: 2.3G
```

---

### Step 3: Copy ONE Sample to Mac

Once the job completes, copy a single audio file to test locally:

```bash
# On your Mac, create directory
mkdir -p ~/Desktop/MTS-2/slurm_data

# Copy one audio file from SLURM
scp your-username@spartan.hpc.unimelb.edu.au:'/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/data/augmented/audio/ccmusic_0001_aug_01.mp3' ~/Desktop/MTS-2/slurm_data/

# Or copy a few samples
scp 'your-username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2072/MTS/MTS/MTS-2/data/augmented/audio/ccmusic_*.mp3' ~/Desktop/MTS-2/slurm_data/
```

**Tips**:
- Copy just 1-3 files for quick testing
- Choose both original and augmented versions
- Total size: ~5-10 MB per file

---

### Step 4: Test Audio Quality on Mac

Run the test script on your local Mac:

```bash
cd ~/Desktop/MTS-2

# Automatic (finds audio in slurm_data/)
python3 test_with_slurm_audio.py

# Or specify file explicitly
python3 test_with_slurm_audio.py slurm_data/ccmusic_0001_aug_01.mp3
```

**What it does**:
- Loads the SLURM-generated audio
- Analyzes quality metrics (clipping, SNR, dynamic range)
- Tests augmentation quality
- Saves test results to `test_output/`

**Expected output**:
```
🎵 Testing SLURM-Generated Audio
✅ No clipping. Peak: 0.854
✅ Good signal level. RMS: 0.123
   Dynamic range: 18.79 dB
   Augmentation SNR: 22.45 dB (Excellent)

✅ QUALITY: GOOD - Audio from SLURM dataset is high quality!
```

---

### Step 5: Verify Quality

Listen to the generated files:

```bash
cd ~/Desktop/MTS-2/test_output

# Play original
afplay slurm_original_ccmusic_0001_aug_01.wav

# Play augmented version
afplay slurm_augmented_ccmusic_0001_aug_01.wav
```

**Quality checklist**:
- [ ] Audio is audible and clear
- [ ] No clipping or distortion
- [ ] Augmented version sounds natural
- [ ] SNR > 15 dB (Good) or > 25 dB (Excellent)
- [ ] No artifacts or weird sounds

---

### Step 6: Decision

Based on the quality test:

**If quality is GOOD** ✅:
- Continue with full SLURM pipeline
- Use the full dataset for training
- You're ready for model training phase

**If quality needs improvement** ⚠️:
- Adjust augmentation settings in `slurm_data_prep_only.sh`
- Install pyrubberband on SLURM for better quality
- Re-run data preparation
- Test again

---

## 🎯 Quick Reference

### Files Created

| File | Purpose |
|------|---------|
| `slurm_data_prep_only.sh` | SLURM script to prepare dataset |
| `test_with_slurm_audio.py` | Local Mac test script |
| `config/data_prep_only.yaml` | Config for data prep (auto-generated) |

### Directory Structure

```
MTS-2/
├── slurm_data/                    # Audio copied from SLURM
│   └── ccmusic_0001_aug_01.mp3
├── test_output/                   # Test results
│   ├── slurm_original_*.wav
│   └── slurm_augmented_*.wav
└── data/                          # SLURM output (on HPC)
    └── augmented/
        └── audio/                 # Full dataset here
            ├── ccmusic_0001.mp3
            ├── ccmusic_0001_aug_01.mp3
            ├── ccmusic_0001_aug_02.mp3
            └── ... (thousands more)
```

---

## 🔧 Troubleshooting

### Issue: No audio files generated on SLURM

**Check**:
1. Look at the SLURM output log
2. Verify `save_audio: true` in config
3. Check disk space: `df -h`
4. Check permissions: `ls -la data/augmented/audio/`

**Fix**:
```bash
# Make sure directory exists and is writable
mkdir -p data/augmented/audio
chmod 755 data/augmented/audio
```

---

### Issue: Audio quality is poor

**Possible causes**:
1. Using librosa fallback (not pyrubberband)
2. Aggressive augmentation settings
3. Dataset issues

**Fix**:
```bash
# Install pyrubberband on SLURM
module load GCC/11.3.0
pip install --user pyrubberband

# Or reduce augmentation strength in config:
# range_semitones: [-2, 2] → [-1, 1]
# range_factor: [0.9, 1.1] → [0.95, 1.05]
```

---

### Issue: File too large to copy

**Solution**: Compress before copying
```bash
# On SLURM
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2/data/augmented/audio
tar -czf sample_audio.tar.gz ccmusic_000*.mp3

# Copy to Mac
scp user@spartan:/.../sample_audio.tar.gz ~/Desktop/MTS-2/slurm_data/

# Extract
cd ~/Desktop/MTS-2/slurm_data
tar -xzf sample_audio.tar.gz
```

---

## 💡 Advantages of This Workflow

1. **Use SLURM Resources**: Download large dataset on HPC, not on Mac
2. **Save Bandwidth**: Only copy 1 file to test, not entire dataset
3. **Test Quickly**: Verify quality on Mac in seconds
4. **Iterate Fast**: Adjust settings and re-test easily
5. **Real Data**: Test with actual dataset, not synthetic

---

## 📊 Expected Timeline

| Step | Duration | Notes |
|------|----------|-------|
| 1. Submit SLURM job | 5 min | Just submit |
| 2. Wait for completion | 2-6 hrs | Depends on dataset |
| 3. Copy to Mac | 1-2 min | One file only |
| 4. Test on Mac | 30 sec | Fast local test |
| 5. Verify quality | 2 min | Listen to audio |

**Total active time**: ~10 minutes
**Total wait time**: 2-6 hours (SLURM processing)

---

## ✅ Success Criteria

You'll know it's working when:

1. ✅ SLURM job completes successfully
2. ✅ Audio files exist on SLURM: `ls data/augmented/audio/`
3. ✅ Sample file copied to Mac
4. ✅ Test script runs without errors
5. ✅ Quality metrics show: SNR > 15 dB, no clipping
6. ✅ Audio sounds good when played

---

## 🚀 Next Steps After Verification

Once you confirm quality is good:

1. **Use full dataset**: All audio files on SLURM are ready
2. **Run training**: Submit training job on SLURM
3. **Local testing**: Continue using Mac for quick experiments
4. **Production**: Full pipeline on SLURM cluster

---

**Ready to start?**

```bash
# 1. Submit SLURM job
ssh spartan
cd /data/gpfs/projects/punim2072/MTS/MTS/MTS-2
sbatch slurm_data_prep_only.sh

# 2. Wait...

# 3. Copy one file to Mac
scp user@spartan:/.../augmented/audio/ccmusic_0001_aug_01.mp3 ./slurm_data/

# 4. Test on Mac
python3 test_with_slurm_audio.py
```

Good luck! 🎵
