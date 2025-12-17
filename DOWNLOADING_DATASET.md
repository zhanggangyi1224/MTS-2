# FMA Dataset Download in Progress

**Status**: 🔄 Downloading FMA Small Dataset (8GB)

---

## 📥 What's Happening

The FMA (Free Music Archive) dataset is being downloaded automatically:

1. **Metadata** (342 MB) - Track info, genres, artists
2. **Audio Files** (7.2 GB) - 8,000 music tracks, 30-second clips

**Total download**: ~8 GB
**Estimated time**: 30-60 minutes (depending on internet speed)

---

## 📊 Progress

The download is running in the background. To check progress:

```bash
# Check if still downloading
ps aux | grep fetch_fma

# View download progress (if running)
tail -f /tmp/claude/tasks/bf9354d.output

# Or check manually
python3 fetch_fma_dataset.py --size small
```

---

## 📁 Output Location

```
MTS-2/
└── fma_data/
    ├── fma_metadata/          # Track metadata (CSV files)
    │   ├── tracks.csv
    │   ├── genres.csv
    │   ├── artists.csv
    │   └── ...
    └── fma_small/             # Audio files
        ├── 000/
        │   ├── 000002.mp3
        │   ├── 000005.mp3
        │   └── ...
        ├── 001/
        └── ... (8,000 total files)
```

---

## ✅ When Complete

You'll see:
```
✅ Download Complete!
Audio files: 8000
Location: /Users/zhanggangyi/Desktop/MTS-2/fma_data/fma_small
Total size: 7.20 GB
```

---

## 🎵 Next Steps After Download

### 1. Test with One Audio File
```bash
# Find a sample file
ls fma_data/fma_small/000/*.mp3 | head -1

# Test it
python3 test_with_slurm_audio.py fma_data/fma_small/000/000002.mp3
```

### 2. Run Full Pipeline
```bash
# Uses FMA dataset automatically
python3 run_pipeline.py --config config/config_local_mac.yaml
```

### 3. Generate Augmented Audio
```bash
# Create augmented versions
python3 -c "
from src.data_loader import MTSDataLoader
from src.augmentation import MTSAudioAugmentation

loader = MTSDataLoader(target_sr=24000)
dataset = loader.load_fma_dataset(
    audio_dir='./fma_data/fma_small',
    metadata_path='./fma_data/fma_metadata/tracks.csv',
    max_files=50  # Start with 50 files
)

augmenter = MTSAudioAugmentation(sample_rate=24000)
augmented = augmenter.augment_dataset(
    dataset['preprocessed_songs'],
    output_dir='./data/augmented',
    save_audio=True,
    audio_format='wav'
)

print(f'Generated {len(augmented)} augmented audio files!')
"
```

---

## 🔧 Troubleshooting

### Download Fails or Times Out

**Option 1**: Resume download
```bash
python3 fetch_fma_dataset.py --size small
# It will detect existing files and resume
```

**Option 2**: Manual download
```bash
# Download with wget (if installed)
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip

# Extract
unzip fma_metadata.zip -d fma_data/
unzip fma_small.zip -d fma_data/
```

**Option 3**: Use browser
- Visit: https://github.com/mdeff/fma
- Click "Get the data" section
- Download fma_small.zip and fma_metadata.zip
- Extract to `fma_data/` folder

---

### Not Enough Disk Space

**Check space**:
```bash
df -h .
```

**Need**: ~10 GB free (8 GB data + 2 GB working space)

**If low on space**:
- Delete test_output/ folder: `rm -rf test_output/`
- Clean cache: `rm -rf cache/`
- Or use smaller subset in config

---

### Download Too Slow

**Alternatives**:
1. Use `fma_metadata` only (342 MB) + simulated audio
2. Download on faster connection, transfer to Mac
3. Download on SLURM cluster (faster connection)

---

## 📋 Dataset Info

**FMA Small**:
- 8,000 tracks
- 30-second clips
- 8 genres
- 16 GB total (8 GB per track at 320kbps)
- Compressed to 7.2 GB

**Genres included**:
- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Pop
- Rock

**Perfect for**: Testing, development, training small models

---

## ⏱️ Estimated Time

| Connection | Time |
|------------|------|
| Fast (50 Mbps) | 20-30 min |
| Medium (25 Mbps) | 40-60 min |
| Slow (10 Mbps) | 90-120 min |

Current download started at: Check the output file for timestamp

---

## 💡 While Waiting

You can continue testing with the synthetic audio already generated:

```bash
# Listen to test files
cd test_output
afplay original.wav
afplay augmented_combined.wav

# Or generate more synthetic tests
python3 test_single_audio.py
```

---

**The download is running! Check back in 30-60 minutes.** ⏳

You'll have 8,000 real music tracks to work with for high-quality audio testing and training!
