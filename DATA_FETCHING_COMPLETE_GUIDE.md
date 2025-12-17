# Complete Data Fetching Guide for MTS-2

## 🎯 Quick Summary

**Current Status**: Background download is **RUNNING** ✅
- Downloaded: 55.4 MB / 7.68 GB (~1%)
- Speed: ~800 KB/s
- ETA: 2-3 hours

**What You Have**:
1. ✅ Metadata (complete)
2. 🔄 Audio files (downloading)
3. ✅ All code ready to use data

---

## 📊 Your Codebase Overview

### Key Files for Data Fetching

1. **[fetch_fma_dataset.py](fetch_fma_dataset.py)** - Main download script
   - Downloads FMA dataset from https://os.unil.cloud.switch.ch/fma/
   - Handles metadata + audio
   - Usage: `python3 fetch_fma_dataset.py --size small`

2. **[src/data_loader.py](src/data_loader.py)** - Data loading
   - Line 211-288: `load_fma_dataset()` method
   - Loads MP3 files with librosa
   - Returns dataset with real audio arrays
   - Usage:
     ```python
     from src.data_loader import MTSDataLoader
     loader = MTSDataLoader(target_sr=24000)
     dataset = loader.load_fma_dataset(
         audio_dir='./fma_data/fma_small',
         metadata_path='./fma_data/fma_metadata/tracks.csv',
         max_files=50
     )
     ```

3. **[src/augmentation.py](src/augmentation.py)** - Audio augmentation
   - Line 635-639: Checks for `song["audio"]` key
   - If audio exists: uses real audio ✅
   - If missing: falls back to synthetic (you want to avoid this!)

4. **[config/config_local_mac.yaml](config/config_local_mac.yaml)** - Configuration
   - Line 13-18: FMA dataset settings
   - Already configured to use FMA data
   - Set to load 50 files for testing

### How Data Flows

```
fetch_fma_dataset.py
    ↓ (downloads)
fma_data/fma_small/*.mp3
    ↓ (loaded by)
src/data_loader.py::load_fma_dataset()
    ↓ (returns)
dataset['preprocessed_songs'] = [
    {
        "id": "fma_000002",
        "audio": numpy.array([...]),  # ✅ REAL audio!
        "has_audio": True,
        ...
    },
    ...
]
    ↓ (passed to)
src/augmentation.py::augment_dataset()
    ↓ (creates)
Augmented audio files in outputs/
```

---

## 🔄 Current Download Progress

### Check Download Status

```bash
# Monitor real-time progress
tail -f fma_manual_download.log

# Check background task
cat /tmp/claude/tasks/bec88bd.output

# Check if process is still running
ps aux | grep fetch_fma
```

### What's Being Downloaded

**FMA Small Dataset**:
- **Source**: https://os.unil.cloud.switch.ch/fma/fma_small.zip
- **Size**: 7.68 GB compressed
- **Contains**: 8,000 MP3 files
- **Format**: 320 kbps, 44.1 kHz, stereo
- **Duration**: 30 seconds each
- **Genres**: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock

**Expected Timeline**:
- At 800 KB/s: ~2.5-3 hours
- At 1 MB/s: ~2 hours
- At 2 MB/s: ~1 hour

---

## ✅ When Download Completes

### Step 1: Verify Download

```bash
cd /Users/zhanggangyi/Desktop/MTS-2

# Check file size (should be ~7.2 GB)
ls -lh fma_data/fma_small.zip

# Extract audio files
cd fma_data
unzip fma_small.zip

# Count files (should be 8000)
find fma_small -name "*.mp3" | wc -l

# Check first file
FIRST=$(find fma_small -name "*.mp3" | head -1)
echo "First file: $FIRST"
file "$FIRST"
```

Expected output:
```
-rw-r--r-- 1 user staff 7.2G Dec 14 17:00 fma_data/fma_small.zip
8000
First file: fma_data/fma_small/000/000002.mp3
fma_data/fma_small/000/000002.mp3: Audio file with ID3 version 2.3.0
```

### Step 2: Test Loading One File

```bash
python3 << 'EOF'
from src.data_loader import MTSDataLoader

# Initialize loader
loader = MTSDataLoader(target_sr=24000)

# Load just ONE file to test
dataset = loader.load_fma_dataset(
    audio_dir='./fma_data/fma_small',
    metadata_path='./fma_data/fma_metadata/tracks.csv',
    max_files=1
)

# Check the song
song = dataset['preprocessed_songs'][0]

print("=" * 60)
print("✅ FMA Dataset Test")
print("=" * 60)
print(f"Song ID:      {song['id']}")
print(f"Title:        {song['title']}")
print(f"Artist:       {song['artist']}")
print(f"Genre:        {song['genre']}")
print(f"Duration:     {song['duration']:.2f}s")
print(f"Sample rate:  {song['sample_rate']} Hz")
print(f"Has audio:    {song['has_audio']}")
if 'audio' in song and song['audio'] is not None:
    print(f"Audio shape:  {song['audio'].shape}")
    print(f"Audio dtype:  {song['audio'].dtype}")
    print("\n✅ SUCCESS! Real audio is loaded!")
else:
    print("\n❌ ERROR! No audio found")
print("=" * 60)
EOF
```

Expected output:
```
🎵 Loading FMA audio from ./fma_data/fma_small (1 files)...
Loading FMA audio: 100%|██████████| 1/1 [00:01<00:00,  1.23s/it]
✅ Loaded 1 FMA tracks with audio.
============================================================
✅ FMA Dataset Test
============================================================
Song ID:      fma_000002
Title:        Real Song Title
Artist:       Real Artist Name
Genre:        Electronic
Duration:     30.00s
Sample rate:  24000 Hz
Has audio:    True
Audio shape:  (720000,)
Audio dtype:  float32

✅ SUCCESS! Real audio is loaded!
============================================================
```

### Step 3: Test Augmentation

```bash
python3 << 'EOF'
from src.data_loader import MTSDataLoader
from src.augmentation import MTSAudioAugmentation
import soundfile as sf

# Load 5 songs
loader = MTSDataLoader(target_sr=24000)
dataset = loader.load_fma_dataset(
    audio_dir='./fma_data/fma_small',
    metadata_path='./fma_data/fma_metadata/tracks.csv',
    max_files=5
)

print(f"\n✅ Loaded {len(dataset['preprocessed_songs'])} songs")

# Create augmenter
augmenter = MTSAudioAugmentation(
    sample_rate=24000,
    augmentation_factor=2,
    preserve_original=True
)

# Generate augmented versions
print("\n🎵 Generating augmented audio...")
augmented = augmenter.augment_dataset(
    dataset['preprocessed_songs'],
    output_dir='./test_output/fma_augmented',
    save_audio=True,
    audio_format='wav'
)

print(f"\n✅ Generated {len(augmented)} augmented files!")
print(f"   Location: test_output/fma_augmented/")

# Show first few files
import os
files = sorted(os.listdir('test_output/fma_augmented'))[:5]
print("\nFirst 5 files:")
for f in files:
    path = f"test_output/fma_augmented/{f}"
    size = os.path.getsize(path) / (1024 * 1024)
    print(f"  {f} ({size:.2f} MB)")
EOF
```

### Step 4: Run Full Pipeline

```bash
# Process 50 songs with augmentation
python3 run_pipeline.py --config config/config_local_mac.yaml
```

This will:
1. Load 50 FMA audio files
2. Generate 2x augmented versions (100 total)
3. Create train/val/test splits
4. Save to `outputs/` directory

---

## 🛠️ Manual Data Fetching Options

### Option 1: Wait for Current Download ⭐ RECOMMENDED

The download is currently running. Just wait for it to complete (~2-3 hours).

```bash
# Monitor progress
tail -f fma_manual_download.log
```

### Option 2: Use curl (If Download Fails)

If the current download fails, resume with curl:

```bash
cd /Users/zhanggangyi/Desktop/MTS-2

# Remove incomplete file if needed
rm -f fma_data/fma_small.zip

# Download with resume support
curl -C - -o fma_data/fma_small.zip \
  https://os.unil.cloud.switch.ch/fma/fma_small.zip

# Verify size (should be ~7.2 GB)
ls -lh fma_data/fma_small.zip

# Extract
cd fma_data && unzip fma_small.zip
```

### Option 3: Use wget (If Installed)

```bash
cd /Users/zhanggangyi/Desktop/MTS-2

# Check if wget is installed
which wget || brew install wget

# Download with auto-resume
wget -c https://os.unil.cloud.switch.ch/fma/fma_small.zip \
  -O fma_data/fma_small.zip

# Extract
cd fma_data && unzip fma_small.zip
```

### Option 4: Browser Download

1. Open: https://os.unil.cloud.switch.ch/fma/fma_small.zip
2. Save to: `/Users/zhanggangyi/Desktop/MTS-2/fma_data/fma_small.zip`
3. Extract manually

### Option 5: Smaller Test Dataset

If 7.2 GB is too large, use your own audio files:

```bash
# Create custom audio directory
mkdir -p custom_audio

# Copy 10-50 of your MP3/WAV files to custom_audio/
cp /path/to/your/music/*.mp3 custom_audio/
```

Edit [config/config_local_mac.yaml](config/config_local_mac.yaml):
```yaml
data:
  use_fma_dataset: false
  custom_audio_dir: "./custom_audio"
```

---

## 🧪 Testing After Download

### Test 1: Single File Load

```bash
python3 test_with_slurm_audio.py $(find fma_data/fma_small -name "*.mp3" | head -1)
```

### Test 2: Quality Check

```bash
# Play a file (Mac)
afplay $(find fma_data/fma_small -name "*.mp3" | shuf | head -1)

# Check audio info
ffmpeg -i $(find fma_data/fma_small -name "*.mp3" | head -1) 2>&1 | grep "Audio:"
```

Expected output:
```
Audio: mp3, 44100 Hz, stereo, fltp, 320 kb/s
```

### Test 3: Load Multiple Files

```bash
python3 -c "
from src.data_loader import MTSDataLoader
loader = MTSDataLoader(target_sr=24000)
dataset = loader.load_fma_dataset(
    audio_dir='./fma_data/fma_small',
    metadata_path='./fma_data/fma_metadata/tracks.csv',
    max_files=10
)
print(f'✅ Loaded {len(dataset[\"preprocessed_songs\"])} songs')
for i, song in enumerate(dataset['preprocessed_songs'][:3], 1):
    print(f'{i}. {song[\"title\"]} by {song[\"artist\"]} ({song[\"genre\"]})')
"
```

---

## 📊 Understanding the Dataset

### FMA Metadata Structure

The metadata you already downloaded includes:

**tracks.csv** - Main metadata
```python
import pandas as pd
tracks = pd.read_csv('fma_data/fma_metadata/tracks.csv', index_col=0, header=[0,1])

# Access track info
track_id = 2
title = tracks.loc[track_id, ('track', 'title')]
artist = tracks.loc[track_id, ('artist', 'name')]
genre = tracks.loc[track_id, ('track', 'genre_top')]
```

**genres.csv** - Genre hierarchy
```
genre_id,title,parent,top_level
1,Electronic,,21
2,Experimental,,38
12,Hip-Hop,,21
...
```

### Audio File Organization

```
fma_small/
├── 000/
│   ├── 000002.mp3  ← Track ID 2
│   ├── 000005.mp3  ← Track ID 5
│   └── ...
├── 001/
│   ├── 000010.mp3  ← Track ID 10
│   └── ...
└── 155/
    └── 155999.mp3  ← Track ID 155999
```

### How load_fma_dataset() Works

```python
# From src/data_loader.py:259-273
def load_fma_dataset(self, audio_dir, metadata_path, max_files=500):
    # 1. Load metadata from tracks.csv
    tracks_df = pd.read_csv(metadata_path, index_col=0, header=[0, 1])

    # 2. Find all MP3 files
    files = glob.glob(f"{audio_dir}/**/*.mp3", recursive=True)

    # 3. Load each audio file
    for filepath in files[:max_files]:
        tid = int(Path(filepath).stem)  # Extract track ID from filename
        audio, sr = librosa.load(filepath, sr=self.target_sr, duration=30.0)

        # 4. Create song dict with REAL audio
        songs.append({
            "id": f"fma_{tid:06d}",
            "audio": audio,  # ✅ REAL AUDIO HERE!
            "has_audio": True,
            "title": tracks_df.loc[tid, ('track', 'title')],
            "genre": tracks_df.loc[tid, ('track', 'genre_top')],
            ...
        })

    return dataset
```

---

## 🔧 Troubleshooting

### Download is Slow

**Current speed**: ~800 KB/s
**Ways to speed up**:
1. Connect to faster network
2. Download during off-peak hours
3. Use wget/curl with parallel connections
4. Download on another machine, transfer via USB

### Download Fails

```bash
# Check if still running
ps aux | grep fetch_fma

# If stopped, restart
python3 fetch_fma_dataset.py --size small

# The script detects existing files and resumes
```

### Zip Extraction Fails

```bash
# Test zip integrity
unzip -t fma_data/fma_small.zip | tail -20

# If corrupted, re-download
rm fma_data/fma_small.zip
curl -C - -o fma_data/fma_small.zip \
  https://os.unil.cloud.switch.ch/fma/fma_small.zip
```

### Out of Disk Space

```bash
# Check available space
df -h /Users/zhanggangyi/Desktop

# Need: 15 GB free
# - 7.2 GB: zip file
# - 7.2 GB: extracted files
# - 1 GB: working space

# Free up space
rm -rf test_output/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

### Audio Files Don't Load

```bash
# Test file format
file fma_data/fma_small/000/000002.mp3

# Should output:
# Audio file with ID3 version 2.3.0

# Test with librosa
python3 -c "
import librosa
audio, sr = librosa.load('fma_data/fma_small/000/000002.mp3', sr=24000, duration=5.0)
print(f'Loaded: {len(audio)} samples, {sr} Hz')
"
```

---

## 📋 Quick Reference

### Check Download Progress
```bash
tail -f fma_manual_download.log
cat /tmp/claude/tasks/bec88bd.output | tail -50
ps aux | grep fetch_fma
```

### After Download Completes
```bash
# Extract
cd fma_data && unzip fma_small.zip

# Verify
find fma_small -name "*.mp3" | wc -l  # Should be 8000

# Test load
python3 -c "from src.data_loader import MTSDataLoader; loader = MTSDataLoader(target_sr=24000); ds = loader.load_fma_dataset('./fma_data/fma_small', './fma_data/fma_metadata/tracks.csv', max_files=1); print('✅ Success!' if ds['preprocessed_songs'][0]['has_audio'] else '❌ Failed')"

# Run pipeline
python3 run_pipeline.py --config config/config_local_mac.yaml
```

### File Paths
- Download script: [fetch_fma_dataset.py](fetch_fma_dataset.py)
- Data loader: [src/data_loader.py](src/data_loader.py#L211-L288)
- Augmentation: [src/augmentation.py](src/augmentation.py#L635-L639)
- Config: [config/config_local_mac.yaml](config/config_local_mac.yaml#L13-L18)
- Download log: `fma_manual_download.log`
- Background task: `/tmp/claude/tasks/bec88bd.output`

---

## 🎯 What Happens Next

### Immediate (Next 2-3 hours)
- Download completes ✅
- You'll have 8,000 real music files
- Total size: ~7-8 GB

### After Download
1. Extract the zip file
2. Test loading one file
3. Run pipeline with 50 files
4. Generate augmented audio
5. Train your model with REAL music!

### Why This Matters

Your code checks for `song["audio"]`:
- ✅ **With FMA data**: Uses real audio from MP3 files
- ❌ **Without FMA data**: Falls back to synthetic sine waves

The download ensures you get **real music** instead of **synthetic beeps**!

---

## 📞 Support

If you have issues:
1. Check [MANUAL_DATA_FETCH_GUIDE.md](MANUAL_DATA_FETCH_GUIDE.md)
2. Check [WHY_NO_REAL_AUDIO.md](WHY_NO_REAL_AUDIO.md)
3. Monitor: `tail -f fma_manual_download.log`

---

## ✅ Summary

**Current Status**: Download running (55.4 MB / 7.68 GB)
**ETA**: 2-3 hours
**Next Step**: Wait for download, then test with real audio!

**Your codebase is ready!** All the infrastructure is in place to load and process FMA data. Just need to wait for the download to finish.

---

**Last Updated**: 2025-12-14
**Download Progress**: 1% (55.4 MB / 7.68 GB)
**Estimated Completion**: ~2-3 hours from now
