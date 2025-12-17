# 🎵 FMA Dataset Download - IN PROGRESS

**Started**: Just now
**Status**: 🔄 Downloading
**Size**: 8 GB (8,000 audio tracks)
**ETA**: 30-60 minutes

---

## ✅ What's Downloaded

The script is fetching the FMA (Free Music Archive) dataset:

### Step 1: Metadata (342 MB) - Starting Now
- Track information (titles, artists, genres)
- File: `fma_metadata.zip`
- Extracts to: `fma_data/fma_metadata/`

### Step 2: Audio Files (7.2 GB) - Next
- 8,000 music tracks (30-second clips)
- File: `fma_small.zip`
- Extracts to: `fma_data/fma_small/`

---

## 📊 Monitor Progress

```bash
# Watch the download log
tail -f fma_download.log

# Or check the progress file
cat /tmp/claude/tasks/b073085.output

# Check if files are arriving
ls -lh fma_data/
```

---

## 📁 Expected Output Structure

```
MTS-2/
└── fma_data/
    ├── fma_metadata/           # ← Downloading now
    │   ├── tracks.csv          # Track metadata
    │   ├── genres.csv          # Genre information
    │   ├── features.csv        # Audio features
    │   └── echonest.csv        # Echo Nest features
    │
    └── fma_small/              # ← Downloads after metadata
        ├── 000/                # Organized by first 3 digits of track ID
        │   ├── 000002.mp3
        │   ├── 000005.mp3
        │   ├── 000010.mp3
        │   └── ...
        ├── 001/
        ├── 002/
        └── ... (155 folders, 8,000 files total)
```

---

## ⏱️ Timeline

| Step | Size | ETA |
|------|------|-----|
| 1. Metadata download | 342 MB | 2-5 min |
| 2. Metadata extraction | - | 1 min |
| 3. Audio download | 7.2 GB | 25-50 min |
| 4. Audio extraction | - | 3-5 min |
| **Total** | **~8 GB** | **30-60 min** |

---

## 🎯 What You'll Get

### Real Music Audio
- **8,000 tracks** from 8 different genres
- **30-second clips** (perfect for training)
- **MP3 format**, 320 kbps
- **High quality** recordings

### Genres
- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Pop
- Rock

### Metadata
- Track ID, title, artist, album
- Genre tags (top and full genre tree)
- Audio features (tempo, key, etc.)
- Licensing information

---

## ✅ When Download Completes

You'll see this message:
```
✅ Download Complete!
Audio files: 8000
Location: /Users/zhanggangyi/Desktop/MTS-2/fma_data/fma_small
Total size: 7.20 GB
```

### Then You Can:

**1. Test One Audio File**
```bash
# Find first audio file
FIRST_FILE=$(find fma_data/fma_small -name "*.mp3" | head -1)

# Test it
python3 test_with_slurm_audio.py "$FIRST_FILE"
```

**2. Run Quick Test (10 files)**
```bash
python3 run_local_test.py --all
```

**3. Generate Augmented Dataset**
```bash
# Update config to use FMA
# Then run:
python3 run_pipeline.py --config config/config_local_mac.yaml
```

---

## 🎧 Quality Check

Once download completes, verify quality:

```bash
# Pick a random file
RANDOM_FILE=$(find fma_data/fma_small -name "*.mp3" | shuf | head -1)

# Listen to it
afplay "$RANDOM_FILE"

# Get info
ffmpeg -i "$RANDOM_FILE" 2>&1 | grep "Audio:"
```

Expected output:
```
Audio: mp3, 44100 Hz, stereo, fltp, 320 kb/s
```

---

## 💾 Disk Usage

**Current**:
- Available: 1804 GB ✅ (plenty of space)
- Needed: 10 GB (8 GB + working space)

**After download**:
- FMA dataset: ~8 GB
- Test outputs: ~0.1 GB
- Augmented data (if generated): ~24 GB (8GB × 3x factor)

---

## 🔧 If Download Fails

The download is running in background. If it fails:

**Check what happened**:
```bash
cat fma_download.log
tail -50 /tmp/claude/tasks/b073085.output
```

**Resume/Restart**:
```bash
python3 fetch_fma_dataset.py --size small
# Script will detect existing files and skip re-downloading
```

**Manual download** (if automatic fails):
```bash
# Download files directly
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip

# Extract
mkdir -p fma_data
unzip fma_metadata.zip -d fma_data/
unzip fma_small.zip -d fma_data/
```

---

## 📝 Background Download

The download is running in the background. You can:

✅ Close this terminal (download continues)
✅ Work on other things
✅ Check progress anytime: `tail -f fma_download.log`

❌ Don't shut down your Mac
❌ Don't put Mac to sleep (may pause download)

---

## 🎯 Summary

**What's happening**: Downloading FMA dataset (8GB, 8000 tracks)
**Where**: `fma_data/` directory
**How long**: 30-60 minutes
**Log file**: `fma_download.log`

**Next**: Wait for completion, then test audio quality!

---

**Check back in 30 minutes!** ⏳

The download is running automatically. Once complete, you'll have 8,000 real music tracks to work with for your MTS-2 project!
