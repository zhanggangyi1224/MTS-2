# Manual Data Fetching Guide for MTS-2

## Current Status
- ✅ Metadata: Downloaded (1.4 GB)
- ⚠️ Audio: Incomplete (1.3 GB / 7.2 GB)
- 📥 Background download: Running now to complete the audio files

---

## Option 1: Wait for Automatic Download (RECOMMENDED)

A background download is currently running. To check progress:

```bash
# Check if download is still running
ps aux | grep fetch_fma

# View download progress
tail -f fma_manual_download.log

# Or check the background task
cat /tmp/claude/tasks/bec88bd.output
```

**Wait time**: 30-60 minutes depending on your internet speed

---

## Option 2: Use `curl` to Download

If the automatic download fails, use `curl`:

```bash
cd /Users/zhanggangyi/Desktop/MTS-2

# Remove incomplete file
rm -f fma_data/fma_small.zip

# Download with curl (resumable)
curl -C - -o fma_data/fma_small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip

# Verify download size (should be ~7.2 GB)
ls -lh fma_data/fma_small.zip

# Extract
cd fma_data
unzip fma_small.zip

# Verify extraction
find fma_small -name "*.mp3" | wc -l
# Should output: 8000
```

---

## Option 3: Use `wget` (If Installed)

```bash
cd /Users/zhanggangyi/Desktop/MTS-2

# Remove incomplete file
rm -f fma_data/fma_small.zip

# Download with wget (auto-resume)
wget -c https://os.unil.cloud.switch.ch/fma/fma_small.zip -O fma_data/fma_small.zip

# Extract
cd fma_data
unzip fma_small.zip

# Verify
find fma_small -name "*.mp3" | wc -l
# Should output: 8000
```

---

## Option 4: Browser Download

1. **Open in browser**: https://os.unil.cloud.switch.ch/fma/fma_small.zip
2. **Save to**: `/Users/zhanggangyi/Desktop/MTS-2/fma_data/fma_small.zip`
3. **Extract manually**:
   ```bash
   cd /Users/zhanggangyi/Desktop/MTS-2/fma_data
   unzip fma_small.zip
   ```

---

## Option 5: Use Python Script Directly

```bash
cd /Users/zhanggangyi/Desktop/MTS-2

# Remove incomplete zip
rm -f fma_data/fma_small.zip

# Run the fetch script
python3 fetch_fma_dataset.py --size small

# It will:
# 1. Skip metadata (already downloaded)
# 2. Download fma_small.zip (7.2 GB)
# 3. Extract 8,000 MP3 files
# 4. Verify the files
```

---

## Verify Download Success

After download completes, verify:

```bash
cd /Users/zhanggangyi/Desktop/MTS-2

# Check metadata
ls -lh fma_data/fma_metadata/tracks.csv
# Should show ~260 MB

# Check audio files
find fma_data/fma_small -name "*.mp3" | wc -l
# Should output: 8000

# Check total size
du -sh fma_data/fma_small
# Should show ~7-8 GB

# Test one file
FIRST_FILE=$(find fma_data/fma_small -name "*.mp3" | head -1)
echo "Testing: $FIRST_FILE"
afplay "$FIRST_FILE" &
sleep 5
killall afplay
echo "✅ Audio file is valid!"
```

---

## After Download: Test with Real Audio

Once you have the audio files, test the pipeline:

### Test 1: Load One Audio File

```bash
python3 -c "
from src.data_loader import MTSDataLoader

loader = MTSDataLoader(target_sr=24000)
dataset = loader.load_fma_dataset(
    audio_dir='./fma_data/fma_small',
    metadata_path='./fma_data/fma_metadata/tracks.csv',
    max_files=1
)

song = dataset['preprocessed_songs'][0]
print(f'✅ Song: {song[\"title\"]}')
print(f'   Artist: {song[\"artist\"]}')
print(f'   Genre: {song[\"genre\"]}')
print(f'   Duration: {song[\"duration\"]:.1f}s')
print(f'   Has audio: {song[\"has_audio\"]}')
print(f'   Audio shape: {song[\"audio\"].shape}')
"
```

Expected output:
```
✅ Song: [Real song title]
   Artist: [Artist name]
   Genre: [Genre]
   Duration: 30.0s
   Has audio: True
   Audio shape: (720000,)
```

### Test 2: Generate Augmented Audio

```bash
python3 -c "
from src.data_loader import MTSDataLoader
from src.augmentation import MTSAudioAugmentation

# Load 5 songs
loader = MTSDataLoader(target_sr=24000)
dataset = loader.load_fma_dataset(
    audio_dir='./fma_data/fma_small',
    metadata_path='./fma_data/fma_metadata/tracks.csv',
    max_files=5
)

# Augment
augmenter = MTSAudioAugmentation(
    sample_rate=24000,
    augmentation_factor=2,
    preserve_original=True
)

augmented = augmenter.augment_dataset(
    dataset['preprocessed_songs'],
    output_dir='./test_output/fma_augmented',
    save_audio=True,
    audio_format='wav'
)

print(f'✅ Generated {len(augmented)} augmented files!')
print(f'   Location: test_output/fma_augmented/')
"
```

### Test 3: Run Full Pipeline

```bash
# Run with 50 FMA songs
python3 run_pipeline.py --config config/config_local_mac.yaml
```

This will:
1. Load 50 real FMA audio files
2. Generate augmented versions (2x factor = 100 total)
3. Save to `outputs/` directory
4. Create dataset splits (train/val/test)

---

## Troubleshooting

### Download Too Slow
- Try `curl -C -` (supports resume)
- Download on faster network, then copy file
- Use browser download (more reliable for large files)

### Not Enough Disk Space
```bash
# Check available space
df -h /Users/zhanggangyi/Desktop

# Need at least 15 GB free:
# - 7.2 GB: FMA audio zip
# - 7.2 GB: Extracted audio
# - 1 GB: Working space
```

To free space:
```bash
# Remove old test outputs
rm -rf test_output/

# Remove incomplete zip
rm -f fma_data/fma_small.zip

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

### Zip Extraction Fails
```bash
# Try with different tools
cd fma_data

# Option 1: unzip (default)
unzip fma_small.zip

# Option 2: Python zipfile
python3 -c "
import zipfile
with zipfile.ZipFile('fma_small.zip', 'r') as z:
    z.extractall('.')
"

# Option 3: 7zip (if installed)
7z x fma_small.zip
```

### Audio Files Don't Load
```bash
# Check file format
file fma_data/fma_small/000/000002.mp3

# Should output:
# fma_data/fma_small/000/000002.mp3: Audio file with ID3 version 2.3.0

# Test with ffmpeg
ffmpeg -i fma_data/fma_small/000/000002.mp3 2>&1 | grep "Audio:"
```

---

## Alternative: Smaller Test Dataset

If FMA download is too large/slow, use a smaller custom dataset:

```bash
# Create test audio directory
mkdir -p custom_audio

# Copy 10 of your own MP3/WAV files to custom_audio/
# Then update config:
```

Edit [config/config_local_mac.yaml](config/config_local_mac.yaml):
```yaml
data:
  use_fma_dataset: false
  custom_audio_dir: "./custom_audio"
```

Then run:
```bash
python3 run_pipeline.py --config config/config_local_mac.yaml
```

---

## Dataset Information

**FMA Small Dataset**:
- **Size**: 7.2 GB compressed, ~7-8 GB extracted
- **Tracks**: 8,000 songs
- **Duration**: 30 seconds per track
- **Format**: MP3, 320 kbps, 44.1 kHz
- **Genres**: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock
- **License**: Creative Commons

**Source**: https://github.com/mdeff/fma

**Paper**:
> Defferrard et al., "FMA: A Dataset For Music Analysis", 2017
> https://arxiv.org/abs/1612.01840

---

## Next Steps After Download

1. ✅ **Verify download**: Run verification commands above
2. 🧪 **Test single file**: Load one audio file to confirm
3. 🎵 **Test augmentation**: Generate augmented versions
4. 🚀 **Run pipeline**: Process full dataset
5. 📊 **Check quality**: Listen to augmented outputs

---

## Quick Reference Commands

```bash
# Download status
tail -f fma_manual_download.log

# Verify files
find fma_data/fma_small -name "*.mp3" | wc -l

# Test load
python3 -c "from src.data_loader import MTSDataLoader; loader = MTSDataLoader(target_sr=24000); ds = loader.load_fma_dataset('./fma_data/fma_small', './fma_data/fma_metadata/tracks.csv', max_files=1); print(ds)"

# Run pipeline
python3 run_pipeline.py --config config/config_local_mac.yaml
```

---

**Current Status**: Background download running (check with `tail -f fma_manual_download.log`)

**ETA**: 30-60 minutes for 7.2 GB download

**What to do now**: Wait for download, or use manual curl/wget commands above
