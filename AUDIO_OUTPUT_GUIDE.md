# Audio Output Guide - MTS-2 Pipeline

## ✅ **AUDIO FILES ARE NOW ENABLED!**

The pipeline has been configured to save audio files in addition to JSON/CSV outputs.

---

## 📁 **Where Audio Files Are Saved**

### **Default Output Structure**
```
test_outputs/
├── logs/
│   └── pipeline.log
├── augmentation/
│   ├── step2_augmentation_results.json  # Metadata
│   └── (Audio files saved to data_dir/augmented)
├── configurations/
│   └── *.csv (train/val/test splits)
└── statistics/
    └── pipeline_stats.json

test_data/
└── augmented/
    ├── original_song_001.wav          # Original audio
    ├── original_song_001_pitch+1.wav  # Augmented version 1
    ├── original_song_001_noise.wav    # Augmented version 2
    ├── original_song_002.wav
    ├── original_song_002_pitch-1.wav
    └── ...
```

---

## ⚙️ **Configuration Settings**

### **In `config/test_config.yaml`**

```yaml
augmentation:
  # ✅ ENABLED - Audio files will be saved
  save_audio: true

  # Format options: wav, mp3, flac
  audio_format: "wav"

  # Keep original files in addition to augmented
  preserve_original: true
```

### **Key Settings:**

| Setting | Value | Effect |
|---------|-------|--------|
| `save_audio: true` | ✅ Enabled | Audio files **WILL** be saved |
| `save_audio: false` | ❌ Disabled | Only JSON metadata saved |
| `audio_format: "wav"` | WAV | Uncompressed, best quality |
| `audio_format: "mp3"` | MP3 | Compressed (needs ffmpeg) |
| `audio_format: "flac"` | FLAC | Lossless compression |
| `preserve_original: true` | Yes | Saves both original & augmented |

---

## 🎵 **Audio File Naming**

### **Pattern:**
```
{original_id}_{augmentation_technique}.{format}
```

### **Examples:**
```
song_001.wav                    # Original
song_001_pitch+2.wav           # Pitch shift +2 semitones
song_001_pitch-1.wav           # Pitch shift -1 semitone
song_001_tempo0.95.wav         # Tempo scaled to 95%
song_001_noise_snr35.wav       # Noise added at 35dB SNR
song_001_eq_low.wav            # Low shelf EQ applied
song_001_reverb_medium.wav     # Medium reverb
```

---

## 📊 **What Gets Saved**

### **1. Audio Files (NEW!)** ✅
- **Location:** `test_data/augmented/*.wav`
- **Content:** Actual audio waveforms
- **Format:** WAV, MP3, or FLAC
- **Size:** ~1-5 MB per 30-second file

### **2. Metadata (JSON)**
- **Location:** `test_outputs/augmentation/step2_augmentation_results.json`
- **Content:**
  - List of all augmented songs
  - Augmentation parameters
  - Quality metrics
  - Statistics

### **3. Dataset CSVs**
- **Location:** `test_outputs/configurations/*.csv`
- **Content:**
  - `train.csv` - Training set with file paths
  - `val.csv` - Validation set
  - `test.csv` - Test set
  - Each CSV includes path to audio file

### **4. Text Prompts**
- **Location:** `test_outputs/step3_labeled_dataset.csv`
- **Content:** Generated text descriptions for each audio

---

## 🔍 **How to Verify Audio Output**

### **After Running Pipeline:**

```bash
# Check if audio files were created
ls -lh test_data/augmented/*.wav

# Count audio files
ls test_data/augmented/*.wav | wc -l

# Play a sample (macOS)
afplay test_data/augmented/song_001.wav

# Check file info
file test_data/augmented/song_001.wav
soxi test_data/augmented/song_001.wav  # If you have sox
```

---

## 💻 **Programmatic Access**

### **Loading Audio in Python:**

```python
import numpy as np
import soundfile as sf  # If available
from pathlib import Path

# Method 1: Using soundfile
audio, sr = sf.read('test_data/augmented/song_001.wav')
print(f"Shape: {audio.shape}, Sample rate: {sr}")

# Method 2: Using numpy (fallback)
audio = np.load('test_data/augmented/song_001.npy')  # If saved as .npy
```

### **Reading CSV with Audio Paths:**

```python
import pandas as pd

# Load dataset CSV
df = pd.read_csv('test_outputs/configurations/train.csv')

# Check audio file paths
print(df['audio_path'].head())

# Load specific audio file
first_audio_path = df['audio_path'].iloc[0]
audio, sr = sf.read(first_audio_path)
```

---

## ⚠️ **Important Notes**

### **Dependency Requirements:**

**For WAV files (recommended):**
```bash
pip install soundfile  # Recommended
# OR
pip install scipy  # Fallback
```

**For MP3 files:**
```bash
# Install ffmpeg (system-level)
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Linux
# Windows: Download from ffmpeg.org

# Then install Python library
pip install soundfile
```

**For FLAC files:**
```bash
pip install soundfile
```

### **Fallback Behavior:**

If `soundfile` is not available:
- ✅ Audio will be saved as `.npy` (numpy arrays)
- ⚠️ Not playable in media players
- ✅ Can still be loaded in Python

---

## 🎛️ **Controlling Audio Output**

### **Disable Audio Saving (JSON only):**
```yaml
augmentation:
  save_audio: false  # Only save metadata
```

### **Change Output Location:**
```yaml
data:
  data_dir: "./my_custom_location"
  # Audio will be saved to: ./my_custom_location/augmented/
```

### **Change Format:**
```yaml
augmentation:
  audio_format: "mp3"  # Options: wav, mp3, flac
```

---

## 📈 **Expected Output Sizes**

For a test run with 10 songs, 30 seconds each, augmentation factor 2:

| Item | Count | Size per Item | Total Size |
|------|-------|---------------|------------|
| Original WAV | 10 | ~1.4 MB | ~14 MB |
| Augmented WAV | 10 | ~1.4 MB | ~14 MB |
| **Total Audio** | **20** | - | **~28 MB** |
| JSON metadata | 1 | ~500 KB | 500 KB |
| CSV files | 3 | ~10 KB | 30 KB |
| **Grand Total** | - | - | **~29 MB** |

---

## 🚀 **Quick Test**

### **Run Pipeline and Check Audio:**

```bash
cd /Users/zhanggangyi/Desktop/MTS/MTS-2

# Run pipeline
python run_pipeline.py --config config/test_config.yaml

# Check outputs
echo "Audio files created:"
ls -lh test_data/augmented/*.wav 2>/dev/null | wc -l

echo "Dataset CSVs created:"
ls -lh test_outputs/configurations/*.csv 2>/dev/null

echo "Sample audio info:"
file test_data/augmented/*.wav 2>/dev/null | head -3
```

---

## 🎯 **Summary**

### ✅ **What's Enabled:**
1. Audio file saving (`save_audio: true`)
2. WAV format (uncompressed)
3. Original preservation
4. Intermediate results saving
5. Complete metadata tracking

### 📁 **Where to Find Files:**
- **Audio:** `test_data/augmented/*.wav`
- **Metadata:** `test_outputs/augmentation/*.json`
- **Datasets:** `test_outputs/configurations/*.csv`
- **Logs:** `test_outputs/logs/pipeline.log`

### 🎵 **File Formats:**
- **Recommended:** WAV (works without extra deps)
- **Compressed:** MP3 (needs ffmpeg)
- **Lossless:** FLAC (needs soundfile)

---

## 🔧 **Troubleshooting**

### **No Audio Files Created?**

1. **Check config:**
   ```bash
   grep "save_audio" config/test_config.yaml
   # Should show: save_audio: true
   ```

2. **Check logs:**
   ```bash
   cat test_outputs/logs/pipeline.log | grep -i audio
   ```

3. **Check output directory:**
   ```bash
   ls -la test_data/augmented/
   ```

### **Audio Files Are .npy Instead of .wav?**

This means `soundfile` is not installed:
```bash
pip install soundfile
```

### **Want MP3 but Getting WAV?**

Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Then run pipeline again
```

---

**AUDIO OUTPUT IS NOW FULLY CONFIGURED! 🎉**

Run the pipeline and you'll get:
- ✅ Audio files (WAV)
- ✅ Metadata (JSON)
- ✅ Dataset splits (CSV)
- ✅ Text prompts
- ✅ Complete logs
