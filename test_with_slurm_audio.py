#!/usr/bin/env python3
"""
Test Audio Quality with SLURM-Generated Data

Workflow:
1. Run slurm_data_prep_only.sh on SLURM to generate dataset
2. Copy one audio file to local Mac
3. Run this script to verify quality

This lets you test with REAL audio from the dataset!
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def find_slurm_audio_files():
    """Find audio files that were copied from SLURM."""

    search_paths = [
        Path("./slurm_data"),           # Recommended location
        Path("./data/slurm_audio"),     # Alternative
        Path("./test_audio"),           # Alternative
        Path("~/Downloads"),            # If you copied to Downloads
    ]

    audio_files = []
    for path in search_paths:
        path = path.expanduser()
        if path.exists():
            found = list(path.glob("*.mp3")) + list(path.glob("*.wav"))
            audio_files.extend(found)

    return audio_files


def test_slurm_audio(audio_file: Path):
    """Test a single audio file from SLURM."""

    print("=" * 60)
    print("🎵 Testing SLURM-Generated Audio")
    print("=" * 60)
    print()

    # Check dependencies
    print("Checking dependencies...")
    try:
        import librosa
        print("  ✅ librosa")
    except ImportError:
        print("  ❌ librosa not available")
        return False

    try:
        import soundfile as sf
        print("  ✅ soundfile")
        has_soundfile = True
    except ImportError:
        print("  ⚠️  soundfile not available")
        has_soundfile = False

    print()

    # Load the audio
    print(f"Loading: {audio_file.name}")
    print(f"Source: {audio_file}")
    print()

    try:
        # Load at original sample rate first
        audio_orig, sr_orig = librosa.load(str(audio_file), sr=None)
        print(f"  Original sample rate: {sr_orig} Hz")
        print(f"  Duration: {len(audio_orig)/sr_orig:.2f} seconds")
        print(f"  Samples: {len(audio_orig):,}")

        # Resample to 24kHz if needed (EnCodec)
        if sr_orig != 24000:
            print(f"  Resampling: {sr_orig} Hz → 24000 Hz")
            audio = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=24000)
            sr = 24000
        else:
            audio = audio_orig
            sr = sr_orig

        print()

        # Analyze audio quality
        print("Audio Quality Analysis:")
        print()

        # 1. Check for clipping
        max_val = np.max(np.abs(audio))
        if max_val >= 0.99:
            print(f"  ⚠️  Clipping detected! Peak: {max_val:.3f}")
        else:
            print(f"  ✅ No clipping. Peak: {max_val:.3f}")

        # 2. Check for silence
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.001:
            print(f"  ⚠️  Audio very quiet. RMS: {rms:.6f}")
        else:
            print(f"  ✅ Good signal level. RMS: {rms:.6f}")

        # 3. Dynamic range
        dynamic_range_db = 20 * np.log10(max_val / (rms + 1e-10))
        print(f"  Dynamic range: {dynamic_range_db:.2f} dB")

        # 4. Spectral analysis
        spec = np.abs(librosa.stft(audio))
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        mean_centroid = np.mean(spectral_centroid)
        print(f"  Spectral centroid: {mean_centroid:.1f} Hz")

        # 5. Check for artifacts
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        mean_zcr = np.mean(zcr)
        print(f"  Zero crossing rate: {mean_zcr:.4f}")

        print()

        # Save analysis results
        output_dir = Path("./test_output")
        output_dir.mkdir(exist_ok=True)

        # Save original (resampled to 24kHz)
        original_out = output_dir / f"slurm_original_{audio_file.stem}.wav"
        if has_soundfile:
            sf.write(str(original_out), audio, sr)
        else:
            from scipy.io import wavfile
            audio_int = (audio * 32767).astype(np.int16)
            wavfile.write(str(original_out), sr, audio_int)

        print(f"✅ Saved to: {original_out}")
        print(f"   Size: {original_out.stat().st_size / 1024:.1f} KB")
        print()

        # Apply augmentation to test quality preservation
        print("Testing Augmentation Quality:")
        print()

        # Pitch shift test
        print("  🎵 Applying pitch shift +1 semitone...")
        augmented = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)

        # Calculate SNR
        if len(augmented) > len(audio):
            augmented = augmented[:len(audio)]
        elif len(augmented) < len(audio):
            augmented = np.pad(augmented, (0, len(audio) - len(augmented)))

        noise = augmented - audio
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

        print(f"     Augmentation SNR: {snr:.2f} dB", end="")
        if snr > 25:
            print(" (Excellent)")
        elif snr > 15:
            print(" (Good)")
        elif snr > 10:
            print(" (Fair)")
        else:
            print(" (Poor)")

        # Save augmented version
        augmented_out = output_dir / f"slurm_augmented_{audio_file.stem}.wav"
        augmented = augmented / np.max(np.abs(augmented)) * 0.9  # Normalize

        if has_soundfile:
            sf.write(str(augmented_out), augmented, sr)
        else:
            from scipy.io import wavfile
            audio_int = (augmented * 32767).astype(np.int16)
            wavfile.write(str(augmented_out), sr, audio_int)

        print(f"     Saved to: {augmented_out}")
        print()

        # Summary
        print("=" * 60)
        print("✅ Analysis Complete!")
        print("=" * 60)
        print()
        print("Results:")
        print(f"  Source: SLURM-generated dataset")
        print(f"  Sample rate: {sr} Hz (EnCodec-compatible)")
        print(f"  Peak level: {max_val:.3f}")
        print(f"  RMS level: {rms:.6f}")
        print(f"  Dynamic range: {dynamic_range_db:.2f} dB")
        print(f"  Augmentation SNR: {snr:.2f} dB")
        print()
        print("Saved files:")
        print(f"  1. {original_out.name}")
        print(f"  2. {augmented_out.name}")
        print()
        print("🎧 Listen to verify quality!")
        print()

        # Quality verdict
        if max_val < 0.99 and rms > 0.01 and snr > 15:
            print("✅ QUALITY: GOOD - Audio from SLURM dataset is high quality!")
        elif snr > 10:
            print("⚠️  QUALITY: FAIR - May need quality improvements")
        else:
            print("❌ QUALITY: POOR - Check SLURM augmentation settings")

        print()

        return True

    except Exception as e:
        print(f"❌ Error loading/analyzing audio: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test audio quality from SLURM-generated dataset"
    )
    parser.add_argument("audio_file", nargs="?",
                       help="Path to audio file from SLURM (optional)")
    parser.add_argument("--search", action="store_true",
                       help="Search for SLURM audio files")

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("🎵 SLURM Audio Quality Test")
    print("=" * 60)
    print()

    # If no file specified, search for files
    if not args.audio_file or args.search:
        print("Searching for SLURM audio files...")
        audio_files = find_slurm_audio_files()

        if not audio_files:
            print()
            print("❌ No audio files found!")
            print()
            print("Please copy audio from SLURM first:")
            print()
            print("1. Run on SLURM:")
            print("   sbatch slurm_data_prep_only.sh")
            print()
            print("2. Wait for job to complete, then copy one file:")
            print("   scp your-user@spartan:'/data/.../augmented/audio/*.mp3' ./slurm_data/")
            print()
            print("   Or create directory and copy manually:")
            print("   mkdir slurm_data")
            print("   # Copy one .mp3 or .wav file to slurm_data/")
            print()
            print("3. Run this script again:")
            print("   python3 test_with_slurm_audio.py")
            print()
            return 1

        print(f"Found {len(audio_files)} audio file(s):")
        for i, f in enumerate(audio_files, 1):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {i}. {f.name} ({size_mb:.2f} MB)")

        # Use first file
        audio_file = audio_files[0]
        print()
        print(f"Testing: {audio_file.name}")
        print()
    else:
        audio_file = Path(args.audio_file)
        if not audio_file.exists():
            print(f"❌ File not found: {audio_file}")
            return 1

    # Test the audio
    success = test_slurm_audio(audio_file)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
