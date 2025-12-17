#!/usr/bin/env python3
"""
Simple test to generate a single augmented audio file locally
This helps verify audio quality before running the full pipeline
"""

import sys
import os
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_single_audio_generation():
    """Generate a single augmented audio file to test quality."""

    print("=" * 60)
    print("🎵 Single Audio Quality Test")
    print("=" * 60)
    print()

    # Check dependencies
    print("Checking dependencies...")
    try:
        import librosa
        print("  ✅ librosa")
    except ImportError:
        print("  ❌ librosa - Install: pip install librosa")
        return False

    try:
        import soundfile as sf
        print("  ✅ soundfile")
        has_soundfile = True
    except ImportError:
        print("  ⚠️  soundfile - Install: pip install soundfile (recommended)")
        has_soundfile = False

    try:
        import pyrubberband
        print("  ✅ pyrubberband (high quality)")
        has_pyrubberband = True
    except ImportError:
        print("  ⚠️  pyrubberband - Will use librosa (lower quality)")
        print("     Install: pip install pyrubberband")
        has_pyrubberband = False

    import shutil
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print(f"  ✅ ffmpeg: {ffmpeg}")
    else:
        print("  ⚠️  ffmpeg - Install: brew install ffmpeg (for MP3 output)")

    print()

    # Step 1: Load a test audio file or generate one
    print("Step 1: Loading/generating audio...")
    print()

    # Check for FMA or any audio file
    audio = None
    sr = 24000  # EnCodec-compatible sample rate
    source_file = None

    # Try to find an audio file in FMA
    fma_dir = Path("./fma_data/fma_small")
    if fma_dir.exists():
        audio_files = list(fma_dir.rglob("*.mp3"))[:1]
        if audio_files:
            source_file = audio_files[0]
            print(f"  Found FMA audio: {source_file.name}")
            audio, sr_orig = librosa.load(str(source_file), sr=sr, duration=10.0)
            print(f"  Loaded {len(audio)/sr:.2f}s at {sr} Hz")

    # Try data/raw directory
    if audio is None:
        data_raw = Path("./data/raw")
        if data_raw.exists():
            audio_files = list(data_raw.glob("*.mp3")) + list(data_raw.glob("*.wav"))
            if audio_files:
                source_file = audio_files[0]
                print(f"  Found audio in data/raw: {source_file.name}")
                audio, sr_orig = librosa.load(str(source_file), sr=sr, duration=10.0)
                print(f"  Loaded {len(audio)/sr:.2f}s at {sr} Hz")

    # Generate a musical test signal if no audio found
    if audio is None:
        print("  No audio files found - generating test signal...")
        duration = 10.0  # 10 seconds
        t = np.linspace(0, duration, int(sr * duration))

        # Create a more musical test signal (chord progression)
        freqs = [261.63, 329.63, 392.00]  # C major chord (C, E, G)
        audio = np.zeros_like(t)
        for freq in freqs:
            audio += 0.3 * np.sin(2 * np.pi * freq * t)

        # Add some envelope
        envelope = np.exp(-t / (duration * 0.8))
        audio *= (0.5 + 0.5 * envelope)

        # Add slight vibrato
        vibrato = 0.02 * np.sin(2 * np.pi * 5 * t)
        for i, freq in enumerate(freqs):
            audio += 0.1 * np.sin(2 * np.pi * freq * t * (1 + vibrato))

        audio = audio / np.max(np.abs(audio)) * 0.8  # Normalize
        audio = audio.astype(np.float32)

        print(f"  Generated {duration}s test signal (C major chord)")
        source_file = "generated_test_signal"

    print()

    # Step 2: Save original
    print("Step 2: Saving original audio...")
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)

    original_path = output_dir / "original.wav"
    if has_soundfile:
        sf.write(str(original_path), audio, sr)
    else:
        # Fallback to scipy
        from scipy.io import wavfile
        audio_int = (audio * 32767).astype(np.int16)
        wavfile.write(str(original_path), sr, audio_int)

    print(f"  ✅ Saved: {original_path}")
    print(f"     Size: {original_path.stat().st_size / 1024:.1f} KB")
    print()

    # Step 3: Apply augmentation
    print("Step 3: Applying augmentation...")
    print()

    augmented_versions = []

    # Augmentation 1: Pitch shift +1 semitone
    print("  🎵 Pitch shift: +1 semitone")
    if has_pyrubberband:
        try:
            import pyrubberband as pyrb
            aug1 = pyrb.pitch_shift(audio, sr, 1)
            print("     Using pyrubberband (high quality)")
        except Exception as e:
            print(f"     pyrubberband failed ({e}), using librosa")
            aug1 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)
    else:
        aug1 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)
        print("     Using librosa")

    # Calculate quality metrics
    if len(aug1) > len(audio):
        aug1 = aug1[:len(audio)]
    elif len(aug1) < len(audio):
        aug1 = np.pad(aug1, (0, len(audio) - len(aug1)))

    # SNR calculation
    noise = aug1 - audio
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    print(f"     Quality: SNR = {snr:.2f} dB", end="")
    if snr > 25:
        print(" (Excellent)")
    elif snr > 15:
        print(" (Good)")
    elif snr > 10:
        print(" (Fair)")
    else:
        print(" (Poor)")

    augmented_versions.append(("pitch_shift_+1", aug1))

    # Augmentation 2: Tempo scale 0.95x (slower)
    print("  ⏱️  Tempo scale: 0.95x (5% slower)")
    if has_pyrubberband:
        try:
            import pyrubberband as pyrb
            aug2 = pyrb.time_stretch(audio, sr, 1.0/0.95)  # Rate is inverse
            print("     Using pyrubberband (high quality)")
        except Exception as e:
            print(f"     pyrubberband failed ({e}), using librosa")
            aug2 = librosa.effects.time_stretch(audio, rate=0.95)
    else:
        aug2 = librosa.effects.time_stretch(audio, rate=0.95)
        print("     Using librosa")

    # Resample back to original length for comparison
    if len(aug2) != len(audio):
        from scipy import signal as scipy_signal
        aug2 = scipy_signal.resample(aug2, len(audio))

    print(f"     Duration: {len(aug2)/sr:.2f}s")
    augmented_versions.append(("tempo_scale_0.95", aug2))

    # Augmentation 3: Add gentle noise (SNR=35dB)
    print("  🔊 Add noise: SNR = 35 dB")
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (35 / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    aug3 = audio + noise
    aug3 = aug3.astype(np.float32)

    # Calculate actual SNR
    actual_snr = 10 * np.log10(signal_power / np.mean(noise ** 2))
    print(f"     Actual SNR: {actual_snr:.2f} dB")

    augmented_versions.append(("with_noise_snr35", aug3))

    # Augmentation 4: Combined (pitch + tempo + gentle EQ)
    print("  🎛️  Combined: pitch +0.5, tempo 0.98x, EQ")

    # Start with pitch shift
    if has_pyrubberband:
        try:
            import pyrubberband as pyrb
            aug4 = pyrb.pitch_shift(audio, sr, 0.5)
            aug4 = pyrb.time_stretch(aug4, sr, 1.0/0.98)
        except:
            aug4 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=0.5)
            aug4 = librosa.effects.time_stretch(aug4, rate=0.98)
    else:
        aug4 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=0.5)
        aug4 = librosa.effects.time_stretch(aug4, rate=0.98)

    # Gentle EQ boost at 1kHz
    from scipy import signal as scipy_signal
    # Peaking filter at 1kHz, +1.5dB, Q=1.0
    sos = scipy_signal.butter(2, [800/(sr/2), 1200/(sr/2)], btype='band', output='sos')
    eq_boost = scipy_signal.sosfilt(sos, aug4)
    aug4 = aug4 + 0.15 * eq_boost  # +1.5dB boost

    # Normalize length
    if len(aug4) != len(audio):
        aug4 = scipy_signal.resample(aug4, len(audio))

    print(f"     Applied 3 transforms")
    augmented_versions.append(("combined", aug4))

    print()

    # Step 4: Save all augmented versions
    print("Step 4: Saving augmented audio files...")
    print()

    for name, aug_audio in augmented_versions:
        # Normalize to prevent clipping
        aug_audio = aug_audio / np.max(np.abs(aug_audio)) * 0.9

        # Save as WAV
        wav_path = output_dir / f"augmented_{name}.wav"
        if has_soundfile:
            sf.write(str(wav_path), aug_audio, sr)
        else:
            from scipy.io import wavfile
            audio_int = (aug_audio * 32767).astype(np.int16)
            wavfile.write(str(wav_path), sr, audio_int)

        wav_size = wav_path.stat().st_size / 1024
        print(f"  ✅ {wav_path.name}")
        print(f"     Size: {wav_size:.1f} KB, Duration: {len(aug_audio)/sr:.2f}s")

        # Also save as MP3 if ffmpeg available
        if ffmpeg:
            mp3_path = output_dir / f"augmented_{name}.mp3"
            try:
                import subprocess
                subprocess.run([
                    ffmpeg, "-y", "-i", str(wav_path),
                    "-vn", "-ar", str(sr), "-ac", "1", "-b:a", "192k",
                    str(mp3_path)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                mp3_size = mp3_path.stat().st_size / 1024
                print(f"     MP3: {mp3_size:.1f} KB (compression: {wav_size/mp3_size:.1f}x)")
            except:
                pass

        print()

    # Step 5: Summary
    print("=" * 60)
    print("✅ Test Complete!")
    print("=" * 60)
    print()
    print(f"Original audio source: {source_file if isinstance(source_file, str) else source_file.name}")
    print(f"Sample rate: {sr} Hz (EnCodec-compatible)")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    print("Generated files:")
    print(f"  1. original.wav - Original audio")
    for i, (name, _) in enumerate(augmented_versions, 2):
        print(f"  {i}. augmented_{name}.wav - Augmented version")
        if ffmpeg:
            print(f"     augmented_{name}.mp3 - MP3 version")
    print()
    print("🎧 Listen to the files to verify audio quality!")
    print()
    print("Quality notes:")
    print("  - Original should sound clean")
    print("  - Pitch shifted should maintain clarity")
    print("  - Tempo scaled should sound natural")
    print("  - Noise added should be subtle")
    print("  - Combined should preserve musicality")
    print()

    if has_pyrubberband:
        print("✅ Using high-quality pyrubberband for pitch/tempo")
    else:
        print("⚠️  Using librosa fallback (lower quality)")
        print("   Install pyrubberband for better quality: pip install pyrubberband")

    print()

    return True


if __name__ == "__main__":
    print()
    success = test_single_audio_generation()
    sys.exit(0 if success else 1)
