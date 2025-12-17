#!/usr/bin/env python3
"""
Local Test Runner for MTS-2 on Mac
Tests the pipeline with a small dataset to verify audio quality
"""

import sys
import os
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_environment():
    """Setup Mac-optimized environment."""
    print("🍎 Setting up Mac environment...")

    # Try to import Mac optimization
    try:
        from mac_optimization import setup_mac_environment
        optimizer = setup_mac_environment()
        return optimizer
    except ImportError:
        print("⚠️  Mac optimization module not found, using standard setup")
        return None

def test_audio_quality():
    """Test audio loading and augmentation quality."""
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    print("\n" + "="*60)
    print("🎵 Testing Audio Quality")
    print("="*60)

    # Test 1: Check librosa
    try:
        import librosa
        print("✅ librosa available")

        # Generate test audio
        sr = 24000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

        # Test pitch shift
        print("   Testing pitch shift...")
        shifted = librosa.effects.pitch_shift(test_audio, sr=sr, n_steps=2)

        # Calculate SNR
        noise = shifted[:len(test_audio)] - test_audio
        signal_power = np.mean(test_audio ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

        print(f"   Pitch shift SNR: {snr:.2f} dB")

        if snr > 20:
            print("   ✅ High quality pitch shifting")
        elif snr > 10:
            print("   ⚠️  Medium quality pitch shifting")
        else:
            print("   ❌ Low quality pitch shifting")

    except ImportError:
        print("❌ librosa not available - install with: pip install librosa")
        return False

    # Test 2: Check soundfile
    try:
        import soundfile as sf
        print("✅ soundfile available for high-quality I/O")
    except ImportError:
        print("⚠️  soundfile not available - install with: pip install soundfile")

    # Test 3: Check pyrubberband
    try:
        import pyrubberband
        print("✅ pyrubberband available for high-quality time/pitch shifting")
    except ImportError:
        print("⚠️  pyrubberband not available - will use librosa fallback")
        print("   Install with: pip install pyrubberband")

    # Test 4: Check ffmpeg
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"✅ ffmpeg available at: {ffmpeg_path}")
    else:
        print("⚠️  ffmpeg not found - install with: brew install ffmpeg")

    return True

def run_quick_test(config_path: str = "config/config_local_mac.yaml"):
    """Run a quick test of the pipeline."""
    from pipeline import MTSDataPipeline
    import yaml

    print("\n" + "="*60)
    print("🚀 Running Quick Pipeline Test")
    print("="*60)

    # Load config
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"❌ Config file not found: {config_path}")
        print(f"   Using default config")
        config_path = "config/config.yaml"

    # Update config for quick test
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Make it a quick test
    config['data']['fma_max_files'] = 10  # Only 10 files
    config['augmentation']['augmentation_factor'] = 2  # Only 2x augmentation
    config['text_generation']['prompt_pool_size'] = 50
    config['structure_processing']['enabled'] = False

    # Save temporary config
    test_config_path = Path("config/test_config_quick.yaml")
    with open(test_config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"📝 Using test config with:")
    print(f"   - Max files: {config['data']['fma_max_files']}")
    print(f"   - Augmentation factor: {config['augmentation']['augmentation_factor']}")
    print(f"   - Sample rate: {config['data']['target_sample_rate']} Hz")

    # Run pipeline
    try:
        pipeline = MTSDataPipeline(config_path=str(test_config_path))
        results = pipeline.run_complete_pipeline()

        print("\n" + "="*60)
        print("✅ Pipeline test completed successfully!")
        print("="*60)
        print(f"Status: {results.get('status')}")
        print(f"Total songs processed: {results.get('data_counts', {}).get('original_songs', 0)}")
        print(f"Augmented songs created: {results.get('data_counts', {}).get('augmented_songs', 0)}")
        print(f"Execution time: {results.get('total_execution_time', 0):.2f}s")

        # Check output files
        output_dir = Path(config['data']['data_dir']) / "augmented" / "audio"
        if output_dir.exists():
            audio_files = list(output_dir.glob("*.wav")) + list(output_dir.glob("*.mp3"))
            print(f"\n🎵 Audio files generated: {len(audio_files)}")
            if audio_files:
                print(f"   Sample: {audio_files[0].name}")
                # Check file size
                size_mb = audio_files[0].stat().st_size / (1024 * 1024)
                print(f"   File size: {size_mb:.2f} MB")

                if size_mb > 0.1:
                    print("   ✅ Audio file has reasonable size")
                else:
                    print("   ⚠️  Audio file seems too small")

        return True

    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_directory():
    """Check if FMA data is available."""
    print("\n" + "="*60)
    print("📂 Checking Data Directory")
    print("="*60)

    fma_dir = Path("./fma_data/fma_small")

    if fma_dir.exists():
        audio_files = list(fma_dir.rglob("*.mp3"))
        print(f"✅ FMA directory found: {fma_dir}")
        print(f"   Audio files: {len(audio_files)}")

        if len(audio_files) > 0:
            print(f"   Sample file: {audio_files[0]}")
            return True
        else:
            print("   ⚠️  No audio files found in FMA directory")
    else:
        print(f"❌ FMA directory not found: {fma_dir}")
        print("\nTo download FMA dataset:")
        print("   1. Visit: https://github.com/mdeff/fma")
        print("   2. Download fma_small.zip (8 hours, 8GB)")
        print("   3. Extract to ./fma_data/fma_small")
        print("\nOr use your own audio files:")
        print("   - Place .mp3 or .wav files in ./data/audio/")
        print("   - Update config: custom_audio_dir: './data/audio'")

    return False

def main():
    parser = argparse.ArgumentParser(description="Local test runner for MTS-2 on Mac")
    parser.add_argument("--test-audio", action="store_true",
                       help="Test audio processing quality")
    parser.add_argument("--test-pipeline", action="store_true",
                       help="Run quick pipeline test")
    parser.add_argument("--check-data", action="store_true",
                       help="Check data directory")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests")
    parser.add_argument("--config", default="config/config_local_mac.yaml",
                       help="Config file to use")

    args = parser.parse_args()

    # If no specific test, run all
    if not (args.test_audio or args.test_pipeline or args.check_data):
        args.all = True

    print("\n" + "="*60)
    print("🍎 MTS-2 Local Test Runner for Mac")
    print("="*60)

    # Setup Mac environment
    optimizer = setup_environment()

    success = True

    # Run tests
    if args.all or args.check_data:
        if not check_data_directory():
            print("\n⚠️  No audio data available. Please download FMA or add your own audio.")
            success = False

    if args.all or args.test_audio:
        if not test_audio_quality():
            success = False

    if args.all or args.test_pipeline:
        if not run_quick_test(args.config):
            success = False

    # Summary
    print("\n" + "="*60)
    if success:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed. Check output above.")
    print("="*60 + "\n")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
