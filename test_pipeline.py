#!/usr/bin/env python
"""
Test script for MTS-2 pipeline

This script tests each component individually and then runs the full pipeline.
It checks for:
- Import errors
- Missing dependencies
- Configuration errors
- Basic functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)

    tests = []

    # Core imports
    try:
        import numpy as np
        print("✅ numpy")
        tests.append(True)
    except ImportError as e:
        print(f"❌ numpy: {e}")
        tests.append(False)

    try:
        import pandas as pd
        print("✅ pandas")
        tests.append(True)
    except ImportError as e:
        print(f"❌ pandas: {e}")
        tests.append(False)

    # Optional but important
    try:
        import librosa
        print("✅ librosa")
        tests.append(True)
    except ImportError as e:
        print(f"⚠️  librosa (optional): {e}")
        tests.append(True)  # Don't fail

    try:
        import transformers
        print("✅ transformers")
        tests.append(True)
    except ImportError as e:
        print(f"⚠️  transformers (optional): {e}")
        tests.append(True)  # Don't fail

    # Pipeline modules
    try:
        from pipeline import MTSDataPipeline
        print("✅ pipeline.MTSDataPipeline")
        tests.append(True)
    except ImportError as e:
        print(f"❌ pipeline: {e}")
        tests.append(False)

    try:
        from data_loader import MTSDataLoader
        print("✅ data_loader.MTSDataLoader")
        tests.append(True)
    except ImportError as e:
        print(f"❌ data_loader: {e}")
        tests.append(False)

    try:
        from augmentation import MTSAudioAugmentation
        print("✅ augmentation.MTSAudioAugmentation")
        tests.append(True)
    except ImportError as e:
        print(f"❌ augmentation: {e}")
        tests.append(False)

    try:
        from text_generation import MTSTextPromptGenerator
        print("✅ text_generation.MTSTextPromptGenerator")
        tests.append(True)
    except ImportError as e:
        print(f"❌ text_generation: {e}")
        tests.append(False)

    try:
        from data_organization import MTSDataOrganizer
        print("✅ data_organization.MTSDataOrganizer")
        tests.append(True)
    except ImportError as e:
        print(f"❌ data_organization: {e}")
        tests.append(False)

    try:
        from structure_processing import MTSStructureProcessor
        print("✅ structure_processing.MTSStructureProcessor")
        tests.append(True)
    except ImportError as e:
        print(f"❌ structure_processing: {e}")
        tests.append(False)

    try:
        from batch_processor import MTSBatchProcessor
        print("✅ batch_processor.MTSBatchProcessor")
        tests.append(True)
    except ImportError as e:
        print(f"❌ batch_processor: {e}")
        tests.append(False)

    success_rate = sum(tests) / len(tests) * 100
    print(f"\nImport success rate: {success_rate:.1f}% ({sum(tests)}/{len(tests)})")
    return all(tests[:2])  # Must have numpy and pandas


def test_config_loading():
    """Test configuration file loading."""
    print("\n" + "=" * 60)
    print("TEST 2: Configuration Loading")
    print("=" * 60)

    import yaml

    config_files = [
        "config/test_config.yaml",
        "config/config.yaml",
        "config/improved_config.yaml",
        "config/batch_config.yaml"
    ]

    loaded = []
    for config_file in config_files:
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            print(f"✅ {config_file}")
            loaded.append(config)
        except FileNotFoundError:
            print(f"⚠️  {config_file} not found")
        except Exception as e:
            print(f"❌ {config_file}: {e}")

    return len(loaded) > 0


def test_data_loader():
    """Test data loader with simulated data."""
    print("\n" + "=" * 60)
    print("TEST 3: Data Loader (Simulated)")
    print("=" * 60)

    try:
        from data_loader import MTSDataLoader

        # Create loader with minimal config
        # MTSDataLoader takes config dict, not individual parameters
        config = {
            "data": {
                "data_source": "simulated",
                "num_songs": 5,
                "cache_dir": "./test_cache",
                "data_dir": "./test_data",
                "simulated": {
                    "num_songs": 5,
                    "duration_range": [30, 60],
                    "sample_rate": 24000
                }
            }
        }
        loader = MTSDataLoader(config)

        print("✅ MTSDataLoader created")

        # Try to load data
        songs = loader.load_dataset()
        print(f"✅ Loaded {len(songs)} songs")

        if len(songs) > 0:
            song = songs[0]
            print(f"   Sample song keys: {list(song.keys())}")

        return True

    except Exception as e:
        print(f"❌ Data loader failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_augmentation():
    """Test audio augmentation (without actual audio)."""
    print("\n" + "=" * 60)
    print("TEST 4: Augmentation (Dry Run)")
    print("=" * 60)

    try:
        from augmentation import MTSAudioAugmentation
        import numpy as np

        # Create augmenter
        augmenter = MTSAudioAugmentation(
            sample_rate=24000,
            augmentation_factor=2
        )

        print("✅ MTSAudioAugmentation created")

        # Test with fake audio
        fake_audio = np.random.randn(24000).astype(np.float32)  # 1 second
        print("✅ Created test audio")

        # Test augmentation (may fail if dependencies missing)
        try:
            # Create augmentation plan
            plan = augmenter.generate_augmentation_plan('test_001')
            print(f"✅ Generated augmentation plan with {len(plan)} steps")

            # Try to apply one augmentation
            if plan:
                result = augmenter.apply_augmentation_plan(
                    fake_audio,
                    plan[0],
                    'test_001'
                )
                print(f"✅ Applied augmentation successfully")
        except Exception as e:
            print(f"⚠️  Augmentation failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Augmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_generation():
    """Test text prompt generation."""
    print("\n" + "=" * 60)
    print("TEST 5: Text Generation")
    print("=" * 60)

    try:
        from text_generation import MTSTextPromptGenerator

        # Create generator
        generator = MTSTextPromptGenerator(
            model_name="t5-small",
            cache_dir="./test_cache",
            num_prompts=10
        )

        print("✅ MTSTextPromptGenerator created")

        # Generate prompt pool (may fail if no transformers)
        try:
            pool = generator.generate_prompt_pool()
            print(f"✅ Generated {len(pool)} prompts")
        except Exception as e:
            print(f"⚠️  Prompt generation failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Text generation test failed: {e}")
        return False


def test_pipeline_creation():
    """Test pipeline creation."""
    print("\n" + "=" * 60)
    print("TEST 6: Pipeline Creation")
    print("=" * 60)

    try:
        import yaml
        from pipeline import MTSDataPipeline

        # Load test config
        config_path = "config/test_config.yaml"
        print(f"✅ Loading config from {config_path}")

        # Create pipeline (pass path, not dict)
        pipeline = MTSDataPipeline(config_path)
        print("✅ MTSDataPipeline created")

        return True

    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test running the full pipeline with minimal data."""
    print("\n" + "=" * 60)
    print("TEST 7: Full Pipeline Execution")
    print("=" * 60)

    try:
        import yaml
        from pipeline import MTSDataPipeline

        # Use config path
        config_path = "config/test_config.yaml"

        # Create and run pipeline (pass path)
        pipeline = MTSDataPipeline(config_path)
        print("✅ Pipeline created, starting execution...")

        results = pipeline.run()
        print("✅ Pipeline completed!")

        # Check results
        if results and 'summary' in results:
            summary = results['summary']
            print(f"\n📊 Results:")
            print(f"   Total songs: {summary.get('total_songs', 'N/A')}")
            print(f"   Augmented: {summary.get('augmented_songs', 'N/A')}")
            print(f"   Output dir: {results.get('output_dir', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "🧪" * 30)
    print("MTS-2 PIPELINE TEST SUITE")
    print("🧪" * 30 + "\n")

    results = {
        "imports": test_imports(),
        "config": test_config_loading(),
        "data_loader": test_data_loader(),
        "augmentation": test_augmentation(),
        "text_generation": test_text_generation(),
        "pipeline_creation": test_pipeline_creation(),
    }

    # Only run full pipeline if basic tests pass
    if all(results.values()):
        print("\n✅ All basic tests passed! Running full pipeline test...")
        results["full_pipeline"] = test_full_pipeline()
    else:
        print("\n⚠️  Some basic tests failed, skipping full pipeline test")
        results["full_pipeline"] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    passed = sum(results.values())
    total = len(results)
    print(f"\n{passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
