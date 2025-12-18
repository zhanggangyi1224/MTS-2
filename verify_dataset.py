#!/usr/bin/env python3
"""
Dataset Verification Script
Verifies that audio files exist at paths specified in mts_final_dataset.csv
"""

import pandas as pd
from pathlib import Path
import sys
import argparse

def verify_dataset(csv_path, max_check=50):
    """
    Verify dataset integrity by checking audio file paths

    Args:
        csv_path: Path to the dataset CSV
        max_check: Maximum number of files to check (0 = check all)

    Returns:
        True if verification passed, False otherwise
    """

    print("=" * 70)
    print("Dataset Verification")
    print("=" * 70)
    print(f"CSV: {csv_path}")
    print()

    # Check if CSV exists
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ ERROR: CSV file not found: {csv_path}")
        return False

    # Load CSV
    print(f"📖 Loading CSV...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ ERROR: Failed to load CSV: {e}")
        return False

    print(f"✅ Loaded {len(df)} rows")

    # Check for audio_path column
    if 'audio_path' not in df.columns:
        print(f"\n❌ ERROR: CSV missing 'audio_path' column!")
        print(f"   Available columns: {list(df.columns)}")
        print(f"\n💡 Fix: Run regenerate_dataset_csv.py to add audio paths")
        return False

    print(f"✅ audio_path column found")

    # Verify audio paths
    print(f"\n🔍 Verifying audio file paths...")

    total_rows = len(df)
    check_rows = total_rows if max_check == 0 else min(max_check, total_rows)

    empty_paths = []
    missing_files = []
    valid_files = []
    path_errors = []

    print(f"   Checking {check_rows} of {total_rows} samples...")

    for idx, row in df.head(check_rows).iterrows():
        sample_id = row.get('id', f'row_{idx}')
        audio_path = row.get('audio_path', '')

        # Check for empty path (including NaN and non-string values)
        if pd.isna(audio_path) or not isinstance(audio_path, str) or not audio_path.strip():
            empty_paths.append((idx, sample_id))
            continue

        # Check if file exists
        audio_file = Path(audio_path)
        try:
            if audio_file.exists():
                valid_files.append((idx, sample_id, str(audio_path)))
            else:
                missing_files.append((idx, sample_id, str(audio_path)))
        except Exception as e:
            path_errors.append((idx, sample_id, str(audio_path), str(e)))

    # Report results
    print()
    print("=" * 70)
    print("Verification Results")
    print("=" * 70)
    print(f"Total checked:  {check_rows}")
    print(f"Valid files:    {len(valid_files)} ✅")
    print(f"Missing files:  {len(missing_files)} {'⚠️' if missing_files else '✅'}")
    print(f"Empty paths:    {len(empty_paths)} {'⚠️' if empty_paths else '✅'}")
    print(f"Path errors:    {len(path_errors)} {'❌' if path_errors else '✅'}")
    print()

    # Show examples
    if valid_files:
        print("✅ Sample valid files:")
        for idx, sample_id, path in valid_files[:3]:
            print(f"   - {sample_id}: {path}")
        print()

    if empty_paths:
        print("⚠️  Empty audio paths:")
        for idx, sample_id in empty_paths[:5]:
            print(f"   - Row {idx}: {sample_id}")
        if len(empty_paths) > 5:
            print(f"   ... and {len(empty_paths) - 5} more")
        print()

    if missing_files:
        print("⚠️  Missing audio files:")
        for idx, sample_id, path in missing_files[:5]:
            print(f"   - {sample_id}: {path}")
        if len(missing_files) > 5:
            print(f"   ... and {len(missing_files) - 5} more")
        print()

    if path_errors:
        print("❌ Path errors:")
        for idx, sample_id, path, error in path_errors[:5]:
            print(f"   - {sample_id}: {error}")
        if len(path_errors) > 5:
            print(f"   ... and {len(path_errors) - 5} more")
        print()

    # Determine pass/fail
    success_rate = len(valid_files) / check_rows if check_rows > 0 else 0

    print("=" * 70)
    print(f"Success Rate: {success_rate * 100:.1f}%")

    if success_rate == 1.0:
        print("✅ PASSED: All checked files exist")
        return True
    elif success_rate >= 0.8:
        print("⚠️  WARNING: Some files missing but >80% valid")
        print("   Training may proceed with fallback audio for missing files")
        return True
    elif success_rate > 0:
        print("⚠️  WARNING: Many files missing (<80% valid)")
        print("   Training quality will be significantly impacted")
        print("\n💡 Recommended actions:")
        if empty_paths:
            print("   1. Regenerate CSV: python regenerate_dataset_csv.py")
        if missing_files:
            # Check if files are FMA or augmented
            sample_missing = missing_files[0][2] if missing_files else ""
            if "fma_data" in sample_missing:
                print("   2. Download FMA data (will happen automatically in SLURM Phase 0)")
            elif "augmented" in sample_missing:
                print("   2. Run data preparation to generate augmented audio")
        return False
    else:
        print("❌ FAILED: No valid files found")
        print("\n💡 Required actions:")
        print("   1. Check if audio_path column has correct values")
        print("   2. Verify FMA data is in correct location (fma_data/fma_small/)")
        print("   3. Verify augmented audio exists (data/augmented/audio/)")
        print("   4. Regenerate CSV: python regenerate_dataset_csv.py")
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify MTS dataset integrity")
    parser.add_argument("--csv", type=str, default="outputs/mts_final_dataset.csv",
                       help="Path to dataset CSV")
    parser.add_argument("--max-check", type=int, default=50,
                       help="Maximum number of files to check (0 = all)")

    args = parser.parse_args()

    success = verify_dataset(args.csv, args.max_check)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
