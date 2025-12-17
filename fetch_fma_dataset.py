#!/usr/bin/env python3
"""
Fetch FMA (Free Music Archive) Dataset
Downloads audio files and metadata for MTS-2 project

FMA Dataset: https://github.com/mdeff/fma
- fma_small: 8,000 tracks, 30s clips, 8GB (RECOMMENDED)
- fma_medium: 25,000 tracks, 30s clips, 25GB
- fma_large: 106,574 tracks, 30s clips, 106GB
- fma_full: 106,574 full tracks, 879GB

This script downloads fma_small by default (manageable size, good quality)
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path, description="Downloading"):
    """Download a file with progress bar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{description}...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=description) as t:
        urllib.request.urlretrieve(url, filename=str(output_path), reporthook=t.update_to)

    print(f"✅ Downloaded: {output_path}")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")

    return output_path


def extract_zip(zip_path, extract_to, description="Extracting"):
    """Extract a zip file with progress."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    print(f"\n{description}...")
    print(f"From: {zip_path}")
    print(f"To: {extract_to}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        print(f"Files in archive: {len(members)}")

        for member in tqdm(members, desc=description):
            zip_ref.extract(member, extract_to)

    print(f"✅ Extracted to: {extract_to}")

    return extract_to


def fetch_fma_dataset(dataset_size="small", output_dir="./fma_data", keep_zip=False):
    """
    Fetch FMA dataset.

    Args:
        dataset_size: "small" (8GB), "medium" (25GB), "large" (106GB), or "full" (879GB)
        output_dir: Where to save the dataset
        keep_zip: Keep the zip file after extraction
    """

    # Dataset URLs (from FMA repository)
    base_url = "https://os.unil.cloud.switch.ch/fma"

    urls = {
        "metadata": f"{base_url}/fma_metadata.zip",
        "small": f"{base_url}/fma_small.zip",
        "medium": f"{base_url}/fma_medium.zip",
        "large": f"{base_url}/fma_large.zip",
        "full": f"{base_url}/fma_full.zip"
    }

    sizes = {
        "metadata": "342 MB",
        "small": "7.2 GB",
        "medium": "22 GB",
        "large": "93 GB",
        "full": "879 GB"
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FMA Dataset Download")
    print("=" * 60)
    print(f"Dataset: fma_{dataset_size}")
    print(f"Expected size: {sizes.get(dataset_size, 'Unknown')}")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)

    # Check if already downloaded
    audio_dir = output_dir / f"fma_{dataset_size}"
    if audio_dir.exists():
        audio_files = list(audio_dir.rglob("*.mp3"))
        if len(audio_files) > 0:
            print(f"\n⚠️  Dataset already exists!")
            print(f"Location: {audio_dir}")
            print(f"Audio files: {len(audio_files)}")
            response = input("\nRe-download? (y/N): ").strip().lower()
            if response != 'y':
                print("Skipping download.")
                return audio_dir

    # Step 1: Download metadata
    print("\n" + "=" * 60)
    print("Step 1: Downloading Metadata")
    print("=" * 60)

    metadata_zip = output_dir / "fma_metadata.zip"
    metadata_dir = output_dir / "fma_metadata"

    if not metadata_dir.exists():
        download_file(urls["metadata"], metadata_zip, "Downloading metadata")
        extract_zip(metadata_zip, output_dir, "Extracting metadata")

        if not keep_zip:
            metadata_zip.unlink()
            print(f"Removed: {metadata_zip}")
    else:
        print(f"✅ Metadata already exists: {metadata_dir}")

    # Step 2: Download audio
    print("\n" + "=" * 60)
    print(f"Step 2: Downloading Audio (fma_{dataset_size})")
    print("=" * 60)

    if dataset_size == "full":
        print("\n⚠️  WARNING: fma_full is 879 GB!")
        print("   This will take many hours and requires lots of disk space.")
        response = input("\nContinue? (y/N): ").strip().lower()
        if response != 'y':
            print("Download cancelled.")
            return None

    audio_zip = output_dir / f"fma_{dataset_size}.zip"

    try:
        download_file(urls[dataset_size], audio_zip, f"Downloading fma_{dataset_size}")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nAlternative download methods:")
        print(f"1. Use wget: wget {urls[dataset_size]}")
        print(f"2. Use curl: curl -O {urls[dataset_size]}")
        print(f"3. Download manually from: https://github.com/mdeff/fma")
        return None

    # Step 3: Extract audio
    print("\n" + "=" * 60)
    print("Step 3: Extracting Audio")
    print("=" * 60)

    extract_zip(audio_zip, output_dir, f"Extracting fma_{dataset_size}")

    if not keep_zip:
        print(f"\nRemoving zip file...")
        audio_zip.unlink()
        print(f"Removed: {audio_zip}")

    # Step 4: Verify
    print("\n" + "=" * 60)
    print("Step 4: Verification")
    print("=" * 60)

    audio_dir = output_dir / f"fma_{dataset_size}"
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        return None

    print(f"Counting audio files...")
    audio_files = list(audio_dir.rglob("*.mp3"))
    print(f"✅ Found {len(audio_files)} audio files")

    if len(audio_files) > 0:
        # Check a sample file
        sample_file = audio_files[0]
        size_kb = sample_file.stat().st_size / 1024
        print(f"\nSample file:")
        print(f"  {sample_file.name}")
        print(f"  Size: {size_kb:.1f} KB")
        print(f"  Path: {sample_file}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ Download Complete!")
    print("=" * 60)
    print(f"Audio files: {len(audio_files)}")
    print(f"Location: {audio_dir}")
    print(f"Metadata: {metadata_dir}")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in audio_files)
    total_gb = total_size / (1024 ** 3)
    print(f"Total size: {total_gb:.2f} GB")

    print("\nNext steps:")
    print("1. Test with one file:")
    print(f"   python3 test_with_slurm_audio.py {audio_files[0]}")
    print("\n2. Run pipeline with FMA data:")
    print("   python3 run_pipeline.py --config config/config_local_mac.yaml")
    print("\n3. Or test augmentation:")
    print("   python3 test_single_audio.py")

    return audio_dir


def main():
    parser = argparse.ArgumentParser(description="Download FMA dataset")
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "full"],
        default="small",
        help="Dataset size (default: small = 8GB)"
    )
    parser.add_argument(
        "--output-dir",
        default="./fma_data",
        help="Output directory (default: ./fma_data)"
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep zip files after extraction"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only metadata (342 MB)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("FMA Dataset Downloader")
    print("=" * 60)
    print()
    print("Dataset Information:")
    print("  fma_small:  8,000 tracks,  30s clips,  7.2 GB  ← Recommended")
    print("  fma_medium: 25,000 tracks, 30s clips, 22 GB")
    print("  fma_large:  106k tracks,   30s clips, 93 GB")
    print("  fma_full:   106k tracks,   full,      879 GB")
    print()
    print(f"Selected: fma_{args.size}")
    print(f"Output: {Path(args.output_dir).absolute()}")
    print()

    # Check disk space
    import shutil
    stat = shutil.disk_usage(Path(args.output_dir).parent if Path(args.output_dir).exists() else ".")
    free_gb = stat.free / (1024 ** 3)
    print(f"Available disk space: {free_gb:.1f} GB")

    size_requirements = {
        "small": 8,
        "medium": 25,
        "large": 100,
        "full": 900
    }

    required = size_requirements.get(args.size, 8)
    if free_gb < required:
        print(f"⚠️  WARNING: Need ~{required} GB free, you have {free_gb:.1f} GB")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Download cancelled.")
            return 1

    # Download
    try:
        if args.metadata_only:
            print("\nDownloading metadata only...")
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            base_url = "https://os.unil.cloud.switch.ch/fma"
            metadata_zip = output_dir / "fma_metadata.zip"

            download_file(f"{base_url}/fma_metadata.zip", metadata_zip, "Downloading metadata")
            extract_zip(metadata_zip, output_dir, "Extracting metadata")

            if not args.keep_zip:
                metadata_zip.unlink()

            print(f"\n✅ Metadata downloaded to: {output_dir / 'fma_metadata'}")
        else:
            result = fetch_fma_dataset(
                dataset_size=args.size,
                output_dir=args.output_dir,
                keep_zip=args.keep_zip
            )

            if result is None:
                return 1

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
