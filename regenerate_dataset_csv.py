#!/usr/bin/env python3
"""
Regenerate mts_final_dataset.csv with audio_path column included
This script reads existing intermediate files and creates a new CSV with audio paths
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def regenerate_csv():
    """Regenerate the final dataset CSV with audio paths"""

    print("=" * 70)
    print("Regenerating mts_final_dataset.csv with audio paths")
    print("=" * 70)

    # Paths
    project_dir = Path(__file__).parent
    outputs_dir = project_dir / "outputs"

    # Load the existing labeled dataset CSV
    labeled_csv = outputs_dir / "step3_labeled_dataset.csv"
    if not labeled_csv.exists():
        print(f"❌ ERROR: {labeled_csv} not found!")
        return False

    print(f"\n📖 Reading labeled dataset: {labeled_csv}")
    labeled_df = pd.read_csv(labeled_csv)
    print(f"   Found {len(labeled_df)} labeled songs")

    # Load augmentation results JSON to get audio file paths
    aug_json = outputs_dir / "augmentation" / "step2_augmentation_results.json"
    if not aug_json.exists():
        print(f"❌ ERROR: {aug_json} not found!")
        return False

    print(f"\n📖 Reading augmentation results: {aug_json}")
    with open(aug_json) as f:
        aug_data = json.load(f)

    augmented_songs = aug_data.get("augmented_songs", [])
    print(f"   Found {len(augmented_songs)} augmented songs")

    # Create a mapping of song ID to audio path
    audio_path_map = {}

    # Add augmented song paths
    for song in augmented_songs:
        song_id = song.get("id")
        # Check both possible audio path fields
        audio_path = song.get("audio_file_path", "") or song.get("audio_path", "")
        if song_id and audio_path:
            audio_path_map[song_id] = audio_path

    print(f"   Built audio path map with {len(audio_path_map)} entries")

    # Load structure annotations if available
    structure_json = outputs_dir / "structure" / "structure_annotations.json"
    structure_map = {}

    if structure_json.exists():
        print(f"\n📖 Reading structure annotations: {structure_json}")
        with open(structure_json) as f:
            structure_data = json.load(f)

        annotated_songs = structure_data.get("annotated_songs", [])
        for song in annotated_songs:
            song_id = song.get("id")
            if song_id:
                structure_map[song_id] = song

        print(f"   Found {len(structure_map)} structured songs")
    else:
        print(f"\n⚠️  Structure annotations not found: {structure_json}")
        print("   Proceeding without structure data")

    # Build the final CSV data
    print("\n🔨 Building final dataset CSV...")
    csv_data = []
    missing_paths = []

    for _, row in labeled_df.iterrows():
        song_id = row['id']

        # Get audio path
        audio_path = audio_path_map.get(song_id, "")

        # If not found in augmented, check if it's an original song
        if not audio_path:
            # For original songs, construct path from FMA data
            # This assumes original songs follow the pattern fma_XXXXXX
            if song_id.startswith("fma_") and not song_id.endswith(("_aug_01", "_aug_02")):
                # Extract track ID
                try:
                    tid = int(song_id.replace("fma_", ""))
                    # FMA path pattern: fma_data/fma_small/XXX/00XXXX.mp3
                    audio_path = f"fma_data/fma_small/{tid:03d}/{tid:06d}.mp3"
                except ValueError:
                    pass

        if not audio_path:
            missing_paths.append(song_id)

        # Get structure info
        structure_info = structure_map.get(song_id, {})

        # Build row
        csv_row = {
            "id": song_id,
            "original_id": row.get('original_id', ''),
            "title": row.get('title', ''),
            "artist": row.get('artist', ''),
            "genre": row.get('genre', ''),
            "duration": row.get('duration', 0),
            "language": row.get('language', ''),
            "text_prompt": row.get('text_prompt', ''),
            "is_augmented": row.get('is_augmented', False),
            "augmentation_quality": row.get('augmentation_quality', 'original'),
            "has_structure": bool(structure_info),
            "structure_template": structure_info.get("structure_template", ""),
            "total_sections": structure_info.get("total_sections", 0),
            "structure_coherence": structure_info.get("structure_coherence_score", 0.0),
            "prompt_length": row.get('prompt_length', len(row.get('text_prompt', '').split())),
            "sample_rate": 24000,  # Default sample rate
            "processing_date": row.get('processing_date', datetime.now().isoformat()),
            "audio_path": audio_path
        }

        csv_data.append(csv_row)

    # Create DataFrame and save
    final_df = pd.DataFrame(csv_data)
    output_csv = outputs_dir / "mts_final_dataset.csv"

    print(f"\n💾 Saving to: {output_csv}")
    final_df.to_csv(output_csv, index=False)

    # Summary
    print("\n" + "=" * 70)
    print("✅ CSV Regeneration Complete!")
    print("=" * 70)
    print(f"Total rows: {len(final_df)}")
    print(f"Rows with audio paths: {len(final_df[final_df['audio_path'] != ''])}")
    print(f"Rows missing audio paths: {len(missing_paths)}")

    if missing_paths:
        print(f"\n⚠️  Warning: {len(missing_paths)} songs missing audio paths:")
        for song_id in missing_paths[:10]:
            print(f"   - {song_id}")
        if len(missing_paths) > 10:
            print(f"   ... and {len(missing_paths) - 10} more")

    print(f"\n📊 Sample rows:")
    print(final_df[['id', 'is_augmented', 'audio_path']].head(10).to_string())

    print(f"\n✅ Dataset saved to: {output_csv}")
    print("=" * 70)

    return True

if __name__ == "__main__":
    import sys
    success = regenerate_csv()
    sys.exit(0 if success else 1)
