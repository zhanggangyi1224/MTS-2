"""
MTS Data Organization - Stratified splitting and training configuration creation
"""

import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class MTSDataOrganizer:
    """
    Data organization system implementing stratified splitting by genre
    and creating multiple training configurations as specified in project plan
    """
    
    def __init__(self, 
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 random_state: int = 42):
        """
        Initialize data organizer
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set  
            test_ratio: Proportion for test set
            random_state: Random seed for reproducibility
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total}"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Set random seeds
        random.seed(random_state)
        np.random.seed(random_state)
        
        self.stats = {
            "total_songs": 0,
            "genres_processed": 0,
            "splits_created": 0,
            "configurations_created": 0
        }
    
    def stratified_split_by_genre(self, songs: List[Dict]) -> Dict:
        """
        Split data ensuring genre diversity in each split
        As specified in project plan: "ensure genre diversity in each split 
        to avoid overfitting to specific genres"
        """
        print(f"🎯 Performing stratified split on {len(songs)} songs...")
        
        # Group songs by genre
        genre_groups = defaultdict(list)
        for song in songs:
            genre = song.get("genre", "unknown")
            genre_groups[genre].append(song)
        
        print(f"   Found {len(genre_groups)} genres")
        
        # Initialize splits
        train_data = []
        val_data = []
        test_data = []
        
        split_info = {
            "genre_distribution": {},
            "original_vs_augmented": {
                "train": {"original": 0, "augmented": 0},
                "val": {"original": 0, "augmented": 0},
                "test": {"original": 0, "augmented": 0}
            },
            "quality_distribution": {
                "train": defaultdict(int),
                "val": defaultdict(int),
                "test": defaultdict(int)
            }
        }
        
        # Split each genre proportionally
        for genre, genre_songs in genre_groups.items():
            # Shuffle songs within genre
            random.shuffle(genre_songs)
            
            n_songs = len(genre_songs)
            n_train = int(n_songs * self.train_ratio)
            n_val = int(n_songs * self.val_ratio)
            n_test = n_songs - n_train - n_val  # Remainder to test
            
            # Ensure minimum 1 song in each split if possible
            if n_songs >= 3:
                n_train = max(1, n_train)
                n_val = max(1, n_val)
                n_test = max(1, n_test)
                
                # Adjust if totals don't match
                total_assigned = n_train + n_val + n_test
                if total_assigned != n_songs:
                    diff = n_songs - total_assigned
                    n_train += diff  # Add difference to training
            
            # Split the genre
            train_songs = genre_songs[:n_train]
            val_songs = genre_songs[n_train:n_train + n_val]
            test_songs = genre_songs[n_train + n_val:]
            
            # Add to main splits
            train_data.extend(train_songs)
            val_data.extend(val_songs)
            test_data.extend(test_songs)
            
            # Track statistics
            split_info["genre_distribution"][genre] = {
                "total": n_songs,
                "train": len(train_songs),
                "val": len(val_songs),
                "test": len(test_songs)
            }
            
            # Track original vs augmented distribution
            for split_name, split_songs in [("train", train_songs), 
                                           ("val", val_songs), 
                                           ("test", test_songs)]:
                for song in split_songs:
                    if song.get("is_augmented", False):
                        split_info["original_vs_augmented"][split_name]["augmented"] += 1
                    else:
                        split_info["original_vs_augmented"][split_name]["original"] += 1
                    
                    # Track quality distribution
                    quality = song.get("augmentation_metadata", {}).get("quality_category", "high")
                    split_info["quality_distribution"][split_name][quality] += 1
        
        # Final shuffle of splits
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        self.stats["total_songs"] = len(songs)
        self.stats["genres_processed"] = len(genre_groups)
        self.stats["splits_created"] = 3
        
        result = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "split_info": split_info,
            "split_sizes": {
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data)
            }
        }
        
        print(f"✅ Split complete:")
        print(f"   Train: {len(train_data)} ({len(train_data)/len(songs)*100:.1f}%)")
        print(f"   Val: {len(val_data)} ({len(val_data)/len(songs)*100:.1f}%)")
        print(f"   Test: {len(test_data)} ({len(test_data)/len(songs)*100:.1f}%)")
        
        return result
    
    def create_segment_datasets(self, 
                               songs: List[Dict],
                               segment_durations: List[int] = [30, 60]) -> Dict:
        """
        Create segmented datasets for different training stages
        As specified: "segment songs into shorter chunks for certain training stages
        (e.g. 30 s segments for initial model training)"
        """
        print(f"✂️  Creating segment datasets: {segment_durations}s segments...")
        
        segment_datasets = {}
        
        for duration in segment_durations:
            print(f"   Processing {duration}s segments...")
            segments = []
            
            for song in songs:
                song_duration = song.get("duration", 275.0)
                n_segments = int(song_duration // duration)
                
                # Create segments for this song
                for seg_idx in range(n_segments):
                    start_time = seg_idx * duration
                    end_time = min((seg_idx + 1) * duration, song_duration)
                    actual_duration = end_time - start_time
                    
                    # Only keep segments >= 75% of target duration
                    if actual_duration >= duration * 0.75:
                        segment = {
                            **song,  # Copy all original metadata
                            "id": f"{song.get('id', 'unknown')}_{seg_idx:03d}",
                            "segment_id": f"{song.get('id', 'unknown')}_seg_{duration}s_{seg_idx:03d}",
                            "parent_song_id": song.get("id", "unknown"),
                            "segment_start_time": start_time,
                            "segment_end_time": end_time,
                            "segment_duration": actual_duration,
                            "segment_index": seg_idx,
                            "total_segments_in_song": n_segments,
                            "target_segment_duration": duration,
                            "is_segment": True,
                            "is_full_song": False
                        }
                        segments.append(segment)
            
            segment_datasets[f"{duration}s_segments"] = {
                "segments": segments,
                "total_segments": len(segments),
                "average_duration": np.mean([s["segment_duration"] for s in segments]),
                "target_duration": duration,
                "songs_segmented": len([s for s in songs if s.get("duration", 0) >= duration * 0.75])
            }
            
            print(f"   Created {len(segments)} segments of {duration}s")
        
        return segment_datasets
    
    def create_training_configurations(self, split_data: Dict) -> Dict:
        """
        Create different training configurations for different training phases
        """
        print(f"⚙️  Creating training configurations...")
        
        configurations = {}
        
        # Configuration 1: Full songs for long-form evaluation
        print("   Creating full songs configuration...")
        full_songs_config = {
            "train": [s for s in split_data["train"] if s.get("duration", 0) >= 240],
            "val": [s for s in split_data["val"] if s.get("duration", 0) >= 240],
            "test": [s for s in split_data["test"] if s.get("duration", 0) >= 240],
            "description": "Full-length songs (4+ minutes) for long-form coherence evaluation",
            "min_duration": 240,
            "use_case": "Final model evaluation and structure assessment"
        }
        configurations["full_songs"] = full_songs_config
        
        # Configuration 2: High-quality only
        print("   Creating high-quality only configuration...")
        high_quality_filter = lambda x: (
            x.get("augmentation_metadata", {}).get("quality_category", "high") == "high" or
            not x.get("is_augmented", False)  # Original songs are high quality
        )
        
        high_quality_config = {
            "train": [s for s in split_data["train"] if high_quality_filter(s)],
            "val": [s for s in split_data["val"] if high_quality_filter(s)],
            "test": [s for s in split_data["test"] if high_quality_filter(s)],
            "description": "High-quality samples only for final training",
            "quality_threshold": "high",
            "use_case": "Fine-tuning and quality optimization"
        }
        configurations["high_quality_only"] = high_quality_config
        
        # Configuration 3: Original songs only (no augmentation)
        print("   Creating original songs only configuration...")
        original_only_config = {
            "train": [s for s in split_data["train"] if not s.get("is_augmented", False)],
            "val": [s for s in split_data["val"] if not s.get("is_augmented", False)],
            "test": [s for s in split_data["test"] if not s.get("is_augmented", False)],
            "description": "Original songs only (no augmentation)",
            "augmentation_level": "none",
            "use_case": "Baseline comparison and ablation studies"
        }
        configurations["original_only"] = original_only_config
        
        # Configuration 4: Balanced genre distribution
        print("   Creating balanced genre configuration...")
        balanced_config = self._create_balanced_genre_config(split_data)
        configurations["balanced_genres"] = balanced_config
        
        # Create segment configurations for each main split
        segment_configs = self._create_segment_configurations(split_data)
        configurations.update(segment_configs)
        
        self.stats["configurations_created"] = len(configurations)
        
        print(f"✅ Created {len(configurations)} training configurations")
        return configurations
    
    def _create_balanced_genre_config(self, split_data: Dict) -> Dict:
        """Create configuration with balanced genre distribution"""
        
        def balance_genres(songs: List[Dict], max_per_genre: int = 50) -> List[Dict]:
            genre_groups = defaultdict(list)
            for song in songs:
                genre = song.get("genre", "unknown")
                genre_groups[genre].append(song)
            
            balanced = []
            for genre, genre_songs in genre_groups.items():
                # Sample up to max_per_genre songs from this genre
                n_samples = min(len(genre_songs), max_per_genre)
                sampled = random.sample(genre_songs, n_samples)
                balanced.extend(sampled)
            
            return balanced
        
        return {
            "train": balance_genres(split_data["train"], max_per_genre=100),
            "val": balance_genres(split_data["val"], max_per_genre=20),
            "test": balance_genres(split_data["test"], max_per_genre=20),
            "description": "Balanced genre distribution to prevent genre bias",
            "balancing_method": "max_samples_per_genre",
            "use_case": "Training without genre imbalance"
        }
    
    def _create_segment_configurations(self, split_data: Dict) -> Dict:
        """Create configurations for different segment lengths"""
        
        segment_configs = {}
        
        # Create segmented versions of each split
        for duration in [30, 60, 120]:
            print(f"   Creating {duration}s segment configuration...")
            
            train_segments = self._create_segments_from_songs(split_data["train"], duration)
            val_segments = self._create_segments_from_songs(split_data["val"], duration)  
            test_segments = self._create_segments_from_songs(split_data["test"], duration)
            
            segment_configs[f"{duration}s_segments"] = {
                "train": train_segments,
                "val": val_segments,
                "test": test_segments,
                "description": f"{duration}-second segments for model training",
                "segment_duration": duration,
                "use_case": f"Initial training on {duration}s clips" if duration <= 30 else f"Progressive training on {duration}s clips"
            }
        
        return segment_configs
    
    def _create_segments_from_songs(self, songs: List[Dict], duration: int) -> List[Dict]:
        """Create segments from list of songs"""
        segments = []
        
        for song in songs:
            song_duration = song.get("duration", 275.0)
            n_segments = int(song_duration // duration)
            
            for seg_idx in range(n_segments):
                start_time = seg_idx * duration
                end_time = min((seg_idx + 1) * duration, song_duration)
                actual_duration = end_time - start_time
                
                # Only keep segments >= 75% of target duration
                if actual_duration >= duration * 0.75:
                    segment = {
                        **song,
                        "id": f"{song.get('id', 'unknown')}_seg{duration}s_{seg_idx:03d}",
                        "parent_song_id": song.get("id", "unknown"),
                        "segment_start_time": start_time,
                        "segment_end_time": end_time,
                        "segment_duration": actual_duration,
                        "segment_index": seg_idx,
                        "is_segment": True,
                        "target_segment_duration": duration
                    }
                    segments.append(segment)
        
        return segments
    
    def analyze_dataset_statistics(self, configurations: Dict) -> Dict:
        """Generate comprehensive statistics for all configurations"""
        
        print(f"📊 Analyzing dataset statistics...")
        
        statistics = {}
        
        for config_name, config_data in configurations.items():
            print(f"   Analyzing {config_name}...")
            
            train_data = config_data.get("train", [])
            val_data = config_data.get("val", [])
            test_data = config_data.get("test", [])
            all_data = train_data + val_data + test_data
            
            if not all_data:
                continue
            
            # Basic statistics
            stats = {
                "total_samples": len(all_data),
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "test_samples": len(test_data),
                "configuration_type": config_name,
                "description": config_data.get("description", "")
            }
            
            # Duration statistics
            durations = []
            for item in all_data:
                if item.get("is_segment", False):
                    durations.append(item.get("segment_duration", 0))
                else:
                    durations.append(item.get("duration", 0))
            
            if durations:
                stats["duration_stats"] = {
                    "mean": float(np.mean(durations)),
                    "median": float(np.median(durations)),
                    "std": float(np.std(durations)),
                    "min": float(np.min(durations)),
                    "max": float(np.max(durations)),
                    "total_hours": float(np.sum(durations) / 3600)
                }
            
            # Genre distribution
            genre_counts = defaultdict(int)
            for item in all_data:
                genre = item.get("genre", "unknown")
                genre_counts[genre] += 1
            stats["genre_distribution"] = dict(genre_counts)
            
            # Quality distribution
            quality_counts = defaultdict(int)
            for item in all_data:
                if item.get("is_augmented", False):
                    quality = item.get("augmentation_metadata", {}).get("quality_category", "medium")
                else:
                    quality = "original"
                quality_counts[quality] += 1
            stats["quality_distribution"] = dict(quality_counts)
            
            # Augmentation statistics
            augmented_count = sum(1 for item in all_data if item.get("is_augmented", False))
            stats["augmentation_stats"] = {
                "original_samples": len(all_data) - augmented_count,
                "augmented_samples": augmented_count,
                "augmentation_ratio": augmented_count / len(all_data) if all_data else 0
            }
            
            statistics[config_name] = stats
        
        return statistics
    
    def save_configurations(self, 
                          configurations: Dict,
                          statistics: Dict,
                          output_dir: str = "./outputs") -> Dict:
        """Save all configurations and statistics to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        print(f"💾 Saving configurations to {output_path}...")
        
        # Save each configuration
        for config_name, config_data in configurations.items():
            filename = f"mts_{config_name}_dataset.json"
            filepath = output_path / filename
            
            # Prepare data for JSON serialization
            serializable_config = {
                "description": config_data.get("description", ""),
                "configuration_metadata": {
                    key: value for key, value in config_data.items() 
                    if key not in ["train", "val", "test"]
                },
                "statistics": statistics.get(config_name, {}),
                "sample_counts": {
                    "train": len(config_data.get("train", [])),
                    "val": len(config_data.get("val", [])),
                    "test": len(config_data.get("test", []))
                },
                # Save first few samples as examples
                "sample_data": {
                    "train_examples": config_data.get("train", [])[:3],
                    "val_examples": config_data.get("val", [])[:2],
                    "test_examples": config_data.get("test", [])[:2]
                }
            }
            
            # Save configuration
            with open(filepath, 'w') as f:
                json.dump(serializable_config, f, indent=2, default=str)
            
            saved_files[config_name] = str(filepath)
        
        # Save comprehensive statistics
        stats_file = output_path / "dataset_statistics_summary.json"
        with open(stats_file, 'w') as f:
            json.dump({
                "overall_statistics": statistics,
                "processing_summary": self.stats,
                "generation_timestamp": pd.Timestamp.now().isoformat()
            }, f, indent=2, default=str)
        
        saved_files["statistics_summary"] = str(stats_file)
        
        print(f"✅ Saved {len(configurations)} configurations and statistics")
        
        return saved_files

def main():
    """Test the data organization system"""
    
    # Create test dataset
    test_songs = []
    genres = ["Pop", "Rock", "Classical", "Jazz", "Electronic"]
    
    for i in range(100):  # 100 test songs
        song = {
            "id": f"test_song_{i:03d}",
            "title": f"Test Song {i}",
            "genre": random.choice(genres),
            "duration": random.uniform(180, 300),
            "is_augmented": random.random() < 0.7,  # 70% augmented
            "augmentation_metadata": {
                "quality_category": random.choice(["high", "medium", "low"])
            } if random.random() < 0.7 else {}
        }
        test_songs.append(song)
    
    print("🎵 Testing MTS Data Organization System...")
    
    # Initialize organizer
    organizer = MTSDataOrganizer(
        train_ratio=0.8,
        val_ratio=0.1, 
        test_ratio=0.1,
        random_state=42
    )
    
    # Perform stratified split
    split_data = organizer.stratified_split_by_genre(test_songs)
    
    # Create training configurations
    configurations = organizer.create_training_configurations(split_data)
    
    # Analyze statistics
    statistics = organizer.analyze_dataset_statistics(configurations)
    
    # Save results
    saved_files = organizer.save_configurations(configurations, statistics, "./test_outputs")
    
    print(f"✅ Test complete!")
    print(f"   Configurations created: {len(configurations)}")
    print(f"   Files saved: {len(saved_files)}")
    
    # Print summary
    print(f"\n📊 Configuration Summary:")
    for config_name, stats in statistics.items():
        print(f"   {config_name}: {stats['total_samples']} samples")

if __name__ == "__main__":
    main()
