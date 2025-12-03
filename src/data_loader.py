"""
MTS Data Loader - Handles CCMusic dataset loading and preprocessing
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import glob

# Optional dependencies with fallbacks
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️  librosa not available. Audio loading will be limited.")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    print("⚠️  soundfile not available. Using numpy for audio I/O.")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️  datasets not available. CCMusic/HuggingFace datasets disabled.")

class MTSDataLoader:
    """
    Comprehensive data loader for MTS project
    Handles CCMusic dataset loading, format conversion, and initial processing
    """
    
    def __init__(self, 
                 cache_dir: str = "./cache",
                 data_dir: str = "./data",
                 target_sr: int = 22050):
        """
        Initialize the data loader
        
        Args:
            cache_dir: Directory for caching datasets
            data_dir: Directory for processed data
            target_sr: Target sample rate for audio
        """
        self.cache_dir = Path(cache_dir)
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        
        # Create directories
        self.cache_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "augmented").mkdir(exist_ok=True)
        
        # Genre mapping from CCMusic documentation
        self.genre_mapping = self._create_genre_mapping()
        
    def _create_genre_mapping(self) -> Dict:
        """Create genre ID mapping"""
        return {
            "Classical": {
                "Symphony": 0,
                "Opera": 1,
                "Solo": 2,
                "Chamber": 3
            },
            "Popular": {
                "Pop": 4,
                "Pop_vocal_ballad": 5,
                "Adult_contemporary": 6,
                "Teen_pop": 7,
                "Dance_and_house": 8,
                "Contemporary_dance_pop": 9,
                "Dance_pop": 10,
                "Indie": 11,
                "Classic_indie_pop": 12,
                "Chamber_cabaret_and_art_pop": 13,
                "Soul_or_RnB": 14,
                "Rock": 15,
                "Adult_alternative_rock": 16,
                "Uplifting_anthemic_rock": 17,
                "Soft_rock": 18,
                "Acoustic_pop": 19
            }
        }
    
    def load_ccmusic_dataset(self, 
                           dataset_name: str = "ccmusic-database/music_genre",
                           split: str = "train",
                           use_auth_token: Optional[str] = None) -> Dict:
        """
        Load CCMusic dataset from Hugging Face
        
        Args:
            dataset_name: Hugging Face dataset identifier
            split: Dataset split to load
            use_auth_token: Authentication token if required
            
        Returns:
            Dictionary containing dataset information
        """
        try:
            print(f"Loading CCMusic dataset: {dataset_name}")
            
            # Load dataset
            if use_auth_token:
                dataset = load_dataset(
                    dataset_name, 
                    use_auth_token=use_auth_token,
                    cache_dir=str(self.cache_dir)
                )
            else:
                dataset = load_dataset(
                    dataset_name,
                    cache_dir=str(self.cache_dir)
                )
            
            # Get available splits
            available_splits = list(dataset.keys())
            print(f"Available splits: {available_splits}")
            
            if split not in available_splits:
                split = available_splits[0]
                print(f"Using split: {split}")
            
            data = dataset[split]
            
            # Analyze dataset structure
            sample = data[0] if len(data) > 0 else {}
            
            dataset_info = {
                "dataset": data,
                "size": len(data),
                "sample_keys": list(sample.keys()) if sample else [],
                "splits": available_splits,
                "current_split": split
            }
            
            print(f"✅ Dataset loaded: {len(data)} samples")
            print(f"Sample features: {list(sample.keys()) if sample else 'No samples'}")
            
            return dataset_info
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return self._create_fallback_dataset()
    
    def _create_fallback_dataset(self) -> Dict:
        """
        Create simulated dataset for development/testing
        """
        print("📝 Creating simulated CCMusic dataset for development...")
        
        # Create simulated dataset matching CCMusic structure
        songs = []
        genres = list(self.genre_mapping["Popular"].keys()) + list(self.genre_mapping["Classical"].keys())
        
        for i in tqdm(range(1700), desc="Generating simulated songs"):
            song = {
                "id": f"ccmusic_{i:04d}",
                "title": f"Song_{i:04d}",
                "artist": f"Artist_{i % 100}",
                "genre": np.random.choice(genres),
                "duration": np.random.uniform(270, 300),  # 4.5-5 minutes
                "sample_rate": self.target_sr,
                "language": np.random.choice(["Chinese", "English", "Korean", "Japanese"], p=[0.69, 0.29, 0.01, 0.01]),
                # Simulate spectrograms (in real implementation, these would be actual spectrograms)
                "mel_spectrogram_path": f"spectrograms/mel/{i:04d}.npy",
                "cqt_spectrogram_path": f"spectrograms/cqt/{i:04d}.npy",
                "chroma_path": f"spectrograms/chroma/{i:04d}.npy",
                # Audio path (would be actual audio files)
                "audio_path": f"audio/{i:04d}.wav",
                "has_audio": False,  # Simulate copyright restrictions
                "has_spectrograms": True
            }
            songs.append(song)
        
        class SimulatedDataset:
            def __init__(self, songs):
                self.songs = songs
            
            def __len__(self):
                return len(self.songs)
            
            def __getitem__(self, idx):
                return self.songs[idx]
        
        dataset = SimulatedDataset(songs)
        
        return {
            "dataset": dataset,
            "size": len(songs),
            "sample_keys": list(songs[0].keys()),
            "splits": ["train"],
            "current_split": "train",
            "is_simulated": True
        }

    def load_fma_dataset(self,
                         audio_dir: str,
                         metadata_path: Optional[str] = None,
                         max_files: Optional[int] = 500,
                         clip_duration: Optional[float] = 30.0) -> Dict:
        """
        Load FMA dataset (pre-downloaded) with real audio.

        Args:
            audio_dir: Path to FMA audio folder (e.g., ./fma_data/fma_small)
            metadata_path: Path to tracks.csv (optional; used for genre/title)
            max_files: Limit number of files to load (controls memory)
            clip_duration: Seconds to load per track (None for full)

        Returns:
            Dataset info dict with preprocessed songs containing audio arrays.
        """
        audio_dir = Path(audio_dir)
        if not audio_dir.exists():
            raise FileNotFoundError(f"FMA audio directory not found: {audio_dir}")

        genre_lookup = {}
        title_lookup = {}
        artist_lookup = {}
        if metadata_path and Path(metadata_path).exists():
            try:
                tracks_df = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
                # The subset column tells which split (small/medium/etc.)
                for tid, row in tracks_df.iterrows():
                    try:
                        genre_lookup[tid] = row["track"]["genre_top"]
                        title_lookup[tid] = row["track"]["title"]
                        artist_lookup[tid] = row["artist"]["name"]
                    except Exception:
                        continue
            except Exception as e:
                print(f"⚠️  Unable to read FMA metadata ({e}); proceeding without detailed labels.")

        # Collect audio file paths
        files = sorted(glob.glob(str(audio_dir / "**" / "*.mp3"), recursive=True))
        if max_files:
            files = files[:max_files]

        songs = []
        print(f"🎵 Loading FMA audio from {audio_dir} ({len(files)} files)...")
        for idx, filepath in enumerate(tqdm(files, desc="Loading FMA audio")):
            try:
                tid = int(Path(filepath).stem)
                audio, sr = librosa.load(filepath, sr=self.target_sr, duration=clip_duration)
                songs.append({
                    "id": f"fma_{tid:06d}",
                    "title": title_lookup.get(tid, f"FMA_{tid:06d}"),
                    "artist": artist_lookup.get(tid, "Unknown"),
                    "genre": genre_lookup.get(tid, "Unknown"),
                    "duration": len(audio) / self.target_sr,
                    "sample_rate": self.target_sr,
                    "language": "instrumental",
                    "original_index": idx,
                    "audio": audio,
                    "has_audio": True,
                    "data_source": "fma",
                    "audio_path": filepath
                })
            except Exception as e:
                print(f"⚠️  Error loading {filepath}: {e}")
                continue

        dataset = {
            "preprocessed_songs": songs,
            "size": len(songs),
            "sample_keys": list(songs[0].keys()) if songs else [],
            "splits": ["train"],
            "current_split": "train",
            "is_fma": True,
            "metadata_used": bool(genre_lookup)
        }
        print(f"✅ Loaded {len(songs)} FMA tracks with audio.")
        return dataset
    
    def extract_features_from_spectrograms(self, 
                                         dataset_info: Dict,
                                         feature_type: str = "mel") -> List[Dict]:
        """
        Extract features from spectrograms when audio is not available
        
        Args:
            dataset_info: Dataset information from load_ccmusic_dataset
            feature_type: Type of features to extract ("mel", "cqt", "chroma")
            
        Returns:
            List of songs with extracted features
        """
        dataset = dataset_info["dataset"]
        processed_songs = []
        
        print(f"🎵 Processing {len(dataset)} songs - extracting {feature_type} features...")
        
        for i, sample in enumerate(tqdm(dataset)):
            try:
                song_data = {
                    "id": sample.get("id", f"song_{i:04d}"),
                    "title": sample.get("title", f"Unknown_Title_{i}"),
                    "artist": sample.get("artist", "Unknown_Artist"),
                    "genre": sample.get("genre", "Unknown"),
                    "duration": sample.get("duration", 275.0),
                    "sample_rate": self.target_sr,
                    "language": sample.get("language", "Unknown"),
                    "original_index": i
                }
                
                # In real implementation, load actual spectrograms
                if feature_type == "mel" and sample.get("has_spectrograms", False):
                    # Simulate mel spectrogram features
                    # In real code: features = np.load(sample["mel_spectrogram_path"])
                    features = self._simulate_mel_features(song_data["duration"])
                    song_data["mel_features"] = features
                    song_data["feature_shape"] = features.shape
                
                elif feature_type == "audio" and sample.get("has_audio", False):
                    # Load actual audio file
                    audio_path = sample.get("audio_path")
                    if audio_path and os.path.exists(audio_path):
                        audio, sr = librosa.load(audio_path, sr=self.target_sr)
                        song_data["audio"] = audio
                        song_data["audio_length"] = len(audio)
                        song_data["actual_duration"] = len(audio) / sr
                
                # Add metadata
                song_data["data_source"] = "ccmusic"
                song_data["processing_timestamp"] = pd.Timestamp.now().isoformat()
                song_data["has_features"] = feature_type in ["mel", "cqt", "chroma"]
                song_data["has_audio"] = feature_type == "audio"
                
                processed_songs.append(song_data)
                
            except Exception as e:
                print(f"⚠️  Error processing sample {i}: {e}")
                continue
        
        print(f"✅ Processed {len(processed_songs)} songs successfully")
        return processed_songs
    
    def _simulate_mel_features(self, duration: float) -> np.ndarray:
        """
        Simulate mel spectrogram features for development
        In production, this would load actual spectrograms
        """
        # Simulate mel spectrogram: (n_mels, time_frames)
        n_mels = 128
        hop_length = 512
        time_frames = int(duration * self.target_sr / hop_length)
        
        # Create realistic-looking mel spectrogram simulation
        features = np.random.lognormal(mean=-2, sigma=1, size=(n_mels, time_frames))
        features = np.clip(features, 0, 10)  # Reasonable dB range
        
        return features.astype(np.float32)
    
    def save_processed_dataset(self, 
                             processed_songs: List[Dict],
                             filename: str = "ccmusic_processed.json") -> str:
        """Save processed dataset to file"""
        
        output_path = self.data_dir / "processed" / filename
        
        # Prepare data for JSON serialization
        serializable_songs = []
        for song in processed_songs:
            serializable_song = song.copy()
            
            # Convert numpy arrays to lists for JSON
            for key, value in serializable_song.items():
                if isinstance(value, np.ndarray):
                    serializable_song[key] = {
                        "data": value.tolist(),
                        "shape": value.shape,
                        "dtype": str(value.dtype)
                    }
            
            serializable_songs.append(serializable_song)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump({
                "songs": serializable_songs,
                "metadata": {
                    "total_songs": len(processed_songs),
                    "processing_date": pd.Timestamp.now().isoformat(),
                    "target_sample_rate": self.target_sr,
                    "data_source": "ccmusic-database"
                }
            }, f, indent=2)
        
        print(f"💾 Saved processed dataset: {output_path}")
        return str(output_path)
    
    def load_processed_dataset(self, filepath: str) -> List[Dict]:
        """Load previously processed dataset"""
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        songs = data["songs"]
        
        # Convert data back to numpy arrays
        for song in songs:
            for key, value in song.items():
                if isinstance(value, dict) and "data" in value:
                    song[key] = np.array(value["data"], dtype=value.get("dtype", "float32"))
        
        print(f"📂 Loaded {len(songs)} songs from {filepath}")
        return songs

def main():
    """Test the data loader"""
    
    # Initialize data loader
    loader = MTSDataLoader(
        cache_dir="./cache",
        data_dir="./data", 
        target_sr=22050
    )
    
    # Load dataset
    print("🚀 Starting CCMusic dataset loading...")
    dataset_info = loader.load_ccmusic_dataset()
    
    # Process features
    processed_songs = loader.extract_features_from_spectrograms(
        dataset_info, 
        feature_type="mel"
    )
    
    # Save processed data
    output_file = loader.save_processed_dataset(processed_songs)
    
    # Test loading
    loaded_songs = loader.load_processed_dataset(output_file)
    
    print(f"🎉 Data loading complete!")
    print(f"Total songs: {len(loaded_songs)}")
    print(f"Sample song keys: {list(loaded_songs[0].keys()) if loaded_songs else 'None'}")

if __name__ == "__main__":
    main()
