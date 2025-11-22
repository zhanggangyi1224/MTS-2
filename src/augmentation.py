"""
MTS Audio Augmentation - Comprehensive audio data augmentation system
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
import json
from tqdm import tqdm
import warnings

# Optional: high-quality audio processing libraries
try:
    import pyrubberband as pyrb
    HAS_PYRUBBERBAND = True
except ImportError:
    HAS_PYRUBBERBAND = False
    print("⚠️  pyrubberband not available. Using librosa for pitch/tempo changes.")

try:
    from pedalboard import Pedalboard, Reverb, LowShelfFilter, HighShelfFilter, PeakFilter
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False
    print("⚠️  pedalboard not available. Using basic audio effects.")

class MTSAudioAugmentation:
    """
    Comprehensive audio augmentation system for MTS project
    Implements all augmentation techniques specified in the project plan
    """
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 augmentation_factor: int = 3,
                 preserve_original: bool = True):
        """
        Initialize augmentation system
        
        Args:
            sample_rate: Target sample rate for audio
            augmentation_factor: Number of augmented versions per original
            preserve_original: Whether to keep original alongside augmented versions
        """
        self.sample_rate = sample_rate
        self.augmentation_factor = augmentation_factor
        self.preserve_original = preserve_original
        
        # Augmentation parameters from project plan
        self.config = self._create_augmentation_config()
        
        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "total_augmented": 0,
            "techniques_applied": {
                "pitch_shift": 0,
                "tempo_scale": 0,
                "noise_addition": 0,
                "eq_filter": 0,
                "reverb": 0
            },
            "quality_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
    
    def _create_augmentation_config(self) -> Dict:
        """Create augmentation configuration"""
        return {
            "pitch_shift": {
                "enabled": True,
                "range_semitones": (-2, 2),  # ±2 semitones as specified
                "probability": 0.8,
                "preserve_tempo": True
            },
            "tempo_scale": {
                "enabled": True,
                "range_factor": (0.9, 1.1),  # ±10% as specified
                "probability": 0.8,
                "preserve_pitch": True
            },
            "noise_addition": {
                "enabled": True,
                "snr_db_range": (20, 40),  # 20-40 dB SNR
                "noise_types": ["gaussian", "uniform", "pink"],
                "probability": 0.4
            },
            "eq_filter": {
                "enabled": True,
                "types": ["low_shelf", "high_shelf", "peaking"],
                "frequency_ranges": {
                    "low": (60, 250),    # Bass
                    "mid": (250, 4000),  # Midrange  
                    "high": (4000, 8000) # Treble
                },
                "gain_range_db": (-3, 3),
                "q_range": (0.5, 2.0),
                "probability": 0.3
            },
            "reverb": {
                "enabled": True,
                "room_sizes": ["small", "medium", "large"],
                "wet_mix_range": (0.1, 0.4),  # 10-40% wet
                "decay_time_range": (0.5, 3.0),
                "probability": 0.25
            }
        }
    
    def pitch_shift_audio(self, 
                         audio: np.ndarray, 
                         semitones: float) -> np.ndarray:
        """
        Pitch shift audio by specified semitones
        
        Args:
            audio: Input audio signal
            semitones: Number of semitones to shift (+ for up, - for down)
            
        Returns:
            Pitch-shifted audio
        """
        if HAS_PYRUBBERBAND:
            # High-quality pitch shifting
            return pyrb.pitch_shift(audio, self.sample_rate, semitones)
        else:
            # Fallback to librosa
            return librosa.effects.pitch_shift(
                audio, 
                sr=self.sample_rate, 
                n_steps=semitones
            )
    
    def tempo_scale_audio(self, 
                         audio: np.ndarray, 
                         scale_factor: float) -> np.ndarray:
        """
        Scale tempo without changing pitch
        
        Args:
            audio: Input audio signal
            scale_factor: Tempo scaling factor (>1 faster, <1 slower)
            
        Returns:
            Tempo-scaled audio
        """
        if HAS_PYRUBBERBAND:
            # High-quality time stretching
            return pyrb.time_stretch(audio, self.sample_rate, scale_factor)
        else:
            # Fallback to librosa
            return librosa.effects.time_stretch(audio, rate=scale_factor)
    
    def add_noise(self, 
                  audio: np.ndarray, 
                  snr_db: float, 
                  noise_type: str = "gaussian") -> np.ndarray:
        """
        Add noise to audio signal
        
        Args:
            audio: Input audio signal
            snr_db: Signal-to-noise ratio in dB
            noise_type: Type of noise ("gaussian", "uniform", "pink")
            
        Returns:
            Audio with added noise
        """
        # Calculate noise power
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate noise
        if noise_type == "gaussian":
            noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        elif noise_type == "uniform":
            noise_std = np.sqrt(noise_power)
            noise = np.random.uniform(-noise_std * np.sqrt(3), 
                                    noise_std * np.sqrt(3), 
                                    len(audio))
        elif noise_type == "pink":
            # Generate pink noise (1/f noise)
            noise = self._generate_pink_noise(len(audio))
            noise = noise * np.sqrt(noise_power) / np.std(noise)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return audio + noise
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink noise (1/f noise)"""
        # Simple pink noise generation
        white = np.random.randn(length)
        
        # Apply 1/f filter in frequency domain
        fft = np.fft.fft(white)
        freqs = np.fft.fftfreq(length)
        
        # Avoid division by zero
        freqs[0] = 1e-10
        
        # Apply 1/f scaling
        fft = fft / np.sqrt(np.abs(freqs))
        
        # Convert back to time domain
        pink = np.real(np.fft.ifft(fft))
        return pink
    
    def apply_eq_filter(self, 
                       audio: np.ndarray,
                       filter_type: str,
                       frequency: float,
                       gain_db: float,
                       q_factor: float = 1.0) -> np.ndarray:
        """
        Apply EQ filter to audio
        
        Args:
            audio: Input audio signal
            filter_type: Type of filter ("low_shelf", "high_shelf", "peaking")
            frequency: Center frequency for filter
            gain_db: Gain in dB
            q_factor: Quality factor for peaking filter
            
        Returns:
            Filtered audio
        """
        if HAS_PEDALBOARD:
            # High-quality EQ using pedalboard
            board = Pedalboard([])
            
            if filter_type == "low_shelf":
                board.append(LowShelfFilter(cutoff_frequency_hz=frequency, gain_db=gain_db))
            elif filter_type == "high_shelf":
                board.append(HighShelfFilter(cutoff_frequency_hz=frequency, gain_db=gain_db))
            elif filter_type == "peaking":
                board.append(PeakFilter(
                    cutoff_frequency_hz=frequency, 
                    gain_db=gain_db, 
                    q=q_factor
                ))
            
            return board(audio, sample_rate=self.sample_rate)
        
        else:
            # Fallback: simple frequency domain filtering
            return self._apply_simple_eq(audio, filter_type, frequency, gain_db)
    
    def _apply_simple_eq(self, 
                        audio: np.ndarray,
                        filter_type: str,
                        frequency: float,
                        gain_db: float) -> np.ndarray:
        """Simple EQ implementation using scipy"""
        from scipy import signal
        
        # Convert gain to linear scale
        gain_linear = 10 ** (gain_db / 20)
        
        if filter_type == "low_shelf":
            # Low shelf filter
            sos = signal.butter(2, frequency / (self.sample_rate / 2), 
                               btype='low', output='sos')
            filtered = signal.sosfilt(sos, audio)
            return audio + (filtered - audio) * (gain_linear - 1)
        
        elif filter_type == "high_shelf":
            # High shelf filter  
            sos = signal.butter(2, frequency / (self.sample_rate / 2), 
                               btype='high', output='sos')
            filtered = signal.sosfilt(sos, audio)
            return audio + (filtered - audio) * (gain_linear - 1)
        
        elif filter_type == "peaking":
            # Peaking filter (bandpass)
            low_freq = frequency * 0.8
            high_freq = frequency * 1.2
            sos = signal.butter(2, [low_freq / (self.sample_rate / 2), 
                                   high_freq / (self.sample_rate / 2)], 
                               btype='band', output='sos')
            filtered = signal.sosfilt(sos, audio)
            return audio + (filtered - audio) * (gain_linear - 1)
        
        return audio
    
    def apply_reverb(self, 
                    audio: np.ndarray,
                    room_size: str,
                    wet_mix: float,
                    decay_time: float) -> np.ndarray:
        """
        Apply reverb effect to audio
        
        Args:
            audio: Input audio signal
            room_size: Size of room ("small", "medium", "large")
            wet_mix: Wet/dry mix ratio (0.0 to 1.0)
            decay_time: Reverb decay time in seconds
            
        Returns:
            Audio with reverb applied
        """
        if HAS_PEDALBOARD:
            # High-quality reverb using pedalboard
            room_size_map = {"small": 0.3, "medium": 0.6, "large": 0.9}
            
            reverb = Reverb(
                room_size=room_size_map.get(room_size, 0.5),
                wet_level=wet_mix,
                dry_level=1.0 - wet_mix
            )
            
            return reverb(audio, sample_rate=self.sample_rate)
        
        else:
            # Simple convolution reverb
            return self._apply_simple_reverb(audio, room_size, wet_mix, decay_time)
    
    def _apply_simple_reverb(self, 
                           audio: np.ndarray,
                           room_size: str,
                           wet_mix: float,
                           decay_time: float) -> np.ndarray:
        """Simple reverb implementation"""
        
        # Create impulse response
        room_size_map = {"small": 0.05, "medium": 0.15, "large": 0.3}
        ir_length = int(self.sample_rate * room_size_map.get(room_size, 0.15))
        
        # Generate simple exponentially decaying impulse response
        t = np.arange(ir_length) / self.sample_rate
        ir = np.exp(-t / decay_time) * np.random.normal(0, 0.1, ir_length)
        
        # Apply convolution
        reverb_audio = np.convolve(audio, ir, mode='same')
        
        # Mix wet and dry signals
        return (1 - wet_mix) * audio + wet_mix * reverb_audio
    
    def generate_augmentation_plan(self, song_id: str) -> List[Dict]:
        """
        Generate augmentation plan for a single song
        
        Args:
            song_id: Unique identifier for the song
            
        Returns:
            List of augmentation configurations
        """
        plans = []
        
        for i in range(self.augmentation_factor):
            plan = {
                "original_id": song_id,
                "augmented_id": f"{song_id}_aug_{i+1:02d}",
                "transformations": []
            }
            
            # Decide which augmentations to apply
            config = self.config
            
            # Pitch shift
            if (config["pitch_shift"]["enabled"] and 
                random.random() < config["pitch_shift"]["probability"]):
                
                semitones = random.uniform(*config["pitch_shift"]["range_semitones"])
                plan["transformations"].append({
                    "type": "pitch_shift",
                    "parameters": {
                        "semitones": semitones,
                        "preserve_tempo": config["pitch_shift"]["preserve_tempo"]
                    }
                })
            
            # Tempo scaling
            if (config["tempo_scale"]["enabled"] and 
                random.random() < config["tempo_scale"]["probability"]):
                
                scale_factor = random.uniform(*config["tempo_scale"]["range_factor"])
                plan["transformations"].append({
                    "type": "tempo_scale",
                    "parameters": {
                        "scale_factor": scale_factor,
                        "preserve_pitch": config["tempo_scale"]["preserve_pitch"]
                    }
                })
            
            # Noise addition
            if (config["noise_addition"]["enabled"] and 
                random.random() < config["noise_addition"]["probability"]):
                
                snr_db = random.uniform(*config["noise_addition"]["snr_db_range"])
                noise_type = random.choice(config["noise_addition"]["noise_types"])
                plan["transformations"].append({
                    "type": "noise_addition",
                    "parameters": {
                        "snr_db": snr_db,
                        "noise_type": noise_type
                    }
                })
            
            # EQ filter
            if (config["eq_filter"]["enabled"] and 
                random.random() < config["eq_filter"]["probability"]):
                
                filter_type = random.choice(config["eq_filter"]["types"])
                gain_db = random.uniform(*config["eq_filter"]["gain_range_db"])
                q_factor = random.uniform(*config["eq_filter"]["q_range"])
                
                # Choose appropriate frequency based on filter type
                if filter_type == "low_shelf":
                    freq_range = config["eq_filter"]["frequency_ranges"]["low"]
                elif filter_type == "high_shelf":
                    freq_range = config["eq_filter"]["frequency_ranges"]["high"]
                else:  # peaking
                    freq_range = config["eq_filter"]["frequency_ranges"]["mid"]
                
                frequency = random.uniform(*freq_range)
                
                plan["transformations"].append({
                    "type": "eq_filter",
                    "parameters": {
                        "filter_type": filter_type,
                        "frequency": frequency,
                        "gain_db": gain_db,
                        "q_factor": q_factor
                    }
                })
            
            # Reverb
            if (config["reverb"]["enabled"] and 
                random.random() < config["reverb"]["probability"]):
                
                room_size = random.choice(config["reverb"]["room_sizes"])
                wet_mix = random.uniform(*config["reverb"]["wet_mix_range"])
                decay_time = random.uniform(*config["reverb"]["decay_time_range"])
                
                plan["transformations"].append({
                    "type": "reverb",
                    "parameters": {
                        "room_size": room_size,
                        "wet_mix": wet_mix,
                        "decay_time": decay_time
                    }
                })
            
            plans.append(plan)
        
        return plans
    
    def apply_augmentation_plan(self, 
                              audio: np.ndarray,
                              plan: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Apply augmentation plan to audio
        
        Args:
            audio: Input audio signal
            plan: Augmentation plan from generate_augmentation_plan
            
        Returns:
            Tuple of (augmented_audio, metadata)
        """
        augmented_audio = audio.copy()
        applied_transforms = []
        quality_score = 1.0  # Start with perfect quality
        
        for transform in plan["transformations"]:
            transform_type = transform["type"]
            params = transform["parameters"]
            
            try:
                if transform_type == "pitch_shift":
                    augmented_audio = self.pitch_shift_audio(
                        augmented_audio, 
                        params["semitones"]
                    )
                    quality_impact = abs(params["semitones"]) * 0.1
                    self.stats["techniques_applied"]["pitch_shift"] += 1
                
                elif transform_type == "tempo_scale":
                    augmented_audio = self.tempo_scale_audio(
                        augmented_audio,
                        params["scale_factor"]
                    )
                    quality_impact = abs(params["scale_factor"] - 1.0) * 0.5
                    self.stats["techniques_applied"]["tempo_scale"] += 1
                
                elif transform_type == "noise_addition":
                    augmented_audio = self.add_noise(
                        augmented_audio,
                        params["snr_db"],
                        params["noise_type"]
                    )
                    quality_impact = (40 - params["snr_db"]) * 0.02
                    self.stats["techniques_applied"]["noise_addition"] += 1
                
                elif transform_type == "eq_filter":
                    augmented_audio = self.apply_eq_filter(
                        augmented_audio,
                        params["filter_type"],
                        params["frequency"],
                        params["gain_db"],
                        params.get("q_factor", 1.0)
                    )
                    quality_impact = abs(params["gain_db"]) * 0.05
                    self.stats["techniques_applied"]["eq_filter"] += 1
                
                elif transform_type == "reverb":
                    augmented_audio = self.apply_reverb(
                        augmented_audio,
                        params["room_size"],
                        params["wet_mix"],
                        params["decay_time"]
                    )
                    quality_impact = params["wet_mix"] * 0.3
                    self.stats["techniques_applied"]["reverb"] += 1
                
                quality_score -= quality_impact
                applied_transforms.append({
                    **transform,
                    "quality_impact": quality_impact,
                    "success": True
                })
                
            except Exception as e:
                print(f"⚠️  Error applying {transform_type}: {e}")
                applied_transforms.append({
                    **transform,
                    "error": str(e),
                    "success": False
                })
        
        # Determine quality category
        if quality_score > 0.8:
            quality_category = "high"
        elif quality_score > 0.5:
            quality_category = "medium"
        else:
            quality_category = "low"
        
        self.stats["quality_distribution"][quality_category] += 1
        
        metadata = {
            "original_id": plan["original_id"],
            "augmented_id": plan["augmented_id"],
            "applied_transforms": applied_transforms,
            "quality_score": quality_score,
            "quality_category": quality_category,
            "final_duration": len(augmented_audio) / self.sample_rate,
            "processing_successful": True
        }
        
        return augmented_audio, metadata
    
    def augment_dataset(self, 
                       songs: List[Dict],
                       output_dir: str,
                       save_audio: bool = False) -> List[Dict]:
        """
        Augment entire dataset
        
        Args:
            songs: List of song dictionaries with audio data
            output_dir: Directory to save augmented data
            save_audio: Whether to save audio files to disk
            
        Returns:
            List of augmented song metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if save_audio:
            (output_path / "audio").mkdir(exist_ok=True)
        
        all_augmented = []
        
        print(f"🎵 Augmenting {len(songs)} songs...")
        
        for song in tqdm(songs, desc="Processing songs"):
            try:
                # Generate augmentation plans
                plans = self.generate_augmentation_plan(song["id"])
                
                # Get audio data (simulated for development)
                if "audio" in song:
                    audio = song["audio"]
                else:
                    # Simulate audio from features or create synthetic audio
                    audio = self._simulate_audio_from_song(song)
                
                # Apply each augmentation plan
                for plan in plans:
                    augmented_audio, metadata = self.apply_augmentation_plan(audio, plan)
                    
                    # Create augmented song entry
                    augmented_song = {
                        **song,  # Copy original metadata
                        "id": plan["augmented_id"],
                        "original_id": plan["original_id"],
                        "augmentation_metadata": metadata,
                        "is_augmented": True,
                        # Do not keep raw audio in memory when not explicitly saving to disk
                        "augmented_audio": None if not save_audio else None,
                        "duration": metadata["final_duration"]
                    }
                    
                    # Save audio file if requested
                    if save_audio:
                        audio_filename = f"{plan['augmented_id']}.wav"
                        audio_path = output_path / "audio" / audio_filename
                        sf.write(str(audio_path), augmented_audio, self.sample_rate)
                        augmented_song["audio_file_path"] = str(audio_path)
                    
                    all_augmented.append(augmented_song)
                    self.stats["total_augmented"] += 1
                
                self.stats["total_processed"] += 1
                
            except Exception as e:
                print(f"❌ Error processing song {song.get('id', 'unknown')}: {e}")
                continue
        
        # Save augmentation manifest
        manifest = {
            "augmented_songs": all_augmented,
            "statistics": self.stats,
            "configuration": self.config,
            "processing_info": {
                "total_original_songs": len(songs),
                "total_augmented_songs": len(all_augmented),
                "augmentation_factor": len(all_augmented) / len(songs) if songs else 0,
                "output_directory": str(output_path),
                "audio_files_saved": save_audio
            }
        }
        
        manifest_path = output_path / "augmentation_manifest.json"
        with open(manifest_path, 'w') as f:
            # Prepare for JSON serialization
            serializable_manifest = self._prepare_for_json(manifest)
            json.dump(serializable_manifest, f, indent=2)
        
        print(f"✅ Augmentation complete!")
        print(f"   Original songs: {len(songs)}")
        print(f"   Augmented songs: {len(all_augmented)}")
        print(f"   Expansion factor: {len(all_augmented) / len(songs):.1f}x")
        print(f"   Manifest saved: {manifest_path}")
        
        return all_augmented
    
    def _simulate_audio_from_song(self, song: Dict) -> np.ndarray:
        """
        Simulate audio signal from song metadata for development
        In production, this would load actual audio files
        """
        duration = song.get("duration", 275.0)
        length = int(duration * self.sample_rate)
        
        # Generate synthetic audio with some musical characteristics
        t = np.linspace(0, duration, length)
        
        # Create a simple musical signal with harmonics
        fundamental = 440 * (2 ** ((random.randint(-12, 12)) / 12))  # Random note
        audio = (0.3 * np.sin(2 * np.pi * fundamental * t) +
                0.2 * np.sin(2 * np.pi * fundamental * 2 * t) +
                0.1 * np.sin(2 * np.pi * fundamental * 3 * t))
        
        # Add some envelope and variation
        envelope = np.exp(-t / (duration * 0.8))  # Decay envelope
        audio *= envelope
        
        # Add some noise for realism
        audio += np.random.normal(0, 0.02, length)
        
        return audio.astype(np.float32)
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization by converting numpy arrays"""
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return {
                "data": obj.tolist(),
                "shape": obj.shape,
                "dtype": str(obj.dtype)
            }
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

def main():
    """Test the augmentation system"""
    
    # Create test song
    test_song = {
        "id": "test_001",
        "title": "Test Song",
        "genre": "Pop",
        "duration": 30.0  # Short for testing
    }
    
    # Initialize augmentation system
    augmenter = MTSAudioAugmentation(
        sample_rate=22050,
        augmentation_factor=2,  # Create 2 augmented versions
        preserve_original=True
    )
    
    print("🎵 Testing MTS Audio Augmentation System...")
    
    # Test augmentation
    augmented_songs = augmenter.augment_dataset(
        songs=[test_song],
        output_dir="./test_augmentation",
        save_audio=True
    )
    
    print(f"✅ Test complete! Generated {len(augmented_songs)} augmented versions")
    
    # Print statistics
    print("\n📊 Augmentation Statistics:")
    for technique, count in augmenter.stats["techniques_applied"].items():
        print(f"   {technique}: {count}")

if __name__ == "__main__":
    main()
