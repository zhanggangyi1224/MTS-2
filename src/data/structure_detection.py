"""
MTS Real Structure Detection - Audio-based structure analysis

Unlike the simulated structure in the original pipeline, this module
performs actual audio analysis to detect:
- Beat and tempo
- Section boundaries
- Structural segments (verse, chorus, etc.)

Uses librosa for beat tracking and madmom for structure detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum
import warnings
warnings.filterwarnings('ignore')

# Required audio library
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️  librosa not available. Structure detection will be simulated.")

# Optional advanced libraries
try:
    import madmom
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    HAS_MADMOM = True
except ImportError:
    HAS_MADMOM = False
    print("⚠️  madmom not available, using librosa-only analysis")

try:
    from msaf.algorithms import interface as msaf_interface
    HAS_MSAF = True
except ImportError:
    HAS_MSAF = False


class SectionType(IntEnum):
    """Enumeration of section types."""
    INTRO = 0
    VERSE = 1
    CHORUS = 2
    PRE_CHORUS = 3
    POST_CHORUS = 4
    BRIDGE = 5
    INTERLUDE = 6
    OUTRO = 7
    UNKNOWN = 8


@dataclass
class Section:
    """Represents a detected musical section."""
    section_type: SectionType
    start_time: float
    end_time: float
    confidence: float
    features: Optional[Dict] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class StructureAnalysis:
    """Complete structure analysis result."""
    sections: List[Section]
    beats: np.ndarray
    downbeats: np.ndarray
    tempo: float
    time_signature: Tuple[int, int]
    key: str
    total_duration: float
    analysis_confidence: float


class AudioStructureDetector:
    """
    Detect musical structure from audio using signal processing.
    
    Methods:
    1. Beat tracking (tempo, beats, downbeats)
    2. Segment boundary detection (novelty-based)
    3. Section classification (feature-based)
    4. Structure template matching
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        use_madmom: bool = True
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.use_madmom = use_madmom and HAS_MADMOM
        
        # Beat tracking processors (if madmom available)
        if self.use_madmom:
            self.beat_processor = RNNBeatProcessor()
            self.beat_tracker = DBNBeatTrackingProcessor(fps=100)
        
        # Feature computation settings
        self.n_mfcc = 20
        self.n_chroma = 12
    
    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> StructureAnalysis:
        """
        Perform complete structure analysis on audio.
        
        Args:
            audio: Audio waveform (mono)
            sample_rate: Sample rate (uses default if None)
            
        Returns:
            StructureAnalysis with all detected elements
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        total_duration = len(audio) / self.sample_rate
        
        # 1. Beat and tempo analysis
        beats, downbeats, tempo = self._analyze_beats(audio)
        
        # 2. Key detection
        key = self._detect_key(audio)
        
        # 3. Time signature estimation
        time_sig = self._estimate_time_signature(beats, downbeats)
        
        # 4. Segment boundary detection
        boundaries = self._detect_boundaries(audio)
        
        # 5. Extract features per segment
        segment_features = self._extract_segment_features(audio, boundaries)
        
        # 6. Classify sections
        sections = self._classify_sections(boundaries, segment_features, total_duration)
        
        # 7. Compute overall confidence
        confidence = self._compute_analysis_confidence(sections, boundaries)
        
        return StructureAnalysis(
            sections=sections,
            beats=beats,
            downbeats=downbeats,
            tempo=tempo,
            time_signature=time_sig,
            key=key,
            total_duration=total_duration,
            analysis_confidence=confidence
        )
    
    def _analyze_beats(
        self,
        audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Detect beats, downbeats, and tempo."""
        
        if self.use_madmom:
            try:
                # Use madmom for more accurate beat tracking
                activations = self.beat_processor(audio)
                beats = self.beat_tracker(activations)
                
                # Estimate downbeats
                # For simplicity, assume every 4th beat is a downbeat
                if len(beats) >= 4:
                    downbeats = beats[::4]
                else:
                    downbeats = beats[:1] if len(beats) > 0 else np.array([])
                
                # Compute tempo from beat intervals
                if len(beats) > 1:
                    intervals = np.diff(beats)
                    tempo = 60.0 / np.median(intervals)
                else:
                    tempo = 120.0
                
                return beats, downbeats, tempo
                
            except Exception as e:
                print(f"⚠️  madmom beat tracking failed: {e}, falling back to librosa")
        
        # Fallback: librosa beat tracking
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            units='time'
        )
        
        # Estimate downbeats (every 4th beat for 4/4)
        if len(beats) >= 4:
            downbeats = beats[::4]
        else:
            downbeats = beats[:1] if len(beats) > 0 else np.array([])
        
        return beats, downbeats, float(tempo)
    
    def _detect_key(self, audio: np.ndarray) -> str:
        """Detect musical key using chroma features."""
        
        # Compute chroma
        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Average chroma profile
        chroma_avg = np.mean(chroma, axis=1)
        
        # Key profiles (Krumhansl-Schmuckler)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        best_corr = -1
        best_key = 'C major'
        
        for i in range(12):
            # Rotate chroma to align with key
            rotated = np.roll(chroma_avg, -i)
            
            # Correlate with major profile
            major_corr = np.corrcoef(rotated, major_profile)[0, 1]
            if major_corr > best_corr:
                best_corr = major_corr
                best_key = f"{key_names[i]} major"
            
            # Correlate with minor profile
            minor_corr = np.corrcoef(rotated, minor_profile)[0, 1]
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = f"{key_names[i]} minor"
        
        return best_key
    
    def _estimate_time_signature(
        self,
        beats: np.ndarray,
        downbeats: np.ndarray
    ) -> Tuple[int, int]:
        """Estimate time signature from beat patterns."""
        
        if len(beats) < 4 or len(downbeats) < 2:
            return (4, 4)  # Default
        
        # Count beats between downbeats
        beats_per_bar = []
        for i in range(len(downbeats) - 1):
            start_db = downbeats[i]
            end_db = downbeats[i + 1]
            count = np.sum((beats >= start_db) & (beats < end_db))
            beats_per_bar.append(count)
        
        if not beats_per_bar:
            return (4, 4)
        
        # Most common beats per bar
        from collections import Counter
        most_common = Counter(beats_per_bar).most_common(1)[0][0]
        
        # Map to time signature
        if most_common == 3:
            return (3, 4)
        elif most_common == 6:
            return (6, 8)
        else:
            return (4, 4)
    
    def _detect_boundaries(self, audio: np.ndarray) -> np.ndarray:
        """Detect segment boundaries using novelty function."""
        
        # Compute features for novelty
        # 1. MFCC for timbral changes
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        
        # 2. Chroma for harmonic changes
        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Stack features
        features = np.vstack([mfcc, chroma])
        
        # Self-similarity matrix
        ssm = librosa.segment.recurrence_matrix(
            features,
            mode='affinity',
            metric='cosine',
            sparse=False
        )
        
        # Novelty curve from checkerboard kernel
        novelty = librosa.segment.novelty_detection(ssm, kernel_size=64)
        
        # Find peaks in novelty curve
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(
            novelty,
            height=np.mean(novelty) + 0.5 * np.std(novelty),
            distance=int(2.0 * self.sample_rate / self.hop_length)  # Min 2 seconds between boundaries
        )
        
        # Convert to time
        times = librosa.frames_to_time(peaks, sr=self.sample_rate, hop_length=self.hop_length)
        
        # Add start and end
        duration = len(audio) / self.sample_rate
        boundaries = np.concatenate([[0], times, [duration]])
        
        return boundaries
    
    def _extract_segment_features(
        self,
        audio: np.ndarray,
        boundaries: np.ndarray
    ) -> List[Dict]:
        """Extract features for each segment."""
        
        segment_features = []
        
        for i in range(len(boundaries) - 1):
            start_sample = int(boundaries[i] * self.sample_rate)
            end_sample = int(boundaries[i + 1] * self.sample_rate)
            segment = audio[start_sample:end_sample]
            
            if len(segment) < self.hop_length:
                segment_features.append({})
                continue
            
            # Compute features
            features = {}
            
            # Energy (RMS)
            rms = librosa.feature.rms(y=segment, hop_length=self.hop_length)
            features['energy'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            
            # Spectral centroid (brightness)
            cent = librosa.feature.spectral_centroid(y=segment, sr=self.sample_rate, hop_length=self.hop_length)
            features['brightness'] = float(np.mean(cent))
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=segment, sr=self.sample_rate, hop_length=self.hop_length)
            features['contrast'] = float(np.mean(contrast))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(segment, hop_length=self.hop_length)
            features['zcr'] = float(np.mean(zcr))
            
            # Chroma
            chroma = librosa.feature.chroma_cqt(y=segment, sr=self.sample_rate, hop_length=self.hop_length)
            features['chroma_mean'] = chroma.mean(axis=1).tolist()
            
            # Duration
            features['duration'] = boundaries[i + 1] - boundaries[i]
            features['start_time'] = boundaries[i]
            features['end_time'] = boundaries[i + 1]
            
            segment_features.append(features)
        
        return segment_features
    
    def _classify_sections(
        self,
        boundaries: np.ndarray,
        segment_features: List[Dict],
        total_duration: float
    ) -> List[Section]:
        """Classify each segment into section type."""
        
        sections = []
        
        # Normalize features for classification
        energies = [f.get('energy', 0) for f in segment_features if f]
        if not energies:
            energies = [0.5]
        
        energy_mean = np.mean(energies)
        energy_std = np.std(energies) if len(energies) > 1 else 0.1
        
        for i, features in enumerate(segment_features):
            if not features:
                section_type = SectionType.UNKNOWN
                confidence = 0.3
            else:
                # Simple rule-based classification
                start_time = features['start_time']
                end_time = features['end_time']
                duration = features['duration']
                energy = features.get('energy', 0.5)
                
                # Normalize energy
                energy_norm = (energy - energy_mean) / (energy_std + 1e-6)
                
                # Position-based heuristics
                relative_position = start_time / total_duration
                
                # Classify based on position and energy
                if relative_position < 0.1:
                    section_type = SectionType.INTRO
                    confidence = 0.8
                elif relative_position > 0.9:
                    section_type = SectionType.OUTRO
                    confidence = 0.8
                elif energy_norm > 0.5:
                    # High energy sections
                    section_type = SectionType.CHORUS
                    confidence = 0.7
                elif energy_norm < -0.3:
                    # Low energy sections
                    section_type = SectionType.BRIDGE if 0.5 < relative_position < 0.8 else SectionType.VERSE
                    confidence = 0.6
                else:
                    # Medium energy
                    section_type = SectionType.VERSE
                    confidence = 0.5
            
            sections.append(Section(
                section_type=section_type,
                start_time=boundaries[i],
                end_time=boundaries[i + 1],
                confidence=confidence,
                features=features
            ))
        
        return sections
    
    def _compute_analysis_confidence(
        self,
        sections: List[Section],
        boundaries: np.ndarray
    ) -> float:
        """Compute overall analysis confidence."""
        
        if not sections:
            return 0.0
        
        # Average section confidence
        section_conf = np.mean([s.confidence for s in sections])
        
        # Penalize too many or too few sections
        n_sections = len(sections)
        section_count_conf = 1.0 - abs(n_sections - 8) / 10  # Expect ~8 sections
        section_count_conf = max(0, section_count_conf)
        
        # Penalize very short sections
        durations = [s.duration for s in sections]
        short_sections = sum(1 for d in durations if d < 10) / len(sections)
        duration_conf = 1.0 - short_sections
        
        return float(np.mean([section_conf, section_count_conf, duration_conf]))


def convert_structure_to_conditioning(
    analysis: StructureAnalysis
) -> Dict:
    """
    Convert structure analysis to model conditioning format.
    
    Returns format compatible with MTSModel.encode_structure()
    """
    section_types = [int(s.section_type) for s in analysis.sections]
    section_props = []
    
    for s in analysis.sections:
        energy = s.features.get('energy', 0.5) if s.features else 0.5
        importance = 1.0 if s.section_type == SectionType.CHORUS else 0.5
        
        section_props.append([
            s.start_time,
            s.end_time,
            s.duration,
            energy,
            importance
        ])
    
    return {
        "section_types": section_types,
        "section_props": section_props,
        "tempo": analysis.tempo,
        "key": analysis.key,
        "time_signature": analysis.time_signature
    }


def test_structure_detection():
    """Test structure detection on synthetic audio."""
    print("🧪 Testing Structure Detection...")
    
    # Create test audio (2 minutes of varying tones)
    sr = 22050
    duration = 120.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create sections with different characteristics
    audio = np.zeros_like(t)
    
    # Intro (0-10s): quiet, building
    mask = t < 10
    audio[mask] = 0.3 * np.sin(2 * np.pi * 220 * t[mask]) * (t[mask] / 10)
    
    # Verse 1 (10-30s): moderate
    mask = (t >= 10) & (t < 30)
    audio[mask] = 0.5 * np.sin(2 * np.pi * 330 * t[mask])
    
    # Chorus 1 (30-50s): loud, energetic
    mask = (t >= 30) & (t < 50)
    audio[mask] = 0.9 * np.sin(2 * np.pi * 440 * t[mask])
    
    # Verse 2 (50-70s): moderate
    mask = (t >= 50) & (t < 70)
    audio[mask] = 0.5 * np.sin(2 * np.pi * 330 * t[mask])
    
    # Chorus 2 (70-90s): loud
    mask = (t >= 70) & (t < 90)
    audio[mask] = 0.9 * np.sin(2 * np.pi * 440 * t[mask])
    
    # Bridge (90-105s): different
    mask = (t >= 90) & (t < 105)
    audio[mask] = 0.4 * np.sin(2 * np.pi * 370 * t[mask])
    
    # Outro (105-120s): fade out
    mask = t >= 105
    fade = 1.0 - (t[mask] - 105) / 15
    audio[mask] = 0.5 * np.sin(2 * np.pi * 220 * t[mask]) * fade
    
    # Add some noise
    audio += 0.01 * np.random.randn(len(audio))
    
    # Run detection
    detector = AudioStructureDetector(sample_rate=sr)
    analysis = detector.analyze(audio)
    
    print(f"\nDetected {len(analysis.sections)} sections:")
    for i, section in enumerate(analysis.sections):
        print(f"  {i+1}. {section.section_type.name}: "
              f"{section.start_time:.1f}s - {section.end_time:.1f}s "
              f"(conf: {section.confidence:.2f})")
    
    print(f"\nTempo: {analysis.tempo:.1f} BPM")
    print(f"Key: {analysis.key}")
    print(f"Time signature: {analysis.time_signature}")
    print(f"Overall confidence: {analysis.analysis_confidence:.2f}")
    
    # Convert to conditioning format
    conditioning = convert_structure_to_conditioning(analysis)
    print(f"\nConditioning format:")
    print(f"  Section types: {conditioning['section_types']}")
    
    print("\n✅ Structure detection test complete!")


if __name__ == "__main__":
    test_structure_detection()
