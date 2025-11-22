"""
MTS Structure Processing - Song structure annotation and analysis system
"""

import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MTSStructureProcessor:
    """
    Structure annotation processing system
    Implements structure analysis and conditioning as specified in project plan section 1.5
    """
    
    def __init__(self, structure_subset_size: int = 300):
        """
        Initialize structure processor
        
        Args:
            structure_subset_size: Number of songs to annotate (300 as per project plan)
        """
        self.structure_subset_size = structure_subset_size
        
        # Structure labels from CCMusic song_structure dataset
        self.structure_labels = [
            "intro", "re-intro", "verse", "chorus", "pre-chorus",
            "post-chorus", "bridge", "interlude", "ending", "outro"
        ]
        
        # Common song structures
        self.structure_templates = {
            "verse_chorus": {
                "template": ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "ending"],
                "description": "Standard verse-chorus form",
                "typical_genres": ["pop", "rock", "country"]
            },
            "aaba": {
                "template": ["intro", "verse", "verse", "chorus", "verse", "ending"],
                "description": "AABA form (32-bar song form)",
                "typical_genres": ["jazz", "standards", "folk"]
            },
            "pop_standard": {
                "template": ["intro", "verse", "pre-chorus", "chorus", "verse", "pre-chorus", "chorus", "bridge", "chorus", "ending"],
                "description": "Modern pop standard form",
                "typical_genres": ["pop", "contemporary", "adult_contemporary"]
            },
            "extended_pop": {
                "template": ["intro", "verse", "pre-chorus", "chorus", "verse", "pre-chorus", "chorus", "bridge", "chorus", "post-chorus", "ending"],
                "description": "Extended pop form with post-chorus",
                "typical_genres": ["pop", "dance_pop", "teen_pop"]
            },
            "blues_form": {
                "template": ["intro", "verse", "verse", "verse", "bridge", "verse", "ending"],
                "description": "12-bar blues form",
                "typical_genres": ["blues", "jazz", "rock"]
            },
            "classical_form": {
                "template": ["intro", "exposition", "development", "recapitulation", "coda"],
                "description": "Classical sonata form (adapted)",
                "typical_genres": ["symphony", "classical", "chamber"]
            }
        }
        
        # Typical section durations (in seconds)
        self.section_durations = {
            "intro": (8, 24),
            "re-intro": (4, 12),
            "verse": (20, 40),
            "chorus": (20, 40),
            "pre-chorus": (8, 20),
            "post-chorus": (8, 20),
            "bridge": (20, 40),
            "interlude": (8, 24),
            "ending": (8, 30),
            "outro": (10, 40),
            # Classical sections
            "exposition": (60, 120),
            "development": (80, 180),
            "recapitulation": (60, 120),
            "coda": (20, 60)
        }
        
        self.stats = {
            "songs_annotated": 0,
            "total_sections": 0,
            "avg_sections_per_song": 0.0,
            "structure_templates_used": defaultdict(int),
            "section_frequency": defaultdict(int)
        }
    
    def select_songs_for_annotation(self, songs: List[Dict]) -> List[Dict]:
        """
        Select subset of songs for structure annotation
        Simulates CCMusic song_structure subset (300 pop songs)
        """
        print(f"🎼 Selecting {self.structure_subset_size} songs for structure annotation...")
        
        # Prioritize certain genres for structure annotation
        priority_genres = ["pop", "rock", "dance", "contemporary"]
        
        # Separate songs by priority
        priority_songs = []
        other_songs = []
        
        for song in songs:
            genre = song.get("genre", "").lower()
            if any(pg in genre for pg in priority_genres):
                priority_songs.append(song)
            else:
                other_songs.append(song)
        
        # Select songs
        selected = []
        
        # First, select from priority genres
        n_priority = min(len(priority_songs), int(self.structure_subset_size * 0.7))
        selected.extend(random.sample(priority_songs, n_priority))
        
        # Then select from other genres
        remaining = self.structure_subset_size - len(selected)
        if remaining > 0 and other_songs:
            n_other = min(len(other_songs), remaining)
            selected.extend(random.sample(other_songs, n_other))
        
        print(f"   Selected {len(selected)} songs ({len(selected)/len(songs)*100:.1f}% of dataset)")
        
        return selected
    
    def generate_structure_annotations(self, songs: List[Dict]) -> List[Dict]:
        """
        Generate detailed structure annotations for selected songs
        """
        print(f"🏗️  Generating structure annotations for {len(songs)} songs...")
        
        annotated_songs = []
        
        for song in tqdm(songs, desc="Annotating structures"):
            try:
                # Get song metadata
                genre = song.get("genre", "unknown").lower()
                duration = song.get("duration", 275.0)
                song_id = song.get("id", f"song_{len(annotated_songs)}")
                
                # Select appropriate structure template
                template_name, template_info = self._select_structure_template(genre, duration)
                
                # Generate section annotations
                structure_annotations = self._generate_section_annotations(
                    template_info["template"],
                    duration,
                    song_id
                )
                
                # Create annotated song
                annotated_song = {
                    **song,
                    "structure_annotations": structure_annotations,
                    "structure_template": template_name,
                    "structure_description": template_info["description"],
                    "total_sections": len(structure_annotations),
                    "annotated_duration": sum(s["duration"] for s in structure_annotations),
                    "structure_coherence_score": self._calculate_coherence_score(structure_annotations),
                    "has_structure_annotation": True,
                    "annotation_quality": "simulated_high"
                }
                
                annotated_songs.append(annotated_song)
                
                # Update statistics
                self.stats["songs_annotated"] += 1
                self.stats["total_sections"] += len(structure_annotations)
                self.stats["structure_templates_used"][template_name] += 1
                
                for section in structure_annotations:
                    self.stats["section_frequency"][section["section_type"]] += 1
                
            except Exception as e:
                print(f"⚠️  Error annotating song {song.get('id', 'unknown')}: {e}")
                continue
        
        # Calculate final statistics
        if annotated_songs:
            self.stats["avg_sections_per_song"] = self.stats["total_sections"] / len(annotated_songs)
        
        print(f"✅ Structure annotation complete!")
        print(f"   Songs annotated: {len(annotated_songs)}")
        print(f"   Average sections per song: {self.stats['avg_sections_per_song']:.1f}")
        
        return annotated_songs
    
    def _select_structure_template(self, genre: str, duration: float) -> Tuple[str, Dict]:
        """Select appropriate structure template based on genre and duration"""
        
        # Genre-based template selection
        if any(g in genre for g in ["pop", "contemporary"]):
            if duration > 300:  # Long songs get extended form
                template_name = "extended_pop"
            else:
                template_name = "pop_standard"
        
        elif any(g in genre for g in ["rock", "alternative"]):
            template_name = "verse_chorus"
        
        elif any(g in genre for g in ["dance", "electronic"]):
            template_name = "verse_chorus"  # Simple structure for dance music
        
        elif any(g in genre for g in ["jazz", "standards"]):
            template_name = "aaba"
        
        elif any(g in genre for g in ["blues"]):
            template_name = "blues_form"
        
        elif any(g in genre for g in ["symphony", "classical", "chamber"]):
            template_name = "classical_form"
        
        else:
            # Default to verse-chorus for unknown genres
            template_name = "verse_chorus"
        
        # Duration-based adjustments
        if duration < 180:  # Short songs
            # Use simpler structure
            if template_name in ["extended_pop", "pop_standard"]:
                template_name = "verse_chorus"
        
        elif duration > 360:  # Very long songs
            # Use more complex structure
            if template_name == "verse_chorus":
                template_name = "extended_pop"
        
        return template_name, self.structure_templates[template_name]
    
    def _generate_section_annotations(self, 
                                    template: List[str],
                                    total_duration: float,
                                    song_id: str) -> List[Dict]:
        """Generate detailed section annotations from template"""
        
        annotations = []
        current_time = 0.0
        
        # Calculate total typical duration for scaling
        typical_total = sum(
            np.mean(self.section_durations.get(section, (20, 40)))
            for section in template
        )
        
        # Scale factor to fit target duration
        scale_factor = total_duration / typical_total if typical_total > 0 else 1.0
        
        for i, section_type in enumerate(template):
            # Get typical duration range for this section
            min_dur, max_dur = self.section_durations.get(section_type, (20, 40))
            
            # Calculate target duration with scaling
            typical_duration = (min_dur + max_dur) / 2
            scaled_duration = typical_duration * scale_factor
            
            # Add some variation (±20%)
            variation = random.uniform(0.8, 1.2)
            section_duration = scaled_duration * variation
            
            # Ensure we don't exceed remaining time
            remaining_time = total_duration - current_time
            sections_left = len(template) - i
            
            if sections_left > 1:
                # Not the last section - limit duration
                max_allowed = remaining_time * 0.8  # Leave room for remaining sections
                section_duration = min(section_duration, max_allowed)
            else:
                # Last section gets remaining time
                section_duration = remaining_time
            
            # Minimum duration constraint
            section_duration = max(section_duration, 5.0)
            
            # Create annotation
            annotation = {
                "section_id": f"{song_id}_{section_type}_{i:02d}",
                "section_type": section_type,
                "section_index": i,
                "start_time": current_time,
                "end_time": current_time + section_duration,
                "duration": section_duration,
                "confidence": random.uniform(0.85, 0.98),  # Simulated annotation confidence
                "key_signature": self._generate_key_signature(),
                "tempo_bpm": random.randint(80, 140),
                "time_signature": random.choice(["4/4", "3/4", "2/4", "6/8"]),
                "energy_level": self._calculate_section_energy(section_type),
                "structural_importance": self._calculate_structural_importance(section_type)
            }
            
            annotations.append(annotation)
            current_time += section_duration
            
            # Break if we've used up the duration
            if current_time >= total_duration:
                break
        
        return annotations
    
    def _generate_key_signature(self) -> str:
        """Generate random key signature"""
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        modes = ["major", "minor"]
        return f"{random.choice(keys)} {random.choice(modes)}"
    
    def _calculate_section_energy(self, section_type: str) -> float:
        """Calculate energy level for section type"""
        energy_map = {
            "intro": 0.3,
            "verse": 0.5,
            "pre-chorus": 0.7,
            "chorus": 0.9,
            "post-chorus": 0.8,
            "bridge": 0.6,
            "interlude": 0.4,
            "ending": 0.4,
            "outro": 0.2
        }
        
        base_energy = energy_map.get(section_type, 0.5)
        # Add some variation
        return max(0.1, min(1.0, base_energy + random.uniform(-0.1, 0.1)))
    
    def _calculate_structural_importance(self, section_type: str) -> float:
        """Calculate structural importance of section"""
        importance_map = {
            "intro": 0.6,
            "verse": 0.8,
            "pre-chorus": 0.7,
            "chorus": 1.0,  # Most important
            "post-chorus": 0.6,
            "bridge": 0.9,  # High importance for contrast
            "interlude": 0.4,
            "ending": 0.7,
            "outro": 0.5
        }
        
        return importance_map.get(section_type, 0.5)
    
    def _calculate_coherence_score(self, annotations: List[Dict]) -> float:
        """Calculate structural coherence score"""
        
        if not annotations:
            return 0.0
        
        score = 0.0
        factors = 0
        
        # Factor 1: Appropriate section count
        n_sections = len(annotations)
        if 5 <= n_sections <= 12:  # Ideal range
            score += 1.0
        elif 3 <= n_sections <= 15:  # Acceptable range
            score += 0.7
        else:
            score += 0.3
        factors += 1
        
        # Factor 2: Balanced section durations
        durations = [s["duration"] for s in annotations]
        duration_std = np.std(durations)
        duration_mean = np.mean(durations)
        cv = duration_std / duration_mean if duration_mean > 0 else 1
        
        if cv < 0.5:  # Well-balanced
            score += 1.0
        elif cv < 0.8:  # Reasonably balanced
            score += 0.7
        else:
            score += 0.4
        factors += 1
        
        # Factor 3: Presence of key sections
        section_types = [s["section_type"] for s in annotations]
        key_sections = ["chorus", "verse"]
        key_present = sum(1 for key in key_sections if key in section_types)
        score += key_present / len(key_sections)
        factors += 1
        
        # Factor 4: Logical progression
        # Check for intro at beginning and ending at end
        progression_score = 0.0
        if annotations[0]["section_type"] in ["intro", "verse"]:
            progression_score += 0.5
        if annotations[-1]["section_type"] in ["ending", "outro", "chorus"]:
            progression_score += 0.5
        score += progression_score
        factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def analyze_structure_patterns(self, annotated_songs: List[Dict]) -> Dict:
        """
        Analyze common structural patterns for model training
        As mentioned in project plan: "analyze common patterns (e.g. verse-chorus repetition)"
        """
        print(f"📈 Analyzing structure patterns from {len(annotated_songs)} annotated songs...")
        
        patterns = {
            "section_frequency": defaultdict(int),
            "section_duration_stats": {},
            "transition_frequency": defaultdict(int),
            "structure_templates": defaultdict(int),
            "temporal_patterns": {},
            "coherence_distribution": []
        }
        
        all_sections = []
        all_transitions = []
        
        for song in annotated_songs:
            annotations = song.get("structure_annotations", [])
            template = song.get("structure_template", "unknown")
            
            # Count template usage
            patterns["structure_templates"][template] += 1
            
            # Track coherence scores
            coherence = song.get("structure_coherence_score", 0.0)
            patterns["coherence_distribution"].append(coherence)
            
            # Analyze sections
            for i, section in enumerate(annotations):
                section_type = section["section_type"]
                duration = section["duration"]
                
                # Count section frequency
                patterns["section_frequency"][section_type] += 1
                
                # Collect section data
                all_sections.append({
                    "type": section_type,
                    "duration": duration,
                    "position": i,
                    "relative_position": i / len(annotations) if annotations else 0,
                    "energy_level": section.get("energy_level", 0.5),
                    "importance": section.get("structural_importance", 0.5)
                })
                
                # Track transitions
                if i < len(annotations) - 1:
                    next_section = annotations[i + 1]["section_type"]
                    transition = f"{section_type} -> {next_section}"
                    patterns["transition_frequency"][transition] += 1
                    all_transitions.append({
                        "from": section_type,
                        "to": next_section,
                        "from_duration": duration,
                        "position": i
                    })
        
        # Calculate section duration statistics
        section_types = set(s["type"] for s in all_sections)
        for section_type in section_types:
            type_durations = [s["duration"] for s in all_sections if s["type"] == section_type]
            
            if type_durations:
                patterns["section_duration_stats"][section_type] = {
                    "count": len(type_durations),
                    "mean": float(np.mean(type_durations)),
                    "std": float(np.std(type_durations)),
                    "min": float(np.min(type_durations)),
                    "max": float(np.max(type_durations)),
                    "median": float(np.median(type_durations))
                }
        
        # Analyze temporal patterns
        patterns["temporal_patterns"] = {
            "avg_sections_per_song": len(all_sections) / len(annotated_songs) if annotated_songs else 0,
            "most_common_first_section": self._get_most_common_first_section(annotated_songs),
            "most_common_last_section": self._get_most_common_last_section(annotated_songs),
            "avg_coherence_score": float(np.mean(patterns["coherence_distribution"])) if patterns["coherence_distribution"] else 0.0
        }
        
        # Convert defaultdicts to regular dicts for JSON serialization
        patterns["section_frequency"] = dict(patterns["section_frequency"])
        patterns["transition_frequency"] = dict(patterns["transition_frequency"])
        patterns["structure_templates"] = dict(patterns["structure_templates"])
        
        print(f"✅ Pattern analysis complete!")
        return patterns
    
    def _get_most_common_first_section(self, songs: List[Dict]) -> str:
        """Get most common first section type"""
        first_sections = []
        for song in songs:
            annotations = song.get("structure_annotations", [])
            if annotations:
                first_sections.append(annotations[0]["section_type"])
        
        if first_sections:
            from collections import Counter
            return Counter(first_sections).most_common(1)[0][0]
        return "unknown"
    
    def _get_most_common_last_section(self, songs: List[Dict]) -> str:
        """Get most common last section type"""
        last_sections = []
        for song in songs:
            annotations = song.get("structure_annotations", [])
            if annotations:
                last_sections.append(annotations[-1]["section_type"])
        
        if last_sections:
            from collections import Counter
            return Counter(last_sections).most_common(1)[0][0]
        return "unknown"
    
    def create_structure_conditioning_data(self, annotated_songs: List[Dict]) -> List[Dict]:
        """
        Create structure conditioning data for model training
        Implements both evaluation and conditioning approaches from project plan
        """
        print(f"🎯 Creating structure conditioning data for {len(annotated_songs)} songs...")
        
        conditioning_data = []
        
        for song in tqdm(annotated_songs, desc="Creating conditioning data"):
            annotations = song.get("structure_annotations", [])
            
            if not annotations:
                continue
            
            # Approach 1: Section timeline for conditioning
            section_timeline = []
            for section in annotations:
                section_timeline.append({
                    "time": section["start_time"],
                    "section": section["section_type"],
                    "duration": section["duration"],
                    "energy": section.get("energy_level", 0.5),
                    "importance": section.get("structural_importance", 0.5)
                })
            
            # Approach 2: High-level structure description
            section_sequence = [s["section_type"] for s in annotations]
            structure_description = " -> ".join(section_sequence)
            
            # Create structure-conditioned text prompt
            original_prompt = song.get("text_prompt", "")
            structure_hint = f"[Structure: {structure_description}]"
            structure_conditioned_prompt = f"{original_prompt} {structure_hint}"
            
            # Approach 3: Create 30-second compressed representation plan
            compressed_plan = self._create_compressed_structure_plan(annotations, target_duration=30.0)
            
            # Approach 4: Structure control tokens
            structure_tokens = self._create_structure_control_tokens(annotations)
            
            # Create conditioning item
            conditioning_item = {
                **song,
                "section_timeline": section_timeline,
                "structure_description": structure_description,
                "structure_conditioned_prompt": structure_conditioned_prompt,
                "compressed_structure_plan": compressed_plan,
                "structure_control_tokens": structure_tokens,
                "conditioning_methods": [
                    "section_timeline",
                    "structure_description", 
                    "compressed_representation",
                    "control_tokens"
                ],
                "can_use_for_structure_conditioning": True,
                "structure_conditioning_quality": "high"
            }
            
            conditioning_data.append(conditioning_item)
        
        print(f"✅ Created conditioning data for {len(conditioning_data)} songs")
        return conditioning_data
    
    def _create_compressed_structure_plan(self, 
                                        annotations: List[Dict],
                                        target_duration: float = 30.0) -> List[Dict]:
        """
        Create 30-second compressed representation plan
        As mentioned: "compress each major section to a few seconds"
        """
        compressed_sections = []
        total_original_duration = sum(s["duration"] for s in annotations)
        
        if total_original_duration == 0:
            return compressed_sections
        
        current_compressed_time = 0.0
        
        for section in annotations:
            # Calculate compression ratio
            original_duration = section["duration"]
            compression_ratio = original_duration / total_original_duration
            compressed_duration = compression_ratio * target_duration
            
            # Ensure minimum duration of 1 second per section
            compressed_duration = max(1.0, compressed_duration)
            
            compressed_section = {
                "original_section": section["section_type"],
                "original_start": section["start_time"],
                "original_duration": original_duration,
                "compressed_start": current_compressed_time,
                "compressed_duration": compressed_duration,
                "compression_ratio": compressed_duration / original_duration if original_duration > 0 else 1.0,
                "energy_preserved": section.get("energy_level", 0.5),
                "importance_weight": section.get("structural_importance", 0.5)
            }
            
            compressed_sections.append(compressed_section)
            current_compressed_time += compressed_duration
        
        # Normalize to exactly 30 seconds
        actual_total = current_compressed_time
        if actual_total > 0:
            scale_factor = target_duration / actual_total
            for section in compressed_sections:
                section["compressed_start"] *= scale_factor
                section["compressed_duration"] *= scale_factor
        
        return compressed_sections
    
    def _create_structure_control_tokens(self, annotations: List[Dict]) -> List[str]:
        """Create control tokens for structure conditioning"""
        
        tokens = []
        
        for section in annotations:
            section_type = section["section_type"]
            start_time = section["start_time"]
            
            # Create time-stamped structure token
            # Format: [SECTION_TYPE@TIME]
            time_token = f"[{section_type.upper()}@{int(start_time)}s]"
            tokens.append(time_token)
        
        return tokens
    
    def create_evaluation_metrics(self, annotated_songs: List[Dict], patterns: Dict) -> Dict:
        """
        Create evaluation metrics for structure coherence assessment
        """
        print(f"📏 Creating structure evaluation metrics...")
        
        evaluation_metrics = {
            "reference_structures": {},
            "evaluation_criteria": {},
            "coherence_benchmarks": {},
            "quality_thresholds": {}
        }
        
        # Reference structures for comparison
        template_references = defaultdict(list)
        for song in annotated_songs:
            template = song.get("structure_template", "unknown")
            if template != "unknown":
                template_references[template].append({
                    "song_id": song.get("id", ""),
                    "sections": [s["section_type"] for s in song.get("structure_annotations", [])],
                    "durations": [s["duration"] for s in song.get("structure_annotations", [])],
                    "total_duration": song.get("duration", 0),
                    "coherence_score": song.get("structure_coherence_score", 0.0)
                })
        
        evaluation_metrics["reference_structures"] = dict(template_references)
        
        # Evaluation criteria with weights
        evaluation_metrics["evaluation_criteria"] = {
            "section_presence": {
                "description": "Check if generated music contains expected sections",
                "metric_type": "binary_classification",
                "weight": 0.25,
                "expected_sections": ["verse", "chorus"],
                "threshold": 0.8
            },
            "section_repetition_similarity": {
                "description": "Verify that repeated sections sound similar",
                "metric_type": "similarity_score",
                "weight": 0.25,
                "target_similarity": 0.7,
                "sections_to_check": ["chorus", "verse"]
            },
            "structural_transitions": {
                "description": "Evaluate smoothness of transitions between sections",
                "metric_type": "continuity_score",
                "weight": 0.20,
                "acceptable_transitions": list(patterns["transition_frequency"].keys())[:20]
            },
            "overall_coherence": {
                "description": "Assess overall structural coherence",
                "metric_type": "coherence_score",
                "weight": 0.30,
                "min_coherence": 0.6,
                "target_coherence": 0.8
            }
        }
        
        # Coherence benchmarks based on analyzed patterns
        avg_coherence = patterns["temporal_patterns"]["avg_coherence_score"]
        
        evaluation_metrics["coherence_benchmarks"] = {
            "section_count": {
                "min": 3,
                "typical": int(patterns["temporal_patterns"]["avg_sections_per_song"]),
                "max": 15,
                "optimal_range": (5, 10)
            },
            "duration_ratios": self._create_duration_ratio_benchmarks(patterns),
            "common_transitions": {
                transition: count / len(annotated_songs)
                for transition, count in patterns["transition_frequency"].items()
                if count >= len(annotated_songs) * 0.1  # At least 10% of songs
            },
            "coherence_thresholds": {
                "excellent": max(0.85, avg_coherence + 0.1),
                "good": max(0.75, avg_coherence),
                "acceptable": max(0.60, avg_coherence - 0.1),
                "poor": max(0.40, avg_coherence - 0.2)
            }
        }
        
        # Quality thresholds for different use cases
        evaluation_metrics["quality_thresholds"] = {
            "production_ready": {
                "min_coherence": 0.8,
                "required_sections": ["verse", "chorus"],
                "max_section_count": 12,
                "min_section_count": 4
            },
            "research_acceptable": {
                "min_coherence": 0.6,
                "required_sections": ["verse"],
                "max_section_count": 15,
                "min_section_count": 3
            },
            "experimental": {
                "min_coherence": 0.4,
                "required_sections": [],
                "max_section_count": 20,
                "min_section_count": 1
            }
        }
        
        print(f"✅ Created comprehensive evaluation metrics")
        return evaluation_metrics
    
    def _create_duration_ratio_benchmarks(self, patterns: Dict) -> Dict:
        """Create duration ratio benchmarks from patterns"""
        
        benchmarks = {}
        duration_stats = patterns.get("section_duration_stats", {})
        
        # Typical song duration for ratio calculation
        typical_song_duration = 275.0  # seconds
        
        for section_type, stats in duration_stats.items():
            mean_duration = stats["mean"]
            std_duration = stats["std"]
            
            benchmarks[section_type] = {
                "mean_ratio": mean_duration / typical_song_duration,
                "std_ratio": std_duration / typical_song_duration,
                "min_ratio": stats["min"] / typical_song_duration,
                "max_ratio": stats["max"] / typical_song_duration,
                "acceptable_range": (
                    max(0.01, (mean_duration - std_duration) / typical_song_duration),
                    min(0.50, (mean_duration + std_duration) / typical_song_duration)
                )
            }
        
        return benchmarks
    
    def save_structure_data(self, 
                          annotated_songs: List[Dict],
                          patterns: Dict,
                          conditioning_data: List[Dict],
                          evaluation_metrics: Dict,
                          output_dir: str = "./outputs") -> Dict:
        """Save all structure processing results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        print(f"💾 Saving structure data to {output_path}...")
        
        # Save annotated songs (sample)
        annotations_file = output_path / "structure_annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump({
                "annotated_songs": annotated_songs[:50],  # Sample for size
                "total_annotated": len(annotated_songs),
                "annotation_stats": self.stats
            }, f, indent=2, default=str)
        saved_files["annotations"] = str(annotations_file)
        
        # Save structure patterns
        patterns_file = output_path / "structure_patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump(patterns, f, indent=2, default=str)
        saved_files["patterns"] = str(patterns_file)
        
        # Save conditioning data (sample)
        conditioning_file = output_path / "structure_conditioning.json"
        with open(conditioning_file, 'w') as f:
            json.dump({
                "conditioning_data": conditioning_data[:50],  # Sample
                "total_conditioning_samples": len(conditioning_data),
                "conditioning_methods": [
                    "section_timeline",
                    "structure_description",
                    "compressed_representation", 
                    "control_tokens"
                ]
            }, f, indent=2, default=str)
        saved_files["conditioning"] = str(conditioning_file)
        
        # Save evaluation metrics
        evaluation_file = output_path / "structure_evaluation_metrics.json"
        with open(evaluation_file, 'w') as f:
            json.dump(evaluation_metrics, f, indent=2, default=str)
        saved_files["evaluation"] = str(evaluation_file)
        
        # Save comprehensive summary
        summary_file = output_path / "structure_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "processing_summary": {
                    "songs_processed": len(annotated_songs),
                    "patterns_identified": len(patterns["structure_templates"]),
                    "conditioning_samples": len(conditioning_data),
                    "evaluation_criteria": len(evaluation_metrics["evaluation_criteria"])
                },
                "statistics": self.stats,
                "file_locations": saved_files,
                "processing_timestamp": pd.Timestamp.now().isoformat()
            }, f, indent=2, default=str)
        saved_files["summary"] = str(summary_file)
        
        print(f"✅ Saved structure processing data to {len(saved_files)} files")
        return saved_files

def main():
    """Test the structure processing system"""
    
    # Create test songs
    test_songs = []
    genres = ["Pop", "Rock", "Classical", "Jazz", "Dance_pop"]
    
    for i in range(50):
        song = {
            "id": f"test_structure_{i:03d}",
            "title": f"Test Structure Song {i}",
            "genre": random.choice(genres),
            "duration": random.uniform(180, 360),
            "text_prompt": f"A {random.choice(genres).lower()} song with musical structure"
        }
        test_songs.append(song)
    
    print("🎼 Testing MTS Structure Processing System...")
    
    # Initialize processor
    processor = MTSStructureProcessor(structure_subset_size=30)
    
    # Select songs for annotation
    selected_songs = processor.select_songs_for_annotation(test_songs)
    
    # Generate structure annotations
    annotated_songs = processor.generate_structure_annotations(selected_songs)
    
    # Analyze patterns
    patterns = processor.analyze_structure_patterns(annotated_songs)
    
    # Create conditioning data
    conditioning_data = processor.create_structure_conditioning_data(annotated_songs)
    
    # Create evaluation metrics
    evaluation_metrics = processor.create_evaluation_metrics(annotated_songs, patterns)
    
    # Save results
    saved_files = processor.save_structure_data(
        annotated_songs, patterns, conditioning_data, evaluation_metrics, "./test_outputs"
    )
    
    print(f"✅ Structure processing test complete!")
    print(f"   Songs annotated: {len(annotated_songs)}")
    print(f"   Structure templates: {len(patterns['structure_templates'])}")
    print(f"   Conditioning samples: {len(conditioning_data)}")
    print(f"   Files saved: {len(saved_files)}")

if __name__ == "__main__":
    main()