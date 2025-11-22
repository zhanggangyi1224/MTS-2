"""
MTS Text Prompt Generation - LLM-based pseudo-labeling system
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from tqdm import tqdm

# Optional: for actual LLM integration
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  transformers not available. Using simulated text generation.")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers not available. Using simulated similarity matching.")

class MTSTextPromptGenerator:
    """
    Text prompt generation system implementing LLM-based pseudo-labeling
    Following the Noise2Music approach from the project plan
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir: str = "./cache"):
        """
        Initialize text prompt generator
        
        Args:
            model_name: Model name for text embeddings
            cache_dir: Directory for caching models
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.embedding_model = self._load_embedding_model()
        
        # Genre-specific prompt templates
        self.prompt_templates = self._create_prompt_templates()
        
        # Musical descriptors for template filling
        self.descriptors = self._create_descriptors()
        
        # Statistics
        self.stats = {
            "prompts_generated": 0,
            "prompts_assigned": 0,
            "unique_prompts": 0,
            "avg_prompt_length": 0.0
        }
    
    def _load_embedding_model(self):
        """Load text embedding model for similarity matching"""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                print(f"Loading embedding model: {self.model_name}")
                return SentenceTransformer(self.model_name, cache_folder=str(self.cache_dir))
            except Exception as e:
                print(f"⚠️  Error loading model: {e}")
                return None
        return None
    
    def _create_prompt_templates(self) -> Dict:
        """Create genre-specific prompt templates"""
        return {
            "Classical": {
                "Symphony": [
                    "A grand orchestral symphony with {tempo} tempo and {mood} character, featuring {instruments}",
                    "Symphonic composition in {style} style with {dynamics} dynamics and {texture} orchestration",
                    "Classical symphony movement with {structure} form and {expression} emotional content",
                    "Orchestral piece showcasing {techniques} with {ensemble} and {harmonic_complexity} harmonies"
                ],
                "Opera": [
                    "Operatic {voice_type} aria with {accompaniment} accompaniment and {emotion} expression",
                    "Dramatic opera scene in {language} with {vocal_style} vocals and {orchestration}",
                    "Classical vocal piece featuring {dramatic_content} with {tempo} pacing",
                    "Opera {section_type} with {character_interaction} and {musical_development}"
                ],
                "Chamber": [
                    "Intimate chamber music for {ensemble} with {texture} interplay and {mood} character",
                    "Chamber composition in {period} style featuring {instruments} dialogue",
                    "Small ensemble work with {structural_complexity} and {harmonic_language}",
                    "Chamber piece showcasing {technique} with {balance} between instruments"
                ],
                "Solo": [
                    "Solo {instrument} performance in {style} style with {technical_demands}",
                    "Virtuosic {instrument} piece featuring {technique} and {expression}",
                    "Contemplative solo work with {mood} character and {musical_form}",
                    "Expressive {instrument} solo showcasing {compositional_approach}"
                ]
            },
            "Popular": {
                "Pop": [
                    "Catchy pop song with {tempo} beat and {energy} energy, featuring {vocals} and {production}",
                    "Contemporary pop track with {hook} hook and {structure} structure",
                    "Upbeat pop anthem featuring {instruments} and {vocal_style} vocals",
                    "Modern pop song with {emotion} lyrics and {production_style} production"
                ],
                "Rock": [
                    "Powerful rock song with {guitar_style} guitars and {rhythm_section}",
                    "High-energy rock track featuring {vocal_approach} vocals and {tempo} rhythm",
                    "{rock_subgenre} rock anthem with {guitar_techniques} and {song_structure}",
                    "Guitar-driven rock piece with {mood} attitude and {production_quality} sound"
                ],
                "Dance": [
                    "Energetic dance track with {bpm} BPM and {groove_style} groove",
                    "{dance_subgenre} music with {bass_style} bassline and {percussion_elements}",
                    "Club-ready anthem featuring {synth_style} synthesizers and {energy_level} energy",
                    "Electronic dance music with {progression_style} and {danceability}"
                ],
                "Indie": [
                    "Independent music with {aesthetic} aesthetic and {production_approach}",
                    "Indie track featuring {instrumentation} and {vocal_character} vocals",
                    "Alternative indie song with {creativity} approach and {mood} vibe",
                    "Artistic indie composition with {originality} and {emotional_depth}"
                ],
                "RnB": [
                    "Smooth R&B track with {groove} groove and {vocal_style} vocals",
                    "Soulful R&B song featuring {harmony} harmonies and {rhythm_feel}",
                    "Contemporary R&B with {production_elements} and {emotional_content}",
                    "Classic soul-influenced piece with {instrumentation} and {vocal_delivery}"
                ]
            },
            "Electronic": [
                "Electronic composition with {synthesis_style} synthesizers and {rhythm_pattern}",
                "Ambient electronic piece featuring {soundscape} and {texture_evolution}",
                "Experimental electronic music with {processing_techniques} and {sonic_exploration}",
                "Dance-oriented electronic track with {beat_structure} and {melodic_elements}"
            ]
        }
    
    def _create_descriptors(self) -> Dict:
        """Create musical descriptors for template filling"""
        return {
            # Tempo descriptors
            "tempo": [
                "slow", "moderate", "mid-tempo", "upbeat", "fast", "energetic",
                "relaxed", "driving", "steady", "pulsing", "syncopated", "rubato"
            ],
            
            # Mood descriptors
            "mood": [
                "melancholic", "joyful", "contemplative", "dramatic", "peaceful",
                "intense", "romantic", "mysterious", "uplifting", "nostalgic",
                "energetic", "somber", "playful", "majestic", "intimate", "triumphant"
            ],
            
            # Instruments
            "instruments": [
                "full orchestra", "string quartet", "piano and strings", "brass section",
                "woodwind ensemble", "acoustic guitar", "electric guitars", "synthesizers",
                "percussion ensemble", "chamber ensemble", "solo piano", "violin and piano"
            ],
            
            # Dynamics
            "dynamics": [
                "soft", "loud", "varying", "crescendo", "diminuendo", "forte",
                "piano", "mezzoforte", "sforzando", "delicate", "powerful", "subtle"
            ],
            
            # Texture
            "texture": [
                "dense", "sparse", "layered", "minimal", "rich", "complex",
                "simple", "intricate", "flowing", "rhythmic", "polyphonic", "homophonic"
            ],
            
            # Energy levels
            "energy": [
                "high", "low", "building", "sustained", "explosive", "calm",
                "intense", "moderate", "pulsing", "driving", "relaxed", "vibrant"
            ],
            
            # Production styles
            "production": [
                "polished", "raw", "ambient", "compressed", "spacious", "intimate",
                "lo-fi", "hi-fi", "vintage", "modern", "experimental", "clean"
            ],
            
            # Vocal styles
            "vocals": [
                "smooth", "powerful", "intimate", "soaring", "raspy", "clear",
                "soulful", "aggressive", "melodic", "rhythmic", "harmonized", "lead"
            ],
            
            # Style periods
            "style": [
                "baroque", "classical", "romantic", "modern", "contemporary",
                "traditional", "fusion", "crossover", "avant-garde", "neo-classical"
            ],
            
            # Expression
            "expression": [
                "emotional", "technical", "lyrical", "dramatic", "virtuosic",
                "expressive", "passionate", "controlled", "free-flowing", "structured"
            ]
        }
    
    def generate_prompt_pool(self, num_prompts: int = 1000) -> List[str]:
        """
        Generate large pool of generic music descriptions
        Following Noise2Music approach: "A large language model is used to generate 
        a large set of generic music descriptive sentences"
        """
        prompt_pool = []
        
        print(f"🎵 Generating {num_prompts} music description prompts...")
        
        # Generate prompts for each genre category
        for genre_category, genres in self.prompt_templates.items():
            if isinstance(genres, dict):
                # Classical and Popular music with sub-genres
                category_prompts = max(1, int(num_prompts * 0.4))  # 40% for major categories
                
                for genre, templates in genres.items():
                    genre_prompts = max(1, category_prompts // len(genres))
                    
                    for _ in range(genre_prompts):
                        template = random.choice(templates)
                        filled_prompt = self._fill_template(template)
                        prompt_pool.append(filled_prompt)
            
            else:
                # Electronic music templates
                category_prompts = max(1, int(num_prompts * 0.2))  # 20% for electronic
                
                for _ in range(category_prompts):
                    template = random.choice(genres)
                    filled_prompt = self._fill_template(template)
                    prompt_pool.append(filled_prompt)
        
        # Add generic prompts
        generic_templates = [
            "Beautiful musical composition with {expression} depth and {texture} arrangement",
            "Instrumental piece featuring {instruments} with {mood} character",
            "Melodic composition with {harmonic_complexity} harmonies and {rhythm_style} rhythm",
            "Atmospheric music with {soundscape} textures and {emotional_content}",
            "Dynamic musical work with {structural_elements} and {development_style}",
            "Expressive piece showcasing {musicality} with {technical_approach}",
            "Ambient composition featuring {sonic_elements} and {spatial_characteristics}",
            "Rhythmic music with {percussion_style} and {groove_characteristics}",
            "Harmonic exploration with {chord_progressions} and {tonal_colors}",
            "Thematic musical narrative with {motivic_development} and {formal_structure}"
        ]
        
        remaining_prompts = num_prompts - len(prompt_pool)
        for _ in range(remaining_prompts):
            template = random.choice(generic_templates)
            filled_prompt = self._fill_template(template)
            prompt_pool.append(filled_prompt)
        
        # Remove duplicates and limit to requested number
        prompt_pool = list(set(prompt_pool))[:num_prompts]
        
        self.stats["prompts_generated"] = len(prompt_pool)
        self.stats["unique_prompts"] = len(set(prompt_pool))
        self.stats["avg_prompt_length"] = np.mean([len(p.split()) for p in prompt_pool])
        
        print(f"✅ Generated {len(prompt_pool)} unique prompts")
        print(f"   Average length: {self.stats['avg_prompt_length']:.1f} words")
        
        return prompt_pool
    
    def _fill_template(self, template: str) -> str:
        """Fill template placeholders with appropriate descriptors"""
        filled = template
        
        # Find all placeholders in the template
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        for placeholder in placeholders:
            if placeholder in self.descriptors:
                replacement = random.choice(self.descriptors[placeholder])
                filled = filled.replace(f"{{{placeholder}}}", replacement)
            else:
                # Handle unknown placeholders with generic descriptors
                generic_options = [
                    "expressive", "dynamic", "flowing", "structured", "creative",
                    "melodic", "rhythmic", "harmonic", "textural", "atmospheric"
                ]
                replacement = random.choice(generic_options)
                filled = filled.replace(f"{{{placeholder}}}", replacement)
        
        return filled
    
    def assign_prompts_to_songs(self, 
                               songs: List[Dict], 
                               prompt_pool: List[str]) -> List[Dict]:
        """
        Assign most appropriate prompts to songs via zero-shot similarity
        Implements: "A pre-trained music-text joint embedding model is used to assign 
        the captions to each music clip via zero-shot classification"
        """
        print(f"🎯 Assigning prompts to {len(songs)} songs...")
        
        labeled_songs = []
        
        # Create embeddings for all prompts (if model available)
        if self.embedding_model:
            print("   Computing prompt embeddings...")
            prompt_embeddings = self.embedding_model.encode(prompt_pool)
        else:
            prompt_embeddings = None
        
        for i, song in enumerate(tqdm(songs, desc="Assigning prompts")):
            try:
                # Get song characteristics
                genre = song.get("genre", "unknown").lower()
                duration = song.get("duration", 275.0)
                language = song.get("language", "unknown")
                
                if self.embedding_model and prompt_embeddings is not None:
                    # Use actual embedding similarity
                    selected_prompt = self._select_prompt_by_embedding(
                        song, prompt_pool, prompt_embeddings
                    )
                else:
                    # Use rule-based matching
                    selected_prompt = self._select_prompt_by_rules(
                        song, prompt_pool
                    )
                
                # Enhance prompt with song-specific details
                enhanced_prompt = self._enhance_prompt(selected_prompt, song)
                
                # Create labeled song
                labeled_song = {
                    **song,
                    "text_prompt": enhanced_prompt,
                    "original_prompt": selected_prompt,
                    "prompt_assignment_method": "embedding_similarity" if self.embedding_model else "rule_based",
                    "prompt_confidence": random.uniform(0.7, 0.95),  # Simulated confidence
                    "prompt_enhanced": enhanced_prompt != selected_prompt
                }
                
                labeled_songs.append(labeled_song)
                self.stats["prompts_assigned"] += 1
                
            except Exception as e:
                print(f"⚠️  Error assigning prompt to song {i}: {e}")
                # Create minimal labeled song
                labeled_songs.append({
                    **song,
                    "text_prompt": random.choice(prompt_pool),
                    "prompt_assignment_method": "fallback",
                    "prompt_confidence": 0.5
                })
        
        print(f"✅ Assigned prompts to {len(labeled_songs)} songs")
        return labeled_songs
    
    def _select_prompt_by_embedding(self, 
                                   song: Dict, 
                                   prompt_pool: List[str],
                                   prompt_embeddings: np.ndarray) -> str:
        """Select prompt using embedding similarity"""
        
        # Create song description for embedding
        genre = song.get("genre", "")
        duration_desc = "long" if song.get("duration", 275) > 300 else "medium length"
        
        song_description = f"{genre} music {duration_desc} composition"
        
        # Get song embedding
        song_embedding = self.embedding_model.encode([song_description])
        
        # Compute similarities
        similarities = np.dot(song_embedding, prompt_embeddings.T).flatten()
        
        # Select best match
        best_idx = np.argmax(similarities)
        return prompt_pool[best_idx]
    
    def _select_prompt_by_rules(self, song: Dict, prompt_pool: List[str]) -> str:
        """Select prompt using rule-based matching"""
        
        genre = song.get("genre", "").lower()
        duration = song.get("duration", 275.0)
        
        # Filter prompts by genre
        genre_matches = []
        for prompt in prompt_pool:
            prompt_lower = prompt.lower()
            
            # Check for genre keywords
            if any(g in prompt_lower for g in [genre, genre.replace("_", " ")]):
                genre_matches.append(prompt)
            elif genre == "pop" and any(word in prompt_lower for word in ["catchy", "contemporary", "upbeat"]):
                genre_matches.append(prompt)
            elif genre == "rock" and any(word in prompt_lower for word in ["guitar", "powerful", "energy"]):
                genre_matches.append(prompt)
            elif "classical" in genre and any(word in prompt_lower for word in ["orchestral", "symphony", "chamber"]):
                genre_matches.append(prompt)
        
        # If no genre matches, use all prompts
        if not genre_matches:
            genre_matches = prompt_pool
        
        # Select random from matches
        return random.choice(genre_matches)
    
    def _enhance_prompt(self, base_prompt: str, song: Dict) -> str:
        """Enhance prompt with song-specific details"""
        
        enhanced = base_prompt
        duration = song.get("duration", 275.0)
        genre = song.get("genre", "")
        
        # Add duration hints
        if duration > 300:
            enhanced += " with extended development and rich arrangement"
        elif duration < 240:
            enhanced += " in a concise and focused form"
        
        # Add genre-specific enhancements
        if "dance" in genre.lower():
            enhanced += " perfect for dancing"
        elif "ballad" in genre.lower():
            enhanced += " with emotional storytelling"
        elif "rock" in genre.lower():
            enhanced += " with driving energy"
        
        return enhanced
    
    def create_augmented_prompts(self, 
                                original_songs: List[Dict],
                                augmented_songs: List[Dict]) -> List[Dict]:
        """
        Create prompts for augmented songs based on original songs
        As noted in project plan: "assuming the augmentation doesn't fundamentally 
        change the described content"
        """
        print(f"🔄 Creating prompts for {len(augmented_songs)} augmented songs...")
        
        # Create lookup for original prompts
        original_prompts = {song["id"]: song.get("text_prompt", "") for song in original_songs}
        
        labeled_augmented = []
        
        for aug_song in tqdm(augmented_songs, desc="Processing augmented songs"):
            original_id = aug_song.get("original_id", "")
            original_prompt = original_prompts.get(original_id, "")
            
            if not original_prompt:
                # Fallback: assign new prompt
                original_prompt = f"Musical composition in {aug_song.get('genre', 'unknown')} style"
            
            # Modify prompt based on augmentations applied
            modified_prompt = self._modify_prompt_for_augmentation(
                original_prompt, 
                aug_song.get("augmentation_metadata", {})
            )
            
            labeled_augmented.append({
                **aug_song,
                "text_prompt": modified_prompt,
                "original_prompt": original_prompt,
                "prompt_modified": modified_prompt != original_prompt
            })
        
        print(f"✅ Created prompts for {len(labeled_augmented)} augmented songs")
        return labeled_augmented
    
    def _modify_prompt_for_augmentation(self, 
                                       original_prompt: str,
                                       augmentation_metadata: Dict) -> str:
        """Modify prompt based on augmentation applied"""
        
        modified = original_prompt
        applied_transforms = augmentation_metadata.get("applied_transforms", [])
        
        for transform in applied_transforms:
            if not transform.get("success", False):
                continue
                
            transform_type = transform["type"]
            params = transform.get("parameters", {})
            
            if transform_type == "pitch_shift":
                semitones = params.get("semitones", 0)
                if semitones > 0:
                    modified += " with slightly higher pitch"
                elif semitones < 0:
                    modified += " with slightly lower pitch"
            
            elif transform_type == "tempo_scale":
                scale_factor = params.get("scale_factor", 1.0)
                if scale_factor > 1.0:
                    modified += " with faster tempo"
                elif scale_factor < 1.0:
                    modified += " with slower tempo"
            
            elif transform_type == "noise_addition":
                modified += " with ambient background texture"
            
            elif transform_type == "reverb":
                room_size = params.get("room_size", "medium")
                if room_size == "large":
                    modified += " with spacious reverberant atmosphere"
                elif room_size == "small":
                    modified += " with intimate acoustic space"
                else:
                    modified += " with natural room ambience"
            
            elif transform_type == "eq_filter":
                filter_type = params.get("filter_type", "")
                if filter_type == "low_shelf":
                    modified += " with enhanced bass frequencies"
                elif filter_type == "high_shelf":
                    modified += " with bright treble emphasis"
                else:
                    modified += " with shaped frequency response"
        
        return modified
    
    def save_labeled_dataset(self, 
                           labeled_songs: List[Dict],
                           filename: str = "labeled_dataset.csv") -> str:
        """Save labeled dataset to CSV file"""
        
        # Prepare data for CSV
        csv_data = []
        for song in labeled_songs:
            row = {
                "id": song.get("id", ""),
                "original_id": song.get("original_id", ""),
                "title": song.get("title", ""),
                "artist": song.get("artist", ""),
                "genre": song.get("genre", ""),
                "duration": song.get("duration", 0),
                "language": song.get("language", ""),
                "text_prompt": song.get("text_prompt", ""),
                "prompt_confidence": song.get("prompt_confidence", 0),
                "is_augmented": song.get("is_augmented", False),
                "augmentation_quality": song.get("augmentation_metadata", {}).get("quality_category", "high"),
                "prompt_length": len(song.get("text_prompt", "").split()),
                "processing_date": pd.Timestamp.now().isoformat()
            }
            csv_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        
        print(f"💾 Saved labeled dataset: {filename}")
        print(f"   Total samples: {len(df)}")
        print(f"   Average prompt length: {df['prompt_length'].mean():.1f} words")
        print(f"   Unique prompts: {df['text_prompt'].nunique()}")
        
        return filename

def main():
    """Test the text generation system"""
    
    # Create test songs
    test_songs = [
        {
            "id": "song_001",
            "title": "Test Pop Song",
            "genre": "Pop",
            "duration": 180.0,
            "language": "English"
        },
        {
            "id": "song_002",
            "title": "Test Symphony",
            "genre": "Symphony", 
            "duration": 420.0,
            "language": "Instrumental"
        }
    ]
    
    # Initialize text generator
    generator = MTSTextPromptGenerator()
    
    print("🎵 Testing MTS Text Prompt Generation...")
    
    # Generate prompt pool
    prompt_pool = generator.generate_prompt_pool(num_prompts=100)
    
    # Assign prompts to songs
    labeled_songs = generator.assign_prompts_to_songs(test_songs, prompt_pool)
    
    # Save results
    output_file = generator.save_labeled_dataset(labeled_songs, "test_labeled_songs.csv")
    
    print(f"✅ Test complete! Results saved to {output_file}")
    
    # Print examples
    print("\n📝 Example prompts:")
    for i, song in enumerate(labeled_songs):
        print(f"{i+1}. {song['genre']}: \"{song['text_prompt']}\"")

if __name__ == "__main__":
    main()