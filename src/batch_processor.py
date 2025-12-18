"""
Memory-efficient batch processor for MTS pipeline
Handles large datasets by processing in small batches
"""

import gc
import os
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import json
import yaml
from tqdm import tqdm
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

class MemoryMonitor:
    """Monitor and manage memory usage during processing"""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            "process_mb": memory_info.rss / 1024 / 1024,
            "system_percent": system_memory.percent,
            "system_available_mb": system_memory.available / 1024 / 1024,
            "system_total_mb": system_memory.total / 1024 / 1024
        }
    
    def is_memory_safe(self) -> bool:
        """Check if memory usage is within safe limits"""
        usage = self.get_memory_usage()
        return usage["system_percent"] < self.max_memory_percent
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        time.sleep(0.1)  # Brief pause for cleanup

class BatchProcessor:
    """
    Memory-efficient batch processor for MTS data pipeline
    """
    
    def __init__(self, 
                 batch_size: int = 100,
                 max_memory_percent: float = 80.0,
                 checkpoint_dir: str = "./checkpoints"):
        """
        Initialize batch processor
        
        Args:
            batch_size: Number of songs to process per batch
            max_memory_percent: Maximum memory usage percentage
            checkpoint_dir: Directory for saving checkpoints
        """
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.memory_monitor = MemoryMonitor(max_memory_percent)
        
        # Processing state
        self.state = {
            "current_batch": 0,
            "processed_songs": 0,
            "total_songs": 0,
            "completed_steps": [],
            "errors": [],
            "start_time": None,
            "last_checkpoint": None
        }
        
        # Load existing state if available
        self.load_state()
    
    def create_batches(self, data: List[Dict], batch_size: Optional[int] = None) -> Iterator[List[Dict]]:
        """
        Create batches from data list
        
        Args:
            data: List of data items
            batch_size: Override default batch size
            
        Yields:
            Batches of data
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    def save_state(self):
        """Save current processing state"""
        state_file = self.checkpoint_dir / "processing_state.json"
        
        state_to_save = {
            **self.state,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_to_save, f, indent=2, default=str)
    
    def load_state(self) -> bool:
        """Load previous processing state"""
        state_file = self.checkpoint_dir / "processing_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    saved_state = json.load(f)
                
                self.state.update(saved_state)
                print(f"📂 Loaded previous state: batch {self.state['current_batch']}, "
                      f"{self.state['processed_songs']} songs processed")
                return True
            except Exception as e:
                print(f"⚠️  Error loading state: {e}")
                return False
        
        return False
    
    def save_batch_results(self, batch_idx: int, results: Dict, step_name: str):
        """Save results from a batch"""
        batch_file = self.checkpoint_dir / f"{step_name}_batch_{batch_idx:04d}.json"
        
        # Prepare results for JSON serialization
        serializable_results = self._prepare_for_json(self._strip_large_fields(results))
        
        with open(batch_file, 'w') as f:
            json.dump({
                "batch_index": batch_idx,
                "step_name": step_name,
                "timestamp": datetime.now().isoformat(),
                "results": serializable_results
            }, f, indent=2)
        
        print(f"💾 Saved {step_name} batch {batch_idx} results")
    
    def load_batch_results(self, step_name: str) -> List[Dict]:
        """Load all batch results for a step"""
        results = []
        batch_files = list(self.checkpoint_dir.glob(f"{step_name}_batch_*.json"))
        batch_files.sort()  # Process in order
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r') as f:
                    batch_data = json.load(f)
                    restored_results = self._restore_from_json(batch_data["results"])
                    results.extend(restored_results)
            except Exception as e:
                print(f"⚠️  Error loading {batch_file}: {e}")
        
        return results
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": obj.tolist(),
                "shape": obj.shape,
                "dtype": str(obj.dtype)
            }
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def _strip_large_fields(self, obj):
        """
        Remove or shrink large fields before serialization to keep checkpoints light.
        We drop raw audio/features that are not needed for downstream steps.
        """
        heavy_keys = {"mel_features", "audio", "augmented_audio"}
        
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                if key in heavy_keys:
                    # Preserve minimal metadata if available
                    if isinstance(value, dict) and "shape" in value:
                        cleaned[f"{key}_shape"] = value.get("shape")
                    else:
                        cleaned[f"{key}_removed"] = True
                    continue
                cleaned[key] = self._strip_large_fields(value)
            return cleaned
        
        if isinstance(obj, list):
            return [self._strip_large_fields(item) for item in obj]
        
        return obj
    
    def _restore_from_json(self, obj):
        """Restore numpy arrays from JSON"""
        if isinstance(obj, dict):
            if obj.get("__numpy_array__"):
                return np.array(obj["data"], dtype=obj.get("dtype", "float32")).reshape(obj["shape"])
            else:
                return {key: self._restore_from_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._restore_from_json(item) for item in obj]
        else:
            return obj
    
    def check_memory_and_cleanup(self):
        """Check memory usage and cleanup if necessary"""
        if not self.memory_monitor.is_memory_safe():
            print("⚠️  High memory usage detected, forcing cleanup...")
            self.memory_monitor.force_cleanup()
            
            # Wait and check again
            time.sleep(1)
            if not self.memory_monitor.is_memory_safe():
                print("🔴 Memory usage still high after cleanup!")
                usage = self.memory_monitor.get_memory_usage()
                print(f"   System memory: {usage['system_percent']:.1f}%")
                print(f"   Process memory: {usage['process_mb']:.1f} MB")

class MTSBatchPipeline:
    """
    Memory-efficient MTS pipeline using batch processing
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize batch pipeline"""
        self.config = self._load_config(config_path)
        self.batch_processor = BatchProcessor(
            batch_size=self.config.get("batch_processing", {}).get("batch_size", 100),
            max_memory_percent=self.config.get("batch_processing", {}).get("max_memory_percent", 80.0),
            checkpoint_dir=self.config.get("batch_processing", {}).get("checkpoint_dir", "./checkpoints")
        )
        
        # Import components (will be imported as needed)
        self.components = {}
        
        # Setup logging
        self._setup_logging()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration with batch processing defaults"""
        default_config = {
            "batch_processing": {
                "batch_size": 100,
                "max_memory_percent": 80.0,
                "checkpoint_dir": "./checkpoints",
                "cleanup_frequency": 10,  # Cleanup every N batches
                "save_intermediate": True,
                "resume_from_checkpoint": True
            },
            "data": {
                "use_simulated_data": True,
                "target_sample_rate": 22050,
                "output_dir": "./outputs"
            },
            "augmentation": {
                "enabled": True,
                "augmentation_factor": 3,
                "batch_augmentation": True,  # Process augmentation in batches
                "save_audio": False,
                "audio_format": "wav",
                "preserve_original": True,
                "suppress_warnings": False
            },
            "text_generation": {
                "enabled": True,
                "batch_text_generation": True
            },
            "structure_processing": {
                "enabled": True,
                "structure_subset_size": 300
            }
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Merge configurations
            def merge_dicts(default, user):
                for key, value in user.items():
                    if isinstance(value, dict) and key in default:
                        merge_dicts(default[key], value)
                    else:
                        default[key] = value
            
            merge_dicts(default_config, user_config)
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging for batch processing"""
        import logging
        
        log_dir = Path(self.config["data"]["output_dir"]) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"batch_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def run_batch_pipeline(self) -> Dict:
        """Run the complete pipeline with batch processing"""
        
        self.logger.info("🚀 Starting MTS Batch Pipeline")
        self.logger.info(f"   Batch size: {self.batch_processor.batch_size}")
        self.logger.info(f"   Max memory: {self.batch_processor.memory_monitor.max_memory_percent}%")
        
        start_time = datetime.now()
        self.batch_processor.state["start_time"] = start_time.isoformat()
        
        try:
            organized_data = {}
            structure_data = {}
            # Step 1: Data Loading (in batches)
            if "data_loading" not in self.batch_processor.state["completed_steps"]:
                original_songs = self._batch_data_loading()
                self.batch_processor.state["completed_steps"].append("data_loading")
                self.batch_processor.state["total_songs"] = len(original_songs)
            else:
                self.logger.info("📂 Loading previous data loading results...")
                original_songs = self.batch_processor.load_batch_results("data_loading")
            
            # Step 2: Batch Augmentation
            if self.config["augmentation"]["enabled"]:
                if "augmentation" not in self.batch_processor.state["completed_steps"]:
                    augmented_songs = self._batch_augmentation(original_songs)
                    self.batch_processor.state["completed_steps"].append("augmentation")
                else:
                    self.logger.info("📂 Loading previous augmentation results...")
                    augmented_songs = self.batch_processor.load_batch_results("augmentation")
            else:
                augmented_songs = []
            
            # Step 3: Batch Text Generation
            if self.config["text_generation"]["enabled"]:
                if "text_generation" not in self.batch_processor.state["completed_steps"]:
                    all_labeled_songs = self._batch_text_generation(original_songs, augmented_songs)
                    self.batch_processor.state["completed_steps"].append("text_generation")
                else:
                    self.logger.info("📂 Loading previous text generation results...")
                    all_labeled_songs = self.batch_processor.load_batch_results("text_generation")
            else:
                all_labeled_songs = original_songs + augmented_songs
            
            # Step 4: Data Organization (can handle full dataset as it's just metadata)
            if "data_organization" not in self.batch_processor.state["completed_steps"]:
                organized_data = self._batch_data_organization(all_labeled_songs)
                self.batch_processor.state["completed_steps"].append("data_organization")
            
            # Step 5: Structure Processing (on subset)
            if self.config["structure_processing"]["enabled"]:
                if "structure_processing" not in self.batch_processor.state["completed_steps"]:
                    structure_data = self._batch_structure_processing(all_labeled_songs)
                    self.batch_processor.state["completed_steps"].append("structure_processing")
                else:
                    self.logger.info("📂 Loading previous structure processing results...")
                    loaded_structures = self.batch_processor.load_batch_results("structure_processing")
                    # Maintain consistent schema for downstream consumers
                    structure_data = {"annotated_songs": loaded_structures}
            else:
                structure_data = {}
            
            # Step 6: Final integration
            final_results = self._finalize_batch_results(
                original_songs, augmented_songs, all_labeled_songs, organized_data, structure_data
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            self.logger.info("✅ Batch pipeline completed successfully!")
            self.logger.info(f"   Total execution time: {execution_time:.2f} seconds")
            self.logger.info(f"   Peak memory usage: {self.batch_processor.memory_monitor.get_memory_usage()['process_mb']:.1f} MB")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Batch pipeline failed: {str(e)}")
            self.batch_processor.state["errors"].append(str(e))
            self.batch_processor.save_state()
            raise
    
    def _batch_data_loading(self) -> List[Dict]:
        """Load data in batches"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: BATCH DATA LOADING")
        self.logger.info("=" * 60)
        
        # Import data loader
        try:
            from .data_loader import MTSDataLoader
        except ImportError:
            from data_loader import MTSDataLoader
        
        loader = MTSDataLoader(
            cache_dir=self.config["data"].get("cache_dir", "./cache"),
            data_dir=self.config["data"].get("data_dir", "./data"),
            target_sr=self.config["data"].get("target_sample_rate", 22050)
        )
        
        # Load dataset info (this is lightweight)
        dataset_info = loader.load_ccmusic_dataset() if not self.config["data"].get("use_simulated_data") else loader._create_fallback_dataset()
        
        # Process in batches
        dataset = dataset_info["dataset"]
        all_processed_songs = []
        
        total_batches = (len(dataset) + self.batch_processor.batch_size - 1) // self.batch_processor.batch_size
        
        for batch_idx, batch_start in enumerate(tqdm(
            range(0, len(dataset), self.batch_processor.batch_size),
            desc="Processing data loading batches",
            total=total_batches
        )):
            batch_end = min(batch_start + self.batch_processor.batch_size, len(dataset))
            batch_data = [dataset[i] for i in range(batch_start, batch_end)]
            
            # Create temporary dataset info for this batch
            batch_dataset_info = {
                **dataset_info,
                "dataset": batch_data,
                "size": len(batch_data)
            }
            
            # Process this batch
            batch_songs = loader.extract_features_from_spectrograms(batch_dataset_info)
            
            # Save batch results
            self.batch_processor.save_batch_results(batch_idx, batch_songs, "data_loading")
            all_processed_songs.extend(batch_songs)
            
            # Memory management
            if batch_idx % self.config["batch_processing"]["cleanup_frequency"] == 0:
                self.batch_processor.check_memory_and_cleanup()
            
            # Update state
            self.batch_processor.state["current_batch"] = batch_idx
            self.batch_processor.state["processed_songs"] = len(all_processed_songs)
            
            if batch_idx % 10 == 0:  # Save state every 10 batches
                self.batch_processor.save_state()
        
        self.logger.info(f"✅ Data loading complete: {len(all_processed_songs)} songs")
        return all_processed_songs
    
    def _batch_augmentation(self, original_songs: List[Dict]) -> List[Dict]:
        """Perform augmentation in batches"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: BATCH AUGMENTATION")
        self.logger.info("=" * 60)
        
        # Import augmentation module
        try:
            from .augmentation import MTSAudioAugmentation
        except ImportError:
            from augmentation import MTSAudioAugmentation
        
        augmenter = MTSAudioAugmentation(
            sample_rate=self.config["data"]["target_sample_rate"],
            augmentation_factor=self.config["augmentation"]["augmentation_factor"],
            preserve_original=self.config["augmentation"].get("preserve_original", True),
            suppress_warnings=self.config["augmentation"].get("suppress_warnings", False)
        )

        # Apply user overrides for augmentation techniques if provided
        if "techniques" in self.config["augmentation"]:
            for technique, settings in self.config["augmentation"]["techniques"].items():
                if technique in augmenter.config:
                    augmenter.config[technique].update(settings)
        
        save_audio = self.config["augmentation"].get("save_audio", False)
        audio_format = self.config["augmentation"].get("audio_format", "wav")
        
        all_augmented_songs = []
        
        # Process in smaller batches for augmentation (memory intensive)
        aug_batch_size = max(1, self.batch_processor.batch_size // 3)  # Smaller batches for augmentation
        
        for batch_idx, batch in enumerate(tqdm(
            self.batch_processor.create_batches(original_songs, aug_batch_size),
            desc="Processing augmentation batches",
            total=(len(original_songs) + aug_batch_size - 1) // aug_batch_size
        )):
            
            # Process this batch
            batch_augmented = augmenter.augment_dataset(
                batch,
                output_dir=f"{self.config['data']['data_dir']}/augmented",
                save_audio=save_audio,
                audio_format=audio_format
            )
            
            # Save batch results
            self.batch_processor.save_batch_results(batch_idx, batch_augmented, "augmentation")
            all_augmented_songs.extend(batch_augmented)
            
            # Aggressive memory cleanup for augmentation
            self.batch_processor.check_memory_and_cleanup()
            
            # Update state
            self.batch_processor.state["current_batch"] = batch_idx
            if batch_idx % 5 == 0:
                self.batch_processor.save_state()
        
        self.logger.info(f"✅ Augmentation complete: {len(all_augmented_songs)} songs")
        return all_augmented_songs
    
    def _batch_text_generation(self, original_songs: List[Dict], augmented_songs: List[Dict]) -> List[Dict]:
        """Generate text prompts in batches"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: BATCH TEXT GENERATION")
        self.logger.info("=" * 60)
        
        # Import text generation module
        try:
            from .text_generation import MTSTextPromptGenerator
        except ImportError:
            from text_generation import MTSTextPromptGenerator
        
        generator = MTSTextPromptGenerator(
            model_name=self.config["text_generation"]["model_name"],
            cache_dir=self.config["data"]["cache_dir"]
        )
        
        # Generate prompt pool once (this is lightweight)
        prompt_pool = generator.generate_prompt_pool(
            num_prompts=self.config["text_generation"]["prompt_pool_size"]
        )
        
        # Process original songs in batches
        all_labeled_songs = []
        
        # Original songs
        for batch_idx, batch in enumerate(tqdm(
            self.batch_processor.create_batches(original_songs),
            desc="Processing original songs text generation",
            total=(len(original_songs) + self.batch_processor.batch_size - 1) // self.batch_processor.batch_size
        )):
            
            labeled_batch = generator.assign_prompts_to_songs(batch, prompt_pool)
            
            # Save batch results
            self.batch_processor.save_batch_results(
                batch_idx, labeled_batch, "text_generation_original"
            )
            all_labeled_songs.extend(labeled_batch)
            
            if batch_idx % self.config["batch_processing"]["cleanup_frequency"] == 0:
                self.batch_processor.check_memory_and_cleanup()
        
        # Augmented songs
        if augmented_songs:
            for batch_idx, batch in enumerate(tqdm(
                self.batch_processor.create_batches(augmented_songs),
                desc="Processing augmented songs text generation",
                total=(len(augmented_songs) + self.batch_processor.batch_size - 1) // self.batch_processor.batch_size
            )):
                
                labeled_batch = generator.create_augmented_prompts(all_labeled_songs[:len(original_songs)], batch)
                
                # Save batch results
                self.batch_processor.save_batch_results(
                    batch_idx, labeled_batch, "text_generation_augmented"
                )
                all_labeled_songs.extend(labeled_batch)
                
                if batch_idx % self.config["batch_processing"]["cleanup_frequency"] == 0:
                    self.batch_processor.check_memory_and_cleanup()
        
        # Save all text generation results
        self.batch_processor.save_batch_results(0, all_labeled_songs, "text_generation")
        
        self.logger.info(f"✅ Text generation complete: {len(all_labeled_songs)} labeled songs")
        return all_labeled_songs
    
    def _batch_data_organization(self, all_songs: List[Dict]) -> Dict:
        """Organize data (lightweight operation)"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: DATA ORGANIZATION")
        self.logger.info("=" * 60)
        
        # Import organization module
        try:
            from .data_organization import MTSDataOrganizer
        except ImportError:
            from data_organization import MTSDataOrganizer
        
        organizer = MTSDataOrganizer(
            train_ratio=self.config["data_organization"]["train_ratio"],
            val_ratio=self.config["data_organization"]["val_ratio"],
            test_ratio=self.config["data_organization"]["test_ratio"],
            random_state=self.config["data_organization"]["random_state"]
        )
        
        # This is lightweight since it only reorganizes metadata
        split_data = organizer.stratified_split_by_genre(all_songs)
        configurations = organizer.create_training_configurations(split_data)
        statistics = organizer.analyze_dataset_statistics(configurations)
        
        # Save organization results
        saved_files = organizer.save_configurations(
            configurations,
            statistics,
            f"{self.config['data']['output_dir']}/configurations"
        )
        
        organized_data = {
            "split_data": split_data,
            "configurations": configurations,
            "statistics": statistics,
            "saved_files": saved_files
        }
        
        self.logger.info(f"✅ Data organization complete: {len(configurations)} configurations")
        return organized_data
    
    def _batch_structure_processing(self, all_songs: List[Dict]) -> Dict:
        """Process structure annotations on subset"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: STRUCTURE PROCESSING")
        self.logger.info("=" * 60)
        
        # Import structure processing module
        try:
            from .structure_processing import MTSStructureProcessor
        except ImportError:
            from structure_processing import MTSStructureProcessor
        
        processor = MTSStructureProcessor(
            structure_subset_size=self.config["structure_processing"]["structure_subset_size"]
        )
        
        # Select subset (lightweight)
        selected_songs = processor.select_songs_for_annotation(all_songs)
        
        # Process in smaller batches
        struct_batch_size = max(1, self.batch_processor.batch_size // 2)
        all_annotated_songs = []
        
        for batch_idx, batch in enumerate(tqdm(
            self.batch_processor.create_batches(selected_songs, struct_batch_size),
            desc="Processing structure annotation batches",
            total=(len(selected_songs) + struct_batch_size - 1) // struct_batch_size
        )):
            
            annotated_batch = processor.generate_structure_annotations(batch)
            all_annotated_songs.extend(annotated_batch)
            
            # Save batch results
            self.batch_processor.save_batch_results(
                batch_idx, annotated_batch, "structure_processing"
            )
            
            if batch_idx % 5 == 0:
                self.batch_processor.check_memory_and_cleanup()
        
        # Analyze patterns on full set (lightweight)
        patterns = processor.analyze_structure_patterns(all_annotated_songs)
        conditioning_data = processor.create_structure_conditioning_data(all_annotated_songs)
        evaluation_metrics = processor.create_evaluation_metrics(all_annotated_songs, patterns)
        
        # Save structure results
        saved_files = processor.save_structure_data(
            all_annotated_songs, patterns, conditioning_data, evaluation_metrics,
            f"{self.config['data']['output_dir']}/structure"
        )
        
        structure_data = {
            "annotated_songs": all_annotated_songs,
            "patterns": patterns,
            "conditioning_data": conditioning_data,
            "evaluation_metrics": evaluation_metrics,
            "saved_files": saved_files
        }
        
        self.logger.info(f"✅ Structure processing complete: {len(all_annotated_songs)} annotated songs")
        return structure_data
    
    def _create_final_csv(self, all_songs: List[Dict], structure_data: Dict) -> Path:
        """Create final comprehensive CSV dataset for batch pipeline"""
        annotated_songs = []
        if isinstance(structure_data, dict):
            annotated_songs = structure_data.get("annotated_songs", [])
        elif isinstance(structure_data, list):
            annotated_songs = structure_data
        
        structured_songs = {s.get("id"): s for s in annotated_songs if isinstance(s, dict)}
        
        csv_data = []
        for song in all_songs:
            if not isinstance(song, dict):
                continue
            song_id = song.get("id", "")
            structure_info = structured_songs.get(song_id, {})
            
            # Get audio path - check both fields (augmented uses audio_file_path, original uses audio_path)
            audio_path = song.get("audio_file_path", "") or song.get("audio_path", "")

            row = {
                "id": song_id,
                "original_id": song.get("original_id", ""),
                "title": song.get("title", ""),
                "artist": song.get("artist", ""),
                "genre": song.get("genre", ""),
                "duration": song.get("duration", 0),
                "language": song.get("language", ""),
                "text_prompt": song.get("text_prompt", ""),
                "is_augmented": song.get("is_augmented", False),
                "augmentation_quality": song.get("augmentation_metadata", {}).get("quality_category", "original"),
                "has_structure": bool(structure_info),
                "structure_template": structure_info.get("structure_template", ""),
                "total_sections": structure_info.get("total_sections", 0),
                "structure_coherence": structure_info.get("structure_coherence_score", 0.0),
                "prompt_length": len(song.get("text_prompt", "").split()),
                "sample_rate": self.config["data"]["target_sample_rate"],
                "processing_date": datetime.now().isoformat(),
                "audio_path": audio_path
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_file = Path(self.config["data"]["output_dir"]) / "mts_final_dataset.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"💾 Final CSV saved: {csv_file}")
        return csv_file
    
    def _finalize_batch_results(self,
                                original_songs: List[Dict],
                                augmented_songs: List[Dict],
                                all_labeled_songs: List[Dict],
                                organized_data: Dict,
                                structure_data: Dict) -> Dict:
        """Finalize and summarize all batch results"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 6: FINALIZING RESULTS")
        self.logger.info("=" * 60)

        n_original = len(original_songs)
        n_augmented = len(augmented_songs)
        n_labeled = len(all_labeled_songs)

        final_csv = self._create_final_csv(all_labeled_songs, structure_data)
        
        # Create comprehensive summary
        summary = {
            "batch_processing_summary": {
                "batch_size_used": self.batch_processor.batch_size,
                "total_batches_processed": self.batch_processor.state["current_batch"] + 1,
                "memory_management": "enabled",
                "checkpoint_recovery": "enabled"
            },
            "data_summary": {
                "original_songs": n_original,
                "augmented_songs": n_augmented,
                "total_labeled_songs": n_labeled,
                "expansion_factor": n_labeled / n_original if n_original > 0 else 0
            },
            "processing_state": self.batch_processor.state,
            "peak_memory_usage": self.batch_processor.memory_monitor.get_memory_usage(),
            "output_files": {
                "configurations": f"{self.config['data']['output_dir']}/configurations",
                "structure": f"{self.config['data']['output_dir']}/structure",
                "checkpoints": str(self.batch_processor.checkpoint_dir),
                "final_csv": str(final_csv)
            },
            "organized_data": {
                "statistics": organized_data.get("statistics", {}),
                "saved_files": organized_data.get("saved_files", {})
            },
            "status": "completed"
        }
        
        # Save final summary
        summary_file = Path(self.config["data"]["output_dir"]) / "batch_pipeline_summary.json"
        summary["output_files"]["summary_file"] = str(summary_file)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"💾 Final summary saved: {summary_file}")
        return summary

# Update config with batch processing options
def create_batch_config():
    """Create configuration optimized for batch processing"""
    return {
        "batch_processing": {
            "batch_size": 100,              # Songs per batch
            "max_memory_percent": 75.0,     # Maximum memory usage
            "checkpoint_dir": "./checkpoints",
            "cleanup_frequency": 10,        # Cleanup every N batches
            "save_intermediate": True,
            "resume_from_checkpoint": True
        },
        "data": {
            "use_simulated_data": True,
            "target_sample_rate": 22050,
            "output_dir": "./outputs",
            "cache_dir": "./cache",
            "data_dir": "./data"
        },
        "augmentation": {
            "enabled": True,
            "augmentation_factor": 3,
            "batch_augmentation": True,
            "save_audio": True,
            "audio_format": "mp3",
            "preserve_original": True,
            "suppress_warnings": True
        },
        "text_generation": {
            "enabled": True,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "prompt_pool_size": 1000,
            "batch_text_generation": True
        },
        "data_organization": {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "random_state": 42,
            "stratify_by_genre": True
        },
        "structure_processing": {
            "enabled": True,
            "structure_subset_size": 300,
            "create_conditioning_data": True
        }
    }

def main():
    """Test batch pipeline"""
    
    # Create batch config
    config = create_batch_config()
    
    # Save config
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    with open("config/batch_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run batch pipeline
    pipeline = MTSBatchPipeline("config/batch_config.yaml")
    results = pipeline.run_batch_pipeline()
    
    print(f"\n🎉 Batch Pipeline Complete!")
    print(f"Status: {results['status']}")
    print(f"Original songs: {results['data_summary']['original_songs']:,}")
    print(f"Final dataset: {results['data_summary']['total_labeled_songs']:,}")
    print(f"Expansion factor: {results['data_summary']['expansion_factor']:.1f}x")
    print(f"Peak memory: {results['peak_memory_usage']['process_mb']:.1f} MB")

# Alias for backward compatibility
MTSBatchProcessor = MTSBatchPipeline

if __name__ == "__main__":
    main()
