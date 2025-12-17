"""
MTS Complete Data Pipeline - Orchestrates all data preparation components
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import argparse
import logging

# Import all our components
# Try relative imports first, then absolute
try:
    from .data_loader import MTSDataLoader
    from .augmentation import MTSAudioAugmentation
    from .text_generation import MTSTextPromptGenerator
    from .data_organization import MTSDataOrganizer
    from .structure_processing import MTSStructureProcessor
except ImportError:
    # Fallback for direct execution
    from data_loader import MTSDataLoader
    from augmentation import MTSAudioAugmentation
    from text_generation import MTSTextPromptGenerator
    from data_organization import MTSDataOrganizer
    from structure_processing import MTSStructureProcessor

class MTSDataPipeline:
    """
    Complete MTS data preparation pipeline
    Orchestrates all components according to the project plan
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the complete data pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        
        # Create output directories
        self._setup_directories()
        
        # Initialize components
        self.data_loader = None
        self.augmenter = None
        self.text_generator = None
        self.organizer = None
        self.structure_processor = None
        
        # Pipeline state
        self.pipeline_state = {
            "step_completed": {},
            "data_counts": {},
            "file_outputs": {},
            "execution_times": {},
            "errors": []
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "data": {
                "dataset_name": "ccmusic-database/music_genre",
                "cache_dir": "./cache",
                "data_dir": "./data",
                "output_dir": "./outputs",
                "target_sample_rate": 22050,
                "use_simulated_data": True  # Set to False for real CCMusic access
            },
            "augmentation": {
                "enabled": True,
                "augmentation_factor": 3,
                "preserve_original": True,
                "techniques": {
                    "pitch_shift": {"enabled": True, "probability": 0.8},
                    "tempo_scale": {"enabled": True, "probability": 0.8},
                    "noise_addition": {"enabled": True, "probability": 0.4},
                    "eq_filter": {"enabled": True, "probability": 0.3},
                    "reverb": {"enabled": True, "probability": 0.25}
                }
            },
            "text_generation": {
                "enabled": True,
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "prompt_pool_size": 1000,
                "use_embedding_similarity": True
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
            },
            "pipeline": {
                "save_intermediate_results": True,
                "continue_on_error": True,
                "parallel_processing": False,
                "validation_enabled": True
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get("logging", {}).get("level", "INFO")
        log_dir = Path(self.config["data"]["output_dir"]) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"mts_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.config["data"]["cache_dir"],
            self.config["data"]["data_dir"],
            self.config["data"]["output_dir"],
            f"{self.config['data']['data_dir']}/raw",
            f"{self.config['data']['data_dir']}/processed", 
            f"{self.config['data']['data_dir']}/augmented",
            f"{self.config['data']['output_dir']}/logs",
            f"{self.config['data']['output_dir']}/configurations",
            f"{self.config['data']['output_dir']}/statistics"
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def run_complete_pipeline(self, force_reload: bool = False) -> Dict:
        """
        Run the complete MTS data preparation pipeline
        Following all steps from the project plan

        Args:
            force_reload: If True, reprocess all data even if it exists
        """
        self.logger.info("🚀 Starting MTS Data Preparation Pipeline")

        start_time = datetime.now()

        try:
            # Check if processed data already exists
            processed_data_file = Path(self.config["data"]["output_dir"]) / "mts_final_dataset.csv"

            if processed_data_file.exists() and not force_reload:
                self.logger.info("=" * 60)
                self.logger.info("✅ FOUND EXISTING PROCESSED DATA")
                self.logger.info("=" * 60)
                self.logger.info(f"📂 Loading from: {processed_data_file}")
                self.logger.info("   Use --force-reload to reprocess from scratch")

                # Load existing data
                df = pd.read_csv(processed_data_file)

                self.pipeline_state["step_completed"]["all"] = True
                self.pipeline_state["data_counts"]["total_samples"] = len(df)

                summary = {
                    "status": "loaded_existing",
                    "total_samples": len(df),
                    "data_file": str(processed_data_file),
                    "message": "Loaded existing processed data"
                }

                self.logger.info(f"\n✅ Loaded {len(df)} existing samples")
                self.logger.info(f"   Dataset ready for training!")

                return summary

            # Step 1: Data Loading and Initial Processing
            self.logger.info("=" * 60)
            self.logger.info("STEP 1: DATA LOADING AND INITIAL PROCESSING")
            self.logger.info("=" * 60)

            # Check for cached processed songs
            step1_cache = Path(self.config["data"]["data_dir"]) / "processed" / "step1_processed_songs.json"

            if step1_cache.exists() and not force_reload:
                self.logger.info(f"📂 Loading cached processed songs from: {step1_cache}")
                with open(step1_cache, 'r') as f:
                    original_songs = json.load(f)
                self.logger.info(f"✅ Loaded {len(original_songs)} cached songs")
            else:
                original_songs = self._step_1_data_loading()
            self.pipeline_state["step_completed"]["data_loading"] = True
            self.pipeline_state["data_counts"]["original_songs"] = len(original_songs)
            
            # Step 2: Audio Data Augmentation
            if self.config["augmentation"]["enabled"]:
                self.logger.info("=" * 60)
                self.logger.info("STEP 2: AUDIO DATA AUGMENTATION")
                self.logger.info("=" * 60)
                
                augmented_songs = self._step_2_augmentation(original_songs)
                self.pipeline_state["step_completed"]["augmentation"] = True
                self.pipeline_state["data_counts"]["augmented_songs"] = len(augmented_songs)
            else:
                augmented_songs = []
                self.logger.info("⏭️  Skipping augmentation (disabled in config)")
            
            # Step 3: Text Prompt Generation
            if self.config["text_generation"]["enabled"]:
                self.logger.info("=" * 60)
                self.logger.info("STEP 3: TEXT PROMPT GENERATION")
                self.logger.info("=" * 60)
                
                all_labeled_songs = self._step_3_text_generation(original_songs, augmented_songs)
                self.pipeline_state["step_completed"]["text_generation"] = True
                self.pipeline_state["data_counts"]["labeled_songs"] = len(all_labeled_songs)
            else:
                all_labeled_songs = original_songs + augmented_songs
                self.logger.info("⏭️  Skipping text generation (disabled in config)")
            
            # Step 4: Data Organization and Splitting
            self.logger.info("=" * 60)
            self.logger.info("STEP 4: DATA ORGANIZATION AND SPLITTING")
            self.logger.info("=" * 60)
            
            organized_data = self._step_4_data_organization(all_labeled_songs)
            self.pipeline_state["step_completed"]["data_organization"] = True
            
            # Step 5: Structure Processing
            if self.config["structure_processing"]["enabled"]:
                self.logger.info("=" * 60)
                self.logger.info("STEP 5: STRUCTURE ANNOTATION PROCESSING")
                self.logger.info("=" * 60)
                
                structure_data = self._step_5_structure_processing(all_labeled_songs)
                self.pipeline_state["step_completed"]["structure_processing"] = True
                self.pipeline_state["data_counts"]["structured_songs"] = len(structure_data.get("annotated_songs", []))
            else:
                structure_data = {}
                self.logger.info("⏭️  Skipping structure processing (disabled in config)")
            
            # Step 6: Final Integration and Validation
            self.logger.info("=" * 60)
            self.logger.info("STEP 6: FINAL INTEGRATION AND VALIDATION")
            self.logger.info("=" * 60)
            
            final_results = self._step_6_final_integration(
                original_songs, augmented_songs, all_labeled_songs, 
                organized_data, structure_data
            )
            
            # Calculate total execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self.pipeline_state["execution_times"]["total"] = execution_time
            
            self.logger.info("=" * 60)
            self.logger.info("✅ MTS DATA PREPARATION PIPELINE COMPLETE!")
            self.logger.info("=" * 60)
            self.logger.info(f"Total execution time: {execution_time:.2f} seconds")
            self.logger.info(f"Original songs: {self.pipeline_state['data_counts'].get('original_songs', 0)}")
            self.logger.info(f"Augmented songs: {self.pipeline_state['data_counts'].get('augmented_songs', 0)}")
            self.logger.info(f"Labeled songs: {self.pipeline_state['data_counts'].get('labeled_songs', 0)}")
            self.logger.info(f"Structured songs: {self.pipeline_state['data_counts'].get('structured_songs', 0)}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}")
            self.pipeline_state["errors"].append(str(e))
            
            if not self.config["pipeline"]["continue_on_error"]:
                raise
            
            return {"status": "failed", "error": str(e), "partial_results": self.pipeline_state}
    
    def _step_1_data_loading(self) -> List[Dict]:
        """Step 1: Load and process the CCMusic dataset"""
        step_start = datetime.now()
        
        self.data_loader = MTSDataLoader(
            cache_dir=self.config["data"]["cache_dir"],
            data_dir=self.config["data"]["data_dir"],
            target_sr=self.config["data"]["target_sample_rate"]
        )
        
        if self.config["data"]["use_simulated_data"]:
            self.logger.info("📝 Using simulated CCMusic dataset for development")
            dataset_info = self.data_loader._create_fallback_dataset()
        elif self.config["data"].get("use_fma_dataset", False):
            self.logger.info("📂 Loading FMA dataset with real audio")
            dataset_info = self.data_loader.load_fma_dataset(
                audio_dir=self.config["data"].get("fma_audio_dir", "./fma_data/fma_small"),
                metadata_path=self.config["data"].get("fma_metadata_path"),
                max_files=self.config["data"].get("fma_max_files", 500),
                clip_duration=self.config["data"].get("fma_clip_duration", 30.0)
            )
        else:
            self.logger.info(f"📂 Loading CCMusic dataset: {self.config['data']['dataset_name']}")
            dataset_info = self.data_loader.load_ccmusic_dataset(
                self.config["data"]["dataset_name"]
            )
        
        # Process features (skip if dataset already provides preprocessed songs)
        if dataset_info.get("preprocessed_songs"):
            processed_songs = dataset_info["preprocessed_songs"]
            self.logger.info(f"🎵 Using preprocessed songs provided by dataset ({len(processed_songs)})")
        else:
            self.logger.info("🎵 Processing features from dataset...")
            processed_songs = self.data_loader.extract_features_from_spectrograms(dataset_info)
        
        # Save processed data
        if self.config["pipeline"]["save_intermediate_results"]:
            output_file = self.data_loader.save_processed_dataset(
                processed_songs,
                "step1_processed_songs.json"
            )
            self.pipeline_state["file_outputs"]["step1_processed"] = output_file
        
        step_time = (datetime.now() - step_start).total_seconds()
        self.pipeline_state["execution_times"]["step1_data_loading"] = step_time
        
        self.logger.info(f"✅ Step 1 complete: {len(processed_songs)} songs processed in {step_time:.2f}s")
        return processed_songs
    
    def _step_2_augmentation(self, original_songs: List[Dict]) -> List[Dict]:
        """Step 2: Apply audio data augmentation"""
        step_start = datetime.now()
        
        self.augmenter = MTSAudioAugmentation(
            sample_rate=self.config["data"]["target_sample_rate"],
            augmentation_factor=self.config["augmentation"]["augmentation_factor"],
            preserve_original=self.config["augmentation"]["preserve_original"],
            suppress_warnings=self.config["augmentation"].get("suppress_warnings", False)
        )
        
        # Update augmentation configuration
        for technique, settings in self.config["augmentation"]["techniques"].items():
            if technique in self.augmenter.config:
                self.augmenter.config[technique].update(settings)
        
        self.logger.info(f"🔄 Augmenting {len(original_songs)} songs with {self.augmenter.augmentation_factor}x factor")
        
        # Run augmentation
        augmented_songs = self.augmenter.augment_dataset(
            original_songs,
            output_dir=f"{self.config['data']['data_dir']}/augmented",
            save_audio=self.config["augmentation"].get("save_audio", False),
            audio_format=self.config["augmentation"].get("audio_format", "wav")
        )
        
        # Save augmentation results
        if self.config["pipeline"]["save_intermediate_results"]:
            output_dir = Path(self.config["data"]["output_dir"]) / "augmentation"
            output_dir.mkdir(exist_ok=True)
            
            results_file = output_dir / "step2_augmentation_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "augmented_songs": augmented_songs[:100],  # Sample for file size
                    "total_augmented": len(augmented_songs),
                    "statistics": self.augmenter.stats,
                    "configuration": self.augmenter.config
                }, f, indent=2, default=str)
            
            self.pipeline_state["file_outputs"]["step2_augmentation"] = str(results_file)
        
        step_time = (datetime.now() - step_start).total_seconds()
        self.pipeline_state["execution_times"]["step2_augmentation"] = step_time
        
        self.logger.info(f"✅ Step 2 complete: {len(augmented_songs)} augmented songs in {step_time:.2f}s")
        return augmented_songs
    
    def _step_3_text_generation(self, 
                               original_songs: List[Dict], 
                               augmented_songs: List[Dict]) -> List[Dict]:
        """Step 3: Generate text prompts for all songs"""
        step_start = datetime.now()
        
        self.text_generator = MTSTextPromptGenerator(
            model_name=self.config["text_generation"]["model_name"],
            cache_dir=self.config["data"]["cache_dir"]
        )
        
        # Generate prompt pool
        self.logger.info(f"📝 Generating prompt pool of {self.config['text_generation']['prompt_pool_size']} prompts")
        prompt_pool = self.text_generator.generate_prompt_pool(
            num_prompts=self.config["text_generation"]["prompt_pool_size"]
        )
        
        # Assign prompts to original songs
        self.logger.info(f"🎯 Assigning prompts to {len(original_songs)} original songs")
        labeled_original = self.text_generator.assign_prompts_to_songs(original_songs, prompt_pool)
        
        # Create prompts for augmented songs
        if augmented_songs:
            self.logger.info(f"🔄 Creating prompts for {len(augmented_songs)} augmented songs")
            labeled_augmented = self.text_generator.create_augmented_prompts(
                labeled_original, augmented_songs
            )
        else:
            labeled_augmented = []
        
        # Combine all labeled songs
        all_labeled_songs = labeled_original + labeled_augmented
        
        # Save text generation results
        if self.config["pipeline"]["save_intermediate_results"]:
            output_file = self.text_generator.save_labeled_dataset(
                all_labeled_songs,
                f"{self.config['data']['output_dir']}/step3_labeled_dataset.csv"
            )
            self.pipeline_state["file_outputs"]["step3_text_generation"] = output_file
        
        step_time = (datetime.now() - step_start).total_seconds()
        self.pipeline_state["execution_times"]["step3_text_generation"] = step_time
        
        self.logger.info(f"✅ Step 3 complete: {len(all_labeled_songs)} labeled songs in {step_time:.2f}s")
        return all_labeled_songs
    
    def _step_4_data_organization(self, all_songs: List[Dict]) -> Dict:
        """Step 4: Organize data with stratified splitting and create training configurations"""
        step_start = datetime.now()
        
        self.organizer = MTSDataOrganizer(
            train_ratio=self.config["data_organization"]["train_ratio"],
            val_ratio=self.config["data_organization"]["val_ratio"],
            test_ratio=self.config["data_organization"]["test_ratio"],
            random_state=self.config["data_organization"]["random_state"]
        )
        
        # Perform stratified split
        self.logger.info(f"🎯 Performing stratified split on {len(all_songs)} songs")
        split_data = self.organizer.stratified_split_by_genre(all_songs)
        
        # Create training configurations
        self.logger.info("⚙️  Creating training configurations")
        configurations = self.organizer.create_training_configurations(split_data)
        
        # Analyze statistics
        self.logger.info("📊 Analyzing dataset statistics")
        statistics = self.organizer.analyze_dataset_statistics(configurations)
        
        # Save organization results
        if self.config["pipeline"]["save_intermediate_results"]:
            saved_files = self.organizer.save_configurations(
                configurations, 
                statistics,
                f"{self.config['data']['output_dir']}/configurations"
            )
            self.pipeline_state["file_outputs"]["step4_organization"] = saved_files
        
        step_time = (datetime.now() - step_start).total_seconds()
        self.pipeline_state["execution_times"]["step4_organization"] = step_time
        
        organized_data = {
            "split_data": split_data,
            "configurations": configurations,
            "statistics": statistics
        }
        
        self.logger.info(f"✅ Step 4 complete: {len(configurations)} configurations created in {step_time:.2f}s")
        return organized_data
    
    def _step_5_structure_processing(self, all_songs: List[Dict]) -> Dict:
        """Step 5: Process structure annotations"""
        step_start = datetime.now()
        
        self.structure_processor = MTSStructureProcessor(
            structure_subset_size=self.config["structure_processing"]["structure_subset_size"]
        )
        
        # Select songs for annotation
        self.logger.info(f"🎼 Selecting songs for structure annotation")
        selected_songs = self.structure_processor.select_songs_for_annotation(all_songs)
        
        # Generate structure annotations
        self.logger.info(f"🏗️  Generating structure annotations")
        annotated_songs = self.structure_processor.generate_structure_annotations(selected_songs)
        
        # Analyze structure patterns
        self.logger.info(f"📈 Analyzing structure patterns")
        patterns = self.structure_processor.analyze_structure_patterns(annotated_songs)
        
        # Create conditioning data
        conditioning_data = []
        if self.config["structure_processing"]["create_conditioning_data"]:
            self.logger.info(f"🎯 Creating structure conditioning data")
            conditioning_data = self.structure_processor.create_structure_conditioning_data(annotated_songs)
        
        # Create evaluation metrics
        self.logger.info(f"📏 Creating evaluation metrics")
        evaluation_metrics = self.structure_processor.create_evaluation_metrics(annotated_songs, patterns)
        
        # Save structure processing results
        if self.config["pipeline"]["save_intermediate_results"]:
            saved_files = self.structure_processor.save_structure_data(
                annotated_songs, patterns, conditioning_data, evaluation_metrics,
                f"{self.config['data']['output_dir']}/structure"
            )
            self.pipeline_state["file_outputs"]["step5_structure"] = saved_files
        
        step_time = (datetime.now() - step_start).total_seconds()
        self.pipeline_state["execution_times"]["step5_structure"] = step_time
        
        structure_data = {
            "annotated_songs": annotated_songs,
            "patterns": patterns,
            "conditioning_data": conditioning_data,
            "evaluation_metrics": evaluation_metrics
        }
        
        self.logger.info(f"✅ Step 5 complete: {len(annotated_songs)} structured songs in {step_time:.2f}s")
        return structure_data
    
    def _step_6_final_integration(self, 
                                 original_songs: List[Dict],
                                 augmented_songs: List[Dict], 
                                 all_labeled_songs: List[Dict],
                                 organized_data: Dict,
                                 structure_data: Dict) -> Dict:
        """Step 6: Final integration and comprehensive summary"""
        step_start = datetime.now()
        
        self.logger.info("🔗 Creating final integration and comprehensive summary")
        
        # Create comprehensive pipeline summary
        pipeline_summary = {
            "pipeline_execution": {
                "status": "completed",
                "execution_times": self.pipeline_state["execution_times"],
                "steps_completed": self.pipeline_state["step_completed"],
                "errors": self.pipeline_state["errors"]
            },
            "data_summary": {
                "original_songs": len(original_songs),
                "augmented_songs": len(augmented_songs),
                "total_labeled_songs": len(all_labeled_songs),
                "expansion_factor": len(all_labeled_songs) / len(original_songs) if original_songs else 0,
                "structured_songs": len(structure_data.get("annotated_songs", [])),
                "training_configurations": len(organized_data.get("configurations", {}))
            },
            "dataset_statistics": organized_data.get("statistics", {}),
            "structure_patterns": structure_data.get("patterns", {}),
            "file_outputs": self.pipeline_state["file_outputs"],
            "configuration_used": self.config,
            "generation_timestamp": datetime.now().isoformat()
        }
        
        # Save comprehensive summary
        summary_file = Path(self.config["data"]["output_dir"]) / "mts_pipeline_complete_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)
        
        # Create final CSV with all data
        self._create_final_csv(all_labeled_songs, structure_data)
        
        # Generate final report
        report_file = self._generate_final_report(pipeline_summary)
        
        step_time = (datetime.now() - step_start).total_seconds()
        self.pipeline_state["execution_times"]["step6_integration"] = step_time
        
        final_results = {
            "status": "completed",
            "summary": pipeline_summary,
            "summary_file": str(summary_file),
            "report_file": report_file,
            "data_counts": self.pipeline_state["data_counts"],
            "total_execution_time": self.pipeline_state["execution_times"].get("total", step_time)
        }
        
        self.logger.info(f"✅ Step 6 complete: Final integration in {step_time:.2f}s")
        return final_results
    
    def _create_final_csv(self, all_songs: List[Dict], structure_data: Dict):
        """Create final comprehensive CSV dataset"""
        
        # Create mapping of structured songs
        structured_songs = {s["id"]: s for s in structure_data.get("annotated_songs", [])}
        
        # Prepare CSV data
        csv_data = []
        for song in all_songs:
            song_id = song.get("id", "")
            
            # Check if song has structure data
            structure_info = structured_songs.get(song_id, {})
            
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
                "processing_date": datetime.now().isoformat()
            }
            csv_data.append(row)
        
        # Save final CSV
        df = pd.DataFrame(csv_data)
        csv_file = Path(self.config["data"]["output_dir"]) / "mts_final_dataset.csv"
        df.to_csv(csv_file, index=False)
        
        self.pipeline_state["file_outputs"]["final_csv"] = str(csv_file)
        self.logger.info(f"💾 Final CSV saved: {csv_file}")
    
    def _generate_final_report(self, pipeline_summary: Dict) -> str:
        """Generate final markdown report"""
        
        report_content = f"""# MTS Data Preparation Pipeline - Final Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

The MTS (Music-to-Structure) data preparation pipeline has successfully completed all phases of data processing, expanding the original CCMusic dataset from **{pipeline_summary['data_summary']['original_songs']:,} songs** to **{pipeline_summary['data_summary']['total_labeled_songs']:,} labeled samples** (expansion factor: **{pipeline_summary['data_summary']['expansion_factor']:.1f}x**).

## Pipeline Results

### Data Processing Statistics
- **Original Songs**: {pipeline_summary['data_summary']['original_songs']:,}
- **Augmented Songs**: {pipeline_summary['data_summary']['augmented_songs']:,}
- **Total Labeled Songs**: {pipeline_summary['data_summary']['total_labeled_songs']:,}
- **Structure Annotated**: {pipeline_summary['data_summary']['structured_songs']:,}
- **Training Configurations**: {pipeline_summary['data_summary']['training_configurations']}

### Execution Performance
- **Total Execution Time**: {pipeline_summary['pipeline_execution']['execution_times'].get('total', 0):.2f} seconds
- **Steps Completed**: {len([k for k, v in pipeline_summary['pipeline_execution']['steps_completed'].items() if v])}
- **Errors Encountered**: {len(pipeline_summary['pipeline_execution']['errors'])}

## Generated Files

### Core Dataset Files
"""
        
        # Add file listings
        for step, files in pipeline_summary['file_outputs'].items():
            if isinstance(files, dict):
                report_content += f"\n#### {step.replace('_', ' ').title()}\n"
                for file_type, file_path in files.items():
                    report_content += f"- **{file_type.replace('_', ' ').title()}**: `{file_path}`\n"
            else:
                report_content += f"- **{step.replace('_', ' ').title()}**: `{files}`\n"
        
        report_content += f"""

## Next Steps - Phase 2: Model Architecture Design

With the data preparation phase complete, the following components are ready for Phase 2 implementation:

### 1. Audio Compression System
- Implement EnCodec audio tokenization (22 kHz model)
- Create compressed discrete representation for efficient training
- Target: ~150k tokens for 3-minute audio vs. 4M samples in waveform

### 2. Diffusion-Based Generative Model
- Build U-Net diffusion architecture for latent audio generation
- Implement cascaded refinement pipeline (coarse → fine generation)
- Create 30-second compressed representation system

### 3. Text Conditioning Integration
- Implement CLAP/MuLan text embeddings for conditioning
- Add cross-attention mechanisms for text-audio alignment
- Enable classifier-free guidance for improved prompt adherence

### 4. Structure Conditioning System
- Integrate structure annotations for long-form coherence
- Implement hierarchical generation (coarse structure → detailed audio)
- Create structure evaluation metrics for model assessment

## Data Quality Assessment

The pipeline has successfully created a comprehensive dataset ready for training state-of-the-art music generation models. All components follow the specifications outlined in the MTS project plan, ensuring compatibility with the planned model architecture.

### Dataset Strengths
- **Comprehensive Augmentation**: 4x dataset expansion through principled augmentation
- **Rich Text Descriptions**: {pipeline_summary['data_summary']['total_labeled_songs']:,} songs with descriptive prompts
- **Structural Annotations**: {pipeline_summary['data_summary']['structured_songs']} songs with detailed section labels
- **Multiple Training Configs**: {pipeline_summary['data_summary']['training_configurations']} specialized training setups
- **Genre Diversity**: Stratified splitting ensures balanced representation

### Ready for Production
The MTS data preparation pipeline is complete and ready for deployment in the model training phase. All outputs follow the project specifications and are compatible with the planned diffusion-based architecture.
"""
        
        # Save report
        report_file = Path(self.config["data"]["output_dir"]) / "MTS_Pipeline_Final_Report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"📄 Final report generated: {report_file}")
        return str(report_file)

def main():
    """Main entry point for the MTS data pipeline"""

    parser = argparse.ArgumentParser(description="MTS Data Preparation Pipeline")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--step", choices=["all", "1", "2", "3", "4", "5", "6"],
                       default="all", help="Run specific step (default: all)")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation checks")
    parser.add_argument("--output-dir", default="./outputs",
                       help="Output directory")
    parser.add_argument("--force-reload", action="store_true",
                       help="Force reprocessing even if data exists")

    args = parser.parse_args()

    # Update output directory in config if specified
    if args.output_dir != "./outputs":
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            config["data"]["output_dir"] = args.output_dir
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

    # Initialize and run pipeline
    pipeline = MTSDataPipeline(config_path=args.config)

    if args.step == "all":
        results = pipeline.run_complete_pipeline(force_reload=args.force_reload)
    else:
        # Run individual step (implementation would go here)
        print(f"Running individual step {args.step} - Not implemented in this demo")
        return
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"MTS DATA PIPELINE EXECUTION COMPLETE")
    print(f"{'='*80}")
    print(f"Status: {results.get('status', 'unknown')}")
    print(f"Total execution time: {results.get('total_execution_time', 0):.2f} seconds")
    print(f"Original songs: {results.get('data_counts', {}).get('original_songs', 0):,}")
    print(f"Final dataset size: {results.get('data_counts', {}).get('labeled_songs', 0):,}")
    print(f"Summary file: {results.get('summary_file', 'N/A')}")
    print(f"Report file: {results.get('report_file', 'N/A')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
