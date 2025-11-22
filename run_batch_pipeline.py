#!/usr/bin/env python3
"""
Memory-efficient batch pipeline runner for MTS
Processes large datasets in manageable chunks
"""

import sys
import argparse
from pathlib import Path
import yaml

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from batch_processor import MTSBatchPipeline

def main():
    parser = argparse.ArgumentParser(description="MTS Batch Data Pipeline")
    parser.add_argument("--config", default="config/batch_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--max-memory", type=float, default=None,
                       help="Maximum memory usage percentage")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--clean-start", action="store_true",
                       help="Start fresh (ignore checkpoints)")
    
    args = parser.parse_args()
    
    # Load and modify config
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    # Apply command line overrides
    if args.batch_size:
        config["batch_processing"]["batch_size"] = args.batch_size
    
    if args.max_memory:
        config["batch_processing"]["max_memory_percent"] = args.max_memory
    
    if args.clean_start:
        config["batch_processing"]["resume_from_checkpoint"] = False
        # Clean checkpoint directory
        import shutil
        checkpoint_dir = Path(config["batch_processing"]["checkpoint_dir"])
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            print("🧹 Cleaned checkpoint directory")
    
    # Save modified config
    temp_config = "config/temp_batch_config.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Print configuration
    print("🎵 MTS Batch Pipeline Configuration:")
    print(f"   Batch size: {config['batch_processing']['batch_size']}")
    print(f"   Max memory: {config['batch_processing']['max_memory_percent']}%")
    print(f"   Checkpoints: {config['batch_processing']['checkpoint_dir']}")
    print(f"   Resume mode: {config['batch_processing']['resume_from_checkpoint']}")
    
    # Initialize and run pipeline
    try:
        pipeline = MTSBatchPipeline(temp_config)
        results = pipeline.run_batch_pipeline()
        
        # Print results
        print(f"\n{'='*60}")
        print(f"🎉 BATCH PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Original songs: {results['data_summary']['original_songs']:,}")
        print(f"Augmented songs: {results['data_summary']['augmented_songs']:,}")
        print(f"Total labeled songs: {results['data_summary']['total_labeled_songs']:,}")
        print(f"Expansion factor: {results['data_summary']['expansion_factor']:.1f}x")
        print(f"Peak memory usage: {results['peak_memory_usage']['process_mb']:.1f} MB")
        print(f"Total batches processed: {results['batch_processing_summary']['total_batches_processed']}")
        print(f"{'='*60}")
        
        # Clean up temp config
        Path(temp_config).unlink()
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())