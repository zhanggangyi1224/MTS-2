"""
Wrapper for the fixed batch processor.
Exports the updated BatchProcessor and MTSBatchPipeline that honor
config-driven audio saving and emit the final CSV output.
"""

from batch_processor import BatchProcessor, MTSBatchPipeline, create_batch_config, main

__all__ = ["BatchProcessor", "MTSBatchPipeline", "create_batch_config", "main"]
