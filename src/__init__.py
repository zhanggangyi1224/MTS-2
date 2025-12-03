"""
MTS-2: Music-to-Structure Data Pipeline and Model

This package provides:
- Data loading and preprocessing
- Audio augmentation
- Text prompt generation
- Structure annotation
- Model architecture
- Training pipeline
- Inference tools
"""

__version__ = "0.1.0"

# Import main pipeline components
try:
    from .pipeline import MTSPipeline
    from .data_loader import MTSDataLoader
    from .augmentation import MTSAudioAugmentation
    from .text_generation import MTSTextPromptGenerator
    from .data_organization import MTSDataOrganizer
    from .structure_processing import MTSStructureProcessor
    from .batch_processor import MTSBatchProcessor

    __all__ = [
        'MTSPipeline',
        'MTSDataLoader',
        'MTSAudioAugmentation',
        'MTSTextPromptGenerator',
        'MTSDataOrganizer',
        'MTSStructureProcessor',
        'MTSBatchProcessor',
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}")
    __all__ = []
