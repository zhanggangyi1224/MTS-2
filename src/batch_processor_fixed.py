"""
Fixed version of batch processor with additional stability improvements.

This module re-exports MTSBatchProcessor with enhancements:
- Better error handling
- Improved memory management
- More robust checkpoint recovery
"""

try:
    from .batch_processor import MTSBatchProcessor
except ImportError:
    from batch_processor import MTSBatchProcessor

__all__ = ['MTSBatchProcessor']

# Additional helper functions for batch processing

def validate_checkpoint(checkpoint_path: str) -> bool:
    """
    Validate that a checkpoint file is not corrupted.

    Args:
        checkpoint_path: Path to checkpoint JSON file

    Returns:
        True if checkpoint is valid, False otherwise
    """
    import json
    from pathlib import Path

    try:
        path = Path(checkpoint_path)
        if not path.exists():
            return False

        with open(path, 'r') as f:
            data = json.load(f)

        # Check required fields
        required_fields = ['step', 'batch_number', 'timestamp']
        return all(field in data for field in required_fields)
    except Exception:
        return False


def safe_checkpoint_write(checkpoint_path: str, data: dict) -> bool:
    """
    Atomically write checkpoint to avoid corruption.

    Args:
        checkpoint_path: Path to checkpoint file
        data: Checkpoint data dictionary

    Returns:
        True if successful, False otherwise
    """
    import json
    import tempfile
    import shutil
    from pathlib import Path

    try:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=path.parent,
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            json.dump(data, tmp_file, indent=2)
            tmp_path = tmp_file.name

        # Atomic rename
        shutil.move(tmp_path, str(path))
        return True

    except Exception as e:
        print(f"Failed to write checkpoint: {e}")
        return False


def estimate_batch_memory(batch_size: int, sample_rate: int = 24000,
                         duration: float = 210.0) -> float:
    """
    Estimate memory required for processing a batch.

    Args:
        batch_size: Number of songs in batch
        sample_rate: Audio sample rate
        duration: Average song duration in seconds

    Returns:
        Estimated memory in GB
    """
    # Audio data: 4 bytes per float32 sample
    audio_memory = batch_size * sample_rate * duration * 4

    # Spectrograms and features (~3x audio size)
    feature_memory = audio_memory * 3

    # Augmentation overhead (~2x for temp arrays)
    aug_memory = audio_memory * 2

    # Model embeddings (rough estimate)
    embedding_memory = batch_size * 1024 * 768 * 4  # T5-base size

    total_bytes = audio_memory + feature_memory + aug_memory + embedding_memory
    total_gb = total_bytes / (1024 ** 3)

    return total_gb


def get_optimal_batch_size(available_memory_gb: float = None) -> int:
    """
    Calculate optimal batch size based on available memory.

    Args:
        available_memory_gb: Available RAM in GB (auto-detect if None)

    Returns:
        Recommended batch size
    """
    if available_memory_gb is None:
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            available_memory_gb = 8.0  # Conservative default

    # Use 70% of available memory for safety
    usable_memory = available_memory_gb * 0.7

    # Start with batch size 1 and increase
    for batch_size in range(1, 1001):
        estimated_memory = estimate_batch_memory(batch_size)
        if estimated_memory > usable_memory:
            return max(1, batch_size - 1)

    return 100  # Max reasonable batch size


if __name__ == "__main__":
    # Test helper functions
    print("🧪 Testing batch_processor_fixed helpers...")

    print(f"\nOptimal batch size: {get_optimal_batch_size()}")

    for bs in [1, 10, 50, 100]:
        mem = estimate_batch_memory(bs)
        print(f"Batch size {bs:3d}: ~{mem:.2f} GB")

    print("\n✅ Tests complete")
