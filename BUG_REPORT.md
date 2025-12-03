# MTS-2 Bug Report and Fix Plan

## Date: 2025-12-01

## Critical Issues Found

### 1. IMPORT ISSUES (High Priority)

#### Problem: Relative imports not working
**Files affected:** All pipeline files
**Issue:** Files use `from data_loader import` instead of `from .data_loader import` or proper package imports

**Locations:**
- `src/pipeline.py:17-21` - Uses bare imports
- `src/batch_processor.py` - Same issue
- All model files - Cross-module imports failing

**Fix:**
```python
# WRONG:
from data_loader import MTSDataLoader

# CORRECT:
from .data_loader import MTSDataLoader
# OR
from src.data_loader import MTSDataLoader
```

---

### 2. MISSING DEPENDENCIES

**Issue:** Code assumes all optional dependencies are installed

**Missing checks for:**
- `librosa` (required in data_loader.py, augmentation.py, structure_detection.py)
- `pyrubberband` (optional in augmentation.py)
- `pedalboard` (optional in augmentation.py)
- `encodec` (optional in encodec_wrapper.py)
- `transformers` (required in text_generation.py)
- `sentence-transformers` (required in text_generation.py)
- `datasets` (required in data_loader.py)
- `madmom` (optional in structure_detection.py)

**Fix:** Add try-except wrappers and fallback implementations

---

### 3. INCOMPLETE IMPLEMENTATIONS

#### 3.1 batch_processor_fixed.py
**File:** `src/batch_processor_fixed.py`
**Issue:** Only exports MTSBatchProcessor but doesn't implement any fixes

**Current code:**
```python
from batch_processor import MTSBatchProcessor
__all__ = ['MTSBatchProcessor']
```

**Missing:** Actual fixes mentioned in filename

---

#### 3.2 Missing main execution blocks
**Files affected:**
- `run_pipeline.py` - Only 16 lines, very minimal
- `run_batch_pipeline.py` - Doesn't handle errors properly

---

### 4. CONFIGURATION ISSUES

#### 4.1 Config file structure inconsistency
**Files:** `config/*.yaml`
**Issue:** Different config files have incompatible structures

**Found configs:**
- `config.yaml` - Basic config
- `improved_config.yaml` - Enhanced version
- `batch_config.yaml` - Batch processing
- `batch_config_fixed.yaml` - Another variant
- `temp_batch_config.yaml` - Temporary file

**Problem:** Pipeline doesn't validate config structure

---

### 5. DATA HANDLING BUGS

#### 5.1 CCMusic dataset handling
**File:** `src/data_loader.py`
**Line:** ~150-200
**Issue:** Assumes dataset structure without validation

#### 5.2 FMA dataset path issues
**File:** `src/data_loader.py:252-300`
**Issue:** Hard-coded paths and missing error handling

```python
def _load_fma_real_audio(self, song_id: str, metadata: Dict):
    # Bug: No validation if path exists
    audio_path = Path(self.config.get('fma_audio_path'))
```

---

### 6. AUGMENTATION ISSUES

#### 6.1 Memory leaks in augmentation
**File:** `src/augmentation.py`
**Issue:** Audio arrays not released after processing

#### 6.2 Fallback chains incomplete
**Lines:** 200-250
**Issue:** Some augmentation methods don't have proper fallbacks when optional deps missing

---

### 7. STRUCTURE PROCESSING BUGS

#### 7.1 Frame rate calculation errors
**File:** `src/structure_processing.py:400-450`
**Issue:** Frame rate mismatch between different components

```python
# Bug: Assumes 75 fps but doesn't validate
frame_rate = 75  # Should be from config
```

#### 7.2 Section boundary validation missing
**Issue:** No checks for overlapping sections or invalid timestamps

---

### 8. TEXT GENERATION ISSUES

#### 8.1 Transformer model loading
**File:** `src/text_generation.py:100-150`
**Issue:** No error handling if models fail to download

#### 8.2 Prompt cache not thread-safe
**Issue:** Shared prompt pool could cause race conditions in batch mode

---

### 9. BATCH PROCESSING BUGS

#### 9.1 Checkpoint corruption risk
**File:** `src/batch_processor.py:500-550`
**Issue:** No atomic writes for checkpoints

```python
# Bug: Could corrupt checkpoint if interrupted
with open(checkpoint_file, 'w') as f:
    json.dump(state, f)
```

**Fix:** Use atomic write with temp file + rename

#### 9.2 Memory monitoring incorrect
**Line:** ~200
**Issue:** Memory calculation doesn't account for GPU memory

---

### 10. MODEL ARCHITECTURE ISSUES

#### 10.1 EnCodec wrapper token mismatch
**File:** `src/models/encodec_wrapper.py:104-140`
**Issue:** Returns inconsistent tuple vs tensor

```python
def encode(self, audio):
    # Sometimes returns (frames, tokens)
    # Sometimes returns just tokens
```

#### 10.2 Diffusion timesteps off-by-one
**File:** `src/models/diffusion_unet.py:566-584`
**Issue:** Cosine schedule calculation might have edge case bugs

---

### 11. TRAINING SYSTEM ISSUES

#### 11.1 Mixed precision scaler not initialized properly
**File:** `src/training/trainer.py:221`
**Issue:** GradScaler might not work on CPU

#### 11.2 Checkpoint loading doesn't validate config
**File:** `src/training/trainer.py:391-402`
**Issue:** Could load checkpoint with incompatible config

---

### 12. INFERENCE ISSUES

#### 12.1 DDIM sampler timestep calculation
**File:** `src/inference/generator.py:102-110`
**Issue:** Integer division could cause issues with small step counts

#### 12.2 Audio export path validation missing
**File:** `src/inference/generator.py:435-517`
**Issue:** Doesn't check if output directory exists

---

## Missing Components

### 1. Missing Files
- `src/data/prepare_datasets.py` - Dataset preparation scripts
- `src/evaluation/metrics.py` - Evaluation metrics
- `src/utils/audio_utils.py` - Shared audio utilities
- `src/utils/config_utils.py` - Config validation
- `tests/` - No test suite exists

### 2. Missing Functionality
- Dataset validation
- Model checkpoint validation
- Audio quality checks
- Progress saving/resumption for pipeline
- Distributed training support
- Multi-GPU support
- Tensorboard logging
- Weights & Biases integration

---

## Performance Issues

### 1. Memory Inefficiency
- Augmentation loads entire audio into memory
- No streaming support for long audio
- Batch processing could be optimized

### 2. Speed Issues
- Text generation is sequential (could be parallelized)
- Structure processing recomputes features
- No caching for repeated operations

---

## Documentation Issues

### 1. Missing Docstrings
- Many functions lack proper docstrings
- Type hints incomplete in some files
- Parameter descriptions missing

### 2. Missing Examples
- No example scripts
- No tutorial notebooks
- No quick start guide (README is minimal)

---

## Priority Fix Order

### P0 - Critical (Blocks execution)
1. Fix import statements in all files
2. Add dependency checks with fallbacks
3. Fix batch_processor_fixed.py
4. Validate config file loading

### P1 - High (Causes errors)
1. Fix EnCodec token return inconsistency
2. Add checkpoint atomic writes
3. Fix memory leaks in augmentation
4. Add path validation everywhere

### P2 - Medium (Quality issues)
1. Add proper error handling throughout
2. Fix frame rate calculations
3. Implement missing fallbacks
4. Add progress saving

### P3 - Low (Nice to have)
1. Add comprehensive tests
2. Improve documentation
3. Add logging throughout
4. Optimize performance

---

## Estimated Effort

- **P0 fixes:** 4-6 hours
- **P1 fixes:** 8-12 hours
- **P2 fixes:** 12-16 hours
- **P3 fixes:** 20+ hours

**Total for basic functionality:** ~1-2 days of focused work

---

## Next Steps

1. Create a test config that works
2. Fix import statements
3. Add dependency checks
4. Test each pipeline step individually
5. Run end-to-end test
6. Document remaining issues
