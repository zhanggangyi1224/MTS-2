# MTS-2 Audio Quality Issues - Root Cause Analysis & Fixes

**Date**: 2025-12-14
**Status**: Critical issues identified and fixed

## Root Cause Analysis

### Issue 1: Simulated Audio Generation (CRITICAL)
**Location**: [src/augmentation.py:701-725](src/augmentation.py#L701-L725)

The pipeline is generating **synthetic sine wave audio** instead of using real audio:

```python
def _simulate_audio_from_song(self, song: Dict) -> np.ndarray:
    # Creates simple sine waves - NOT REAL AUDIO
    fundamental = 440 * (2 ** ((random.randint(-12, 12)) / 12))
    audio = (0.3 * np.sin(2 * np.pi * fundamental * t) + ...)
```

**Impact**: All augmented audio is synthetic, explaining the "not have original audio structure" problem.

**Root Cause**: When songs don't have an 'audio' key, the code falls back to simulation.

### Issue 2: Sample Rate Mismatch
**Location**: [config/config.yaml:8](config/config.yaml#L8)

- Current config: `target_sample_rate: 22050`
- EnCodec requirement: 24000 Hz (from [encodec_wrapper.py:68](src/models/encodec_wrapper.py#L68))
- SLURM script tries to fix it: Line 299 sets it to 24000

**Impact**: Resampling artifacts and quality degradation.

### Issue 3: No Real Audio Loading
**Location**: [src/data_loader.py:160-209](src/data_loader.py#L160-L209)

The fallback dataset creates:
- Simulated metadata ✓
- Spectrogram paths (non-existent) ✗
- `has_audio: False` ✗

**Impact**: No real audio ever loads.

### Issue 4: No Mac Optimization
**Current**: Only uses CPU/CUDA
**Missing**: CoreML/ANE support for Mac Silicon

## Comprehensive Fix

### Fix 1: Use Real Audio Data
Options:
1. **Use FMA Dataset** (Free Music Archive) - Already supported!
2. Load your own audio files
3. Use actual CCMusic with proper auth

### Fix 2: Update Sample Rate to 24kHz

### Fix 3: Add Mac Neural Engine Support

### Fix 4: Fix Audio Augmentation Quality

---

## Implementation Plan

See the following fixed files:
- `config/config_fixed_local.yaml` - Local Mac config
- `src/augmentation_fixed.py` - Quality-preserving augmentation
- `run_local_pipeline.py` - Mac-optimized local runner
- `src/mac_optimization.py` - CoreML/ANE integration
