"""
Mac-Specific Optimizations for MTS-2 Pipeline
Leverages Apple Silicon Neural Engine and Metal Performance Shaders
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import platform
import subprocess

# Check if we're on Mac
IS_MAC = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MAC and platform.machine() == "arm64"

# Try to import Mac-specific modules
HAS_MPS = False
HAS_COREML = False

if IS_MAC:
    try:
        # Metal Performance Shaders support (PyTorch 1.12+)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            HAS_MPS = True
            print("✅ Metal Performance Shaders (MPS) available")
        else:
            print("⚠️  MPS not available (requires PyTorch 1.12+ on macOS 12.3+)")
    except:
        print("⚠️  MPS backend not found")

    try:
        import coremltools as ct
        HAS_COREML = True
        print("✅ CoreML Tools available for Neural Engine optimization")
    except ImportError:
        print("⚠️  coremltools not installed. Run: pip install coremltools")


class MacOptimizer:
    """
    Optimizer for Mac-specific acceleration
    """

    def __init__(self, enable_neural_engine: bool = True):
        self.is_mac = IS_MAC
        self.is_apple_silicon = IS_APPLE_SILICON
        self.has_mps = HAS_MPS
        self.has_coreml = HAS_COREML
        self.enable_neural_engine = enable_neural_engine and IS_APPLE_SILICON

        # Determine best device
        self.device = self._get_optimal_device()

        print(f"\n{'='*60}")
        print(f"Mac Optimization Status")
        print(f"{'='*60}")
        print(f"Platform: {platform.platform()}")
        print(f"Machine: {platform.machine()}")
        print(f"Apple Silicon: {self.is_apple_silicon}")
        print(f"MPS Available: {self.has_mps}")
        print(f"CoreML Available: {self.has_coreml}")
        print(f"Optimal Device: {self.device}")
        print(f"{'='*60}\n")

    def _get_optimal_device(self) -> str:
        """Determine the best device for computation."""
        if self.has_mps:
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Optimize a PyTorch model for Mac.

        Args:
            model: PyTorch model to optimize

        Returns:
            Optimized model
        """
        # Move to optimal device
        model = model.to(self.device)

        # Set to eval mode for inference
        model.eval()

        # Enable inference mode optimizations
        if hasattr(torch, 'inference_mode'):
            model = torch.jit.optimize_for_inference(
                torch.jit.script(model) if hasattr(model, '__torch_function__') else model
            )

        return model

    def convert_to_coreml(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        output_path: str,
        compute_units: str = "ALL"  # "ALL", "CPU_AND_NE", "CPU_ONLY"
    ) -> Optional[Any]:
        """
        Convert PyTorch model to CoreML for Neural Engine acceleration.

        Args:
            model: PyTorch model
            example_input: Example input tensor for tracing
            output_path: Path to save .mlpackage or .mlmodel
            compute_units: "ALL" (CPU+GPU+NE), "CPU_AND_NE", or "CPU_ONLY"

        Returns:
            CoreML model if successful, None otherwise
        """
        if not self.has_coreml:
            print("⚠️  CoreML not available, skipping conversion")
            return None

        if not self.is_apple_silicon:
            print("⚠️  Not Apple Silicon, Neural Engine not available")
            return None

        try:
            import coremltools as ct

            print(f"🔄 Converting model to CoreML ({compute_units})...")

            # Trace the model
            model.eval()
            model = model.cpu()  # CoreML conversion requires CPU

            traced_model = torch.jit.trace(model, example_input)

            # Convert to CoreML
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=example_input.shape)],
                compute_units=getattr(ct.ComputeUnit, compute_units, ct.ComputeUnit.ALL),
                minimum_deployment_target=ct.target.macOS13 if IS_APPLE_SILICON else ct.target.macOS12
            )

            # Save the model
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            mlmodel.save(str(output_path))
            print(f"✅ CoreML model saved to {output_path}")

            # Print model info
            spec = mlmodel.get_spec()
            print(f"   Input: {[i.name for i in spec.description.input]}")
            print(f"   Output: {[o.name for o in spec.description.output]}")

            return mlmodel

        except Exception as e:
            print(f"❌ CoreML conversion failed: {e}")
            return None

    def get_audio_processing_config(self) -> Dict[str, Any]:
        """Get optimized audio processing configuration for Mac."""
        config = {
            "use_numba": False,  # Numba may not work well on Apple Silicon
            "use_native_resample": True,
            "use_accelerate": True if IS_MAC else False,  # Apple Accelerate framework
            "chunk_size": 2048 if self.is_apple_silicon else 1024,
            "num_threads": self._get_optimal_threads(),
        }
        return config

    def _get_optimal_threads(self) -> int:
        """Get optimal number of threads for audio processing."""
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()

            # On Apple Silicon, use performance cores efficiently
            if self.is_apple_silicon:
                # M1/M2 typically have 4-8 performance cores
                # Leave some headroom for OS
                return max(2, min(cpu_count - 2, 6))
            else:
                return max(1, cpu_count // 2)
        except:
            return 2

    def optimize_librosa_for_mac(self):
        """Configure librosa for optimal Mac performance."""
        try:
            import librosa

            # Use faster FFT backends on Mac
            if IS_MAC:
                # Try to use Apple's Accelerate framework
                import scipy
                print("📊 Configuring audio processing for Mac...")
                print(f"   NumPy uses: {np.__config__.show()}")
        except:
            pass

    def benchmark_device(self, size: tuple = (1, 128, 1024)) -> Dict[str, float]:
        """
        Benchmark different devices to verify acceleration.

        Args:
            size: Tensor size to test

        Returns:
            Dictionary with timing results
        """
        import time

        results = {}

        # Test CPU
        x = torch.randn(size)
        start = time.time()
        for _ in range(100):
            y = torch.matmul(x, x.transpose(-2, -1))
        results['cpu'] = time.time() - start

        # Test MPS if available
        if self.has_mps:
            x_mps = x.to('mps')
            torch.mps.synchronize()  # Ensure transfer is complete
            start = time.time()
            for _ in range(100):
                y = torch.matmul(x_mps, x_mps.transpose(-2, -1))
            torch.mps.synchronize()  # Wait for completion
            results['mps'] = time.time() - start

        # Print results
        print(f"\n📊 Device Benchmark Results:")
        print(f"   CPU: {results['cpu']:.4f}s")
        if 'mps' in results:
            speedup = results['cpu'] / results['mps']
            print(f"   MPS: {results['mps']:.4f}s (Speedup: {speedup:.2f}x)")

        return results

    def get_ffmpeg_config_for_mac(self) -> Dict[str, Any]:
        """Get optimized FFmpeg settings for Mac."""
        config = {
            "audio_codec": "pcm_s16le",  # Lossless PCM
            "video_codec": "h264_videotoolbox" if IS_MAC else "libx264",  # Hardware encoding
            "use_videotoolbox": IS_MAC,  # Use Apple's VideoToolbox
            "threads": self._get_optimal_threads(),
        }

        # Check if ffmpeg has videotoolbox support
        if IS_MAC:
            try:
                result = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-encoders'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if 'h264_videotoolbox' in result.stdout:
                    print("✅ FFmpeg VideoToolbox hardware acceleration available")
                else:
                    print("⚠️  FFmpeg VideoToolbox not available")
                    config["video_codec"] = "libx264"
            except:
                pass

        return config


def setup_mac_environment():
    """
    Setup optimal environment for Mac.
    Call this at the start of your pipeline.
    """
    if not IS_MAC:
        print("ℹ️  Not running on Mac, skipping Mac optimizations")
        return None

    optimizer = MacOptimizer(enable_neural_engine=True)

    # Set PyTorch to use optimal number of threads
    num_threads = optimizer._get_optimal_threads()
    torch.set_num_threads(num_threads)
    print(f"🔧 Set PyTorch threads: {num_threads}")

    # Disable PyTorch JIT on MPS if needed (some ops not supported)
    if optimizer.has_mps:
        torch.jit.enable_onednn_fusion(False)

    # Run benchmark to verify acceleration
    if optimizer.has_mps:
        print("\n🏃 Running device benchmark...")
        optimizer.benchmark_device()

    return optimizer


def get_optimal_audio_backend():
    """Get the best audio I/O backend for Mac."""
    backends = []

    try:
        import soundfile
        backends.append("soundfile")
    except:
        pass

    try:
        import scipy.io.wavfile
        backends.append("scipy")
    except:
        pass

    if IS_MAC:
        try:
            import audioread
            backends.append("audioread")  # Uses Core Audio on Mac
        except:
            pass

    print(f"📻 Available audio backends: {backends}")
    return backends[0] if backends else "numpy"


# Example usage
if __name__ == "__main__":
    print("🍎 Mac Optimization Module Test")
    print("=" * 60)

    # Setup environment
    optimizer = setup_mac_environment()

    if optimizer:
        # Get configs
        audio_config = optimizer.get_audio_processing_config()
        print(f"\n🎵 Audio Processing Config:")
        for k, v in audio_config.items():
            print(f"   {k}: {v}")

        ffmpeg_config = optimizer.get_ffmpeg_config_for_mac()
        print(f"\n🎬 FFmpeg Config:")
        for k, v in ffmpeg_config.items():
            print(f"   {k}: {v}")

        # Test model optimization
        print(f"\n🧪 Testing model optimization...")
        test_model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )

        optimized_model = optimizer.optimize_model(test_model)
        print(f"✅ Model optimized and moved to: {next(optimized_model.parameters()).device}")

        # Test CoreML conversion if available
        if optimizer.has_coreml:
            example_input = torch.randn(1, 128)
            coreml_model = optimizer.convert_to_coreml(
                test_model,
                example_input,
                "./test_model.mlpackage",
                compute_units="ALL"
            )
            if coreml_model:
                print("✅ CoreML conversion successful!")

    print("\n" + "=" * 60)
    print("✅ Mac optimization test complete!")
