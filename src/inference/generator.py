"""
MTS Generator - Inference pipeline for music generation

Features:
1. DDIM sampling (faster than DDPM)
2. Classifier-free guidance
3. Structure-guided generation
4. Multiple output formats (WAV, MP3, FLAC)
5. Batch generation
6. 30-second preview generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import time
from tqdm import tqdm

# Audio I/O
import soundfile as sf

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

# Import model components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mts_model import MTSModel, MTSConfig, create_mts_model
from models.encodec_wrapper import EnCodecWrapper


@dataclass
class GenerationConfig:
    """Configuration for music generation."""
    
    # Duration
    duration: float = 210.0  # 3.5 minutes default
    
    # Sampling
    num_inference_steps: int = 50  # DDIM steps
    cfg_scale: float = 7.5  # Classifier-free guidance scale
    eta: float = 0.0  # DDIM eta (0 = deterministic)
    
    # Output
    output_format: str = "mp3"  # wav, mp3, flac
    sample_rate: int = 24000
    mp3_bitrate: str = "320k"
    
    # Structure
    use_structure: bool = True
    
    # Generation
    batch_size: int = 1
    seed: Optional[int] = None


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) sampler.
    
    Much faster than DDPM while maintaining quality.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine"
    ):
        self.model = model
        self.num_train_timesteps = num_train_timesteps
        
        # Compute schedule
        betas = self._make_beta_schedule(beta_schedule, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    def _make_beta_schedule(self, schedule: str, num_timesteps: int) -> torch.Tensor:
        """Create noise schedule."""
        if schedule == "linear":
            return torch.linspace(1e-4, 0.02, num_timesteps)
        elif schedule == "cosine":
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * 3.14159 * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def set_timesteps(self, num_inference_steps: int, device: str = "cuda"):
        """Set timesteps for inference."""
        
        # Create evenly spaced timesteps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device).long()
        
        # Move alphas to device
        self.alphas_cumprod = self.alphas_cumprod.to(device)
    
    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        prev_timestep: Optional[int] = None
    ) -> torch.Tensor:
        """
        Single DDIM step.
        
        Args:
            model_output: Predicted noise from model
            timestep: Current timestep
            sample: Current noisy sample
            eta: Controls stochasticity (0 = deterministic)
            prev_timestep: Previous timestep (for step calculation)
            
        Returns:
            Denoised sample at prev_timestep
        """
        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep is not None and prev_timestep >= 0 else torch.tensor(1.0)
        
        # Compute predicted x_0
        pred_original = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        
        # Clip predicted x_0
        pred_original = torch.clamp(pred_original, -1, 1)
        
        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev = eta * torch.sqrt(variance)
        
        # Compute direction pointing to x_t
        pred_dir = torch.sqrt(1 - alpha_prod_t_prev - std_dev ** 2) * model_output
        
        # Compute x_{t-1}
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original + pred_dir
        
        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + std_dev * noise
        
        return prev_sample


class MTSGenerator:
    """
    Complete inference pipeline for MTS music generation.
    """
    
    def __init__(
        self,
        model: Optional[MTSModel] = None,
        model_path: Optional[str] = None,
        model_size: str = "base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize generator.
        
        Args:
            model: Pre-loaded MTSModel (if None, will create from scratch)
            model_path: Path to checkpoint (loads if provided)
            model_size: Model size if creating new
            device: Device for generation
        """
        self.device = device
        
        # Load or create model
        if model is not None:
            self.model = model.to(device)
        else:
            self.model = create_mts_model(model_size, use_structure=True).to(device)
        
        # Load checkpoint if provided
        if model_path is not None:
            self.load_checkpoint(model_path)
        
        self.model.eval()
        
        # Initialize sampler
        self.sampler = DDIMSampler(
            self.model.unet,
            num_train_timesteps=self.model.config.diffusion_timesteps
        )
        
        # EnCodec wrapper for audio encoding/decoding
        self.encodec = self.model.encodec
        
        print(f"✅ Generator initialized on {device}")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"✅ Loaded checkpoint from {path}")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
        structure: Optional[Dict] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Generate music from text prompt.
        
        Args:
            prompt: Text prompt(s)
            config: Generation configuration
            structure: Optional structure annotation
            progress_callback: Optional callback(step, total_steps)
            
        Returns:
            Dict with:
                - audio: Generated audio tensor
                - compressed: 30-second compressed representation
                - metadata: Generation metadata
        """
        if config is None:
            config = GenerationConfig()
        
        # Set seed for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        
        start_time = time.time()
        
        # Handle batch
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        batch_size = len(prompts)
        
        # Calculate sequence length
        frame_rate = self.model.config.encodec_frame_rate
        seq_len = int(config.duration * frame_rate)
        latent_dim = self.model.config.encodec_latent_dim
        
        # Encode text
        text_emb = self.model.encode_text(prompts, self.device)
        
        # Encode structure (if provided)
        struct_emb = None
        if structure is not None and config.use_structure:
            struct_emb = self.model.encode_structure(structure, self.device)
        
        # Prepare null embeddings for CFG
        null_text = self.model.null_text_emb.expand(batch_size, -1, -1)
        
        # Initialize noise
        latents = torch.randn(
            batch_size, latent_dim, seq_len,
            device=self.device
        )
        
        # Set up sampler
        self.sampler.set_timesteps(config.num_inference_steps, self.device)
        
        # DDIM sampling loop
        for i, t in enumerate(tqdm(self.sampler.timesteps, desc="Generating")):
            # Expand timestep for batch
            t_batch = t.expand(batch_size)
            
            # Classifier-free guidance
            # Predict noise for conditional and unconditional
            if config.cfg_scale != 1.0:
                # Conditional prediction
                noise_pred_cond = self.model.unet(latents, t_batch, text_emb, struct_emb)
                
                # Unconditional prediction
                noise_pred_uncond = self.model.unet(latents, t_batch, null_text, struct_emb)
                
                # CFG interpolation
                noise_pred = noise_pred_uncond + config.cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.model.unet(latents, t_batch, text_emb, struct_emb)
            
            # DDIM step
            prev_t = self.sampler.timesteps[i + 1] if i + 1 < len(self.sampler.timesteps) else -1
            latents = self.sampler.step(noise_pred, t, latents, config.eta, prev_t)
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(self.sampler.timesteps))
        
        # Get 30-second compression
        compressed = self.model.compression.compress(latents)
        
        # Decode to audio
        audio = self.model.decode_audio(latents)
        
        generation_time = time.time() - start_time
        
        return {
            "audio": audio,
            "latents": latents,
            "compressed": compressed,
            "metadata": {
                "prompts": prompts,
                "duration": config.duration,
                "num_inference_steps": config.num_inference_steps,
                "cfg_scale": config.cfg_scale,
                "seed": config.seed,
                "generation_time": generation_time,
                "real_time_factor": generation_time / config.duration
            }
        }
    
    @torch.no_grad()
    def generate_preview(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        structure: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Generate 30-second preview directly.
        
        Faster than full generation - useful for quick iteration.
        """
        if config is None:
            config = GenerationConfig()
        
        # Generate compressed representation
        result = self.generate(
            prompt,
            GenerationConfig(
                duration=config.duration,
                num_inference_steps=config.num_inference_steps // 2,  # Faster
                cfg_scale=config.cfg_scale
            ),
            structure
        )
        
        # Expand to 30 seconds
        preview_len = int(30.0 * self.model.config.encodec_frame_rate)
        preview_latents = self.model.compression.expand(result["compressed"], preview_len)
        
        return self.model.decode_audio(preview_latents)
    
    @torch.no_grad()
    def generate_variations(
        self,
        prompt: str,
        num_variations: int = 4,
        config: Optional[GenerationConfig] = None,
        structure: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Generate multiple variations of the same prompt.
        """
        variations = []
        
        for i in range(num_variations):
            var_config = config or GenerationConfig()
            var_config.seed = (config.seed or 0) + i if config else i
            
            result = self.generate(prompt, var_config, structure)
            variations.append(result)
        
        return variations
    
    @torch.no_grad()
    def continue_generation(
        self,
        audio: torch.Tensor,
        prompt: str,
        additional_duration: float = 60.0,
        config: Optional[GenerationConfig] = None
    ) -> torch.Tensor:
        """
        Continue/extend existing audio.
        
        Encodes the existing audio and generates continuation.
        """
        if config is None:
            config = GenerationConfig()
        
        # Encode existing audio to latents
        existing_latents = self.model.encode_audio(audio)
        existing_len = existing_latents.shape[-1]
        
        # Generate new content
        new_len = int(additional_duration * self.model.config.encodec_frame_rate)
        total_len = existing_len + new_len
        
        # Start with existing content + noise for continuation
        text_emb = self.model.encode_text([prompt], self.device)
        
        # Initialize with existing + noise
        continuation_latents = torch.randn(
            1, self.model.config.encodec_latent_dim, new_len,
            device=self.device
        )
        
        # Generate continuation (simplified - full implementation would use inpainting)
        self.sampler.set_timesteps(config.num_inference_steps, self.device)
        
        for t in tqdm(self.sampler.timesteps, desc="Continuing"):
            t_batch = t.expand(1)
            noise_pred = self.model.unet(continuation_latents, t_batch, text_emb, None)
            prev_t = self.sampler.timesteps[list(self.sampler.timesteps).index(t) + 1] if t != self.sampler.timesteps[-1] else -1
            continuation_latents = self.sampler.step(noise_pred, t, continuation_latents, config.eta, prev_t)
        
        # Concatenate
        full_latents = torch.cat([existing_latents, continuation_latents], dim=-1)
        
        return self.model.decode_audio(full_latents)
    
    def save_audio(
        self,
        audio: torch.Tensor,
        path: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Save generated audio to file.
        
        Supports WAV, MP3, and FLAC formats.
        """
        if config is None:
            config = GenerationConfig()
        
        path = Path(path)
        
        # Convert to numpy
        audio_np = audio.squeeze().cpu().numpy()
        
        # Normalize
        audio_np = audio_np / (np.abs(audio_np).max() + 1e-8)
        audio_np = (audio_np * 0.95).astype(np.float32)
        
        output_format = config.output_format.lower()
        
        if output_format == "wav":
            # Direct WAV output
            output_path = path.with_suffix(".wav")
            sf.write(str(output_path), audio_np, config.sample_rate)
        
        elif output_format == "flac":
            # FLAC output
            output_path = path.with_suffix(".flac")
            sf.write(str(output_path), audio_np, config.sample_rate)
        
        elif output_format == "mp3":
            # MP3 output
            output_path = path.with_suffix(".mp3")
            
            if HAS_PYDUB:
                # Use pydub for MP3 encoding
                # First save as WAV
                temp_wav = path.with_suffix(".temp.wav")
                sf.write(str(temp_wav), audio_np, config.sample_rate)
                
                # Convert to MP3
                audio_segment = AudioSegment.from_wav(str(temp_wav))
                audio_segment.export(
                    str(output_path),
                    format="mp3",
                    bitrate=config.mp3_bitrate
                )
                
                # Clean up temp file
                temp_wav.unlink()
            else:
                # Fallback: try ffmpeg directly
                import subprocess
                import tempfile
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio_np, config.sample_rate)
                    
                    try:
                        subprocess.run([
                            "ffmpeg", "-y",
                            "-i", tmp.name,
                            "-b:a", config.mp3_bitrate,
                            str(output_path)
                        ], check=True, capture_output=True)
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # Fallback to WAV
                        print("⚠️  MP3 encoding failed, saving as WAV")
                        output_path = path.with_suffix(".wav")
                        sf.write(str(output_path), audio_np, config.sample_rate)
                    finally:
                        Path(tmp.name).unlink()
        
        else:
            raise ValueError(f"Unknown format: {output_format}")
        
        print(f"💾 Saved audio: {output_path}")
        return str(output_path)
    
    def save_compressed(
        self,
        compressed: torch.Tensor,
        path: str
    ) -> str:
        """Save compressed representation for later use."""
        path = Path(path).with_suffix(".pt")
        torch.save(compressed.cpu(), path)
        print(f"💾 Saved compressed: {path}")
        return str(path)
    
    def load_compressed(self, path: str) -> torch.Tensor:
        """Load compressed representation."""
        return torch.load(path, map_location=self.device)
    
    @torch.no_grad()
    def expand_compressed(
        self,
        compressed: torch.Tensor,
        target_duration: float = 210.0
    ) -> torch.Tensor:
        """Expand compressed representation to full audio."""
        target_len = int(target_duration * self.model.config.encodec_frame_rate)
        latents = self.model.compression.expand(compressed.to(self.device), target_len)
        return self.model.decode_audio(latents)


def create_generator(
    model_path: Optional[str] = None,
    model_size: str = "base",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> MTSGenerator:
    """
    Factory function to create generator.
    
    Args:
        model_path: Path to trained checkpoint
        model_size: Model size if no checkpoint
        device: Device for generation
        
    Returns:
        Configured MTSGenerator
    """
    return MTSGenerator(
        model_path=model_path,
        model_size=model_size,
        device=device
    )


def main():
    """Demo generation."""
    print("🎵 MTS Music Generator Demo")
    print("=" * 50)
    
    # Create generator
    generator = create_generator(model_size="small")
    
    # Generation config
    config = GenerationConfig(
        duration=30.0,  # Short for demo
        num_inference_steps=20,  # Fast for demo
        cfg_scale=5.0,
        output_format="mp3",
        seed=42
    )
    
    # Structure annotation
    structure = {
        "section_types": [0, 1, 2],  # intro, verse, chorus
        "section_props": [
            [0, 8, 8, 0.3, 0.5],    # intro
            [8, 20, 12, 0.5, 0.7],  # verse
            [20, 30, 10, 0.9, 1.0]  # chorus
        ]
    }
    
    # Generate
    print("\n🎹 Generating music...")
    result = generator.generate(
        prompt="A peaceful piano melody with gentle strings and soft ambient textures",
        config=config,
        structure=structure
    )
    
    print(f"\n📊 Generation Results:")
    print(f"   Audio shape: {result['audio'].shape}")
    print(f"   Compressed shape: {result['compressed'].shape}")
    print(f"   Generation time: {result['metadata']['generation_time']:.2f}s")
    print(f"   Real-time factor: {result['metadata']['real_time_factor']:.2f}x")
    
    # Save outputs
    output_dir = Path("./generated")
    output_dir.mkdir(exist_ok=True)
    
    audio_path = generator.save_audio(
        result["audio"],
        output_dir / "demo_song",
        config
    )
    
    compressed_path = generator.save_compressed(
        result["compressed"],
        output_dir / "demo_compressed"
    )
    
    print(f"\n✅ Demo complete!")
    print(f"   Audio: {audio_path}")
    print(f"   Compressed: {compressed_path}")


if __name__ == "__main__":
    main()
