"""
MTS EnCodec Wrapper - Audio Tokenization for Music Generation

This module wraps Meta's EnCodec for converting audio waveforms to discrete tokens
and back, enabling efficient training of music generation models.

Key features:
- 24kHz audio codec with ~150x compression
- Residual Vector Quantization (RVQ) with configurable codebooks
- Streaming encode/decode for long-form audio
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Union
from pathlib import Path

try:
    from encodec import EncodecModel
    from encodec.utils import convert_audio
    HAS_ENCODEC = True
except ImportError:
    HAS_ENCODEC = False
    print("⚠️  encodec not installed. Run: pip install encodec")


class EnCodecWrapper(nn.Module):
    
    def __init__(
        self,
        model_type: str = "encodec_24khz",
        target_bandwidth: float = 6.0,  # kbps
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize EnCodec wrapper.
        
        Args:
            model_type: Model variant ("encodec_24khz" or "encodec_48khz")
            target_bandwidth: Target bandwidth in kbps (1.5, 3.0, 6.0, 12.0, 24.0)
            device: Device to run model on
            cache_dir: Directory to cache model weights
        """
        super().__init__()
        
        self.model_type = model_type
        self.target_bandwidth = target_bandwidth
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Model parameters based on bandwidth
        self.bandwidth_to_codebooks = {
            1.5: 2,
            3.0: 4,
            6.0: 8,
            12.0: 16,
            24.0: 32
        }
        self.n_codebooks = self.bandwidth_to_codebooks.get(target_bandwidth, 8)
        
        # Load model
        self.model = self._load_model()
        
        # Audio parameters
        if model_type == "encodec_24khz":
            self.sample_rate = 24000
            self.frame_rate = 75  # tokens per second
            self.codebook_size = 1024
        else:
            self.sample_rate = 48000
            self.frame_rate = 50
            self.codebook_size = 1024
    
    def _load_model(self) -> Optional[nn.Module]:
        """Load EnCodec model."""
        if not HAS_ENCODEC:
            print("❌ EnCodec not available, using mock tokenization")
            return None
        
        try:
            if self.model_type == "encodec_24khz":
                model = EncodecModel.encodec_model_24khz()
            else:
                model = EncodecModel.encodec_model_48khz()
            
            model.set_target_bandwidth(self.target_bandwidth)
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ Loaded EnCodec {self.model_type} @ {self.target_bandwidth}kbps")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load EnCodec: {e}")
            return None
    
    @torch.no_grad()
    def encode(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None
    ) -> Tuple[list, torch.Tensor]:
        """
        Encode audio waveform to discrete tokens.
        
        Args:
            audio: Audio tensor of shape (batch, channels, samples) or (batch, samples)
            sample_rate: Sample rate of input audio (will resample if different)
            
        Returns:
            frames: list of (codes, scale) tuples for decode
            tokens: Token tensor of shape (batch, n_codebooks, seq_len)
        """
        # Ensure correct shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
        
        # Convert to mono if stereo
        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)
        
        # Resample if necessary
        if sample_rate and sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)
        
        audio = audio.to(self.device)
        
        if self.model is None:
            # Mock tokenization for development
            return [], self._mock_encode(audio)
        
        # EnCodec encoding
        with torch.no_grad():
            encoded_frames = self.model.encode(audio)

        # Extract codes from frames
        codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)
        # Ensure codes are on the same device as the model
        codes = codes.to(self.device)

        return encoded_frames, codes  # Shape: (batch, n_codebooks, seq_len)
    
    @torch.no_grad()
    def decode(
        self,
        frames_or_tokens
    ) -> torch.Tensor:
        """
        Decode discrete tokens back to audio waveform.
        
        Args:
            frames_or_tokens: list of (codes, scale) tuples from encode; or tokens tensor (mock)
        """
        if self.model is None:
            if isinstance(frames_or_tokens, torch.Tensor):
                tokens = frames_or_tokens
            else:
                tokens = torch.zeros(1, 1, 1)
            return self._mock_decode(tokens)
        
        if isinstance(frames_or_tokens, list):
            encoded_frames = frames_or_tokens
        else:
            encoded_frames = [(frames_or_tokens.to(self.device), None)]
        
        with torch.no_grad():
            audio = self.model.decode(encoded_frames)
        
        return audio
    
    def encode_to_flat(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
        interleave: bool = True
    ) -> torch.Tensor:
        """
        Encode audio to flattened token sequence for transformer input.
        
        Args:
            audio: Audio tensor
            sample_rate: Input sample rate
            interleave: If True, interleave codebook tokens; else concatenate
            
        Returns:
            flat_tokens: Flattened token sequence (batch, seq_len * n_codebooks)
        """
        tokens = self.encode(audio, sample_rate)
        batch, n_codebooks, seq_len = tokens.shape
        
        if interleave:
            # Interleave: [c0_t0, c1_t0, c2_t0, c3_t0, c0_t1, c1_t1, ...]
            flat_tokens = tokens.permute(0, 2, 1).reshape(batch, -1)
        else:
            # Concatenate: [c0_t0, c0_t1, ..., c1_t0, c1_t1, ...]
            flat_tokens = tokens.reshape(batch, -1)
        
        return flat_tokens
    
    def decode_from_flat(
        self,
        flat_tokens: torch.Tensor,
        interleave: bool = True
    ) -> torch.Tensor:
        """
        Decode flattened token sequence back to audio.
        
        Args:
            flat_tokens: Flattened token sequence
            interleave: Whether tokens were interleaved during encoding
            
        Returns:
            audio: Decoded audio waveform
        """
        batch = flat_tokens.shape[0]
        seq_len = flat_tokens.shape[1] // self.n_codebooks
        
        if interleave:
            tokens = flat_tokens.reshape(batch, seq_len, self.n_codebooks).permute(0, 2, 1)
        else:
            tokens = flat_tokens.reshape(batch, self.n_codebooks, seq_len)
        
        return self.decode(tokens)
    
    def get_token_duration(self, num_tokens: int) -> float:
        """Get duration in seconds for a given number of tokens."""
        return num_tokens / self.frame_rate
    
    def get_num_tokens(self, duration_seconds: float) -> int:
        """Get number of tokens needed for a given duration."""
        return int(duration_seconds * self.frame_rate)
    
    def _resample(
        self,
        audio: torch.Tensor,
        orig_sr: int,
        target_sr: int
    ) -> torch.Tensor:
        """Resample audio to target sample rate."""
        try:
            import torchaudio
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            return resampler(audio)
        except ImportError:
            # Fallback to simple interpolation
            ratio = target_sr / orig_sr
            new_length = int(audio.shape[-1] * ratio)
            return torch.nn.functional.interpolate(
                audio, size=new_length, mode='linear', align_corners=False
            )
    
    def _mock_encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Mock encoding for development without EnCodec."""
        batch = audio.shape[0]
        samples = audio.shape[-1]
        seq_len = samples // (self.sample_rate // self.frame_rate)
        
        # Generate random tokens
        tokens = torch.randint(
            0, self.codebook_size,
            (batch, self.n_codebooks, seq_len),
            device=audio.device
        )
        return tokens
    
    def _mock_decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mock decoding for development without EnCodec."""
        batch, n_codebooks, seq_len = tokens.shape
        samples = seq_len * (self.sample_rate // self.frame_rate)
        
        # Generate random audio
        audio = torch.randn(batch, 1, samples, device=tokens.device) * 0.1
        return audio
    
    def streaming_encode(
        self,
        audio: torch.Tensor,
        chunk_duration: float = 30.0,
        overlap: float = 0.5
    ) -> List[torch.Tensor]:
        """
        Encode long audio in streaming chunks with overlap.
        
        Args:
            audio: Long audio tensor
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            
        Returns:
            List of token chunks
        """
        chunk_samples = int(chunk_duration * self.sample_rate)
        overlap_samples = int(overlap * self.sample_rate)
        hop_samples = chunk_samples - overlap_samples
        
        total_samples = audio.shape[-1]
        chunks = []
        
        for start in range(0, total_samples - chunk_samples + 1, hop_samples):
            chunk = audio[..., start:start + chunk_samples]
            tokens = self.encode(chunk)
            chunks.append(tokens)
        
        # Handle last chunk
        if total_samples > start + chunk_samples:
            last_chunk = audio[..., -chunk_samples:]
            chunks.append(self.encode(last_chunk))
        
        return chunks


class EnCodecFeatureExtractor:
    """
    Extract continuous features from EnCodec for conditioning.
    
    Instead of discrete tokens, extract the continuous latent representations
    for use in diffusion models.
    """
    
    def __init__(self, wrapper: EnCodecWrapper):
        self.wrapper = wrapper
        self.model = wrapper.model
    
    @torch.no_grad()
    def extract_latents(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract continuous latent representations before quantization.
        
        Args:
            audio: Audio tensor
            
        Returns:
            latents: Continuous latent tensor
        """
        if self.model is None:
            # Mock latents
            batch = audio.shape[0]
            samples = audio.shape[-1]
            seq_len = samples // (self.wrapper.sample_rate // self.wrapper.frame_rate)
            latent_dim = 128  # EnCodec latent dimension
            return torch.randn(batch, latent_dim, seq_len, device=audio.device)
        
        # Get encoder output before quantization
        audio = audio.to(self.wrapper.device)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        # Access encoder directly
        latents = self.model.encoder(audio)
        return latents


def test_encodec_wrapper():
    """Test the EnCodec wrapper."""
    print("🧪 Testing EnCodec Wrapper...")
    
    # Initialize wrapper
    wrapper = EnCodecWrapper(
        model_type="encodec_24khz",
        target_bandwidth=6.0
    )
    
    # Create test audio (3 minutes)
    duration = 180.0  # 3 minutes
    samples = int(duration * wrapper.sample_rate)
    test_audio = torch.randn(1, samples)
    
    print(f"Input audio shape: {test_audio.shape}")
    print(f"Duration: {duration}s")
    
    # Encode
    tokens = wrapper.encode(test_audio)
    print(f"Token shape: {tokens.shape}")
    print(f"Expected tokens: {wrapper.get_num_tokens(duration)}")
    
    # Decode
    reconstructed = wrapper.decode(tokens)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Flat encoding
    flat = wrapper.encode_to_flat(test_audio, interleave=True)
    print(f"Flat token shape: {flat.shape}")
    
    # Verify round-trip
    reconstructed_flat = wrapper.decode_from_flat(flat, interleave=True)
    print(f"Round-trip shape: {reconstructed_flat.shape}")
    
    print("✅ EnCodec wrapper test complete!")


if __name__ == "__main__":
    test_encodec_wrapper()
