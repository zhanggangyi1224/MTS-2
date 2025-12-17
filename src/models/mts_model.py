"""
MTS Complete Model - Music-to-Structure Generative Model

This module implements the complete MTS system that:
1. Generates 3-4 minute music from text prompts
2. Maintains coherent musical structure
3. Provides 30-second compressed representations

Architecture follows Noise2Music + MusicGen hybrid approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
import math
from pathlib import Path

# Import our components
try:
    from .encodec_wrapper import EnCodecWrapper
    from .diffusion_unet import MTSDiffusionUNet, GaussianDiffusion
except ImportError:
    # For standalone testing
    from encodec_wrapper import EnCodecWrapper
    from diffusion_unet import MTSDiffusionUNet, GaussianDiffusion


@dataclass
class MTSConfig:
    """Configuration for MTS model."""
    
    # Audio settings
    sample_rate: int = 24000
    audio_channels: int = 1
    target_duration: float = 210.0  # 3.5 minutes default
    
    # EnCodec settings
    encodec_bandwidth: float = 6.0  # kbps
    encodec_latent_dim: int = 128
    encodec_frame_rate: int = 75
    
    # Diffusion settings
    diffusion_timesteps: int = 1000
    beta_schedule: str = "cosine"
    
    # U-Net settings
    model_channels: int = 256
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (2, 4, 8)
    num_heads: int = 8
    dropout: float = 0.1
    
    # Text encoder settings
    text_encoder: str = "t5-base"  # or "clap"
    text_dim: int = 768
    max_text_len: int = 256
    
    # Structure conditioning
    use_structure: bool = True
    structure_dim: int = 256
    max_sections: int = 20
    
    # 30-second compression
    compression_target_duration: float = 30.0
    compression_latent_dim: int = 64
    
    # Training
    cfg_dropout: float = 0.1  # Classifier-free guidance dropout


class TextEncoder(nn.Module):
    """
    Text encoder for conditioning.
    
    Supports T5 (semantic) or CLAP (audio-aligned) embeddings.
    """
    
    def __init__(
        self,
        encoder_type: str = "t5-base",
        output_dim: int = 768,
        max_length: int = 256,
        freeze: bool = True
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.output_dim = output_dim
        self.max_length = max_length
        
        self.encoder = None
        self.tokenizer = None
        
        self._load_encoder(encoder_type, freeze)
    
    def _load_encoder(self, encoder_type: str, freeze: bool):
        """Load pre-trained text encoder."""
        try:
            from transformers import T5EncoderModel, T5Tokenizer
            
            if "t5" in encoder_type.lower():
                self.tokenizer = T5Tokenizer.from_pretrained(encoder_type)
                self.encoder = T5EncoderModel.from_pretrained(encoder_type)
                
                if freeze:
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                
                print(f"✅ Loaded {encoder_type} text encoder")
            else:
                print(f"⚠️  Encoder type {encoder_type} not supported, using mock encoder")
                self.encoder = None
                
        except ImportError:
            print("⚠️  transformers not available, using mock text encoder")
            self.encoder = None
    
    def forward(
        self,
        text: Union[str, List[str], torch.Tensor],
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text: Input text (string, list of strings, or pre-tokenized tensor)
            device: Target device
            
        Returns:
            Text embeddings (batch, seq_len, dim)
        """
        if isinstance(text, torch.Tensor):
            # Already encoded
            return text
        
        if self.encoder is None:
            # Mock encoding
            if isinstance(text, str):
                batch_size = 1
            else:
                batch_size = len(text)
            return torch.randn(batch_size, self.max_length, self.output_dim, device=device)
        
        # Tokenize
        if isinstance(text, str):
            text = [text]
        
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Encode
        with torch.no_grad():
            outputs = self.encoder(**tokens)
        
        return outputs.last_hidden_state


class StructureEncoder(nn.Module):
    """
    Encode musical structure annotations to conditioning embeddings.
    
    Structure format:
    - Section types (intro, verse, chorus, etc.)
    - Section timestamps
    - Section properties (energy, key, tempo)
    """
    
    def __init__(
        self,
        num_sections: int = 20,
        section_vocab_size: int = 20,  # Number of section types
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.num_sections = num_sections
        self.hidden_dim = hidden_dim
        
        # Section type embedding
        self.section_embed = nn.Embedding(section_vocab_size, hidden_dim)
        
        # Continuous property encoding
        self.property_proj = nn.Linear(5, hidden_dim)  # start, end, duration, energy, importance
        
        # Positional encoding for section order
        self.pos_embed = nn.Parameter(torch.randn(1, num_sections, hidden_dim) * 0.02)
        
        # Transformer for section relationships
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        section_types: torch.Tensor,  # (batch, num_sections)
        section_props: torch.Tensor,  # (batch, num_sections, 5)
        mask: Optional[torch.Tensor] = None  # (batch, num_sections)
    ) -> torch.Tensor:
        """
        Encode structure to embeddings.
        
        Returns:
            Structure embeddings (batch, num_sections, output_dim)
        """
        batch_size = section_types.shape[0]
        
        # Embed section types
        type_emb = self.section_embed(section_types)
        
        # Encode properties
        prop_emb = self.property_proj(section_props)
        
        # Combine
        h = type_emb + prop_emb + self.pos_embed[:, :section_types.shape[1]]
        
        # Transform
        if mask is not None:
            h = self.transformer(h, src_key_padding_mask=~mask)
        else:
            h = self.transformer(h)
        
        return self.output_proj(h)


class CompressionNetwork(nn.Module):
    """
    Network to compress full song to 30-second representation.
    
    This creates a condensed representation that captures:
    - Main musical themes
    - Section transitions
    - Overall structure
    
    Like a "musical abstract" of the full piece.
    """
    
    def __init__(
        self,
        input_dim: int = 128,  # EnCodec latent dim
        hidden_dim: int = 256,
        output_dim: int = 64,
        compression_ratio: float = 7.0,  # 3.5 min -> 30 sec
        num_layers: int = 4
    ):
        super().__init__()
        
        self.compression_ratio = compression_ratio
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Encoder (full song -> compressed)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 7, padding=3),
            nn.SiLU(),
            nn.GroupNorm(8, hidden_dim),
        )
        
        # Learned compression via pooling + conv
        self.compress_blocks = nn.ModuleList()
        current_dim = hidden_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            self.compress_blocks.append(nn.Sequential(
                nn.Conv1d(current_dim, out_dim, 3, padding=1),
                nn.SiLU(),
                nn.GroupNorm(8 if out_dim >= 64 else 4, out_dim),
                nn.AvgPool1d(2)  # Halve sequence length
            ))
            current_dim = out_dim
        
        # Decoder (compressed -> expanded)
        self.decoder_blocks = nn.ModuleList()
        current_dim = output_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else input_dim
            self.decoder_blocks.append(nn.Sequential(
                nn.ConvTranspose1d(current_dim, out_dim, 4, stride=2, padding=1),
                nn.SiLU(),
                nn.GroupNorm(8 if out_dim >= 64 else 4, out_dim),
            ))
            current_dim = out_dim
        
        # Final refinement
        self.output_proj = nn.Conv1d(input_dim, input_dim, 3, padding=1)
    
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress full-length latent to 30-second representation.
        
        Args:
            x: Full latent (batch, input_dim, full_seq_len)
            
        Returns:
            Compressed latent (batch, output_dim, compressed_seq_len)
        """
        h = self.encoder(x)
        
        for block in self.compress_blocks:
            h = block(h)
        
        return h
    
    def expand(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Expand 30-second representation to full length.
        
        Args:
            z: Compressed latent (batch, output_dim, compressed_seq_len)
            target_len: Target sequence length
            
        Returns:
            Expanded latent (batch, input_dim, target_len)
        """
        h = z
        
        for block in self.decoder_blocks:
            h = block(h)
        
        # Adjust to exact target length
        if h.shape[-1] != target_len:
            h = F.interpolate(h, size=target_len, mode='linear', align_corners=False)
        
        return self.output_proj(h)
    
    def forward(
        self,
        x: torch.Tensor,
        return_compressed: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass: compress then expand (for training).
        
        Args:
            x: Full latent
            return_compressed: Whether to return compressed representation
            
        Returns:
            Reconstructed latent (and optionally compressed)
        """
        z = self.compress(x)
        x_recon = self.expand(z, x.shape[-1])
        
        if return_compressed:
            return x_recon, z
        return x_recon


class MTSModel(nn.Module):
    """
    Complete Music-to-Structure generative model.
    
    Generates full-length (3-4 min) music from:
    - Text prompts
    - Optional structure conditioning
    - Noise input
    
    Also provides 30-second compressed representations.
    """
    
    def __init__(self, config: MTSConfig):
        super().__init__()
        
        self.config = config
        
        # Audio codec
        self.encodec = EnCodecWrapper(
            model_type="encodec_24khz",
            target_bandwidth=config.encodec_bandwidth
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            encoder_type=config.text_encoder,
            output_dim=config.text_dim,
            max_length=config.max_text_len
        )
        
        # Structure encoder
        if config.use_structure:
            self.structure_encoder = StructureEncoder(
                num_sections=config.max_sections,
                hidden_dim=config.structure_dim,
                output_dim=config.structure_dim
            )
        else:
            self.structure_encoder = None
        
        # Diffusion model
        self.unet = MTSDiffusionUNet(
            in_channels=config.encodec_latent_dim,
            out_channels=config.encodec_latent_dim,
            model_channels=config.model_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attention_resolutions,
            context_dim=config.text_dim,
            structure_dim=config.structure_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            use_structure_conditioning=config.use_structure
        )
        
        self.diffusion = GaussianDiffusion(
            self.unet,
            num_timesteps=config.diffusion_timesteps,
            beta_schedule=config.beta_schedule
        )
        
        # Projection from EnCodec codebooks to latent space
        # EnCodec uses 8 codebooks at 6.0kbps, we need to project to latent_dim
        self.encodec_proj = nn.Conv1d(
            self.encodec.n_codebooks,  # 8 codebooks
            config.encodec_latent_dim,  # 128 latent dim
            kernel_size=1
        )

        # 30-second compression network
        self.compression = CompressionNetwork(
            input_dim=config.encodec_latent_dim,
            hidden_dim=256,
            output_dim=config.compression_latent_dim
        )

        # CFG dropout embedding
        self.null_text_emb = nn.Parameter(torch.randn(1, 1, config.text_dim) * 0.02)
        self.null_struct_emb = nn.Parameter(torch.randn(1, 1, config.structure_dim) * 0.02)
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to latent tokens."""
        frames, codes = self.encodec.encode(audio)
        # codes shape: (batch, n_codebooks=8, seq_len)
        # Project to latent space: (batch, latent_dim=128, seq_len)
        latents = self.encodec_proj(codes.float())
        return latents
    
    def decode_audio(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent tokens to audio waveform."""
        return self.encodec.decode(latents)
    
    def encode_text(
        self,
        text: Union[str, List[str]],
        device: str = "cuda"
    ) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        return self.text_encoder(text, device)
    
    def encode_structure(
        self,
        structure: Dict,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Encode structure annotation to embeddings.
        
        Args:
            structure: Dict with keys:
                - section_types: List of section type IDs
                - section_props: List of [start, end, duration, energy, importance]
        """
        if self.structure_encoder is None:
            return None
        
        # Convert to tensors
        section_types = torch.tensor(structure["section_types"], device=device).unsqueeze(0)
        section_props = torch.tensor(structure["section_props"], device=device, dtype=torch.float32).unsqueeze(0)
        
        return self.structure_encoder(section_types, section_props)
    
    def compute_loss(
        self,
        audio: torch.Tensor,
        text: Union[str, List[str]],
        structure: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Compute training loss (simplified interface for training loop).

        Args:
            audio: Audio waveform (batch, samples)
            text: Text prompts
            structure: Optional structure annotations

        Returns:
            Total loss value
        """
        losses = self.training_step(audio, text, structure)
        return losses["loss"]

    def training_step(
        self,
        audio: torch.Tensor,
        text: Union[str, List[str]],
        structure: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step.

        Args:
            audio: Audio waveform (batch, samples)
            text: Text prompts
            structure: Optional structure annotations

        Returns:
            Dict with losses
        """
        device = audio.device
        batch_size = audio.shape[0]

        # Encode audio to latents
        # encode_audio now projects from codebooks to latent space
        latents = self.encode_audio(audio)

        # Encode conditioning
        text_emb = self.encode_text(text, device)

        struct_emb = None
        if structure is not None and self.structure_encoder is not None:
            struct_emb = self.encode_structure(structure, device)

        # Classifier-free guidance dropout
        if self.training and self.config.cfg_dropout > 0:
            drop_text = torch.rand(batch_size, device=device) < self.config.cfg_dropout
            drop_struct = torch.rand(batch_size, device=device) < self.config.cfg_dropout

            null_text = self.null_text_emb.expand(batch_size, -1, -1)
            text_emb = torch.where(drop_text[:, None, None], null_text, text_emb)

            if struct_emb is not None:
                null_struct = self.null_struct_emb.expand(batch_size, -1, -1)
                struct_emb = torch.where(drop_struct[:, None, None], null_struct, struct_emb)

        # Sample timesteps
        t = torch.randint(0, self.config.diffusion_timesteps, (batch_size,), device=device)

        # Diffusion loss
        diff_loss = self.diffusion.p_losses(latents, t, text_emb, struct_emb)

        # Compression reconstruction loss
        latents_recon, compressed = self.compression(latents, return_compressed=True)
        compress_loss = F.mse_loss(latents_recon, latents)

        # Total loss
        total_loss = diff_loss + 0.1 * compress_loss

        return {
            "loss": total_loss,
            "diffusion_loss": diff_loss,
            "compression_loss": compress_loss
        }
    
    @torch.no_grad()
    def generate(
        self,
        text: Union[str, List[str]],
        structure: Optional[Dict] = None,
        duration: float = 210.0,
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate music from text prompt.
        
        Args:
            text: Text prompt(s)
            structure: Optional structure annotation
            duration: Target duration in seconds
            cfg_scale: Classifier-free guidance scale
            num_inference_steps: Number of diffusion steps (for DDIM)
            device: Generation device
            
        Returns:
            audio: Generated audio waveform (batch, samples)
            compressed: 30-second compressed representation
        """
        self.eval()
        
        # Calculate sequence length
        seq_len = int(duration * self.config.encodec_frame_rate)
        batch_size = 1 if isinstance(text, str) else len(text)
        
        # Encode conditioning
        text_emb = self.encode_text(text, device)
        
        struct_emb = None
        if structure is not None and self.structure_encoder is not None:
            struct_emb = self.encode_structure(structure, device)
        
        # Generate latents via diffusion
        shape = (batch_size, self.config.encodec_latent_dim, seq_len)
        latents = self.diffusion.sample(
            shape,
            text_emb,
            struct_emb,
            cfg_scale=cfg_scale,
            device=device
        )
        
        # Get 30-second compression
        compressed = self.compression.compress(latents)
        
        # Decode to audio
        audio = self.decode_audio(latents)
        
        return audio, compressed
    
    @torch.no_grad()
    def generate_from_compressed(
        self,
        compressed: torch.Tensor,
        target_duration: float = 210.0,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Generate full song from 30-second compressed representation.
        
        Args:
            compressed: Compressed representation from generate()
            target_duration: Target duration to expand to
            device: Device
            
        Returns:
            Full audio waveform
        """
        self.eval()
        
        target_len = int(target_duration * self.config.encodec_frame_rate)
        
        # Expand compressed representation
        latents = self.compression.expand(compressed, target_len)
        
        # Decode to audio
        audio = self.decode_audio(latents)
        
        return audio
    
    def get_30_second_preview(
        self,
        audio_or_latents: torch.Tensor,
        is_audio: bool = True
    ) -> torch.Tensor:
        """
        Get 30-second compressed preview of a song.
        
        Args:
            audio_or_latents: Either audio waveform or EnCodec latents
            is_audio: True if input is audio, False if latents
            
        Returns:
            30-second audio preview
        """
        if is_audio:
            latents = self.encode_audio(audio_or_latents)
        else:
            latents = audio_or_latents
        
        # Compress
        compressed = self.compression.compress(latents)
        
        # Generate 30-second audio from compressed
        target_len = int(30.0 * self.config.encodec_frame_rate)
        preview_latents = self.compression.expand(compressed, target_len)
        
        return self.decode_audio(preview_latents)


def create_mts_model(
    model_size: str = "base",
    use_structure: bool = True
) -> MTSModel:
    """
    Factory function to create MTS model with preset configurations.
    
    Args:
        model_size: One of "small", "base", "large"
        use_structure: Whether to use structure conditioning
        
    Returns:
        Configured MTSModel
    """
    configs = {
        "small": MTSConfig(
            model_channels=128,
            channel_multipliers=(1, 2, 4),
            num_heads=4,
            diffusion_timesteps=500
        ),
        "base": MTSConfig(
            model_channels=256,
            channel_multipliers=(1, 2, 4, 8),
            num_heads=8,
            diffusion_timesteps=1000
        ),
        "large": MTSConfig(
            model_channels=384,
            channel_multipliers=(1, 2, 4, 8, 8),
            num_heads=12,
            diffusion_timesteps=1000
        )
    }
    
    config = configs.get(model_size, configs["base"])
    config.use_structure = use_structure
    
    return MTSModel(config)


def test_mts_model():
    """Test the complete MTS model."""
    print("🧪 Testing Complete MTS Model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = create_mts_model("small", use_structure=True).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test generation (short duration for testing)
    print("\nTesting generation...")
    
    text = "A peaceful piano melody with gentle strings and soft percussion"
    structure = {
        "section_types": [0, 1, 2, 1, 2, 3],  # intro, verse, chorus, verse, chorus, outro
        "section_props": [
            [0, 20, 20, 0.3, 0.5],   # intro
            [20, 60, 40, 0.5, 0.7],  # verse
            [60, 100, 40, 0.9, 1.0], # chorus
            [100, 140, 40, 0.5, 0.7], # verse
            [140, 180, 40, 0.9, 1.0], # chorus
            [180, 210, 30, 0.4, 0.5]  # outro
        ]
    }
    
    # Generate (very short for testing)
    audio, compressed = model.generate(
        text=text,
        structure=structure,
        duration=10.0,  # 10 seconds for testing
        cfg_scale=3.0,
        device=device
    )
    
    print(f"Generated audio shape: {audio.shape}")
    print(f"Compressed shape: {compressed.shape}")
    
    # Test preview generation
    preview = model.get_30_second_preview(audio, is_audio=True)
    print(f"Preview shape: {preview.shape}")
    
    print("\n✅ MTS Model test complete!")


if __name__ == "__main__":
    test_mts_model()
