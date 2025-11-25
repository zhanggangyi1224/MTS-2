"""
MTS Diffusion Model - U-Net based diffusion for music generation

Implements a diffusion model following Noise2Music architecture:
- U-Net backbone with cross-attention for text conditioning
- Structure conditioning via additional embeddings
- Cascaded refinement for high-quality output

The model operates in EnCodec's latent space for efficiency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from einops import rearrange, repeat


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Tensor of shape (batch,) with timestep values
            
        Returns:
            Embeddings of shape (batch, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        
        return embedding


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm and SiLU activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 8
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResNetBlock(nn.Module):
    """ResNet block with timestep conditioning."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        groups: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels, groups=groups)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, seq_len)
            time_emb: Timestep embedding (batch, time_emb_dim)
        """
        h = self.conv1(x)
        
        # Add time conditioning
        time_emb = self.time_mlp(time_emb)
        scale, shift = time_emb.chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        
        h = self.dropout(self.norm2(self.conv2(F.silu(h))))
        
        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Cross-attention layer for text conditioning."""
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        
        inner_dim = heads * dim_head
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor (batch, seq_len, query_dim)
            context: Context tensor (batch, context_len, context_dim)
        """
        if context is None:
            context = x
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and cross-attention."""
    
    def __init__(
        self,
        dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CrossAttention(dim, dim, heads, dim_head, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, context_dim, heads, dim_head, dropout)
        
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.norm1(x))
        
        # Cross-attention (if context provided)
        if context is not None:
            x = x + self.cross_attn(self.norm2(x), context)
        
        # Feed-forward
        x = x + self.ff(self.norm3(x))
        
        return x


class DownBlock(nn.Module):
    """Downsampling block for U-Net."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        context_dim: int,
        num_layers: int = 2,
        has_attention: bool = True,
        downsample: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResNetBlock(in_ch, out_channels, time_emb_dim, dropout=dropout))
            
            if has_attention:
                self.attn_blocks.append(TransformerBlock(out_channels, context_dim, dropout=dropout))
            else:
                self.attn_blocks.append(nn.Identity())
        
        self.downsample = nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1) if downsample else None
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hiddens = []
        
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x, time_emb)
            
            if not isinstance(attn_block, nn.Identity):
                # Reshape for attention: (batch, channels, seq) -> (batch, seq, channels)
                x = rearrange(x, 'b c s -> b s c')
                x = attn_block(x, context)
                x = rearrange(x, 'b s c -> b c s')
            
            hiddens.append(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x, hiddens


class UpBlock(nn.Module):
    """Upsampling block for U-Net."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_emb_dim: int,
        context_dim: int,
        num_layers: int = 2,
        has_attention: bool = True,
        upsample: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = in_channels + skip_channels if i == 0 else out_channels + skip_channels
            self.res_blocks.append(ResNetBlock(in_ch, out_channels, time_emb_dim, dropout=dropout))
            
            if has_attention:
                self.attn_blocks.append(TransformerBlock(out_channels, context_dim, dropout=dropout))
            else:
                self.attn_blocks.append(nn.Identity())
        
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, 4, stride=2, padding=1) if upsample else None
    
    def forward(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor],
        time_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        
        for i, (res_block, attn_block) in enumerate(zip(self.res_blocks, self.attn_blocks)):
            skip = skip_connections.pop() if skip_connections else None
            if skip is not None:
                # Handle size mismatch
                if x.shape[-1] != skip.shape[-1]:
                    x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            
            x = res_block(x, time_emb)
            
            if not isinstance(attn_block, nn.Identity):
                x = rearrange(x, 'b c s -> b s c')
                x = attn_block(x, context)
                x = rearrange(x, 'b s c -> b c s')
        
        return x


class MTSDiffusionUNet(nn.Module):
    """
    U-Net diffusion model for MTS music generation.
    
    Architecture:
    - Encoder: Series of down blocks with attention
    - Bottleneck: Transformer blocks
    - Decoder: Series of up blocks with skip connections
    - Conditioning: Timestep + Text + Structure
    """
    
    def __init__(
        self,
        in_channels: int = 128,  # EnCodec latent dim
        out_channels: int = 128,
        model_channels: int = 256,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (2, 4, 8),  # Which levels have attention
        context_dim: int = 768,  # Text embedding dim (T5-base)
        structure_dim: int = 256,  # Structure embedding dim
        num_heads: int = 8,
        dropout: float = 0.1,
        use_structure_conditioning: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.use_structure_conditioning = use_structure_conditioning
        
        # Timestep embedding
        time_emb_dim = model_channels * 4
        self.time_emb = nn.Sequential(
            SinusoidalPositionalEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Structure embedding (optional)
        if use_structure_conditioning:
            self.structure_proj = nn.Sequential(
                nn.Linear(structure_dim, context_dim),
                nn.SiLU(),
                nn.Linear(context_dim, context_dim)
            )
        
        # Input projection
        self.input_proj = nn.Conv1d(in_channels, model_channels, 3, padding=1)
        
        # Encoder (down path)
        self.down_blocks = nn.ModuleList()
        channels = [model_channels]
        ch = model_channels
        
        for level, mult in enumerate(channel_multipliers):
            out_ch = model_channels * mult
            has_attn = (level + 1) in attention_resolutions
            downsample = level < len(channel_multipliers) - 1
            
            self.down_blocks.append(
                DownBlock(
                    ch, out_ch, time_emb_dim, context_dim,
                    num_layers=num_res_blocks,
                    has_attention=has_attn,
                    downsample=downsample,
                    dropout=dropout
                )
            )
            ch = out_ch
            channels.append(ch)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResNetBlock(ch, ch, time_emb_dim, dropout=dropout),
            TransformerBlock(ch, context_dim, heads=num_heads, dropout=dropout),
            ResNetBlock(ch, ch, time_emb_dim, dropout=dropout)
        ])
        
        # Decoder (up path)
        self.up_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = model_channels * mult
            has_attn = (level + 1) in attention_resolutions
            upsample = level > 0
            
            # Calculate skip channels
            skip_ch = channels.pop()
            
            in_ch = ch
            self.up_blocks.append(
                UpBlock(
                    in_ch, out_ch, skip_ch, time_emb_dim, context_dim,
                    num_layers=num_res_blocks,
                    has_attention=has_attn,
                    upsample=upsample,
                    dropout=dropout
                )
            )
            ch = out_ch
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv1d(ch, out_channels, 3, padding=1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        structure_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of diffusion U-Net.
        
        Args:
            x: Noisy latent (batch, in_channels, seq_len)
            timesteps: Diffusion timesteps (batch,)
            text_emb: Text conditioning (batch, text_len, context_dim)
            structure_emb: Structure conditioning (batch, struct_len, structure_dim)
            
        Returns:
            Noise prediction (batch, out_channels, seq_len)
        """
        # Timestep embedding
        t_emb = self.time_emb(timesteps)
        
        # Combine context embeddings
        context = text_emb
        
        if self.use_structure_conditioning and structure_emb is not None:
            struct_proj = self.structure_proj(structure_emb)
            if context is not None:
                context = torch.cat([context, struct_proj], dim=1)
            else:
                context = struct_proj
        
        # Input projection
        h = self.input_proj(x)
        
        # Encoder
        skip_connections = []
        for down_block in self.down_blocks:
            h, hiddens = down_block(h, t_emb, context)
            skip_connections.extend(hiddens)
        
        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, ResNetBlock):
                h = block(h, t_emb)
            else:
                h = rearrange(h, 'b c s -> b s c')
                h = block(h, context)
                h = rearrange(h, 'b s c -> b c s')
        
        # Decoder
        for up_block in self.up_blocks:
            h = up_block(h, skip_connections, t_emb, context)
        
        # Output
        return self.output_proj(h)


class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process for training and sampling.
    
    Implements:
    - Forward diffusion (q): Add noise to data
    - Reverse diffusion (p): Denoise to generate data
    - Various sampling methods (DDPM, DDIM)
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        loss_type: str = "l2",
        clip_denoised: bool = True
    ):
        super().__init__()
        
        self.model = model
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type
        self.clip_denoised = clip_denoised
        
        # Compute diffusion schedule
        betas = self._make_beta_schedule(beta_schedule, num_timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def _make_beta_schedule(self, schedule: str, num_timesteps: int) -> torch.Tensor:
        """Create noise schedule."""
        if schedule == "linear":
            beta_start = 1e-4
            beta_end = 0.02
            return torch.linspace(beta_start, beta_end, num_timesteps)
        
        elif schedule == "cosine":
            # Cosine schedule from Improved DDPM
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0)
        
        Args:
            x_0: Clean data
            t: Timesteps
            noise: Optional pre-sampled noise
            
        Returns:
            Noisy data x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def p_losses(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        structure_emb: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            x_0: Clean data
            t: Timesteps
            text_emb: Text conditioning
            structure_emb: Structure conditioning
            noise: Optional pre-sampled noise
            
        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        noise_pred = self.model(x_t, t, text_emb, structure_emb)
        
        # Compute loss
        if self.loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise_pred, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(noise_pred, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        text_emb: Optional[torch.Tensor] = None,
        structure_emb: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Single step of reverse diffusion.
        
        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            text_emb: Text conditioning
            structure_emb: Structure conditioning
            cfg_scale: Classifier-free guidance scale
            
        Returns:
            Denoised data at timestep t-1
        """
        batch_size = x_t.shape[0]
        t_batch = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        
        # Predict noise
        noise_pred = self.model(x_t, t_batch, text_emb, structure_emb)
        
        # Classifier-free guidance
        if cfg_scale != 1.0 and text_emb is not None:
            noise_pred_uncond = self.model(x_t, t_batch, None, structure_emb)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)
        
        # Compute mean
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        
        pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod * noise_pred) / self.sqrt_alphas_cumprod[t]
        
        if self.clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        # Compute posterior mean
        mean = self.posterior_mean_coef1[t] * pred_x0 + self.posterior_mean_coef2[t] * x_t
        
        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t]
            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            x_t_minus_1 = mean
        
        return x_t_minus_1
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        text_emb: Optional[torch.Tensor] = None,
        structure_emb: Optional[torch.Tensor] = None,
        cfg_scale: float = 7.5,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Generate samples via reverse diffusion.
        
        Args:
            shape: Shape of samples to generate (batch, channels, seq_len)
            text_emb: Text conditioning
            structure_emb: Structure conditioning
            cfg_scale: Classifier-free guidance scale
            device: Device for generation
            
        Returns:
            Generated samples
        """
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t, text_emb, structure_emb, cfg_scale)
        
        return x


def test_diffusion_model():
    """Test the diffusion model."""
    print("🧪 Testing Diffusion Model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    unet = MTSDiffusionUNet(
        in_channels=128,
        out_channels=128,
        model_channels=256,
        channel_multipliers=(1, 2, 4),
        context_dim=768,
        structure_dim=256
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # Create diffusion wrapper
    diffusion = GaussianDiffusion(unet, num_timesteps=1000).to(device)
    
    # Test forward pass
    batch_size = 2
    seq_len = 750  # ~10 seconds at 75 fps
    
    x = torch.randn(batch_size, 128, seq_len, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    text_emb = torch.randn(batch_size, 77, 768, device=device)
    structure_emb = torch.randn(batch_size, 10, 256, device=device)
    
    # Training loss
    loss = diffusion.p_losses(x, t, text_emb, structure_emb)
    print(f"Training loss: {loss.item():.4f}")
    
    # Sampling (just a few steps for testing)
    print("Testing sampling...")
    with torch.no_grad():
        sample = diffusion.sample(
            (1, 128, 100),
            text_emb[:1],
            structure_emb[:1],
            cfg_scale=7.5,
            device=device
        )
    print(f"Sample shape: {sample.shape}")
    
    print("✅ Diffusion model test complete!")


if __name__ == "__main__":
    test_diffusion_model()
