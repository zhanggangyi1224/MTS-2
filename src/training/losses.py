"""
MTS Losses - Comprehensive loss functions for music generation

Implements multiple loss types:
1. Diffusion loss (noise prediction)
2. Multi-scale spectral loss (audio quality)
3. Perceptual loss (high-level features)
4. Structure consistency loss (section coherence)
5. Compression reconstruction loss (30-sec representation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-scale STFT loss for audio quality.
    
    Computes spectral convergence and log magnitude loss at multiple
    STFT resolutions for better frequency coverage.
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [128, 256, 512],
        win_sizes: List[int] = [512, 1024, 2048],
        sample_rate: int = 24000,
        mag_weight: float = 1.0,
        convergence_weight: float = 1.0
    ):
        super().__init__()
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.sample_rate = sample_rate
        self.mag_weight = mag_weight
        self.convergence_weight = convergence_weight
        
        # Pre-compute windows
        self.windows = nn.ParameterList([
            nn.Parameter(torch.hann_window(win_size), requires_grad=False)
            for win_size in win_sizes
        ])
    
    def stft(
        self,
        x: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_size: int,
        window: torch.Tensor
    ) -> torch.Tensor:
        """Compute STFT magnitude."""
        # Ensure correct window device
        window = window.to(x.device)
        
        # Pad if necessary
        if x.shape[-1] < fft_size:
            x = F.pad(x, (0, fft_size - x.shape[-1]))
        
        # Compute STFT
        stft = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_size,
            window=window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        
        return torch.abs(stft)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-scale spectral loss.
        
        Args:
            pred: Predicted audio (batch, samples) or (batch, 1, samples)
            target: Target audio (batch, samples) or (batch, 1, samples)
            
        Returns:
            Total loss and dict of individual losses
        """
        # Ensure 2D input
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        
        total_loss = 0.0
        losses = {}
        
        for i, (fft_size, hop_size, win_size) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_sizes)
        ):
            window = self.windows[i]
            
            # Compute spectrograms
            pred_mag = self.stft(pred, fft_size, hop_size, win_size, window)
            target_mag = self.stft(target, fft_size, hop_size, win_size, window)
            
            # Spectral convergence loss
            sc_loss = torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)
            
            # Log magnitude loss
            log_pred = torch.log(pred_mag + 1e-8)
            log_target = torch.log(target_mag + 1e-8)
            mag_loss = F.l1_loss(log_pred, log_target)
            
            # Combine
            scale_loss = self.convergence_weight * sc_loss + self.mag_weight * mag_loss
            total_loss = total_loss + scale_loss
            
            losses[f'spectral_convergence_{fft_size}'] = sc_loss
            losses[f'log_magnitude_{fft_size}'] = mag_loss
        
        # Average over scales
        total_loss = total_loss / len(self.fft_sizes)
        losses['total_spectral'] = total_loss
        
        return total_loss, losses


class MelSpectrogramLoss(nn.Module):
    """
    Mel spectrogram loss for perceptual similarity.
    
    Uses mel-scale which better matches human perception.
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        if HAS_TORCHAUDIO:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max or sample_rate // 2
            )
        else:
            self.mel_transform = None
            print("⚠️  torchaudio not available, using simple spectral loss")
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute mel spectrogram L1 loss."""
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        
        if self.mel_transform is not None:
            self.mel_transform = self.mel_transform.to(pred.device)
            
            pred_mel = self.mel_transform(pred)
            target_mel = self.mel_transform(target)
            
            # Log mel spectrogram
            pred_mel = torch.log(pred_mel + 1e-8)
            target_mel = torch.log(target_mel + 1e-8)
            
            return F.l1_loss(pred_mel, target_mel)
        else:
            # Fallback: simple L1
            return F.l1_loss(pred, target)


class StructureConsistencyLoss(nn.Module):
    """
    Loss for maintaining structure consistency.
    
    Encourages:
    1. Similar sections to sound similar (e.g., chorus repetitions)
    2. Smooth transitions between sections
    3. Energy profile matching
    """
    
    def __init__(
        self,
        frame_rate: int = 75,  # EnCodec frame rate
        section_similarity_weight: float = 1.0,
        transition_smoothness_weight: float = 0.5,
        energy_profile_weight: float = 0.5
    ):
        super().__init__()
        
        self.frame_rate = frame_rate
        self.section_similarity_weight = section_similarity_weight
        self.transition_smoothness_weight = transition_smoothness_weight
        self.energy_profile_weight = energy_profile_weight
    
    def forward(
        self,
        latents: torch.Tensor,
        structure: Dict,
        target_latents: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute structure consistency loss.
        
        Args:
            latents: Generated latents (batch, channels, seq_len)
            structure: Structure annotation dict with section_types and section_props
            target_latents: Optional target latents for supervised comparison
            
        Returns:
            Total loss and dict of components
        """
        losses = {}
        total_loss = 0.0
        
        batch_size = latents.shape[0]
        
        # 1. Section similarity loss
        # Sections of the same type should be similar
        if self.section_similarity_weight > 0:
            sim_loss = self._section_similarity_loss(latents, structure)
            total_loss = total_loss + self.section_similarity_weight * sim_loss
            losses['section_similarity'] = sim_loss
        
        # 2. Transition smoothness loss
        # Transitions between sections should be smooth
        if self.transition_smoothness_weight > 0:
            trans_loss = self._transition_smoothness_loss(latents, structure)
            total_loss = total_loss + self.transition_smoothness_weight * trans_loss
            losses['transition_smoothness'] = trans_loss
        
        # 3. Energy profile loss
        # Match expected energy profile from structure
        if self.energy_profile_weight > 0:
            energy_loss = self._energy_profile_loss(latents, structure)
            total_loss = total_loss + self.energy_profile_weight * energy_loss
            losses['energy_profile'] = energy_loss
        
        losses['total_structure'] = total_loss
        
        return total_loss, losses
    
    def _section_similarity_loss(
        self,
        latents: torch.Tensor,
        structure: Dict
    ) -> torch.Tensor:
        """Encourage similar sections to have similar latent representations."""
        
        section_types = structure.get('section_types', [])
        section_props = structure.get('section_props', [])
        
        if len(section_types) < 2:
            return torch.tensor(0.0, device=latents.device)
        
        # Group sections by type
        type_to_sections = {}
        for i, (sec_type, props) in enumerate(zip(section_types, section_props)):
            if sec_type not in type_to_sections:
                type_to_sections[sec_type] = []
            
            start_frame = int(props[0] * self.frame_rate)
            end_frame = int(props[1] * self.frame_rate)
            type_to_sections[sec_type].append((start_frame, end_frame))
        
        # Compute similarity loss for sections of same type
        similarity_losses = []
        
        for sec_type, sections in type_to_sections.items():
            if len(sections) < 2:
                continue
            
            # Extract section latents and compute pairwise similarity
            for i in range(len(sections) - 1):
                for j in range(i + 1, len(sections)):
                    start_i, end_i = sections[i]
                    start_j, end_j = sections[j]
                    
                    # Ensure valid indices
                    end_i = min(end_i, latents.shape[-1])
                    end_j = min(end_j, latents.shape[-1])
                    
                    if end_i <= start_i or end_j <= start_j:
                        continue
                    
                    # Extract sections
                    sec_i = latents[..., start_i:end_i]
                    sec_j = latents[..., start_j:end_j]
                    
                    # Pool to same size for comparison
                    min_len = min(sec_i.shape[-1], sec_j.shape[-1])
                    if min_len < 1:
                        continue
                    
                    sec_i = F.adaptive_avg_pool1d(sec_i, min_len)
                    sec_j = F.adaptive_avg_pool1d(sec_j, min_len)
                    
                    # Similarity loss (we want them similar, so minimize distance)
                    sim_loss = F.mse_loss(sec_i, sec_j)
                    similarity_losses.append(sim_loss)
        
        if similarity_losses:
            return torch.stack(similarity_losses).mean()
        return torch.tensor(0.0, device=latents.device)
    
    def _transition_smoothness_loss(
        self,
        latents: torch.Tensor,
        structure: Dict
    ) -> torch.Tensor:
        """Encourage smooth transitions at section boundaries."""
        
        section_props = structure.get('section_props', [])
        
        if len(section_props) < 2:
            return torch.tensor(0.0, device=latents.device)
        
        smoothness_losses = []
        transition_window = int(1.0 * self.frame_rate)  # 1 second window
        
        for i in range(len(section_props) - 1):
            boundary_frame = int(section_props[i][1] * self.frame_rate)
            
            # Get frames around boundary
            start = max(0, boundary_frame - transition_window)
            end = min(latents.shape[-1], boundary_frame + transition_window)
            
            if end <= start + 1:
                continue
            
            # Compute local variation (smoothness = low variation)
            transition_region = latents[..., start:end]
            diff = torch.diff(transition_region, dim=-1)
            smoothness_loss = torch.mean(diff ** 2)
            smoothness_losses.append(smoothness_loss)
        
        if smoothness_losses:
            return torch.stack(smoothness_losses).mean()
        return torch.tensor(0.0, device=latents.device)
    
    def _energy_profile_loss(
        self,
        latents: torch.Tensor,
        structure: Dict
    ) -> torch.Tensor:
        """Match energy profile to structure annotations."""
        
        section_props = structure.get('section_props', [])
        
        if not section_props:
            return torch.tensor(0.0, device=latents.device)
        
        # Compute latent energy
        latent_energy = torch.mean(latents ** 2, dim=1)  # (batch, seq_len)
        
        # Build target energy profile
        seq_len = latents.shape[-1]
        target_energy = torch.zeros(latents.shape[0], seq_len, device=latents.device)
        
        for props in section_props:
            start_frame = int(props[0] * self.frame_rate)
            end_frame = min(int(props[1] * self.frame_rate), seq_len)
            expected_energy = props[3]  # Energy from structure annotation
            
            if end_frame > start_frame:
                target_energy[:, start_frame:end_frame] = expected_energy
        
        # Normalize
        latent_energy = latent_energy / (latent_energy.max() + 1e-8)
        
        return F.mse_loss(latent_energy, target_energy)


class CompressionReconstructionLoss(nn.Module):
    """
    Loss for 30-second compression quality.
    
    Ensures the compressed representation preserves important musical content.
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        structure_preservation_weight: float = 0.3
    ):
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.structure_preservation_weight = structure_preservation_weight
        
        # Perceptual loss uses multiple scales
        self.spectral_loss = MultiScaleSpectralLoss() if perceptual_weight > 0 else None
    
    def forward(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        compressed: torch.Tensor,
        structure: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute compression quality loss.
        
        Args:
            original: Original latents
            reconstructed: Reconstructed latents from compression
            compressed: Compressed representation
            structure: Optional structure for preservation loss
            
        Returns:
            Total loss and components
        """
        losses = {}
        total_loss = 0.0
        
        # 1. Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, original)
        total_loss = total_loss + self.reconstruction_weight * recon_loss
        losses['compression_recon'] = recon_loss
        
        # 2. Perceptual loss (if we have spectrograms)
        if self.spectral_loss is not None and self.perceptual_weight > 0:
            # Convert latents to pseudo-audio for spectral comparison
            # In practice, you'd decode to audio first
            perc_loss, _ = self.spectral_loss(
                reconstructed.mean(dim=1),
                original.mean(dim=1)
            )
            total_loss = total_loss + self.perceptual_weight * perc_loss
            losses['compression_perceptual'] = perc_loss
        
        # 3. Structure preservation (compressed should maintain structure info)
        if structure and self.structure_preservation_weight > 0:
            struct_loss = self._structure_preservation_loss(original, compressed, structure)
            total_loss = total_loss + self.structure_preservation_weight * struct_loss
            losses['compression_structure'] = struct_loss
        
        losses['total_compression'] = total_loss
        
        return total_loss, losses
    
    def _structure_preservation_loss(
        self,
        original: torch.Tensor,
        compressed: torch.Tensor,
        structure: Dict
    ) -> torch.Tensor:
        """Ensure compressed representation preserves structural information."""
        
        # Compute structural features from original
        section_props = structure.get('section_props', [])
        if not section_props:
            return torch.tensor(0.0, device=original.device)
        
        # Energy variance should be preserved
        original_energy = torch.mean(original ** 2, dim=1)
        compressed_energy = torch.mean(compressed ** 2, dim=1)
        
        # Compare energy statistics
        orig_var = torch.var(original_energy, dim=-1)
        comp_var = torch.var(compressed_energy, dim=-1)
        
        return F.mse_loss(comp_var, orig_var)


class MTSCombinedLoss(nn.Module):
    """
    Combined loss for complete MTS training.
    
    Combines all loss components with configurable weights.
    """
    
    def __init__(
        self,
        diffusion_weight: float = 1.0,
        spectral_weight: float = 0.1,
        mel_weight: float = 0.1,
        structure_weight: float = 0.2,
        compression_weight: float = 0.1,
        sample_rate: int = 24000
    ):
        super().__init__()
        
        self.diffusion_weight = diffusion_weight
        self.spectral_weight = spectral_weight
        self.mel_weight = mel_weight
        self.structure_weight = structure_weight
        self.compression_weight = compression_weight
        
        # Initialize loss components
        self.spectral_loss = MultiScaleSpectralLoss(sample_rate=sample_rate)
        self.mel_loss = MelSpectrogramLoss(sample_rate=sample_rate)
        self.structure_loss = StructureConsistencyLoss()
        self.compression_loss = CompressionReconstructionLoss()
    
    def forward(
        self,
        diffusion_loss: torch.Tensor,
        pred_audio: Optional[torch.Tensor] = None,
        target_audio: Optional[torch.Tensor] = None,
        pred_latents: Optional[torch.Tensor] = None,
        target_latents: Optional[torch.Tensor] = None,
        compressed: Optional[torch.Tensor] = None,
        reconstructed: Optional[torch.Tensor] = None,
        structure: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            diffusion_loss: Base diffusion loss
            pred_audio: Predicted audio (for spectral losses)
            target_audio: Target audio
            pred_latents: Predicted latents (for structure loss)
            target_latents: Target latents
            compressed: Compressed representation
            reconstructed: Reconstructed from compression
            structure: Structure annotation
            
        Returns:
            Total loss and all components
        """
        losses = {'diffusion': diffusion_loss}
        total_loss = self.diffusion_weight * diffusion_loss
        
        # Spectral loss
        if pred_audio is not None and target_audio is not None and self.spectral_weight > 0:
            spec_loss, spec_losses = self.spectral_loss(pred_audio, target_audio)
            total_loss = total_loss + self.spectral_weight * spec_loss
            losses.update(spec_losses)
        
        # Mel loss
        if pred_audio is not None and target_audio is not None and self.mel_weight > 0:
            mel_loss = self.mel_loss(pred_audio, target_audio)
            total_loss = total_loss + self.mel_weight * mel_loss
            losses['mel'] = mel_loss
        
        # Structure loss
        if pred_latents is not None and structure is not None and self.structure_weight > 0:
            struct_loss, struct_losses = self.structure_loss(pred_latents, structure)
            total_loss = total_loss + self.structure_weight * struct_loss
            losses.update(struct_losses)
        
        # Compression loss
        if compressed is not None and reconstructed is not None and target_latents is not None:
            if self.compression_weight > 0:
                comp_loss, comp_losses = self.compression_loss(
                    target_latents, reconstructed, compressed, structure
                )
                total_loss = total_loss + self.compression_weight * comp_loss
                losses.update(comp_losses)
        
        losses['total'] = total_loss
        
        return total_loss, losses


def test_losses():
    """Test loss functions."""
    print("🧪 Testing Loss Functions...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test data
    batch_size = 2
    seq_len = 1000
    latent_dim = 128
    audio_len = 24000  # 1 second at 24kHz
    
    pred_latents = torch.randn(batch_size, latent_dim, seq_len, device=device)
    target_latents = torch.randn(batch_size, latent_dim, seq_len, device=device)
    pred_audio = torch.randn(batch_size, audio_len, device=device)
    target_audio = torch.randn(batch_size, audio_len, device=device)
    
    structure = {
        'section_types': [0, 1, 2, 1, 2],
        'section_props': [
            [0, 2, 2, 0.3, 0.5],
            [2, 5, 3, 0.5, 0.7],
            [5, 8, 3, 0.9, 1.0],
            [8, 11, 3, 0.5, 0.7],
            [11, 13, 2, 0.9, 1.0]
        ]
    }
    
    # Test individual losses
    print("\n1. Multi-scale Spectral Loss:")
    spectral_loss = MultiScaleSpectralLoss()
    loss, details = spectral_loss(pred_audio, target_audio)
    print(f"   Total: {loss.item():.4f}")
    
    print("\n2. Mel Spectrogram Loss:")
    mel_loss = MelSpectrogramLoss()
    loss = mel_loss(pred_audio, target_audio)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n3. Structure Consistency Loss:")
    struct_loss = StructureConsistencyLoss()
    loss, details = struct_loss(pred_latents, structure)
    print(f"   Total: {loss.item():.4f}")
    for k, v in details.items():
        if k != 'total_structure':
            print(f"   {k}: {v.item():.4f}")
    
    print("\n4. Compression Reconstruction Loss:")
    comp_loss = CompressionReconstructionLoss()
    compressed = torch.randn(batch_size, 64, seq_len // 16, device=device)
    loss, details = comp_loss(target_latents, pred_latents, compressed, structure)
    print(f"   Total: {loss.item():.4f}")
    
    print("\n5. Combined Loss:")
    combined_loss = MTSCombinedLoss()
    diffusion_loss = torch.tensor(0.5, device=device)
    total, all_losses = combined_loss(
        diffusion_loss=diffusion_loss,
        pred_audio=pred_audio,
        target_audio=target_audio,
        pred_latents=pred_latents,
        target_latents=target_latents,
        compressed=compressed,
        reconstructed=pred_latents,
        structure=structure
    )
    print(f"   Total: {total.item():.4f}")
    
    print("\n✅ All loss tests passed!")


if __name__ == "__main__":
    test_losses()
