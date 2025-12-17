"""
MTS Model Training Script - Mac Optimized (FIXED)
Trains the diffusion model on prepared FMA dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import pandas as pd

# Import MTS model
from src.models.mts_model import MTSModel, MTSConfig

print("=" * 60)
print("MTS Model Training - Mac Optimized")
print("=" * 60)

# Force CPU due to EnCodec MPS compatibility issues
# TODO: Re-enable MPS once EnCodec fully supports it
device = torch.device("cpu")
print("⚠️  Using CPU (EnCodec has MPS compatibility issues)")

print(f"Device: {device}")
print("=" * 60)

# Training configuration
BATCH_SIZE = 2
EPOCHS = 50  # Train for 50 epochs
LEARNING_RATE = 1e-4
SAVE_EVERY = 10  # Save every 10 epochs
CHECKPOINT_DIR = "checkpoints"

Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

class MTSDataset(Dataset):
    """Dataset for MTS training - FIXED to use CSV"""
    
    def __init__(self, csv_path, split='train'):
        # Load the main CSV file
        df = pd.read_csv(csv_path)

        # Create train/val/test splits (80/10/10)
        from sklearn.model_selection import train_test_split

        # First split: train vs (val+test)
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        # Second split: val vs test
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        # Select the right split
        if split == 'train':
            selected_df = train_df
        elif split == 'val':
            selected_df = val_df
        else:
            selected_df = test_df

        self.samples = selected_df.to_dict('records')
        self.sample_rate = 24000

        print(f"✅ Loaded {split} set: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get audio path
        audio_path = sample.get('audio_file_path', '')
        
        # Try to load audio
        try:
            if audio_path and Path(audio_path).exists():
                audio, sr = sf.read(audio_path)
            else:
                # Generate dummy audio for missing files
                audio = np.random.randn(30 * self.sample_rate) * 0.01
            
            # Convert to mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Pad or trim to 30 seconds
            target_length = 30 * self.sample_rate
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
            
            # Get text prompt
            text_prompt = sample.get('text_prompt', '')
            
            return {
                'audio': audio_tensor,
                'text': text_prompt,
                'id': sample.get('id', f'sample_{idx}')
            }
        
        except Exception as e:
            # Return dummy data on error
            return {
                'audio': torch.randn(1, 30 * self.sample_rate) * 0.01,
                'text': '',
                'id': f'error_{idx}'
            }

def train_epoch(model, dataloader, optimizer, epoch, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    count = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device)
        text = batch['text']
        
        try:
            loss = model.compute_loss(audio, text)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        except Exception as e:
            print(f"\n⚠️  Error in batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / count if count > 0 else 0
    return avg_loss

def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            audio = batch['audio'].to(device)
            text = batch['text']
            
            try:
                loss = model.compute_loss(audio, text)
                total_loss += loss.item()
                count += 1
            except:
                continue
    
    avg_loss = total_loss / count if count > 0 else 0
    return avg_loss

def main():
    print("\n📊 Loading dataset...")
    
    # Use the final CSV file
    csv_path = "outputs/mts_final_dataset.csv"
    
    # Create datasets
    train_dataset = MTSDataset(csv_path, split='train')
    val_dataset = MTSDataset(csv_path, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\n📊 Dataset ready:")
    print(f"   Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"   Val: {len(val_dataset)} samples ({len(val_loader)} batches)")
    
    # Create model
    print("\n🔨 Creating MTS model...")
    config = MTSConfig(
        sample_rate=24000,
        target_duration=30.0,
        model_channels=64,
        diffusion_timesteps=100,
        use_structure=False,
        dropout=0.1
    )
    
    model = MTSModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {total_params/1e6:.1f}M parameters")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n📍 Epoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, epoch, device)
        print(f"   Train loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"   Val loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            checkpoint_path = f"{CHECKPOINT_DIR}/mts_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"   💾 Saved: {checkpoint_path}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = f"{CHECKPOINT_DIR}/mts_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_path)
            print(f"   🌟 New best! Val loss: {val_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints: {CHECKPOINT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
