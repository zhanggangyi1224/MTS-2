"""
MTS Model Training Script - HPC Optimized
GPU-accelerated training for high-performance computing environments
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
import argparse
from datetime import datetime

# Import MTS model
from src.models.mts_model import MTSModel, MTSConfig

def parse_args():
    parser = argparse.ArgumentParser(description="MTS Model Training - HPC")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-every", type=int, default=20, help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--dataset", type=str, default="outputs/mts_final_dataset.csv", help="Dataset CSV path")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()

print("=" * 70)
print(" " * 20 + "MTS Model Training - HPC")
print("=" * 70)

args = parse_args()

# Check device
if args.device == "cuda" and not torch.cuda.is_available():
    print("⚠️  CUDA not available, falling back to CPU")
    device = torch.device("cpu")
else:
    device = torch.device(args.device)

print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("=" * 70)

# Dataset class with path verification and better error handling
class MTSDataset(Dataset):
    def __init__(self, csv_path, split='train', sample_rate=24000, verify_paths=True):
        self.sample_rate = sample_rate
        self.split = split
        self.csv_path = csv_path

        # Load CSV
        print(f"📖 Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)

        # Verify audio_path column exists
        if 'audio_path' not in df.columns:
            raise ValueError(f"❌ ERROR: CSV missing 'audio_path' column!\n"
                           f"   Available columns: {list(df.columns)}\n"
                           f"   Please regenerate CSV with: python regenerate_dataset_csv.py")

        # Create splits
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        if split == 'train':
            selected_df = train_df
        elif split == 'val':
            selected_df = val_df
        else:
            selected_df = test_df

        self.samples = selected_df.to_dict('records')
        print(f"✅ Loaded {split} set: {len(self.samples)} samples")

        # Verify audio file paths if requested
        if verify_paths:
            self._verify_audio_paths()

    def _verify_audio_paths(self):
        """Verify that audio files exist at the specified paths"""
        print(f"\n🔍 Verifying audio file paths for {self.split} set...")

        missing_files = []
        empty_paths = []
        valid_files = 0

        # Check a sample of files (first 10 to avoid long delays)
        sample_size = min(10, len(self.samples))

        for i, sample in enumerate(self.samples[:sample_size]):
            audio_path = sample.get('audio_path', '')

            # Handle NaN or non-string values
            if pd.isna(audio_path) or not isinstance(audio_path, str) or not audio_path.strip():
                empty_paths.append(sample.get('id', f'sample_{i}'))
            elif not Path(audio_path).exists():
                missing_files.append((sample.get('id', f'sample_{i}'), audio_path))
            else:
                valid_files += 1

        # Report results
        if empty_paths:
            print(f"⚠️  WARNING: {len(empty_paths)} samples with empty audio_path in sample check")
            print(f"   Examples: {empty_paths[:3]}")
            print(f"   This will cause samples to use fallback/zero audio")

        if missing_files:
            print(f"⚠️  WARNING: {len(missing_files)} files not found in sample check")
            for sample_id, path in missing_files[:3]:
                print(f"   - {sample_id}: {path}")
            print(f"\n💡 Possible fixes:")
            print(f"   1. Check if FMA data is downloaded to correct location")
            print(f"   2. Verify augmented audio was generated")
            print(f"   3. Regenerate CSV: python regenerate_dataset_csv.py")

        if valid_files > 0:
            print(f"✅ {valid_files}/{sample_size} sample files verified successfully")
        else:
            print(f"❌ ERROR: No valid audio files found!")
            print(f"   Cannot proceed with training")
            raise FileNotFoundError(f"No audio files found at paths specified in CSV")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample.get('id', f'sample_{idx}')

        try:
            # Load audio
            audio_path = sample.get('audio_path', '')

            # Check if path is empty or NaN
            if pd.isna(audio_path) or not isinstance(audio_path, str) or not audio_path.strip():
                print(f"⚠️  Error loading sample {idx} ({sample_id}): Empty audio_path")
                # Generate zero audio as fallback
                audio = np.zeros(30 * self.sample_rate)
                sr = self.sample_rate
            elif Path(audio_path).exists():
                # Load the audio file
                audio, sr = sf.read(audio_path)
            else:
                # File doesn't exist - log detailed error
                print(f"⚠️  Error loading sample {idx} ({sample_id}): File not found '{audio_path}'")
                # Generate zero audio as fallback
                audio = np.zeros(30 * self.sample_rate)
                sr = self.sample_rate

            # Convert to mono if stereo
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
                'id': sample_id
            }

        except Exception as e:
            # Return dummy data on error
            print(f"⚠️  Error loading sample {idx} ({sample_id}): {type(e).__name__}: {e}")
            return {
                'audio': torch.randn(1, 30 * self.sample_rate) * 0.01,
                'text': '',
                'id': f'error_{sample_id}'
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

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            count += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            audio = batch['audio'].to(device)
            text = batch['text']

            try:
                loss = model.compute_loss(audio, text)
                total_loss += loss.item()
                count += 1
            except Exception as e:
                continue

    avg_loss = total_loss / count if count > 0 else 0
    return avg_loss


# Main training
print("\n📊 Loading dataset...")
train_dataset = MTSDataset(args.dataset, split='train')
val_dataset = MTSDataset(args.dataset, split='val')

print(f"\n📊 Dataset ready:")
print(f"   Train: {len(train_dataset)} samples ({len(train_dataset)//args.batch_size} batches)")
print(f"   Val: {len(val_dataset)} samples ({len(val_dataset)//args.batch_size} batches)")

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=(device.type == "cuda")
)

# Model
print(f"\n🔨 Creating MTS model...")
config = MTSConfig()
model = MTSModel(config).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model created: {trainable_params/1e6:.1f}M parameters")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs,
    eta_min=1e-6
)

# Resume from checkpoint if specified
start_epoch = 0
best_val_loss = float('inf')

if args.resume:
    print(f"\n📥 Resuming from checkpoint: {args.resume}")
    try:
        checkpoint = torch.load(args.resume, map_location=device)

        # Try to load state dict with strict=False to handle model architecture changes
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if missing_keys or unexpected_keys:
            print(f"⚠️  Checkpoint has different model architecture:")
            if missing_keys:
                print(f"   Missing keys: {len(missing_keys)} (new model has these, checkpoint doesn't)")
            if unexpected_keys:
                print(f"   Unexpected keys: {len(unexpected_keys)} (checkpoint has these, new model doesn't)")
            print(f"   Continuing with partial weights loaded...")

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"✅ Resumed from epoch {start_epoch}")

    except Exception as e:
        print(f"⚠️  Warning: Could not load checkpoint: {e}")
        print(f"   Starting training from scratch instead...")
        start_epoch = 0
        best_val_loss = float('inf')

# Create checkpoint directory
Path(args.checkpoint_dir).mkdir(exist_ok=True)

# Training loop
print(f"\n🚀 Starting training for {args.epochs} epochs...")
print("=" * 70)

training_start = datetime.now()

for epoch in range(start_epoch, args.epochs):
    print(f"\n📍 Epoch {epoch+1}/{args.epochs}")

    # Train
    train_loss = train_epoch(model, train_loader, optimizer, epoch, device)

    # Validate
    val_loss = validate(model, val_loader, device)

    # Update learning rate
    scheduler.step()

    # Print stats
    print(f"   Train loss: {train_loss:.4f}")
    print(f"   Val loss: {val_loss:.4f}")
    print(f"   LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save checkpoint
    if (epoch + 1) % args.save_every == 0 or val_loss < best_val_loss:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config.__dict__
        }

        # Regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"mts_epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"   💾 Saved: {checkpoint_path}")

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = Path(args.checkpoint_dir) / "mts_best.pt"
            torch.save(checkpoint, best_path)
            print(f"   🌟 New best! Val loss: {val_loss:.4f}")

training_end = datetime.now()
training_duration = (training_end - training_start).total_seconds()

print("\n" + "=" * 70)
print("✅ Training complete!")
print(f"   Total time: {training_duration/3600:.2f} hours")
print(f"   Best val loss: {best_val_loss:.4f}")
print(f"   Checkpoints: {args.checkpoint_dir}/")
print("=" * 70)
