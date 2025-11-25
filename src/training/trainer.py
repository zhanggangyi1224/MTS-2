"""
MTS Training Script - Train the Music-to-Structure model

This script handles:
1. Data loading from prepared pipeline outputs
2. Model training with mixed precision
3. Checkpointing and logging
4. Evaluation metrics
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mts_model import MTSModel, MTSConfig, create_mts_model


class MTSDataset(Dataset):
    """
    Dataset for MTS training.
    
    Loads preprocessed data from the data pipeline outputs.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_duration: float = 210.0,  # 3.5 minutes
        sample_rate: int = 24000,
        load_audio: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.load_audio = load_audio
        
        # Load metadata
        self.samples = self._load_metadata()
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_metadata(self) -> List[Dict]:
        """Load dataset metadata."""
        
        # Try to load from pipeline outputs
        metadata_file = self.data_dir / f"mts_{self.split}_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file) as f:
                return json.load(f)["samples"]
        
        # Fallback: load from CSV
        csv_file = self.data_dir / "mts_final_dataset.csv"
        if csv_file.exists():
            import pandas as pd
            df = pd.read_csv(csv_file)
            
            # Filter by split (if column exists)
            if "split" in df.columns:
                df = df[df["split"] == self.split]
            
            return df.to_dict("records")
        
        # Create dummy data for testing
        print(f"⚠️  No metadata found, creating dummy data")
        return self._create_dummy_samples(100)
    
    def _create_dummy_samples(self, n: int) -> List[Dict]:
        """Create dummy samples for testing."""
        samples = []
        for i in range(n):
            samples.append({
                "id": f"dummy_{i:04d}",
                "text_prompt": f"A beautiful piece of music in the style of {['classical', 'pop', 'jazz', 'rock'][i % 4]}",
                "duration": np.random.uniform(180, 240),
                "structure": {
                    "section_types": [0, 1, 2, 1, 2, 7],  # intro, verse, chorus, verse, chorus, outro
                    "section_props": [
                        [0, 20, 20, 0.3, 0.5],
                        [20, 60, 40, 0.5, 0.7],
                        [60, 100, 40, 0.9, 1.0],
                        [100, 140, 40, 0.5, 0.7],
                        [140, 180, 40, 0.9, 1.0],
                        [180, 210, 30, 0.4, 0.5]
                    ]
                }
            })
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load or generate audio
        if self.load_audio and "audio_path" in sample:
            audio = self._load_audio(sample["audio_path"])
        else:
            # Generate dummy audio for training
            duration = min(sample.get("duration", 210), self.max_duration)
            audio = torch.randn(1, int(duration * self.sample_rate)) * 0.1
        
        # Get text prompt
        text = sample.get("text_prompt", "A piece of music")
        
        # Get structure (if available)
        structure = sample.get("structure", None)
        
        return {
            "id": sample.get("id", f"sample_{idx}"),
            "audio": audio,
            "text": text,
            "structure": structure
        }
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file."""
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.max_duration)
            return torch.from_numpy(audio).unsqueeze(0).float()
        except Exception as e:
            print(f"⚠️  Failed to load {audio_path}: {e}")
            return torch.randn(1, int(self.max_duration * self.sample_rate)) * 0.1


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
    
    # Pad audio to same length
    max_len = max(b["audio"].shape[-1] for b in batch)
    
    audios = []
    for b in batch:
        audio = b["audio"]
        if audio.shape[-1] < max_len:
            padding = max_len - audio.shape[-1]
            audio = F.pad(audio, (0, padding))
        audios.append(audio)
    
    return {
        "ids": [b["id"] for b in batch],
        "audio": torch.stack(audios),
        "text": [b["text"] for b in batch],
        "structure": [b["structure"] for b in batch]
    }


class MTSTrainer:
    """
    Trainer for MTS model.
    """
    
    def __init__(
        self,
        model: MTSModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        save_every: int = 5000,
        eval_every: int = 1000,
        log_every: int = 100,
        checkpoint_dir: str = "./checkpoints",
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_clip = gradient_clip
        
        self.max_steps = max_steps
        self.save_every = save_every
        self.eval_every = eval_every
        self.log_every = log_every
        self.warmup_steps = warmup_steps
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps,
            eta_min=learning_rate * 0.1
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.log_history = []
    
    def train(self):
        """Run training loop."""
        
        print(f"Starting training for {self.max_steps} steps...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        
        self.model.train()
        data_iter = iter(self.train_dataloader)
        
        pbar = tqdm(range(self.max_steps), desc="Training")
        
        for step in pbar:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            # Training step
            loss_dict = self._training_step(batch)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['loss']:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if step % self.log_every == 0:
                self._log_metrics(step, loss_dict)
            
            # Evaluation
            if self.val_dataloader and step % self.eval_every == 0 and step > 0:
                val_loss = self._evaluate()
                self._log_metrics(step, {"val_loss": val_loss}, prefix="val")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(step, is_best=True)
                
                self.model.train()
            
            # Save checkpoint
            if step % self.save_every == 0 and step > 0:
                self._save_checkpoint(step)
            
            self.global_step = step
        
        # Final save
        self._save_checkpoint(self.max_steps, is_final=True)
        print("Training complete!")
    
    def _training_step(self, batch: Dict) -> Dict:
        """Single training step."""
        
        # Move to device
        audio = batch["audio"].to(self.device)
        text = batch["text"]
        structure = batch["structure"]
        
        # Forward pass with mixed precision
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with autocast():
                loss_dict = self.model.training_step(audio, text, structure)
            
            # Backward pass
            self.scaler.scale(loss_dict["loss"]).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_dict = self.model.training_step(audio, text, structure)
            loss_dict["loss"].backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
        
        # Learning rate scheduling
        if self.global_step < self.warmup_steps:
            # Linear warmup
            lr_scale = (self.global_step + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * lr_scale if 'initial_lr' in pg else pg['lr'] * lr_scale
        else:
            self.scheduler.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
    @torch.no_grad()
    def _evaluate(self) -> float:
        """Run evaluation."""
        
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            audio = batch["audio"].to(self.device)
            text = batch["text"]
            structure = batch["structure"]
            
            if self.use_amp:
                with autocast():
                    loss_dict = self.model.training_step(audio, text, structure)
            else:
                loss_dict = self.model.training_step(audio, text, structure)
            
            total_loss += loss_dict["loss"].item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else float('inf')
    
    def _log_metrics(self, step: int, metrics: Dict, prefix: str = "train"):
        """Log metrics."""
        
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **{f"{prefix}/{k}": v for k, v in metrics.items()}
        }
        
        self.log_history.append(log_entry)
        
        # Save log
        log_file = self.checkpoint_dir / "training_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.log_history, f, indent=2)
    
    def _save_checkpoint(
        self,
        step: int,
        is_best: bool = False,
        is_final: bool = False
    ):
        """Save model checkpoint."""
        
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.model.config.__dict__,
            "best_val_loss": self.best_val_loss
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        elif is_final:
            path = self.checkpoint_dir / "final_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_{step:08d}.pt"
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        print(f"Loaded checkpoint from step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description="Train MTS Model")
    parser.add_argument("--data-dir", default="./outputs", help="Data directory")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--model-size", default="base", choices=["small", "base", "large"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--use-structure", action="store_true", default=True)
    parser.add_argument("--no-structure", action="store_false", dest="use_structure")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Create model
    print(f"Creating {args.model_size} model...")
    model = create_mts_model(args.model_size, use_structure=args.use_structure)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = MTSDataset(args.data_dir, split="train")
    val_dataset = MTSDataset(args.data_dir, split="val")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create trainer
    trainer = MTSTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
