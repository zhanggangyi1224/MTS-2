"""
Test 30-Second Music Generation
Tests the trained MTS model's ability to generate music from text prompts
"""

import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from src.models.mts_model import MTSModel, MTSConfig

print("=" * 60)
print("MTS 30-Second Generation Test")
print("=" * 60)

# Device
device = torch.device("cpu")
print(f"Device: {device}")

# Create output directory
output_dir = Path("generated_samples")
output_dir.mkdir(exist_ok=True)

# Load trained model
print("\n📥 Loading trained model...")
config = MTSConfig()
model = MTSModel(config).to(device)

# Find best checkpoint
checkpoint_dir = Path("checkpoints")
checkpoints = list(checkpoint_dir.glob("mts_*.pt"))

if not checkpoints:
    print("❌ No checkpoints found! Please train the model first.")
    print("   Run: python3 train_mts_local.py")
    exit(1)

# Load the most recent checkpoint
latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
print(f"✅ Loading checkpoint: {latest_checkpoint}")

try:
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"   Train loss: {checkpoint.get('train_loss', '?'):.4f}")
    print(f"   Val loss: {checkpoint.get('val_loss', '?'):.4f}")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    exit(1)

model.eval()

# Test prompts
test_prompts = [
    "A peaceful piano melody with gentle strings and soft percussion",
    "An upbeat electronic dance track with strong bass and energetic synths",
    "A melancholic acoustic guitar with ambient background sounds",
    "A powerful rock song with electric guitars and driving drums",
    "A calm jazz piece with saxophone and light piano accompaniment"
]

print(f"\n🎵 Generating {len(test_prompts)} samples...")
print("=" * 60)

for i, prompt in enumerate(test_prompts):
    print(f"\n📝 Prompt {i+1}: {prompt}")

    try:
        # Generate audio
        with torch.no_grad():
            audio, compressed = model.generate(
                text=prompt,
                duration=30.0,
                cfg_scale=7.5,
                num_inference_steps=50,
                device=device
            )

        # Convert to numpy
        audio_np = audio[0, 0].cpu().numpy()

        # Normalize to prevent clipping
        audio_np = audio_np / (np.abs(audio_np).max() + 1e-7)
        audio_np = audio_np * 0.9  # Leave headroom

        # Save
        output_file = output_dir / f"generated_{i+1:02d}.wav"
        sf.write(output_file, audio_np, 24000)

        print(f"✅ Generated: {output_file}")
        print(f"   Duration: {len(audio_np)/24000:.1f}s")
        print(f"   Shape: {audio.shape}")
        print(f"   Compressed shape: {compressed.shape}")

    except Exception as e:
        print(f"❌ Error generating sample {i+1}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Generation complete!")
print(f"   Samples saved to: {output_dir}/")
print("=" * 60)

# Test 30-second compression on generated audio
print("\n🔬 Testing 30-second compression feature...")
try:
    # Load one of the generated files
    test_file = output_dir / "generated_01.wav"
    if test_file.exists():
        audio_data, sr = sf.read(test_file)
        audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            preview = model.get_30_second_preview(audio_tensor, is_audio=True)

        preview_np = preview[0, 0].cpu().numpy()
        preview_np = preview_np / (np.abs(preview_np).max() + 1e-7) * 0.9

        preview_file = output_dir / "compression_test_preview.wav"
        sf.write(preview_file, preview_np, 24000)

        print(f"✅ Compression test complete!")
        print(f"   Original: {audio_tensor.shape}")
        print(f"   Preview: {preview.shape}")
        print(f"   Saved to: {preview_file}")
    else:
        print("⚠️  No generated file to test compression")
except Exception as e:
    print(f"❌ Error testing compression: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🎵 Play generated samples:")
print(f"   afplay {output_dir}/generated_01.wav")
print("=" * 60)
