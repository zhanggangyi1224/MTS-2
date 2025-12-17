"""
Complete MTS Model Demonstration
Shows all capabilities: 30s generation, 3-4 min generation, compression, structure
"""

import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from src.models.mts_model import MTSModel, MTSConfig

print("=" * 70)
print(" " * 20 + "MTS MODEL FULL DEMONSTRATION")
print("=" * 70)
print("\nDemonstrating:")
print("  ✓ Text-to-Music generation (30 seconds)")
print("  ✓ Structure-aware generation")
print("  ✓ Long-form generation (3-4 minutes)")
print("  ✓ 30-second compression/preview")
print("=" * 70)

# Setup
device = torch.device("cpu")
output_dir = Path("demo_output")
output_dir.mkdir(exist_ok=True)

# Load model
print("\n📥 Loading trained MTS model...")
config = MTSConfig()
model = MTSModel(config).to(device)

checkpoint_dir = Path("checkpoints")
checkpoints = list(checkpoint_dir.glob("mts_*.pt"))

if not checkpoints:
    print("❌ No checkpoints found!")
    print("   Please train the model first: python3 train_mts_local.py")
    exit(1)

latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
checkpoint = torch.load(latest_checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded from: {latest_checkpoint.name}")
print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"   Val loss: {checkpoint.get('val_loss', 0):.4f}")


# ============================================================================
# DEMO 1: Basic 30-Second Generation
# ============================================================================
print("\n" + "=" * 70)
print("DEMO 1: Basic 30-Second Generation")
print("=" * 70)

prompt = "A calm piano melody with gentle strings"
print(f"\n📝 Prompt: \"{prompt}\"")
print("⏱️  Duration: 30 seconds")
print("🎵 Generating...")

with torch.no_grad():
    audio, compressed = model.generate(
        text=prompt,
        duration=30.0,
        cfg_scale=7.5,
        device=device
    )

audio_np = audio[0, 0].cpu().numpy()
audio_np = audio_np / (np.abs(audio_np).max() + 1e-7) * 0.9

output_file = output_dir / "demo1_basic_30s.wav"
sf.write(output_file, audio_np, 24000)

print(f"✅ Generated: {output_file}")
print(f"   Audio shape: {audio.shape}")
print(f"   Compressed representation: {compressed.shape}")


# ============================================================================
# DEMO 2: Structure-Aware 30-Second Generation
# ============================================================================
print("\n" + "=" * 70)
print("DEMO 2: Structure-Aware Generation (30s)")
print("=" * 70)

# Define simple structure: intro -> main -> outro
structure_30s = {
    "section_types": [0, 1, 3],  # intro, main, outro
    "section_props": [
        [0, 8, 8, 0.3, 0.5],     # intro: 0-8s, low energy
        [8, 24, 16, 0.8, 1.0],   # main: 8-24s, high energy
        [24, 30, 6, 0.4, 0.5]    # outro: 24-30s, low energy
    ]
}

prompt = "An energetic rock song with powerful guitars"
print(f"\n📝 Prompt: \"{prompt}\"")
print("🎼 Structure:")
print("   - Intro (0-8s): Low energy")
print("   - Main (8-24s): High energy")
print("   - Outro (24-30s): Low energy")
print("🎵 Generating...")

with torch.no_grad():
    audio, compressed = model.generate(
        text=prompt,
        structure=structure_30s,
        duration=30.0,
        cfg_scale=7.5,
        device=device
    )

audio_np = audio[0, 0].cpu().numpy()
audio_np = audio_np / (np.abs(audio_np).max() + 1e-7) * 0.9

output_file = output_dir / "demo2_structured_30s.wav"
sf.write(output_file, audio_np, 24000)

print(f"✅ Generated: {output_file}")


# ============================================================================
# DEMO 3: 30-Second Compression Feature
# ============================================================================
print("\n" + "=" * 70)
print("DEMO 3: 30-Second Compression/Preview")
print("=" * 70)

print("\n🔬 Testing compression on real FMA audio...")

# Load a real FMA track
fma_files = list(Path("fma_data/fma_small").rglob("*.mp3"))
if fma_files:
    test_file = fma_files[0]
    print(f"📂 Loading: {test_file}")

    import librosa
    audio_real, sr = librosa.load(test_file, sr=24000, duration=30.0)

    # Pad if needed
    target_length = 30 * 24000
    if len(audio_real) < target_length:
        audio_real = np.pad(audio_real, (0, target_length - len(audio_real)))
    else:
        audio_real = audio_real[:target_length]

    # Get compressed preview
    audio_tensor = torch.FloatTensor(audio_real).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        preview = model.get_30_second_preview(audio_tensor, is_audio=True)

    # Save original and preview
    sf.write(output_dir / "demo3_original.wav", audio_real, 24000)

    preview_np = preview[0, 0].cpu().numpy()
    preview_np = preview_np / (np.abs(preview_np).max() + 1e-7) * 0.9
    sf.write(output_dir / "demo3_compressed_preview.wav", preview_np, 24000)

    print(f"✅ Original saved: {output_dir}/demo3_original.wav")
    print(f"✅ Preview saved: {output_dir}/demo3_compressed_preview.wav")
    print(f"   Original shape: {audio_tensor.shape}")
    print(f"   Preview shape: {preview.shape}")
    print("\n   Compare the two files to hear compression quality!")
else:
    print("⚠️  No FMA files found, skipping compression demo")


# ============================================================================
# DEMO 4: Long-Form Generation (3.5 minutes with structure)
# ============================================================================
print("\n" + "=" * 70)
print("DEMO 4: Long-Form Generation (3.5 minutes)")
print("=" * 70)

# Define full song structure
structure_long = {
    "section_types": [0, 1, 2, 1, 2, 3],
    "section_props": [
        [0, 20, 20, 0.3, 0.5],      # intro
        [20, 60, 40, 0.5, 0.7],     # verse 1
        [60, 100, 40, 0.9, 1.0],    # chorus 1
        [100, 140, 40, 0.5, 0.7],   # verse 2
        [140, 180, 40, 0.9, 1.0],   # chorus 2
        [180, 210, 30, 0.4, 0.5]    # outro
    ]
}

prompt = "An uplifting pop song with catchy melody and driving beat"
print(f"\n📝 Prompt: \"{prompt}\"")
print("⏱️  Duration: 210 seconds (3.5 minutes)")
print("🎼 Structure: Intro → Verse → Chorus → Verse → Chorus → Outro")
print("\n⚠️  Note: Long-form generation uses sliding window approach")
print("   This will take several minutes...")

# Import the sliding window function
import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate_long_form import generate_long_form

print("\n🎵 Generating long-form track...")
full_audio = generate_long_form(
    model=model,
    text=prompt,
    structure=structure_long,
    duration=210.0,
    segment_duration=30.0,
    overlap=5.0,
    cfg_scale=7.5,
    device=device
)

# Normalize and save
full_audio = full_audio / (np.abs(full_audio).max() + 1e-7) * 0.9
output_file = output_dir / "demo4_longform_210s.wav"
sf.write(output_file, full_audio, 24000)

print(f"\n✅ Long-form track generated: {output_file}")
print(f"   Duration: {len(full_audio)/24000:.1f}s")

# Generate 30s preview of the long track
print("\n   📦 Generating 30s compressed preview...")
audio_tensor = torch.FloatTensor(full_audio).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    preview = model.get_30_second_preview(audio_tensor, is_audio=True)

preview_np = preview[0, 0].cpu().numpy()
preview_np = preview_np / (np.abs(preview_np).max() + 1e-7) * 0.9

preview_file = output_dir / "demo4_longform_preview_30s.wav"
sf.write(preview_file, preview_np, 24000)

print(f"   ✅ 30s preview saved: {preview_file}")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE! 🎉")
print("=" * 70)

print("\n📁 Generated Files:")
print(f"   {output_dir}/")
for file in sorted(output_dir.glob("*.wav")):
    size_mb = file.stat().st_size / 1024 / 1024
    print(f"   ├─ {file.name} ({size_mb:.1f} MB)")

print("\n🎵 Play Examples:")
print(f"   # Basic 30s generation")
print(f"   afplay {output_dir}/demo1_basic_30s.wav")
print(f"\n   # Structure-aware 30s")
print(f"   afplay {output_dir}/demo2_structured_30s.wav")
print(f"\n   # Compression comparison")
print(f"   afplay {output_dir}/demo3_original.wav")
print(f"   afplay {output_dir}/demo3_compressed_preview.wav")
print(f"\n   # Long-form (3.5 min)")
print(f"   afplay {output_dir}/demo4_longform_210s.wav")
print(f"\n   # Long-form preview (30s summary)")
print(f"   afplay {output_dir}/demo4_longform_preview_30s.wav")

print("\n" + "=" * 70)
print("✅ All MTS features demonstrated successfully!")
print("=" * 70)
