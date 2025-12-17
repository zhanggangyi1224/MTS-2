"""
Long-Form Music Generation (3-4 Minutes)
Implements sliding-window generation for full-length tracks with structure
"""

import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List
from src.models.mts_model import MTSModel, MTSConfig

print("=" * 60)
print("MTS Long-Form Generation (3-4 Minutes)")
print("=" * 60)

# Device
device = torch.device("cpu")
print(f"Device: {device}")

# Create output directory
output_dir = Path("generated_samples")
output_dir.mkdir(exist_ok=True)


def crossfade(audio1: np.ndarray, audio2: np.ndarray, fade_samples: int) -> np.ndarray:
    """
    Crossfade two audio segments.

    Args:
        audio1: First audio segment
        audio2: Second audio segment
        fade_samples: Number of samples to crossfade

    Returns:
        Blended audio
    """
    if len(audio1) < fade_samples or len(audio2) < fade_samples:
        # Not enough samples, simple concatenation
        return np.concatenate([audio1, audio2])

    # Extract fade regions
    fade_out = audio1[-fade_samples:]
    fade_in = audio2[:fade_samples]

    # Create fade curves
    t = np.linspace(0, 1, fade_samples)
    fade_out_curve = np.cos(t * np.pi / 2)  # Cosine fade out
    fade_in_curve = np.sin(t * np.pi / 2)   # Sine fade in

    # Apply fades
    faded_out = fade_out * fade_out_curve
    faded_in = fade_in * fade_in_curve
    crossfaded = faded_out + faded_in

    # Combine: audio1 (without fade region) + crossfade + audio2 (without fade region)
    result = np.concatenate([
        audio1[:-fade_samples],
        crossfaded,
        audio2[fade_samples:]
    ])

    return result


def extract_segment_structure(
    full_structure: Dict,
    segment_idx: int,
    num_segments: int
) -> Dict:
    """
    Extract structure information for a specific segment.

    Args:
        full_structure: Full song structure
        segment_idx: Index of current segment
        num_segments: Total number of segments

    Returns:
        Structure dict for this segment
    """
    # For simplicity, distribute sections evenly across segments
    # In practice, you'd want more sophisticated section alignment

    section_types = full_structure["section_types"]
    section_props = full_structure["section_props"]

    # Calculate which section this segment belongs to
    sections_per_segment = len(section_types) / num_segments
    start_section = int(segment_idx * sections_per_segment)
    end_section = int((segment_idx + 1) * sections_per_segment)

    # Extract relevant sections
    segment_types = section_types[start_section:end_section] or [section_types[start_section]]
    segment_props = section_props[start_section:end_section] or [section_props[start_section]]

    # Adjust timing to 0-30s range
    adjusted_props = []
    for props in segment_props:
        # Normalize to 30-second segment
        adjusted_props.append([
            props[0] % 30,  # start
            props[1] % 30,  # end
            props[2],       # duration
            props[3],       # energy
            props[4]        # importance
        ])

    return {
        "section_types": segment_types,
        "section_props": adjusted_props
    }


def generate_long_form(
    model: MTSModel,
    text: str,
    structure: Dict,
    duration: float = 210.0,
    segment_duration: float = 30.0,
    overlap: float = 5.0,
    cfg_scale: float = 7.5,
    device: str = "cpu"
) -> np.ndarray:
    """
    Generate long-form audio using sliding window approach.

    Args:
        model: Trained MTS model
        text: Text prompt
        structure: Musical structure specification
        duration: Total duration in seconds
        segment_duration: Duration of each segment
        overlap: Overlap between segments in seconds
        cfg_scale: Classifier-free guidance scale
        device: Device to run on

    Returns:
        Full audio as numpy array
    """
    print(f"\n🎵 Generating {duration}s audio...")
    print(f"   Segment duration: {segment_duration}s")
    print(f"   Overlap: {overlap}s")

    # Calculate number of segments
    effective_segment = segment_duration - overlap
    num_segments = int(np.ceil((duration - overlap) / effective_segment))

    print(f"   Total segments: {num_segments}")

    segments = []
    sample_rate = 24000
    fade_samples = int(overlap * sample_rate)

    for i in range(num_segments):
        print(f"\n   📍 Generating segment {i+1}/{num_segments}...")

        # Get structure for this segment
        segment_structure = extract_segment_structure(structure, i, num_segments)

        try:
            # Generate segment
            with torch.no_grad():
                audio_seg, _ = model.generate(
                    text=text,
                    structure=segment_structure,
                    duration=segment_duration,
                    cfg_scale=cfg_scale,
                    num_inference_steps=50,
                    device=device
                )

            # Convert to numpy
            audio_np = audio_seg[0, 0].cpu().numpy()
            segments.append(audio_np)

            print(f"      ✅ Segment {i+1} complete ({len(audio_np)/sample_rate:.1f}s)")

        except Exception as e:
            print(f"      ❌ Error generating segment {i+1}: {e}")
            # Generate silence as fallback
            silence = np.zeros(int(segment_duration * sample_rate))
            segments.append(silence)

    # Blend all segments
    print(f"\n   🔗 Blending {len(segments)} segments...")
    full_audio = segments[0]

    for i in range(1, len(segments)):
        full_audio = crossfade(full_audio, segments[i], fade_samples)
        print(f"      ✅ Blended segment {i+1}")

    # Trim to exact duration
    target_samples = int(duration * sample_rate)
    if len(full_audio) > target_samples:
        full_audio = full_audio[:target_samples]
    elif len(full_audio) < target_samples:
        # Pad with silence if needed
        padding = np.zeros(target_samples - len(full_audio))
        full_audio = np.concatenate([full_audio, padding])

    print(f"\n   ✅ Blending complete! Total duration: {len(full_audio)/sample_rate:.1f}s")

    return full_audio


# Load model
print("\n📥 Loading trained model...")
config = MTSConfig()
model = MTSModel(config).to(device)

checkpoint_dir = Path("checkpoints")
checkpoints = list(checkpoint_dir.glob("mts_*.pt"))

if not checkpoints:
    print("❌ No checkpoints found! Please train the model first.")
    exit(1)

latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
print(f"✅ Loading checkpoint: {latest_checkpoint}")

checkpoint = torch.load(latest_checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define musical structure (3.5 minutes)
structure = {
    "section_types": [0, 1, 2, 1, 2, 3],  # intro, verse, chorus, verse, chorus, outro
    "section_props": [
        [0, 20, 20, 0.3, 0.5],      # intro: 0-20s, low energy
        [20, 60, 40, 0.5, 0.7],     # verse: 20-60s, medium energy
        [60, 100, 40, 0.9, 1.0],    # chorus: 60-100s, high energy
        [100, 140, 40, 0.5, 0.7],   # verse: 100-140s, medium energy
        [140, 180, 40, 0.9, 1.0],   # chorus: 140-180s, high energy
        [180, 210, 30, 0.4, 0.5]    # outro: 180-210s, low energy
    ]
}

# Test prompts for long-form generation
test_prompts = [
    {
        "text": "An uplifting electronic dance track with strong bass and melodic synths",
        "duration": 210.0,
        "filename": "longform_edm_210s.wav"
    },
    {
        "text": "A peaceful ambient soundscape with gentle piano and atmospheric pads",
        "duration": 180.0,
        "filename": "longform_ambient_180s.wav"
    }
]

print("\n" + "=" * 60)
print("🎵 Generating long-form tracks...")
print("=" * 60)

for i, prompt_config in enumerate(test_prompts):
    print(f"\n{'='*60}")
    print(f"Track {i+1}/{len(test_prompts)}")
    print(f"📝 Prompt: {prompt_config['text']}")
    print(f"⏱️  Duration: {prompt_config['duration']}s")
    print(f"{'='*60}")

    try:
        # Generate long-form audio
        full_audio = generate_long_form(
            model=model,
            text=prompt_config["text"],
            structure=structure,
            duration=prompt_config["duration"],
            segment_duration=30.0,
            overlap=5.0,
            cfg_scale=7.5,
            device=device
        )

        # Normalize
        full_audio = full_audio / (np.abs(full_audio).max() + 1e-7)
        full_audio = full_audio * 0.9

        # Save
        output_file = output_dir / prompt_config["filename"]
        sf.write(output_file, full_audio, 24000)

        print(f"\n✅ Track {i+1} complete!")
        print(f"   Saved to: {output_file}")
        print(f"   Duration: {len(full_audio)/24000:.1f}s")
        print(f"   Size: {len(full_audio):,} samples")

        # Also generate 30-second compressed preview
        print(f"\n   🔬 Generating 30s preview...")
        audio_tensor = torch.FloatTensor(full_audio).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            preview = model.get_30_second_preview(audio_tensor, is_audio=True)

        preview_np = preview[0, 0].cpu().numpy()
        preview_np = preview_np / (np.abs(preview_np).max() + 1e-7) * 0.9

        preview_file = output_dir / f"preview_{prompt_config['filename']}"
        sf.write(preview_file, preview_np, 24000)

        print(f"   ✅ 30s preview saved to: {preview_file}")

    except Exception as e:
        print(f"\n❌ Error generating track {i+1}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Long-form generation complete!")
print(f"   Tracks saved to: {output_dir}/")
print("=" * 60)
print("\n🎵 Play generated tracks:")
print(f"   afplay {output_dir}/longform_edm_210s.wav")
print(f"   afplay {output_dir}/longform_ambient_180s.wav")
print("=" * 60)
