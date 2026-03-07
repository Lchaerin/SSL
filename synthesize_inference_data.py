"""
Binaural inference clip synthesis for BinauralSSLNet.

Generates a small number of long (~1 min) binaural WAV files from real sound
effects convolved with HRTF databases, together with their ground-truth heatmaps.

Differences from data_generation.py:
  - Clips are ~60 s long instead of 128 ms (suitable for inference.py)
  - Only a small number of clips are generated (default: 10)
  - No data augmentation applied
  - Output goes to ./data/inference/ by default

Usage:
    python synthesize_inference_data.py [--n_clips 10] [--duration_sec 60]
                                        [--out_dir ./data/inference]
"""

import argparse
import json
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from utils.audio_processing import SAMPLE_RATE, FEATURE_SR, compute_rms_db
from utils.hrtf_synthesis import HRTFDatabasePool
from utils.heatmap_generator import generate_heatmap
from data_generation import (
    AudioCache,
    sample_n_sources,
    sample_source_positions,
    sample_db,
    BUFFER_SAMPLES,
)

warnings.filterwarnings('ignore')

BASE_DIR          = Path(__file__).parent
SOUND_EFFECTS_DIR = BASE_DIR / 'data' / 'sound_effects'
HRIR_DIR          = BASE_DIR / 'data' / 'hrir'


# ── Core synthesis ────────────────────────────────────────────────────────────

def synthesize_clip(
    audio_cache: AudioCache,
    hrtf_pool: HRTFDatabasePool,
    duration_sec: float,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float, float]]]:
    """
    Synthesize one long binaural clip with fixed source positions.

    A short leading buffer (200 ms at 44100 Hz) is added before synthesis
    to let the HRTF convolution settle, then stripped before saving.

    Returns:
        audio:   [2, n_samples] float32 at FEATURE_SR (16 kHz)
        heatmap: [72, 37]       float32 ground-truth heatmap
        sources: list of (azimuth_deg, elevation_deg, level_db)
    """
    n_sources = sample_n_sources()
    positions = sample_source_positions(n_sources)

    # Total samples at synthesis rate with a leading buffer for HRTF transients
    total_synth = int(duration_sec * SAMPLE_RATE) + BUFFER_SAMPLES
    binaural_mix = np.zeros((2, total_synth), dtype=np.float32)
    active_sources: List[Tuple[float, float, float]] = []

    hrtf_db = hrtf_pool.get_random()

    for az, el in positions:
        db = sample_db()
        mono = audio_cache.get_random_segment(total_synth)

        binaural = hrtf_db.synthesize(mono, az, el)  # [2, total_synth]

        # Scale to desired dB level relative to the left channel RMS
        current_db = compute_rms_db(binaural[0])
        if current_db > -70:
            scale = 10.0 ** ((db - current_db) / 20.0)
            binaural = binaural * scale

        binaural_mix += binaural
        active_sources.append((az, el, db))

    # Peak normalise to prevent clipping
    peak = np.max(np.abs(binaural_mix))
    if peak > 1e-6:
        binaural_mix /= peak

    # Strip leading HRTF buffer → [2, duration * SAMPLE_RATE]
    binaural_mix = binaural_mix[:, BUFFER_SAMPLES:]

    # Downsample to FEATURE_SR (16 kHz) for compatibility with inference.py
    audio = np.stack([
        librosa.resample(binaural_mix[0], orig_sr=SAMPLE_RATE, target_sr=FEATURE_SR),
        librosa.resample(binaural_mix[1], orig_sr=SAMPLE_RATE, target_sr=FEATURE_SR),
    ], axis=0).astype(np.float32)  # [2, duration * FEATURE_SR]

    heatmap = generate_heatmap(active_sources)  # [72, 37]

    return audio, heatmap, active_sources


# ── Main loop ─────────────────────────────────────────────────────────────────

def generate_inference_clips(
    n_clips: int = 10,
    duration_sec: float = 60.0,
    out_dir: str = None,
):
    out_path     = Path(out_dir) if out_dir else BASE_DIR / 'data' / 'inference'
    audio_dir    = out_path / 'audio'
    heatmaps_dir = out_path / 'heatmaps'
    audio_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading audio cache from {SOUND_EFFECTS_DIR} ...")
    audio_cache = AudioCache(str(SOUND_EFFECTS_DIR))
    audio_cache.load_all()

    print(f"Loading HRTF pool from {HRIR_DIR} ...")
    hrtf_pool = HRTFDatabasePool(str(HRIR_DIR))
    hrtf_pool.get(hrtf_pool.sofa_paths[0])   # validate first file early
    print(f"  Found {hrtf_pool.n_databases} SOFA file(s).")

    metadata = []
    expected_samples = int(duration_sec * FEATURE_SR)

    print(f"\nSynthesizing {n_clips} clip(s) of ~{duration_sec:.0f} s each ...")
    for i in tqdm(range(n_clips), desc='Synthesizing', unit='clip'):
        try:
            audio, heatmap, sources = synthesize_clip(
                audio_cache, hrtf_pool, duration_sec)

            # Trim or pad to exact duration (resampling may drift by 1-2 samples)
            if audio.shape[1] > expected_samples:
                audio = audio[:, :expected_samples]
            elif audio.shape[1] < expected_samples:
                pad = expected_samples - audio.shape[1]
                audio = np.pad(audio, ((0, 0), (0, pad)))

            # Save binaural WAV [samples, 2] and ground-truth heatmap
            sf.write(audio_dir    / f'{i:06d}.wav', audio.T, FEATURE_SR)
            np.save(heatmaps_dir  / f'{i:06d}.npy', heatmap)

            entry = {
                'id': i,
                'duration_sec': duration_sec,
                'n_samples': audio.shape[1],
                'sample_rate': FEATURE_SR,
                'n_sources': len(sources),
                'sources': [
                    {'azimuth': float(az), 'elevation': float(el), 'db': float(db)}
                    for az, el, db in sources
                ],
            }
            metadata.append(entry)

            src_str = ', '.join(
                f'az={az:+.0f}° el={el:+.0f}° {db:.1f}dB'
                for az, el, db in sources
            )
            tqdm.write(f'  [{i:03d}] {len(sources)} source(s): {src_str}')

        except Exception as e:
            tqdm.write(f'\n[Warning] Clip {i} failed: {e}')
            metadata.append({'id': i, 'error': str(e)})

    meta_path = out_path / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    valid = sum(1 for m in metadata if 'error' not in m)
    print(f"\nDone. {valid}/{n_clips} clip(s) saved → {out_path}")
    if valid:
        print(f"\nRun inference on the first clip:")
        print(f"  python inference.py --audio {audio_dir / '000000.wav'}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Synthesize long binaural clips for inference evaluation')
    parser.add_argument('--n_clips', type=int, default=10,
                        help='Number of clips to generate (default: 10)')
    parser.add_argument('--duration_sec', type=float, default=60.0,
                        help='Duration of each clip in seconds (default: 60)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: ./data/inference)')
    args = parser.parse_args()

    generate_inference_clips(
        n_clips=args.n_clips,
        duration_sec=args.duration_sec,
        out_dir=args.out_dir,
    )
