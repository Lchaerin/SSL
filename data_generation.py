"""
Dataset generation script for Binaural Sound Source Localization.

Generates 30,000 training samples:
  - Each sample: 100 ms binaural audio → IPD/ILD features + Gaussian heatmap label
  - Multiple simultaneous sources per scene
  - Diverse azimuth/elevation/SNR distributions per CLAUDE.md spec

Usage:
    python data_generation.py [--n_samples 30000] [--workers 4]
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from utils.audio_processing import (
    SAMPLE_RATE, FEATURE_SR, WINDOW_SAMPLES_SYNTH,
    compute_rms_db,
)
from utils.hrtf_synthesis import HRTFDatabasePool
from utils.heatmap_generator import generate_heatmap

warnings.filterwarnings('ignore')

# ── Output paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'generated'
HEATMAPS_DIR = DATA_DIR / 'heatmaps'
AUDIO_DIR    = DATA_DIR / 'audio'
METADATA_PATH = DATA_DIR / 'metadata.json'

SOUND_EFFECTS_DIR = BASE_DIR / 'data' / 'sound_effects'
HRIR_DIR = BASE_DIR / 'data' / 'hrir'

# ── Dataset composition (CLAUDE.md spec) ─────────────────────────────────────
SOURCE_COUNTS = [1, 2, 3, 4, 5]
SOURCE_WEIGHTS = [0.20, 0.333, 0.333, 0.10, 0.034]

# Azimuth distribution (ranges in degrees)
AZ_RANGES = [(-30, 30), (30, 90), (-90, -30), (90, 150), (-150, -90), (150, 180), (-180, -150)]
AZ_WEIGHTS = [0.25, 0.35 / 2, 0.35 / 2, 0.25 / 2, 0.25 / 2, 0.15 / 2, 0.15 / 2]

# Elevation distribution
EL_RANGES = [(-10, 10), (10, 45), (-45, -10)]
EL_WEIGHTS = [0.50, 0.25, 0.25]

# Angular separation between sources (when multiple)
SEP_RANGES = [(5, 20), (20, 60), (60, 180)]
SEP_WEIGHTS = [0.30, 0.40, 0.30]

# SNR / dB level distribution
DB_RANGES = [(20, 30), (10, 20), (0, 10), (-5, 0)]
DB_WEIGHTS = [0.30, 0.40, 0.20, 0.10]

# Data augmentation probability
AUG_PROB = 0.5


# ── Sampling helpers ─────────────────────────────────────────────────────────

def sample_from_ranges(ranges, weights):
    """Draw one float uniformly from a weighted set of (lo, hi) ranges."""
    lo, hi = random.choices(ranges, weights=weights, k=1)[0]
    return random.uniform(lo, hi)


def sample_azimuth() -> float:
    return sample_from_ranges(AZ_RANGES, AZ_WEIGHTS)


def sample_elevation() -> float:
    return sample_from_ranges(EL_RANGES, EL_WEIGHTS)


def sample_db() -> float:
    return sample_from_ranges(DB_RANGES, DB_WEIGHTS)


def sample_n_sources() -> int:
    return random.choices(SOURCE_COUNTS, weights=SOURCE_WEIGHTS, k=1)[0]


def sample_angular_separation() -> float:
    return sample_from_ranges(SEP_RANGES, SEP_WEIGHTS)


def sample_source_positions(n_sources: int) -> List[Tuple[float, float]]:
    """
    Sample (azimuth, elevation) for n_sources, respecting angular separation
    constraints for the multi-source case.
    """
    positions = []
    az = sample_azimuth()
    el = sample_elevation()
    positions.append((az, el))

    for _ in range(n_sources - 1):
        # Try to place next source at the requested angular separation
        target_sep = sample_angular_separation()
        best = None
        best_dist = np.inf
        for _ in range(30):
            az2 = sample_azimuth()
            el2 = sample_elevation()
            # Compute angular distance from all existing sources
            min_sep = np.inf
            for az1, el1 in positions:
                daz = np.radians(az2 - az1)
                del1 = np.radians(el1)
                del2 = np.radians(el2)
                dot = (np.cos(del1) * np.cos(del2) * np.cos(daz)
                       + np.sin(del1) * np.sin(del2))
                dot = np.clip(dot, -1.0, 1.0)
                sep = np.degrees(np.arccos(dot))
                min_sep = min(min_sep, sep)
            dist = abs(min_sep - target_sep)
            if dist < best_dist:
                best_dist = dist
                best = (az2, el2)
        positions.append(best)

    return positions


# ── Audio cache ───────────────────────────────────────────────────────────────

class AudioCache:
    """Loads and caches all sound effect files in memory."""

    def __init__(self, sound_dir: str):
        self.files = sorted(Path(sound_dir).glob('*.mp3'))
        if not self.files:
            raise FileNotFoundError(f"No .mp3 files in {sound_dir}")
        self._cache: Dict[str, np.ndarray] = {}

    def load_all(self):
        for f in tqdm(self.files, desc='Loading sound effects', leave=False):
            self._get(str(f))

    def _get(self, path: str) -> np.ndarray:
        if path not in self._cache:
            audio, sr = librosa.load(path, sr=None, mono=True)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            self._cache[path] = audio.astype(np.float32)
        return self._cache[path]

    def get_random_segment(self, length: int) -> np.ndarray:
        """Return a random segment of the requested length from a random file."""
        path = str(random.choice(self.files))
        audio = self._get(path)
        if len(audio) <= length:
            # Loop the audio to fill the segment
            repeats = (length // len(audio)) + 2
            audio = np.tile(audio, repeats)
        start = random.randint(0, len(audio) - length)
        return audio[start:start + length].copy()


# ── Data augmentation ─────────────────────────────────────────────────────────

def augment_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply random augmentations (pitch shift, time stretch, volume scale)."""
    import librosa

    # Volume scale ±3 dB
    if random.random() < AUG_PROB:
        db_shift = random.uniform(-3.0, 3.0)
        audio = audio * (10.0 ** (db_shift / 20.0))

    # Pitch shift ±2 semitones
    if random.random() < AUG_PROB:
        n_steps = random.uniform(-2.0, 2.0)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    # Time stretch 0.9 ~ 1.1x
    if random.random() < AUG_PROB:
        rate = random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=rate)

    return audio


# ── Sample generation ─────────────────────────────────────────────────────────

# Extra buffer around the window for realistic HRTF convolution (at 44100 Hz)
BUFFER_SAMPLES = SAMPLE_RATE // 5                        # 200 ms at 44100 Hz
MIX_LENGTH     = WINDOW_SAMPLES_SYNTH + 2 * BUFFER_SAMPLES


def generate_one_sample(
    audio_cache: AudioCache,
    hrtf_pool: HRTFDatabasePool,
    augment: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Generate a single training sample.

    Returns:
        window:  [2, WINDOW_SAMPLES] float32  binaural audio
        heatmap: [72, 37] float32
        sources: list of (azimuth, elevation, db)
    """
    n_sources = sample_n_sources()
    positions = sample_source_positions(n_sources)

    binaural_mix = np.zeros((2, MIX_LENGTH), dtype=np.float32)  # at SAMPLE_RATE (44100 Hz)
    active_sources: List[Tuple[float, float, float]] = []
    hrtf_db = hrtf_pool.get_random()

    for az, el in positions:
        db = sample_db()
        mono = audio_cache.get_random_segment(MIX_LENGTH)

        # Optional augmentation on the dry mono signal
        if augment:
            mono = augment_audio(mono)
            mono = mono[:MIX_LENGTH]
            if len(mono) < MIX_LENGTH:
                mono = np.pad(mono, (0, MIX_LENGTH - len(mono)))

        # Synthesize binaural
        binaural = hrtf_db.synthesize(mono, az, el)  # [2, MIX_LENGTH]

        # Scale to desired dB level
        current_db = compute_rms_db(binaural[0])
        if current_db > -70:
            scale = 10.0 ** ((db - current_db) / 20.0)
            binaural = binaural * scale

        binaural_mix += binaural
        active_sources.append((az, el, db))

    # Peak normalization to prevent clipping
    peak = np.max(np.abs(binaural_mix))
    if peak > 1e-6:
        binaural_mix /= peak

    # Extract centre window (at 44100 Hz) where all sources overlap
    start = BUFFER_SAMPLES
    window_synth = binaural_mix[:, start:start + WINDOW_SAMPLES_SYNTH]  # [2, 5644]

    # Downsample to FEATURE_SR (16 kHz) for feature extraction and storage
    window = np.stack([
        librosa.resample(window_synth[0], orig_sr=SAMPLE_RATE, target_sr=FEATURE_SR),
        librosa.resample(window_synth[1], orig_sr=SAMPLE_RATE, target_sr=FEATURE_SR),
    ], axis=0)  # [2, 2048]

    # Generate ground truth heatmap
    heatmap = generate_heatmap(active_sources)   # [72, 37]

    return window, heatmap, active_sources


# ── Main generation loop ──────────────────────────────────────────────────────

def generate_dataset(n_samples: int = 30_000, augment: bool = True):
    """Generate the full dataset and save to disk."""
    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Initialising audio cache from {SOUND_EFFECTS_DIR} ...")
    audio_cache = AudioCache(str(SOUND_EFFECTS_DIR))
    audio_cache.load_all()

    print(f"Loading HRTF pool from {HRIR_DIR} ...")
    hrtf_pool = HRTFDatabasePool(str(HRIR_DIR))
    # Pre-load first SOFA file to catch errors early
    hrtf_pool.get(hrtf_pool.sofa_paths[0])
    print(f"  Found {hrtf_pool.n_databases} SOFA files.")

    metadata = []
    errors = 0

    print(f"\nGenerating {n_samples:,} samples ...")
    for i in tqdm(range(n_samples), desc='Generating', unit='sample'):
        try:
            # Augment on training portion (first 80%)
            do_aug = augment and (i < int(n_samples * 0.8))
            window, heatmap, sources = generate_one_sample(
                audio_cache, hrtf_pool, augment=do_aug)

            # Save binaural audio (.wav) at FEATURE_SR (16 kHz) and ground truth heatmap (.npy)
            # window: [2, 2048] → soundfile expects [samples, channels]
            sf.write(AUDIO_DIR / f'{i:06d}.wav', window.T, FEATURE_SR)
            np.save(HEATMAPS_DIR / f'{i:06d}.npy', heatmap)

            metadata.append({
                'id': i,
                'n_sources': len(sources),
                'sources': [
                    {'azimuth': az, 'elevation': el, 'db': db}
                    for az, el, db in sources
                ],
            })

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"\n[Warning] Sample {i} failed: {e}")
            metadata.append({'id': i, 'error': str(e)})

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    valid = n_samples - errors
    print(f"\nDone. {valid:,}/{n_samples:,} samples generated successfully.")
    print(f"Output directory: {DATA_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate binaural SSL dataset')
    parser.add_argument('--n_samples', type=int, default=30_000,
                        help='Number of samples to generate (default: 30000)')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.n_samples,
        augment=not args.no_augment,
    )
