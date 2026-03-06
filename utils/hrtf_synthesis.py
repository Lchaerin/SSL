"""
HRTF synthesis from SOFA files.

Loads Head-Related Impulse Response (HRIR) data from IRCAM Listen SOFA files
and synthesizes binaural audio by convolving mono audio with HRIRs.

SOFA coordinate convention:
  Azimuth:   0 = front, 90 = left, 180/-180 = back, -90/270 = right
  Elevation: positive = up, negative = down
"""

import os
import random
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import scipy.signal
from scipy.spatial import KDTree


def _load_sofa(sofa_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load SOFA file using netCDF4.

    Returns:
        source_positions: [M, 3] (azimuth, elevation, radius) in degrees
        hrir:             [M, 2, N] impulse responses
        sample_rate:      int
    """
    import netCDF4 as nc4
    ds = nc4.Dataset(sofa_path, 'r')
    source_positions = np.array(ds.variables['SourcePosition'][:], dtype=np.float64)  # [M, 3]
    hrir = np.array(ds.variables['Data.IR'][:], dtype=np.float32)                     # [M, 2, N]
    sr_var = ds.variables['Data.SamplingRate'][:]
    sample_rate = int(np.round(float(np.asarray(sr_var).flatten()[0])))
    ds.close()
    return source_positions, hrir, sample_rate


def _az_to_standard(az: np.ndarray) -> np.ndarray:
    """Convert azimuth from [0, 360) to [-180, 180) range."""
    az = az.copy()
    az[az > 180] -= 360
    return az


def _spherical_to_cartesian(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """Convert spherical (azimuth, elevation) in degrees to unit Cartesian vectors."""
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.stack([x, y, z], axis=-1)


class HRTFDatabase:
    """Loads a single SOFA HRTF file and provides HRIR retrieval with interpolation."""

    def __init__(self, sofa_path: str, n_neighbors: int = 3):
        self.sofa_path = sofa_path
        self.n_neighbors = n_neighbors
        self._load()

    def _load(self):
        positions, hrir, sr = _load_sofa(self.sofa_path)

        # Normalize azimuth to [-180, 180)
        positions[:, 0] = _az_to_standard(positions[:, 0])

        self.source_positions = positions  # [M, 3]
        self.hrir = hrir                   # [M, 2, N]
        self.sample_rate = sr
        self.hrir_len = hrir.shape[2]

        # Build KDTree in Cartesian space for nearest-neighbor lookup
        cart = _spherical_to_cartesian(positions[:, 0], positions[:, 1])  # [M, 3]
        self.tree = KDTree(cart)

    def get_hrir(self, azimuth: float, elevation: float) -> np.ndarray:
        """
        Retrieve interpolated HRIR for a given (azimuth, elevation) in degrees.

        Uses inverse-distance weighted interpolation over the k nearest measured positions.

        Returns:
            hrir: [2, N] float32 impulse response
        """
        query = _spherical_to_cartesian(np.array([azimuth]), np.array([elevation]))[0]  # [3]
        distances, indices = self.tree.query(query, k=self.n_neighbors)

        if distances[0] < 1e-6:
            # Exact match
            return self.hrir[indices[0]].copy()

        # Inverse distance weighting
        weights = 1.0 / distances
        weights /= weights.sum()

        hrir = np.zeros((2, self.hrir_len), dtype=np.float32)
        for w, idx in zip(weights, indices):
            hrir += w * self.hrir[idx]
        return hrir

    def synthesize(self, mono_audio: np.ndarray, azimuth: float, elevation: float) -> np.ndarray:
        """
        Convolve mono audio with HRIR to produce binaural audio.

        Args:
            mono_audio: [n_samples] float32 mono signal
            azimuth:    degrees, -180 to +180
            elevation:  degrees, -90 to +90
        Returns:
            binaural: [2, n_samples] float32
        """
        hrir = self.get_hrir(azimuth, elevation)  # [2, N]
        left = scipy.signal.fftconvolve(mono_audio, hrir[0])[:len(mono_audio)]
        right = scipy.signal.fftconvolve(mono_audio, hrir[1])[:len(mono_audio)]
        return np.stack([left, right], axis=0).astype(np.float32)


class HRTFDatabasePool:
    """
    Manages a pool of HRTF databases (one per SOFA file).
    Lazily loads databases on first access to save memory.
    """

    def __init__(self, hrir_dir: str):
        self.hrir_dir = Path(hrir_dir)
        sofa_files = sorted(self.hrir_dir.glob('*.sofa'))
        if not sofa_files:
            raise FileNotFoundError(f"No .sofa files found in {hrir_dir}")
        self.sofa_paths = [str(p) for p in sofa_files]
        self._cache: dict = {}

    def get(self, sofa_path: str) -> HRTFDatabase:
        if sofa_path not in self._cache:
            self._cache[sofa_path] = HRTFDatabase(sofa_path)
        return self._cache[sofa_path]

    def get_random(self) -> HRTFDatabase:
        """Return a randomly selected HRTF database."""
        return self.get(random.choice(self.sofa_paths))

    def preload_all(self):
        """Load all SOFA files into memory."""
        for path in self.sofa_paths:
            self.get(path)

    @property
    def n_databases(self) -> int:
        return len(self.sofa_paths)

    def synthesize(
        self,
        mono_audio: np.ndarray,
        azimuth: float,
        elevation: float,
        sofa_path: Optional[str] = None,
    ) -> np.ndarray:
        """Synthesize binaural audio using a specific or random HRTF."""
        db = self.get(sofa_path) if sofa_path else self.get_random()
        return db.synthesize(mono_audio, azimuth, elevation)
