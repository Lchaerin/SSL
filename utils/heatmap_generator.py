"""
Spherical sound energy heatmap generation.

Output heatmap shape: [N_AZ, N_EL] = [72, 37]
  Azimuth bins:   -180° to +175° in 5° steps (72 bins)
  Elevation bins: -90°  to +90°  in 5° steps (37 bins)
"""

from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

# Output grid constants
N_AZ = 72
N_EL = 37
AZ_STEP = 5     # degrees
EL_STEP = 5     # degrees
AZ_MIN = -180   # degrees
EL_MIN = -90    # degrees

AZ_ANGLES = np.arange(AZ_MIN, AZ_MIN + N_AZ * AZ_STEP, AZ_STEP)   # -180 ... 175
EL_ANGLES = np.arange(EL_MIN, EL_MIN + N_EL * EL_STEP, EL_STEP)   # -90  ... +90

SILENCE_DB = -40.0  # sources below this level are omitted from the heatmap


def az_to_index(azimuth: float) -> int:
    """Convert azimuth (degrees) to grid index [0, N_AZ). Wraps around."""
    idx = round((azimuth - AZ_MIN) / AZ_STEP) % N_AZ
    return int(idx)


def el_to_index(elevation: float) -> int:
    """Convert elevation (degrees) to grid index [0, N_EL]. Clipped."""
    idx = round((elevation - EL_MIN) / EL_STEP)
    return int(np.clip(idx, 0, N_EL - 1))


def _db_to_sigma(db: float) -> float:
    """Gaussian sigma grows with source loudness (in dB)."""
    # sigma = 2.0 at 0 dB, 5.0 at 30 dB (in grid bins)
    return 2.0 + max(0.0, db / 30.0) * 3.0


def _make_source_heatmap(az_idx: int, el_idx: int, sigma: float) -> np.ndarray:
    """
    Create a Gaussian blob on the heatmap grid at (az_idx, el_idx).
    Azimuth dimension is wrapped (toroidal) to handle the -180/+180 boundary.
    """
    # Work on a tiled azimuth to allow wrap-around Gaussian
    tiled = np.zeros((3 * N_AZ, N_EL), dtype=np.float32)
    tiled[N_AZ + az_idx, el_idx] = 1.0
    blurred = gaussian_filter(tiled, sigma=[sigma, sigma])
    # Take the middle tile (wrapping absorbed)
    hmap = blurred[N_AZ:2 * N_AZ, :]
    # Normalize the blob so its peak is 1.0
    if hmap.max() > 0:
        hmap /= hmap.max()
    return hmap


def generate_heatmap(
    sources: List[Tuple[float, float, float]],
    n_az: int = N_AZ,
    n_el: int = N_EL,
) -> np.ndarray:
    """
    Generate a spherical energy heatmap from a list of sound sources.

    Args:
        sources: list of (azimuth_deg, elevation_deg, level_db)

    Returns:
        heatmap: [n_az, n_el] float32 in [0, 1]
    """
    heatmap = np.zeros((n_az, n_el), dtype=np.float32)

    # Group sources at the same grid cell and combine their energy
    cell_energy: dict = {}
    for az, el, db in sources:
        if db < SILENCE_DB:
            continue
        az_idx = az_to_index(az)
        el_idx = el_to_index(el)
        energy = 10.0 ** (db / 10.0)
        cell_energy[(az_idx, el_idx)] = cell_energy.get((az_idx, el_idx), 0.0) + energy

    for (az_idx, el_idx), energy in cell_energy.items():
        db_combined = 10.0 * np.log10(energy + 1e-10)
        sigma = _db_to_sigma(db_combined)
        blob = _make_source_heatmap(az_idx, el_idx, sigma)
        # Element-wise maximum (as recommended in spec)
        heatmap = np.maximum(heatmap, blob * min(1.0, energy ** 0.5))

    # Final normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def compute_peak_position(heatmap: np.ndarray) -> Tuple[float, float]:
    """Return the (azimuth, elevation) in degrees of the heatmap peak."""
    idx = np.argmax(heatmap)
    az_idx, el_idx = np.unravel_index(idx, heatmap.shape)
    return float(AZ_ANGLES[az_idx]), float(EL_ANGLES[el_idx])


def angular_error(az1: float, el1: float, az2: float, el2: float) -> float:
    """
    Great-circle angular distance between two (azimuth, elevation) positions.
    All values in degrees. Returns degrees.
    """
    az1, el1 = np.radians(az1), np.radians(el1)
    az2, el2 = np.radians(az2), np.radians(el2)
    # Dot product of unit vectors on the sphere
    dot = (np.cos(el1) * np.cos(az1) * np.cos(el2) * np.cos(az2)
           + np.cos(el1) * np.sin(az1) * np.cos(el2) * np.sin(az2)
           + np.sin(el1) * np.sin(el2))
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))
