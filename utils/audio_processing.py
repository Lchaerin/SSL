"""
IPD (Inter-aural Phase Difference) and ILD (Inter-aural Level Difference) computation.

Input binaural audio shape: [2, n_samples]
Output feature shape:       [2, freq_bins, time_frames]
  channel 0 = IPD (normalized to [-1, 1])
  channel 1 = ILD (tanh-normalized)
"""

import numpy as np
import torch
import torchaudio
import librosa

# Sample rates
SAMPLE_RATE = 44100   # HRTF synthesis rate (do not change)
FEATURE_SR  = 16000   # Downsample to this rate before computing IPD/ILD

# STFT parameters (applied at FEATURE_SR)
FFT_SIZE    = 256
HOP_LENGTH  = 64
WINDOW_TYPE = 'hann'

# Window / overlap spec
WINDOW_MS  = 128   # analysis window length  [ms]
OVERLAP_MS = 64    # step between windows    [ms]  (50 % overlap)

# Derived dimensions
FREQ_BINS          = FFT_SIZE // 2 + 1                              # 129
WINDOW_SAMPLES     = int(WINDOW_MS  / 1000 * FEATURE_SR)           # 2048  (at 16 kHz)
WINDOW_SAMPLES_SYNTH = int(WINDOW_MS / 1000 * SAMPLE_RATE)         # 5644  (at 44100 Hz)
TIME_FRAMES        = 1 + (WINDOW_SAMPLES - FFT_SIZE) // HOP_LENGTH # 29


def load_audio(file_path: str, target_sr: int = FEATURE_SR) -> np.ndarray:
    """Load mono audio file and resample if needed. Returns [n_samples] float32 array."""
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def compute_stft(audio: np.ndarray, n_fft: int = FFT_SIZE, hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Compute complex STFT of a mono signal.

    Args:
        audio: [n_samples] float32
    Returns:
        stft: [freq_bins, time_frames] complex64
    """
    window = torch.hann_window(n_fft)
    x = torch.from_numpy(audio).float()
    stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length,
                      win_length=n_fft, window=window,
                      center=False, return_complex=True)
    return stft.numpy()  # [freq_bins, time_frames]


def compute_ipd_ild(
    binaural_audio: np.ndarray,
    n_fft: int = FFT_SIZE,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Compute IPD and ILD features from binaural audio.

    Args:
        binaural_audio: [2, n_samples] float32
    Returns:
        features: [2, freq_bins, time_frames] float32
            features[0] = IPD / pi  -> [-1, 1]
            features[1] = tanh(ILD / 20)  -> ~[-1, 1]
    """
    assert binaural_audio.shape[0] == 2, "Input must be 2-channel binaural audio"

    L = compute_stft(binaural_audio[0], n_fft, hop_length)  # [F, T] complex
    R = compute_stft(binaural_audio[1], n_fft, hop_length)  # [F, T] complex

    # IPD: phase difference L - R, range [-pi, pi]
    cross_spectrum = L * np.conj(R)
    ipd = np.angle(cross_spectrum)           # [F, T]
    ipd_norm = ipd / np.pi                   # normalize to [-1, 1]

    # ILD: 20 * log10(|L| / |R|) in dB
    eps = 1e-8
    mag_L = np.abs(L) + eps
    mag_R = np.abs(R) + eps
    # Set ILD to 0 when both channels are silent
    silence_mask = (np.abs(L) + np.abs(R)) < eps * 10
    ild = 20.0 * np.log10(mag_L / mag_R)    # [F, T]
    ild[silence_mask] = 0.0
    ild_norm = np.tanh(ild / 20.0)           # normalize to ~[-1, 1]

    features = np.stack([ipd_norm, ild_norm], axis=0).astype(np.float32)  # [2, F, T]
    return features


def pad_or_trim_features(features: np.ndarray, target_frames: int = TIME_FRAMES) -> np.ndarray:
    """Pad or trim time dimension to target_frames."""
    current_frames = features.shape[2]
    if current_frames == target_frames:
        return features
    elif current_frames > target_frames:
        return features[:, :, :target_frames]
    else:
        pad_width = target_frames - current_frames
        return np.pad(features, ((0, 0), (0, 0), (0, pad_width)), mode='constant')


def extract_window(audio: np.ndarray, start: int, window_samples: int = WINDOW_SAMPLES) -> np.ndarray:
    """Extract a window from binaural audio with zero-padding if needed."""
    end = start + window_samples
    if end <= audio.shape[1]:
        return audio[:, start:end]
    # Zero-pad if audio is too short
    segment = audio[:, start:]
    pad = np.zeros((2, end - audio.shape[1]), dtype=audio.dtype)
    return np.concatenate([segment, pad], axis=1)


def compute_rms_db(audio: np.ndarray, eps: float = 1e-8) -> float:
    """Compute RMS level in dB."""
    rms = np.sqrt(np.mean(audio ** 2) + eps)
    return 20.0 * np.log10(rms)
