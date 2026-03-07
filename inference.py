"""
Real-time binaural sound source localization inference.

Processes a binaural WAV file with a sliding window (50% overlap, 64 ms step) and
displays an animated spherical energy heatmap.

Usage:
    python inference.py --audio <input.wav> [--model ./checkpoints/model_best.pth]
                        [--window_ms 100] [--device cpu]
                        [--save_frames ./frames/]
                        [--no_display]
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Generator

import numpy as np
import torch
import soundfile as sf
import librosa

from models import BinauralSSLNet
from utils.audio_processing import (
    FEATURE_SR, FFT_SIZE, HOP_LENGTH,
    compute_ipd_ild,
)
from utils.heatmap_generator import AZ_ANGLES, EL_ANGLES, compute_peak_position

BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_binaural_wav(path: str, target_sr: int = FEATURE_SR) -> np.ndarray:
    """
    Load a binaural (stereo) audio file.

    Returns:
        audio: [2, n_samples] float32
    """
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    audio = audio.T  # [channels, samples]
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    if sr != target_sr:
        left = librosa.resample(audio[0], orig_sr=sr, target_sr=target_sr)
        right = librosa.resample(audio[1], orig_sr=sr, target_sr=target_sr)
        audio = np.stack([left, right], axis=0)

    return audio.astype(np.float32)


# ── Sliding window ────────────────────────────────────────────────────────────

def sliding_windows(
    audio: np.ndarray,
    window_ms: int = 128,
    overlap: float = 0.5,
    sr: int = FEATURE_SR,
) -> Generator[np.ndarray, None, None]:
    """
    Yield overlapping windows from a binaural audio array.

    Args:
        audio:      [2, n_samples]
        window_ms:  window length in milliseconds (default: 128)
        overlap:    fractional overlap (default: 0.5 → 64 ms step)
    Yields:
        window: [2, window_samples]
    """
    window_samples = int(window_ms / 1000 * sr)
    step_samples = int(window_samples * (1.0 - overlap))
    n_samples = audio.shape[1]

    start = 0
    while start + window_samples <= n_samples:
        yield audio[:, start:start + window_samples]
        start += step_samples

    # Final partial window (zero-padded)
    if start < n_samples:
        segment = audio[:, start:]
        pad = np.zeros((2, window_samples - segment.shape[1]), dtype=audio.dtype)
        yield np.concatenate([segment, pad], axis=1)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(window: np.ndarray, target_frames: Optional[int] = None) -> np.ndarray:
    """
    Compute IPD/ILD features from a [2, window_samples] window.
    Returns [2, freq_bins, time_frames] float32.
    """
    features = compute_ipd_ild(window)  # [2, F, T]
    if target_frames is not None:
        T = features.shape[2]
        if T < target_frames:
            features = np.pad(features, ((0, 0), (0, 0), (0, target_frames - T)))
        else:
            features = features[:, :, :target_frames]
    return features


# ── Inference engine ──────────────────────────────────────────────────────────

class SSLInference:
    """
    Wrapper for streaming inference with BinauralSSLNet.

    Supports single-window and batch inference for throughput.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        config_path = str(Path(checkpoint_path).parent / 'model_config.json')
        self.model = BinauralSSLNet.load(
            checkpoint_path, config_path, device=device)
        self.model.eval()
        print(f"Model loaded on {device}. Parameters: {self.model.count_parameters():,}")

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict heatmap for a single feature array or a batch.

        Args:
            features: [2, F, T] or [B, 2, F, T]
        Returns:
            heatmap: [72, 37] or [B, 72, 37]
        """
        if features.ndim == 3:
            x = torch.from_numpy(features).unsqueeze(0).to(self.device)
            out = self.model(x)
            return out.squeeze(0).cpu().numpy()
        else:
            x = torch.from_numpy(features).to(self.device)
            return self.model(x).cpu().numpy()

    @torch.no_grad()
    def predict_batch(self, feature_list: list, batch_size: int = 32) -> list:
        """Predict heatmaps for a list of feature arrays in batches."""
        results = []
        for i in range(0, len(feature_list), batch_size):
            batch = np.stack(feature_list[i:i + batch_size], axis=0)
            results.extend(self.predict(batch).tolist())
        return results


# ── Visualization ─────────────────────────────────────────────────────────────

def setup_matplotlib_animation():
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlabel('Azimuth (°)')
    ax.set_ylabel('Elevation (°)')
    ax.set_title('Real-time Binaural Sound Source Localization')
    ax.set_xticks(np.linspace(0, 71, 9))
    ax.set_xticklabels([f'{int(a)}°' for a in np.linspace(-180, 180, 9)])
    ax.set_yticks(np.linspace(0, 36, 7))
    ax.set_yticklabels([f'{int(e)}°' for e in np.linspace(-90, 90, 7)])

    dummy = np.zeros((72, 37))
    im = ax.imshow(dummy.T, origin='lower', aspect='auto',
                   cmap='hot', vmin=0, vmax=1, animated=True)
    plt.colorbar(im, ax=ax, label='Energy')
    peak_dot, = ax.plot([], [], 'c+', markersize=12, markeredgewidth=2)

    return fig, ax, im, peak_dot


def run_realtime(args):
    """Run inference with live matplotlib animation."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    print(f"Loading audio: {args.audio}")
    audio = load_binaural_wav(args.audio)
    print(f"  Duration: {audio.shape[1] / FEATURE_SR:.2f}s  SR: {FEATURE_SR} Hz")

    engine = SSLInference(args.model, device=args.device)

    step_ms = args.window_ms // 2   # 50 % overlap → step = window / 2
    windows = list(sliding_windows(audio, window_ms=args.window_ms, overlap=0.5))
    print(f"  Windows: {len(windows)} (step={step_ms} ms)")

    # Pre-compute target time frames
    target_T = 1 + (int(args.window_ms / 1000 * FEATURE_SR) - FFT_SIZE) // HOP_LENGTH

    # Pre-compute all heatmaps for smooth playback
    print("Pre-computing heatmaps...")
    heatmaps = []
    t_start = time.time()
    for w in windows:
        feat = extract_features(w, target_frames=target_T)
        hm = engine.predict(feat)
        heatmaps.append(hm)
    elapsed = time.time() - t_start
    print(f"  Computed {len(heatmaps)} frames in {elapsed:.2f}s "
          f"({elapsed / len(heatmaps) * 1000:.1f} ms/frame)")

    # Save frames if requested
    if args.save_frames:
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt2
        os.makedirs(args.save_frames, exist_ok=True)
        for i, hm in enumerate(heatmaps):
            fig2, ax2 = plt2.subplots(figsize=(10, 4))
            ax2.imshow(hm.T, origin='lower', aspect='auto',
                       cmap='hot', vmin=0, vmax=1)
            az, el = compute_peak_position(hm)
            ax2.set_title(f'Frame {i:04d} | Peak: az={az:.0f}° el={el:.0f}°')
            fig2.savefig(Path(args.save_frames) / f'frame_{i:04d}.png', dpi=80)
            plt2.close(fig2)
        print(f"Saved {len(heatmaps)} frames to {args.save_frames}")

    if args.no_display:
        return

    # Animate
    import matplotlib
    matplotlib.use('TkAgg' if not args.no_display else 'Agg')

    fig, ax, im, peak_dot = setup_matplotlib_animation()

    # step_ms already defined above (window_ms // 2)
    frame_idx = [0]

    def update(_):
        i = frame_idx[0] % len(heatmaps)
        hm = heatmaps[i]
        im.set_data(hm.T)
        az, el = compute_peak_position(hm)
        az_i = np.argmin(np.abs(AZ_ANGLES - az))
        el_i = np.argmin(np.abs(EL_ANGLES - el))
        peak_dot.set_data([az_i], [el_i])
        ax.set_title(
            f'Frame {i+1}/{len(heatmaps)} | '
            f't={i * step_ms / 1000:.2f}s | '
            f'Peak: az={az:.0f}° el={el:.0f}°'
        )
        frame_idx[0] += 1
        return im, peak_dot

    anim = animation.FuncAnimation(
        fig, update, frames=len(heatmaps),
        interval=max(step_ms, 40), blit=True, repeat=False)

    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time binaural SSL inference')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to binaural WAV file')
    parser.add_argument('--model', type=str,
                        default=str(CHECKPOINT_DIR / 'model_best.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--window_ms', type=int, default=128,
                        help='Analysis window size in ms (default: 128)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Inference device (default: cuda if available)')
    parser.add_argument('--save_frames', type=str, default=None,
                        help='Directory to save heatmap frames as PNG images')
    parser.add_argument('--no_display', action='store_true',
                        help='Skip live matplotlib display (useful for headless)')
    args = parser.parse_args()

    run_realtime(args)
