"""
BinauralSSLNet: Hybrid 3D-CNN + 2D-CNN model for binaural sound source localization.

Architecture overview:
  1. 3D-CNN stage: treats [B, 2(IPD/ILD), F, T] as [B, 1, 2, F, T] and applies
     2x2x2 3D convolutions to jointly encode the inter-channel cues.
  2. 2D-CNN stage: extracts spatial-spectral features with progressively larger
     receptive field.
  3. FC head: decodes feature vector to a 72x37 spherical energy heatmap.

Input:  [B, 2, freq_bins, time_frames]  (IPD and ILD as 2 channels)
Output: [B, 72, 37]                     (normalized energy heatmap, values in [0, 1])
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# ── Grid constants ──────────────────────────────────────────────────────────
N_AZ = 72
N_EL = 37
FREQ_BINS   = 129  # FFT_SIZE=256 → 129 one-sided bins
TIME_FRAMES = 29   # 128 ms window at 16 kHz, FFT_SIZE=256, hop=64

MODEL_CONFIG = {
    'freq_bins': FREQ_BINS,
    'time_frames': TIME_FRAMES,
    'n_azimuth': N_AZ,
    'n_elevation': N_EL,
    'base_channels': 32,
}


# ── Building blocks ──────────────────────────────────────────────────────────

class Conv2dBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock2d(nn.Module):
    """Lightweight residual block for 2D feature maps."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dBnRelu(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


# ── Main model ───────────────────────────────────────────────────────────────

class BinauralSSLNet(nn.Module):
    """
    Binaural Sound Source Localization Network.

    Args:
        freq_bins:    Number of frequency bins (default 257)
        time_frames:  Number of time frames (default 16)
        base_ch:      Base number of channels for the 2D-CNN stage
    """

    def __init__(
        self,
        freq_bins: int = FREQ_BINS,
        time_frames: int = TIME_FRAMES,
        base_ch: int = 32,
    ):
        super().__init__()
        self.freq_bins = freq_bins
        self.time_frames = time_frames

        # ── Stage 1: 3D-CNN ─────────────────────────────────────────────────
        # Input viewed as [B, 1, 2(channel_depth), F, T]
        # Kernel (2, 2, 2): collapses the IPD/ILD depth dimension entirely
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, base_ch, kernel_size=(2, 2, 2),
                      stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
            # Second 3D conv to enrich cross-channel features (depth stays 1)
            nn.Conv3d(base_ch, base_ch, kernel_size=(1, 3, 3),
                      stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
        )
        # After conv3d: [B, base_ch, 1, F, T] → squeeze depth → [B, base_ch, F, T]

        # ── Stage 2: 2D-CNN ─────────────────────────────────────────────────
        # With FREQ_BINS≈129, TIME_FRAMES≈29 after 3D-CNN:
        #   Block1 MaxPool(2): [B, 64,  ~65, ~15]
        #   Block2 MaxPool(2): [B, 128, ~32, ~7 ]
        #   Block3 (no pool):  [B, 256, ~32, ~7 ]  ← keep spatial size for AdaptivePool
        #   Block4:            [B, 256, ~32, ~7 ]
        #   AdaptiveAvgPool:   [B, 256,   4,  4 ]  ← 7≥4 and 32≥4 ✓
        self.cnn = nn.Sequential(
            # Block 1
            Conv2dBnRelu(base_ch, base_ch * 2),
            ResBlock2d(base_ch * 2),
            nn.MaxPool2d(2),                         # freq/2, time/2

            # Block 2
            Conv2dBnRelu(base_ch * 2, base_ch * 4),
            ResBlock2d(base_ch * 4),
            nn.MaxPool2d(2),                         # freq/4, time/4

            # Block 3  (no MaxPool — time dim is ~7, too small for another /2)
            Conv2dBnRelu(base_ch * 4, base_ch * 8),
            ResBlock2d(base_ch * 8),

            # Block 4
            Conv2dBnRelu(base_ch * 8, base_ch * 8),
        )

        # Adaptive pooling ensures model works with any input time length
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        flat_size = (base_ch * 8) * 4 * 4  # 256 * 16 = 4096

        # ── Stage 3: FC head ─────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, N_AZ * N_EL),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 2, F, T]  IPD and ILD features
        Returns:
            heatmap: [B, 72, 37]
        """
        B = x.shape[0]
        # Treat 2 feature channels as 3D depth dimension
        x = x.unsqueeze(1)           # [B, 1, 2, F, T]
        x = self.conv3d(x)           # [B, C, 1, F', T']
        x = x.squeeze(2)             # [B, C, F', T']

        x = self.cnn(x)              # [B, 256, ...]
        x = self.global_pool(x)      # [B, 256, 4, 4]
        x = self.head(x)             # [B, 72*37]
        return x.view(B, N_AZ, N_EL)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_config(self, path: str):
        cfg = {
            'freq_bins': self.freq_bins,
            'time_frames': self.time_frames,
            'base_ch': MODEL_CONFIG['base_channels'],
            'n_azimuth': N_AZ,
            'n_elevation': N_EL,
        }
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_config(cls, config_path: str) -> 'BinauralSSLNet':
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        return cls(
            freq_bins=cfg['freq_bins'],
            time_frames=cfg['time_frames'],
            base_ch=cfg.get('base_ch', 32),
        )

    @classmethod
    def load(cls, checkpoint_path: str, config_path: Optional[str] = None,
             device: str = 'cpu') -> 'BinauralSSLNet':
        """Load model from checkpoint (and optionally a config JSON)."""
        if config_path is None:
            config_path = str(Path(checkpoint_path).parent / 'model_config.json')
        if Path(config_path).exists():
            model = cls.from_config(config_path)
        else:
            model = cls()
        state = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        model.to(device)
        return model
