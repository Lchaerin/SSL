# Binaural Sound Source Localization (SSL)

Deep learning model that takes binaural (2-channel) audio as input and outputs a real-time spherical sound energy heatmap showing the estimated positions of active sound sources.

---

## Project Structure

```
SSL/
├── data/
│   ├── sound_effects/          # Input .mp3 effect files
│   ├── hrir/                   # .sofa HRTF files (IRCAM Listen)
│   └── generated/              # Generated training data
│       ├── features/           # IPD/ILD feature npy files
│       ├── heatmaps/           # Ground-truth heatmap npy files
│       └── metadata.json
├── models/
│   ├── __init__.py
│   └── model.py                # BinauralSSLNet architecture
├── utils/
│   ├── __init__.py
│   ├── audio_processing.py     # IPD/ILD computation
│   ├── hrtf_synthesis.py       # SOFA loading + binaural synthesis
│   └── heatmap_generator.py    # Gaussian heatmap generation
├── data_generation.py          # Dataset generation (30,000 samples)
├── train.py                    # Training script
├── evaluate.py                 # Evaluation + report generation
├── inference.py                # Real-time streaming inference
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Step 1 — Generate Dataset

```bash
python data_generation.py
# Creates 30,000 samples in ./data/generated/
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--n_samples` | 30000 | Number of samples |
| `--no_augment` | off | Disable data augmentation |

---

### Step 2 — Train

```bash
python train.py
# Saves checkpoints to ./checkpoints/
# Logs to ./logs/ (view with tensorboard --logdir logs)
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--batch_size` | 32 | Batch size |
| `--epochs` | 100 | Max epochs |
| `--lr` | 1e-3 | Learning rate |
| `--patience` | 15 | Early stopping patience |
| `--workers` | 4 | DataLoader worker processes |

---

### Step 3 — Evaluate

```bash
python evaluate.py --checkpoint ./checkpoints/model_best.pth
# Prints metrics and saves evaluation_report.md
```

---

### Step 4 — Real-time Inference

```bash
python inference.py --audio <input.wav> --model ./checkpoints/model_best.pth
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--window_ms` | 100 | Analysis window (ms) |
| `--device` | auto | `cpu` or `cuda` |
| `--save_frames` | None | Save heatmap PNGs to directory |
| `--no_display` | off | Headless mode (no GUI) |

---

## Model Architecture

**BinauralSSLNet** — Hybrid 3D-CNN + 2D-CNN

```
Input:  [B, 2, 257, 16]          (IPD / ILD features)

   3D-CNN Stage
   unsqueeze -> [B, 1, 2, 257, 16]
   Conv3D(1->32, 2x2x2) — collapses IPD/ILD depth
   Conv3D(32->32, 1x3x3)
   squeeze -> [B, 32, 257, 16]

   2D-CNN Stage  (4 blocks + residual connections)
   32 -> 64 -> 128 -> 256 channels
   AdaptiveAvgPool2D(4, 4) -> [B, 256, 4, 4]

   FC Head
   Flatten -> Linear(4096, 1024) -> Linear(1024, 512)
   -> Linear(512, 72x37) -> Sigmoid

Output: [B, 72, 37]              (azimuth x elevation heatmap)
```

**Parameters:** ~7.5 M
**Inference latency:** <50 ms on GPU, <200 ms on CPU (100 ms window)

---

## Output Heatmap Specification

| Dimension | Range | Resolution | Bins |
|-----------|-------|------------|------|
| Azimuth | -180 to +175 deg | 5 deg | 72 |
| Elevation | -90 to +90 deg | 5 deg | 37 |

Values are normalized to **[0, 1]** — brighter = higher acoustic energy.

---

## Input Preprocessing

| Parameter | Value |
|-----------|-------|
| Sample Rate | 44,100 Hz |
| FFT Size | 512 |
| Hop Length | 256 |
| Window | Hann |
| Window Duration | 100 ms |
| Overlap | 75% |

Features per window:
- **IPD** (Inter-aural Phase Difference): `angle(L x R*) / pi` in [-1, 1]
- **ILD** (Inter-aural Level Difference): `tanh(20 log10(|L|/|R|) / 20)` in ~[-1, 1]

---

## Dataset Details

- **Total samples:** 30,000
- **Source counts:** 1 (20%), 2 (33%), 3 (33%), 4 (10%), 5+ (4%)
- **HRTF files:** 35 IRCAM Listen SOFA files, randomly selected per sample
- **Augmentation:** pitch shift +-2 semitones, time stretch x0.9-1.1, volume +-3 dB
- **Train/Val split:** 80/20 (seed=42)

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| 2-source angular error | < 10 deg |
| Inference latency (GPU) | < 100 ms |
| Real-time streaming | Supported |
| SNR >= 10 dB performance | Stable |

---

## Reproducibility

```bash
# All random seeds fixed to 42
python train.py
```

Model config is saved to `checkpoints/model_config.json` for exact reconstruction.
