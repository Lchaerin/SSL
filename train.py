"""
Training script for BinauralSSLNet.

Usage:
    python train.py [--batch_size 32] [--epochs 100] [--lr 1e-3]
                    [--data_dir ./data/generated] [--out_dir ./checkpoints]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from models import BinauralSSLNet, MODEL_CONFIG
from utils.audio_processing import compute_ipd_ild, FFT_SIZE, HOP_LENGTH, WINDOW_SAMPLES
from utils.heatmap_generator import compute_peak_position, angular_error

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'generated'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
LOG_DIR = BASE_DIR / 'logs'

RANDOM_SEED = 42


# ── Dataset ───────────────────────────────────────────────────────────────────

TARGET_T = 1 + (WINDOW_SAMPLES - FFT_SIZE) // HOP_LENGTH  # expected time frames


class BinauralSSLDataset(Dataset):
    """
    Loads binaural WAV files and ground-truth heatmap npy files.
    IPD/ILD features are computed on-the-fly in __getitem__.

    audio/XXXXXX.wav     →  [2, WINDOW_SAMPLES]  binaural audio
    heatmaps/XXXXXX.npy  →  [72, 37]             ground-truth heatmap
    """

    def __init__(self, data_dir: str, indices: list):
        self.audio_dir   = Path(data_dir) / 'audio'
        self.heatmaps_dir = Path(data_dir) / 'heatmaps'
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        # Load binaural audio  [samples, 2] → transpose → [2, samples]
        audio, _ = sf.read(self.audio_dir / f'{idx:06d}.wav', dtype='float32',
                           always_2d=True)
        audio = audio.T  # [2, samples]

        # Compute IPD/ILD on-the-fly  →  [2, freq_bins, time_frames]
        feat = compute_ipd_ild(audio)
        T = feat.shape[2]
        if T < TARGET_T:
            feat = np.pad(feat, ((0, 0), (0, 0), (0, TARGET_T - T)))
        else:
            feat = feat[:, :, :TARGET_T]

        hmap = np.load(self.heatmaps_dir / f'{idx:06d}.npy')    # [72, 37]
        return torch.from_numpy(feat), torch.from_numpy(hmap)


def build_datasets(
    data_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    """
    Discover valid samples and split into train / val / test sets.

    Default split: 70% train, 15% val, 15% test.
    Returns:
        (train_ds, val_ds, test_ds)
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    audio_dir = Path(data_dir) / 'audio'
    all_indices = sorted([
        int(p.stem) for p in audio_dir.glob('*.wav')
        if (Path(data_dir) / 'heatmaps' / p.with_suffix('.npy').name).exists()
    ])
    if not all_indices:
        raise FileNotFoundError(
            f"No audio/heatmap pairs found in {data_dir}. "
            "Run data_generation.py first."
        )

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(all_indices := np.array(all_indices))
    n = len(all_indices)
    split_train = int(n * train_ratio)
    split_val   = int(n * (train_ratio + val_ratio))

    train_ds = BinauralSSLDataset(data_dir, all_indices[:split_train].tolist())
    val_ds   = BinauralSSLDataset(data_dir, all_indices[split_train:split_val].tolist())
    test_ds  = BinauralSSLDataset(data_dir, all_indices[split_val:].tolist())
    return train_ds, val_ds, test_ds


# ── Loss ──────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """MSE + cosine-similarity + KL-divergence loss for heatmap regression.

    KL divergence treats heatmaps as probability distributions, directly
    penalising peak displacement and improving angular accuracy.
    """

    def __init__(self, mse_weight: float = 1.0, cos_weight: float = 0.1,
                 kl_weight: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.cos_weight = cos_weight
        self.kl_weight = kl_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: [B, 72, 37]
        loss = self.mse_weight * self.mse(pred, target)

        p_flat = pred.view(pred.shape[0], -1)
        t_flat = target.view(target.shape[0], -1)

        if self.cos_weight > 0:
            cos_sim = nn.functional.cosine_similarity(p_flat, t_flat, dim=1)
            loss = loss + self.cos_weight * (1.0 - cos_sim.mean())

        if self.kl_weight > 0:
            # Normalise to probability distributions, then compute KL(target || pred)
            p_prob = p_flat / (p_flat.sum(dim=1, keepdim=True) + 1e-8)
            t_prob = t_flat / (t_flat.sum(dim=1, keepdim=True) + 1e-8)
            kl = nn.functional.kl_div(
                torch.log(p_prob + 1e-8), t_prob, reduction='batchmean')
            loss = loss + self.kl_weight * kl

        return loss


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute MSE, MAE, and peak angular error over a batch."""
    mse = nn.functional.mse_loss(pred, target).item()
    mae = nn.functional.l1_loss(pred, target).item()

    ang_errors = []
    for p, t in zip(pred.cpu().numpy(), target.cpu().numpy()):
        p_az, p_el = compute_peak_position(p)
        t_az, t_el = compute_peak_position(t)
        ang_errors.append(angular_error(p_az, p_el, t_az, t_el))

    return {
        'mse': mse,
        'mae': mae,
        'peak_angular_error': float(np.mean(ang_errors)),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Datasets
    train_ds, val_ds, test_ds = build_datasets(
        args.data_dir, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == 'cuda'),
        prefetch_factor=2 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == 'cuda'),
        prefetch_factor=2 if args.workers > 0 else None,
    )

    # Model
    model = BinauralSSLNet().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = CombinedLoss()

    # Output dirs
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(LOG_DIR))

    # Save model config
    model.save_config(str(CHECKPOINT_DIR / 'model_config.json'))

    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch_feat, batch_hmap in train_loader:
            batch_feat = batch_feat.to(device, non_blocking=True)
            batch_hmap = batch_hmap.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_feat)
            loss = criterion(pred, batch_hmap)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            global_step += 1

            # ── Step checkpoint ────────────────────────────────────────────
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                step_ckpt = CHECKPOINT_DIR / f'step_{global_step:07d}.pth'
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, step_ckpt)
                print(f"\n  [step {global_step}] checkpoint saved → {step_ckpt.name}")

        train_loss /= len(train_loader)
        train_time = time.time() - t0

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_metrics = {'mse': 0.0, 'mae': 0.0, 'peak_angular_error': 0.0}

        with torch.no_grad():
            for batch_feat, batch_hmap in val_loader:
                batch_feat = batch_feat.to(device, non_blocking=True)
                batch_hmap = batch_hmap.to(device, non_blocking=True)
                pred = model(batch_feat)
                val_loss += criterion(pred, batch_hmap).item()
                m = compute_metrics(pred, batch_hmap)
                for k in val_metrics:
                    val_metrics[k] += m[k]

        val_loss /= len(val_loader)
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        # ── Logging ────────────────────────────────────────────────────────
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', lr, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'Val/{k}', v, epoch)

        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"ang_err={val_metrics['peak_angular_error']:.1f}°  "
            f"lr={lr:.2e}  t={train_time:.1f}s"
        )

        # ── Checkpointing ──────────────────────────────────────────────────
        state = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
        }
        torch.save(state, CHECKPOINT_DIR / 'model_last.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(state, CHECKPOINT_DIR / 'model_best.pth')
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        # Log sample heatmaps every 10 epochs
        if epoch % 10 == 0:
            _log_sample_heatmaps(writer, model, val_loader, device, epoch)

    writer.close()
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")


def _log_sample_heatmaps(writer, model, loader, device, epoch, n=4):
    """Log predicted vs. ground-truth heatmap images to TensorBoard."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    import torchvision

    model.eval()
    feat, hmap = next(iter(loader))
    feat = feat[:n].to(device)
    hmap = hmap[:n]
    with torch.no_grad():
        pred = model(feat).cpu()

    fig, axes = plt.subplots(n, 2, figsize=(8, n * 2))
    for i in range(n):
        axes[i, 0].imshow(hmap[i].T, origin='lower', aspect='auto', cmap='hot')
        axes[i, 0].set_title('Ground Truth')
        axes[i, 1].imshow(pred[i].T, origin='lower', aspect='auto', cmap='hot')
        axes[i, 1].set_title('Predicted')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80)
    buf.seek(0)
    img = np.array(Image.open(buf)).transpose(2, 0, 1)[:3]  # CHW
    writer.add_image('Heatmaps/comparison', img, epoch)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BinauralSSLNet')
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR))
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--train_ratio', type=float, default=0.70,
                        help='Fraction of data for training (default: 0.70)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Fraction of data for validation (default: 0.15)')
    parser.add_argument('--save_steps', type=int, default=0,
                        help='Save a step checkpoint every N steps (0 to disable)')
    args = parser.parse_args()

    train(args)
