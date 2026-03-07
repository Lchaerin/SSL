"""
Evaluation script for BinauralSSLNet.

Computes per-condition metrics:
  - MSE / MAE on heatmap values
  - Peak localization angular error (degrees)
  - Breakdown by azimuth region, elevation region, n_sources, SNR

Usage:
    python evaluate.py --checkpoint ./checkpoints/model_best.pth
                       [--data_dir ./data/generated]
                       [--batch_size 64]
                       [--out ./evaluation_report.md]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import BinauralSSLNet
from train import BinauralSSLDataset, build_datasets
from utils.heatmap_generator import compute_peak_position, angular_error

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'generated'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'


# ── Condition classifiers ─────────────────────────────────────────────────────

def azimuth_region(az: float) -> str:
    az = abs(az)
    if az <= 30:
        return 'front (0-30°)'
    elif az <= 90:
        return 'side (30-90°)'
    elif az <= 150:
        return 'rear (90-150°)'
    else:
        return 'back (150-180°)'


def elevation_region(el: float) -> str:
    if -10 <= el <= 10:
        return 'horizontal (-10~+10°)'
    elif el > 10:
        return 'upper (+10~+45°)'
    else:
        return 'lower (-10~-45°)'


def snr_region(db: float) -> str:
    if db >= 20:
        return 'high SNR (20~30 dB)'
    elif db >= 10:
        return 'medium SNR (10~20 dB)'
    elif db >= 0:
        return 'low SNR (0~10 dB)'
    else:
        return 'very low SNR (<0 dB)'


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    config_path = str(Path(args.checkpoint).parent / 'model_config.json')
    model = BinauralSSLNet.load(args.checkpoint, config_path, device=str(device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load test set (same split ratios used during training)
    _, _, test_ds = build_datasets(
        args.data_dir, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    print(f"Test samples: {len(test_ds):,}")

    val_loader = DataLoader(test_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    # Load metadata for condition breakdown
    meta_path = Path(args.data_dir) / 'metadata.json'
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            for entry in json.load(f):
                if 'error' not in entry:
                    metadata[entry['id']] = entry

    # ── Collect predictions ───────────────────────────────────────────────
    all_mse, all_mae, all_ang_err = [], [], []
    cond_ang_err: dict = defaultdict(list)
    cond_mse: dict = defaultdict(list)

    for batch_feat, batch_hmap in tqdm(val_loader, desc='Evaluating'):
        batch_feat = batch_feat.to(device)
        pred = model(batch_feat).cpu()
        target = batch_hmap

        mse_batch = nn.functional.mse_loss(pred, target, reduction='none') \
                      .mean(dim=(1, 2)).numpy()
        mae_batch = nn.functional.l1_loss(pred, target, reduction='none') \
                      .mean(dim=(1, 2)).numpy()

        all_mse.extend(mse_batch.tolist())
        all_mae.extend(mae_batch.tolist())

        for p, t in zip(pred.numpy(), target.numpy()):
            p_az, p_el = compute_peak_position(p)
            t_az, t_el = compute_peak_position(t)
            err = angular_error(p_az, p_el, t_az, t_el)
            all_ang_err.append(err)

    # ── Per-condition breakdown from metadata ─────────────────────────────
    val_indices = test_ds.indices
    for i, idx in enumerate(val_indices):
        if idx not in metadata:
            continue
        meta = metadata[idx]
        ang_err = all_ang_err[i]
        mse_val = all_mse[i]

        n_src = meta.get('n_sources', 1)
        cond_ang_err[f'n_sources={n_src}'].append(ang_err)
        cond_mse[f'n_sources={n_src}'].append(mse_val)

        for src in meta.get('sources', []):
            az = src['azimuth']
            el = src['elevation']
            db = src['db']
            cond_ang_err[azimuth_region(az)].append(ang_err)
            cond_ang_err[elevation_region(el)].append(ang_err)
            cond_ang_err[snr_region(db)].append(ang_err)

    # ── Build report ──────────────────────────────────────────────────────
    lines = []
    lines.append('# Binaural SSL Evaluation Report\n')
    lines.append(f'Checkpoint: `{args.checkpoint}`\n')
    lines.append(f'Test samples: {len(all_mse):,}\n')
    lines.append('')

    lines.append('## Overall Metrics\n')
    lines.append(f'| Metric | Value |')
    lines.append(f'|--------|-------|')
    lines.append(f'| MSE    | {np.mean(all_mse):.5f} |')
    lines.append(f'| MAE    | {np.mean(all_mae):.5f} |')
    lines.append(f'| Peak Angular Error (mean) | {np.mean(all_ang_err):.2f}° |')
    lines.append(f'| Peak Angular Error (median) | {np.median(all_ang_err):.2f}° |')
    lines.append(f'| Angular Error < 5° | {np.mean(np.array(all_ang_err) < 5):.1%} |')
    lines.append(f'| Angular Error < 10° | {np.mean(np.array(all_ang_err) < 10):.1%} |')
    lines.append('')

    lines.append('## Per-Condition Angular Error\n')
    lines.append('| Condition | Mean Error | Median Error | N |')
    lines.append('|-----------|-----------|--------------|---|')
    for cond, errs in sorted(cond_ang_err.items()):
        lines.append(
            f'| {cond} | {np.mean(errs):.2f}° | {np.median(errs):.2f}° | {len(errs)} |'
        )
    lines.append('')

    report_text = '\n'.join(lines)
    print('\n' + report_text)

    if args.out:
        Path(args.out).write_text(report_text)
        print(f"Report saved to: {args.out}")

    # ── Summary for pass/fail criteria ───────────────────────────────────
    mean_err_2src = np.mean(cond_ang_err.get('n_sources=2', [float('nan')]))
    print(f"\nSuccess criterion (2-source angular error < 10°): ", end='')
    if np.isnan(mean_err_2src):
        print("N/A (no 2-source samples in test set)")
    elif mean_err_2src < 10.0:
        print(f"PASS ({mean_err_2src:.2f}°)")
    else:
        print(f"FAIL ({mean_err_2src:.2f}°)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate BinauralSSLNet')
    parser.add_argument('--checkpoint', type=str,
                        default=str(CHECKPOINT_DIR / 'model_best.pth'))
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_ratio', type=float, default=0.70,
                        help='Must match the ratio used during training (default: 0.70)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Must match the ratio used during training (default: 0.15)')
    parser.add_argument('--out', type=str, default='evaluation_report.md',
                        help='Path to save the markdown report')
    args = parser.parse_args()
    evaluate(args)
