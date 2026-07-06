"""
run_classifier_experiment.py
============================
Trains and compares two facial expression classifiers:

  Baseline              — plain images, no uncertainty weighting
  Uncertainty-weighted  — images multiplied by a 2D confidence mask derived from
                          per-vertex FLAME uncertainty (TTA by default)

Both models use the same ResNet backbone and are trained from the same pre-trained
checkpoint.  The only difference is the input distribution: the uncertainty-weighted
model sees occluded / low-confidence regions suppressed at the pixel level.

Usage
-----
    # Full comparison (requires pre-computed confidence maps)
    python -m src.run_classifier_experiment \\
        --data_root       ./data/affectnet \\
        --uncertainty_root ./data/affectnet_confidence_maps \\
        --epochs 20 --batch_size 32

    # Baseline only
    python -m src.run_classifier_experiment \\
        --data_root ./data/affectnet \\
        --epochs 20

Output
------
Best model checkpoints and a CSV summary table are written to --output_dir.
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.downstream import UncertaintyWeightedClassifier, EXPRESSION_CLASSES
from src.emotion_dataset import EmotionDataset


@dataclass
class ExperimentConfig:
    """Bundles every knob needed to run one train+eval comparison (see main())."""
    data_root: str
    uncertainty_root: Optional[str] = None
    output_dir: str = './outputs/classifier'
    backbone: str = 'resnet18'
    pretrained: bool = True
    fusion_mode: str = 'input'
    num_classes: int = 7
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def _build_loader(
    config: ExperimentConfig,
    split: str,
    mode: str,
) -> DataLoader:
    """Build the train/test DataLoader for one split and dataset `mode`."""
    dataset = EmotionDataset(
        root=config.data_root,
        split=split,
        mode=mode,
        uncertainty_root=config.uncertainty_root if mode == 'uncertainty_weighted' else None,
        image_size=config.image_size,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == 'train'),
        num_workers=config.num_workers,
        pin_memory=(config.device == 'cuda'),
        drop_last=(split == 'train'),
    )


def _build_model(config: ExperimentConfig) -> UncertaintyWeightedClassifier:
    """Instantiate the CNN classifier with the fusion mode from `config`."""
    return UncertaintyWeightedClassifier(
        num_classes=config.num_classes,
        architecture_type='CNN',
        backbone=config.backbone,
        pretrained=config.pretrained,
        fusion_mode=config.fusion_mode,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Run one training epoch; returns the mean per-sample cross-entropy loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict:
    """Evaluate `model` on `loader`; returns overall and per-class accuracy."""
    model.eval()
    correct = 0
    total   = 0
    per_class_correct = [0] * len(EXPRESSION_CLASSES)
    per_class_total   = [0] * len(EXPRESSION_CLASSES)

    for images, labels in loader:
        images     = images.to(device)
        preds      = model(images).argmax(dim=1).cpu()
        correct   += (preds == labels).sum().item()
        total     += labels.size(0)
        for c in range(len(EXPRESSION_CLASSES)):
            mask = (labels == c)
            per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            per_class_total[c]   += mask.sum().item()

    per_class = {
        EXPRESSION_CLASSES[c]: per_class_correct[c] / max(per_class_total[c], 1)
        for c in range(len(EXPRESSION_CLASSES))
    }
    return {'accuracy': correct / max(total, 1), 'per_class': per_class}


def _train_mode(config: ExperimentConfig, mode: str) -> Dict:
    """Full train + eval loop for one mode.  Returns best-epoch metrics."""
    print(f"\n{'='*62}")
    print(f"  Training mode: {mode}")
    print(f"{'='*62}")

    train_loader = _build_loader(config, 'train', mode)
    test_loader  = _build_loader(config, 'test',  mode)

    model     = _build_model(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=config.epochs)
    best_acc   = 0.0
    best_ckpt  = os.path.join(config.output_dir, f'best_{mode}.pth')
    best_metrics: Dict = {}

    for epoch in range(1, config.epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, config.device)
        metrics    = evaluate(model, test_loader, config.device)
        scheduler.step()

        acc = metrics['accuracy']
        if acc > best_acc:
            best_acc     = acc
            best_metrics = metrics
            torch.save(model.state_dict(), best_ckpt)

        print(
            f"  Epoch {epoch:3d}/{config.epochs}  "
            f"loss={train_loss:.4f}  acc={acc:.4f}  "
            f"best={best_acc:.4f}  ({time.time()-t0:.1f}s)"
        )

    print(f"\n  Best accuracy ({mode}): {best_acc:.4f}")
    return {'best_accuracy': best_acc, **best_metrics}


def _print_comparison(results: Dict[str, Dict]) -> None:
    """Print the plain-vs-uncertainty-weighted accuracy delta, overall and per-class."""
    if 'plain' not in results or 'uncertainty_weighted' not in results:
        return
    base  = results['plain']['best_accuracy']
    uw    = results['uncertainty_weighted']['best_accuracy']
    delta = uw - base
    print(f"\n{'='*62}")
    print(f"  Results Summary")
    print(f"{'='*62}")
    print(f"  Baseline (plain)            : {base:.4f}")
    print(f"  Uncertainty-weighted (ours) : {uw:.4f}")
    print(f"  Delta                       : {delta:+.4f}")
    print(f"\n  Per-class accuracy comparison:")
    print(f"  {'Class':10s}  {'Baseline':>8s}  {'UW':>8s}  {'Delta':>8s}")
    print(f"  {'-'*42}")
    for cls in EXPRESSION_CLASSES:
        b = results['plain']['per_class'].get(cls, 0.0)
        u = results['uncertainty_weighted']['per_class'].get(cls, 0.0)
        print(f"  {cls:10s}  {b:8.4f}  {u:8.4f}  {u-b:+8.4f}")


def _save_csv(results: Dict[str, Dict], output_dir: str) -> None:
    """Write one row per mode (best accuracy + per-class accuracy) to results.csv."""
    csv_path = os.path.join(output_dir, 'results.csv')
    rows = []
    for mode, m in results.items():
        row = {'mode': mode, 'best_accuracy': m['best_accuracy']}
        for cls, acc in m.get('per_class', {}).items():
            row[f'acc_{cls}'] = acc
        rows.append(row)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results saved to {csv_path}")


def run_experiment(config: ExperimentConfig) -> Dict[str, Dict]:
    """
    Train and evaluate baseline and uncertainty-weighted classifiers.

    Skips 'uncertainty_weighted' if config.uncertainty_root is None.
    Returns a dict keyed by mode name with best-epoch metrics.
    """
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}

    results['plain'] = _train_mode(config, 'plain')

    if config.uncertainty_root is not None:
        results['uncertainty_weighted'] = _train_mode(config, 'uncertainty_weighted')
    else:
        print(
            "\nSkipping 'uncertainty_weighted' — no --uncertainty_root provided.\n"
            "Run src/precompute_uncertainty_maps.py first, then re-run with "
            "--uncertainty_root pointing to the output directory."
        )

    _print_comparison(results)
    _save_csv(results, config.output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Uncertainty-Weighted Expression Classifier Experiment'
    )
    parser.add_argument('--data_root',        required=True)
    parser.add_argument('--uncertainty_root', default=None)
    parser.add_argument('--output_dir',       default='./outputs/classifier')
    parser.add_argument('--backbone',         default='resnet18')
    parser.add_argument('--epochs',           type=int,   default=20)
    parser.add_argument('--batch_size',       type=int,   default=32)
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--weight_decay',     type=float, default=1e-4)
    parser.add_argument('--image_size',       type=int,   default=224)
    parser.add_argument('--num_workers',      type=int,   default=4)
    parser.add_argument('--fusion_mode',      default='input',
                        choices=['input', 'feature'])
    parser.add_argument('--no_pretrained',    action='store_true')
    args = parser.parse_args()

    config = ExperimentConfig(
        data_root=args.data_root,
        uncertainty_root=args.uncertainty_root,
        output_dir=args.output_dir,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        fusion_mode=args.fusion_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    run_experiment(config)


if __name__ == '__main__':
    main()
