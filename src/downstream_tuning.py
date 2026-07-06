"""
src/downstream_tuning.py
Hyperparameter tuning for the downstream expression classifier.

Design
------
Two sequential phases eliminate confounding:

  Phase 1 — training-config search (plain mode, no uncertainty maps).
    Find the best hyperparameter set for this dataset/backbone combination.
    CPU path: backbone features are pre-cached once; only the head is optimised
              per trial (~50 ms/trial). With n_trials=100 the search completes
              in ~10 seconds total.
    GPU path: two-stage fine-tuning (frozen warmup → full fine-tune) sweeps
              all parameters in a single search (n_trials=40 ≈ 20–40 min).

  Phase 2 — uncertainty-source comparison (uses Phase-1 best config).
    For every available uncertainty method tests two training modes:
      'uncertainty_weighted' — image multiplied by confidence map before backbone
      'loss_weighted'        — plain image with per-sample CE loss scaled by
                               mean image confidence (no pixel modification)
    Reports ΔACC = mode_acc − plain_acc for each method and mode.
    Each (config, mode, method) is evaluated across n_seeds for mean and std.

Search strategies
-----------------
  search='auto'   : try Optuna TPE; fall back to log-uniform random search.
  search='tpe'    : Optuna Tree-structured Parzen Estimator (requires optuna).
  search='random' : log-uniform random sampling (no extra dependencies).
  search='grid'   : exhaustive 96-point grid (original behaviour, CPU only).

Tunable parameters (random / TPE)
-----------------------------------
  lr              : log-uniform  [1e-5, 1e-3]
  weight_decay    : log-uniform  [1e-5, 1e-2]
  head_dropout    : uniform      [0.10, 0.70]
  label_smoothing : uniform      [0.00, 0.20]
  head_arch       : categorical  ['linear', 'mlp']
  head_hidden_dim : categorical  [64, 128, 256]  (MLP only)
  optimizer       : categorical  ['adamw', 'sgd']  (SGD uses momentum=0.9+Nesterov)

Notes on loss_weighted vs uncertainty_weighted (CPU path)
---------------------------------------------------------
For 'uncertainty_weighted', backbone features differ from plain (the input image
is multiplied by the confidence map before the frozen backbone), so features are
re-cached per method.

For 'loss_weighted', the input image is UNCHANGED — only the training loss is
scaled by the mean confidence.  Backbone features are therefore identical to the
plain cache; only the per-sample weights differ.  These scalar weights are
extracted by reading the .npy confidence files directly (fast, no GPU).

Estimated wall times
--------------------
  CPU (feature-cached, n_trials=100)  :  ~10 min total (all phases)
  GPU (full fine-tune, n_trials=40)   :  ~25–45 min total

Entry points
------------
  tune_downstream_cpu(raf_root, maps_dict, output_dir, n_trials=100, search='auto')
  tune_downstream_gpu(raf_root, maps_dict, output_dir, n_trials=40,  search='auto')

  maps_dict : dict[method_name → maps_root_path] or None
      Pre-computed confidence-map directories from Stage-5 of main.py.
      If None, Phase 2 is skipped.

  Both functions write results to output_dir/results.json and return the same dict.
"""

from __future__ import annotations

import io
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPRESSION_CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
_N_CLS    = len(EXPRESSION_CLASSES)
_IMG_SIZE = 224
_MEAN     = [0.485, 0.456, 0.406]
_STD      = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Hyperparameter grid (kept for search='grid' backward-compat, CPU only)
# ---------------------------------------------------------------------------

_CPU_P1_GRID: List[Dict] = [
    {
        'lr':              lr,
        'weight_decay':    wd,
        'head_dropout':    do,
        'label_smoothing': ls,
        'head_arch':       arch,
        'head_hidden_dim': 128,
        'optimizer':       'adamw',
    }
    for lr   in [5e-5, 1e-4, 2e-4, 5e-4]
    for wd   in [1e-4, 5e-4, 2e-3]
    for do   in [0.2, 0.5]
    for arch in ['linear', 'mlp']
    for ls   in [0.05, 0.15]
]

# Phase 2 modes tested per uncertainty method.
_P2_MODES = ['uncertainty_weighted', 'loss_weighted']

# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def _val_transform():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((_IMG_SIZE, _IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])


def _train_transform():
    import torchvision.transforms as T
    return T.Compose([
        T.RandomResizedCrop(_IMG_SIZE, scale=(0.80, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])

# ---------------------------------------------------------------------------
# Shared dataset helpers
# ---------------------------------------------------------------------------

def _stratify(ds, n_per_class: int) -> Subset:
    """Cap each class to n_per_class samples (deterministic: first N per class)."""
    buckets: Dict[int, List[int]] = defaultdict(list)
    for i, (_, lbl) in enumerate(ds.samples):
        buckets[lbl].append(i)
    idxs: List[int] = []
    for lbl in range(_N_CLS):
        idxs.extend(buckets[lbl][:n_per_class])
    return Subset(ds, sorted(idxs))


def _class_weights_from_subset(subset: Subset) -> torch.Tensor:
    """Inverse-frequency class weights from a Subset, normalised to sum to N_CLS."""
    counts = np.zeros(_N_CLS, dtype=np.float32)
    for i in subset.indices:
        _, lbl = subset.dataset.samples[i]
        counts[lbl] += 1.0
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    w = w / w.sum() * _N_CLS
    return torch.tensor(w, dtype=torch.float32)


def _macro_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    f1s = []
    for c in range(_N_CLS):
        tp = float(((preds == c) & (labels == c)).sum())
        fp = float(((preds == c) & (labels != c)).sum())
        fn = float(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    return float(np.mean(f1s))


def _per_class_acc(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    acc = {}
    for c, name in enumerate(EXPRESSION_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            acc[name] = float('nan')
        else:
            acc[name] = round(100.0 * float((preds[mask] == c).float().mean()), 2)
    return acc

# ---------------------------------------------------------------------------
# Head factory (CPU path)
# ---------------------------------------------------------------------------

def _make_head(feature_dim: int,
               head_arch: str = 'mlp',
               dropout: float = 0.5,
               hidden_dim: int = 128) -> nn.Module:
    """
    'linear' — LayerNorm → Dropout → Linear(D, 7)                [~5 K params for D=768]
    'mlp'    — LayerNorm → Linear(D, hidden_dim) → GELU → Dropout → Linear(hidden_dim, 7)
    With 175 training samples, 'linear' typically outperforms 'mlp' due to
    far fewer parameters and less overfitting risk.
    """
    if head_arch == 'linear':
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, _N_CLS),
        )
    return nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, _N_CLS),
    )

# ---------------------------------------------------------------------------
# CPU: backbone feature extraction
# ---------------------------------------------------------------------------

def _build_frozen_backbone(backbone_name: str, device: str):
    """
    Returns (backbone_module, pool_or_None, feature_dim).
    The backbone is frozen, on device, and in eval mode.

    ViT  classifier layout: LayerNorm[0] → Linear[1] → GELU[2] → Dropout[3] → Linear[4]
    ResNet classifier layout: Flatten[0] → Dropout[1] → Linear[2]
    """
    from src.downstream import UncertaintyWeightedClassifier
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = UncertaintyWeightedClassifier(
            num_classes=_N_CLS,
            backbone=backbone_name,
            pretrained=True,
        )
    backbone = clf.backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    if clf._backbone_type == 'vit':
        feature_dim = clf.classifier[1].in_features   # Linear[1].in_features
        return backbone, None, feature_dim
    else:
        feature_dim = clf.classifier[2].in_features   # Linear[2].in_features
        pool = clf.pool.to(device).eval()
        for p in pool.parameters():
            p.requires_grad_(False)
        return backbone, pool, feature_dim


def _extract_features(
    backbone: nn.Module,
    pool: Optional[nn.Module],
    loader: DataLoader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract frozen backbone features for all images in loader.
    Returns (features: (N,D), labels: (N,)) both on CPU.
    Handles 2-item (plain / uncertainty_weighted) and 3-item (loss_weighted) batches.
    """
    backbone.eval()
    all_feats:  List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            imgs   = batch[0].to(device)
            labels = batch[1]
            feats  = backbone(imgs)
            if pool is not None:
                feats = pool(feats).view(feats.size(0), -1)
            all_feats.append(feats.cpu())
            all_labels.append(
                labels.cpu() if torch.is_tensor(labels) else torch.tensor(labels)
            )

    return torch.cat(all_feats, 0), torch.cat(all_labels, 0)


def _extract_loss_weights(subset: Subset, maps_root: str) -> torch.Tensor:
    """
    For loss_weighted mode on CPU: read pre-computed .npy confidence maps for
    each training sample and return per-sample 10th-percentile confidence as a
    float tensor.

    The raw .npy maps are stored at their original image resolution (H×W).
    Feature caching uses _val_transform() which resizes to (_IMG_SIZE, _IMG_SIZE)
    with no random crop.  To make the weight signal consistent with what the
    backbone actually saw, we resize the confidence map to _IMG_SIZE×_IMG_SIZE
    before computing the percentile.

    Note: actual training (EmotionDataset + _train_one_variant) uses RandomResized-
    Crop augmentation and computes the percentile over the crop-aligned map.  The
    CPU tuning path cannot replicate per-crop weights (features are pre-cached once
    in val-transform mode), so a small signal mismatch is inherent in the design.

    Missing maps default to weight 1.0 (no downweighting).
    """
    maps_root = Path(maps_root)
    weights: List[float] = []
    warned = False

    for i in subset.indices:
        img_path, _ = subset.dataset.samples[i]
        img_path = Path(img_path)
        rel      = img_path.relative_to(subset.dataset.root)
        npy_path = (maps_root / rel).with_suffix('.npy')
        if npy_path.exists():
            conf = np.load(str(npy_path)).astype(np.float32)
            # Resize to the standard processing size so the weight is computed
            # on the same spatial region as the cached backbone features.
            if conf.shape != (_IMG_SIZE, _IMG_SIZE):
                conf_t = torch.from_numpy(conf).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                conf_t = torch.nn.functional.interpolate(
                    conf_t, size=(_IMG_SIZE, _IMG_SIZE),
                    mode='bilinear', align_corners=False,
                )
                conf = conf_t.squeeze().numpy()
            weights.append(float(np.percentile(conf, 10)))
        else:
            if not warned:
                warnings.warn(
                    f"Confidence map not found for loss_weighted: {npy_path}. "
                    "Using weight=1.0 for missing samples.",
                    UserWarning, stacklevel=2,
                )
                warned = True
            weights.append(1.0)

    return torch.tensor(weights, dtype=torch.float32)

# ---------------------------------------------------------------------------
# CPU: head training on cached features
# ---------------------------------------------------------------------------

def _train_head(
    feats_tr: torch.Tensor,
    lbls_tr:  torch.Tensor,
    feats_te: torch.Tensor,
    lbls_te:  torch.Tensor,
    feature_dim: int,
    lr: float,
    weight_decay: float,
    head_dropout: float,
    head_arch: str,
    epochs: int,
    patience: int,
    label_smoothing: float,
    device: str,
    seed: int,
    sample_weights: Optional[torch.Tensor] = None,
    hidden_dim: int = 128,
    optimizer_type: str = 'adamw',
) -> Dict[str, Any]:
    """
    Train a standalone head on pre-cached backbone features.

    sample_weights  : (N_train,) float tensor for loss_weighted mode, or None.
    hidden_dim      : intermediate size for 'mlp' head (ignored for 'linear').
    optimizer_type  : 'adamw' or 'sgd' (SGD uses momentum=0.9, Nesterov=True).
    """
    torch.manual_seed(seed)
    head = _make_head(feature_dim, head_arch, head_dropout, hidden_dim).to(device)

    # Inverse-frequency class weights.
    counts = torch.zeros(_N_CLS)
    for lbl in lbls_tr:
        counts[int(lbl)] += 1.0
    counts = counts.clamp(min=1.0)
    cls_w  = ((1.0 / counts) / (1.0 / counts).sum() * _N_CLS).to(device)

    criterion_mean = nn.CrossEntropyLoss(
        weight=cls_w, label_smoothing=label_smoothing, reduction='mean'
    )
    criterion_none = nn.CrossEntropyLoss(
        weight=cls_w, label_smoothing=label_smoothing, reduction='none'
    )
    use_sw = sample_weights is not None

    if optimizer_type == 'sgd':
        opt = torch.optim.SGD(
            head.parameters(), lr=lr, momentum=0.9,
            weight_decay=weight_decay, nesterov=True,
        )
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.01
    )

    gen = torch.Generator()
    gen.manual_seed(seed)
    ds = (TensorDataset(feats_tr, lbls_tr, sample_weights)
          if use_sw else TensorDataset(feats_tr, lbls_tr))
    loader = DataLoader(ds, batch_size=64, shuffle=True, generator=gen)

    best_acc   = 0.0
    best_f1    = 0.0
    best_pca:  Dict = {}
    no_improve = 0
    epoch_log: List[Dict] = []

    for ep in range(1, epochs + 1):
        head.train()
        for batch in loader:
            opt.zero_grad()
            if use_sw:
                xb, yb, wb = (batch[0].to(device), batch[1].to(device),
                               batch[2].to(device))
                logits = head(xb)
                loss   = (criterion_none(logits, yb) * wb).mean()
            else:
                xb, yb = batch[0].to(device), batch[1].to(device)
                logits = head(xb)
                loss   = criterion_mean(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        sch.step()

        head.eval()
        with torch.no_grad():
            preds = head(feats_te.to(device)).argmax(1).cpu()
        acc = round(100.0 * float((preds == lbls_te).float().mean()), 2)
        f1  = _macro_f1(preds, lbls_te)
        epoch_log.append({'epoch': ep, 'acc': acc, 'f1': f1})

        if acc > best_acc + 0.3:
            best_acc   = acc
            best_f1    = f1
            best_pca   = _per_class_acc(preds, lbls_te)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    return {
        'best_acc':      best_acc,
        'best_f1':       best_f1,
        'per_class_acc': best_pca,
        'n_epochs':      len(epoch_log),
        'epoch_log':     epoch_log,
    }

# ---------------------------------------------------------------------------
# GPU: two-stage full fine-tune
# ---------------------------------------------------------------------------

def _train_full(
    mode: str,
    raf_root: str,
    maps_root: Optional[str],
    backbone_name: str,
    lr: float,
    weight_decay: float,
    head_dropout: float,
    label_smoothing: float,
    n_subset: int,
    n_epochs: int,
    patience: int,
    device: str,
    seed: int,
    backbone_lr_ratio: float = 0.1,
) -> Dict[str, Any]:
    """
    Full two-stage fine-tune (frozen warmup → backbone unfreeze).
    Handles 'loss_weighted' 3-item batches from EmotionDataset.
    Test uses the same mode as training (consistent train/test distributions).
    For 'uncertainty_weighted', test images are attenuated by confidence maps —
    matching deployment.  For 'loss_weighted' and 'plain', test images are raw.

    backbone_lr_ratio : backbone LR = lr × backbone_lr_ratio in Stage 2.
    """
    from src.emotion_dataset import EmotionDataset
    from src.downstream import UncertaintyWeightedClassifier

    torch.manual_seed(seed)
    n_test = max(5, min(n_subset // 3, 50))

    try:
        tr_raw = EmotionDataset(raf_root, 'train', mode=mode,
                                uncertainty_root=maps_root,
                                transform=_train_transform())
        te_raw = EmotionDataset(raf_root, 'test',  mode=mode,
                                uncertainty_root=maps_root,
                                transform=_val_transform())
    except Exception as exc:
        return {'error': str(exc)}

    tr_sub = _stratify(tr_raw, n_subset)
    te_sub = _stratify(te_raw, n_test)

    pin = (device == 'cuda')
    tr_loader = DataLoader(tr_sub, batch_size=32, shuffle=True,
                           num_workers=0, pin_memory=pin)
    te_loader = DataLoader(te_sub, batch_size=64, shuffle=False,
                           num_workers=0, pin_memory=pin)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = UncertaintyWeightedClassifier(
            num_classes=_N_CLS, backbone=backbone_name, pretrained=True,
        ).to(device)

    # Override head dropout.
    if clf._backbone_type == 'vit':
        clf.classifier[3] = nn.Dropout(head_dropout)   # index 3: Dropout in ViT head
    else:
        clf.classifier[1] = nn.Dropout(head_dropout)   # index 1: Dropout in ResNet head

    cls_w = _class_weights_from_subset(tr_sub).to(device)
    crit_mean = nn.CrossEntropyLoss(
        weight=cls_w, label_smoothing=label_smoothing, reduction='mean'
    )
    crit_none = nn.CrossEntropyLoss(
        weight=cls_w, label_smoothing=label_smoothing, reduction='none'
    )
    is_loss_wt = (mode == 'loss_weighted')

    n_warmup   = max(1, n_epochs // 3)
    n_finetune = n_epochs - n_warmup

    def _run_epoch(opt: torch.optim.Optimizer) -> float:
        clf.train()
        n_correct = n_total = 0
        for batch in tr_loader:
            imgs = batch[0].to(device)
            lbls = batch[1].to(device)
            opt.zero_grad()
            logits = clf(image=imgs)
            if is_loss_wt and len(batch) >= 3:
                sw   = batch[2].to(device)
                loss = (crit_none(logits, lbls) * sw).mean()
            else:
                loss = crit_mean(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                n_correct += int((logits.detach().argmax(1) == lbls).sum())
            n_total += imgs.size(0)
        return 100.0 * n_correct / max(n_total, 1)

    def _eval() -> Tuple[float, float, Dict]:
        clf.eval()
        all_preds: List[torch.Tensor] = []
        all_lbls:  List[torch.Tensor] = []
        with torch.no_grad():
            for batch in te_loader:
                imgs = batch[0].to(device)
                lbls = batch[1]
                all_preds.append(clf(image=imgs).argmax(1).cpu())
                all_lbls.append(
                    lbls.cpu() if torch.is_tensor(lbls) else torch.tensor(lbls)
                )
        preds  = torch.cat(all_preds)
        labels = torch.cat(all_lbls)
        acc = round(100.0 * float((preds == labels).float().mean()), 2)
        return acc, _macro_f1(preds, labels), _per_class_acc(preds, labels)

    best_acc = 0.0; best_f1 = 0.0; best_pca: Dict = {}
    _s1_best_state: Optional[bytes] = None  # best Stage-1 model snapshot (in-memory)
    no_improve = 0; epoch_log: List[Dict] = []

    # Stage 1: frozen backbone → warm up head.
    for p in clf.backbone.parameters():
        p.requires_grad_(False)
    if hasattr(clf, 'pool') and clf.pool is not None:
        for p in clf.pool.parameters():
            p.requires_grad_(False)
    opt1 = torch.optim.AdamW(clf.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=n_warmup, eta_min=lr * 0.05)

    for ep in range(1, n_warmup + 1):
        tr_acc = _run_epoch(opt1)
        sch1.step()
        acc, f1, pca = _eval()
        epoch_log.append({'epoch': ep, 'stage': 1,
                          'train_acc': round(tr_acc, 2), 'test_acc': acc})
        if acc > best_acc + 0.3:
            best_acc = acc; best_f1 = f1; best_pca = pca; no_improve = 0
            _buf = io.BytesIO()
            torch.save(clf.state_dict(), _buf)
            _s1_best_state = _buf.getvalue()
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    # Stage 2: full fine-tune with backbone LR scaled by backbone_lr_ratio.
    # Skip only when Stage 1 made zero genuine progress (model still at or below
    # random-chance accuracy).  If Stage 1 peaked above random but then triggered
    # early stopping due to a post-peak plateau, Stage 2 is still worthwhile.
    if n_finetune > 0 and (no_improve < patience or best_acc > 100.0 / _N_CLS):
        # Reload the best Stage-1 state so Stage 2 starts from the peak, not from
        # the last (possibly degraded) epoch that triggered early stopping.
        if _s1_best_state is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                clf.load_state_dict(
                    torch.load(io.BytesIO(_s1_best_state), map_location=device)
                )
        for p in clf.backbone.parameters():
            p.requires_grad_(True)
        if hasattr(clf, 'pool') and clf.pool is not None:
            for p in clf.pool.parameters():
                p.requires_grad_(True)
        opt2 = torch.optim.AdamW([
            {'params': clf.backbone.parameters(),   'lr': lr * backbone_lr_ratio},
            {'params': clf.classifier.parameters(), 'lr': lr},
        ], weight_decay=weight_decay)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2, T_max=n_finetune, eta_min=lr * 0.005
        )
        no_improve = 0
        for ep in range(1, n_finetune + 1):
            tr_acc = _run_epoch(opt2)
            sch2.step()
            acc, f1, pca = _eval()
            epoch_log.append({'epoch': n_warmup + ep, 'stage': 2,
                              'train_acc': round(tr_acc, 2), 'test_acc': acc})
            if acc > best_acc + 0.3:
                best_acc = acc; best_f1 = f1; best_pca = pca; no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

    return {
        'best_acc':      best_acc,
        'best_f1':       best_f1,
        'per_class_acc': best_pca,
        'n_epochs':      len(epoch_log),
        'epoch_log':     epoch_log,
    }

# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def _try_optuna():
    """Return the optuna module or None if not installed."""
    try:
        import optuna
        return optuna
    except ImportError:
        return None


def _resolve_search(search: str) -> str:
    """Resolve 'auto' to 'tpe' or 'random' based on optuna availability."""
    if search != 'auto':
        return search
    return 'tpe' if _try_optuna() is not None else 'random'


def _suggest_cpu_config(src) -> Dict[str, Any]:
    """
    Generate one CPU head config from either an optuna Trial or a numpy RandomState.

    Covers:
      lr, weight_decay          — log-uniform [1e-5..1e-3] / [1e-5..1e-2]
      head_dropout              — uniform [0.10, 0.70]
      label_smoothing           — uniform [0.00, 0.20]
      head_arch                 — categorical ['linear', 'mlp']
      head_hidden_dim           — categorical [64, 128, 256]  (relevant for 'mlp')
      optimizer                 — categorical ['adamw', 'sgd']
    """
    optuna = _try_optuna()
    if optuna is not None and isinstance(src, optuna.Trial):
        t = src
        lr   = t.suggest_float('lr',           1e-5, 1e-3, log=True)
        wd   = t.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        do   = t.suggest_float('head_dropout',    0.10, 0.70)
        ls   = t.suggest_float('label_smoothing', 0.00, 0.20)
        arch = t.suggest_categorical('head_arch', ['linear', 'mlp'])
        hdim = (t.suggest_categorical('head_hidden_dim', [64, 128, 256])
                if arch == 'mlp' else 128)
        opt  = t.suggest_categorical('optimizer', ['adamw', 'sgd'])
    else:
        rng  = src
        lr   = float(np.exp(rng.uniform(np.log(1e-5), np.log(1e-3))))
        wd   = float(np.exp(rng.uniform(np.log(1e-5), np.log(1e-2))))
        do   = float(rng.uniform(0.10, 0.70))
        ls   = float(rng.uniform(0.00, 0.20))
        arch = str(rng.choice(['linear', 'mlp']))
        hdim = int(rng.choice([64, 128, 256]))
        opt  = str(rng.choice(['adamw', 'sgd']))
    return {
        'lr':              float(lr),
        'weight_decay':    float(wd),
        'head_dropout':    float(do),
        'label_smoothing': float(ls),
        'head_arch':       str(arch),
        'head_hidden_dim': int(hdim),
        'optimizer':       str(opt),
    }


def _suggest_gpu_config(src) -> Dict[str, Any]:
    """
    Generate one GPU full-finetune config.

    Covers:
      backbone          — categorical ['vit_b_32', 'vit_b_16']
      lr                — log-uniform [1e-5, 5e-4]
      weight_decay      — log-uniform [1e-5, 1e-2]
      head_dropout      — uniform [0.10, 0.70]
      label_smoothing   — uniform [0.00, 0.20]
      n_subset          — categorical [50, 75, 100]
      backbone_lr_ratio — log-uniform [0.01, 0.50]
    """
    optuna = _try_optuna()
    if optuna is not None and isinstance(src, optuna.Trial):
        t  = src
        bb  = t.suggest_categorical('backbone',     ['vit_b_32', 'vit_b_16'])
        lr  = t.suggest_float('lr',                 1e-5, 5e-4, log=True)
        wd  = t.suggest_float('weight_decay',       1e-5, 1e-2, log=True)
        do  = t.suggest_float('head_dropout',       0.10, 0.70)
        ls  = t.suggest_float('label_smoothing',    0.00, 0.20)
        ns  = t.suggest_categorical('n_subset',     [50, 75, 100])
        blr = t.suggest_float('backbone_lr_ratio',  0.01, 0.50, log=True)
    else:
        rng = src
        bb  = str(rng.choice(['vit_b_32', 'vit_b_16']))
        lr  = float(np.exp(rng.uniform(np.log(1e-5), np.log(5e-4))))
        wd  = float(np.exp(rng.uniform(np.log(1e-5), np.log(1e-2))))
        do  = float(rng.uniform(0.10, 0.70))
        ls  = float(rng.uniform(0.00, 0.20))
        ns  = int(rng.choice([50, 75, 100]))
        blr = float(np.exp(rng.uniform(np.log(0.01), np.log(0.50))))
    return {
        'backbone_name':      str(bb),
        'lr':                 float(lr),
        'weight_decay':       float(wd),
        'head_dropout':       float(do),
        'label_smoothing':    float(ls),
        'n_subset':           int(ns),
        'backbone_lr_ratio':  float(blr),
    }

# ---------------------------------------------------------------------------
# Seed averaging
# ---------------------------------------------------------------------------

def _run_seeds(trial_fn, n_seeds: int, **kwargs) -> Dict[str, Any]:
    """
    Run trial_fn(seed=i, **kwargs) for i in range(n_seeds).
    Returns mean/std of best_acc and best_f1, plus raw per-seed results.
    """
    results = []
    for s in range(n_seeds):
        r = trial_fn(seed=s, **kwargs)
        if 'error' in r:
            warnings.warn(f"Seed {s} failed: {r['error']}", stacklevel=2)
            continue
        results.append(r)

    if not results:
        return {'error': 'all seeds failed', 'mean_acc': 0.0, 'std_acc': 0.0,
                'mean_f1': 0.0, 'std_f1': 0.0, 'per_seed': []}

    accs = [r['best_acc'] for r in results]
    f1s  = [r['best_f1']  for r in results]
    return {
        'mean_acc':  round(float(np.mean(accs)), 2),
        'std_acc':   round(float(np.std(accs)),  2),
        'mean_f1':   round(float(np.mean(f1s)),  4),
        'std_f1':    round(float(np.std(f1s)),   4),
        'per_seed':  results,
        'n_seeds':   len(results),
    }

# ---------------------------------------------------------------------------
# Result I/O and printing
# ---------------------------------------------------------------------------

def _build_summary(results: dict) -> dict:
    """
    Extract a concise summary from the full results dict.
    This is the first thing shown in the JSON so it can be read at a glance.
    """
    summary: Dict[str, Any] = {}

    # ── Phase 1: best training config ────────────────────────────────────────
    p1 = results.get('phase1') or results.get('phase1a')
    if p1:
        summary['best_training_config'] = p1.get('best_config') or {
            'backbone':     p1.get('best_backbone'),
            'lr':           p1.get('best_lr'),
            'weight_decay': p1.get('best_wd'),
        }

    # Phase 1b regularisation winner (GPU path).
    p1b = results.get('phase1b')
    if p1b and p1b.get('best_regularisation'):
        summary['best_training_config'] = {
            **summary.get('best_training_config', {}),
            **p1b['best_regularisation'],
        }

    # Full best config (GPU path has this directly).
    if results.get('best_config'):
        summary['best_training_config'] = results['best_config']

    # Best plain accuracy from Phase 1.
    if p1:
        summary['best_plain_acc']   = p1.get('best_acc')
        summary['best_plain_f1']    = p1.get('best_f1')

    # ── Phase 2: uncertainty method comparison ────────────────────────────────
    p2 = results.get('phase2', {})
    plain_acc = (p2.get('plain') or {}).get('mean_acc')

    if p2 and plain_acc is not None:
        summary['plain_baseline_acc'] = plain_acc
        method_rows = []
        best_delta  = -999.0
        best_entry  = None

        for method, mres in p2.items():
            if method == 'plain':
                continue
            for mode in _P2_MODES:
                r = mres.get(mode, {})
                if 'mean_acc' not in r:
                    continue
                delta = r.get('delta_vs_plain', r['mean_acc'] - plain_acc)
                entry = {
                    'method':          method,
                    'mode':            mode,
                    'acc':             r['mean_acc'],
                    'std':             r.get('std_acc'),
                    'f1':              r.get('mean_f1'),
                    'delta_vs_plain':  round(delta, 2),
                }
                method_rows.append(entry)
                if delta > best_delta:
                    best_delta = delta
                    best_entry = entry

        # Sort by delta descending.
        method_rows.sort(key=lambda x: -x['delta_vs_plain'])
        summary['uncertainty_comparison'] = method_rows
        if best_entry:
            summary['best_uncertainty_source'] = best_entry

    return summary


def _strip_epoch_logs(obj: Any) -> Any:
    """Recursively remove 'epoch_log' keys to keep the file readable."""
    if isinstance(obj, dict):
        return {k: _strip_epoch_logs(v) for k, v in obj.items()
                if k != 'epoch_log'}
    if isinstance(obj, list):
        return [_strip_epoch_logs(v) for v in obj]
    return obj


def _save_results(results: dict, output_dir: str) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = str(out / 'results.json')

    def _safe(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_safe(v) for v in obj]
        return obj

    # Build a readable output: summary first, then stripped detail.
    summary   = _build_summary(results)
    stripped  = _strip_epoch_logs(results)
    output    = {'summary': summary, **stripped}

    with open(path, 'w') as f:
        json.dump(_safe(output), f, indent=2)
    return path


def _print_p1_summary(configs: List[Dict], n: int = 10) -> None:
    print(f"  {'Rank':<5} {'Acc':>8} {'Std':>6} {'F1':>7} | Config")
    print("  " + "-" * 70)
    for rank, cfg in enumerate(configs[:n], 1):
        kv = '  '.join(
            f"{k}={v:.2e}" if isinstance(v, float) and abs(v) < 1e-2
            else f"{k}={v:.3f}" if isinstance(v, float)
            else f"{k}={v}"
            for k, v in cfg['params'].items()
        )
        print(f"  {rank:<5} {cfg['mean_acc']:>7.2f}%  "
              f"{cfg['std_acc']:>5.2f}  {cfg['mean_f1']:>7.4f} | {kv}")

# ---------------------------------------------------------------------------
# CPU tuning entry point
# ---------------------------------------------------------------------------

def tune_downstream_cpu(
    raf_root: str,
    maps_dict: Optional[Dict[str, str]] = None,
    output_dir: str = './outputs',
    n_seeds: int = 2,
    n_subset: int = 25,
    n_trials: int = 100,
    search: str = 'auto',
    max_epochs: int = 60,
    patience: int = 12,
    backbone: str = 'vit_b_32',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    CPU-safe downstream hyperparameter tuning.

    Strategy
    --------
    All Phase-1 trials share pre-cached backbone features.  The backbone is
    run once per dataset variant (plain + one per uncertainty method) before
    the search starts.  Each head training trial then takes ~50 ms.

    Phase 1 search (n_trials configs × n_seeds seeds):
      Tunable: lr [1e-5..1e-3], weight_decay [1e-5..1e-2], head_dropout [0.1..0.7],
               label_smoothing [0..0.2], head_arch, head_hidden_dim, optimizer.
      search='auto'  : Optuna TPE if available, otherwise log-uniform random.
      search='grid'  : 96-point grid (4lr × 3wd × 2dropout × 2arch × 2label_smoothing).
      search='random': log-uniform random, n_trials trials.
      search='tpe'   : Optuna Tree-structured Parzen Estimator, n_trials trials.

    Phase 2: best Phase-1 config × N methods × 2 modes:
      'uncertainty_weighted' — backbone sees weighted images (new cached features)
      'loss_weighted'        — backbone sees plain images; per-sample weights
                               loaded from .npy files; reuses plain feature cache.

    Parameters
    ----------
    raf_root        : RAF-DB root containing train/ and test/ subdirs.
    maps_dict       : {method_name: maps_root_path}.  None → skip Phase 2.
    output_dir      : Results written to output_dir/results.json.
    n_seeds         : Seeds per config (default 2).
    n_subset        : Training images per class (default 25).
    n_trials        : Number of search trials for random/TPE (default 100).
                      Ignored when search='grid' (always 96 configs).
    search          : Search strategy: 'auto' | 'tpe' | 'random' | 'grid'.
    max_epochs      : Max head training epochs.
    patience        : Early stopping patience.
    backbone        : Frozen backbone name.
    """
    from src.emotion_dataset import EmotionDataset

    raf_root = str(Path(raf_root).resolve())
    if not (Path(raf_root) / 'train').exists():
        raise FileNotFoundError(f"RAF-DB train/ not found: {raf_root}")

    resolved_search = _resolve_search(search)
    n_grid          = len(_CPU_P1_GRID) if resolved_search == 'grid' else n_trials

    t_start = time.time()
    if verbose:
        print("=" * 65)
        print("[Downstream Tuning — CPU]")
        print(f"  RAF-DB    : {raf_root}")
        print(f"  Backbone  : {backbone}  (frozen, feature-cached)")
        print(f"  n_subset  : {n_subset}/class  ({n_subset * _N_CLS} train samples)")
        print(f"  Seeds     : {n_seeds}  |  epochs={max_epochs}  patience={patience}")
        print(f"  Search    : {resolved_search}  |  n_trials={n_grid}")
        if resolved_search == 'grid':
            print(f"            : lr×wd×dropout×arch×label_smoothing (hidden=128, adamw)")
        else:
            print(f"            : lr, wd, dropout, label_smoothing, head_arch,")
            print(f"              head_hidden_dim, optimizer  (all tuned simultaneously)")
        print(f"  Phase 2   : {len(maps_dict or {})} methods × {len(_P2_MODES)} modes × {n_seeds} seeds")
        print("=" * 65)

    device = 'cpu'
    n_test = max(5, min(n_subset // 3, 30))

    # ── 0. Build frozen backbone ───────────────────────────────────────────────
    if verbose:
        print(f"\n[0] Loading {backbone} backbone …", end='', flush=True)
    t0 = time.time()
    bb_module, pool, feature_dim = _build_frozen_backbone(backbone, device)
    if verbose:
        print(f"  feature_dim={feature_dim}  ({time.time()-t0:.1f} s)")

    def _make_sub(split: str, mode: str, maps_root: Optional[str]) -> Subset:
        cap = n_subset if split == 'train' else n_test
        ds  = EmotionDataset(raf_root, split, mode=mode,
                             uncertainty_root=maps_root,
                             transform=_val_transform())
        return _stratify(ds, cap)

    def _cache(split: str, mode: str, maps_root: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        sub    = _make_sub(split, mode, maps_root)
        loader = DataLoader(sub, batch_size=64, shuffle=False, num_workers=0)
        return _extract_features(bb_module, pool, loader, device)

    # ── 1. Phase 1 — plain features + search ──────────────────────────────────
    if verbose:
        print("\n[1] Phase 1 — caching plain features …", end='', flush=True)
    t0 = time.time()
    feats_tr, lbls_tr = _cache('train', 'plain', None)
    feats_te, lbls_te = _cache('test',  'plain', None)
    if verbose:
        print(f"  tr={feats_tr.shape}  te={feats_te.shape}  ({time.time()-t0:.1f} s)")
        print(f"    Running {n_grid} trials × {n_seeds} seeds …")

    def _eval_config(cfg: Dict) -> Dict:
        agg = _run_seeds(
            lambda seed, c=cfg: _train_head(
                feats_tr, lbls_tr, feats_te, lbls_te,
                feature_dim     = feature_dim,
                lr              = c['lr'],
                weight_decay    = c['weight_decay'],
                head_dropout    = c['head_dropout'],
                head_arch       = c['head_arch'],
                epochs          = max_epochs,
                patience        = patience,
                label_smoothing = c['label_smoothing'],
                device          = device,
                seed            = seed,
                sample_weights  = None,
                hidden_dim      = c['head_hidden_dim'],
                optimizer_type  = c['optimizer'],
            ),
            n_seeds=n_seeds,
        )
        return {
            'params':   cfg,
            'mean_acc': agg['mean_acc'],
            'std_acc':  agg['std_acc'],
            'mean_f1':  agg['mean_f1'],
            'std_f1':   agg['std_f1'],
            'per_seed': agg.get('per_seed', []),
        }

    p1_configs: List[Dict] = []

    if resolved_search == 'grid':
        for i, cfg in enumerate(_CPU_P1_GRID):
            p1_configs.append(_eval_config(cfg))
            if verbose and (i + 1) % 16 == 0:
                best = sorted(p1_configs, key=lambda x: -x['mean_acc'])[0]
                print(f"    [{i+1}/{n_grid}] best so far: "
                      f"{best['mean_acc']:.1f}%  {best['params']}")

    elif resolved_search == 'random':
        rng = np.random.RandomState(0)
        for i in range(n_trials):
            cfg = _suggest_cpu_config(rng)
            p1_configs.append(_eval_config(cfg))
            if verbose and (i + 1) % 25 == 0:
                best = sorted(p1_configs, key=lambda x: -x['mean_acc'])[0]
                print(f"    [{i+1}/{n_trials}] best so far: "
                      f"{best['mean_acc']:.1f}%  lr={best['params']['lr']:.2e}"
                      f"  opt={best['params']['optimizer']}"
                      f"  arch={best['params']['head_arch']}")

    else:  # tpe
        optuna = _try_optuna()
        if optuna is None:
            raise ImportError(
                "search='tpe' requires optuna.  Install with: pip install optuna"
            )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _objective(trial) -> float:
            cfg = _suggest_cpu_config(trial)
            row = _eval_config(cfg)
            p1_configs.append(row)
            if verbose and len(p1_configs) % 25 == 0:
                best = sorted(p1_configs, key=lambda x: -x['mean_acc'])[0]
                print(f"    [{len(p1_configs)}/{n_trials}] best so far: "
                      f"{best['mean_acc']:.1f}%  lr={best['params']['lr']:.2e}"
                      f"  opt={best['params']['optimizer']}"
                      f"  arch={best['params']['head_arch']}")
            return row['mean_acc']

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=0),
        )
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    p1_sorted = sorted(p1_configs, key=lambda x: (-x['mean_acc'], -x['mean_f1']))
    best_cfg  = p1_sorted[0]['params']

    if verbose:
        print(f"\n  Phase 1 top configs ({n_seeds} seeds each):")
        _print_p1_summary(p1_sorted)
        print(f"\n  Best config: {best_cfg}")

    # ── 2. Phase 2 — uncertainty method comparison ─────────────────────────────
    p2_results: Dict[str, Any] = {}

    if not maps_dict:
        if verbose:
            print("\n[2] Phase 2 skipped — maps_dict is None.")
    else:
        if verbose:
            print(f"\n[2] Phase 2 — {len(maps_dict)} methods × {len(_P2_MODES)} modes …")

        def _run_best(ft_tr, lb_tr, ft_te, lb_te,
                      sw: Optional[torch.Tensor] = None) -> Dict[str, Any]:
            return _run_seeds(
                lambda seed, ftr=ft_tr, ltr=lb_tr, fte=ft_te, lte=lb_te, s=sw: _train_head(
                    ftr, ltr, fte, lte,
                    feature_dim     = feature_dim,
                    lr              = best_cfg['lr'],
                    weight_decay    = best_cfg['weight_decay'],
                    head_dropout    = best_cfg['head_dropout'],
                    head_arch       = best_cfg['head_arch'],
                    epochs          = max_epochs,
                    patience        = patience,
                    label_smoothing = best_cfg['label_smoothing'],
                    device          = device,
                    seed            = seed,
                    sample_weights  = s,
                    hidden_dim      = best_cfg.get('head_hidden_dim', 128),
                    optimizer_type  = best_cfg.get('optimizer', 'adamw'),
                ),
                n_seeds=n_seeds,
            )

        # Baseline: plain mode with best config (reuse cached features).
        plain_agg = _run_best(feats_tr, lbls_tr, feats_te, lbls_te, None)
        p2_results['plain'] = {'mode': 'plain', **plain_agg}
        plain_acc = plain_agg['mean_acc']
        if verbose:
            print(f"    plain: {plain_acc:.1f}% ± {plain_agg['std_acc']:.1f}%")

        for method, maps_root in maps_dict.items():
            maps_root = str(Path(maps_root).resolve())
            if not Path(maps_root).exists():
                if verbose:
                    print(f"    [{method}] maps_root not found — skip")
                continue

            method_res: Dict[str, Any] = {}

            # uncertainty_weighted: backbone sees pixel-attenuated images → new features.
            if verbose:
                print(f"    [{method}/uncertainty_weighted] caching features …",
                      end='', flush=True)
            try:
                ft_uw_tr, lb_uw_tr = _cache('train', 'uncertainty_weighted', maps_root)
                ft_uw_te, lb_uw_te = _cache('test',  'uncertainty_weighted', maps_root)
                agg   = _run_best(ft_uw_tr, lb_uw_tr, ft_uw_te, lb_uw_te, None)
                delta = round(agg['mean_acc'] - plain_acc, 2)
                method_res['uncertainty_weighted'] = {**agg, 'delta_vs_plain': delta}
                sign = '+' if delta >= 0 else ''
                if verbose:
                    print(f" acc={agg['mean_acc']:.1f}% ({sign}{delta:.1f}%)")
            except Exception as exc:
                method_res['uncertainty_weighted'] = {'error': str(exc)}
                if verbose:
                    print(f" FAILED: {exc}")

            # loss_weighted: plain features + per-sample scalar weights from .npy.
            if verbose:
                print(f"    [{method}/loss_weighted] loading weights …",
                      end='', flush=True)
            try:
                tr_sub = _make_sub('train', 'plain', None)
                sw     = _extract_loss_weights(tr_sub, maps_root)
                agg    = _run_best(feats_tr, lbls_tr, feats_te, lbls_te, sw)
                delta  = round(agg['mean_acc'] - plain_acc, 2)
                method_res['loss_weighted'] = {**agg, 'delta_vs_plain': delta}
                sign = '+' if delta >= 0 else ''
                if verbose:
                    print(f" acc={agg['mean_acc']:.1f}% ({sign}{delta:.1f}%)")
            except Exception as exc:
                method_res['loss_weighted'] = {'error': str(exc)}
                if verbose:
                    print(f" FAILED: {exc}")

            method_res['best_mode'] = max(
                (m for m in method_res if 'mean_acc' in method_res[m]),
                key=lambda m: method_res[m].get('mean_acc', -1),
                default=None,
            )
            p2_results[method] = method_res

        if verbose:
            print(f"\n  Phase 2 summary  (ΔACC vs plain={plain_acc:.1f}%):")
            print(f"  {'Method':<16} {'Mode':<22} {'Acc':>8} {'ΔACC':>7} {'F1':>7}")
            print("  " + "-" * 65)
            for method, mres in p2_results.items():
                if method == 'plain':
                    continue
                for mode in _P2_MODES:
                    r = mres.get(mode, {})
                    if 'mean_acc' not in r:
                        continue
                    sign = '+' if r['delta_vs_plain'] >= 0 else ''
                    print(f"  {method:<16} {mode:<22} {r['mean_acc']:>7.2f}%  "
                          f"{sign}{r['delta_vs_plain']:>5.2f}%  {r['mean_f1']:>7.4f}")

    # ── 3. Save ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    results = {
        'meta': {
            'device':          'cpu',
            'backbone':        backbone,
            'feature_dim':     feature_dim,
            'n_subset':        n_subset,
            'n_seeds':         n_seeds,
            'max_epochs':      max_epochs,
            'patience':        patience,
            'search':          resolved_search,
            'n_trials':        n_grid,
            'elapsed_s':       round(elapsed, 1),
        },
        'phase1': {
            'best_config': best_cfg,
            'best_acc':    p1_sorted[0]['mean_acc'],
            'best_f1':     p1_sorted[0]['mean_f1'],
            'all_configs': p1_sorted,
        },
        'phase2': p2_results,
    }
    path = _save_results(results, output_dir)
    if verbose:
        print(f"\n  Results → {path}")
        print(f"  Total time: {elapsed/60:.1f} min")
    return results

# ---------------------------------------------------------------------------
# GPU tuning entry point
# ---------------------------------------------------------------------------

def tune_downstream_gpu(
    raf_root: str,
    maps_dict: Optional[Dict[str, str]] = None,
    output_dir: str = './outputs',
    n_seeds: int = 3,
    n_epochs: int = 30,
    patience: int = 7,
    n_trials: int = 40,
    search: str = 'auto',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    GPU downstream hyperparameter tuning with full two-stage fine-tuning.

    Strategy
    --------
    Phase 1  — single search over ALL parameters simultaneously (n_trials configs).
      Tunable: backbone, lr, weight_decay, head_dropout, label_smoothing,
               n_subset, backbone_lr_ratio.
      search='auto' : Optuna TPE if available, otherwise random.
      Each trial does frozen-warmup + full fine-tune and reports test accuracy.

    Phase 2  — best full config × N methods × 2 modes × n_seeds.
      Test uses the same mode as training (deployment-consistent distributions).

    Parameters
    ----------
    raf_root   : RAF-DB root.
    maps_dict  : {method_name: maps_root_path}.  None → skip Phase 2.
    output_dir : Results directory.
    n_seeds    : Seeds per config (default 3).
    n_epochs   : Max epochs per trial (early stopping usually kicks in at 10–20).
    patience   : Early stopping patience.
    n_trials   : Number of Phase-1 search trials (default 40).
    search     : Search strategy: 'auto' | 'tpe' | 'random'.
    """
    import torch
    if not torch.cuda.is_available():
        warnings.warn(
            "No CUDA device detected.  tune_downstream_gpu() will run on CPU — very slow. "
            "Use tune_downstream_cpu() instead.",
            UserWarning, stacklevel=2,
        )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    raf_root = str(Path(raf_root).resolve())
    if not (Path(raf_root) / 'train').exists():
        raise FileNotFoundError(f"RAF-DB train/ not found: {raf_root}")

    resolved_search = _resolve_search(search)
    if resolved_search == 'grid':
        warnings.warn(
            "search='grid' is not supported for tune_downstream_gpu; using 'random'.",
            UserWarning, stacklevel=2,
        )
        resolved_search = 'random'

    t_start = time.time()
    if verbose:
        gpu_name = (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else 'cpu')
        print("=" * 65)
        print("[Downstream Tuning — GPU]")
        print(f"  Device    : {device}  ({gpu_name})")
        print(f"  RAF-DB    : {raf_root}")
        print(f"  Seeds     : {n_seeds}  |  epochs={n_epochs}  patience={patience}")
        print(f"  Search    : {resolved_search}  |  n_trials={n_trials}")
        print(f"  Tunable   : backbone, lr, wd, dropout, label_smoothing,")
        print(f"              n_subset, backbone_lr_ratio")
        print(f"  Phase 2   : {len(maps_dict or {})} methods × {len(_P2_MODES)} modes × {n_seeds} seeds")
        print("=" * 65)

    def _run_config(cfg: Dict) -> Dict[str, Any]:
        full = dict(mode='plain', raf_root=raf_root, maps_root=None,
                    n_epochs=n_epochs, patience=patience, device=device, **cfg)
        agg = _run_seeds(lambda seed, c=full: _train_full(seed=seed, **c), n_seeds=n_seeds)
        return {
            'params':   cfg,
            'mean_acc': agg['mean_acc'],
            'std_acc':  agg['std_acc'],
            'mean_f1':  agg['mean_f1'],
            'std_f1':   agg['std_f1'],
            'per_seed': agg.get('per_seed', []),
        }

    # ── Phase 1: joint search over all parameters ──────────────────────────────
    if verbose:
        print(f"\n[1] Phase 1 — {n_trials} trials …")

    p1_configs: List[Dict] = []

    def _log_progress(i: int) -> None:
        if not verbose:
            return
        best = sorted(p1_configs, key=lambda x: -x['mean_acc'])[0]
        print(f"  [{i:>3}/{n_trials}] best so far: "
              f"{best['mean_acc']:.1f}±{best['std_acc']:.1f}%  "
              f"backbone={best['params']['backbone_name']}  "
              f"lr={best['params']['lr']:.2e}")

    if resolved_search == 'random':
        rng = np.random.RandomState(0)
        for i in range(n_trials):
            t0  = time.time()
            cfg = _suggest_gpu_config(rng)
            row = _run_config(cfg)
            p1_configs.append(row)
            if verbose:
                sign = ''
                print(f"  [{i+1:>3}/{n_trials}]  acc={row['mean_acc']:.1f}±{row['std_acc']:.1f}%"
                      f"  bb={cfg['backbone_name']}  lr={cfg['lr']:.2e}"
                      f"  blr={cfg['backbone_lr_ratio']:.3f}"
                      f"  ({time.time()-t0:.0f}s)")

    else:  # tpe
        optuna = _try_optuna()
        if optuna is None:
            raise ImportError(
                "search='tpe' requires optuna.  Install with: pip install optuna"
            )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _objective(trial) -> float:
            cfg = _suggest_gpu_config(trial)
            t0  = time.time()
            row = _run_config(cfg)
            p1_configs.append(row)
            if verbose:
                print(f"  [{len(p1_configs):>3}/{n_trials}]  "
                      f"acc={row['mean_acc']:.1f}±{row['std_acc']:.1f}%"
                      f"  bb={cfg['backbone_name']}  lr={cfg['lr']:.2e}"
                      f"  blr={cfg['backbone_lr_ratio']:.3f}"
                      f"  ({time.time()-t0:.0f}s)")
            return row['mean_acc']

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=0),
        )
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    p1_sorted   = sorted(p1_configs, key=lambda x: (-x['mean_acc'], -x['mean_f1']))
    best_config = p1_sorted[0]['params']

    if verbose:
        print(f"\n  Phase 1 top configs:")
        _print_p1_summary(p1_sorted)
        print(f"\n  Best config: {best_config}")

    # Add fixed training params to best_config for Phase 2 / save.
    best_config_full = {**best_config, 'n_epochs': n_epochs, 'patience': patience}

    # ── Phase 2: uncertainty method comparison ─────────────────────────────────
    p2_results: Dict[str, Any] = {}

    if not maps_dict:
        if verbose:
            print("\n[2] Phase 2 skipped — maps_dict is None.")
    else:
        if verbose:
            print(f"\n[2] Phase 2 — {len(maps_dict)} methods × {len(_P2_MODES)} modes …")

        if verbose:
            print("    plain: training …", end='', flush=True)
        t0 = time.time()
        plain_full = dict(mode='plain', raf_root=raf_root, maps_root=None,
                          device=device, **best_config_full)
        plain_agg = _run_seeds(
            lambda seed, c=plain_full: _train_full(seed=seed, **c), n_seeds=n_seeds
        )
        p2_results['plain'] = {'mode': 'plain', **plain_agg}
        plain_acc = plain_agg['mean_acc']
        if verbose:
            print(f" acc={plain_acc:.1f}±{plain_agg['std_acc']:.1f}%  ({time.time()-t0:.0f}s)")

        for method, maps_root in maps_dict.items():
            maps_root = str(Path(maps_root).resolve())
            if not Path(maps_root).exists():
                if verbose:
                    print(f"    [{method}] maps_root not found — skip")
                continue

            method_res: Dict[str, Any] = {}
            for mode in _P2_MODES:
                if verbose:
                    print(f"    [{method}/{mode}] training …", end='', flush=True)
                t0 = time.time()
                try:
                    full = dict(mode=mode, raf_root=raf_root, maps_root=maps_root,
                                device=device, **best_config_full)
                    agg   = _run_seeds(
                        lambda seed, c=full: _train_full(seed=seed, **c), n_seeds=n_seeds
                    )
                    delta = round(agg['mean_acc'] - plain_acc, 2)
                    method_res[mode] = {**agg, 'delta_vs_plain': delta}
                    sign = '+' if delta >= 0 else ''
                    if verbose:
                        print(f" acc={agg['mean_acc']:.1f}±{agg['std_acc']:.1f}%  "
                              f"({sign}{delta:.1f}%)  ({time.time()-t0:.0f}s)")
                except Exception as exc:
                    method_res[mode] = {'error': str(exc)}
                    if verbose:
                        print(f" FAILED: {exc}")

            method_res['best_mode'] = max(
                (m for m in method_res if 'mean_acc' in method_res[m]),
                key=lambda m: method_res[m].get('mean_acc', -1),
                default=None,
            )
            p2_results[method] = method_res

        if verbose:
            print(f"\n  Phase 2 summary  (ΔACC vs plain={plain_acc:.1f}%):")
            print(f"  {'Method':<16} {'Mode':<22} {'Acc':>8} {'ΔACC':>7} {'F1':>7}")
            print("  " + "-" * 65)
            for method, mres in p2_results.items():
                if method == 'plain':
                    continue
                for mode in _P2_MODES:
                    r = mres.get(mode, {})
                    if 'mean_acc' not in r:
                        continue
                    sign = '+' if r['delta_vs_plain'] >= 0 else ''
                    print(f"  {method:<16} {mode:<22} {r['mean_acc']:>7.2f}%  "
                          f"{sign}{r['delta_vs_plain']:>5.2f}%  {r['mean_f1']:>7.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    results = {
        'meta': {
            'device':      device,
            'n_seeds':     n_seeds,
            'max_epochs':  n_epochs,
            'patience':    patience,
            'search':      resolved_search,
            'n_trials':    n_trials,
            'elapsed_s':   round(elapsed, 1),
        },
        'phase1': {
            'best_config': best_config,
            'best_acc':    p1_sorted[0]['mean_acc'],
            'best_f1':     p1_sorted[0]['mean_f1'],
            'all_configs': p1_sorted,
        },
        'best_config':  best_config_full,
        'phase2':       p2_results,
    }
    path = _save_results(results, output_dir)
    if verbose:
        print(f"\n  Results → {path}")
        print(f"  Total time: {elapsed/60:.1f} min")
    return results
