"""
main.py — 3D Face Reconstruction Uncertainty Pipeline
======================================================
Stages
------
  1  eda         — dataset EDA: subset counts, image stats, class distribution
  2  inference   — FLAME parameter regression with SMIRK / DECA / EMOCA / SHeaP
  3  uncertainty — uncertainty methods (TTA, MCD, Cross, Jacobian,
                   Mahalanobis, SOL-MCD, A-MCD)
  4  evaluation  — geometric error + uncertainty quality metrics (Spearman ρ,
                   AUSE, ECE, NLL) with full visualisation report
  5  downstream  — (reserved) uncertainty-weighted emotion classifier

CPU / GPU
---------
All model wrappers accept a ``device`` argument; no code path is GPU-only.
Pass ``--cpu`` to force CPU even when CUDA is detected.

Model selection
---------------
    --models all                       # all four regressors (default)
    --models SMIRK DECA                # subset
    --models SHeaP                     # single model

Method selection
----------------
    --methods all                      # all 7 methods (default)
    --methods all_no_dropout           # TTA, CrossMethod, Jacobian, Mahalanobis
    --methods all_dropout              # MCD, SOL-MCD, A-MCD (need smirk_checkpoint_data/trained.pt)
    --methods tta jacobian             # explicit list of individual methods
    --methods mahalanobis              # single method

Valid method names: tta  mcd  crossmethod  jacobian  mahalanobis  sol_mcd  amcd

Examples
--------
    # Single image, CPU only, no dropout:
    python main.py --stage uncertainty --cpu \\
                   --models SMIRK --methods all_no_dropout

    # Batch evaluation on NoW with all models and all methods:
    python main.py --stage all --dataset tempeh --partition_size 5 \\
                   --models all --methods all --output_dir ./outputs

    # Quick: just TTA + Jacobian on DECA
    python main.py --stage uncertainty --models DECA \\
                   --methods tta jacobian
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# ── src imports ───────────────────────────────────────────────────────────────
from src.data_loader import FaceDatasetLoader, perform_basic_eda, SUBSET_SIZES
from src.inference import UnifiedFaceRegressor
from src.uncertainty import (
    calculate_tta_uncertainty,
    calculate_mcd_uncertainty,
    calculate_cross_method_disagreement,
    calculate_jacobian_sensitivity_uncertainty,
    calculate_mahalanobis_uncertainty,
    calculate_sol_mcd_uncertainty,
    calculate_antithetic_mcd_uncertainty,
)
from src.evaluation import (
    calculate_geometric_error,
    calculate_vertex_rmse,
    calculate_scan_to_mesh_distance,
    calculate_region_wise_geometric_error,
    correlate_error_and_uncertainty,
    calculate_sparsification_error_curve,
    calculate_uncertainty_calibration,
    calculate_nll,
    compare_uncertainty_methods,
    compute_static_region_baseline,
    calculate_rank_stability,
)
from src.visualization import (
    render_uncertainty_heatmap,
    plot_uncertainty_spatial_maps,
    plot_uncertainty_comparison_violin,
    plot_sparsification_curves,
    plot_calibration_diagram,
    plot_uncertainty_vs_error_scatter,
    plot_method_comparison_table,
    plot_region_error_breakdown,
    plot_error_distributions,
    plot_multi_model_reconstruction,
    plot_image_uncertainty_row,
    plot_per_image_uncertainty_gallery,
    plot_model_comparison_gallery,
    plot_uncertainty_winner_summary,
    plot_per_image_rho_distribution,
    plot_tta_n_ablation,
    plot_paper_style_comparison_panel,
    plot_method_correlation_matrix,
    create_full_analysis_report,
)
from src.uncertainty import calculate_tta_uncertainty_n_ablation


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

_ALL_MODELS   = ['SMIRK', 'DECA', 'EMOCA', 'SHeaP']
_ALL_METHODS  = ['tta', 'mcd', 'crossmethod', 'jacobian', 'mahalanobis', 'sol_mcd', 'amcd']
_NO_DROPOUT   = ['tta', 'crossmethod', 'jacobian', 'mahalanobis']
_DROPOUT_ONLY = ['mcd', 'sol_mcd', 'amcd']


def _resolve_models(model_args: list) -> list:
    """Expand 'all' shorthand; deduplicate; warn on unknown names."""
    if 'all' in [m.lower() for m in model_args]:
        return _ALL_MODELS[:]
    valid = set(_ALL_MODELS)
    result: list = []
    for m in model_args:
        if m not in valid:
            warnings.warn(
                f"Unknown model '{m}' — ignored. "
                f"Valid: {_ALL_MODELS + ['all']}",
                RuntimeWarning, stacklevel=2,
            )
        elif m not in result:
            result.append(m)
    return result or _ALL_MODELS[:]


def _resolve_methods(method_args: list) -> set:
    """Expand group aliases to a set of canonical method keys."""
    result: set = set()
    for m in method_args:
        key = m.lower()
        if key == 'all':
            result.update(_ALL_METHODS)
        elif key == 'all_no_dropout':
            result.update(_NO_DROPOUT)
        elif key == 'all_dropout':
            result.update(_DROPOUT_ONLY)
        elif key in _ALL_METHODS:
            result.add(key)
        else:
            warnings.warn(
                f"Unknown method '{m}' — ignored. "
                f"Valid: {_ALL_METHODS + ['all', 'all_no_dropout', 'all_dropout']}",
                RuntimeWarning, stacklevel=2,
            )
    return result if result else set(_ALL_METHODS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Uncertainty-Aware 3D Face Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Method names:  tta  mcd  crossmethod  jacobian  mahalanobis  sol_mcd  amcd\n"
            "Method groups: all  all_no_dropout  all_dropout\n"
            "Model names:   SMIRK  DECA  EMOCA  SHeaP  (or 'all')"
        ),
    )
    p.add_argument('--stage', default='all',
                   choices=['eda', 'inference', 'uncertainty', 'evaluation',
                            'downstream', 'all', 'no_downstream', 'tune',
                            'downstream_tune', 'plot_downstream'],
                   help=(
                       'Pipeline stage to run.  '
                       '"all" = EDA→inference→uncertainty→evaluation→downstream.  '
                       '"no_downstream" = EDA→inference→uncertainty→evaluation (skip Stage 5).  '
                       '"tune" = hyperparameter search for uncertainty methods (Stage 0).  '
                       '"downstream_tune" = hyperparameter tuning for the downstream classifier '
                       '(CPU: feature-cached ~5 min; GPU: full fine-tune ~25–45 min).  '
                       '"plot_downstream" = parse a Stage-5 text log (--downstream_log) and write '
                       'plain-vs-weighted comparison plots (--downstream_plots_out); does nothing '
                       'else, no model loading, exits immediately after.  '
                       'Individual: eda | inference | uncertainty | evaluation | downstream.'
                   ))
    p.add_argument('--downstream_log', default='figures/downstream.txt',
                   help='[plot_downstream] Path to the Stage-5 text log to parse.')
    p.add_argument('--downstream_plots_out', default='figures/downstream_plots',
                   help='[plot_downstream] Directory to write the comparison plots into.')
    p.add_argument('--tune_n_images', type=int, default=8,
                   help='Number of GT-paired images to use for hyperparameter tuning (Stage 0).')
    p.add_argument('--tune_objective', default='spearman_rho',
                   choices=['spearman_rho', 'ause'],
                   help='Objective for hyperparameter tuning: spearman_rho (default) or ause.')
    p.add_argument('--tune_methods', nargs='+',
                   default=['tta', 'cross', 'jacobian', 'mahalanobis'],
                   help='Which methods to tune (default: all non-MCD methods). '
                        'Valid: tta mcd cross jacobian mahalanobis sol_mcd a_mcd')
    p.add_argument('--image_path', default='./data/sample_image.jpg',
                   help='Path to the primary test image.')
    p.add_argument('--output_dir', default='./outputs',
                   help='Root directory for all output files.')
    p.add_argument('--dataset', default='now',
                   choices=['now', 'coma', 'tempeh', 'utkface', 'lfw'],
                   help='Dataset to use for batch evaluation and Mahalanobis reference.')
    p.add_argument('--partition_size', type=int, default=5,
                   help='Number of dataset samples for batch evaluation, gallery, and Mahalanobis reference.')
    p.add_argument('--models', nargs='+', default=['all'],
                   metavar='MODEL',
                   help=(
                       'FLAME regressors to load.  '
                       '"all" selects all four.  '
                       'Examples: --models all | --models SMIRK DECA | --models SHeaP'
                   ))
    p.add_argument('--methods', nargs='+', default=['all'],
                   metavar='METHOD',
                   help=(
                       'Uncertainty methods to run.  '
                       'Groups: all | all_no_dropout | all_dropout.  '
                       'Individual: tta mcd crossmethod jacobian mahalanobis sol_mcd amcd.  '
                       'Examples: --methods all_no_dropout | --methods tta jacobian'
                   ))
    p.add_argument('--primary_model', default='SMIRK',
                   choices=_ALL_MODELS,
                   help='Model for single-model methods (TTA, Jacobian, Mahalanobis).')
    p.add_argument('--n_tta', type=int, default=10,
                   help='Number of augmented forward passes for TTA.')
    p.add_argument('--n_mcd', type=int, default=15,
                   help='MC Dropout: n_passes for MCD/SOL-MCD; n_pairs for A-MCD.')
    p.add_argument('--n_jacobian', type=int, default=8,
                   help='Random projection directions for the Jacobian method.')
    p.add_argument('--n_mahal_ref', type=int, default=20,
                   help='Minimum number of reference images for Mahalanobis covariance estimation (default 20).')
    p.add_argument('--cpu', action='store_true',
                   help='Force CPU even when CUDA is available.')

    # ── Stage 5: downstream classifier ────────────────────────────────────────
    p.add_argument('--raf_db_root', default='./datasets/fer2013',
                   help='Root of the emotion dataset (FER2013 after organise_fer2013.py). '
                        'Must contain train/{anger,disgust,fear,happy,neutral,sad,surprise}/ '
                        'and test/(same). Set EMOTION_DB_ROOT in run_experiments.sh.')
    p.add_argument('--maps_cache_dir', default='./datasets/maps',
                   help='Stable directory for pre-computed confidence maps. '
                        'Keyed by model/method/n_tta/subset so maps are reused across '
                        'runs even when --output_dir changes. Set MAPS_CACHE_DIR in '
                        'run_experiments.sh to an absolute HPC path to survive job restarts.')
    p.add_argument('--downstream_epochs', type=int, default=20,
                   help='Training epochs for each classifier variant.')
    p.add_argument('--downstream_backbone', default='vit_b_32',
                   choices=['vit_b_32', 'vit_b_16', 'vit_l_16', 'vit_l_32',
                            'vit_h_14', 'resnet18', 'resnet50'],
                   help=(
                       'Backbone for UncertaintyWeightedClassifier. '
                       'vit_b_32 = CPU-friendly (default); vit_b_16 = balanced (GPU); '
                       'vit_l_16 = GPU required; vit_h_14 = HPC only; '
                       'resnet18/50 = legacy baselines.'
                   ))
    p.add_argument('--downstream_batch_size', type=int, default=32,
                   help='Batch size for downstream training.')
    p.add_argument('--downstream_lr', type=float, default=2e-4,
                   help='Initial learning rate for the downstream optimizer.')
    p.add_argument('--downstream_methods', nargs='+', default=['tta'],
                   metavar='METHOD',
                   help=(
                       'Uncertainty methods to use for Stage 5 confidence-map generation. '
                       'Only single-wrapper methods are supported: tta, jacobian. '
                       'Default: tta (CPU-friendly). Example: --downstream_methods tta jacobian'
                   ))
    p.add_argument('--downstream_subset', type=int, default=25, metavar='N',
                   help=(
                       'Legacy per-class cap applied to both train (N) and test (N//5). '
                       'Ignored when --downstream_train_subset is set explicitly. '
                       'CPU default: 25 (~175 train + 35 test = 210 total, ~15-20 min).'
                   ))
    p.add_argument('--downstream_train_subset', type=int, default=0, metavar='N',
                   help='Cap training set at N images per class (0 = no cap, use full train). '
                        'Takes priority over --downstream_subset when set.')
    p.add_argument('--downstream_test_subset', type=int, default=0, metavar='N',
                   help='Cap test set at N images per class (0 = no cap, use full test set). '
                        'Default 0 keeps all test images for maximum statistical power.')
    p.add_argument('--downstream_n_tta', type=int, default=None, metavar='K',
                   help=(
                       'TTA passes used specifically for Stage-5 confidence-map '
                       'precomputation — independent of --n_tta used in Stage 3. '
                       'Lower = much faster precomputation. '
                       'CPU default: 3.  GPU: 5.  Set explicitly to override.'
                   ))
    p.add_argument('--downstream_mode', default='loss_weighted',
                   choices=['loss_weighted', 'pixel_masked'],
                   help=(
                       'How uncertainty confidence maps are used in Stage 5. '
                       '"loss_weighted" (default): pixels are untouched; per-sample '
                       '10th-percentile confidence of the cropped region is used as a '
                       'loss weight so high-uncertainty images contribute less to gradient '
                       'updates. '
                       '"pixel_masked": multiply the normalised image tensor by the '
                       'confidence map to attenuate uncertain face regions directly.'
                   ))
    p.add_argument('--downstream_fusion', default='input',
                   choices=['input', 'patch_embed', 'attn_bias', 'key_scale',
                            'value_scale', 'all'],
                   help=(
                       'Uncertainty injection point for the Stage-5 ViT classifier. '
                       '"input" (default): multiply input image by (1 − α·U_2D). '
                       '"patch_embed": scale conv_proj patch tokens before the encoder. '
                       '"key_scale": scale MHA key vectors in every encoder layer. '
                       '"value_scale": scale MHA value vectors in every encoder layer. '
                       '"attn_bias": subtract α·U_j from pre-softmax attention logits '
                       '(requires PyTorch ≥ 2.0). '
                       '"all": run all five modes as separate sub-experiments and '
                       'compare results. '
                       'ViT-internal modes are silently ignored for ResNet backbones.'
                   ))
    p.add_argument('--downstream_fusion_alpha', type=float, default=1.0,
                   help='Suppression strength α for downstream uncertainty injection (default: 1.0).')
    p.add_argument('--downstream_weight_decay', type=float, default=5e-2,
                   help='Weight decay for downstream optimizer (default: 5e-2; ViT fine-tuning needs strong L2).')
    p.add_argument('--downstream_label_smoothing', type=float, default=0.1,
                   help='Label smoothing for downstream cross-entropy loss (default: 0.1).')
    p.add_argument('--downstream_head_dropout', type=float, default=0.5,
                   help='Dropout probability in the classifier head (default: 0.5).')
    p.add_argument('--downstream_head_arch', default='mlp',
                   choices=['mlp', 'linear'],
                   help=(
                       'Classification head architecture for Stage 5. '
                       '"mlp" (default): LayerNorm→Linear(D,128)→GELU→Dropout→Linear(128,C). '
                       '"linear": LayerNorm→Dropout→Linear(D,C) — fewer params, '
                       'better for small datasets.'
                   ))
    p.add_argument('--downstream_patience', type=int, default=3,
                   help=(
                       'Early-stopping patience for Stage-5 training: number of epochs '
                       'without a strict test-accuracy improvement before training stops. '
                       'Default: 3.  Keep low on small datasets to prevent overfitting.'
                   ))
    p.add_argument('--downstream_unfreeze_blocks', type=int, default=12,
                   help=(
                       'Number of trailing ViT transformer blocks to unfreeze in Stage 2 '
                       '(counted from the top of the encoder). -1 = unfreeze all. '
                       'Default: 6.  Partial unfreezing prevents 303M-param overfitting on '
                       'small datasets.'
                   ))
    p.add_argument('--downstream_mixup_alpha', type=float, default=0.1,
                   help=(
                       'Beta-distribution α for Mixup regularization applied during '
                       'Stage-2 fine-tuning (0 = disabled). Default: 0.2.'
                   ))
    # Manual --flag/--no-flag pair instead of argparse.BooleanOptionalAction
    # (Python 3.9+ only) — this repo also runs on Python 3.8 (e.g. local WSL dev).
    p.add_argument('--downstream_curriculum', dest='downstream_curriculum',
                   action='store_true', default=True,
                   help='Enable curriculum learning in Stage 2: sort training samples by uncertainty '
                        'score (easy-first) and gradually expose harder samples. Default: True.')
    p.add_argument('--no-downstream_curriculum', dest='downstream_curriculum',
                   action='store_false',
                   help='Disable curriculum learning (see --downstream_curriculum).')
    p.add_argument('--downstream_curriculum_start', type=float, default=0.5,
                   help='Fraction of training data visible in the first Stage-2 epoch. '
                        'Grows linearly to 1.0 by the last epoch. Default: 0.5.')
    p.add_argument('--force_recompute_maps', action='store_true',
                   help=(
                       'Delete and regenerate all Stage-5 confidence maps even when '
                       'cached .npy files already exist.  Use this after changing '
                       '--downstream_n_tta or after updating the pipeline code.'
                   ))
    p.add_argument('--downstream_reuse_checkpoint', action='store_true',
                   help=(
                       'If a {tag}_best.pth checkpoint already exists under '
                       '{output_dir}/downstream/checkpoints/ for a given classifier '
                       'variant (e.g. the PLAIN baseline, or one particular fusion mode), '
                       'load it and evaluate once instead of retraining from scratch. '
                       'Use this to resume a combo (e.g. re-running only the remaining '
                       'fusion modes for a combo) without paying to retrain PLAIN again, '
                       'as long as --output_dir points at the same directory that holds '
                       'the earlier checkpoint.'
                   ))

    # ── Stage downstream_tune (downstream hyperparameter tuning) ──────────────
    p.add_argument('--downstream_maps_dir', default='',
                   help=(
                       'Root directory containing pre-computed confidence-map trees '
                       '(one sub-directory per model×method combo, each mirroring the '
                       'RAF-DB image tree with .npy files).  Used by --stage downstream_tune '
                       'for Phase 2.  When empty the pipeline looks for maps under '
                       '{output_dir}/downstream/maps/ (the default Stage-5 location).  '
                       'Phase 2 is skipped when no maps are found.'
                   ))
    p.add_argument('--downstream_tune_n_seeds', type=int, default=None,
                   help=(
                       'Random seeds per config in downstream_tune (default: 2 on CPU, '
                       '3 on GPU).  More seeds give more reliable estimates.'
                   ))
    p.add_argument('--downstream_tune_max_epochs', type=int, default=None,
                   help=(
                       'Max training epochs per trial in downstream_tune '
                       '(default: 60 on CPU with feature caching; 30 on GPU).'
                   ))
    p.add_argument('--downstream_tune_patience', type=int, default=None,
                   help=(
                       'Early-stopping patience for downstream_tune '
                       '(default: 12 on CPU; 7 on GPU).'
                   ))
    p.add_argument('--downstream_tune_n_trials', type=int, default=None,
                   help=(
                       'Number of search trials for downstream_tune Phase 1 '
                       '(default: 100 on CPU; 40 on GPU).  Ignored when '
                       '--downstream_tune_search=grid.'
                   ))
    p.add_argument('--downstream_tune_search', default=None,
                   choices=['auto', 'tpe', 'random', 'grid'],
                   help=(
                       'Hyperparameter search strategy for downstream_tune Phase 1.  '
                       'auto (default): Optuna TPE when optuna is installed, otherwise '
                       'log-uniform random.  tpe: Optuna TPE (requires pip install optuna).  '
                       'random: log-uniform random sampling.  '
                       'grid: exhaustive 96-point grid (CPU only, original behaviour).'
                   ))
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _load_image(path: str) -> Optional[np.ndarray]:
    """Load an image as RGB uint8.  Returns None if the file does not exist."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _get_flame_faces(regressor: UnifiedFaceRegressor) -> Optional[np.ndarray]:
    """
    Best-effort extraction of FLAME face topology (F, 3) int64 from any loaded wrapper.
    Returns None if not discoverable.
    """
    for wrapper in regressor.models.values():
        # SMIRK: wrapper.flame holds the FLAME nn.Module
        flame = getattr(wrapper, 'flame', None)
        if flame is not None:
            for attr in ('faces_tensor', 'faces', 'triangles'):
                f = getattr(flame, attr, None)
                if f is not None:
                    if isinstance(f, torch.Tensor):
                        return f.cpu().numpy().astype(np.int64)
                    try:
                        return np.asarray(f, dtype=np.int64)
                    except Exception:
                        pass
        # DECA/EMOCA: nested under model.flame or model.flame_model
        model = getattr(wrapper, 'model', None)
        if model is not None:
            for sub_attr in ('flame', 'flame_model', '_flame'):
                sub = getattr(model, sub_attr, None)
                if sub is not None:
                    for attr in ('faces_tensor', 'faces', 'triangles'):
                        f = getattr(sub, attr, None)
                        if f is not None:
                            if isinstance(f, torch.Tensor):
                                return f.cpu().numpy().astype(np.int64)
                            try:
                                return np.asarray(f, dtype=np.int64)
                            except Exception:
                                pass
    return None


def _save_json(data: dict, path: str) -> None:
    """Save a dict of Python scalars / arrays to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_convert)
    print(f"  [saved] {os.path.relpath(path)}")


def _try_load_flame_masks(
    regressor: UnifiedFaceRegressor,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Try to load FLAME facial region masks (vertex index arrays) from the SMIRK
    assets directory.  Returns {region_name: int64 vertex-index array} or None.

    The file `FLAME_masks.pkl` (or `flame_masks.pkl`) ships with the FLAME
    model package.  It maps region names (e.g. 'face', 'neck', 'scalp', 'nose',
    'left_eyeball', 'lips', …) to arrays of FLAME vertex indices.

    If the file is absent or unparsable the function returns None gracefully;
    the caller must handle None by skipping region breakdown.
    """
    import pickle
    candidates_per_wrapper = []
    for wrapper in regressor.models.values():
        smirk_root = getattr(wrapper, '_smirk_root', None)
        if smirk_root:
            candidates_per_wrapper.extend([
                os.path.join(smirk_root, 'assets', 'FLAME2020', 'FLAME_masks.pkl'),
                os.path.join(smirk_root, 'assets', 'FLAME2020', 'flame_masks.pkl'),
                os.path.join(smirk_root, 'assets', 'FLAME_masks.pkl'),
                os.path.join(smirk_root, 'assets', 'flame_masks.pkl'),
                os.path.join(smirk_root, 'FLAME_masks.pkl'),
            ])

    for path in candidates_per_wrapper:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'rb') as fh:
                raw = pickle.load(fh, encoding='latin1')
            masks: Dict[str, np.ndarray] = {}
            for name, idxs in raw.items():
                arr = np.asarray(idxs, dtype=np.int64).ravel()
                if len(arr) > 0:
                    masks[name] = arr
            if masks:
                print(f"  [info] FLAME region masks: {list(masks.keys())}")
                return masks
        except Exception as exc:
            print(f"  [warn] Could not parse FLAME masks at {path}: {exc}")

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: EDA
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("[Stage 1] Exploratory Data Analysis")
    print("=" * 60)

    loader = FaceDatasetLoader(data_root='./datasets', subset_sizes=SUBSET_SIZES,
                               render_coma=(args.dataset == 'coma'))

    selected = args.dataset if args.dataset != 'all' else None
    eda_datasets = [selected] if selected else ['now', 'coma', 'tempeh', 'utkface', 'lfw']
    for ds_name in eda_datasets:
        print(f"\n  [{ds_name}]")
        try:
            subsets = loader.create_subsets(ds_name)
            for n, items in subsets.items():
                if items:
                    print(f"    subset n={n}: {len(items)} samples")
                    perform_basic_eda(items)
                else:
                    print(f"    subset n={n}: 0 samples (dataset not downloaded?)")
        except Exception as exc:
            print(f"    [skip] {exc}")

    try:
        raf_train = loader.load_raf_db(split='train')
        raf_test  = loader.load_raf_db(split='test')
        print(f"\n  [raf-db] train={len(raf_train)}  test={len(raf_test)}")
    except FileNotFoundError:
        print("\n  [raf-db] not found")
    except Exception as exc:
        print(f"\n  [raf-db] {exc}")

    print("\n[Stage 1] EDA complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Inference
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(
    args: argparse.Namespace,
    image: np.ndarray,
    device: str,
) -> Tuple[UnifiedFaceRegressor, Dict[str, np.ndarray]]:
    """
    Load each FLAME regressor sequentially, run inference, then free it from
    RAM before loading the next one.  Only the primary model stays in memory
    after this stage — it is needed by TTA, Jacobian, and Mahalanobis.

    CrossMethod disagreement uses the saved vertex arrays (not live models),
    so all models never need to coexist in RAM simultaneously.
    """
    import gc

    print("\n" + "=" * 60)
    print(f"[Stage 2] Unified Inference  (device={device})  [sequential loading]")
    print("=" * 60)

    mesh_results: Dict[str, np.ndarray] = {}
    primary_regressor: Optional[UnifiedFaceRegressor] = None
    verts_dir = os.path.join(args.output_dir, 'vertices')
    os.makedirs(verts_dir, exist_ok=True)

    # Determine which model to keep in memory for single-model uncertainty methods
    primary_name = args.primary_model
    if primary_name not in args.models:
        primary_name = args.models[0]

    for name in args.models:
        print(f"  Loading {name} …", end=' ', flush=True)
        try:
            r = UnifiedFaceRegressor(device=device, models=[name])
            if name not in r.models:
                print(f"not loaded — skipped.")
                del r
                continue
            verts = r.models[name].get_vertices(image)       # (5023, 3)
            mesh_results[name] = verts
            np.save(os.path.join(verts_dir, f'{name}_vertices.npy'), verts)
            print(f"done  shape={verts.shape}  z̄={verts[:, 2].mean():.3f}")

            if name == primary_name:
                primary_regressor = r                        # keep in RAM
            else:
                del r                                        # free weights
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as exc:
            print(f"FAILED ({exc})")
            try:
                del r
            except NameError:
                pass
            gc.collect()

    # If primary model never loaded, fall back to the first successful one
    if primary_regressor is None and mesh_results:
        fallback = next(iter(mesh_results))
        print(f"  [info] Primary model '{primary_name}' unavailable; "
              f"reloading '{fallback}' for uncertainty methods.")
        try:
            primary_regressor = UnifiedFaceRegressor(device=device, models=[fallback])
        except Exception as exc:
            print(f"  [warn] Fallback reload failed: {exc}")

    if not mesh_results:
        print("\n  [warn] No models produced output.  "
              "Verify that checkpoints are installed under models/.")
    else:
        print(f"\n  Vertex arrays saved → {verts_dir}/")

    print(f"\n[Stage 2] Inference complete.  "
          f"{len(mesh_results)}/{len(args.models)} models ran.")
    return primary_regressor, mesh_results


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: Uncertainty Quantification
# ══════════════════════════════════════════════════════════════════════════════

def run_uncertainty(
    args: argparse.Namespace,
    regressor: UnifiedFaceRegressor,
    image: np.ndarray,
    mesh_results: Dict[str, np.ndarray],
    reference_images: Optional[List[np.ndarray]],
    device: str,
    active_methods: Optional[set] = None,
) -> Dict[str, np.ndarray]:
    """
    Run the requested uncertainty methods and return {method_name: (5023,1)}.

    active_methods is the resolved set of lowercase method keys from
    _resolve_methods(args.methods).  When None, all 7 methods are attempted.

    Methods 2/6/7 (MCD variants) require SMIRK retrained with nn.Dropout.
    Method 5 (Mahalanobis) requires ≥4 reference images.
    All other methods need only the standard checkpoints and the test image.
    """
    if active_methods is None:
        active_methods = set(_ALL_METHODS)

    n_requested = len(active_methods)

    print("\n" + "=" * 60)
    print("[Stage 3] Uncertainty Quantification")
    print(f"  Methods requested ({n_requested}): {sorted(active_methods)}")
    print("=" * 60)

    unc: Dict[str, np.ndarray] = {}

    # Primary single-model wrapper
    primary = args.primary_model
    if primary not in regressor.models or primary not in mesh_results:
        if mesh_results:
            primary = next(iter(mesh_results))
            print(f"  [info] Primary model '{args.primary_model}' not available; "
                  f"using '{primary}'.")
        else:
            print("  [warn] No models available — skipping single-model methods.")
            primary = None

    # ── 1. TTA ────────────────────────────────────────────────────────────────
    if 'tta' not in active_methods:
        print("\n  [1/7] TTA — skipped (not in --methods).")
    elif primary is None:
        print("\n  [1/7] TTA — skipped (no primary model).")
    else:
        print(f"\n  [1/7] TTA  (model={primary}, n_passes={args.n_tta})")
        try:
            u = calculate_tta_uncertainty(
                regressor.models[primary], image, n_passes=args.n_tta)
            unc['TTA'] = u
            print(f"        shape={u.shape}  μ={u.mean():.4f}  max={u.max():.4f}")
        except Exception as exc:
            print(f"        FAILED: {exc}")

    # ── 2. MCD ────────────────────────────────────────────────────────────────
    smirk_mcd_wrapper = None

    if 'mcd' not in active_methods and 'sol_mcd' not in active_methods \
            and 'amcd' not in active_methods:
        print("\n  [2/7] MC Dropout — skipped (not in --methods).")
    else:
        # Load the MCD wrapper once — reused for SOL-MCD and A-MCD.
        # Wrapper loads from smirk_checkpoint_data/trained.pt; raises FileNotFoundError
        # if the checkpoint or encoder definition is absent.
        try:
            from wrappers.smirk_wrapper import SMIRKWrapper
            smirk_mcd_wrapper = SMIRKWrapper(device=device, use_mcd_checkpoint=True)
            smirk_mcd_wrapper.enable_dropout_for_inference()
            print("        [ok] MCD checkpoint loaded from smirk_checkpoint_data/trained.pt")
        except Exception as exc:
            print(f"  [2/7] MC Dropout wrapper failed to load: {exc}")
            smirk_mcd_wrapper = None

        if 'mcd' not in active_methods:
            print("\n  [2/7] MCD — skipped (not in --methods).")
        elif smirk_mcd_wrapper is None:
            print("\n  [2/7] MCD — skipped (wrapper unavailable).")
        else:
            print(f"\n  [2/7] MC Dropout  (n_passes={args.n_mcd})")
            try:
                u = calculate_mcd_uncertainty(
                    smirk_mcd_wrapper, image, n_passes=args.n_mcd)
                unc['MCD'] = u
                print(f"        shape={u.shape}  μ={u.mean():.4f}  max={u.max():.4f}")
            except Exception as exc:
                print(f"        FAILED: {exc}")

    # ── 3. Cross-Method Disagreement ──────────────────────────────────────────
    available_models = list(mesh_results.keys())
    if 'crossmethod' not in active_methods:
        print("\n  [3/7] CrossMethod — skipped (not in --methods).")
    else:
        print(f"\n  [3/7] Cross-Method Disagreement  (models={available_models})")
        if len(available_models) >= 2:
            try:
                # Use pre-computed vertices so all models need not be in RAM at once
                u = calculate_cross_method_disagreement(
                    vertices_dict=mesh_results,
                    normalise=True,
                )
                unc['CrossMethod'] = u
                print(f"        shape={u.shape}  μ={u.mean():.4f}  max={u.max():.4f}")
            except Exception as exc:
                print(f"        FAILED: {exc}")
        else:
            print("        Skipped — need ≥2 loaded models.")

    # ── 4. Jacobian Sensitivity ───────────────────────────────────────────────
    if 'jacobian' not in active_methods:
        print("\n  [4/7] Jacobian — skipped (not in --methods).")
    elif primary is None:
        print("\n  [4/7] Jacobian — skipped (no primary model).")
    else:
        print(f"\n  [4/7] Jacobian  (model={primary}, n_directions={args.n_jacobian})")
        try:
            u = calculate_jacobian_sensitivity_uncertainty(
                regressor.models[primary], image,
                n_directions=args.n_jacobian)
            unc['Jacobian'] = u
            print(f"        shape={u.shape}  μ={u.mean():.4f}  max={u.max():.4f}")
        except Exception as exc:
            print(f"        FAILED: {exc}")

    # ── 5. Mahalanobis Distance ───────────────────────────────────────────────
    if 'mahalanobis' not in active_methods:
        print("\n  [5/7] Mahalanobis — skipped (not in --methods).")
    elif primary is None:
        print("\n  [5/7] Mahalanobis — skipped (no primary model).")
    elif not reference_images or len(reference_images) < 4:
        n_ref = len(reference_images) if reference_images else 0
        print(f"\n  [5/7] Mahalanobis — skipped "
              f"(need ≥4 reference images, have {n_ref}).")
    else:
        ref = reference_images   # use all loaded refs for best covariance estimation
        print(f"\n  [5/7] Mahalanobis  (model={primary}, n_ref={len(ref)})")
        try:
            u = calculate_mahalanobis_uncertainty(
                regressor.models[primary], image,
                reference_images=ref)
            unc['Mahalanobis'] = u
            print(f"        shape={u.shape}  μ={u.mean():.4f}  max={u.max():.4f}")
        except Exception as exc:
            print(f"        FAILED: {exc}")

    # ── 6. SOL-MCD ────────────────────────────────────────────────────────────
    if 'sol_mcd' not in active_methods:
        print("\n  [6/7] SOL-MCD — skipped (not in --methods).")
    elif smirk_mcd_wrapper is None:
        print("\n  [6/7] SOL-MCD — skipped (MCD wrapper unavailable).")
    else:
        print(f"\n  [6/7] SOL-MCD  (n_passes={args.n_mcd}, n_stable=1)")
        try:
            u = calculate_sol_mcd_uncertainty(
                smirk_mcd_wrapper, image,
                n_passes=args.n_mcd, n_stable_layers=1)
            unc['SOL-MCD'] = u
            print(f"        shape={u.shape}  μ={u.mean():.4f}  max={u.max():.4f}")
        except Exception as exc:
            print(f"        FAILED: {exc}")

    # ── 7. A-MCD ──────────────────────────────────────────────────────────────
    if 'amcd' not in active_methods:
        print("\n  [7/7] A-MCD — skipped (not in --methods).")
    elif smirk_mcd_wrapper is None:
        print("\n  [7/7] A-MCD — skipped (MCD wrapper unavailable).")
    else:
        print(f"\n  [7/7] A-MCD  (n_pairs={args.n_mcd})")
        try:
            u = calculate_antithetic_mcd_uncertainty(
                smirk_mcd_wrapper, image, n_pairs=args.n_mcd)
            unc['A-MCD'] = u
            print(f"        shape={u.shape}  μ={u.mean():.4f}  max={u.max():.4f}")
        except Exception as exc:
            print(f"        FAILED: {exc}")

    # ── Normalise all uncertainty maps to [0, 1] ──────────────────────────────
    # Different methods output vastly different magnitude scales:
    #   TTA ≈ 1e-4,  Jacobian ≈ 1e-2 or larger,  CrossMethod ≈ 1e-3, …
    # Min-max normalisation makes cross-method visualisation and downstream
    # confidence maps comparable.  Rank-based metrics (Spearman ρ, AUSE) are
    # unaffected.  ECE/NLL are computed on normalised values — they remain
    # informative as a calibration measure on the [0, 1] scale.
    for method in list(unc):
        u = unc[method].astype(np.float32)
        u_min, u_max = float(u.min()), float(u.max())
        if u_max > u_min:
            u = (u - u_min) / (u_max - u_min)
        unc[method] = u.astype(np.float32)
        print(f"  [norm] {method}: [{u_min:.4e}, {u_max:.4e}] → [0, 1]")

    # ── Persist uncertainty maps ───────────────────────────────────────────────
    if unc:
        unc_dir = os.path.join(args.output_dir, 'uncertainty')
        os.makedirs(unc_dir, exist_ok=True)
        for method, u in unc.items():
            np.save(os.path.join(unc_dir, f'{method}_uncertainty.npy'), u)
        print(f"\n  Maps saved → {unc_dir}/")

    n_total = len(active_methods)
    print(f"\n[Stage 3] Uncertainty complete.  "
          f"{len(unc)}/{n_total} methods succeeded: {list(unc)}")
    return unc


# ══════════════════════════════════════════════════════════════════════════════
# Per-image gallery helper
# ══════════════════════════════════════════════════════════════════════════════

def _run_per_image_gallery(
    args: argparse.Namespace,
    regressor: "UnifiedFaceRegressor",
    gt_data: List[dict],
    flame_faces: np.ndarray,
    active_methods: set,
    device: str,
) -> None:
    """
    For each dataset image that has a real face photo:
      1. Run inference for ALL requested models sequentially (reload each to
         avoid requiring all 4 wrappers in RAM simultaneously).
      2. Compute TTA uncertainty on the primary model (up to 5 passes).
      3. Save per-image [photo | mesh | uncertainty] for the primary model.
      4. Save a multi-model comparison gallery (photo + all models).
      5. Save a combined TTA uncertainty gallery sheet.

    Models are loaded one-at-a-time and freed after their inference pass,
    matching the memory strategy used in Stage 2 and the batch eval.
    """
    import gc

    items_with_images = [s for s in gt_data if s.get('image') is not None]
    if not items_with_images:
        print("  [gallery] No dataset images with photos — gallery skipped.")
        return

    primary_name = args.primary_model
    # Fall back to the first still-loaded model if primary isn't available
    if primary_name not in regressor.models and regressor.models:
        primary_name = next(iter(regressor.models))

    n_images    = len(items_with_images)
    do_tta      = 'tta' in active_methods
    n_tta_fast  = args.n_tta
    gallery_dir = os.path.join(args.output_dir, 'plots', 'per_image')
    os.makedirs(gallery_dir, exist_ok=True)

    print(f"\n  Per-image gallery: {n_images} images, models={args.models}"
          + (f", TTA n={n_tta_fast}" if do_tta else ", no uncertainty"))

    # ── Step 1: collect vertices for EVERY model by sequential loading ─────────
    # {model_name: [verts_or_None, …]} indexed parallel to items_with_images
    all_model_verts: Dict[str, List] = {}

    for mname in args.models:
        print(f"    Gallery inference  {mname} …", end=' ', flush=True)
        try:
            r = UnifiedFaceRegressor(device=device, models=[mname])
            if mname not in r.models:
                print("not loaded — skipped.")
                del r
                continue
            wrapper = r.models[mname]
            row: List = []
            n_ok = 0
            for sample in items_with_images:
                img_224 = cv2.resize(
                    np.asarray(sample['image'], dtype=np.uint8), (224, 224))
                try:
                    v = wrapper.get_vertices(img_224)
                    row.append(v)
                    n_ok += 1
                except Exception:
                    row.append(None)
            all_model_verts[mname] = row
            del r; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"{n_ok}/{n_images} ok")
        except Exception as exc:
            print(f"FAILED ({exc})")

    if not all_model_verts:
        print("  [gallery] All model loads failed — gallery skipped.")
        return

    # Ensure primary_name is among the models that actually loaded
    if primary_name not in all_model_verts:
        primary_name = next(iter(all_model_verts))

    # ── Step 2: TTA uncertainty for the primary model (still in regressor) ─────
    primary_uncertainty: List[Optional[np.ndarray]] = [None] * n_images
    if do_tta and primary_name in regressor.models:
        for i, sample in enumerate(items_with_images):
            img_224 = cv2.resize(
                np.asarray(sample['image'], dtype=np.uint8), (224, 224))
            try:
                primary_uncertainty[i] = calculate_tta_uncertainty(
                    regressor.models[primary_name], img_224, n_passes=n_tta_fast)
            except Exception:
                pass

    # ── Step 3: per-image rows + individual PNG saves ──────────────────────────
    tta_gallery_rows: List = []
    model_gallery_rows: List = []

    for i, sample in enumerate(items_with_images):
        img_224  = cv2.resize(np.asarray(sample['image'], dtype=np.uint8), (224, 224))
        subj     = sample.get('subject_id', 'sample')
        cond     = sample.get('condition', '')
        unc      = primary_uncertainty[i]

        primary_verts = all_model_verts.get(primary_name, [None] * n_images)[i]
        if primary_verts is None:
            print(f"    [{subj[:20]}] primary ({primary_name}) produced no vertices — skip.")
            continue

        # Build {model: verts} dict for this image
        per_image_verts = {
            mname: all_model_verts[mname][i]
            for mname in all_model_verts
            if all_model_verts[mname][i] is not None
        }

        tta_gallery_rows.append((img_224, primary_verts, unc))
        model_gallery_rows.append((img_224, per_image_verts, unc))

        idx      = len(tta_gallery_rows)
        fname    = f"{idx:03d}_{subj[:20].replace('/', '_')}.png"
        row_path = os.path.join(gallery_dir, fname)
        title    = f"{primary_name}  ·  {subj[:35]}  ·  {cond}"
        try:
            unc_label = 'TTA Uncertainty' if unc is not None else 'N/A'
            plot_image_uncertainty_row(
                img_224, primary_verts, flame_faces, unc, row_path,
                title=title, uncertainty_label=unc_label)
        except Exception as exc:
            print(f"    [{subj[:20]}] row plot failed: {exc}")

    if not tta_gallery_rows:
        return

    # ── Combined TTA uncertainty gallery (primary model) ─────────────────────
    tta_gallery_path = os.path.join(args.output_dir, 'plots', 'per_image_gallery.png')
    try:
        plot_per_image_uncertainty_gallery(
            tta_gallery_rows, flame_faces, tta_gallery_path,
            model_name=primary_name)
        print(f"    TTA gallery: {len(tta_gallery_rows)} images → {tta_gallery_path}")
    except Exception as exc:
        print(f"    TTA gallery failed: {exc}")

    # ── All-models comparison gallery ─────────────────────────────────────────
    if len(all_model_verts) > 1:
        cmp_gallery_path = os.path.join(
            args.output_dir, 'plots', 'per_image_model_comparison_gallery.png')
        try:
            plot_model_comparison_gallery(
                model_gallery_rows, flame_faces, cmp_gallery_path,
                primary_model=primary_name)
            print(f"    Model comparison gallery → {cmp_gallery_path}")
        except Exception as exc:
            print(f"    Model comparison gallery failed: {exc}")

    # ── Paper-style comparison panel (dark background, high DPI) ─────────────
    if len(all_model_verts) >= 1 and model_gallery_rows:
        paper_path = os.path.join(
            args.output_dir, 'plots', 'paper_style_comparison.png')
        try:
            paper_rows = [(r[0], r[1]) for r in model_gallery_rows if r[0] is not None]
            plot_paper_style_comparison_panel(
                paper_rows, flame_faces, paper_path,
                model_order=list(all_model_verts.keys()),
            )
            print(f"    Paper-style panel → {paper_path}")
        except Exception as exc:
            print(f"    Paper-style panel failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Batch dataset evaluation (replaces single-image metrics)
# ══════════════════════════════════════════════════════════════════════════════

def _run_batch_dataset_eval(
    args: argparse.Namespace,
    gt_data: List[dict],
    device: str,
    active_methods: set,
    reference_images: Optional[List[np.ndarray]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List]]:
    """
    Run inference + uncertainty for every dataset image with a face photo,
    then aggregate per-vertex errors and uncertainties across all samples.

    Returns
    -------
    error_dict      : {model_name : (5023,) mean per-vertex error}
    unc_dict        : {method_name: (5023,) mean per-vertex uncertainty}
    per_image_pairs : {method_name: list of (error_5023, unc_5023) tuples,
                       one per GT-paired image} — used for per-image Spearman ρ

    All three dicts may be empty when no paired data is available.
    """
    import gc

    samples = [s for s in (gt_data or []) if s.get('image') is not None]
    if not samples:
        print("  [batch eval] No samples with images — skipping.")
        return {}, {}, {}

    N = len(samples)
    print(f"\n  [batch eval] {N} images × {args.models} models")

    images_224 = [
        cv2.resize(np.asarray(s['image'], dtype=np.uint8), (224, 224))
        for s in samples
    ]
    gt_verts_list = []
    for s in samples:
        gv = s.get('gt_vertices')
        if gv is not None:
            arr = np.asarray(gv, dtype=np.float32)
            gt_verts_list.append(arr if arr.shape == (5023, 3) else None)
        else:
            gt_verts_list.append(None)

    # ── Step 1: per-model inference over all images ───────────────────────────
    all_verts: Dict[str, List] = {}   # {model_name: [array|None, …]}

    for name in args.models:
        print(f"    {name}: batch inference …", end=' ', flush=True)
        try:
            r = UnifiedFaceRegressor(device=device, models=[name])
            if name not in r.models:
                print("not loaded — skipped.")
                del r
                continue
            wrapper = r.models[name]
            row, n_ok = [], 0
            for img in images_224:
                try:
                    v = np.asarray(wrapper.get_vertices(img), dtype=np.float32)
                    row.append(v)
                    n_ok += 1
                except Exception:
                    row.append(None)
            all_verts[name] = row
            del r; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"{n_ok}/{N} ok")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if not all_verts:
        return {}, {}, {}

    primary_name = args.primary_model
    if primary_name not in all_verts:
        primary_name = next(iter(all_verts))

    # ── Step 2: paired geometric error per sample ─────────────────────────────
    err_acc: Dict[str, List[np.ndarray]] = {m: [] for m in all_verts}
    # gt_paired_indices[j] = image index i of the j-th GT-paired sample; used
    # to align per-image uncertainty (indexed over all N images) with per-image
    # error (indexed over GT images only) when computing per-image Spearman ρ.
    gt_paired_indices: List[int] = []
    n_paired = 0
    for i, gt_v in enumerate(gt_verts_list):
        if gt_v is None:
            continue
        n_paired += 1
        gt_paired_indices.append(i)
        for m in all_verts:
            pred_v = all_verts[m][i]
            if pred_v is not None:
                err_acc[m].append(calculate_geometric_error(pred_v, gt_v, align=True))

    error_dict: Dict[str, np.ndarray] = {}
    for m, errs in err_acc.items():
        if errs:
            error_dict[m] = np.mean(np.stack(errs), axis=0)
            rmse = float(np.sqrt(np.mean(error_dict[m] ** 2)))
            med  = float(np.median(error_dict[m]))
            print(f"    {m:<10} RMSE={rmse:.4f}  median={med:.4f}  "
                  f"({len(errs)}/{N} paired)")

    if not error_dict:
        print("  [batch eval] No GT pairs found — metrics will be skipped.")

    # ── Step 3: CrossMethod per image ─────────────────────────────────────────
    # Accumulators are dicts keyed by image index so that per-image pairing
    # (for per-image Spearman ρ) is correct even when individual images fail.
    cross_acc: Dict[int, np.ndarray] = {}
    if 'crossmethod' in active_methods and len(all_verts) >= 2:
        for i in range(N):
            mesh_dict = {m: all_verts[m][i]
                         for m in all_verts if all_verts[m][i] is not None}
            if len(mesh_dict) >= 2:
                try:
                    u = calculate_cross_method_disagreement(vertices_dict=mesh_dict, normalise=True)
                    cross_acc[i] = np.asarray(u, dtype=np.float32).ravel()
                except Exception:
                    pass

    # ── Step 4: TTA / Jacobian / Mahalanobis over all images (primary only) ───
    tta_acc:   Dict[int, np.ndarray] = {}
    jac_acc:   Dict[int, np.ndarray] = {}
    mahal_acc: Dict[int, np.ndarray] = {}

    need_primary = (
        ('tta'         in active_methods) or
        ('jacobian'    in active_methods) or
        ('mahalanobis' in active_methods)
    )
    if need_primary:
        print(f"    {primary_name}: loading for uncertainty over {N} images …",
              end=' ', flush=True)
        try:
            p_reg = UnifiedFaceRegressor(device=device, models=[primary_name])
            p_wrap = p_reg.models.get(primary_name)
            print("ok")
        except Exception as exc:
            print(f"FAILED: {exc}")
            p_wrap = None

        if p_wrap is not None:
            n_tta_batch = args.n_tta
            n_jac_batch = args.n_jacobian
            for i, img in enumerate(images_224):
                if 'tta' in active_methods:
                    try:
                        u = calculate_tta_uncertainty(p_wrap, img, n_passes=n_tta_batch)
                        tta_acc[i] = np.asarray(u, dtype=np.float32).ravel()
                    except Exception:
                        pass
                if 'jacobian' in active_methods:
                    try:
                        u = calculate_jacobian_sensitivity_uncertainty(
                            p_wrap, img, n_directions=n_jac_batch)
                        jac_acc[i] = np.asarray(u, dtype=np.float32).ravel()
                    except Exception:
                        pass
                if ('mahalanobis' in active_methods
                        and reference_images and len(reference_images) >= 4):
                    try:
                        u = calculate_mahalanobis_uncertainty(
                            p_wrap, img,
                            reference_images=reference_images)
                        mahal_acc[i] = np.asarray(u, dtype=np.float32).ravel()
                    except Exception:
                        pass
                if (i + 1) % max(1, N // 4) == 0 or i == N - 1:
                    print(f"    uncertainty: {i+1}/{N} images done")
            del p_reg; gc.collect()

    # ── Step 5: aggregate and normalise ───────────────────────────────────────
    unc_dict: Dict[str, np.ndarray] = {}
    if tta_acc:
        unc_dict['TTA']         = np.mean(np.stack(list(tta_acc.values())),   axis=0)
    if cross_acc:
        unc_dict['CrossMethod'] = np.mean(np.stack(list(cross_acc.values())), axis=0)
    if jac_acc:
        unc_dict['Jacobian']    = np.mean(np.stack(list(jac_acc.values())),   axis=0)
    if mahal_acc:
        unc_dict['Mahalanobis'] = np.mean(np.stack(list(mahal_acc.values())), axis=0)

    # Normalise aggregated maps to [0, 1] for cross-method comparability.
    for k in list(unc_dict):
        u = unc_dict[k].astype(np.float32)
        u_min, u_max = float(u.min()), float(u.max())
        if u_max > u_min:
            u = (u - u_min) / (u_max - u_min)
        unc_dict[k] = u.astype(np.float32)

    # ── Step 6: build per-image (error, uncertainty) pairs for per-image ρ ──────
    # For each GT-paired image j (at original image index gt_paired_indices[j]):
    #   - error comes from err_acc[primary][j]    (already computed in Step 2)
    #   - uncertainty comes from unc_acc[i] where i = gt_paired_indices[j]
    # CrossMethod uses mean error across all models as its reference.
    per_image_pairs: Dict[str, List] = {}

    primary_errs = err_acc.get(primary_name, [])

    # Mean-error reference for CrossMethod (per GT image)
    if len(err_acc) >= 2:
        mean_errs_gt: List[Optional[np.ndarray]] = []
        for j in range(len(gt_paired_indices)):
            per_model = [err_acc[m][j] for m in err_acc
                         if j < len(err_acc[m])]
            if per_model:
                mean_errs_gt.append(np.mean(np.stack(per_model), axis=0))
            else:
                mean_errs_gt.append(None)
    else:
        mean_errs_gt = [primary_errs[j] if j < len(primary_errs) else None
                        for j in range(len(gt_paired_indices))]

    def _build_pairs(unc_acc: Dict[int, np.ndarray],
                     ref_errs: List[Optional[np.ndarray]]) -> List:
        pairs = []
        for j, i in enumerate(gt_paired_indices):
            if i in unc_acc and j < len(ref_errs) and ref_errs[j] is not None:
                pairs.append((ref_errs[j].ravel(), unc_acc[i].ravel()))
        return pairs

    if tta_acc and primary_errs:
        p = _build_pairs(tta_acc, primary_errs)
        if p:
            per_image_pairs['TTA'] = p
    if cross_acc:
        p = _build_pairs(cross_acc, mean_errs_gt)
        if p:
            per_image_pairs['CrossMethod'] = p
    if jac_acc and primary_errs:
        p = _build_pairs(jac_acc, primary_errs)
        if p:
            per_image_pairs['Jacobian'] = p
    if mahal_acc and primary_errs:
        p = _build_pairs(mahal_acc, primary_errs)
        if p:
            per_image_pairs['Mahalanobis'] = p

    # ── StaticRegion baseline: mean per-vertex error as non-adaptive baseline ──
    # Any live method that cannot beat StaticRegion in AUSE/Spearman ρ is merely
    # recovering dataset-level difficulty, not per-image input-adaptive uncertainty.
    #
    # IMPORTANT: StaticRegion is NOT added to unc_dict.  Adding it there and then
    # calling compare_uncertainty_methods(mean_error, {'StaticRegion': baseline})
    # is circular: static_baseline = mean(primary_errs) = mean_error (up to linear
    # normalisation), so ρ(mean_error, baseline) = 1.0 and AUSE = 0.0 trivially.
    # Instead, StaticRegion is evaluated in per_image_pairs where pair (i) compares
    # image-i's error against the global mean map — a genuinely non-trivial signal
    # (measures whether static difficulty predicts per-image vertex hardness).
    # compare_uncertainty_methods is then called on the concatenated per-image pairs
    # inside run_evaluation after the batch results are returned.
    if len(primary_errs) >= 2:
        static_baseline = compute_static_region_baseline(primary_errs)
        # Normalise to [0, 1] for display consistency with other uncertainty maps.
        sb_min, sb_max = float(static_baseline.min()), float(static_baseline.max())
        if sb_max > sb_min:
            static_baseline = (static_baseline - sb_min) / (sb_max - sb_min)
        # Per-image pairs: same (5023,) baseline paired with each image's own error.
        static_pairs = [(err.ravel(), static_baseline.ravel()) for err in primary_errs]
        per_image_pairs['StaticRegion'] = static_pairs
        print(f"    StaticRegion: non-adaptive baseline built from "
              f"{len(primary_errs)} images  (evaluated via per-image pairs only)")

    print(f"  [batch eval] Complete — "
          f"{len(error_dict)} error maps, {len(unc_dict)} uncertainty maps, "
          f"{sum(len(v) for v in per_image_pairs.values())} per-image pairs")
    return error_dict, unc_dict, per_image_pairs


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4: Evaluation & Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    args: argparse.Namespace,
    mesh_results: Dict[str, np.ndarray],
    uncertainty_results: Dict[str, np.ndarray],
    flame_faces: Optional[np.ndarray],
    gt_data: Optional[List[dict]],
    regressor: Optional["UnifiedFaceRegressor"] = None,
    image: Optional[np.ndarray] = None,
    active_methods: Optional[set] = None,
    device: str = 'cpu',
    reference_images: Optional[List[np.ndarray]] = None,
) -> dict:
    """
    Compute geometric error (if GT vertices available) and all uncertainty
    quality metrics, then generate the full visualisation report.

    Evaluation summary
    ------------------
    • Geometric quality : per-vertex RMSE, median L2 distance (mm)
    • Uncertainty quality: Spearman ρ, AUSE, ECE, NLL  (require GT error)
    • Visualisation     : 10+ plot types saved to output_dir/plots/

    When GT is absent only the uncertainty distribution plots are generated;
    all error-dependent metrics are skipped gracefully.
    """
    print("\n" + "=" * 60)
    print("[Stage 4] Evaluation & Visualisation")
    print("=" * 60)

    primary = args.primary_model
    if primary not in mesh_results and mesh_results:
        primary = next(iter(mesh_results))

    # ── Dataset capability report ─────────────────────────────────────────────
    has_gt = bool(gt_data and any(
        s.get('gt_vertices') is not None for s in gt_data))
    has_images = bool(gt_data and any(
        s.get('image') is not None for s in gt_data))
    if args.dataset == 'tempeh':
        print("\n  [note] TEMPEH uses near-infrared images; SMIRK (RGB MobileNet backbone) "
              "may produce mesh artifacts. SHeaP and DECA are more reliable on this dataset.")

    print(f"\n  Dataset capabilities  (dataset={args.dataset}):")
    print(f"    GT vertices available : {'YES — geometric + quality metrics enabled' if has_gt else 'NO  — geometric error and Spearman/AUSE/ECE/NLL require GT and will be skipped'}")
    print(f"    Images available      : {'YES — per-image gallery and batch uncertainty enabled' if has_images else 'NO'}")
    print(f"    Uncertainty maps      : always computed (TTA / Jacobian / CrossMethod)")
    if not has_gt:
        print("  NOTE: Datasets without ground-truth FLAME vertices cannot produce geometric")
        print("        error or uncertainty quality metrics (Spearman ρ, AUSE, ECE, NLL).")
        print("        UTKFace, LFW, RAF-DB: no GT at all.")
        print("        NoW: raw 3-D scans only → scan-to-mesh distance available but not")
        print("        vertex-level Spearman/AUSE/ECE/NLL.")
        print("        Use --dataset tempeh or --dataset coma for full metric evaluation.")

    # ── Geometric error (requires GT vertices) ────────────────────────────────
    error_dict: Dict[str, np.ndarray] = {}
    gt_verts_single: Optional[np.ndarray] = None

    if gt_data:
        valid_gt = [
            s for s in gt_data
            if s.get('gt_vertices') is not None
               and np.asarray(s['gt_vertices']).shape == (5023, 3)
        ]
        if valid_gt and mesh_results:
            # Geometric error is computed correctly (image-paired) in the batch
            # eval below.  Stage 2 mesh_results contain predictions for a single
            # random test image, so comparing them against the full GT set would
            # pair the wrong identities and produce invalid, run-to-run-variable
            # numbers.  Rely on batch eval exclusively.
            print(f"  {len(valid_gt)} GT samples available — "
                  f"geometric error computed via paired batch evaluation below.")
        elif valid_gt:
            print(f"  {len(valid_gt)} GT samples found; no model predictions — "
                  "geometric error skipped.")

        # Collect first valid GT for visualisation (e.g. scatter plot reference)
        for sample in valid_gt:
            gt_v = np.asarray(sample['gt_vertices'], dtype=np.float32)
            if gt_verts_single is None:
                gt_verts_single = gt_v
                break
    else:
        print("  No GT data available — geometric evaluation skipped.")

    primary_error: Optional[np.ndarray] = error_dict.get(primary)

    # ── Batch dataset evaluation (replaces single-image metrics) ──────────────
    # Run all dataset images through inference and uncertainty, pair each
    # prediction_i with its GT_i, and aggregate.  Results override the
    # single-image fallback for Spearman / AUSE / ECE / NLL.
    batch_error_dict: Dict[str, np.ndarray] = {}
    batch_unc_dict:   Dict[str, np.ndarray] = {}
    per_image_pairs:  Dict[str, List]       = {}
    if gt_data:
        dataset_has_images = any(s.get('image') is not None for s in gt_data)
        if dataset_has_images:
            print("\n  Running batch dataset evaluation (all images)…")
            batch_error_dict, batch_unc_dict, per_image_pairs = _run_batch_dataset_eval(
                args, gt_data, device,
                active_methods if active_methods is not None else set(_ALL_METHODS),
                reference_images,
            )
            if batch_error_dict:
                error_dict = batch_error_dict
                # Auto-select the model with the lowest median per-vertex error as
                # primary — avoids using a weak model (e.g. SMIRK on NIR data) as
                # the reference for uncertainty quality metrics.
                best_model = min(
                    batch_error_dict,
                    key=lambda m: float(np.median(batch_error_dict[m]))
                )
                if best_model != primary:
                    print(f"  [auto] Switching primary model: {primary} → {best_model} "
                          f"(median error {np.median(batch_error_dict[best_model]):.3f} mm "
                          f"< {np.median(batch_error_dict[primary]):.3f} mm)")
                    primary = best_model
                if primary in batch_error_dict:
                    primary_error = batch_error_dict[primary]
                elif batch_error_dict:
                    primary_error = next(iter(batch_error_dict.values()))

    # Merge batch uncertainty into uncertainty_results for metrics; keep
    # single-image uncertainty_results for visualisation (spatial heatmaps).
    metrics_unc = dict(uncertainty_results)
    if batch_unc_dict:
        metrics_unc.update(batch_unc_dict)

    # ── Uncertainty quality metrics (require GT error) ─────────────────────────
    summary_dict:       Dict[str, Dict[str, float]] = {}
    sparsification_dict: Dict[str, dict]            = {}
    calibration_dict:    Dict[str, dict]            = {}
    n_ablation_ause:     Dict[str, Dict[int, float]] = {}   # populated below if TTA active

    if primary_error is not None and metrics_unc:
        e_primary = primary_error.ravel().astype(np.float64)

        # CrossMethod disagreement measures divergence across all regressors, so
        # its natural reference is the mean per-vertex error across all models —
        # not just the primary model's error (which would introduce a directional
        # bias toward whichever model was selected as primary).
        if len(error_dict) >= 2:
            mean_error_all = np.mean(
                np.stack([v.ravel() for v in error_dict.values()]), axis=0
            ).astype(np.float64)
            print(f"\n  Uncertainty quality metrics "
                  f"(CrossMethod vs. mean-model error; others vs. {primary} error) …")
        else:
            mean_error_all = e_primary
            print(f"\n  Uncertainty quality metrics vs. {primary} error …")

        # Split into CrossMethod (vs mean error) and all others (vs primary error).
        cross_unc = {m: np.asarray(u, dtype=np.float64).ravel()
                     for m, u in metrics_unc.items() if m == 'CrossMethod'}
        other_unc = {m: np.asarray(u, dtype=np.float64).ravel()
                     for m, u in metrics_unc.items() if m != 'CrossMethod'}

        if cross_unc:
            summary_cross = compare_uncertainty_methods(mean_error_all, cross_unc)
            for m, u in cross_unc.items():
                sparsification_dict[m] = calculate_sparsification_error_curve(mean_error_all, u)
                calibration_dict[m]    = calculate_uncertainty_calibration(mean_error_all, u)
        else:
            summary_cross = {}

        if other_unc:
            summary_other = compare_uncertainty_methods(e_primary, other_unc)
            for m, u in other_unc.items():
                sparsification_dict[m] = calculate_sparsification_error_curve(e_primary, u)
                calibration_dict[m]    = calculate_uncertainty_calibration(e_primary, u)
        else:
            summary_other = {}

        summary_dict = {**summary_other, **summary_cross}

        # ── StaticRegion from concatenated per-image pairs (non-circular) ─────
        # compare_uncertainty_methods(mean_error, static_baseline) is circular
        # because static_baseline IS mean_error.  Instead, concatenate all
        # per-image (error_i, static_baseline) pairs: this compares each image's
        # own error distribution against the globally-averaged difficulty map,
        # giving a meaningful (and non-trivially achievable) reference score.
        if 'StaticRegion' in per_image_pairs:
            sr_all_e = np.concatenate(
                [p[0] for p in per_image_pairs['StaticRegion']]
            ).astype(np.float64)
            sr_all_u = np.concatenate(
                [p[1] for p in per_image_pairs['StaticRegion']]
            ).astype(np.float64)
            sr_metrics = compare_uncertainty_methods(
                sr_all_e, {'StaticRegion': sr_all_u}
            )
            summary_dict.update(sr_metrics)

        # Console summary
        print(f"\n  {'Method':<16} {'Spearman ρ':>11} {'AUSE':>9} {'NLL':>9}")
        print("  " + "-" * 48)
        for method, mets in summary_dict.items():
            ref_label = "(mean)" if method == 'CrossMethod' and len(error_dict) >= 2 else ""
            print(f"  {method:<16} "
                  f"{mets.get('spearman_rho', float('nan')):>11.4f} "
                  f"{mets.get('ause',         float('nan')):>9.4f} "
                  f"{mets.get('nll',          float('nan')):>9.4f}"
                  f"  {ref_label}")

        metrics_path = os.path.join(args.output_dir, 'evaluation', 'metrics.json')
        _save_json(summary_dict, metrics_path)

        # ── Per-image Spearman ρ (N×5023 concatenated) ──────────────────────
        # The summary_dict above computes ρ on the mean-over-N spatial maps,
        # which can be inflated by the spatial structure of the face.  The flat
        # ρ concatenates all N×5023 per-vertex pairs and therefore measures true
        # per-image predictive power of each uncertainty method.
        if per_image_pairs:
            try:
                from scipy.stats import spearmanr as _spearmanr
                print(f"\n  Per-image Spearman ρ  "
                      f"({'N×5023 concat' if per_image_pairs else ''}):")
                print(f"  {'Method':<16} {'flat ρ (N×5023)':>16} "
                      f"{'per-img ρ mean':>16} {'per-img ρ std':>14}")
                print("  " + "-" * 65)
                for method, pairs in per_image_pairs.items():
                    # flat ρ: concatenate all pairs
                    all_e = np.concatenate([p[0] for p in pairs]).astype(np.float64)
                    all_u = np.concatenate([p[1] for p in pairs]).astype(np.float64)
                    flat_rho = float(_spearmanr(all_e, all_u).statistic)

                    # per-image ρ: compute ρ per image, then mean ± std
                    per_img_rhos = []
                    for e_img, u_img in pairs:
                        if len(e_img) >= 4:
                            r = float(_spearmanr(
                                e_img.astype(np.float64),
                                u_img.astype(np.float64)
                            ).statistic)
                            if not np.isnan(r):
                                per_img_rhos.append(r)
                    if per_img_rhos:
                        pi_mean = float(np.mean(per_img_rhos))
                        pi_std  = float(np.std(per_img_rhos))
                    else:
                        pi_mean = float('nan')
                        pi_std  = float('nan')

                    print(f"  {method:<16} {flat_rho:>16.4f} "
                          f"{pi_mean:>16.4f} {pi_std:>14.4f}")

                    # Append flat_rho into summary_dict for JSON export
                    if method in summary_dict:
                        summary_dict[method]['flat_spearman_rho']     = flat_rho
                        summary_dict[method]['per_image_rho_mean']    = pi_mean
                        summary_dict[method]['per_image_rho_std']     = pi_std
                        summary_dict[method]['n_image_pairs']         = len(pairs)

                _save_json(summary_dict, metrics_path)
            except Exception as exc:
                print(f"  [warn] Per-image ρ failed: {exc}")

        # ── Kendall's W: rank stability of uncertainty across images ───────────
        # W measures how consistently each method ranks the 5023 FLAME vertices
        # across evaluation images.  High ρ + high W = reliably input-adaptive.
        # High ρ + low W = good on average but not trustworthy per-image.
        if per_image_pairs:
            print(f"\n  Kendall's W (rank stability across images):")
            print(f"  {'Method':<16} {'W':>8}  (1=stable, 0=random; NaN=<2 images)")
            print("  " + "-" * 35)
            for method, pairs in per_image_pairs.items():
                w = calculate_rank_stability(pairs)
                if method in summary_dict:
                    summary_dict[method]['kendall_w'] = w
                w_str = f"{w:.4f}" if not (isinstance(w, float) and w != w) else "NaN"
                print(f"  {method:<16} {w_str:>8}")
            _save_json(summary_dict, metrics_path)

        # ── TTA N ablation: AUSE vs number of passes ─────────────────────────
        # Shows whether N=10 (default) is sufficient or whether more passes
        # would improve the uncertainty estimate.
        if (image is not None
                and regressor is not None
                and primary_error is not None
                and primary in regressor.models
                and active_methods is not None
                and 'tta' in active_methods):
            try:
                print(f"\n  TTA N ablation (N ∈ {{2, 5, 10, 15, 20}}) …",
                      end=' ', flush=True)
                tta_n_unc = calculate_tta_uncertainty_n_ablation(
                    regressor.models[primary], image,
                    n_values=[2, 5, 10, 15, 20])
                tta_n_ause_: Dict[int, float] = {}
                e_ref = primary_error.ravel().astype(np.float64)
                for n_passes, u_arr in tta_n_unc.items():
                    sparse_n = calculate_sparsification_error_curve(
                        e_ref, np.asarray(u_arr, dtype=np.float64).ravel())
                    tta_n_ause_[n_passes] = sparse_n['ause']
                n_ablation_ause['TTA'] = tta_n_ause_
                ause_vals_str = "  ".join(
                    f"N={n}→{a:.4f}" for n, a in sorted(tta_n_ause_.items()))
                print(f"done  [{ause_vals_str}]")
            except Exception as exc:
                print(f"FAILED: {exc}")

    elif metrics_unc:
        print("  No GT → skipping Spearman ρ / AUSE / NLL metrics.")

    # ── Region-wise geometric error breakdown ──────────────────────────────────
    region_errors_dict: Dict[str, Dict] = {}
    if primary_error is not None and regressor is not None:
        region_masks = _try_load_flame_masks(regressor)
        if region_masks:
            print(f"\n  Computing region-wise error for {len(region_masks)} FLAME regions …")
            try:
                region_errors_dict[primary] = calculate_region_wise_geometric_error(
                    primary_error, region_masks
                )
                print(f"    {primary}: "
                      + ", ".join(
                          f"{r}={v['mean_mm']:.3f}mm"
                          for r, v in list(region_errors_dict[primary].items())[:4]
                      ) + " …")
            except Exception as exc:
                print(f"    [warn] Region breakdown failed: {exc}")
        else:
            print("  [info] FLAME_masks.pkl not found — region breakdown skipped.")

    # ── Scan-to-mesh evaluation (NoW-style raw-scan GT) ───────────────────────
    s2m_results: Dict[str, Dict[str, float]] = {}
    if flame_faces is not None and gt_data:
        scan_samples = [
            s for s in gt_data
            if s.get('gt_scan') is not None
            and np.asarray(s['gt_scan']).ndim == 2
            and np.asarray(s['gt_scan']).shape[1] == 3
            and len(s['gt_scan']) > 50
        ]
        if scan_samples:
            print(f"\n  Scan-to-mesh evaluation over {len(scan_samples)} raw-scan GT samples …")
            for model_name, pred_v in mesh_results.items():
                medians: List[float] = []
                means:   List[float] = []
                p90s:    List[float] = []
                for sample in scan_samples[:args.partition_size]:
                    gt_scan = np.asarray(sample['gt_scan'], dtype=np.float32)
                    try:
                        s2m = calculate_scan_to_mesh_distance(
                            pred_v, flame_faces, gt_scan, align=True)
                        medians.append(s2m['median_mm'])
                        means.append(s2m['mean_mm'])
                        p90s.append(s2m['p90_mm'])
                    except Exception:
                        pass
                if medians:
                    s2m_results[model_name] = {
                        'median_mm': float(np.mean(medians)),
                        'mean_mm':   float(np.mean(means)),
                        'p90_mm':    float(np.mean(p90s)),
                    }
                    print(f"    {model_name:<10}  s2m_median={s2m_results[model_name]['median_mm']:.3f} mm"
                          f"  s2m_p90={s2m_results[model_name]['p90_mm']:.3f} mm")

            if s2m_results:
                s2m_path = os.path.join(args.output_dir, 'evaluation', 's2m_metrics.json')
                _save_json(s2m_results, s2m_path)

    # ── Per-image gallery (real face photos from dataset) ─────────────────────
    if regressor is not None and flame_faces is not None and gt_data:
        _run_per_image_gallery(
            args, regressor, gt_data, flame_faces,
            active_methods=active_methods if active_methods is not None
            else set(_ALL_METHODS),
            device=device,
        )

    # ── Full visualisation report ──────────────────────────────────────────────
    plots_dir = os.path.join(args.output_dir, 'plots')
    print(f"\n  Generating plots → {plots_dir}/")

    unc_flat = {m: np.asarray(u, dtype=np.float32).ravel()
                for m, u in uncertainty_results.items()}

    # Standalone plots that create_full_analysis_report also generates, but
    # called here explicitly so they appear in the plots_dir even when the
    # report itself is skipped due to a partial run.
    if len(unc_flat) >= 2:
        try:
            plot_method_correlation_matrix(
                unc_flat,
                os.path.join(plots_dir, 'method_correlation_matrix.png'),
            )
        except Exception as _exc:
            warnings.warn(f"[vis] Standalone correlation matrix failed: {_exc}")

    create_full_analysis_report(
        output_dir          = plots_dir,
        vertices_dict       = mesh_results,
        faces               = flame_faces,
        uncertainty_dict    = unc_flat,
        error_dict          = error_dict or None,
        summary_dict        = summary_dict or None,
        sparsification_dict = sparsification_dict or None,
        calibration_dict    = calibration_dict or None,
        region_errors_dict  = region_errors_dict or None,
        s2m_results         = s2m_results or None,
        primary_error       = primary_error,
        gt_vertices         = gt_verts_single,
        image               = image,
        per_image_pairs     = per_image_pairs or None,
        n_ablation_ause     = n_ablation_ause or None,
    )

    print("\n[Stage 4] Evaluation complete.")
    return {
        'error_dict':          error_dict,
        'summary_dict':        summary_dict,
        'sparsification_dict': sparsification_dict,
        'calibration_dict':    calibration_dict,
        'region_errors_dict':  region_errors_dict,
        's2m_results':         s2m_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helper: load dataset partition
# ══════════════════════════════════════════════════════════════════════════════

def _load_partition_data(
    args: argparse.Namespace,
    loader: FaceDatasetLoader,
) -> Tuple[Optional[List[np.ndarray]], Optional[List[dict]]]:
    """
    Load the dataset partition for evaluation and reference images for Mahalanobis.

    Loads a combined set of partition_size + n_mahal_ref + 5 items, then splits:
      - first partition_size items → gt_data (test set)
      - remaining items → reference images (resized to ≤256px to save memory)

    This ensures no overlap between test images and Mahalanobis reference images.

    Returns (reference_images, gt_data).  Both may be None if unavailable.
    """
    _MIN_MAHAL_REF = args.n_mahal_ref

    # Load a combined set: test items first, then extra items for Mahalanobis reference.
    # A single loader with a larger subset_size ensures sequential, non-overlapping splits.
    combined_size = args.partition_size + _MIN_MAHAL_REF + 5
    try:
        combined_loader = FaceDatasetLoader(
            data_root='./datasets',
            subset_sizes=[combined_size],
            render_coma=(args.dataset == 'coma'),
        )
        combined_subsets = combined_loader.create_subsets(args.dataset)
        all_items = combined_subsets.get(combined_size, [])
        if not all_items:
            for n in sorted(combined_subsets, reverse=True):
                if combined_subsets[n]:
                    all_items = combined_subsets[n]
                    print(f"  [info] Combined subset size {combined_size} empty; "
                          f"using n={n}.")
                    break
    except Exception as exc:
        print(f"  [warn] Could not load combined partition for '{args.dataset}': {exc}")
        # Fall back to original loader for the test partition only
        try:
            subsets = loader.create_subsets(args.dataset)
            all_items = subsets.get(args.partition_size, [])
            if not all_items:
                for n in sorted(subsets):
                    if subsets[n]:
                        all_items = subsets[n]
                        break
        except Exception:
            return None, None

    if not all_items:
        return None, None

    # Split: first partition_size → test (gt_data); remainder → Mahalanobis reference
    gt_data = all_items[:args.partition_size]
    ref_items = all_items[args.partition_size:]

    # Build reference images at reduced resolution to avoid OOM on large-image datasets.
    reference_images: List[np.ndarray] = []
    for item in ref_items:
        img = item.get('image')
        if img is not None:
            img_np = np.asarray(img, dtype=np.uint8)
            h, w = img_np.shape[:2]
            if max(h, w) > 256:
                scale = 256.0 / max(h, w)
                img_np = cv2.resize(img_np,
                                    (max(1, int(w * scale)), max(1, int(h * scale))),
                                    interpolation=cv2.INTER_AREA)
            reference_images.append(img_np)

    if len(reference_images) < _MIN_MAHAL_REF:
        print(f"  [warn] Only {len(reference_images)} reference images for Mahalanobis "
              f"(≥{_MIN_MAHAL_REF} recommended); covariance estimates will be noisy.")

    print(f"  [data] Loaded {len(gt_data)} test samples + "
          f"{len(reference_images)} reference images from '{args.dataset}' "
          f"(no overlap between test and reference).")
    return (reference_images if reference_images else None), gt_data


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5: Downstream classifier — RAF-DB expression recognition
# ══════════════════════════════════════════════════════════════════════════════

def _clf_evaluate(
    model: torch.nn.Module,
    loader,
    device: str,
    num_classes: int,
) -> Tuple[float, List[float]]:
    """Return (overall_accuracy %, [per_class_accuracy %, …]).

    Also writes per-class F1 scores to the returned list's .f1 attribute
    so callers can access them without changing the return signature.
    (Stored as a list attribute on the returned list object.)
    """
    model.eval()
    correct = torch.zeros(num_classes)
    total   = torch.zeros(num_classes)
    # Confusion matrix for F1 computation: conf[true][pred]
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0], batch[1]   # ignore scalar weight if present
            images = images.to(device)
            preds  = model(image=images).argmax(dim=1).cpu()
            for i in range(num_classes):
                mask = (labels == i)
                correct[i] += (preds[mask] == labels[mask]).sum()
                total[i]   += mask.sum()
            for t, p in zip(labels.numpy(), preds.numpy()):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    conf_mat[t, p] += 1
    safe = total.clamp(min=1)
    overall = float(correct.sum() / safe.sum() * 100)
    per_cls = [float(correct[i] / safe[i] * 100) for i in range(num_classes)]

    # Per-class F1 from confusion matrix
    f1_scores = []
    for i in range(num_classes):
        tp = int(conf_mat[i, i])
        fp = int(conf_mat[:, i].sum()) - tp
        fn = int(conf_mat[i, :].sum()) - tp
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        f1_scores.append(float(f1))

    result = per_cls
    result_list = list(result)
    result_list_obj = result_list
    # Attach F1 as attribute using a simple wrapper trick
    class _AccList(list):
        pass
    acc_list = _AccList(result_list)
    acc_list.f1 = f1_scores
    acc_list.conf_mat = conf_mat
    return overall, acc_list


def _raf_subset_paths(
    raf_root: str,
    split: str,
    n_cap: Optional[int],
) -> Dict[str, List]:
    """Return {class_name: [Path, …]} sorted and capped at n_cap per class."""
    from pathlib import Path
    from src.emotion_dataset import EXPRESSION_CLASSES, _IMG_EXTS
    root = Path(raf_root) / split
    result: Dict[str, List] = {}
    for cls in EXPRESSION_CLASSES:
        d = root / cls
        if not d.exists():
            result[cls] = []
            continue
        paths = sorted(p for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)
        result[cls] = paths[:n_cap] if n_cap else paths
    return result


def _sample_raf_reference_images(
    raf_root: str,
    n: int = 60,
    seed: int = 42,
) -> List[np.ndarray]:
    """Sample n images uniformly from RAF-DB training split for Mahalanobis reference."""
    from pathlib import Path
    from PIL import Image as PILImage
    from src.emotion_dataset import _IMG_EXTS
    import random as _random
    rng = _random.Random(seed)
    all_paths: List = []
    for cls_dir in sorted((Path(raf_root) / 'train').iterdir()):
        if not cls_dir.is_dir():
            continue
        all_paths.extend(p for p in cls_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)
    if not all_paths:
        raise FileNotFoundError(f"No images found in RAF-DB train split: {raf_root}")
    rng.shuffle(all_paths)
    images: List[np.ndarray] = []
    for p in all_paths[:n]:
        try:
            images.append(np.asarray(PILImage.open(p).convert('RGB'), dtype=np.uint8))
        except Exception:
            pass
    return images


def _precompute_raf_maps(
    wrapper,
    method_name: str,
    raf_root: str,
    maps_root: str,
    n_cap: Optional[int],
    n_tta: int = 5,
    n_jacobian: int = 5,
    n_mcd: int = 15,
    reference_images: Optional[List] = None,
    force_recompute: bool = False,
) -> Tuple[int, int]:
    """
    For each RAF-DB image in the working subset compute the 2-D confidence
    map (projected from 3-D per-vertex uncertainty) and save as .npy.

    Maps are stored at:
        maps_root/{split}/{class}/{stem}.npy

    A manifest.json records the parameters used.  If parameters have changed
    since the last run (or force_recompute=True), all maps are deleted and
    recomputed.  Returns (n_computed, n_cached).
    """
    import shutil
    from pathlib import Path
    from PIL import Image as PILImage
    from src.downstream import project_variance_to_2d

    maps_path = Path(maps_root)

    # ── Manifest check: invalidate stale cache ─────────────────────────────────
    manifest_path = maps_path / 'manifest.json'
    current_params = {
        'method': method_name,
        'n_tta': n_tta,
        'n_jacobian': n_jacobian,
        'n_mcd': n_mcd,
        'n_mahal_ref': len(reference_images) if reference_images else 0,
        'n_cap': n_cap,
    }
    if force_recompute and maps_path.exists():
        shutil.rmtree(str(maps_path))
        print(f"        [force] Deleted existing maps for fresh recompute.")
    elif maps_path.exists() and manifest_path.exists():
        try:
            saved = json.loads(manifest_path.read_text())
            if saved != current_params:
                print(f"        [stale] Map parameters changed {saved} → {current_params}.")
                print(f"        [stale] Deleting cached maps and recomputing.")
                shutil.rmtree(str(maps_path))
        except Exception:
            pass  # corrupt manifest → let files be re-evaluated individually

    if method_name == 'tta':
        from src.uncertainty import calculate_tta_uncertainty
        def _unc(w, img): return calculate_tta_uncertainty(w, img, n_passes=n_tta)
    elif method_name == 'jacobian':
        from src.uncertainty import calculate_jacobian_sensitivity_uncertainty
        def _unc(w, img): return calculate_jacobian_sensitivity_uncertainty(
            w, img, n_directions=n_jacobian)
    elif method_name == 'mahalanobis':
        if not reference_images:
            raise ValueError("Mahalanobis downstream maps require reference_images.")
        from src.uncertainty import calculate_mahalanobis_uncertainty
        _ref = reference_images  # capture for closure
        def _unc(w, img): return calculate_mahalanobis_uncertainty(w, img, _ref)
    elif method_name == 'mcd':
        from src.uncertainty import calculate_mcd_uncertainty
        def _unc(w, img): return calculate_mcd_uncertainty(w, img, n_passes=n_mcd)
    elif method_name == 'sol_mcd':
        from src.uncertainty import calculate_sol_mcd_uncertainty
        def _unc(w, img): return calculate_sol_mcd_uncertainty(w, img, n_passes=n_mcd)
    elif method_name == 'amcd':
        from src.uncertainty import calculate_antithetic_mcd_uncertainty
        def _unc(w, img): return calculate_antithetic_mcd_uncertainty(w, img, n_pairs=n_mcd)
    else:
        raise ValueError(f"Unsupported Stage-5 method: {method_name!r}")

    # Compute per-split caps: train uses n_cap; test uses n_cap//5 (matching the
    # EmotionDataset stratification in _train_one_variant).  This avoids computing
    # up to 5× more test maps than will ever be loaded.
    split_caps = {
        'train': n_cap,
        'test':  (max(1, n_cap // 5) if n_cap is not None else None),
    }

    # Pre-count total work so we can print progress / ETA.
    total_to_compute = 0
    for split, cap in split_caps.items():
        for _, paths in _raf_subset_paths(raf_root, split, cap).items():
            for p in paths:
                if not (Path(maps_root) / split / p.parent.name / (Path(p).stem + '.npy')).exists():
                    total_to_compute += 1
    if method_name == 'tta':
        passes_per_img = n_tta + 1
    elif method_name == 'jacobian':
        passes_per_img = n_jacobian + 1
    elif method_name in ('mcd', 'sol_mcd'):
        passes_per_img = n_mcd
    elif method_name == 'amcd':
        passes_per_img = n_mcd * 2  # antithetic pairs
    else:
        passes_per_img = 1
    print(f"        {total_to_compute} maps to compute "
          f"({passes_per_img} passes each) — "
          f"est. {total_to_compute * passes_per_img:.0f} model calls")

    import time as _time
    t0 = _time.time()
    n_computed = n_cached = 0

    for split, cap in split_caps.items():
        for cls_name, img_paths in _raf_subset_paths(raf_root, split, cap).items():
            for img_path in img_paths:
                img_path = Path(img_path)
                out = Path(maps_root) / split / cls_name / (img_path.stem + '.npy')
                if out.exists():
                    n_cached += 1
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                img_np = np.asarray(PILImage.open(img_path).convert('RGB'), dtype=np.uint8)
                H, W   = img_np.shape[:2]
                try:
                    verts = np.asarray(wrapper.get_vertices(img_np), dtype=np.float32)
                    unc   = np.asarray(_unc(wrapper, img_np), dtype=np.float32)
                    conf  = project_variance_to_2d(verts, unc, (H, W))
                    np.save(str(out), conf)
                    n_computed += 1
                    # Progress every 10 computed images
                    if n_computed % 10 == 0:
                        elapsed = _time.time() - t0
                        rate = n_computed / elapsed if elapsed > 0 else 0
                        remaining = (total_to_compute - n_computed) / rate if rate > 0 else 0
                        print(f"        [{n_computed}/{total_to_compute}] "
                              f"{rate:.1f} img/s — "
                              f"~{remaining/60:.1f} min remaining")
                except Exception as exc:
                    warnings.warn(f"Map failed for {img_path.name}: {exc}")

    # Write manifest so future runs can detect parameter changes.
    maps_path.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(current_params, indent=2))
    return n_computed, n_cached


def _save_overlay_grid(
    raf_root: str,
    maps_root: str,
    output_path: str,
    n_per_class: int = 3,
) -> None:
    """
    PNG grid: left column = original image, right column = confidence-masked image.
    One row per emotion class, n_per_class pairs per row.
    Saves to output_path for visual inspection of the confidence maps.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    from PIL import Image as PILImage
    from src.emotion_dataset import EXPRESSION_CLASSES, _IMG_EXTS

    N = len(EXPRESSION_CLASSES)
    fig, axes = plt.subplots(
        N, n_per_class * 2,
        figsize=(n_per_class * 4, N * 2.4),
        squeeze=False,
    )
    fig.suptitle(
        'Original  |  Uncertainty-Masked  (per column pair)\n'
        'Bright pixels = confident region seen by classifier; '
        'dark pixels = high uncertainty → dimmed',
        fontsize=10, fontweight='bold',
    )

    for row, cls_name in enumerate(EXPRESSION_CLASSES):
        cls_dir = Path(raf_root) / 'test' / cls_name
        map_dir = Path(maps_root) / 'test' / cls_name
        img_paths = (
            sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)
            if cls_dir.exists() else []
        )[:n_per_class]

        for col_i, img_path in enumerate(img_paths):
            img  = np.asarray(PILImage.open(img_path).convert('RGB'))
            npy  = map_dir / (img_path.stem + '.npy')

            ax_o = axes[row][col_i * 2]
            ax_m = axes[row][col_i * 2 + 1]

            ax_o.imshow(img)
            ax_o.set_axis_off()
            if col_i == 0:
                ax_o.set_ylabel(cls_name, fontsize=8, fontweight='bold', rotation=0,
                                labelpad=40, va='center')

            if npy.exists():
                conf = np.load(str(npy)).astype(np.float32)             # (H, W) [0.5,1]
                # Float-preserving resize — skip uint8 round-trip to
                # avoid visible quantisation bands across the 128-level
                # [0.5, 1.0] confidence range.
                if conf.shape != (img.shape[0], img.shape[1]):
                    conf = cv2.resize(
                        conf, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
                masked = np.clip(
                    img.astype(np.float32) / 255.0 * conf[:, :, None],
                    0.0, 1.0,
                )
                ax_m.imshow(masked)
            else:
                ax_m.text(0.5, 0.5, 'no map', ha='center', va='center', fontsize=7)
            ax_m.set_axis_off()

        # Blank unused columns
        for col_i in range(len(img_paths), n_per_class):
            axes[row][col_i * 2].set_axis_off()
            axes[row][col_i * 2 + 1].set_axis_off()

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def _plot_downstream_comparison(
    comparison: Dict[str, dict],
    output_path: str,
) -> None:
    """
    Three-panel comparison figure + separate training-curves figure.

    Panel 1 — overall test accuracy: plain vs weighted for each combo.
    Panel 2 — per-class accuracy for the best combo.
    Panel 3 — per-class F1 score for the best combo (if available).

    A separate figure (output_path → _training_curves.png) shows train/test
    accuracy and training loss per epoch for plain and weighted runs.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.emotion_dataset import EXPRESSION_CLASSES

    combos = list(comparison.keys())
    if not combos:
        return

    plain_accs    = [comparison[c]['plain'].get('best_test_acc', 0)    for c in combos]
    weighted_accs = [comparison[c]['weighted'].get('best_test_acc', 0) for c in combos]
    best_combo = max(combos, key=lambda c: comparison[c]['weighted'].get('best_test_acc', 0))

    has_f1 = bool(comparison[best_combo]['plain'].get('per_class_f1'))
    n_panels = 3 if has_f1 else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), layout='constrained')
    axes = list(axes)
    fig.suptitle('Plain vs Uncertainty-Weighted Classifier — RAF-DB',
                 fontsize=13, fontweight='bold')

    # ── Panel 1: Overall accuracy ─────────────────────────────────────────────
    ax1 = axes[0]
    x = np.arange(len(combos))
    w = 0.35
    b1 = ax1.bar(x - w/2, plain_accs,    w, label='Plain',                color='#4C72B0', alpha=0.85)
    b2 = ax1.bar(x + w/2, weighted_accs, w, label='Uncertainty-Weighted', color='#DD8452', alpha=0.85)
    for bar, val in zip(b1, plain_accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(b2, weighted_accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color='#DD8452')
    for i, (p, ww) in enumerate(zip(plain_accs, weighted_accs)):
        delta = ww - p
        ax1.text(i, max(p, ww) + 2.5, f'Δ{delta:+.1f}%',
                 ha='center', fontsize=7, color='green' if delta >= 0 else 'red')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace('_', '\n') for c in combos], fontsize=9)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Overall Test Accuracy\nΔ = weighted − plain')
    ax1.legend()
    ceil = max(max(plain_accs), max(weighted_accs), 1.0)
    ax1.set_ylim(0, ceil * 1.20)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # ── Panel 2: Per-class accuracy for best combo ────────────────────────────
    ax2 = axes[1]
    plain_pc    = [comparison[best_combo]['plain']['per_class_acc'].get(cl, 0)
                   for cl in EXPRESSION_CLASSES]
    weighted_pc = [comparison[best_combo]['weighted']['per_class_acc'].get(cl, 0)
                   for cl in EXPRESSION_CLASSES]
    x2 = np.arange(len(EXPRESSION_CLASSES))
    ax2.bar(x2 - w/2, plain_pc,    w, label='Plain',                color='#4C72B0', alpha=0.85)
    ax2.bar(x2 + w/2, weighted_pc, w, label='Uncertainty-Weighted', color='#DD8452', alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(EXPRESSION_CLASSES, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Per-class Accuracy (%)')
    ax2.set_title(f'Per-class Accuracy Breakdown\n({best_combo})')
    ax2.legend()
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # ── Panel 3: Per-class F1 score for best combo (if available) ────────────
    if has_f1:
        ax3 = axes[2]
        plain_f1    = [comparison[best_combo]['plain']['per_class_f1'].get(cl, 0)
                       for cl in EXPRESSION_CLASSES]
        weighted_f1 = [comparison[best_combo]['weighted']['per_class_f1'].get(cl, 0)
                       for cl in EXPRESSION_CLASSES]
        ax3.bar(x2 - w/2, plain_f1,    w, label='Plain',                color='#4C72B0', alpha=0.85)
        ax3.bar(x2 + w/2, weighted_f1, w, label='Uncertainty-Weighted', color='#DD8452', alpha=0.85)
        ax3.set_xticks(x2)
        ax3.set_xticklabels(EXPRESSION_CLASSES, rotation=30, ha='right', fontsize=8)
        ax3.set_ylabel('F1 Score')
        ax3.set_ylim(0, 1.15)
        ax3.set_title(f'Per-class F1 Score\n({best_combo})')
        ax3.legend()
        ax3.spines[['top', 'right']].set_visible(False)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Training curves figure ────────────────────────────────────────────────
    # Save to <output_path base>_training_curves.png
    base, ext = os.path.splitext(output_path)
    curves_path = base + '_training_curves' + (ext or '.png')

    any_epochs = any(
        comparison[c][v].get('epochs')
        for c in combos for v in ('plain', 'weighted')
    )
    if not any_epochs:
        return

    n_combos = len(combos)
    fig2, axes2 = plt.subplots(n_combos, 2, figsize=(12, 4 * n_combos),
                                squeeze=False, layout='constrained')
    fig2.suptitle('Training Curves — Plain vs Uncertainty-Weighted',
                   fontsize=13, fontweight='bold')

    for ri, combo in enumerate(combos):
        ax_acc  = axes2[ri, 0]
        ax_loss = axes2[ri, 1]

        for variant, color, ls in [('plain', '#4C72B0', '-'), ('weighted', '#DD8452', '--')]:
            epochs_data = comparison[combo][variant].get('epochs', [])
            if not epochs_data:
                continue
            ep  = [d['epoch']     for d in epochs_data]
            ta  = [d['train_acc'] for d in epochs_data]
            tea = [d['test_acc']  for d in epochs_data]
            tl  = [d.get('train_loss', float('nan')) for d in epochs_data]
            lbl = variant.capitalize()
            ax_acc.plot(ep, ta,  color=color, ls=ls,  lw=1.5, alpha=0.7, label=f'{lbl} train')
            ax_acc.plot(ep, tea, color=color, ls='-',  lw=2.0, marker='o', ms=3,
                         label=f'{lbl} test')
            if any(not np.isnan(v) for v in tl):
                ax_loss.plot(ep, tl, color=color, ls=ls, lw=2.0, label=lbl)

        ax_acc.set_title(f'{combo} — Accuracy', fontsize=10)
        ax_acc.set_xlabel('Epoch'); ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.legend(fontsize=7); ax_acc.grid(alpha=0.3, linestyle='--')
        ax_acc.spines[['top', 'right']].set_visible(False)

        ax_loss.set_title(f'{combo} — Training Loss', fontsize=10)
        ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('Cross-Entropy Loss')
        ax_loss.legend(fontsize=8); ax_loss.grid(alpha=0.3, linestyle='--')
        ax_loss.spines[['top', 'right']].set_visible(False)

    fig2.savefig(curves_path, dpi=120, bbox_inches='tight')
    plt.close(fig2)
    print(f"    Training curves → {os.path.relpath(curves_path)}")


def _train_one_variant(
    mode: str,
    raf_root: str,
    maps_root: Optional[str],
    args: argparse.Namespace,
    device: str,
    tag: str,
) -> dict:
    """
    Train UncertaintyWeightedClassifier in one mode ('plain' or
    'uncertainty_weighted').

    Two-stage fine-tuning for CNN/ViT backbones:
      Stage 1 — backbone frozen, head-only training (warm-up).
      Stage 2 — full fine-tune with backbone LR × 0.1, head at full LR.

    Both stages use a cosine-annealing schedule and gradient clipping.
    When n_finetune==0 (backbone stays frozen for all epochs; always the case on
    CPU), training uses val_transform (resize + normalise) so the frozen backbone
    produces deterministic features each epoch, matching the pre-cached features
    that tune_downstream_cpu produces.  When Stage 2 runs, train_transform
    (RandomResizedCrop, HorizontalFlip, ColorJitter) is active throughout.

    Returns metrics dict.
    """
    import torch.nn as nn
    import torchvision.transforms as T
    from pathlib import Path
    from torch.utils.data import DataLoader, Subset
    from collections import defaultdict
    from src.emotion_dataset import EmotionDataset
    from src.downstream import UncertaintyWeightedClassifier, EXPRESSION_CLASSES

    NUM_CLASSES  = len(EXPRESSION_CLASSES)
    # ViT-H/14 SWAG weights were trained at 518×518; all other backbones use 224.
    _BACKBONE_SIZES = {'vit_h_14': 518}
    IMAGE_SIZE   = _BACKBONE_SIZES.get(getattr(args, 'downstream_backbone', ''), 224)
    MEAN, STD    = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # ── Transforms ────────────────────────────────────────────────────────────
    train_transform = T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.80, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    val_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    try:
        train_ds = EmotionDataset(root=raf_root, split='train', mode=mode,
                                   uncertainty_root=maps_root,
                                   transform=train_transform,
                                   image_size=IMAGE_SIZE)
        test_ds  = EmotionDataset(root=raf_root, split='test',  mode=mode,
                                   uncertainty_root=maps_root,
                                   transform=val_transform,
                                   image_size=IMAGE_SIZE)
    except Exception as exc:
        print(f"    Dataset load failed ({mode}): {exc}")
        return {}

    # Per-class caps — train and test are controlled independently.
    # --downstream_train_subset takes priority; falls back to --downstream_subset.
    _raw_train_cap = getattr(args, 'downstream_train_subset', 0) or 0
    if _raw_train_cap == 0 and (args.downstream_subset or 0) > 0:
        _raw_train_cap = args.downstream_subset
    _raw_test_cap = getattr(args, 'downstream_test_subset', 0) or 0

    n_train_cap = _raw_train_cap if _raw_train_cap > 0 else None
    n_test_cap  = _raw_test_cap  if _raw_test_cap  > 0 else None

    if n_train_cap is not None or n_test_cap is not None:
        def _stratify(ds: EmotionDataset, n: int) -> Subset:
            buckets: dict = defaultdict(list)
            for i, (_, lbl) in enumerate(ds.samples):
                buckets[lbl].append(i)
            idxs: List[int] = []
            for lbl in range(NUM_CLASSES):
                idxs.extend(buckets[lbl][:n])
            return Subset(ds, sorted(idxs))
        if n_train_cap is not None:
            train_ds = _stratify(train_ds, n_train_cap)
        if n_test_cap is not None:
            test_ds  = _stratify(test_ds, n_test_cap)

    # Inverse-frequency class weights
    raw = (train_ds.samples if isinstance(train_ds, EmotionDataset)
           else [train_ds.dataset.samples[i] for i in train_ds.indices])
    counts = np.zeros(NUM_CLASSES, dtype=np.float32)
    for _, lbl in raw:
        counts[lbl] += 1
    counts = np.maximum(counts, 1.0)
    cls_w  = (1.0 / counts)
    cls_w  = (cls_w / cls_w.sum() * NUM_CLASSES).astype(np.float32)

    train_loader = DataLoader(train_ds, batch_size=args.downstream_batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.downstream_batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    _fusion_mode  = getattr(args, 'downstream_fusion',       'input')
    if _fusion_mode == 'all':
        _fusion_mode = 'input'  # 'all' is a run_downstream meta-value; plain mode ignores fusion
    _fusion_alpha = getattr(args, 'downstream_fusion_alpha', 1.0)
    clf = UncertaintyWeightedClassifier(
        num_classes=NUM_CLASSES, backbone=args.downstream_backbone,
        pretrained=True, fusion_mode=_fusion_mode, fusion_alpha=_fusion_alpha,
        head_dropout=args.downstream_head_dropout,
        head_arch=args.downstream_head_arch,
    ).to(device)
    total_p = sum(p.numel() for p in clf.parameters()) / 1e6
    print(f"    {args.downstream_backbone}  {total_p:.1f}M params  "
          f"{len(train_ds)} train / {len(test_ds)} test  mode={mode}")

    is_loss_weighted = (mode == 'loss_weighted')
    cls_w_t   = torch.tensor(cls_w, device=device)
    # loss_weighted needs per-sample losses to multiply by confidence weights.
    criterion = nn.CrossEntropyLoss(weight=cls_w_t,
                                    label_smoothing=args.downstream_label_smoothing,
                                    reduction='none' if is_loss_weighted else 'mean')
    ckpt_dir  = os.path.join(args.output_dir, 'downstream', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f'{tag}_best.pth')
    done_path = ckpt_path + '.done'
    if getattr(args, 'downstream_reuse_checkpoint', False) and os.path.exists(ckpt_path):
        if not os.path.exists(done_path):
            print(f"    [reuse] Checkpoint '{ckpt_path}' exists but has no "
                  f"'.done' marker — a prior run must have been interrupted "
                  f"mid-training. Retraining from scratch instead of reusing "
                  f"a partially-trained checkpoint.")
        else:
            print(f"    [reuse] Found existing checkpoint '{ckpt_path}' — "
                  f"loading and evaluating once instead of retraining.")
            _ckpt = torch.load(ckpt_path, map_location=device)
            clf.load_state_dict(_ckpt['model_state'])
            _reuse_acc, _reuse_per_class = _clf_evaluate(clf, test_loader, device, NUM_CLASSES)
            print(f"    [reuse] test={_reuse_acc:.1f}%  "
                  f"(checkpoint saved at epoch {_ckpt.get('epoch', '?')}, "
                  f"best_acc={_ckpt.get('best_acc', float('nan')):.1f}%)")
            return {
                'best_test_acc': float(_reuse_acc),
                'per_class_acc': {c: float(a) for c, a in zip(EXPRESSION_CLASSES, _reuse_per_class)},
                'per_class_f1':  {c: float(f) for c, f in zip(EXPRESSION_CLASSES, getattr(_reuse_per_class, 'f1', []))}
                                 if getattr(_reuse_per_class, 'f1', None) else {},
            }

    best_acc          = 0.0
    per_class_acc     = [0.0] * NUM_CLASSES
    best_per_class_acc: List[float] = [0.0] * NUM_CLASSES
    best_f1: List[float] = []
    epoch_log: list      = []
    log_every  = max(1, len(train_loader) // 4)
    no_improve = 0                                   # epochs since last strict improvement
    patience   = args.downstream_patience            # configurable via --downstream_patience

    total_epochs = args.downstream_epochs
    is_cnn = (getattr(clf, 'architecture_type', 'CNN') == 'CNN')

    # Two-stage split: ⌈1/3⌉ for head warm-up, rest for full fine-tune.
    # GCN mode runs as a single stage since there is no distinct backbone.
    n_warmup   = max(1, total_epochs // 3) if is_cnn else 0
    n_finetune = total_epochs - n_warmup

    # On CPU with small subsets, Stage 2 (backbone unfreeze) causes immediate
    # catastrophic overfitting: 87M params vs. ~175 training samples.
    # Keep backbone frozen for all epochs on CPU.
    if device == 'cpu' and is_cnn:
        n_warmup   = total_epochs
        n_finetune = 0
        print(f"    [CPU] Backbone frozen for all {total_epochs} epochs "
              f"(Stage 2 skipped — too few samples for full fine-tune).")

    if n_finetune == 0:
        # Backbone is frozen for all epochs. Replace augmented train_transform with
        # val_transform (resize + normalise only) so the frozen backbone produces
        # deterministic features each epoch — matching the pre-cached features that
        # tune_downstream_cpu produces.  Augmentation only helps when the backbone
        # is also being fine-tuned; applying it to a frozen backbone adds noise that
        # breaks the train/test feature distribution alignment.
        _underlying = (train_ds.dataset if isinstance(train_ds, Subset)
                       else train_ds)
        _underlying.transform = val_transform

    # ── Helper: run one training epoch ────────────────────────────────────────
    def _run_epoch(opt, epoch_idx, stage_label, mixup_alpha: float = 0.0, loader=None):
        _loader = loader if loader is not None else train_loader
        _log_every = max(1, len(_loader) // 4)
        clf.train()
        run_loss = n_correct = n_total = 0
        for bi, batch in enumerate(_loader):
            if is_loss_weighted:
                images, labels, sample_weights = batch
                sample_weights = sample_weights.to(device)
            else:
                images, labels = batch
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()

            if mixup_alpha > 0.0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(len(images), device=device)
                images = lam * images + (1.0 - lam) * images[idx]
                labels_b = labels[idx]
                logits = clf(image=images)
                if is_loss_weighted:
                    per_sample = (lam * criterion(logits, labels)
                                  + (1.0 - lam) * criterion(logits, labels_b))
                    loss = (per_sample * sample_weights).mean()
                else:
                    loss = (lam * criterion(logits, labels)
                            + (1.0 - lam) * criterion(logits, labels_b))
            else:
                logits = clf(image=images)
                if is_loss_weighted:
                    loss = (criterion(logits, labels) * sample_weights).mean()
                else:
                    loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step()
            run_loss  += loss.item() * len(labels)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_total   += len(labels)
            if (bi + 1) % _log_every == 0 or bi == len(_loader) - 1:
                print(f"    [{tag}][{stage_label} e{epoch_idx}] "
                      f"b{bi+1}/{len(_loader)}  "
                      f"loss={run_loss/n_total:.4f}  "
                      f"acc={100*n_correct/n_total:.1f}%")
        return float(run_loss / max(n_total, 1)), float(100.0 * n_correct / n_total)

    # ── Stage 1: freeze backbone, train head only ─────────────────────────────
    if is_cnn and n_warmup > 0:
        print(f"    Stage 1/2: freeze backbone, train head ({n_warmup} epoch(s))")
        for param in clf.backbone.parameters():
            param.requires_grad = False

        opt1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, clf.parameters()),
            lr=args.downstream_lr, weight_decay=args.downstream_weight_decay)
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt1, T_max=n_warmup, eta_min=args.downstream_lr * 0.05)

        for ep in range(1, n_warmup + 1):
            tr_loss, tr_acc = _run_epoch(opt1, ep, 'S1')
            sch1.step()
            test_acc, per_class_acc = _clf_evaluate(clf, test_loader, device, NUM_CLASSES)
            marker = ' ◄' if test_acc > best_acc else ''
            print(f"    [S1] Epoch {ep:2d}/{n_warmup}  "
                  f"train={tr_acc:.1f}%  test={test_acc:.1f}%{marker}")
            if test_acc > best_acc:
                best_acc           = test_acc
                no_improve         = 0
                best_per_class_acc = list(per_class_acc)
                best_f1            = list(getattr(per_class_acc, 'f1', []))
                torch.save({'epoch': ep, 'model_state': clf.state_dict(),
                            'best_acc': best_acc, 'mode': mode,
                            'backbone': args.downstream_backbone},
                           os.path.join(ckpt_dir, f'{tag}_best.pth'))
            else:
                no_improve += 1
            epoch_log.append({'epoch': ep, 'stage': 1,
                              'train_acc': tr_acc, 'train_loss': tr_loss,
                              'test_acc': float(test_acc)})
            if no_improve >= patience:
                print(f"    [S1] Early stop — no improvement for {patience} epochs.")
                break

        # Reload the best Stage-1 checkpoint before Stage-2 begins.
        _s1_best = os.path.join(ckpt_dir, f'{tag}_best.pth')
        if os.path.exists(_s1_best):
            clf.load_state_dict(
                torch.load(_s1_best, map_location=device)['model_state']
            )
            print(f"    [S1→S2] Reloaded best Stage-1 checkpoint "
                  f"(acc={best_acc:.1f}%) — Stage 2 fine-tunes from peak state.")

        # ── Partial unfreeze: only the last N transformer blocks + final LN ──
        # Fine-tuning all 303M ViT params on ~3K samples → severe overfitting.
        # Keeping early layers frozen acts as a structural regularizer.
        _n_unfreeze = getattr(args, 'downstream_unfreeze_blocks', 6)
        backbone    = clf.backbone
        # Start with everything frozen; selectively open later layers.
        for p in backbone.parameters():
            p.requires_grad = False
        if _n_unfreeze < 0:
            # -1 = caller explicitly wants full fine-tune (large dataset)
            for p in backbone.parameters():
                p.requires_grad = True
        elif hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layers'):
            enc = backbone.encoder
            layers = list(enc.layers.children())
            for layer in layers[-_n_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True
            # Always unfreeze the encoder's final LayerNorm
            if hasattr(enc, 'ln'):
                for p in enc.ln.parameters():
                    p.requires_grad = True
            n_unfreeze_actual = min(_n_unfreeze, len(layers))
            print(f"    [S1→S2] Unfreezing last {n_unfreeze_actual}/{len(layers)} "
                  f"transformer blocks + encoder LN.")
        else:
            # CNN backbone: unfreeze all (partial-freeze makes less sense for ResNet)
            for p in backbone.parameters():
                p.requires_grad = True

    # ── Curriculum: pre-score training samples by uncertainty (low = easy) ──
    # "Easy" samples are those where the confidence map is mostly 1.0 (the
    # regressor is confident about the face geometry everywhere).  Hard samples
    # have lower mean confidence.  Score = 1 - mean(conf) so ascending sort
    # gives easy-first ordering.  We fall back gracefully (score=0.5) when a
    # map is missing, so the sample is treated as medium-difficulty.
    _curriculum = getattr(args, 'downstream_curriculum', True)
    _curr_sorted_indices = None
    _curr_start_frac = getattr(args, 'downstream_curriculum_start', 0.5)
    if _curriculum and maps_root and os.path.isdir(maps_root):
        print("    [Curriculum] Scoring training samples by uncertainty …")
        _base_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
        _base_indices = list(train_ds.indices) if isinstance(train_ds, Subset) else list(range(len(train_ds)))
        _scores = np.full(len(_base_indices), 0.5, dtype=np.float32)
        _maps_root_p = Path(maps_root)
        n_found = 0
        for _li, _bi in enumerate(_base_indices):
            img_path, _ = _base_ds.samples[_bi]
            try:
                _rel = img_path.relative_to(_base_ds.root)
                _npy = (_maps_root_p / _rel).with_suffix('.npy')
                if _npy.exists():
                    _conf = np.load(str(_npy), mmap_mode='r')
                    _scores[_li] = 1.0 - float(np.mean(_conf))
                    n_found += 1
            except Exception:
                pass
        _curr_sorted_indices = np.argsort(_scores).tolist()
        print(f"    [Curriculum] {n_found}/{len(_base_indices)} maps scored — "
              f"pacing {int(_curr_start_frac * 100)}%→100% over "
              f"{n_finetune} epoch(s).")

    # ── Stage 2: LLRD fine-tune of unfrozen layers ───────────────────────────
    stage2_label = '2/2' if (is_cnn and n_warmup > 0) else '1/1'
    _mixup_alpha = getattr(args, 'downstream_mixup_alpha', 0.2)

    if is_cnn:
        backbone     = clf.backbone
        backbone_lr  = args.downstream_lr * 0.1
        param_groups = [{'params': clf.classifier.parameters(), 'lr': args.downstream_lr}]
        # LLRD: assign decaying LR to each unfrozen transformer block
        if (hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layers')
                and getattr(args, 'downstream_unfreeze_blocks', 6) >= 0):
            enc    = backbone.encoder
            layers = list(enc.layers.children())
            _n_ub  = getattr(args, 'downstream_unfreeze_blocks', 6)
            unfrozen_layers = layers[-_n_ub:] if _n_ub > 0 else layers
            n_ub   = len(unfrozen_layers)
            llrd_decay = 0.75   # LR halves roughly every ~3 layers
            for i, layer in enumerate(unfrozen_layers):
                layer_ps = [p for p in layer.parameters() if p.requires_grad]
                if layer_ps:
                    lr_i = backbone_lr * (llrd_decay ** (n_ub - 1 - i))
                    param_groups.append({'params': layer_ps, 'lr': lr_i})
            # Final encoder LN at full backbone LR
            if hasattr(enc, 'ln'):
                ln_ps = [p for p in enc.ln.parameters() if p.requires_grad]
                if ln_ps:
                    param_groups.append({'params': ln_ps, 'lr': backbone_lr})
        else:
            # CNN or full-unfreeze fallback: single backbone group
            bb_ps = [p for p in backbone.parameters() if p.requires_grad]
            if bb_ps:
                param_groups.append({'params': bb_ps, 'lr': backbone_lr})
    else:
        param_groups = [{'params': clf.parameters(), 'lr': args.downstream_lr}]

    opt2 = torch.optim.AdamW(param_groups, weight_decay=args.downstream_weight_decay)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=max(1, n_finetune), eta_min=args.downstream_lr * 0.005)

    # Guard: skip Stage 2 only when Stage 1 made ZERO genuine progress —
    # i.e. the model is still at or below random-chance accuracy after early
    # stopping.  If Stage 1 peaked above random (e.g. best_acc=25%) and then
    # plateaued, Stage 2 should still run: clf.load_state_dict(_s1_best) above
    # already reloaded the peak state, giving Stage 2 a good starting point.
    # Using no_improve >= patience alone was too aggressive — it skipped Stage 2
    # even after a healthy Stage 1 that simply converged and stopped early.
    _s1_stagnated = (is_cnn and n_warmup > 0 and no_improve >= patience
                     and best_acc <= 100.0 / NUM_CLASSES)
    no_improve = 0  # reset counter for Stage 2
    _s2_epochs = 0 if _s1_stagnated else n_finetune
    if _s1_stagnated and n_finetune > 0:
        print(f"    [S2] Skipped — Stage 1 early-stopped without improvement for "
              f"{patience} consecutive epochs; backbone fine-tune not attempted.")
    elif _s2_epochs > 0:
        _n_ub_display = getattr(args, 'downstream_unfreeze_blocks', 6)
        print(f"    Stage {stage2_label}: partial fine-tune ({_s2_epochs} epoch(s))  "
              f"blocks={_n_ub_display}  backbone_lr(top)={args.downstream_lr * 0.1:.1e}  "
              f"head_lr={args.downstream_lr:.1e}  mixup={_mixup_alpha}")
    for ep in range(1, _s2_epochs + 1):
        global_ep = n_warmup + ep
        _curr_loader = None
        if _curr_sorted_indices is not None:
            _progress = (ep - 1) / max(1, _s2_epochs - 1)
            _frac = _curr_start_frac + (1.0 - _curr_start_frac) * _progress
            _n_active = max(args.downstream_batch_size,
                            int(len(_curr_sorted_indices) * _frac))
            _curr_subset = Subset(train_ds, _curr_sorted_indices[:_n_active])
            _curr_loader = DataLoader(
                _curr_subset, batch_size=args.downstream_batch_size,
                shuffle=True, num_workers=0, pin_memory=False)
            print(f"    [Curriculum] ep {ep}/{_s2_epochs}: "
                  f"active={_n_active}/{len(_curr_sorted_indices)} "
                  f"({int(_frac * 100)}%)")
        tr_loss, tr_acc = _run_epoch(opt2, ep, f'S{stage2_label[0]}',
                                     mixup_alpha=_mixup_alpha, loader=_curr_loader)
        sch2.step()
        test_acc, per_class_acc = _clf_evaluate(clf, test_loader, device, NUM_CLASSES)
        marker = ' ◄' if test_acc > best_acc else ''
        print(f"    [S{stage2_label[0]}] Epoch {ep:2d}/{n_finetune}  "
              f"(global {global_ep}/{total_epochs})  "
              f"train={tr_acc:.1f}%  test={test_acc:.1f}%{marker}")
        if test_acc > best_acc:
            best_acc           = test_acc
            no_improve         = 0
            best_per_class_acc = list(per_class_acc)
            best_f1            = list(getattr(per_class_acc, 'f1', []))
            torch.save({'epoch': global_ep, 'model_state': clf.state_dict(),
                        'best_acc': best_acc, 'mode': mode,
                        'backbone': args.downstream_backbone},
                       os.path.join(ckpt_dir, f'{tag}_best.pth'))
        else:
            no_improve += 1
        epoch_log.append({'epoch': global_ep, 'stage': int(stage2_label[0]),
                          'train_acc': tr_acc, 'train_loss': tr_loss,
                          'test_acc': float(test_acc)})
        if no_improve >= patience:
            print(f"    [S{stage2_label[0]}] Early stop — no improvement for {patience} epochs.")
            break

    # Mark this checkpoint as belonging to a run that finished on its own terms
    # (ran out of epochs or legitimately early-stopped) rather than one that
    # was killed mid-training by e.g. a walltime limit -- only a run that
    # reaches this point should ever be eligible for --downstream_reuse_checkpoint.
    try:
        with open(done_path, 'w') as _f:
            _f.write('1')
    except OSError:
        pass

    return {
        'best_test_acc': float(best_acc),
        'per_class_acc': {c: float(a) for c, a in zip(EXPRESSION_CLASSES, best_per_class_acc)},
        'per_class_f1':  {c: float(f) for c, f in zip(EXPRESSION_CLASSES, best_f1)}
                         if best_f1 else {},
        'epochs':        epoch_log,
    }


def run_hyperparam_tuning(args: argparse.Namespace, device: str) -> dict:
    """
    Stage 0 — Hyperparameter tuning for uncertainty methods.

    Loads a small subset of GT-paired images (--tune_n_images, default 8)
    from the configured dataset, runs a grid search over each uncertainty
    method's hyperparameters, and saves a descriptive JSON report.

    Runs on CPU or GPU depending on the wrapper's device.  Estimated time
    on CPU with default 8 images and no MCD methods: ~11 min.  With GPU: ~1 min.

    Outputs
    -------
    <output_dir>/tuning/hyperparams.json  — full results with interpretation
    Console: tabular report of best params per method
    """
    from src.hyperparam_tuning import (
        tune_uncertainty_hyperparams,
        save_tuning_results_json,
        print_tuning_report,
    )
    from src.inference import UnifiedFaceRegressor

    print("\n" + "=" * 60)
    print("[Stage 0] Hyperparameter Tuning")
    print(f"  Dataset      : {args.dataset}")
    print(f"  N eval images: {args.tune_n_images}")
    print(f"  Objective    : {args.tune_objective}")
    print(f"  Methods      : {args.tune_methods}")
    print(f"  Primary model: {args.primary_model}")
    print("=" * 60)

    # ── 1. Load dataset partition ──────────────────────────────────────────────
    loader = FaceDatasetLoader(data_root='./datasets')
    try:
        if args.dataset == 'now':
            items = loader.load_now_subset(args.tune_n_images + 5, seed=99)
        elif args.dataset == 'coma':
            items = loader.load_coma_subset(args.tune_n_images + 5, seed=99)
        elif args.dataset == 'tempeh':
            items = loader.load_tempeh_subset(args.tune_n_images + 5, seed=99)
        else:
            print(f"  [warn] Dataset '{args.dataset}' has no GT vertices — "
                  "falling back to tempeh.")
            items = loader.load_tempeh_subset(args.tune_n_images + 5, seed=99)
    except Exception as exc:
        print(f"  [error] Dataset load failed: {exc}")
        return {}

    valid = [it for it in items
             if it.get('image') is not None and it.get('gt_vertices') is not None]
    if len(valid) < 2:
        print(f"  [skip] Only {len(valid)} GT-paired items found — need ≥ 2.")
        return {}
    eval_items = valid[:args.tune_n_images]
    print(f"  Using {len(eval_items)} GT-paired images for tuning.")

    # ── 2. Load models ─────────────────────────────────────────────────────────
    # CrossMethod disagreement requires ≥2 models; use the full args.models list
    # so the bash caller can control which models are loaded.
    need_cross = 'cross' in args.tune_methods
    models_to_load = args.models if need_cross else [args.primary_model]
    print(f"\n  Loading {' + '.join(models_to_load)} …")
    try:
        reg = UnifiedFaceRegressor(device=device, models=models_to_load)
    except Exception as exc:
        print(f"  [error] Model load failed: {exc}")
        return {}

    if args.primary_model not in reg.models:
        print(f"  [error] {args.primary_model} did not load.")
        return {}

    if need_cross and len(reg.models) < 2:
        print(f"  [warn] CrossMethod requires ≥2 models but only "
              f"{list(reg.models.keys())} loaded — CrossMethod tuning will be invalid. "
              "Pass --models SMIRK DECA EMOCA SHeaP to get a real result.")

    # MCD wrapper: try to load if mcd / sol_mcd / a_mcd are requested
    mcd_methods = {'mcd', 'sol_mcd', 'a_mcd'}
    need_mcd    = bool(set(args.tune_methods) & mcd_methods)
    mcd_wrapper = None
    if need_mcd:
        print("  Loading SMIRK MCD checkpoint …")
        try:
            from wrappers.smirk_wrapper import SMIRKWrapper
            mcd_wrapper = SMIRKWrapper(device=device, use_mcd_checkpoint=True)
            print("  MCD wrapper loaded.")
        except Exception as exc:
            print(f"  [warn] MCD wrapper failed ({exc}) — MCD methods skipped.")

    # ── 3. Run tuning ──────────────────────────────────────────────────────────
    results = tune_uncertainty_hyperparams(
        wrapper       = reg,
        eval_items    = eval_items,
        primary_model = args.primary_model,
        mcd_wrapper   = mcd_wrapper,
        methods       = args.tune_methods,
        objective     = args.tune_objective,
        verbose       = True,
    )

    # ── 4. Save JSON ───────────────────────────────────────────────────────────
    tuning_dir  = args.output_dir
    json_path   = os.path.join(tuning_dir, 'hyperparams.json')
    meta = {
        'primary_model': args.primary_model,
        'dataset':       args.dataset,
        'device':        device,
        'n_eval_images': len(eval_items),
        'methods_tuned': args.tune_methods,
    }
    save_tuning_results_json(results, json_path, meta=meta)
    print_tuning_report(results)

    print(f"\n[Stage 0] Done.  Results → {json_path}")
    return results


def run_downstream(args: argparse.Namespace, device: str) -> dict:
    """
    Stage 5 orchestrator.

    For each (model, uncertainty_method) combo:
      1. Load the FLAME regressor wrapper (MCD methods load the SMIRK MCD checkpoint).
      2. Pre-compute 2-D confidence maps for all RAF-DB images in the working
         subset and cache them as .npy files (skipped if already on disk).
      3. Save an overlay PNG grid (original vs masked) for visual verification.
      4. Train a PLAIN ViT classifier (no uncertainty) on the same subset.
      5. Train an UNCERTAINTY-WEIGHTED ViT classifier (image × confidence map).
      6. Compare both classifiers and save a comparison bar chart.

    GPU/HPC: all models from --models × all applicable methods from --downstream_methods.
    MCD variants (mcd, sol_mcd, amcd) are filtered to SMIRK only (require retrained checkpoint).
    Mahalanobis samples n_mahal_ref reference images from the RAF-DB training split.

    CPU: primary_model only × {tta, jacobian} to keep runtime manageable.
    """
    from src.inference import UnifiedFaceRegressor
    from src.emotion_dataset import EXPRESSION_CLASSES
    import gc

    NUM_CLASSES = len(EXPRESSION_CLASSES)
    ckpt_dir    = os.path.join(args.output_dir, 'downstream')
    os.makedirs(ckpt_dir, exist_ok=True)

    # GPU/HPC: full method set across all selected models.
    # CPU: tta + jacobian only, primary model only (keeps runtime manageable).
    _MCD_METHODS = {'mcd', 'sol_mcd', 'amcd'}
    if device == 'cpu':
        _SUPPORTED = {'tta', 'jacobian'}
        ds_models = [args.primary_model]
    else:
        _SUPPORTED = {'tta', 'jacobian', 'mahalanobis', 'mcd', 'sol_mcd', 'amcd'}
        ds_models = list(args.models)

    methods = [m.lower() for m in args.downstream_methods if m.lower() in _SUPPORTED]
    if not methods:
        warnings.warn(
            f"No supported Stage-5 methods in {args.downstream_methods}. "
            f"Supported: {sorted(_SUPPORTED)}.  Defaulting to TTA.",
            UserWarning, stacklevel=2,
        )
        methods = ['tta']

    # MCD variants (methods 2/6/7) require the SMIRK checkpoint retrained with
    # nn.Dropout layers — not valid for DECA/EMOCA/SHeaP.
    combos = [
        (mo, me)
        for mo in ds_models
        for me in methods
        if not (me in _MCD_METHODS and mo != 'SMIRK')
    ]

    # ── Apply tuning best_config to args before anything is printed or run ────
    # tune_downstream_cpu/gpu saves to {output_dir}/results.json.  When that
    # file exists we apply phase1.best_config (lr, wd, dropout, ls, head_arch)
    # so the actual training run uses the hyperparameters that were found to be
    # optimal.  CLI flags always override because argparse sets args first; we
    # only mutate args here for the parameters found in best_config.
    _tune_json = os.path.join(args.output_dir, 'results.json')
    _tune_data: dict = {}
    if os.path.exists(_tune_json):
        try:
            with open(_tune_json) as _f:
                _tune_data = json.load(_f)
            _best_cfg: dict = (_tune_data.get('phase1') or {}).get('best_config') or {}
            if _best_cfg:
                _applied: list = []
                for _key, _attr in [
                    ('lr',              'downstream_lr'),
                    ('weight_decay',    'downstream_weight_decay'),
                    ('head_dropout',    'downstream_head_dropout'),
                    ('label_smoothing', 'downstream_label_smoothing'),
                    ('head_arch',       'downstream_head_arch'),
                ]:
                    if _key in _best_cfg:
                        # Cast to the same type already in args to avoid type errors.
                        _cast = type(getattr(args, _attr))
                        setattr(args, _attr, _cast(_best_cfg[_key]))
                        _applied.append(f"--{_attr} {_best_cfg[_key]}")
                if _applied:
                    print(f"  [tuning] Loaded best_config from '{_tune_json}':")
                    for _a in _applied:
                        print(f"           {_a}")
                    print(f"           (pass any of these flags explicitly to override)")
        except Exception as _exc:
            print(f"  [warn] Could not read tuning results '{_tune_json}': {_exc}")

    _ds_fusion_requested = getattr(args, 'downstream_fusion', 'input')
    _ALL_FUSION_MODES_LIST = ['input', 'patch_embed', 'attn_bias', 'key_scale', 'value_scale']
    _fusion_modes_to_run = (
        _ALL_FUSION_MODES_LIST if _ds_fusion_requested == 'all'
        else [_ds_fusion_requested]
    )

    print("\n" + "=" * 60)
    print("[Stage 5] Downstream Classifier — RAF-DB Expression Recognition")
    _ds_n_tta_display = (
        getattr(args, 'downstream_n_tta', None) or str(args.n_tta)
    )
    print(f"  Combos         : {[f'{mo}×{me}' for mo, me in combos]}")
    print(f"  Backbone       : {args.downstream_backbone}  head={args.downstream_head_arch}  dropout={args.downstream_head_dropout}")
    print(f"  Epochs         : {args.downstream_epochs}  LR={args.downstream_lr}  WD={args.downstream_weight_decay}  LS={args.downstream_label_smoothing}")
    print(f"  Regularization : patience={args.downstream_patience}  unfreeze_blocks={getattr(args, 'downstream_unfreeze_blocks', 6)}  mixup_alpha={getattr(args, 'downstream_mixup_alpha', 0.2)}")
    print(f"  Train subset   : {getattr(args, 'downstream_train_subset', 0) or args.downstream_subset or 'FULL'} imgs/class  test subset: {getattr(args, 'downstream_test_subset', 0) or 'FULL'}")
    print(f"  Precomp n_tta  : {_ds_n_tta_display}")
    print(f"  Fusion modes   : {_fusion_modes_to_run}")
    print(f"  Fusion alpha   : {getattr(args, 'downstream_fusion_alpha', 1.0)}")
    print("=" * 60)

    # Verify RAF-DB is accessible
    from pathlib import Path
    if not (Path(args.raf_db_root) / 'train').exists():
        print(f"\n  [skip] RAF-DB not found at '{args.raf_db_root}'.")
        return {}

    all_comparisons: Dict[str, dict] = {}

    for model_name, method_name in combos:
        combo_tag = f"{model_name}_{method_name}"
        print(f"\n{'─'*60}")
        print(f"  Combo: {combo_tag}")
        print(f"{'─'*60}")

        # ── 1. Load wrapper ────────────────────────────────────────────────────
        print(f"\n  [1/4] Loading {model_name} ({method_name}) …")
        reg = None
        try:
            if method_name in _MCD_METHODS:
                # MCD variants need the SMIRK checkpoint retrained with nn.Dropout
                from wrappers.smirk_wrapper import SMIRKWrapper
                wrapper = SMIRKWrapper(device=device, use_mcd_checkpoint=True)
                wrapper.enable_dropout_for_inference()
            else:
                reg = UnifiedFaceRegressor(device=device, models=[model_name])
                if model_name not in reg.models:
                    print(f"        {model_name} failed to load — skipping.")
                    continue
                wrapper = reg.models[model_name]
        except Exception as exc:
            print(f"        Load failed: {exc} — skipping.")
            continue

        # ── 1b. Mahalanobis reference images (sampled once per combo) ──────────
        ref_images = None
        if method_name == 'mahalanobis':
            print(f"        Sampling {args.n_mahal_ref} reference images from RAF-DB train …")
            try:
                ref_images = _sample_raf_reference_images(
                    args.raf_db_root, n=args.n_mahal_ref)
                print(f"        {len(ref_images)} reference images loaded.")
            except Exception as exc:
                print(f"        Reference sampling failed: {exc} — skipping {combo_tag}.")
                if reg is not None:
                    del reg
                del wrapper
                continue

        # ── 2. Pre-compute confidence maps ─────────────────────────────────────
        # Resolve parameters that determine map content BEFORE building the
        # cache path, so the path encodes exactly what was used to generate maps.
        ds_n_tta = getattr(args, 'downstream_n_tta', None)
        if ds_n_tta is None:
            ds_n_tta = 3 if device == 'cpu' else args.n_tta

        # Use train subset for map precomputation — same cap as training data.
        _raw_train_cap = getattr(args, 'downstream_train_subset', 0) or 0
        if _raw_train_cap == 0 and (args.downstream_subset or 0) > 0:
            _raw_train_cap = args.downstream_subset
        _ds_subset = _raw_train_cap if _raw_train_cap > 0 else None
        _subset_key = str(_ds_subset) if _ds_subset else 'all'

        # Maps are stored outside the timestamped output_dir so they survive
        # across runs.  Cache key encodes every parameter that affects map
        # content: dataset, model, method, n_tta, subset cap.
        _ds_stem   = Path(args.raf_db_root).resolve().name
        _cache_key = f"{combo_tag}_tta{ds_n_tta}_sub{_subset_key}"
        maps_root  = os.path.join(
            getattr(args, 'maps_cache_dir', './datasets/maps'),
            _ds_stem,
            _cache_key,
        )
        print(f"\n  [2/4] Pre-computing confidence maps → {maps_root}")
        print(f"        method={method_name}, cap={_subset_key}/class")

        n_computed, n_cached = _precompute_raf_maps(
            wrapper, method_name, args.raf_db_root, maps_root,
            _ds_subset,
            n_tta            = ds_n_tta,
            n_jacobian       = min(args.n_jacobian, 5),
            n_mcd            = args.n_mcd,
            reference_images = ref_images,
            force_recompute  = getattr(args, 'force_recompute_maps', False),
        )
        print(f"        {n_computed} maps computed, {n_cached} loaded from cache.")

        # Overlay grid for visual verification
        overlay_path = os.path.join(ckpt_dir, 'overlays', f'{combo_tag}_grid.png')
        try:
            _save_overlay_grid(args.raf_db_root, maps_root, overlay_path, n_per_class=3)
            print(f"        Overlay grid → {overlay_path}")
        except Exception as exc:
            print(f"        [warn] Overlay grid failed: {exc}")

        # Free wrapper RAM before training
        if reg is not None:
            del reg
        del wrapper
        gc.collect()

        # ── 3. Train PLAIN baseline (once per model×method combo) ─────────────
        print(f"\n  [3/{3 + len(_fusion_modes_to_run)}] Training PLAIN classifier …")
        plain_res = _train_one_variant(
            mode='plain', raf_root=args.raf_db_root, maps_root=None,
            args=args, device=device, tag=f'{combo_tag}_plain',
        )

        # ── 4+. Train UNCERTAINTY-WEIGHTED classifier for each fusion mode ─────
        # Phase 2 in the tuning results stores best_mode keyed by combo_tag.
        _bm = (_tune_data
               .get('phase2', {})
               .get(combo_tag, {})
               .get('best_mode'))
        if _bm in ('uncertainty_weighted', 'loss_weighted'):
            train_mode = _bm
            print(f"        Weighted mode from tuning results: {train_mode}")
        else:
            ds_mode    = getattr(args, 'downstream_mode', 'loss_weighted')
            train_mode = ds_mode if ds_mode == 'loss_weighted' else 'uncertainty_weighted'
            print(f"        Weighted mode from --downstream_mode: {train_mode}")

        fusion_results: dict = {}
        for _fi, _fm in enumerate(_fusion_modes_to_run, start=4):
            print(f"\n  [{_fi}/{3 + len(_fusion_modes_to_run)}] Training WEIGHTED "
                  f"classifier  fusion={_fm}  mode={train_mode} …")
            import copy as _copy
            _args_f = _copy.copy(args)
            _args_f.downstream_fusion = _fm
            _weighted_res = _train_one_variant(
                mode=train_mode, raf_root=args.raf_db_root, maps_root=maps_root,
                args=_args_f, device=device,
                tag=f'{combo_tag}_{_fm}_weighted',
            )
            fusion_results[_fm] = _weighted_res

            p_acc = plain_res.get('best_test_acc', 0.0)
            w_acc = _weighted_res.get('best_test_acc', 0.0)
            delta = w_acc - p_acc
            print(f"        [{combo_tag}|{_fm}]  plain={p_acc:.1f}%  "
                  f"weighted={w_acc:.1f}%  Δ={delta:+.1f}%  "
                  f"({'weighted wins' if delta > 0 else 'plain wins' if delta < 0 else 'tie'})")

        # Use the best weighted result as the canonical 'weighted' entry so that
        # downstream comparison plots and the summary table still work.
        best_fm = max(fusion_results, key=lambda k: fusion_results[k].get('best_test_acc', 0.0))
        weighted_res = fusion_results[best_fm]

        all_comparisons[combo_tag] = {
            'plain': plain_res,
            'weighted': weighted_res,          # best fusion mode
            'fusion_results': fusion_results,  # full per-mode breakdown
        }

    if not all_comparisons:
        print("\n[Stage 5] No combos completed.")
        return {}

    # ── Comparison plot ────────────────────────────────────────────────────────
    cmp_path = os.path.join(ckpt_dir, 'plain_vs_weighted_comparison.png')
    try:
        _plot_downstream_comparison(all_comparisons, cmp_path)
        print(f"\n  Comparison plot → {cmp_path}")
    except Exception as exc:
        print(f"  [warn] Comparison plot failed: {exc}")

    _save_json(all_comparisons, os.path.join(ckpt_dir, 'downstream_comparison.json'))

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[Stage 5] Summary")
    print(f"  {'Combo':<25}  {'Plain':>8}  {'Best weighted':>14}  {'Δ':>6}")
    print(f"  {'─'*60}")
    for tag, res in all_comparisons.items():
        p = res['plain'].get('best_test_acc', 0.0)
        w = res['weighted'].get('best_test_acc', 0.0)
        print(f"  {tag:<25}  {p:>7.1f}%  {w:>13.1f}%  {w-p:>+6.1f}%")
        # Per-fusion-mode breakdown (only when multiple modes were run)
        fr = res.get('fusion_results', {})
        if len(fr) > 1:
            for fm, fres in fr.items():
                fa = fres.get('best_test_acc', 0.0)
                marker = ' ◀ best' if fa == w else ''
                print(f"    {fm:<20}  {fa:>13.1f}%{marker}")

    # Per-class report for each combo
    for tag, res in all_comparisons.items():
        print(f"\n  Per-class accuracy — {tag}")
        print(f"  {'Class':<10}  {'Plain':>8}  {'Weighted':>10}  {'Δ':>6}")
        for cls in EXPRESSION_CLASSES:
            p = res['plain']['per_class_acc'].get(cls, 0.0)
            w = res['weighted']['per_class_acc'].get(cls, 0.0)
            bar = '█' * int(w / 5) + '░' * (20 - int(w / 5))
            print(f"  {cls:<10}  {p:>7.1f}%  {w:>9.1f}%  {w-p:>+6.1f}%  {bar}")

    print(f"\n[Stage 5] Complete.")
    return all_comparisons


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# Stage downstream_tune: downstream hyperparameter tuning
# ══════════════════════════════════════════════════════════════════════════════

def run_downstream_tune(args: argparse.Namespace, device: str) -> dict:
    """
    Hyperparameter tuning for the downstream expression classifier.

    CPU path (~5 min)
    -----------------
    Pre-caches frozen ViT-B/32 backbone features once, then sweeps 96 head
    configs (4 lr × 3 wd × 2 dropout × 2 head_arch × 2 label_smoothing) at millisecond scale.
    Phase 2 compares uncertainty methods using the best training config.

    GPU path (~25–45 min)
    ---------------------
    Full two-stage fine-tuning (frozen warmup → backbone unfreeze) over a
    broader grid: backbone × lr × wd (Phase 1a) + dropout × label_smoothing
    × n_subset (Phase 1b).  Phase 2 tests every available uncertainty method.

    Maps auto-discovery
    -------------------
    Pre-computed confidence maps are expected at:
        {downstream_maps_dir}/{model}_{method}/{split}/{class}/{stem}.npy
    or (if --downstream_maps_dir is empty):
        {output_dir}/downstream/maps/{model}_{method}/...
    These are generated by Stage 5 (--stage downstream).  When no maps are
    found, Phase 2 is silently skipped.
    """
    from src.downstream_tuning import tune_downstream_cpu, tune_downstream_gpu
    from pathlib import Path

    raf_root = args.raf_db_root
    if not (Path(raf_root) / 'train').exists():
        print(f"\n  [skip] RAF-DB not found at '{raf_root}'.")
        print("         Set --raf_db_root to the dataset root (must contain train/ and test/).")
        return {}

    # ── Auto-discover pre-computed confidence maps ─────────────────────────────
    maps_base = args.downstream_maps_dir or os.path.join(
        args.output_dir, 'downstream', 'maps'
    )
    maps_dict: dict = {}
    if os.path.isdir(maps_base):
        for entry in sorted(os.listdir(maps_base)):
            entry_path = os.path.join(maps_base, entry)
            if os.path.isdir(entry_path):
                maps_dict[entry] = entry_path
        if maps_dict:
            print(f"\n  [info] Found {len(maps_dict)} map set(s) in {maps_base}:")
            for k in maps_dict:
                print(f"           {k}")
        else:
            print(f"\n  [info] No map sub-directories found in '{maps_base}'.")

    if not maps_dict:
        print("         Phase 2 (uncertainty comparison) will be skipped.")
        print("         Run --stage downstream first to generate maps, then re-run downstream_tune.")

    is_cpu = (device == 'cpu')

    if is_cpu:
        n_seeds    = args.downstream_tune_n_seeds    or 2
        max_epochs = args.downstream_tune_max_epochs or 60
        patience   = args.downstream_tune_patience   or 12
        n_trials   = args.downstream_tune_n_trials   or 100
        search     = args.downstream_tune_search     or 'auto'
        return tune_downstream_cpu(
            raf_root         = raf_root,
            maps_dict        = maps_dict or None,
            output_dir       = args.output_dir,
            n_seeds          = n_seeds,
            n_subset         = getattr(args, 'downstream_train_subset', 0) or args.downstream_subset or 25,
            n_trials         = n_trials,
            search           = search,
            max_epochs       = max_epochs,
            patience         = patience,
            backbone         = args.downstream_backbone,
            verbose          = True,
        )
    else:
        n_seeds    = args.downstream_tune_n_seeds    or 3
        max_epochs = args.downstream_tune_max_epochs or 30
        patience   = args.downstream_tune_patience   or 7
        n_trials   = args.downstream_tune_n_trials   or 40
        search     = args.downstream_tune_search     or 'auto'
        return tune_downstream_gpu(
            raf_root   = raf_root,
            maps_dict  = maps_dict or None,
            output_dir = args.output_dir,
            n_seeds    = n_seeds,
            n_epochs   = max_epochs,
            patience   = patience,
            n_trials   = n_trials,
            search     = search,
            verbose    = True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── plot_downstream short-circuits everything else: no device selection,
    # no model/method resolution, no other stage runs. Just parse the log and
    # write plots, then exit. ──────────────────────────────────────────────
    if args.stage == 'plot_downstream':
        from scripts.plot_downstream_results import run as _plot_downstream_run
        _plot_downstream_run(Path(args.downstream_log), Path(args.downstream_plots_out))
        return

    # Auto-name output dir per dataset + partition size unless user specified one
    if args.output_dir == './outputs':
        args.output_dir = f'./outputs/{args.dataset}_ps{args.partition_size}'

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Resolve model / method selections ─────────────────────────────────────
    args.models = _resolve_models(args.models)
    active_methods: set = _resolve_methods(args.methods)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}")
    print(f"  3D Face Reconstruction Uncertainty Pipeline")
    print(f"  Device  : {device.upper()}"
          + (" (forced via --cpu)" if args.cpu else ""))
    print(f"  Stage   : {args.stage}")
    print(f"  Models  : {args.models}")
    print(f"  Methods : {sorted(active_methods)}")
    print(f"  Output  : {args.output_dir}")
    print(f"{'='*60}")

    # Shared state accumulated across stages
    image:               Optional[np.ndarray]          = None
    regressor:           Optional[UnifiedFaceRegressor] = None
    mesh_results:        Dict[str, np.ndarray]          = {}
    uncertainty_results: Dict[str, np.ndarray]          = {}
    flame_faces:         Optional[np.ndarray]           = None
    reference_images:    Optional[List[np.ndarray]]     = None
    gt_data:             Optional[List[dict]]           = None

    # ── Stage 1: EDA ──────────────────────────────────────────────────────────
    if args.stage in ('eda', 'all', 'no_downstream'):
        run_eda(args)

    # ── Load image + partition (shared across stages 2-4) ─────────────────────
    if args.stage in ('inference', 'uncertainty', 'evaluation', 'all', 'no_downstream'):
        image = _load_image(args.image_path)
        _explicit_image = image is not None
        if _explicit_image:
            h, w = image.shape[:2]
            print(f"\n[info] Test image loaded: {args.image_path}  ({h}×{w})")

        loader = FaceDatasetLoader(data_root='./datasets', subset_sizes=[args.partition_size],
                                   render_coma=(args.dataset == 'coma'))
        print(f"[info] Loading partition "
              f"(dataset={args.dataset}, size={args.partition_size}) …")
        reference_images, gt_data = _load_partition_data(args, loader)

        if image is None:
            import random as _random
            # Prefer full-resolution partition images; reference_images may contain
            # downsampled (256-px max) ref_loader images which degrade inference quality.
            _img_candidates = [it.get('image') for it in (gt_data or [])
                               if it.get('image') is not None]
            if not _img_candidates:
                _img_candidates = list(reference_images) if reference_images else []
            if _img_candidates:
                raw = _random.choice(_img_candidates)
                image = cv2.resize(raw, (224, 224))
                print(f"\n[info] No test image at '{args.image_path}' — "
                      f"using random dataset image ({raw.shape[1]}×{raw.shape[0]}) "
                      "as test input for global analysis.")
            else:
                print(f"\n[warn] Dataset '{args.dataset}' has no face photos and "
                      f"'{args.image_path}' not found. Inference will be skipped; "
                      "GT mesh statistics will still be computed.")
                image = None  # kept as None — stages 2-3 skip when image is None

    # ── Stage 2: Inference ────────────────────────────────────────────────────
    if args.stage in ('inference', 'uncertainty', 'evaluation', 'all', 'no_downstream'):
        if image is None:
            print("\n[Stage 2] Skipped — no face image available.")
        else:
            regressor, mesh_results = run_inference(args, image, device)
            flame_faces = _get_flame_faces(regressor) if regressor else None
            if flame_faces is not None:
                print(f"[info] FLAME face topology: {flame_faces.shape}")
            else:
                print("[info] FLAME faces not found — 3-D mesh plots disabled.")

    # ── Stage 3: Uncertainty ──────────────────────────────────────────────────
    if args.stage in ('uncertainty', 'evaluation', 'all', 'no_downstream'):
        if image is not None and regressor is not None:
            uncertainty_results = run_uncertainty(
                args, regressor, image, mesh_results,
                reference_images, device,
                active_methods=active_methods,
            )
        else:
            print("\n[Stage 3] Skipped — no regressor / image available.")

    # ── Stage 4: Evaluation ───────────────────────────────────────────────────
    if args.stage in ('uncertainty', 'evaluation', 'all', 'no_downstream'):
        if mesh_results or uncertainty_results or gt_data:
            run_evaluation(
                args,
                mesh_results,
                uncertainty_results,
                flame_faces,
                gt_data,
                regressor=regressor,
                image=image,
                active_methods=active_methods,
                device=device,
                reference_images=reference_images,
            )
        else:
            print("\n[Stage 4] Skipped — nothing to evaluate.")

    # ── Stage 0: Hyperparameter tuning (standalone) ──────────────────────────
    if args.stage == 'tune':
        run_hyperparam_tuning(args, device)

    # ── Stage 5: Downstream classifier ────────────────────────────────────────
    if args.stage in ('downstream', 'all'):
        run_downstream(args, device)

    # ── Stage downstream_tune: downstream hyperparameter tuning ───────────────
    if args.stage == 'downstream_tune':
        run_downstream_tune(args, device)

    print(f"\n{'='*60}")
    print("  Pipeline finished.")
    print(f"  All results → {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
