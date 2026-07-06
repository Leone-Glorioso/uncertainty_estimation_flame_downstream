"""
hyperparam_tuning.py
====================
Hyperparameter search for per-vertex uncertainty methods.

Objective
---------
Each uncertainty method exposes a handful of numerical hyperparameters
(e.g. n_passes for TTA, epsilon for Jacobian).  The "right" values depend on
the particular models and data; they cannot be reasoned from first principles.

This module provides ``tune_uncertainty_hyperparams``: given a small held-out
set of images with ground-truth FLAME vertices, it sweeps each method's
hyperparameter grid, evaluates the **average Spearman ρ** between predicted
uncertainty and actual per-vertex L2 error across the eval images, and returns
the configuration that maximises ρ (or minimises AUSE, selectable via
``objective``).

Companion: ``save_tuning_results_json`` serialises the full results — including
every config's score and a per-parameter interpretation — to a JSON file.

CPU vs GPU
----------
This module is **fully CPU-compatible**.  The only compute is model forward
passes through the already-loaded wrapper.  Estimated wall-clock times on a
modern CPU (Intel i7, ResNet-50 backbone, ~100 ms/pass):

  Method       Grid size   Images   Estimated time
  ──────────────────────────────────────────────────────────────────
  TTA          15 configs     8      ~4 min  (15 × 8 × avg 20 passes)
  Jacobian      4 configs     8      ~0.5 min (4 × 8 × avg 10 passes)
  Mahalanobis   6 configs     8      ~0.5 min (1 test + ref passes per img)
  CrossMethod   2 configs     8      ~1 min  (2 × 8 × 4 models)
  MCD           5 configs     8      ~1.5 min (5 × 8 × avg 25 passes) *
  SOL-MCD      10 configs     8      ~3.5 min (10 × 8 × avg 25 passes) *
  A-MCD         7 configs     8      ~2 min  (7 × 8 × avg 25 pairs)   *
  signal check  —             —      ~0.5 min (40 probe passes on 2 images)

  * Requires MCD checkpoint; skipped automatically if mcd_wrapper is None or
    the signal check fails.  Check runs once and is shared across all three.

  Total (no MCD):   ~6 min CPU.
  Total (with MCD): ~13 min CPU.  With GPU: ~1 min.

MCD signal check
----------------
Before running the MCD grid, ``_check_mcd_signal`` probes the checkpoint with
``n_probe_passes`` stochastic forward passes on two eval images and verifies:

  Hard failures (skip tuning):
    • No nn.Dropout layers in mcd_wrapper.model
    • All dropout passes produce identical predictions

  Soft warnings (tuning proceeds, ρ may be weak):
    • cross-image r > 0.95 — uncertainty pattern is model-intrinsic, not
      image-specific.  This was observed empirically on the trained.pt
      checkpoint (r ≈ 0.99 across three UTKFace images), driven by aggressive
      dropout (p = 0.5, three layers) in the expression encoder.  The spatial
      pattern still carries a *structural* prior over which FLAME vertices are
      sensitive to encoder perturbation; tuning will determine whether that
      prior correlates with per-vertex reconstruction error.
    • mean_unc / face_diameter > 0.05 — dropout magnitude is very large.

Usage
-----
    from src.hyperparam_tuning import (
        tune_uncertainty_hyperparams,
        save_tuning_results_json,
        print_tuning_report,
    )
    from src.data_loader import FaceDatasetLoader

    loader = FaceDatasetLoader('./datasets')
    items  = loader.load_now_subset(8)               # 8 GT-paired images

    results = tune_uncertainty_hyperparams(
        wrapper       = regressor,                   # UnifiedFaceRegressor
        eval_items    = items,
        primary_model = 'SMIRK',
        methods       = ['tta', 'jacobian', 'mahalanobis'],
        objective     = 'spearman_rho',
    )

    save_tuning_results_json(results, './outputs/tuning/hyperparams.json')
    print_tuning_report(results)

Design constraints
------------------
* This is a **separate**, self-contained process.  It does NOT modify any
  uncertainty function or the main pipeline.  The results are printed, saved,
  and returned; applying them is the caller's responsibility.
* Only the explicitly exposed hyperparameters are searched.  Internal choices
  (e.g. augmentation probability within a pass) are not changed.
* Methods that require the MCD checkpoint (mcd, sol_mcd, a_mcd) are skipped
  automatically if mcd_wrapper is None.
* Each method's search is exhaustive over its grid (no Bayesian optimisation);
  the grids are small enough (2–20 configs) that exhaustive search is fast.
"""

import json
import warnings
import itertools
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without scipy dependency."""
    n = len(x)
    if n < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(np.dot(rx, ry) / denom) if denom > 1e-12 else float('nan')


def _sparsification_ause(error: np.ndarray, uncertainty: np.ndarray,
                          n_bins: int = 20) -> float:
    """
    Normalised AUSE ∈ [0, 1].  Lower = better.

    Normalised by the random-baseline area so that a method that gives no
    information relative to random removal scores 1.0, and oracle scores 0.0.
    """
    err = np.asarray(error,       dtype=np.float64).ravel()
    unc = np.asarray(uncertainty, dtype=np.float64).ravel()
    n = len(err)
    if n < n_bins:
        return float('nan')

    fracs = np.linspace(0.0, 1.0, n_bins, endpoint=False)
    mean_all = float(err.mean())
    method_errs: List[float] = []
    oracle_errs: List[float] = []
    random_errs: List[float] = []

    for frac in fracs:
        k = int(np.floor(frac * n))
        remaining = n - k
        if remaining <= 0:
            method_errs.append(0.0); oracle_errs.append(0.0)
            random_errs.append(mean_all); continue
        method_errs.append(float(err[np.argsort(unc)[:remaining]].mean()))
        oracle_errs.append(float(err[np.argsort(err)[:remaining]].mean()))
        random_errs.append(mean_all)

    m = np.array(method_errs); o = np.array(oracle_errs); r = np.array(random_errs)
    ause        = float(np.trapz(m - o, fracs))
    ause_random = float(np.trapz(r - o, fracs))
    if abs(ause_random) < 1e-12:
        return float('nan')
    return float(np.clip(ause / ause_random, 0.0, 2.0))


def _per_vertex_l2(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Procrustes-aligned per-vertex L2 error (5023,) float32."""
    src_c = pred - pred.mean(axis=0)
    tgt_c = gt   - gt.mean(axis=0)
    sn = np.linalg.norm(src_c); tn = np.linalg.norm(tgt_c)
    if sn < 1e-10 or tn < 1e-10:
        return np.linalg.norm(pred - gt, axis=1).astype(np.float32)
    H  = (src_c / sn).T @ (tgt_c / tn)
    U, _, Vt = np.linalg.svd(H)
    d  = np.linalg.det(Vt.T @ U.T)
    R  = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    aligned = (tn / sn) * (src_c @ R.T) + gt.mean(axis=0)
    return np.linalg.norm(aligned - gt, axis=1).astype(np.float32)


def _score_uncertainty(error: np.ndarray, uncertainty: np.ndarray,
                        objective: str) -> float:
    """Scalar score; ALWAYS higher = better (AUSE is negated)."""
    e = np.asarray(error,       dtype=np.float64).ravel()
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    if objective == 'ause':
        raw = _sparsification_ause(e, u)
        return -raw if not np.isnan(raw) else float('-inf')
    return _spearman_rho(e, u)


def _cartesian(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo))
            for combo in itertools.product(*[grid[k] for k in keys])]


def _check_mcd_signal(
    mcd_wrapper,
    images: List[np.ndarray],
    n_probe_passes: int = 20,
    verbose: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Quick sanity check before running the full MCD hyperparameter grid.

    Runs `n_probe_passes` stochastic forward passes on the first two eval images,
    then reports stochasticity (are passes different?) and spatial structure
    (is the uncertainty map non-uniform and image-specific?).

    Returns
    -------
    ok : bool
        True  — checkpoint produces real stochastic variance; proceed with grid.
        False — hard failure: no dropout layers found, or all passes identical.
                (Soft warnings do NOT flip this to False.)
    diagnostics : dict
        n_dropout_layers, dropout_p, mean_unc_mm, max_unc_mm, cv,
        cross_image_r, warnings.

    Soft warnings (ok stays True)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cross_image_r > 0.95
        The uncertainty map is nearly identical across images — the pattern is
        model-intrinsic (fixed by network topology and FLAME basis functions)
        rather than image-specific difficulty.  MCD will still produce a useful
        *structural* prior (which face regions the encoder is most uncertain
        about) even when r ≈ 1, and that prior can correlate with average
        per-vertex error.  Tuning proceeds; the ρ score will determine utility.

    mean_unc / face_diameter > 0.05
        Dropout collapses predictions very aggressively.  Useful signal may
        still exist (the spatial pattern can still rank vertices correctly), but
        calibration quality will be poor.
    """
    try:
        import torch as _torch
    except ImportError:
        if verbose:
            print("  [MCD check] FAILED — torch not available.")
        return False, {'reason': 'torch not importable'}

    pytorch_model = getattr(mcd_wrapper, 'model', None)
    if pytorch_model is None:
        if verbose:
            print("  [MCD check] FAILED — mcd_wrapper has no .model attribute.")
        return False, {'reason': 'no .model attribute on mcd_wrapper'}

    dropout_layers = [m for m in pytorch_model.modules()
                      if isinstance(m, _torch.nn.Dropout)]
    n_dropout = len(dropout_layers)

    if n_dropout == 0:
        msg = ("No nn.Dropout layers found in mcd_wrapper.model — "
               "use a checkpoint retrained with dropout in the expression encoder.")
        if verbose:
            print(f"  [MCD check] FAILED — {msg}")
        return False, {'reason': msg}

    def _run_stochastic(img: np.ndarray) -> np.ndarray:
        """N stochastic passes; returns stacked vertices (N, 5023, 3)."""
        passes = []
        try:
            for m in pytorch_model.modules():
                if isinstance(m, _torch.nn.Dropout):
                    m.training = True
            with _torch.no_grad():
                for _ in range(n_probe_passes):
                    v = np.asarray(mcd_wrapper.get_vertices(img), dtype=np.float32)
                    passes.append(v)
        finally:
            pytorch_model.eval()
        return np.stack(passes)                                   # (N, 5023, 3)

    stack1 = _run_stochastic(images[0])
    unc1   = np.linalg.norm(stack1.std(axis=0), axis=-1)         # (5023,)

    # Hard failure 1: all dropout passes are literally identical
    if float(unc1.max()) < 1e-8:
        msg = "All dropout passes give identical predictions — dropout is not active."
        if verbose:
            print(f"  [MCD check] FAILED — {msg}")
        return False, {'reason': msg}

    # Hard failure 2: uncertainty is negligible relative to face scale.
    # float32 clone operations produce ~1e-5 mm of numerical noise even for
    # deterministic models — an absolute threshold misses this.  A relative
    # threshold (uncertainty must be ≥ 0.01 % of face diameter) catches both
    # cases: real signal is typically 1–10 mm on a ~450 mm face (ratio 0.003),
    # while floating-point noise is < 0.0001 mm (ratio < 1e-6).
    face_diam_m = float(np.linalg.norm(stack1[0].max(axis=0) - stack1[0].min(axis=0)))
    if face_diam_m > 1e-6:
        rel_unc = float(unc1.mean()) / face_diam_m
        if rel_unc < 1e-4:
            msg = (
                f"MCD uncertainty ({unc1.mean()*1000:.6f} mm mean) is negligible "
                f"relative to face scale ({face_diam_m*1000:.1f} mm diameter, "
                f"ratio = {rel_unc:.2e} < 1e-4) — dropout is not affecting "
                "predictions (likely forward pass does not use Dropout layers)."
            )
            if verbose:
                print(f"  [MCD check] FAILED — {msg}")
            return False, {'reason': msg}

    cv          = float(unc1.std() / (unc1.mean() + 1e-10))
    mean_unc_mm = float(unc1.mean() * 1000)
    max_unc_mm  = float(unc1.max() * 1000)

    warnings_out: List[str] = []

    # Cross-image correlation: model-intrinsic vs instance-specific
    cross_r: Optional[float] = None
    if len(images) >= 2:
        stack2 = _run_stochastic(images[1])
        unc2   = np.linalg.norm(stack2.std(axis=0), axis=-1)
        cross_r = float(np.corrcoef(unc1, unc2)[0, 1])
        if cross_r > 0.95:
            warnings_out.append(
                f"cross-image r = {cross_r:.3f} > 0.95: uncertainty map is nearly "
                "identical across different face images — the pattern is model-intrinsic "
                "(driven by FLAME basis sensitivity and dropout placement) rather than "
                "image-specific difficulty.  Tuning continues; ρ will reflect whether "
                "this structural prior correlates with mean per-vertex error."
            )

    # High-magnitude warning
    face_diam_m = float(np.linalg.norm(stack1[0].max(0) - stack1[0].min(0)))
    if face_diam_m > 1e-6 and (unc1.mean() / face_diam_m) > 0.05:
        warnings_out.append(
            f"mean_unc / face_diameter = {unc1.mean()/face_diam_m:.3f} > 0.05: "
            "dropout is very aggressive — uncertainty may be dominated by random "
            "prediction collapse rather than structured epistemic signal."
        )

    diag: Dict[str, Any] = {
        'n_dropout_layers': n_dropout,
        'dropout_p':        [float(d.p) for d in dropout_layers],
        'mean_unc_mm':      mean_unc_mm,
        'max_unc_mm':       max_unc_mm,
        'cv':               cv,
        'cross_image_r':    cross_r,
        'warnings':         warnings_out,
    }

    if verbose:
        status = "PASSED" if not warnings_out else "PASSED (with warnings)"
        print(f"  [MCD check] {status} — {n_dropout} dropout layer(s)  "
              f"p={diag['dropout_p']}  mean_unc={mean_unc_mm:.4f}mm  CV={cv:.3f}")
        if cross_r is not None:
            print(f"              cross-image r = {cross_r:.3f}")
        for w in warnings_out:
            print(f"  [warn] {w}")

    return True, diag


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter grids
# ──────────────────────────────────────────────────────────────────────────────

_TTA_GRID: Dict[str, List[Any]] = {
    'n_passes':  [5, 10, 15, 20, 30],
    'aggregate': ['std', 'norm_std', 'var'],
}

_MCD_GRID: Dict[str, List[Any]] = {
    # Lower bound extended to 5: previous runs peaked at n_passes=10 (the old
    # minimum) because the dynamic-mean Procrustes alignment suppressed signal
    # at higher N.  Now that alignment uses a fixed eval-mode base, convergence
    # should be monotone; including n_passes=5 probes whether the estimate is
    # already stable with fewer passes.
    'n_passes': [5, 10, 20, 30, 50],
}

_SOL_MCD_GRID: Dict[str, List[Any]] = {
    # Same lower-bound extension as MCD for the same reason.
    'n_passes':        [5, 10, 20, 30, 50],
    # n_stable_layers now freezes the FIRST N dropout layers (closest to input),
    # keeping only the output-proximal layer(s) stochastic.  This is the
    # reverse of the original SOL-MCD paper; see calculate_sol_mcd_uncertainty
    # for the rationale.
    'n_stable_layers': [1, 2],
}

_AMCD_GRID: Dict[str, List[Any]] = {
    # Extended upper range: the original grid peaked at n_pairs=15 (ρ = -0.050)
    # and dropped at 25, but that was before basis-normalisation removed the
    # structural anti-correlation.  More pairs reduce estimator variance and
    # may expose a genuine positive signal once the confound is removed.
    'n_pairs': [5, 10, 15, 20, 25, 30, 40],
}

_JACOBIAN_GRID: Dict[str, List[Any]] = {
    'n_directions': [5, 10, 20, 30],
    # epsilon is accepted by the function for API compatibility but is NOT
    # used by the augmentation-direction secant method; augmentation magnitudes
    # are determined by _augment_image().  Only n_directions matters here.
    'epsilon': [None],
}

_MAHALANOBIS_GRID: Dict[str, List[Any]] = {
    # regularise_cov is now in units of the mean PCA eigenvalue (global PCA).
    # The PCA subspace is already low-rank (rank ≤ N−1 = 6 with N≈7 images),
    # so the condition number is much lower than for the old per-vertex 3×3
    # matrices.  Lower values can now preserve genuine covariance structure
    # instead of forcing near-Euclidean behaviour.
    'regularise_cov': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
}

_CROSS_GRID: Dict[str, List[Any]] = {
    'normalise': [False, True],
}


# ──────────────────────────────────────────────────────────────────────────────
# Per-parameter interpretation texts
# ──────────────────────────────────────────────────────────────────────────────

def _interpret_tta(best: Dict[str, Any], all_scores: Dict[str, float]) -> Dict[str, str]:
    n = best['n_passes']
    agg = best['aggregate']
    agg_text = {
        'std':      ('mean of per-coordinate standard deviations treats X/Y/Z '
                     'independently — sensitive to axes that move independently'),
        'norm_std': ('L2 norm of the per-coordinate std vector measures total 3D '
                     'spread — usually best when uncertainty is roughly isotropic'),
        'var':      ('mean of per-coordinate variances weights large outlier deviations '
                     'more heavily — useful when a few passes differ drastically'),
    }.get(agg, agg)
    n_trend = ('converges quickly — fewer passes than the maximum suffice'
               if n <= 10 else
               'benefits from many samples — the model is sensitive to augmentation or'
               ' the data is challenging' if n >= 20 else
               'balanced between stability and runtime')
    return {
        'n_passes': (f"Best: {n} (searched {_TTA_GRID['n_passes']}). "
                     f"The estimate {n_trend}. "
                     "Each pass applies one random augmentation (blur, jitter, noise, "
                     "crop, rotation) and re-runs inference; more passes = smoother "
                     "variance estimate at the cost of extra forward passes."),
        'aggregate': (f"Best: '{agg}'. "
                      f"The {agg_text}. "
                      "All three collapse the (5023, 3) spread to (5023, 1) scalars "
                      "but weight the three spatial coordinates differently."),
        'recommendation': (
            f"Pass --n_tta {n} to main.py. "
            f"The aggregate='{agg}' choice is fixed inside calculate_tta_uncertainty "
            "and must be passed explicitly when calling the function directly."
        ),
    }


def _interpret_mcd(best: Dict[str, Any]) -> Dict[str, str]:
    n = best['n_passes']
    trend = 'converges with few samples' if n <= 10 else 'needs many samples for stable epistemic estimates'
    return {
        'n_passes': (f"Best: {n} (searched {_MCD_GRID['n_passes']}). "
                     f"MC Dropout {trend}. "
                     "Each pass activates different dropout masks, sampling a different "
                     "sub-network from the approximate posterior over weights. "
                     "Pass --n_mcd {n} to main.py."),
        'recommendation': f"Pass --n_mcd {n} to main.py.",
    }


def _interpret_sol_mcd(best: Dict[str, Any]) -> Dict[str, str]:
    n = best['n_passes']; sl = best['n_stable_layers']
    return {
        'n_passes': (f"Best: {n}. "
                     "After basis normalisation and Procrustes alignment, SOL-MCD "
                     "convergence depends on the residual signal magnitude; "
                     "more passes help stabilise the normalised variance estimate."),
        'n_stable_layers': (f"Best: {sl} stable layer(s) frozen at the INPUT end "
                            "(note: freeze direction is reversed relative to the original "
                            f"SOL-MCD paper — the FIRST {sl} dropout layer(s) are frozen, "
                            "keeping the output-proximal layer(s) stochastic). "
                            "Freezing 1 input layer leaves the final 2 layers active, "
                            "which carry more expression-specific features and produce "
                            "a less uniformly expression-basis-driven uncertainty pattern. "
                            "Freezing 2 input layers leaves only the final output-adjacent "
                            "dropout active — maximum concentration of late-layer signal."),
        'recommendation': f"Pass --n_mcd {n} and set n_stable_layers={sl} in the SOL-MCD call.",
    }


def _interpret_amcd(best: Dict[str, Any]) -> Dict[str, str]:
    n = best['n_pairs']
    trend = ('converges quickly with few pairs'
             if n <= 10 else
             'benefits from more pairs — higher n reduces estimator variance '
             'and stabilises the basis-normalised uncertainty map'
             if n >= 20 else
             'balanced between stability and runtime')
    return {
        'n_pairs': (f"Best: {n} antithetic pairs (= {n*2} total forward passes, "
                    f"searched {_AMCD_GRID['n_pairs']}). "
                    "Antithetic variates pair each dropout mask with its bitwise complement, "
                    "halving the estimator variance for the same compute budget. "
                    f"The estimate {trend}. "
                    "After expression-basis normalisation, more pairs help stabilise the "
                    "residual uncertainty signal that remains after removing the structural "
                    "expression-sensitivity confound."),
        'recommendation': f"Pass n_pairs={n} to calculate_antithetic_mcd_uncertainty.",
    }


def _interpret_jacobian(best: Dict[str, Any]) -> Dict[str, str]:
    nd = best['n_directions']
    n_per_type = max(1, nd // 4)
    trend = ('converges quickly — fewer passes than the maximum suffice'
             if nd <= 10 else
             'benefits from more passes — variance in per-augmentation responses is high'
             if nd >= 20 else
             'balanced between stability and runtime')
    return {
        'n_directions': (f"Best: {nd} augmentation-direction probes "
                         f"(searched {_JACOBIAN_GRID['n_directions']}). "
                         f"Cycling through 4 types (blur / jitter / crop_scale / rotate) "
                         f"gives ~{n_per_type} pass(es) per type with independently re-sampled "
                         f"augmentation parameters.  The estimate {trend}.  "
                         "Each probe runs the full augmented image through the model "
                         "(secant approximation) rather than a finite-difference ε-step, "
                         "so the signal is proportional to the genuine model response "
                         "to augmentation rather than the local gradient."),
        'recommendation': f"Pass --n_jacobian {nd} to main.py.",
    }


def _interpret_mahalanobis(best: Dict[str, Any]) -> Dict[str, str]:
    rc = best['regularise_cov']
    regime = ('strong regularisation (near-Euclidean)' if rc >= 1.0 else
              'moderate regularisation' if rc >= 0.1 else
              'light regularisation (PCA structure preserved)')
    return {
        'regularise_cov': (f"Best: {rc} ({regime}). "
                           "The ridge coefficient λ adds λ × mean(eigenvalue) to all "
                           "PCA eigenvalues, preventing singular directions in the global "
                           "covariance.  "
                           f"{'Low λ means the PCA covariance structure is informative — the reference set is large or consistent enough that the top PCs are well-estimated.' if rc < 0.1 else 'Higher λ shrinks toward scaled Euclidean distance, the stable regime when N is small (<7 reference images).'} "
                           "Unlike the old per-vertex approach, even λ = 0.01 is stable "
                           "because the SVD naturally regularises by truncating to the "
                           "K = N − 1 dimensional PCA subspace."),
        'recommendation': f"Set regularise_cov={rc} in the Mahalanobis call.",
    }


def _interpret_cross(best: Dict[str, Any]) -> Dict[str, str]:
    norm = best['normalise']
    return {
        'normalise': (
            f"Best: normalise={norm}. " + (
                "Generalised Procrustes Analysis (GPA) alignment aligns all four "
                "methods (SMIRK/DECA/EMOCA/SHeaP) to their shared Fréchet mean "
                "before computing disagreement.  This removes global scale/pose "
                "differences without anchoring to any single method, so the "
                "measured variance reflects symmetric inter-method disagreement "
                "rather than 'disagreement with SMIRK'."
                if norm else
                "Raw (un-aligned) disagreement was better — the global pose offsets "
                "between methods are small enough that GPA alignment would remove "
                "meaningful signal alongside the noise."
            )
        ),
        'recommendation': f"Set normalise={norm} in the CrossMethod call.",
    }


_INTERPRET_FNS = {
    'tta':          lambda best, all_s: _interpret_tta(best, all_s),
    'mcd':          lambda best, _:    _interpret_mcd(best),
    'sol_mcd':      lambda best, _:    _interpret_sol_mcd(best),
    'a_mcd':        lambda best, _:    _interpret_amcd(best),
    'jacobian':     lambda best, _:    _interpret_jacobian(best),
    'mahalanobis':  lambda best, _:    _interpret_mahalanobis(best),
    'cross':        lambda best, _:    _interpret_cross(best),
}

_METHOD_DISPLAY = {
    'tta':         'TTA (Test-Time Augmentation)',
    'mcd':         'MCD (Monte Carlo Dropout)',
    'sol_mcd':     'SOL-MCD (Stable Output Layers MCD)',
    'a_mcd':       'A-MCD (Antithetic MCD)',
    'jacobian':    'Jacobian Sensitivity',
    'mahalanobis': 'Mahalanobis Distance',
    'cross':       'Cross-Method Disagreement',
}


# ──────────────────────────────────────────────────────────────────────────────
# Public: main tuning function
# ──────────────────────────────────────────────────────────────────────────────

def tune_uncertainty_hyperparams(
    wrapper,
    eval_items:    List[Dict],
    primary_model: str = 'SMIRK',
    mcd_wrapper=None,
    methods:       Optional[List[str]] = None,
    objective:     str = 'spearman_rho',
    verbose:       bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Grid-search hyperparameters for each uncertainty method so that the
    predicted uncertainty best matches the true per-vertex geometric error.

    Runs entirely on the device of the already-loaded wrapper (CPU or GPU).

    Parameters
    ----------
    wrapper       : UnifiedFaceRegressor instance (must contain primary_model).
    eval_items    : list of dicts, each with 'image' (H×W×3 uint8) and
                    'gt_vertices' (5023, 3) float32.  Minimum 2; 5–10 recommended.
    primary_model : which model to use for single-model methods.
    mcd_wrapper   : SMIRK wrapper with MCD checkpoint; needed for mcd / sol_mcd /
                    a_mcd.  If None, those methods are skipped.
    methods       : which methods to tune.  Valid keys: 'tta', 'mcd', 'cross',
                    'jacobian', 'mahalanobis', 'sol_mcd', 'a_mcd'.
                    Defaults to all feasible given the provided data and wrappers.
    objective     : 'spearman_rho' (maximise) or 'ause' (minimise, negated internally
                    so higher is always better).
    verbose       : print progress.

    Returns
    -------
    Dict[method_name, result_dict] where each result_dict has:
        best_params  : {param: value} — plug these into the uncertainty function
        best_score   : float — higher = better regardless of objective
        metric       : str — 'spearman_rho' or 'ause'
        n_configs    : int — total configs evaluated
        n_images     : int — eval images used
        all_configs  : list of {params, score, rank} — every config tried, ranked
        interpretation: dict — per-parameter and overall explanations (added by
                        save_tuning_results_json)
    """
    valid_items = [
        it for it in eval_items
        if it.get('image') is not None and it.get('gt_vertices') is not None
    ]
    if len(valid_items) < 2:
        raise ValueError(
            f"Need ≥ 2 items with 'image' and 'gt_vertices'; got {len(valid_items)}."
        )
    if len(valid_items) < 5 and verbose:
        warnings.warn(
            f"Only {len(valid_items)} eval images — estimates may be noisy. "
            "Recommend ≥ 5 for stable Spearman ρ.", UserWarning, stacklevel=2,
        )

    _ALL = ['tta', 'mcd', 'cross', 'jacobian', 'mahalanobis', 'sol_mcd', 'a_mcd']
    if methods is None:
        methods = _ALL
    methods = [m.lower() for m in methods]

    wmap = getattr(wrapper, 'models', {})
    pw = wmap.get(primary_model)
    if pw is None:
        raise ValueError(
            f"Primary model '{primary_model}' not in wrapper.models "
            f"(available: {list(wmap.keys())})."
        )

    results: Dict[str, Dict[str, Any]] = {}

    # ── Pre-compute primary model predictions ─────────────────────────────────
    if verbose:
        print(f"\n[hyperparam_tuning] Precomputing {primary_model} on "
              f"{len(valid_items)} images …")
    primary_preds, gt_list, images = [], [], []
    for item in valid_items:
        try:
            pred = np.asarray(pw.get_vertices(item['image']), dtype=np.float32)
        except Exception as exc:
            if verbose: print(f"    [warn] forward pass failed: {exc}")
            continue
        primary_preds.append(pred)
        gt_list.append(np.asarray(item['gt_vertices'], dtype=np.float32))
        images.append(item['image'])
    if len(images) < 2:
        raise ValueError("Fewer than 2 successful forward passes.")
    if verbose:
        print(f"    OK — {len(images)} images processed.")

    errors = [_per_vertex_l2(gt, pred) for gt, pred in zip(gt_list, primary_preds)]

    # ── Lazy MCD gate ─────────────────────────────────────────────────────────
    # Shared across MCD, SOL-MCD, A-MCD so the signal check runs at most once.
    _mcd_ok:   Optional[bool]        = None
    _mcd_diag: Dict[str, Any]        = {}

    def _ensure_mcd_checked() -> bool:
        nonlocal _mcd_ok, _mcd_diag
        if _mcd_ok is None:
            if verbose:
                print("\n[MCD signal check] Probing checkpoint stochasticity …")
            _mcd_ok, _mcd_diag = _check_mcd_signal(
                mcd_wrapper, images, verbose=verbose)
        return bool(_mcd_ok)

    def _avg_score(unc_list: List[np.ndarray]) -> float:
        scores = [_score_uncertainty(e, u.ravel(), objective)
                  for e, u in zip(errors, unc_list)]
        good = [s for s in scores if not np.isnan(s)]
        return float(np.mean(good)) if good else float('-inf')

    def _run_configs(name: str, grid: Dict, call_fn) -> Dict[str, Any]:
        configs = _cartesian(grid)
        if verbose:
            print(f"\n[{name}] {len(configs)} config(s) × {len(images)} images …")
        all_cfg_scores: List[Dict] = []
        best_cfg, best_score = None, float('-inf')
        for cfg in configs:
            unc_list = []
            for img in images:
                try:
                    u = call_fn(img, cfg)
                    unc_list.append(np.asarray(u, dtype=np.float32))
                except Exception:
                    unc_list.append(np.zeros(5023, dtype=np.float32))
            sc = _avg_score(unc_list)
            all_cfg_scores.append({'params': dict(cfg), 'score': float(sc) if not np.isnan(sc) else None})
            if sc > best_score:
                best_score, best_cfg = sc, cfg
            if verbose:
                sc_str = f'{sc:.4f}' if not np.isnan(sc) else 'nan'
                print(f"    {cfg}  →  {objective}={sc_str}")
        # Add rank (1 = best)
        sorted_by_score = sorted(
            all_cfg_scores,
            key=lambda x: x['score'] if x['score'] is not None else float('-inf'),
            reverse=True,
        )
        rank_map = {id(c): i + 1 for i, c in enumerate(sorted_by_score)}
        for c in all_cfg_scores:
            c['rank'] = rank_map[id(c)]
        if verbose:
            print(f"  ★ Best {name}: {best_cfg}  {objective}={best_score:.4f}")
        return {
            'best_params': dict(best_cfg) if best_cfg else {},
            'best_score':  float(best_score),
            'metric':      objective,
            'n_configs':   len(configs),
            'n_images':    len(images),
            'all_configs': all_cfg_scores,
        }

    # ════════════════════════════════════════════════════════════════════════════
    # 1. TTA
    # ════════════════════════════════════════════════════════════════════════════
    if 'tta' in methods:
        from src.uncertainty import calculate_tta_uncertainty

        def _tta_seed(img: np.ndarray, n_passes: int) -> int:
            # Stable seed derived from image content + n_passes so all aggregate
            # variants with the same (image, n_passes) see identical augmentations.
            # This eliminates augmentation-luck noise from aggregate comparisons
            # without changing TTA's default random behaviour in production.
            return int(
                (int(img.ravel()[:64].astype(np.int64).sum()) + n_passes * 7919)
                % (2 ** 31)
            )

        results['tta'] = _run_configs(
            'TTA', _TTA_GRID,
            lambda img, cfg: calculate_tta_uncertainty(
                pw, img,
                n_passes=cfg['n_passes'],
                aggregate=cfg['aggregate'],
                seed=_tta_seed(img, cfg['n_passes'])),
        )

    # ════════════════════════════════════════════════════════════════════════════
    # 2. MCD
    # ════════════════════════════════════════════════════════════════════════════
    if 'mcd' in methods:
        if mcd_wrapper is None:
            if verbose: print("\n[MCD] Skipped — no mcd_wrapper provided.")
        elif not _ensure_mcd_checked():
            if verbose: print("\n[MCD] Skipped — signal check failed (see above).")
        else:
            from src.uncertainty import calculate_mcd_uncertainty
            results['mcd'] = _run_configs(
                'MCD', _MCD_GRID,
                lambda img, cfg: calculate_mcd_uncertainty(
                    mcd_wrapper, img, n_passes=cfg['n_passes']),
            )

    # ════════════════════════════════════════════════════════════════════════════
    # 3. Cross-Method Disagreement
    # ════════════════════════════════════════════════════════════════════════════
    if 'cross' in methods:
        from src.uncertainty import calculate_cross_method_disagreement
        results['cross'] = _run_configs(
            'CrossMethod', _CROSS_GRID,
            lambda img, cfg: calculate_cross_method_disagreement(
                wrapper=wrapper, image=img, normalise=cfg['normalise']),
        )

    # ════════════════════════════════════════════════════════════════════════════
    # 4. Jacobian
    # ════════════════════════════════════════════════════════════════════════════
    if 'jacobian' in methods:
        from src.uncertainty import calculate_jacobian_sensitivity_uncertainty

        def _jacobian_seed(img: np.ndarray, n_directions: int) -> int:
            # Stable seed derived from image content + n_directions so all
            # configs with the same (image, n_directions) see identical aug params.
            return int(
                (int(img.ravel()[:64].astype(np.int64).sum()) + n_directions * 6271)
                % (2 ** 31)
            )

        results['jacobian'] = _run_configs(
            'Jacobian', _JACOBIAN_GRID,
            lambda img, cfg: calculate_jacobian_sensitivity_uncertainty(
                pw, img,
                n_directions=cfg['n_directions'],
                epsilon=cfg['epsilon'],
                seed=_jacobian_seed(img, cfg['n_directions'])),
        )

    # ════════════════════════════════════════════════════════════════════════════
    # 5. Mahalanobis  (leave-one-out reference)
    # ════════════════════════════════════════════════════════════════════════════
    if 'mahalanobis' in methods:
        from src.uncertainty import calculate_mahalanobis_uncertainty

        def _mahal_call(img, cfg):
            i = next((j for j, im in enumerate(images) if im is img), None)
            ref = [im for j, im in enumerate(images) if j != i] if i is not None else images
            if len(ref) < 4:
                ref = images
            return calculate_mahalanobis_uncertainty(
                pw, img, reference_images=ref,
                regularise_cov=cfg['regularise_cov'])

        results['mahalanobis'] = _run_configs(
            'Mahalanobis', _MAHALANOBIS_GRID, _mahal_call,
        )

    # ════════════════════════════════════════════════════════════════════════════
    # 6. SOL-MCD
    # ════════════════════════════════════════════════════════════════════════════
    if 'sol_mcd' in methods:
        if mcd_wrapper is None:
            if verbose: print("\n[SOL-MCD] Skipped — no mcd_wrapper provided.")
        elif not _ensure_mcd_checked():
            if verbose: print("\n[SOL-MCD] Skipped — signal check failed (see above).")
        else:
            from src.uncertainty import calculate_sol_mcd_uncertainty
            results['sol_mcd'] = _run_configs(
                'SOL-MCD', _SOL_MCD_GRID,
                lambda img, cfg: calculate_sol_mcd_uncertainty(
                    mcd_wrapper, img,
                    n_passes=cfg['n_passes'],
                    n_stable_layers=cfg['n_stable_layers']),
            )

    # ════════════════════════════════════════════════════════════════════════════
    # 7. A-MCD
    # ════════════════════════════════════════════════════════════════════════════
    if 'a_mcd' in methods:
        if mcd_wrapper is None:
            if verbose: print("\n[A-MCD] Skipped — no mcd_wrapper provided.")
        elif not _ensure_mcd_checked():
            if verbose: print("\n[A-MCD] Skipped — signal check failed (see above).")
        else:
            from src.uncertainty import calculate_antithetic_mcd_uncertainty
            results['a_mcd'] = _run_configs(
                'A-MCD', _AMCD_GRID,
                lambda img, cfg: calculate_antithetic_mcd_uncertainty(
                    mcd_wrapper, img, n_pairs=cfg['n_pairs']),
            )

    # ── Summary ────────────────────────────────────────────────────────────────
    if verbose and results:
        print("\n" + "═" * 60)
        print(f"  Tuning Summary  (objective: {objective}, higher = better)")
        print("  " + "─" * 56)
        for mname, r in sorted(results.items(), key=lambda x: -x[1]['best_score']):
            sc = r['best_score']
            bp = r['best_params']
            print(f"  {mname:<16s}  {objective}={sc:+.4f}  best={bp}")
        print("═" * 60)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Public: JSON serialisation with interpretation
# ──────────────────────────────────────────────────────────────────────────────

def save_tuning_results_json(
    results: Dict[str, Dict[str, Any]],
    save_path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Serialise the full tuning results to a JSON file.

    The file is self-documenting: every parameter choice is explained in plain
    English under an ``interpretation`` key so the file can be read without
    reference to the source code.

    Structure
    ---------
    {
      "meta": { objective, primary_model, n_eval_images, timestamp, … },
      "results": {
        "<method>": {
          "display_name":  str,
          "best_params":   {param: value, …},
          "best_score":    float,
          "metric":        "spearman_rho" | "ause",
          "n_configs":     int,
          "n_images":      int,
          "all_configs":   [{params, score, rank}, …],
          "interpretation": {
            "summary":        str,   ← one-paragraph plain-English summary
            "per_param":      {param_name: explanation_str, …},
            "recommendation": str,   ← copy-paste command for main.py
          }
        }
      },
      "summary": {
        "ranking":          [{method, score}, …],   ← best-to-worst
        "best_method":      str,
        "score_scale_note": str,
        "notes":            [str, …],
      }
    }

    Parameters
    ----------
    results   : dict returned by ``tune_uncertainty_hyperparams``.
    save_path : output path (parent directories are created as needed).
    meta      : optional extra metadata added verbatim to the "meta" block.
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    objective = next(
        (r['metric'] for r in results.values()), 'spearman_rho'
    )
    n_images = next((r['n_images'] for r in results.values()), 0)

    # ── Build meta block ──────────────────────────────────────────────────────
    meta_block: Dict[str, Any] = {
        'objective':     objective,
        'n_eval_images': n_images,
        'timestamp':     datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'score_convention': (
            'Spearman ρ: range [−1, +1], higher = better.  '
            'A method with ρ < 0 is anti-correlated with true error (wrong direction).  '
            'ρ ∈ [0.1, 0.3] = weak signal; [0.3, 0.6] = moderate; > 0.6 = strong.'
        ) if objective == 'spearman_rho' else (
            'AUSE (normalised): range [0, 1], lower = better.  '
            'Stored as −AUSE so that higher is always better.  '
            'A method near 0 (stored as ≈ 0) removes vertices as well as the oracle.'
        ),
    }
    if meta:
        meta_block.update(meta)

    # ── Build per-method result blocks ────────────────────────────────────────
    results_block: Dict[str, Any] = {}
    for mname, r in results.items():
        best  = r.get('best_params', {})
        score = r.get('best_score', float('nan'))
        interp_fn = _INTERPRET_FNS.get(mname)
        all_scores_map = {
            str(c['params']): c['score']
            for c in r.get('all_configs', [])
            if c['score'] is not None
        }
        if interp_fn is not None:
            try:
                per_param = interp_fn(best, all_scores_map)
                recommendation = per_param.pop('recommendation', '')
            except Exception:
                per_param = {}; recommendation = ''
        else:
            per_param = {}; recommendation = ''

        # Terse one-line summary
        score_str = f'{score:+.4f}' if not np.isnan(score) else 'n/a'
        if objective == 'spearman_rho':
            quality = ('strong' if score > 0.5 else
                       'moderate' if score > 0.25 else
                       'weak' if score > 0.05 else
                       'near-zero (uninformative)' if score >= -0.05 else
                       'NEGATIVE (anti-correlated — uncertainty points away from true error)')
        else:
            quality = ('near-oracle' if score > -0.1 else
                       'good' if score > -0.3 else
                       'moderate' if score > -0.6 else
                       'weak')

        summary = (
            f"{_METHOD_DISPLAY.get(mname, mname)} achieved "
            f"{objective.replace('_', ' ')} = {score_str} ({quality}) "
            f"with the best configuration {best}. "
            f"Evaluated {r.get('n_configs', '?')} configurations over "
            f"{r.get('n_images', '?')} ground-truth-paired images."
        )

        results_block[mname] = {
            'display_name':  _METHOD_DISPLAY.get(mname, mname),
            'best_params':   best,
            'best_score':    float(score) if not np.isnan(score) else None,
            'metric':        objective,
            'n_configs':     r.get('n_configs', 0),
            'n_images':      r.get('n_images', 0),
            'all_configs':   r.get('all_configs', []),
            'interpretation': {
                'summary':       summary,
                'quality_label': quality,
                'per_param':     per_param,
                'recommendation': recommendation,
            },
        }

    # ── Build summary block ───────────────────────────────────────────────────
    ranked = sorted(
        [(m, r.get('best_score', float('-inf'))) for m, r in results.items()],
        key=lambda x: x[1], reverse=True,
    )
    best_method = ranked[0][0] if ranked else None

    anti_corr = [m for m, s in ranked if s < -0.05]
    uninform   = [m for m, s in ranked if -0.05 <= s < 0.05]
    notes: List[str] = []
    if anti_corr:
        notes.append(
            f"Methods with ρ < 0 are ANTI-CORRELATED with true error — they assign "
            f"highest uncertainty to the LEAST erroneous vertices: {anti_corr}. "
            "Check that gt_vertices and predicted vertices are in the same coordinate "
            "frame and that the reference set for Mahalanobis is not polluted by "
            "test images."
        )
    if uninform:
        notes.append(
            f"Methods near ρ = 0 are effectively uninformative on this data: "
            f"{uninform}. Consider whether these methods are appropriate for the "
            "dataset or whether the hyperparameter grid needs widening."
        )
    if not notes:
        notes.append(
            "All methods show positive correlation with true error — "
            "hyperparameter search succeeded. Use the ranked best_score values "
            "to decide which method to deploy."
        )

    summary_block: Dict[str, Any] = {
        'ranking': [
            {'method': m, 'display_name': _METHOD_DISPLAY.get(m, m),
             'score': float(s) if not np.isnan(s) else None}
            for m, s in ranked
        ],
        'best_method':      best_method,
        'best_method_name': _METHOD_DISPLAY.get(best_method, best_method) if best_method else None,
        'score_scale_note': meta_block['score_convention'],
        'notes':            notes,
    }

    payload = {
        'meta':    meta_block,
        'results': results_block,
        'summary': summary_block,
    }

    with open(save_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"[tuning] Results saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Public: pretty-print helper
# ──────────────────────────────────────────────────────────────────────────────

def print_tuning_report(results: Dict[str, Dict[str, Any]]) -> None:
    """Pretty-print the dict returned by tune_uncertainty_hyperparams."""
    if not results:
        print("No tuning results available.")
        return
    ranked = sorted(results.items(), key=lambda x: x[1].get('best_score', float('-inf')), reverse=True)
    print("\n┌────────────────────────────────────────────────────────────────────┐")
    print("│           Uncertainty Hyperparameter Tuning — Final Report        │")
    print("├───────────────┬──────────┬─────────────────────────────────────────┤")
    print("│ Method        │ Score    │ Best Hyperparameters                    │")
    print("├───────────────┼──────────┼─────────────────────────────────────────┤")
    for mname, r in ranked:
        sc = r.get('best_score', float('nan'))
        bp = r.get('best_params', {})
        param_str = ', '.join(f"{k}={v}" for k, v in bp.items())[:41]
        sc_str = f'{sc:+.4f}' if not np.isnan(sc) else '   n/a'
        print(f"│ {mname:<13s} │ {sc_str}  │ {param_str:<41s} │")
    print("└───────────────┴──────────┴─────────────────────────────────────────┘")
