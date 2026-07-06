"""
evaluation.py
=============
Geometric error computation and uncertainty quality metrics for FLAME-based
3D face reconstruction.

This module answers two separate but deeply connected questions:

  Q1 (Geometric quality): How accurately does a regressor reconstruct the
     ground-truth FLAME mesh, vertex by vertex?

  Q2 (Uncertainty quality): Does the estimated uncertainty actually *predict*
     where the geometric error is large?  A method that says "I am uncertain
     here" but turns out to be wrong there is useless in practice.

The functions below are split into four groups:

  Group A — Geometric error functions
       compute distances between predicted and ground-truth meshes.

  Group B — Uncertainty-error correlation
       ask whether uncertainty predicts error (Q2 above).

  Group C — Uncertainty calibration and ranking quality
       answer the deeper question: are the uncertainty *magnitudes* correct,
       not just their relative ordering?

  Group D — Aggregation helpers
       combine everything into a single summary dict for reporting.

FLAME vertex count: nv = 5023 (fixed topology, same for all four methods).

All distance metrics are reported in millimetres (mm) to match the NoW
benchmark convention used in the SHeaP paper.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

# np.trapezoid was added in NumPy 2.0 (np.trapz is deprecated there but still
# works); older NumPy (<2.0, e.g. some local/CPU dev environments) only has
# np.trapz. Use whichever is available so this module works on both.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz
from scipy import stats
from scipy.spatial import KDTree
from scipy.stats import norm as _norm
from scipy.stats import chi as _chi


# ---------------------------------------------------------------------------
# Constants: FLAME facial region masks
# ---------------------------------------------------------------------------
# These index sets partition the 5023 FLAME vertices into semantically
# meaningful facial regions.  They are used by Group A and Group C functions
# that compute region-wise breakdowns.
#
# NOTE: the actual index arrays must be loaded from the FLAME model package
# (e.g. from flame_masks.pkl provided with the official FLAME release).
# The names here are symbolic; replace with loaded numpy arrays before use.

REGION_NAMES = ["forehead", "nose", "left_eye", "right_eye",
                "left_cheek", "right_cheek", "mouth", "chin", "neck"]
# REGION_MASKS : Dict[str, np.ndarray]  (populated at import time from FLAME)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _align_meshes(source: np.ndarray, target: np.ndarray,
                  method: str = "procrustes") -> np.ndarray:
    """
    Align `source` vertices onto `target` vertices.  Both have shape (N, 3).

    "procrustes" : scale + rotation + translation  (non-metrical, NoW convention)
                   minimises ||s · R · source_centred + t − target||_F
    "rigid"      : rotation + translation only  (metrical evaluation)
                   same as procrustes but scale is fixed at 1.

    The rotation is found via SVD with a determinant correction to guarantee
    a proper rotation (det = +1) and avoid reflections.
    """
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_c = source - src_mean
    tgt_c = target - tgt_mean

    src_norm = np.sqrt((src_c ** 2).sum())
    tgt_norm = np.sqrt((tgt_c ** 2).sum())
    if src_norm < 1e-10 or tgt_norm < 1e-10:
        return source.copy()

    src_n = src_c / src_norm
    tgt_n = tgt_c / tgt_norm

    # Optimal rotation: H = src_n.T @ tgt_n,  SVD(H) = U S Vt
    H = src_n.T @ tgt_n                             # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)                   # +1 or -1
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T         # proper rotation (3, 3)

    if method == "procrustes":
        scale = tgt_norm / src_norm
        aligned = scale * (src_c @ R.T) + tgt_mean
    elif method == "rigid":
        aligned = src_c @ R.T + tgt_mean
    else:
        raise ValueError(
            f"Unknown alignment_method '{method}'. Choose 'procrustes' or 'rigid'."
        )
    return aligned.astype(source.dtype)


def _scale_translate_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Align `source` to `target` using only scale and translation.

    Used for scan-to-mesh distance where `source` is a FLAME mesh (5023 × 3)
    and `target` is an unstructured GT point cloud (N × 3) — different sizes
    make Procrustes ill-defined, so we match centroid and RMS radius instead.
    This mirrors the scale-invariant evaluation convention used in NoW /
    NoW: the regressor handles orientation explicitly via camera params,
    so rotation ambiguity is not an issue.
    """
    src_c = source.mean(axis=0)
    tgt_c = target.mean(axis=0)

    src_rms = np.sqrt(np.mean(np.sum((source - src_c) ** 2, axis=-1)))
    tgt_rms = np.sqrt(np.mean(np.sum((target - tgt_c) ** 2, axis=-1)))

    if src_rms < 1e-10:
        return source.copy()

    scale = tgt_rms / src_rms
    return (scale * (source - src_c) + tgt_c).astype(source.dtype)


def _sample_points_on_mesh(vertices: np.ndarray, faces: np.ndarray,
                            n_points: int) -> np.ndarray:
    """
    Sample `n_points` 3-D points uniformly on the triangle mesh surface.

    Algorithm
    ---------
    1. Compute the area of every triangle via the cross-product formula.
    2. Randomly select `n_points` triangle indices, weighted by area.
    3. For each selected triangle draw a uniformly random point using the
       (sqrt(r1), r2) barycentric parameterisation that avoids the clustering
       artefact of the naive (r1, r2) scheme.

    Returns
    -------
    points : (n_points, 3) float32
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)  # (F,)
    total_area = areas.sum()

    if total_area < 1e-10:
        # Degenerate mesh: fall back to vertex sampling
        idx = np.random.default_rng().integers(0, len(vertices), n_points)
        return vertices[idx].astype(np.float32)

    rng = np.random.default_rng()
    face_idx = rng.choice(len(faces), size=n_points, p=areas / total_area)

    # Uniform barycentric coordinates via the square-root trick
    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    s = np.sqrt(r1)
    lam0 = 1.0 - s
    lam1 = s * (1.0 - r2)
    lam2 = s * r2

    sel = faces[face_idx]    # (n_points, 3)  — vertex indices per sampled point
    points = (lam0[:, None] * vertices[sel[:, 0]] +
              lam1[:, None] * vertices[sel[:, 1]] +
              lam2[:, None] * vertices[sel[:, 2]])
    return points.astype(np.float32)


# ===========================================================================
# GROUP A — Geometric Error Functions
# ===========================================================================

def calculate_geometric_error(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    align: bool = True,
    alignment_method: str = "procrustes",
) -> np.ndarray:
    """
    Compute the per-vertex Euclidean (L2) distance between a predicted mesh
    and the ground-truth FLAME mesh.

    This is the most fundamental geometric metric: it tells us, for each of
    the 5023 vertices, exactly how far off the prediction is in 3-D space.
    It is the foundation for every downstream metric in this module.

    Alignment note
    --------------
    Before computing distances, the predicted mesh is rigidly aligned to the
    ground truth.  Without this step, global errors in camera/pose estimation
    dominate the vertex errors and obscure expression/shape quality.  Two
    alignment strategies are supported:

      "procrustes" (default) : orthogonal Procrustes — finds the rotation R,
           translation t, and optionally scale s that minimise
           ||s·R·P + t − G||_F.  When scale is fitted this is *non-metrical*
           evaluation (as in the NoW benchmark); when s=1 it is metrical.
      "rigid"  : fit R and t only (no scale).  Use this for metrical eval
           when the regressor has been given camera intrinsics.

    Parameters
    ----------
    pred_vertices : np.ndarray, shape (5023, 3)
        Predicted FLAME vertices in mm (or normalised units).
    gt_vertices : np.ndarray, shape (5023, 3)
        Ground-truth FLAME vertices.
    align : bool, default True
        Whether to Procrustes-align `pred_vertices` to `gt_vertices` before
        computing distances.  Set to False only if meshes are already in the
        same canonical space.
    alignment_method : {"procrustes", "rigid"}

    Returns
    -------
    error : np.ndarray, shape (5023,), dtype float32
        Per-vertex L2 distance in mm.  Index i corresponds to FLAME vertex i.
    """
    pred = np.asarray(pred_vertices, dtype=np.float64)
    gt   = np.asarray(gt_vertices,   dtype=np.float64)

    if align:
        pred = _align_meshes(pred, gt, method=alignment_method)

    # Per-vertex Euclidean distance — FLAME vertices are in metres (±0.15 m);
    # multiply by 1000 to return mm as documented (NoW/NeRSemble convention).
    error = np.linalg.norm(pred - gt, axis=-1)   # (5023,) in metres
    return (error * 1000).astype(np.float32)     # → mm


def calculate_scan_to_mesh_distance(
    pred_vertices: np.ndarray,
    pred_faces: np.ndarray,
    gt_point_cloud: np.ndarray,
    n_sample_points: int = 10_000,
    align: bool = True,
) -> Dict[str, float]:
    """
    Compute the *scan-to-mesh* distance used by the NoW benchmark.

    Why scan-to-mesh instead of vertex-to-vertex?
    -----------------------------------------------
    Ground-truth data in NoW and TEMPEH are *unstructured* point clouds —
    they have no predefined vertex topology matching FLAME's 5023 vertices.
    A vertex-to-vertex comparison would require explicit correspondence, which
    is non-trivial to establish accurately.

    Instead, for each ground-truth point g_j we find its nearest point on the
    *surface* of the predicted mesh (not just the nearest vertex), giving a
    sub-vertex-resolution distance.  This metric is therefore:
      • topology-free on the GT side (works with any dense point cloud)
      • sensitive to fine-grained surface shape, not just coarse vertex placement
      • directly comparable with published NoW scores

    Algorithm
    ---------
    1. Sample `n_sample_points` random points uniformly on the predicted mesh
       surface using barycentric interpolation on randomly selected triangles.
    2. For each GT point cloud point g_j, compute the distance to its nearest
       sampled surface point:  d_j = min_k ||g_j − s_k||₂.
    3. Aggregate:  median(d), mean(d), std(d), 90th-percentile(d).

    Parameters
    ----------
    pred_vertices : np.ndarray, shape (5023, 3)
    pred_faces : np.ndarray, shape (F, 3), dtype int
        Triangle face indices into pred_vertices.
    gt_point_cloud : np.ndarray, shape (N, 3)
        Ground-truth 3D points (from COLMAP / structured light scan).
    n_sample_points : int, default 10_000
        More points → more accurate distance but slower.  10k is the NoW
        default and gives stable estimates.
    align : bool, default True
        Procrustes-align pred_vertices to gt_point_cloud centroid/scale before
        computing distances (non-metrical evaluation).

    Returns
    -------
    metrics : dict with keys
        "median_mm"   : float  — primary NoW / SHeaP metric
        "mean_mm"     : float  — secondary metric
        "std_mm"      : float  — spread across GT points
        "p90_mm"      : float  — 90th percentile; sensitive to bad outlier regions
        "per_point_d" : np.ndarray, shape (N,) — raw distances, for histogramming
    """
    pred_v = np.asarray(pred_vertices,  dtype=np.float64)
    gt_pts = np.asarray(gt_point_cloud, dtype=np.float64)
    faces  = np.asarray(pred_faces,     dtype=int)

    if align:
        # Scale + translation only: GT and pred can have very different N values
        # so full Procrustes rotation is ill-defined; matching centroid + RMS scale
        # removes the metric ambiguity inherent in monocular reconstruction.
        pred_v = _scale_translate_align(pred_v, gt_pts)

    # Sample dense set of surface points on the aligned predicted mesh
    sampled = _sample_points_on_mesh(
        pred_v.astype(np.float32), faces, n_sample_points
    ).astype(np.float64)                             # (n_sample_points, 3)

    # Nearest-neighbour distance from each GT point to the sampled surface
    # KDTree gives O((N + n_sample_points) log n_sample_points) runtime
    tree = KDTree(sampled)
    per_point_d, _ = tree.query(gt_pts, k=1)         # (N,)

    return {
        "median_mm":   float(np.median(per_point_d)),
        "mean_mm":     float(np.mean(per_point_d)),
        "std_mm":      float(np.std(per_point_d)),
        "p90_mm":      float(np.percentile(per_point_d, 90)),
        "per_point_d": per_point_d.astype(np.float32),
    }


def calculate_region_wise_geometric_error(
    error: np.ndarray,
    region_masks: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Break down the per-vertex error array by facial region to reveal *where*
    reconstruction fails.

    Motivation
    ----------
    A single scalar (mean or median error) over all 5023 vertices hides
    important spatial structure.  The mouth, eyes, and nose tip are perceptually
    the most salient regions, yet they are often the hardest to reconstruct
    because they are articulated (jaw, eyelids) and small relative to the full
    head.  Region-wise metrics let us say, for instance, "SMIRK has lower mouth
    error than DECA because of the cycle-augmentation path" — which is a much
    richer claim than a single benchmark number.

    Parameters
    ----------
    error : np.ndarray, shape (5023,)
        Per-vertex L2 error from `calculate_geometric_error`.
    region_masks : dict[str → np.ndarray of int]
        Maps each region name to the array of FLAME vertex indices belonging
        to it (loaded from FLAME model masks file).

    Returns
    -------
    region_metrics : dict[str → dict]
        For each region name, returns:
          "mean_mm", "median_mm", "max_mm", "std_mm"
        e.g. region_metrics["mouth"]["mean_mm"] = 2.4
    """
    error = np.asarray(error, dtype=np.float64).ravel()
    region_metrics: Dict[str, Dict[str, float]] = {}

    for region_name, vertex_idx in region_masks.items():
        region_error = error[np.asarray(vertex_idx)]
        if len(region_error) == 0:
            region_metrics[region_name] = {
                "mean_mm": 0.0, "median_mm": 0.0, "max_mm": 0.0, "std_mm": 0.0,
            }
            continue
        region_metrics[region_name] = {
            "mean_mm":   float(np.mean(region_error)),
            "median_mm": float(np.median(region_error)),
            "max_mm":    float(np.max(region_error)),
            "std_mm":    float(np.std(region_error)),
        }

    return region_metrics


def calculate_vertex_rmse(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    align: bool = True,
) -> float:
    """
    Root Mean Squared Error over all vertices (scalar summary).

    RMSE penalises large outlier errors more than the median, making it a
    complementary metric: a method might have low median error but one badly
    reconstructed region (e.g. a wide-open jaw) will push RMSE up.  Reporting
    both median and RMSE together gives a more complete picture of the error
    distribution than either alone.

    Parameters
    ----------
    pred_vertices : np.ndarray, shape (5023, 3)
    gt_vertices   : np.ndarray, shape (5023, 3)
    align : bool, default True

    Returns
    -------
    rmse : float  (in mm if inputs are in mm)
    """
    # Per-vertex L2 errors, shape (5023,)
    error = calculate_geometric_error(pred_vertices, gt_vertices, align=align)
    # RMSE = sqrt( mean( ||v_pred_i − v_gt_i||² ) )
    # Note: error[i] = ||...||, so error[i]² = squared distance; mean then sqrt.
    return float(np.sqrt(np.mean(error ** 2)))


# ===========================================================================
# GROUP B — Uncertainty-Error Correlation
# ===========================================================================

def correlate_error_and_uncertainty(
    error: np.ndarray,
    uncertainty: np.ndarray,
    method: str = "both",
) -> Dict[str, float]:
    """
    Compute statistical correlation between the per-vertex geometric error and
    the per-vertex uncertainty heatmap.

    This is the central evaluation question: does the uncertainty estimator
    *know* where it is wrong?  A well-calibrated estimator should assign high
    uncertainty to exactly those vertices where the geometric error is large.

    Two correlation coefficients are computed because they measure different
    things:

      Pearson r  : measures *linear* association between error and uncertainty.
                   Good when both are roughly normally distributed.  Sensitive
                   to outlier vertices.

      Spearman ρ : rank-based, measures *monotonic* association.  More robust
                   to the heavy-tailed distributions typical of vertex errors
                   (a few badly predicted vertices dominate).  This is the
                   recommended primary metric.

    Both are bounded in [−1, +1].  We want values close to +1: high uncertainty
    should co-occur with high error.  Values near 0 mean the uncertainty is
    uncorrelated with where the errors actually are — i.e. useless.  Negative
    values would indicate a systematically misaligned estimator.

    Parameters
    ----------
    error : np.ndarray, shape (5023,) or (N_images × 5023,)
        Per-vertex geometric error (L2 in mm).
    uncertainty : np.ndarray, same shape as error
        Per-vertex uncertainty scalar from any of the methods in uncertainty.py.
    method : {"pearson", "spearman", "both"}
        Which coefficient(s) to return.

    Returns
    -------
    results : dict with (subset of) keys
        "pearson_r"      : float  — linear correlation coefficient
        "pearson_p"      : float  — two-tailed p-value for H0: r=0
        "spearman_rho"   : float  — rank correlation coefficient
        "spearman_p"     : float  — two-tailed p-value for H0: ρ=0
        "kendall_tau"    : float  — included when method=="both" as a tie-robust
                                    alternative to Spearman
    """
    e = np.asarray(error,       dtype=np.float64).ravel()
    u = np.asarray(uncertainty, dtype=np.float64).ravel()

    if len(e) != len(u):
        raise ValueError(
            f"error and uncertainty must have the same number of elements "
            f"(got {len(e)} and {len(u)})."
        )

    result: Dict[str, float] = {}

    if method in ("pearson", "both"):
        r, p = stats.pearsonr(e, u)
        result["pearson_r"] = float(r)
        result["pearson_p"] = float(p)

    if method in ("spearman", "both"):
        rho, p = stats.spearmanr(e, u)
        result["spearman_rho"] = float(rho)
        result["spearman_p"]   = float(p)

    if method == "both":
        tau, p = stats.kendalltau(e, u)
        result["kendall_tau"] = float(tau)

    return result


# ===========================================================================
# GROUP C — Uncertainty Calibration and Ranking Quality
# ===========================================================================

def calculate_sparsification_error_curve(
    error: np.ndarray,
    uncertainty: np.ndarray,
    n_bins: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute the Sparsification Error Curve and the Area Under it (AUSE),
    the gold-standard metric for uncertainty ranking quality in regression.

    What is sparsification?
    -----------------------
    Suppose we have N predictions with errors e_1,...,e_N and uncertainty
    estimates u_1,...,u_N.  If u is a *perfect oracle* for error, then
    removing the fraction f of predictions with the highest uncertainty should
    remove exactly the fraction f with the highest error.

    The sparsification curve plots how the *mean error of the remaining
    predictions* decreases as we progressively remove the most-uncertain
    fraction.  There are three curves:

      1. Method curve  : remove by *predicted* uncertainty (what we have).
      2. Oracle curve  : remove by *true* error (the ideal we cannot achieve
                         at deployment time but use as upper bound).
      3. Random curve  : remove randomly (lower bound; uninformative estimator).

    A good uncertainty estimator produces a curve close to the oracle.

    Area Under the Sparsification Error (AUSE)
    ------------------------------------------
    AUSE is the integral of (Method curve − Oracle curve).  Lower is better
    (0 = perfect oracle).  It summarises how close the uncertainty ranking is
    to the ideal across all sparsification levels, without cherry-picking a
    single threshold.

    AUSE is the primary metric from the uncertainty-quality literature (see e.g.
    Ilg et al., "Uncertainty Estimates and Multi-Hypotheses Networks for Optical
    Flow", ECCV 2018).

    Parameters
    ----------
    error : np.ndarray, shape (N,)
        Per-sample (or per-vertex, flattened) absolute errors.
    uncertainty : np.ndarray, shape (N,)
        Corresponding uncertainty estimates.
    n_bins : int, default 20
        Number of equally-spaced sparsification fractions in [0, 1).

    Returns
    -------
    curves : dict with keys
        "fractions"      : np.ndarray, shape (n_bins,) — removal fractions
        "method_errors"  : np.ndarray, shape (n_bins,) — method curve values
        "oracle_errors"  : np.ndarray, shape (n_bins,) — oracle curve values
        "random_errors"  : np.ndarray, shape (n_bins,) — random curve values
        "ause"           : float — Area Under Sparsification Error (lower=better)
        "ause_normalised": float — AUSE / AUSE_random, useful for cross-method
                                    comparison when error magnitudes differ
    """
    e = np.asarray(error,       dtype=np.float64).ravel()
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    N = len(e)

    # Fractions from 0% to (n_bins-1)/n_bins removed
    fractions = np.linspace(0.0, 1.0, n_bins, endpoint=False)

    # Pre-sort: descending uncertainty → remove high-uncertainty first
    unc_desc  = np.argsort(u)[::-1]   # index 0 = most uncertain
    # Pre-sort: descending error → oracle removes hardest first
    err_desc  = np.argsort(e)[::-1]

    overall_mean = float(np.mean(e))  # random curve is flat at this value

    method_errors = np.empty(n_bins)
    oracle_errors = np.empty(n_bins)
    random_errors = np.full(n_bins, overall_mean)

    for i, f in enumerate(fractions):
        n_remove = int(np.round(f * N))
        n_keep   = N - n_remove

        if n_keep <= 0:
            # Degenerate case at the very end: define error as 0
            method_errors[i] = 0.0
            oracle_errors[i] = 0.0
            random_errors[i] = 0.0
            continue

        # Method: keep the n_keep LEAST uncertain predictions
        keep_method = unc_desc[n_remove:]        # tail of descending-unc sort
        method_errors[i] = float(np.mean(e[keep_method]))

        # Oracle: keep the n_keep LOWEST-error predictions
        keep_oracle = err_desc[n_remove:]        # tail of descending-err sort
        oracle_errors[i] = float(np.mean(e[keep_oracle]))

    # AUSE = ∫ (method − oracle) df  via trapezoidal rule
    # Lower is better; 0 = method ranking identical to oracle ranking.
    ause = float(_trapezoid(method_errors - oracle_errors, fractions))

    # Normalised AUSE ∈ [0, 1]:  0 = perfect, 1 = no better than random
    ause_random = float(_trapezoid(random_errors - oracle_errors, fractions))
    ause_normalised = ause / max(ause_random, 1e-10)

    return {
        "fractions":       fractions,
        "method_errors":   method_errors,
        "oracle_errors":   oracle_errors,
        "random_errors":   random_errors,
        "ause":            ause,
        "ause_normalised": ause_normalised,
    }


def calculate_uncertainty_calibration(
    error: np.ndarray,
    uncertainty_std: np.ndarray,
    confidence_levels: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Assess whether the predicted uncertainty magnitudes are *calibrated* — i.e.
    whether a predicted σ corresponds to an actual standard deviation of the
    error distribution.

    Why calibration matters beyond correlation
    ------------------------------------------
    Good Spearman ρ means the uncertainty *ranks* errors correctly.  But a
    model could rank perfectly and yet predict σ=0.001mm everywhere — a useless
    signal because the absolute magnitudes are off by orders of magnitude.
    Calibration asks the harder question: if I predict σ=2mm, does the actual
    error really lie within ±2mm with ~68% probability?

    Calibration test (regression coverage)
    ----------------------------------------
    Under the assumption that the error at each vertex is approximately
    Gaussian with mean 0 and std σ_hat (the predicted uncertainty), we check
    empirical coverage:

        coverage(α) = fraction of vertices where |error_i| ≤ z_α · σ_hat_i

    where z_α = Φ⁻¹((1+α)/2) is the two-tailed normal quantile.
    For α=0.90: z_0.90 = Φ⁻¹(0.95) ≈ 1.645.
    For α=0.68: z_0.68 = Φ⁻¹(0.84) ≈ 1.00  (the familiar ±1σ rule).

    A perfectly calibrated model satisfies coverage(α) = α for all α.

    Expected Calibration Error (ECE): the average |coverage(α) − α| over a
    grid of α values.  Lower ECE = better calibrated.

    Sharpness: mean(σ_hat²).  A perfectly calibrated but high-sharpness model
    predicts narrow (confident) intervals and is therefore more *useful*.
    Two models can have identical ECE; the sharper one is preferred.

    Parameters
    ----------
    error : np.ndarray, shape (N,)
        Absolute per-vertex / per-image geometric errors (mm).
    uncertainty_std : np.ndarray, shape (N,)
        Predicted standard deviation (not variance) at each sample.
    confidence_levels : list[float] or None
        Confidence levels α to evaluate coverage at.
        Default: [0.5, 0.68, 0.80, 0.90, 0.95, 0.99].

    Returns
    -------
    calibration : dict with keys
        "ece"                    : float  — Expected Calibration Error (lower=better)
        "coverage_at_levels"     : dict[float → float]  — empirical coverage per α
        "sharpness"              : float  — mean calibrated variance (lower=sharper)
        "reliability_diagram_x"  : np.ndarray — expected coverage values (the α grid)
        "reliability_diagram_y"  : np.ndarray — empirical coverage values
        "overconfidence_fraction": float  — fraction of α levels where empirical
                                   coverage < expected (model σ too small → overconfident)
        "underconfidence_fraction": float — fraction where empirical > expected
                                    (model σ too large → underconfident / conservative)
        "temperature"            : float  — scaling factor T = RMS(e) / RMS(σ)
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.68, 0.80, 0.90, 0.95, 0.99]

    e    = np.asarray(error,           dtype=np.float64).ravel()
    sig  = np.asarray(uncertainty_std, dtype=np.float64).ravel()

    # Temperature scaling: find T such that sig_cal = T·σ has the same RMS as
    # the error.  This makes the coverage check |e| ≤ z_α·σ_cal dimensionally
    # consistent without distorting rank order (T is a monotonic rescaling).
    # Prior approach (max-normalize e + rank-normalize σ) gave ECE ≡ 0.196 for
    # ALL methods: rank-normalized σ ∈ [1/N,1] dwarfed max-normalized errors for
    # skewed error distributions → every vertex trivially covered at every α.
    rms_e   = float(np.sqrt(np.mean(e ** 2)))
    rms_sig = float(np.sqrt(np.mean(sig ** 2)))
    T = rms_e / rms_sig if rms_sig > 1e-10 else 1.0
    sig_cal = sig * T                                   # same scale as error

    coverage_at_levels: Dict[float, float] = {}
    for alpha in confidence_levels:
        # Per-vertex error e = ||pred - gt||_2 is the L2 norm of a 3-D
        # displacement vector.  Under isotropic Gaussian noise with per-coord
        # std σ, e/σ ~ chi(3).  After temperature scaling, sig_cal has the
        # same RMS as e, so sig_cal ≈ sqrt(3)*σ for a well-calibrated model.
        # The correct alpha-quantile threshold in units of sig_cal is therefore
        # chi(3).ppf(alpha) / sqrt(3).  Using the Gaussian ppf((1+α)/2) gives
        # the wrong coverage (underestimates at low α, overestimates at high α).
        z_alpha = float(_chi.ppf(alpha, df=3) / np.sqrt(3))
        covered = float(np.mean(e <= z_alpha * sig_cal))  # e >= 0 always
        coverage_at_levels[alpha] = covered

    # ECE: mean absolute gap between predicted and empirical coverage
    ece = float(np.mean([
        abs(coverage_at_levels[a] - a) for a in confidence_levels
    ]))

    # Sharpness in calibrated units (lower = tighter / more useful intervals)
    sharpness = float(np.mean(sig_cal ** 2))

    # Reliability diagram arrays
    expected_coverage = np.array(confidence_levels)
    empirical_coverage = np.array([coverage_at_levels[a] for a in confidence_levels])

    # Standard naming: OVERCONFIDENT = intervals too narrow = empirical < expected
    # (σ_cal too small); UNDERCONFIDENT = intervals too wide = empirical > expected.
    overconfidence_fraction = float(
        np.mean(empirical_coverage < expected_coverage)
    )
    underconfidence_fraction = float(
        np.mean(empirical_coverage > expected_coverage)
    )

    return {
        "ece":                     ece,
        "coverage_at_levels":      coverage_at_levels,
        "sharpness":               sharpness,
        "reliability_diagram_x":   expected_coverage,
        "reliability_diagram_y":   empirical_coverage,
        "overconfidence_fraction":  overconfidence_fraction,
        "underconfidence_fraction": underconfidence_fraction,
        "temperature":             T,
    }


def calculate_nll(
    error: np.ndarray,
    uncertainty_std: np.ndarray,
) -> float:
    """
    Compute the Gaussian Negative Log-Likelihood (NLL) of the observed errors
    under the predicted uncertainty model.

    This is the probabilistic evaluation metric: it asks "how probable were
    the actual errors under the predictive distribution we put forward?"
    Under the assumption that error_i ~ N(0, σ_hat_i²):

        NLL = (1/N) Σ_i  [ log(σ_hat_i) + e_i² / (2 σ_hat_i²) ]

    Jointly penalises two failure modes:
      • Overconfident predictions (small σ_hat, large e): large e_i²/σ_hat_i².
      • Underconfident predictions (large σ_hat, small e): penalised by log(σ_hat).

    Sigma is temperature-scaled to have the same RMS as error before computing
    NLL.  This preserves rank ordering (ρ, AUSE unchanged) while making the
    formula dimensionally consistent and the output comparable across methods.

    Parameters
    ----------
    error : np.ndarray, shape (N,)
        Absolute per-vertex errors in mm.
    uncertainty_std : np.ndarray, shape (N,)
        Predicted standard deviation at each vertex.

    Returns
    -------
    nll : float
        Lower is better.  A useful baseline is the NLL achieved by setting
        σ_hat = std(error) everywhere (the non-adaptive constant predictor).
    """
    e   = np.asarray(error,           dtype=np.float64).ravel()
    sig = np.asarray(uncertainty_std, dtype=np.float64).ravel()

    # Temperature scaling: rescale sigma so both e and σ_cal share the same RMS.
    # T is the single optimal scalar under a homoscedastic Gaussian; it preserves
    # rank ordering (ρ and AUSE unchanged) while making the NLL interpretable.
    # Prior approach (max-normalize e + rank-normalize σ): inflated NLL for MCD
    # (238–432) vs TTA (47–54) because rank-normalized σ assigned tiny values to
    # some vertices that happened to have large errors, making (e/σ)² explode.
    rms_e   = float(np.sqrt(np.mean(e ** 2)))
    rms_sig = float(np.sqrt(np.mean(sig ** 2)))
    T = rms_e / rms_sig if rms_sig > 1e-10 else 1.0
    # Floor: the 1st percentile of calibrated sigmas, with an absolute minimum
    # of 1e-3 * rms_e (≈ 5 µm for typical face meshes).  This prevents the
    # (e / sig_cal)^2 term from exploding to ~1e20 when a method assigns
    # near-zero uncertainty to some vertices (e.g. StaticRegion near-zero
    # vertices), which was producing NLL values of ~1e15.
    floor = max(float(np.percentile(sig * T, 1)), 1e-3 * rms_e)
    sig_cal = np.maximum(sig * T, floor)

    # Gaussian NLL (constant 0.5·log(2π) omitted; cancels in method comparisons)
    return float(np.mean(np.log(sig_cal) + 0.5 * (e / sig_cal) ** 2))


def calculate_uncertainty_sharpness(
    uncertainty: np.ndarray,
) -> Dict[str, float]:
    """
    Compute sharpness (resolution / spread) statistics of an uncertainty map.

    Sharpness is a property of the *predicted* uncertainty alone, independent
    of the ground-truth error.  Together with ECE it forms a complete picture:
      • Well-calibrated + sharp  → the best outcome: confident and right.
      • Well-calibrated + diffuse → acceptable: correct but unhelpfully vague.
      • Poorly calibrated + sharp → dangerous: confidently wrong.

    Sharpness metrics
    -----------------
    - Mean uncertainty  : overall level of predicted uncertainty.
    - Std of uncertainty : spread — does the model differentiate between
                            hard and easy vertices?
    - Spatial entropy   : if we normalise uncertainty to a probability simplex,
                           how uniformly is uncertainty spread across vertices?
                           High entropy = uncertainty is uniform (useless).
                           Low entropy = uncertainty is concentrated on a small
                           region (potentially informative).

    Parameters
    ----------
    uncertainty : np.ndarray, shape (5023,) or (5023, 1)
        Per-vertex uncertainty scalar.

    Returns
    -------
    sharpness : dict with keys
        "mean"             : float
        "std"              : float
        "median"           : float
        "p10", "p90"       : float  — 10th / 90th percentile
        "spatial_entropy"  : float  — normalised entropy ∈ [0, 1]
        "coefficient_var"  : float  — std/mean, scale-free sharpness measure
    """
    u = np.asarray(uncertainty, dtype=np.float64).ravel()

    mean   = float(np.mean(u))
    std_u  = float(np.std(u))
    median = float(np.median(u))
    p10    = float(np.percentile(u, 10))
    p90    = float(np.percentile(u, 90))

    # Coefficient of variation: scale-free measure of how spread the uncertainty is
    coefficient_var = float(std_u / mean) if mean > 1e-10 else 0.0

    # Spatial entropy: treat the normalised uncertainty map as a probability
    # distribution over vertices and compute its entropy, normalised by the
    # maximum possible entropy log(N) so the result lies in [0, 1].
    # High entropy ≈ 1: uncertainty is uniform across all vertices (uninformative).
    # Low entropy ≈ 0: uncertainty is concentrated on a few vertices (informative).
    u_sum = u.sum()
    if u_sum > 1e-10:
        p = u / u_sum                                    # probability simplex
        # Shannon entropy in nats, with a small epsilon inside log for stability
        entropy = -float(np.sum(p * np.log(np.maximum(p, 1e-15))))
        max_entropy = np.log(len(u))
        spatial_entropy = float(entropy / max_entropy) if max_entropy > 1e-10 else 0.0
    else:
        spatial_entropy = 1.0   # all-zero uncertainty → perfectly uniform (degenerate)

    return {
        "mean":            mean,
        "std":             std_u,
        "median":          median,
        "p10":             p10,
        "p90":             p90,
        "spatial_entropy": spatial_entropy,
        "coefficient_var": coefficient_var,
    }


# ===========================================================================
# GROUP D — Aggregation / Summary
# ===========================================================================

def compute_static_region_baseline(error_list: List[np.ndarray]) -> np.ndarray:
    """
    Build the 'StaticRegion' uncertainty baseline from a list of per-vertex
    error arrays.

    Motivation
    ----------
    Some FLAME vertices (eye corners, lip edges, chin, nostrils) are
    systematically harder to reconstruct regardless of the input image.
    A trivially correct uncertainty estimator can simply memorise "these
    vertices are always hard" without ever looking at the test image and still
    achieve non-trivial Spearman ρ.

    By including the mean per-vertex error across all evaluation images as a
    synthetic eighth "method" in compare_uncertainty_methods, we expose whether
    any live method (TTA, MCD, …) is doing better than this non-adaptive
    baseline.  If a method cannot beat StaticRegion it is only recovering
    dataset-level difficulty, not per-image uncertainty.

    Usage in main pipeline
    ----------------------
    ::
        static_baseline = compute_static_region_baseline(per_image_error_list)
        uncertainty_dict['StaticRegion'] = static_baseline
        summary = compare_uncertainty_methods(mean_error, uncertainty_dict)

    Parameters
    ----------
    error_list : list of np.ndarray, each shape (5023,) or (5023, 1)
        Per-vertex geometric errors for individual evaluation images.

    Returns
    -------
    baseline : np.ndarray, shape (5023,), float32
        Mean per-vertex error across all images — used as a static uncertainty
        map for cross-method baseline comparison.
    """
    if not error_list:
        return np.zeros(5023, dtype=np.float32)
    stacked = np.stack(
        [np.asarray(e, dtype=np.float32).ravel() for e in error_list], axis=0
    )
    return stacked.mean(axis=0)                  # (5023,)


def calculate_rank_stability(
    per_image_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
    """
    Compute Kendall's coefficient of concordance W for uncertainty rank
    stability across evaluation images.

    W ∈ [0, 1] measures how consistently a single uncertainty method ranks
    FLAME vertices across different evaluation images.

      W ≈ 1 — the method assigns the same relative rank to each vertex on every
              image: the ranking is stable and reproducible.
      W ≈ 0 — vertex rankings differ arbitrarily between images: the method is
              noisy and image-specific, making per-image predictions unreliable.

    A method with high Spearman ρ (good average correlation with error) but low
    W cannot be trusted on individual images.  A method with both high ρ and
    high W is reliably predictive at the per-image level.

    Formula (Kendall 1948)
    ----------------------
    Given m images and n vertices, let r_{i,j} = rank of vertex j on image i.
    Rank sum for vertex j:     R_j = Σ_i r_{i,j}
    Mean rank sum:             R̄  = m(n+1)/2
    Sum of squared deviations: S   = Σ_j (R_j − R̄)²
    W = 12·S / [m²·(n³ − n)]        ∈ [0, 1]

    Parameters
    ----------
    per_image_pairs : list of (error_array, uncertainty_array) tuples.
        Each tuple contains per-vertex arrays of shape (5023,), one per image.

    Returns
    -------
    w : float
        Kendall's W in [0, 1].  Returns NaN when fewer than 2 images are
        provided (W is undefined for a single image).
    """
    if len(per_image_pairs) < 2:
        return float('nan')

    from scipy.stats import rankdata as _rankdata

    # Build rank matrix: rows = images, cols = vertices (rank over vertices)
    rank_matrix = np.stack(
        [_rankdata(u.ravel(), method='average') for _, u in per_image_pairs],
        axis=0,
    )                                    # (m, n)

    m, n = rank_matrix.shape
    R     = rank_matrix.sum(axis=0)      # rank sum per vertex, shape (n,)
    R_bar = m * (n + 1) / 2.0           # expected rank sum under uniform agreement
    S     = float(np.sum((R - R_bar) ** 2))
    denom = float(m ** 2) * float(n ** 3 - n)
    if denom < 1e-10:
        return float('nan')
    return float(12.0 * S / denom)


def compare_uncertainty_methods(
    error: np.ndarray,
    uncertainty_dict: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Run the full evaluation suite for each uncertainty method and return a
    side-by-side comparison table.

    This is the top-level function for the "does uncertainty predict error?"
    question.  It calls `correlate_error_and_uncertainty`,
    `calculate_sparsification_error_curve`, `calculate_uncertainty_calibration`,
    and `calculate_nll` for every method in `uncertainty_dict` and packages
    the results into a single nested dict suitable for display as a table.

    Parameters
    ----------
    error : np.ndarray, shape (N,)
        Flattened per-vertex errors over all evaluation images.
    uncertainty_dict : dict[str → np.ndarray of shape (N,)]
        Keys are method names (e.g. "TTA", "MCD", "CrossMethod",
        "DeepEnsemble", "Heteroscedastic", "ConcreteMCD").
        Values are the corresponding uncertainty estimates.

    Returns
    -------
    summary : dict[method_name → dict of metrics]
        Each inner dict contains:
          "spearman_rho", "pearson_r", "ause", "ause_normalised",
          "ece", "nll", "sharpness_mean", "sharpness_entropy"
        Ready to be converted to a pandas DataFrame for the report.
    """
    e_flat = np.asarray(error, dtype=np.float64).ravel()
    summary: Dict[str, Dict[str, float]] = {}

    for method_name, uncertainty in uncertainty_dict.items():
        u_flat = np.asarray(uncertainty, dtype=np.float64).ravel()

        # Group B: correlation
        corr = correlate_error_and_uncertainty(e_flat, u_flat, method="both")

        # Group C: sparsification → AUSE
        sparse = calculate_sparsification_error_curve(e_flat, u_flat)

        # Group C: probabilistic score
        nll = calculate_nll(e_flat, u_flat)

        # Group C: calibration (ECE)
        calib = calculate_uncertainty_calibration(e_flat, u_flat)

        # Group C: sharpness
        sharp = calculate_uncertainty_sharpness(u_flat)

        summary[method_name] = {
            "spearman_rho":      corr.get("spearman_rho",  float("nan")),
            "pearson_r":         corr.get("pearson_r",     float("nan")),
            "ause":              sparse["ause"],
            "ause_normalised":   sparse["ause_normalised"],
            "ece":               calib["ece"],
            "nll":               nll,
            "sharpness_mean":    sharp["mean"],
            "sharpness_entropy": sharp["spatial_entropy"],
            "sharpness_cv":      sharp["coefficient_var"],
        }

    return summary


def compute_full_evaluation_summary(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    pred_faces: np.ndarray,
    gt_point_cloud: np.ndarray,
    uncertainty: np.ndarray,
    region_masks: Optional[Dict[str, np.ndarray]] = None,
    method_name: str = "unknown",
) -> Dict[str, object]:
    """
    End-to-end evaluation pipeline for a single (method, image) pair.

    Calls all Group A and Group C metrics in order and packages the results
    into a flat dict.  Intended to be mapped over a dataset of evaluation images
    and then aggregated (mean/median over images).

    Parameters
    ----------
    pred_vertices : np.ndarray, shape (5023, 3)
    gt_vertices   : np.ndarray, shape (5023, 3)
    pred_faces    : np.ndarray, shape (F, 3)   — for scan-to-mesh
    gt_point_cloud: np.ndarray, shape (N, 3)   — for scan-to-mesh
    uncertainty   : np.ndarray, shape (5023,)  — from any uncertainty.py method
    region_masks  : optional dict of FLAME region vertex indices
    method_name   : str   — label for this method (e.g. "SMIRK_TTA")

    Returns
    -------
    summary : dict with all computed scalar metrics plus the raw error and
              uncertainty arrays for further visualisation.

        "method"               : str
        "per_vertex_error"     : np.ndarray (5023,)
        "vertex_rmse"          : float
        "s2m_median_mm"        : float
        "s2m_mean_mm"          : float
        "region_errors"        : dict  (if region_masks provided)
        "spearman_rho"         : float
        "pearson_r"            : float
        "ause"                 : float
        "ece"                  : float
        "nll"                  : float
        "sharpness"            : float
    """
    # ------------------------------------------------------------------ #
    # Group A: geometric quality                                           #
    # ------------------------------------------------------------------ #
    per_vertex_error = calculate_geometric_error(
        pred_vertices, gt_vertices, align=True
    )                                                       # (5023,)

    vertex_rmse = calculate_vertex_rmse(
        pred_vertices, gt_vertices, align=True
    )

    s2m = calculate_scan_to_mesh_distance(
        pred_vertices, pred_faces, gt_point_cloud, align=True
    )

    region_errors: Dict = {}
    if region_masks is not None:
        region_errors = calculate_region_wise_geometric_error(
            per_vertex_error, region_masks
        )

    # ------------------------------------------------------------------ #
    # Groups B & C: uncertainty quality                                    #
    # ------------------------------------------------------------------ #
    u_flat = np.asarray(uncertainty, dtype=np.float64).ravel()

    corr   = correlate_error_and_uncertainty(per_vertex_error, u_flat, method="both")
    sparse = calculate_sparsification_error_curve(per_vertex_error, u_flat)
    calib  = calculate_uncertainty_calibration(per_vertex_error, u_flat)
    nll    = calculate_nll(per_vertex_error, u_flat)
    sharp  = calculate_uncertainty_sharpness(u_flat)

    return {
        "method":           method_name,
        "per_vertex_error": per_vertex_error,
        "vertex_rmse":      vertex_rmse,
        "s2m_median_mm":    s2m["median_mm"],
        "s2m_mean_mm":      s2m["mean_mm"],
        "region_errors":    region_errors,
        "spearman_rho":     corr.get("spearman_rho", float("nan")),
        "pearson_r":        corr.get("pearson_r",    float("nan")),
        "ause":             sparse["ause"],
        "ece":              calib["ece"],
        "nll":              nll,
        "sharpness":        sharp["mean"],
    }
