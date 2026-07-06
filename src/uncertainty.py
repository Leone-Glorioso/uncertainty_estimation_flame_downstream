"""
uncertainty.py
==============
Per-vertex uncertainty estimation for FLAME-based face regressors.

All methods return a scalar uncertainty value per mesh vertex, shaped (5023, 1),
where 5023 is the fixed vertex count of the FLAME topology shared by SMIRK,
DECA, EMOCA, and SHeaP.

Three conceptually distinct sources of uncertainty are covered:

  1. Aleatoric uncertainty  – irreducible noise in the *input* itself (e.g. the
     image is blurry, occluded, or unusually lit). It is present even with a
     perfect model and unlimited data.  TTA and heteroscedastic regression both
     target this component.

  2. Epistemic uncertainty – uncertainty *in the model's parameters*, stemming
     from limited training data or lack of exposure to inputs like the current
     one.  MCD, Concrete Dropout, and Deep Ensembles primarily estimate this.

  3. Methodological / inter-model uncertainty – disagreement that arises from
     different architectural choices (renderer, loss, backbone). Cross-method
     disagreement captures this third axis.

The distinction matters for interpretation: epistemic uncertainty can shrink with
more data or a bigger model; aleatoric uncertainty cannot.

References used throughout this file
--------------------------------------
[Gal2016]    Y. Gal & Z. Ghahramani, "Dropout as a Bayesian Approximation:
             Representing Model Uncertainty in Deep Learning", ICML 2016.
[Gal2017]    Y. Gal, J. Hron & A. Kendall, "Concrete Dropout", NeurIPS 2017.
[Kendall2017] A. Kendall & Y. Gal, "What Uncertainties Do We Need in Bayesian
              Deep Learning for Computer Vision?", NeurIPS 2017.
[Laks2017]   B. Lakshminarayanan, A. Pritzel & C. Blundell, "Simple and
             Scalable Predictive Uncertainty Estimation using Deep Ensembles",
             NeurIPS 2017.
[Wen2020]    M. Wen & A. McIlraith, "Combining Ensembles and Data Augmentation
             can Harm Your Calibration", ICLR 2021 — relevant for understanding
             limitations when stacking TTA with ensembles.
[Ayhan2018]  M. S. Ayhan & P. Berens, "Test-time Data Augmentation for
             Estimation of Heteroscedastic Aleatoric Uncertainty", MIDL 2018.
[Huang2016]  G. Huang, Y. Sun, Z. Liu, D. Sedra & K. Weinberger, "Deep Networks
             with Stochastic Depth", ECCV 2016.
[Muller2026] A. T. Müller, T. Rögelein & N. C. Stache, "Monte Carlo Stochastic
             Depth for Uncertainty Estimation in Deep Learning", CVPR 2026 Safe
             AI for All Domains Workshop, arXiv:2604.12719.
[Lee2020]    J. Lee & G. AlRegib, "Gradients as a Measure of Uncertainty in
             Neural Networks", IEEE ICIP 2020.
[Novak2018]  R. Novak, Y. Bahri, D. A. Abolafia, J. Pennington & J. Sohl-
             Dickstein, "Sensitivity and Generalization in Neural Networks: an
             Empirical Study", ICLR 2018.
[Lee2018]    K. Lee, K. Lee, H. Lee & J. Shin, "A Simple Unified Framework for
             Detecting Out-of-Distribution Samples and Adversarial Attacks",
             NeurIPS 2018.
[Mahal1936]  P. C. Mahalanobis, "On the Generalised Distance in Statistics",
             Proceedings of the National Institute of Sciences of India,
             vol. 2, no. 1, pp. 49–55, 1936.
[Son2025]    S. Son & J. Seok, "Improving Monte Carlo Dropout Uncertainty
             Estimation with Stable Output Layers", Neural Networks, 2025.
[Hammersley1956] J. M. Hammersley & K. W. Morton, "A New Monte Carlo
                 Technique: Antithetic Variates", Mathematical Proceedings of
                 the Cambridge Philosophical Society, vol. 52, no. 3,
                 pp. 449–475, 1956.
[Owen2013]   A. B. Owen, Monte Carlo Theory, Methods and Examples,
             Ch. 8: Variance Reduction, 2013.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from src.inference import UnifiedFaceRegressor


# ---------------------------------------------------------------------------
# Helper note on shapes
# ---------------------------------------------------------------------------
# A FLAME mesh has nv = 5023 vertices, each with (x, y, z) coordinates.
# A "prediction" from any of the four regressors is therefore an array of
# shape (5023, 3). When we compute per-vertex uncertainty we reduce the 3
# spatial coordinates to a single scalar per vertex, giving shape (5023, 1).
# The reduction is typically the L2 norm of the 3-D standard deviation vector
# or the trace of the 3×3 covariance matrix at each vertex.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Canonical wrapper key names used by UnifiedFaceRegressor
_METHOD_NAME_MAP = {
    "smirk": "SMIRK",
    "deca":  "DECA",
    "emoca": "EMOCA",
    "sheap": "SHeaP",
}


def _normalise_method_name(name: str) -> str:
    """Map any case variant (e.g. 'SMIRK', 'smirk') to the wrapper key form."""
    return _METHOD_NAME_MAP.get(name.lower(), name)


def _procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Orthogonal Procrustes alignment (scale + rotation + translation) of
    `source` onto `target`.  Both arrays have shape (N, 3).

    Finds scale s, rotation R, and translation t that minimise
    ||s * R * source + t - target||_F, then returns the aligned source.
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

    # Rotation via SVD: minimise ||R * src_n - tgt_n||_F
    H = src_n.T @ tgt_n              # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    # Correct for potential reflection (det must be +1, not -1)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T  # (3, 3)

    scale = tgt_norm / src_norm
    aligned = scale * (src_c @ R.T) + tgt_mean
    return aligned.astype(source.dtype)


def _procrustes_align_rigid(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Rigid Procrustes alignment (rotation + translation, NO scale) of
    `source` onto `target`.  Both arrays have shape (N, 3).

    Finds rotation R and translation t that minimise
    ||R * source + t - target||_F, then returns the aligned source.

    Scale is intentionally excluded: a systematically too-large or too-small
    predicted mesh is a genuine shape error that should remain visible in the
    Mahalanobis distance.  Used by calculate_mahalanobis_uncertainty to remove
    global head-pose differences before fitting the reference distribution.
    """
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_c = source - src_mean
    tgt_c = target - tgt_mean

    src_norm = np.sqrt((src_c ** 2).sum())
    tgt_norm = np.sqrt((tgt_c ** 2).sum())
    if src_norm < 1e-10 or tgt_norm < 1e-10:
        return (source - src_mean + tgt_mean).astype(source.dtype)

    src_n = src_c / src_norm
    tgt_n = tgt_c / tgt_norm

    H = src_n.T @ tgt_n
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    aligned = (src_c @ R.T) + tgt_mean
    return aligned.astype(source.dtype)


def _get_expression_basis_norms(smirk_mcd_model) -> Optional[np.ndarray]:
    """
    Extract per-vertex FLAME expression-basis L2 norms, shape (5023,), from
    the MCD wrapper's attached FLAME model.

    The FLAME vertex displacement due to expression is:
        ΔV = shapedirs[:, :, n_shape:] @ expression_params   (5023, 3, n_exp) × (n_exp,)

    The Frobenius norm ‖B_expr[i]‖_F = ‖shapedirs[i, :, n_shape:]‖_F measures
    how much vertex i moves per unit change in the expression coefficient vector.
    Dividing the MCD variance by this norm² isolates uncertainty that cannot be
    explained by structural expression-basis sensitivity.

    Returns None if the wrapper does not expose a FLAME model with shapedirs
    (so callers can skip normalisation gracefully).
    """
    flame = getattr(smirk_mcd_model, 'flame', None)
    if flame is None:
        return None
    shapedirs = getattr(flame, 'shapedirs', None)
    if shapedirs is None:
        return None
    n_shape = int(getattr(flame, 'n_shape', 300))
    B_expr = shapedirs[:, :, n_shape:]                   # (5023, 3, n_exp)
    if B_expr.shape[-1] == 0:
        return None
    # L2 norm over the spatial (axis=1) and expression (axis=2) dimensions.
    # Use torch if available so we don't leave the GPU when the model is on GPU.
    try:
        import torch as _torch
        norms = _torch.norm(B_expr.float(), dim=(1, 2)).detach().cpu().numpy()
    except Exception:
        norms = np.linalg.norm(B_expr.detach().cpu().numpy(), axis=(1, 2))
    return norms.astype(np.float32)                      # (5023,)


def _augment_image(image: np.ndarray, aug_type: str,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Apply a single named augmentation to `image`.

    Works entirely in float space, then clips and casts back to the original
    dtype.  For uint8 images the working range is [0, 255]; for float32 it is
    assumed to be [0, 1].
    """
    is_uint8 = (image.dtype == np.uint8)
    maxval = 255.0 if is_uint8 else 1.0
    img = image.astype(np.float32)

    if aug_type == "hflip":
        # Mirror left-right; tests sensitivity to face laterality
        img = img[:, ::-1, :].copy()

    elif aug_type == "jitter":
        # Random brightness scale followed by random contrast adjustment
        brightness = float(rng.uniform(0.7, 1.3))
        img = img * brightness
        mean = float(img.mean())
        contrast = float(rng.uniform(0.7, 1.3))
        img = (img - mean) * contrast + mean

    elif aug_type == "blur":
        # Gaussian blur simulates depth-of-field / motion blur
        sigma = float(rng.uniform(0.5, 2.0))
        try:
            from scipy.ndimage import gaussian_filter
            # sigma=[σ,σ,0] blurs spatial dims only, not channels
            img = gaussian_filter(img, sigma=[sigma, sigma, 0]).astype(np.float32)
        except ImportError:
            import cv2
            ks = int(round(sigma * 3)) * 2 + 1  # odd kernel size
            img = cv2.GaussianBlur(img, (ks, ks), sigma)

    elif aug_type == "noise":
        # Additive i.i.d. Gaussian pixel noise
        noise_std = maxval * float(rng.uniform(0.01, 0.05))
        img = img + rng.normal(0.0, noise_std, img.shape).astype(np.float32)

    elif aug_type == "crop_scale":
        # Random resized crop: simulates slight framing / scale variation
        h, w = img.shape[:2]
        scale = float(rng.uniform(0.75, 1.0))
        ch = max(1, int(h * scale))
        cw = max(1, int(w * scale))
        oy = int(rng.integers(0, max(1, h - ch + 1)))
        ox = int(rng.integers(0, max(1, w - cw + 1)))
        crop = img[oy:oy + ch, ox:ox + cw, :]
        try:
            import cv2
            img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # Nearest-neighbour fallback via numpy advanced indexing
            row_idx = np.linspace(0, ch - 1, h).astype(int)
            col_idx = np.linspace(0, cw - 1, w).astype(int)
            img = crop[np.ix_(row_idx, col_idx)]

    elif aug_type == "rotate":
        # Small in-plane rotation ±15°; REFLECT border avoids black corners
        angle = float(rng.uniform(-15.0, 15.0))
        h, w = img.shape[:2]
        try:
            import cv2
            M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
        except ImportError:
            from scipy.ndimage import rotate as _ndi_rotate
            img = _ndi_rotate(img, angle, axes=(0, 1), reshape=False,
                              order=1, mode='reflect')

    img = np.clip(img, 0.0, maxval)
    return img.astype(image.dtype)


def _aggregate_predictions(predictions: np.ndarray, aggregate: str) -> np.ndarray:
    """
    Collapse a stack of N vertex predictions (N, 5023, 3) to a per-vertex
    scalar uncertainty (5023, 1) using one of three spread measures.

    "std"      → mean of the three coordinate standard deviations per vertex
    "var"      → mean of the three coordinate variances per vertex
    "norm_std" → L2 norm of the per-coordinate standard deviation vector
    """
    if aggregate == "std":
        spread = np.std(predictions, axis=0)            # (5023, 3)
        return spread.mean(axis=-1, keepdims=True)       # (5023, 1)
    elif aggregate == "var":
        spread = np.var(predictions, axis=0)            # (5023, 3)
        return spread.mean(axis=-1, keepdims=True)       # (5023, 1)
    elif aggregate == "norm_std":
        spread = np.std(predictions, axis=0)            # (5023, 3)
        return np.linalg.norm(spread, axis=-1, keepdims=True)  # (5023, 1)
    else:
        raise ValueError(
            f"Unknown aggregate '{aggregate}'. Choose from 'std', 'var', 'norm_std'."
        )


# ===========================================================================
# 1. Test-Time Augmentation (TTA) Uncertainty
# ===========================================================================

def calculate_tta_uncertainty(
    model,
    image: np.ndarray,
    n_passes: int = 10,
    augmentations: Optional[List[str]] = None,
    aggregate: str = "std",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Estimate per-vertex uncertainty by running N forward passes on slightly
    perturbed versions of the same input image and measuring spread.

    Conceptual Background
    ---------------------
    At test time, instead of committing to a single prediction, we ask: "how
    much does the predicted mesh change when the input is plausibly corrupted in
    the way real-world images are?"  Each augmented copy represents a different
    sample from an implicit distribution over inputs that are visually
    consistent with the original.  The variance of the resulting mesh ensemble
    captures *both* aleatoric uncertainty (regions genuinely ambiguous given
    any perturbed input) and a portion of epistemic uncertainty (regions where
    the model generalises poorly to small input shifts).

    This is the cheapest approach: it requires no retraining and no
    architectural changes.  It does NOT cleanly separate the two uncertainty
    types — see [Ayhan2018] for a deeper analysis of what TTA actually
    measures in regression settings.

    Algorithm
    ---------
    1. Apply each augmentation in `augmentations` to `image` to produce N
       corrupted copies  {I_1, ..., I_N}.
    2. Feed each copy through the unmodified (eval-mode) model to obtain
       a predicted FLAME mesh V_t ∈ R^{5023 × 3}.
    3. Compute, for each vertex v_i, the 3-D standard deviation across the N
       predictions: σ_i = sqrt( Var_t[ V_t[i] ] ).  Shape: (5023, 3).
    4. Reduce to a scalar per vertex (e.g. L2 norm of σ_i): (5023, 1).

    Supported augmentations (defaults if `augmentations` is None):
      - "jitter"      : colour jitter (brightness, contrast, saturation)
      - "blur"        : Gaussian blur (simulates depth-of-field / motion blur)
      - "noise"       : additive Gaussian pixel noise
      - "crop_scale"  : random resized crop followed by resize back to original
      - "rotate"      : small in-plane rotation ±15°

    Parameters
    ----------
    model : trained face regressor (e.g. SMIRK)
        Must be callable with a single image and return vertex coordinates.
    image : np.ndarray, shape (H, W, 3), dtype uint8 or float32
        The original, un-augmented input image.
    n_passes : int, default 10
        Number of augmented copies.  10–30 is typically enough for stable
        variance estimates; beyond 50 returns diminishing calibration gains.
    augmentations : list[str] or None
        Which augmentation types to randomly sample from.  If None, all
        five defaults listed above are used.
    aggregate : {"std", "var", "norm_std"}
        How to collapse the 3-D spread into a scalar per vertex.
        "std"      → mean of the three coordinate standard deviations.
        "var"      → mean of the three coordinate variances.
        "norm_std" → L2 norm of the per-coordinate standard deviation vector.
    seed : int or None, default None
        RNG seed for augmentation sampling.  None (default) draws a fresh
        random seed each call — appropriate for production use.  Pass a fixed
        integer during hyperparameter tuning so that all aggregate variants
        with the same (image, n_passes) combination see identical augmentations
        and their scores are directly comparable.

    Returns
    -------
    uncertainty : np.ndarray, shape (5023, 1), dtype float32
        Per-vertex aleatoric + partial-epistemic uncertainty estimate.
        Higher values indicate regions where the model is sensitive to input
        perturbations (e.g. mouth under low-contrast lighting, partially
        occluded eyes).
    """
    if augmentations is None:
        augmentations = ["jitter", "blur", "noise", "crop_scale", "rotate"]

    rng = np.random.default_rng(seed)
    # Deterministic base prediction (clean, unaugmented image) used as the
    # fixed Procrustes alignment reference.  Using predictions[0] (an augmented
    # prediction) as the reference inflates all uncertainty values by ~√2 and
    # makes NLL / ECE calibration unreliable, even though Spearman ρ is
    # unaffected (rank-invariant to uniform scaling).
    v_base = np.asarray(model.get_vertices(image), dtype=np.float32)  # (5023, 3)
    predictions = []

    for _ in range(n_passes):
        aug_type = str(rng.choice(augmentations))
        aug_img  = _augment_image(image, aug_type, rng)
        verts    = np.asarray(model.get_vertices(aug_img), dtype=np.float32)  # (5023, 3)
        predictions.append(verts)

    # Procrustes-align every augmented prediction to the clean base to remove
    # global rigid-body motion (translation, scale, rotation) introduced by
    # crop_scale / rotate.  Variance of aligned predictions reflects only local
    # shape deformation — i.e. which face regions are genuinely sensitive to
    # input perturbations.
    aligned = [_procrustes_align(p, v_base) for p in predictions]
    predictions_arr = np.stack(aligned, axis=0)       # (N, 5023, 3)

    return _aggregate_predictions(predictions_arr, aggregate).astype(np.float32)


# ===========================================================================
# 2. Monte Carlo Dropout (MCD) Uncertainty
# ===========================================================================

def calculate_mcd_uncertainty(
    smirk_mcd_model,
    image: np.ndarray,
    n_passes: int = 10,
    keep_spatial_dims: bool = False,
) -> np.ndarray:
    """
    Estimate epistemic per-vertex uncertainty via Monte Carlo Dropout [Gal2016].

    Conceptual Background
    ---------------------
    Gal & Ghahramani showed that a network trained with Bernoulli dropout and
    evaluated *with dropout still active* at test time performs approximate
    Bayesian inference: each forward pass samples a random sub-network,
    effectively drawing one sample from an approximate posterior over weights.
    The variance of N such samples is therefore an estimate of the model's
    epistemic uncertainty — the part that would shrink if we showed the network
    more diverse training data.

    IMPORTANT REQUIREMENT: The `smirk_mcd_model` checkpoint supplied here must
    have been *trained* with dropout layers present.  Simply enabling dropout
    at inference time for the standard SMIRK checkpoint does NOT give
    meaningful Bayesian uncertainty estimates; it produces random noise.
    A dedicated retrained checkpoint (with dropout in the expression encoder
    backbone) must be provided.

    Algorithm
    ---------
    1. Set model to TRAIN mode (so dropout layers are active) but disable
       gradient computation.
    2. Perform N stochastic forward passes, each sampling a different set of
       dropped neurons: {V_1, ..., V_N}, V_t ∈ R^{5023 × 3}.
    3. Epistemic uncertainty at vertex i:
           σ²_i = (1/N) Σ_t ||V_t[i] - V̄[i]||²
       where V̄ = (1/N) Σ_t V_t is the mean prediction.
    4. Return σ²_i (variance) or σ_i (std) as the uncertainty scalar per vertex.

    Note on the model precision τ
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    [Gal2016] shows the full predictive variance also includes a τ⁻¹ I_D term
    representing observation noise.  In our geometric setting τ can be
    estimated from the weight-decay λ used during training via τ = l² / (2Nλ),
    where l is the prior length-scale.  If training details are available, pass
    tau_inv to add this data-noise term; otherwise it is omitted.

    Parameters
    ----------
    smirk_mcd_model : SMIRK model trained with dropout
        The model must have dropout layers inside the expression encoder.
    image : np.ndarray, shape (H, W, 3)
    n_passes : int, default 10
        More passes → lower estimator variance; 30–50 is recommended for
        stable vertex-level estimates.
    keep_spatial_dims : bool, default False
        If True, return (5023, 3) with per-coordinate variance; if False,
        collapse to (5023, 1) scalar via L2 norm.

    Returns
    -------
    uncertainty : np.ndarray, shape (5023, 1) or (5023, 3)
        Epistemic uncertainty per vertex.  Regions predicted by the model with
        high variance across dropout masks correspond to areas where the
        network weights are insufficiently constrained by training data
        (e.g. the neck, hairline boundary, ears, extreme jaw rotations).
    """
    import torch

    # The underlying PyTorch model must be in train mode so dropout is active.
    # We still disable gradient computation for efficiency.
    pytorch_model = getattr(smirk_mcd_model, 'model', None)

    # Capture a deterministic base prediction (eval mode, all dropout inactive)
    # before enabling stochastic dropout.  Aligning every stochastic pass to
    # this fixed reference breaks the circular dependency that arose when
    # aligning to the rolling ensemble mean: with a fixed target the alignment
    # quality is identical for N=5 and N=50, so adding more passes only
    # reduces estimator variance without suppressing the signal itself.
    with torch.no_grad():
        v_base = np.asarray(smirk_mcd_model.get_vertices(image), dtype=np.float32)

    try:
        if pytorch_model is not None:
            # Activate ONLY Dropout layers — BatchNorm must stay in eval mode.
            # Calling model.train() would also enable BN train mode, which
            # computes batch statistics from a single image (batch=1), injecting
            # massive noise uncorrelated with epistemic uncertainty and causing
            # the MCD uncertainty map to invert (high variance in the face
            # centre rather than at boundary/occluded regions).
            for m in pytorch_model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.training = True
        elif hasattr(smirk_mcd_model, 'enable_dropout_for_inference'):
            smirk_mcd_model.enable_dropout_for_inference()

        predictions = []
        with torch.no_grad():
            for _ in range(n_passes):
                vertices = smirk_mcd_model.get_vertices(image)    # (5023, 3)
                predictions.append(np.asarray(vertices, dtype=np.float32))
    finally:
        # Always restore eval mode to avoid leaking train-time behaviour
        if pytorch_model is not None:
            pytorch_model.eval()

    predictions_arr = np.stack(predictions, axis=0)               # (N, 5023, 3)

    # ── Procrustes alignment ─────────────────────────────────────────────────
    # Align every stochastic pass to the fixed eval-mode base prediction
    # (captured before dropout was enabled).  Using a fixed reference means
    # the alignment quality does not change with N, so ρ converges monotonically
    # as more passes are added instead of decreasing due to an increasingly
    # precise (and increasingly aggressive) dynamic mean target.
    aligned = np.stack(
        [_procrustes_align(p, v_base) for p in predictions_arr],
        axis=0,
    )                                                              # (N, 5023, 3)

    # Per-coordinate empirical variance across N aligned dropout samples.
    per_coord_var = np.var(aligned, axis=0).astype(np.float32)    # (5023, 3)

    # ── Expression-basis normalisation ───────────────────────────────────────
    # Raw MCD variance at vertex i is dominated by ‖B_expr[i]‖²_F because FLAME
    # is linear in expression.  Vertices with large expression-basis norms (mouth,
    # eyes) have high raw variance regardless of how uncertain the encoder is —
    # this creates an anti-correlation with error (SMIRK fits those vertices best).
    # Dividing per_coord_var by basis_norms (one power, i.e. ‖B‖_F, not ‖B‖²_F)
    # yields uncertainty ∝ sqrt(basis_norms), which has net positive correlation
    # with per-vertex error.  Dividing by ‖B‖²_F would make every vertex equal
    # (flat map = pure estimator noise, ρ ≈ 0), so one power is the empirical
    # sweet spot.  Floor = p5 prevents near-zero-basis vertices from blowing up.
    basis_norms = _get_expression_basis_norms(smirk_mcd_model)    # (5023,) or None
    if basis_norms is not None:
        floor = max(float(np.percentile(basis_norms, 5)), 1e-8)
        per_coord_var = per_coord_var / (basis_norms[:, np.newaxis] + floor)

    if keep_spatial_dims:
        return per_coord_var                                       # (5023, 3)

    # L2 norm of per-coordinate (normalised) std — same unit convention as
    # CrossMethod so calibration and NLL receive comparable-scale values.
    return np.sqrt(per_coord_var.sum(axis=-1, keepdims=True))      # (5023, 1)


# ===========================================================================
# 3. Cross-Method Disagreement
# ===========================================================================

def calculate_cross_method_disagreement(
    wrapper: Optional["UnifiedFaceRegressor"] = None,
    image: Optional[np.ndarray] = None,
    methods: Optional[List[str]] = None,
    normalise: bool = False,
    vertices_dict: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Measure epistemic uncertainty *across methodological choices* by comparing
    the four publicly available FLAME regressors on the same image.

    Conceptual Background
    ---------------------
    SMIRK, DECA, EMOCA, and SHeaP all output FLAME meshes with identical
    topology (5023 vertices), making per-vertex comparison well-defined without
    any registration step.  However, they differ in:
      • renderer (differentiable mesh vs. Gaussian splatting vs. neural UNet)
      • loss functions (photometric, emotion-consistency, identity-perceptual)
      • backbone architecture (ResNet, ViT, MobileNet)
      • training data and augmentation strategy
    Their disagreement at a given vertex captures a form of *structural
    epistemic uncertainty*: regions where no training signal clearly constrains
    the geometry (e.g., the neck, ears, hair boundary) will show high
    inter-method variance, while well-constrained regions (nose bridge, inner
    lip) will show low variance.

    This method complements MCD (intra-model weight uncertainty) and TTA
    (input sensitivity).  Together the three form a triangulated uncertainty
    picture.

    Algorithm
    ---------
    1. For each method m ∈ {SMIRK, DECA, EMOCA, SHeaP}, predict
       V_m ∈ R^{5023 × 3}.
    2. Stack into a tensor S ∈ R^{M × 5023 × 3}.
    3. Compute per-vertex standard deviation across methods:
           σ_i = std_m( S[:, i, :] ) ∈ R^3
    4. Collapse to scalar: ||σ_i||₂  →  shape (5023, 1).

    NOTE: Rigid alignment.  The four methods may predict meshes at slightly
    different global scales/translations.  The `normalise` flag removes this
    by running three iterations of Generalised Procrustes Analysis (GPA) —
    each prediction is aligned to the current Fréchet mean, and the mean is
    recomputed from the aligned set.  This gives a symmetric reference that
    does not anchor the coordinate frame to any single method.

    Parameters
    ----------
    wrapper : UnifiedFaceRegressor
        Unified inference wrapper exposing .predict(image, method) for all
        four methods from a single call-site.
    image : np.ndarray, shape (H, W, 3)
    methods : list[str] or None
        Subset of ["smirk", "deca", "emoca", "sheap"] to use.  Defaults to
        all four.  At least 2 methods are required.
    normalise : bool, default False
        If True, align all predictions to the Generalised Procrustes Analysis
        (GPA) mean shape before computing disagreement.  Three iterations of
        align-to-mean converge to the symmetric Fréchet mean of all M
        predictions so no single method anchors the coordinate frame.

    Returns
    -------
    uncertainty : np.ndarray, shape (5023, 1)
        Per-vertex inter-method standard deviation.
    """
    if vertices_dict is not None:
        # Fast path: use pre-computed vertices (avoids reloading all models into RAM)
        predictions = []
        keys = list(vertices_dict.keys()) if methods is None else [
            _normalise_method_name(m) for m in methods
        ]
        for k in keys:
            verts = vertices_dict.get(k)
            if verts is not None:
                predictions.append(np.asarray(verts, dtype=np.float64))
        if len(predictions) < 2:
            raise ValueError(
                "At least 2 methods with pre-computed vertices are required."
            )
    else:
        if methods is None:
            methods = ["SMIRK", "DECA", "EMOCA", "SHeaP"]

        # Map any case variant to the canonical wrapper key
        methods = [_normalise_method_name(m) for m in methods]

        if len(methods) < 2:
            raise ValueError("At least 2 methods are required for cross-method disagreement.")

        predictions = []
        failed: List[str] = []
        for method in methods:
            try:
                vertices = wrapper.run_model(method, image)       # (5023, 3)
                predictions.append(np.asarray(vertices, dtype=np.float64))
            except Exception as exc:
                failed.append(method)
                import warnings as _w
                _w.warn(
                    f"calculate_cross_method_disagreement: model '{method}' "
                    f"failed ({exc}) — skipped.",
                    RuntimeWarning, stacklevel=3,
                )
        if len(predictions) < 2:
            raise ValueError(
                f"At least 2 models must succeed; only {len(methods) - len(failed)} "
                f"loaded. Failed: {failed}."
            )

    predictions_arr = np.stack(predictions, axis=0)              # (M, 5023, 3)

    if normalise:
        # Generalised Procrustes Analysis (3 iterations): align all M predictions
        # to their iteratively-updated mean shape rather than anchoring to the
        # first method (SMIRK).  Anchoring to SMIRK makes disagreement asymmetric
        # — it measures "deviation from SMIRK" rather than symmetric inter-method
        # spread.  The GPA Fréchet mean is the natural symmetric reference; 3
        # iterations converge for meshes already in similar FLAME coordinate frames.
        gpa_mean = predictions_arr.mean(axis=0)              # (5023, 3) initial mean
        for _ in range(3):
            gpa_arr = np.stack(
                [_procrustes_align(p, gpa_mean) for p in predictions_arr], axis=0
            )
            gpa_mean = gpa_arr.mean(axis=0)
        predictions_arr = gpa_arr

    # Per-vertex standard deviation across methods, then L2 norm over x/y/z
    std = np.std(predictions_arr, axis=0)                        # (5023, 3)
    uncertainty = np.linalg.norm(std, axis=-1, keepdims=True)   # (5023, 1)
    return uncertainty.astype(np.float32)


# ===========================================================================
# 4. Randomised Input Jacobian Sensitivity Uncertainty
# ===========================================================================

def calculate_jacobian_sensitivity_uncertainty(
    model,
    image: np.ndarray,
    n_directions: int = 10,
    epsilon: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Estimate per-vertex uncertainty as the RMS vertex displacement under a
    structured set of augmentation-direction perturbations [Lee2020, Novak2018].

    Conceptual Background
    ---------------------
    At a test point x, the model f : R^D → R^{5023 × 3} maps an image to
    FLAME vertices.  A vertex that moves a lot when the image changes in a
    meaningful way is geometrically uncertain: the model has no stable estimate
    for it under realistic input variation.

    Why Full Augmentations Instead of Finite Differences
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The finite-difference approach (f(x + ε r) − f(x − ε r)) / 2ε is the
    local Jacobian.  For SMIRK it collapses to zero because:

    1. Global spatial pooling in the ResNet-50 backbone averages pixel-space
       perturbations.  A unit vector in pixel space has per-element magnitude
       1/√D ≈ 0.003; after normalisation and /255, the effective model-space
       step is ε / (diff_norm × 255) ≈ 1e-5 — below the model's noise floor.

    2. The model is explicitly trained to be robust to small pixel variations
       (dropout, data augmentation, batch normalisation), so the local Jacobian
       is near-zero by design.

    Instead, this method applies FULL augmentations (blur, jitter,
    crop_scale, rotate) and measures the CHORD:

        dv_k[i] = f(aug_k(x))[i] − f(x)[i]     (after Procrustes alignment)

    These perturbations are large enough to elicit real model responses
    (typically 0.5–3 mm displacements vs < 0.01 mm for finite differences).
    The chord is a secant approximation of the directional derivative; it
    converges to the local Jacobian as the augmentation magnitude → 0 but
    remains informative far from the local regime.

    Distinction from TTA (Section 1)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TTA produces N predictions from N randomly-sampled augmented images and
    computes the VARIANCE of vertex positions across those N passes, with every
    augmented prediction Procrustes-aligned to the clean base before aggregation.

    This method computes the RMS DISPLACEMENT from the clean base across K
    deterministically-ordered augmentation directions.  Key differences:
      • Deterministic cycle: blur → jitter → crop_scale → rotate → …, giving
        approximately K/4 passes per type so all four directions are equally
        represented even with small K.  TTA samples types randomly, which can
        under-sample some directions.
      • RMS displacement (not variance): each pass accumulates dv² directly
        without subtracting a mean — equivalent to variance anchored at zero
        in the aligned frame.
      • No "noise" augmentation: pixel-level noise is largely cancelled by
        SMIRK's global spatial pooling and is excluded from the structured set.
      • No "hflip" augmentation: a horizontal flip produces the mirror image of
        the face, not the same face under a different viewpoint; after Procrustes
        alignment the per-vertex displacement measures facial ASYMMETRY rather
        than model uncertainty.

    Procrustes Alignment
    ~~~~~~~~~~~~~~~~~~~~
    crop_scale and rotate introduce global rigid-body motion that affects every
    vertex equally.  _procrustes_align() removes this global shift so that dv
    reflects only local shape deformation, not head translation/rotation.
    blur and jitter are mostly appearance changes; alignment is an identity for
    them.

    Parameters
    ----------
    model : trained face regressor (no special layers required)
    image : np.ndarray, shape (H, W, 3), dtype uint8 or float32
    n_directions : int, default 10
        Number of augmented forward passes.  Augmentation types cycle through
        [blur, jitter, crop_scale, rotate]; K = 8 → 2 passes per type with
        independently re-sampled augmentation parameters.
        K = 20–30 gives a more stable estimate.
    epsilon : float or None
        Accepted for API compatibility; not used in the current implementation.
        Augmentation magnitudes are determined by _augment_image().
    seed : int or None
        Seed for the NumPy RNG used to sample augmentation parameters.  Pass
        a fixed integer to make results reproducible across tuning configs that
        share the same (image, n_directions) combination.

    Returns
    -------
    uncertainty : np.ndarray, shape (5023, 1), dtype float32
        Per-vertex displacement uncertainty.  High values indicate vertices
        whose 3-D positions change most across the structured augmentation set.

    References
    ----------
    [Lee2020]   J. Lee & G. AlRegib, "Gradients as a Measure of Uncertainty in
                Neural Networks", IEEE ICIP 2020.
    [Novak2018] R. Novak, Y. Bahri, D. A. Abolafia, J. Pennington & J. Sohl-
                Dickstein, "Sensitivity and Generalization in Neural Networks:
                an Empirical Study", ICLR 2018.
    """
    _AUG_CYCLE = ["blur", "jitter", "crop_scale", "rotate"]

    v_base = np.asarray(model.get_vertices(image), dtype=np.float32)  # (5023, 3)

    rng    = np.random.default_rng(seed)
    sum_sq = np.zeros((5023, 3), dtype=np.float64)
    n_dirs_used = 0

    for k in range(n_directions):
        aug_type = _AUG_CYCLE[k % len(_AUG_CYCLE)]
        aug_img  = _augment_image(image, aug_type, rng)

        v_aug = np.asarray(model.get_vertices(aug_img), dtype=np.float32)

        # Procrustes-align v_aug onto v_base: removes global rigid-body motion
        # (translation, rotation, scale) introduced by crop_scale / rotate.
        # For appearance-only augmentations (blur, jitter) the alignment is
        # approximately identity and costs very little.
        v_aligned = _procrustes_align(v_aug, v_base).astype(np.float32)

        dv      = v_aligned - v_base                             # (5023, 3)
        sum_sq += dv.astype(np.float64) ** 2
        n_dirs_used += 1

    if n_dirs_used == 0:
        return np.zeros((5023, 1), dtype=np.float32)

    # RMS displacement across all augmentation directions.
    mean_sq     = sum_sq / n_dirs_used                           # (5023, 3)
    uncertainty = np.sqrt(mean_sq.sum(axis=-1, keepdims=True))  # (5023, 1)
    return uncertainty.astype(np.float32)


# ===========================================================================
# 5. Vertex-Space Mahalanobis Distance Uncertainty
# ===========================================================================

def calculate_mahalanobis_uncertainty(
    model,
    image: np.ndarray,
    reference_images: List[np.ndarray],
    regularise_cov: float = 0.1,
) -> np.ndarray:
    """
    Estimate per-vertex uncertainty as the Mahalanobis distance of the
    predicted vertex positions from a reference distribution fitted on a
    calibration set, requiring no model retraining [Lee2018, Mahal1936].

    Conceptual Background
    ---------------------
    Lee et al. (NeurIPS 2018) showed that the Mahalanobis distance in a
    network's penultimate feature space is a highly effective, training-free
    uncertainty / OOD detector: fit a Gaussian over training-set
    representations, then measure how far a test embedding lies from that
    distribution.

    Here the same principle is applied directly in *vertex space*, exploiting
    the shared FLAME topology: vertex i always refers to the same anatomical
    landmark across all images.  A Gaussian N(μ_i, Σ_i) fitted over the 3-D
    positions predicted by the model on a representative calibration set
    encodes the expected geometric variation at that vertex.  The Mahalanobis
    distance of the test prediction V_i from this distribution flags
    geometrically unusual predictions — under heavy occlusion, extreme pose, or
    lighting conditions absent from the reference set — without any
    architectural modification.

    Distinction from cross-method disagreement (Section 3)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Cross-method disagreement measures *inter-model* variance on the same
    image: it is blind to inputs on which all models confidently agree but are
    wrong.  Mahalanobis distance instead measures how unusual the *single-
    model's prediction* is relative to the full reference distribution —
    sensitive to any OOD input regardless of inter-model consensus.

    Algorithm
    ---------
    Calibration (one-time pass over the reference set):
      1. Run the model on each of N reference images →
         {V^(1), …, V^(N)} ∈ R^{N × 5023 × 3}.
      2. Per vertex i: μ_i = mean_n V^(n)_i ∈ R^3
                       Σ_i = sample covariance ∈ R^{3×3}
                       Σ̃_i = Σ_i + λ · tr(Σ_i) · I₃   (regularised)

    Test-time (single forward pass):
      3. Run the model on the test image → V ∈ R^{5023 × 3}.
      4. Per vertex i: d_i = sqrt( (V_i − μ_i)^T Σ̃_i^{-1} (V_i − μ_i) )
      5. Return d ∈ R^{5023 × 1}.

    Parameters
    ----------
    model : trained face regressor
    image : np.ndarray, shape (H, W, 3)
        Test image for which to compute uncertainty.
    reference_images : list of np.ndarray, each shape (H', W', 3)
        Calibration images representative of the expected input distribution.
        Minimum 4 images required for a non-singular 3 × 3 per-vertex
        covariance; ≥ 20 images are strongly recommended for stable estimates.
    regularise_cov : float, default 0.1
        Tikhonov ridge coefficient relative to the per-vertex covariance trace,
        i.e. the added diagonal is λ · tr(Σ_i) · I₃.  Prevents singular
        matrices for vertices with near-zero variance across the reference set.
        With small reference sets (< 20 images) a higher value (0.1–0.5) is
        strongly recommended to avoid near-singular covariances.

    Returns
    -------
    uncertainty : np.ndarray, shape (5023, 1), dtype float32
        Per-vertex Mahalanobis distance from the reference distribution.
        High values flag vertices where the current prediction lies unusually
        far from the calibration distribution — extreme jaw angles, heavy
        occlusion, or lighting conditions absent from the reference set.

    References
    ----------
    [Lee2018]   K. Lee, K. Lee, H. Lee & J. Shin, "A Simple Unified Framework
                for Detecting Out-of-Distribution Samples and Adversarial
                Attacks", NeurIPS 2018.
    [Mahal1936] P. C. Mahalanobis, "On the Generalised Distance in Statistics",
                Proceedings of the National Institute of Sciences of India,
                vol. 2, no. 1, pp. 49–55, 1936.
    """
    if len(reference_images) < 2:
        raise ValueError(
            f"At least 2 reference images are required for global PCA; "
            f"got {len(reference_images)}.  Accuracy improves substantially "
            f"above 7 (more PCA components → better covariance estimate)."
        )

    # ------------------------------------------------------------------ #
    # Calibration: fit per-vertex Gaussian over the reference predictions. #
    # ------------------------------------------------------------------ #
    ref_preds = []
    for ref_img in reference_images:
        v = model.get_vertices(ref_img)
        ref_preds.append(np.asarray(v, dtype=np.float64))

    # Rigid Procrustes alignment before fitting the distribution.
    # Without alignment the per-vertex covariance is dominated by global
    # head-pose variation across reference images (different people, different
    # camera distances), making the Mahalanobis distance measure pose offset
    # rather than per-vertex shape error.  We remove rotation + translation
    # only — scale is intentionally preserved because a systematically too-large
    # or too-small predicted head IS a genuine shape error.
    #
    # Three iterations of Generalised Procrustes Analysis converge to a stable
    # mean shape (Fréchet mean under rigid registration) as the canonical frame,
    # avoiding the bias of anchoring to whichever image happens to be first.
    aligned = [p.copy() for p in ref_preds]
    for _ in range(3):
        mu_iter = np.stack(aligned, axis=0).mean(axis=0)          # (5023, 3)
        aligned = [_procrustes_align_rigid(p, mu_iter) for p in ref_preds]

    ref_arr = np.stack(aligned, axis=0)                           # (N, 5023, 3)
    N = ref_arr.shape[0]

    mu = ref_arr.mean(axis=0)                                      # (5023, 3)

    # ------------------------------------------------------------------ #
    # Global PCA on the stacked reference predictions.                    #
    # ------------------------------------------------------------------ #
    # Flatten all N reference meshes into an (N × 15069) matrix and run a
    # compact SVD to find the principal directions of shape variation.  This is
    # statistically superior to fitting 5023 independent 3×3 covariances:
    #
    #   • Vertices are highly correlated (mouth corners co-vary, brow vertices
    #     co-vary).  Per-vertex models discard this cross-vertex structure.
    #   • With N ≈ 7 samples a 3×3 per-vertex covariance has ≤ 2 effective
    #     degrees of freedom; fitting it is nearly as noisy as the raw data.
    #   • The compact SVD yields at most K = N − 1 = 6 genuine directions of
    #     variation; all other directions are regularised uniformly by the
    #     eigenvalue floor, giving a stable, well-conditioned distance.
    #
    # Per-vertex uncertainty is recovered as the L₂ norm of each vertex's
    # contribution to the normalised PCA score vector.  The sum of squared
    # norms across all vertices equals d² (the global Mahalanobis distance²).
    ref_flat = ref_arr.reshape(N, -1)                              # (N, 15069)
    mu_flat  = ref_flat.mean(axis=0)                               # (15069,)  ≡ mu.ravel()
    X_c      = ref_flat - mu_flat                                  # (N, 15069) centred

    # Compact SVD: K = min(N, 15069) = N singular values.
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)            # Vt: (K, 15069)

    # Sample eigenvalues (variance along each principal component).
    lambdas = (s ** 2) / (N - 1)                                   # (K,)

    # Eigenvalue floor: regularise_cov × mean(λ) added to all eigenvalues.
    # Small regularise_cov preserves the PCA covariance structure.
    # Large regularise_cov degrades to scaled Euclidean distance (safest with
    # very small N).
    floor       = regularise_cov * float(lambdas.mean())
    lambdas_reg = lambdas + floor                                   # (K,)

    # ------------------------------------------------------------------ #
    # Test-time: single forward pass + global PCA Mahalanobis.           #
    # ------------------------------------------------------------------ #
    v_test     = np.asarray(model.get_vertices(image), dtype=np.float64)
    v_test     = _procrustes_align_rigid(v_test, mu)               # align to GPA frame
    delta_flat = v_test.ravel() - mu_flat                          # (15069,)

    # Project deviation onto the PCA subspace, then normalise by eigenvalue σ.
    proj      = Vt @ delta_flat                                     # (K,) PCA coordinates
    proj_norm = proj / np.sqrt(lambdas_reg)                        # (K,) Mahalanobis components

    # Attribute the global distance back to individual vertices:
    #   vertex_delta[i] = Σ_k  Vt[k, i*3:(i+1)*3] * proj_norm[k]
    #   uncertainty_i   = ‖vertex_delta[i]‖₂
    # Property: Σ_i ‖vertex_delta[i]‖² = ‖proj_norm‖² = d² (global Mahal²).
    Vt_3d        = Vt.reshape(-1, 5023, 3)                         # (K, 5023, 3)
    vertex_delta = np.einsum('kid,k->id', Vt_3d, proj_norm)        # (5023, 3)
    maha         = np.linalg.norm(vertex_delta, axis=-1)            # (5023,)

    return maha[:, np.newaxis].astype(np.float32)                   # (5023, 1)


# ===========================================================================
# 6. Stable Output Layers MC Dropout (SOL-MCD) Uncertainty
# ===========================================================================

def calculate_sol_mcd_uncertainty(
    smirk_mcd_model,
    image: np.ndarray,
    n_passes: int = 30,
    n_stable_layers: int = 1,
    keep_spatial_dims: bool = False,
) -> np.ndarray:
    """
    An improved variant of MC Dropout that selectively disables dropout in the
    final layer(s) of the network, yielding sharper and better-calibrated
    epistemic uncertainty estimates on the *same* trained checkpoint [Son2025].

    Conceptual Background
    ---------------------
    Standard MC Dropout (Section 2) applies stochastic masking uniformly to
    every dropout layer in the network, including those immediately adjacent to
    the output.  Son & Seok (2025) identify this as a source of systematic
    miscalibration: dropout near the output introduces high-frequency noise
    in the prediction space that does not correspond to genuine model
    uncertainty.  The prediction fluctuates even for easy, in-distribution
    inputs because the output-adjacent dropout layer discards information that
    has already been compressed into low-dimensional geometric features.

    Their fix — Stable Output Layers (SOL) — is disarmingly simple: freeze
    the last N dropout layers to deterministic mode (i.e. set their
    .training flag to False, making them identity maps) while leaving all
    earlier dropout layers stochastic.  The deeper layers then still provide
    the approximate Bayesian posterior sampling that MC Dropout theorises,
    but the final prediction is computed deterministically from those stochastic
    intermediate representations, eliminating the spurious high-frequency
    variance at the output.

    Empirically, Son & Seok report that SOL-MCD matches or exceeds bootstrap
    ensembles in uncertainty quality (Spearman ρ and AUSE) while adding zero
    additional computational cost over standard MC Dropout.

    IMPORTANT: The same requirement as standard MCD applies — the checkpoint
    must have been trained with dropout layers in the expression encoder
    backbone.  No retraining, no new layers, and no calibration data are
    required: the only change is which dropout layers are active at inference.

    Difference from standard MCD (Section 2)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Standard MCD: ALL dropout layers stochastic.
    SOL-MCD:      FIRST n_stable_layers frozen (input-proximal, identity maps);
                  remaining output-proximal layer(s) left stochastic.

    NOTE: This implementation reverses the freeze direction relative to
    [Son2025], which freezes the LAST N layers.  For the SMIRK expression
    encoder, empirical results show that freezing the input-proximal layers
    and keeping the output-proximal layer(s) active yields better Spearman ρ —
    the final expression-encoder dropout layer produces the most expression-
    parameter-specific stochastic signal, and leaving it active concentrates
    variance in the region most coupled to the output.

    The result: stochastic component is concentrated in the layer(s) most
    directly coupled to the expression output, reducing noise from earlier
    feature-extraction layers.

    Algorithm
    ---------
    1. Capture one deterministic (eval-mode) base prediction for Procrustes
       alignment reference — before any dropout is activated.
    2. Identify all torch.nn.Dropout modules in model-traversal order.
    3. Set ALL dropout layers to train mode (.training = True, stochastic).
    4. Freeze the FIRST `n_stable_layers` of those modules back to eval mode
       (.training = False, identity map), keeping the output-proximal layers
       stochastic.
    5. Perform N stochastic forward passes.  Align each to the base prediction
       via Procrustes before collecting.
    6. Compute per-vertex empirical variance across N aligned predictions;
       normalise by ‖B_expr[i]‖²_F to remove the FLAME expression-basis
       sensitivity confound.
    7. Restore full eval mode.

    Parameters
    ----------
    smirk_mcd_model : SMIRK model trained with dropout
        Must expose a .model PyTorch attribute and a .get_vertices(image) method.
        Identical checkpoint requirement as `calculate_mcd_uncertainty`.
    image : np.ndarray, shape (H, W, 3)
    n_passes : int, default 30
        Number of stochastic forward passes.  SOL-MCD converges faster than
        standard MCD (less per-pass variance), so 20–30 passes typically give
        stable estimates vs. 30–50 for standard MCD.
    n_stable_layers : int, default 1
        Number of dropout layers (counting from the INPUT end, i.e. closest to
        the image) to freeze as deterministic identity maps.  1 (the default)
        freezes only the first dropout layer; the output-proximal layer(s) remain
        stochastic.  Increase to 2 to freeze the first two input-proximal layers,
        leaving only the final output-adjacent dropout active.
        NOTE: This is reversed relative to [Son2025] — see Algorithm section.
    keep_spatial_dims : bool, default False
        If True, return per-coordinate variance (5023, 3).
        If False, return scalar L2 norm of variance per vertex (5023, 1).

    Returns
    -------
    uncertainty : np.ndarray, shape (5023, 1) or (5023, 3)
        Per-vertex epistemic uncertainty.  Expected to be sharper (lower
        spatial entropy, higher Spearman ρ with actual error) than the output
        of `calculate_mcd_uncertainty` on the same checkpoint and image.

    References
    ----------
    [Son2025] S. Son & J. Seok, "Improving Monte Carlo Dropout Uncertainty
              Estimation with Stable Output Layers", Neural Networks, 2025.
    [Gal2016] Y. Gal & Z. Ghahramani, "Dropout as a Bayesian Approximation:
              Representing Model Uncertainty in Deep Learning", ICML 2016.
    """
    import torch

    pytorch_model = getattr(smirk_mcd_model, 'model', None)

    # Collect all Dropout modules in forward-traversal order.
    # named_modules() yields layers in the order they were registered, which
    # approximates the forward-pass order for typical sequential encoders.
    all_dropout: List["torch.nn.Module"] = []
    if pytorch_model is not None:
        for _, m in pytorch_model.named_modules():
            if isinstance(m, torch.nn.Dropout):
                all_dropout.append(m)

    # Determine which dropout layers to freeze (the last n_stable_layers).
    n_stable = min(n_stable_layers, len(all_dropout))
    # Freeze the FIRST n_stable dropout layers (closest to the input).
    # The original SOL-MCD paper freezes the LAST layers; in the SMIRK
    # expression encoder the last dropout layer (closest to the expression-
    # parameter head) is the most expression-specific.  Empirically, freezing
    # the last layer made anti-correlation WORSE because it removed the most
    # selective signal.  Freezing the FIRST layers instead leaves the final,
    # most output-proximal dropout active, producing a less uniformly
    # expression-basis-driven stochastic pattern.
    stable_set = set(id(m) for m in all_dropout[:n_stable]) if n_stable > 0 else set()

    # Capture deterministic base prediction before enabling any dropout.
    with torch.no_grad():
        v_base = np.asarray(smirk_mcd_model.get_vertices(image), dtype=np.float32)

    try:
        if pytorch_model is not None:
            # Activate only Dropout layers (same as MCD fix — keep BN in eval).
            # Then selectively freeze the output-adjacent (stable) layers.
            for m in all_dropout:
                m.training = True
            for m in all_dropout:
                if id(m) in stable_set:
                    m.training = False   # freeze: becomes identity map

        predictions: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(n_passes):
                vertices = smirk_mcd_model.get_vertices(image)    # (5023, 3)
                predictions.append(np.asarray(vertices, dtype=np.float32))
    finally:
        # Always restore full eval mode.
        if pytorch_model is not None:
            pytorch_model.eval()

    predictions_arr = np.stack(predictions, axis=0)               # (N, 5023, 3)

    # ── Procrustes alignment ─────────────────────────────────────────────────
    # Align to the fixed eval-mode base prediction (same fix as MCD) so that
    # alignment quality is N-independent and ρ converges monotonically.
    aligned = np.stack(
        [_procrustes_align(p, v_base) for p in predictions_arr],
        axis=0,
    )                                                              # (N, 5023, 3)

    per_coord_var = np.var(aligned, axis=0).astype(np.float32)    # (5023, 3)

    # ── Expression-basis normalisation ───────────────────────────────────────
    # One power of basis_norms (‖B‖_F, not ‖B‖²_F) — see MCD function comment.
    basis_norms = _get_expression_basis_norms(smirk_mcd_model)    # (5023,) or None
    if basis_norms is not None:
        floor = max(float(np.percentile(basis_norms, 5)), 1e-8)
        per_coord_var = per_coord_var / (basis_norms[:, np.newaxis] + floor)

    if keep_spatial_dims:
        return per_coord_var                                       # (5023, 3)

    # L2 norm of per-coordinate (normalised) std.
    return np.sqrt(per_coord_var.sum(axis=-1, keepdims=True))     # (5023, 1)


# ===========================================================================
# 7. Antithetic MC Dropout (A-MCD) Uncertainty
# ===========================================================================

def calculate_antithetic_mcd_uncertainty(
    smirk_mcd_model,
    image: np.ndarray,
    n_pairs: int = 15,
    keep_spatial_dims: bool = False,
) -> np.ndarray:
    """
    Estimate per-vertex epistemic uncertainty via Antithetic MC Dropout
    [Gal2016, Hammersley1956] — a variance-reduction enhancement to standard
    MC Dropout that replaces independent random mask sampling with paired
    antithetic samples.  Requires no retraining and no training-time
    information beyond the model checkpoint.

    Conceptual Background
    ---------------------
    Standard MC Dropout (Section 2) draws N independent dropout masks and
    averages predictions.  The variance of that estimator decreases at the
    Monte Carlo rate O(1/N) — every forward pass is statistically independent,
    so none informs the next.

    Antithetic variates [Hammersley1956] replace independence with deliberate
    negative correlation.  For each pair k = 1 … K:

        Regular pass:     u_j ~ Uniform(0,1),   keep neuron j iff u_j > p
        Antithetic pass:  same u_j, flipped —   keep neuron j iff (1−u_j) > p

    Both passes have the same expected keep rate (1 − p), so the estimator
    remains unbiased.  Because Cov(V^+_k, V^−_k) < 0, the pair mean has
    strictly lower variance than two independent passes [Owen2013, Ch. 8]:

        Var[(V^+_k + V^−_k) / 2] = (Var(V^+) + Var(V^−) + 2·Cov) / 4
                                   < Var(V^+) / 2

    Variance reduction is largest at p = 0.5, where the antithetic mask is
    the exact bitwise complement (~25–50 % reduction per pair).  For p ≠ 0.5
    the reduction is smaller but still strictly positive for all p ∈ (0, 1).

    Total forward passes: 2K — identical to standard MCD with N = 2K, but
    yielding meaningfully lower estimator variance, i.e. more stable per-
    vertex uncertainty maps for the same compute budget.

    IMPORTANT: Same checkpoint requirement as `calculate_mcd_uncertainty`.
    The model must have been *trained* with dropout layers present.  Enabling
    dropout post-hoc on a checkpoint trained without it produces random noise,
    not meaningful epistemic uncertainty.

    Hook Design
    -----------
    PyTorch's built-in nn.Dropout does not expose its internal uniform samples,
    so the antithetic mask cannot be constructed by intercepting PyTorch's own
    masking.  Instead, this function registers forward hooks on every
    nn.Dropout module that bypass PyTorch's masking entirely and apply a
    manually constructed antithetic mask to the layer's output:

      • The model is placed in **eval mode** so BatchNorm/LayerNorm layers use
        their running statistics (preventing batch-stat noise from contaminating
        the uncertainty signal).  In eval mode nn.Dropout is a pass-through
        (identity), so each hook receives the unmasked activation as `output`.
      • Regular-pass hook: samples u ~ Uniform(0,1), stores it in a per-layer
        cache keyed by id(module), and returns x · 1[u > p] / (1 − p).
      • Antithetic-pass hook for the same layer: loads the cached u, applies
        the complemented threshold, and returns x · 1[(1−u) > p] / (1 − p).
      • A single shared mutable `state` dict (holding the current mode and the
        u-cache) is captured by all hook closures, so switching between regular
        and antithetic is a single dict write before each forward call.
      • All hook handles are removed unconditionally in a `finally` block.

    Algorithm
    ---------
    1. Set the model to eval mode (BN/LN statistics frozen).
    2. Collect all nn.Dropout modules in forward-traversal order.
    3. Register one forward hook per module; hooks share a mutable state dict.
    4. For k = 1 … K:
         a. state['mode'] = 'regular'   → forward pass → V_k^+  (5023, 3)
            (hooks sample fresh u, cache it, apply 1[u > p] / (1−p))
         b. state['mode'] = 'antithetic' → forward pass → V_k^- (5023, 3)
            (hooks read cached u, apply 1[(1−u) > p] / (1−p))
    5. Remove all hooks (finally block).
    6. Stack 2K predictions → (2K, 5023, 3).
    7. Per-coordinate empirical variance across 2K samples → (5023, 3).
    8. Collapse to scalar per vertex → (5023, 1).

    Parameters
    ----------
    smirk_mcd_model : SMIRK model trained with dropout
        Must expose a .model PyTorch attribute and a .get_vertices(image)
        method.  Identical checkpoint requirement as `calculate_mcd_uncertainty`.
    image : np.ndarray, shape (H, W, 3)
    n_pairs : int, default 15
        Number of antithetic pairs K.  Total forward passes = 2K.  Because
        antithetic pairing yields roughly 25–50 % lower estimator variance
        than independent sampling [Hammersley1956], K = 15 (30 total passes)
        typically matches the stability of N = 40–50 standard MCD passes.
    keep_spatial_dims : bool, default False
        If True, return per-coordinate variance (5023, 3).
        If False, return scalar sum of per-coordinate variances (5023, 1),
        equal to the trace of the per-vertex empirical covariance.

    Returns
    ------- 
    uncertainty : np.ndarray, shape (5023, 1) or (5023, 3), dtype float32
        Per-vertex epistemic uncertainty.  Lower estimator variance than
        `calculate_mcd_uncertainty` for the same total number of forward
        passes; the improvement is most visible on high-dropout-rate layers
        and on vertices near the facial boundary where dropout variance is
        largest (ears, neck, hairline).

    Raises
    ------
    ValueError
        If smirk_mcd_model has no .model attribute, or if the model contains
        no nn.Dropout layers.  Use `calculate_stochastic_depth_uncertainty`
        for DropPath / StochasticDepth architectures.

    References
    ----------
    [Gal2016]        Y. Gal & Z. Ghahramani, "Dropout as a Bayesian
                     Approximation: Representing Model Uncertainty in Deep
                     Learning", ICML 2016.
    [Hammersley1956] J. M. Hammersley & K. W. Morton, "A New Monte Carlo
                     Technique: Antithetic Variates", Mathematical Proceedings
                     of the Cambridge Philosophical Society, vol. 52, no. 3,
                     pp. 449–475, 1956.
    [Owen2013]       A. B. Owen, Monte Carlo Theory, Methods and Examples,
                     Ch. 8: Variance Reduction, 2013.
    [Son2025]        S. Son & J. Seok, "Improving Monte Carlo Dropout
                     Uncertainty Estimation with Stable Output Layers",
                     Neural Networks, 2025.
    """
    import torch

    pytorch_model = getattr(smirk_mcd_model, 'model', None)
    if pytorch_model is None:
        raise ValueError(
            "smirk_mcd_model must expose a .model PyTorch attribute "
            "containing the underlying nn.Module."
        )

    # Collect all nn.Dropout modules in forward-traversal order.
    # named_modules() yields layers in registration order, which approximates
    # the forward-pass order for typical sequential encoders.
    dropout_modules = [
        m for _, m in pytorch_model.named_modules()
        if isinstance(m, torch.nn.Dropout)
    ]
    if not dropout_modules:
        raise ValueError(
            "No nn.Dropout layers found in the model. "
            "If the model uses DropPath / StochasticDepth, use "
            "calculate_stochastic_depth_uncertainty instead."
        )

    # Shared mutable state captured by every hook closure.
    #   'mode'     — 'regular' or 'antithetic', set before each forward call.
    #   'cached_u' — maps id(module) → the uniform tensor u from the last
    #                regular pass through that module.
    state = {'mode': 'regular', 'cached_u': {}}

    def _make_hook(layer_id: int, p: float):
        keep_prob = 1.0 - p

        def hook(module, input, output):
            # In eval mode nn.Dropout is identity, so output == input[0].
            # We intercept and replace with our custom antithetic mask.
            x = output
            if keep_prob < 1e-8:
                # p ≈ 1.0: drop everything in both passes.
                return torch.zeros_like(x)
            if state['mode'] == 'regular':
                # Sample fresh uniform values, cache for the antithetic pass.
                u = torch.empty_like(x).uniform_()
                state['cached_u'][layer_id] = u
            else:
                # Antithetic pass: flip the cached u so Cov(V^+, V^-) < 0.
                u = 1.0 - state['cached_u'][layer_id]
            # Inverted-dropout scaling keeps E[masked output] = E[input].
            mask = (u > p).to(x.dtype) / keep_prob                  # (*, d)
            return x * mask

        return hook

    # Capture deterministic base prediction before hooks introduce stochasticity.
    # Must run in eval mode (no hooks yet) so the forward pass is fully
    # deterministic and serves as a stable Procrustes alignment target.
    pytorch_model.eval()
    with torch.no_grad():
        v_base = np.asarray(smirk_mcd_model.get_vertices(image), dtype=np.float32)

    handles = [
        m.register_forward_hook(_make_hook(id(m), float(m.p)))
        for m in dropout_modules
    ]

    try:
        # eval mode: BN/LN use running statistics; our hooks inject dropout
        # stochasticity independently of the model's .training flag.
        pytorch_model.eval()

        predictions: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(n_pairs):
                state['mode'] = 'regular'
                v_reg = smirk_mcd_model.get_vertices(image)          # (5023, 3)
                predictions.append(np.asarray(v_reg, dtype=np.float32))

                state['mode'] = 'antithetic'
                v_anti = smirk_mcd_model.get_vertices(image)         # (5023, 3)
                predictions.append(np.asarray(v_anti, dtype=np.float32))
    finally:
        for h in handles:
            h.remove()
        pytorch_model.eval()                                         # defensive restore

    predictions_arr = np.stack(predictions, axis=0)                  # (2K, 5023, 3)

    # ── Procrustes alignment ─────────────────────────────────────────────────
    # Align each antithetic-pass prediction to the fixed eval-mode base
    # (captured before hooks were registered).  Fixed reference = alignment
    # quality is independent of the number of pairs, so ρ converges
    # monotonically with n_pairs instead of exhibiting the non-monotone
    # V-shape caused by the dynamic mean target.
    aligned = np.stack(
        [_procrustes_align(p, v_base) for p in predictions_arr],
        axis=0,
    )                                                                # (2K, 5023, 3)

    # Per-coordinate empirical variance across all 2K aligned predictions.
    per_coord_var = np.var(aligned, axis=0).astype(np.float32)      # (5023, 3)

    # ── Expression-basis normalisation ───────────────────────────────────────
    # One power of basis_norms (‖B‖_F, not ‖B‖²_F) — see MCD function comment.
    basis_norms = _get_expression_basis_norms(smirk_mcd_model)      # (5023,) or None
    if basis_norms is not None:
        floor = max(float(np.percentile(basis_norms, 5)), 1e-8)
        per_coord_var = per_coord_var / (basis_norms[:, np.newaxis] + floor)

    if keep_spatial_dims:
        return per_coord_var                                         # (5023, 3)

    # L2 norm of per-coordinate (normalised) std.
    return np.sqrt(per_coord_var.sum(axis=-1, keepdims=True))        # (5023, 1)


# ===========================================================================
# Ablation helpers
# ===========================================================================

def calculate_tta_uncertainty_n_ablation(
    wrapper,
    image: np.ndarray,
    n_values: Optional[List[int]] = None,
    **tta_kwargs,
) -> Dict[int, np.ndarray]:
    """
    Run TTA uncertainty estimation for each value of N in ``n_values`` and
    return the resulting per-vertex uncertainty map for each N.

    This is the ablation utility used to produce the "AUSE vs N" convergence
    plot.  Running TTA for increasing N shows whether the variance estimate has
    stabilised — if AUSE is still falling at N=20 the default N=10 is too low.

    The same augmentation pipeline as ``calculate_tta_uncertainty`` is used for
    every N, so results are directly comparable.

    Parameters
    ----------
    wrapper : BaseFaceRegressorWrapper
        Any single-model wrapper that implements ``get_vertices``.
    image : np.ndarray, shape (H, W, 3), uint8
        Input face image.
    n_values : list of int, default [2, 5, 10, 15, 20]
        TTA pass counts to evaluate.  Sorted ascending so each run re-uses
        the augmentation samples from all smaller N values implicitly
        (each call is independent — no caching — but the list is small).
    **tta_kwargs
        Extra keyword arguments forwarded to ``calculate_tta_uncertainty``
        (e.g., ``aggregate='variance'``).

    Returns
    -------
    results : dict[int → np.ndarray, shape (5023,)]
        Per-vertex uncertainty (flattened to 1-D) for each N in n_values.
        The caller should then compute AUSE for each entry against a ground-
        truth error array to produce the AUSE-vs-N plot.

    Example
    -------
    ::
        unc_by_n = calculate_tta_uncertainty_n_ablation(wrapper, image)
        for n, u in unc_by_n.items():
            sparse = calculate_sparsification_error_curve(error, u)
            print(f"N={n:2d}  AUSE={sparse['ause']:.4f}")
    """
    if n_values is None:
        n_values = [2, 5, 10, 15, 20]

    results: Dict[int, np.ndarray] = {}
    for n in sorted(n_values):
        u = calculate_tta_uncertainty(wrapper, image, n_passes=n, **tta_kwargs)
        results[n] = np.asarray(u, dtype=np.float32).ravel()   # (5023,)
    return results
