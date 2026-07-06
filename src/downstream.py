"""
downstream.py
=============
Downstream facial expression classifier: projects per-vertex FLAME uncertainty
into image space and fuses it into a Vision Transformer / ResNet / GCN
classifier to test whether geometric uncertainty carries task-relevant signal
for 2D facial expression recognition.

Key building blocks
--------------------
  project_variance_to_2d         — weak-perspective splat of (5023, 1) vertex
                                    uncertainty onto an (H, W) confidence map.
  UncertaintyWeightedClassifier   — the classifier itself; supports a CNN/ViT
                                    path (five uncertainty fusion points, see
                                    the class docstring) and a GCN path that
                                    operates directly on FLAME vertices.
  build_flame_adjacency          — normalised graph adjacency for the GCN path.
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
try:
    import torchvision.models as _tv_models
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

EXPRESSION_CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ViT backbone names supported by torchvision (≥0.12).
# Tier guidance:
#   vit_b_32 — Base model, 32-pixel patches → fewest sequence tokens → fastest; CPU-friendly.
#   vit_b_16 — Base model, 16-pixel patches → 4× more tokens; balanced (GPU recommended).
#   vit_l_16 — Large model, 16-pixel patches → highest capacity; GPU required.
#   vit_h_14 — Huge model, 14-pixel patches → HPC only.
_VIT_BACKBONES = frozenset({'vit_b_32', 'vit_b_16', 'vit_l_16', 'vit_l_32', 'vit_h_14'})

# ViT-internal uncertainty fusion modes: inject after the pixel stage, inside the transformer.
#   patch_embed — scale patch tokens produced by conv_proj (before positional embedding)
#   attn_bias   — add -α·U_j as a key-side bias to pre-softmax attention logits (requires PyTorch ≥ 2.0)
#   key_scale   — scale key projections so uncertain patches attract less attention
#   value_scale — scale value projections so uncertain patches contribute less to aggregated output
_VIT_INTERNAL_FUSION_MODES = frozenset({'patch_embed', 'attn_bias', 'key_scale', 'value_scale'})

_ALL_FUSION_MODES = {'input', 'feature'} | _VIT_INTERNAL_FUSION_MODES


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def project_variance_to_2d(
    vertices: np.ndarray,
    variance: np.ndarray,
    image_size: Tuple[int, int],
    camera_params: Optional[Dict] = None,
    splat_sigma: float = 4.0,
) -> np.ndarray:
    """
    Weak-perspective projection of per-vertex 3D uncertainty onto a 2D confidence map.

    Uses the SMIRK/DECA/EMOCA weak-perspective model:
        x_norm = s * (R @ v)[0] + tx
        y_norm = s * (R @ v)[1] + ty
    where normalised coordinates are in [-1, 1] (origin = image centre, y up).
    Converts to pixel coordinates, scatters variance values, applies Gaussian
    smoothing to fill gaps, then inverts to produce a *confidence* map where
    high uncertainty → low confidence.

    camera_params keys (all optional, safe defaults shown):
        'scale' : float    weak-perspective scale s            (default 1.0)
        'tx'    : float    x translation in normalised [-1,1]  (default 0.0)
        'ty'    : float    y translation in normalised [-1,1]  (default 0.0)
        'R'     : (3, 3)   camera rotation matrix              (default I_3)

    Parameters
    ----------
    vertices   : (5023, 3) FLAME vertex positions
    variance   : (5023, 1) or (5023,) per-vertex scalar uncertainty
    image_size : (H, W) pixel resolution of the target confidence map
    camera_params : dict or None
    splat_sigma : float
        Gaussian smoothing radius in pixels used to fill inter-vertex gaps.

    Returns
    -------
    confidence_map : np.ndarray, shape (H, W), float32, values in [0, 1]
        1 = confident (low uncertainty), 0 = maximally uncertain.
        Pixels with no vertex projection default to 1 (confident).
    """
    from scipy.ndimage import gaussian_filter

    H, W = image_size
    verts = np.asarray(vertices, dtype=np.float64)   # (5023, 3)
    var   = np.asarray(variance, dtype=np.float64).ravel()  # (5023,)

    if camera_params is not None:
        R     = np.asarray(camera_params.get('R', np.eye(3)), dtype=np.float64)
        scale = float(camera_params.get('scale', 1.0))
        tx    = float(camera_params.get('tx', 0.0))
        ty    = float(camera_params.get('ty', 0.0))
        verts_cam = (R @ verts.T).T                      # (5023, 3)
        x_norm = scale * verts_cam[:, 0] + tx            # ∈ [-1, 1] if camera is valid
        y_norm = scale * verts_cam[:, 1] + ty
    else:
        # Auto-fit: centre and normalise the XY extent so the face fills
        # the image regardless of each regressor's native vertex scale.
        # FLAME vertices span ≈ ±0.15 in metres; without this the face
        # would be projected into a tiny central strip (≈15 % of image width),
        # leaving the rest of the map at confidence=1 and the face region
        # near-zero — producing black boxes on the face.
        v_xy     = verts[:, :2]                          # (N, 2)
        v_center = v_xy.mean(0)                          # (2,)
        v_extent = np.abs(v_xy - v_center).max()         # half-range scalar
        if v_extent < 1e-8:
            v_extent = 1.0
        x_norm = (verts[:, 0] - v_center[0]) / v_extent  # ∈ [-1, 1]
        y_norm = (verts[:, 1] - v_center[1]) / v_extent  # ∈ [-1, 1], y-up

    # Convert to pixel coordinates: origin top-left, y-down
    px = np.round((x_norm + 1.0) * 0.5 * (W - 1)).astype(int)
    py = np.round((1.0 - (y_norm + 1.0) * 0.5) * (H - 1)).astype(int)

    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)

    unc_map = np.zeros((H, W), dtype=np.float64)
    weight  = np.zeros((H, W), dtype=np.float64)
    np.add.at(unc_map, (py[valid], px[valid]), var[valid])
    np.add.at(weight,  (py[valid], px[valid]), 1.0)

    unc_map = gaussian_filter(unc_map, sigma=splat_sigma)
    weight  = gaussian_filter(weight,  sigma=splat_sigma)

    with np.errstate(divide='ignore', invalid='ignore'):
        unc_map = np.where(weight > 1e-8, unc_map / weight, 0.0)

    unc_max = unc_map.max()
    if unc_max > 1e-12:
        unc_map /= unc_max

    # Confidence = 1 − normalised uncertainty.
    # Apply a minimum floor so even maximally-uncertain vertices are never
    # fully zeroed out (no black boxes).  Background pixels (outside the
    # vertex projection) have unc_map=0, so they stay at confidence=1.0.
    _MIN_CONF = 0.5
    confidence = _MIN_CONF + (1.0 - _MIN_CONF) * (1.0 - unc_map)
    return confidence.astype(np.float32)


def _compute_patch_uncertainty_batch(
    attention_map: torch.Tensor,
    patch_size: int,
    img_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a batch of 2-D confidence maps into patch-level uncertainty tensors.

    Parameters
    ----------
    attention_map : (B, 1, H, W) float in [0, 1], where 1 = fully confident.
    patch_size    : ViT patch size in pixels (e.g. 16 for ViT-B/16).
    img_size      : square side length the ViT was trained on (e.g. 224).

    Returns
    -------
    u_patch_2d : (B, n_h, n_h) per-patch uncertainty,  n_h = img_size // patch_size
    u_padded   : (B, n_h*n_h + 1) flat patches with a leading CLS-token zero
    """
    n_h = img_size // patch_size
    u_2d = 1.0 - attention_map[:, 0:1]                           # (B, 1, H, W)
    u_grid = F.interpolate(
        u_2d, size=(n_h, n_h), mode='bilinear', align_corners=False,
    ).squeeze(1)                                                   # (B, n_h, n_h)
    u_flat = u_grid.reshape(u_grid.shape[0], -1)                  # (B, n_patches)
    cls_z  = torch.zeros(u_flat.shape[0], 1,
                         device=u_flat.device, dtype=u_flat.dtype)
    u_padded = torch.cat([cls_z, u_flat], dim=1)                  # (B, n_patches+1)
    return u_grid, u_padded


def _build_uncertainty_hooks(
    backbone: nn.Module,
    u_patch_2d: torch.Tensor,   # (B, n_h, n_h) — used by patch_embed
    u_padded: torch.Tensor,     # (B, N)  N = n_patches+1, index 0 = CLS (zero)
    alpha: float,
    fusion_mode: str,
) -> List:
    """
    Register forward hooks on *backbone* for ViT-internal uncertainty injection.

    All four modes keep the input image intact and modulate internal representations:

    patch_embed  — scale the (B, D, n_h, n_h) conv_proj output; uncertain patches
                   enter the encoder with lower-magnitude token vectors.

    key_scale    — scale key projections in every self-attention layer so uncertain
                   patches produce smaller dot-products with all queries, reducing
                   how strongly they are attended to.

    value_scale  — scale value projections so uncertain patches contribute less to
                   the aggregated attention output even when heavily attended.

    attn_bias    — subtract α·U_j from the (B*H, tgt, src) pre-softmax attention
                   logits for each key position j.  Requires PyTorch ≥ 2.0
                   (uses register_forward_pre_hook with with_kwargs=True).

    Returns a list of RemovableHook handles.  The caller is responsible for
    calling handle.remove() in a finally block after the forward pass.
    """
    handles: List = []

    if fusion_mode == 'patch_embed':
        scale_2d = (1.0 - alpha * u_patch_2d).clamp(0.0, 1.0)   # (B, n_h, n_h)

        def _pe_hook(module, inp, output, _s=scale_2d):
            # output: (B, D, n_h, n_h) — broadcast scale over D channel
            return output * _s.to(output.device).unsqueeze(1)

        handles.append(backbone.conv_proj.register_forward_hook(_pe_hook))

    elif fusion_mode == 'key_scale':
        scale = (1.0 - alpha * u_padded).clamp(0.0, 1.0)         # (B, N)

        def _make_key_hook(s):
            def _hook(module, args):
                q, k, v = args[0], args[1], args[2]
                # k: (B, N, D) — scale each token independently
                k = k * s.to(k.device).unsqueeze(-1)
                return (q, k, v) + args[3:]
            return _hook

        for layer in backbone.encoder.layers:
            handles.append(
                layer.self_attention.register_forward_pre_hook(_make_key_hook(scale))
            )

    elif fusion_mode == 'value_scale':
        scale = (1.0 - alpha * u_padded).clamp(0.0, 1.0)         # (B, N)

        def _make_val_hook(s):
            def _hook(module, args):
                q, k, v = args[0], args[1], args[2]
                v = v * s.to(v.device).unsqueeze(-1)
                return (q, k, v) + args[3:]
            return _hook

        for layer in backbone.encoder.layers:
            handles.append(
                layer.self_attention.register_forward_pre_hook(_make_val_hook(scale))
            )

    elif fusion_mode == 'attn_bias':
        def _make_bias_hook(u_b, a):
            def _hook(module, args, kwargs):
                B_cur  = args[0].shape[0]
                N_cur  = args[0].shape[1]
                n_head = module.num_heads
                bias   = -a * u_b.to(args[0].device)              # (B, N)
                # Per key-position bias broadcast over all query positions → (B, N, N)
                bias_kj = bias.unsqueeze(1).expand(-1, N_cur, -1)
                # Expand over heads: (B*heads, N, N)
                attn_mask = (
                    bias_kj.unsqueeze(1)
                    .expand(-1, n_head, -1, -1)
                    .reshape(B_cur * n_head, N_cur, N_cur)
                )
                kwargs['attn_mask'] = attn_mask
                return args, kwargs
            return _hook

        for layer in backbone.encoder.layers:
            handles.append(
                layer.self_attention.register_forward_pre_hook(
                    _make_bias_hook(u_padded, alpha), with_kwargs=True
                )
            )

    return handles


def build_flame_adjacency(faces: np.ndarray, n_vertices: int = 5023) -> torch.Tensor:
    """
    Build the symmetric normalised adjacency matrix A_hat for the FLAME mesh graph.

    A_hat = D^{-1/2} (A + I) D^{-1/2}

    Used by UncertaintyWeightedClassifier when architecture_type='GCN'.

    Parameters
    ----------
    faces      : (F, 3) int — FLAME triangle face indices
    n_vertices : int        — 5023 for FLAME

    Returns
    -------
    torch.sparse_coo_tensor, shape (n_vertices, n_vertices), float32
    """
    f = np.asarray(faces, dtype=np.int64)

    # All directed edges from every triangle + self-loops
    edges = np.concatenate([
        f[:, [0, 1]], f[:, [1, 0]],
        f[:, [1, 2]], f[:, [2, 1]],
        f[:, [0, 2]], f[:, [2, 0]],
        np.stack([np.arange(n_vertices, dtype=np.int64)] * 2, axis=1),
    ], axis=0)

    rows = edges[:, 0]
    cols = edges[:, 1]

    degree = np.bincount(rows, minlength=n_vertices).astype(np.float32)
    d_inv  = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    vals   = (d_inv[rows] * d_inv[cols]).astype(np.float32)

    idx = torch.from_numpy(np.stack([rows, cols], axis=0))
    val = torch.from_numpy(vals)
    return torch.sparse_coo_tensor(idx, val, (n_vertices, n_vertices)).coalesce()


# ---------------------------------------------------------------------------
# GCN layer (no external dependency)
# ---------------------------------------------------------------------------

class _GCNLayer(nn.Module):
    """H' = ReLU(A_hat H W)  — single graph convolutional layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(torch.sparse.mm(adj, x)))


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class UncertaintyWeightedClassifier(nn.Module):
    """
    Facial expression classifier that integrates spatial uncertainty to suppress
    hallucinated features in occluded facial regions.

    Two architecture modes
    ----------------------
    'CNN'  — ViT or ResNet backbone.

      ViT backbones (recommended; torchvision ≥ 0.12):
        'vit_b_32'  32-pixel patches, fewest tokens → fastest; CPU-friendly.
        'vit_b_16'  16-pixel patches; balanced (GPU recommended).
        'vit_l_16'  Large model; GPU required.
        'vit_h_14'  Huge model; HPC only.
      ResNet backbones (legacy):
        'resnet18', 'resnet50', …

      fusion_mode controls how the 2-D uncertainty map is injected:
        'input'       — multiply image by (1 − α·U_2D) before backbone  [default]
        'feature'     — multiply ResNet spatial feature map (ViT falls back to 'input')
        'patch_embed' — scale conv_proj patch tokens (ViT only)
        'key_scale'   — scale MHA key projections in every encoder layer (ViT only)
        'value_scale' — scale MHA value projections in every encoder layer (ViT only)
        'attn_bias'   — subtract α·U_j from pre-softmax attention logits (ViT only, PyTorch ≥ 2.0)

    'GCN'  — 2-layer Graph Convolutional Network operating directly on FLAME
      vertex positions.  Per-vertex uncertainty is appended as a 4th node
      feature [x, y, z, σ].  Requires a pre-built FLAME adjacency matrix
      (see build_flame_adjacency).

    Parameters
    ----------
    num_classes       : int   — 7 (RAF-DB / AffectNet convention) or custom
    architecture_type : str   — 'CNN' or 'GCN'
    backbone          : str   — torchvision model name
    pretrained        : bool  — load ImageNet weights for the CNN backbone
    fusion_mode       : str   — see table above; default 'input'
    fusion_alpha      : float — suppression strength α for all internal modes (default 1.0)
    gcn_hidden        : int   — hidden dim for each GCN layer  (GCN path only)
    head_dropout      : float — dropout probability in the classification head (default 0.5)
    head_arch         : str   — 'mlp' (LayerNorm→Linear(D,128)→GELU→Dropout→Linear(128,C))
                                or 'linear' (LayerNorm→Dropout→Linear(D,C))
                                (CNN/ViT path only; default 'mlp')
    """

    def __init__(
        self,
        num_classes: int = 7,
        architecture_type: str = 'CNN',
        backbone: str = 'vit_b_32',
        pretrained: bool = True,
        fusion_mode: str = 'input',
        fusion_alpha: float = 1.0,
        gcn_hidden: int = 128,
        head_dropout: float = 0.5,
        head_arch: str = 'mlp',
    ):
        super().__init__()
        self.architecture_type = architecture_type.upper()
        self.fusion_mode = fusion_mode
        self.fusion_alpha = float(fusion_alpha)
        self.num_classes = num_classes

        if fusion_mode not in _ALL_FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {sorted(_ALL_FUSION_MODES)}, "
                f"got '{fusion_mode}'."
            )

        if self.architecture_type == 'CNN':
            if not _TORCHVISION_AVAILABLE:
                raise ImportError("torchvision is required for architecture_type='CNN'.")

            if backbone in _VIT_BACKBONES:
                # ── ViT backbone ─────────────────────────────────────────────
                # ViT outputs the CLS-token representation (B, hidden_dim);
                # there is no spatial feature map, so 'feature' fusion is
                # not applicable.
                self._backbone_type = 'vit'
                if fusion_mode == 'feature':
                    warnings.warn(
                        "fusion_mode='feature' is not supported for ViT backbones "
                        "(no spatial feature map after encoder). Falling back to "
                        "fusion_mode='input'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.fusion_mode = 'input'

                try:
                    # vit_h_14 has only SWAG weights (no IMAGENET1K_V1); all others use V1.
                    if not pretrained:
                        weights = None
                    elif backbone == 'vit_h_14':
                        weights = 'DEFAULT'
                    else:
                        weights = 'IMAGENET1K_V1'
                    vit_model = getattr(_tv_models, backbone)(weights=weights)
                except TypeError:
                    vit_model = getattr(_tv_models, backbone)(pretrained=pretrained)

                # Determine feature dimension.
                feature_dim = getattr(vit_model, 'hidden_dim', None)
                if feature_dim is None:
                    head = vit_model.heads.head
                    if isinstance(head, nn.Linear):
                        feature_dim = head.in_features
                    else:
                        raise ValueError(
                            f"Cannot determine feature_dim for ViT backbone '{backbone}'."
                        )

                # Strip the classification head; forward() now returns (B, feature_dim).
                vit_model.heads.head = nn.Identity()
                self.backbone = vit_model
                self.pool = None   # not used for ViT

                if head_arch == 'linear':
                    self.classifier = nn.Sequential(
                        nn.LayerNorm(feature_dim),
                        nn.Dropout(head_dropout),
                        nn.Linear(feature_dim, num_classes),
                    )
                else:
                    self.classifier = nn.Sequential(
                        nn.LayerNorm(feature_dim),
                        nn.Linear(feature_dim, 128),
                        nn.GELU(),
                        nn.Dropout(head_dropout),
                        nn.Linear(128, num_classes),
                    )

            else:
                # ── ResNet backbone (legacy) ──────────────────────────────────
                self._backbone_type = 'resnet'
                if fusion_mode in _VIT_INTERNAL_FUSION_MODES:
                    warnings.warn(
                        f"fusion_mode='{fusion_mode}' is a ViT-internal method and is not "
                        f"supported for ResNet backbones. Falling back to 'input'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.fusion_mode = 'input'
                try:
                    weights = 'DEFAULT' if pretrained else None
                    resnet = getattr(_tv_models, backbone)(weights=weights)  # ResNet only has DEFAULT/IMAGENET1K_V1
                except TypeError:
                    resnet = getattr(_tv_models, backbone)(pretrained=pretrained)
                feature_dim = resnet.fc.in_features
                self.backbone = nn.Sequential(*list(resnet.children())[:-2])
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(head_dropout),
                    nn.Linear(feature_dim, num_classes),
                )

        elif self.architecture_type == 'GCN':
            self.gcn1 = _GCNLayer(4, gcn_hidden)
            self.gcn2 = _GCNLayer(gcn_hidden, gcn_hidden)
            self.classifier = nn.Sequential(
                nn.Linear(gcn_hidden, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )

        else:
            raise ValueError(
                f"architecture_type must be 'CNN' or 'GCN', got '{architecture_type}'."
            )

    def project_variance_to_2d(
        self,
        vertices: np.ndarray,
        variance: np.ndarray,
        camera_params: Optional[Dict] = None,
        image_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """Convenience wrapper around the module-level project_variance_to_2d."""
        return project_variance_to_2d(vertices, variance, image_size, camera_params)

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        attention_map: Optional[torch.Tensor] = None,
        vertices: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        CNN path
        --------
        image        : (B, 3, H, W)   — input image, ImageNet-normalised
        attention_map: (B, 1, H', W') — confidence map from project_variance_to_2d,
                                         already as a float tensor in [0, 1].
                                         Pass None for the plain / baseline run.
                                         Resized automatically to match image or
                                         feature map resolution.

        GCN path
        --------
        vertices     : (B, 5023, 3)   — FLAME vertex positions per sample
        uncertainty  : (B, 5023, 1)   — per-vertex uncertainty scalars
        adj          : torch.sparse   — normalised adjacency (5023, 5023);
                                         build once via build_flame_adjacency(faces).

        Returns
        -------
        logits : (B, num_classes)
        """
        if self.architecture_type == 'CNN':
            return self._forward_cnn(image, attention_map)
        return self._forward_gcn(vertices, uncertainty, adj)

    def _forward_cnn(
        self,
        image: torch.Tensor,
        attention_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # ── ViT-internal injection (patch_embed / key_scale / value_scale / attn_bias) ──
        if (
            self.fusion_mode in _VIT_INTERNAL_FUSION_MODES
            and self._backbone_type == 'vit'
            and attention_map is not None
        ):
            # Resize confidence map to image resolution if needed.
            if attention_map.shape[-2:] != image.shape[-2:]:
                attention_map = F.interpolate(
                    attention_map, size=image.shape[-2:],
                    mode='bilinear', align_corners=False,
                )
            # Derive per-patch uncertainty from the confidence map.
            patch_size = getattr(
                self.backbone, 'patch_size',
                self.backbone.conv_proj.kernel_size[0],
            )
            img_size = getattr(self.backbone, 'image_size', image.shape[-1])
            u_patch_2d, u_padded = _compute_patch_uncertainty_batch(
                attention_map, patch_size, img_size,
            )
            handles = _build_uncertainty_hooks(
                self.backbone, u_patch_2d, u_padded, self.fusion_alpha, self.fusion_mode,
            )
            try:
                feat = self.backbone(image)       # (B, D)
            finally:
                for h in handles:
                    h.remove()
            return self.classifier(feat)

        # ── Input-level fusion: multiply image by confidence map before backbone ──
        if attention_map is not None and self.fusion_mode == 'input':
            if attention_map.shape[-2:] != image.shape[-2:]:
                attention_map = F.interpolate(
                    attention_map, size=image.shape[-2:],
                    mode='bilinear', align_corners=False,
                )
            image = image * attention_map

        if self._backbone_type == 'vit':
            feat = self.backbone(image)           # (B, D)
            return self.classifier(feat)

        # ── ResNet path ──────────────────────────────────────────────────────────
        feat = self.backbone(image)               # (B, C, h, w)

        if attention_map is not None and self.fusion_mode == 'feature':
            if attention_map.shape[-2:] != feat.shape[-2:]:
                attention_map = F.interpolate(
                    attention_map, size=feat.shape[-2:],
                    mode='bilinear', align_corners=False,
                )
            feat = feat * attention_map

        feat = self.pool(feat)                    # (B, C, 1, 1)
        return self.classifier(feat)              # Flatten → Dropout → Linear

    def _forward_gcn(
        self,
        vertices: torch.Tensor,
        uncertainty: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        B = vertices.shape[0]
        outs = []
        for b in range(B):
            x = torch.cat([vertices[b], uncertainty[b]], dim=-1)   # (N, 4)
            x = self.gcn1(x, adj)                                   # (N, hidden)
            x = self.gcn2(x, adj)                                   # (N, hidden)
            outs.append(x.mean(dim=0))                              # global mean pool
        return self.classifier(torch.stack(outs, dim=0))            # (B, num_classes)
