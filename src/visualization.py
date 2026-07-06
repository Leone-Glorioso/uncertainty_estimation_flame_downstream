"""
visualization.py
================
Uncertainty visualisation and analysis suite for FLAME 3D face reconstruction.

All rendering functions use only numpy and matplotlib (Agg backend) — no CUDA,
PyTorch3D, or OpenGL dependency; runs identically on CPU-only and GPU hosts.

Public API
----------
Mesh heatmaps
  render_uncertainty_heatmap         – 3-view mesh colour map for one scalar field
  plot_uncertainty_spatial_maps      – multi-method mesh panel (side-by-side)

Method comparison
  plot_uncertainty_comparison_violin – distribution violin / box-whisker across methods
  plot_method_comparison_table       – colour-coded metric table heatmap

Sparsification / AUSE
  plot_sparsification_curves         – method / oracle / random curves + bar inset

Calibration
  plot_calibration_diagram           – reliability diagram for multiple methods

Error–uncertainty scatter
  plot_uncertainty_vs_error_scatter  – per-vertex scatter with Spearman ρ annotated

Region-wise breakdown
  plot_region_error_breakdown        – grouped bar chart across 9 FLAME regions

Error distributions
  plot_error_distributions           – histogram + KDE of per-vertex errors

Multi-model mesh comparison
  plot_multi_model_reconstruction    – 4-panel front-view per FLAME regressor
  plot_paper_style_comparison_panel  – paper-style panel (dark BG, high DPI) matching published figures
  plot_method_correlation_matrix     – Spearman ρ matrix between all uncertainty methods

Master report
  create_full_analysis_report        – call all plots above, save to one directory
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless backend; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.tri as mtri
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ─────────────────────────────── constants ────────────────────────────────────

METHOD_COLORS: Dict[str, str] = {
    "TTA":          "#4C72B0",
    "MCD":          "#DD8452",
    "CrossMethod":  "#55A868",
    "Jacobian":     "#C44E52",
    "Mahalanobis":  "#8172B2",
    "SOL-MCD":      "#937860",
    "A-MCD":        "#DA8BC3",
    "Oracle":       "#111111",
    "Random":       "#AAAAAA",
}

# Display-name overrides for plot labels only — internal dict keys are unchanged.
_DISPLAY_NAMES: Dict[str, str] = {
    "Jacobian": "Jacobian (Sensitivity)",
}

def _dn(name: str) -> str:
    """Return the display-friendly label for a method name."""
    return _DISPLAY_NAMES.get(name, name)

_MESH_CMAP = 'plasma'
_ERR_CMAP  = 'RdYlGn_r'       # red = high error, green = low

# Metric direction: True = higher is better, False = lower is better, None = informational
_METRIC_BETTER: Dict[str, Optional[bool]] = {
    "spearman_rho":      True,
    "pearson_r":         True,
    "ause":              False,
    "ause_normalised":   False,
    "nll":               False,
    "sharpness_mean":    None,
    "sharpness_entropy": False,
    "sharpness_cv":      None,
}

_METRIC_LABELS: Dict[str, str] = {
    "spearman_rho":      "Spearman ρ ↑",
    "pearson_r":         "Pearson r ↑",
    "ause":              "AUSE ↓",
    "ause_normalised":   "AUSE (norm.) ↓",
    "nll":               "NLL ↓",
    "sharpness_mean":    "Sharpness (mean)",
    "sharpness_entropy": "Spatial Entropy ↓",
    "sharpness_cv":      "Sharpness CV",
}


# ─────────────────────────── private helpers ──────────────────────────────────

def _save(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save *fig* to *path* and close it; creates parent directories as needed."""
    if path:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _method_color(name: str) -> str:
    """Return a stable colour for *name*, falling back to a hash-derived hex."""
    return METHOD_COLORS.get(name, f"#{abs(hash(name)) & 0xFFFFFF:06x}")


def _project_vertices(
    vertices: np.ndarray, azim_deg: float, elev_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthographic camera projection of 3-D FLAME vertices → (u, v) screen coords.

    Rotation convention: azimuth around the Y axis (left-right turn), then
    elevation around the X axis (up-down tilt).  Positive azimuth rotates the
    face to the RIGHT so the viewer sees the LEFT cheek (nose moves right in
    screen space; "Left 45°" = az=+45°, "Right 45°" = az=−45°).
    """
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    Ry = np.array([[np.cos(az),  0, np.sin(az)],
                   [0,           1, 0          ],
                   [-np.sin(az), 0, np.cos(az)]])
    Rx = np.array([[1, 0,           0          ],
                   [0, np.cos(el), -np.sin(el) ],
                   [0, np.sin(el),  np.cos(el) ]])
    vr = vertices @ (Rx @ Ry).T     # (N, 3) in camera space
    return vr[:, 0], vr[:, 1]      # u = x_cam,  v = y_cam  (matplotlib Y-up, no flip)


def _backface_mask(
    vertices: np.ndarray, faces: np.ndarray,
    azim_deg: float, elev_deg: float
) -> np.ndarray:
    """
    Boolean mask shape (F,): True = triangle is back-facing in screen space.
    A triangle is back-facing when its projected area is negative (CW winding).
    These triangles are masked out to show a clean front-only silhouette.
    """
    u, v = _project_vertices(vertices, azim_deg, elev_deg)
    ax_, ay_ = u[faces[:, 0]], v[faces[:, 0]]
    bx_, by_ = u[faces[:, 1]], v[faces[:, 1]]
    cx_, cy_ = u[faces[:, 2]], v[faces[:, 2]]
    signed_area = (bx_ - ax_) * (cy_ - ay_) - (by_ - ay_) * (cx_ - ax_)
    # Negative signed area → CW winding in Y-up screen space → back-facing.
    # Mask (True) hides the triangle; mask front-facing positives only when
    # viewing from inside would be the bug — we mask the negatives (back faces).
    return signed_area < 0


def _depth_sort_faces(
    vertices: np.ndarray, faces: np.ndarray,
    azim_deg: float, elev_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Painter's algorithm: sort faces far-to-near in camera space.

    matplotlib's tripcolor has no z-buffer — it draws triangles in the order
    they appear in the face array.  Without sorting, interior triangles (inner
    mouth cavity, eye sockets) that happen to have lower face indices get drawn
    on top of the visible face surface, producing the "broken mesh" look.

    This function computes each triangle's centroid Z in camera space and
    returns the faces sorted ascending (most-negative Z first = farthest from
    camera = drawn first), so that near triangles paint over far ones.

    Returns
    -------
    faces_sorted : (F, 3) int — face array in draw order
    sort_idx     : (F,)  int — original-face indices in draw order
    """
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    Ry = np.array([[np.cos(az), 0.0,  np.sin(az)],
                   [0.0,        1.0,  0.0        ],
                   [-np.sin(az),0.0,  np.cos(az) ]])
    Rx = np.array([[1.0, 0.0,           0.0          ],
                   [0.0, np.cos(el), -np.sin(el)     ],
                   [0.0, np.sin(el),  np.cos(el)     ]])
    R = Rx @ Ry                                       # world → camera

    # Centroid of each triangle in camera space
    c = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0
    centroid_z = (c @ R.T)[:, 2]                     # (F,) — camera Z axis

    sort_idx = np.argsort(centroid_z)                 # ascending: far first
    return faces[sort_idx], sort_idx


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Area-weighted per-vertex normals for a triangulated mesh.

    For each triangle the unnormalised face normal (cross product of two edges,
    area = half its magnitude) is accumulated at all three vertices.  Dividing
    by the accumulated magnitude gives a unit normal that is biased toward the
    larger adjacent faces — the standard weighting for smooth-shading.

    Returns (N, 3) unit-length normals.
    """
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)          # (F, 3) unnormalised (2× area)
    vnormals = np.zeros_like(vertices, dtype=np.float64)
    np.add.at(vnormals, faces[:, 0], face_normals)
    np.add.at(vnormals, faces[:, 1], face_normals)
    np.add.at(vnormals, faces[:, 2], face_normals)
    norms = np.linalg.norm(vnormals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return (vnormals / norms).astype(np.float64)


def _canonical_pose_align(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Rotate FLAME mesh to canonical front-facing pose using two anatomical
    landmark vertices that are consistent across all FLAME regressors.

    Landmark indices (FLAME 2020, 5023-vertex template):
      3564 — nose tip (global max-Z vertex in template, at +Z in canonical space)
      3559 — crown   (global max-Y vertex in template, at +Y in canonical space)

    The alignment constructs a right-handed coordinate frame:
      +Z = direction from centroid to nose tip
      +Y = component of (centroid→crown) perpendicular to Z
      +X = Y × Z  (right-hand rule → anatomical right side)

    Vertices are centred before the rotation so the output is centred at the
    origin with the face pointing forward (+Z) and crown pointing up (+Y),
    regardless of the global rotation baked in by each regressor's pose
    prediction.  Only rigid rotation is applied — face shape is unchanged.
    """
    _NOSE = 3564   # max-Z vertex in FLAME 2020 template
    _CROWN = 3559  # max-Y vertex in FLAME 2020 template

    v = np.asarray(vertices, dtype=np.float64)
    v = v - v.mean(axis=0)

    nose_idx  = min(_NOSE,  len(v) - 1)
    crown_idx = min(_CROWN, len(v) - 1)

    # +Z axis: unit vector from centroid toward nose tip
    d_nose = v[nose_idx]
    nose_len = np.linalg.norm(d_nose)
    if nose_len < 1e-10:          # degenerate / completely flat mesh
        return v.astype(vertices.dtype)
    z_axis = d_nose / nose_len

    # +Y axis: crown direction, Gram-Schmidt-orthogonalised against Z
    d_crown = v[crown_idx]
    y_raw   = d_crown - np.dot(d_crown, z_axis) * z_axis
    y_len   = np.linalg.norm(y_raw)
    if y_len < 1e-8:              # crown collinear with nose — use world +Y as fallback
        y_raw = np.array([0.0, 1.0, 0.0])
        y_raw = y_raw - np.dot(y_raw, z_axis) * z_axis
        y_len = np.linalg.norm(y_raw)
    y_axis = y_raw / y_len

    # +X axis: right-hand rule (Y × Z gives anatomical right)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)

    # Re-orthogonalise Y to eliminate numerical drift
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-10)

    # R rows = new basis vectors in original space; v_aligned = v @ R.T
    R = np.stack([x_axis, y_axis, z_axis], axis=0)   # (3, 3)
    return (v @ R.T).astype(vertices.dtype)


def _lambertian_intensity(
    vertices: np.ndarray,
    faces: np.ndarray,
    azim: float,
    elev: float,
    ambient: float = 0.42,
    diffuse: float = 0.65,
) -> np.ndarray:
    """
    Per-vertex Lambertian shading intensity in [0, 1] for a neutral gray render.

    Light direction is fixed in *camera* space: mostly from the front (+Z),
    slightly from the upper-right (+Y, +X).  After rotating the vertex normals
    into camera space via the same view matrix used for projection, we compute
    intensity = ambient + diffuse × max(0, dot(n_cam, light_cam)).

    This produces bright highlights on the nose bridge and forehead, mid-gray
    cheeks, and darker shadowing on the side/back of the head — closely
    matching the SHeaP offscreen-render appearance.
    """
    az = np.radians(azim)
    el = np.radians(elev)
    Ry = np.array([[np.cos(az),  0, np.sin(az)],
                   [0,           1, 0          ],
                   [-np.sin(az), 0, np.cos(az)]])
    Rx = np.array([[1, 0,            0           ],
                   [0, np.cos(el), -np.sin(el)   ],
                   [0, np.sin(el),  np.cos(el)   ]])
    R = Rx @ Ry                                     # world → camera rotation

    normals_world = _compute_vertex_normals(vertices, faces)  # (N, 3)
    normals_cam   = normals_world @ R.T                        # (N, 3)

    # Key light: mostly from front (+Z_cam), slight upper-right (+Y_cam, +X_cam)
    light = np.array([0.30, 0.55, 1.0], dtype=np.float64)
    light /= np.linalg.norm(light)

    # Fill light from opposite side to avoid pure black shadows
    fill  = np.array([-0.20, 0.10, 0.50], dtype=np.float64)
    fill  /= np.linalg.norm(fill)

    key_contrib  = diffuse        * np.clip(normals_cam @ light, 0, 1)
    fill_contrib = diffuse * 0.25 * np.clip(normals_cam @ fill,  0, 1)
    intensity    = ambient + key_contrib + fill_contrib
    return np.clip(intensity, 0.0, 1.0).astype(np.float64)


def _draw_gray_mesh_view(
    ax: plt.Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    azim: float = 0.0,
    elev: float = 10.0,
    bg: str = '#aaaaaa',
    wireframe: bool = False,
    ambient: float = 0.42,
    diffuse: float = 0.65,
    title: str = '',
) -> None:
    """
    Render a FLAME mesh as a neutral gray solid using Lambertian shading.

    This reproduces the SHeaP-style offscreen render appearance without
    requiring pyrender or an OpenGL context.  The mesh is shaded by
    per-vertex Lambertian intensity mapped onto a gray colourmap, with
    back-face culling so only the visible hemisphere is drawn.

    Use for *reconstruction* panels where the goal is to show face shape,
    not encode a scalar quantity.  For uncertainty/error heatmaps use
    _draw_mesh_view with an appropriate colourmap instead.
    """
    ax.set_facecolor(bg)
    verts = _canonical_pose_align(np.asarray(vertices, dtype=np.float64), faces)
    intensity = _lambertian_intensity(verts, faces, azim, elev, ambient, diffuse)
    # Painter's algorithm: sort faces far-to-near so near triangles paint over
    # far ones — eliminates interior-triangle z-ordering artifacts.
    faces_sorted, _ = _depth_sort_faces(verts, faces, azim, elev)
    u, v = _project_vertices(verts, azim, elev)
    triang = mtri.Triangulation(u, v, faces_sorted)
    triang.set_mask(_backface_mask(verts, faces_sorted, azim, elev))
    ax.tripcolor(triang, intensity, cmap='gray', vmin=0.0, vmax=1.0,
                 shading='gouraud', rasterized=True)
    if wireframe:
        ax.triplot(triang, color='white', lw=0.10, alpha=0.06, rasterized=True)
    ax.set_aspect('equal')
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, pad=3)


def _draw_mesh_view(
    ax: plt.Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    azim: float = 0.0,
    elev: float = 10.0,
    title: str = '',
    bg: str = '#f0f0f0',
    wireframe: bool = True,
) -> None:
    """
    Render one camera view of a FLAME mesh onto *ax* using Gouraud-shaded
    matplotlib tripcolor.  Back-facing triangles are culled automatically.

    The colour at each triangle vertex is linearly interpolated across the face
    ('gouraud' shading), giving a smooth appearance even with relatively few
    vertices.  No 3-D axes or tick marks are drawn — only the mesh silhouette.

    When wireframe=True (default) a faint white edge overlay is drawn on top of
    the colour fill.  This makes anatomical regions (nose bridge, eye sockets,
    jaw line) legible regardless of what the scalar colourmap encodes, so the
    viewer can always identify whether the front or back of the head is shown.
    """
    ax.set_facecolor(bg)
    verts_canon = _canonical_pose_align(np.asarray(vertices, dtype=np.float64), faces)
    # Painter's algorithm: sort faces far-to-near so near triangles paint over
    # far ones — eliminates interior-triangle z-ordering artifacts.
    faces_sorted, _ = _depth_sort_faces(verts_canon, faces, azim, elev)
    u, v = _project_vertices(verts_canon, azim, elev)
    triang = mtri.Triangulation(u, v, faces_sorted)
    triang.set_mask(_backface_mask(verts_canon, faces_sorted, azim, elev))
    ax.tripcolor(triang, scalars,
                 cmap=cmap, vmin=vmin, vmax=vmax,
                 shading='gouraud', rasterized=True)
    if wireframe:
        ax.triplot(triang, color='white', lw=0.15, alpha=0.12, rasterized=True)
    ax.set_aspect('equal')
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, pad=3)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  render_uncertainty_heatmap
# ══════════════════════════════════════════════════════════════════════════════

def render_uncertainty_heatmap(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalar_values: np.ndarray,
    save_path: str,
    colormap: str = _MESH_CMAP,
    title: str = 'Uncertainty Heatmap',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (16, 5),
) -> None:
    """
    Render a FLAME mesh colour-coded by a per-vertex scalar field — uncertainty
    or geometric error — showing three orthographic viewpoints in one figure.

    Layout
    ------
    ┌──────────────────────────────────────────┐
    │  Front view  │  Left 45°  │  Right 45°  │  ← shared colourbar on right
    └──────────────────────────────────────────┘

    The colourmap is shared and normalised to [vmin, vmax] (auto if None).
    Gouraud shading interpolates vertex colours across each triangle, giving a
    smooth heatmap even with the coarse FLAME topology.

    Parameters
    ----------
    vertices     : (5023, 3) FLAME vertices
    faces        : (F, 3) triangle indices
    scalar_values: (5023,) or (5023, 1) per-vertex scalars to colour-map
    save_path    : output file (PNG / PDF / SVG)
    colormap     : matplotlib colourmap name (default 'plasma')
    title        : figure super-title
    vmin, vmax   : colourbar limits; None = auto-scale to [min, max]
    """
    scalars = np.asarray(scalar_values, dtype=np.float64).ravel()
    vmin = float(scalars.min()) if vmin is None else float(vmin)
    vmax = float(scalars.max()) if vmax is None else float(vmax)
    if vmin == vmax:
        vmax = vmin + 1e-8

    # Centre the mesh so it renders correctly regardless of model-specific offset.
    vertices = np.asarray(vertices, dtype=np.float64)
    vertices = vertices - vertices.mean(axis=0, keepdims=True)

    fig, axes = plt.subplots(1, 3, figsize=figsize,
                              layout='constrained')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.patch.set_facecolor('white')

    views = [("Front", 0.0, 10.0), ("Left 45°", 45.0, 10.0), ("Right 45°", -45.0, 10.0)]
    for ax, (view_name, azim, elev) in zip(axes, views):
        _draw_mesh_view(ax, vertices, faces, scalars, colormap,
                        vmin, vmax, azim=azim, elev=elev, title=view_name)

    sm = plt.cm.ScalarMappable(cmap=colormap,
                                norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label('Value (a.u.)', fontsize=10)

    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  plot_uncertainty_spatial_maps
# ══════════════════════════════════════════════════════════════════════════════

def plot_uncertainty_spatial_maps(
    vertices: np.ndarray,
    faces: np.ndarray,
    uncertainty_dict: Dict[str, np.ndarray],
    save_path: str,
    colormap: str = _MESH_CMAP,
    share_scale: bool = False,
    figsize_per_panel: Tuple[float, float] = (3.5, 3.5),
) -> None:
    """
    Side-by-side front-view mesh panels, one per uncertainty method.

    Each panel uses an independent colourmap range by default (so the *pattern*
    within each method is visible regardless of absolute scale).  Setting
    ``share_scale=True`` maps all panels onto the same [global_min, global_max]
    for a magnitude-comparable view.

    This plot reveals spatial patterns across methods at a glance — e.g., TTA
    may concentrate uncertainty along the hair boundary while MCD concentrates
    it around the jaw hinge.

    Parameters
    ----------
    vertices         : (5023, 3)
    faces            : (F, 3)
    uncertainty_dict : {method_name: (5023,) or (5023, 1)}
    share_scale      : False = per-method scale (pattern visible), True = shared (magnitude comparable)
    """
    methods = list(uncertainty_dict.keys())
    n = len(methods)
    if n == 0:
        warnings.warn("plot_uncertainty_spatial_maps: empty uncertainty_dict.")
        return

    # Centre the mesh so it renders correctly regardless of model-specific offset.
    vertices = np.asarray(vertices, dtype=np.float64)
    vertices = vertices - vertices.mean(axis=0, keepdims=True)

    all_scalars = [np.asarray(uncertainty_dict[m], dtype=np.float64).ravel()
                   for m in methods]
    global_vmin = float(min(s.min() for s in all_scalars))
    global_vmax = float(max(s.max() for s in all_scalars))

    fig, axes = plt.subplots(1, n,
                              figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
                              layout='constrained')
    if n == 1:
        axes = [axes]

    fig.suptitle('Per-Vertex Uncertainty Maps (Front View)', fontsize=13,
                 fontweight='bold')

    for ax, method, scalars in zip(axes, methods, all_scalars):
        if share_scale:
            vmin, vmax = global_vmin, global_vmax
        else:
            vmin = float(scalars.min())
            vmax = float(scalars.max())
            if vmin == vmax:
                vmax = vmin + 1e-8

        _draw_mesh_view(ax, vertices, faces, scalars, colormap,
                        vmin, vmax, azim=0.0, elev=10.0)
        ax.set_title(_dn(method), fontsize=10, fontweight='bold',
                     color=_method_color(method))

        if not share_scale:
            sm = plt.cm.ScalarMappable(cmap=colormap,
                                        norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                                 shrink=0.85, pad=0.02, aspect=20)
            cbar.ax.tick_params(labelsize=7)

    if share_scale:
        sm = plt.cm.ScalarMappable(
            cmap=colormap,
            norm=mcolors.Normalize(vmin=global_vmin, vmax=global_vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02, aspect=30)
        cbar.set_label('Uncertainty (shared scale)', fontsize=9)

    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  plot_uncertainty_comparison_violin
# ══════════════════════════════════════════════════════════════════════════════

def plot_uncertainty_comparison_violin(
    uncertainty_dict: Dict[str, np.ndarray],
    save_path: str,
    error: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """
    Violin + box-whisker comparison of per-vertex uncertainty distributions.

    The *distribution shape* reveals calibration properties that scalar summaries
    miss:
    - Narrow violin (low variance across vertices) → diffuse / uniform uncertainty
    - Wide / bimodal violin → concentrated on specific anatomical regions

    If ``error`` is provided, a reference panel shows the geometric error
    distribution for magnitude comparison.

    Parameters
    ----------
    uncertainty_dict : {method_name: (N,)} per-vertex scalars
    error            : optional (N,) per-vertex geometric error in mm
    """
    methods = list(uncertainty_dict.keys())
    n = len(methods)
    if n == 0:
        return

    ncols = 2 if error is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize,
                              layout='constrained',
                              gridspec_kw={'width_ratios': [3, 1] if ncols == 2 else [1]})
    if ncols == 1:
        axes = [axes]

    ax = axes[0]
    data   = [np.asarray(uncertainty_dict[m], dtype=np.float64).ravel() for m in methods]
    colors = [_method_color(m) for m in methods]

    parts = ax.violinplot(data, positions=range(n),
                           showmedians=True, showextrema=False, widths=0.7)
    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
        body.set_alpha(0.65)

    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')

    # IQR box overlay
    for i, (d, color) in enumerate(zip(data, colors)):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(i, q1, q3, color='black', linewidth=3, zorder=3)
        ax.scatter(i, med, color='white', s=30, zorder=4,
                    edgecolors='black', linewidth=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels([_dn(m) for m in methods], rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Per-vertex uncertainty (a.u.)', fontsize=10)
    ax.set_title('Uncertainty Distribution per Method\n'
                  '(narrower violin = more spatially concentrated)', fontsize=11,
                  fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if error is not None:
        ax2 = axes[1]
        e = np.asarray(error, dtype=np.float64).ravel()
        ax2.violinplot([e], positions=[0], showmedians=True, widths=0.6)
        ax2.set_xticks([0])
        ax2.set_xticklabels(['GT Error'], fontsize=9)
        ax2.set_ylabel('Per-vertex error (mm)', fontsize=10)
        ax2.set_title('Geometric\nError Ref.', fontsize=10, fontweight='bold')
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  plot_sparsification_curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_sparsification_curves(
    sparsification_dict: Dict[str, Dict[str, np.ndarray]],
    save_path: str,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """
    Sparsification error curves for all uncertainty methods + AUSE bar inset.

    Layout
    ------
    Left (large): sparsification curves
      X = fraction of highest-uncertainty vertices removed
      Y = mean error of the remaining vertices
      Three curve types:
        Coloured solid  – method curve (what the estimator achieves)
        Black dashed    – oracle curve (ideal: remove by true error)
        Grey dashed     – random curve (baseline: uninformative estimator)
      Semi-transparent fill between method and oracle = the AUSE area.

    Right (small): AUSE bar chart — direct ranking across all methods.
    Lower AUSE = the method's ranking of uncertain vertices is closer to ideal.

    Parameters
    ----------
    sparsification_dict : {method_name: output_of(calculate_sparsification_error_curve)}
        Required keys per method: 'fractions', 'method_errors', 'oracle_errors', 'ause'.
    """
    if not sparsification_dict:
        return

    fig = plt.figure(figsize=figsize, layout='constrained')
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[2.5, 1])
    ax_curve = fig.add_subplot(gs[0])
    ax_bar   = fig.add_subplot(gs[1])

    oracle_plotted = False
    random_plotted = False
    ause_vals = {}

    for method, res in sparsification_dict.items():
        fracs  = np.asarray(res['fractions'])
        me     = np.asarray(res['method_errors'])
        oracle = np.asarray(res['oracle_errors'])
        rand   = np.asarray(res.get('random_errors', np.full_like(me, np.nan)))
        ause   = float(res.get('ause', np.nan))
        ause_vals[method] = ause
        color  = _method_color(method)

        ax_curve.plot(fracs, me, color=color, linewidth=2,
                       label=f"{method} (AUSE={ause:.4f})")
        ax_curve.fill_between(fracs, oracle, me,
                               where=me >= oracle, alpha=0.09, color=color)

        if not oracle_plotted:
            ax_curve.plot(fracs, oracle, 'k--', linewidth=1.5,
                           label='Oracle', zorder=10)
            oracle_plotted = True
        if not random_plotted and not np.all(np.isnan(rand)):
            ax_curve.plot(fracs, rand, color='#AAAAAA', linestyle='--',
                           linewidth=1.2, label='Random')
            random_plotted = True

    ax_curve.set_xlabel('Fraction of most-uncertain vertices removed', fontsize=10)
    ax_curve.set_ylabel('Mean error of remaining vertices (mm)', fontsize=10)
    ax_curve.set_title('Sparsification Error Curves\n'
                        '(shaded gap above oracle = AUSE; smaller gap = better)',
                        fontsize=11, fontweight='bold')
    ax_curve.legend(fontsize=8, loc='upper right')
    ax_curve.spines[['top', 'right']].set_visible(False)
    ax_curve.grid(alpha=0.25, linestyle='--')

    # AUSE bar chart (sorted best → worst)
    methods_sorted = sorted(ause_vals, key=lambda m: ause_vals[m])
    vals   = [ause_vals[m] for m in methods_sorted]
    colors = [_method_color(m) for m in methods_sorted]
    bars = ax_bar.barh(range(len(methods_sorted)), vals,
                        color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax_bar.set_yticks(range(len(methods_sorted)))
    ax_bar.set_yticklabels(methods_sorted, fontsize=9)
    ax_bar.set_xlabel('AUSE ↓', fontsize=9)
    ax_bar.set_title('AUSE\nRanking', fontsize=10, fontweight='bold')
    ax_bar.spines[['top', 'right']].set_visible(False)
    ax_bar.grid(axis='x', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, vals):
        ax_bar.text(val + max(vals) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f'{val:.4f}', va='center', fontsize=7)

    fig.suptitle('Sparsification Error Analysis (AUSE)',
                  fontsize=13, fontweight='bold')
    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  plot_calibration_diagram
# ══════════════════════════════════════════════════════════════════════════════

def plot_calibration_diagram(
    calibration_dict: Dict[str, Dict],
    save_path: str,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Reliability (calibration) diagram for multiple uncertainty methods.

    Each curve plots empirical coverage at a grid of confidence levels α.
    Perfect calibration = diagonal (empirical coverage = α for all α).

    - Curve above diagonal → under-confident (intervals too wide / conservative)
    - Curve below diagonal → over-confident (intervals too narrow)

    The Expected Calibration Error (ECE) = mean |empirical − expected| is
    annotated in the legend and ranked in the inset bar chart.

    Parameters
    ----------
    calibration_dict : {method_name: output_of(calculate_uncertainty_calibration)}
        Required keys: 'reliability_diagram_x', 'reliability_diagram_y', 'ece'.
    """
    if not calibration_dict:
        return

    fig = plt.figure(figsize=figsize, layout='constrained')
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
    ax  = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration', zorder=10)
    ax.fill_between([0, 1], [0, 1], 1, color='#FFF3E0', alpha=0.4, label='Under-confident region')
    ax.fill_between([0, 1], 0, [0, 1], color='#E3F2FD', alpha=0.4, label='Over-confident region')

    ece_vals = {}
    for method, res in calibration_dict.items():
        x   = np.asarray(res['reliability_diagram_x'])
        y   = np.asarray(res['reliability_diagram_y'])
        ece = float(res.get('ece', np.nan))
        ece_vals[method] = ece
        color = _method_color(method)
        ax.plot(x, y, 'o-', color=color, linewidth=2,
                 markersize=6, label=f"{method}", zorder=5)

        # Annotate over/underconfidence fractions at the curve end-point
        overc  = float(res.get('overconfidence_fraction',  float('nan')))
        underc = float(res.get('underconfidence_fraction', float('nan')))
        if not (np.isnan(overc) or np.isnan(underc)) and len(x) > 0:
            ax.annotate(
                f"cons={overc:.0%} | overc={underc:.0%}",
                xy=(x[-1], y[-1]),
                xytext=(x[-1] - 0.35, y[-1] + 0.03),
                fontsize=6.5, color=color, alpha=0.85,
                arrowprops=dict(arrowstyle='-', color=color, alpha=0.4, lw=0.8),
            )

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.08)
    ax.set_xlabel('Expected confidence level α', fontsize=10)
    ax.set_ylabel('Empirical coverage fraction', fontsize=10)
    ax.set_title('Calibration Reliability Diagram\n'
                  '(on diagonal = perfect; note: ECE is not meaningful after min-max normalisation)',
                  fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, linestyle='--')

    # ECE ranking bar
    methods_sorted = sorted(ece_vals, key=lambda m: ece_vals[m])
    vals   = [ece_vals[m] for m in methods_sorted]
    colors = [_method_color(m) for m in methods_sorted]
    bars = ax2.barh(range(len(methods_sorted)), vals,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(methods_sorted)))
    ax2.set_yticklabels(methods_sorted, fontsize=9)
    ax2.set_xlabel('ECE (not meaningful\nafter normalisation)', fontsize=9)
    ax2.set_title('ECE\n(informational)', fontsize=10, fontweight='bold')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    fig.suptitle('Uncertainty Calibration Analysis', fontsize=13, fontweight='bold')
    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  plot_uncertainty_vs_error_scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_uncertainty_vs_error_scatter(
    error: np.ndarray,
    uncertainty_dict: Dict[str, np.ndarray],
    save_path: str,
    spearman_rho_dict: Optional[Dict[str, float]] = None,
    max_points: int = 5023,
    figsize_per: Tuple[float, float] = (3.5, 3.5),
) -> None:
    """
    Per-vertex scatter: geometric error (X) vs uncertainty (Y), one panel per method.

    Each point is one FLAME vertex.  A well-calibrated estimator shows a
    positive trend: high-error vertices → high uncertainty.

    Spearman ρ (rank correlation) is annotated on each panel — the primary
    metric for uncertainty ranking quality.  A linear trend line is overlaid.

    Parameters
    ----------
    error            : (N,) per-vertex geometric error in mm
    uncertainty_dict : {method_name: (N,)} per-vertex uncertainty
    spearman_rho_dict: pre-computed Spearman ρ per method (avoids recomputation)
    max_points       : subsample to avoid overplotting (5023 = all FLAME vertices)
    """
    from scipy import stats as _stats

    methods = list(uncertainty_dict.keys())
    n = len(methods)
    if n == 0:
        return

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        squeeze=False, layout='constrained',
    )
    fig.suptitle(
        'Uncertainty vs. Geometric Error (per vertex)\n'
        'Spearman ρ annotated — positive trend = uncertainty tracks error',
        fontsize=12, fontweight='bold',
    )

    e_all = np.asarray(error, dtype=np.float64).ravel()

    for idx, method in enumerate(methods):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        u_all = np.asarray(uncertainty_dict[method], dtype=np.float64).ravel()

        if len(e_all) > max_points:
            rng = np.random.default_rng(42)
            sel = rng.choice(len(e_all), max_points, replace=False)
            e_plot, u_plot = e_all[sel], u_all[sel]
        else:
            e_plot, u_plot = e_all, u_all

        color = _method_color(method)
        ax.scatter(e_plot, u_plot, s=4, alpha=0.35, color=color, rasterized=True)

        try:
            slope, intercept, *_ = _stats.linregress(e_plot, u_plot)
            x_line = np.linspace(e_plot.min(), e_plot.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.5, alpha=0.8)
        except Exception:
            pass

        if spearman_rho_dict and method in spearman_rho_dict:
            rho = spearman_rho_dict[method]
        else:
            rho, _ = _stats.spearmanr(e_all, u_all)

        ax.set_title(f"{_dn(method)}\nSpearman ρ = {rho:.3f}", fontsize=9,
                      fontweight='bold', color=color)
        ax.set_xlabel('Error (mm)', fontsize=8)
        ax.set_ylabel('Uncertainty', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.2, linestyle='--')

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  plot_method_comparison_table
# ══════════════════════════════════════════════════════════════════════════════

def plot_method_comparison_table(
    summary_dict: Dict[str, Dict[str, float]],
    save_path: str,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """
    Colour-coded table (heatmap) of uncertainty quality metrics across methods.

    Cell colour encodes performance relative to other methods per metric:
      Green  = better-than-average  (green definition depends on metric direction)
      Red    = worse-than-average
      Grey   = informational (no direction)

    Numerical values are printed in each cell.  Column headers are the method
    names (coloured by their global palette), row headers are metric labels.

    Parameters
    ----------
    summary_dict : output of compare_uncertainty_methods()
        {method_name: {"spearman_rho": ..., "ause": ..., ...}}
    metrics      : subset of metric keys to display; None = show all available
    """
    if not summary_dict:
        return

    all_metrics = metrics or list(_METRIC_LABELS.keys())
    first_val = next(iter(summary_dict.values()))
    available = [m for m in all_metrics if m in first_val]
    methods = list(summary_dict.keys())
    n_methods = len(methods)
    n_metrics = len(available)
    if n_metrics == 0:
        return

    data = np.full((n_metrics, n_methods), np.nan)
    for j, method in enumerate(methods):
        for i, metric in enumerate(available):
            data[i, j] = summary_dict[method].get(metric, np.nan)

    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.axis('off')
    fig.suptitle('Uncertainty Method Comparison — Full Metric Table',
                  fontsize=13, fontweight='bold')

    for i, metric in enumerate(available):
        row = data[i]
        direction = _METRIC_BETTER.get(metric, None)
        valid = row[~np.isnan(row)]

        if len(valid) >= 2 and direction is not None:
            rmin, rmax = valid.min(), valid.max()
            norm_vals = (row - rmin) / (rmax - rmin + 1e-10)
            row_cmap = cm.get_cmap('RdYlGn') if direction else cm.get_cmap('RdYlGn_r')
        else:
            norm_vals = np.full(n_methods, 0.5)
            row_cmap  = None

        for j, (val, nv) in enumerate(zip(row, norm_vals)):
            bg = row_cmap(float(nv)) if (row_cmap is not None and not np.isnan(val)) else (0.96, 0.96, 0.96, 1.0)
            rect = plt.Rectangle([j, n_metrics - 1 - i], 1, 1,
                                   color=bg, ec='white', lw=1.5)
            ax.add_patch(rect)
            brightness = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            txt_color  = 'white' if brightness < 0.5 else '#111111'
            cell_text  = f"{val:.4f}" if not np.isnan(val) else "—"
            ax.text(j + 0.5, n_metrics - 0.5 - i, cell_text,
                     ha='center', va='center', fontsize=9,
                     color=txt_color, fontweight='bold')

    for j, method in enumerate(methods):
        ax.text(j + 0.5, n_metrics + 0.2, _dn(method), ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color=_method_color(method))

    for i, metric in enumerate(available):
        label = _METRIC_LABELS.get(metric, metric)
        ax.text(-0.1, n_metrics - 0.5 - i, label, ha='right', va='center', fontsize=9)

    ax.set_xlim(-0.6, n_methods)
    ax.set_ylim(-0.3, n_metrics + 0.6)

    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  plot_region_error_breakdown
# ══════════════════════════════════════════════════════════════════════════════

def plot_region_error_breakdown(
    region_errors_dict: Dict[str, Dict[str, Dict[str, float]]],
    save_path: str,
    metric: str = 'mean_mm',
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """
    Grouped bar chart of geometric error across nine FLAME facial regions.

    Each bar group = one region; bars within = one method / regressor.
    Error bars show standard deviation within the region.

    This reveals *where* each method succeeds or fails.  A method tuned on
    expression data might do well around the mouth but poorly at the hairline.

    Parameters
    ----------
    region_errors_dict : {method_name: {region_name: {"mean_mm": ..., "std_mm": ...}}}
        Nested dict from calculate_region_wise_geometric_error per method.
    metric : 'mean_mm', 'median_mm', or 'max_mm'
    """
    if not region_errors_dict:
        return

    methods = list(region_errors_dict.keys())
    all_regions: List[str] = []
    for m in methods:
        for r in region_errors_dict[m]:
            if r not in all_regions:
                all_regions.append(r)

    n_regions = len(all_regions)
    n_methods = len(methods)
    if n_regions == 0:
        return

    bar_width = 0.8 / n_methods
    x = np.arange(n_regions)

    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    for i, method in enumerate(methods):
        vals = []
        errs = []
        for region in all_regions:
            rd = region_errors_dict[method].get(region, {})
            vals.append(float(rd.get(metric, 0.0)))
            errs.append(float(rd.get('std_mm', 0.0)))
        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, width=bar_width * 0.9,
                label=method, color=_method_color(method),
                alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.errorbar(x + offset, vals, yerr=errs,
                     fmt='none', ecolor='black', elinewidth=1, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('_', '\n') for r in all_regions],
                        fontsize=9, ha='center')
    ax.set_ylabel(f'Error ({metric.replace("_", " ")}) [mm]', fontsize=10)
    ax.set_title('Region-wise Geometric Error Breakdown\n'
                  '(error bars = within-region σ; lower = better)', fontsize=12,
                  fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  plot_error_distributions
# ══════════════════════════════════════════════════════════════════════════════

def plot_error_distributions(
    error_dict: Dict[str, np.ndarray],
    save_path: str,
    figsize: Tuple[int, int] = (14, 4),
    bins: int = 60,
) -> None:
    """
    Overlaid histograms + KDE of per-vertex geometric error for each regressor.

    Dashed vertical lines mark the median of each distribution.  Reveals both
    the *central tendency* (median) and *tail behaviour* — invisible in a
    single-number summary.  A narrow, left-skewed distribution indicates a
    method that is consistently accurate with few catastrophic failures.

    Parameters
    ----------
    error_dict : {model_name: (N,) per-vertex error array in mm}
    bins       : histogram bins (default 60)
    """
    if not error_dict:
        return

    try:
        from scipy.stats import gaussian_kde
        has_kde = True
    except ImportError:
        has_kde = False

    fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    for method, err in error_dict.items():
        e = np.asarray(err, dtype=np.float64).ravel()
        color  = _method_color(method)
        median = float(np.median(e))

        ax.hist(e, bins=bins, density=True, alpha=0.20, color=color)

        if has_kde:
            try:
                kde   = gaussian_kde(e, bw_method='scott')
                x_kde = np.linspace(e.min(), e.max(), 500)
                ax.plot(x_kde, kde(x_kde), color=color, linewidth=2,
                         label=f"{method}  (med={median:.2f} mm)")
            except Exception:
                ax.axvline(median, color=color, linewidth=2, linestyle='-',
                            label=f"{method}  (med={median:.2f} mm)")
        else:
            ax.axvline(median, color=color, linewidth=2,
                        label=f"{method}  (med={median:.2f} mm)")

        ax.axvline(median, color=color, linewidth=1.5, linestyle='--', alpha=0.7)

    ax.set_xlabel('Per-vertex geometric error (mm)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Error Distribution per Regressor\n'
                  '(dashed = median; narrow, left-skewed = better)',
                  fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, linestyle='--')

    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 10. plot_scan_to_mesh_comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_scan_to_mesh_comparison(
    s2m_results: Dict[str, Dict[str, float]],
    save_path: str,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Grouped bar chart of scan-to-mesh distances per FLAME regressor.

    Scan-to-mesh (NoW benchmark convention) is the primary geometric quality
    metric when ground truth is a raw 3D scan rather than a FLAME registration.
    For each GT scan point, the nearest surface point on the predicted mesh is
    found, and distances are aggregated into median, mean, and 90th percentile.

    Lower bars = better reconstruction.

    Parameters
    ----------
    s2m_results : {model_name: {"median_mm": ..., "mean_mm": ..., "p90_mm": ...}}
        Output of calculate_scan_to_mesh_distance averaged across evaluation images.
    """
    if not s2m_results:
        return

    methods = list(s2m_results.keys())
    metrics_to_show = [
        ('median_mm', 'Median (mm)', 0.6),
        ('mean_mm',   'Mean (mm)',   0.3),
        ('p90_mm',    'P90 (mm)',    0.15),
    ]

    fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    n_methods = len(methods)
    n_metrics = len(metrics_to_show)
    bar_w     = 0.7 / n_metrics
    x         = np.arange(n_methods)

    for j, (key, label, alpha_base) in enumerate(metrics_to_show):
        vals = [float(s2m_results[m].get(key, 0.0)) for m in methods]
        colors = [_method_color(m) for m in methods]
        offset = (j - n_metrics / 2 + 0.5) * bar_w
        bars = ax.bar(
            x + offset, vals,
            width=bar_w * 0.92,
            label=label,
            color=[mcolors.to_rgba(c, alpha_base + 0.4) for c in colors],
            edgecolor=colors, linewidth=1.0,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 * max(
                        s2m_results[m].get('p90_mm', 1) for m in methods
                    ),
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('Distance (mm) — lower is better', fontsize=10)
    ax.set_title(
        'Scan-to-Mesh Distance per Regressor (NoW convention)\n'
        'Median = primary metric; P90 = tail quality',
        fontsize=12, fontweight='bold',
    )
    ax.legend(fontsize=9, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 11. plot_sharpness_comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_sharpness_comparison(
    summary_dict: Dict[str, Dict[str, float]],
    save_path: str,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Bar chart comparing three sharpness facets across uncertainty methods.

    Sharpness measures whether the predicted uncertainty is spatially
    informative — i.e. concentrated on vertices that are genuinely hard —
    rather than being a uniform low-signal heatmap.

    Three facets are shown side by side:
      • Mean uncertainty  — overall level of predicted uncertainty.
      • Spatial entropy   — how uniformly uncertainty is spread (lower = sharper).
      • CV (std/mean)     — scale-free spread; higher = more spatially discriminative.

    A perfectly useless estimator assigns equal uncertainty to every vertex
    (entropy ≈ 1, CV ≈ 0).  A perfectly focused estimator concentrates all
    uncertainty on a small anatomical region (low entropy, high CV).

    Parameters
    ----------
    summary_dict : output of compare_uncertainty_methods(), must contain keys
        "sharpness_mean", "sharpness_entropy", "sharpness_cv".
    """
    if not summary_dict:
        return

    methods = list(summary_dict.keys())
    facets = [
        ("sharpness_mean",    "Mean uncertainty",    "Mean uncertainty (a.u.)"),
        ("sharpness_entropy", "Spatial entropy ↓",   "Spatial entropy [0=sharp, 1=uniform]"),
        ("sharpness_cv",      "CV (std/mean) →high", "Coefficient of variation (higher=sharper)"),
    ]

    n_avail = sum(
        1 for _, key, _ in [(k, k, '') for k, _, _ in facets]
        if any(key in summary_dict[m] for m in methods)
    )
    if n_avail == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=figsize, layout='constrained')
    fig.suptitle('Uncertainty Sharpness Analysis\n'
                 '(how spatially concentrated / informative each method is)',
                 fontsize=12, fontweight='bold')

    for ax, (key, label, ylabel) in zip(axes, facets):
        vals   = [float(summary_dict[m].get(key, 0.0)) for m in methods]
        colors = [_method_color(m) for m in methods]
        bars = ax.bar(range(len(methods)), vals, color=colors,
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=28, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 12. plot_multi_model_reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def plot_multi_model_reconstruction(
    vertices_dict: Dict[str, np.ndarray],
    faces: np.ndarray,
    save_path: str,
    gt_vertices: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None,
    colormap: str = 'Blues',
    figsize_per: Tuple[float, float] = (3.5, 4.0),
) -> None:
    """
    Side-by-side front-view mesh renders for all FLAME regressors.

    Colour encodes the Z coordinate (depth): brighter = closer to camera.
    Cheekbone prominence, nose tip projection, and jaw depth are directly
    visible.  If ``image`` is provided it is shown first as a reference photo.
    If ``gt_vertices`` is provided, a final GT panel is appended.

    Parameters
    ----------
    vertices_dict : {model_name: (5023, 3)} predicted vertices
    faces         : (F, 3) FLAME face topology (shared across all methods)
    gt_vertices   : optional (5023, 3) ground-truth FLAME vertices
    image         : optional (H, W, 3) uint8 input photo shown as first panel
    colormap      : colourmap for depth; 'Blues' or 'viridis' both work well
    """
    panels = dict(vertices_dict)
    if gt_vertices is not None:
        panels['GT'] = gt_vertices

    methods = list(panels.keys())
    n_mesh = len(methods)
    if n_mesh == 0:
        return

    n = n_mesh + (1 if image is not None else 0)
    fig, axes = plt.subplots(1, n,
                              figsize=(figsize_per[0] * n, figsize_per[1]),
                              layout='constrained')
    axes = [axes] if n == 1 else list(axes)

    ax_iter = iter(axes)

    # Optional input photo
    if image is not None:
        ax_photo = next(ax_iter)
        ax_photo.imshow(np.asarray(image, dtype=np.uint8))
        ax_photo.set_axis_off()
        ax_photo.set_title('Input Image', fontsize=10, fontweight='bold', color='#333333')

    for method in methods:
        ax = next(ax_iter)
        verts = np.asarray(panels[method], dtype=np.float64)
        verts = verts - verts.mean(axis=0, keepdims=True)
        label_color = '#333333' if method == 'GT' else _method_color(method)
        _draw_gray_mesh_view(ax, verts, faces, azim=0.0, elev=10.0, bg='#b0b0b0')
        ax.set_title(method, fontsize=10, fontweight='bold', color=label_color)

    fig.suptitle('Reconstructed 3-D Meshes — Lambertian Gray (front view)',
                  fontsize=12, fontweight='bold')
    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 11. Per-image gallery  (photo | uncertainty heatmap on mesh)
# ══════════════════════════════════════════════════════════════════════════════

def plot_image_uncertainty_row(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    uncertainty: Optional[np.ndarray],
    save_path: str,
    colormap: str = _MESH_CMAP,
    title: str = '',
    uncertainty_label: str = 'TTA Uncertainty',
) -> None:
    """
    Two- or three-panel figure for a single dataset image.

    Layout when uncertainty is provided (3 panels)
    -----------------------------------------------
    Input Image  |  Reconstruction (depth)  |  Uncertainty heatmap

    Layout when uncertainty is None (2 panels)
    ------------------------------------------
    Input Image  |  Reconstruction (depth)

    Showing the depth panel alongside the uncertainty panel gives the viewer
    anatomical context: the depth map makes the face structure legible (nose
    bump, eye sockets), which otherwise disappears when uncertainty coloring
    happens to be dark at the face centre.

    Parameters
    ----------
    image             : (H, W, 3) uint8 face photo
    vertices          : (5023, 3) FLAME vertices predicted from *this* image
    faces             : (F, 3) FLAME topology
    uncertainty       : (5023,) or (5023, 1) per-vertex uncertainty, or None
    save_path         : output PNG path
    uncertainty_label : panel title for the uncertainty column
    """
    has_unc = uncertainty is not None
    n_panels = 3 if has_unc else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4),
                              layout='constrained')
    axes = list(axes) if n_panels > 1 else [axes, axes]
    if title:
        fig.suptitle(title, fontsize=9)

    # ── Photo ────────────────────────────────────────────────────────────────
    axes[0].imshow(np.asarray(image, dtype=np.uint8))
    axes[0].set_axis_off()
    axes[0].set_title('Input Image', fontsize=10, fontweight='bold')

    # ── Gray mesh — anatomical reference with Lambertian shading ─────────────
    verts = np.asarray(vertices, dtype=np.float64)
    verts = verts - verts.mean(axis=0, keepdims=True)
    _draw_gray_mesh_view(axes[1], verts, faces, azim=0.0, elev=10.0, bg='#b0b0b0')
    axes[1].set_title('Reconstruction', fontsize=10, fontweight='bold')

    # ── Uncertainty heatmap ───────────────────────────────────────────────────
    if has_unc:
        scalars = np.asarray(uncertainty, dtype=np.float64).ravel()
        vmin = float(np.percentile(scalars, 2))
        vmax = float(np.percentile(scalars, 98))
        if vmin == vmax:
            vmax = vmin + 1e-8
        _draw_mesh_view(axes[2], verts, faces, scalars, colormap,
                         vmin, vmax, azim=0.0, elev=10.0)
        axes[2].set_title(uncertainty_label, fontsize=10, fontweight='bold')
        sm = plt.cm.ScalarMappable(cmap=colormap,
                                    norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[2], shrink=0.8, pad=0.02,
                      aspect=20).ax.tick_params(labelsize=7)

    _save(fig, save_path)


def plot_per_image_uncertainty_gallery(
    rows: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    faces: np.ndarray,
    save_path: str,
    model_name: str = '',
    colormap: str = _MESH_CMAP,
    max_images: int = 10,
) -> None:
    """
    Combined gallery: one row per dataset image, two panels per row.

    Layout
    ------
    ┌─────────────────┬──────────────────────────────┐
    │  Input Image    │  FLAME mesh (TTA uncertainty) │  ← row 0
    ├─────────────────┼──────────────────────────────┤
    │  …              │  …                            │  ← row 1 … N
    └─────────────────┴──────────────────────────────┘

    All mesh panels share the same colourbar scale so per-image magnitudes
    are directly comparable.  Faces with higher average uncertainty appear
    brighter (plasma colormap: dark-purple = low, yellow = high).

    Parameters
    ----------
    rows      : list of (image, vertices, uncertainty) — one element per image.
                ``image``:       (H, W, 3) uint8 face photo
                ``vertices``:    (5023, 3) float32 predicted for THIS image
                ``uncertainty``: (5023,) float32, or None
    faces     : (F, 3) FLAME face topology (shared)
    save_path : output PNG path
    model_name: label for the plot title
    max_images: maximum rows in the gallery
    """
    rows = rows[:max_images]
    n = len(rows)
    if n == 0:
        return

    # Always show 3 columns: photo | depth | uncertainty (or just depth if no unc).
    has_any_unc = any(unc is not None for _, _, unc in rows)
    ncols = 3 if has_any_unc else 2
    col_w = 3.0
    fig, axes = plt.subplots(n, ncols, figsize=(col_w * ncols, 3.8 * n),
                              squeeze=False, layout='constrained')

    tag = f'  [{model_name}]' if model_name else ''
    unc_tag = ', TTA' if has_any_unc else ''
    fig.suptitle(f'Per-Image Gallery: Photo  |  Depth  |  Uncertainty{tag}{unc_tag}',
                  fontsize=12, fontweight='bold')

    for ri, (img, verts, unc) in enumerate(rows):
        ax_img   = axes[ri, 0]
        ax_depth = axes[ri, 1]

        ax_img.imshow(np.asarray(img, dtype=np.uint8))
        ax_img.set_axis_off()
        if ri == 0:
            ax_img.set_title('Input Image', fontsize=10, fontweight='bold')

        # Centre vertices per-row so each mesh renders correctly.
        v = np.asarray(verts, dtype=np.float64)
        v = v - v.mean(axis=0, keepdims=True)

        # Gray mesh — Lambertian anatomical reference
        _draw_gray_mesh_view(ax_depth, v, faces, azim=0.0, elev=10.0, bg='#b0b0b0')
        if ri == 0:
            ax_depth.set_title('Reconstruction', fontsize=10, fontweight='bold')

        # Uncertainty panel
        if has_any_unc:
            ax_unc = axes[ri, 2]
            if unc is not None:
                scalars = np.asarray(unc, dtype=np.float64).ravel()
                vmin = float(np.percentile(scalars, 2))
                vmax = float(np.percentile(scalars, 98))
                if vmin == vmax:
                    vmax = vmin + 1e-8
                _draw_mesh_view(ax_unc, v, faces, scalars, colormap,
                                 vmin, vmax, azim=0.0, elev=10.0)
                sm = plt.cm.ScalarMappable(cmap=colormap,
                                            norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                fig.colorbar(sm, ax=ax_unc, shrink=0.7, pad=0.02, aspect=16,
                              format='%.4f').ax.tick_params(labelsize=6)
            else:
                ax_unc.set_facecolor('#f0f0f0')
                ax_unc.text(0.5, 0.5, 'n/a', ha='center', va='center',
                             fontsize=9, color='grey', transform=ax_unc.transAxes)
                ax_unc.set_axis_off()
            if ri == 0:
                ax_unc.set_title('TTA Uncertainty (per-image scale)', fontsize=10,
                                  fontweight='bold')

    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 12a. plot_model_comparison_gallery  (all models × all images)
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison_gallery(
    rows: List[Tuple[np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray]]],
    faces: np.ndarray,
    save_path: str,
    primary_model: str = '',
    colormap: str = 'Blues',
    max_images: int = 10,
) -> None:
    """
    Gallery showing ALL loaded models' reconstructions per dataset image.

    Layout
    ------
    ┌──────┬────────┬────────┬─────────┬───────┐
    │Photo │ SMIRK  │  DECA  │  EMOCA  │ SHeaP │  ← row 0
    ├──────┼────────┼────────┼─────────┼───────┤
    │  …   │   …    │   …    │    …    │   …   │  ← row 1 … N
    └──────┴────────┴────────┴─────────┴───────┘

    Meshes are depth-coloured (Z) so the face anatomy is visible regardless
    of pose or identity.  Models that failed to load are shown as grey blanks.

    Parameters
    ----------
    rows          : list of (image, vertices_dict, primary_uncertainty).
                    ``image``:            (H, W, 3) uint8 face photo
                    ``vertices_dict``:    {model_name: (5023, 3)} — may be partial
                    ``primary_uncertainty``: (5023,) uncertainty array or None
    faces         : (F, 3) FLAME face topology
    save_path     : output PNG path
    primary_model : name of the primary model (underlined title)
    max_images    : maximum rows in the gallery
    """
    rows = rows[:max_images]
    n = len(rows)
    if n == 0:
        return

    all_model_names: List[str] = []
    for _, vd, _ in rows:
        for k in vd:
            if k not in all_model_names:
                all_model_names.append(k)
    if not all_model_names:
        return

    ncols = 1 + len(all_model_names)   # photo + one per model
    col_w = 3.2
    fig, axes = plt.subplots(n, ncols,
                              figsize=(col_w * ncols, 3.8 * n),
                              squeeze=False, layout='constrained')
    fig.suptitle('Per-Image Reconstruction Gallery — All Models (depth colour-coded)',
                  fontsize=12, fontweight='bold')

    for ri, (img, vd, _) in enumerate(rows):
        axes[ri, 0].imshow(np.asarray(img, dtype=np.uint8))
        axes[ri, 0].set_axis_off()
        if ri == 0:
            axes[ri, 0].set_title('Input Image', fontsize=10, fontweight='bold')

        for ci, mname in enumerate(all_model_names, start=1):
            ax = axes[ri, ci]
            if ri == 0:
                style = 'bold' if mname == primary_model else 'normal'
                ax.set_title(mname, fontsize=10, fontweight=style,
                              color=_method_color(mname))
            if mname not in vd:
                ax.set_facecolor('#d8d8d8')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                         fontsize=10, color='#888888', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            verts = np.asarray(vd[mname], dtype=np.float64)
            verts -= verts.mean(axis=0, keepdims=True)
            _draw_gray_mesh_view(ax, verts, faces, azim=0.0, elev=10.0, bg='#b0b0b0')

    fig.suptitle('Per-Image Reconstruction Gallery — All Models (Lambertian shading)',
                  fontsize=12, fontweight='bold')
    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 12b. plot_uncertainty_winner_summary  (best model + best method at a glance)
# ══════════════════════════════════════════════════════════════════════════════

def plot_uncertainty_winner_summary(
    s2m_results: Optional[Dict[str, Dict[str, float]]],
    summary_dict: Optional[Dict[str, Dict[str, float]]],
    save_path: str,
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """
    Two-panel summary answering the two core project questions:

      Left  — "Which FLAME regressor is most accurate?"
               Bar chart of scan-to-mesh median distance per model (↓ better).
               Requires s2m_results.

      Right — "Which uncertainty method best predicts vertex error?"
               Bar chart of Spearman ρ per method (↑ better) with AUSE as
               a secondary label (↓ better).  Requires summary_dict.

    Either panel is omitted if its data is not available.
    """
    has_s2m     = bool(s2m_results)
    has_summary = bool(summary_dict)
    if not has_s2m and not has_summary:
        warnings.warn("[vis] plot_uncertainty_winner_summary: no data to plot.")
        return

    n_panels = has_s2m + has_summary
    fig, raw_axes = plt.subplots(1, n_panels, figsize=figsize, layout='constrained')
    raw_axes_list = [raw_axes] if n_panels == 1 else list(raw_axes)
    fig.suptitle('Summary: Best Model & Best Uncertainty Method',
                  fontsize=13, fontweight='bold')

    ax_iter = iter(raw_axes_list)

    # ── Left: model geometric quality (S2M) ─────────────────────────────────
    if has_s2m:
        ax = next(ax_iter)
        models  = list(s2m_results.keys())
        medians = [float(s2m_results[m].get('median_mm', 0)) for m in models]
        colors  = [_method_color(m) for m in models]
        bars = ax.bar(models, medians, color=colors, alpha=0.85, edgecolor='white', lw=1.2)
        for bar, val in zip(bars, medians):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{val:.2f} mm', ha='center', va='bottom', fontsize=9, fontweight='bold')
        best_model = models[int(np.argmin(medians))]
        ax.set_title(f'Model Geometric Quality (↓ better)\nBest: {best_model}',
                      fontsize=11, fontweight='bold')
        ax.set_ylabel('Scan-to-Mesh Median Distance (mm)', fontsize=10)
        ax.set_ylim(0, max(medians) * 1.25 if medians else 1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', labelsize=10)

    # ── Right: uncertainty method quality (Spearman ρ + AUSE) ───────────────
    if has_summary:
        ax = next(ax_iter)
        methods = [m for m in summary_dict
                   if not np.isnan(float(summary_dict[m].get('spearman_rho', np.nan)))]
        if not methods:
            ax.text(0.5, 0.5, 'No Spearman ρ data\n(needs FLAME GT vertices)',
                     ha='center', va='center', fontsize=11, color='grey',
                     transform=ax.transAxes, wrap=True)
            ax.set_title('Uncertainty Method Quality', fontsize=11, fontweight='bold')
            ax.set_axis_off()
        else:
            rhos   = [float(summary_dict[m].get('spearman_rho', 0)) for m in methods]
            auses  = [float(summary_dict[m].get('ause', np.nan))     for m in methods]
            colors = [_method_color(m) for m in methods]
            bars = ax.bar([_dn(m) for m in methods], rhos, color=colors, alpha=0.85, edgecolor='white', lw=1.2)
            for bar, rho, ause in zip(bars, rhos, auses):
                lbl = f'ρ={rho:.3f}'
                if not np.isnan(ause):
                    lbl += f'\nAUSE={ause:.4f}'
                ax.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.005 if rho >= 0 else bar.get_height() - 0.02,
                         lbl, ha='center', va='bottom', fontsize=7.5)
            best_method = methods[int(np.argmax(rhos))]
            ax.axhline(0, color='black', lw=0.8, linestyle='--')
            ax.set_title(f'Uncertainty Method Quality (Spearman ρ ↑)\nBest: {_dn(best_method)}',
                          fontsize=11, fontweight='bold')
            ax.set_ylabel('Spearman ρ (rank correlation with vertex error)', fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(axis='x', labelsize=9, rotation=20)

    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 13. plot_per_image_rho_distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_per_image_rho_distribution(
    per_image_pairs: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    save_path: str,
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """
    Box plot of the per-image Spearman ρ distribution for each uncertainty method.

    Why this matters
    ----------------
    Reporting a single Spearman ρ computed by flattening all N×5023 pairs into
    one vector hides image-to-image variance.  A method that scores ρ=0.4 on
    average might be reliable (all images near 0.4) or volatile (some images
    at 0.8, others at –0.1).  This plot shows the full distribution.

    Each box covers [Q1, Q3]; the centre line is the median; the whiskers
    extend to 1.5×IQR.  A tight box at high ρ = reliable and accurate method.
    A wide box or large fraction of negative values = unreliable despite decent
    average.  The 'StaticRegion' baseline (if present) shows the non-adaptive
    floor — any method whose median exceeds the baseline's median is genuinely
    input-adaptive.

    Parameters
    ----------
    per_image_pairs : {method_name: list of (error_5023, uncertainty_5023) tuples}
        One tuple per evaluation image.  From _run_batch_dataset_eval.
    """
    from scipy.stats import spearmanr as _spearmanr

    methods = list(per_image_pairs.keys())
    if not methods:
        return

    rho_data = {}
    for method, pairs in per_image_pairs.items():
        rhos = []
        for e_img, u_img in pairs:
            e_ = np.asarray(e_img, dtype=np.float64).ravel()
            u_ = np.asarray(u_img, dtype=np.float64).ravel()
            if len(e_) >= 4:
                r = float(_spearmanr(e_, u_).statistic)
                if not np.isnan(r):
                    rhos.append(r)
        rho_data[method] = rhos

    n_methods = len(methods)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    # Separate StaticRegion visually (dashed box)
    bp_data   = [rho_data[m] for m in methods]
    positions = list(range(n_methods))
    colors    = [_method_color(m) for m in methods]

    bp = ax.boxplot(
        bp_data,
        positions=positions,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color='black', linewidth=2.0),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=3, alpha=0.5),
        showfliers=True,
    )
    for patch, color, method in zip(bp['boxes'], colors, methods):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        if method == 'StaticRegion':
            patch.set_linestyle('--')
            patch.set_linewidth(1.8)

    # Annotate with N (number of images)
    for i, method in enumerate(methods):
        n = len(rho_data[method])
        ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] > -1 else -1,
                f'n={n}', ha='center', va='top', fontsize=7, color='grey')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([_dn(m) for m in methods], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Spearman ρ (per image)', fontsize=10)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(
        'Per-Image Spearman ρ Distribution\n'
        '(tight box at high median = reliable & accurate; '
        'StaticRegion = non-adaptive floor)',
        fontsize=11, fontweight='bold',
    )
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    # Legend entries
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=_method_color(m), alpha=0.65, label=_dn(m))
        for m in methods
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right', ncol=min(4, n_methods))

    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 14. plot_tta_n_ablation
# ══════════════════════════════════════════════════════════════════════════════

def plot_tta_n_ablation(
    n_ause_dict: Dict[str, Dict[int, float]],
    save_path: str,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """
    AUSE vs N (number of stochastic forward passes) convergence plot.

    Motivation
    ----------
    TTA (N=10) and MCD (N=15) use a fixed number of passes chosen without
    rigorous justification.  This plot shows how AUSE changes as N increases
    from 2 to 20+.  A method that has converged by N=10 validates that choice;
    one still falling at N=20 suggests more passes are needed.

    Layout
    ------
    One line per method (e.g., TTA, MCD).  X = N, Y = AUSE (↓ better).
    A dashed horizontal line marks the AUSE at the default N (10 for TTA,
    15 for MCD), annotated with "default N".

    Parameters
    ----------
    n_ause_dict : {method_name: {n_passes: ause_value}}
        Keyed by method name (e.g., 'TTA', 'MCD').
        Inner dict maps number of passes → AUSE at that N.
    """
    if not n_ause_dict:
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize, layout='constrained',
                              gridspec_kw={'width_ratios': [2.5, 1]})
    ax_line, ax_bar = axes

    default_n = {'TTA': 10, 'MCD': 15, 'SOL-MCD': 15, 'A-MCD': 15}

    for method, n_ause in n_ause_dict.items():
        ns   = sorted(n_ause.keys())
        vals = [n_ause[n] for n in ns]
        color = _method_color(method)

        ax_line.plot(ns, vals, 'o-', color=color, linewidth=2, markersize=6,
                     label=f'{method}')

        d_n = default_n.get(method, None)
        if d_n is not None and d_n in n_ause:
            ax_line.axvline(d_n, color=color, linestyle=':', alpha=0.5, linewidth=1)
            ax_line.annotate(
                f'default N={d_n}',
                xy=(d_n, n_ause[d_n]),
                xytext=(d_n + 0.5, n_ause[d_n] * 1.05),
                fontsize=7, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
            )

    ax_line.set_xlabel('Number of stochastic passes (N)', fontsize=10)
    ax_line.set_ylabel('AUSE ↓ (lower = better)', fontsize=10)
    ax_line.set_title('AUSE vs N — Convergence Ablation\n'
                       '(flat curve from N=10 onwards validates the default choice)',
                       fontsize=11, fontweight='bold')
    ax_line.legend(fontsize=9)
    ax_line.spines[['top', 'right']].set_visible(False)
    ax_line.grid(alpha=0.25, linestyle='--')

    # Right panel: AUSE at final N (highest N tested)
    final_auses = {}
    for method, n_ause in n_ause_dict.items():
        max_n = max(n_ause.keys())
        final_auses[method] = n_ause[max_n]

    methods_sorted = sorted(final_auses, key=lambda m: final_auses[m])
    vals   = [final_auses[m] for m in methods_sorted]
    colors = [_method_color(m) for m in methods_sorted]
    bars = ax_bar.barh(range(len(methods_sorted)), vals,
                        color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax_bar.set_yticks(range(len(methods_sorted)))
    ax_bar.set_yticklabels(methods_sorted, fontsize=9)
    ax_bar.set_xlabel('AUSE at max N ↓', fontsize=9)
    ax_bar.set_title('Final\nAUSE', fontsize=10, fontweight='bold')
    ax_bar.spines[['top', 'right']].set_visible(False)
    ax_bar.grid(axis='x', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, vals):
        ax_bar.text(val + max(vals) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f'{val:.4f}', va='center', fontsize=7)

    fig.suptitle('Stochastic Pass Count Ablation (AUSE vs N)',
                  fontsize=13, fontweight='bold')
    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 15. plot_paper_style_comparison_panel
# ══════════════════════════════════════════════════════════════════════════════

def plot_paper_style_comparison_panel(
    rows: List[Tuple[np.ndarray, Dict[str, np.ndarray]]],
    faces: np.ndarray,
    save_path: str,
    model_order: Optional[List[str]] = None,
    max_images: int = 6,
    dpi: int = 200,
) -> None:
    """
    Paper-style comparison panel: input photo beside Lambertian-shaded mesh
    reconstructions for every loaded FLAME regressor.

    Layout (matches the style in DECA / EMOCA / SHeaP papers)
    -----------------------------------------------------------
    ┌──────────┬────────┬────────┬─────────┬───────┐
    │  Input   │ SMIRK  │  DECA  │  EMOCA  │ SHeaP │  ← row 0
    ├──────────┼────────┼────────┼─────────┼───────┤
    │  …       │   …    │   …    │    …    │   …   │  ← row 1 … N
    └──────────┴────────┴────────┴─────────┴───────┘

    Visual style differences from the default ``plot_model_comparison_gallery``:
    * Dark background (#3a3a3a) — face shape "pops" against the dark field,
      matching the offscreen-renderer look in the published papers.
    * Stronger Lambertian contrast (ambient=0.30, diffuse=0.80) — more dramatic
      highlights on the nose bridge and forehead, darker side shadowing.
    * Higher figure DPI (default 200) for print-quality output.
    * Front view only — removes the per-view overhead; the single front view is
      what appears in every paper comparison figure.
    * Column widths equal to the photo column — square-ish mesh panels.

    Parameters
    ----------
    rows        : list of (image, vertices_dict) tuples.
                  ``image``        : (H, W, 3) uint8 input photo.
                  ``vertices_dict``: {model_name: (5023, 3)} predicted meshes.
                  The dict may be partial; missing models are shown as blanks.
    faces       : (F, 3) shared FLAME face topology.
    save_path   : output PNG path.
    model_order : explicit ordering of model columns; defaults to the order
                  first seen in rows.
    max_images  : cap on the number of rows rendered (default 6).
    dpi         : output DPI (default 200).
    """
    rows = rows[:max_images]
    n = len(rows)
    if n == 0:
        return

    # Collect model names in a consistent order
    if model_order is not None:
        all_models = model_order
    else:
        all_models = []
        for _, vd in rows:
            for k in vd:
                if k not in all_models:
                    all_models.append(k)
    if not all_models:
        return

    ncols   = 1 + len(all_models)     # photo + one per model
    col_w   = 2.8                      # inches per column
    row_h   = 3.0                      # inches per row
    _DARK_BG = '#3a3a3a'               # dark panel background (paper style)

    fig, axes = plt.subplots(
        n, ncols,
        figsize=(col_w * ncols, row_h * n),
        squeeze=False,
        layout='constrained',
    )
    fig.patch.set_facecolor('#1a1a1a')   # outer figure background

    for ri, (img, vd) in enumerate(rows):
        # ── Photo ──────────────────────────────────────────────────────────────
        ax_ph = axes[ri, 0]
        ax_ph.imshow(np.asarray(img, dtype=np.uint8))
        ax_ph.set_axis_off()
        ax_ph.set_facecolor('#1a1a1a')
        if ri == 0:
            ax_ph.set_title('Input', fontsize=9, color='white', fontweight='bold', pad=3)

        # ── Model columns ──────────────────────────────────────────────────────
        for ci, mname in enumerate(all_models, start=1):
            ax = axes[ri, ci]
            if ri == 0:
                ax.set_title(mname, fontsize=9, color='white', fontweight='bold', pad=3)
            if mname not in vd or vd[mname] is None:
                ax.set_facecolor(_DARK_BG)
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        fontsize=9, color='#888888', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            verts = np.asarray(vd[mname], dtype=np.float64)
            verts -= verts.mean(axis=0, keepdims=True)
            # Stronger contrast: lower ambient, higher diffuse → more dramatic
            _draw_gray_mesh_view(
                ax, verts, faces,
                azim=0.0, elev=10.0,
                bg=_DARK_BG,
                ambient=0.30,
                diffuse=0.80,
            )

    fig.suptitle(
        'Reconstructed 3-D Face Meshes — Lambertian Shading (front view)',
        fontsize=11, fontweight='bold', color='white',
    )
    _save(fig, save_path, dpi=dpi)


# ══════════════════════════════════════════════════════════════════════════════
# 16. plot_method_correlation_matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_method_correlation_matrix(
    uncertainty_dict: Dict[str, np.ndarray],
    save_path: str,
    figsize: Tuple[int, int] = (8, 7),
) -> None:
    """
    Spearman rank-correlation matrix between all pairs of uncertainty methods.

    Why this matters
    ----------------
    If two methods are highly correlated (ρ ≈ 1) they are effectively measuring
    the same phenomenon: reporting both adds no new information.  If they are
    nearly uncorrelated (ρ ≈ 0) or anti-correlated they capture different
    aspects of uncertainty, making them genuinely complementary.

    This matrix tells us:
    * Which methods can be safely dropped without information loss (ρ > 0.9).
    * Which pairs are complementary and could usefully be combined (ρ < 0.5).
    * Whether the MCD variants (MCD, SOL-MCD, A-MCD) collapse to a single
      signal or remain distinct.

    Parameters
    ----------
    uncertainty_dict : {method_name: (5023,) or (5023,1) uncertainty array}
        All arrays are flattened and rank-normalised before correlation.
    save_path : output PNG path.
    """
    names = list(uncertainty_dict.keys())
    n = len(names)
    if n < 2:
        warnings.warn("[vis] plot_method_correlation_matrix: need ≥ 2 methods.")
        return

    # Rank-normalise each method to [0,1] to make the comparison unit-free
    vecs: List[np.ndarray] = []
    for name in names:
        u = np.asarray(uncertainty_dict[name], dtype=np.float64).ravel()
        rk = np.argsort(np.argsort(u)).astype(np.float64)
        vecs.append(rk / max(len(rk) - 1, 1))

    # Build Spearman ρ matrix
    rho = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = vecs[i], vecs[j]
            xi_c = xi - xi.mean(); yi_c = yi - yi.mean()
            denom = np.linalg.norm(xi_c) * np.linalg.norm(yi_c)
            r = float(xi_c @ yi_c / denom) if denom > 1e-12 else float('nan')
            rho[i, j] = rho[j, i] = r

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')

    # Diverging colourmap: blue = negative, white = zero, red = positive
    im = ax.imshow(rho, cmap='RdBu_r', vmin=-1.0, vmax=1.0, aspect='auto')

    # Annotate each cell
    for i in range(n):
        for j in range(n):
            val = rho[i, j]
            txt_color = 'white' if abs(val) > 0.65 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=txt_color, fontweight='bold')

    disp_names = [_dn(nm) for nm in names]
    ax.set_xticks(range(n)); ax.set_xticklabels(disp_names, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(disp_names, fontsize=9)
    ax.set_title(
        'Uncertainty Method Correlation Matrix\n'
        'Spearman ρ (rank correlation of per-vertex values)\n'
        'ρ ≈ 1 = redundant  |  ρ ≈ 0 = complementary  |  ρ < 0 = anti-correlated',
        fontsize=10, fontweight='bold', pad=10,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label='Spearman ρ')
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 12. create_full_analysis_report
# ══════════════════════════════════════════════════════════════════════════════

def create_full_analysis_report(
    output_dir: str,
    vertices_dict: Optional[Dict[str, np.ndarray]] = None,
    faces: Optional[np.ndarray] = None,
    uncertainty_dict: Optional[Dict[str, np.ndarray]] = None,
    error_dict: Optional[Dict[str, np.ndarray]] = None,
    summary_dict: Optional[Dict[str, Dict[str, float]]] = None,
    sparsification_dict: Optional[Dict[str, Dict]] = None,
    calibration_dict: Optional[Dict[str, Dict]] = None,
    region_errors_dict: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    s2m_results: Optional[Dict[str, Dict[str, float]]] = None,
    primary_error: Optional[np.ndarray] = None,
    gt_vertices: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None,
    per_image_pairs: Optional[Dict[str, List]] = None,
    n_ablation_ause: Optional[Dict[str, Dict[int, float]]] = None,
) -> Dict[str, str]:
    """
    Run all available visualisations and save them to *output_dir*.

    Each plot is generated only when its required inputs are present — missing
    data does not cause an error, just a skip.  The returned dict maps plot
    names to absolute file paths for logging.

    Typical call from main.py
    -------------------------
    ::

        paths = create_full_analysis_report(
            output_dir        = os.path.join(args.output_dir, 'plots'),
            vertices_dict     = mesh_results,
            faces             = flame_faces,
            uncertainty_dict  = uncertainty_results,
            error_dict        = {'SMIRK': smirk_error},
            summary_dict      = compare_uncertainty_methods(smirk_error, uncertainty_results),
            sparsification_dict = sparse_dict,
            calibration_dict    = calib_dict,
            region_errors_dict  = region_dict,
            primary_error       = smirk_error,
            gt_vertices         = gt_verts,
            image               = rgb_image,
        )

    Parameters
    ----------
    output_dir        : root directory for all saved plots
    vertices_dict     : {model_name: (5023, 3)} — for mesh renders & reconstruction
    faces             : (F, 3) FLAME face topology
    uncertainty_dict  : {method_name: (5023,)} uncertainty estimates
    error_dict        : {model_name: (5023,)} per-vertex geometric errors
    summary_dict      : output of compare_uncertainty_methods()
    sparsification_dict: per-method output of calculate_sparsification_error_curve()
    calibration_dict  : per-method output of calculate_uncertainty_calibration()
    region_errors_dict: per-method output of calculate_region_wise_geometric_error()
    s2m_results       : {model_name: {"median_mm", "mean_mm", "p90_mm"}} scan-to-mesh
                        distances for NoW-style raw-scan GT (None if not available)
    primary_error     : (5023,) error array for scatter / violin reference panels
    gt_vertices       : (5023, 3) ground-truth vertices for reconstruction comparison
    image             : (H, W, 3) original input image (saved as reference)

    Returns
    -------
    saved_paths : {plot_name: absolute_file_path}
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: Dict[str, str] = {}

    def _p(name: str) -> str:
        return os.path.join(output_dir, name)

    # Pick reference vertices for mesh rendering (first prediction or GT)
    ref_verts: Optional[np.ndarray] = None
    if vertices_dict:
        ref_verts = next(iter(vertices_dict.values()))
    elif gt_vertices is not None:
        ref_verts = gt_vertices

    # ── 1. Per-method uncertainty heatmaps ───────────────────────────────────
    if faces is not None and uncertainty_dict and ref_verts is not None:
        hm_dir = _p('heatmaps')
        os.makedirs(hm_dir, exist_ok=True)
        for method, u in uncertainty_dict.items():
            p = os.path.join(hm_dir, f'{method}_heatmap.png')
            try:
                render_uncertainty_heatmap(
                    ref_verts, faces, u, p,
                    title=f'Uncertainty Heatmap — {_dn(method)}',
                )
                saved[f'heatmap_{method}'] = p
            except Exception as exc:
                warnings.warn(f"[vis] Heatmap for '{method}' failed: {exc}")

    # ── 2. Side-by-side spatial maps ──────────────────────────────────────────
    if faces is not None and uncertainty_dict and ref_verts is not None:
        p = _p('uncertainty_spatial_maps.png')
        try:
            plot_uncertainty_spatial_maps(ref_verts, faces, uncertainty_dict, p)
            saved['spatial_maps'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Spatial maps failed: {exc}")

    # ── 3. Violin comparison ──────────────────────────────────────────────────
    if uncertainty_dict:
        p = _p('uncertainty_violin.png')
        try:
            plot_uncertainty_comparison_violin(
                uncertainty_dict, p, error=primary_error)
            saved['violin'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Violin plot failed: {exc}")

    # ── 4. Sparsification curves ──────────────────────────────────────────────
    if sparsification_dict:
        p = _p('sparsification_curves.png')
        try:
            plot_sparsification_curves(sparsification_dict, p)
            saved['sparsification'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Sparsification failed: {exc}")

    # ── 5. Calibration diagram ────────────────────────────────────────────────
    if calibration_dict:
        p = _p('calibration_diagram.png')
        try:
            plot_calibration_diagram(calibration_dict, p)
            saved['calibration'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Calibration failed: {exc}")

    # ── 6. Error-uncertainty scatter ──────────────────────────────────────────
    if primary_error is not None and uncertainty_dict:
        p = _p('uncertainty_vs_error_scatter.png')
        rho_dict: Dict[str, float] = {}
        if summary_dict:
            for m, mets in summary_dict.items():
                if m in uncertainty_dict:
                    rho_dict[m] = float(mets.get('spearman_rho', np.nan))
        try:
            plot_uncertainty_vs_error_scatter(
                primary_error, uncertainty_dict, p,
                spearman_rho_dict=rho_dict or None,
            )
            saved['scatter'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Scatter plot failed: {exc}")

    # ── 7. Method comparison table ────────────────────────────────────────────
    if summary_dict:
        p = _p('method_comparison_table.png')
        try:
            plot_method_comparison_table(summary_dict, p)
            saved['comparison_table'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Comparison table failed: {exc}")

    # ── 8. Region-wise error breakdown ────────────────────────────────────────
    if region_errors_dict:
        p = _p('region_error_breakdown.png')
        try:
            plot_region_error_breakdown(region_errors_dict, p)
            saved['region_breakdown'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Region breakdown failed: {exc}")

    # ── 8b. Scan-to-mesh comparison (NoW-style raw-scan GT) ───────────────────
    if s2m_results:
        p = _p('scan_to_mesh_comparison.png')
        try:
            plot_scan_to_mesh_comparison(s2m_results, p)
            saved['scan_to_mesh'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Scan-to-mesh plot failed: {exc}")

    # ── 8c. Sharpness facets comparison ───────────────────────────────────────
    if summary_dict:
        p = _p('sharpness_comparison.png')
        try:
            plot_sharpness_comparison(summary_dict, p)
            saved['sharpness_comparison'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Sharpness comparison failed: {exc}")

    # ── 9. Error distributions ────────────────────────────────────────────────
    if error_dict:
        p = _p('error_distributions.png')
        try:
            plot_error_distributions(error_dict, p)
            saved['error_distributions'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Error distributions failed: {exc}")

    # ── 10. Multi-model reconstruction ────────────────────────────────────────
    if faces is not None and vertices_dict and len(vertices_dict) >= 1:
        p = _p('multi_model_reconstruction.png')
        try:
            plot_multi_model_reconstruction(
                vertices_dict, faces, p,
                gt_vertices=gt_vertices, image=image)
            saved['reconstruction'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Reconstruction failed: {exc}")

    # ── 11. Input image reference ─────────────────────────────────────────────
    if image is not None:
        p = _p('input_image.png')
        try:
            fig, ax = plt.subplots(figsize=(3, 4))
            ax.imshow(image)
            ax.set_axis_off()
            ax.set_title('Input Image', fontsize=10)
            _save(fig, p)
            saved['input_image'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Input image save failed: {exc}")

    # ── 12. Summary: best model + best uncertainty method ─────────────────────
    if s2m_results or summary_dict:
        p = _p('summary_winner.png')
        try:
            plot_uncertainty_winner_summary(s2m_results, summary_dict, p)
            saved['summary_winner'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Winner summary failed: {exc}")

    # ── 13. Per-image Spearman ρ distribution (box plot) ─────────────────────
    if per_image_pairs:
        p = _p('per_image_rho_distribution.png')
        try:
            plot_per_image_rho_distribution(per_image_pairs, p)
            saved['per_image_rho'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Per-image ρ distribution failed: {exc}")

    # ── 14. TTA / MCD N ablation: AUSE vs N ──────────────────────────────────
    if n_ablation_ause:
        p = _p('n_ablation_ause.png')
        try:
            plot_tta_n_ablation(n_ablation_ause, p)
            saved['n_ablation'] = p
        except Exception as exc:
            warnings.warn(f"[vis] N ablation plot failed: {exc}")

    # ── 15. Paper-style comparison panel ─────────────────────────────────────
    if faces is not None and vertices_dict and image is not None:
        p = _p('paper_style_comparison.png')
        try:
            plot_paper_style_comparison_panel(
                [(image, vertices_dict)], faces, p, max_images=1,
            )
            saved['paper_comparison'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Paper-style panel failed: {exc}")

    # ── 16. Uncertainty method correlation matrix ─────────────────────────────
    if uncertainty_dict and len(uncertainty_dict) >= 2:
        p = _p('method_correlation_matrix.png')
        try:
            plot_method_correlation_matrix(uncertainty_dict, p)
            saved['method_correlation'] = p
        except Exception as exc:
            warnings.warn(f"[vis] Correlation matrix failed: {exc}")

    print(f"[vis] Report complete — {len(saved)} plots saved to {output_dir}/")
    for name, path in sorted(saved.items()):
        print(f"      {name:35s} → {os.path.relpath(path)}")

    return saved
