"""
Parse a Stage-5 downstream-classifier training log (e.g. figures/downstream.txt)
and produce plain-vs-uncertainty-weighted classifier comparison plots.

The log is produced by main.py's `run_downstream()` / `_train_one_variant()`.
Per combo (model x uncertainty method, e.g. "SMIRK_tta") it trains one PLAIN
classifier and five WEIGHTED classifiers (one per fusion mode: input,
patch_embed, attn_bias, key_scale, value_scale), each for the same number of
epochs (a frozen-backbone "Stage 1" warm-up followed by a partial fine-tune
"Stage 2", numbered on one continuous global-epoch axis).

Only combos that have actual epoch-level training data in the log are used --
a combo whose log block stops during confidence-map precomputation (no
"Training PLAIN classifier" line reached) is silently skipped.

Metrics available per epoch, per classifier:
  - train_acc, train_loss  (from the last per-batch line of that epoch --
    per-batch losses/accuracies are running averages within the epoch, so the
    final batch's value equals the epoch's aggregate, verified against the
    "[S1]/[S2] Epoch ..." summary line's own train=% figure).
  - test_acc               (printed once per epoch on the summary line).
  - test_loss is NEVER printed anywhere in this log -- only test accuracy.
    Loss plots (e, f) are therefore training loss only; this is intentional,
    not a bug, and is labelled as such on every loss plot.

Usage:
    python3 scripts/plot_downstream_results.py \
        --log figures/downstream.txt \
        --out figures/downstream_plots
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# --------------------------------------------------------------------------
# Palette (validated categorical set, light-surface, fixed slot order --
# see dataviz skill references/palette.md). Plain takes slot 6 (red): it's
# the reference baseline in every plot, so it needs to be the most vivid,
# most immediately-findable color -- and red is maximally distinct in hue
# from the cool blue/aqua/green/violet cluster used by the five fusion
# modes below, so it never gets lost among them.
# --------------------------------------------------------------------------
PLAIN_COLOR = "#e34948"  # red    (slot 6)
MUTED = "#898781"        # gridlines / secondary
WEIGHTED_AVG = "#2a78d6"  # combined "Weighted" series in the 2-way plots (a,c,e)

FUSION_MODES = ["input", "patch_embed", "attn_bias", "key_scale", "value_scale"]
FUSION_COLORS = {
    "input": "#2a78d6",        # blue   (slot 1)
    "patch_embed": "#1baf7a",  # aqua   (slot 2)
    "attn_bias": "#eda100",    # yellow (slot 3)
    "key_scale": "#008300",    # green  (slot 4)
    "value_scale": "#4a3aa7",  # violet (slot 5)
}

# Longest-first so "sol_mcd" isn't mis-split as "mcd" + "_sol" etc.
KNOWN_METHODS = ["sol_mcd", "mahalanobis", "jacobian", "amcd", "mcd", "tta"]

# Fixed display/color order for METHOD comparison plots -- grouped the same
# way the paper's own taxonomy does (aleatoric: tta, jacobian, mahalanobis;
# epistemic: mcd, sol_mcd, amcd), not alphabetically. A method's color is
# always the same slot regardless of which subset of methods is present in
# a given log, so a reader can't be misled by color reuse across figures.
CANONICAL_METHOD_ORDER = ["tta", "jacobian", "mahalanobis", "mcd", "sol_mcd", "amcd"]
METHOD_COLORS = {
    "tta": "#2a78d6",         # blue    (slot 1)
    "jacobian": "#1baf7a",    # aqua    (slot 2)
    "mahalanobis": "#eda100",  # yellow (slot 3)
    "mcd": "#008300",         # green   (slot 4)
    "sol_mcd": "#4a3aa7",     # violet  (slot 5)
    "amcd": "#e87ba4",        # magenta (slot 7)
}


def ordered_methods(methods) -> list[str]:
    """Canonical taxonomy order, filtered to whichever methods are present."""
    return [m for m in CANONICAL_METHOD_ORDER if m in methods]

BATCH_RE = re.compile(
    r"\[(?P<tag>[A-Za-z0-9_]+)\]\[S(?P<stage>\d)\s+e(?P<epoch>\d+)\]\s+"
    r"b(?P<batch>\d+)/(?P<nbatches>\d+)\s+loss=(?P<loss>[\d.]+)\s+acc=(?P<acc>[\d.]+)%"
)
EPOCH_RE = re.compile(
    r"\[S(?P<stage>\d)\]\s+Epoch\s+(?P<ep>\d+)/(?P<total>\d+)"
    r"(?:\s+\(global\s+(?P<global>\d+)/(?P<gtotal>\d+)\))?\s+"
    r"train=(?P<train>[\d.]+)%\s+test=(?P<test>[\d.]+)%"
)


def split_tag(tag: str):
    """'SMIRK_tta_plain' -> ('tta', 'plain'); 'SMIRK_tta_key_scale_weighted' -> ('tta', 'key_scale')."""
    if not tag.startswith("SMIRK_"):
        raise ValueError(f"Unexpected tag (expected SMIRK_ prefix): {tag!r}")
    rest = tag[len("SMIRK_"):]
    if rest.endswith("_plain"):
        return rest[: -len("_plain")], "plain"
    if rest.endswith("_weighted"):
        body = rest[: -len("_weighted")]
        for m in KNOWN_METHODS:
            if body == m:
                raise ValueError(f"Weighted tag with no fusion mode: {tag!r}")
            if body.startswith(m + "_"):
                return m, body[len(m) + 1:]
        raise ValueError(f"Could not identify method in weighted tag body {body!r}")
    raise ValueError(f"Unrecognised tag suffix: {tag!r}")


def parse_log(path: Path):
    """
    Returns: data[method][variant] -> sorted list of per-epoch dicts:
        {'global_epoch': int, 'stage': int, 'train_acc': float,
         'train_loss': float, 'test_acc': float}
    variant is 'plain' or one of FUSION_MODES.
    """
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    pending: dict[str, dict] = {}
    current_tag = None

    with open(path, "r", errors="replace") as f:
        for line in f:
            mb = BATCH_RE.search(line)
            if mb:
                tag = mb.group("tag")
                stage = int(mb.group("stage"))
                batch = int(mb.group("batch"))
                nbatches = int(mb.group("nbatches"))
                current_tag = tag
                if batch == nbatches:
                    pending[tag] = {
                        "stage": stage,
                        "local_epoch": int(mb.group("epoch")),
                        "train_acc": float(mb.group("acc")),
                        "train_loss": float(mb.group("loss")),
                    }
                continue

            me = EPOCH_RE.search(line)
            if me and current_tag is not None:
                stage = int(me.group("stage"))
                local_ep = int(me.group("ep"))
                global_ep = int(me.group("global")) if me.group("global") else local_ep
                test_acc = float(me.group("test"))

                pe = pending.get(current_tag)
                if pe is None or pe["stage"] != stage or pe["local_epoch"] != local_ep:
                    # Summary line we can't confidently attribute to a batch
                    # block (e.g. an early-stop line) -- skip rather than guess.
                    continue

                method, variant = split_tag(current_tag)
                data[method][variant].append({
                    "global_epoch": global_ep,
                    "stage": stage,
                    "train_acc": pe["train_acc"],
                    "train_loss": pe["train_loss"],
                    "test_acc": test_acc,
                })

    for method, variants in data.items():
        for key, records in variants.items():
            records.sort(key=lambda r: r["global_epoch"])
    return data


def usable_methods(data) -> list[str]:
    """A combo is usable only if it has a plain series and >=1 fusion series."""
    ok = []
    for method, variants in data.items():
        if "plain" in variants and any(fm in variants for fm in FUSION_MODES):
            ok.append(method)
    return sorted(ok)


def best_test_acc(records: list[dict]) -> float:
    """Highest test accuracy reached across all logged epochs."""
    return max(r["test_acc"] for r in records)


def loss_at_best_epoch(records: list[dict]) -> float:
    """Training loss AT the epoch that produced the best test accuracy -- i.e.
    the loss of the checkpoint that's actually reported/deployed, not just
    whatever the last epoch happened to log (training can regress after its
    peak epoch, and the pipeline reloads the best checkpoint, not the last
    one -- see '[S1→S2] Reloaded best Stage-1 checkpoint' in the log)."""
    return max(records, key=lambda r: r["test_acc"])["train_loss"]


def stack_series(list_of_record_lists: list[list[dict]], field: str):
    """
    Align multiple per-epoch record lists on a common global-epoch axis.
    Returns (epochs: sorted np.ndarray, arr: (n_series, n_epochs) with NaN
    for missing epochs).
    """
    all_epochs = sorted({r["global_epoch"] for recs in list_of_record_lists for r in recs})
    arr = np.full((len(list_of_record_lists), len(all_epochs)), np.nan)
    for i, recs in enumerate(list_of_record_lists):
        m = {r["global_epoch"]: r[field] for r in recs}
        for j, ep in enumerate(all_epochs):
            if ep in m:
                arr[i, j] = m[ep]
    return np.array(all_epochs), arr


# --------------------------------------------------------------------------
# Shared plot chrome
# --------------------------------------------------------------------------

def _style_axes(ax, ylabel: str, title: str):
    """Apply the shared title/label/grid/spine styling used by every line plot."""
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, color=MUTED, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _mark_stage_boundary(ax, n_warmup_epochs: float = 3.5):
    """Draw a dotted vertical line + label where Stage 1 (frozen warm-up) ends
    and Stage 2 (partial fine-tune) begins on the shared global-epoch axis."""
    ax.axvline(n_warmup_epochs, color=MUTED, linestyle=":", linewidth=1)
    ymin, ymax = ax.get_ylim()
    ax.text(n_warmup_epochs, ymax, " Stage 1→2", fontsize=8, color=MUTED,
             va="top", ha="left", style="italic")


def _plot_shaded_line(ax, epochs, arr, label, color, linestyle="-", shade=True):
    """Plot the per-epoch mean of `arr` (n_series, n_epochs) with an optional
    min-max shaded band showing the spread across the stacked series."""
    avg = np.nanmean(arr, axis=0)
    ax.plot(epochs, avg, label=label, color=color, linewidth=2, linestyle=linestyle)
    if shade and arr.shape[0] > 1:
        mn = np.nanmin(arr, axis=0)
        mx = np.nanmax(arr, axis=0)
        ax.fill_between(epochs, mn, mx, color=color, alpha=0.15, linewidth=0)


def _bar_with_err(ax, labels, values, errs, colors):
    """Bar chart with error bars and a value label printed above each bar."""
    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=errs, capsize=4, color=colors,
                   edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20 if len(labels) > 3 else 0, ha="right" if len(labels) > 3 else "center")
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, color=MUTED, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --------------------------------------------------------------------------
# Plot builders (a)-(f) -- each takes the set of methods to include, so the
# same functions serve both the "all versions combined" and "single version"
# cases requested in (g).
# --------------------------------------------------------------------------

def plot_a_bar_overall(data, methods, out_path, title_suffix=""):
    """(a) Single bar pair: Plain vs. Weighted, each averaged over all `methods`
    (and, for Weighted, over all five fusion modes)."""
    plain_vals = [best_test_acc(data[m]["plain"]) for m in methods]
    weighted_vals = [best_test_acc(data[m][fm])
                      for m in methods for fm in FUSION_MODES if fm in data[m]]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    means = [np.mean(plain_vals), np.mean(weighted_vals)]
    errs = [np.std(plain_vals) if len(plain_vals) > 1 else 0,
            np.std(weighted_vals) if len(weighted_vals) > 1 else 0]
    _bar_with_err(ax, ["Plain", "Weighted"], means, errs, [PLAIN_COLOR, WEIGHTED_AVG])
    ax.set_ylabel("Best test accuracy (%)")
    ax.set_title(f"Plain vs. Weighted — overall average{title_suffix}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_b_bar_by_fusion(data, methods, out_path, title_suffix=""):
    """(b) One bar per fusion mode (plus Plain), each averaged over `methods`."""
    plain_vals = [best_test_acc(data[m]["plain"]) for m in methods]
    labels = ["Plain"] + FUSION_MODES
    colors = [PLAIN_COLOR] + [FUSION_COLORS[fm] for fm in FUSION_MODES]
    means, errs = [np.mean(plain_vals)], [np.std(plain_vals) if len(plain_vals) > 1 else 0]
    for fm in FUSION_MODES:
        vals = [best_test_acc(data[m][fm]) for m in methods if fm in data[m]]
        means.append(np.mean(vals) if vals else np.nan)
        errs.append(np.std(vals) if len(vals) > 1 else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    _bar_with_err(ax, labels, means, errs, colors)
    ax.set_ylabel("Best test accuracy (%)")
    ax.set_title(f"Plain vs. each weighted fusion mode{title_suffix}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _acc_lines(ax, series_defs, field, mark_stage=True, legend_loc="lower right"):
    """series_defs: list of (label, color, list_of_record_lists)."""
    for label, color, recs_list in series_defs:
        if not recs_list:
            continue
        ep, arr = stack_series(recs_list, field)
        _plot_shaded_line(ax, ep, arr, label, color, "-", shade=len(recs_list) > 1)
    if mark_stage:
        _mark_stage_boundary(ax)
    ax.legend(fontsize=8, loc=legend_loc)


def plot_c_line_acc_all(data, methods, out_path, title_suffix=""):
    """(c) Test/train accuracy vs. global epoch, Plain vs. pooled Weighted
    (all fusion modes merged into one shaded band), averaged over `methods`."""
    plain_recs = [data[m]["plain"] for m in methods]
    weighted_recs = [data[m][fm] for m in methods for fm in FUSION_MODES if fm in data[m]]
    series = [("Plain", PLAIN_COLOR, plain_recs), ("Weighted", WEIGHTED_AVG, weighted_recs)]

    fig, (ax_test, ax_train) = plt.subplots(1, 2, figsize=(13, 5.5))
    _acc_lines(ax_test, series, "test_acc")
    _style_axes(ax_test, "Test accuracy (%)", "Test accuracy")
    _acc_lines(ax_train, series, "train_acc", legend_loc="upper left")
    _style_axes(ax_train, "Train accuracy (%)", "Train accuracy")
    fig.suptitle(f"Accuracy per epoch: Plain vs. Weighted{title_suffix}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_d_line_acc_by_fusion(data, methods, out_path, title_suffix=""):
    """(d) Same as (c) but with each fusion mode plotted as its own line
    instead of being pooled into a single Weighted band."""
    plain_recs = [data[m]["plain"] for m in methods]
    series = [("Plain", PLAIN_COLOR, plain_recs)]
    for fm in FUSION_MODES:
        recs = [data[m][fm] for m in methods if fm in data[m]]
        series.append((fm, FUSION_COLORS[fm], recs))

    fig, (ax_test, ax_train) = plt.subplots(1, 2, figsize=(15, 6))
    _acc_lines(ax_test, series, "test_acc")
    _style_axes(ax_test, "Test accuracy (%)", "Test accuracy")
    _acc_lines(ax_train, series, "train_acc", legend_loc="upper left")
    _style_axes(ax_train, "Train accuracy (%)", "Train accuracy")
    fig.suptitle(f"Accuracy per epoch by fusion mode{title_suffix}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_e_line_loss_all(data, methods, out_path, title_suffix=""):
    """(e) Training loss vs. global epoch, Plain vs. pooled Weighted — the
    loss counterpart of (c). No test loss is available (see module docstring)."""
    plain_recs = [data[m]["plain"] for m in methods]
    weighted_recs = [data[m][fm] for m in methods for fm in FUSION_MODES if fm in data[m]]

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color, recs_list in [("Plain", PLAIN_COLOR, plain_recs), ("Weighted", WEIGHTED_AVG, weighted_recs)]:
        if not recs_list:
            continue
        ep, arr = stack_series(recs_list, "train_loss")
        _plot_shaded_line(ax, ep, arr, label, color, "-", shade=len(recs_list) > 1)
    _mark_stage_boundary(ax)
    ax.legend(fontsize=9, loc="upper right")
    _style_axes(ax, "Training loss", f"Training loss per epoch: Plain vs. Weighted{title_suffix}\n(test loss is not logged by this pipeline)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_f_line_loss_by_fusion(data, methods, out_path, title_suffix=""):
    """(f) Training loss vs. global epoch, one line per fusion mode plus
    Plain — the loss counterpart of (d)."""
    plain_recs = [data[m]["plain"] for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, color, recs_list in [("Plain", PLAIN_COLOR, plain_recs)] + [
        (fm, FUSION_COLORS[fm], [data[m][fm] for m in methods if fm in data[m]]) for fm in FUSION_MODES
    ]:
        if not recs_list:
            continue
        ep, arr = stack_series(recs_list, "train_loss")
        _plot_shaded_line(ax, ep, arr, label, color, "-", shade=len(recs_list) > 1)
    _mark_stage_boundary(ax)
    ax.legend(fontsize=8, loc="upper right")
    _style_axes(ax, "Training loss", f"Training loss per epoch by fusion mode{title_suffix}\n(test loss is not logged by this pipeline)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


ALL_PLOT_FUNCS = [
    ("a_bar_overall", plot_a_bar_overall),
    ("b_bar_by_fusion", plot_b_bar_by_fusion),
    ("c_line_acc_all", plot_c_line_acc_all),
    ("d_line_acc_by_fusion", plot_d_line_acc_by_fusion),
    ("e_line_loss_all", plot_e_line_loss_all),
    ("f_line_loss_by_fusion", plot_f_line_loss_by_fusion),
]


# ==========================================================================
# Method-comparison plots: TTA vs. Jacobian vs. Mahalanobis vs. MCD (etc.),
# each method's value being the AVERAGE of its five weighted (fusion-mode)
# classifiers only -- plain is never averaged into a method's own number,
# it only appears as a single grey-red reference line/bar spanning all
# methods, exactly as in the (a)-(f) plots, so the two families of plot
# stay visually and semantically consistent.
# ==========================================================================

def plot_method_bar_accuracy(data, methods, out_path, title_suffix=""):
    """One bar per uncertainty method (averaged over its 5 fusion-mode
    classifiers), plus a single Plain reference bar spanning all methods."""
    methods = ordered_methods(methods)
    labels, means, errs, colors = [], [], [], []
    for m in methods:
        vals = [best_test_acc(data[m][fm]) for fm in FUSION_MODES if fm in data[m]]
        labels.append(m)
        means.append(np.mean(vals))
        errs.append(np.std(vals) if len(vals) > 1 else 0)
        colors.append(METHOD_COLORS[m])

    plain_vals = [best_test_acc(data[m]["plain"]) for m in methods]
    labels.append("Plain\n(reference)")
    means.append(np.mean(plain_vals))
    errs.append(np.std(plain_vals) if len(plain_vals) > 1 else 0)
    colors.append(PLAIN_COLOR)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _bar_with_err(ax, labels, means, errs, colors)
    ax.axvline(len(methods) - 0.5, color=MUTED, linestyle=":", linewidth=1)
    ax.set_ylabel("Best test accuracy (%) — avg. over 5 fusion modes")
    ax.set_title(f"Uncertainty method comparison: weighted-classifier accuracy{title_suffix}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_method_line_accuracy(data, methods, out_path, title_suffix=""):
    """Test/train accuracy vs. epoch, one line per uncertainty method (each
    averaged over its 5 fusion modes), plus a single Plain reference line."""
    methods = ordered_methods(methods)
    series = [(m, METHOD_COLORS[m], [data[m][fm] for fm in FUSION_MODES if fm in data[m]]) for m in methods]
    series.append(("Plain (reference)", PLAIN_COLOR, [data[m]["plain"] for m in methods]))

    fig, (ax_test, ax_train) = plt.subplots(1, 2, figsize=(13, 5.5))
    _acc_lines(ax_test, series, "test_acc")
    _style_axes(ax_test, "Test accuracy (%)", "Test accuracy")
    _acc_lines(ax_train, series, "train_acc", legend_loc="upper left")
    _style_axes(ax_train, "Train accuracy (%)", "Train accuracy")
    fig.suptitle(f"Uncertainty method comparison: accuracy per epoch (avg. over 5 fusion modes){title_suffix}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_method_line_loss(data, methods, out_path, title_suffix=""):
    """Training loss vs. epoch, one line per uncertainty method (averaged
    over its 5 fusion modes), plus a single Plain reference line."""
    methods = ordered_methods(methods)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in methods:
        recs = [data[m][fm] for fm in FUSION_MODES if fm in data[m]]
        ep, arr = stack_series(recs, "train_loss")
        _plot_shaded_line(ax, ep, arr, m, METHOD_COLORS[m], "-", shade=len(recs) > 1)
    plain_recs = [data[m]["plain"] for m in methods]
    ep, arr = stack_series(plain_recs, "train_loss")
    _plot_shaded_line(ax, ep, arr, "Plain (reference)", PLAIN_COLOR, "-", shade=len(plain_recs) > 1)
    _mark_stage_boundary(ax)
    ax.legend(fontsize=9, loc="upper right")
    _style_axes(ax, "Training loss",
                f"Uncertainty method comparison: training loss per epoch{title_suffix}\n"
                "(avg. over 5 fusion modes; test loss is not logged by this pipeline)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_method_by_fusion_heatmap(data, methods, out_path, title_suffix=""):
    """The 'both axes at once' diagram: method x fusion mode, un-averaged in
    either direction -- every one of the (up to) 20 individually-trained
    weighted classifiers gets its own cell."""
    methods = ordered_methods(methods)
    mat = np.full((len(methods), len(FUSION_MODES)), np.nan)
    for i, m in enumerate(methods):
        for j, fm in enumerate(FUSION_MODES):
            if fm in data[m]:
                mat[i, j] = best_test_acc(data[m][fm])

    plain_vals = [best_test_acc(data[m]["plain"]) for m in methods]
    plain_avg = np.mean(plain_vals)

    # Sequential blue ramp, exact steps from the validated palette (light->dark).
    seq_hex = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", seq_hex)

    fig, ax = plt.subplots(figsize=(9, 1.2 + 1.0 * len(methods)))
    finite = mat[np.isfinite(mat)]
    vmin, vmax = (finite.min(), finite.max()) if finite.size else (0, 1)
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(FUSION_MODES)))
    ax.set_xticklabels(FUSION_MODES, rotation=20, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    thresh = vmin + 0.6 * (vmax - vmin) if vmax > vmin else vmin
    for i in range(len(methods)):
        for j in range(len(FUSION_MODES)):
            v = mat[i, j]
            if np.isfinite(v):
                txt_color = "white" if v >= thresh else "#0b0b0b"
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center", color=txt_color, fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Best test accuracy (%)")
    ax.set_title(
        f"Weighted-classifier accuracy: method × fusion mode{title_suffix}\n"
        f"(Plain reference average: {plain_avg:.1f}%)",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


METHOD_COMPARISON_PLOT_FUNCS = [
    ("bar_accuracy", plot_method_bar_accuracy),
    ("line_accuracy", plot_method_line_accuracy),
    ("line_loss", plot_method_line_loss),
    ("heatmap_method_by_fusion", plot_method_by_fusion_heatmap),
]


# ==========================================================================
# LaTeX tables -- three tables, one file. Column set is always
# [Plain, input, patch_embed, attn_bias, key_scale, value_scale, Weighted Avg]
# so the accuracy and loss tables line up cell-for-cell and can be read side
# by side; the third table is the mean/std/min/max summary that backs
# bar_accuracy.png with exact numbers instead of eyeballed error bars.
# ==========================================================================

_TABLE_COLS = ["Plain"] + FUSION_MODES + ["Weighted Avg"]

# Display name for method row labels in LaTeX output. Needed because raw
# method keys like "sol_mcd" contain an underscore -- outside math mode LaTeX
# reads "_" as "start a subscript", which cascades into exactly the
# "Missing $ inserted" / "Extra }, or forgotten $" errors a bare "sol_mcd &"
# in a tabular row produces. Plots are unaffected (matplotlib renders "_"
# literally, no mathtext subscript by default), so only this mapping -- used
# solely when writing .tex -- was needed, not a change to the color/legend
# keys used elsewhere.
METHOD_DISPLAY = {
    "tta": "TTA",
    "jacobian": "Jacobian",
    "mahalanobis": "Mahalanobis",
    "mcd": "MCD",
    "sol_mcd": "SOL-MCD",
    "amcd": "A-MCD",
}


def _tex_method_label(m: str) -> str:
    """LaTeX-safe row label for method key `m` (escapes stray underscores)."""
    return METHOD_DISPLAY.get(m, m.replace("_", r"\_"))


def _col_header(c: str) -> str:
    """LaTeX column header for a fusion-mode key, e.g. 'key_scale' -> 'Key Scale'."""
    if c in ("Plain", "Weighted Avg"):
        return c
    return c.replace("_", " ").title()


def _fmt(v, decimals: int, suffix: str = "") -> str:
    """Format a numeric table cell, rendering missing/NaN values as '--'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{decimals}f}{suffix}"


def _method_row_values(data, m: str, metric_fn):
    """[Plain, input, ..., value_scale, Weighted Avg] for one method, where
    metric_fn(records) -> float is either best_test_acc or loss_at_best_epoch."""
    plain_v = metric_fn(data[m]["plain"])
    fusion_vs = [metric_fn(data[m][fm]) if fm in data[m] else None for fm in FUSION_MODES]
    present = [v for v in fusion_vs if v is not None]
    avg_v = float(np.mean(present)) if present else None
    return [plain_v] + fusion_vs + [avg_v]


def _wrap_table(caption: str, label: str, col_spec: str, header_cells: list[str],
                 body_rows: list[str], footnote: str = "") -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]
    lines.extend(body_rows)
    lines += [r"\bottomrule", r"\end{tabular}%", "}"]
    if footnote:
        lines.append(rf"\vspace{{2pt}}\\{{\footnotesize {footnote}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_accuracy_table(data, methods) -> str:
    methods = ordered_methods(methods)
    header = [r"\textbf{Method}"] + [rf"\textbf{{{_col_header(c)}}}" for c in _TABLE_COLS]
    rows = []
    for m in methods:
        vals = _method_row_values(data, m, best_test_acc)
        rows.append(f"{_tex_method_label(m)} & " + " & ".join(_fmt(v, 1, r"\%") for v in vals) + r" \\")
    return _wrap_table(
        caption="Downstream classifier best test accuracy (\\%) by uncertainty method and classifier type.",
        label="tab:downstream_accuracy",
        col_spec="l" + "c" * len(_TABLE_COLS),
        header_cells=header,
        body_rows=rows,
    )


def build_loss_table(data, methods) -> str:
    methods = ordered_methods(methods)
    header = [r"\textbf{Method}"] + [rf"\textbf{{{_col_header(c)}}}" for c in _TABLE_COLS]
    rows = []
    for m in methods:
        vals = _method_row_values(data, m, loss_at_best_epoch)
        rows.append(f"{_tex_method_label(m)} & " + " & ".join(_fmt(v, 3) for v in vals) + r" \\")
    return _wrap_table(
        caption="Training loss at the best-test-accuracy epoch, by uncertainty method and classifier type.",
        label="tab:downstream_loss",
        col_spec="l" + "c" * len(_TABLE_COLS),
        header_cells=header,
        body_rows=rows,
        footnote=(
            "Test loss is not logged by this pipeline (only test accuracy); values shown are "
            "training loss, read at the epoch that achieved the best test accuracy for that "
            "classifier -- i.e. the loss of the checkpoint actually reported elsewhere, not "
            "necessarily the final epoch's. Plain and weighted use different loss objectives "
            "(standard cross-entropy vs.\\ uncertainty-weighted cross-entropy), so absolute "
            "magnitudes are not comparable across the Plain column and the rest of the row."
        ),
    )


def _stats_row(label: str, vals: np.ndarray) -> str:
    """One 'Group & Mean & Std & Min & Max & n \\\\' row. Kept as a helper (rather
    than inline f-strings) because Python <3.12 rejects a backslash inside an
    f-string's {} expression part, and these cells all need a literal '\%'."""
    cells = [_fmt(vals.mean(), 1, r"\%"), _fmt(vals.std(), 1, r"\%"),
             _fmt(vals.min(), 1, r"\%"), _fmt(vals.max(), 1, r"\%"), str(len(vals))]
    return label + " & " + " & ".join(cells) + r" \\"


def build_stats_table(data, methods) -> str:
    methods = ordered_methods(methods)
    header = [r"\textbf{Group}", r"\textbf{Mean}", r"\textbf{Std}", r"\textbf{Min}", r"\textbf{Max}", r"\textbf{n}"]
    rows = []
    for m in methods:
        vals = np.array([best_test_acc(data[m][fm]) for fm in FUSION_MODES if fm in data[m]], dtype=float)
        rows.append(_stats_row(f"{_tex_method_label(m)} (weighted)", vals))

    all_weighted = np.array(
        [best_test_acc(data[m][fm]) for m in methods for fm in FUSION_MODES if fm in data[m]], dtype=float)
    rows.append(r"\midrule")
    rows.append(_stats_row("All weighted (pooled)", all_weighted))

    plain_vals = np.array([best_test_acc(data[m]["plain"]) for m in methods], dtype=float)
    rows.append(_stats_row("Plain (reference)", plain_vals))

    return _wrap_table(
        caption="Best test accuracy (\\%) summary statistics: per method (pooled over its 5 fusion-mode "
                "classifiers), all weighted classifiers pooled together, and the Plain baseline (pooled "
                "over its 4 independently-trained runs, one per method combo).",
        label="tab:downstream_stats",
        col_spec="lccccc",
        header_cells=header,
        body_rows=rows,
    )


def write_latex_tables(data, methods, out_path: Path) -> None:
    tables = [
        "% Auto-generated by scripts/plot_downstream_results.py -- do not hand-edit,\n"
        "% re-run the script instead so this file and the plots never drift apart.\n",
        "% ---- Table 1: accuracy ----",
        build_accuracy_table(data, methods),
        "",
        "% ---- Table 2: loss ----",
        build_loss_table(data, methods),
        "",
        "% ---- Table 3: summary statistics ----",
        build_stats_table(data, methods),
        "",
    ]
    out_path.write_text("\n\n".join(tables))


def run(log: Path, out: Path) -> None:
    """Parse `log` and write every plot (a)-(g) under `out`. Callable directly
    from other Python code (e.g. main.py's --stage plot_downstream), not just
    from the CLI below."""
    data = parse_log(log)
    methods = usable_methods(data)
    if not methods:
        raise SystemExit(f"No combos with epoch-level training data found in {log}")

    print(f"Usable combos (have epoch data): {methods}")
    for m in methods:
        fms_present = [fm for fm in FUSION_MODES if fm in data[m]]
        print(f"  SMIRK_{m}: plain + {len(fms_present)}/5 fusion modes ({fms_present})")

    overall_dir = out / "overall"
    overall_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in ALL_PLOT_FUNCS:
        fn(data, methods, overall_dir / f"{name}.png", title_suffix=f" (n={len(methods)} combos)")

    for m in methods:
        per_dir = out / "per_version" / f"SMIRK_{m}"
        per_dir.mkdir(parents=True, exist_ok=True)
        for name, fn in ALL_PLOT_FUNCS:
            fn(data, [m], per_dir / f"{name}.png", title_suffix=f" — SMIRK×{m}")

    method_dir = out / "method_comparison"
    if len(ordered_methods(methods)) >= 2:
        method_dir.mkdir(parents=True, exist_ok=True)
        for name, fn in METHOD_COMPARISON_PLOT_FUNCS:
            fn(data, methods, method_dir / f"{name}.png")
    else:
        print("\nSkipping method_comparison/ -- fewer than 2 usable methods, nothing to compare.")

    tables_path = out / "tables.tex"
    out.mkdir(parents=True, exist_ok=True)
    write_latex_tables(data, methods, tables_path)

    print(f"\nSaved plots under {out}/")
    print(f"  overall/            -- combined across {methods}")
    for m in methods:
        print(f"  per_version/SMIRK_{m}/")
    if len(ordered_methods(methods)) >= 2:
        print(f"  method_comparison/  -- {ordered_methods(methods)} compared to each other")
    print(f"  tables.tex          -- accuracy, loss, and summary-stats LaTeX tables")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=Path("figures/downstream.txt"))
    ap.add_argument("--out", type=Path, default=Path("figures/downstream_plots"))
    args = ap.parse_args()
    run(args.log, args.out)


if __name__ == "__main__":
    main()
