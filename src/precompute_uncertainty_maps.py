"""
precompute_uncertainty_maps.py
==============================
Pre-computes 2D confidence maps for all images in an emotion dataset and saves
them as .npy files, mirroring the dataset directory structure.

Run this once before training with ``EmotionDataset(mode='uncertainty_weighted')``.

For each image the script:
  1. Runs SMIRK to get FLAME vertex positions.
  2. Estimates TTA uncertainty (N augmented forward passes).
  3. Projects per-vertex variance to a 2D confidence map with project_variance_to_2d.
  4. Saves the (H, W) float32 array as <output_root>/<split>/<class>/<stem>.npy.

Existing .npy files are skipped so the script is safe to re-run after interruption.

Usage
-----
    python -m src.precompute_uncertainty_maps \\
        --data_root   ./data/affectnet \\
        --output_root ./data/affectnet_confidence_maps \\
        --n_tta_passes 10

    # Limit to specific splits:
    python -m src.precompute_uncertainty_maps \\
        --data_root ./data/affectnet \\
        --output_root ./data/affectnet_confidence_maps \\
        --splits train test
"""

import argparse
import warnings
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def precompute_confidence_maps(
    data_root: str,
    output_root: str,
    smirk_checkpoint: Optional[str] = None,
    n_tta_passes: int = 10,
    splits: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> None:
    """
    Iterate over every image in data_root and write a confidence-map .npy file
    to the mirrored location under output_root.

    Parameters
    ----------
    data_root         : root of the emotion dataset (parent of split dirs)
    output_root       : where to write .npy files
    smirk_checkpoint  : path to SMIRK .pth checkpoint; None uses wrapper default
    n_tta_passes      : number of TTA augmented passes per image (10 is fast)
    splits            : list of split subdirectory names; None → all subdirs
    device            : 'cuda' or 'cpu'; None → auto-detect
    """
    from src.downstream import project_variance_to_2d
    from src.uncertainty import calculate_tta_uncertainty

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load SMIRK wrapper
    try:
        from wrappers.smirk_wrapper import SMIRKWrapper
        smirk = SMIRKWrapper(device=device)
        if smirk_checkpoint is not None:
            smirk.load_checkpoint(smirk_checkpoint)
        print(f"SMIRK loaded on {device}.")
    except ImportError as exc:
        raise ImportError(
            "SMIRKWrapper not importable.  Ensure wrappers/smirk_wrapper.py "
            f"exists and SMIRK dependencies are installed.  Original: {exc}"
        ) from exc

    data_root   = Path(data_root).resolve()
    output_root = Path(output_root).resolve()

    if splits is None:
        splits = [p.name for p in sorted(data_root.iterdir()) if p.is_dir()]

    n_saved   = 0
    n_skipped = 0
    n_errors  = 0

    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            print(f"  Split '{split}' not found, skipping.")
            continue

        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue

            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in _IMG_EXTS:
                    continue

                rel      = img_path.relative_to(data_root)
                npy_path = (output_root / rel).with_suffix('.npy')

                if npy_path.exists():
                    n_skipped += 1
                    continue

                npy_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        warnings.warn(f"Could not read {img_path}, skipping.")
                        n_errors += 1
                        continue

                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8
                    H, W    = img_rgb.shape[:2]

                    verts = smirk.get_vertices(img_rgb)                  # (5023, 3)
                    unc   = calculate_tta_uncertainty(smirk, img_rgb,
                                                      n_passes=n_tta_passes)  # (5023, 1)
                    conf  = project_variance_to_2d(verts, unc, (H, W))  # (H, W) float32

                    np.save(str(npy_path), conf)
                    n_saved += 1

                    if n_saved % 100 == 0:
                        print(f"  Saved {n_saved} maps "
                              f"({n_skipped} cached, {n_errors} errors)...")

                except Exception as exc:
                    warnings.warn(f"Error processing {img_path}: {exc}")
                    n_errors += 1

    print(
        f"\nDone.  Saved {n_saved} confidence maps to {output_root}."
        f"  Skipped {n_skipped} (already existed).  Errors: {n_errors}."
    )


def main() -> None:
    """CLI entry point: parse arguments and call precompute_confidence_maps()."""
    parser = argparse.ArgumentParser(
        description='Pre-compute 2D confidence maps for an emotion dataset'
    )
    parser.add_argument('--data_root',        required=True)
    parser.add_argument('--output_root',      required=True)
    parser.add_argument('--smirk_checkpoint', default=None)
    parser.add_argument('--n_tta_passes',     type=int,   default=10)
    parser.add_argument('--splits',           nargs='+',  default=None)
    parser.add_argument('--device',           default=None)
    args = parser.parse_args()

    precompute_confidence_maps(
        data_root=args.data_root,
        output_root=args.output_root,
        smirk_checkpoint=args.smirk_checkpoint,
        n_tta_passes=args.n_tta_passes,
        splits=args.splits,
        device=args.device,
    )


if __name__ == '__main__':
    main()
