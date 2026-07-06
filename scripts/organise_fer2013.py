"""
Reorganise a Kaggle FER2013 download into the class-subfolder layout that
EmotionDataset expects.

Kaggle dataset: https://www.kaggle.com/datasets/msambare/fer2013
Download with:  kaggle datasets download -d msambare/fer2013

The Kaggle archive already has train/ and test/ with one subfolder per class.
The only mismatch is that FER2013 calls the folder "angry" while EmotionDataset
expects "anger".  This script creates an organised copy (or symlink tree) with
the correct name.

Usage
-----
    python scripts/organise_fer2013.py \
        --src  /path/to/fer2013_extracted \
        --dst  /path/to/datasets/fer2013 \
        [--copy]   # default: symlink (no extra disk space)

Expected src layout
-------------------
    src/
      train/
        angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
      test/
        angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/

Output dst layout
-----------------
    dst/
      train/
        anger/ disgust/ fear/ happy/ neutral/ sad/ surprise/
      test/
        anger/ disgust/ fear/ happy/ neutral/ sad/ surprise/
"""

import argparse
import os
import shutil
from pathlib import Path

# FER2013 folder name → EmotionDataset class name
_RENAME = {'angry': 'anger'}

_IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def organise_split(src_split: Path, dst_split: Path, use_copy: bool) -> int:
    """Copy or symlink one split (train/ or test/) into the renamed class layout.
    Returns the number of images now present (copied plus already-existing)."""
    total = 0
    for cls_dir in sorted(src_split.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = _RENAME.get(cls_dir.name.lower(), cls_dir.name.lower())
        out_dir = dst_split / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in _IMG_EXTS:
                continue
            dst = out_dir / img.name
            if dst.exists() or dst.is_symlink():
                total += 1
                continue
            if use_copy:
                shutil.copy2(img, dst)
            else:
                os.symlink(img.resolve(), dst)
            total += 1
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True,
                    help='Root of the extracted FER2013 Kaggle download '
                         '(contains train/ and test/).')
    ap.add_argument('--dst', required=True,
                    help='Output root for the reorganised dataset.')
    ap.add_argument('--copy', action='store_true', default=False,
                    help='Copy files instead of symlinking.')
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    mode = 'copy' if args.copy else 'symlink'
    print(f"FER2013 organiser  [{mode} mode]")
    print(f"  src : {src}")
    print(f"  dst : {dst}")

    for split in ('train', 'test'):
        src_split = src / split
        if not src_split.exists():
            print(f"  [skip] {src_split} not found")
            continue
        n = organise_split(src_split, dst / split, args.copy)
        print(f"  [{split}] {n:,} images")

    print(f"\nDone. Set EMOTION_DB_ROOT=\"{dst.resolve()}\" in run_experiments.sh")


if __name__ == '__main__':
    main()
