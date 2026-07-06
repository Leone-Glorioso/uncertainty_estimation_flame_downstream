#!/usr/bin/env python3
"""
extract_now_images.py
=====================
Selectively extract the 352 NoW validation images from a TEMPEH
Training Images zip package without unpacking the whole archive.

The zip stays intact — only the 352 needed files are read out of it.

Usage
-----
    python scripts/extract_now_images.py \
        --zip      "/mnt/c/Users/Κωνσταντινος/Downloads/Images - Package 01.zip" \
        --out-dir  ./datasets/now

If not all 352 images are in Package 01, run again with Package 02, etc.
The script skips images already extracted.
"""

import argparse
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT  = Path(__file__).resolve().parent.parent
IMG_LIST_PATH = PROJECT_ROOT / 'datasets' / 'now' / 'imagepathsvalidation.txt'


def load_needed(img_list: Path):
    """Return list of relative image paths from imagepathsvalidation.txt."""
    with open(img_list) as f:
        return [line.strip() for line in f if line.strip()]


def build_zip_index(zf: zipfile.ZipFile):
    """
    Build a suffix-index of zip entries so we can match
    'FaMoS_XXX/multiview_neutral/IMG_0041.jpg' regardless of any
    leading prefix inside the zip.
    """
    index = {}
    for name in zf.namelist():
        # index by the last 3 path components (subject/condition/file)
        parts = Path(name).parts
        if len(parts) >= 3:
            key = '/'.join(parts[-3:])
            index[key] = name
        if len(parts) >= 2:
            key2 = '/'.join(parts[-2:])
            index.setdefault(key2, name)
    return index


def main():
    parser = argparse.ArgumentParser(description='Extract NoW validation images from TEMPEH zip')
    parser.add_argument('--zip', required=True,
                        help='Path to a TEMPEH Training Images zip package')
    parser.add_argument('--out-dir', default=str(PROJECT_ROOT / 'datasets' / 'now'),
                        help='Destination directory (default: datasets/now/)')
    parser.add_argument('--img-list', default=str(IMG_LIST_PATH),
                        help='Path to imagepathsvalidation.txt')
    args = parser.parse_args()

    zip_path  = Path(args.zip)
    out_dir   = Path(args.out_dir)
    list_path = Path(args.img_list)

    if not zip_path.exists():
        sys.exit(f'[error] Zip not found: {zip_path}')
    if not list_path.exists():
        sys.exit(f'[error] Image list not found: {list_path}\n'
                 f'  Expected at: {IMG_LIST_PATH}')

    needed     = load_needed(list_path)
    n_needed   = len(needed)
    n_already  = sum(1 for p in needed if (out_dir / p).exists())

    if n_already == n_needed:
        print(f'All {n_needed} images already extracted. Nothing to do.')
        return

    print(f'[extract] Opening {zip_path.name} ({zip_path.stat().st_size / 1e9:.1f} GB)...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        idx = build_zip_index(zf)
        n_found   = 0
        n_skipped = 0
        n_missing = 0

        for rel_path in needed:
            dst = out_dir / rel_path
            if dst.exists():
                n_skipped += 1
                continue

            # Try exact match, then suffix match
            zip_entry = idx.get(rel_path) or idx.get('/'.join(Path(rel_path).parts[-3:]))
            if zip_entry is None:
                n_missing += 1
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zip_entry) as src, open(dst, 'wb') as f:
                f.write(src.read())
            n_found += 1

    print(f'[extract] Extracted : {n_found}')
    print(f'[extract] Skipped   : {n_skipped} (already present)')
    print(f'[extract] Not found : {n_missing} (try next package)')

    total_present = n_skipped + n_found
    print(f'\n[extract] Total extracted so far: {total_present} / {n_needed}')
    if n_missing > 0:
        print(f'  Run again with the next zip package to find the remaining {n_missing}.')
    else:
        print('  All images extracted!')


if __name__ == '__main__':
    main()
