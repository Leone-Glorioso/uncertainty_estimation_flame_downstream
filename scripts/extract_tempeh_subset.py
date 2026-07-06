#!/usr/bin/env python3
"""
extract_tempeh_subset.py
========================
Extract N paired (image + FLAME registration) samples from downloaded
TEMPEH / FaMoS packages and copy them into datasets/tempeh/.

After running this script you can delete the original downloaded packages.

Usage
-----
    python scripts/extract_tempeh_subset.py \
        --images-dir    /tmp/tempeh_raw/images \
        --reg-dir       /tmp/tempeh_raw/registrations \
        --out-dir       ./datasets/tempeh \
        --n-samples     500

If the script cannot find pairs automatically, run with --scan-only first
to print the discovered directory structure, then adjust --images-dir /
--reg-dir to point at the right subdirectories.

    python scripts/extract_tempeh_subset.py \
        --images-dir /tmp/tempeh_raw \
        --reg-dir    /tmp/tempeh_raw \
        --scan-only
"""

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────

IMG_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp'}
REG_EXTS  = {'.ply', '.npz', '.obj'}
MAX_DEPTH = 6   # how deep to search for files


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_files(root: Path, extensions: set, max_depth: int = MAX_DEPTH):
    """Yield (relative_path, absolute_path) for all files with given extensions."""
    def _walk(path: Path, depth: int):
        if depth > max_depth:
            return
        try:
            for child in sorted(path.iterdir()):
                if child.is_dir():
                    yield from _walk(child, depth + 1)
                elif child.suffix.lower() in extensions:
                    yield child.relative_to(root), child
        except PermissionError:
            pass

    yield from _walk(root, 0)


def _infer_structure(rel_path: Path):
    """
    Decompose a relative path into (subject, sequence, frame_id).

    TEMPEH image layout:        subject/expression/frame_dir/camera_file.png
                                 → frame_id = frame_dir  (e.g. '000013')
    TEMPEH registration layout: subject/expression/expression.000013.ply
                                 → frame_id = last dot-component of stem
    """
    parts = rel_path.parts  # includes filename
    stem  = rel_path.stem

    if len(parts) >= 4:
        # Images: use the frame directory (parts[2]) as the canonical frame id
        return parts[0], parts[1], parts[2]
    if len(parts) == 3:
        # Registrations: strip expression prefix, keep numeric frame id
        frame = stem.rsplit('.', 1)[-1] if '.' in stem else stem
        return parts[0], parts[1], frame
    if len(parts) == 2:
        return parts[0], '', stem
    return '', '', stem


def scan_structure(root: Path, extensions: set, label: str):
    """Print a summary of the directory tree and file counts."""
    print(f"\n[scan] {label}: {root}")
    counts: dict = defaultdict(int)
    total = 0
    for rel, _ in _find_files(root, extensions):
        subj, seq, _ = _infer_structure(rel)
        counts[(subj, seq)] += 1
        total += 1
        if total > 5000:
            print("  (stopping scan at 5000 files)")
            break

    if not counts:
        print("  No files found.")
        return

    for (subj, seq), cnt in sorted(counts.items())[:20]:
        print(f"  subject={subj!r:30s}  seq={seq!r:30s}  files={cnt}")
    if len(counts) > 20:
        print(f"  ... and {len(counts) - 20} more (subject, sequence) pairs")
    print(f"  TOTAL files found: {total}")


# ── main logic ────────────────────────────────────────────────────────────────

def collect_pairs(images_dir: Path, reg_dir: Path):
    """
    Find all (image_path, registration_path) pairs that share the same
    (subject, sequence, frame_stem) triple.

    Returns list of dicts with keys:
        subject, sequence, frame, img_path, reg_path
    """
    print("\n[collect] Indexing registration files …")
    reg_index: dict = {}   # (subject, sequence, frame_stem) → Path
    for rel, abs_path in _find_files(reg_dir, REG_EXTS):
        key = _infer_structure(rel)
        reg_index[key] = abs_path
    print(f"  Found {len(reg_index)} registration files.")

    print("[collect] Indexing image files …")
    img_index: dict = {}   # (subject, sequence, frame_stem) → Path
    for rel, abs_path in _find_files(images_dir, IMG_EXTS):
        key = _infer_structure(rel)
        # Keep one image per (subject, sequence, frame) — prefer earlier cameras
        if key not in img_index:
            img_index[key] = abs_path
    print(f"  Found {len(img_index)} image files (one per frame).")

    print("[collect] Matching pairs …")
    pairs = []
    for key, img_path in img_index.items():
        if key in reg_index:
            subj, seq, frame = key
            pairs.append({
                'subject'  : subj,
                'sequence' : seq,
                'frame'    : frame,
                'img_path' : img_path,
                'reg_path' : reg_index[key],
            })

    print(f"  Matched {len(pairs)} pairs.")
    return pairs


def stratified_sample(pairs: list, n: int, seed: int) -> list:
    """
    Sample n pairs, proportionally from each (subject, sequence) stratum.
    If n >= len(pairs), returns all pairs shuffled.
    """
    if n >= len(pairs):
        rng = random.Random(seed)
        result = list(pairs)
        rng.shuffle(result)
        return result

    strata: dict = defaultdict(list)
    for p in pairs:
        strata[(p['subject'], p['sequence'])].append(p)

    rng   = random.Random(seed)
    n_s   = len(strata)
    base  = n // n_s
    rem   = n - base * n_s

    sorted_strata = sorted(strata.items(), key=lambda x: -len(x[1]))
    selected = []
    for i, (_key, group) in enumerate(sorted_strata):
        cnt = base + (1 if i < rem else 0)
        cnt = min(cnt, len(group))
        selected.extend(rng.sample(group, cnt))

    rng.shuffle(selected)
    return selected


def copy_subset(pairs: list, out_dir: Path, dry_run: bool = False):
    """Copy selected pairs into datasets/tempeh/{subject}/{sequence}/images/ and .../flame/."""
    n_copied = 0
    total_bytes = 0

    for p in pairs:
        subj = p['subject'] or 'unknown_subject'
        seq  = p['sequence'] or 'seq_01'
        frame = p['frame']

        img_dst_dir  = out_dir / subj / seq / 'images'
        flame_dst_dir = out_dir / subj / seq / 'flame'

        if not dry_run:
            img_dst_dir.mkdir(parents=True, exist_ok=True)
            flame_dst_dir.mkdir(parents=True, exist_ok=True)

        img_dst  = img_dst_dir  / (frame + p['img_path'].suffix)
        reg_dst  = flame_dst_dir / (frame + p['reg_path'].suffix)

        if not dry_run:
            shutil.copy2(p['img_path'], img_dst)
            shutil.copy2(p['reg_path'], reg_dst)

        total_bytes += p['img_path'].stat().st_size + p['reg_path'].stat().st_size
        n_copied += 1

    return n_copied, total_bytes


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Extract TEMPEH/FaMoS subset")
    p.add_argument('--images-dir', required=True,
                   help='Root directory of extracted images package(s).')
    p.add_argument('--reg-dir', required=True,
                   help='Root directory of extracted FLAME registrations package(s).')
    p.add_argument('--out-dir', default='./datasets/tempeh',
                   help='Destination directory (default: ./datasets/tempeh).')
    p.add_argument('--n-samples', type=int, default=500,
                   help='Number of paired samples to extract (default: 500).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--scan-only', action='store_true',
                   help='Just print directory structure and exit — no files copied.')
    p.add_argument('--dry-run', action='store_true',
                   help='Show what would be copied without copying anything.')
    return p.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images_dir).resolve()
    reg_dir    = Path(args.reg_dir).resolve()
    out_dir    = Path(args.out_dir).resolve()

    if not images_dir.exists():
        sys.exit(f"[error] --images-dir does not exist: {images_dir}")
    if not reg_dir.exists():
        sys.exit(f"[error] --reg-dir does not exist: {reg_dir}")

    # ── scan mode: just print structure ──────────────────────────────────────
    if args.scan_only:
        scan_structure(images_dir, IMG_EXTS, "Images")
        scan_structure(reg_dir,    REG_EXTS, "Registrations")
        return

    # ── collect and match pairs ───────────────────────────────────────────────
    pairs = collect_pairs(images_dir, reg_dir)

    if not pairs:
        print("\n[error] No paired (image + registration) files found.")
        print("  Try running with --scan-only to inspect the directory structure,")
        print("  then adjust --images-dir and --reg-dir accordingly.")
        sys.exit(1)

    # ── sample ────────────────────────────────────────────────────────────────
    selected = stratified_sample(pairs, args.n_samples, args.seed)
    print(f"\n[sample] Selected {len(selected)} / {len(pairs)} pairs "
          f"(requested {args.n_samples}, seed={args.seed})")

    # Breakdown by subject
    subj_counts: dict = defaultdict(int)
    for p in selected:
        subj_counts[p['subject']] += 1
    for subj, cnt in sorted(subj_counts.items()):
        print(f"  subject={subj!r:30s}  selected={cnt}")

    # ── dry-run / copy ────────────────────────────────────────────────────────
    label = "[dry-run]" if args.dry_run else "[copy]"
    print(f"\n{label} Copying to {out_dir} …")
    n_copied, total_bytes = copy_subset(selected, out_dir, dry_run=args.dry_run)

    mb = total_bytes / (1024 ** 2)
    print(f"{label} Done — {n_copied} pairs, {mb:.1f} MB on disk.")

    if not args.dry_run:
        print(f"\nYou can now delete the original packages to free space.")
        print(f"Kept data is at: {out_dir}")


if __name__ == '__main__':
    main()
