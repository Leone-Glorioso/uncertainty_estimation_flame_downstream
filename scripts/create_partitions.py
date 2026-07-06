#!/usr/bin/env python3
"""
create_partitions.py
====================
Create fixed, reproducible dataset partitions and save them as JSON index files.

Each JSON file lists the file paths and metadata for a given (dataset, size) combination.
Partitions are prefix-consistent: the n=10 partition is always a subset of the n=50 one.

Usage examples
--------------
# Single dataset, default sizes (10 / 50 / 100), stratified:
    python scripts/create_partitions.py --datasets coma --stratify

# Multiple datasets separately, custom sizes:
    python scripts/create_partitions.py --datasets utkface lfw --sizes 50 200 500

# All available datasets + a combined mixed partition:
    python scripts/create_partitions.py --datasets all --sizes 10 50 100 --mix --stratify

# Non-stratified, custom output dir:
    python scripts/create_partitions.py --datasets coma --sizes 100 --output-dir my_splits/

Loading a saved partition
-------------------------
    import json
    records = json.load(open('partitions/coma/n00100.json'))['samples']
    # each record has: img_path, ply_path, subject_id, expression, dataset, ...
"""

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
sys.path.insert(0, str(PROJECT_ROOT))

AVAILABLE_DATASETS = ["now", "coma", "tempeh", "utkface", "lfw"]

# ---------------------------------------------------------------------------
# Stratification keys
# ---------------------------------------------------------------------------

def _stratum(item: dict) -> str:
    """Stratification-key for one catalogue item, chosen per source dataset
    (expression for CoMA, condition for NoW, age-decade x gender for UTKFace,
    subject for LFW/TEMPEH, source dataset name for mixed partitions)."""
    ds = item.get("dataset", "")
    if ds == "coma":
        return item.get("expression", "unknown")
    if ds == "now":
        return item.get("condition", "unknown")
    if ds == "tempeh":
        return item.get("expression", item.get("subject_id", "unknown"))
    if ds == "utkface":
        try:
            decade = (int(item.get("age", 0)) // 10) * 10
        except (ValueError, TypeError):
            decade = 0
        return f"{decade}s_{item.get('gender', 'unknown')}"
    if ds == "lfw":
        return item.get("subject_id", "unknown")
    # mixed — stratify by source dataset
    return item.get("dataset", "unknown")


# ---------------------------------------------------------------------------
# Sampling: prefix-consistent ordering
# ---------------------------------------------------------------------------

def _ordered_list(items: List[dict], stratify: bool, rng: np.random.Generator,
                   stratum_fn: Optional[Callable] = None) -> List[dict]:
    """
    Return a single fixed ordering of all items.
    Taking the first N items of this list gives a valid N-sample partition,
    and smaller partitions are always subsets of larger ones (prefix-consistent).

    If stratify=True, items are interleaved round-robin across strata so that
    any prefix has proportional representation from every stratum.
    stratum_fn defaults to _stratum; pass a custom fn for mixed partitions.
    """
    if not stratify:
        shuffled = list(items)
        rng.shuffle(shuffled)
        return shuffled

    key_fn = stratum_fn or _stratum

    groups: Dict[str, list] = defaultdict(list)
    for item in items:
        groups[key_fn(item)].append(item)

    for grp in groups.values():
        rng.shuffle(grp)

    # Round-robin interleave: ensures proportionality for any prefix
    strata  = sorted(groups.keys())
    queues  = [groups[s] for s in strata]
    result  = []
    while any(queues):
        for q in queues:
            if q:
                result.append(q.pop(0))
    return result


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _to_record(item: dict) -> dict:
    """Convert a loader item dict to a JSON-serialisable record."""
    record: dict = {}
    for key, val in item.items():
        if val is None:
            record[key.lstrip("_")] = None
        elif isinstance(val, Path):
            pub = key.lstrip("_")
            try:
                record[pub] = str(val.relative_to(PROJECT_ROOT))
            except ValueError:
                record[pub] = str(val)
        elif isinstance(val, (str, int, float, bool)):
            record[key.lstrip("_")] = val
        elif hasattr(val, "tolist"):
            pass  # skip large numpy arrays (gt_vertices etc.)
        # skip anything else (loaded images, etc.)
    return record


def _save(items: List[dict], path: Path, meta: dict) -> None:
    """Serialise `items` (as JSON-safe records) and `meta` to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "samples": [_to_record(item) for item in items]}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create fixed dataset partitions and save as JSON index files."
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        choices=AVAILABLE_DATASETS + ["all"],
        metavar="DATASET",
        help=f"One or more of: {AVAILABLE_DATASETS} — or 'all'.",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[10, 50, 100],
        metavar="N",
        help="Partition sizes to create (default: 10 50 100).",
    )
    parser.add_argument(
        "--mix", action="store_true",
        help="Also write a combined mixed partition across all specified datasets.",
    )
    parser.add_argument(
        "--stratify", action="store_true",
        help=(
            "Stratified sampling: CoMA→expression, NoW→condition, "
            "UTKFace→age-decade×gender, LFW/TEMPEH→subject."
        ),
    )
    parser.add_argument(
        "--output-dir", default="partitions",
        help="Output directory for JSON files (default: partitions/).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--data-root", default=str(DATASETS_ROOT),
        help="Datasets root directory (default: datasets/).",
    )
    args = parser.parse_args()

    datasets   = AVAILABLE_DATASETS if "all" in args.datasets else args.datasets
    sizes      = sorted(set(args.sizes))
    output_dir = PROJECT_ROOT / args.output_dir

    # Lazy import so the script starts fast even if torch isn't installed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from src.data_loader import FaceDatasetLoader

    loader = FaceDatasetLoader(data_root=args.data_root)
    collectors: Dict[str, Callable] = {
        "now":     loader._collect_now,
        "coma":    loader._collect_coma,
        "tempeh":  loader._collect_tempeh,
        "utkface": loader._collect_utkface,
        "lfw":     loader._collect_lfw,
    }

    # ── Collect ─────────────────────────────────────────────────────────────
    collected: Dict[str, List[dict]] = {}
    for ds in datasets:
        print(f"[collect] {ds} ...", end=" ", flush=True)
        items = collectors[ds]()
        collected[ds] = items
        if items:
            print(f"{len(items)} items")
        else:
            print("0 items — skipping (dataset not downloaded yet?)")

    # ── Per-dataset partitions ───────────────────────────────────────────────
    print()
    for ds, items in collected.items():
        if not items:
            continue

        ordering = _ordered_list(items, args.stratify, np.random.default_rng(args.seed))
        strat_tag = " [stratified]" if args.stratify else ""

        for n in sizes:
            chosen = ordering[:n] if n <= len(ordering) else ordering
            actual = len(chosen)
            if actual < n:
                print(f"  [{ds}] n={n}: only {actual} items available, using all.")

            out_path = output_dir / ds / f"n{n:05d}.json"
            _save(chosen, out_path, {
                "dataset":    ds,
                "n":          actual,
                "requested":  n,
                "stratified": args.stratify,
                "seed":       args.seed,
            })

            breakdown = Counter(_stratum(x) for x in chosen)
            top = ", ".join(f"{k}:{v}" for k, v in list(breakdown.most_common(4)))
            print(f"  [{ds}] n={n:>5}: {actual} samples → "
                  f"{out_path.relative_to(PROJECT_ROOT)}{strat_tag}  ({top}{'…' if len(breakdown) > 4 else ''})")

    # ── Mixed partition ──────────────────────────────────────────────────────
    if args.mix:
        ready = {ds: items for ds, items in collected.items() if items}
        if len(ready) < 2:
            print("\n[mix] Need at least 2 non-empty datasets for a mixed partition — skipping.")
        else:
            print(f"\n[mix] Combining {list(ready.keys())} ...")
            all_items = [item for items in ready.values() for item in items]

            # For mixed, always stratify by dataset name (not per-item strata)
            # so each source dataset gets equal round-robin representation.
            ordering = _ordered_list(
                all_items, stratify=True,
                rng=np.random.default_rng(args.seed),
                stratum_fn=lambda item: item.get("dataset", "unknown"),
            )

            for n in sizes:
                chosen = ordering[:n] if n <= len(ordering) else ordering
                actual = len(chosen)

                out_path = output_dir / "mixed" / f"n{n:05d}.json"
                _save(chosen, out_path, {
                    "dataset":    "mixed",
                    "sources":    list(ready.keys()),
                    "n":          actual,
                    "requested":  n,
                    "stratified": True,
                    "seed":       args.seed,
                })

                breakdown = Counter(x.get("dataset", "?") for x in chosen)
                detail = "  ".join(f"{k}:{v}" for k, v in sorted(breakdown.items()))
                print(f"  [mixed] n={n:>5}: {actual} samples → "
                      f"{out_path.relative_to(PROJECT_ROOT)}  [{detail}]")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\nDone. Partitions written to {output_dir}/")
    print("\nTo load a partition in Python:")
    print("  import json")
    print("  records = json.load(open('partitions/coma/n00100.json'))['samples']")
    print("  print(records[0]['img_path'], records[0]['expression'])")


if __name__ == "__main__":
    main()
