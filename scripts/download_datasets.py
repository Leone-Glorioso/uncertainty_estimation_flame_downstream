"""
download_datasets.py
====================
Dataset setup helper for the 3D face uncertainty project.

All four datasets (NoW, CoMA, TEMPEH, RAF-DB) require **manual
registration** before download.  This script does the following:

  1. Creates the expected directory skeleton under datasets/.
  2. Prints exact download instructions with registration links and
     anticipated file layouts.
  3. Reorganises RAF-DB from its native flat+label-list format into the
     class-folder format expected by src/emotion_dataset.EmotionDataset.
  4. Optionally generates synthetic sample data so the pipeline can be
     smoke-tested before the real data arrives.

Usage
-----
# Print all instructions and create folder skeleton:
    python scripts/download_datasets.py

# Create tiny synthetic data for unit-testing the loaders:
    python scripts/download_datasets.py --create-sample

# Organise RAF-DB after you have downloaded and extracted the zip:
    python scripts/download_datasets.py --organise-rafdb \\
        --rafdb-raw   ./datasets/raf-db/raw \\
        --rafdb-out   ./datasets/raf-db/organised

# Validate that all expected files exist:
    python scripts/download_datasets.py --validate
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"

# RAF-DB label integer → class folder name (EmotionDataset convention)
RAFDB_LABEL_TO_CLASS = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "anger",
    7: "neutral",
}

EXPECTED_DIRS = {
    "now": [
        "datasets/now/",
        "  {subject_id}/",
        "    neutral/            # jpg images",
        "    expression/",
        "    occlusion/",
        "    selfie/",
        "  scans/",
        "    {subject_id}.ply   # ~120K vertex neutral scan",
        "  scan_landmarks/",
        "    {subject_id}.npy   # (7,3) face landmarks for alignment",
    ],
    "coma": [
        "datasets/coma/",
        "  {subject}/            # e.g. FaceTalk_170725_00137_TA",
        "    {expression}/       # e.g. bareteeth",
        "      {idx:05d}.ply    # FLAME-registered mesh, 5023 vertices",
        "      {idx:05d}.jpg    # optional rendered image (if absent the",
        "                       # loader skips the image or renders on-the-fly)",
    ],
    "tempeh": [
        "datasets/tempeh/",
        "  {subject}/",
        "    {sequence}/",
        "      images/           # jpg or png frames",
        "      scans/            # .ply scan files (raw, not FLAME-registered)",
    ],
    "utkface": [
        "datasets/utkface/",
        "  {age}_{gender}_{race}_{datetime}.jpg.chip.jpg",
        "  (all images flat in this folder)",
    ],
    "lfw": [
        "datasets/lfw/",
        "  {person_name}/",
        "    {person_name}_{XXXX}.jpg",
    ],
    "raf-db": [
        "datasets/raf-db/",
        "  train/",
        "    anger/",
        "    disgust/",
        "    fear/",
        "    happy/",
        "    neutral/",
        "    sad/",
        "    surprise/",
        "  test/",
        "    (same structure)",
    ],
}


# ---------------------------------------------------------------------------
# Directory skeleton
# ---------------------------------------------------------------------------

def setup_directories() -> None:
    """Create all expected dataset directories."""
    dirs = [
        DATASETS_ROOT / "now",
        DATASETS_ROOT / "now" / "scans",
        DATASETS_ROOT / "now" / "scan_landmarks",
        DATASETS_ROOT / "utkface",
        DATASETS_ROOT / "lfw",
        DATASETS_ROOT / "coma",
        DATASETS_ROOT / "tempeh",
        DATASETS_ROOT / "raf-db" / "train",
        DATASETS_ROOT / "raf-db" / "test",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    for cls in RAFDB_LABEL_TO_CLASS.values():
        (DATASETS_ROOT / "raf-db" / "train" / cls).mkdir(parents=True, exist_ok=True)
        (DATASETS_ROOT / "raf-db" / "test" / cls).mkdir(parents=True, exist_ok=True)
    print(f"[setup] Directory skeleton created under {DATASETS_ROOT}")


# ---------------------------------------------------------------------------
# Download instructions
# ---------------------------------------------------------------------------

def print_now_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                     NoW Benchmark Dataset                       ║
╚══════════════════════════════════════════════════════════════════╝
Registration required:  https://now.is.tue.mpg.de/

Steps:
  1. Register and accept the licence on the website above.
  2. On the Downloads page, download EXACTLY these files:
       - Validation set (image list)
       - Validation set (scans)
       - Validation set (scan landmarks)
  3. Extract and place under datasets/now/:
       unzip "NoW_validation_images.zip"     -d datasets/now/
       unzip "NoW_validation_scans.zip"      -d datasets/now/scans/
       unzip "NoW_scan_landmarks.zip"        -d datasets/now/scan_landmarks/
  4. See NOW_SETUP.md for the full step-by-step guide including
     how to verify the layout and run the data loader.

Expected format after setup:
  """ + "\n  ".join(EXPECTED_DIRS["now"]))


def print_coma_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                       CoMA Dataset                              ║
╚══════════════════════════════════════════════════════════════════╝
Registration required:  https://coma.is.tue.mpg.de/

Steps:
  1. Create a free account and accept the licence on the website.
  2. Download "CoMA data" (the raw .ply split) and extract:
       unzip COMA_data.zip -d datasets/coma/
  3. The extracted layout should match:
       datasets/coma/{subject}/{expression}/{idx:05d}.ply
  4. (Optional) Render frontal-view images for each mesh by running:
       python scripts/render_coma_images.py --coma-root datasets/coma/
     The loader works without images (image=None) but regressors need them.

Expected format after setup:
  """ + "\n  ".join(EXPECTED_DIRS["coma"]))


def print_tempeh_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                     TEMPEH Dataset                              ║
╚══════════════════════════════════════════════════════════════════╝
Registration required:  https://tempeh.is.tue.mpg.de/

Steps:
  1. Accept the licence agreement on the website.
  2. Download the data archives and extract:
       tar -xzf tempeh_subject_001.tar.gz -C datasets/tempeh/
  3. Organise files so the layout is:
       datasets/tempeh/{subject}/{sequence}/images/{frame:06d}.png
       datasets/tempeh/{subject}/{sequence}/scans/{frame:06d}.ply
     The scans are raw (not FLAME-registered); scan-to-mesh distance
     will be used for evaluation instead of per-vertex L2.

Expected format after setup:
  """ + "\n  ".join(EXPECTED_DIRS["tempeh"]))


def print_utkface_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  UTKFace Dataset (~360 MB)                      ║
╚══════════════════════════════════════════════════════════════════╝
Source: https://www.kaggle.com/datasets/jangedoo/utkface-new
Downloaded as: archive(1).zip

Steps (in WSL terminal):
  1. Extract — the zip contains a UTKFace/ subfolder, strip it:
       mkdir -p datasets/utkface
       unzip "/mnt/c/Users/Κωνσταντινος/Downloads/archive(1).zip" -d /tmp/utkface_raw/
       mv /tmp/utkface_raw/UTKFace/* datasets/utkface/
       rm -rf /tmp/utkface_raw/
  2. Verify:
       ls datasets/utkface/ | head -5
     Should show filenames like: 25_0_2_20170116174525125.jpg.chip.jpg

  Filenames follow the pattern: {age}_{gender}_{race}_{datetime}.jpg.chip.jpg
  All images sit flat inside datasets/utkface/.

Expected format after setup:
  """ + "\n  ".join(EXPECTED_DIRS["utkface"]))


def print_lfw_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║               LFW — Labeled Faces in the Wild (~173 MB)        ║
╚══════════════════════════════════════════════════════════════════╝
Source: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
Downloaded as: archive.zip  (Windows browser → Downloads folder)

Steps (in WSL terminal):
  1. Extract:
       mkdir -p datasets/lfw
       unzip "/mnt/c/Users/Κωνσταντινος/Downloads/archive.zip" -d /tmp/lfw_raw/
  2. Move images into place (zip contains a lfw/ subfolder):
       mv /tmp/lfw_raw/lfw/* datasets/lfw/
       rm -rf /tmp/lfw_raw/
  3. Verify:
       ls datasets/lfw/ | head -5
     Should show person-name folders like: Aaron_Eckhart/

Expected format after setup:
  """ + "\n  ".join(EXPECTED_DIRS["lfw"]))


def print_rafdb_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                       RAF-DB Dataset                            ║
╚══════════════════════════════════════════════════════════════════╝
Registration required:  http://www.whdeng.cn/RAF/model1.html

Alternatively, a community upload is available on Kaggle:
  kaggle datasets download -d shuvoalok/raf-db-dataset
  unzip raf-db-dataset.zip -d datasets/raf-db/raw/

Steps after download:
  1. Locate the extracted directory containing:
       basic/Image/aligned/       (face-aligned images)
       basic/EmoLabel/list_patition_label.txt
  2. Run the organiser to convert to class-folder layout:
       python scripts/download_datasets.py \\
           --organise-rafdb \\
           --rafdb-raw  <path-to-extracted-RAF-DB/basic> \\
           --rafdb-out  datasets/raf-db/

  Label mapping (RAF-DB integers → class folders):
    1=surprise  2=fear  3=disgust  4=happy  5=sad  6=anger  7=neutral

Expected format after organisation:
  """ + "\n  ".join(EXPECTED_DIRS["raf-db"]))


# ---------------------------------------------------------------------------
# RAF-DB organiser
# ---------------------------------------------------------------------------

def organise_rafdb(raw_root: Path, out_root: Path) -> None:
    """
    Reorganise extracted RAF-DB into class-folder layout.

    Auto-detects two source formats:
      - Kaggle upload: raw_root/DATASET/{train,test}/{1-7}/*.jpg
      - Original RAF-DB: raw_root/EmoLabel/list_patition_label.txt
                         raw_root/Image/aligned/*.jpg

    Pass raw_root as the top-level extracted directory in either case.
    """
    # ── Kaggle format: DATASET/{train,test}/{1-7}/ ──────────────────────
    kaggle_root = raw_root / "DATASET"
    if not kaggle_root.exists() and (raw_root / "train" / "1").exists():
        kaggle_root = raw_root  # user pointed directly at DATASET/

    if (kaggle_root / "train" / "1").exists():
        _organise_rafdb_kaggle(kaggle_root, out_root)
        return

    # ── Original format: EmoLabel/ + Image/aligned/ ─────────────────────
    label_file = raw_root / "EmoLabel" / "list_patition_label.txt"
    img_dir    = raw_root / "Image" / "aligned"

    if not label_file.exists():
        sys.exit(
            f"[error] Could not detect RAF-DB format under: {raw_root}\n"
            "  Kaggle format expects: {raw_root}/DATASET/train/1/\n"
            "  Original format expects: {raw_root}/EmoLabel/list_patition_label.txt"
        )
    if not img_dir.exists():
        sys.exit(f"[error] Image directory not found: {img_dir}")

    for split in ("train", "test"):
        for cls in RAFDB_LABEL_TO_CLASS.values():
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

    n_copied, n_errors = 0, 0
    with open(label_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            fname, label_str = parts
            try:
                label = int(label_str)
            except ValueError:
                continue
            cls = RAFDB_LABEL_TO_CLASS.get(label)
            if cls is None:
                continue
            split = "train" if fname.startswith("train") else "test"
            src   = img_dir / fname
            dst   = out_root / split / cls / fname
            if not src.exists():
                n_errors += 1
                continue
            shutil.copy2(src, dst)
            n_copied += 1

    print(f"[organise-rafdb] Copied {n_copied} images to {out_root} ({n_errors} missing).")


def _organise_rafdb_kaggle(kaggle_root: Path, out_root: Path) -> None:
    """Handle Kaggle RAF-DB layout: {train,test}/{1-7}/*.jpg → {train,test}/{class_name}/."""
    n_copied, n_errors = 0, 0
    for split in ("train", "test"):
        split_src = kaggle_root / split
        if not split_src.exists():
            print(f"[organise-rafdb] Warning: {split_src} not found, skipping.")
            continue
        for label, cls in RAFDB_LABEL_TO_CLASS.items():
            src_dir = split_src / str(label)
            dst_dir = out_root / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            if not src_dir.exists():
                continue
            for img_path in src_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    shutil.copy2(img_path, dst_dir / img_path.name)
                    n_copied += 1

    print(f"[organise-rafdb] Copied {n_copied} images to {out_root} ({n_errors} missing).")


# ---------------------------------------------------------------------------
# Synthetic sample data
# ---------------------------------------------------------------------------

def create_sample_data(n_per_dataset: int = 5) -> None:
    """
    Generate tiny synthetic datasets for smoke-testing the loaders.
    Creates random images and FLAME-shaped vertex arrays.
    """
    rng = np.random.default_rng(0)
    N_VERTS = 5023

    # ── NoW ───────────────────────────────────────────────────────────
    import cv2
    now_scans_dir = DATASETS_ROOT / "now" / "scans"
    now_lmk_dir   = DATASETS_ROOT / "now" / "scan_landmarks"
    now_scans_dir.mkdir(parents=True, exist_ok=True)
    now_lmk_dir.mkdir(parents=True, exist_ok=True)
    for subj in ["subject_001", "subject_002"]:
        for condition in ["neutral", "expression", "occlusion", "selfie"]:
            cond_dir = DATASETS_ROOT / "now" / subj / condition
            cond_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_dataset):
                img = rng.integers(50, 220, (224, 224, 3), dtype=np.uint8)
                cv2.imwrite(str(cond_dir / f"{i:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # one scan + landmarks per subject
        scan_verts = rng.standard_normal((5000, 3)).astype(np.float32) * 0.05
        _write_minimal_ply(now_scans_dir / f"{subj}.ply", scan_verts)
        landmarks = rng.standard_normal((7, 3)).astype(np.float32) * 0.05
        np.save(str(now_lmk_dir / f"{subj}.npy"), landmarks)

    # ── CoMA ──────────────────────────────────────────────────────────
    try:
        import trimesh
        _TRIMESH = True
    except ImportError:
        _TRIMESH = False

    FLAME_FACES = _get_sample_faces()

    expressions = ["bareteeth", "eyeblink", "mouth_extreme"]
    for subj in ["FaceTalk_001", "FaceTalk_002"]:
        for expr in expressions:
            expr_dir = DATASETS_ROOT / "coma" / subj / expr
            expr_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_dataset):
                verts = rng.standard_normal((N_VERTS, 3)).astype(np.float32) * 0.05
                ply_path = expr_dir / f"{i:05d}.ply"
                if _TRIMESH:
                    mesh = trimesh.Trimesh(vertices=verts, faces=FLAME_FACES, process=False)
                    mesh.export(str(ply_path))
                else:
                    # fallback: write minimal ascii ply
                    _write_minimal_ply(ply_path, verts)

    # ── TEMPEH ────────────────────────────────────────────────────────
    for subj in ["subject_T01", "subject_T02"]:
        for seq in ["seq_001"]:
            img_dir  = DATASETS_ROOT / "tempeh" / subj / seq / "images"
            scan_dir = DATASETS_ROOT / "tempeh" / subj / seq / "scans"
            img_dir.mkdir(parents=True, exist_ok=True)
            scan_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_dataset):
                frame = f"{i:06d}"
                img   = rng.integers(50, 220, (224, 224, 3), dtype=np.uint8)
                import cv2
                cv2.imwrite(str(img_dir / f"{frame}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                pts  = rng.standard_normal((800, 3)).astype(np.float32) * 0.05
                scan_path = scan_dir / f"{frame}.ply"
                if _TRIMESH:
                    cloud = trimesh.PointCloud(pts)
                    cloud.export(str(scan_path))
                else:
                    _write_minimal_ply(scan_path, pts)

    # ── RAF-DB ────────────────────────────────────────────────────────
    import cv2
    utkface_dir = DATASETS_ROOT / "utkface"
    utkface_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_per_dataset):
        age    = rng.integers(18, 80)
        gender = rng.integers(0, 2)
        race   = rng.integers(0, 5)
        fname  = f"{age}_{gender}_{race}_20170101{i:06d}.jpg.chip.jpg"
        img    = rng.integers(50, 220, (200, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(utkface_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    for person in ["John_Doe", "Jane_Smith", "Alex_Brown"]:
        person_dir = DATASETS_ROOT / "lfw" / person
        person_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_dataset):
            img = rng.integers(50, 220, (250, 250, 3), dtype=np.uint8)
            cv2.imwrite(str(person_dir / f"{person}_{i+1:04d}.jpg"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    for split in ("train", "test"):
        for cls in RAFDB_LABEL_TO_CLASS.values():
            cls_dir = DATASETS_ROOT / "raf-db" / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_dataset):
                img = rng.integers(50, 220, (112, 112, 3), dtype=np.uint8)
                cv2.imwrite(str(cls_dir / f"{i:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"[sample] Synthetic sample data written under {DATASETS_ROOT}.")


def _get_sample_faces() -> np.ndarray:
    """Return a tiny placeholder face array; real FLAME has 9976 faces."""
    # Minimal triangulation for 5023 vertices (just enough to be a valid mesh)
    faces = np.array([[i, i + 1, i + 2] for i in range(0, 5021, 3)], dtype=np.int32)
    return faces


def _write_minimal_ply(path: Path, vertices: np.ndarray) -> None:
    """Write a minimal ASCII .ply point cloud without trimesh."""
    n = len(vertices)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_datasets() -> None:
    """Print a simple presence-check for each dataset."""
    checks = {
        "NoW":      DATASETS_ROOT / "now",
        "CoMA":     DATASETS_ROOT / "coma",
        "TEMPEH":   DATASETS_ROOT / "tempeh",
        "UTKFace":  DATASETS_ROOT / "utkface",
        "LFW":      DATASETS_ROOT / "lfw",
        "RAF-DB":   DATASETS_ROOT / "raf-db",
    }
    for name, path in checks.items():
        n_files = sum(1 for _ in path.rglob("*") if _.is_file()) if path.exists() else 0
        status  = "✓" if n_files > 0 else "✗ (empty or missing)"
        print(f"  {name:12s}  {status}  ({n_files} files at {path})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset download helper for 3D face uncertainty project"
    )
    parser.add_argument(
        "--create-sample", action="store_true",
        help="Generate synthetic sample data for smoke-testing the loaders.",
    )
    parser.add_argument(
        "--organise-rafdb", action="store_true",
        help="Reorganise downloaded RAF-DB into class-folder layout.",
    )
    parser.add_argument(
        "--rafdb-raw", type=str, default=None,
        help="Path to extracted RAF-DB 'basic/' directory.",
    )
    parser.add_argument(
        "--rafdb-out", type=str, default=str(DATASETS_ROOT / "raf-db"),
        help="Output directory for organised RAF-DB.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Check which datasets are present.",
    )
    args = parser.parse_args()

    setup_directories()

    if args.create_sample:
        create_sample_data()
        return

    if args.organise_rafdb:
        if args.rafdb_raw is None:
            parser.error("--rafdb-raw is required when --organise-rafdb is set.")
        organise_rafdb(Path(args.rafdb_raw), Path(args.rafdb_out))
        return

    if args.validate:
        print("\nDataset presence check:")
        validate_datasets()
        return

    # Default: print all instructions
    print_now_instructions()
    print_coma_instructions()
    print_tempeh_instructions()
    print_utkface_instructions()
    print_lfw_instructions()
    print_rafdb_instructions()
    print(f"""
─────────────────────────────────────────────────────────────────────
After downloading, verify everything is in place with:
    python scripts/download_datasets.py --validate

To generate synthetic sample data for testing (no download needed):
    python scripts/download_datasets.py --create-sample
─────────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
