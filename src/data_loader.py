"""
data_loader.py
==============
Unified dataset loader for the 3D face uncertainty pipeline.

Three evaluation datasets return paired (image, ground-truth 3D data):

  NoW        — images (neutral/expression/occlusion/selfie) + raw 3D scan per
               subject (~120K pts).  The standard benchmark used by DECA,
               EMOCA, SMIRK, and SHeaP; evaluation is scan-to-mesh distance.
  CoMA       — FLAME-registered meshes (5023, 3) + optional rendered images.
               Mesh-only if no rendered images are available.
  TEMPEH     — images + FLAME registrations (5023, 3).
               Stored under flame/ sub-directory by extract_tempeh_subset.py.

One classifier dataset:
  RAF-DB     — face images organised into 7 expression class folders.
               Handled by src.emotion_dataset.EmotionDataset.

Key public API
--------------
  loader = FaceDatasetLoader(data_root='./datasets')

  # Fixed-size subsets
  items_100 = loader.load_now_subset(100)
  items_50  = loader.load_coma_subset(50)

  # All configured sizes at once (nested: 10 ⊆ 50 ⊆ 100)
  subsets   = loader.create_subsets('now')   # → {10: [...], 50: [...], 100: [...]}

  # Mixed subset from several datasets
  mixed     = loader.create_mixed_subset(n_total=100)

  # RAF-DB classifier dataset
  train_ds  = loader.load_raf_db(split='train')           # EmotionDataset
  small_sub = loader.create_raf_db_subset('train', n_per_class=10)  # Subset

Every evaluation sample dict has:
  'image'       : np.ndarray (H, W, 3) uint8 RGB | None
  'gt_vertices' : np.ndarray (5023, 3) float32   | None  (FLAME-registered)
  'gt_scan'     : np.ndarray (N, 3) float32       | None  (raw scan)
  'subject_id'  : str
  'expression'  : str
  'frame_id'    : str
  'gt_scan_landmarks' : np.ndarray (7, 3) float32 | None  (NoW alignment pts)
  'condition'   : str                              (NoW: neutral/expression/occlusion/selfie)
  'dataset'     : 'now' | 'coma' | 'tempeh' | 'utkface' | 'lfw'

Change SUBSET_SIZES at the top of this file to reconfigure default sizes.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# ── configurable subset sizes ────────────────────────────────────────────────
SUBSET_SIZES: List[int] = [10, 50, 100]

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
N_FLAME_VERTS   = 5023
_PROJECT_ROOT   = Path(__file__).resolve().parent.parent
_PARTITIONS_ROOT = _PROJECT_ROOT / 'partitions'

# Keys that become Path objects when reconstructing a JSON record into a raw item.
_PATH_FIELDS = {'img_path', 'ply_path', 'flame_path', 'scan_path', 'lmk_path'}


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_image(path: Path) -> Optional[np.ndarray]:
    """Read an image as uint8 RGB (H, W, 3), or return None on failure."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_ply_vertices(path: Path) -> Optional[np.ndarray]:
    """
    Load vertices from a mesh file (.ply, .obj, or any format trimesh supports).
    Tries trimesh first; falls back to format-specific ASCII parsers.
    Returns (N, 3) float32 or None on failure.
    """
    try:
        import trimesh
        mesh = trimesh.load(str(path), process=False)
        if hasattr(mesh, 'vertices'):
            return np.asarray(mesh.vertices, dtype=np.float32)
        return np.asarray(mesh, dtype=np.float32)
    except Exception:
        pass

    ext = path.suffix.lower()

    # OBJ fallback: lines starting with "v " are vertices
    if ext == '.obj':
        try:
            verts = []
            with open(path, 'r') as fh:
                for line in fh:
                    if line.startswith('v '):
                        parts = line.split()
                        if len(parts) >= 4:
                            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            return np.array(verts, dtype=np.float32) if verts else None
        except Exception:
            return None

    # PLY ASCII fallback
    try:
        verts = []
        in_data = False
        with open(path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if line == 'end_header':
                    in_data = True
                    continue
                if in_data:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        except ValueError:
                            pass
        return np.array(verts, dtype=np.float32) if verts else None
    except Exception:
        return None


def _load_pp_landmarks(path: Path) -> Optional[np.ndarray]:
    """
    Parse a MeshLab .pp (PickedPoints XML) file into an (N, 3) float32 array.
    Returns None on any parse failure.
    """
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(path))
        pts = []
        for pt in tree.getroot().iter('point'):
            if pt.get('active', '1') == '1':
                pts.append([float(pt.get('x')), float(pt.get('y')), float(pt.get('z'))])
        return np.array(pts, dtype=np.float32) if pts else None
    except Exception:
        return None


def _load_flame_npz(path: Path) -> Optional[np.ndarray]:
    """
    Load (5023, 3) FLAME vertices from a .npz file.
    Tries common key names: 'vertices', 'verts', 'v'.
    Returns float32 array or None.
    """
    try:
        data = np.load(str(path), allow_pickle=True)
        for key in ('vertices', 'verts', 'v', 'vertex'):
            if key in data:
                arr = data[key].astype(np.float32)
                if arr.shape == (N_FLAME_VERTS, 3):
                    return arr
                # batch dim (1, 5023, 3)
                if arr.ndim == 3 and arr.shape[1] == N_FLAME_VERTS:
                    return arr[0]
        # last resort: first array with right shape
        for key in data.files:
            arr = data[key]
            if hasattr(arr, 'shape'):
                if arr.shape == (N_FLAME_VERTS, 3):
                    return arr.astype(np.float32)
        return None
    except Exception:
        return None


def _stratified_sample(
    items: List,
    n: int,
    key_fn,
    rng: np.random.Generator,
) -> List:
    """
    Stratified random sample of `n` items from `items`.
    key_fn(item) → stratum label.  Samples are drawn proportionally from
    each stratum; remainder allocated to strata with the most samples.
    """
    if n >= len(items):
        return list(items)

    from collections import defaultdict
    strata: Dict[str, List] = defaultdict(list)
    for item in items:
        strata[key_fn(item)].append(item)

    n_strata   = len(strata)
    base_count = n // n_strata
    remainder  = n - base_count * n_strata

    # Sort strata by size descending so remainder goes to largest
    sorted_strata = sorted(strata.items(), key=lambda x: -len(x[1]))
    selected = []
    for i, (_label, group) in enumerate(sorted_strata):
        cnt = base_count + (1 if i < remainder else 0)
        cnt = min(cnt, len(group))
        perm = rng.permutation(len(group))[:cnt]
        selected.extend([group[j] for j in perm])
    rng.shuffle(selected)
    return selected


# ── main class ───────────────────────────────────────────────────────────────

class FaceDatasetLoader:
    """
    Unified loader for NoW, CoMA, TEMPEH (evaluation), UTKFace, LFW
    (uncertainty estimation) and RAF-DB (classifier).

    Parameters
    ----------
    data_root     : root directory containing dataset sub-folders.
    subset_sizes  : list of N values for create_subsets(); defaults to SUBSET_SIZES.
    render_coma   : if True, attempt to render a frontal-view image for CoMA meshes
                    that have no accompanying .jpg/.png (requires trimesh).
    """

    def __init__(
        self,
        data_root: str = './datasets',
        subset_sizes: Optional[List[int]] = None,
        render_coma: bool = False,
        max_img_dim: int = 0,
    ):
        self.data_root    = Path(data_root).resolve()
        self.subset_sizes = sorted(subset_sizes or SUBSET_SIZES)
        self.render_coma  = render_coma
        self.max_img_dim  = max_img_dim  # 0 = no resize; >0 = resize longest edge

    # ────────────────────────────────────────────────────────────────────────
    # NoW benchmark
    # ────────────────────────────────────────────────────────────────────────

    def load_now_subset(
        self,
        num_samples: int,
        seed: int = 42,
    ) -> List[Dict]:
        """
        Load `num_samples` (image, raw-scan) pairs from the NoW validation set.

        Each image is paired with its subject's neutral 3D scan (~120K pts).
        Evaluation uses scan-to-mesh distance, same as the published NoW metric.

        Expected layout::
            now/{subject}/{condition}/*.jpg     (condition: neutral/expression/occlusion/selfie)
            now/scans/{subject}.ply             (one scan per subject)
            now/scan_landmarks/{subject}.npy    (7 face landmarks, shape (7,3))

        The loader auto-detects common alternate layouts (e.g. an extra
        'iphone_pictures/' or 'final_release_version/' prefix).
        """
        all_items = self._collect_now()
        return self._sample(all_items, num_samples, seed, dataset='now')

    def _collect_now(self) -> List[Dict]:
        root = self.data_root / 'now'
        if not root.exists():
            warnings.warn(f"NoW directory not found: {root}", UserWarning)
            return []

        # ── index scans: scans/{subject}/*.obj (or *.ply) ──────────────────
        scan_root = root / 'scans'
        scan_map: Dict[str, Path] = {}
        if scan_root.is_dir():
            for subj_dir in scan_root.iterdir():
                if not subj_dir.is_dir():
                    continue
                for p in subj_dir.iterdir():
                    if p.suffix.lower() in {'.obj', '.ply'}:
                        scan_map[subj_dir.name] = p
                        break

        # ── index landmarks: scans_lmks_onlypp/{subject}/*.pp (or *.npy) ──
        lmk_root: Optional[Path] = None
        for candidate in ['scans_lmks_onlypp', 'scan_landmarks']:
            p = root / candidate
            if p.is_dir():
                lmk_root = p
                break

        lmk_map: Dict[str, Path] = {}
        if lmk_root:
            for subj_dir in lmk_root.iterdir():
                if not subj_dir.is_dir():
                    continue
                for p in subj_dir.iterdir():
                    if p.suffix.lower() in {'.pp', '.npy'}:
                        lmk_map[subj_dir.name] = p
                        break

        # ── build item list from imagepathsvalidation.txt ──────────────────
        # Each line: {subject_id}/{condition}/{frame}.jpg
        # Images may not exist yet (downloaded separately); items are still
        # created so that scan/landmark data is accessible immediately.
        img_list = root / 'imagepathsvalidation.txt'
        items: List[Dict] = []

        if img_list.exists():
            with open(img_list) as fh:
                for line in fh:
                    rel = line.strip()
                    if not rel:
                        continue
                    parts = Path(rel).parts
                    subj_id   = parts[0]
                    condition = parts[1] if len(parts) >= 3 else ''
                    frame     = Path(rel).stem
                    items.append({
                        '_img_path'  : root / rel,
                        '_scan_path' : scan_map.get(subj_id),
                        '_lmk_path'  : lmk_map.get(subj_id),
                        'gt_vertices': None,
                        'gt_scan'    : None,
                        'subject_id' : subj_id,
                        'condition'  : condition,
                        'expression' : condition,
                        'frame_id'   : frame,
                        'dataset'    : 'now',
                    })
        else:
            # Fallback: scan image directories if txt not present
            for subj_dir in sorted(root.iterdir()):
                if not subj_dir.is_dir() or subj_dir.name in {'scans', 'scans_lmks_onlypp',
                                                               'scan_landmarks'}:
                    continue
                subj_id = subj_dir.name
                for cond_dir in sorted(subj_dir.iterdir()):
                    if not cond_dir.is_dir():
                        continue
                    for img in sorted(cond_dir.iterdir()):
                        if img.suffix.lower() in _IMG_EXTS:
                            items.append({
                                '_img_path'  : img,
                                '_scan_path' : scan_map.get(subj_id),
                                '_lmk_path'  : lmk_map.get(subj_id),
                                'gt_vertices': None,
                                'gt_scan'    : None,
                                'subject_id' : subj_id,
                                'condition'  : cond_dir.name,
                                'expression' : cond_dir.name,
                                'frame_id'   : img.stem,
                                'dataset'    : 'now',
                            })

        if not items:
            warnings.warn("NoW: no items found. Images not yet downloaded — "
                          "run scripts/extract_now_images.py to get them.", UserWarning)
        return items

    # ────────────────────────────────────────────────────────────────────────
    # CoMA
    # ────────────────────────────────────────────────────────────────────────

    def load_coma_subset(
        self,
        num_samples: int,
        seed: int = 42,
    ) -> List[Dict]:
        """
        Load `num_samples` (optional-image, FLAME-vertices) pairs from CoMA.

        Expected layout::
            coma/{subject}/{expression}/{idx:05d}.ply
            coma/{subject}/{expression}/{idx:05d}.jpg   (optional)

        If no jpg/png is found alongside the .ply, image is None unless
        render_coma=True was passed to the constructor.
        """
        all_items = self._collect_coma()
        return self._sample(
            all_items, num_samples, seed, dataset='coma',
            stratum_key=lambda x: x['expression'],
        )

    def _collect_coma(self) -> List[Dict]:
        root = self.data_root / 'coma'
        if not root.exists():
            warnings.warn(f"CoMA directory not found: {root}", UserWarning)
            return []

        items: List[Dict] = []

        for subj_dir in sorted(root.iterdir()):
            if not subj_dir.is_dir():
                continue
            for expr_dir in sorted(subj_dir.iterdir()):
                if not expr_dir.is_dir():
                    continue

                ply_paths = sorted(
                    p for p in expr_dir.iterdir()
                    if p.suffix.lower() == '.ply'
                )

                for ply_path in ply_paths:
                    stem     = ply_path.stem
                    img_path = None
                    for ext in ('.jpg', '.jpeg', '.png'):
                        candidate = expr_dir / f"{stem}{ext}"
                        if candidate.exists():
                            img_path = candidate
                            break

                    items.append({
                        '_img_path'    : img_path,
                        '_ply_path'    : ply_path,
                        'gt_scan'      : None,
                        'subject_id'   : subj_dir.name,
                        'expression'   : expr_dir.name,
                        'frame_id'     : stem,
                        'dataset'      : 'coma',
                    })
        return items

    def _resolve_coma_item(self, item: Dict) -> Dict:
        """Load gt_vertices from PLY and optionally render an image."""
        ply_path  = item.pop('_ply_path', None)
        img_path  = item.pop('_img_path', None)

        if ply_path and ply_path.exists():
            item['gt_vertices'] = _load_ply_vertices(ply_path)
        else:
            item['gt_vertices'] = None

        # Image: load from disk if present, else render frontal view
        if img_path and img_path.exists():
            item['image'] = _read_image(img_path)
        elif self.render_coma and item.get('gt_vertices') is not None:
            item['image'] = _render_mesh_frontal(item['gt_vertices'])
        else:
            item['image'] = None

        return item

    # ────────────────────────────────────────────────────────────────────────
    # TEMPEH
    # ────────────────────────────────────────────────────────────────────────

    def load_tempeh_subset(
        self,
        num_samples: int,
        seed: int = 42,
    ) -> List[Dict]:
        """
        Load `num_samples` (image, raw-scan) pairs from TEMPEH.

        Expected layout::
            tempeh/{subject}/{sequence}/images/{frame}.png
            tempeh/{subject}/{sequence}/scans/{frame}.ply
        """
        all_items = self._collect_tempeh()
        return self._sample(all_items, num_samples, seed, dataset='tempeh')

    def _collect_tempeh(self) -> List[Dict]:
        root = self.data_root / 'tempeh'
        if not root.exists():
            warnings.warn(f"TEMPEH directory not found: {root}", UserWarning)
            return []

        items: List[Dict] = []

        for subj_dir in sorted(root.iterdir()):
            if not subj_dir.is_dir():
                continue
            for seq_dir in sorted(subj_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue

                img_dir   = seq_dir / 'images'
                flame_dir = seq_dir / 'flame'   # FLAME registrations (from extract script)
                scan_dir  = seq_dir / 'scans'   # raw scan fallback

                if not img_dir.is_dir():
                    continue

                img_paths = sorted(
                    p for p in img_dir.iterdir()
                    if p.suffix.lower() in _IMG_EXTS
                )

                for img_path in img_paths:
                    stem = img_path.stem

                    # Prefer FLAME registration; fall back to raw scan
                    flame_path = None
                    if flame_dir.is_dir():
                        for ext in ('.npz', '.ply', '.obj'):
                            cand = flame_dir / f"{stem}{ext}"
                            if cand.exists():
                                flame_path = cand
                                break

                    scan_path = scan_dir / f"{stem}.ply" if scan_dir.is_dir() else None

                    items.append({
                        '_img_path'   : img_path,
                        '_flame_path' : flame_path,
                        '_scan_path'  : scan_path,
                        'gt_vertices' : None,
                        'subject_id'  : subj_dir.name,
                        'expression'  : seq_dir.name,
                        'frame_id'    : stem,
                        'dataset'     : 'tempeh',
                    })
        return items

    # ────────────────────────────────────────────────────────────────────────
    # Subset factories
    # ────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # UTKFace
    # ────────────────────────────────────────────────────────────────────────

    def load_utkface_subset(self, num_samples: int, seed: int = 42) -> List[Dict]:
        """
        Load `num_samples` images from UTKFace (aligned & cropped).

        No ground-truth 3D — use for uncertainty estimation only (methods 1–4).

        Expected layout::
            utkface/{age}_{gender}_{race}_{datetime}.jpg.chip.jpg
            OR utkface/UTKFace/*.jpg.chip.jpg  (if zip extracted with folder)
        """
        all_items = self._collect_utkface()
        return self._sample(all_items, num_samples, seed, dataset='utkface')

    def _collect_utkface(self) -> List[Dict]:
        root = self.data_root / 'utkface'
        if not root.exists():
            warnings.warn(f"UTKFace directory not found: {root}", UserWarning)
            return []

        # Images may be flat in root or inside a single subdirectory
        img_paths = sorted(
            p for p in root.rglob('*')
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )

        items: List[Dict] = []
        for img_path in img_paths:
            # Filename: {age}_{gender}_{race}_{datetime}.jpg.chip.jpg
            name_parts = img_path.name.split('_')
            age    = name_parts[0] if len(name_parts) > 0 else 'unknown'
            gender = {'0': 'male', '1': 'female'}.get(
                name_parts[1] if len(name_parts) > 1 else '', 'unknown'
            )
            items.append({
                '_img_path'         : img_path,
                'gt_vertices'       : None,
                'gt_scan'           : None,
                'gt_scan_landmarks' : None,
                'subject_id'        : img_path.stem,
                'condition'         : '',
                'expression'        : '',
                'frame_id'          : img_path.stem,
                'dataset'           : 'utkface',
                'age'               : age,
                'gender'            : gender,
            })
        return items

    # ────────────────────────────────────────────────────────────────────────
    # LFW
    # ────────────────────────────────────────────────────────────────────────

    def load_lfw_subset(self, num_samples: int, seed: int = 42) -> List[Dict]:
        """
        Load `num_samples` images from LFW (Labeled Faces in the Wild).

        No ground-truth 3D — use for uncertainty estimation only (methods 1–4).

        Expected layout::
            lfw/{person_name}/{person_name}_{XXXX}.jpg
        """
        all_items = self._collect_lfw()
        return self._sample(all_items, num_samples, seed, dataset='lfw')

    def _collect_lfw(self) -> List[Dict]:
        root = self.data_root / 'lfw'
        if not root.exists():
            warnings.warn(f"LFW directory not found: {root}", UserWarning)
            return []

        items: List[Dict] = []
        for person_dir in sorted(root.iterdir()):
            if not person_dir.is_dir():
                continue
            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() in _IMG_EXTS:
                    items.append({
                        '_img_path'         : img_path,
                        'gt_vertices'       : None,
                        'gt_scan'           : None,
                        'gt_scan_landmarks' : None,
                        'subject_id'        : person_dir.name,
                        'condition'         : '',
                        'expression'        : '',
                        'frame_id'          : img_path.stem,
                        'dataset'           : 'lfw',
                    })
        return items

    # ────────────────────────────────────────────────────────────────────────
    # Partition loader
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _record_to_item(record: dict) -> dict:
        """Reconstruct a raw item dict from a JSON partition record."""
        import json as _json
        item: dict = {}
        for key, val in record.items():
            if key in _PATH_FIELDS:
                item[f'_{key}'] = (_PROJECT_ROOT / val) if val is not None else None
            else:
                item[key] = val
        return item

    def load_partition(self, json_path) -> List[Dict]:
        """
        Load and materialise a partition saved by scripts/create_partitions.py.

        Parameters
        ----------
        json_path : str or Path — path to a partition JSON file.

        Returns
        -------
        List[Dict] — same format as create_subsets / load_*_subset output.
        """
        import json as _json
        json_path = Path(json_path)
        payload   = _json.loads(json_path.read_text())
        dataset   = payload['meta'].get('dataset', None)
        raw_items = [self._record_to_item(r) for r in payload['samples']]
        return self._materialise(raw_items, dataset)

    # ────────────────────────────────────────────────────────────────────────
    # Subset factories
    # ────────────────────────────────────────────────────────────────────────

    def create_subsets(
        self,
        dataset: str = 'now',
        seed: int = 42,
    ) -> Dict[int, List[Dict]]:
        """
        Return nested subsets for every size in self.subset_sizes.

        Subsets are nested: the 10-sample subset is always a prefix of the
        50-sample subset, which is a prefix of the 100-sample subset.  This
        makes incremental experiments reproducible.

        Returns
        -------
        dict : {10: [<10 samples>], 50: [<50 samples>], 100: [<100 samples>]}
        """
        loaders = {
            'now'    : self._collect_now,
            'coma'   : self._collect_coma,
            'tempeh' : self._collect_tempeh,
            'utkface': self._collect_utkface,
            'lfw'    : self._collect_lfw,
        }
        if dataset not in loaders:
            raise ValueError(
                f"dataset must be one of {list(loaders)}, got '{dataset}'."
            )

        # ── Prefer pre-built partition files when available ──────────────────
        result: Dict[int, List[Dict]] = {}
        missing_sizes = []
        for n in self.subset_sizes:
            p = _PARTITIONS_ROOT / dataset / f'n{n:05d}.json'
            if p.exists():
                result[n] = self.load_partition(p)
            else:
                missing_sizes.append(n)

        if not missing_sizes:
            return result

        # ── Fall back to in-memory collection for any size without a file ────
        all_items = loaders[dataset]()
        if not all_items:
            warnings.warn(f"No items found for dataset '{dataset}'.", UserWarning)
            for n in missing_sizes:
                result[n] = []
            return result

        rng      = np.random.default_rng(seed)
        indices  = rng.permutation(len(all_items))
        max_n    = min(max(missing_sizes), len(all_items))
        shuffled = [all_items[i] for i in indices[:max_n]]

        for n in missing_sizes:
            slice_n = shuffled[:min(n, len(shuffled))]
            result[n] = self._materialise(slice_n, dataset)

        if max(self.subset_sizes) > len(all_items):
            warnings.warn(
                f"Requested max subset size {max(self.subset_sizes)} but only "
                f"{len(all_items)} samples available for '{dataset}'.",
                UserWarning,
            )
        return result

    def create_mixed_subset(
        self,
        n_total: int,
        datasets: Optional[List[str]] = None,
        seed: int = 42,
    ) -> List[Dict]:
        """
        Sample proportionally from each available evaluation dataset.

        Parameters
        ----------
        n_total  : total number of samples requested.
        datasets : list subset of ['now', 'coma', 'tempeh', 'utkface', 'lfw'].
                   Defaults to the three GT datasets; datasets with no local
                   files are skipped silently.
        seed     : random seed.

        Returns
        -------
        List[Dict] — combined samples, shuffled, length ≤ n_total.
        """
        if datasets is None:
            datasets = ['now', 'coma', 'tempeh']

        collectors = {
            'now'    : self._collect_now,
            'coma'   : self._collect_coma,
            'tempeh' : self._collect_tempeh,
            'utkface': self._collect_utkface,
            'lfw'    : self._collect_lfw,
        }

        pools: Dict[str, List] = {}
        for ds in datasets:
            items = collectors[ds]()
            if items:
                pools[ds] = items

        if not pools:
            warnings.warn("No data found for any requested dataset.", UserWarning)
            return []

        rng = np.random.default_rng(seed)

        # Proportional allocation
        n_ds   = len(pools)
        base   = n_total // n_ds
        rem    = n_total - base * n_ds
        sorted_ds = sorted(pools.keys(), key=lambda k: -len(pools[k]))

        mixed_raw: List[Dict] = []
        for i, ds in enumerate(sorted_ds):
            cnt   = base + (1 if i < rem else 0)
            items = pools[ds]
            idx   = rng.permutation(len(items))[:min(cnt, len(items))]
            mixed_raw.extend([items[j] for j in idx])

        rng.shuffle(mixed_raw)
        return self._materialise(mixed_raw, dataset=None)

    # ────────────────────────────────────────────────────────────────────────
    # RAF-DB / classifier dataset
    # ────────────────────────────────────────────────────────────────────────

    def load_raf_db(
        self,
        split: str = 'train',
        mode: str = 'plain',
        uncertainty_root: Optional[str] = None,
        face_regressor=None,
        uncertainty_fn=None,
        camera_params: Optional[dict] = None,
        image_size: int = 224,
    ):
        """
        Return an EmotionDataset wrapping the RAF-DB class-folder layout.

        Parameters
        ----------
        split            : 'train' or 'test'.
        mode             : 'plain' or 'uncertainty_weighted' (see EmotionDataset).
        uncertainty_root : pre-computed .npy confidence maps root (optional).
        face_regressor   : on-the-fly regressor (optional).
        uncertainty_fn   : on-the-fly uncertainty function (optional).
        camera_params    : weak-perspective camera parameters (optional).
        image_size       : image resize target (default 224).
        """
        from src.emotion_dataset import EmotionDataset

        rafdb_root = self.data_root / 'raf-db'
        if not rafdb_root.exists():
            raise FileNotFoundError(
                f"RAF-DB not found at {rafdb_root}. "
                "Run: python scripts/download_datasets.py for instructions."
            )

        return EmotionDataset(
            root=str(rafdb_root),
            split=split,
            mode=mode,
            uncertainty_root=uncertainty_root,
            face_regressor=face_regressor,
            uncertainty_fn=uncertainty_fn,
            camera_params=camera_params,
            image_size=image_size,
        )

    def create_raf_db_subset(
        self,
        split: str = 'train',
        n_per_class: int = 10,
        seed: int = 42,
        mode: str = 'plain',
        **kwargs,
    ):
        """
        Return a torch Subset of RAF-DB with exactly n_per_class samples per
        expression class (7 classes → total = 7 * n_per_class at most).

        Useful for creating fixed-size subsets of sizes 10, 50, 100, etc.
        The number of samples per class is: ceil(n_total / 7).

        Parameters
        ----------
        split       : 'train' or 'test'.
        n_per_class : maximum images to include from each class.
        seed        : random seed for per-class shuffling.
        mode        : forwarded to EmotionDataset.
        **kwargs    : forwarded to EmotionDataset.
        """
        import torch
        from src.emotion_dataset import EmotionDataset, _CLASS_TO_IDX

        dataset = self.load_raf_db(split=split, mode=mode, **kwargs)
        if len(dataset) == 0:
            return torch.utils.data.Subset(dataset, [])

        rng = np.random.default_rng(seed)

        # Group sample indices by class
        from collections import defaultdict
        class_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, (_path, label) in enumerate(dataset.samples):
            class_indices[label].append(idx)

        selected: List[int] = []
        for label in sorted(class_indices.keys()):
            idx_list = class_indices[label]
            perm     = rng.permutation(len(idx_list))
            take     = min(n_per_class, len(idx_list))
            selected.extend([idx_list[perm[i]] for i in range(take)])

        rng.shuffle(selected)
        return torch.utils.data.Subset(dataset, selected)

    # ────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────────────────────────────────────

    def _sample(
        self,
        all_items: List[Dict],
        n: int,
        seed: int,
        dataset: str,
        stratum_key=None,
    ) -> List[Dict]:
        """
        Randomly sample `n` items, optionally with stratification.
        Then materialise (load images / vertices) and return.
        """
        if not all_items:
            return []

        rng = np.random.default_rng(seed)

        if n >= len(all_items):
            sampled = list(all_items)
        elif stratum_key is not None:
            sampled = _stratified_sample(all_items, n, stratum_key, rng)
        else:
            perm    = rng.permutation(len(all_items))
            sampled = [all_items[i] for i in perm[:n]]

        if n > len(all_items):
            warnings.warn(
                f"Requested {n} samples from '{dataset}' but only "
                f"{len(all_items)} are available.",
                UserWarning,
            )

        return self._materialise(sampled, dataset)

    def _materialise(
        self,
        raw_items: List[Dict],
        dataset: Optional[str],
    ) -> List[Dict]:
        """
        Convert raw catalogue entries (which may hold Path objects) into
        fully loaded dicts with 'image', 'gt_vertices', 'gt_scan'.
        """
        result = []
        for item in raw_items:
            item = dict(item)  # shallow copy
            ds = item.get('dataset', dataset)

            if ds == 'coma':
                item = self._resolve_coma_item(item)
            else:
                # Load image from path if available
                img_path = item.pop('_img_path', None)
                if img_path is not None and img_path.exists():
                    img = _read_image(img_path)
                    if img is not None and self.max_img_dim > 0:
                        h, w = img.shape[:2]
                        if max(h, w) > self.max_img_dim:
                            scale = self.max_img_dim / max(h, w)
                            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                                             interpolation=cv2.INTER_AREA)
                    item['image'] = img
                elif 'image' not in item:
                    item['image'] = None

                # TEMPEH: FLAME registration takes priority over raw scan
                flame_path = item.pop('_flame_path', None)
                if flame_path is not None and flame_path.exists():
                    if flame_path.suffix == '.npz':
                        item['gt_vertices'] = _load_flame_npz(flame_path)
                    else:
                        verts = _load_ply_vertices(flame_path)
                        if verts is not None and verts.shape == (N_FLAME_VERTS, 3):
                            item['gt_vertices'] = verts
                if 'gt_vertices' not in item:
                    item['gt_vertices'] = None

                scan_path = item.pop('_scan_path', None)
                if scan_path is not None and scan_path.exists():
                    item['gt_scan'] = _load_ply_vertices(scan_path)
                elif 'gt_scan' not in item:
                    item['gt_scan'] = None

                # NoW: scan landmarks (.pp XML or .npy) for evaluation alignment
                lmk_path = item.pop('_lmk_path', None)
                if lmk_path is not None and lmk_path.exists():
                    if lmk_path.suffix.lower() == '.pp':
                        item['gt_scan_landmarks'] = _load_pp_landmarks(lmk_path)
                    else:
                        try:
                            item['gt_scan_landmarks'] = np.load(
                                str(lmk_path), allow_pickle=True
                            ).astype(np.float32)
                        except Exception:
                            item['gt_scan_landmarks'] = None
                elif 'gt_scan_landmarks' not in item:
                    item['gt_scan_landmarks'] = None

            result.append(item)
        return result


# ── CoMA frontal renderer ─────────────────────────────────────────────────────

def _render_mesh_frontal(
    vertices: np.ndarray,
    image_size: int = 224,
) -> Optional[np.ndarray]:
    """
    Render a frontal-view depth-scatter image of a FLAME mesh using matplotlib Agg.
    Works in headless / WSL2 environments with no OpenGL requirement.
    Returns (H, W, 3) uint8 RGB, or None on failure.
    """
    try:
        import io
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        v = np.asarray(vertices, dtype=np.float32)
        # FLAME: X = right, Y = up, Z = toward camera.
        # Plot X (right) vs -Y (down) so the face is upright; colour by Z depth.
        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.scatter(v[:, 0], -v[:, 1], c=v[:, 2], cmap='gray', s=0.6, alpha=0.9)
        ax.set_aspect('equal')
        ax.axis('off')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        buf.seek(0)
        img_arr = np.frombuffer(buf.read(), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (image_size, image_size))
    except Exception:
        return None


# ── EDA helper ────────────────────────────────────────────────────────────────

def perform_basic_eda(dataset_subset: List[Dict]) -> Dict:
    """
    Compute summary statistics for a loaded dataset subset.

    Checks image resolution, face bounding boxes (from non-background pixels),
    and vertex spatial extents.

    Returns
    -------
    dict with keys:
      'n_samples'          : int
      'n_with_image'       : int
      'n_with_gt_vertices' : int
      'n_with_gt_scan'     : int
      'image_sizes'        : list of (H, W) tuples
      'vertex_bbox_mm'     : {'min': (3,), 'max': (3,)} over all valid vertices
      'dataset_counts'     : {dataset_name: count}
    """
    from collections import Counter

    n_total    = len(dataset_subset)
    n_img      = sum(1 for s in dataset_subset if s.get('image') is not None)
    n_gt_verts = sum(1 for s in dataset_subset if s.get('gt_vertices') is not None)
    n_gt_scan  = sum(1 for s in dataset_subset if s.get('gt_scan') is not None)

    img_sizes = [
        s['image'].shape[:2]  # (H, W)
        for s in dataset_subset
        if s.get('image') is not None
    ]

    all_verts = np.concatenate(
        [s['gt_vertices'] for s in dataset_subset if s.get('gt_vertices') is not None],
        axis=0,
    ) if n_gt_verts > 0 else None

    dataset_counts = dict(Counter(s.get('dataset', 'unknown') for s in dataset_subset))

    summary = {
        'n_samples'          : n_total,
        'n_with_image'       : n_img,
        'n_with_gt_vertices' : n_gt_verts,
        'n_with_gt_scan'     : n_gt_scan,
        'image_sizes'        : img_sizes,
        'vertex_bbox_mm'     : {
            'min': all_verts.min(axis=0).tolist() if all_verts is not None else None,
            'max': all_verts.max(axis=0).tolist() if all_verts is not None else None,
        },
        'dataset_counts'     : dataset_counts,
    }

    print(f"EDA summary ({n_total} samples):")
    print(f"  Images available  : {n_img}")
    print(f"  GT vertices (FLAME): {n_gt_verts}")
    print(f"  GT scans (raw)    : {n_gt_scan}")
    if img_sizes:
        hs, ws = zip(*img_sizes)
        print(f"  Image H range     : {min(hs)}–{max(hs)} px")
        print(f"  Image W range     : {min(ws)}–{max(ws)} px")
    if all_verts is not None:
        print(f"  Vertex X range    : {all_verts[:,0].min():.3f} – {all_verts[:,0].max():.3f}")
        print(f"  Vertex Y range    : {all_verts[:,1].min():.3f} – {all_verts[:,1].max():.3f}")
        print(f"  Vertex Z range    : {all_verts[:,2].min():.3f} – {all_verts[:,2].max():.3f}")
    print(f"  Dataset breakdown : {dataset_counts}")

    return summary
