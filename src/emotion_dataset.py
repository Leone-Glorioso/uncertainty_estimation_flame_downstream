"""
emotion_dataset.py
===================
PyTorch Dataset for RAF-DB / AffectNet / FER2013-style facial expression
datasets, with optional per-pixel uncertainty weighting fed by pre-computed
FLAME confidence maps (see src/precompute_uncertainty_maps.py) or an
on-the-fly face regressor.  Used by src/run_classifier_experiment.py,
src/downstream_tuning.py, and main.py's downstream stage.
"""

import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

EXPRESSION_CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(EXPRESSION_CLASSES)}

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


class EmotionDataset(Dataset):
    """
    Facial expression dataset compatible with RAF-DB and AffectNet-style layouts.

    Expected directory layout::

        root/
          {split}/
            anger/    img1.jpg  img2.jpg  ...
            disgust/  ...
            fear/     ...
            happy/    ...
            neutral/  ...
            sad/      ...
            surprise/ ...

    Two modes
    ---------
    'plain'
        Returns the raw image tensor and the class label.

    'uncertainty_weighted'
        Multiplies each image by a per-pixel confidence map (inverted uncertainty)
        before returning it.  This forces the classifier to down-weight occluded
        or ambiguous facial regions flagged with high uncertainty.

        Confidence maps are loaded from ``uncertainty_root`` as .npy files that
        mirror the image tree exactly (same relative path, different root,
        extension replaced with .npy).  If a .npy file is missing the sample is
        returned with an all-ones map and a warning is emitted once.

        Alternatively, maps can be computed on-the-fly by supplying a
        ``face_regressor`` and ``uncertainty_fn`` (slow; recommended only for
        debugging with a few images).

    Pre-computing maps (recommended for training)
    ---------------------------------------------
    Run ``src/precompute_uncertainty_maps.py`` once to populate ``uncertainty_root``
    before starting a training run.

    Parameters
    ----------
    root : str
        Dataset root — parent of split subdirectories.
    split : str
        Name of the split subdirectory, e.g. 'train' or 'test'.
    mode : str
        'plain' or 'uncertainty_weighted'.
    uncertainty_root : str, optional
        Root for pre-computed confidence-map .npy files.  Required when
        ``mode='uncertainty_weighted'`` and no on-the-fly regressor is given.
    face_regressor : object, optional
        On-the-fly regressor; must expose ``get_vertices(image_np) → (5023, 3)``.
    uncertainty_fn : callable, optional
        Called as ``uncertainty_fn(regressor, image_np) → (5023, 1)`` variance.
    camera_params : dict, optional
        Weak-perspective camera parameters forwarded to project_variance_to_2d.
    transform : callable, optional
        torchvision transform applied to the (possibly masked) PIL image.
        Defaults to Resize(image_size) + ToTensor + ImageNet normalisation.
    image_size : int
        Target image size (square).  Used only when ``transform`` is None.
    """

    CLASSES = EXPRESSION_CLASSES

    def __init__(
        self,
        root: str,
        split: str = 'train',
        mode: str = 'plain',
        uncertainty_root: Optional[str] = None,
        face_regressor=None,
        uncertainty_fn: Optional[Callable] = None,
        camera_params: Optional[dict] = None,
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        self.mode = mode
        self.root = Path(root).resolve()
        self.uncertainty_root = Path(uncertainty_root).resolve() if uncertainty_root else None
        self.face_regressor = face_regressor
        self.uncertainty_fn = uncertainty_fn
        self.camera_params  = camera_params

        if mode in ('uncertainty_weighted', 'loss_weighted') and uncertainty_root is None and face_regressor is None:
            raise ValueError(
                f"mode='{mode}' requires either uncertainty_root "
                "(pre-computed .npy maps) or face_regressor + uncertainty_fn."
            )

        if transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples: List[Tuple[Path, int]] = []
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            label_key = cls_dir.name.lower()
            if label_key not in _CLASS_TO_IDX:
                continue
            label = _CLASS_TO_IDX[label_key]
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in _IMG_EXTS:
                    self.samples.append((img_path, label))

        self._warned_missing: set = set()
        self._warned_unreadable: set = set()

        # class label -> list of sample indices, for picking a same-class
        # fallback when a file turns out to be unreadable (see __getitem__).
        self._class_to_indices: Dict[int, List[int]] = {}
        for i, (_, label) in enumerate(self.samples):
            self._class_to_indices.setdefault(label, []).append(i)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int, _retries: int = 5):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, PermissionError) as e:
            if img_path not in self._warned_unreadable:
                warnings.warn(
                    f"Could not read image {img_path} ({e}). Substituting a "
                    "different sample from the same class for this fetch — "
                    "this file will be skipped for the rest of training. "
                    "Check filesystem permissions if this recurs often.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_unreadable.add(img_path)
            if _retries <= 0:
                raise
            candidates = [i for i in self._class_to_indices.get(label, []) if i != idx]
            fallback_idx = (
                candidates[np.random.randint(len(candidates))]
                if candidates else (idx + 1) % len(self.samples)
            )
            return self.__getitem__(fallback_idx, _retries=_retries - 1)

        if self.mode in ('uncertainty_weighted', 'loss_weighted'):
            # Get the confidence map in original image space before transform.
            img_np = np.asarray(image, dtype=np.float32) / 255.0   # (H, W, 3)
            conf   = self._get_confidence(img_path, img_np)          # (H, W) [0.5,1]

            # Save both the PyTorch and Python random states before applying
            # the full image transform.  torchvision spatial transforms
            # (RandomResizedCrop, RandomHorizontalFlip) consume torch.rand
            # internally; Python's random module is also saved for older
            # torchvision versions that use random.random() for the flip.
            # Replaying from the same states on the confidence map below
            # guarantees the map receives the IDENTICAL crop and flip as the
            # image, preserving spatial correspondence.
            import random as _random
            _rng_torch = torch.get_rng_state()
            _rng_py    = _random.getstate()

            # Apply transform to the UNMASKED image so ImageNet normalisation
            # operates on the original pixel distribution.  Applying the mask
            # before normalisation would push uncertain pixels to large negative
            # values (mean-subtracted black ≈ −2σ), creating out-of-distribution
            # inputs for the pretrained ViT backbone.
            img_tensor = self.transform(image)          # (3, H', W') normalised

            # Identify spatial (geometric) transforms for RNG replay.  Both
            # loss_weighted and uncertainty_weighted need to align the confidence
            # map with the same crop and flip applied to img_tensor.
            # Appearance transforms (ColorJitter) and tensor ops (ToTensor,
            # Normalize) are excluded — the confidence map stays float32.
            _GEO_TYPES = (
                T.RandomResizedCrop, T.RandomCrop, T.Resize, T.CenterCrop,
                T.RandomHorizontalFlip, T.RandomVerticalFlip,
            )
            geo_tfms = (
                [t for t in self.transform.transforms if isinstance(t, _GEO_TYPES)]
                if isinstance(self.transform, T.Compose) else []
            )

            if self.mode == 'loss_weighted':
                # Replay geometric transforms on the confidence map so the scalar
                # weight reflects uncertainty in the ACTUAL cropped region being
                # trained on, not the full pre-crop image.
                if geo_tfms:
                    torch.set_rng_state(_rng_torch)
                    _random.setstate(_rng_py)
                    conf_pil = Image.fromarray(
                        np.clip(conf * 255, 0, 255).astype(np.uint8), mode='L'
                    )
                    for t in geo_tfms:
                        conf_pil = t(conf_pil)
                    conf_for_weight = np.array(conf_pil, dtype=np.float32) / 255.0
                else:
                    conf_for_weight = conf
                # Use the 10th-percentile confidence rather than the mean.
                # Background pixels (no FLAME vertex projected) carry conf=1.0
                # and dominate the mean, compressing the effective weight range
                # to only ~7-17%.  The percentile targets the most-uncertain
                # face-region pixels and gives a ~5× wider range [0.5, 1.0].
                w = float(np.percentile(conf_for_weight, 10))
                scalar_weight = torch.tensor(w, dtype=torch.float32)
                return img_tensor, label, scalar_weight

            # uncertainty_weighted: pixel attenuation path.
            # Replay the same geometric (spatial) transforms on the confidence
            # map so that crop and flip match the image exactly.  Only spatial
            # transforms are replayed; appearance transforms (ColorJitter) and
            # tensor transforms (ToTensor, Normalize) are skipped — the map
            # stays as a float32 array throughout.
            if geo_tfms:
                torch.set_rng_state(_rng_torch)
                _random.setstate(_rng_py)
                conf_pil = Image.fromarray(
                    np.clip(conf * 255, 0, 255).astype(np.uint8), mode='L'
                )
                for t in geo_tfms:
                    conf_pil = t(conf_pil)
                conf = np.array(conf_pil, dtype=np.float32) / 255.0

            # Resize confidence map to match the output spatial size if the
            # geometric replay did not already produce the right dimensions
            # (e.g. non-Compose transform, or no spatial transform at all).
            _, th, tw = img_tensor.shape
            if conf.shape != (th, tw):
                import cv2 as _cv2
                conf = _cv2.resize(conf, (tw, th), interpolation=_cv2.INTER_LINEAR)

            # Multiply the normalised image by the confidence map.
            # In normalised space the channel mean ≈ 0, so multiplying by
            # conf ∈ [0.5, 1] attenuates uncertain regions toward the mean
            # (neutral / uninformative), rather than toward black (−2σ).
            conf_t = torch.from_numpy(conf).unsqueeze(0)  # (1, H', W')
            return img_tensor * conf_t, label

        return self.transform(image), label

    def _get_confidence(self, img_path: Path, img_np: np.ndarray) -> np.ndarray:
        """Load or compute the (H, W) float32 confidence map for img_path."""
        if self.uncertainty_root is not None:
            rel      = img_path.relative_to(self.root)
            npy_path = (self.uncertainty_root / rel).with_suffix('.npy')
            if npy_path.exists():
                return np.load(str(npy_path)).astype(np.float32)
            if npy_path not in self._warned_missing:
                warnings.warn(
                    f"Confidence map not found: {npy_path}. "
                    "Using all-ones fallback (no masking). "
                    "Run precompute_uncertainty_maps.py to generate maps.",
                    UserWarning,
                    stacklevel=4,
                )
                self._warned_missing.add(npy_path)

        if self.face_regressor is not None and self.uncertainty_fn is not None:
            from src.downstream import project_variance_to_2d
            unc   = self.uncertainty_fn(self.face_regressor, img_np)
            verts = self.face_regressor.get_vertices(img_np)
            H, W  = img_np.shape[:2]
            return project_variance_to_2d(verts, unc, (H, W), self.camera_params)

        return np.ones(img_np.shape[:2], dtype=np.float32)
