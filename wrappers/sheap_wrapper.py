import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
from .base_wrapper import BaseFaceRegressorWrapper


class SHeaPWrapper(BaseFaceRegressorWrapper):
    """
    Wrapper for SHeaP (Schoneveld et al., 2025).
    Self-Supervised Head Geometry Predictor via 2D Gaussian Splatting.

    self.model is the TorchScript JIT SHeaP regressor.
    self.flame is TinyFlame — a lightweight FLAME implementation that accepts
    the parameter dict directly from the SHeaP model output.

    The JIT model is auto-downloaded on first use (~200 MB) to
    models/SHeaP/SHeaP/models/model_paper.pt if not already present.
    """

    def __init__(self, device='cuda'):
        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # The sheap/ Python package lives at models/SHeaP/SHeaP/ (add to sys.path).
        # The .pt checkpoint files live at models/SHeaP/data/ (pass as models_dir).
        # FLAME2020 assets (.pt) are at models/SHeaP/SHeaP/FLAME2020/.
        self._sheap_root = os.path.join(self._project_root, 'models', 'SHeaP', 'SHeaP')
        self._sheap_models_dir = os.path.join(self._project_root, 'models', 'SHeaP', 'data')
        self._sheap_flame_dir = os.path.join(self._sheap_root, 'FLAME2020')
        super().__init__(device)

    def _load_model(self):
        if self._sheap_root not in sys.path:
            sys.path.insert(0, self._sheap_root)

        from sheap import load_sheap_model
        from sheap.tiny_flame import TinyFlame

        # ── 1. Load SHeaP JIT model ─────────────────────────────────────────
        # model_paper.pt lives in models/SHeaP/data/ (_sheap_models_dir)
        models_dir = Path(self._sheap_models_dir)
        sheap_model = load_sheap_model(model_type='paper', models_dir=models_dir)
        sheap_model = sheap_model.to(self.device)

        # ── 2. Load TinyFlame with explicit absolute paths ──────────────────
        flame_dir = Path(self._sheap_flame_dir)
        self.flame = TinyFlame(
            ckpt=flame_dir / 'generic_model.pt',
            eyelids_ckpt=flame_dir / 'eyelids.pt',
        ).to(self.device)
        self.flame.eval()

        return sheap_model  # stored as self.model

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Converts HxWx3 uint8 RGB numpy array to (1,3,224,224) float32 tensor in [0,1]."""
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        image = cv2.resize(image, (224, 224))
        img_float = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor

    def predict_parameters(self, image: np.ndarray) -> dict:
        """
        Runs SHeaP regressor.
        Returns standardised keys: 'shape' (1,300), 'expression' (1,100),
        'pose' (dict of 5 rotvecs), 'cam' (1,3 translation), plus full
        SHeaP prediction dict under 'raw'.
        """
        img_tensor = self._preprocess_image(image)
        with torch.no_grad():
            predictions = self.model(img_tensor)
        return {
            'shape': predictions['shape_from_facenet'].cpu().numpy(),
            'expression': predictions['expr'].cpu().numpy(),
            'pose': {
                'torso_pose': predictions['torso_pose'].cpu().numpy(),
                'neck_pose': predictions['neck_pose'].cpu().numpy(),
                'jaw_pose': predictions['jaw_pose'].cpu().numpy(),
                'eye_l_pose': predictions['eye_l_pose'].cpu().numpy(),
                'eye_r_pose': predictions['eye_r_pose'].cpu().numpy(),
            },
            'cam': predictions['cam_trans'].cpu().numpy(),
            'eyelids': predictions['eyelids'].cpu().numpy(),
        }

    def get_vertices(self, image: np.ndarray) -> np.ndarray:
        """
        Returns FLAME mesh vertices of shape (5023, 3) float32.
        SHeaP JIT model → FLAME parameter dict → TinyFlame → vertices.
        """
        from sheap.tiny_flame import pose_components_to_rotmats

        img_tensor = self._preprocess_image(image)
        with torch.no_grad():
            predictions = self.model(img_tensor)
            pose_mats = pose_components_to_rotmats(predictions)
            verts = self.flame(
                shape=predictions['shape_from_facenet'],
                expression=predictions['expr'],
                pose=pose_mats,
                eyelids=predictions['eyelids'],
                translation=predictions['cam_trans'],
            )
        return verts[0].detach().cpu().numpy().astype(np.float32)
