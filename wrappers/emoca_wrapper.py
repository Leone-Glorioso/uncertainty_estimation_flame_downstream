import os
import sys
import numpy as np
import torch
import cv2
from .base_wrapper import BaseFaceRegressorWrapper


class EMOCAWrapper(BaseFaceRegressorWrapper):
    """
    Wrapper for EMOCA (Daněček et al., CVPR 2022).
    Emotion-aware reconstruction built on the DECA architecture.
    Returns (5023, 3) vertices.

    NOTE: EMOCA's decode pipeline runs the pytorch3d renderer internally.
    Full CPU support is limited — CUDA is recommended.
    """

    def __init__(self, device='cuda'):
        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # EMOCA repo cloned at models/EMOCA/emoca/
        self._emoca_root = os.path.join(self._project_root, 'models', 'EMOCA', 'emoca')
        super().__init__(device)

    def _load_model(self):
        if self._emoca_root not in sys.path:
            sys.path.insert(0, self._emoca_root)
        # Also keep project root reachable (EMOCA imports nothing from our src/)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        try:
            from gdl_apps.EMOCA.utils.load import load_model
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                f"EMOCA failed to import gdl_apps: {exc}.  "
                "Exclude EMOCA via --models SMIRK DECA SHeaP to suppress this warning."
            ) from exc

        path_to_models = os.path.join(self._emoca_root, 'assets', 'EMOCA', 'models')
        run_name = 'EMOCA_v2_lr_mse_20'
        stage = 'detail'

        emoca, _conf = load_model(path_to_models, run_name, stage)
        emoca = emoca.to(self.device)
        emoca.eval()
        return emoca

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
        Runs EMOCA encoder.
        Returns standardised keys: 'shape' (1,100), 'expression' (1,50),
        'pose' (1,6), 'cam' inferred from 'posecode'.
        """
        img_tensor = self._preprocess_image(image)
        batch = {"image": img_tensor}
        with torch.no_grad():
            vals = self.model.encode(batch, training=False)
        return {
            'shape': vals['shapecode'].cpu().numpy(),
            'expression': vals['expcode'].cpu().numpy(),
            'pose': vals['posecode'].cpu().numpy(),
            'cam': vals['posecode'].cpu().numpy()[:, :3],  # first 3 dims approximate cam
        }

    def get_vertices(self, image: np.ndarray) -> np.ndarray:
        """
        Returns FLAME mesh vertices of shape (5023, 3) float32.
        Calls encode to get FLAME parameters, then runs FLAME directly.
        This bypasses the pytorch3d renderer, making CPU inference viable.
        """
        img_tensor = self._preprocess_image(image)
        batch = {"image": img_tensor}
        with torch.no_grad():
            vals = self.model.encode(batch, training=False)
            # Run FLAME directly instead of the full decode (which invokes the renderer)
            flame = self.model.deca.flame
            shape = vals['shapecode']
            expr  = vals['expcode']
            pose  = vals['posecode']
            from gdl.models.DecaFLAME import FLAME_mediapipe
            if not isinstance(flame, FLAME_mediapipe):
                verts, _, _ = flame(shape_params=shape, expression_params=expr, pose_params=pose)
            else:
                verts, _, _, _ = flame(shape, expr, pose)
        return verts[0].detach().cpu().numpy().astype(np.float32)
