import os
import sys
import numpy as np
import torch
import cv2
from .base_wrapper import BaseFaceRegressorWrapper


class DECAWrapper(BaseFaceRegressorWrapper):
    """
    Wrapper for DECA (Feng et al., SIGGRAPH 2021).
    Predicts FLAME parameters from a monocular image via a coarse encoder.
    Returns (5023, 3) vertices on both CPU and GPU.
    """

    def __init__(self, device='cuda'):
        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Nested clone: git clone put the repo at models/DECA/models/DECA/
        self._deca_root = os.path.join(self._project_root, 'models', 'DECA', 'models', 'DECA')
        super().__init__(device)

    def _load_model(self):
        if self._deca_root not in sys.path:
            sys.path.insert(0, self._deca_root)

        from decalib.deca import DECA
        from decalib.utils.config import cfg as deca_cfg

        # Override device and disable texture (avoids missing FLAME_albedo_from_BFM.npz)
        deca_cfg.device = self.device
        deca_cfg.model.use_tex = False
        deca_cfg.rasterizer_type = 'pytorch3d'

        # The fetch_data.sh places generic_model.pkl in data/FLAME2020/FLAME2020/
        # but config expects data/generic_model.pkl — override to correct path
        deca_cfg.model.flame_model_path = os.path.join(
            self._deca_root, 'data', 'FLAME2020', 'FLAME2020', 'generic_model.pkl'
        )

        model = DECA(config=deca_cfg, device=self.device)
        model.eval()
        return model

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
        Runs DECA encoder.
        Returns standardised keys: 'shape' (1,100), 'expression' (1,50),
        'pose' (1,6), 'cam' (1,3).
        """
        img_tensor = self._preprocess_image(image)
        with torch.no_grad():
            codedict = self.model.encode(img_tensor, use_detail=False)
        return {
            'shape': codedict['shape'].cpu().numpy(),
            'expression': codedict['exp'].cpu().numpy(),
            'pose': codedict['pose'].cpu().numpy(),
            'cam': codedict['cam'].cpu().numpy(),
        }

    def get_vertices(self, image: np.ndarray) -> np.ndarray:
        """
        Returns FLAME mesh vertices of shape (5023, 3) float32.
        decode() is called with rendering=False, use_detail=False, return_vis=False
        to avoid the rendering / detail pipeline and return opdict directly.
        """
        img_tensor = self._preprocess_image(image)
        with torch.no_grad():
            codedict = self.model.encode(img_tensor, use_detail=False)
            opdict = self.model.decode(
                codedict,
                rendering=False,
                use_detail=False,
                return_vis=False,
                vis_lmk=False,
            )
        return opdict['verts'][0].detach().cpu().numpy().astype(np.float32)
