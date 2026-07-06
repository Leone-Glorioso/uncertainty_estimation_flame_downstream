import os
import sys
import importlib.util
import numpy as np
import torch
import cv2
from .base_wrapper import BaseFaceRegressorWrapper

# Dropout probability that smirk_checkpoint_data/trained.pt was trained with
_MCD_DROPOUT_RATE = 0.5


class SMIRKWrapper(BaseFaceRegressorWrapper):
    """
    Wrapper for SMIRK (Retsinas et al., CVPR 2024).
    Neural renderer replaces the differentiable rasterizer.

    self.model is the SmirkEncoder nn.Module — this is intentional so that
    MC Dropout uncertainty methods can walk model.modules() to find Dropout layers.
    self.flame holds the FLAME model separately.

    For MC Dropout (methods 2, 6, 7) a retrained checkpoint with nn.Dropout
    inserted in the expression encoder backbone is required.
    enable_dropout_for_inference() must be called after instantiation.
    """

    def __init__(self, device='cuda', use_mcd_checkpoint=False):
        self.use_mcd_checkpoint = use_mcd_checkpoint
        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Nested clone: git clone put the repo at models/SMIRK/smirk/
        self._smirk_root = os.path.join(self._project_root, 'models', 'SMIRK', 'smirk')
        super().__init__(device)  # sets self.device then calls self.model = self._load_model()

    def _load_model(self):
        smirk_src = os.path.join(self._smirk_root, 'src')
        # smirk_root itself contains src/__init__.py which shadows the project's
        # src/ package — do NOT add smirk_root to sys.path.  Only add smirk_src
        # (where smirk_encoder.py and FLAME/ live).  Also purge smirk_root if a
        # previous call accidentally inserted it.
        if self._smirk_root in sys.path:
            sys.path.remove(self._smirk_root)
        if smirk_src not in sys.path:
            sys.path.insert(0, smirk_src)
        # Keep the project root first so our own from src.xxx imports always win.
        if self._project_root not in sys.path:
            sys.path.insert(0, self._project_root)

        from FLAME.FLAME import FLAME

        # ── 1. Load SmirkEncoder ────────────────────────────────────────────
        if self.use_mcd_checkpoint:
            # The MCD checkpoint was trained with smirk_checkpoint_data/smirk_encoder.py,
            # which adds dropout_rate to each sub-encoder.  The stock encoder at
            # smirk_src/smirk_encoder.py has no dropout_rate parameter.
            # Use importlib to load the specific file by path — avoids sys.path pollution
            # and name collision with the already-imported stock module.
            mcd_enc_path = os.path.join(
                self._project_root, 'smirk_checkpoint_data', 'smirk_encoder.py')
            if not os.path.isfile(mcd_enc_path):
                raise FileNotFoundError(
                    f"MCD encoder definition not found: {mcd_enc_path}\n"
                    "Ensure smirk_checkpoint_data/ is present at the project root."
                )
            spec = importlib.util.spec_from_file_location('smirk_encoder_mcd', mcd_enc_path)
            _mcd_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_mcd_mod)
            smirk_encoder = _mcd_mod.SmirkEncoder(dropout_rate=_MCD_DROPOUT_RATE).to(self.device)
            ckpt_path = os.path.join(
                self._project_root, 'smirk_checkpoint_data', 'trained.pt')
        else:
            from smirk_encoder import SmirkEncoder
            smirk_encoder = SmirkEncoder().to(self.device)
            ckpt_path = os.path.join(self._smirk_root, 'pretrained_models', 'SMIRK_em1.pt')

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"SMIRK checkpoint not found at: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        # Checkpoint bundles the full training state; extract only the encoder weights
        encoder_state = {
            k.replace('smirk_encoder.', ''): v
            for k, v in checkpoint.items()
            if k.startswith('smirk_encoder.')
        }
        if not encoder_state:
            raise RuntimeError(
                f"No 'smirk_encoder.*' keys found in {ckpt_path}. "
                "Check the checkpoint was saved with the correct key prefix."
            )
        smirk_encoder.load_state_dict(encoder_state)
        smirk_encoder.eval()

        # ── 2. Load FLAME ───────────────────────────────────────────────────
        # SMIRK's FLAME.__init__ uses hardcoded relative paths:
        #   np.load('assets/l_eyelid.npy'), np.load('assets/r_eyelid.npy'),
        #   np.load('assets/mediapipe_landmark_embedding/...')
        # These are relative to CWD, not to FLAME.py.
        # Temporarily chdir to _smirk_root so the relative loads resolve correctly.
        flame_model_path = os.path.join(self._smirk_root, 'assets', 'FLAME2020', 'generic_model.pkl')
        flame_lmk_path = os.path.join(self._smirk_root, 'assets', 'landmark_embedding.npy')

        old_cwd = os.getcwd()
        try:
            os.chdir(self._smirk_root)
            self.flame = FLAME(
                flame_model_path=flame_model_path,
                flame_lmk_embedding_path=flame_lmk_path,
            ).to(self.device)
        finally:
            os.chdir(old_cwd)

        self.flame.eval()

        return smirk_encoder  # stored as self.model for MCD compatibility

    def enable_dropout_for_inference(self):
        """
        Forces all nn.Dropout layers in self.model to remain active during eval.
        Call this after instantiation when using the MCD checkpoint for methods 2/6/7.
        Has no effect unless the checkpoint was trained with Dropout layers.
        """
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.training = True

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
        Runs SMIRK encoder.
        Returns standardised keys: 'shape' (1,300), 'expression' (1,50),
        'pose' (1,3 global + 1,3 jaw), 'cam' (1,3), plus SMIRK-specific
        'eyelid_params' (1,2) and 'jaw_params' (1,3).
        """
        img_tensor = self._preprocess_image(image)
        with torch.no_grad():
            outputs = self.model(img_tensor)
        return {
            'shape': outputs['shape_params'].cpu().numpy(),
            'expression': outputs['expression_params'].cpu().numpy(),
            'pose': outputs['pose_params'].cpu().numpy(),
            'cam': outputs['cam'].cpu().numpy(),
            'jaw_params': outputs['jaw_params'].cpu().numpy(),
            'eyelid_params': outputs['eyelid_params'].cpu().numpy(),
        }

    def get_vertices(self, image: np.ndarray) -> np.ndarray:
        """
        Returns FLAME mesh vertices of shape (5023, 3) float32.
        self.model (SmirkEncoder) → FLAME parameter dict → self.flame → vertices.
        """
        img_tensor = self._preprocess_image(image)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            flame_output = self.flame.forward(outputs)
        return flame_output['vertices'][0].detach().cpu().numpy().astype(np.float32)
