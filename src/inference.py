import numpy as np
import os
from wrappers.smirk_wrapper import SMIRKWrapper
from wrappers.deca_wrapper import DECAWrapper
from wrappers.emoca_wrapper import EMOCAWrapper
from wrappers.sheap_wrapper import SHeaPWrapper

class UnifiedFaceRegressor:
    """
    Orchestrates the individual wrappers for DECA, EMOCA, SMIRK, and SHeaP.
    Used to generate unified outputs to calculate cross-method disagreement.

    Wrappers whose checkpoints are missing are skipped with a warning rather
    than crashing the whole object, so the pipeline can run with whatever
    models are currently installed.
    """
    def __init__(self, device='cuda', models=None):
        self.device = device

        _all_wrappers = {
            'SMIRK': lambda: SMIRKWrapper(device=self.device, use_mcd_checkpoint=False),
            'DECA':  lambda: DECAWrapper(device=self.device),
            'EMOCA': lambda: EMOCAWrapper(device=self.device),
            'SHeaP': lambda: SHeaPWrapper(device=self.device),
        }
        requested = models if models is not None else list(_all_wrappers.keys())

        self.models = {}
        for name in requested:
            if name not in _all_wrappers:
                import warnings
                warnings.warn(f"UnifiedFaceRegressor: unknown model '{name}' — skipped.")
                continue
            try:
                self.models[name] = _all_wrappers[name]()
            except FileNotFoundError as exc:
                import warnings
                warnings.warn(
                    f"UnifiedFaceRegressor: checkpoint for '{name}' not found "
                    f"({exc}) — skipped. Provide the checkpoint or remove '{name}' "
                    f"from --models to suppress this warning.",
                    RuntimeWarning,
                )
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"UnifiedFaceRegressor: failed to load '{name}' ({exc}) — skipped.",
                    RuntimeWarning,
                )

    def run_all_models(self, image: np.ndarray, save_dir: str = None) -> dict:
        """
        Runs the image through all four models. 
        Saves their per-vertex arrays (5023 x 3) to a common format (.npy).
        
        Returns:
            dict: Mapping of model names to their (5023, 3) vertex arrays.
        """
        results = {}
        for name, wrapper in self.models.items():
            vertices = wrapper.get_vertices(image)
            results[name] = vertices
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, f"{name}_vertices.npy"), vertices)
                
        return results
    def run_model(self, model_name: str, image: np.ndarray) -> np.ndarray:
        """
        Runs a single specified model on the input image.
        
        Args:
            model_name: One of 'SMIRK', 'DECA', 'EMOCA', 'SHeaP'.
            image: Input image as a numpy array.
        
        Returns:
            np.ndarray: The (5023, 3) vertex array output by the specified model.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not recognized. Choose from {list(self.models.keys())}.")
        
        return self.models[model_name].get_vertices(image)