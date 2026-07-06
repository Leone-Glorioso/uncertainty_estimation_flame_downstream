from abc import ABC, abstractmethod
import numpy as np
import torch

class BaseFaceRegressorWrapper(ABC):
    """
    Abstract base class for all monocular 3D face regressors.
    Ensures all models return standardized FLAME outputs for cross-method comparison.
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self):
        """Loads the specific model architecture and weights."""
        pass

    @abstractmethod
    def predict_parameters(self, image: np.ndarray) -> dict:
        """
        Runs the forward pass.
        
        Returns:
            dict: Standardized dictionary containing keys:
                  'shape' (beta), 'expression' (psi), 'pose' (theta), 'cam'.
        """
        pass

    @abstractmethod
    def get_vertices(self, image: np.ndarray) -> np.ndarray:
        """
        Converts the predicted parameters into a 3D mesh.
        
        Returns:
            np.ndarray: Per-vertex array of shape (5023, 3).
        """
        pass