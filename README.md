# 3D Face Reconstruction Uncertainty and Downstream Weighting

This project investigates per-vertex uncertainty estimation for monocular 3D face regressors (DECA, EMOCA, SMIRK, SHeaP) and leverages this uncertainty to robustify downstream facial expression classification under occlusions[cite: 38, 97, 100].

## Core Objectives
1. **Mesh Reconstruction:** Extract 5023-vertex FLAME meshes from in-the-wild images[cite: 30, 44].
2. **Uncertainty Quantification:** Estimate aleatoric and epistemic uncertainty without retraining standard models using Test-Time Augmentation (TTA) and Cross-Method Disagreement, plus Monte Carlo (MC) Dropout on a custom SMIRK checkpoint[cite: 59, 60, 66, 68, 90].
3. **Downstream Extension:** Project 3D variance back to the 2D image plane to act as an attention mask for CNN classifiers, or append as explicit node features for GCNs, down-weighting occluded/hallucinated regions[cite: 103].