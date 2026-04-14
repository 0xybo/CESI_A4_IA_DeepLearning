"""
Implementation of SHAP (SHapley Additive exPlanations) 
    for explaining the output of machine learning models.
"""
import numpy as np
from .base import Explainatinator

class SHAP(Explainatinator):
    """
    Class implementing SHAP (SHapley Additive exPlanations) for explaining the output of machine learning models.
    """
    def explain(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...
