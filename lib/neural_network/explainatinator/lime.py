"""
Implementation of LIME (Local Interpretable Model-agnostic Explanations) 
    for explaining the output of machine learning models.
"""
import numpy as np
from .base import Explainatinator

class LIME(Explainatinator):
    """
    Class implementing LIME (Local Interpretable Model-agnostic Explanations) 
        for explaining the output of machine learning models.
    """
    def explain(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...
