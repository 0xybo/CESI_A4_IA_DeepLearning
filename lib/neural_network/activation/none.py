"""
Implementation for no activation function
"""

import numpy as np
from .base import ActivationFunction  # pylint: disable=relative-beyond-top-level


class NoActivation(ActivationFunction):
    """
    No Activation Function
    """

    def compute(self, x: np.ndarray) -> np.ndarray:
        print("NONE : ", x)
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x

    def __str__(self) -> str:
        return "None"

    def __repr__(self) -> str:
        return self.__str__()
