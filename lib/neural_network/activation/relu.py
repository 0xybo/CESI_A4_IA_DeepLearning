"""
ReLU Activation Function Implementation
"""

import numpy as np
from .base import ActivationFunction  # pylint: disable=relative-beyond-top-level


class Relu(ActivationFunction):
    """
    ReLU Activation Function
    """

    def compute(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the ReLU activation function on the input data.

        Args:
            x (np.ndarray): The input data.

        Returns:
            np.ndarray: The output of the ReLU activation function.
        """
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the ReLU activation function on the input data.

        Args:
            x (np.ndarray): The input data.

        Returns:
            np.ndarray: The output of the derivative of the ReLU activation function.
        """
        return (x > 0).astype(float)

    def __str__(self) -> str:
        return "Relu"

    def __repr__(self) -> str:
        return self.__str__()
