"""
Tanh Activation Function Implementation
"""
from .base import ActivationFunction
import numpy as np

class Tanh(ActivationFunction):
    """
    Tanh Activation Function
    """
    def compute(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Tanh activation function on the input data.

        Args:
            x (np.ndarray): The input data to the activation function.

        Returns:
            np.ndarray: The output of the Tanh activation function.
        """
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the Tanh activation function.

        Args:
            x (np.ndarray): The input data to compute the derivative on.

        Returns:
            np.ndarray: The derivative of the Tanh activation function.
        """
        tanh_x = self.compute(x)
        return 1 - tanh_x ** 2
