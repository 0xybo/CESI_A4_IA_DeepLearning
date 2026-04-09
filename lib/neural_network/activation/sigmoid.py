"""
Caca Activation Function Implementation
"""
from .base import ActivationFunction
import numpy as np

class Sigmoid(ActivationFunction):
    """
    Sigmoid Activation Function
    """
    def compute(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Sigmoid activation function on the input data.

        Args:
            x (np.ndarray): The input data to the activation function.

        Returns:
            np.ndarray: The output of the Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the Sigmoid activation function.

        Args:
            x (np.ndarray): The input data to compute the derivative on.

        Returns:
            np.ndarray: The derivative of the Sigmoid activation function.
        """
        sigmoid_x = self.compute(x)
        return sigmoid_x * (1 - sigmoid_x)
