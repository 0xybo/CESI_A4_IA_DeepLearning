"""
Tanh Activation Function Implementation
"""

import numpy as np
from .base import ActivationFunction  # pylint: disable=relative-beyond-top-level


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
        return 1 - tanh_x**2

    def __str__(self) -> str:
        """
        Returns a string representation of the Tanh activation function.

        Returns:
            str: The name of the activation function.
        """
        return "tanh"

    def __repr__(self) -> str:
        return self.__str__()
