"""
Implementation of the base class for activation functions.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

# class ActivationFunction {
#     + float compute(float x)
#     + float derivative(float x)
# }

class ActivationFunction(ABC):
    """
    Abstract base class representing an activation function in a neural network.
    """
    @abstractmethod
    def compute(self, x: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray: ...