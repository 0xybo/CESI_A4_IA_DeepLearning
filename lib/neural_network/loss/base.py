"""
Neural Network Loss Functions Package
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    """
    Base class for loss functions in neural network training.

    This class defines the interface for loss functions, which includes methods to compute the loss
    value and its derivative with respect to the predicted outputs. Specific loss functions, such
    as mean squared error or cross-entropy, should inherit from this base class and implement the
    compute and derivative methods.
    """

    @abstractmethod
    def compute(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss value given the true labels and predicted outputs.

        Args:
        - y_train (np.ndarray): The true labels for the training data.
        - y_pred (np.ndarray): The predicted outputs for the training data.
        Returns:
        - float: The computed loss value.
        """

    @abstractmethod
    def derivative(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the derivative of the loss function with respect to the predicted outputs.

        Args:
        - y_train (np.ndarray): The true labels for the training data.
        - y_pred (np.ndarray): The predicted outputs for the training data.
        Returns:
        - float: The computed derivative value.
        """
