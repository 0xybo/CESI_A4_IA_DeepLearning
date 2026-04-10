"""
Binary Cross-Entropy Loss Function

The Binary Cross-Entropy loss is defined as:
L = -1/N * sum(y_train * log(y_pred) + (1 - y_train) * log(1 - y_pred))
where:
- N is the number of samples
- y_train is the true binary label (0 or 1)
- y_pred is the predicted probability of the positive class (between 0 and 1)

The derivative of the Binary Cross-Entropy loss with respect to the predictions is:
dL/dy_pred = (y_pred - y_train) / (y_pred * (1 - y_pred) * N)
"""

from __future__ import annotations

import numpy as np
from .base import LossFunction


class BinaryCrossEntropySigmoid(LossFunction):
    """
    Binary Cross-Entropy loss function for binary classification tasks.
    It considers that the sigmoid is the last activation function used, thus it is not needed to add it to the last layer

    The Binary Cross-Entropy loss is defined as:
    L = -1/N * sum(y_train * log(y_pred) + (1 - y_train) * log(1 - y_pred))
    where:
    - N is the number of samples
    - y_train is the true binary label (0 or 1)
    - y_pred is the predicted probability of the positive class (between 0 and 1)

    The derivative of the Binary Cross-Entropy loss with respect to the predictions is:
    dL/dy_pred = (y_pred - y_train) / (y_pred * (1 - y_pred) * N)
    """

    def compute(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip y_pred to prevent log(0)
        y_pred = 1 / (1 + np.exp(-y_pred))  # Apply sigmoid to y_pred
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        print("LOSS :", y_pred)
        return -np.mean(
            y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred)
        )  # pyright: ignore[reportReturnType]

    def derivative(
        self,
        y_train: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        # Clip y_pred to prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        batch_size = y_pred.shape[1] if y_pred.ndim > 1 else y_pred.shape[0]
        return (y_pred - y_train) / (batch_size)

    def __str__(self) -> str:
        return "Binary Cross-Entropy (Sigmoid) Loss"

    def __repr__(self) -> str:
        return self.__str__()
