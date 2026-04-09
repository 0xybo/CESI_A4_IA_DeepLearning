"""
Categorical Cross-Entropy Loss Function

The Categorical Cross-Entropy loss function is used for multi-class classification tasks. It
measures the dissimilarity between the true labels and the predicted probabilities. The loss is
defined as:
L = -1/N * sum(sum(y_train * log(y_pred)))
where:
- N is the number of samples
- y_train is the true one-hot encoded label (a binary vector where the index of the true class is
  1 and the rest are 0)
- y_pred is the predicted probability distribution over the classes (each value is between 0 and 1,
  and the sum of all values is 1)

The derivative of the Categorical Cross-Entropy loss with respect to the predictions is:
dL/dy_pred = (y_pred - y_train) / N
"""

import numpy as np
from .base import LossFunction


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical Cross-Entropy loss function for multi-class classification tasks.

    The Categorical Cross-Entropy loss is defined as:
    L = -1/N * sum(sum(y_train * log(y_pred)))
    where:
    - N is the number of samples
    - y_train is the true one-hot encoded label (a binary vector where the index of the true class
      is 1 and the rest are 0)
    - y_pred is the predicted probability distribution over the classes (each value is between 0
      and 1, and the sum of all values is 1)

    The derivative of the Categorical Cross-Entropy loss with respect to the predictions is:
    dL/dy_pred = (y_pred - y_train) / N
    """

    def compute(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip y_pred to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(
            np.sum(y_train * np.log(y_pred), axis=1)
        )  # pyright: ignore[reportReturnType]

    def derivative(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip y_pred to prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_train) / y_train.shape[0]
