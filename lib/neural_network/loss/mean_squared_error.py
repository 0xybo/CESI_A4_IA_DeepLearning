"""
Mean Squared Error Loss Function

The Mean Squared Error loss function is used for regression tasks. It measures the average squared
difference between the true values and the predicted values. The loss is defined as:
L = 1/N * sum(y_train - y_pred)^2
where:
- N is the number of samples
- y_train is the true value (a scalar or array of scalars)
- y_pred is the predicted value (a scalar or array of scalars)

The derivative of the Mean Squared Error loss with respect to the predictions is:
dL/dy_pred = 2 * (y_pred - y_train) / N
"""

import numpy as np
from .base import LossFunction


class MeanSquaredError(LossFunction):
    """
    Mean Squared Error loss function for regression tasks.

    The Mean Squared Error loss is defined as:
    L = 1/N * sum(y_train - y_pred)^2
    where:
    - N is the number of samples
    - y_train is the true value (a scalar or array of scalars)
    - y_pred is the predicted value (a scalar or array of scalars)
    The derivative of the Mean Squared Error loss with respect to the predictions is:
    dL/dy_pred = 2 * (y_pred - y_train) / N
    """

    def compute(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_train - y_pred) ** 2)  # pyright: ignore[reportReturnType]

    def derivative(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        return 2 * (y_pred - y_train) / y_train.shape[0]
