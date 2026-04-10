"""
Mean Absolute Error Loss Function

The Mean Absolute Error loss function is used for regression tasks. It measures the average
absolute difference between the true values and the predicted values. The loss is defined as:
L = 1/N * sum|y_train - y_pred|
where:
- N is the number of samples
- y_train is the true value (a scalar or array of scalars)
- y_pred is the predicted value (a scalar or array of scalars)

The derivative of the Mean Absolute Error loss with respect to the predictions is:
dL/dy_pred = sign(y_pred - y_train) / N

"""

import numpy as np
from .base import LossFunction


class MeanAbsoluteError(LossFunction):
    """
    Mean Absolute Error loss function for regression tasks.

    The Mean Absolute Error loss is defined as:
    L = 1/N * sum|y_train - y_pred|
    where:
    - N is the number of samples
    - y_train is the true value (a scalar or array of scalars)
    - y_pred is the predicted value (a scalar or array of scalars)

    The derivative of the Mean Absolute Error loss with respect to the predictions is:
    dL/dy_pred = sign(y_pred - y_train) / N
    """

    def compute(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_train - y_pred))  # pyright: ignore[reportReturnType]

    def derivative(self, y_train: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sign(y_pred - y_train) / y_train.shape[0]

    def __str__(self) -> str:
        return "Mean Absolute Error"

    def __repr__(self) -> str:
        return self.__str__()
