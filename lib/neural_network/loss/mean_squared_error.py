from .base import LossFunction
import numpy as np


class MeanSquaredError(LossFunction):
    def compute(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_train - y_pred) ** 2)  # pyright: ignore[reportReturnType]

    def derivative(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        return 2 * (y_pred - y_train) / y_train.shape[0]
